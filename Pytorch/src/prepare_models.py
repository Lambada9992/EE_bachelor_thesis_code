import copy
import torch
import os
import torchvision
import utils.eval_model as eval_model
from torch.utils.data import DataLoader
import utils.prune_model as prune_model
import utils.qunatize_utils as quantize_utils
import utils.convert_utils as convert_utils
import utils.model_size_utils as model_size_utils
import utils.save_stats_utils as save_stats_utils

from utils.qunatize_utils import QuantizeMode

imagenet_root = "/home/marcinwsl/tensorflow_datasets/downloads/manual"
excel_path = f'output/result.xlsx'
if not os.path.exists('output'):
    os.makedirs('output')

models = [
    {"name":"densenet121","model": torchvision.models.densenet121(weights="IMAGENET1K_V1"), "imagesize":224, "transform": torchvision.models.DenseNet121_Weights.IMAGENET1K_V1.transforms()},
    {"name":"densenet169","model": torchvision.models.densenet169(weights="IMAGENET1K_V1"), "imagesize":224, "transform": torchvision.models.DenseNet169_Weights.IMAGENET1K_V1.transforms()},
    {"name":"densenet201","model": torchvision.models.densenet201(weights="IMAGENET1K_V1"), "imagesize":224, "transform": torchvision.models.DenseNet201_Weights.IMAGENET1K_V1.transforms()},
    {"name":"inception_v3","model": torchvision.models.inception_v3(weights="IMAGENET1K_V1"), "imagesize":299, "transform": torchvision.models.Inception_V3_Weights.IMAGENET1K_V1.transforms()},
    {"name":"mobilenet_v2","model": torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1"), "imagesize":224, "transform": torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1.transforms()},
    {"name":"resnet101","model": torchvision.models.resnet101(weights="IMAGENET1K_V1"), "imagesize":224, "transform": torchvision.models.ResNet101_Weights.IMAGENET1K_V1.transforms()},
    {"name":"resnet152","model": torchvision.models.resnet152(weights="IMAGENET1K_V1"), "imagesize":224, "transform": torchvision.models.ResNet152_Weights.IMAGENET1K_V1.transforms()},
    {"name":"resnet50","model": torchvision.models.resnet50(weights="IMAGENET1K_V1"), "imagesize":224, "transform": torchvision.models.ResNet50_Weights.IMAGENET1K_V1.transforms()},
]


def prepare_data_loaders(transform):
    train_dataset = torchvision.datasets.ImageNet(root=imagenet_root, split='train', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)
    val_dataset = torchvision.datasets.ImageNet(root=imagenet_root, split='val', transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=10)

    return train_dataloader, val_dataloader

def prepare_model_name(model_name, sparse, quantization: QuantizeMode):
    _name = f"{model_name}"
    sparse_text = sparse if sparse is not None else "0_0"
    quantization_name = next(enum_name for enum_name, value in vars(QuantizeMode).items() if value == quantization)
    _name = f"{_name}_quantized_{quantization_name}_sparse_{sparse_text}".replace(".", "_")
    return f"{_name}.ptl"

def process_model(
        name,
        model,
        imagesize,
        train_dataloader,
        val_dataloader,
        sparse=None,
        quantization: QuantizeMode = QuantizeMode.NONE,
):
    try:
        save_path = f"output/{prepare_model_name(name, sparse, quantization)}"
        if os.path.exists(save_path) :
            return

        print(f"Processing {name} with sparse {sparse} and quantization {quantization}")

        _model = copy.deepcopy(model)

        if sparse is not None:
            _model = prune_model.prune_model(_model, sparse)

        _model = quantize_utils.qunatize(_model, quantization, train_dataloader,
                                         example_input=torch.rand(1, 3, imagesize, imagesize))


        model_accuracy = eval_model.evaluate_accuracy(_model, val_dataloader)
        print(f"Accuracy: {model_accuracy}")

        convert_utils.save_as_ptl(save_path, _model, torch.rand(1, 3, imagesize, imagesize))
        model_size = model_size_utils.get_gzipped_model_size(save_path)
        print(f"Size: {model_size}")
        sparse_string = sparse if sparse is not None else 0.0
        save_stats_utils.save_model_stats_to_file(excel_path, name, sparse_string, quantization, model_accuracy, model_size)

    except Exception as e:
        print(f"Error in {name} with sparse {sparse} and quantization {quantization}: {e}")
        # raise e



for data in models:
    try:
        name, model, imagesize = data["name"], data["model"], data["imagesize"]
        model.eval()

        train_data, val_data = prepare_data_loaders(data['transform'])

        for quantization_mode in QuantizeMode:
            for sparse_value in [None, 0.1, 0.2, 0.3, 0.4, 0.5]:
                process_model(name, model, imagesize, train_data, val_data, sparse=sparse_value, quantization=quantization_mode)
    except Exception as e:
        print(f"Error in {data['name']}: {e}")
