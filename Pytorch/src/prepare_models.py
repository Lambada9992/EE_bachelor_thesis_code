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

test_models = [
    {
        "name": "mobilnet_v2",
        # "model": torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1),
        "model": torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1"),
        "imagesize": 224,
        "transform": torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1.transforms()
    }
]

models = [
    # {"name":"alexnet","model": torchvision.models.alexnet(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"convnext_base","model": torchvision.models.convnext_base(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"convnext_large","model": torchvision.models.convnext_large(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"convnext_small","model": torchvision.models.convnext_small(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"convnext_tiny","model": torchvision.models.convnext_tiny(weights="IMAGENET1K_V1"), "imagesize":224},
    {"name":"densenet121","model": torchvision.models.densenet121(weights="IMAGENET1K_V1"), "imagesize":224, "transform": torchvision.models.DenseNet121_Weights.IMAGENET1K_V1.transforms()},
    # {"name":"densenet161","model": torchvision.models.densenet161(weights="IMAGENET1K_V1"), "imagesize":224},
    {"name":"densenet169","model": torchvision.models.densenet169(weights="IMAGENET1K_V1"), "imagesize":224, "transform": torchvision.models.DenseNet169_Weights.IMAGENET1K_V1.transforms()},
    {"name":"densenet201","model": torchvision.models.densenet201(weights="IMAGENET1K_V1"), "imagesize":224, "transform": torchvision.models.DenseNet201_Weights.IMAGENET1K_V1.transforms()},
    # {"name":"efficientnet_b0","model": torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1"), "imagesize":224, "transform": torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()},
    # {"name":"efficientnet_b1","model": torchvision.models.efficientnet_b1(weights="IMAGENET1K_V1"), "imagesize":224, "transform": torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V1.transforms()},
    # {"name":"efficientnet_b2","model": torchvision.models.efficientnet_b2(weights="IMAGENET1K_V1"), "imagesize":224, "transform": torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1.transforms()},
    # {"name":"efficientnet_b3","model": torchvision.models.efficientnet_b3(weights="IMAGENET1K_V1"), "imagesize":224, "transform": torchvision.models.EfficientNet_B3_Weights.IMAGENET1K_V1.transforms()},
    # {"name":"efficientnet_b4","model": torchvision.models.efficientnet_b4(weights="IMAGENET1K_V1"), "imagesize":224, "transform": torchvision.models.EfficientNet_B4_Weights.IMAGENET1K_V1.transforms()},
    # {"name":"efficientnet_b5","model": torchvision.models.efficientnet_b5(weights="IMAGENET1K_V1"), "imagesize":224, "transform": torchvision.models.EfficientNet_B5_Weights.IMAGENET1K_V1.transforms()},
    # {"name":"efficientnet_b6","model": torchvision.models.efficientnet_b6(weights="IMAGENET1K_V1"), "imagesize":224, "transform": torchvision.models.EfficientNet_B6_Weights.IMAGENET1K_V1.transforms()},
    # {"name":"efficientnet_b7","model": torchvision.models.efficientnet_b7(weights="IMAGENET1K_V1"), "imagesize":224, "transform": torchvision.models.EfficientNet_B7_Weights.IMAGENET1K_V1.transforms()},
    # {"name":"efficientnet_v2_l","model": torchvision.models.efficientnet_v2_l(weights="IMAGENET1K_V1"), "imagesize":224, "transform": torchvision.models.},
    # {"name":"efficientnet_v2_m","model": torchvision.models.efficientnet_v2_m(weights="IMAGENET1K_V1"), "imagesize":224, "transform": torchvision.models.},
    # {"name":"efficientnet_v2_s","model": torchvision.models.efficientnet_v2_s(weights="IMAGENET1K_V1"), "imagesize":224, "transform": torchvision.models.},
    # {"name":"googlenet","model": torchvision.models.googlenet(weights="IMAGENET1K_V1"), "imagesize":224},
    {"name":"inception_v3","model": torchvision.models.inception_v3(weights="IMAGENET1K_V1"), "imagesize":299, "transform": torchvision.models.Inception_V3_Weights.IMAGENET1K_V1.transforms()},
    # {"name":"mnasnet0_5","model": torchvision.models.mnasnet0_5(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"mnasnet0_75","model": torchvision.models.mnasnet0_75(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"mnasnet1_0","model": torchvision.models.mnasnet1_0(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"mnasnet1_3","model": torchvision.models.mnasnet1_3(weights="IMAGENET1K_V1"), "imagesize":224},
    {"name":"mobilenet_v2","model": torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1"), "imagesize":224, "transform": torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1.transforms()},
    # {"name":"mobilenet_v3_large","model": torchvision.models.mobilenet_v3_large(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"mobilenet_v3_small","model": torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"regnet_x_16gf","model": torchvision.models.regnet_x_16gf(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"regnet_x_1_6gf","model": torchvision.models.regnet_x_1_6gf(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"regnet_x_32gf","model": torchvision.models.regnet_x_32gf(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"regnet_x_3_2gf","model": torchvision.models.regnet_x_3_2gf(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"regnet_x_400mf","model": torchvision.models.regnet_x_400mf(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"regnet_x_800mf","model": torchvision.models.regnet_x_800mf(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"regnet_x_8gf","model": torchvision.models.regnet_x_8gf(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"regnet_y_1_6gf","model": torchvision.models.regnet_y_1_6gf(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"regnet_y_32gf","model": torchvision.models.regnet_y_32gf(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"regnet_y_3_2gf","model": torchvision.models.regnet_y_3_2gf(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"regnet_y_400mf","model": torchvision.models.regnet_y_400mf(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"regnet_y_800mf","model": torchvision.models.regnet_y_800mf(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"regnet_y_8gf","model": torchvision.models.regnet_y_8gf(weights="IMAGENET1K_V1"), "imagesize":224},
    {"name":"resnet101","model": torchvision.models.resnet101(weights="IMAGENET1K_V1"), "imagesize":224, "transform": torchvision.models.ResNet101_Weights.IMAGENET1K_V1.transforms()},
    {"name":"resnet152","model": torchvision.models.resnet152(weights="IMAGENET1K_V1"), "imagesize":224, "transform": torchvision.models.ResNet152_Weights.IMAGENET1K_V1.transforms()},
    # {"name":"resnet18","model": torchvision.models.resnet18(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"resnet34","model": torchvision.models.resnet34(weights="IMAGENET1K_V1"), "imagesize":224},
    {"name":"resnet50","model": torchvision.models.resnet50(weights="IMAGENET1K_V1"), "imagesize":224, "transform": torchvision.models.ResNet50_Weights.IMAGENET1K_V1.transforms()},
    # {"name":"resnext101_32x8d","model": torchvision.models.resnext101_32x8d(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"resnext101_64x4d","model": torchvision.models.resnext101_64x4d(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"resnext50_32x4d","model": torchvision.models.resnext50_32x4d(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"shufflenet_v2_x0_5","model": torchvision.models.shufflenet_v2_x0_5(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"shufflenet_v2_x1_0","model": torchvision.models.shufflenet_v2_x1_0(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"shufflenet_v2_x1_5","model": torchvision.models.shufflenet_v2_x1_5(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"shufflenet_v2_x2_0","model": torchvision.models.shufflenet_v2_x2_0(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"squeezenet1_0","model": torchvision.models.squeezenet1_0(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"squeezenet1_1","model": torchvision.models.squeezenet1_1(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"swin_b","model": torchvision.models.swin_b(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"swin_s","model": torchvision.models.swin_s(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"swin_t","model": torchvision.models.swin_t(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"swin_v2_b","model": torchvision.models.swin_v2_b(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"swin_v2_s","model": torchvision.models.swin_v2_s(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"swin_v2_t","model": torchvision.models.swin_v2_t(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"vgg11","model": torchvision.models.vgg11(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"vgg11_bn","model": torchvision.models.vgg11_bn(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"vgg13","model": torchvision.models.vgg13(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"vgg13_bn","model": torchvision.models.vgg13_bn(weights="IMAGENET1K_V1"), "imagesize":224},
    {"name":"vgg16","model": torchvision.models.vgg16(weights="IMAGENET1K_V1"), "imagesize":224, "transform": torchvision.models.VGG16_Weights.IMAGENET1K_V1.transforms()},
    # {"name":"vgg16_bn","model": torchvision.models.vgg16_bn(weights="IMAGENET1K_V1"), "imagesize":224},
    {"name":"vgg19","model": torchvision.models.vgg19(weights="IMAGENET1K_V1"), "imagesize":224, "transform":  torchvision.models.VGG19_Weights.IMAGENET1K_V1.transforms()},
    # {"name":"vgg19_bn","model": torchvision.models.vgg19_bn(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"vit_b_16","model": torchvision.models.vit_b_16(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"vit_b_32","model": torchvision.models.vit_b_32(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"vit_l_16","model": torchvision.models.vit_l_16(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"vit_l_32","model": torchvision.models.vit_l_32(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"wide_resnet101_2","model": torchvision.models.wide_resnet101_2(weights="IMAGENET1K_V1"), "imagesize":224},
    # {"name":"wide_resnet50_2","model": torchvision.models.wide_resnet50_2(weights="IMAGENET1K_V1"), "imagesize":224},
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

        _model = model

        if sparse is not None:
            _model = prune_model.prune_model(_model, sparse)

        _model = quantize_utils.qunatize(_model, quantization, train_dataloader,
                                         example_input=torch.rand(1, 3, imagesize, imagesize))


        model_accuracy = eval_model.evaluate_accuracy(_model, val_dataloader)
        # model_accuracy = 0.0
        print(f"Accuracy: {model_accuracy}")

        convert_utils.save_as_ptl(save_path, _model, torch.rand(1, 3, imagesize, imagesize))
        model_size = model_size_utils.get_gzipped_model_size(save_path)
        print(f"Size: {model_size}")
        sparse_string = sparse if sparse is not None else 0.0
        save_stats_utils.save_model_stats_to_file(excel_path, name, sparse_string, quantization, model_accuracy, model_size)

    except Exception as e:
        print(f"Error in {name} with sparse {sparse} and quantization {quantization}: {e}")
        # raise e



for data in test_models:
    try:
        name, model, imagesize = data["name"], data["model"], data["imagesize"]
        model.eval()

        train_data, val_data = prepare_data_loaders(data['transform'])

        for quantization_mode in QuantizeMode:
            for sparse_value in [None, 0.1, 0.2, 0.3, 0.4, 0.5]:
                process_model(name, model, imagesize, train_data, val_data, sparse=sparse_value, quantization=quantization_mode)


        pass
    except Exception as e:
        print(f"Error in {data['name']}: {e}")
        # raise e