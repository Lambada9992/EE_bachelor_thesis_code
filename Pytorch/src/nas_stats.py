import torch
import os
import torchvision
import utils.eval_model as eval_model
from torch.utils.data import DataLoader
import utils.convert_utils as convert_utils
import utils.model_size_utils as model_size_utils
from mmcls import init_model

def prepare_data_loader():
    imagenet_root = "/home/marcinwsl/tensorflow_datasets/downloads/manual"
    transform = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1.transforms()
    val_dataset = torchvision.datasets.ImageNet(root=imagenet_root, split='val', transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=10)
    return val_dataloader


config_path =  "mmrazor_configs/spos/spos_mobilenet_subnet_8xb128_in1k.py"
model_path = "my_nas_subnet/epoch_24.pth"

student = init_model(config_path, model_path)

save_path = model_path.replace(".pth",".ptl")
convert_utils.save_as_ptl(save_path, student, torch.rand(1, 3, 224, 224))

print(f"Size: {os.path.getsize(save_path)}")
print(f"Zip Size: {model_size_utils.get_gzipped_model_size(save_path)}")
print(f"Accuracy: {eval_model.evaluate_accuracy(student, prepare_data_loader())}")