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


# student
model_config = "mmrazor_configs/mobilenet_v2_config.py"
model_path = "output/kd_mobilenetv2_epoch_55.pth"

student = init_model(model_config)
state_dict = torch.load(model_path)['state_dict']
state_dict = {key.replace('architecture.', ''): value for key, value in state_dict.items()}
state_dict = {k: v for k, v in state_dict.items() if k in student.state_dict()}
student.load_state_dict(state_dict)

save_path = model_path.replace(".pth",".ptl")
convert_utils.save_as_ptl(save_path, student, torch.rand(1, 3, 224, 224))


print(f"Size: {os.path.getsize(save_path)}")
print(f"Zip Size: {model_size_utils.get_gzipped_model_size(save_path)}")
print(f"Accuracy: {eval_model.evaluate_accuracy(student, prepare_data_loader())}")


# teacher
teacher = init_model("mmrazor_configs/densenet201_config.py", 'https://download.openmmlab.com/mmclassification/v0/densenet/densenet201_4xb256_in1k_20220426-05cae4ef.pth')
teacher_save_path = "output/kd_densenet201.ptl"
convert_utils.save_as_ptl(teacher_save_path, teacher, torch.rand(1, 3, 224, 224))

print(f"Size: {os.path.getsize(teacher_save_path)}")
print(f"Zip Size: {model_size_utils.get_gzipped_model_size(teacher_save_path)}")
print(f"Accuracy: {eval_model.evaluate_accuracy(teacher, prepare_data_loader())}")
