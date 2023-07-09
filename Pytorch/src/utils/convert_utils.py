import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
def save_as_ptl(file_path, model, example_input):
    example_input.cpu()
    traced_script_module = torch.jit.trace(model.cpu(), example_input)
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    traced_script_module_optimized._save_for_lite_interpreter(file_path)