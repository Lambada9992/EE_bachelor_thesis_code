import copy
from torch.nn.utils import prune
import torch
def is_prunable(layer):
    return isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear))
def prune_model(model, amount):
    model_pruned = copy.deepcopy(model)

    prunable_layers = [layer for name, layer in model_pruned.named_modules() if is_prunable(layer)]

    parameters_to_prune = []
    for layer in prunable_layers:
        parameters_to_prune.append((layer, "weight"))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    for layer in prunable_layers:
        torch.nn.utils.prune.remove(layer, "weight")

    return model_pruned