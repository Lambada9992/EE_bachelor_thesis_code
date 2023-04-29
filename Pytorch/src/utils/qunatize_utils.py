from enum import Enum
import copy
import torch
from torch.ao.quantization.qconfig import float16_static_qconfig
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization import get_default_qconfig_mapping
import itertools


class QuantizeMode(Enum):
    NONE = 0
    FP16 = 1
    DYNAMIC_INT8 = 2
    STATIC_INT8 = 3


def qunatize(model, mode, train_dataloader = None, example_input = torch.rand(1, 3, 224, 224)):
    if mode == QuantizeMode.FP16:
        return quantize_fp16(model, example_input)
    elif mode == QuantizeMode.DYNAMIC_INT8:
        return quantize_dynamic_int8(model, example_input)
    elif mode == QuantizeMode.STATIC_INT8 and train_dataloader is not None:
        return quantize_static_int8(model, train_dataloader, example_input)
    elif mode == QuantizeMode.NONE:
        return model
    else:
        raise ValueError(f"Unsupported quantize mode: {mode} or train_dataloader is None")


def quantize_fp16(model, example_input = torch.rand(1, 3, 224, 224)):
    model_fp16 = copy.deepcopy(model)
    model_fp16 = quantize_fx.prepare_fx(model_fp16, {"": float16_static_qconfig}, example_input)
    model_fp16 = quantize_fx.convert_to_reference_fx(model_fp16)
    return model_fp16

def quantize_dynamic_int8(model, example_input = torch.rand(1, 3, 224, 224)):
    model_dint8 = copy.deepcopy(model)
    model_dint8.eval()
    qconfig_mapping = QConfigMapping().set_global(torch.ao.quantization.default_dynamic_qconfig)
    model_dint8 = quantize_fx.prepare_fx(model_dint8, qconfig_mapping, example_input)
    model_dint8 = quantize_fx.convert_to_reference_fx(model_dint8)
    return model_dint8

def quantize_static_int8(model, train_dataloader, example_input = torch.rand(1, 3, 224, 224)):
    model_sint8 = copy.deepcopy(model)
    model_sint8.eval()
    qconfig_mapping = get_default_qconfig_mapping("qnnpack")

    model_sint8 = quantize_fx.prepare_fx(model_sint8, qconfig_mapping, example_input)

    # Calibrating model
    for (image, label) in itertools.islice(train_dataloader, 200):
        model_sint8(image)

    model_sint8 = quantize_fx.convert_to_reference_fx(model_sint8)

    return model_sint8
