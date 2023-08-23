# Adapted from
# https://github.com/PanQiWei/AutoGPTQ/blob/main/auto_gptq/modeling/_utils.py
"""Utilities for quantizing models."""
from typing import List, Dict

import torch.nn as nn
import transformers

from auto_gptq.utils.import_utils import dynamically_import_QuantLinear

from vllm.model_executor.parallel_utils.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)

def find_layers(
    module: nn.Module,
    layers: List[nn.Module] = None,
    name: str = ''
) -> Dict[str, nn.Module]:
    if not layers:
        layers = [transformers.pytorch_utils.Conv1D, nn.Conv2d, nn.Linear,
                  ColumnParallelLinear, RowParallelLinear]
    for layer in layers:
        if isinstance(module,layer):
            return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res


def make_quant(
    module: nn.Module,
    names: List[str],
    bits: int,
    group_size: int,
    name: str = '',
    use_triton: bool = False,
    disable_exllama: bool = False,
    use_cuda_fp16: bool = True,
    desc_act: bool = False,
    trainable: bool = False
) -> None:
    QuantLinear = dynamically_import_QuantLinear(use_triton=use_triton, desc_act=desc_act, group_size=group_size, bits=bits, disable_exllama=disable_exllama)

    class QuantLinearWrapper(QuantLinear):
        def forward(self, *args, **kwargs):
            return super().forward(*args, **kwargs), None

    if isinstance(module, QuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            delattr(module, attr)
            if isinstance(tmp, nn.Linear):
                in_features = tmp.in_features
                out_features = tmp.out_features
            elif isinstance(tmp, nn.Conv2d):
                in_features = tmp.in_channels
                out_features = tmp.out_channels
            elif isinstance(tmp, transformers.pytorch_utils.Conv1D):
                in_features = tmp.weight.shape[0]
                out_features = tmp.weight.shape[1]
            elif isinstance(tmp, ColumnParallelLinear) or isinstance(tmp, RowParallelLinear):
                in_features = tmp.input_size
                out_features = tmp.output_size
            if (not(desc_act) or group_size == -1) and not use_triton:
                new_layer = QuantLinearWrapper(
                    bits, group_size, in_features, out_features, True, use_cuda_fp16=use_cuda_fp16, trainable=trainable
                )
            else:
                new_layer = QuantLinearWrapper(bits, group_size, in_features, out_features, True, trainable=trainable)
            setattr(module, attr, new_layer)
    for name1, child in module.named_children():
        make_quant(
            child,
            names,
            bits,
            group_size,
            name + '.' + name1 if name != '' else name1,
            use_triton=use_triton,
            use_cuda_fp16=use_cuda_fp16,
            desc_act=desc_act,
            trainable=trainable,
            disable_exllama=disable_exllama,
        )
