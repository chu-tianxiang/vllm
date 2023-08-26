# Adapted from
# https://github.com/PanQiWei/AutoGPTQ/blob/main/auto_gptq/modeling/_utils.py
"""Utilities for quantizing models."""
from typing import List, Dict

import torch.nn as nn
import transformers

from auto_gptq.utils.import_utils import dynamically_import_QuantLinear

from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.parallel_utils.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
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

    class ColumnParallelQuantLinear(QuantLinear):
        def __init__(
            self,
            bits,
            group_size,
            infeatures,
            outfeatures,
            bias,
            gather_output=True,
            **kwargs
        ):
            self.gather_output = gather_output
            world_size = get_tensor_model_parallel_world_size()
            super().__init__(bits, group_size, infeatures,
                             outfeatures // world_size, bias, **kwargs)

        def forward(self, input_):
            output_parallel = super().forward(input_)
            if self.gather_output:
                # All-gather across the partitions.
                output = gather_from_tensor_model_parallel_region(output_parallel)
            else:
                output = output_parallel
            return output, None

    class RowParallelQuantLinear(QuantLinear):
        def __init__(
            self,
            bits,
            group_size,
            infeatures,
            outfeatures,
            bias,
            input_is_parallel=False,
            reduce_results=True,
            **kwargs
        ):
            self.input_is_parallel = input_is_parallel
            self.reduce_results = reduce_results
            self.world_size = get_tensor_model_parallel_world_size()
            super().__init__(bits, group_size, infeatures // self.world_size,
                             outfeatures, bias, **kwargs)

        def forward(self, input_):
            if self.input_is_parallel:
                input_parallel = input_
            else:
                input_parallel = scatter_to_tensor_model_parallel_region(input_)
            output_parallel = super().forward(input_parallel)
            if self.reduce_results and self.world_size > 1:
                output = reduce_from_tensor_model_parallel_region(output_parallel)
            else:
                output = output_parallel
            return output, None


    class GatherQuantLinear(QuantLinear):
        def __init__(
            self,
            bits,
            group_size,
            infeatures,
            outfeatures,
            bias,
            input_is_parallel=False,
            **kwargs
        ):
            self.input_is_parallel = input_is_parallel
            super().__init__(bits, group_size, infeatures, outfeatures, bias, **kwargs)

        def forward(self, input_):
            # All-gather across the partitions.
            if self.input_is_parallel:
                input_ = gather_from_tensor_model_parallel_region(input_)
            output = super().forward(input_)
            return output, None

    if isinstance(module, QuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            delattr(module, attr)
            quant_class = QuantLinear
            kwargs = {"trainable": trainable}
            if isinstance(tmp, nn.Linear):
                in_features = tmp.in_features
                out_features = tmp.out_features
            elif isinstance(tmp, nn.Conv2d):
                in_features = tmp.in_channels
                out_features = tmp.out_channels
            elif isinstance(tmp, transformers.pytorch_utils.Conv1D):
                in_features = tmp.weight.shape[0]
                out_features = tmp.weight.shape[1]
            elif isinstance(tmp, ColumnParallelLinear):
                in_features = tmp.input_size
                out_features = tmp.output_size
                quant_class = ColumnParallelQuantLinear
                kwargs.update({"gather_output": tmp.gather_output})
            elif isinstance(tmp, RowParallelLinear):
                in_features = tmp.input_size
                out_features = tmp.output_size
                if not desc_act or group_size == -1:
                    quant_class = RowParallelQuantLinear
                    kwargs.update({"input_is_parallel": tmp.input_is_parallel,
                                   "reduce_results": tmp.reduce_results})
                else:
                    quant_class = GatherQuantLinear
                    kwargs.update({"input_is_parallel": tmp.input_is_parallel})
            if (not(desc_act) or group_size == -1) and not use_triton:
                kwargs["use_cuda_fp16"] = use_cuda_fp16
            new_layer = quant_class(bits, group_size, in_features, out_features, True,
                                    **kwargs)
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
