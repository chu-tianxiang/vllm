# Adapted from
# https://github.com/huggingface/optimum/blob/main/optimum/gptq/quantizer.py
"""Utilities for quantizing models."""
from typing import List, Optional
from functools import partialmethod

from torch import nn
from transformers.pytorch_utils import Conv1D
from transformers.utils import (
    is_auto_gptq_available,
    is_optimum_available,
)

from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size, )
from vllm.model_executor.parallel_utils.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from vllm.model_executor.parallel_utils.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.parallel_utils.tensor_parallel.utils import divide

if is_optimum_available() and is_auto_gptq_available():
    from optimum.gptq import GPTQQuantizer
    from optimum.gptq.utils import (
        get_block_name_with_pattern,
        get_layers,
    )
    from auto_gptq.modeling._utils import autogptq_post_init
    from auto_gptq.utils.import_utils import dynamically_import_QuantLinear
else:
    GPTQQuantizer = object


class TpGPTQQuantizer(GPTQQuantizer):
    """
    Simple GPTQ Quantization with added support of vllm tensor-parallel layer.
    """

    def convert_model(self, model: nn.Module):
        """
        Convert the model to a GPTQ model by getting and replacing the layers.

        Args:
            model (`nn.Module`):
                Model to be converted

        """
        if self.block_name_to_quantize is None:
            self.block_name_to_quantize = get_block_name_with_pattern(model)
        block_name = self.block_name_to_quantize
        layers_to_be_replaced = get_layers(model,
                                           layers=[
                                               Conv1D, nn.Conv2d, nn.Linear,
                                               ColumnParallelLinear,
                                               RowParallelLinear
                                           ],
                                           prefix=block_name)
        self._replace_by_quant_layers(model, layers_to_be_replaced)
        return model

    def _replace_by_quant_layers(self,
                                 module: nn.Module,
                                 names: List[str],
                                 name: str = ""):
        """
        Replaces linear layers in `module` by `QuantLinear`

        Args:
            module (`nn.Module`):
                Module to quantize
            names (`List[str]`):
                List of names of the module to quantize
            name (`str`, defaults to `""`):
                To keep track of the name of the current module
        """
        QuantLinear = dynamically_import_QuantLinear(
            use_triton=False,
            desc_act=self.desc_act,
            group_size=self.group_size,
            bits=self.bits,
            disable_exllama=self.disable_exllama,
        )

        class ColumnParallelQuantLinear(QuantLinear):

            def __init__(self,
                         bits,
                         group_size,
                         infeatures,
                         outfeatures,
                         bias,
                         gather_output=True,
                         **kwargs):
                self.gather_output = gather_output
                world_size = get_tensor_model_parallel_world_size()
                super().__init__(bits, group_size, infeatures,
                                 divide(outfeatures, world_size), bias,
                                 **kwargs)

            def forward(self, input_):
                output_parallel = super().forward(input_).to(input_.dtype)
                if self.gather_output:
                    # All-gather across the partitions.
                    output = gather_from_tensor_model_parallel_region(
                        output_parallel)
                else:
                    output = output_parallel
                return output, None

        class RowParallelQuantLinear(QuantLinear):

            def __init__(self,
                         bits,
                         group_size,
                         infeatures,
                         outfeatures,
                         bias,
                         input_is_parallel=False,
                         reduce_results=True,
                         **kwargs):
                self.input_is_parallel = input_is_parallel
                self.reduce_results = reduce_results
                self.world_size = get_tensor_model_parallel_world_size()
                super().__init__(bits, group_size,
                                 divide(infeatures, self.world_size),
                                 outfeatures, bias, **kwargs)

            def forward(self, input_):
                if self.input_is_parallel:
                    input_parallel = input_
                else:
                    input_parallel = scatter_to_tensor_model_parallel_region(
                        input_)
                output_parallel = super().forward(input_parallel).to(
                    input_.dtype)
                if self.reduce_results and self.world_size > 1:
                    output = reduce_from_tensor_model_parallel_region(
                        output_parallel)
                else:
                    output = output_parallel
                return output, None

        class GatherQuantLinear(QuantLinear):

            def __init__(self,
                         bits,
                         group_size,
                         infeatures,
                         outfeatures,
                         bias,
                         input_is_parallel=False,
                         **kwargs):
                self.input_is_parallel = input_is_parallel
                super().__init__(bits, group_size, infeatures, outfeatures,
                                 bias, **kwargs)

            def forward(self, input_):
                # All-gather across the partitions.
                if self.input_is_parallel:
                    input_ = gather_from_tensor_model_parallel_region(input_)
                output = super().forward(input_).to(input_.dtype)
                return output, None

        if isinstance(module, QuantLinear):
            return
        for attr in dir(module):
            tmp = getattr(module, attr)
            name1 = name + "." + attr if name != "" else attr
            if name1 in names:
                delattr(module, attr)
                quant_class = QuantLinear
                kwargs = {}
                if isinstance(tmp, nn.Linear):
                    in_features = tmp.in_features
                    out_features = tmp.out_features
                elif isinstance(tmp, nn.Conv2d):
                    in_features = tmp.in_channels
                    out_features = tmp.out_channels
                elif isinstance(tmp, Conv1D):
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
                    # Quant linear with group_size and desc_act cannot be
                    # splitted. So we simply gather the input and calculate
                    # without tensor parallel.
                    if not self.desc_act or self.group_size == -1:
                        quant_class = RowParallelQuantLinear
                        kwargs.update({
                            "input_is_parallel": tmp.input_is_parallel,
                            "reduce_results": tmp.reduce_results
                        })
                    else:
                        quant_class = GatherQuantLinear
                        kwargs.update(
                            {"input_is_parallel": tmp.input_is_parallel})
                if not (self.desc_act) or self.group_size == -1:
                    kwargs["use_cuda_fp16"] = self.use_cuda_fp16
                new_layer = quant_class(self.bits, self.group_size,
                                        in_features, out_features, True,
                                        **kwargs)
                setattr(module, attr, new_layer)
        for name1, child in module.named_children():
            self._replace_by_quant_layers(
                child, names, name + "." + name1 if name != "" else name1)

    def post_init_model(self, model, max_input_length: Optional[int] = None):
        """
        Post-initialization that require device information, for example buffers
        initialization on device.

        Args:
            model (`nn.Module`):
                The input model
            max_input_length (`Optional[int]`, optional):
                Maximum token number used during inference. Defaults to `None`.
        """
        return autogptq_post_init(model,
                                  use_act_order=self.desc_act,
                                  max_input_length=max_input_length)


def patch_tp_linear_layer():
    """ Patch linear layer to use cpu initialization."""
    ColumnParallelLinear.__init__ = partialmethod(
        ColumnParallelLinear.__init__, use_cpu_initialization=True)
    RowParallelLinear.__init__ = partialmethod(RowParallelLinear.__init__,
                                               use_cpu_initialization=True)
