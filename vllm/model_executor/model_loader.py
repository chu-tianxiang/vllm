"""Utilities for selecting and loading models."""
from typing import Type

import torch
import torch.nn as nn
from accelerate import init_on_device
from transformers import PretrainedConfig
from auto_gptq.modeling._utils import autogptq_post_init
from auto_gptq import exllama_set_max_input_length

from vllm.config import ModelConfig
from vllm.model_executor.models import *  # pylint: disable=wildcard-import
from vllm.model_executor.weight_utils import initialize_dummy_weights
from vllm.model_executor.quantize import make_quant, find_layers

# TODO(woosuk): Lazy-load the model classes.
_MODEL_REGISTRY = {
    "AquilaModel": AquilaForCausalLM,
    "BaiChuanForCausalLM": BaiChuanForCausalLM,  # baichuan-7b
    "BaichuanForCausalLM": BaichuanForCausalLM,  # baichuan-13b
    "BloomForCausalLM": BloomForCausalLM,
    "FalconForCausalLM": FalconForCausalLM,
    "GPT2LMHeadModel": GPT2LMHeadModel,
    "GPTBigCodeForCausalLM": GPTBigCodeForCausalLM,
    "GPTJForCausalLM": GPTJForCausalLM,
    "GPTNeoXForCausalLM": GPTNeoXForCausalLM,
    "InternLMForCausalLM": InternLMForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "LLaMAForCausalLM": LlamaForCausalLM,  # For decapoda-research/llama-*
    "MPTForCausalLM": MPTForCausalLM,
    "OPTForCausalLM": OPTForCausalLM,
    "QWenLMHeadModel": QWenLMHeadModel,
    "RWForCausalLM": FalconForCausalLM,
}


def _get_model_architecture(config: PretrainedConfig) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[arch]
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_MODEL_REGISTRY.keys())}")


def get_model(model_config: ModelConfig, max_tokens: int) -> nn.Module:
    model_class = _get_model_architecture(model_config.hf_config)
    torch.set_default_dtype(model_config.dtype)

    # Create a model instance.
    # The weights will be initialized as empty tensors.
    if model_config.quantize_config:
        with init_on_device(device=torch.device("cpu")):
            model = model_class(model_config.hf_config)
        layers = find_layers(model)
        ignore_layers = [model_class.lm_head_name] + model_class.outside_layer_modules
        for name in list(layers.keys()):
            if any([name.startswith(ignore_layer) for ignore_layer in ignore_layers]):
                del layers[name]

        make_quant(
            model,
            layers,
            model_config.quantize_config.bits,
            model_config.quantize_config.group_size,
            desc_act=model_config.quantize_config.desc_act,
        )
    else:
        model = model_class(model_config.hf_config)
    model.quantize_config = model_config.quantize_config
    if model_config.use_dummy_weights:
        model = model.cuda()
        # NOTE(woosuk): For accurate performance evaluation, we assign
        # random values to the weights.
        initialize_dummy_weights(model)
    else:
        # Load the weights from the cached or downloaded files.
        model.load_weights(model_config.model, model_config.download_dir,
                           model_config.use_np_weights)
        model = model.cuda()
    if model_config.quantize_config:
        model = autogptq_post_init(model, use_act_order=model_config.quantize_config.desc_act)
        if model_config.quantize_config.desc_act:
            model = exllama_set_max_input_length(model, max_tokens)
    return model.eval()
