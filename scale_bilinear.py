from dataclasses import dataclass
from typing import Optional, Union, Dict
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaAttention,
    LlamaDecoderLayer
)
import copy
import logging
import math

class ModelScalingError(Exception):
    """Width scaling related errors"""
    pass

def create_scaled_config(
    base_config: LlamaConfig,
    scale_factor: float = 1.2
) -> LlamaConfig:
    """Create a new config with unified hidden size scaling"""
    config = copy.deepcopy(base_config)

    # Calculate new dimensions ensuring divisibility
    base_head_dim = base_config.hidden_size // base_config.num_attention_heads
    new_num_heads = math.ceil(config.num_attention_heads * scale_factor)

    # Set hidden size based on number of heads
    config.hidden_size = new_num_heads * base_head_dim
    config.num_attention_heads = new_num_heads

    if hasattr(config, 'num_key_value_heads'):
        config.num_key_value_heads = new_num_heads

    # Scale intermediate size proportionally
    config.intermediate_size = int(config.hidden_size * (base_config.intermediate_size / base_config.hidden_size))

    return config

def create_width_scaled_model(
    base_model_path: str,
    scale_factor: float = 1.2,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None
) -> LlamaForCausalLM:
    try:
        logging.info(f"Loading base model from {base_model_path}")
        base_model = LlamaForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
        )

        scaled_config = create_scaled_config(base_model.config, scale_factor)
        scaled_model = LlamaForCausalLM(scaled_config)
        initialize_scaled_parameters(scaled_model, base_model)

        if device:
            scaled_model = scaled_model.to(device)

        return scaled_model

    except Exception as e:
        raise ModelScalingError(f"Error during model scaling: {str(e)}")

def initialize_scaled_parameters(
    scaled_model: LlamaForCausalLM,
    base_model: LlamaForCausalLM
):
    """Initialize parameters with unified hidden size"""
    hidden_size = scaled_model.config.hidden_size

    # Embedding layer
    scaled_model.model.embed_tokens.weight.data = expand_weight_matrix(
        base_model.model.embed_tokens.weight.data,
        base_model.config.vocab_size,
        hidden_size
    )

    # Initialize each layer
    for scaled_layer, base_layer in zip(scaled_model.model.layers, base_model.model.layers):
        # Attention weights
        for name in ['q_proj', 'k_proj', 'v_proj']:
            base_weight = getattr(base_layer.self_attn, name).weight.data
            getattr(scaled_layer.self_attn, name).weight.data = expand_weight_matrix(
                base_weight, hidden_size, hidden_size
            )

        # Output projection
        base_weight = base_layer.self_attn.o_proj.weight.data
        scaled_layer.self_attn.o_proj.weight.data = expand_weight_matrix(
            base_weight, hidden_size, hidden_size
        )

        # MLP weights
        gate_weight = base_layer.mlp.gate_proj.weight.data
        scaled_layer.mlp.gate_proj.weight.data = expand_weight_matrix(
            gate_weight, scaled_model.config.intermediate_size, hidden_size
        )

        up_weight = base_layer.mlp.up_proj.weight.data
        scaled_layer.mlp.up_proj.weight.data = expand_weight_matrix(
            up_weight, scaled_model.config.intermediate_size, hidden_size
        )

        down_weight = base_layer.mlp.down_proj.weight.data
        scaled_layer.mlp.down_proj.weight.data = expand_weight_matrix(
            down_weight, hidden_size, scaled_model.config.intermediate_size
        )

        # Layer norms
        for name in ['input_layernorm', 'post_attention_layernorm']:
            base_norm = getattr(base_layer, name)
            scaled_norm = getattr(scaled_layer, name)
            scaled_norm.weight.data = torch.nn.functional.interpolate(
                base_norm.weight.data.view(1, 1, -1),
                size=hidden_size,
                mode='linear'
            ).view(-1)

    # Final layer norm
    scaled_model.model.norm.weight.data = torch.nn.functional.interpolate(
        base_model.model.norm.weight.data.view(1, 1, -1),
        size=hidden_size,
        mode='linear'
    ).view(-1)

    # LM head
    scaled_model.lm_head.weight.data = expand_weight_matrix(
        base_model.lm_head.weight.data,
        base_model.config.vocab_size,
        hidden_size
    )

def expand_weight_matrix(base_weight: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """Expand weight matrix with float scale factor"""
    out_features, in_features = base_weight.shape
    new_out_features = int(out_features * scale_factor)
    new_in_features = int(in_features * scale_factor)

    expanded_weight = torch.zeros(
        (new_out_features, new_in_features),
        device=base_weight.device,
        dtype=base_weight.dtype
    )

    # Linear interpolation for weight values
    scale = 1.0 / math.sqrt(scale_factor)
    expanded_weight = torch.nn.functional.interpolate(
        base_weight.unsqueeze(0).unsqueeze(0),
        size=(new_out_features, new_in_features),
        mode='bilinear',
        align_corners=True
    ).squeeze(0).squeeze(0) * scale

    return expanded_weight

def expand_weight_matrix(
    base_weight: torch.Tensor,
    out_features: int,
    in_features: int
) -> torch.Tensor:
    """Expand weight matrix to exact dimensions"""
    expanded_weight = torch.nn.functional.interpolate(
        base_weight.unsqueeze(0).unsqueeze(0),
        size=(out_features, in_features),
        mode='bilinear',
        align_corners=True
    ).squeeze(0).squeeze(0) * math.sqrt(base_weight.size(0) / out_features)

    return expanded_weight

def get_model_info(model: LlamaForCausalLM) -> Dict[str, int]:
    """Get information about the scaled model"""
    return {
        "hidden_size": model.config.hidden_size,
        "num_attention_heads": model.config.num_attention_heads,
        "intermediate_size": model.config.intermediate_size,
        "num_layers": model.config.num_hidden_layers,
        "vocab_size": model.config.vocab_size
    }
