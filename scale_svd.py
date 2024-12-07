from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Union, Dict, Callable
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM
import logging
import math
import copy

# Custom exception for scaling errors
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

class ScalingMethod(Enum):
    INTERPOLATION = auto()  # Original bilinear interpolation
    QUADRANT = auto()      # Quadrant-based copying
    SVD = auto()           # SVD-based expansion
    HYBRID = auto()        # Hybrid SVD + interpolation

@dataclass
class ScalingConfig:
    method: ScalingMethod
    scale_factor: float = 1.2
    svd_ratio: float = 0.7  # Only used for HYBRID method

def expand_weight_matrix_interpolation(
    base_weight: torch.Tensor,
    out_features: int,
    in_features: int
) -> torch.Tensor:
    """Original bilinear interpolation method"""
    expanded_weight = torch.nn.functional.interpolate(
        base_weight.unsqueeze(0).unsqueeze(0),
        size=(out_features, in_features),
        mode='bilinear',
        align_corners=True
    ).squeeze(0).squeeze(0) * math.sqrt(base_weight.size(0) / out_features)

    return expanded_weight

def expand_weight_matrix_quadrant(
    base_weight: torch.Tensor,
    out_features: int,
    in_features: int
) -> torch.Tensor:
    """Quadrant-based copying method"""
    base_out, base_in = base_weight.shape

    expanded_weight = torch.zeros(
        (out_features, in_features),
        device=base_weight.device,
        dtype=base_weight.dtype
    )

    # Copy original weights to top-left quadrant
    expanded_weight[:base_out, :base_in] = base_weight

    # Copy right side
    if in_features > base_in:
        expanded_weight[:base_out, base_in:] = base_weight[:, -(in_features-base_in):]

    # Copy bottom side
    if out_features > base_out:
        expanded_weight[base_out:, :base_in] = base_weight[-(out_features-base_out):, :]

    # Copy bottom-right corner
    if out_features > base_out and in_features > base_in:
        expanded_weight[base_out:, base_in:] = base_weight[
            -(out_features-base_out):,
            -(in_features-base_in):
        ]

    scale_factor = math.sqrt(base_out / out_features)
    expanded_weight *= scale_factor

    return expanded_weight

def expand_weight_matrix_svd(
    base_weight: torch.Tensor,
    out_features: int,
    in_features: int
) -> torch.Tensor:
    """
    SVD-based expansion method with proper rank handling
    """
    # Perform SVD
    U, S, V = torch.svd(base_weight)

    # Determine rank to preserve (minimum of dimensions)
    rank = min(U.shape[1], V.shape[0])

    # Extract core matrices with proper dimensions
    U_core = U[:, :rank]
    S_core = S[:rank]
    V_core = V[:, :rank]

    # Expand U matrix (left singular vectors)
    expanded_U = torch.nn.functional.interpolate(
        U_core.unsqueeze(0).unsqueeze(0),
        size=(out_features, rank),
        mode='bilinear',
        align_corners=True
    ).squeeze(0).squeeze(0)

    # Expand V matrix (right singular vectors)
    expanded_V = torch.nn.functional.interpolate(
        V_core.unsqueeze(0).unsqueeze(0),
        size=(in_features, rank),
        mode='bilinear',
        align_corners=True
    ).squeeze(0).squeeze(0)

    # Scale singular values
    out_ratio = out_features / base_weight.shape[0]
    in_ratio = in_features / base_weight.shape[1]
    scale_factor = math.sqrt(1.0 / (out_ratio * in_ratio))
    scaled_S = S_core * scale_factor

    # Reconstruct expanded matrix
    expanded_weight = expanded_U @ torch.diag(scaled_S) @ expanded_V.T

    return expanded_weight

def expand_weight_matrix_hybrid(
    base_weight: torch.Tensor,
    out_features: int,
    in_features: int,
    svd_ratio: float = 0.7
) -> torch.Tensor:
    """
    Hybrid SVD + interpolation method with proper dimension handling
    """
    # Get SVD-based expansion
    svd_weight = expand_weight_matrix_svd(base_weight, out_features, in_features)

    # Get interpolation-based expansion
    interp_weight = torch.nn.functional.interpolate(
        base_weight.unsqueeze(0).unsqueeze(0),
        size=(out_features, in_features),
        mode='bilinear',
        align_corners=True
    ).squeeze(0).squeeze(0)

    # Blend the two approaches
    expanded_weight = svd_ratio * svd_weight + (1 - svd_ratio) * interp_weight

    # Scale to maintain overall magnitude
    scale_factor = math.sqrt(base_weight.shape[0] / out_features)
    expanded_weight *= scale_factor

    return expanded_weight

def expand_weight_matrix_hybrid(
    base_weight: torch.Tensor,
    out_features: int,
    in_features: int,
    svd_ratio: float = 0.7
) -> torch.Tensor:
    """Hybrid SVD + interpolation method"""
    svd_weight = expand_weight_matrix_svd(base_weight, out_features, in_features)

    interp_weight = torch.nn.functional.interpolate(
        base_weight.unsqueeze(0).unsqueeze(0),
        size=(out_features, in_features),
        mode='bilinear',
        align_corners=True
    ).squeeze(0).squeeze(0)

    expanded_weight = svd_ratio * svd_weight + (1 - svd_ratio) * interp_weight
    scale_factor = math.sqrt(base_weight.shape[0] / out_features)
    expanded_weight *= scale_factor

    return expanded_weight

def get_expansion_function(method: ScalingMethod, svd_ratio: float = 0.7) -> Callable:
    """Get the appropriate expansion function based on scaling method"""
    if method == ScalingMethod.INTERPOLATION:
        return expand_weight_matrix_interpolation
    elif method == ScalingMethod.QUADRANT:
        return expand_weight_matrix_quadrant
    elif method == ScalingMethod.SVD:
        return expand_weight_matrix_svd
    elif method == ScalingMethod.HYBRID:
        return lambda base_weight, out_features, in_features: expand_weight_matrix_hybrid(
            base_weight, out_features, in_features, svd_ratio
        )
    else:
        raise ValueError(f"Unknown scaling method: {method}")

def create_scaled_model(
    base_model_path: str,
    scaling_config: ScalingConfig,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None
) -> LlamaForCausalLM:
    """Create a width-scaled model using the specified scaling method"""
    try:
        logging.info(f"Loading base model from {base_model_path}")
        logging.info(f"Using scaling method: {scaling_config.method.name}")

        base_model = LlamaForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
        )

        scaled_config = create_scaled_config(base_model.config, scaling_config.scale_factor)
        scaled_model = LlamaForCausalLM(scaled_config)

        # Get appropriate expansion function
        expand_fn = get_expansion_function(scaling_config.method, scaling_config.svd_ratio)

        # Initialize parameters using the selected method
        initialize_scaled_parameters(scaled_model, base_model, expand_fn)

        if device:
            scaled_model = scaled_model.to(device)

        return scaled_model

    except Exception as e:
        raise ModelScalingError(f"Error during model scaling: {str(e)}")

def initialize_scaled_parameters(
    scaled_model: LlamaForCausalLM,
    base_model: LlamaForCausalLM,
    expand_fn: Callable
):
    """Initialize parameters using the specified expansion function"""
    hidden_size = scaled_model.config.hidden_size

    # Embedding layer
    scaled_model.model.embed_tokens.weight.data = expand_fn(
        base_model.model.embed_tokens.weight.data,
        base_model.config.vocab_size,
        hidden_size
    )

    # Initialize each layer
    for scaled_layer, base_layer in zip(scaled_model.model.layers, base_model.model.layers):
        # Attention weights
        for name in ['q_proj', 'k_proj', 'v_proj']:
            base_weight = getattr(base_layer.self_attn, name).weight.data
            getattr(scaled_layer.self_attn, name).weight.data = expand_fn(
                base_weight, hidden_size, hidden_size
            )

        # Output projection
        base_weight = base_layer.self_attn.o_proj.weight.data
        scaled_layer.self_attn.o_proj.weight.data = expand_fn(
            base_weight, hidden_size, hidden_size
        )

        # MLP weights
        gate_weight = base_layer.mlp.gate_proj.weight.data
        scaled_layer.mlp.gate_proj.weight.data = expand_fn(
            gate_weight, scaled_model.config.intermediate_size, hidden_size
        )

        up_weight = base_layer.mlp.up_proj.weight.data
        scaled_layer.mlp.up_proj.weight.data = expand_fn(
            up_weight, scaled_model.config.intermediate_size, hidden_size
        )

        down_weight = base_layer.mlp.down_proj.weight.data
        scaled_layer.mlp.down_proj.weight.data = expand_fn(
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
    scaled_model.lm_head.weight.data = expand_fn(
        base_model.lm_head.weight.data,
        base_model.config.vocab_size,
        hidden_size
    )


config = ScalingConfig(
    method=ScalingMethod.SVD,
    scale_factor=1.1
)
model_svd = create_scaled_model("llm-jp/llm-jp-3-1.8b", config)
