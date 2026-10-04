"""Unscaled E4M3 quantization matching the OptiX cooperative-vector path."""

import torch
import torch.nn.functional as F


def fake_quantize_fp16(tensor: torch.Tensor) -> torch.Tensor:
    """Round through FP16 in the forward pass with an STE backward."""
    quantized = tensor.to(torch.float16).to(tensor.dtype)
    return tensor + (quantized - tensor).detach()


def fake_quantize_e4m3(tensor: torch.Tensor) -> torch.Tensor:
    """Round through finite E4M3 in the forward pass with an STE backward."""
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("FP8 simulation requires PyTorch float8_e4m3fn support")
    quantized = tensor.to(torch.float8_e4m3fn).to(tensor.dtype)
    return tensor + (quantized - tensor).detach()


def fake_fp8_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Simulate the renderer's unscaled E4M3 operands and FP16 output."""
    output = F.linear(
        fake_quantize_e4m3(inputs),
        fake_quantize_e4m3(fake_quantize_fp16(weight)),
        None if bias is None else bias.to(torch.float16).to(inputs.dtype),
    )
    return output.to(torch.float16).to(output.dtype)
