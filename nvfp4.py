"""NVFP4 fake quantization and logical checkpoint export.

The exported scales are row-major, not backend-swizzled. A CUDA deployment
backend must transform them to the layout required by its matrix API.
"""

from dataclasses import dataclass
from pathlib import Path

import torch


E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
E4M3_MAX = 448.0
E2M1_MAX = 6.0


@dataclass
class NVFP4Tensor:
    dequantized: torch.Tensor
    codes: torch.Tensor
    block_scales: torch.Tensor
    global_scale: torch.Tensor
    logical_shape: tuple[int, int]
    padded_shape: tuple[int, int]
    block_shape: tuple[int, int]


def _round_e4m3(values: torch.Tensor) -> torch.Tensor:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("NVFP4 simulation requires PyTorch float8_e4m3fn support")
    return values.to(torch.float8_e4m3fn).to(torch.float32)


def quantize_nvfp4(tensor: torch.Tensor, block_shape: tuple[int, int]) -> NVFP4Tensor:
    """Quantize a matrix using logical NVFP4 two-level block scaling."""
    if tensor.dim() != 2:
        raise ValueError("NVFP4 quantization expects a matrix")
    block_rows, block_columns = block_shape
    if block_rows < 1 or block_columns < 1:
        raise ValueError("NVFP4 block dimensions must be positive")

    rows, columns = tensor.shape
    padded_rows = ((rows + block_rows - 1) // block_rows) * block_rows
    padded_columns = ((columns + block_columns - 1) // block_columns) * block_columns
    values = torch.zeros(
        (padded_rows, padded_columns), device=tensor.device, dtype=torch.float32
    )
    values[:rows, :columns] = tensor.detach().to(torch.float32)
    tiles = values.view(
        padded_rows // block_rows, block_rows,
        padded_columns // block_columns, block_columns,
    ).permute(0, 2, 1, 3)

    global_amax = values.abs().amax()
    global_scale = torch.where(
        global_amax > 0,
        global_amax / (E4M3_MAX * E2M1_MAX),
        torch.ones_like(global_amax),
    )
    block_amax = tiles.abs().amax(dim=(-2, -1))
    block_scales = _round_e4m3((block_amax / E2M1_MAX) / global_scale)
    effective_scales = block_scales * global_scale
    normalized = torch.where(
        effective_scales[..., None, None] > 0,
        tiles / effective_scales[..., None, None],
        torch.zeros_like(tiles),
    )

    boundaries = normalized.new_tensor(E2M1_MIDPOINTS)
    magnitudes = normalized.new_tensor(E2M1_MAGNITUDES)
    magnitude_codes = torch.bucketize(
        normalized.abs().contiguous(), boundaries
    ).to(torch.uint8)
    signs = torch.signbit(normalized).to(torch.uint8)
    codes = magnitude_codes | (signs << 3)
    quantized = magnitudes[magnitude_codes.long()]
    quantized = torch.where(signs.bool(), -quantized, quantized)
    dequantized_tiles = quantized * effective_scales[..., None, None]
    dequantized = dequantized_tiles.permute(0, 2, 1, 3).reshape(
        padded_rows, padded_columns
    )[:rows, :columns]
    code_matrix = codes.permute(0, 2, 1, 3).reshape(padded_rows, padded_columns)

    return NVFP4Tensor(
        dequantized=dequantized.to(tensor.dtype),
        codes=code_matrix,
        block_scales=block_scales.to(torch.float8_e4m3fn),
        global_scale=global_scale,
        logical_shape=(rows, columns),
        padded_shape=(padded_rows, padded_columns),
        block_shape=block_shape,
    )


def fake_quantize_nvfp4(tensor: torch.Tensor, block_shape: tuple[int, int]) -> torch.Tensor:
    """Apply NVFP4 in the forward pass and STE identity in the backward pass."""
    matrix = tensor.reshape(-1, tensor.shape[-1])
    quantized = quantize_nvfp4(matrix, block_shape).dequantized.reshape_as(tensor)
    return tensor + (quantized - tensor).detach()


def pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    """Pack row-major E2M1 codes with the earlier element in the low nibble."""
    flat = codes.reshape(-1).to(torch.uint8)
    if flat.numel() % 2:
        flat = torch.cat((flat, torch.zeros(1, device=flat.device, dtype=torch.uint8)))
    return (flat[0::2] | (flat[1::2] << 4)).cpu()


def export_nvfp4(model, output: str | Path) -> None:
    """Export packed MLP weights and logical scales for a future CUDA backend."""
    layers = {}
    for index, layer in enumerate(model.layers):
        quantized = quantize_nvfp4(layer.weight, (16, 16))
        layers[str(index)] = {
            "values": pack_e2m1(quantized.codes),
            "block_scales_e4m3": quantized.block_scales.view(torch.uint8).cpu(),
            "global_scale_fp32": quantized.global_scale.cpu(),
            "logical_shape": quantized.logical_shape,
            "padded_shape": quantized.padded_shape,
            "block_shape": quantized.block_shape,
            "bias_fp16": None if layer.bias is None else layer.bias.detach().half().cpu(),
        }
    torch.save(
        {
            "format": "nvfp4-e2m1",
            "version": 1,
            "weight_scaling": "2d-16x16-e4m3-with-fp32-global",
            "activation_scaling": "1d-1x16-e4m3-with-fp32-global",
            "scale_layout": "logical-row-major-unswizzled",
            "layers": layers,
        },
        Path(output),
    )
