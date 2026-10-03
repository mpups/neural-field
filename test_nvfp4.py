import torch

from model import HashGridEncoding, NIFModel, load_model_checkpoint, model_from_config
from nvfp4 import export_nvfp4, fake_quantize_nvfp4, pack_e2m1, quantize_nvfp4


def test_nvfp4_uses_e2m1_values_and_two_level_scaling():
    values = torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]])
    quantized = quantize_nvfp4(values, (1, 16))
    normalized = quantized.dequantized / (
        quantized.block_scales.float()[0, 0] * quantized.global_scale
    )

    torch.testing.assert_close(normalized, values, rtol=0, atol=1e-6)
    assert quantized.padded_shape == (1, 16)
    assert quantized.codes[0, :8].tolist() == list(range(8))


def test_nvfp4_weight_scaling_uses_independent_16_by_16_tiles():
    values = torch.ones((17, 17))
    values[16, 16] = 64.0
    quantized = quantize_nvfp4(values, (16, 16))

    assert quantized.block_scales.shape == (2, 2)
    assert quantized.padded_shape == (32, 32)
    assert quantized.block_scales.float()[0, 0] < quantized.block_scales.float()[1, 1]


def test_nvfp4_fake_quantization_preserves_gradients():
    values = torch.randn((3, 24), requires_grad=True)
    fake_quantize_nvfp4(values, (1, 16)).sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))


def test_e2m1_packing_places_first_code_in_low_nibble():
    packed = pack_e2m1(torch.tensor([[1, 2, 3]], dtype=torch.uint8))
    assert packed.tolist() == [0x21, 0x03]


def test_export_contains_packed_weights_and_scaling_contract(tmp_path):
    model = NIFModel(
        input_dim=2,
        layer_size=16,
        num_layers=2,
        encoding=HashGridEncoding(n_levels=2, n_features=2, log2_hashmap_size=4),
    )
    output = tmp_path / "nvfp4.pt"
    export_nvfp4(model, output)
    checkpoint = torch.load(output, weights_only=True)

    assert checkpoint["format"] == "nvfp4-e2m1"
    assert checkpoint["scale_layout"] == "logical-row-major-unswizzled"
    assert checkpoint["layers"]["0"]["logical_shape"] == (16, 4)
    assert checkpoint["layers"]["0"]["padded_shape"] == (16, 16)
    assert checkpoint["layers"]["0"]["values"].numel() == 128


def test_quantized_checkpoint_restores_fake_quantized_forward(tmp_path):
    config = {
        "input_dim": 2,
        "layer_size": 16,
        "num_layers": 2,
        "hashgrid_config": {
            "n_levels": 2,
            "n_features": 2,
            "log2_hashmap_size": 4,
            "base_resolution": 2,
            "max_resolution": 8,
        },
        "quantization": {"mlp": "nvfp4"},
    }
    model = model_from_config(config)
    path = tmp_path / "model.pt"
    torch.save({"model_config": config, "model_state_dict": model.state_dict()}, path)

    restored, checkpoint = load_model_checkpoint(path)
    assert checkpoint["model_config"]["quantization"]["mlp"] == "nvfp4"
    assert restored.nvfp4_qat
    assert restored(torch.rand(3, 2)).shape == (3, 3)
