import torch

from model import (
    HashGridEncoding,
    NIFModel,
    calibrate_nvfp4_activation_scales,
    load_model_checkpoint,
    model_from_config,
)
from fp8 import fake_fp8_linear, fake_quantize_e4m3, fake_quantize_fp16
from nvfp4 import export_nvfp4, fake_quantize_nvfp4, pack_e2m1, quantize_nvfp4


def test_fp8_fake_quantization_matches_e4m3_and_preserves_gradients():
    values = torch.tensor([[0.1, 1.1, 448.0]], requires_grad=True)
    quantized = fake_quantize_e4m3(values)

    torch.testing.assert_close(
        quantized, values.detach().to(torch.float8_e4m3fn).float(), rtol=0, atol=0
    )
    quantized.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))


def test_fp8_linear_rounds_operands_to_e4m3_and_output_to_fp16():
    inputs = torch.tensor([[1.1, 0.1]])
    weight = torch.tensor([[1.1, 0.1]])
    bias = torch.tensor([0.0001])

    expected = torch.nn.functional.linear(
        inputs.to(torch.float8_e4m3fn).float(),
        weight.half().to(torch.float8_e4m3fn).float(),
        bias.half().float(),
    ).half().float()
    torch.testing.assert_close(
        fake_fp8_linear(inputs, weight, bias), expected, rtol=0, atol=0
    )


def test_fp16_hash_fake_quantization_preserves_gradients():
    values = torch.tensor([0.1, 1.1], requires_grad=True)
    quantized = fake_quantize_fp16(values)

    torch.testing.assert_close(quantized, values.detach().half().float(), rtol=0, atol=0)
    quantized.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))


def test_fp8_checkpoint_restores_quantized_forward(tmp_path):
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
        "quantization": {"mlp": "fp8-e4m3", "hash": "fp16"},
    }
    network = model_from_config(config)
    path = tmp_path / "model.pt"
    torch.save({"model_config": config, "model_state_dict": network.state_dict()}, path)

    restored, _ = load_model_checkpoint(path)
    assert restored.fp8_qat
    assert not restored.nvfp4_qat
    assert restored.encoding.fp16_qat
    assert restored(torch.rand(3, 2)).shape == (3, 3)


def test_fp8_colour_conversion_uses_fixed_fp16_matrix():
    colour_matrix = torch.tensor(
        [[1.0, 2.032, 0.0], [1.0, -0.395, -0.581], [1.0, 0.0, 1.140]]
    ).numpy()
    network = NIFModel(
        input_dim=2,
        layer_size=16,
        num_layers=2,
        color_matrix=colour_matrix,
        encoding=HashGridEncoding(n_levels=2, n_features=2, log2_hashmap_size=4),
    )
    inputs = torch.rand(3, 2)
    network.set_fp8_qat(True)
    without_matrix = network.color_matrix
    network.color_matrix = None
    encoded = network(inputs)
    network.color_matrix = without_matrix

    expected = torch.nn.functional.linear(
        encoded, torch.from_numpy(colour_matrix).half().float()
    ).half().float()
    torch.testing.assert_close(network(inputs), expected, rtol=0, atol=0)


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


def test_nvfp4_can_reuse_a_fixed_activation_global_scale():
    values = torch.tensor([[0.5] * 16])
    scale = torch.tensor(6.0 / (448.0 * 6.0))
    quantized = quantize_nvfp4(values, (1, 16), scale)
    with_outlier = quantize_nvfp4(
        torch.cat((values, torch.full_like(values, 100.0))), (1, 16), scale
    )

    assert quantized.global_scale == scale
    torch.testing.assert_close(with_outlier.dequantized[:1], quantized.dequantized)


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

    assert checkpoint["format"] == "nvfp4-hidden-fp8-output"
    assert checkpoint["version"] == 2
    assert checkpoint["scale_layout"] == "logical-row-major-unswizzled"
    assert checkpoint["activation_global_scales_fp32"].tolist() == [1.0]
    assert checkpoint["layers"]["0"]["logical_shape"] == (16, 4)
    assert checkpoint["layers"]["0"]["padded_shape"] == (16, 16)
    assert checkpoint["layers"]["0"]["values"].numel() == 128
    assert "1" not in checkpoint["layers"]
    assert checkpoint["output_layer"]["format"] == "fp8-e4m3fn"
    assert checkpoint["output_layer"]["logical_shape"] == (3, 16)
    assert checkpoint["output_layer"]["weight"].numel() == 48


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
        "quantization": {
            "mlp": "nvfp4",
            "activation_global_scales": [0.001, 0.002],
        },
    }
    model = model_from_config(config)
    path = tmp_path / "model.pt"
    torch.save({"model_config": config, "model_state_dict": model.state_dict()}, path)

    restored, checkpoint = load_model_checkpoint(path)
    assert checkpoint["model_config"]["quantization"]["mlp"] == "nvfp4"
    assert restored.nvfp4_qat
    assert restored.nvfp4_static_activation_scales
    torch.testing.assert_close(
        restored.nvfp4_activation_global_scales,
        torch.tensor([0.001, 0.002]),
    )
    assert restored(torch.rand(3, 2)).shape == (3, 3)


def test_activation_scale_calibration_covers_each_mlp_input():
    model = NIFModel(
        input_dim=2,
        layer_size=16,
        num_layers=2,
        encoding=HashGridEncoding(n_levels=2, n_features=2, log2_hashmap_size=4),
    )
    scales = calibrate_nvfp4_activation_scales(model, torch.rand(32, 2), batch_size=7)

    assert scales.shape == (2,)
    assert torch.isfinite(scales).all()
    assert torch.all(scales > 0)
