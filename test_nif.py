import numpy as np
import torch

import nif


def test_log1p_transfer_round_trips_hdr_samples():
    image = np.array([[[0.0, 0.1, 32.0], [1.0, 4.0, 16.0]]], dtype=np.float32)
    encoded, params = nif.prepare_image_for_sampling(image, "log1p", torch.device("cpu"))
    pixels = np.array([[0, 0], [0, 1]])

    decoded = nif.decode_samples(image.shape, pixels, encoded.reshape(-1, 3), params)

    assert params["transfer_function"] == "log1p"
    np.testing.assert_allclose(decoded, image, rtol=1e-6, atol=1e-6)


def test_training_decode_round_trips_log_hdr_samples_with_gradients():
    image = np.array([[[0.01, 0.1, 32.0], [1.0, 4.0, 16.0]]], dtype=np.float32)
    encoded, params = nif.prepare_image_for_sampling(image, "log", torch.device("cpu"))
    encoded = encoded.reshape(-1, 3).requires_grad_()

    decoded = nif.decode_training_values(encoded, params)

    torch.testing.assert_close(decoded, torch.from_numpy(image).reshape(-1, 3))
    decoded.sum().backward()
    assert torch.isfinite(encoded.grad).all()
    assert torch.all(encoded.grad > 0)


def test_highlight_linear_loss_ignores_dark_samples_and_has_finite_gradients():
    params = {
        "mean": [0.0, 0.0, 0.0],
        "max": 1.0,
        "transfer_function": "linear",
        "eps": 1e-8,
    }
    targets = torch.tensor([[0.1, 0.1, 0.1], [2.0, 2.0, 2.0]])
    outputs = torch.tensor([[9.0, 9.0, 9.0], [1.0, 1.0, 1.0]], requires_grad=True)

    loss = nif.highlight_linear_mse(outputs, targets, params, 1.0, 2.0)

    torch.testing.assert_close(loss, torch.tensor(0.25))
    loss.backward()
    torch.testing.assert_close(outputs.grad[0], torch.zeros(3))
    assert torch.isfinite(outputs.grad).all()


def test_highlight_linear_loss_is_zero_when_batch_has_no_highlights():
    params = {
        "mean": [0.0, 0.0, 0.0],
        "max": 1.0,
        "transfer_function": "linear",
        "eps": 1e-8,
    }
    targets = torch.full((2, 3), 0.1)
    outputs = torch.zeros((2, 3), requires_grad=True)

    loss = nif.highlight_linear_mse(outputs, targets, params, 1.0, 1.0)

    torch.testing.assert_close(loss, torch.tensor(0.0))
    loss.backward()
    torch.testing.assert_close(outputs.grad, torch.zeros_like(outputs))
