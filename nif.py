"""Neural Image Field Utilities - PyTorch-based data and I/O functions"""

import torch
import cv2
import numpy as np
import json
import os


def value_stats(values):
    """Compute max and mean statistics of values (tensor or numpy)."""
    if isinstance(values, torch.Tensor):
        max_val = values.max().item()
        mean = values.mean(dim=0)
        return max_val, mean
    else:
        max_val = values.max().astype(float)
        mean = np.mean(values, axis=0).astype(float)
        return max_val, mean


def make_image_grid(width, height, device):
    """Generate UV grid coordinates for full image.

    Args:
        width: Image width
        height: Image height
        device: torch.device

    Returns:
        pixel_coords: (H*W, 2) integer pixel coordinates
        uv_coords: (H*W, 2) normalized UV coordinates [0, 1]
    """
    v_coords, u_coords = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )

    pixel_coords = torch.stack([v_coords.flatten(), u_coords.flatten()], dim=1)

    uv_coords = torch.stack(
        [v_coords.flatten() / (height - 1), u_coords.flatten() / (width - 1)], dim=1
    )

    return pixel_coords, uv_coords


def deterministic_uv_samples(image_shape, device):
    """Generate one UV sample per pixel in deterministic grid order.

    Args:
        image_shape: (height, width, channels)
        device: torch.device

    Returns:
        uv: (H*W, 2) tensor of UV coordinates
    """
    height, width = image_shape[0], image_shape[1]
    _, uv = make_image_grid(width, height, device)
    return uv


def stochastic_uv_samples(sample_count, device):
    """Generate random UV samples from uniform distribution.

    Args:
        sample_count: Number of samples to generate
        device: torch.device

    Returns:
        uv: (sample_count, 2) tensor of UV coordinates in [0, 1]
    """
    uv = torch.rand(sample_count, 2, device=device, dtype=torch.float32)
    return uv


def bilinear_interpolate(img, coords):
    """Bilinear interpolation for image sampling at arbitrary coordinates.

    Args:
        img: (H, W, C) image tensor
        coords: (N, 2) coordinates as (row, col) in pixel space

    Returns:
        (N, C) sampled values
    """
    y = coords[:, 0]
    x = coords[:, 1]

    h, w, c = img.shape

    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0_safe = torch.clamp(x0, 0, w - 1)
    x1_safe = torch.clamp(x1, 0, w - 1)
    y0_safe = torch.clamp(y0, 0, h - 1)
    y1_safe = torch.clamp(y1, 0, h - 1)

    wa = ((x1.float() - x) * (y1.float() - y)).unsqueeze(1)
    wb = ((x1.float() - x) * (y - y0.float())).unsqueeze(1)
    wc = ((x - x0.float()) * (y1.float() - y)).unsqueeze(1)
    wd = ((x - x0.float()) * (y - y0.float())).unsqueeze(1)

    Ia = img[y0_safe, x0_safe]
    Ib = img[y1_safe, x0_safe]
    Ic = img[y0_safe, x1_safe]
    Id = img[y1_safe, x1_safe]

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def prepare_image_for_sampling(image, transfer_function, device):
    """Prepare image tensor for repeated sampling.

    Call once at start of training to get a GPU tensor and encoding params.

    Args:
        image: numpy array (H, W, C)
        transfer_function: 'linear' or 'log'
        device: torch.device

    Returns:
        img_tensor: Normalized image tensor on device
        encode_params: dict with encoding parameters for decoding
    """
    img_tensor = torch.from_numpy(image.astype(np.float32)).to(device)

    max_value = img_tensor.max().item()

    eps = 1e-8
    if transfer_function == "log":
        img_tensor = torch.log(img_tensor + eps)
        max_value = img_tensor.max().item()

    mean_value = img_tensor.mean(dim=(0, 1))
    img_tensor = img_tensor - mean_value

    if max_value == 0:
        max_value = 1.0
    img_tensor = img_tensor / max_value

    encode_params = {
        "mean": mean_value.cpu().tolist(),
        "max": float(max_value),
        "log_tone_map": transfer_function == "log",
        "transfer_function": transfer_function,
        "eps": eps,
    }

    return img_tensor, encode_params


def sample_image(img_tensor, uv_coords):
    """Sample from a prepared image tensor at UV coordinates.

    Args:
        img_tensor: Prepared image tensor from prepare_image_for_sampling
        uv_coords: (N, 2) tensor of UV coordinates in [0, 1]

    Returns:
        (N, C) tensor of sampled values
    """
    height, width = img_tensor.shape[0], img_tensor.shape[1]
    device = img_tensor.device

    remap_coords = uv_coords * torch.tensor([height - 1, width - 1], device=device)
    return bilinear_interpolate(img_tensor, remap_coords)


def encode_samples(image, uv_coords, transfer_function, device, debug_filename=""):
    """Encode image values at UV sample coordinates.

    For repeated sampling, use prepare_image_for_sampling + sample_image instead.

    Args:
        image: numpy array (H, W, C)
        uv_coords: (N, 2) tensor of UV coordinates
        transfer_function: 'linear' or 'log'
        device: torch.device
        debug_filename: optional path to save debug image

    Returns:
        train_values: (N, C) tensor of encoded sample values
        encode_params: dict with encoding parameters for decoding
    """
    img_tensor, encode_params = prepare_image_for_sampling(
        image, transfer_function, device
    )
    train_values = sample_image(img_tensor, uv_coords)

    if debug_filename:
        height, width = image.shape[0], image.shape[1]
        remap_coords = uv_coords * torch.tensor([height - 1, width - 1], device=device)
        pixel_coords = torch.round(remap_coords).long()
        decoded = decode_samples(
            image.shape,
            pixel_coords.cpu().numpy(),
            train_values.cpu().numpy(),
            encode_params,
        )
        cv2.imwrite(debug_filename, decoded)

    return train_values, encode_params


def decode_samples(image_shape, uv, values, params):
    """Reconstruct image from UV coordinates and predicted values.

    Args:
        image_shape: (H, W, C) tuple
        uv: (N, 2) pixel coordinates as numpy array
        values: (N, C) predicted values as numpy array
        params: encoding parameters dict

    Returns:
        Reconstructed image as numpy array
    """
    if isinstance(values, torch.Tensor):
        values = values.cpu().numpy()
    if isinstance(uv, torch.Tensor):
        uv = uv.cpu().numpy()

    if uv.shape[0] != values.shape[0]:
        raise ValueError(
            f"Size mismatch between uv coords: {uv.shape} and values: {values.shape}"
        )

    decoded_values = values * params["max"]

    transfer_function = params["transfer_function"]
    if transfer_function not in ["linear", "log"]:
        raise ValueError(f"Unsupported transfer function: {transfer_function}")

    if transfer_function == "log":
        decoded_values = decoded_values + np.array(params["mean"])
        decoded_values = np.exp(decoded_values) - params["eps"]
    else:
        decoded_values = decoded_values + np.array(params["mean"])

    output = np.zeros(shape=image_shape, dtype=np.float32)
    uv_int = uv.astype(np.int32)
    for i, (r, c) in enumerate(uv_int):
        output[r, c] = decoded_values[i]

    return output


def save_metadata(
    file_name,
    name,
    args,
    shape,
    encode_params,
    model_path,
):
    """Save training metadata alongside model."""
    nif_params = {
        "name": name,
        "train_command": args,
        "original_image_shape": list(shape),
        "pytorch_model": model_path,
        "encode_params": encode_params,
    }
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as file:
        json.dump(nif_params, file, indent=2, sort_keys=True)


def load_metadata(file_name):
    """Load training metadata from file."""
    with open(file_name, "r") as file:
        data = json.load(file)
    return data


def metadata_path_from_model_path(model_path):
    """Determine metadata file path from model path."""
    return os.path.join(model_path, "nif_metadata.json")


def color_space_to_bgr_matrix(color_space: str):
    """Return transformation matrix from color space to BGR."""
    if color_space not in ["rgb", "yuv", "ycocg"]:
        raise ValueError(f"Unsupported color space: {color_space}")

    if color_space == "yuv":
        return np.array([[1, 2.032, 0], [1, -0.395, -0.581], [1, 0, 1.140]])
    if color_space == "ycocg":
        return np.array([[1, -1, -1], [1, 0, 1], [1, -1, 1]])
    return None


def compute_psnr(original_file, reconstructed_image):
    """Compute PSNR metrics between original and reconstructed images.

    Args:
        original_file: Path to original image file
        reconstructed_image: Reconstructed image as numpy array (BGR)

    Returns:
        Dict with 'rgb', 'l', 'ab' PSNR values, or None if computation fails
    """
    from skimage import color, metrics

    bgr_img = cv2.imread(original_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if bgr_img is None:
        print(f"Warning: Could not load original image for PSNR: {original_file}")
        return None

    bgr_img = bgr_img.astype(np.float32)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    recon_rgb = cv2.cvtColor(reconstructed_image.astype(np.float32), cv2.COLOR_BGR2RGB)

    rgb_max = np.max(rgb_img)
    rgb_mse = metrics.mean_squared_error(rgb_img, recon_rgb)
    psnr_rgb = (
        10 * np.log10((rgb_max / rgb_mse) * rgb_max) if rgb_mse > 0 else float("inf")
    )

    lab_img = color.rgb2lab(rgb_img)
    lab_recon = color.rgb2lab(recon_rgb)

    lum_img = lab_img[:, :, 0]
    lum_recon = lab_recon[:, :, 0]
    lum_max = np.max(lum_img)
    lum_mse = metrics.mean_squared_error(lum_img, lum_recon)
    psnr_l = (
        10 * np.log10((lum_max / lum_mse) * lum_max) if lum_mse > 0 else float("inf")
    )

    ab_img = lab_img[:, :, 1:]
    ab_recon = lab_recon[:, :, 1:]
    ab_max = np.max(ab_img)
    ab_mse = metrics.mean_squared_error(ab_img, ab_recon)
    psnr_ab = 10 * np.log10((ab_max / ab_mse) * ab_max) if ab_mse > 0 else float("inf")

    return {"rgb": psnr_rgb, "l": psnr_l, "ab": psnr_ab}


def run_inference(
    model,
    device,
    img_shape,
    encode_params,
    batch_size=2048,
):
    """Run full-image inference using the model.

    Args:
        model: PyTorch model (already on device, includes hash grid encoding)
        device: torch.device
        img_shape: Original image shape (H, W, C)
        encode_params: Encoding parameters from training
        batch_size: Inference batch size

    Returns:
        Reconstructed image as numpy array
    """
    height, width = img_shape[0], img_shape[1]

    # Model handles encoding internally, pass raw UV
    pixel_coords, uv_coords = make_image_grid(width, height, device)

    output_samples = []

    model.eval()
    with torch.no_grad():
        for i in range(0, uv_coords.shape[0], batch_size):
            batch = uv_coords[i : i + batch_size]
            batch_output = model(batch)
            output_samples.append(batch_output)

    result = torch.cat(output_samples, dim=0).cpu().numpy()

    output_sample_count = width * height
    reconstructed = decode_samples(
        tuple(img_shape),
        pixel_coords[0:output_sample_count].cpu().numpy().astype(np.int32),
        result,
        encode_params,
    )

    return reconstructed



