"""Radiometric and display-referred metrics for HDR environment fields."""

import numpy as np


def _psnr(reference: np.ndarray, candidate: np.ndarray, peak: float) -> float:
    mse = np.mean((reference.astype(np.float64) - candidate.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(peak * peak / mse))


def _luminance_bgr(image: np.ndarray) -> np.ndarray:
    return (
        0.0722 * image[..., 0]
        + 0.7152 * image[..., 1]
        + 0.2126 * image[..., 2]
    )


def hdr_quality_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    """Measure direct reconstruction, lighting energy, and displayed exposures."""
    if reference.shape != candidate.shape or reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("HDR metric inputs must be same-shaped HxWx3 images")
    reference = reference.astype(np.float64)
    candidate = candidate.astype(np.float64)
    reference_positive = np.maximum(reference, 0)
    candidate_positive = np.maximum(candidate, 0)
    peak = max(float(reference_positive.max()), 1e-12)
    log_reference = np.log1p(reference_positive)
    log_candidate = np.log1p(candidate_positive)

    reference_luminance = _luminance_bgr(reference_positive)
    candidate_luminance = _luminance_bgr(candidate_positive)
    row_angles = np.pi * (np.arange(reference.shape[0]) + 0.5) / reference.shape[0]
    solid_angle_weights = np.sin(row_angles)[:, None]
    weighted_luminance_mse = float(
        np.sum(solid_angle_weights * (reference_luminance - candidate_luminance) ** 2)
        / (solid_angle_weights.sum() * reference.shape[1])
    )
    weighted_luminance_peak = max(float(reference_luminance.max()), 1e-12)

    weighted_reference_energy = np.sum(
        reference_positive * solid_angle_weights[..., None], axis=(0, 1)
    )
    weighted_candidate_energy = np.sum(
        candidate_positive * solid_angle_weights[..., None], axis=(0, 1)
    )
    energy_relative_error = np.abs(
        weighted_candidate_energy - weighted_reference_energy
    ) / np.maximum(weighted_reference_energy, 1e-12)

    highlight_threshold = np.percentile(reference_luminance, 99.9)
    highlight_mask = reference_luminance >= highlight_threshold
    reference_chroma = reference_positive / np.maximum(
        reference_positive.sum(axis=2, keepdims=True), 1e-12
    )
    candidate_chroma = candidate_positive / np.maximum(
        candidate_positive.sum(axis=2, keepdims=True), 1e-12
    )

    exposure_psnr = {}
    for ev in (-4, -2, 0, 2):
        scale = 2.0**ev
        displayed_reference = np.power(np.clip(reference_positive * scale, 0, 1), 1 / 2.2)
        displayed_candidate = np.power(np.clip(candidate_positive * scale, 0, 1), 1 / 2.2)
        exposure_psnr[str(ev)] = _psnr(displayed_reference, displayed_candidate, 1.0)

    relative_floor = max(float(reference_luminance.mean()) * 0.01, 1e-12)
    return {
        "linear_rgb_psnr_db": _psnr(reference, candidate, peak),
        "log1p_rgb_psnr_db": _psnr(
            log_reference, log_candidate, max(float(log_reference.max()), 1e-12)
        ),
        "mean_relative_absolute_error": float(
            np.mean(np.abs(candidate - reference) / (np.abs(reference) + relative_floor))
        ),
        "solid_angle_luminance_psnr_db": (
            float("inf") if weighted_luminance_mse == 0 else
            float(10 * np.log10(weighted_luminance_peak**2 / weighted_luminance_mse))
        ),
        "solid_angle_energy_relative_error_bgr": energy_relative_error.tolist(),
        "highlight_chromaticity_mae": float(
            np.mean(np.abs(candidate_chroma[highlight_mask] - reference_chroma[highlight_mask]))
        ),
        "negative_output_fraction": float(np.mean(candidate < 0)),
        "exposure_psnr_db": exposure_psnr,
        "mean_exposure_psnr_db": float(np.mean(list(exposure_psnr.values()))),
    }
