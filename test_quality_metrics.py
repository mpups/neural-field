import numpy as np

from quality_metrics import hdr_quality_metrics


def test_hdr_metrics_detect_energy_and_highlight_colour_errors():
    reference = np.ones((8, 16, 3), dtype=np.float32)
    reference[4, 8] = [4, 8, 16]
    candidate = reference.copy()
    candidate[4, 8] = [16, 8, 4]
    candidate *= 0.5

    metrics = hdr_quality_metrics(reference, candidate)

    assert metrics["linear_rgb_psnr_db"] < 30
    assert metrics["highlight_chromaticity_mae"] > 0.1
    assert all(error > 0 for error in metrics["solid_angle_energy_relative_error_bgr"])
    assert set(metrics["exposure_psnr_db"]) == {"-4", "-2", "0", "2"}


def test_hdr_metrics_are_exact_for_identical_images():
    reference = np.full((2, 4, 3), 0.25, dtype=np.float32)
    metrics = hdr_quality_metrics(reference, reference)

    assert np.isinf(metrics["linear_rgb_psnr_db"])
    assert metrics["mean_relative_absolute_error"] == 0
    assert metrics["solid_angle_energy_relative_error_bgr"] == [0, 0, 0]
