from train_nif import detect_transfer_function


def test_radiance_hdr_extension_uses_log_transfer():
    assert detect_transfer_function("environment.HDR", "data") == "log"


def test_openexr_magic_uses_log_transfer_without_known_extension():
    assert detect_transfer_function("environment.bin", "OpenEXR image data") == "log"


def test_ordinary_image_uses_linear_transfer():
    assert detect_transfer_function("photo.png", "PNG image data") == "linear"
