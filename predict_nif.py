"""PyTorch Neural Image Field Inference Script"""

import torch
import argparse
import cv2
import nif
import model as nif_model
import os
import device_utils as du


def parse_args():
    parser = argparse.ArgumentParser("Neural Image Field (NIF) Inference")
    parser.add_argument(
        "--output", type=str, default="mlp_samples.png", help="Output image file name."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./saved_model/",
        help="Input path to load a trained NIF model.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Width of generated image (0 to use original).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="Height of generated image (0 to use original).",
    )
    parser.add_argument(
        "--original",
        type=str,
        default="",
        help="Original reference image for computing error metrics.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for inference.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2048, help="Batch size for inference."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    device = du.get_device(args.device)

    # Load metadata
    metadata_path = nif.metadata_path_from_model_path(args.model)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    metadata = nif.load_metadata(metadata_path)
    print(f"NIF metadata loaded from: {metadata_path}")

    img_shape = metadata["original_image_shape"]
    embedding_dimension = metadata["embedding_dimension"]
    embedding_sigma = metadata["embedding_sigma"]
    encode_params = metadata["encode_params"]

    if args.width == 0 or args.height == 0:
        width = img_shape[1]
        height = img_shape[0]
    else:
        width = args.width
        height = args.height
        img_shape = (
            [height, width, img_shape[2]] if len(img_shape) > 2 else [height, width]
        )

    # Load model
    model_path = os.path.join(args.model, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint.get("model_config", {})

    model = nif_model.NIFModel(
        input_dim=model_config.get("input_dim"),
        layer_size=model_config.get("layer_size"),
        num_layers=model_config.get("num_layers"),
        color_matrix=model_config.get("color_matrix"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Model loaded from: {model_path}")
    print(f"Generating {width}x{height} image...")

    # Run inference using the shared utility
    reconstructed = nif.run_inference(
        model=model,
        device=device,
        img_shape=tuple(img_shape),
        embedding_dimension=embedding_dimension,
        embedding_sigma=embedding_sigma,
        encode_params=encode_params,
        batch_size=args.batch_size,
    )

    cv2.imwrite(args.output, reconstructed)
    print(f"Reconstructed image saved to: {args.output}")

    # Compute metrics if original provided
    if args.original:
        psnr = nif.compute_psnr(args.original, reconstructed)
        if psnr:
            print(f"PSNR RGB: {psnr['rgb']:.2f}")
            print(f"PSNR L: {psnr['l']:.2f}")
            print(f"PSNR AB: {psnr['ab']:.2f}")
