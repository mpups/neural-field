"""Evaluate a NIF checkpoint using deployment precision and HDR metrics."""

import argparse
import json
import os

import cv2
import torch

import device_utils as du
import model as nif_model
import nif
from quality_metrics import hdr_quality_metrics


def parse_args():
    parser = argparse.ArgumentParser("Evaluate neural image field quality")
    parser.add_argument("--model", required=True)
    parser.add_argument("--original", default="")
    parser.add_argument("--precision", choices=["training", "fp8"], default="fp8")
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output", default="")
    parser.add_argument("--metrics", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    device = du.get_device(args.device)
    metadata = nif.load_metadata(nif.metadata_path_from_model_path(args.model))
    original_path = args.original or metadata["name"]
    original = cv2.imread(original_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if original is None:
        raise FileNotFoundError(f"Could not load image: {original_path}")

    network, checkpoint = nif_model.load_model_checkpoint(
        args.model, device, quantized=False
    )
    if args.precision == "fp8":
        network.set_fp8_qat(True)
        network.set_hash_fp16_qat(True)
    network.eval()
    reconstructed = nif.run_inference(
        network,
        device,
        tuple(metadata["original_image_shape"]),
        metadata["encode_params"],
        args.batch_size,
    )

    config = checkpoint["model_config"]
    checkpoint_quantization = config.get("quantization", {}).get("mlp")
    metrics = hdr_quality_metrics(original, reconstructed)
    metrics.update(
        {
            "precision": (
                "fp8-mlp-fp16-hash" if args.precision == "fp8" else "training"
            ),
            "quantization_stage": (
                "qat" if args.precision == "fp8" and checkpoint_quantization == "fp8-e4m3"
                else "ptq" if args.precision == "fp8" else "teacher"
            ),
            "layer_size": config["layer_size"],
            "hidden_layer_count": config.get("hidden_layer_count"),
            "architecture": config.get("architecture", "legacy-v1"),
            "parameter_count": sum(parameter.numel() for parameter in network.parameters()),
            "mlp_macs_per_query": sum(
                layer.in_features * layer.out_features for layer in network.layers
            ),
            "mlp_deployment_bytes": sum(
                layer.weight.numel()
                + (0 if layer.bias is None else 2 * layer.bias.numel())
                for layer in network.layers
            ),
            "hash_deployment_bytes": network.encoding.embedding.weight.numel() * 2,
        }
    )
    metrics_path = args.metrics or os.path.join(
        args.model, f"quality_metrics_{args.precision}.json"
    )
    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=2, sort_keys=True)
    if args.output:
        cv2.imwrite(args.output, reconstructed)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
