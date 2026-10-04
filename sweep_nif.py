"""Run reproducible MLP-capacity sweeps with optional production FP8 QAT."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


DEFAULT_ARCHITECTURES = "64x2,128x2,256x2,128x4,256x4,320x6"


def parse_architectures(specification):
    architectures = []
    for item in specification.split(","):
        width, depth = (int(value) for value in item.lower().split("x", 1))
        if width % 16 or depth < 2 or depth % 2:
            raise ValueError("Architectures require widths divisible by 16 and even depths")
        architectures.append((width, depth))
    return architectures


def parse_args():
    parser = argparse.ArgumentParser("Neural image field capacity sweep")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--architectures", default=DEFAULT_ARCHITECTURES)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--qat-epochs", type=int, default=50)
    parser.add_argument("--train-samples", type=int, default=1048576)
    parser.add_argument("--batch-size", type=int, default=262144)
    parser.add_argument("--evaluation-batch-size", type=int, default=65536)
    parser.add_argument("--color-space", choices=["rgb", "yuv", "ycocg"], default="rgb")
    parser.add_argument("--hashgrid-levels", type=int, default=12)
    parser.add_argument("--hashgrid-features", type=int, default=2)
    parser.add_argument("--hashgrid-log2-size", type=int, default=16)
    parser.add_argument("--hashgrid-max-res", type=int, default=2048)
    parser.add_argument("--fp8-qat", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command, log_path, dry_run):
    print(" ".join(str(item) for item in command), flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def main():
    args = parse_args()
    root = Path(args.output_root)
    script_root = Path(__file__).parent
    architectures = parse_architectures(args.architectures)
    seeds = [int(seed) for seed in args.seeds.split(",")]
    common = [
        "--input", args.input,
        "--epochs", str(args.epochs),
        "--train-samples", str(args.train_samples),
        "--batch-size", str(args.batch_size),
        "--evaluation-batch-size", str(args.evaluation_batch_size),
        "--color-space", args.color_space,
        "--hashgrid-levels", str(args.hashgrid_levels),
        "--hashgrid-features", str(args.hashgrid_features),
        "--hashgrid-log2-size", str(args.hashgrid_log2_size),
        "--hashgrid-max-res", str(args.hashgrid_max_res),
        "--fp16",
    ]

    metric_paths = []
    for width, depth in architectures:
        for seed in seeds:
            name = f"h{width}-l{depth}-seed{seed}"
            teacher = root / "teacher" / name
            train_command = [
                sys.executable, str(script_root / "train_nif.py"),
                "--model", str(teacher),
                "--layer-size", str(width),
                "--hidden-layer-count", str(depth),
                "--seed", str(seed),
                *common,
            ]
            run(train_command, teacher / "train.log", args.dry_run)
            metric_paths.append(teacher / "quality_metrics.json")
            ptq_metrics = teacher / "quality_metrics_fp8.json"
            run(
                [
                    sys.executable, str(script_root / "evaluate_nif.py"),
                    "--model", str(teacher), "--original", args.input,
                    "--precision", "fp8", "--batch-size", str(args.evaluation_batch_size),
                    "--metrics", str(ptq_metrics),
                ],
                teacher / "evaluate-fp8.log",
                args.dry_run,
            )
            metric_paths.append(ptq_metrics)

            if args.fp8_qat:
                qat = root / "fp8-qat" / name
                qat_command = [
                    sys.executable, str(script_root / "train_nif.py"),
                    "--input", args.input, "--model", str(qat),
                    "--resume", str(teacher), "--teacher-model", str(teacher),
                    "--fp8-qat", "--distillation-weight", "1.0",
                    "--learning-rate", "1e-5", "--epochs", str(args.qat_epochs),
                    "--train-samples", str(args.train_samples),
                    "--batch-size", str(args.batch_size),
                    "--evaluation-batch-size", str(args.evaluation_batch_size),
                    "--seed", str(seed), "--fp16",
                ]
                run(qat_command, qat / "train.log", args.dry_run)
                qat_metrics = qat / "quality_metrics_fp8.json"
                run(
                    [
                        sys.executable, str(script_root / "evaluate_nif.py"),
                        "--model", str(qat), "--original", args.input,
                        "--precision", "fp8",
                        "--batch-size", str(args.evaluation_batch_size),
                        "--metrics", str(qat_metrics),
                    ],
                    qat / "evaluate-fp8.log",
                    args.dry_run,
                )
                metric_paths.append(qat_metrics)

    if args.dry_run:
        return
    rows = []
    for path in metric_paths:
        with path.open() as file:
            row = json.load(file)
        row["path"] = str(path.parent)
        rows.append(row)
    with (root / "summary.json").open("w") as file:
        json.dump(rows, file, indent=2, sort_keys=True)
    columns = [
        "path", "precision", "quantization_stage", "layer_size", "hidden_layer_count",
        "linear_rgb_psnr_db", "log1p_rgb_psnr_db", "mean_exposure_psnr_db",
        "solid_angle_luminance_psnr_db", "mean_relative_absolute_error",
        "highlight_chromaticity_mae", "mlp_macs_per_query",
        "mlp_deployment_bytes", "hash_deployment_bytes",
    ]
    with (root / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
