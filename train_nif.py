"""PyTorch Neural Image Field Training Script"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import cv2
import numpy as np
import nif
import model as nif_model
import time
import sys
import os
import device_utils as du
from nvfp4 import export_nvfp4

try:
    import magic
except ImportError:
    magic = None


def detect_transfer_function(path, file_magic=None):
    """Use logarithmic encoding for Radiance HDR and OpenEXR inputs."""
    extension = os.path.splitext(os.fspath(path))[1].lower()
    description = file_magic or ""
    is_hdr = extension in {".hdr", ".exr"} or any(
        marker in description for marker in ("OpenEXR", "Radiance HDR")
    )
    return "log" if is_hdr else "linear"


def parse_args():
    parser = argparse.ArgumentParser("Neural Image Field (NIF) Generator - PyTorch")
    parser.add_argument(
        "--input",
        type=str,
        default="Mandrill_portrait_2_Berlin_Zoo.jpg",
        help="Input image file name.",
    )
    parser.add_argument(
        "--blur",
        type=int,
        default=0,
        help="Size of Gaussian blur kernel applied to the input (0 to disable).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./saved_model/",
        help="Output path to save the trained NIF model.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="The learning rate for ADAM."
    )
    parser.add_argument("--batch-size", type=int, default=1024*1024, help="The batch size.")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Total number of epochs to train for."
    )
    parser.add_argument(
        "--layer-size", type=int, default=64, help="Hidden size of the MLPs."
    )
    parser.add_argument(
        "--layer-count",
        type=int,
        default=2,
        help="Number of MLP layers. Should be multiple of 2 for >= 4 layers.",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=1024*1024,
        help="The number of image samples used to train the NIF.",
    )
    # Hash grid encoding options
    parser.add_argument(
        "--hashgrid-levels",
        type=int,
        default=12,
        help="Number of resolution levels (hashgrid encoding only).",
    )
    parser.add_argument(
        "--hashgrid-features",
        type=int,
        default=2,
        help="Features per hash table entry (hashgrid encoding only).",
    )
    parser.add_argument(
        "--hashgrid-log2-size",
        type=int,
        default=0,
        help="Log2 of hash table size per level (0=auto based on image size). "
        "14=16K, 15=32K, 16=64K entries.",
    )
    parser.add_argument(
        "--hashgrid-level-log2-sizes",
        type=str,
        default="",
        help="comma-separated per-level log2 table sizes; overrides --hashgrid-log2-size",
    )
    parser.add_argument(
        "--hashgrid-layout",
        choices=["hash", "hybrid"],
        default="hash",
        help="use direct dense storage for coarse levels that fit their table budget",
    )
    parser.add_argument(
        "--compression-ratio",
        type=float,
        default=8.0,
        help="Target compression ratio for auto hash table sizing.",
    )
    parser.add_argument(
        "--hashgrid-base-res",
        type=int,
        default=16,
        help="Base (coarsest) resolution (hashgrid encoding only).",
    )
    parser.add_argument(
        "--hashgrid-max-res",
        type=int,
        default=1024,
        help="Maximum (finest) resolution (hashgrid encoding only).",
    )
    parser.add_argument(
        "--deterministic-samples",
        action="store_true",
        help="Create training data from one uv sample per pixel.",
    )
    parser.add_argument(
        "--checkpoint-period",
        type=int,
        default=0,
        help="Save checkpoints every N epochs (0=only at end).",
    )
    parser.add_argument(
        "--single-step", action="store_true", help="Execute a single step then exit."
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Train in fp16 (mixed precision)."
    )
    parser.add_argument(
        "--mse", action="store_true", help="Use MSE loss instead of Huber loss."
    )
    parser.add_argument(
        "--color-space",
        type=str,
        default="rgb",
        choices=["rgb", "yuv", "ycocg"],
        help="Color space for network output.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to train on.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for optimized execution.",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Warm start from a model directory or model.pt checkpoint.",
    )
    parser.add_argument(
        "--resume-optimizer",
        action="store_true",
        help="Restore Adam moments from --resume while keeping --learning-rate.",
    )
    parser.add_argument(
        "--nvfp4-qat",
        action="store_true",
        help="Fine-tune MLP weights and activations with NVFP4 fake quantization.",
    )
    parser.add_argument(
        "--nvfp4-start-epoch",
        type=int,
        default=0,
        help="Absolute epoch at which NVFP4 fake quantization begins.",
    )
    parser.add_argument(
        "--nvfp4-calibration-samples",
        type=int,
        default=262144,
        help="Samples used to fix deployment activation scales before NVFP4 QAT.",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="",
        help="Full-precision model directory or model.pt used for distillation.",
    )
    parser.add_argument(
        "--distillation-weight",
        type=float,
        default=0.0,
        help="Weight applied to MSE against the full-precision teacher output.",
    )
    parser.add_argument(
        "--evaluation-batch-size",
        type=int,
        default=2048,
        help="Batch size used for final full-image PSNR evaluation.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.layer_count >= 4 and args.layer_count % 2:
        raise ValueError(
            "Layer count >= 4 must be a multiple of 2 (for skip connections)."
        )
    if args.nvfp4_start_epoch < 0:
        raise ValueError("--nvfp4-start-epoch must be non-negative")
    if args.nvfp4_calibration_samples < 1:
        raise ValueError("--nvfp4-calibration-samples must be positive")
    if args.distillation_weight < 0:
        raise ValueError("--distillation-weight must be non-negative")
    if args.distillation_weight and not args.teacher_model:
        raise ValueError("--distillation-weight requires --teacher-model")
    if args.resume_optimizer and not args.resume:
        raise ValueError("--resume-optimizer requires --resume")
    level_log2_sizes = None
    if args.hashgrid_level_log2_sizes:
        level_log2_sizes = [int(value) for value in args.hashgrid_level_log2_sizes.split(",")]
        if len(level_log2_sizes) != args.hashgrid_levels:
            raise ValueError("--hashgrid-level-log2-sizes must contain one value per level")

    device = du.get_device(args.device)

    resume_checkpoint = None
    if args.resume:
        model_obj, resume_checkpoint = nif_model.load_model_checkpoint(
            args.resume, device, quantized=False
        )
        model_config = dict(resume_checkpoint["model_config"])
        print(f"Warm starting from: {args.resume}")

    # Load image
    img = cv2.imread(args.input, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {args.input}")

    if args.blur > 0:
        img = cv2.GaussianBlur(img, (args.blur, args.blur), 0)

    height = img.shape[0]
    width = img.shape[1]
    input_mean = np.mean(np.mean(img, axis=0), axis=0)
    input_min = img.min().astype(float)
    input_max = img.max().astype(float)

    print(
        f"Image loaded. Size: {img.shape} Type: {img.dtype} Mean: {input_mean} Min/max: {input_min}/{input_max}"
    )

    # Sample count
    if args.deterministic_samples:
        sample_count = height * width
        print(f"Using deterministic samples: {sample_count}")
    else:
        sample_count = args.train_samples
        print(f"Using {sample_count} stochastic samples (regenerated each epoch)")

    # Detect HDR and choose transfer function. Extensions remain authoritative
    # when python-magic is installed but reports a generic description.
    file_magic = None
    if magic is not None:
        file_magic = magic.from_file(args.input)
        print(f"Input: {file_magic}")
    transfer_function = detect_transfer_function(args.input, file_magic)
    if transfer_function == "log":
        print("HDR input detected: using log transfer function")

    # Prepare image tensor on GPU once (stays for entire training)
    img_tensor, encode_params = nif.prepare_image_for_sampling(
        img, transfer_function, device
    )
    print(f"Image tensor on GPU: {img_tensor.shape}, {img_tensor.device}")
    print(f"Encode params: {encode_params}")

    # Loss function
    if args.mse:
        loss_fn = nn.MSELoss(reduction="mean")
    else:
        loss_fn = nn.HuberLoss(delta=0.001, reduction="mean")

    # Hash grid encoding setup. A resumed checkpoint owns its architecture.
    if args.resume:
        encoding_module = model_obj.encoding
    else:
        log2_size = args.hashgrid_log2_size
        if log2_size == 0:
            image_bytes = height * width * 3 * 4  # FP32 image
            target_bytes = image_bytes / args.compression_ratio
            mlp_bytes = 2 * args.layer_size * args.layer_size * 4
            hash_bytes = max(target_bytes - mlp_bytes, 32768)  # Min 32KB
            total_entries = hash_bytes / (4 * args.hashgrid_features)
            entries_per_level = total_entries / args.hashgrid_levels
            log2_size = max(10, min(18, int(np.ceil(np.log2(entries_per_level)))))
            print(
                f"Auto hash table size: log2={log2_size} ({2**log2_size} entries/level) "
                f"for {args.compression_ratio}:1 target compression"
            )

        encoding_module = nif_model.HashGridEncoding(
            n_levels=args.hashgrid_levels,
            n_features=args.hashgrid_features,
            log2_hashmap_size=log2_size,
            level_log2_hashmap_sizes=level_log2_sizes,
            base_resolution=args.hashgrid_base_res,
            max_resolution=args.hashgrid_max_res,
            layout=args.hashgrid_layout,
        ).to(device)

    total_hash_params = encoding_module.embedding.num_embeddings * encoding_module.n_features
    print(
        f"Hash grid encoding: {encoding_module.n_levels} levels, {encoding_module.layout} layout, "
        f"{encoding_module.level_sizes} entries/level, {encoding_module.n_features} features/entry, "
        f"{total_hash_params:,} hash params"
    )

    # Helper to generate training data (samples from GPU image tensor)
    def generate_training_data():
        if args.deterministic_samples:
            uv = nif.deterministic_uv_samples(img.shape, device)
        else:
            uv = nif.stochastic_uv_samples(sample_count, device)

        values = nif.sample_image(img_tensor, uv)
        return uv, values

    # Initial dataset
    train_uv_encoded, train_values = generate_training_data()
    steps_per_epoch = (sample_count + args.batch_size - 1) // args.batch_size

    if not args.resume:
        color_matrix = nif.color_space_to_bgr_matrix(args.color_space)
        model_obj = nif_model.NIFModel(
            input_dim=2,
            layer_size=args.layer_size,
            num_layers=args.layer_count,
            color_matrix=color_matrix,
            encoding=encoding_module,
        ).to(device)
        model_config = {
            "input_dim": 2,
            "layer_size": args.layer_size,
            "num_layers": args.layer_count,
            "color_matrix": color_matrix,
            "hashgrid_config": {
                "n_levels": args.hashgrid_levels,
                "n_features": args.hashgrid_features,
                "log2_hashmap_size": log2_size,
                "level_log2_hashmap_sizes": list(encoding_module.level_log2_hashmap_sizes),
                "base_resolution": args.hashgrid_base_res,
                "max_resolution": args.hashgrid_max_res,
                "layout": args.hashgrid_layout,
            },
        }

    existing_quantization = model_config.get("quantization", {})
    use_nvfp4 = args.nvfp4_qat or existing_quantization.get("mlp") == "nvfp4"
    if use_nvfp4:
        activation_scales = existing_quantization.get("activation_global_scales")
        if activation_scales is None:
            calibration_count = min(
                args.nvfp4_calibration_samples, train_uv_encoded.shape[0]
            )
            activation_scales = nif_model.calibrate_nvfp4_activation_scales(
                model_obj,
                train_uv_encoded[:calibration_count],
                min(args.batch_size, calibration_count),
            ).tolist()
            print(
                "Calibrated fixed NVFP4 activation global scales: "
                f"{activation_scales}"
            )
        model_obj.set_nvfp4_qat(False, activation_scales)
        model_config["quantization"] = {
            "mlp": "nvfp4",
            "values": "e2m1",
            "block_scale": "e4m3",
            "global_scale": "fp32",
            "weight_block_shape": [16, 16],
            "activation_block_shape": [1, 16],
            "activation_global_scales": activation_scales,
        }

    print(f"\nModel created:")
    total_params = sum(p.numel() for p in model_obj.parameters())
    print(f"Total parameters: {total_params:,}")

    teacher_model = None
    if args.teacher_model:
        teacher_model, _ = nif_model.load_model_checkpoint(
            args.teacher_model, device, quantized=False
        )
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)
        print(f"Full-precision teacher loaded from: {args.teacher_model}")

    first_epoch = resume_checkpoint.get("epoch", -1) + 1 if resume_checkpoint else 0
    num_epochs = 1 if args.single_step else args.epochs
    final_epoch = first_epoch + num_epochs
    if use_nvfp4 and args.nvfp4_start_epoch >= final_epoch:
        raise ValueError("--nvfp4-start-epoch must occur within this training run")
    model_obj.set_nvfp4_qat(use_nvfp4 and first_epoch >= args.nvfp4_start_epoch)
    base_model = model_obj

    # Compile model if requested
    if args.compile:
        model_obj = nif_model.compile_model(model_obj, mode=args.compile_mode)

    optimizer = optim.Adam(model_obj.parameters(), lr=args.learning_rate)
    if args.resume_optimizer and "optimizer_state_dict" not in resume_checkpoint:
        raise ValueError("The resume checkpoint has no optimizer state")
    if args.resume_optimizer:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = args.learning_rate
        print(f"Optimizer state restored with learning rate {args.learning_rate:g}")

    # Mixed precision (CUDA only)
    use_fp16 = args.fp16 and device.type == "cuda"
    if args.fp16 and device.type != "cuda":
        print("WARNING: --fp16 ignored on non-CUDA device")
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16) if use_fp16 else None

    # Save metadata
    metadata_file = nif.metadata_path_from_model_path(args.model)
    nif.save_metadata(
        metadata_file,
        name=args.input,
        args=sys.argv,
        shape=img.shape,
        encode_params=encode_params,
        model_path=args.model,
    )

    print(
        f"\nStarting training: epochs {first_epoch}..{final_epoch - 1}, "
        f"{steps_per_epoch} steps/epoch, device: {device}"
    )
    if not args.deterministic_samples:
        print("Stochastic mode: regenerating samples each epoch")

    epoch_start_time = time.time()
    for epoch in range(first_epoch, final_epoch):
        qat_active = use_nvfp4 and epoch >= args.nvfp4_start_epoch
        base_model.set_nvfp4_qat(qat_active)
        model_obj.train()

        # Regenerate samples each epoch for stochastic mode
        if not args.deterministic_samples and epoch > 0:
            train_uv_encoded, train_values = generate_training_data()

        # Shuffle indices on GPU
        perm = torch.randperm(train_uv_encoded.shape[0], device=device)
        train_uv_shuffled = train_uv_encoded[perm]
        train_values_shuffled = train_values[perm]

        epoch_loss = 0.0
        epoch_reconstruction_loss = 0.0
        epoch_distillation_loss = 0.0
        num_batches = 0

        for batch_idx in range(steps_per_epoch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, train_uv_shuffled.shape[0])

            uv_batch = train_uv_shuffled[start_idx:end_idx]
            values_batch = train_values_shuffled[start_idx:end_idx]

            optimizer.zero_grad()

            teacher_outputs = None
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = teacher_model(uv_batch)

            if use_fp16:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    outputs = model_obj(uv_batch)
                    reconstruction_loss = loss_fn(outputs, values_batch)
                    distillation_loss = (
                        nn.functional.mse_loss(outputs, teacher_outputs)
                        if teacher_outputs is not None else outputs.new_zeros(())
                    )
                    loss = reconstruction_loss + args.distillation_weight * distillation_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model_obj(uv_batch)
                reconstruction_loss = loss_fn(outputs, values_batch)
                distillation_loss = (
                    nn.functional.mse_loss(outputs, teacher_outputs)
                    if teacher_outputs is not None else outputs.new_zeros(())
                )
                loss = reconstruction_loss + args.distillation_weight * distillation_loss
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            epoch_reconstruction_loss += reconstruction_loss.item()
            epoch_distillation_loss += distillation_loss.item()
            num_batches += 1

            if args.single_step:
                break

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_reconstruction_loss = epoch_reconstruction_loss / max(num_batches, 1)
        avg_distillation_loss = epoch_distillation_loss / max(num_batches, 1)
        epoch_time = time.time() - epoch_start_time
        print(
            f"Epoch {epoch}: {epoch_time:.2f}s, loss={avg_loss:.6f}, "
            f"reconstruction={avg_reconstruction_loss:.6f}, "
            f"distillation={avg_distillation_loss:.6f}, nvfp4={qat_active}"
        )
        sys.stdout.flush()
        epoch_start_time = time.time()

        # Save checkpoint periodically
        should_save = (
            args.checkpoint_period > 0 and (epoch + 1) % args.checkpoint_period == 0
        )
        if should_save:
            os.makedirs(args.model, exist_ok=True)
            checkpoint_path = os.path.join(args.model, f"checkpoint_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": base_model.state_dict(),
                    "model_config": model_config,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                checkpoint_path,
            )

            model_pt_path = os.path.join(args.model, "model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": base_model.state_dict(),
                    "model_config": model_config,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                model_pt_path,
            )

        if args.single_step:
            break

    # Final model save
    os.makedirs(args.model, exist_ok=True)
    final_model_path = os.path.join(args.model, "model.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": base_model.state_dict(),
            "model_config": model_config,
            "optimizer_state_dict": optimizer.state_dict(),
        },
        final_model_path,
    )
    print(f"Training complete. Model saved to {final_model_path}")

    if use_nvfp4:
        nvfp4_path = os.path.join(args.model, "nvfp4.pt")
        export_nvfp4(base_model, nvfp4_path)
        print(f"Packed logical NVFP4 MLP saved to {nvfp4_path}")

    # Final PSNR evaluation
    print("\nEvaluating final PSNR...")
    base_model.set_nvfp4_qat(use_nvfp4)
    reconstructed = nif.run_inference(
        model=base_model,
        device=device,
        img_shape=img.shape,
        encode_params=encode_params,
        batch_size=args.evaluation_batch_size,
    )
    psnr = nif.compute_psnr(args.input, reconstructed)
    if psnr:
        print(f"PSNR RGB={psnr['rgb']:.2f} L={psnr['l']:.2f} AB={psnr['ab']:.2f}")
