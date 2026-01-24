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

try:
    import magic
except ImportError:
    magic = None


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
    parser.add_argument("--batch-size", type=int, default=8192, help="The batch size.")
    parser.add_argument(
        "--epochs", type=int, default=2000, help="Total number of epochs to train for."
    )
    parser.add_argument(
        "--layer-size", type=int, default=256, help="Hidden size of the MLPs."
    )
    parser.add_argument(
        "--layer-count",
        type=int,
        default=6,
        help="Number of MLP layers. Should be multiple of 2 for >= 4 layers.",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=1000000,
        help="The number of image samples used to train the NIF.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="fourier",
        choices=["fourier", "hashgrid"],
        help="Input encoding type: fourier (fixed) or hashgrid (learnable).",
    )
    # Fourier encoding options
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=10,
        help="Dimension of Fourier position embedding (fourier encoding only).",
    )
    parser.add_argument(
        "--embedding-sigma",
        type=float,
        default=2.0,
        help="Base for Fourier positional embedding.",
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
        "--compression-ratio",
        type=float,
        default=4.0,
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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.layer_count >= 4 and args.layer_count % 2:
        raise ValueError(
            "Layer count >= 4 must be a multiple of 2 (for skip connections)."
        )

    device = du.get_device(args.device)

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

    # Detect HDR and choose transfer function
    transfer_function = "linear"
    if magic is not None:
        file_magic = magic.from_file(args.input)
        print(f"Input: {file_magic}")
        if "OpenEXR" in file_magic:
            print("HDR input detected: using log transfer function")
            transfer_function = "log"
    else:
        _, file_extension = os.path.splitext(args.input)
        if file_extension.lower() in [".exr"]:
            print("HDR input detected (by extension): using log transfer function")
            transfer_function = "log"

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

    # Encoding setup
    log2_size = args.hashgrid_log2_size  # Will be updated if auto-calculated
    if args.encoding == "fourier":
        # Fourier: apply fixed encoding as preprocessing
        embedding_dimension = args.embedding_dimension
        input_dim = 4 * embedding_dimension  # sin/cos for u and v
        encoding_module = None
        print(
            f"Fourier encoding: dimension={embedding_dimension}, input_dim={input_dim}"
        )
    else:
        # Hash grid: encoding is part of model, pass raw UV
        embedding_dimension = 0  # Not used for hash grid
        input_dim = 2  # Raw UV

        # Auto-calculate hash table size if not specified
        if log2_size == 0:
            # Target model size based on compression ratio
            image_bytes = height * width * 3 * 4  # FP32 image
            target_bytes = image_bytes / args.compression_ratio

            # Estimate MLP params (rough: 2 * layer_size^2)
            mlp_bytes = 2 * args.layer_size * args.layer_size * 4
            hash_bytes = max(target_bytes - mlp_bytes, 32768)  # Min 32KB

            # entries = bytes / (4 bytes * n_features)
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
            base_resolution=args.hashgrid_base_res,
            max_resolution=args.hashgrid_max_res,
        ).to(device)

        total_hash_params = (
            args.hashgrid_levels * (2**log2_size) * args.hashgrid_features
        )
        print(
            f"Hash grid encoding: {args.hashgrid_levels} levels, "
            f"{2**log2_size} entries/level, {args.hashgrid_features} features/entry, "
            f"{total_hash_params:,} hash params"
        )

    # Helper to generate training data (samples from GPU image tensor)
    def generate_training_data():
        if args.deterministic_samples:
            uv = nif.deterministic_uv_samples(img.shape, device)
        else:
            uv = nif.stochastic_uv_samples(sample_count, device)

        values = nif.sample_image(img_tensor, uv)

        if args.encoding == "fourier":
            uv = nif.uv_positional_encode(uv, embedding_dimension, args.embedding_sigma)

        return uv, values

    # Initial dataset
    train_uv_encoded, train_values = generate_training_data()
    steps_per_epoch = sample_count // args.batch_size

    # Color space conversion
    color_matrix = nif.color_space_to_bgr_matrix(args.color_space)

    # Create model
    model_obj = nif_model.NIFModel(
        input_dim=input_dim,
        layer_size=args.layer_size,
        num_layers=args.layer_count,
        color_matrix=color_matrix,
        encoding=encoding_module,
    )
    model_obj = model_obj.to(device)

    print(f"\nModel created:")
    total_params = sum(p.numel() for p in model_obj.parameters())
    print(f"Total parameters: {total_params:,}")

    # Compile model if requested
    if args.compile:
        model_obj = nif_model.compile_model(model_obj, mode=args.compile_mode)

    optimizer = optim.Adam(model_obj.parameters(), lr=args.learning_rate)

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
        embedding_dim=embedding_dimension,
        embedding_sigma=args.embedding_sigma,
        model_path=args.model,
        encoding_type=args.encoding,
    )

    # Training loop
    num_epochs = 1 if args.single_step else args.epochs

    print(
        f"\nStarting training: {num_epochs} epochs, {steps_per_epoch} steps/epoch, device: {device}"
    )
    if not args.deterministic_samples:
        print("Stochastic mode: regenerating samples each epoch")

    epoch_start_time = time.time()
    for epoch in range(num_epochs):
        model_obj.train()

        # Regenerate samples each epoch for stochastic mode
        if not args.deterministic_samples and epoch > 0:
            train_uv_encoded, train_values = generate_training_data()

        # Shuffle indices on GPU
        perm = torch.randperm(train_uv_encoded.shape[0], device=device)
        train_uv_shuffled = train_uv_encoded[perm]
        train_values_shuffled = train_values[perm]

        epoch_loss = 0.0
        num_batches = 0

        for batch_idx in range(steps_per_epoch):
            start_idx = batch_idx * args.batch_size
            end_idx = start_idx + args.batch_size

            uv_batch = train_uv_shuffled[start_idx:end_idx]
            values_batch = train_values_shuffled[start_idx:end_idx]

            optimizer.zero_grad()

            if use_fp16:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    outputs = model_obj(uv_batch)
                    loss = loss_fn(outputs, values_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model_obj(uv_batch)
                loss = loss_fn(outputs, values_batch)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if args.single_step:
                break

        avg_loss = epoch_loss / max(num_batches, 1)
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch}: {epoch_time:.2f}s, loss={avg_loss:.6f}")
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
                    "model_state_dict": model_obj.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                checkpoint_path,
            )

            model_pt_path = os.path.join(args.model, "model.pt")
            torch.save(
                {
                    "model_state_dict": model_obj.state_dict(),
                    "model_config": {
                        "input_dim": input_dim,
                        "layer_size": args.layer_size,
                        "num_layers": args.layer_count,
                        "color_matrix": color_matrix,
                        "encoding_type": args.encoding,
                        "hashgrid_config": (
                            {
                                "n_levels": args.hashgrid_levels,
                                "n_features": args.hashgrid_features,
                                "log2_hashmap_size": log2_size,
                                "base_resolution": args.hashgrid_base_res,
                                "max_resolution": args.hashgrid_max_res,
                            }
                            if args.encoding == "hashgrid"
                            else None
                        ),
                    },
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
            "model_state_dict": model_obj.state_dict(),
            "model_config": {
                "input_dim": input_dim,
                "layer_size": args.layer_size,
                "num_layers": args.layer_count,
                "color_matrix": color_matrix,
                "encoding_type": args.encoding,
                "hashgrid_config": (
                    {
                        "n_levels": args.hashgrid_levels,
                        "n_features": args.hashgrid_features,
                        "log2_hashmap_size": log2_size,
                        "base_resolution": args.hashgrid_base_res,
                        "max_resolution": args.hashgrid_max_res,
                    }
                    if args.encoding == "hashgrid"
                    else None
                ),
            },
        },
        final_model_path,
    )
    print(f"Training complete. Model saved to {final_model_path}")

    # Final PSNR evaluation
    print("\nEvaluating final PSNR...")
    reconstructed = nif.run_inference(
        model=model_obj,
        device=device,
        img_shape=img.shape,
        embedding_dimension=embedding_dimension,
        embedding_sigma=args.embedding_sigma,
        encode_params=encode_params,
        encoding_type=args.encoding,
    )
    psnr = nif.compute_psnr(args.input, reconstructed)
    if psnr:
        print(f"PSNR RGB={psnr['rgb']:.2f} L={psnr['l']:.2f} AB={psnr['ab']:.2f}")
