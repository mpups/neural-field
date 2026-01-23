"""PyTorch Neural Image Field Training Script"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
        help="Number of MLP layers. Should be multiple of 2.",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=1000000,
        help="The number of image samples used to train the NIF.",
    )
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=10,
        help="Dimension of the position embedding space for UV coords.",
    )
    parser.add_argument(
        "--embedding-sigma",
        type=float,
        default=2.0,
        help="Base for the positional embedding (power base for Fourier features).",
    )
    parser.add_argument(
        "--no-position-embedding",
        action="store_true",
        help="Disable the position embedding and train directly on UV coords.",
    )
    parser.add_argument(
        "--deterministic-samples",
        action="store_true",
        help="Create training data from one uv sample per pixel (instead of randomly distributed).",
    )
    parser.add_argument(
        "--disable-psnr",
        action="store_true",
        help="Disable peak signal-to-noise ratio evaluation during training.",
    )
    parser.add_argument(
        "--callback-period",
        type=int,
        default=10,
        help="Interval in epochs at which to log training stats and evaluate PSNR (if enabled).",
    )
    parser.add_argument(
        "--single-step",
        action="store_true",
        help="If set the program will execute a single step then exit.",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Train in fp16 (mixed precision)."
    )
    parser.add_argument(
        "--loss-scale",
        type=float,
        default=32768,
        help="Loss scale (affects fp16 training only).",
    )
    parser.add_argument(
        "--mse",
        action="store_true",
        help="Use MSE loss instead of the default Huber loss.",
    )
    parser.add_argument(
        "--color-space",
        type=str,
        default="rgb",
        choices=["rgb", "yuv", "ycocg"],
        help="Force the network to predict in the specified color-space and convert to RGB.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to train on (cuda or cpu).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.layer_count >= 4 and args.layer_count % 2:
        raise ValueError("Layer count >= 4 must be a multiple of 2 (for skip connections).")

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

    # Generate training UV samples (on device)
    print(f"Generating samples on {device}...")
    t_start = time.time()

    if args.deterministic_samples:
        train_uv = nif.deterministic_uv_samples(img.shape, device)
        sample_count = train_uv.shape[0]
        print(f"Using deterministic samples: {sample_count}")
    else:
        sample_count = args.train_samples
        train_uv = nif.stochastic_uv_samples(sample_count, device)
        print(
            f"Generated {sample_count} stochastic samples in {time.time() - t_start:.2f}s"
        )

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

    # Encode samples (bilinear interpolation on device)
    _, file_extension = os.path.splitext(args.input)
    debug_file = "input_samples" + file_extension
    train_values, encode_params = nif.encode_samples(
        img, train_uv, transfer_function, device, debug_file
    )
    print(f"Encode params: {encode_params}")

    max_encoded, mean_encoded = nif.value_stats(train_values)
    print(f"Max value after encode: {max_encoded} mean after encode: {mean_encoded}")

    # Loss function
    if args.mse:
        loss_fn = nn.MSELoss(reduction="mean")
    else:
        loss_fn = nn.HuberLoss(delta=0.001, reduction="mean")

    # Position encoding (vectorized on device)
    embedding_dimension = 0 if args.no_position_embedding else args.embedding_dimension
    if not args.no_position_embedding:
        t0 = time.time()
        train_uv = nif.uv_positional_encode(
            train_uv, embedding_dimension, args.embedding_sigma
        )
        print(f"UV positional encode time: {time.time() - t0:.2f}s")

    # Create dataset and dataloader
    # Data is already on device, move to CPU for DataLoader (it will move back during iteration)
    dataset = TensorDataset(train_uv.cpu(), train_values.cpu())

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )

    # Color space conversion
    color_matrix = nif.color_space_to_bgr_matrix(args.color_space)

    # Create model
    input_dim = train_uv.shape[-1]
    model_obj = nif_model.NIFModel(
        input_dim=input_dim,
        layer_size=args.layer_size,
        num_layers=args.layer_count,
        color_matrix=color_matrix,
    )
    model_obj = model_obj.to(device)

    print("\nModel created:")
    total_params = sum(p.numel() for p in model_obj.parameters())
    print(f"Total parameters: {total_params}")

    optimizer = optim.Adam(model_obj.parameters(), lr=args.learning_rate)

    # Mixed precision (CUDA only)
    use_fp16 = args.fp16 and device.type == "cuda"
    if args.fp16 and device.type != "cuda":
        print("WARNING: --fp16 ignored on non-CUDA device")
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16) if use_fp16 else None

    # Evaluation callback
    eval_callback = nif.EvalCallback(
        model=model_obj,
        device=device,
        original_file=args.input,
        img_shape=img.shape,
        embedding_dimension=embedding_dimension,
        embedding_sigma=args.embedding_sigma,
        encode_params=encode_params,
        period=args.callback_period,
        compute_psnr_flag=not args.disable_psnr,
    )

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
    )

    # Training loop
    num_epochs = 1 if args.single_step else args.epochs
    steps_per_epoch = 1 if args.single_step else len(train_loader)

    print(
        f"\nStarting training: {num_epochs} epochs, {steps_per_epoch} steps/epoch, device: {device}"
    )

    for epoch in range(num_epochs):
        model_obj.train()
        eval_callback.on_epoch_begin(epoch)

        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (uv_batch, values_batch) in enumerate(train_loader):
            uv_batch = uv_batch.to(device)
            values_batch = values_batch.to(device)

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
        eval_callback.on_epoch_end(epoch, loss=avg_loss)
        sys.stdout.flush()

        # Save checkpoint periodically
        if (epoch + 1) % args.callback_period == 0 or epoch == num_epochs - 1:
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
            },
        },
        final_model_path,
    )
    print(f"Training complete. Model saved to {final_model_path}")
