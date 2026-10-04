# Neural Image Fields

Uses a hashgrid encoding (as in instant NGP) and small neural network trained as a function approximator to go from pixel
coordinates to colour values. With careful choice of parameters this is a form of neural image compression.

## Quick Start

Train and reconstruct the example image with a small number of epochs (increase for higher quality):

```bash
# Install dependencies
pip install -r requirements.txt

# Train
python train_nif.py --input Mandrill_portrait_2_Berlin_Zoo.jpg --epochs 20

# Inference
python predict_nif.py --output reconstruction.png --original Mandrill_portrait_2_Berlin_Zoo.jpg
```

Use an Instant-NGP-style hybrid layout to store coarse levels densely and fine
levels in hash tables:

```bash
python train_nif.py --input image.hdr --hashgrid-layout hybrid
```

Hash budgets can also be assigned independently to each level for architecture
sweeps. The list length must match `--hashgrid-levels`:

```bash
python train_nif.py --input image.hdr --hashgrid-levels 4 \
  --hashgrid-level-log2-sizes 12,13,14,15
```

## NVFP4 quantization-aware fine-tuning

Fine-tune an existing full-precision checkpoint with the NVIDIA NVFP4 E2M1
value format. The simulation uses FP8 E4M3 scales per 16 elements, an FP32
global tensor scale, 16x16 scaling for MLP weights, 1x16 scaling for
activations, and straight-through gradients to FP32 master weights. Activation
global scales are calibrated once and fixed during QAT, allowing the producer
of each activation to write packed NVFP4 and block scales directly without a
runtime tensor-wide reduction:

```bash
python train_nif.py \
  --input image.hdr \
  --model ./saved_model_nvfp4 \
  --resume ./saved_model \
  --teacher-model ./saved_model \
  --nvfp4-qat \
  --distillation-weight 1.0 \
  --learning-rate 1e-5 \
  --epochs 20 \
  --fp16
```

`--resume` restores model weights and architecture. Add `--resume-optimizer`
only when continuing the same optimization run; Adam moments are then restored
but the explicitly requested `--learning-rate` is retained. The QAT checkpoint
contains FP32 master weights and quantization metadata. `predict_nif.py`
automatically applies the fake-NVFP4 forward path for such checkpoints.

Training also writes `nvfp4.pt`, which contains nibble-packed MLP weights and
logical row-major scales. The scale layout is intentionally unswizzled: a
future CUDA/CUTLASS backend must transform it to the hardware matrix layout.
The current ray tracer cannot execute this packed artifact yet.

## HDR quality and FP8 capacity sweeps

The renderer's current production target uses an FP16 hash grid and unscaled
E4M3 MLP weights and activations. Evaluate any checkpoint at that precision:

```bash
python evaluate_nif.py \
  --model ./saved_model \
  --original image.hdr \
  --precision fp8
```

Unlike the legacy `--layer-count`, `--hidden-layer-count` follows the paper:
it counts dense-ReLU hidden layers, adds a separate RGB output layer, and uses
the midpoint skip connection. The count must be positive and even.

Run the initial capacity ablation with FP8 post-training evaluation after every
teacher model. Add `--fp8-qat` to also fine-tune and evaluate an E4M3 model for
each architecture:

```bash
python sweep_nif.py \
  --input image.hdr \
  --output-root ./sweeps/capacity-rgb \
  --architectures 64x2,128x2,256x2,128x4,256x4,320x6 \
  --seeds 1 \
  --fp8-qat
```

The summary contains radiometric PSNR, log-domain PSNR, solid-angle-weighted
luminance and energy errors, highlight chromaticity error, an exposure-stack
display metric, deployment bytes, and MLP MACs per query. Full-precision
teachers and post-training FP8 results are kept separate from FP8-QAT outputs.
