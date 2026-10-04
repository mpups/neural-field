# HDR NIF quality sweep

## Deployment contract

Quality candidates are ranked using the renderer's current production target:

- FP16 hash-table storage and FP16 bilinear interpolation;
- unscaled E4M3 MLP weights and activations;
- FP16 MLP biases and outputs.

FP32-master/FP16-autocast checkpoints are retained as teachers and controls.
Post-training FP8 and FP8-QAT results are reported separately. NVFP4 is not a
target of this sweep.

## Capacity ablation

Urban Alley was trained for 250 epochs with 1,048,576 stochastic samples per
epoch, batch size 262,144, RGB output, logarithmic HDR encoding, Huber loss,
and seed 1. FP8 QAT used another 50 epochs at learning rate 1e-5 with teacher
distillation weight 1.0. The 12-level, two-feature, `2^16`-entry hash grid was
held fixed. `L` is the paper-compatible number of hidden dense-ReLU layers.

| Model | Stage | Linear PSNR | log1p PSNR | Exposure PSNR | Relative error | MLP MACs/query |
|---|---|---:|---:|---:|---:|---:|
| H64 L2 | teacher | 41.71 | 35.26 | 30.86 | 11.54% | 5,248 |
| H64 L2 | FP8 PTQ | 41.11 | 34.88 | 30.65 | 11.94% | 5,248 |
| H64 L2 | FP8 QAT | 41.44 | 35.07 | 30.73 | 11.90% | 5,248 |
| H128 L2 | FP8 QAT | 41.99 | 35.65 | 31.21 | 11.20% | 19,264 |
| H256 L2 | FP8 QAT | 42.33 | 35.90 | 31.39 | 10.99% | 71,872 |
| H128 L4 | FP8 QAT | **42.83** | **36.65** | 32.02 | 10.45% | 49,536 |
| H256 L4 | FP8 QAT | 42.93 | 36.69 | **32.09** | **10.27%** | 197,376 |
| H320 L6 | FP8 QAT | 42.49 | 36.40 | 31.98 | 10.59% | 512,960 |

The existing production model has one H64 hidden layer rather than the paper's
H64/L2 architecture. Under the same deployment simulator it measures 39.46 dB
linear PSNR, 32.68 dB log1p PSNR, 28.76 dB mean exposure PSNR, and 14.40%
relative error. Restoring the missing hidden layer is therefore a substantial
quality gain. FP8 QAT consistently recovers part of the PTQ loss. H128/L4 is
the provisional Pareto knee; H256/L4 offers little extra quality for about four
times its MLP arithmetic, and H320/L6 has not converged to a better solution at
this training budget.

## Next ablations

1. Repeat H128/L2, H128/L4, and H256/L4 with seeds 2 and 3.
2. Hold H128/L4 fixed and compare RGB, YUV, and YCoCg. The original paper found
   YUV was the only representation to preserve the hue of sparse sun lighting,
   even when aggregate PSNR barely distinguished it from YCoCg.
3. On the winning colour representation, compare Huber/MSE and HDR transfer
   functions, then solid-angle and highlight-aware sampling.
4. Sweep hash levels, features, resolutions, and per-level capacities while
   preferring MLP input dimensions compatible with E4M3 matrix kernels.
5. Integrate only the nondominated architectures into the renderer and measure
   actual frame time and memory before selecting the final Pareto frontier.

Every stage retains linear, log-domain, exposure-stack, solid-angle-weighted
luminance and energy, highlight chromaticity, deployment-byte, and MAC metrics.
