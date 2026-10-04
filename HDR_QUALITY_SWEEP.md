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

Three-seed replication gives 42.68±0.18 dB for H128/L4 and 42.86±0.07 dB for
H256/L4 after FP8 QAT. The wider model's 0.18 dB mean advantage still costs
four times the MACs, so H128/L4 remains the quality-efficiency choice. H256/L2
and H256/L4 remain candidates for hardware timing because fewer or wider GEMMs
may execute more efficiently than the arithmetic count predicts.

## Colour-space ablation

H128/L4 was trained and FP8-QAT fine-tuned with three seeds per colour space.
The fixed 3x3 conversion is simulated as an FP16 operation outside the E4M3
MLP, matching the intended renderer implementation.

| Colour | Linear PSNR | Exposure PSNR | Relative error | Highlight chroma MAE |
|---|---:|---:|---:|---:|
| RGB | **42.68±0.18** | **31.98±0.09** | **10.39%** | 0.01784 |
| YUV | 42.56±0.19 | 31.97±0.09 | 10.58% | 0.01746 |
| YCoCg | 42.49±0.29 | 31.89±0.17 | 10.69% | **0.01712** |

RGB leads aggregate reconstruction metrics, but matched exposure inspection
finds YCoCg preserves sparse highlight hue and neutral shadow colour best on
Urban Alley. YCoCg is therefore the provisional representation for HDR-loss
experiments. This is not yet a universal choice: the paper found YUV was the
only correct lighting representation on Borghese Gardens, and the YCoCg result
has greater seed variance. The final colour choice requires a second HDRI with
a concentrated sun and a path-traced lighting comparison.

## Loss and HDR-transfer ablation

Three-seed H128/L4 YCoCg results show that MSE is better than Huber for this
hash-grid field. Unlike the Fourier network in the original paper, MSE remains
stable around the sparse high-radiance pixels and improves every aggregate
metric: FP8-QAT linear PSNR rises from 42.49±0.29 dB to 43.51±0.15 dB and
exposure PSNR from 31.89±0.18 dB to 32.75±0.08 dB.

Replacing `log(x + eps)` with `log1p(x)` raises linear PSNR further to 48.43 dB
and cuts highlight chroma error nearly in half, but matched-exposure images show
flattened shadows and a magenta shift in orange/red graffiti. It also worsens
mean relative error from 10.62% to 19.01% and mean exposure PSNR from 32.75 dB
to 31.29 dB. A pure `log1p` target is therefore rejected despite its strong
radiometric scores.

The selected objective retains log encoding and MSE, then adds weight 0.05 of
linear-radiance MSE over samples at or above the reference image's 95th
luminance percentile. Values are normalized by the reference's peak radiance
before this auxiliary loss. All statistics below are three-seed FP8-QAT means.

| Objective | Linear PSNR | log1p PSNR | Exposure PSNR | Solid-angle luminance PSNR | Relative error | Highlight chroma MAE |
|---|---:|---:|---:|---:|---:|---:|
| log MSE | 43.51±0.15 | 37.75±0.15 | 32.75±0.08 | 41.99±0.09 | **10.62%** | 0.01668 |
| log MSE + highlight 0.05 | **43.97±0.18** | **37.88±0.16** | **32.79±0.08** | **42.45±0.18** | 10.65% | **0.01461** |
| log1p MSE | 48.43 | 40.95 | 31.29 | 46.54 | 19.01% | 0.00856 |

The hybrid loss improves linear and solid-angle luminance PSNR by about 0.46 dB
and reduces highlight chroma error by 12.4% without the visual regressions of
`log1p`. Weight 0.10 increases radiometric scores only marginally over 0.05,
while oversaturating highlights and shifting shadows, so 0.05 is the new
training baseline.

## Next ablations

1. Validate YCoCg versus YUV and the hybrid loss on a sun-dominant HDRI and
   path-traced lighting.
2. Compare uniform sampling with solid-angle and highlight-aware sampling.
3. Sweep hash levels, features, resolutions, and per-level capacities while
   preferring MLP input dimensions compatible with E4M3 matrix kernels.
4. Integrate only the nondominated architectures into the renderer and measure
   actual frame time and memory before selecting the final Pareto frontier.

Every stage retains linear, log-domain, exposure-stack, solid-angle-weighted
luminance and energy, highlight chromaticity, deployment-byte, and MAC metrics.
