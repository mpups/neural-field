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
