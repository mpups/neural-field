# Neural Image Fields

A neural image field (NIF) learns to represent a 2D image as a continuous function mapping pixel coordinates to colour values.
A small neural network is trained as a function approximator using positional encoding (Fourier features) to capture high-frequency detail.
With careful choice of parameters this is a form of neural image compression.

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
