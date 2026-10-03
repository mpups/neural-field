"""PyTorch Neural Image Field (NIF) Model Definition"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nvfp4 import fake_quantize_nvfp4


class HashGridEncoding(nn.Module):
    """Multi-resolution hash grid encoding from Instant-NGP.

    Optimized implementation with:
    - FP16 hash tables
    - Vectorized computation across all levels and corners
    - Fused bilinear interpolation
    """

    def __init__(
        self,
        n_levels=16,
        n_features=2,
        log2_hashmap_size=19,
        level_log2_hashmap_sizes=None,
        base_resolution=16,
        max_resolution=2048,
        layout="hash",
    ):
        super().__init__()

        if layout not in {"hash", "hybrid"}:
            raise ValueError(f"Unknown hash-grid layout: {layout}")
        if level_log2_hashmap_sizes is None:
            level_log2_hashmap_sizes = [log2_hashmap_size] * n_levels
        if len(level_log2_hashmap_sizes) != n_levels:
            raise ValueError("level_log2_hashmap_sizes must contain one value per level")
        if any(size < 1 or size > 30 for size in level_log2_hashmap_sizes):
            raise ValueError("hash-grid log2 sizes must be between 1 and 30")

        self.n_levels = n_levels
        self.n_features = n_features
        self.log2_hashmap_size = log2_hashmap_size
        self.hashmap_size = 1 << log2_hashmap_size
        self.level_log2_hashmap_sizes = tuple(level_log2_hashmap_sizes)
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.layout = layout

        # Compute resolution growth factor
        if n_levels > 1:
            self.growth_factor = np.exp(
                (np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)
            )
        else:
            self.growth_factor = 1.0

        # Precompute resolutions for each level: (L,)
        resolutions = torch.tensor(
            [base_resolution * (self.growth_factor**l) for l in range(n_levels)],
            dtype=torch.float32,
        )
        self.register_buffer("resolutions", resolutions)

        hash_table_sizes = [1 << size for size in self.level_log2_hashmap_sizes]
        dense_strides = [int(resolution) + 2 for resolution in resolutions]
        level_sizes = []
        level_is_dense = []
        for hash_size, dense_stride in zip(hash_table_sizes, dense_strides):
            dense_size = dense_stride * dense_stride
            use_dense = layout == "hybrid" and dense_size <= hash_size
            level_sizes.append(dense_size if use_dense else hash_size)
            level_is_dense.append(use_dense)
        level_offsets = np.cumsum([0] + level_sizes[:-1]).tolist()

        # Single packed embedding containing variable-sized per-level tables.
        # It remains FP32 during optimization; autocast controls compute precision.
        total_entries = sum(level_sizes)
        self.embedding = nn.Embedding(total_entries, n_features)
        nn.init.uniform_(self.embedding.weight, -1e-4, 1e-4)

        self.register_buffer("level_offsets", torch.tensor(level_offsets, dtype=torch.int64))
        self.register_buffer(
            "hash_masks",
            torch.tensor([size - 1 for size in hash_table_sizes], dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "dense_strides",
            torch.tensor(
                [stride if use_dense else 0
                 for stride, use_dense in zip(dense_strides, level_is_dense)],
                dtype=torch.int64,
            ),
            persistent=False,
        )
        self.level_sizes = tuple(level_sizes)

        # Corner offsets for vectorized corner computation: (4, 2)
        corner_offsets = torch.tensor(
            [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.int32
        )
        self.register_buffer("corner_offsets", corner_offsets)

        # Primes for spatial hashing
        self.register_buffer("primes", torch.tensor([1, 2654435761], dtype=torch.int64))

    def forward(self, uv):
        """Encode UV coordinates using multi-resolution hash grids.

        Fully vectorized over levels and corners.

        Args:
            uv: (N, 2) coordinates in [0, 1]

        Returns:
            (N, n_levels * n_features) encoded features
        """
        N = uv.shape[0]
        L = self.n_levels
        F = self.n_features

        # Scale UV to all resolutions at once: (N, 2) * (L,) -> (N, L, 2)
        # resolutions: (L,) -> (1, L, 1)
        scaled = uv.unsqueeze(1) * self.resolutions.view(1, L, 1)

        # Get integer floor and fractional parts: (N, L, 2)
        floor_coords = torch.floor(scaled).int()
        frac = scaled - floor_coords.float()

        # Compute all 4 corners for all levels: (N, L, 4, 2)
        # corner_offsets: (4, 2) -> (1, 1, 4, 2)
        corners = floor_coords.unsqueeze(2) + self.corner_offsets.view(1, 1, 4, 2)

        # Hash all corners: (N, L, 4)
        # Spatial hash: (x * p1) XOR (y * p2) mod T
        hashed = (corners[..., 0].long() * self.primes[0]) ^ (
            corners[..., 1].long() * self.primes[1]
        )
        hashed = hashed & self.hash_masks.view(1, L, 1)
        dense = (
            corners[..., 0].long() * self.dense_strides.view(1, L, 1)
            + corners[..., 1].long()
        )
        indices = torch.where(self.dense_strides.view(1, L, 1) > 0, dense, hashed)

        # Add level offsets to get global indices: (N, L, 4)
        # level_offsets: (L,) -> (1, L, 1)
        global_indices = indices + self.level_offsets.view(1, L, 1)

        # Lookup features for all corners: (N, L, 4) -> (N*L*4, F) -> (N, L, 4, F)
        flat_indices = global_indices.reshape(-1)
        flat_features = self.embedding(flat_indices)
        corner_features = flat_features.view(N, L, 4, F)

        # Fused bilinear interpolation
        # frac: (N, L, 2) -> extract u_frac (x) and v_frac (y)
        u_frac = frac[..., 1:2]  # (N, L, 1) - x direction
        v_frac = frac[..., 0:1]  # (N, L, 1) - y direction

        # corners: [0]=f00, [1]=f01, [2]=f10, [3]=f11
        # f00 at (0,0), f01 at (0,1), f10 at (1,0), f11 at (1,1)
        f00 = corner_features[:, :, 0, :]  # (N, L, F)
        f01 = corner_features[:, :, 1, :]
        f10 = corner_features[:, :, 2, :]
        f11 = corner_features[:, :, 3, :]

        # Bilinear interpolation: lerp in x, then lerp in y
        f0 = f00 * (1 - u_frac) + f01 * u_frac  # bottom edge
        f1 = f10 * (1 - u_frac) + f11 * u_frac  # top edge
        interpolated = f0 * (1 - v_frac) + f1 * v_frac  # (N, L, F)

        # Flatten levels: (N, L, F) -> (N, L*F)
        return interpolated.reshape(N, L * F)

    @property
    def output_dim(self):
        """Output dimension of the encoding."""
        return self.n_levels * self.n_features


class NIFModel(nn.Module):
    """Neural Image Field Model - MLP for image compression/reconstruction.

    Uses learnable hash grid encoding integrated in the model.
    """

    def __init__(
        self,
        input_dim,
        layer_size,
        num_layers,
        color_matrix=None,
        encoding=None,
    ):
        """
        Args:
            input_dim: Input dimension (2 for raw UV)
            layer_size: Hidden layer width
            num_layers: Number of MLP layers
            color_matrix: Optional color space conversion matrix
            encoding: HashGridEncoding module
        """
        super().__init__()

        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2, got {num_layers}")

        self.encoding = encoding

        # Determine actual MLP input dimension
        if encoding is not None:
            mlp_input_dim = encoding.output_dim
        else:
            mlp_input_dim = input_dim

        self.use_skip = num_layers >= 4

        if self.use_skip and layer_size <= mlp_input_dim:
            raise ValueError(
                f"layer_size ({layer_size}) must be > mlp_input_dim ({mlp_input_dim}) "
                f"to accommodate skip connections"
            )

        self.skip_index = num_layers // 2 if self.use_skip else -1
        concat_dim = mlp_input_dim if self.use_skip else 0

        # Build MLP layers
        layers = []
        for i in range(num_layers):
            is_first = i == 0
            is_last = i == num_layers - 1
            is_skip_layer = i == self.skip_index

            if is_first:
                in_dim = mlp_input_dim
            elif i == self.skip_index:
                in_dim = layer_size
            else:
                in_dim = layer_size

            if is_last:
                out_dim = 3
            elif is_skip_layer:
                out_dim = layer_size - concat_dim
            else:
                out_dim = layer_size

            layer = nn.Linear(in_dim, out_dim, bias=is_last or i >= num_layers - 2)
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

            layers.append(layer)

        self.layers = nn.ModuleList(layers)
        self.nvfp4_qat = False
        self.nvfp4_static_activation_scales = False
        self.register_buffer(
            "nvfp4_activation_global_scales",
            torch.ones(len(layers), dtype=torch.float32),
            persistent=False,
        )

        # Color space conversion
        if color_matrix is not None:
            self.register_buffer("color_matrix", torch.from_numpy(color_matrix).float())
        else:
            self.color_matrix = None

    def set_nvfp4_qat(self, enabled=True, activation_global_scales=None):
        if activation_global_scales is not None:
            scales = torch.as_tensor(
                activation_global_scales,
                dtype=torch.float32,
                device=self.nvfp4_activation_global_scales.device,
            )
            if scales.shape != self.nvfp4_activation_global_scales.shape:
                raise ValueError("one NVFP4 activation global scale is required per MLP layer")
            if not torch.isfinite(scales).all() or not torch.all(scales > 0):
                raise ValueError("NVFP4 activation global scales must be finite and positive")
            self.nvfp4_activation_global_scales.copy_(scales)
            self.nvfp4_static_activation_scales = True
        self.nvfp4_qat = enabled

    def forward(self, x):
        # Apply encoding if present
        if self.encoding is not None:
            x = self.encoding(x)

        input_for_skip = x

        for i, layer in enumerate(self.layers):
            if self.nvfp4_qat:
                activation_scale = (
                    self.nvfp4_activation_global_scales[i]
                    if self.nvfp4_static_activation_scales else None
                )
                quantized_input = fake_quantize_nvfp4(x, (1, 16), activation_scale)
                quantized_weight = fake_quantize_nvfp4(layer.weight, (16, 16))
                x = F.linear(quantized_input, quantized_weight, layer.bias)
            else:
                x = layer(x)

            if i == self.skip_index:
                x = torch.relu(x)
                x = torch.cat([x, input_for_skip], dim=1)
            elif i < len(self.layers) - 1:
                x = torch.relu(x)

        if self.color_matrix is not None:
            x = torch.nn.functional.linear(x, self.color_matrix)

        return x


def model_from_config(config):
    """Construct a model from checkpoint metadata."""
    hash_config = config.get("hashgrid_config", {})
    encoding = HashGridEncoding(
        n_levels=hash_config.get("n_levels", 16),
        n_features=hash_config.get("n_features", 2),
        log2_hashmap_size=hash_config.get("log2_hashmap_size", 19),
        level_log2_hashmap_sizes=hash_config.get("level_log2_hashmap_sizes"),
        base_resolution=hash_config.get("base_resolution", 16),
        max_resolution=hash_config.get("max_resolution", 2048),
        layout=hash_config.get("layout", "hash"),
    )
    model = NIFModel(
        input_dim=config.get("input_dim", 2),
        layer_size=config["layer_size"],
        num_layers=config["num_layers"],
        color_matrix=config.get("color_matrix"),
        encoding=encoding,
    )
    quantization = config.get("quantization", {})
    model.set_nvfp4_qat(
        quantization.get("mlp") == "nvfp4",
        quantization.get("activation_global_scales"),
    )
    return model


def calibrate_nvfp4_activation_scales(model, inputs, batch_size=65536):
    """Measure fixed activation tensor scales for direct producer-side packing."""
    maxima = torch.zeros(len(model.layers), device=inputs.device, dtype=torch.float32)
    hooks = []
    for index, layer in enumerate(model.layers):
        def capture(_layer, arguments, layer_index=index):
            maxima[layer_index] = torch.maximum(
                maxima[layer_index], arguments[0].detach().float().abs().amax()
            )
        hooks.append(layer.register_forward_pre_hook(capture))

    was_training = model.training
    was_quantized = model.nvfp4_qat
    model.set_nvfp4_qat(False)
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, inputs.shape[0], batch_size):
                model(inputs[start:start + batch_size])
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)
        model.set_nvfp4_qat(was_quantized)

    denominator = 448.0 * 6.0
    return torch.where(maxima > 0, maxima / denominator, torch.ones_like(maxima))


def load_model_checkpoint(path, device="cpu", quantized=None):
    """Load a model checkpoint from either a model directory or .pt file."""
    checkpoint_path = os.path.join(path, "model.pt") if os.path.isdir(path) else path
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_config" not in checkpoint:
        raise ValueError(f"Checkpoint has no model_config: {checkpoint_path}")
    model = model_from_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    if quantized is not None:
        model.set_nvfp4_qat(quantized)
    return model.to(device), checkpoint


def compile_model(model, mode="reduce-overhead"):
    """Compile model with torch.compile for optimized execution.

    Args:
        model: PyTorch model
        mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')

    Returns:
        Compiled model (or original if compilation unavailable)
    """
    if hasattr(torch, "compile"):
        try:
            compiled = torch.compile(model, mode=mode)
            print(f"Model compiled with torch.compile (mode={mode})")
            return compiled
        except Exception as e:
            print(f"torch.compile failed: {e}, using uncompiled model")
            return model
    else:
        print("torch.compile not available, using uncompiled model")
        return model
