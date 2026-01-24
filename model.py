"""PyTorch Neural Image Field (NIF) Model Definition"""

import torch
import torch.nn as nn
import numpy as np


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
        base_resolution=16,
        max_resolution=2048,
    ):
        super().__init__()

        self.n_levels = n_levels
        self.n_features = n_features
        self.log2_hashmap_size = log2_hashmap_size
        self.hashmap_size = 2**log2_hashmap_size
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution

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

        # Single unified embedding table for all levels
        # Shape: (L * T, F) - kept as FP32, autocast handles FP16 compute
        total_entries = n_levels * self.hashmap_size
        self.embedding = nn.Embedding(total_entries, n_features)
        nn.init.uniform_(self.embedding.weight, -1e-4, 1e-4)

        # Level offsets for indexing into unified table
        level_offsets = torch.arange(n_levels, dtype=torch.int64) * self.hashmap_size
        self.register_buffer("level_offsets", level_offsets)

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
        hashed = hashed % self.hashmap_size

        # Add level offsets to get global indices: (N, L, 4)
        # level_offsets: (L,) -> (1, L, 1)
        global_indices = hashed + self.level_offsets.view(1, L, 1)

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

    Supports Fourier positional encoding (input preprocessing) or learnable
    hash grid encoding (integrated in model).
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
            input_dim: Input dimension (2 for raw UV with hash encoding,
                       or 4*embedding_dim for Fourier encoding)
            layer_size: Hidden layer width
            num_layers: Number of MLP layers
            color_matrix: Optional color space conversion matrix
            encoding: Optional HashGridEncoding module (for hash grid mode)
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

        # Color space conversion
        if color_matrix is not None:
            self.register_buffer("color_matrix", torch.from_numpy(color_matrix).float())
        else:
            self.color_matrix = None

    def forward(self, x):
        # Apply encoding if present
        if self.encoding is not None:
            x = self.encoding(x)

        input_for_skip = x

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == self.skip_index:
                x = torch.relu(x)
                x = torch.cat([x, input_for_skip], dim=1)
            elif i < len(self.layers) - 1:
                x = torch.relu(x)

        if self.color_matrix is not None:
            x = torch.nn.functional.linear(x, self.color_matrix)

        return x


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
