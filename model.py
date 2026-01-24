"""PyTorch Neural Image Field (NIF) Model Definition"""

import torch
import torch.nn as nn
import numpy as np


class HashGridEncoding(nn.Module):
    """Multi-resolution hash grid encoding from Instant-NGP.
    
    Learnable spatial encoding that maps 2D coordinates to feature vectors
    using multiple resolution levels with hash-based lookups.
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
        self.hashmap_size = 2 ** log2_hashmap_size
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        
        # Compute resolution growth factor
        if n_levels > 1:
            self.growth_factor = np.exp(
                (np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)
            )
        else:
            self.growth_factor = 1.0
        
        # Precompute resolutions for each level
        self.register_buffer(
            "resolutions",
            torch.tensor([
                int(base_resolution * (self.growth_factor ** l))
                for l in range(n_levels)
            ], dtype=torch.float32)
        )
        
        # Learnable hash tables for each level
        self.embeddings = nn.ModuleList([
            nn.Embedding(self.hashmap_size, n_features)
            for _ in range(n_levels)
        ])
        
        # Initialize with small random values
        for emb in self.embeddings:
            nn.init.uniform_(emb.weight, -1e-4, 1e-4)
        
        # Prime numbers for spatial hashing
        self.register_buffer("primes", torch.tensor([1, 2654435761], dtype=torch.int64))
    
    def hash_coords(self, coords):
        """Hash 2D integer coordinates to table indices.
        
        Args:
            coords: (N, 2) integer coordinates
        
        Returns:
            (N,) hash indices
        """
        # XOR-based spatial hash
        hashed = coords[:, 0].long() * self.primes[0]
        hashed = hashed ^ (coords[:, 1].long() * self.primes[1])
        return hashed % self.hashmap_size
    
    def forward(self, uv):
        """Encode UV coordinates using multi-resolution hash grids.
        
        Args:
            uv: (N, 2) coordinates in [0, 1]
        
        Returns:
            (N, n_levels * n_features) encoded features
        """
        outputs = []
        
        for level, resolution in enumerate(self.resolutions):
            # Scale coordinates to grid resolution
            scaled = uv * resolution
            
            # Get integer corners and fractional offsets
            floor_coords = torch.floor(scaled).int()
            frac = scaled - floor_coords.float()
            
            # 4 corners of the grid cell: (0,0), (0,1), (1,0), (1,1)
            corners = []
            for dy in [0, 1]:
                for dx in [0, 1]:
                    corner = floor_coords + torch.tensor([[dy, dx]], device=uv.device)
                    corners.append(corner)
            
            # Hash corners and lookup features
            corner_features = []
            for corner in corners:
                indices = self.hash_coords(corner)
                features = self.embeddings[level](indices)
                corner_features.append(features)
            
            # Bilinear interpolation
            # f00, f01, f10, f11 correspond to corners (0,0), (0,1), (1,0), (1,1)
            f00, f01, f10, f11 = corner_features
            
            # Interpolation weights
            u_frac = frac[:, 1:2]  # x direction
            v_frac = frac[:, 0:1]  # y direction
            
            # Bilinear interpolation
            f0 = f00 * (1 - u_frac) + f01 * u_frac  # bottom edge
            f1 = f10 * (1 - u_frac) + f11 * u_frac  # top edge
            interpolated = f0 * (1 - v_frac) + f1 * v_frac
            
            outputs.append(interpolated)
        
        return torch.cat(outputs, dim=1)
    
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
