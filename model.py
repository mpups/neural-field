"""PyTorch Neural Image Field (NIF) Model Definition"""

import torch
import torch.nn as nn


class NIFModel(nn.Module):
    """Neural Image Field Model - MLP for image compression/reconstruction.

    Uses ReLU activation with positional encoding. For >= 4 layers, includes
    input skip connections at the midpoint.
    """

    def __init__(
        self,
        input_dim,
        layer_size,
        num_layers,
        color_matrix=None,
    ):
        super().__init__()

        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2, got {num_layers}")

        self.use_skip = num_layers >= 4

        if self.use_skip and layer_size <= input_dim:
            raise ValueError(
                f"layer_size ({layer_size}) must be > input_dim ({input_dim}) "
                f"to accommodate skip connections"
            )

        self.skip_index = num_layers // 2 if self.use_skip else -1
        concat_dim = input_dim if self.use_skip else 0

        # Build all layers
        layers = []
        for i in range(num_layers):
            is_first = i == 0
            is_last = i == num_layers - 1
            is_skip_layer = i == self.skip_index

            # Determine input/output dimensions
            if is_first:
                in_dim = input_dim
            elif i == self.skip_index:
                in_dim = layer_size  # After skip concat
            else:
                in_dim = layer_size

            if is_last:
                out_dim = 3
            elif is_skip_layer:
                out_dim = layer_size - concat_dim  # Leave room for concat
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
        input_for_skip = x

        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Skip connection: concat input after the skip layer
            if i == self.skip_index:
                x = torch.relu(x)
                x = torch.cat([x, input_for_skip], dim=1)
            elif i < len(self.layers) - 1:
                x = torch.relu(x)

        if self.color_matrix is not None:
            x = torch.nn.functional.linear(x, self.color_matrix)

        return x
