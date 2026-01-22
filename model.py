"""PyTorch Neural Image Field (NIF) Model Definition"""

import torch
import torch.nn as nn


class NIFModel(nn.Module):
    """Neural Image Field Model - MLP for image compression/reconstruction.

    Uses ReLU activation with positional encoding and input skip connections.
    """

    def __init__(
        self,
        input_dim,
        layer_size,
        num_layers,
        color_matrix=None,
    ):
        super().__init__()

        self.layer_size = layer_size
        self.input_dim = input_dim

        if num_layers < 4:
            raise ValueError(f"num_layers must be >= 4, got {num_layers}")

        if layer_size <= input_dim:
            raise ValueError(
                f"layer_size ({layer_size}) must be > input_dim ({input_dim}) "
                f"to accommodate skip connections"
            )

        half_size = num_layers // 2
        concat_dim = input_dim

        # Front-end layers
        front_end = []
        for l in range(half_size - 1):
            layer = nn.Linear(
                layer_size if l > 0 else input_dim, layer_size, bias=False
            )
            nn.init.xavier_uniform_(layer.weight)
            front_end.append(layer)
            front_end.append(nn.ReLU())

        # Last layer of front-end reduces dimension for concatenation
        last_front = nn.Linear(layer_size, layer_size - concat_dim, bias=False)
        nn.init.xavier_uniform_(last_front.weight)
        front_end.append(last_front)
        front_end.append(nn.ReLU())

        self.front_end = nn.Sequential(*front_end)

        # Back-end layers
        back_end = []
        for l in range(half_size - 1):
            layer = nn.Linear(
                layer_size, layer_size, bias=False if l < half_size - 2 else True
            )
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            back_end.append(layer)
            back_end.append(nn.ReLU())

        # Penultimate layer (has bias)
        penultimate = nn.Linear(layer_size, layer_size, bias=True)
        nn.init.xavier_uniform_(penultimate.weight)
        nn.init.zeros_(penultimate.bias)
        back_end.append(penultimate)
        back_end.append(nn.ReLU())

        # Final output layer (3 channels for RGB)
        final = nn.Linear(layer_size, 3, bias=True)
        nn.init.xavier_uniform_(final.weight)
        nn.init.zeros_(final.bias)
        back_end.append(final)

        self.back_end = nn.Sequential(*back_end)

        # Color space conversion layer (if needed)
        if color_matrix is not None:
            self.register_buffer("color_matrix", torch.from_numpy(color_matrix).float())
        else:
            self.color_matrix = None

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, 3) - RGB values
        """
        input_for_skip = x

        x = self.front_end(x)
        x = torch.cat([x, input_for_skip], dim=1)
        x = self.back_end(x)

        if self.color_matrix is not None:
            x = torch.nn.functional.linear(x, self.color_matrix)

        return x
