"""
Factorized Positional Embedding
© Dehghani et al. / Google DeepMind 
"""

import torch
import torch.nn as nn


class NaViTEmbedding(nn.Module):
    """
    The factorized positional embedding for any aspect ratio and resolution.
    """
    def __init__(self, d_model: int = 512, max_len: int = 1000, patch_size: int = 512):
        super().__init__()
        self.patch_size = patch_size
        self.pos_embed_x = nn.Parameter(torch.randn(max_len, d_model))
        self.pos_embed_y = nn.Parameter(torch.randn(max_len, d_model))
        self.pos_embed_z = nn.Parameter(torch.randn(max_len, d_model))

    def get_patch_coords(self, coords_tensor: torch.Tensor):
        """
        Given a coordinates as tensors, return a them as a list.
        """
        # Ensure tensor has at least 2 dimensions
        if coords_tensor.dim() == 1:
            coords_tensor = coords_tensor.unsqueeze(0)
        
        # Squeeze the coordinate tensor in the batch dimension
        coords_tensor = coords_tensor.squeeze(0)
        
        # Handle case where tensor becomes 0-d after squeezing
        if coords_tensor.dim() == 0:
            return [(0, 0, 0)]  # Default coordinate
        
        # Handle single coordinate case
        if coords_tensor.dim() == 1:
            if len(coords_tensor) >= 3:
                return [(int(coords_tensor[2].item()), int(coords_tensor[0].item()), int(coords_tensor[1].item()))]
            else:
                return [(0, 0, 0)]  # Default coordinate
        
        # Extract x, y, and z coordinates into a tuple (z, x, y)
        patch_coords = [
            (
                int(z.item()),
                int(x.item()),
                int(y.item()),
            ) for z, x, y in coords_tensor
        ]
        return patch_coords

    def get_patch_indices(self, patch_coords: list):
        """
        Create a minimum bounding rectangle around a polygon
        made of patches and extract their postion as indices.
        """
        # Get the predefined patch size
        patch_size = self.patch_size

        # Extract the unique levels
        coords = torch.tensor(patch_coords)
        z_levels = coords[:, 0].unique()

        # Initialize a list to hold the results
        indices = []

        # Process each level separately
        for z in z_levels:
            # Filter coordinates for the current level
            level_coords = coords[coords[:, 0] == z][:, 1:]

            # Extract min and max coordinates for the current level
            x_min, y_min = level_coords.min(dim=0).values
            x_max, y_max = level_coords.max(dim=0).values

            # Create grid ranges for the current level
            x_range = torch.arange(x_min, x_max + patch_size, patch_size)
            y_range = torch.arange(y_min, y_max + patch_size, patch_size)

            # Create a mapping of coordinates to indices for the current level
            x_indices = {x.item(): i for i, x in enumerate(x_range)}
            y_indices = {y.item(): i for i, y in enumerate(y_range)}

            # Calculate (x, y) indices for the current level
            level_indices = [
                (z.item(), x_indices[x.item()], y_indices[y.item()])
                for x, y in level_coords
            ]

            # Append the results for the current level to the overall list
            indices.extend(level_indices)

        return indices

    def find_patch_index(self, patch_indices: list, target_index: tuple):
        """
        Find the index of the target patch in the list of patch indices.
        """
        try:
            return patch_indices.index(target_index)
        except ValueError:
            return -1

    def forward(self, x: torch.Tensor, coords_tensor: torch.Tensor):
        patch_coords = self.get_patch_coords(coords_tensor)
        patch_indices = self.get_patch_indices(patch_coords)

        patch_indices = torch.tensor(
            patch_indices,
            dtype=torch.long,
            device=x.device,
        )

        z_indices = patch_indices[:, 0]
        x_indices = patch_indices[:, 1]
        y_indices = patch_indices[:, 2]

        pos_emb_x = self.pos_embed_x[x_indices]
        pos_emb_y = self.pos_embed_y[y_indices]
        pos_emb_z = self.pos_embed_z[z_indices]

        return x + pos_emb_x + pos_emb_y + pos_emb_z
