"""Fallbacks for the torch_scatter operations used by voxel aggregation."""

import torch

try:
    from torch_scatter import scatter_add, scatter_max
except (ImportError, OSError):

    def _expanded_index(index, src, dim):
        if index.ndim == 1:
            shape = [1] * src.ndim
            shape[dim] = index.numel()
            index = index.reshape(shape)
        return index.expand_as(src)

    def _output_shape(src, index, dim, dim_size):
        if dim_size is None:
            dim_size = int(index.max()) + 1 if index.numel() else 0
        shape = list(src.shape)
        shape[dim] = dim_size
        return shape

    def scatter_add(src, index, dim=-1, out=None, dim_size=None):
        dim %= src.ndim
        expanded_index = _expanded_index(index, src, dim)
        if out is None:
            out = src.new_zeros(_output_shape(src, index, dim, dim_size))
        return out.scatter_add_(dim, expanded_index, src)

    def scatter_max(src, index, dim=-1, out=None, dim_size=None):
        dim %= src.ndim
        expanded_index = _expanded_index(index, src, dim)
        if out is None:
            shape = _output_shape(src, index, dim, dim_size)
            out = torch.full(
                shape, -torch.inf, dtype=src.dtype, device=src.device
            )
        out.scatter_reduce_(
            dim, expanded_index, src, reduce="amax", include_self=True
        )
        return out, None
