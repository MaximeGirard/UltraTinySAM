# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Some utilities for backbones, in particular for windowing"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Used only for training, other version used for export
def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.
    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if (pad_h > 0) or (pad_w > 0):
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h)) 
    
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)

def window_partition_listed(x, window_size, x_hard_shape):
    """
    Partition into non-overlapping windows without padding.
    Args:
        x (tensor): input tokens with shape [B, H, W, C].
        window_size (int): window size.
        x_hard_size (list or tuple): hardcoded size of x, same as list(x.shape).
    Returns:
        window_list: list of tensors with shape [B, window_size, window_size, C]
        (Hp, Wp): original height and width (same as H, W since no padding)
    """
    B, H, W, C = x_hard_shape  # Fully static

    num_h = H // window_size
    num_w = W // window_size

    # Partition into windows
    x = x.view(B, num_h, window_size, num_w, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5)  # [B, num_h, num_w, window_size, window_size, C]

    # Use precomputed num_h and num_w — no x.shape call
    window_list = [
        x[:, i, j, :, :, :]  # shape: [B, window_size, window_size, C]
        for i in range(num_h)
        for j in range(num_w)
    ]

    return window_list, (H, W)

def window_partition_tensor(x, window_size, x_hard_shape):
    """
    Partition tensor into non-overlapping windows without padding.
    Args:
        x (tensor): input tensor of shape [B, H, W, C]
        window_size (int): window size
        x_hard_shape (tuple): hardcoded shape [B, H, W, C]
    Returns:
        windows: tensor of shape [B, num_windows, window_size, window_size, C]
    """
    B, H, W, C = x_hard_shape
    assert H % window_size == 0, f"H {H} is not divisable by window_size {window_size}"
    assert W % window_size == 0, f"W {W} is not divisable by window_size {window_size}"
    num_h = H // window_size
    num_w = W // window_size

    # print("[HARDCODING] window_partition_tensor | x: ", x_hard_shape)
    # print("[HARDCODING] window_partition_tensor | window_size: ", window_size)
    # print("[HARDCODING] window_partition_tensor | num_h, num_w: ", num_h, num_w)

    x = x.view(B, num_h, window_size, num_w, window_size, C)                 # [B, num_h, wh, num_w, ww, C]
    x = x.permute(0, 1, 3, 2, 4, 5)                                          # [B, num_h, num_w, wh, ww, C]
    windows = x.reshape(B, num_h * num_w, window_size, window_size, C)      # [B, num_windows, wh, ww, C]
    
    return windows, (H, W)

def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.
    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    # print("windows.shape", windows.shape)
    x = windows.reshape(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    # print(f"window_unpartition: {x.shape}")
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        #print(f"H, W: {H}, {W}")
        x = x[:, :H, :W, :]
    return x

def window_unpartition_listed(windows, window_size, B, H, W, dim_out):
    """
    Reconstruct original tensor from a list of window tensors without padding.
    Args:
        windows (list): list of tensors with shape [B, window_size, window_size, C].
        window_size (int): window size.
        x_hard_size (list or tuple): original tensor shape [B, H, W, C].
    Returns:
        x: tensor of shape [B, H, W, C].
    """

    num_h = H // window_size
    num_w = W // window_size
    num_windows = num_h * num_w

    # Stack windows into [B, num_windows, window_size, window_size, C]
    x = torch.stack(windows, dim=1)

    # Reshape and permute to recover original layout
    x = x.view(B, num_h, num_w, window_size, window_size, dim_out)         # [B, num_h, num_w, wh, ww, C]
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, dim_out)              # [B, H, W, C]

    return x

def window_unpartition_tensor(windows, window_size, B, H, W, dim_out):
    """
    Reconstruct original tensor from a tensor of windowed patches without padding.

    Args:
        windows (tensor): shape [B, num_windows, window_size, window_size, C]
        window_size (int): size of each window
        B (int): batch size
        H (int): original height
        W (int): original width
        dim_out (int): number of output channels (C)

    Returns:
        x: tensor of shape [B, H, W, C]
    """
    assert H % window_size == 0, f"H {H} is not divisable by window_size {window_size}"
    assert W % window_size == 0, f"W {W} is not divisable by window_size {window_size}"
    num_h = H // window_size
    num_w = W // window_size

    x = windows.view(B, num_h, num_w, window_size, window_size, dim_out)     # [B, num_h, num_w, wh, ww, C]
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, dim_out)                # [B, H, W, C]

    return x

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, ...] = (7, 7),
        stride: Tuple[int, ...] = (4, 4),
        padding: Tuple[int, ...] = (3, 3),
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x
