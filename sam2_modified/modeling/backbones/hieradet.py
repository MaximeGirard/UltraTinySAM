# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from iopath.common.file_io import g_pathmgr

from sam2.modeling.backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
    window_partition_listed,
    window_unpartition_listed,
    window_partition_tensor,
    window_unpartition_tensor
)

from sam2.modeling.sam2_utils import DropPath, MLP


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
        export: bool = False,
    ):
        super().__init__()

        self.export = export
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor, hard_x_shape = None) -> torch.Tensor:
        if self.export:
            # x: [1, N, w, w, C]
            # print("[MULTISCALEATTENTION] x shape:", x.shape)
            B, N, H, W, C = x.shape  # B is always 1

            # Flatten windows into sequence: [1, N, H, W, C] → [1, N * H, W, C]
            x = x.reshape(B, N * H, W, C)  # Still 4D, compatible with qkv

            # qkv projection (expects 4D): output shape will be [1, N * H, W, 3 * dim]
            qkv = self.qkv(x)  # [1, N * H, W, 3 * dim_out]

            # Reshape to [1, N * H * W, 3, nH, C]
            qkv = qkv.reshape(B, N * H * W, 3, self.num_heads, -1)
            # print("[MULTISCALEATTENTION] qkv shape:", qkv.shape)

            # Split q, k, v
            q, k, v = torch.unbind(qkv, dim=2)

            # Q pooling if required
            if self.q_pool:
                # Reshape q to [1, H, W, C] using known hard shape before pooling
                q = q.reshape(B, hard_x_shape[1], hard_x_shape[2], -1)  # [1, H, W, C]
                q = do_pool(q, self.q_pool)
                H_, W_ = q.shape[1:3]  # Update H, W after pooling
                q = q.reshape(B, H_ * W_, self.num_heads, -1)  # [1, new_tokens, nH, C]
                # print("[MULTISCALEATTENTION] q shape after pooling:", q.shape)

            # Transpose for attention: [1, tokens, nH, C] → [1, nH, tokens, C]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Scaled dot-product attention
            x = F.scaled_dot_product_attention(q, k, v)
            # print("[MULTISCALEATTENTION] x shape after attention:", x.shape)

            # Transpose back and reshape: [1, nH, tokens, C] → [1, tokens, nH * C]
            x = x.transpose(1, 2).reshape(B, -1, self.num_heads * x.shape[-1])

            # Final projection
            # Need 4d input for hardware
            x = self.proj(x.unsqueeze(1)).squeeze(1)  # [1, tokens, dim_out]

            # Reshape back to [1, N, w, w, dim_out]
            x = x.reshape(B, N, H, W, -1)
            # print("[MULTISCALEATTENTION] x shape after projection:", x.shape)

            return x
        else:
            B, H, W, _ = x.shape
            # qkv with shape (B, H * W, 3, nHead, C)
            qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
            # q, k, v with shape (B, H * W, nheads, C)
            q, k, v = torch.unbind(qkv, 2)

            # Q pooling (for downsample at stage changes)
            if self.q_pool:
                q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
                H, W = q.shape[1:3]  # downsampled shape
                q = q.reshape(B, H * W, self.num_heads, -1)

            # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
            x = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
            )
            
            # Transpose back
            x = x.transpose(1, 2)
            x = x.reshape(B, H, W, -1)

            x = self.proj(x)
            
            return x


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
        export: bool = False,
        hard_x_shape: bool = False,
    ):
        super().__init__()

        self.export = export
        
        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)

        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(
                kernel_size=q_stride, stride=q_stride, ceil_mode=False
            )

        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
            export=export,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=act_layer,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)
            
        self.hard_x_shape = hard_x_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.export:
            # print("[MULTISCALEBLOCK] known hard x shape:", self.hard_x_shape)
            shortcut = x  # B, H, W, C
            x = self.norm1(x)

            # Skip connection
            if self.dim != self.dim_out:
                shortcut = do_pool(self.proj(x), self.pool)

            # Window partition
            window_size = self.window_size
            # print("[MULTISCALEBLOCK] Window size:", window_size)
            if window_size > 0: # For global att layers
                B, H, W = self.hard_x_shape[0], self.hard_x_shape[1], self.hard_x_shape[2] # B = 1
                x, pad_hw = window_partition_tensor(x, window_size, self.hard_x_shape)
            else:
                # print("[MULTISCALEBLOCK] Global attention layer with input shape :", self.hard_x_shape[0], 1, self.hard_x_shape[1], self.hard_x_shape[2], self.hard_x_shape[3])
                B, H, W, C = self.hard_x_shape[0], self.hard_x_shape[1], self.hard_x_shape[2], self.hard_x_shape[3]
                x = x.reshape(B, 1, H, W, C)


            # print("[MULTISCALEBLOCK] x shape after partitionning:", x.shape)

            # Window Attention + Q Pooling (if stage change)
            x_attn = self.attn(x, hard_x_shape=self.hard_x_shape)
                
            # print("[MULTISCALEBLOCK] x shape after attention:", x_attn.shape)
            
            if self.q_stride:
                # Shapes have changed due to Q pooling
                window_size = self.window_size // self.q_stride[0]
                H, W = self.hard_x_shape[1]//2, self.hard_x_shape[2]//2 # Hardcoded for quantization

                pad_h = (window_size - H % window_size) % window_size
                pad_w = (window_size - W % window_size) % window_size
                pad_hw = (H + pad_h, W + pad_w)

            #print("Hard x shape:", self.hard_x_shape)

            # Reverse window partition
            if self.window_size > 0:
                x = window_unpartition_tensor(x_attn, window_size, B, H, W, self.dim_out)
            else:
                # print("[MULTISCALEBLOCK] Global attention layer with output shape :", B, H, W, self.dim_out)
                x = x_attn.reshape(B, H, W, self.dim_out)

            # print("[MULTISCALEBLOCK] x shape after unpartitionning:", x.shape)

            x = shortcut + self.drop_path(x)
            # MLP
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        else:
            shortcut = x  # B, H, W, C
            x = self.norm1(x)

            # Skip connection
            if self.dim != self.dim_out:
                shortcut = do_pool(self.proj(x), self.pool)

            # Window partition
            window_size = self.window_size
            if window_size > 0:
                H, W = x.shape[1], x.shape[2]
                x, pad_hw = window_partition(x, window_size)


            # Window Attention + Q Pooling (if stage change)
            x = self.attn(x)
            
            if self.q_stride:
                # Shapes have changed due to Q pooling
                window_size = self.window_size // self.q_stride[0]
                H, W = shortcut.shape[1:3]

                pad_h = (window_size - H % window_size) % window_size
                pad_w = (window_size - W % window_size) % window_size
                pad_hw = (H + pad_h, W + pad_w)

            #print("Hard x shape:", self.hard_x_shape)

            # Reverse window partition
            if self.window_size > 0:
                x = window_unpartition(x, window_size, pad_hw, (H, W))

            x = shortcut + self.drop_path(x)
            # MLP
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class Hiera(nn.Module):
    """
    Reference: https://arxiv.org/abs/2306.00989
    """

    def __init__(
        self,
        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        drop_path_rate: float = 0.0,  # stochastic depth
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, int] = (2, 2),  # downsample stride bet. stages
        stages: Tuple[int, ...] = (2, 3, 16, 3),  # blocks per stage
        dim_mul: float = 2.0,  # dim_mul factor at stage shift
        head_mul: float = 2.0,  # head_mul factor at stage shift
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        mlp_ratio: float = 4.0,  # mlp ratio
        # window size per stage, when not using global att.
        window_spec: Tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),
        # global attn in these blocks
        global_att_blocks: Tuple[int, ...] = (
            12,
            16,
            20,
        ),
        weights_path=None,
        return_interm_layers=True,  # return feats from every stage
        input_size: Tuple[int, int] = (224, 224),  # input size
        export: bool = False,  # export mode
    ):
        super().__init__()

        self.export = export

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim,
        )
        # Which blocks have global att?
        self.global_att_blocks = global_att_blocks

        # Windowed positional embedding (https://arxiv.org/abs/2311.05613)
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0])
        )
        
        self.embed_dim = embed_dim
        self.input_size = input_size # input_size = 224


        # hard_x_shapes_list = [
        #     [1, 56, 56, 64],
        #     [1, 56, 56, 64],
        #     [1, 56, 56, 64],
        #     [1, 56, 56, 64],
        #     [1, 56, 56, 64],
        #     [1, 28, 28, 128],
        #     [1, 28, 28, 128],
        #     [1, 28, 28, 128],
        #     [1, 28, 28, 128],
        #     [1, 14, 14, 256],
        # ]
        
        # print("[HARDCODING] embed_dim:", embed_dim) # 64
        # print("[HARDCODING] stages:", stages)
        # print("[HARDCODING] stage_ends:", self.stage_ends)
        # print("[HARDCODING] q_pool_blocks:", self.q_pool_blocks) # [4, 8]
        # print("[HARDCODING] window_spec:", self.window_spec)
        
        if self.export:
            shapes = []
            spatial_size = input_size // 4  # initial 4x downsampling
            channels = embed_dim
            
            total_blocks = sum(stages)
            for block in range(total_blocks):
                shapes.append([1, spatial_size, spatial_size, channels])
                if block in self.q_pool_blocks:
                    spatial_size = spatial_size // 2
                    channels = channels * 2
            self.hard_x_shapes_list = shapes
            # print("[HARDCODING] hard_x_shapes_list:", self.hard_x_shapes_list)
        else:
            self.hard_x_shapes_list = None

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        cur_stage = 1
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            # lags by a block, so first block of
            # next stage uses an initial window size
            # of previous stage and final window size of current stage
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                hard_x_shape=self.hard_x_shapes_list[i] if self.export else None, # Hardcoded for quantization
                export=export,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

        if weights_path is not None:
            with g_pathmgr.open(weights_path, "rb") as f:
                chkpt = torch.load(f, map_location="cpu")
            logging.info("loading Hiera", self.load_state_dict(chkpt, strict=False))

    # def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
    #     h, w = hw
    #     window_embed = self.pos_embed_window
    #     pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
    #     pos_embed = pos_embed + window_embed.tile(
    #         [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
    #     )
    #     pos_embed = pos_embed.permute(0, 2, 3, 1)
    #     return pos_embed
    
    # Same but compliant with Torch FX
    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        # print("[HARDCODING] window_embed shape:", window_embed.shape)
        # print("[HARDCODING] window_embed shape estimated:", 1, self.embed_dim, self.window_spec[0], self.window_spec[0])
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        
        # print("[HARDCODING] pos_embed shape:", pos_embed.shape)
        # print("[HARDCODING] pos_embed shape estimated:", 1, self.embed_dim, h, w)
               
        # Compute tiling ratio without iterating over Proxy shapes
        # batch_repeat = pos_embed.shape[0] // window_embed.shape[0]
        # channel_repeat = pos_embed.shape[1] // window_embed.shape[1]
        # height_repeat = pos_embed.shape[2] // window_embed.shape[2]
        # width_repeat = pos_embed.shape[3] // window_embed.shape[3]
        # Let's compute it dynamically (.shape bad with FX Tracer)
        batch_repeat = 1 // 1
        channel_repeat = self.embed_dim // self.embed_dim
        height_repeat = h // self.window_spec[0]
        width_repeat = w // self.window_spec[0]

        repeat_factors = (batch_repeat, channel_repeat, height_repeat, width_repeat)
        # print("[HARDCODING] repeat_factors estimated:", repeat_factors)

        pos_embed = pos_embed + window_embed.repeat(repeat_factors)

        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed


    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        
        # print("[HIERADET] Input shape:", x.shape)
        
        x = self.patch_embed(x)
        
        # print("[HIERADET] After patch embedding shape:", x.shape)
        # x: (B, H, W, C)

        # Add pos embed
        #x = x + self._get_pos_embed(x.shape[1:3])
        x = x + self._get_pos_embed((self.input_size//4, self.input_size//4)) # Equivalently can use X hard sizes

        outputs = []
        for i, blk in enumerate(self.blocks):
            # print("[HIERADET] Block", i, " x input shape", x.shape)
            x = blk(x)
            if (i == self.stage_ends[-1]) or (
                i in self.stage_ends and self.return_interm_layers
            ):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)
                
        return outputs

    def get_layer_id(self, layer_name):
        # https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
        num_layers = self.get_num_layers()

        if layer_name.find("rel_pos") != -1:
            return num_layers + 1
        elif layer_name.find("pos_embed") != -1:
            return 0
        elif layer_name.find("patch_embed") != -1:
            return 0
        elif layer_name.find("blocks") != -1:
            return int(layer_name.split("blocks")[1].split(".")[1]) + 1
        else:
            return num_layers + 1

    def get_num_layers(self) -> int:
        return len(self.blocks)
