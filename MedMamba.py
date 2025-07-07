import re
import time
import math
import numpy as np
from functools import partial
from typing import Optional, Union, Type, List, Tuple, Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from .local_scan import LocalScanTriton, LocalReverseTriton, local_scan, local_scan_bchw, local_reverse

def diagonal_gather(tensor):
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)
    index = (shift + torch.arange(W, device=tensor.device)) % W
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    return tensor.gather(3, expanded_index).transpose(-1, -2).reshape(B, C, H * W)

def antidiagonal_gather(tensor):
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)
    index = (torch.arange(W, device=tensor.device) - shift) % W
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    return tensor.gather(3, expanded_index).transpose(-1, -2).reshape(B, C, H * W)

def diagonal_scatter(tensor_flat, original_shape):
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)
    index = (shift + torch.arange(W, device=tensor_flat.device)) % W
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor

def antidiagonal_scatter(tensor_flat, original_shape):
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)
    index = (torch.arange(W, device=tensor_flat.device) - shift) % W
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor

def zigzag_path_lr(N, M, start_row=0, start_col=0, dir_row=1, dir_col=1):
    path = []
    for i in range(N):
        for j in range(M):
            # If the row number is even, move right; otherwise, move left
            col = j if i % 2 == 0 else M - 1 - j
            path.append((start_row + dir_row * i) * M + start_col + dir_col * col)
    return path

def zigzag_path_tb(N, M, start_row=0, start_col=0, dir_row=1, dir_col=1):
    path = []
    for j in range(M):
        for i in range(N):
            # If the column number is even, move down; otherwise, move up
            row = i if j % 2 == 0 else N - 1 - i
            path.append((start_row + dir_row * row) * M + start_col + dir_col * j)
    return path

def reverse_permut_np(permutation):
    n = len(permutation)
    reverse = np.array([0] * n)
    for i in range(n):
        reverse[permutation[i]] = i
    return reverse


class ScanPool:
    def forward(self, x, index):
        B, C, H, W = x.shape
        x_ = x.new_empty((B, C, H * W))
        if index == 0:
            x_ = x.flatten(2, 3)  # → ↓
        if index == 1:
            x_ = x.transpose(dim0=2, dim1=3).flatten(2, 3)  # ↓ →
        if index == 2:
            x_ = torch.flip(x.flatten(2, 3), dims=[-1])  # ← ↑
        if index == 3:
            x_ = torch.flip(x.transpose(dim0=2, dim1=3).flatten(2, 3), dims=[-1])  # ↑ ←
        if index == 4:
            x_ = diagonal_gather(x)
        if index == 5:
            x_ = antidiagonal_gather(x)
        if index == 6:
            x_ = torch.flip(diagonal_gather(x), dims=[-1])
        if index == 7:
            x_ = torch.flip(antidiagonal_gather(x), dims=[-1])
        if index == 8:
            x_ = x.flatten(2, 3)
            perm = zigzag_path_lr(H, W, 0, 0, 1, 1)  # → ←
            x_ = x_[:, :, perm].contiguous()
        if index == 9:
            x_ = x.flatten(2, 3)
            perm = zigzag_path_lr(H, W, 0, H-1, 1, -1)  #  ← →
            x_ = x_[:, :, perm].contiguous()
        if index == 10:
            x_ = x.flatten(2, 3)
            perm = zigzag_path_tb(H, W, 0, 0, 1, 1) # ↓ ↑
            x_ = x_[:, :, perm].contiguous()
        if index == 11:
            x_ = x.flatten(2, 3)
            perm = zigzag_path_tb(H, W, H-1, 0, -1, 1)  # ↑ ↓
            x_ = x_[:, :, perm].contiguous()
        if index == 12:
            x_ = local_scan_bchw(x, 7, H, W, False, False)
        if index == 13:
            x_ = local_scan_bchw(x, 7, H, W, False, True)
        if index == 14:
            x_ = local_scan_bchw(x, 7, H, W, True, False)
        if index == 15:
            x_ = local_scan_bchw(x, 7, H, W, True, True)
        return x_

    def backward(self, x, y, index):
        _, _, H, W = x.shape
        L = H * W
        B, D, _ = y.shape
        y_ = y.new_empty((B, D, L))
        if index == 0:  # → ↓
            y_ = y
        if index == 1:
            y_ = y.view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)  # ↓ →
        if index == 2:
            y_ = y.flip(dims=[-1])  # ← ↑
        if index == 3:
            y_ = y.flip(dims=[-1]).view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)  # ↑ ←
        if index == 4:
            y_ = diagonal_scatter(y, (B, D, H, W)).view(B, -1, L)
        if index == 5:
            y_ = antidiagonal_scatter(y, (B, D, H, W)).view(B, -1, L)
        if index == 6:
            y_ = diagonal_scatter(y.flip(dims=[-1]), (B, D, H, W)).view(B, -1, L)
        if index == 7:
            y_ = antidiagonal_scatter(y.flip(dims=[-1]), (B, D, H, W)).view(B, -1, L)
        if index == 8:
            perm = zigzag_path_lr(H, W, 0, 0, 1, 1)  # → ←
            reverse_perm = reverse_permut_np(perm)
            y_ = y[:, :, reverse_perm].contiguous()
        if index == 9:
            perm = zigzag_path_lr(H, W, 0, H-1, 1, -1)  #  ← →
            reverse_perm = reverse_permut_np(perm)
            y_ = y[:, :, reverse_perm].contiguous()
        if index == 10:
            perm = zigzag_path_tb(H, W, 0, 0, 1, 1) # ↓ ↑
            reverse_perm = reverse_permut_np(perm)
            y_ = y[:, :, reverse_perm].contiguous()
        if index == 11:
            perm = zigzag_path_tb(H, W, H-1, 0, -1, 1)  # ↑ ↓
            reverse_perm = reverse_permut_np(perm)
            y_ = y[:, :, reverse_perm].contiguous()
        if index == 12:
            y_ = local_reverse(y, 7, H, W, False, False)
        if index == 13:
            y_ = local_reverse(y, 7, H, W, False, True)
        if index == 14:
            y_ = local_reverse(y, 7, H, W, True, False)
        if index == 15:
            y_ = local_reverse(y, 7, H, W, True, True)
        return y_


class DWConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: int = 6,
                 kernel_size: int = 3,
                 do_res: int = True,
                 norm_type: str = 'group',
                 n_groups: int or None = None,
                 dim='2d'
                 ):
        super().__init__()
        hidden_channels = int(in_channels * expand_ratio)
        self.do_res = do_res
        assert dim in ['2d', '3d']
        self.dim = dim
        if self.dim == '2d':
            conv = nn.Conv2d
        elif self.dim == '3d':
            conv = nn.Conv3d
        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.conv2 = conv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=hidden_channels if n_groups is None else n_groups,
        )
        self.conv3 = conv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        if norm_type == 'group':
            self.norm = nn.GroupNorm(
                num_groups=hidden_channels,
                num_channels=hidden_channels
            )
        elif norm_type == 'layer':
            self.norm = LayerNorm(
                normalized_shape=hidden_channels,
                data_format='channels_first'
            )
        self.act = nn.GELU()
        if do_res:
            self.res_conv = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )

    def forward(self, x):
        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))
        x1 = self.conv3(x1)
        if self.do_res:
            x1 = self.res_conv(x) + x1
        return x1


class DWConvBasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, expand_ratio=6, kernel_size=3,
                 do_res=True, norm_type='group', dim='2d'):

        super().__init__()
        self.DWConv1 = DWConvBlock(in_channels, out_channels, expand_ratio=expand_ratio, do_res=False)
        self.DWConv2 = DWConvBlock(out_channels, out_channels, expand_ratio=expand_ratio, do_res=False)
        self.dim = dim
        self.do_res = do_res
        if self.dim == '2d':
            conv = nn.Conv2d
        elif self.dim == '3d':
            conv = nn.Conv3d
        if do_res:
            self.res_conv = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )

    def forward(self, x):
        x1 = self.DWConv1(x)
        x1 = self.DWConv2(x1)
        if self.do_res:
            res = self.res_conv(x)
            x1 = x1 + res
        return x1


class DWConvDownBlock(DWConvBlock):

    def __init__(self, in_channels, out_channels, expand_ratio=6, kernel_size=3,
                 do_res=True, norm_type='group', dim='2d'):

        super().__init__(in_channels, out_channels, expand_ratio, kernel_size,
                         do_res=False, norm_type=norm_type, dim=dim)
        hidden_channels = in_channels * expand_ratio
        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d
        self.do_res = do_res
        if do_res:
            self.res_conv = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2,
                padding=0
            )

        self.conv2 = conv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels
        )

    def forward(self, x):
        x1 = super().forward(x)
        if self.do_res:
            x1 = x1 + self.res_conv(x)
        return x1


class DWConvUpBlock(DWConvBlock):

    def __init__(self, in_channels, out_channels, expand_ratio=6, kernel_size=3,
                 do_res=True, norm_type='group', dim='2d'):
        super().__init__(in_channels, out_channels, expand_ratio, kernel_size,
                         do_res=False, norm_type=norm_type, dim=dim)
        hidden_channels = in_channels * expand_ratio
        self.do_res = do_res
        self.dim = dim
        if dim == '2d':
            conv = nn.ConvTranspose2d
        elif dim == '3d':
            conv = nn.ConvTranspose3d
        if do_res:
            self.res_conv = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2,
                padding=0
            )
        self.conv2 = conv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels
        )

    def forward(self, x):
        x1 = super().forward(x)
        if self.dim == '2d':
            x1 = torch.nn.functional.pad(x1, (1, 0, 1, 0))
        elif self.dim == '3d':
            x1 = torch.nn.functional.pad(x1, (1, 0, 1, 0, 1, 0))
        if self.do_res:
            x = self.res_conv(x)
            if self.dim == '2d':
                x = torch.nn.functional.pad(x, (1, 0, 1, 0))
            elif self.dim == '3d':
                x = torch.nn.functional.pad(x, (1, 0, 1, 0, 1, 0))
            x1 = x1 + x
        return x1


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpand(nn.Module):
    """
    Reference: https://arxiv.org/pdf/2105.05537.pdf
    """

    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # B, C, H, W ==> B, H, W, C
        x = self.expand(x)
        B, H, W, C = x.shape

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        x = x.reshape(B, H * 2, W * 2, C // 4)

        return x

class MambaExpert(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.K = 4
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.d_model
        # self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.x_proj_weight = nn.Parameter(self.x_proj.weight)  # (N, inner)
        self.dt_projs = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs)
        self.dt_projs_weight = nn.Parameter(self.dt_projs.weight)  # (inner, rank)
        self.dt_projs_bias = nn.Parameter(self.dt_projs.bias)  # (inner)
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (D, N)
        self.selective_scan = selective_scan_fn

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward(self, x: torch.Tensor):
        B, C, L = x.shape
        x_dbl = torch.einsum("b d l, c d -> b c l", x, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1)
        dts = torch.einsum("b r l, d r -> b d l", dts.view(B, -1, L), self.dt_projs_weight)
        x = x.float().view(B, -1, L)  # (b, d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, d, l)
        Bs = Bs.float().view(B, -1, L)  # (b, d_state, l)
        Cs = Cs.float().view(B, -1, L)  # (b, d_state, l)
        Ds = self.Ds.float().view(-1)  # (d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (d)

        out_y = self.selective_scan(
            x, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, -1, L)
        assert out_y.dtype == torch.float
        return out_y

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.K = 4
        self.num_experts = 16
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        # self.d_inner = int(self.expand * self.d_model)
        self.d_inner = self.d_model
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_proj_z = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.gate = nn.Linear(self.d_inner, self.num_experts)
        self.noise_gate = nn.Linear(self.d_inner, self.num_experts)
        self.experts = nn.ModuleList([MambaExpert(d_model=d_model, d_state=d_state, **kwargs) for _ in range(self.num_experts)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.scan_pool = ScanPool()
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    def forward_core(self, x: torch.Tensor, scores, indices):
        B, C, H, W = x.shape
        L = H * W
        K = self.K
        final_output = torch.zeros(B,C,L).cuda()
        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1).view(-1)
            if expert_mask.any():
                expert_input = x[expert_mask]
                expert_input = self.scan_pool.forward(expert_input, i)   # b C L
                expert_output = expert(expert_input)
                gating_scores = scores[expert_mask, i].unsqueeze(1).unsqueeze(1)
                weighted_output = expert_output * gating_scores
                weighted_output = self.scan_pool.backward(x, weighted_output, i)
                final_output[expert_mask] += weighted_output.squeeze(1)
        return final_output

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        z = self.in_proj_z(x)
        x = self.in_proj(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)

        # gating
        B, D, _, _ = x.shape
        logits  = self.pool(x).view(B, D, -1).permute(0, 2, 1).contiguous()   # B 1 D
        noise_logits = self.noise_gate(logits).squeeze(1)
        logits = self.gate(logits).squeeze(1)  # B num_experts
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise
        top_k_logits, indices = noisy_logits.topk(self.K, dim=-1)     # B K
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        scores = F.softmax(sparse_logits, dim=-1)      # B num_experts

        y = self.forward_core(x, scores, indices)   # B, C, L
        y = y.view(B, -1, H*W)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class VSSLayer(nn.Module):
    """ A basic layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        if True:  # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class VSSMEncoder(nn.Module):
    def __init__(self, depths=[9, 2],
                 dims=[384, 768], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,  # 20240109
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                self.downsamples.append(PatchMerging2D(dim=dims[i_layer], norm_layer=norm_layer))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless

        Conv2D is not intialized !!!
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x_ret = []
        for s, layer in enumerate(self.layers):
            x = layer(x)
            x_ret.append(x.permute(0, 3, 1, 2).contiguous())
            if s < len(self.downsamples):
                x = self.downsamples[s](x)

        return x_ret


class MedMamba(nn.Module):
    def __init__(
            self,
            in_chans=1,
            out_chans=13,
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            hidden_size: int = 768,
            norm_name="instance",
            res_block: bool = True,
            spatial_dims=2,
            deep_supervision: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, feat_size[0], kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(feat_size[0], eps=1e-5, affine=True),
        )
        self.spatial_dims = spatial_dims
        self.vssm_encoder = VSSMEncoder()
        self.block1 = DWConvBasicBlock(feat_size[1], feat_size[1])
        self.block2 = DWConvBasicBlock(feat_size[2], feat_size[2])
        self.block_down1 = DWConvDownBlock(feat_size[0], feat_size[1])
        self.block_down2 = DWConvDownBlock(feat_size[1], feat_size[2])
        self.block_up2 = DWConvUpBlock(feat_size[2], feat_size[1])
        self.block_up1 = DWConvUpBlock(feat_size[1], feat_size[0])
        self.D_block3 = DWConvBasicBlock(feat_size[2], feat_size[2])
        self.D_block2 = DWConvBasicBlock(feat_size[1], feat_size[1])
        self.D_block1 = DWConvBasicBlock(feat_size[0], feat_size[0])
        self.proj_enc = nn.Sequential(
            nn.Conv2d(feat_size[2], feat_size[2], kernel_size=1, stride=1),
        )
        self.proj_dec = nn.Sequential(
            nn.Conv2d(feat_size[2], feat_size[2], kernel_size=1, stride=1),
        )
        self.embedding = PatchMerging2D(dim=feat_size[2], norm_layer=nn.LayerNorm)
        self.expand_layer = PatchExpand(
            dim=feat_size[3] * 2,
            dim_scale=2,
            norm_layer=nn.LayerNorm,
        )
        self.expand_layer2 = PatchExpand(
            dim=feat_size[3],
            dim_scale=2,
            norm_layer=nn.LayerNorm,
        )
        self.vss_dec_layer = VSSLayer(
            dim=feat_size[3],
            depth=2,
            d_state=16,
            drop=0.,
            attn_drop=0.,
            drop_path=0.1,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
        )
        self.deep_supervision = deep_supervision
        self.out_layers = nn.ModuleList()
        for i in range(4):
            self.out_layers.append(UnetOutBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[i],
                out_channels=self.out_chans
            ))

    def forward(self, x):
        x = self.stem(x)
        temp = self.block_down1(x)
        enc0 = self.block1(temp)
        temp = self.block_down2(enc0)
        enc1 = self.block2(temp)
        vss_encs = self.vssm_encoder(self.embedding(self.proj_enc(enc1).permute(0, 2, 3, 1).contiguous()))

        vss_decs = self.expand_layer(vss_encs[1])
        vss_decs = vss_decs + vss_encs[0].permute(0, 2, 3, 1).contiguous()

        dec3 = self.vss_dec_layer(vss_decs)
        dec3 = dec3.permute(0, 3, 1, 2).contiguous()

        dec2 = self.expand_layer2(dec3)
        dec2 = self.proj_dec(dec2.permute(0, 3, 1, 2).contiguous())
        dec2 = dec2 + enc1

        dec2 = self.D_block3(dec2)
        dec1 = self.block_up2(dec2)
        dec1 = dec1 + enc0

        dec1 = self.D_block2(dec1)
        dec0 = self.block_up1(dec1)
        dec0 = dec0 + x
        dec0 = self.D_block1(dec0)

        if self.deep_supervision:
            feat_out = [dec0, dec1, dec2, dec3]
            out = []
            for i in range(4):
                pred = self.out_layers[i](feat_out[i])
                out.append(pred)
        else:
            out = self.out_layers[0](dec0)
        return out

    @torch.no_grad()
    def freeze_encoder(self):
        for name, param in self.vssm_encoder.named_parameters():
            if "patch_embed" not in name:
                param.requires_grad = False

    @torch.no_grad()
    def unfreeze_encoder(self):
        for param in self.vssm_encoder.parameters():
            param.requires_grad = True


def get_med_umamba_from_plans(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        deep_supervision: bool = True
):
    label_manager = plans_manager.get_label_manager(dataset_json)

    model = MedMamba(
        in_chans=num_input_channels,
        out_chans=label_manager.num_segmentation_heads,
        feat_size=[48, 96, 192, 384],
        deep_supervision=deep_supervision,
        hidden_size=768,
    )

    return model


