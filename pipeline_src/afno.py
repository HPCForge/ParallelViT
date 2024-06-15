# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

import math
import torch
import torch.fft
import torch.nn as nn
from functools import partial #
from collections import OrderedDict
from copy import Error, deepcopy
from re import S
from numpy.lib.arraypad import pad
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
import torch.fft
from torch.nn.modules.container import Sequential
from torch.utils.checkpoint import checkpoint_sequential
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
#from utils.img_utils import PeriodicPad2d
from torchvision.utils import save_image

from vit import PatchEmbed, Mlp, SelfAttentionBlock, get_2d_sincos_pos_embed

class AFNO2D(nn.Module):
	"""
	hidden_size: channel dimension size
	num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
	sparsity_threshold: lambda for softshrink
	hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
	"""
	def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1):
		super().__init__()
		assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

		self.hidden_size = hidden_size
		self.sparsity_threshold = sparsity_threshold
		self.num_blocks = num_blocks
		self.block_size = self.hidden_size // self.num_blocks
		self.hard_thresholding_fraction = hard_thresholding_fraction
		self.hidden_size_factor = hidden_size_factor
		self.scale = 0.02

		self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
		self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
		self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
		self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

	def forward(self, x, spatial_size=None):
		bias = x

		dtype = x.dtype
		x = x.float()
		B, N, C = x.shape

		if spatial_size == None:
			H = W = int(math.sqrt(N))
		else:
			H, W = spatial_size

		x = x.reshape(B, H, W, C)
		x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
		x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

		o1_real = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
		o1_imag = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
		o2_real = torch.zeros(x.shape, device=x.device)
		o2_imag = torch.zeros(x.shape, device=x.device)

		total_modes = N // 2 + 1
		kept_modes = int(total_modes * self.hard_thresholding_fraction)

		o1_real[:, :, :kept_modes] = F.relu(
			torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[0]) - \
			torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[1]) + \
			self.b1[0]
		)

		o1_imag[:, :, :kept_modes] = F.relu(
			torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[0]) + \
			torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[1]) + \
			self.b1[1]
		)

		o2_real[:, :, :kept_modes] = (
			torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[0]) - \
			torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[1]) + \
			self.b2[0]
		)

		o2_imag[:, :, :kept_modes] = (
			torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[0]) + \
			torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[1]) + \
			self.b2[1]
		)

		x = torch.stack([o2_real, o2_imag], dim=-1)
		x = F.softshrink(x, lambd=self.sparsity_threshold)
		x = torch.view_as_complex(x)
		x = x.reshape(B, x.shape[1], x.shape[2], C)
		x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
		x = x.reshape(B, N, C)
		x = x.type(dtype)
		return x + bias


class Block(nn.Module):

	def __init__(
			self,
			dim,
			mlp_ratio=4.,
			drop=0.,
			drop_path=0.,
			act_layer=nn.GELU,
			norm_layer=nn.LayerNorm,
			double_skip=True,
			num_blocks=8,
			sparsity_threshold=0.01,
			hard_thresholding_fraction=1.0
		):
		super().__init__()
		self.norm1 = norm_layer(dim)
		self.filter = AFNO2D(dim, num_blocks, sparsity_threshold, hard_thresholding_fraction) 
		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
		#self.drop_path = nn.Identity()
		self.norm2 = norm_layer(dim)
		mlp_hidden_dim = int(dim * mlp_ratio)
		self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
		self.double_skip = double_skip

	def forward(self, x):
		residual = x
		x = self.norm1(x)
		x = self.filter(x)

		if self.double_skip:
			x = x + residual
			residual = x

		x = self.norm2(x)
		x = self.mlp(x)
		x = self.drop_path(x)
		x = x + residual
		return x


class AFNONet(nn.Module):
	def __init__(self, field_height=512, field_width=512, patch_size=16, pred_channels=2,
				embed_dim=1024, norm_layer=nn.LayerNorm, depth=6, mlp_ratio=4., 
				drop_rate=0., drop_path_rate=0., num_blocks=16, 
				sparsity_threshold=0.01, hard_thresholding_fraction=1.0):
		
		super().__init__()
		
		self.field_height = field_height
		self.field_width = field_height
		self.patch_size = patch_size
		self.pred_channels = pred_channels
		self.num_blocks = num_blocks
		self.embed_dim = embed_dim
		

		self.patch_embed = PatchEmbed(field_height=field_height, field_width=field_width, patch_size=patch_size, in_channels=pred_channels, embed_dim=embed_dim)
		num_patches = self.patch_embed.num_patches

		self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
		self.pos_drop = nn.Dropout(p=drop_rate)

		dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

		#self.h = img_size[0] // self.patch_size[0]
		#self.w = img_size[1] // self.patch_size[1]
		self.h = field_height // patch_size
		self.w = field_width // patch_size

		self.blocks = nn.ModuleList([
			Block(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
			num_blocks=self.num_blocks, sparsity_threshold=sparsity_threshold, hard_thresholding_fraction=hard_thresholding_fraction) 
		for i in range(depth)])

		self.norm = norm_layer(embed_dim)

		#self.head = nn.Linear(embed_dim, (patch_size ** 2) * pred_channels, bias=False)
		self.pred = nn.Sequential(
			nn.Linear(embed_dim, (patch_size ** 2) * pred_channels, bias=False),
			nn.Sigmoid()
		)

		trunc_normal_(self.pos_embed, std=.02)
		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)

	@torch.jit.ignore
	def no_weight_decay(self):
		return {'pos_embed', 'cls_token'}

	def forward_features(self, x):
		B = x.shape[0]
		x = self.patch_embed(x)
		x = x + self.pos_embed
		x = self.pos_drop(x)
		# todo manxin cannot understand, got it
		x = x.reshape(B, self.h, self.w, self.embed_dim)
		for blk in self.blocks:
			x = blk(x)

		return x

	def forward(self, x):
		x = self.forward_features(x)
		x = self.pred(x)
		x = rearrange(
			x,
			"b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
			p1=self.patch_size,
			p2=self.patch_size,
			h=self.field_height // self.patch_size,
			w=self.field_width // self.patch_size,
		)
		return x