from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pos_embed import get_2d_sincos_pos_embed


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, 
                 img_size: Union[int, Tuple[int, int]] = 512,
                 patch_size: int = 16, 
                 in_channels: int = 4,
                 embed_dim: int = 1024, 
                 norm_layer: int = None, 
                 flatten: bool = True, 
                 bias: bool = True):
        super().__init__()
        if type(img_size) == int:
            img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class ViTBlock(nn.Module):
    """ Parallel ViT block (MLP & Attention in parallel)
    Based on:
      'Scaling Vision Transformers to 22 Billion Parameters` - https://arxiv.org/abs/2302.05442
    """
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = True,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        mlp_hidden_dim = int(mlp_ratio * dim)
        in_proj_out_dim = mlp_hidden_dim + 3 * dim

        self.in_norm = norm_layer(dim)
        self.in_proj = nn.Linear(dim, in_proj_out_dim, bias=qkv_bias)
        self.in_split = [mlp_hidden_dim] + [dim] * 3
        
        if qkv_bias:
            self.register_buffer('qkv_bias', None)
            self.register_parameter('mlp_bias', None)
        else:
            self.register_buffer('qkv_bias', torch.zeros(3 * dim), persistent=False)
            self.mlp_bias = nn.Parameter(torch.zeros(mlp_hidden_dim))
        
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_out_proj = nn.Linear(dim, dim)

        self.mlp_act = act_layer()
        self.mlp_out_proj = nn.Linear(mlp_hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        y = self.in_norm(x)
        if self.mlp_bias is not None:
            y = F.linear(y, self.in_proj.weight, torch.cat((self.qkv_bias, self.mlp_bias)))
        else:
            y = self.in_proj(y)
        x_mlp, q, k, v = torch.split(y, self.in_split, dim=-1)

        q = self.q_norm(q.view(B, N, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(B, N, self.num_heads, self.head_dim)).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        x_attn = F.scaled_dot_product_attention(q, k, v)

        x_attn = x_attn.transpose(1, 2).reshape(B, N, C)
        x_attn = self.attn_out_proj(x_attn)

        # MLP activation, dropout, fc2
        x_mlp = self.mlp_act(x_mlp)
        x_mlp = self.mlp_out_proj(x_mlp)

        y = x_attn + x_mlp
        x = x + y

        return x

class VisionTransformer(nn.Module):
    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 512,
            patch_size: int = 16,
            in_channels: int = 4,
            embed_dim: int = 4096,
            depth: int = 14,
            num_heads: int = 32,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = True,
            has_cls_token: bool = True,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.has_cls_token = has_cls_token

        self.patch_embed = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_channels=in_channels, 
            embed_dim=embed_dim
        )
        self.num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size

        if has_cls_token:
            pos_patches = self.num_patches + 1
        else:
            pos_patches = self.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if has_cls_token else None
        self.pos_embed = nn.Parameter(torch.zeros(1, pos_patches, embed_dim))


        self.blocks = nn.Sequential(*[
            ViTBlock(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                qk_norm=qk_norm,
                act_layer=act_layer, 
                norm_layer=norm_layer
            )
            for _ in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.pred = nn.Linear(embed_dim, patch_size**2 * self.in_channels)
        
        self.initialize_weights()
    
    def initialize_weights(self) -> None:
        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, self.grid_size, cls_token=self.has_cls_token)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h, w = self.grid_size
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_channels, h * p, w * p))
        return imgs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)

        if self.has_cls_token:
            x = x + self.pos_embed[:, 1:, :]
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        else:
            x = x + self.pos_embed

        x = self.blocks(x)
        x = self.norm(x)

        x = self.pred(x)
        x = x[:, 1:, :] # remove cls token
        return self.unpatchify(x)

if __name__ == '__main__':
    from torchsummary import summary
    model = VisionTransformer().cuda().float()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    x = torch.randn(1, 4, 512, 512).cuda().float()
    out = model(x)
    print(out.shape)
    # print(model)
    summary(model, (4, 512, 512))