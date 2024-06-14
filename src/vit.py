import torch
import torch.nn as nn
import torch.nn.functional as F

def get_2d_sincos_pos_embed(embed_dim, height_patches, width_patches):
    """
    embed_dim: embedding dimension of the encoder
	field_height: int of the field height
    field_width: int of the field width
	return:
	pos_embed: [field_width*field_height, embed_dim]
	"""
    grid_h = np.arange(height_patches, dtype=np.float32)
    grid_w = np.arange(width_patches, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, height_patches, width_patches])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
	assert embed_dim % 2 == 0

	# use half of dimensions to encode grid_h
	emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
	emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

	emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
	return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
	"""
	embed_dim: output dimension for each position
	pos: a list of positions to be encoded: size (M,)
	out: (M, D)
	"""
	assert embed_dim % 2 == 0
	omega = np.arange(embed_dim // 2, dtype=float)
	omega /= embed_dim / 2.
	omega = 1. / 10000**omega  # (D/2,)

	pos = pos.reshape(-1)  # (M,)
	out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

	emb_sin = np.sin(out) # (M, D/2)
	emb_cos = np.cos(out) # (M, D/2)

	emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
	return emb


class Mlp(nn.Module):
	"""
	MLP as used in Vision Transformer, MLP-Mixer and related networks
	"""
	def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
			norm_layer=None, bias=True):
		super().__init__()
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features
		bias = (bias, bias)

		self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
		self.act = act_layer()
		self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
		self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])

	def forward(self, x):
		x = self.fc1(x)
		x = self.act(x)
		x = self.norm(x)
		x = self.fc2(x) 
		return x
	

class PatchEmbed(nn.Module):
	"""
	2D Image to Patch Embedding
	"""
	def __init__(self, field_height=512, field_width=512, patch_size=16, in_channels=1,
				 embed_dim=1024, norm_layer=None, flatten=True, bias=True):
		super().__init__()
		assert field_height%patch_size == 0, f'Input field height is not divisible by patch size.'
		assert field_width%patch_size == 0, f'Input field width is not divisible by patch size.'
		img_size = (field_height, field_width)
		patch_size = (patch_size, patch_size)
		self.img_size = img_size
		self.patch_size = patch_size
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
	
class SelfAttention(nn.Module):
	def __init__(self, dim, num_heads=12, qkv_bias=True, qk_scale=None):
		super().__init__()
		assert dim % num_heads == 0, 'self attention dim should be divisible by num_heads'
		self.num_heads = num_heads
		self.head_dim = dim // num_heads
		self.scale = qk_scale or (dim // num_heads) ** -0.5

		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.proj = nn.Linear(dim, dim)

	def forward(self, x):
		B, N, C = x.shape
		qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
		q, k, v = qkv.unbind(0)

		attn = (q @ k.transpose(-2, -1)) * self.scale
		attn = F.softmax(attn, dim=-1)
		x = attn @ v

		x = (attn @ v).transpose(1, 2).reshape(B, N, C)
		x = self.proj(x)
		return x


class SelfAttentionBlock(nn.Module):
	def __init__(self, dim, num_heads=12, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
				act_layer=nn.GELU, norm_layer=nn.LayerNorm, mlp_layer=Mlp):
		super().__init__()
		self.norm1 = norm_layer(dim)
		self.attn = SelfAttention(dim, num_heads, qkv_bias, qk_scale)

		self.norm2 = norm_layer(dim)
		self.mlp = mlp_layer(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)
	
	def forward(self, x):
		x = x + self.attn(self.norm1(x))
		x = x + self.mlp(self.norm2(x))
		return x




class ViT(nn.Module):

	def __init__(self, field_height=512, field_width=512, patch_size=16, pred_channels=1,
				 embed_dim=1024, depth=6, num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, pred_activation=nn.GELU):
		super().__init__()

		self.field_height = field_height
		self.field_width = field_width
		self.embed_dim = embed_dim
		self.patch_size = patch_size
		self.pred_channels = pred_channels

		#self.norm = norm_layer(embed_dim)

		self.patch_embed = PatchEmbed(field_height=self.field_height, field_width=self.field_width, patch_size=self.patch_size, embed_dim=self.embed_dim, in_channels=self.pred_channels)
		self.num_patches = self.patch_embed.num_patches

		self.blocks = nn.ModuleList(
			[SelfAttentionBlock(dim=self.embed_dim, num_heads=num_heads) for i in range(depth)]
		)
		
		self.pred = nn.Sequential(
			nn.Linear(self.embed_dim, self.patch_size ** 2 * self.pred_channels),
			pred_activation(),
		)
	
	def forward(self, x):

		x = self.patch_embed(x)
		for block in self.blocks:
			x = block(x)
		#x = self.pred(x)
		#x = self.unpatchify(x)
		return x
	
	def unpatchify(self, x):
		"""
		x: (N, L, patch_size**2 *3)
		imgs: (N, 3, H, W)
		"""
		p = self.patch_embed.patch_size[0]
		h = w = int(x.shape[1]**.5)
		assert h * w == x.shape[1]
		
		x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
		x = torch.einsum('nhwpqc->nchpwq', x)
		imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
		return imgs

