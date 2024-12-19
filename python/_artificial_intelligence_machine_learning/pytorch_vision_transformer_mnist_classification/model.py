"""
Vision transformer model class

Author: Sam Barba
Created 19/12/2024
"""

from torch import cat, nn, randn


def imgs_to_patches(x, patch_size, flatten_channels=True):
	n, c, h, w = x.shape
	h_ = h // patch_size
	w_ = w // patch_size
	x = x.reshape(n, c, h_, patch_size, w_, patch_size)  # -> (N, C, H', p_H, W', p_W)
	x = x.permute(0, 2, 4, 1, 3, 5)                      # -> (N, H', W', C, p_H, p_W)
	x = x.flatten(1, 2)                                  # -> (N, H' x W', C, p_H, p_W)
	if flatten_channels:
		x = x.flatten(2, 4)                              # -> (N, H' x W', C x p_H x p_W)

	return x


class AttentionBlock(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, num_heads):
		super().__init__()

		self.layer_norm_1 = nn.LayerNorm(embedding_dim)
		self.attention = nn.MultiheadAttention(embedding_dim, num_heads)
		self.layer_norm_2 = nn.LayerNorm(embedding_dim)
		self.fc_block = nn.Sequential(
			nn.Linear(embedding_dim, hidden_dim),
			nn.GELU(),
			nn.Dropout(0.2),
			nn.Linear(hidden_dim, embedding_dim),
			nn.Dropout(0.2)
		)

	def forward(self, x):
		x_norm1 = self.layer_norm_1(x)
		attn_out = self.attention(x_norm1, x_norm1, x_norm1)[0]
		x_norm2 = self.layer_norm_2(x + attn_out)
		fc_out = self.fc_block(x_norm2)
		out = fc_out + x

		return out


class VisionTransformer(nn.Module):
	def __init__(
		self, *, num_channels=1, embedding_dim, hidden_dim,
		num_attention_layers, num_heads, patch_size, num_patches
	):
		"""
		Inputs:
			num_channels - number of input channels (1 for grayscale)
			embedding_dim - dimensionality of the input feature vectors to the ViT
			hidden_dim - dimensionality of the hidden layer in the feed-forward networks within the ViT
			num_attention_layers - number of attention layers in the ViT
			num_heads - number of heads to use in the multi-head attention block
			patch_size - number of pixels per dimension of one image patch
			num_patches - number of patches that constitute an image
		"""

		super().__init__()

		self.patch_size = patch_size
		self.input_layer = nn.Linear(num_channels * patch_size * patch_size, embedding_dim)

		# In ViTs, the input image is divided into a sequence of fixed-size patches, then each patch is linearly
		# embedded. The classification (CLS) token is concatenated to the sequence of patch embeddings before being fed
		# into the transformer. The CLS token serves as a global representation of the image. After passing through the
		# transformer layers, the output corresponding to the CLS token is typically used for downstream tasks like
		# image classification, where its representation is used to make predictions.
		self.cls_token = nn.Parameter(randn(1, 1, embedding_dim))

		# Since transformers don't inherently understand the spatial structure of the input data, position embeddings
		# are used to inform about the position of each image patch. Position embeddings are added to the patch
		# embeddings to maintain the spatial relationships between patches, ensuring the model's understanding of where
		# each patch is located in the original image. This spatial information can be used when learning patterns.
		# These embeddings can either be learned during training, or be fixed (e.g. sinusoidal embeddings).
		self.pos_embedding = nn.Parameter(randn(1, num_patches + 1, embedding_dim))

		self.dropout = nn.Dropout(0.2)
		self.transformer = nn.Sequential(
			*[AttentionBlock(embedding_dim, hidden_dim, num_heads) for _ in range(num_attention_layers)]
		)
		self.mlp_head = nn.Sequential(
			nn.LayerNorm(embedding_dim),
			nn.Linear(embedding_dim, 10)  # 10 classes (0-9)
		)

	def forward(self, x):
		# Preprocess input
		x = imgs_to_patches(x, self.patch_size)    # -> (N, num_patches, patch_size ^ 2)
		n, p, *_ = x.shape
		x = self.input_layer(x)                    # -> (N, num_patches, embedding_dim)

		# Concatenate CLS token and add positional embedding
		cls_token = self.cls_token.repeat(n, 1, 1)
		x = cat([cls_token, x], dim=1)             # -> (N, num_patches + 1, embedding_dim)
		x += self.pos_embedding[:, :p + 1]         # -> (N, num_patches + 1, embedding_dim)

		# Apply transformer
		x = self.dropout(x)
		x = x.transpose(0, 1)                      # -> (num_patches + 1, N, embedding_dim)
		x = self.transformer(x)                    # -> (num_patches + 1, N, embedding_dim)

		# Perform classification
		cls = x[0]                                 # -> (N, embedding_dim)
		out = self.mlp_head(cls)                   # -> (N, 10)

		return out
