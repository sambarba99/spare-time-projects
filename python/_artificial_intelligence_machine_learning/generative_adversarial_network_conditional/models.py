"""
Generator and disriminator models

Author: Sam Barba
Created 2026-07-06
"""

from torch import cat, nn
from torch.nn.utils import spectral_norm


AGE_EMBEDDING_DIM = 32
GENDER_EMBEDDING_DIM = 8
RACE_EMBEDDING_DIM = 16
CONDITION_DIM = AGE_EMBEDDING_DIM + GENDER_EMBEDDING_DIM + RACE_EMBEDDING_DIM


class ConditionalBatchNorm2d(nn.Module):
	"""
	Conditional Batch Normalisation (CBN) replaces BatchNorm's fixed learned affine parameters (gamma and beta) with
	values predicted from a conditioning embedding. Although conditioning information could be injected only at the
	generator input, the signal may weaken as it propagates through successive layers. With CBN, after normalisation,
	condition-dependent scaling and shifting parameters are generated from the embedding and applied to the features.
	Applying CBN at each normalisation layer continuously injects the conditioning signal, ensuring features remain
	dependent on the target attributes throughout the generator.
	"""

	def __init__(self, num_features):
		super().__init__()

		# No learned affine parameters (they come from the condition)
		self.bn = nn.BatchNorm2d(num_features, affine=False)
		self.gamma = nn.Linear(CONDITION_DIM, num_features)
		self.beta = nn.Linear(CONDITION_DIM, num_features)

		# Initialise close to normal BatchNorm behaviour
		nn.init.zeros_(self.gamma.weight)
		nn.init.ones_(self.gamma.bias)
		nn.init.zeros_(self.beta.weight)
		nn.init.zeros_(self.beta.bias)

	def forward(self, x, cond):
		bn_out = self.bn(x)
		gamma = self.gamma(cond)[..., None, None]
		beta = self.beta(cond)[..., None, None]

		return gamma * bn_out + beta


class Generator(nn.Module):
	def __init__(self, *, latent_dim):
		super().__init__()

		# Label embeddings
		self.age_embedding = nn.Embedding(10, AGE_EMBEDDING_DIM)       # 10 age bins in training data
		self.gender_embedding = nn.Embedding(2, GENDER_EMBEDDING_DIM)  # Male/female
		self.race_embedding = nn.Embedding(5, RACE_EMBEDDING_DIM)      # 5 race classes

		self.conv1 = nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, bias=False)
		self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
		self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
		self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
		self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False)
		self.conv6 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

		self.cbn1 = ConditionalBatchNorm2d(512)
		self.cbn2 = ConditionalBatchNorm2d(256)
		self.cbn3 = ConditionalBatchNorm2d(128)
		self.cbn4 = ConditionalBatchNorm2d(64)
		self.cbn5 = ConditionalBatchNorm2d(32)

		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()

	def forward(self, z, labels=None, age_embedding=None, race_embedding=None):
		age = self.age_embedding(labels[:, 0]) if age_embedding is None else age_embedding
		gender = self.gender_embedding(labels[:, 1])
		race = self.race_embedding(labels[:, 2]) if race_embedding is None else race_embedding

		y = cat([age, gender, race], dim=1)            # -> (N, CONDITION_DIM)

		conv1_out = self.conv1(z)
		cbn1_out = self.relu(self.cbn1(conv1_out, y))  # -> (N, 512, 4, 4)

		conv2_out = self.conv2(cbn1_out)
		cbn2_out = self.relu(self.cbn2(conv2_out, y))  # -> (N, 256, 8, 8)

		conv3_out = self.conv3(cbn2_out)
		cbn3_out = self.relu(self.cbn3(conv3_out, y))  # -> (N, 128, 16, 16)

		conv4_out = self.conv4(cbn3_out)
		cbn4_out = self.relu(self.cbn4(conv4_out, y))  # -> (N, 64, 32, 32)

		conv5_out = self.conv5(cbn4_out)
		cbn5_out = self.relu(self.cbn5(conv5_out, y))  # -> (N, 32, 64, 64)

		conv6_out = self.tanh(self.conv6(cbn5_out))    # -> (N, 3, 64, 64)

		return conv6_out


class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()

		self.age_embedding = nn.Embedding(10, AGE_EMBEDDING_DIM)
		self.gender_embedding = nn.Embedding(2, GENDER_EMBEDDING_DIM)
		self.race_embedding = nn.Embedding(5, RACE_EMBEDDING_DIM)

		# Spectral normalisation constrains each layer's Lipschitz constant, stabilising the discriminator while
		# avoiding the batch-dependent statistics introduced by BatchNorm

		self.features = nn.Sequential(
			spectral_norm(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)),
			nn.LeakyReLU(0.2),
			spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
			nn.LeakyReLU(0.2),
			spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
			nn.LeakyReLU(0.2),
			spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
			nn.LeakyReLU(0.2)
		)

		# Projection: label embedding into feature space
		self.label_proj = spectral_norm(nn.Linear(CONDITION_DIM, 512))

		# Final real/fake head
		self.adv_head = spectral_norm(nn.Linear(512 * 4 * 4, 1))

	def forward(self, x, y):
		age = self.age_embedding(y[:, 0])
		gender = self.gender_embedding(y[:, 1])
		race = self.race_embedding(y[:, 2])

		cond = cat([age, gender, race], dim=1)                      # -> (N, CONDITION_DIM)

		# Projection term
		cond_emb = self.label_proj(cond)                            # -> (N, 512)

		features = self.features(x)                                 # -> (N, 512, 4, 4)
		features_flat = features.view(x.shape[0], -1)               # -> (N, 8192)
		features_pool = features.mean(dim=(2, 3))                   # -> (N, 512)

		# Real/fake score
		out = self.adv_head(features_flat)                          # -> (N, 1)

		# Inner product conditioning
		proj = (cond_emb * features_pool).sum(dim=1, keepdim=True)  # -> (N, 1)

		return out + proj
