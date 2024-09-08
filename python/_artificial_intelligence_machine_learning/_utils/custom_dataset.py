"""
Custom dataset class

Author: Sam Barba
Created 26/03/2024
"""

from torch.utils.data import Dataset


class CustomDataset(Dataset):
	def __init__(self, *data):
		assert len(data) >= 1, 'Dataset needs at least 1 data iterable'
		assert len(set(map(len, data))) == 1, 'All data iterables must be the same length'

		self.data = data
		self.num_samples = len(data[0])

	def __getitem__(self, index):
		ret = [d[index] for d in self.data]

		return ret[0] if len(ret) == 1 else ret

	def __len__(self):
		return self.num_samples
