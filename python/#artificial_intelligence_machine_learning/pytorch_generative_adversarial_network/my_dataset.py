"""
Dataset class

Author: Sam Barba
Created 01/07/2023
"""

import os

from PIL import Image
import torch


class MyDataset(torch.utils.data.Dataset):
	def __init__(self, root_dir, transform):
		self.root_dir = root_dir
		self.transform = transform
		self.img_paths = [f'{root_dir}/{f}' for f in os.listdir(root_dir) if f.endswith('.jpg')]

	def __len__(self):
		return len(self.img_paths)

	def __getitem__(self, idx):
		image = Image.open(self.img_paths[idx])
		image = self.transform(image)
		return image
