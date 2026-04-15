"""
Prioritised Replay Buffer implementation

Author: Sam Barba
Created 16/02/2023
"""

import random

import torch

from pytorch_proximal_policy_optimisation_racecar.ddqn.constants import *


class SumTree:
	"""
	A SumTree is a binary tree in which the value of a parent node is the sum of its children.
	Leaf nodes store transition (state, action, return, next_state, terminal) priorities, and
	internal nodes are intermediate sums. The root node contains the sum over all priorities
	(p_total). This provides an efficient way of calculating the cumulative sum of priorities,
	allowing O(log N) updates and sampling.
	"""

	def __init__(self, capacity):
		self.capacity = capacity
		self.tree = [0] * (2 * self.capacity - 1)
		self.data_idx = [None] * self.capacity
		self.write = 0

	@property
	def total(self):
		return self.tree[0]

	def add(self, priority, data_idx):
		leaf_idx = self.write + self.capacity - 1
		self.data_idx[self.write] = data_idx
		self.update(leaf_idx, priority)
		self.write = (self.write + 1) % self.capacity

	def update(self, tree_idx, priority):
		"""Update a leaf node and propagate the priority change up to the root"""

		delta = priority - self.tree[tree_idx]
		while True:
			self.tree[tree_idx] += delta
			if tree_idx == 0:
				break
			tree_idx = (tree_idx - 1) // 2

	def get(self, cumulative_sum):
		"""
		Traverse the tree to find the leaf whose cumulative priority range contains `cumulative_sum`.
		Uses a binary search over prefix sums.
		Returns: tree_idx, priority, data_idx
		"""

		# Clamp to guard against floating point drift exceeding the root sum
		cumulative_sum = min(cumulative_sum, self.total - 1e-8)

		idx = 0
		while 2 * idx + 1 < len(self.tree):
			left = 2 * idx + 1
			right = left + 1

			if cumulative_sum <= self.tree[left]:
				idx = left
			else:
				cumulative_sum -= self.tree[left]
				idx = right

		data_idx = idx - (self.capacity - 1)

		return idx, self.tree[idx], self.data_idx[data_idx]


class PrioritisedReplayBuffer:
	def __init__(self):
		"""See ddqn/constants.py for descriptions"""

		self.capacity = PER_CAPACITY
		self.tree = SumTree(self.capacity)
		self.beta = PER_BETA
		self.max_priority = PER_EPSILON  # Priority for new samples, init as epsilon

		# Contains transitions of format: (state, action, return, next_state, terminal)
		self.buffer = [None] * self.capacity
		self.write = 0
		self.size = 0

	def store_transition(self, state, action, return_, next_state, terminal):
		self.buffer[self.write] = (state, action, return_, next_state, terminal)

		# New samples get max priority
		self.tree.add(self.max_priority, self.write)

		self.write = (self.write + 1) % self.capacity
		self.size = min(self.size + 1, self.capacity)

	def sample(self, batch_size):
		assert self.size >= batch_size, f'Buffer only has {self.size} transitions; cannot sample {batch_size}'
		assert self.tree.total > 0, 'SumTree is empty'

		batch, tree_indices, priorities = [], [], []

		# To sample a batch of size k, the range [0, p_total] is divided equally into k ranges.
		# Next, a value is uniformly sampled from each range. Finally, the transitions (s, a, r, s', t)
		# that correspond to each of these sampled values are retrieved from the tree.
		segment_size = self.tree.total / batch_size

		for i in range(batch_size):
			a = segment_size * i
			b = segment_size * (i + 1)
			s = random.uniform(a, b)
			tree_idx, priority, data_idx = self.tree.get(s)

			batch.append(self.buffer[data_idx])
			tree_indices.append(tree_idx)
			priorities.append(priority)

		priorities = torch.tensor(priorities).float()
		probs = priorities / (self.tree.total + 1e-8)

		# The estimation of the expected value with stochastic updates relies on those updates corresponding to the
		# same distribution as its expectation. PER introduces bias as it changes this distribution in an uncontrolled
		# fashion, therefore changing the solution that the estimates will converge to. This can be corrected by using
		# importance sampling weights w = (1/N * 1/P(i))^b (fully compensates for non-uniform probabilities if b = 1).
		weights = (self.size * probs) ** -self.beta

		# Scale weights to [0,1] to avoid large updates
		weights /= weights.max()

		# Anneal beta towards 1
		self.beta = min(self.beta + PER_BETA_INC, 1)

		return batch, weights, tree_indices

	def update_priorities(self, tree_indices, td_errors):
		"""Higher TD error -> higher priority -> sampled more often"""

		# Add epsilon to prevent 0 priority, and apply prioritisation rate
		td_errors = torch.abs(td_errors) + PER_EPSILON
		priorities = td_errors ** PER_ALPHA

		for tree_idx, priority in zip(tree_indices, priorities):
			priority = priority.item()
			self.tree.update(tree_idx, priority)
			self.max_priority = max(self.max_priority, priority)
