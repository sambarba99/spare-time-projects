"""
Prioritised Replay Buffer implementation

Author: Sam Barba
Created 16/02/2023
"""

from collections import deque
import random

import torch

from ddqn.constants import *


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
		self.nodes = [0] * (2 * self.capacity - 1)
		self.transition_priorities = [None] * self.capacity
		self.write_idx = 0

	@property
	def total(self):
		return self.nodes[0]

	def add(self, priority, data):
		self.transition_priorities[self.write_idx] = data
		self.update(self.write_idx, priority)
		self.write_idx = (self.write_idx + 1) % self.capacity

	def update(self, data_idx, priority):
		"""
		Update the SumTree by changing a leaf value (transition priority).
		The change is propagated up to the root.
		"""

		child_idx = data_idx + self.capacity - 1  # Child index in tree array
		delta = priority - self.nodes[child_idx]
		while child_idx >= 0:
			self.nodes[child_idx] += delta
			child_idx = (child_idx - 1) // 2

	def get(self, cumulative_sum):
		"""
		Get the leaf number, leaf value and data value (transition priority)
		that correspond to a given cumulative sum
		"""

		assert cumulative_sum <= self.total

		idx = 0
		while 2 * idx + 1 < len(self.nodes):
			left = 2 * idx + 1
			right = left + 1

			if cumulative_sum <= self.nodes[left]:
				idx = left
			else:
				idx = right
				cumulative_sum -= self.nodes[left]

		data_idx = idx - self.capacity + 1

		return data_idx, self.nodes[idx], self.transition_priorities[data_idx]


class PrioritisedReplayBuffer:
	def __init__(self):
		"""See ddqn/constants.py for attribute descriptions"""

		self.capacity = PER_CAPACITY
		self.tree = SumTree(self.capacity)
		self.beta = PER_BETA
		self.max_priority = PER_EPSILON  # Priority for new samples, init as epsilon

		# Transition: (state, action, return, next_state, terminal)
		self.transitions = deque(maxlen=self.capacity)

		self.write_idx = 0
		self.size = 0

	def store_transition(self, state, action, return_, next_state, terminal):
		# Store transition index with maximum priority in sum tree
		self.tree.add(self.max_priority, self.write_idx)

		# Store transition in the buffer
		self.transitions.append((state, action, return_, next_state, terminal))

		self.write_idx = (self.write_idx + 1) % self.capacity
		self.size = min(self.size + 1, self.capacity)

	def sample(self, batch_size):
		priorities, tree_indices, sample_indices = [], [], []

		# To sample a batch of size k, the range [0, p_total] is divided equally into k ranges.
		# Next, a value is uniformly sampled from each range. Finally, the transitions (s, a, r, s', t)
		# that correspond to each of these sampled values are retrieved from the tree.
		segment_size = self.tree.total / batch_size

		for i in range(batch_size):
			segment_min = segment_size * i
			segment_max = segment_size * (i + 1)
			cumulative_sum = random.uniform(segment_min, segment_max)

			# tree_idx is the sample's index in the tree, needed to update priorities;
			# sample_idx is the sample's index in the buffer, needed to sample actual transitions
			tree_idx, priority, sample_idx = self.tree.get(cumulative_sum)

			priorities.append(priority)
			tree_indices.append(tree_idx)
			sample_indices.append(sample_idx)

		priorities = torch.FloatTensor(priorities).unsqueeze(dim=1)

		probs = priorities / self.tree.total

		# The estimation of the expected value with stochastic updates relies on those updates
		# corresponding to the same distribution as its expectation. PER introduces bias as it
		# changes this distribution in an uncontrolled fashion, therefore changing the solution
		# that the estimates will converge to. This can be corrected by using importance sampling
		# weights w_IS = (1/N * 1/P(i))^b (fully compensates for non-uniform probabilities if b = 1).
		weights_IS = (self.size * probs) ** -self.beta

		# Normalise weights to avoid large updates
		weights_IS /= weights_IS.max()

		# Anneal beta towards 1
		self.beta = min(self.beta + PER_BETA_INC, 1)

		batch = [self.transitions[i] for i in sample_indices]

		return batch, weights_IS, tree_indices

	def update_priorities(self, tree_indices, td_errors):
		# Add epsilon to prevent 0 priority, and apply prioritisation rate
		td_errors = (td_errors + PER_EPSILON) ** PER_ALPHA

		for data_idx, priority in zip(tree_indices, td_errors):
			self.tree.update(data_idx, priority)
			self.max_priority = max(self.max_priority, priority)
