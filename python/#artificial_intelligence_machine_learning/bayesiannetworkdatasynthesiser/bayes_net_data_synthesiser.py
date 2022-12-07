"""
Data synthesiser from a Bayesian network

Author: Sam Barba
Created 02/12/2022
"""

import numpy as np

from bayes_net_plotter import plot_bayes_net

# Network as an adjacency matrix (A[i][j] = 1 means edge from node i to node j)
A = [
	[0, 0, 1, 0, 0, 0, 0, 0],
	[0, 0, 1, 1, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 1, 0, 0],
	[0, 0, 0, 0, 0, 1, 0, 0],
	[0, 0, 0, 0, 0, 1, 0, 0],
	[0, 0, 0, 0, 0, 0, 1, 1],
	[0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0]
]

def generate_data_point(adj_mat):
	dim = adj_mat.shape[0]
	x = np.zeros((dim, dim))

	# Get the source nodes
	sampled_nodes = list(np.where(adj_mat.sum(axis=0) == 0)[0])

	# Generate data for source nodes
	x[sampled_nodes] = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), size=len(sampled_nodes))

	next_gen_nodes = []
	while adj_mat.sum() != 0:
		# Identify the next generation of nodes
		with_only_parents = adj_mat[sampled_nodes].sum(axis=0)
		with_all = adj_mat.sum(axis=0)
		next_gen_nodes.extend(list(np.where(with_only_parents == with_all)[0]))

		# Remove already sampled nodes
		for n in sampled_nodes:
			if n in next_gen_nodes:
				next_gen_nodes.remove(n)

		submatrix_idx = np.ix_(sampled_nodes, next_gen_nodes)
		submatrix = adj_mat[submatrix_idx]
		# Generate data for the next generation nodes
		x[next_gen_nodes] = np.matmul(submatrix.T, x[sampled_nodes])

		sampled_nodes.extend(next_gen_nodes)
		next_gen_nodes = []
		adj_mat[:, sampled_nodes] = 0

	return x

if __name__ == '__main__':
	plot_bayes_net(np.array(A))
	print(generate_data_point(np.array(A)))
