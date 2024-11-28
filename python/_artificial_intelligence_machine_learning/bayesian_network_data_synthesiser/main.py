"""
Demo of data synthesis from a Bayesian network

Author: Sam Barba
Created 02/12/2022
"""

from graphviz import Digraph
import numpy as np
import pandas as pd


# Bayes net structure dictionary format: {node: [children]}
BAYES_STRUCTURE_DICT = {
	'A': ['C'],
	'B': ['C', 'D'],
	'C': ['F'],
	'D': ['F'],
	'E': ['F'],
	'F': []
}

# Marginal distributions (parent nodes)
MARG_A = np.array([0.6, 0.4])  # P(A=0), P(A=1)
MARG_B = np.array([0.3, 0.7])  # P(B=0), P(B=1)
MARG_E = np.array([0.5, 0.5])  # P(E=0), P(E=1)

# Conditional distributions
COND_D_GIVEN_B = np.array([
	[0.7, 0.3],  # P(D=0|B=0), P(D=1|B=0)
	[0.4, 0.6]   # P(D=0|B=1), P(D=1|B=1)
])
COND_C_GIVEN_AB = np.array([
	[[0.9, 0.1],  # P(C=0|A=0,B=0), P(C=1|A=0,B=0)
	[0.3, 0.7]],  # P(C=0|A=0,B=1), P(C=1|A=0,B=1)
	[[0.2, 0.8],  # P(C=0|A=1,B=0), P(C=1|A=1,B=0)
	[0.1, 0.9]]   # P(C=0|A=1,B=1), P(C=1|A=1,B=1)
])
COND_F_GIVEN_CDE = np.array([
	[[[0.88, 0.12],  # P(F=0|C=0,D=0,E=0), P(F=1|C=0,D=0,E=0)
	[0.01, 0.99]],   # P(F=0|C=0,D=0,E=1), P(F=1|C=0,D=0,E=1)
	[[0.64, 0.36],   # P(F=0|C=0,D=1,E=0), P(F=1|C=0,D=1,E=0)
	[0.75, 0.25]]],  # P(F=0|C=0,D=1,E=1), P(F=1|C=0,D=1,E=1)
	[[[0.33, 0.67],  # P(F=0|C=1,D=0,E=0), P(F=1|C=1,D=0,E=0)
	[0.07, 0.93]],   # P(F=0|C=1,D=0,E=1), P(F=1|C=1,D=0,E=1)
	[[0.48, 0.52],   # P(F=0|C=1,D=1,E=0), P(F=1|C=1,D=1,E=0)
	[0.59, 0.41]]]   # P(F=0|C=1,D=1,E=1), P(F=1|C=1,D=1,E=1)
])


def plot_bayes_net():
	# Set up global attributes

	g = Digraph(
		name='bayesian network',
		graph_attr={'fontname': 'consolas', 'labelloc': 't', 'label': 'Bayesian Network'},
		node_attr={'style': 'filled,setlinewidth(0)', 'fontname': 'consolas', 'shape': 'circle'}
	)

	# Create nodes

	for node in BAYES_STRUCTURE_DICT:
		g.node(node, label=node, color='#80c0ff')

	# Create edges

	for node, children in BAYES_STRUCTURE_DICT.items():
		for child in children:
			g.edge(node, child)

	g.render('./bayes_net', view=True, cleanup=True, format='png')


def generate_data(num_samples=1000):
	states = [0, 1]
	samples = []

	for _ in range(num_samples):
		# Sample parent nodes (A,B,E)
		sample_a = np.random.choice(states, p=MARG_A)
		sample_b = np.random.choice(states, p=MARG_B)
		sample_e = np.random.choice(states, p=MARG_E)

		# Sample D given B
		sample_d = np.random.choice(states, p=COND_D_GIVEN_B[sample_b])

		# Sample C given A,B
		sample_c = np.random.choice(states, p=COND_C_GIVEN_AB[sample_a, sample_b])

		# Sample F given C,D,E
		sample_f = np.random.choice(states, p=COND_F_GIVEN_CDE[sample_c, sample_d, sample_e])

		samples.append([sample_a, sample_b, sample_c, sample_d, sample_e, sample_f])

	return pd.DataFrame(data=samples, columns=list(BAYES_STRUCTURE_DICT))


if __name__ == '__main__':
	print()
	plot_bayes_net()
	print(generate_data())
