"""
Belief propagation in a Bayesian network demo

Author: Sam Barba
Created 29/11/2021
"""

import numpy as np
import pandas as pd

from graph_plotter import plot_graphical_model, plot_factor_graph


pd.set_option('display.max_columns', 11)
pd.set_option('display.width', None)

RV_TO_FACTOR = 17  # For converting variables to factors for factor graph

# These variables will ultimately represent marginal and conditional probability distributions.
# The naming convention is that p_ne contains P(ne), p_he_me_fh contains P(he|me,fh) etc.
# Indexing is in the same order as given in the variables' names.

p_ne = np.zeros(2)
p_fp = np.zeros(2)
p_fc = np.zeros(2)
p_wr = np.zeros(2)
p_gh = np.zeros(2)
p_dp = np.zeros(2)
p_fh = np.zeros(2)
p_pw_ne_fp = np.zeros((2, 2, 2))
p_cw_pw_fc = np.zeros((2, 2, 2))
p_gw_cw_wr_dp = np.zeros((2, 2, 2, 2))
p_rl_ne = np.zeros((2, 2))
p_vp_pw = np.zeros((2, 2))
p_pl_cw = np.zeros((2, 2))
p_wv_wr = np.zeros((2, 2))
p_pa_dp = np.zeros((2, 2))
p_me_gw_gh = np.zeros((2, 2, 2))
p_he_me_fh = np.zeros((2, 2, 2))

# Index to name
itn = ['ne', 'fp', 'fc', 'wr', 'gh', 'dp', 'fh', 'pw', 'cw', 'gw', 'rl', 'vp', 'pl', 'wv', 'pa', 'me', 'he']
# Name to index
nti = {name: idx for idx, name in enumerate(itn)}

# This list describes the above in a computer-readable form, as the tuple
# (numpy array, human-readable name, list of RV indices, kind).
# The list of RV indices is aligned with the dimensions of the numpy array.
# Kind is F for failure, M for mechanism, D for diagnostic.
rvs = [
	(p_ne, 'P(ne)', [nti['ne']], 'F'),
	(p_fp, 'P(fp)', [nti['fp']], 'F'),
	(p_fc, 'P(fc)', [nti['fc']], 'F'),
	(p_wr, 'P(wr)', [nti['wr']], 'F'),
	(p_gh, 'P(gh)', [nti['gh']], 'F'),
	(p_dp, 'P(dp)', [nti['dp']], 'F'),
	(p_fh, 'P(fh)', [nti['fh']], 'F'),
	(p_pw_ne_fp, 'P(pw|ne,fp)', [nti['pw'], nti['ne'], nti['fp']], 'M'),
	(p_cw_pw_fc, 'P(cw|pw,fc)', [nti['cw'], nti['pw'], nti['fc']], 'M'),
	(p_gw_cw_wr_dp, 'P(gw|cw,wr,dp)', [nti['gw'], nti['cw'], nti['wr'], nti['dp']], 'M'),
	(p_rl_ne, 'P(rl|ne)', [nti['rl'], nti['ne']], 'D'),
	(p_vp_pw, 'P(vp|pw)', [nti['vp'], nti['pw']], 'D'),
	(p_pl_cw, 'P(pl|cw)', [nti['pl'], nti['cw']], 'D'),
	(p_wv_wr, 'P(wv|wr)', [nti['wv'], nti['wr']], 'D'),
	(p_pa_dp, 'P(pa|dp)', [nti['pa'], nti['dp']], 'D'),
	(p_me_gw_gh, 'P(me|gw,gh)', [nti['me'], nti['gw'], nti['gh']], 'D'),
	(p_he_me_fh, 'P(he|me,fh)', [nti['he'], nti['me'], nti['fh']], 'D')
]

data = None


def generate_bit_permutations(n_bits):
	if n_bits < 1: yield slice(None),
	else:
		for head in generate_bit_permutations(n_bits - 1):
			yield head + (0,)
			yield head + (1,)


def calculate_prob_distributions():
	for idx, rv in enumerate(rvs):
		indices = rv[2]

		for bit_permutation in generate_bit_permutations(len(indices)):
			bits = list(bit_permutation[1:])
			bits2 = bits[:]
			bits2[0] = 1 - bits[0]

			alpha = (data[:, indices] == bits).all(axis=1).sum() + 1
			beta = (data[:, indices] == bits2).all(axis=1).sum() + 1

			rvs[idx][0][tuple(bits)] = alpha / (alpha + beta)


def calculate_edges():
	"""
	Calculate edges of factor graph - much simpler computation than on a Bayesian network
	"""

	edges = []

	for rv in rvs:
		indices = rv[2]
		factor = indices[0] + RV_TO_FACTOR
		for idx in indices:
			edges.append((idx, factor))

	return sorted(edges)


def calculate_message_order(edges):
	"""
	Convert list of edges into a valid message order (a message can only be sent from node A
	to node B once A has received all of its messages, except for the message from B)
	"""

	flags = dict()  # Nested dict
	for a, b in edges:
		if a not in flags: flags[a] = dict()
		if b not in flags: flags[b] = dict()
		flags[a][b] = flags[b][a] = False

	msg_order = []
	check = True

	while check:
		check = False

		for src in flags:
			for dest in flags[src]:  # For all destinations of this source...
				# If received messages from all neighbours except destination...
				if all(flags[src][d] for d in flags[src] if d != dest):
					if (tup := (src, dest)) not in msg_order:
						msg_order.append(tup)
						flags[dest][src] = check = True

	return msg_order


def calculate_marginals(known, msg_order):
	"""
	Belief propagation - a trick is that if an RV is known (observed), then instead of using
	rv_to_factor_msg whenever a message is sent from it, you send the known distribution instead
	(i.e. [1, 0] for False or [0, 1] for True).
	The 'known' parameter is a dictionary, where an RV index existing as a key in the dictionary
	indicates that it has been observed. The value obtained using the key is the value the RV
	has been observed as.
	This function returns a 17x2 matrix, such that [rv, 0] is the probability of RV being False,
	and [rv, 1] is the probability of being True.
	"""

	def rv_to_factor_msg(src, dest, msgs):
		msg = np.ones(2)
		incoming_messages = [msgs[src][d] for d in msgs[src] if d != dest]

		if incoming_messages:
			msg = np.prod(incoming_messages, axis=0)

		return msg

	def factor_to_rv_msg(src, dest, msgs):
		msg = np.ones(2)
		subscript_choices = 'ijkl'

		for rv in rvs:
			if rv[2][0] == src - RV_TO_FACTOR:
				msg = rv[0]
				shape = subscript_choices[:msg.ndim]

				for i in msgs[src]:
					if i != dest:
						axis = subscript_choices[rv[2].index(i)]
						new_shape = shape.replace(axis, '')
						subscripts = f'{shape},{axis}->{new_shape}'
						msg = np.einsum(subscripts, msg, msgs[src][i])
						shape = new_shape

		return msg


	# Nested dict, accessed via [dest][src] = msg
	msgs = {dest: dict() for _, dest in msg_order}

	for src, dest in msg_order:
		if src < RV_TO_FACTOR:  # src is a RV
			if src in known:
				msgs[dest][src] = np.array([0, 1] if known[src] else [1, 0])
			else:  # src is a factor
				msgs[dest][src] = rv_to_factor_msg(src, dest, msgs)
		else:  # Factor
			msgs[dest][src] = factor_to_rv_msg(src, dest, msgs)

	# Calculate and return marginal distributions ('beliefs')
	marginals = np.zeros((17, 2))
	for idx, _ in enumerate(marginals):
		if idx in known:
			marginals[idx, :] = np.array([0, 1] if known[idx] else [1, 0])
		else:
			marginals[idx, :] = rv_to_factor_msg(idx, None, msgs)
			marginals[idx, :] /= marginals[idx, :].sum()  # Normalise to ensure probs sum to 1

	return marginals


if __name__ == '__main__':
	# 1. Get coffee machine data

	df = pd.read_csv('C:/Users/Sam/Desktop/projects/datasets/coffee_machines.csv')
	print(f'\nRaw data:\n{df}\n')
	data = df.to_numpy()

	print('No. working machines:', data[:, nti['he']].sum())  # Works only if 'he' ('makes hot espresso') is true
	print('No. broken machines:', len(data) - data[:, nti['he']].sum())

	# 2. Display graphical model, RVs, and their calculated probability distributions

	plot_graphical_model(rvs)

	calculate_prob_distributions()

	print('\nRVs:\n')
	for rv in rvs:
		indices = rv[2]
		event_name = itn[indices[0]]

		if len(indices) == 1:
			# Marginal
			print(f'{rv[1]} = {rv[0]}')
		else:
			# Conditional
			print(f'{rv[1]}:')
			for bit_permutation in generate_bit_permutations(len(indices) - 1):
				bits = ','.join(str(b) for b in bit_permutation[1:])
				conditionals = rv[0][bit_permutation]
				print(f'\tP({event_name}|{bits}) = {conditionals}')

	# 3. Display factor graph

	edges = calculate_edges()
	plot_factor_graph(edges, itn, rvs)

	# 4. Display message order

	msg_order = calculate_message_order(edges)
	print(f'\nFactor graph message order ({len(msg_order)} messages):')
	for src, dest in msg_order:
		if src < RV_TO_FACTOR:  # src is a RV
			print(f'{src} ({itn[src]}) -> {dest} (factor)')
		else:  # src is a factor
			print(f'{src} (factor) -> {dest} ({itn[dest]})')

	# 5. Display marginals ('beliefs')

	print('\nMarginals with no observations:\n')
	marginals = calculate_marginals(known=dict(), msg_order=msg_order)
	for idx, m in enumerate(marginals):
		print(f'P({itn[idx]}) = {m}')

	print('\nMarginals if e.g. water reservoir empty:\n')
	known = {nti['wr']: True}
	marginals = calculate_marginals(known, msg_order)
	for idx, m in enumerate(marginals):
		print(f'P({itn[idx]}) = {m}')
	print()

	# 6. Test on some example machines

	machines = {
		'Machine A': {nti['me']: True},
		'Machine B': {nti['wv']: False, nti['pl']: True},
		'Machine C': {nti['pa']: False, nti['he']: False},
		'Machine D': {nti['pa']: True, nti['wv']: True, nti['vp']: True}
	}

	for machine_name, observations in machines.items():
		print(f'{machine_name}:')
		for k, v in observations.items():
			print(f'\t{itn[k]}: {v}')
		marginals = calculate_marginals(observations, msg_order)
		idx = marginals[:7, 1].argmax()  # Consider only the failure marginals (first 7 rows)
		print(f'\tMost likely issue = {itn[idx]}')
