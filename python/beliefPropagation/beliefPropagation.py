# Belief propagation demo
# Author: Sam Barba
# Created 29/11/2021

from collections import defaultdict
import numpy as np

# Each dataset row is a fully observed coffee machine, with the state of every random variable.
# These binary RVs are:

# Failures (aim is to detect these)                          |     # The graphical model (Bayes net) showing how the variables are related (given as conditional probs):
# 0 - ne (no electricity)                                    |     p_ne (prob of no electricity)
# 1 - fp (fried power supply unit)                           |     p_fp (prob of fried PSU)
# 2 - fc (fried circuitry)                                   |     p_fc (prob of fried circuitry)
# 3 - wr (water reservoir empty)                             |     p_wr (prob of empty water reservoir)
# 4 - gh (group head gasket broken)                          |     p_gh (prob of broken group head gasket)
# 5 - dp (dead pump)                                         |     p_dp (prob of dead pump)
# 6 - fh (fried heating element)                             |     p_fh (prob of fried heating element)
#                                                            |
# Mechanism (unobservable)                                   |
# 7 - pw (power supply unit works)                           |     p_pw_ne_fp (prob that PSU works, given no electricity & fried PSU)
# 8 - cw (circuitry works)                                   |     p_cw_pw_fc (prob that circuitry works, given PSU works and & fried circuitry)
# 9 - gw (get water out of group head)                       |     p_gw_cw_wr_dp (prob of getting water, given circuitry works & water reservoir empty & dead pump)
#                                                            |
# Diagnostic (the tests a mechanic can run - observable)     |
# 10 - rl (room lights switch on)                            |     p_rl_ne (prob that lights switch on, given no electricity)
# 11 - vp (voltage measured across power supply unit)        |     p_vp_pw (prob that there's voltage across PSU, given PSU works)
# 12 - pl (power light switches on)                          |     p_pl_cw (prob that power light switches on, given circuitry works)
# 13 - wv (water visible in reservoir)                       |     p_wv_wr (prob that water is visible, given water reservoir empty)
# 14 - pa (pump is audible)                                  |     p_pa_dp (prob that pump is audible, given dead pump)
# 15 - me (makes espresso)                                   |     p_me_gw_gh (prob that makes espresso, given getting water & group head gasket broken)
# 16 - he (makes hot espresso)                               |     p_he_me_fh (prob that makes hot espresso, given makes espresso & fried heating element)

# These variables will ultimately represent conditional probability distributions.
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
itn = ["ne", "fp", "fc", "wr", "gh", "dp", "fh", "pw", "cw", "gw", "rl", "vp", "pl", "wv", "pa", "me", "he"]
# Name to index
nti = {name: idx for idx, name in enumerate(itn)}

# This list describes the above in a computer readable form, as the tuple (numpy array, human-readable name, list of RV indices, kind).
# The list of RV indices is aligned with the dimensions of the numpy array.
# Kind is F for failure, M for mechanism, D for diagnostic.
rvs = [(p_ne, "P(ne)", [nti["ne"]], "F"),
	(p_fp, "P(fp)", [nti["fp"]], "F"),
	(p_fc, "P(fc)", [nti["fc"]], "F"),
	(p_wr, "P(wr)", [nti["wr"]], "F"),
	(p_gh, "P(gh)", [nti["gh"]], "F"),
	(p_dp, "P(dp)", [nti["dp"]], "F"),
	(p_fh, "P(fh)", [nti["fh"]], "F"),
	(p_pw_ne_fp, "P(pw|ne,fp)", [nti["pw"], nti["ne"], nti["fp"]], "M"),
	(p_cw_pw_fc, "P(cw|pw,fc)", [nti["cw"], nti["pw"], nti["fc"]], "M"),
	(p_gw_cw_wr_dp, "P(gw|cw,wr,dp)", [nti["gw"], nti["cw"], nti["wr"], nti["dp"]], "M"),
	(p_rl_ne, "P(rl|ne)", [nti["rl"], nti["ne"]], "D"),
	(p_vp_pw, "P(vp|pw)", [nti["vp"], nti["pw"]], "D"),
	(p_pl_cw, "P(pl|cw)", [nti["pl"], nti["cw"]], "D"),
	(p_wv_wr, "P(wv|wr)", [nti["wv"], nti["wr"]], "D"),
	(p_pa_dp, "P(pa|dp)", [nti["pa"], nti["dp"]], "D"),
	(p_me_gw_gh, "P(me|gw,gh)", [nti["me"], nti["gw"], nti["gh"]], "D"),
	(p_he_me_fh, "P(he|me,fh)", [nti["he"], nti["me"], nti["fh"]], "D")]

RV_TO_FACTOR = 17 # For converting variables to factors for factor graph

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def calculateProbDistributions():
	for idx, rv in enumerate(rvs):
		indices = rv[2]

		for bitPermutation in generateBitPermutations(len(indices)):
			bits = list(bitPermutation[1:])
			bits2 = bits[:]
			bits2[0] = 1 - bits[0]

			alpha = np.all(data[:, indices] == bits, axis=1).sum() + 1
			beta = np.all(data[:, indices] == bits2, axis=1).sum() + 1

			rvs[idx][0][tuple(bits)] = alpha / (alpha + beta)

def generateBitPermutations(nBits):
	if nBits < 1:
		yield slice(None),
	else:
		for head in generateBitPermutations(nBits - 1):
			yield head + (0,)
			yield head + (1,)

# Calculate edges of factor graph - much simpler computation than on a Bayesian network
def calculateEdges():
	edges = []

	for rv in rvs:
		indices = rv[2]
		factor = indices[0] + RV_TO_FACTOR
		for idx in indices:
			edges.append((idx, factor))

	return sorted(edges)

# Convert list of edges into a valid message order (a message can only be sent from node A to node B
# once A has received all of its messages, except for the message from B)
def calculateMessageOrder(edges):
	msgOrder = []
	flags = defaultdict(dict)
	for a, b in edges:
		flags[a][b] = flags[b][a] = False

	check = True
	while check:
		check = False

		for src in flags:
			for dest in flags[src]: # For all destinations of this source...
				# If received messages from all neighbours except destination...
				if all(flags[src][d] for d in flags[src] if d != dest):
					tup = (src, dest)
					if tup not in msgOrder:
						msgOrder.append(tup)
						flags[dest][src] = check = True

	return msgOrder

def rvToFactorMsg(src, dest, msgs):
	msg = np.ones(2)
	incomingMessages = [msgs[src][d] for d in msgs[src] if d != dest]

	if incomingMessages:
		msg = np.prod(incomingMessages, axis=0)

	return msg

def factorToRVmsg(src, dest, msgs):
	msg = np.ones(2)
	subscriptChoices = "ijkl"

	for rv in rvs:
		if rv[2][0] == src - RV_TO_FACTOR:
			msg = rv[0]
			shape = subscriptChoices[:msg.ndim]

			for i in msgs[src]:
				if i != dest:
					axis = subscriptChoices[rv[2].index(i)]
					newShape = shape.replace(axis, "")
					subscripts = "{},{}->{}".format(shape, axis, newShape)
					msg = np.einsum(subscripts, msg, msgs[src][i])
					shape = newShape

	return msg

# Belief propagation - a trick is that if an RV is known (observed), then instead of using
# rvToFactorMsg whenever a message is sent from it, you send the known distribution instead
# (i.e. [1,0] for False or [0,1] for True).
# The 'known' parameter is a dictionary, where an RV index existing as a key in the dictionary
# indicates that it has been observed. The value obtained using the key is the value the RV
# has been observed as.
# This function returns a 17x2 matrix, such that [rv, 0] is the probability of RV being False,
# and [rv, 1] is the probability of being True.
def calculateMarginals(known, msgOrder):
	msgs = defaultdict(dict) # [dest][src] = msg

	for src, dest in msgOrder:
		if src < RV_TO_FACTOR: # RV
			if src in known:
				msgs[dest][src] = np.array([0,1] if known[src] else [1,0])
			else:
				msgs[dest][src] = rvToFactorMsg(src, dest, msgs)
		else: # Factor
			msgs[dest][src] = factorToRVmsg(src, dest, msgs)

	# Calculate and return marginal distributions ("beliefs")
	marginals = np.zeros((17, 2))
	for idx, m in enumerate(marginals):
		if idx in known:
			marginals[idx, :] = np.array([0,1] if known[idx] else [1,0])
		else:
			marginals[idx, :] = rvToFactorMsg(idx, None, msgs)
			marginals[idx, :] /= marginals[idx, :].sum() # Needed for numerical stability

	return marginals

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

# 1. Get data as ints

with open("C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\coffeeMachines.txt", "r") as file:
	data = file.readlines()[1:] # Skip header

data = [row.strip("\n").split() for row in data]
data = np.array(data).astype(int)

print("Data: {} exemplars, {} features".format(*data.shape))
print("No. working machines:", data[:, nti["he"]].sum()) # Works only if 'he' is true ("makes hot espresso")
print("No. broken machines:", len(data) - data[:, nti["he"]].sum())

# 2. Display RVs and their calculated probability distributions

calculateProbDistributions()

print("\nRVs:\n")
for rv in rvs:
	indices = rv[2]

	if len(indices) == 1:
		# Marginal
		print(f"{rv[1]} = {rv[0]}")
	else:
		# Conditional
		print(f"{rv[1]}:")
		for bitPermutation in generateBitPermutations(len(indices) - 1):
			eventName = itn[indices[0]]
			bits = ",".join([str(i) for i in bitPermutation[1:]])
			conditionals = rv[0][bitPermutation]
			print(f"    P({eventName}|{bits}) = {conditionals}")

# 3. Display factor graph

edges = calculateEdges()
print(f"\nGenerated {len(edges)} factor graph edges:\n")
for idx, e in enumerate(edges):
	print(f"{e[0]} ({itn[e[0]]}) - {e[1]}", end="    ")
	if (idx + 1) % 3 == 0:
		print()

# 4. Display message order

msgOrder = calculateMessageOrder(edges)
msgOrderStr = ", ".join(str(m[0]) + "->" + str(m[1]) for m in msgOrder)
print(f"\nMessage order ({len(msgOrder)} messages):\n\n{msgOrderStr}")

# 5. Display marginals ("beliefs")

print("\nMarginals with no observations:\n")
known = {}
marginals = calculateMarginals(known, msgOrder)
for idx, m in enumerate(marginals):
	print(f"P({itn[idx]}) = {m}")

print("\nMarginals if water reservoir empty:\n")
known = {nti["wr"]: True}
marginals = calculateMarginals(known, msgOrder)
for idx, m in enumerate(marginals):
	print(f"P({itn[idx]}) = {m}")
print()

# 6. Test on some example machines

machines = {"Machine A": {nti["me"]: True},
	"Machine B": {nti["wv"]: False, nti["pl"]: True},
	"Machine C": {nti["pa"]: False, nti["he"]: False},
	"Machine D": {nti["pa"]: True, nti["wv"]: True, nti["vp"]: True}}

for machineName, observations in machines.items():
	print(machineName + ":")
	for k, v in observations.items():
		print(f"    {itn[k]}: {v}")
	marginals = calculateMarginals(observations, msgOrder)
	idx = np.argmax(marginals[:7,1]) # Consider only the failure marginals (first 7 rows)
	print(f"    Most likely issue = {itn[idx]}")
