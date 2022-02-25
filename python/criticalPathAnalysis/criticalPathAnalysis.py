# Critical Path Analysis demo
# Author: Sam Barba
# Created 15/12/2021

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def forward_pass(data):
	# Early start, early finish
	es = [0] * len(data["CODE"])
	ef = es[:]

	for idx, p in enumerate(data["PREDECESSORS"]):
		if p is not None:
			es[idx] = max(ef[task_code] for task_code in p)
		ef[idx] = es[idx] + data["DURATION"][idx]

	data["ES"] = es
	data["EF"] = ef

def backward_pass(data):
	# Late start, late finish
	ls = [0] * len(data["CODE"])
	lf = ls[:]
	successors = [None] * len(data["CODE"])

	for idx, p in reversed(list(enumerate(data["PREDECESSORS"]))):
		if p is not None:
			for task_code in p:
				if successors[task_code] is None:
					successors[task_code] = [idx] # idx = data["CODE"][idx]
				else:
					successors[task_code].append(idx) # idx = data["CODE"][idx]

	for idx, s in reversed(list(enumerate(successors))):
		if s is None:
			lf[idx] = max(data["EF"])
		else:
			lf[idx] = min(ls[task_code] for task_code in s)
		ls[idx] = lf[idx] - data["DURATION"][idx]

	data["SUCCESSORS"] = successors
	data["LS"] = ls
	data["LF"] = lf

def compute_slack(data):
	slack = [ls - es for ls, es in zip(data["LS"], data["ES"])]
	critical = ["YES" if s == 0 else "NO" for s in slack]

	data["SLACK"] = slack
	data["CRITICAL"] = critical

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

# Task data (example: engineering project)
# | Task description | Code | Duration | Predecessors |
# |------------------+------+----------+--------------|
# |      Design      |  0   |   120    |     None     |
# |     Analysis     |  1   |    60    |      0       |
# |      Layout      |  2   |    15    |      0       |
# | Request material |  3   |    3     |     1,2      |
# |  Request parts   |  4   |    3     |     1,2      |
# | Receive material |  5   |    7     |      3       |
# |  Receive parts   |  6   |    7     |      4       |
# |   Fabrication    |  7   |    25    |     2,5      |
# |     Assembly     |  8   |    60    |    2,6,7     |
# |     Testing      |  9   |    90    |      8       |

data = {"CODE": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
	"DURATION": [120, 60, 15, 3, 3, 7, 7, 25, 60, 90],
	"PREDECESSORS": [None, [0], [0], [1, 2], [1, 2], [3], [4], [2, 5], [2, 6, 7], [8]]}

# 1. Compute CPA

forward_pass(data)
backward_pass(data)
compute_slack(data)

# 2. Print data as table

for k in data:
	print(f"{k:^12}", end="")
print()

for i in data["CODE"]:
	for v in data.values():
		if isinstance(v[i], list):
			# Printing predecessors or successors
			p = ",".join(str(n) for n in sorted(v[i]))
		else:
			# Convert to str in case v[i] is None
			p = str(v[i])
		print(f"{p:^12}", end="")
	print()

# 3. Print critical path info

critical_path = []
critical_duration = 0
for idx, slack_value in enumerate(data["SLACK"]):
	if slack_value == 0:
		critical_path.append(idx) # idx = data["CODE"][idx]
		critical_duration += data["DURATION"][idx]

print(f"\nCritical path: {critical_path}")
print(f"Critical duration: {critical_duration}")
