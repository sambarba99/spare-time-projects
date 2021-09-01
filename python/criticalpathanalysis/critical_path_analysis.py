"""
Demonstration of critical path analysis on this task data:

Example: engineering project
+------------------+------+----------+--------------+
| Task description | Code | Duration | Predecessors |
+------------------+------+----------+--------------+
|     Analysis     |  0   |   120    |     None     |
|      Design      |  1   |    60    |      0       |
|      Layout      |  2   |    15    |      0       |
| Request material |  3   |    3     |     1,2      |
|  Request parts   |  4   |    3     |     1,2      |
| Receive material |  5   |    7     |      3       |
|  Receive parts   |  6   |    7     |      4       |
|   Fabrication    |  7   |    25    |     2,5      |
|     Assembly     |  8   |    60    |    2,6,7     |
|     Testing      |  9   |    90    |      8       |
+------------------+------+----------+--------------+

Author: Sam Barba
Created 15/12/2021
"""

import pandas as pd

DATA = {'CODE': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
	'DURATION': [120, 60, 15, 3, 3, 7, 7, 25, 60, 90],
	'PREDECESSORS': [None, [0], [0], [1, 2], [1, 2], [3], [4], [2, 5], [2, 6, 7], [8]]}

pd.set_option('display.width', None)

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def forward_pass(df):
	# Early start, early finish
	es = [0] * len(df['CODE'])
	ef = es[:]

	for idx, p in enumerate(df['PREDECESSORS']):
		if p is not None:
			es[idx] = max(ef[task_code] for task_code in p)
		ef[idx] = es[idx] + df['DURATION'][idx]

	df['ES'] = es
	df['EF'] = ef

def backward_pass(df):
	# Late start, late finish
	ls = [0] * len(df['CODE'])
	lf = ls[:]
	successors = [None] * len(df['CODE'])

	for idx, p in reversed(list(enumerate(df['PREDECESSORS']))):
		if p is not None:
			for task_code in p:
				if successors[task_code] is None:
					successors[task_code] = [idx]  # idx = data['CODE'][idx]
				else:
					successors[task_code].append(idx)  # idx = data['CODE'][idx]

	for idx, arr in enumerate(successors):
		if arr is not None:
			successors[idx] = sorted(arr)

	for idx, s in reversed(list(enumerate(successors))):
		if s is None:
			lf[idx] = max(df['EF'])
		else:
			lf[idx] = min(ls[task_code] for task_code in s)
		ls[idx] = lf[idx] - df['DURATION'][idx]

	df['SUCCESSORS'] = successors
	df['LS'] = ls
	df['LF'] = lf

def compute_slack(df):
	slack = [ls - es for ls, es in zip(df['LS'], df['ES'])]
	critical = ['YES' if s == 0 else 'NO' for s in slack]

	df['SLACK'] = slack
	df['CRITICAL'] = critical

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def main():
	df = pd.DataFrame(DATA)

	# 1. Compute CPA

	forward_pass(df)
	backward_pass(df)
	compute_slack(df)

	# 2. Print tabular data
	print(df)

	# 3. Print critical path info

	critical_path = []
	critical_duration = 0
	for idx, slack_value in enumerate(df['SLACK']):
		if slack_value == 0:
			critical_path.append(idx)  # idx = data['CODE'][idx]
			critical_duration += df['DURATION'][idx]

	print(f'\nCritical path: {critical_path}')
	print(f'Critical duration: {critical_duration}')

if __name__ == '__main__':
	main()
