"""
Decision tree demo

Author: Sam Barba
Created 03/11/2021
"""

import matplotlib.pyplot as plt
import numpy as np

feature_names = None

plt.rcParams['figure.figsize'] = (7, 5)

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def extract_data(path, train_test_ratio=0.8):
	"""Split file data into train/test"""

	data = np.genfromtxt(path, dtype=str, delimiter='\n')
	feature_names = data[0].strip().split(',')
	# Skip header and convert to floats
	data = [row.split() for row in data[1:]]
	data = np.array(data).astype(float)
	np.random.shuffle(data)

	x, y = data[:, :-1], data[:, -1].astype(int)

	split = int(len(data) * train_test_ratio)

	x_train, y_train = x[:split], y[:split]
	x_test, y_test = x[split:], y[split:]

	return feature_names, x_train, y_train, x_test, y_test

def build_tree(x, y, max_depth=1000):
	def find_best_split(x, y):
		"""
		Given a dataest and its target values, find the optimal combination
		of feature and split point that yields maximum information gain
		"""

		def calculate_entropy(y):
			if len(y) <= 1: return 0

			counts = np.bincount(y)
			probs = counts[np.nonzero(counts)] / len(y)  # np.nonzero ensures that we're not doing log(0) after

			return -(probs * np.log2(probs)).sum()

		parent_entropy = calculate_entropy(y)
		best = {'infoGain': -1}

		# Loop every possible split of every dimension
		for i in range(x.shape[1]):
			for split_threshold in np.unique(x[:, i]):
				left_indices = np.where(x[:, i] <= split_threshold)
				right_indices = np.where(x[:, i] > split_threshold)
				left = y[left_indices]
				right = y[right_indices]
				info_gain = parent_entropy - len(left) / len(y) * calculate_entropy(left) \
					- len(right) / len(y) * calculate_entropy(right)

				if info_gain > best['infoGain']:
					best = {'feature': i,
						'splitThreshold': split_threshold,
						'infoGain': info_gain,
						'leftIndices': left_indices,
						'rightIndices': right_indices}

		return best

	# Generate leaf node if either stopping condition has been reached
	if max_depth == 1 or np.all(y == y[0]):
		classes, counts = np.unique(y, return_counts=True)
		return {'leaf': True, 'class': classes[np.argmax(counts)]}
	else:
		move = find_best_split(x, y)
		left = build_tree(x[move['leftIndices']], y[move['leftIndices']], max_depth - 1)
		right = build_tree(x[move['rightIndices']], y[move['rightIndices']], max_depth - 1)

		return {'leaf': False,
			'feature': move['feature'],
			'splitThreshold': move['splitThreshold'],
			'infoGain': move['infoGain'],
			'left': left,
			'right': right}

def make_best_tree(x_train, y_train, x_test, y_test):
	"""Test different max depth values, and create tree with the best one"""

	def evaluate(x_train, y_train, x_test, y_test, max_depth):
		tree = build_tree(x_train, y_train, max_depth)
		train_predictions = np.array([predict(tree, sample) for sample in x_train])
		test_predictions = np.array([predict(tree, sample) for sample in x_test])
		train_acc = (train_predictions == y_train).sum() / len(y_train)
		test_acc = (test_predictions == y_test).sum() / len(y_test)

		return tree, train_acc, test_acc

	best_tree = None
	best_depth = best_train_acc = best_test_acc = -1
	depth = 2

	while True:
		tree, train_acc, test_acc = evaluate(x_train, y_train, x_test, y_test, depth)
		print(f'Depth {depth}: training accuracy = {train_acc} | test accuracy = {test_acc}')

		conditions = [train_acc >= best_train_acc,
			test_acc >= best_test_acc,
			not (train_acc == best_train_acc and test_acc == best_test_acc)]

		if all(conditions):
			best_tree = tree
			best_depth = depth
			best_train_acc = train_acc
			best_test_acc = test_acc
		else:
			break  # No improvement, so stop

		depth += 1

	return best_tree, best_depth

def predict(tree, sample):
	if tree['leaf']:
		return tree['class']
	else:
		if sample[tree['feature']] <= tree['splitThreshold']:
			return predict(tree['left'], sample)
		else:
			return predict(tree['right'], sample)

def print_tree(tree, classes, indent=0):
	if tree['leaf']:
		print(' ' * indent + classes[tree['class']])
	else:
		f = tree['feature']
		print('{}x{} ({}) <= {}'.format(' ' * indent, f, feature_names[f], tree['splitThreshold']))
		print_tree(tree['left'], classes, indent + 4)
		print('{}x{} ({}) > {}'.format(' ' * indent, f, feature_names[f], tree['splitThreshold']))
		print_tree(tree['right'], classes, indent + 4)

def confusion_matrix(predictions, actual):
	n_classes = len(np.unique(actual))
	conf_mat = np.zeros((n_classes, n_classes)).astype(int)

	for a, p in zip(actual, predictions):
		conf_mat[a][p] += 1

	accuracy = np.trace(conf_mat) / conf_mat.sum()
	return conf_mat, accuracy

def plot_confusion_matrices(train_conf_mat, train_acc, test_conf_mat, test_acc):
	# axes[0] = training confusion matrix
	# axes[1] = test confusion matrix
	_, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
	axes[0].matshow(train_conf_mat, cmap=plt.cm.Blues, alpha=0.7)
	axes[1].matshow(test_conf_mat, cmap=plt.cm.Blues, alpha=0.7)
	axes[0].xaxis.set_ticks_position('bottom')
	axes[1].xaxis.set_ticks_position('bottom')
	for (j, i), val in np.ndenumerate(train_conf_mat):
		axes[0].text(x=i, y=j, s=val, ha='center', va='center')
		axes[1].text(x=i, y=j, s=test_conf_mat[j][i], ha='center', va='center')
	axes[0].set_xlabel('Predictions')
	axes[0].set_ylabel('Actual')
	axes[0].set_title(f'Training Confusion Matrix\nAccuracy = {train_acc:.3f}')
	axes[1].set_title(f'Test Confusion Matrix\nAccuracy = {test_acc:.3f}')
	plt.show()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def main():
	global feature_names

	choice = input('Enter B to use breast tumour dataset,'
		+ '\nI for iris dataset,'
		+ '\nP for pulsar dataset,'
		+ '\nT for Titanic dataset,'
		+ '\nor W for wine dataset\n>>> ').upper()
	print()

	match choice:
		case 'B':
			path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\breastTumourData.txt'
			classes = ['malignant', 'benign']
		case 'I':
			path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\irisData.txt'
			classes = ['setosa', 'versicolor', 'virginica']
		case 'P':
			path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\pulsarData.txt'
			classes = ['not pulsar', 'pulsar']
		case 'T':
			path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\titanicData.txt'
			classes = ['did not survive', 'survived']
		case _:
			path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\wineData.txt'
			classes = ['class 0', 'class 1', 'class 2']

	feature_names, x_train, y_train, x_test, y_test = extract_data(path)

	tree, depth = make_best_tree(x_train, y_train, x_test, y_test)

	print(f'\nOptimal tree (depth {depth}):\n')

	print_tree(tree, classes)

	# Plot confusion matrices

	train_predictions = [predict(tree, i) for i in x_train]
	test_predictions = [predict(tree, i) for i in x_test]
	train_conf_mat, train_acc = confusion_matrix(train_predictions, y_train)
	test_conf_mat, test_acc = confusion_matrix(test_predictions, y_test)

	plot_confusion_matrices(train_conf_mat, train_acc, test_conf_mat, test_acc)

if __name__ == '__main__':
	main()
