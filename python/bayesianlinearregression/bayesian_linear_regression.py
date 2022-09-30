"""
Bayesian linear regression demo

Author: Sam Barba
Created 03/03/2022
"""

import bayesian_utility
import matplotlib.pyplot as plt
import numpy as np

N_TRAIN = 14
N_VAL = N_TRAIN
N_TEST = 300
SIGMA = 0.3  # Noise = SIGMA ^ 2

plt.rcParams['figure.figsize'] = (9, 6)

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def plot_regression(approx_data, x_train, y_train, x_test, y_test, lam, lower_bounds=None, upper_bounds=None):
	plt.plot(x_test, y_test, color='#ff8000', ls='--', zorder=1, label='Ground truth')
	plt.scatter(x_train, y_train, color='black', zorder=2, label='Training samples')
	plt.plot(x_test, approx_data, color='#0080ff', zorder=3, label='Prediction')

	if lower_bounds is not None and upper_bounds is not None:
		plt.fill_between(x_test.flatten(), lower_bounds, upper_bounds, color='#0080ff', alpha=0.2,
			zorder=0, label='Error')
		plot_lim1 = np.min(lower_bounds) - 0.2
		plot_lim2 = np.max(upper_bounds) + 0.2
	else:
		plot_lim1 = np.min(np.append(y_train, y_test)) - 0.2
		plot_lim2 = np.max(np.append(y_train, y_test)) + 0.2

	plt.ylim([plot_lim1, plot_lim2])
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title(f'Regression with lambda = {lam:.3f}\n(alpha = {(lam / SIGMA ** 2):.3f})')
	plt.legend()
	plt.show()

def fit_pls(phi, y, lam):
	"""Partial least squares"""
	return np.linalg.inv(phi.T.dot(phi) + lam * np.identity(phi.shape[1])).dot(phi.T).dot(y)

def compute_posterior(phi, y, alpha, s2):
	"""
	Compute posterior mean (mu) and variance (sigma) for a Bayesian linear regression model with basis matrix
	phi and hyperparameters alpha and sigma^2, where lambda = alpha * sgima^2 (lambda = regularisation parameter)
	"""
	lam = alpha * s2
	mu = np.linalg.inv(phi.T.dot(phi) + lam * np.identity(phi.shape[1])).dot(phi.T).dot(y)
	sigma = s2 * np.linalg.inv(phi.T.dot(phi) + lam * np.identity(phi.shape[1]))
	return mu, sigma

def compute_log_marginal(phi, y, alpha, s2):
	"""
	Compute the logarithm of the marginal likelihood for a Bayesian linear regression model
	with basis matrix phi and hyperparameters alpha and sigma^2
	"""
	y = y.flatten()
	n = phi.shape[0]
	lml1 = (2 * np.pi) ** (-n / 2) * np.linalg.det(s2 * np.identity(n) + phi.dot(phi.T) / alpha) ** -0.5
	lml2 = np.exp(-0.5 * y.T.dot(np.linalg.inv(s2 * np.identity(n) + phi.dot(phi.T) / alpha)).dot(y))
	return np.log(lml1 * lml2)

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def main():
	# Synthesise datasets

	data_generator = bayesian_utility.DataGenerator(SIGMA ** 2)
	x_train, y_train = data_generator.get_data('TRAIN', N_TRAIN)
	x_val, y_val = data_generator.get_data('VALIDATION', N_VAL)
	x_test, y_test = data_generator.get_data('TEST', N_TEST)

	# Compute basis matrices for all 3 datasets - note that because a 'bias' function is used, we need n - 1
	# Gaussians to make the basis 'complete' (i.e. for m = n)

	m = N_TRAIN - 1
	centres = np.linspace(data_generator.min_x, data_generator.max_x, m)
	rbf_generator = bayesian_utility.RBFGenerator(centres=centres, radius=1)

	phi_train = rbf_generator.evaluate(x_train)
	phi_val = rbf_generator.evaluate(x_val)
	phi_test = rbf_generator.evaluate(x_test)

	# Plot regression for different lambda values
	for lam in [0, 0.01, 10]:
		w = fit_pls(phi_train, y_train, lam)
		approx_data = phi_test.dot(w)
		plot_regression(approx_data, x_train, y_train, x_test, y_test, lam)

	# Check consistency of fit_pls and compute_posterior
	# (let lambda = 0.01, so alpha = 0.01 / sigma^2)

	mu, _ = compute_posterior(phi_test, y_test, alpha=0.01 / SIGMA ** 2, s2=SIGMA ** 2)
	w = fit_pls(phi_test, y_test, lam=0.01)
	print('mu = PLS w:', all(mu == w))

	# Compute train, validation, and test set errors for the PLS model
	# - Also compute the negative log marginal likelihood (negative log evidence)
	# - Plot all curves on a graph

	v = np.linspace(-13, 5, 500)
	lam_vals = 10 ** v

	err_train = np.zeros(0)
	err_test = np.zeros(0)
	err_val = np.zeros(0)
	neg_log_evidence = np.zeros(0)

	for lam in lam_vals:
		w = fit_pls(phi_train, y_train, lam)
		train_pred = phi_train.dot(w)
		test_pred = phi_test.dot(w)
		val_pred = phi_val.dot(w)
		err_train = np.append(err_train, bayesian_utility.mae(train_pred, y_train))
		err_test = np.append(err_test, bayesian_utility.mae(test_pred, y_test))
		err_val = np.append(err_val, bayesian_utility.mae(val_pred, y_val))
		neg_log_evidence = np.append(neg_log_evidence,
			-compute_log_marginal(phi_train, y_train, lam / SIGMA ** 2, SIGMA ** 2))

	ax1 = plt.subplot()
	ax2 = ax1.twinx()

	ax1.plot(v, err_train, color='#cc0000', label='Train')
	ax1.plot(v, err_test, color='#0080ff', ls='--', label='Test')
	ax1.plot(v, err_val, color='#ff8000', ls='--', label='Val')
	ax1.set_xlabel('log(lambda)')
	ax1.set_ylabel('MAE')
	ax1.legend(loc='center left')

	ax2.plot(v, neg_log_evidence, color='#008000', label='Train')
	ax2.yaxis.label.set_color('#008000')
	ax2.tick_params(axis='y', colors='#008000')
	ax2.set_ylabel('-log(evidence)')
	ax2.legend(loc='center right')

	plt.show()

	min_err_val_idx = np.argmin(err_val)
	min_neg_log_evidence_idx = np.argmin(neg_log_evidence)

	print('\nMin point on test curve:', min(err_test))
	print('Point on test curve corresponding to min of val curve:', err_test[min_err_val_idx])
	print('Point on test curve corresponding to min of neg log evidence curve:', err_test[min_neg_log_evidence_idx])

	# Plotting regression with optimal lambda including 'error bars'
	# The predictive variance is sum of:
	# - Uncertainty due to noise (sigma^2)
	# - Uncertainty due to the parameter estimate being imprecise, encapsulated by posterior covariance sigma

	best_lam = 10 ** v[min_neg_log_evidence_idx]
	best_alpha = best_lam / SIGMA ** 2
	print('\nOptimal lambda =', best_lam)
	print('Optimal alpha =', best_alpha)

	mu, sigma = compute_posterior(phi_train, y_train, alpha=best_alpha, s2=SIGMA ** 2)
	y_posterior = phi_test.dot(mu).flatten()
	var_matrix = SIGMA ** 2 + phi_test.dot(sigma).dot(phi_test.T)
	var_matrix = np.diagonal(var_matrix)
	sd = var_matrix ** 0.5
	lower_bounds = y_posterior - sd
	upper_bounds = y_posterior + sd

	plot_regression(y_posterior, x_train, y_train, x_test, y_test, best_lam, lower_bounds, upper_bounds)

	# Print log marginal likelihood given optimal lambda and alpha
	print(f'\nLog marginal likelihood with sigma^2 = {SIGMA ** 2}, lambda = optimal, alpha = optimal:')
	print(compute_log_marginal(phi_train, y_train, best_alpha, SIGMA ** 2))

if __name__ == '__main__':
	main()
