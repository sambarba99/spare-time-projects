"""
Kalman filter demo

Author: Sam Barba
Created 06/10/2024
"""

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['figure.figsize'] = (8, 5)

DT = 0.1                               # Timestep (s)
TIME = np.arange(0, 30, DT)            # Time from 0 to 30 secs
ACC = 2                                # Rocket acceleration (m/s^2)
TRUE_ALTITUDE = 0.5 * ACC * TIME ** 2  # Based on kinematic equation
NOISE_STD = 50                         # Standard deviation of measurement noise


if __name__ == '__main__':
	# Simulate noisy altitude measurements

	measurements = TRUE_ALTITUDE + np.random.normal(0, NOISE_STD, len(TIME))

	# Initialise Kalman filter variables

	x = np.array([[0], [0]])          # State vector (containing altitude and velocity, both initially 0)
	F = np.array([[1, DT], [0, 1]])   # State transition matrix
	H = np.array([[1, 0]])            # Measurement matrix (we only measure altitude)
	P = np.eye(2)                     # State covariance matrix
	Q = np.array([[1, 0], [0, 1]])    # Process noise covariance matrix
	R = np.array([[NOISE_STD ** 2]])  # Measurement covariance matrix (1 element, as we're just measuring altitude)

	# Kalman filter loop

	estimated_altitudes = []

	for z in measurements:
		# Prediction step
		x = F.dot(x)               # State prediction
		P = F.dot(P).dot(F.T) + Q  # Covariance prediction

		# Update step
		y = z - H.dot(x)                      # Measurement residual (innovation)
		S = H.dot(P).dot(H.T) + R             # Innovation covariance
		K = P.dot(H.T).dot(np.linalg.inv(S))  # Kalman gain

		x += K.dot(y)         # State update (correct the prediction)
		P -= K.dot(H).dot(P)  # Covariance update

		# Store estimate
		estimated_altitudes.append(x[0, 0])

	# Plot results

	plt.plot(TIME, TRUE_ALTITUDE, label='True altitude', color='black')
	plt.scatter(TIME, measurements, label='Noisy measurements', s=4, color='#6080ff')
	plt.plot(TIME, estimated_altitudes, label='Kalman filter estimate', color='red')
	plt.xlabel('Time (s)')
	plt.ylabel('Altitude (m)')
	plt.title('Rocket Altitude Estimation via Kalman Filtering')
	plt.legend()
	plt.show()
