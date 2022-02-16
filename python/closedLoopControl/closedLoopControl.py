# Closed-Loop Control System demo
# Author: Sam Barba
# Created 12/10/2021

import matplotlib.pyplot as plt

ref_input = 1
controller_gain = 0.67
plant_gain = 1
sensor_gain = 1
time_delay = 5
total_time = 101

y = 0
yLim = (plant_gain * controller_gain * ref_input) / (1 + plant_gain * controller_gain * sensor_gain)

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

x_plot = list(range(total_time))
ref_plot = [ref_input] * total_time
y_lim_plot = [yLim] * total_time
y_plot = []

for i in range(total_time):
	if i % time_delay == 0:
		err = ref_input - y
		y = err * controller_gain * plant_gain

	y_plot.append(y)

plt.figure(figsize=(8, 6))
plt.plot(x_plot, y_plot, color="#0080ff")
plt.plot(x_plot, ref_plot, color="#008000")
plt.plot(x_plot, y_lim_plot, color="#ff8000", ls="--")
plt.legend(["y(t)", "ref", "yLim"])
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title(f"Closed-Loop Control Demo\nrefInput={ref_input}  cGain={controller_gain}  pGain={plant_gain}  sGain={sensor_gain}  delay={time_delay}")
plt.show()
