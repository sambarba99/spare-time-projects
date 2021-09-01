# Closed-Loop Control System demo
# Author: Sam Barba
# Created 12/10/2021

import matplotlib.pyplot as plt

refInput = 1
controllerGain = 0.67
plantGain = 1
sensorGain = 1
timeDelay = 5
totalTime = 101

y = 0
yLim = (plantGain * controllerGain * refInput) / (1 + plantGain * controllerGain * sensorGain)

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

xPlot = [i for i in range(totalTime)]
refPlot = [refInput] * totalTime
yLimPlot = [yLim] * totalTime
yPlot = []

for i in range(totalTime):
	if i % timeDelay == 0:
		err = refInput - y
		y = err * controllerGain * plantGain

	yPlot.append(y)

plt.plot(xPlot, yPlot, color = "#0080ff")
plt.plot(xPlot, refPlot, color = "#008000")
plt.plot(xPlot, yLimPlot, "--", color = "#ff8000")
plt.legend(["y(t)", "ref", "yLim"])
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title("Closed-Loop Control Demo\nrefInput={}  cGain={}  pGain={}  sGain={}  delay={}".format(refInput, controllerGain, plantGain, sensorGain, timeDelay))
plt.show()
