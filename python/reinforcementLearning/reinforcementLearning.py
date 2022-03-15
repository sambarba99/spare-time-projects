# Reinforcement learning algorithms demo
# Author: Sam Barba
# Created 25/02/2022

# - In a grid environment, an agent starts at a random cell and must find gold.
# - Agent can use dynamic programming methods:
# 		- Policy iteration
# 		- Value iteration
#   Or a Monte Carlo method:
#       - On-policy first-visit MC control
#   Or temporal difference methods:
#		- SARSA
# 		- Q-learning
# - Rewards are as follows:
# 		- +10 for locating gold
# 		- -10 for falling into hole
# 		- -1 per time step

from agent import Agent
from gridenvironment import GridEnv
import tkinter as tk

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

agent = Agent()
agent.env = GridEnv()

root = tk.Tk()
root.title("Reinforcement learning demo")
root.configure(width=620, height=400, bg="#141414")

frame = tk.Frame(root, bg="#0080ff")
frame.place(relwidth=0.9, relheight=0.9, relx=0.5, rely=0.5, anchor="center")

dp_methods_lbl = tk.Label(frame, text="Dynamic Programming methods", font="consolas", bg="#0080ff")
mc_methods_lbl = tk.Label(frame, text="Monte Carlo method", font="consolas", bg="#0080ff")
td_methods_lbl = tk.Label(frame, text="Temporal Difference methods", font="consolas", bg="#0080ff")
dp_methods_lbl.place(relwidth=0.85, relheight=0.1, relx=0.5, rely=0.29, anchor="center")
mc_methods_lbl.place(relwidth=0.85, relheight=0.1, relx=0.5, rely=0.51, anchor="center")
td_methods_lbl.place(relwidth=0.85, relheight=0.1, relx=0.5, rely=0.73, anchor="center")

btn_generate_env = tk.Button(frame, text="Generate environment", font="consolas", command=lambda: agent.env.generate())
btn_policy_iteration = tk.Button(frame, text="Do policy iteration", font="consolas", command=lambda: agent.policy_iteration())
btn_value_iteration = tk.Button(frame, text="Do value iteration", font="consolas", command=lambda: agent.value_iteration())
btn_on_policy_mc = tk.Button(frame, text="Do on-policy first-visit MC control", font="consolas", command=lambda: agent.on_policy_first_visit_mc_control())
btn_sarsa = tk.Button(frame, text="Do SARSA", font="consolas", command=lambda: agent.sarsa())
btn_q_learning = tk.Button(frame, text="Do Q-learning", font="consolas", command=lambda: agent.q_learning())
btn_generate_env.place(relwidth=0.85, relheight=0.1, relx=0.5, rely=0.16, anchor="center")
btn_policy_iteration.place(relwidth=0.42, relheight=0.1, relx=0.28, rely=0.39, anchor="center")
btn_value_iteration.place(relwidth=0.42, relheight=0.1, relx=0.72, rely=0.39, anchor="center")
btn_on_policy_mc.place(relwidth=0.85, relheight=0.1, relx=0.5, rely=0.61, anchor="center")
btn_sarsa.place(relwidth=0.42, relheight=0.1, relx=0.28, rely=0.84, anchor="center")
btn_q_learning.place(relwidth=0.42, relheight=0.1, relx=0.72, rely=0.84, anchor="center")

root.mainloop()
