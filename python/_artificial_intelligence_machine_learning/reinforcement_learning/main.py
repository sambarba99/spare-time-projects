"""
Reinforcement learning algorithms demo

Author: Sam Barba
Created 25/02/2022

- In a grid environment, an agent starts at a random cell and must find gold.
- Agent can use dynamic programming methods:
	- Policy iteration
	- Value iteration
- Or a Monte Carlo method:
	- On-policy first-visit MC control
- Or temporal difference methods:
	- SARSA
	- Q-learning
- Rewards are as follows:
	- +10 for locating gold
	- -10 for falling into hole
	- -1 per time step
"""

import tkinter as tk

from agent import Agent


if __name__ == '__main__':
	root = tk.Tk()
	root.title('Reinforcement learning demo')
	root.config(width=550, height=350, background='#101010')
	root.resizable(False, False)

	dp_methods_lbl = tk.Label(root, text='Dynamic Programming methods',
		font='consolas 12', background='#101010', foreground='white')
	mc_methods_lbl = tk.Label(root, text='Monte Carlo method',
		font='consolas 12', background='#101010', foreground='white')
	td_methods_lbl = tk.Label(root, text='Temporal Difference methods',
		font='consolas 12', background='#101010', foreground='white')

	btn_generate_env = tk.Button(root, text='Generate environment', font='consolas 12',
		command=lambda: agent.env.generate())
	btn_policy_iteration = tk.Button(root, text='Do policy iteration', font='consolas 12',
		command=lambda: agent.policy_iteration())
	btn_value_iteration = tk.Button(root, text='Do value iteration', font='consolas 12',
		command=lambda: agent.value_iteration())
	btn_on_policy_mc = tk.Button(root, text='Do on-policy first-visit MC control', font='consolas 12',
		command=lambda: agent.on_policy_first_visit_mc_control())
	btn_sarsa = tk.Button(root, text='Do SARSA', font='consolas 12', command=lambda: agent.sarsa())
	btn_q_learning = tk.Button(root, text='Do Q-learning', font='consolas 12',
		command=lambda: agent.q_learning())

	dp_methods_lbl.place(width=468, height=35, relx=0.5, y=102, anchor='center')
	mc_methods_lbl.place(width=468, height=35, relx=0.5, y=178, anchor='center')
	td_methods_lbl.place(width=468, height=35, relx=0.5, y=256, anchor='center')
	btn_generate_env.place(width=468, height=35, relx=0.5, y=56, anchor='center')
	btn_policy_iteration.place(width=231, height=35, relx=0.28, y=136, anchor='center')
	btn_value_iteration.place(width=231, height=35, relx=0.72, y=136, anchor='center')
	btn_on_policy_mc.place(width=468, height=35, relx=0.5, y=214, anchor='center')
	btn_sarsa.place(width=231, height=35, relx=0.28, y=294, anchor='center')
	btn_q_learning.place(width=231, height=35, relx=0.72, y=294, anchor='center')

	agent = Agent()

	root.mainloop()
