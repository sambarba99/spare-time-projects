"""
Driver code

Controls:
	Left/right arrows: move cart
	D: activate DDQN agent
	Space: pause

Author: Sam Barba
Created 05/06/2026
"""

import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pygame as pg
from scipy.signal import savgol_filter
import torch

from pytorch_dueling_ddqn_cartpole_swingup.game_env.game_env import GameEnv
from pytorch_dueling_ddqn_cartpole_swingup.dueling_ddqn.dueling_ddqn_agent import DuelingDDQNAgent


random.seed(1)
plt.rcParams['figure.figsize'] = (10, 6)
np.random.seed(1)
torch.manual_seed(1)


def train():
	train_env = GameEnv(do_rendering=False, training_mode=True)
	checkpoint_env = GameEnv(do_rendering=False)
	agent = DuelingDDQNAgent(training_mode=True)
	total_reward_per_episode = agent.train(train_env, checkpoint_env)

	# Smooth data
	w = len(total_reward_per_episode) // 20
	smoothed = savgol_filter(total_reward_per_episode, window_length=w, polyorder=3, mode='mirror')

	plt.plot(total_reward_per_episode, alpha=0.4, linewidth=1, color='red')
	plt.plot(smoothed, linewidth=2, color='red')
	plt.xlabel('Episode')
	plt.ylabel('Total reward')
	plt.title('Total reward per training episode')
	plt.show()


def test():
	env = GameEnv(do_rendering=True)
	agent = DuelingDDQNAgent(training_mode=False)
	agent.load_model()
	paused = False
	action = 0

	while True:
		env.reset()
		state = env.get_state()
		agent_active = terminal = False

		while not terminal:
			for event in pg.event.get():
				match event.type:
					case pg.QUIT:
						sys.exit()
					case pg.KEYDOWN:
						match event.key:
							case pg.K_d:
								agent_active = not agent_active
								env.player = 'DDQN agent' if agent_active else 'you'
							case pg.K_SPACE:
								paused = not paused

			if not paused:
				if agent_active:
					action = agent.choose_action(state, greedy=True)
				else:
					keys_pressed = pg.key.get_pressed()

					left = keys_pressed[pg.K_LEFT]
					right = keys_pressed[pg.K_RIGHT]

					if left:
						action = 1
					elif right:
						action = 2
					else:
						action = 0

				_, state, terminal = env.step(action)

			env.render(action, terminal=terminal)


if __name__ == '__main__':
	choice = input(
		'\nEnter 1 to play/test agent,'
		'\nor 2 to train new agent\n>>> '
	)

	match choice:
		case '1': test()
		case '2': train()
		case _: print('\nBad input')
