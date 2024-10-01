"""
Driver code

Author: Sam Barba
Created 16/02/2023

Controls:
	M: toggle metadata rendering (gates/rays)
	Space: toggle animation (during testing)
	WASD: control car (during manual playing)
"""

import random
import sys

import matplotlib.pyplot as plt
import pygame as pg
from scipy.signal import savgol_filter
import torch

from pytorch_proximal_policy_optimisation.ddqn.ddqn_agent import DDQNAgent
from pytorch_proximal_policy_optimisation.ppo.ppo_agent import PPOAgent
from pytorch_proximal_policy_optimisation.game_env.game_env import GameEnv


plt.rcParams['figure.figsize'] = (10, 6)
random.seed(1)
torch.manual_seed(1)


def train(is_ppo):
	env = GameEnv(do_rendering=False)
	agent = PPOAgent(training_mode=True) if is_ppo else DDQNAgent(training_mode=True)
	total_return_per_epoch = agent.do_training(env)

	# Smooth data
	w = int(len(total_return_per_epoch) / 20)
	smoothed = savgol_filter(total_return_per_epoch, window_length=w, polyorder=3)

	plt.plot(total_return_per_epoch, alpha=0.4, linewidth=1, color='red')
	plt.plot(smoothed, linewidth=2, color='red')
	plt.xlabel('Epoch no.')
	plt.ylabel('Total return')
	plt.title('Total return per training epoch')
	plt.show()


def test(is_ppo):
	env = GameEnv(do_rendering=True)
	agent = PPOAgent(training_mode=False) if is_ppo else DDQNAgent(training_mode=False)
	agent.load_model()
	paused = render_meta = True
	action = 0

	while True:
		env.reset()
		state = env.get_state()
		terminal = False

		while not terminal:
			for event in pg.event.get():
				match event.type:
					case pg.QUIT:
						sys.exit()
					case pg.KEYDOWN:
						if event.key == pg.K_m:
							render_meta = not render_meta
						elif event.key == pg.K_SPACE:
							paused = not paused

			if not paused:
				action = agent.choose_action(state)
				_, state, terminal = env.step(action)

			env.render(action, render_meta, terminal)


def play_manually():
	env = GameEnv(do_rendering=True)

	while True:
		env.reset()
		terminal = False

		while not terminal:
			for event in pg.event.get():
				if event.type == pg.QUIT:
					sys.exit()

			keys_pressed = pg.key.get_pressed()

			accelerating = keys_pressed[pg.K_w]
			decelerating = keys_pressed[pg.K_s]
			turning_left = keys_pressed[pg.K_a]
			turning_right = keys_pressed[pg.K_d]

			if accelerating and turning_left: action = 5
			elif accelerating and turning_right: action = 6
			elif decelerating and turning_left: action = 7
			elif decelerating and turning_right: action = 8
			elif accelerating: action = 1
			elif decelerating: action = 2
			elif turning_left: action = 3
			elif turning_right: action = 4
			else: action = 0

			*_, terminal = env.step(action)

			env.render(action, render_meta=False, terminal=terminal)


if __name__ == '__main__':
	choice = input(
		'\nEnter 1 to play manually,'
		'\n2 to test existing PPO agent,'
		'\n3 to test existing DDQN agent,'
		'\n4 to train new PPO agent,'
		'\nor 5 to train new DDQN agent\n>>> '
	)

	match choice:
		case '1': play_manually()
		case '2': test(is_ppo=True)
		case '3': test(is_ppo=False)
		case '4': train(is_ppo=True)
		case '5': train(is_ppo=False)
		case _: print('\nBad input')
