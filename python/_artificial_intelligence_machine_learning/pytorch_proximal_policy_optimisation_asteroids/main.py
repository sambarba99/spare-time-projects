"""
Driver code

Controls:
	Space: toggle animation (during testing)
	Arrow keys/space: control spaceship (during manual playing)

Author: Sam Barba
Created 11/07/2024
"""

import random
import sys

import matplotlib.pyplot as plt
import pygame as pg
from scipy.signal import savgol_filter
import torch

from pytorch_ppo_asteroids.game_env.game_env import GameEnv
from pytorch_ppo_asteroids.ppo.ppo_agent import PPOAgent


plt.rcParams['figure.figsize'] = (10, 6)
torch.manual_seed(1)


def train():
	rand1 = random.Random(1)
	rand2 = random.Random()
	train_env = GameEnv(random_obj=rand1, do_rendering=False, training_mode=True)
	checkpoint_env = GameEnv(random_obj=rand2, do_rendering=False)
	agent = PPOAgent()
	total_return_per_epoch, mean_checkpoint_score = agent.do_training(train_env, checkpoint_env)

	# Smooth data
	w = len(total_return_per_epoch) // 20
	smoothed = savgol_filter(total_return_per_epoch, window_length=w, polyorder=3)

	plt.plot(total_return_per_epoch, alpha=0.4, linewidth=1, color='red')
	plt.plot(smoothed, linewidth=2, color='red')
	plt.xlabel('Epoch')
	plt.ylabel('Total return')
	plt.title('Total return per training epoch')
	plt.savefig('./training_return.png')
	plt.close()

	w = len(mean_checkpoint_score) // 10
	smoothed = savgol_filter(mean_checkpoint_score, window_length=w, polyorder=3)

	plt.plot(mean_checkpoint_score, alpha=0.4, linewidth=1, color='red')
	plt.plot(smoothed, linewidth=2, color='red')
	plt.title('Mean score per model checkpoint')
	plt.savefig('./test_score.png')
	plt.close()


def test():
	env = GameEnv(random_obj=random.Random(), do_rendering=True)
	agent = PPOAgent()
	agent.load_model()
	paused = True
	action = 0

	while True:
		# env.reset(171)  # Best found seed
		env.reset()
		agent.buffer.clear()  # Don't need rollout buffer when testing
		state = env.get_state()
		terminal = False

		while not terminal:
			for event in pg.event.get():
				match event.type:
					case pg.QUIT:
						sys.exit()
					case pg.KEYDOWN:
						if event.key == pg.K_SPACE:
							paused = not paused

			if not paused:
				action = agent.choose_action(state, True)
				_, state, terminal = env.step(action)

			env.render(action, terminal)


def play_manually():
	env = GameEnv(random_obj=random.Random(), do_rendering=True)

	while True:
		env.reset()
		shooting = terminal = False

		while not terminal:
			for event in pg.event.get():
				if event.type == pg.QUIT:
					sys.exit()
				elif event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
					# Player must tap spacebar to shoot, not hold down
					shooting = True

			keys_pressed = pg.key.get_pressed()

			boosting = keys_pressed[pg.K_UP]
			turning_left = keys_pressed[pg.K_LEFT]
			turning_right = keys_pressed[pg.K_RIGHT]

			if boosting and turning_left and shooting: action = 10
			elif boosting and turning_right and shooting: action = 11
			elif boosting and turning_left: action = 5
			elif boosting and turning_right: action = 6
			elif boosting and shooting: action = 7
			elif turning_left and shooting: action = 8
			elif turning_right and shooting: action = 9
			elif boosting: action = 1
			elif turning_left: action = 2
			elif turning_right: action = 3
			elif shooting: action = 4
			else: action = 0

			*_, terminal = env.step(action)

			env.render(action, terminal)

			shooting = False


if __name__ == '__main__':
	choice = input(
		'\nEnter 1 to play manually,'
		'\n2 to test existing agent,'
		'\nor 3 to train new agent\n>>> '
	)

	match choice:
		case '1': play_manually()
		case '2': test()
		case '3': train()
		case _: print('\nBad input')
