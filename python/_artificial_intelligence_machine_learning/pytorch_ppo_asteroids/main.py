"""
Driver code

Author: Sam Barba
Created 11/07/2024

Controls:
	M: toggle metadata rendering (asteroids detected by spaceship)
	Space: toggle animation (during testing)
	Arrow keys/space: control spaceship (during manual playing)
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
	for init_asteroids in [8, 4]:
		for ts_reward in [0.1, -0.1]:
			for mb_size in [1024, 2048, 4096, 8192]:
				for gsteps_per_update in [5, 10, 20, 40, 80]:
					path = f'{init_asteroids}_{ts_reward}_{mb_size}_{gsteps_per_update}'
					rand1 = random.Random(1)
					rand2 = random.Random()
					train_env = GameEnv(random_obj=rand1, do_rendering=False, num_init_asteroids=init_asteroids, ts_reward=ts_reward, training_mode=True)
					checkpoint_env = GameEnv(random_obj=rand2, do_rendering=False, num_init_asteroids=4, ts_reward=ts_reward)
					agent = PPOAgent(path, mb_size, gsteps_per_update)
					total_return_per_epoch = agent.do_training(train_env, checkpoint_env)

					# Smooth data
					w = int(len(total_return_per_epoch) / 20)
					smoothed = savgol_filter(total_return_per_epoch, window_length=w, polyorder=3)

					plt.plot(total_return_per_epoch, alpha=0.4, linewidth=1, color='red')
					plt.plot(smoothed, linewidth=2, color='red')
					plt.xlabel('Epoch no.')
					plt.ylabel('Total return')
					plt.title('Total return per training epoch')
					plt.savefig(f'C:/Users/Sam/Desktop/x/{path}.png')
					plt.close()
					# plt.show()


def test():
	# env = GameEnv(random_obj=random.Random(), do_rendering=True)
	env = GameEnv(random_obj=random.Random(), do_rendering=True, num_init_asteroids=4, ts_reward=0)
	agent = PPOAgent(path='x', mbsize=1, gsteps=1)
	agent.load_model()
	paused = render_meta = True
	action = 0

	while True:
		env.reset()
		agent.buffer.clear()  # Don't need rollout buffer when testing
		state = env.get_state()
		terminal = False

		while not terminal:
			for event in pg.event.get():
				match event.type:
					case pg.QUIT: sys.exit()
					case pg.KEYDOWN:
						if event.key == pg.K_m:
							render_meta = not render_meta
						elif event.key == pg.K_SPACE:
							paused = not paused

			if not paused:
				action = agent.choose_action(state, True)
				_, state, terminal = env.step(action)

			env.render(action, render_meta=render_meta)


def play_manually():
	env = GameEnv(random_obj=random.Random(), do_rendering=True, num_init_asteroids=4, ts_reward=0.1, training_mode=True)

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

			env.render(action, render_meta=False)

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
