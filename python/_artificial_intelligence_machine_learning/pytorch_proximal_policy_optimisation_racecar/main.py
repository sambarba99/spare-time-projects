"""
Driver code

Controls:
	WASD: control car
	Space: play/pause
	M: toggle metadata rendering (gates/rays)
	V: toggle state value rendering
	P: activate PPO agent

Author: Sam Barba
Created 16/02/2023
"""

import sys

import matplotlib.pyplot as plt
import pygame as pg
from scipy.signal import savgol_filter
import torch

from pytorch_proximal_policy_optimisation_racecar.ppo.ppo_agent import PPOAgent
from pytorch_proximal_policy_optimisation_racecar.game_env.game_env import GameEnv


plt.rcParams['figure.figsize'] = (10, 6)
torch.manual_seed(1)


def train():
	train_env = GameEnv(do_rendering=False)
	checkpoint_env = GameEnv(do_rendering=False)
	agent = PPOAgent(training_mode=True)
	total_reward_per_episode = agent.train(train_env, checkpoint_env)

	# Smooth data
	w = int(len(total_reward_per_episode) / 20)
	smoothed = savgol_filter(total_reward_per_episode, window_length=w, polyorder=3, mode='mirror')

	plt.plot(total_reward_per_episode, alpha=0.4, linewidth=1, color='red')
	plt.plot(smoothed, linewidth=2, color='red')
	plt.xlabel('Episode')
	plt.ylabel('Total reward')
	plt.title('Total reward per training episode')
	plt.show()


def test():
	env = GameEnv(do_rendering=True)
	ppo_agent = PPOAgent(training_mode=False)
	ppo_agent.load_model()
	paused = render_meta = render_state_values = False
	action = state_value = 0

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
							case pg.K_SPACE:
								paused = not paused
							case pg.K_m:
								render_meta = not render_meta
							case pg.K_v:
								render_state_values = not render_state_values
							case pg.K_p:
								agent_active = not agent_active
								env.driver = 'PPO agent' if agent_active else 'you'

			if not paused:
				if agent_active:
					action, state_value = ppo_agent.choose_action(state, greedy=True)
				else:
					keys_pressed = pg.key.get_pressed()

					accelerating = keys_pressed[pg.K_w]
					decelerating = keys_pressed[pg.K_s]
					turning_left = keys_pressed[pg.K_a]
					turning_right = keys_pressed[pg.K_d]

					if accelerating and turning_left:
						action = 5
					elif accelerating and turning_right:
						action = 6
					elif decelerating and turning_left:
						action = 7
					elif decelerating and turning_right:
						action = 8
					elif accelerating:
						action = 1
					elif decelerating:
						action = 2
					elif turning_left:
						action = 3
					elif turning_right:
						action = 4
					else:
						action = 0

				_, state, terminal = env.step(action)

			env.render(
				action,
				render_meta=render_meta,
				terminal=terminal,
				state_value=state_value if render_state_values else None
			)


if __name__ == '__main__':
	choice = input(
		'\nEnter 1 to play/test agent,'
		'\nor 2 to train new agent\n>>> '
	)

	match choice:
		case '1': test()
		case '2': train()
		case _: print('\nBad input')
