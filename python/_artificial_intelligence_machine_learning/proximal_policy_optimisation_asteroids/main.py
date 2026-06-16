"""
Driver code

Controls:
	W/A/D/space: control spaceship
	K: pause
	P: activate PPO agent

Author: Sam Barba
Created 2024-07-11
"""

from pathlib import Path
import random
import sys

import matplotlib.pyplot as plt
import pygame as pg
from scipy.signal import savgol_filter
import torch

from proximal_policy_optimisation_asteroids.game_env.game_env import GameEnv
from proximal_policy_optimisation_asteroids.ppo.ppo_agent import PPOAgent


plt.rcParams['figure.figsize'] = (10, 6)
torch.manual_seed(1)


def train():
	rand1 = random.Random(1)
	rand2 = random.Random()
	train_env = GameEnv(random_obj=rand1, do_rendering=False, training_mode=True)
	checkpoint_env = GameEnv(random_obj=rand2, do_rendering=False)
	agent = PPOAgent(training_mode=True)
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


def find_best_test_seeds(num_runs=1000):
	env = GameEnv(random_obj=random.Random(), do_rendering=False)
	agent = PPOAgent(training_mode=False)
	results = []

	for p in Path('./ppo').glob('*.pth'):
		print('\nTesting', p.name)
		agent.load_model(p.resolve())
		total_score = 0
		seeds_and_scores = []

		for i in range(1, num_runs + 1):
			env.reset(i)
			state = env.get_state()
			terminal = False

			while not terminal:
				action = agent.choose_action(state, greedy=True)
				_, state, terminal = env.step(action)

			total_score += env.spaceship.score
			seeds_and_scores.append((i, env.spaceship.score))

			print(f'Seed: {i}/{num_runs}  |  '
				f'score: {env.spaceship.score}  |  '
				f'running mean: {(total_score / i):.1f}')

		# Sort by score descending, and keep the top 5
		seeds_and_scores = sorted(seeds_and_scores, key=lambda i: -i[1])[:5]

		results.append((p.name, total_score / num_runs, tuple(seeds_and_scores)))

	# Sort by best score (desc), then mean score (desc)
	results.sort(key=lambda i: (-i[2][0][1], -i[1]))

	print('\nModel path | Mean score | Best seeds/scores')
	for row in results:
		print(*row)


def test():
	env = GameEnv(random_obj=random.Random(), do_rendering=True)
	agent = PPOAgent(training_mode=False)
	agent.load_model()
	paused = False
	action = 0

	while True:
		# env.reset(482)  # Demo seed
		env.reset()
		state = env.get_state()
		shooting = agent_active = terminal = False
		# agent_active = True  # Uncomment if using a seed

		while not terminal:
			for event in pg.event.get():
				match event.type:
					case pg.QUIT:
						sys.exit()
					case pg.KEYDOWN:
						match event.key:
							case pg.K_SPACE:
								# Player must tap spacebar to shoot, not hold down
								shooting = True
							case pg.K_k:
								paused = not paused
							case pg.K_p:
								agent_active = not agent_active
								env.player = 'PPO agent' if agent_active else 'you'

			if not paused:
				if agent_active:
					action = agent.choose_action(state, greedy=True)
				else:
					keys_pressed = pg.key.get_pressed()

					boosting = keys_pressed[pg.K_w]
					turning_left = keys_pressed[pg.K_a]
					turning_right = keys_pressed[pg.K_d]

					if boosting and turning_left and shooting:
						action = 10
					elif boosting and turning_right and shooting:
						action = 11
					elif boosting and turning_left:
						action = 5
					elif boosting and turning_right:
						action = 6
					elif boosting and shooting:
						action = 7
					elif turning_left and shooting:
						action = 8
					elif turning_right and shooting:
						action = 9
					elif boosting:
						action = 1
					elif turning_left:
						action = 2
					elif turning_right:
						action = 3
					elif shooting:
						action = 4
					else:
						action = 0

				_, state, terminal = env.step(action)

			env.render(action, terminal=terminal)

			shooting = False  # Reset for next loop


if __name__ == '__main__':
	choice = input(
		'\nEnter 1 to play/test agent,'
		'\n2 to find the best test seed,'
		'\nor 3 to train new agent\n>>> '
	)

	match choice:
		case '1': test()
		case '2': find_best_test_seeds()
		case '3': train()
		case _: print('\nBad input')
