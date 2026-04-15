"""
Double Deep Q Network (DDQN) agent implementation

Author: Sam Barba
Created 16/02/2023
"""

import numpy as np
import torch
from torch import nn

from pytorch_proximal_policy_optimisation_racecar.ddqn.constants import *
from pytorch_proximal_policy_optimisation_racecar.ddqn.prioritised_replay_buffer import PrioritisedReplayBuffer


def build_network():
	return nn.Sequential(
		nn.Linear(NUM_INPUTS, LAYER_SIZE),
		nn.LeakyReLU(),
		nn.Linear(LAYER_SIZE, LAYER_SIZE),
		nn.LeakyReLU(),
		nn.Linear(LAYER_SIZE, NUM_ACTIONS)
	)


class DDQNAgent:
	def __init__(self, training_mode):
		# If training, start epsilon at 1 and decay. Otherwise, set to 0 (i.e. act greedily; no exploration)
		self.epsilon = 1 if training_mode else 0
		self.policy = build_network()  # Online model, used for action selection (equivalent to actor in PPO)
		if training_mode:
			self.target = build_network()  # Offline model, used for action evaluation (equivalent to critic in PPO)
			self.replay_buffer = PrioritisedReplayBuffer()
			self.optimiser = torch.optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)

	def do_training(self, train_env, checkpoint_env):
		# Pretraining step: populate agent's memory with transitions

		print('\nPretraining (populating agent memory)...')

		state = train_env.get_state()

		for _ in range(WARMUP_SAMPLES):
			# Choose action given current state
			action = self.choose_action(state)
			return_, next_state, terminal = train_env.step(action)

			# Store transition for experience replay
			self.replay_buffer.store_transition(state, action, return_, next_state, terminal)

			if terminal:
				train_env.reset()
				state = train_env.get_state()
			else:
				# Update for next loop
				state = next_state

		# Now agent can be trained

		print('\nTraining...\n')

		total_return_per_epoch = []

		for i in range(1, NUM_EPOCHS + 1):
			train_env.reset()
			state = train_env.get_state()
			total_return = no_reward_count = 0

			for _ in range(MAX_EP_LENGTH):
				action = self.choose_action(state)
				return_, next_state, terminal = train_env.step(action)
				self.replay_buffer.store_transition(state, action, return_, next_state, terminal)

				total_return += return_
				no_reward_count = no_reward_count + 1 if return_ <= 0 else 0

				if terminal or no_reward_count > NO_REWARD_LIMIT:
					break

				state = next_state

			total_return_per_epoch.append(total_return)

			# Do experience replay (learning) with a batch of data
			batch, weights, tree_indices = self.replay_buffer.sample(BATCH_SIZE)
			td_errors = self.experience_replay(batch, weights)
			self.replay_buffer.update_priorities(tree_indices, td_errors)

			# Target network soft update
			self.update_target()

			if i % 100 == 0:
				print(f'Epoch {i}/{NUM_EPOCHS}  |  epsilon: {self.epsilon:.3f}  |  total return: {total_return:.1f}')
				self.checkpoint(checkpoint_env)

			# Decay exploration rate for next epoch
			self.epsilon = max(self.epsilon * EPSILON_DECAY, MIN_EPSILON)

		return total_return_per_epoch

	def choose_action(self, state, greedy=False):
		if not greedy and np.random.random() < self.epsilon:
			# Random action (exploration)
			return np.random.randint(NUM_ACTIONS)
		else:
			# Choose best action available in this state (exploitation)
			state = torch.tensor(state).float().unsqueeze(dim=0)
			with torch.inference_mode():
				action_values = self.policy(state)
			return action_values.squeeze().argmax().item()

	def experience_replay(self, batch, weights):
		"""
		Experience replay (learning) with a batch of transitions.
		No need to use torch's train(), eval(), or inference_mode() functions,
		as there are no dropout or batch norm layers.
		"""

		states = torch.tensor([s[0] for s in batch]).float()
		actions = torch.tensor([s[1] for s in batch])
		returns = torch.tensor([s[2] for s in batch]).float()
		next_states = torch.tensor([s[3] for s in batch]).float()
		terminal_mask = torch.tensor([s[4] for s in batch])

		# ------------------------------ Double DQN ------------------------------ #

		# Policy model's current approximation of Q(s,a) for `actions`
		policy_q_current = self.policy(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

		policy_q_next = self.policy(next_states).detach()  # Q(s_{t+1},a|θ_t)
		target_q_next = self.target(next_states).detach()  # Q(s_{t+1},a|θ'_t)
		target_q_next[terminal_mask] = 0

		best_actions = policy_q_next.argmax(dim=1).unsqueeze(dim=1)  # argmax_a(Q(s_{t+1},a|θ_t))

		# Double Q-learning update (see README):
		# Q(s_t,a_t|theta) = r_t + γ Q(s_{t+1},argmax_a(Q(s_{t+1},a|θ_t))|θ'_t)
		expected_q_current = returns + GAMMA * target_q_next.gather(dim=1, index=best_actions).squeeze()

		# Policy model gradient step
		loss = (weights * (policy_q_current - expected_q_current) ** 2).mean()
		self.optimiser.zero_grad()
		loss.backward()
		self.optimiser.step()

		# Temporal Difference error (used to update priority replay tree)
		td_errors = (policy_q_current - expected_q_current).abs().detach().cpu()

		return td_errors

	def update_target(self):
		"""
		Soft target model update: θ' = τ * θ + (1 - τ) * θ'
		"""

		for main_param, target_param in zip(self.policy.parameters(), self.target.parameters()):
			target_param.data.copy_(TAU * main_param.data + (1 - TAU) * target_param.data)

	def checkpoint(self, env):
		env.reset()
		state = env.get_state()
		t = total_vel = 0

		for t in range(1, MAX_EP_LENGTH + 1):
			action = self.choose_action(state, True)  # Be greedy when testing
			_, state, terminal = env.step(action)
			if terminal:
				break
			total_vel += env.car.vel

		laps = env.car.num_gates_crossed / len(env.reward_gates)
		mean_vel = total_vel / t
		torch.save(self.policy.state_dict(), f'./ddqn/model_{laps:.2f}_laps_{mean_vel:.1f}_mean_vel.pth')

	def load_model(self):
		# No need to load target model (only need policy model for testing)
		self.policy.load_state_dict(torch.load('./ddqn/model.pth'))
