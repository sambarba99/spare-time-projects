"""
Dueling Double Deep Q Network (Dueling DDQN) agent implementation

Author: Sam Barba
Created 05/06/2026
"""

import numpy as np
import torch
from torch import nn

from pytorch_dueling_ddqn_cartpole_swingup.dueling_ddqn.constants import *
from pytorch_dueling_ddqn_cartpole_swingup.dueling_ddqn.prioritised_replay_buffer import PrioritisedReplayBuffer


class DuelingQNetwork(nn.Module):
	def __init__(self):
		super().__init__()

		# Shared feature extractor
		self.feature_extractor = nn.Sequential(
			nn.Linear(NUM_INPUTS, LAYER_SIZE),
			nn.LeakyReLU(),
			nn.Linear(LAYER_SIZE, LAYER_SIZE),
			nn.LeakyReLU()
		)

		# Value function:
		# V(s) = expected discounted return from starting in state 's' and following current policy
		self.value_func = nn.Linear(LAYER_SIZE, 1)

		# Advantage function:
		# A(s,a) = how much better action 'a' is in state 's' compared to the average action
		self.advantage_func = nn.Linear(LAYER_SIZE, NUM_ACTIONS)

	def forward(self, states):
		features = self.feature_extractor(states)
		state_values = self.value_func(features)
		advantages = self.advantage_func(features)
		q_values = state_values + (
			advantages - advantages.mean(dim=1, keepdim=True)  # Helps stabilisation
		)
		return q_values


class DuelingDDQNAgent:
	def __init__(self, training_mode):
		self.epsilon = 1
		self.policy = DuelingQNetwork().cpu()  # Online model, used for action selection
		if training_mode:
			self.target = DuelingQNetwork().cpu()  # Offline model, used for action evaluation
			self.target.load_state_dict(self.policy.state_dict())
			self.replay_buffer = PrioritisedReplayBuffer()
			self.optimiser = torch.optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
			self.best_cos_angle = -np.inf

	def train(self, train_env, checkpoint_env):
		# Pretraining step: populate agent's memory with transitions
		# (choosing actions randomly, as self.epsilon = 1 initially)

		print('\nPretraining (populating agent memory)...')

		state = train_env.get_state()

		for _ in range(WARMUP_SAMPLES):
			# Choose action given current state
			action = self.choose_action(state)
			reward, next_state, terminal = train_env.step(action)

			# Store transition for experience replay
			self.replay_buffer.store_transition(state, action, reward, next_state, terminal)

			if terminal:
				train_env.reset()
				state = train_env.get_state()
			else:
				# Update for next loop
				state = next_state

		print('\nTraining...\n')

		episode_num = timesteps_done = 0
		total_reward_per_episode = []

		epsilon_decay_rate = np.log(1 / MIN_EPSILON) / TOTAL_TRAIN_TIMESTEPS

		while timesteps_done < TOTAL_TRAIN_TIMESTEPS:
			train_env.reset()
			state = train_env.get_state()
			t = total_episode_reward = 0

			for t in range(1, MAX_EP_LENGTH + 1):
				action = self.choose_action(state)
				reward, next_state, terminal = train_env.step(action)
				self.replay_buffer.store_transition(state, action, reward, next_state, terminal)

				timesteps_done += 1
				total_episode_reward += reward

				if timesteps_done % TRAIN_FREQ == 0:
					# Do experience replay (learning) with a batch of data
					batch, weights, tree_indices = self.replay_buffer.sample(BATCH_SIZE)
					td_errors = self.experience_replay(batch, weights)
					self.replay_buffer.update_priorities(tree_indices, td_errors)

					# Anneal PER beta towards 1
					self.replay_buffer.beta = min(
						PER_BETA + (timesteps_done / TOTAL_TRAIN_TIMESTEPS) * (1 - PER_BETA),
						1
					)

					# Target network soft update
					self.update_target()

				if timesteps_done % CHECKPOINT_FREQ == 0:
					self.checkpoint(checkpoint_env)

				if terminal or timesteps_done == TOTAL_TRAIN_TIMESTEPS:
					break

				state = next_state

			episode_num += 1
			total_reward_per_episode.append(total_episode_reward)

			print(f'Episode {episode_num}  |  '
				f'timesteps: {t} ({(timesteps_done / TOTAL_TRAIN_TIMESTEPS):.1%} done)  |  '
				f'epsilon: {self.epsilon:.4f}  |  '
				f'total reward: {total_episode_reward:.1f}  |  '
				f'best model: {self.best_cos_angle:.4f} cos(angle)')

			# Decay exploration rate for next episode
			self.epsilon = max(np.exp(-epsilon_decay_rate * timesteps_done), MIN_EPSILON)

		return total_reward_per_episode

	def choose_action(self, state, greedy=False):
		if not greedy and np.random.random() < self.epsilon:
			# Choose random action (exploration)
			return np.random.randint(NUM_ACTIONS)
		else:
			# Choose best action (exploitation)
			state = torch.tensor(state).float().unsqueeze(dim=0)
			with torch.inference_mode():
				action_values = self.policy(state)
			return action_values.squeeze().argmax().item()

	def experience_replay(self, batch, weights):
		"""Experience replay (learning) with a batch of transitions"""

		states, actions, rewards, next_states, terminals = zip(*batch)

		states = torch.from_numpy(np.array(states, dtype=np.float32))
		actions = torch.from_numpy(np.array(actions, dtype=np.int64))
		rewards = torch.from_numpy(np.array(rewards, dtype=np.float32))
		next_states = torch.from_numpy(np.array(next_states, dtype=np.float32))
		terminal_mask = torch.from_numpy(np.array(terminals, dtype=np.bool_))

		# ------------------------------ Double DQN ------------------------------ #

		# Policy model's current approximation of Q(s,a) for `actions`
		policy_q_current = self.policy(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

		with torch.inference_mode():
			policy_q_next = self.policy(next_states)  # Q(s_{t+1},a|θ_t)
			target_q_next = self.target(next_states)  # Q(s_{t+1},a|θ'_t)

			best_actions = policy_q_next.argmax(dim=1, keepdim=True)  # argmax_a(Q(s_{t+1},a|θ_t))
			target_values = target_q_next.gather(dim=1, index=best_actions).squeeze()
			target_values[terminal_mask] = 0

			# Double Q-learning update (see README):
			# Q(s_t,a_t|theta) = r_t + γ Q(s_{t+1},argmax_a(Q(s_{t+1},a|θ_t))|θ'_t)
			expected_q_current = rewards + GAMMA * target_values

		# Policy model gradient step
		loss = (weights * (policy_q_current - expected_q_current) ** 2).mean()
		self.optimiser.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.policy.parameters(), GRAD_NORM_THRESHOLD)
		self.optimiser.step()

		# Temporal Difference error (used to update priority replay tree)
		td_errors = (policy_q_current - expected_q_current).abs().detach()

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
		t = total_reward = total_cos_angle = 0

		for t in range(1, MAX_EP_LENGTH + 1):
			action = self.choose_action(state, True)  # Be greedy when testing
			reward, state, terminal = env.step(action)
			total_reward += reward
			total_cos_angle += state[-2]
			if terminal:
				break

		mean_cos_angle = total_cos_angle / t

		if mean_cos_angle > self.best_cos_angle:
			self.best_cos_angle = mean_cos_angle
			torch.save(
				self.policy.state_dict(),
				f'./dueling_ddqn/model_{mean_cos_angle:.4f}_cos_angle_{total_reward:.1f}_reward_{t}_timesteps.pth'
			)

	def load_model(self):
		# No need to load target model (only need policy model for testing)
		self.policy.load_state_dict(torch.load('./dueling_ddqn/model.pth'))
