"""
Proximal Policy Optimisation (PPO) agent implementation

The steps here correspond to those in ppo_clip_pseudocode.png (see 'ALGORITHM STEP N' comments)

Author: Sam Barba
Created 16/02/2023
"""

import os

import torch
from torch import nn
from torch.distributions import Categorical

from ppo.constants import *


class RolloutBuffer:
	def __init__(self):
		self.states = []
		self.state_values = []
		self.actions = []
		self.action_log_probs = []
		self.returns = []
		self.terminals = []

	def __len__(self):
		return len(self.states)

	def rollout(self):
		batch_states = torch.stack(self.states).detach()
		batch_state_values = torch.stack(self.state_values).detach()
		batch_actions = torch.stack(self.actions).detach()
		batch_action_log_probs = torch.stack(self.action_log_probs).detach()

		return batch_states, batch_state_values, batch_actions, batch_action_log_probs

	def clear(self):
		self.__init__()


class ActorCritic(nn.Module):
	def __init__(self, training_mode):                                                                  # ALGORITHM STEP 1
		super().__init__()
		self.training_mode = training_mode

		# Policy (actor) function:
		# Q_pi(s,a) = expected return from starting in state 's', doing action 'a' and following policy 'pi'
		self.actor = nn.Sequential(
			nn.Linear(N_INPUTS, LAYER_SIZE),
			nn.Tanh(),
			nn.Linear(LAYER_SIZE, LAYER_SIZE),
			nn.Tanh(),
			nn.Linear(LAYER_SIZE, N_ACTIONS),
			nn.Softmax(dim=-1)
		)

		# Value (critic) function:
		# V_pi(s) = expected return from starting in state 's' and following policy 'pi'
		self.critic = nn.Sequential(
			nn.Linear(N_INPUTS, LAYER_SIZE),
			nn.Tanh(),
			nn.Linear(LAYER_SIZE, LAYER_SIZE),
			nn.Tanh(),
			nn.Linear(LAYER_SIZE, 1)
		)

	def forward(self):
		raise NotImplementedError

	def act(self, state):
		# Generate action distribution given state
		action_probs = self.actor(state)
		distribution = Categorical(action_probs)

		if self.training_mode:
			# Sample an action from the distribution
			action = distribution.sample()
		else:
			# For testing/visualisation, be greedy
			action = action_probs.argmax()

		# Get the log probability of this action in the distribution
		action_log_prob = distribution.log_prob(action)

		# Get critic's value of state (just for storing in buffer.state_values for training)
		state_value = self.critic(state).squeeze()

		return action, action_log_prob, state_value

	def evaluate(self, states, actions):
		# Get critic's value of states
		state_values = self.critic(states).squeeze()

		# Generate action distribution given states
		action_probs = self.actor(states)
		distribution = Categorical(action_probs)

		# Get the log probability of 'actions' in this distribution
		action_log_probs = distribution.log_prob(actions)

		# Get distribution's entropy
		dist_entropy = distribution.entropy()

		return state_values, action_log_probs, dist_entropy


class PPOAgent:
	def __init__(self, *, training_mode):
		self.training_mode = training_mode

		# In each PPO update, gradient steps are performed on self.trainable_policy,
		# then its weights are copied to self.policy
		self.trainable_policy = ActorCritic(self.training_mode)
		self.policy = ActorCritic(self.training_mode)
		self.policy.load_state_dict(self.trainable_policy.state_dict())

		self.buffer = RolloutBuffer()
		self.optimiser = torch.optim.Adam([
			{'params': self.trainable_policy.actor.parameters(), 'lr': ACTOR_LR},
			{'params': self.trainable_policy.critic.parameters(), 'lr': CRITIC_LR}
		])

	def do_training(self, env):
		timesteps_done = episode_num = 0
		total_return_per_episode = []

		while timesteps_done < TOTAL_TRAIN_TIMESTEPS:                                                   # ALGORITHM STEP 2
			env.reset()
			state = env.get_state()
			t = total_return = total_vel = 0

			for t in range(1, MAX_EP_LENGTH + 1):
				# Calculate action and make a step in the env
				action = self.choose_action(state)
				return_, state, terminal = env.step(action)

				# For returns-to-go computation later
				self.buffer.returns.append(return_)
				self.buffer.terminals.append(terminal)

				total_return += return_
				total_vel += env.car.vel
				timesteps_done += 1

				if len(self.buffer) == BATCH_SIZE:
					# ------------------------------ PPO update ------------------------------ #

					batch_states, batch_state_values, batch_actions, batch_action_log_probs = \
						self.buffer.rollout()                                                           # ALGORITHM STEP 3

					returns_to_go = self.compute_returns_to_go()                                        # ALGORITHM STEP 4

					# Compute expected advantage: A(s,a) = Q(s,a) - V(s)                                  ALGORITHM STEP 5
					adv = returns_to_go - batch_state_values

					for _ in range(GRAD_STEPS_PER_EPOCH):
						state_values, action_log_probs, action_dist_entropy = \
							self.trainable_policy.evaluate(batch_states, batch_actions)

						# Calculate ratio r(t) = pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t).
						# r(t) = (prob. of choosing action a_t in state s_t under the current policy)
						#      ÷ (prob. of choosing a_t | s_t under the prev. policy)
						ratios = torch.exp(action_log_probs - batch_action_log_probs)

						# Calculate surrogate losses (the minimum of these will be used in steps 6/7):
						# surr2 clips the surr1 ratios to ensure the gradient step isn't too big
						surr1 = ratios * adv
						surr2 = torch.clamp(ratios, 1 - EPSILON, 1 + EPSILON) * adv  # Clip

						# Calculate losses and do backpropagation                                         ALGORITHM STEP 6/7
						# Actor loss is negative because we want to maximise (gradient ascent)
						actor_loss = -torch.min(surr1, surr2)
						critic_loss = nn.MSELoss()(state_values, returns_to_go)

						# Critic loss x0.5 cancels the 2 in the MSE derivative.
						# An entropy value is used for regularisation, as it adds a penalty based on
						# the entropy of the policy distribution. By maximising entropy, the agent is
						# encouraged to explore different actions and avoid converging to local minima.
						loss = (actor_loss + 0.5 * critic_loss - ENTROPY_COEFF * action_dist_entropy).mean()
						self.optimiser.zero_grad()
						loss.backward()
						self.optimiser.step()

					self.policy.load_state_dict(self.trainable_policy.state_dict())
					self.buffer.clear()

				if terminal: break

			# Checkpoint and save model in case a PPO update just happened
			episode_num += 1
			percent_done = 100 * timesteps_done / TOTAL_TRAIN_TIMESTEPS
			total_return_per_episode.append(total_return)

			laps = env.car.n_gates_crossed / len(env.reward_gates)
			mean_vel = total_vel / t
			model_path = f'./ppo/model_{laps:.2f}_laps_{mean_vel:.1f}_mean_vel.pth'
			if not os.path.exists(model_path):
				self.save_model(model_path)

			if episode_num % 10 == 0:
				print(f'Episode: {episode_num}  |  '
					f'timesteps: {t} ({percent_done:.1f}% done)  |  '
					f'total return: {total_return:.1f}  |  '
					f'laps: {laps:.2f}  |  '
					f'mean vel: {mean_vel:.1f}')

		return total_return_per_episode

	def choose_action(self, state):
		state = torch.tensor(state).float()
		with torch.inference_mode():
			action, action_log_prob, state_value = self.policy.act(state)

		if self.training_mode:
			# Store data for rollout
			self.buffer.states.append(state)
			self.buffer.state_values.append(state_value)
			self.buffer.actions.append(action)
			self.buffer.action_log_probs.append(action_log_prob)

		return action.item()

	def compute_returns_to_go(self):
		"""Compute 'returns-to-go' (estimated future returns from a given start state)"""

		discounted_return = 0
		returns_to_go = []

		# Iterate through all returns backwards, to correctly apply discount factor GAMMA
		for return_, terminal in reversed(list(zip(self.buffer.returns, self.buffer.terminals))):
			discounted_return = 0 if terminal else return_ + GAMMA * discounted_return
			returns_to_go.insert(0, discounted_return)

		# Standardise for more stable training
		returns_to_go = torch.tensor(returns_to_go).float()
		returns_to_go = (returns_to_go - returns_to_go.mean()) / (returns_to_go.std() + 1e-7)

		return returns_to_go

	def save_model(self, path):
		torch.save(self.policy.state_dict(), path)

	def load_model(self):
		self.policy.load_state_dict(torch.load('./ppo/model.pth'))
