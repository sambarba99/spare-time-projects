"""
Proximal Policy Optimisation (PPO) agent implementation

The steps here correspond to those in ppo_clip_pseudocode.png (see 'ALGORITHM STEP N' comments)

Author: Sam Barba
Created 11/07/2024
"""

import torch
from torch import nn
from torch.distributions import Categorical

from pytorch_ppo_asteroids.ppo.constants import *


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
		batch_returns = torch.tensor(self.returns).float()

		return batch_states, batch_state_values, batch_actions, batch_action_log_probs, batch_returns, self.terminals

	def clear(self):
		self.__init__()


class ActorCritic(nn.Module):
	def __init__(self):                                                                                 # ALGORITHM STEP 1
		super().__init__()

		# Policy (actor) function:
		# Q_pi(s,a) = expected return from starting in state 's', doing action 'a' and following policy 'pi'
		self.actor = nn.Sequential(
			nn.Linear(NUM_INPUTS, LAYER_SIZE),
			nn.LeakyReLU(),
			nn.Linear(LAYER_SIZE, LAYER_SIZE),
			nn.LeakyReLU(),
			nn.Linear(LAYER_SIZE, NUM_ACTIONS),
			nn.Softmax(dim=-1)
		)

		# Value (critic) function:
		# V_pi(s) = expected return from starting in state 's' and following policy 'pi'
		self.critic = nn.Sequential(
			nn.Linear(NUM_INPUTS, LAYER_SIZE),
			nn.LeakyReLU(),
			nn.Linear(LAYER_SIZE, LAYER_SIZE),
			nn.LeakyReLU(),
			nn.Linear(LAYER_SIZE, 1)
		)

	def forward(self):
		raise NotImplementedError

	def act(self, state, greedy):
		# Generate action distribution given state
		action_probs = self.actor(state)
		distribution = Categorical(action_probs)

		if greedy:
			action = action_probs.argmax()
		else:
			# Sample an action from the distribution
			action = distribution.sample()

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
	def __init__(self, path, mbsize, gsteps):
		# In each PPO update, gradient steps are performed on self.trainable_policy,
		# then its weights are copied to self.policy
		self.trainable_policy = ActorCritic()
		self.policy = ActorCritic()
		self.policy.load_state_dict(self.trainable_policy.state_dict())
		self.path = path
		self.mbsize = mbsize
		self.gsteps = gsteps

		self.buffer = RolloutBuffer()
		self.optimiser = torch.optim.Adam([
			{'params': self.trainable_policy.actor.parameters(), 'lr': ACTOR_LR},
			{'params': self.trainable_policy.critic.parameters(), 'lr': CRITIC_LR}
		])

	def do_training(self, train_env, checkpoint_env):
		timesteps_done = episode_num = percent_done = 0
		total_return_per_episode = []

		while timesteps_done < TOTAL_TRAIN_TIMESTEPS:                                                   # ALGORITHM STEP 2
			train_env.reset()
			state = train_env.get_state()
			t = total_episode_return = 0

			for t in range(1, MAX_EP_LENGTH + 1):
				# Calculate action and make a step in the env
				action = self.choose_action(state)
				return_, state, terminal = train_env.step(action)

				# For GAE computation later
				self.buffer.returns.append(return_)
				self.buffer.terminals.append(terminal)

				total_episode_return += return_

				if len(self.buffer) == BATCH_SIZE:
					# ------------------------------ PPO update ------------------------------ #

					batch_states, batch_state_values, batch_actions, batch_action_log_probs, \
						batch_returns, batch_terminals = self.buffer.rollout()                          # ALGORITHM STEP 3

					# Standardise for more stable training
					batch_returns = (batch_returns - batch_returns.mean()) / (batch_returns.std() + 1e-9)

					for i in range(0, BATCH_SIZE, self.mbsize):
						minibatch_states = batch_states[i:i + self.mbsize]
						minibatch_state_values = batch_state_values[i:i + self.mbsize]
						minibatch_actions = batch_actions[i:i + self.mbsize]
						minibatch_action_log_probs = batch_action_log_probs[i:i + self.mbsize]
						minibatch_returns = batch_returns[i:i + self.mbsize]
						minibatch_terminals = batch_terminals[i:i + self.mbsize]

						advantages, returns = self.compute_gae(                                         # ALGORITHM STEP 4/5
							minibatch_state_values, minibatch_returns, minibatch_terminals
						)

						for _ in range(self.gsteps):
							state_values, action_log_probs, action_dist_entropy = \
								self.trainable_policy.evaluate(minibatch_states, minibatch_actions)

							# Calculate ratio r(t) = pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t).
							# r(t) = (prob. of choosing action a_t in state s_t under the current policy)
							#      ÷ (prob. of choosing a_t | s_t under the prev. policy)
							ratios = torch.exp(action_log_probs - minibatch_action_log_probs)

							# Calculate surrogate losses (the minimum of these will be used in steps 6/7):
							# surr2 clips the surr1 ratios to ensure the gradient step isn't too big
							surr1 = ratios * advantages
							surr2 = torch.clamp(ratios, 1 - EPSILON, 1 + EPSILON) * advantages  # Clip

							# Calculate losses and do backpropagation                                     ALGORITHM STEP 6/7
							# Actor loss is negative because we want to maximise (gradient ascent)
							actor_loss = -torch.min(surr1, surr2)
							critic_loss = nn.MSELoss()(state_values, returns)

							# Critic loss x0.5 cancels the 2 in the MSE derivative.
							# An entropy value is used for regularisation, as it adds a penalty based on
							# the entropy of the policy distribution. By maximising entropy, the agent is
							# encouraged to explore different actions and avoid converging to local minima.
							loss = (actor_loss + 0.5 * critic_loss - ENTROPY_COEFF * action_dist_entropy).mean()
							self.optimiser.zero_grad()
							loss.backward()
							self.optimiser.step()

					self.policy.load_state_dict(self.trainable_policy.state_dict())
					self.checkpoint_model(checkpoint_env, percent_done)
					self.buffer.clear()

				if terminal: break

			episode_num += 1
			timesteps_done += t
			percent_done = 100 * timesteps_done / TOTAL_TRAIN_TIMESTEPS
			total_return_per_episode.append(total_episode_return)

			if episode_num % 10 == 0:
				print(f'Episode: {episode_num}  |  '
					f'timesteps: {t} ({percent_done:.1f}% done)  |  '
					f'total return: {total_episode_return:.2f}  |  '
					f'score: {train_env.spaceship.score}')

		return total_return_per_episode

	def choose_action(self, state, greedy=False):
		state = torch.tensor(state).float()
		with torch.inference_mode():
			action, action_log_prob, state_value = self.policy.act(state, greedy)

		# Store data for rollout (training)
		self.buffer.states.append(state)
		self.buffer.state_values.append(state_value)
		self.buffer.actions.append(action)
		self.buffer.action_log_probs.append(action_log_prob)

		return action.item()

	def compute_gae(self, batch_state_values, batch_returns, batch_terminals):
		"""
		Generalised Advantage Estimation: balances bias and variance in advantage estimation
		by combining multi-step returns using a discount factor gamma and a smoothing parameter lambda
		"""

		advantages, returns = [], []
		advantage = next_state_value = rt = 0

		for i in reversed(range(len(batch_state_values))):
			v = batch_state_values[i]
			r = batch_returns[i]
			terminal = batch_terminals[i]

			# Temporal Difference error: difference between the value of s_t and the actual reward + estimated value of s_(t+1)
			td_error = 0 if terminal else r + GAMMA * next_state_value - v
			advantage = td_error + GAMMA * LAMBDA * advantage
			next_state_value = v
			advantages.append(advantage)

			# Discounted return
			rt = 0 if terminal else r + GAMMA * rt
			returns.append(rt)

		advantages.reverse()
		returns.reverse()
		advantages = torch.tensor(advantages).float()
		returns = torch.tensor(returns).float()

		# Standardise for more stable training
		advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

		return advantages, returns

	def checkpoint_model(self, env, percent_done, num_runs=10):
		total_score = 0

		for seed in range(1, num_runs + 1):
			env.rand.seed(seed)  # Test with the same set of random seeds each time
			env.reset()
			state = env.get_state()

			for _ in range(MAX_EP_LENGTH):
				action = self.choose_action(state, True)
				_, state, terminal = env.step(action)
				if terminal:
					break

			total_score += env.spaceship.score

		mean_score = total_score / num_runs
		path = f'./ppo/models/model_{mean_score:.1f}_score_{percent_done:.1f}_percent_done_{self.path}.pth'
		torch.save(self.policy.state_dict(), path)

	def load_model(self):
		self.policy.load_state_dict(torch.load('./ppo/models/model.pth'))
