"""
Proximal Policy Optimisation (PPO) agent implementation

The steps here correspond to those in ppo_clip_pseudocode.png (see 'ALGORITHM STEP N' comments)

Author: Sam Barba
Created 11/07/2024
"""

from datetime import datetime

import torch
from torch import nn
from torch.distributions import Categorical

from pytorch_proximal_policy_optimisation_asteroids.ppo.constants import *


class RolloutBuffer:
	def __init__(self):
		self.states = []
		self.state_values = []
		self.actions = []
		self.action_log_probs = []
		self.rewards = []
		self.terminals = []

	def __len__(self):
		return len(self.states)

	def rollout(self):
		batch_states = torch.stack(self.states).detach()
		batch_state_values = torch.stack(self.state_values).detach()
		batch_actions = torch.stack(self.actions).detach()
		batch_action_log_probs = torch.stack(self.action_log_probs).detach()

		return batch_states, batch_state_values, batch_actions, batch_action_log_probs, self.rewards, self.terminals

	def clear(self):
		self.states.clear()
		self.state_values.clear()
		self.actions.clear()
		self.action_log_probs.clear()
		self.rewards.clear()
		self.terminals.clear()


class ActorCritic(nn.Module):
	def __init__(self):                                                                             # ALGORITHM STEP 1
		super().__init__()

		# Policy (actor) function:
		# π(a|s) = policy distribution over possible actions 'a' given state 's'
		self.actor = nn.Sequential(
			nn.Linear(NUM_INPUTS, LAYER_SIZE_ACTOR),
			nn.LeakyReLU(),
			nn.Linear(LAYER_SIZE_ACTOR, LAYER_SIZE_ACTOR),
			nn.LeakyReLU(),
			nn.Linear(LAYER_SIZE_ACTOR, NUM_ACTIONS)
		)

		# Value (critic) function:
		# V_π(s) = expected discounted return from starting in state 's' and following policy π
		self.critic = nn.Sequential(
			nn.Linear(NUM_INPUTS, LAYER_SIZE_CRITIC),
			nn.LeakyReLU(),
			nn.Linear(LAYER_SIZE_CRITIC, LAYER_SIZE_CRITIC),
			nn.LeakyReLU(),
			nn.Linear(LAYER_SIZE_CRITIC, 1)
		)

	def forward(self, states):
		action_logits = self.actor(states).squeeze()
		state_values = self.critic(states).squeeze()
		return action_logits, state_values

	def act(self, state, greedy):
		action_logits, state_value = self.forward(state)

		distribution = Categorical(logits=action_logits)
		if greedy:
			# Choose best action (exploitation)
			action = action_logits.argmax()
		else:
			# Choose random action (exploration)
			action = distribution.sample()
		action_log_prob = distribution.log_prob(action)

		return action, action_log_prob, state_value

	def evaluate(self, states, actions):
		action_logits, state_values = self.forward(states)

		distribution = Categorical(logits=action_logits)
		action_log_probs = distribution.log_prob(actions)
		dist_entropy = distribution.entropy()

		return state_values, action_log_probs, dist_entropy


class PPOAgent:
	def __init__(self, training_mode):
		self.training_mode = training_mode
		self.policy = ActorCritic().cpu()

		if self.training_mode:
			self.buffer = RolloutBuffer()
			self.optimiser = torch.optim.Adam([
				{'params': self.policy.actor.parameters(), 'lr': ACTOR_LR},
				{'params': self.policy.critic.parameters(), 'lr': CRITIC_LR}
			])

	def train(self, train_env, checkpoint_env):
		episode_num = timesteps_done = approx_kl_div = 0
		total_reward_per_episode = []

		total_anneal_steps = ENTROPY_ANNEAL_STEPS // BATCH_SIZE
		entropy_coefficient_schedule = torch.linspace(START_ENTROPY_COEFF, END_ENTROPY_COEFF, total_anneal_steps)
		entropy_coeff_idx = 0

		while timesteps_done < TOTAL_TRAIN_TIMESTEPS:                                               # ALGORITHM STEP 2
			train_env.reset()
			state = train_env.get_state()
			t = total_episode_reward = 0

			for t in range(1, MAX_EP_LENGTH + 1):
				action = self.choose_action(state)
				reward, state, terminal = train_env.step(action)

				# For advantage/return computation later
				self.buffer.rewards.append(reward)
				self.buffer.terminals.append(terminal)

				timesteps_done += 1
				total_episode_reward += reward

				if len(self.buffer) == BATCH_SIZE:
					# ------------------------------ PPO update ------------------------------ #

					batch_states, batch_state_values, batch_actions, batch_action_log_probs, \
						batch_rewards, batch_terminals = self.buffer.rollout()                      # ALGORITHM STEP 3

					advantages, returns = self.compute_gae_and_returns(                             # ALGORITHM STEP 4/5
						batch_state_values, batch_rewards, batch_terminals, latest_state=state
					)

					# Standardise for more stable training
					advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

					# Anneal entropy coefficient from START_ENTROPY_COEFF to END_ENTROPY_COEFF
					entropy_coeff = entropy_coefficient_schedule[entropy_coeff_idx]
					if entropy_coeff_idx < len(entropy_coefficient_schedule) - 1:
						entropy_coeff_idx += 1

					for _ in range(NUM_EPOCHS):
						indices = torch.randperm(BATCH_SIZE)

						for i in range(0, BATCH_SIZE, MINIBATCH_SIZE):
							mb_idx = indices[i:i + MINIBATCH_SIZE]
							mb_states = batch_states[mb_idx]
							mb_actions = batch_actions[mb_idx]
							mb_action_log_probs = batch_action_log_probs[mb_idx]
							mb_advantages = advantages[mb_idx]
							mb_returns = returns[mb_idx]

							state_values, action_log_probs, action_dist_entropy = \
								self.policy.evaluate(mb_states, mb_actions)

							# Calculate ratio r(t) = pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t)
							# pi_theta_old is the policy before this update loop, captured in batch_action_log_probs
							# i.e. r(t) = (prob. of choosing action a_t in state s_t under the current policy)
							#           ÷ (prob. of choosing a_t | s_t under the old policy)
							log_ratios = action_log_probs - mb_action_log_probs
							ratios = torch.exp(log_ratios)

							approx_kl_div = ((ratios - 1) - log_ratios).mean().item()
							if approx_kl_div > KL_DIVERGENCE_THRESHOLD:
								break

							# Calculate surrogate losses:
							# surr2 clips the surr1 ratios to ensure the gradient step isn't too big
							surr1 = ratios * mb_advantages
							surr2 = ratios.clamp(1 - EPSILON, 1 + EPSILON) * mb_advantages
							actor_loss = -torch.min(surr1, surr2).mean()                            # ALGORITHM STEP 6

							value_pred_clipped = batch_state_values[mb_idx] + (
								state_values - batch_state_values[mb_idx]
							).clamp(-EPSILON, EPSILON)
							value_loss1 = (state_values - mb_returns).pow(2)
							value_loss2 = (value_pred_clipped - mb_returns).pow(2)
							critic_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()          # ALGORITHM STEP 7

							# An entropy value is used for regularisation, as it adds a penalty based on the entropy of
							# the actor distribution. Maximising entropy encourages the agent to explore different
							# actions and avoid converging to local minima.
							entropy = action_dist_entropy.mean()
							loss = actor_loss + VALUE_FUNC_COEFF * critic_loss - entropy_coeff * entropy
							self.optimiser.zero_grad()
							loss.backward()
							torch.nn.utils.clip_grad_norm_(self.policy.parameters(), GRAD_NORM_THRESHOLD)
							self.optimiser.step()

						if approx_kl_div > KL_DIVERGENCE_THRESHOLD:
							break

					self.buffer.clear()
					self.checkpoint(checkpoint_env)

				if terminal or timesteps_done == TOTAL_TRAIN_TIMESTEPS:
					break

			episode_num += 1
			total_reward_per_episode.append(total_episode_reward)

			print(f'Episode: {episode_num}  |  '
				f'timesteps: {t} ({(timesteps_done / TOTAL_TRAIN_TIMESTEPS):.1%} done)  |  '
				f'total reward: {total_episode_reward:.1f}  |  '
				f'score: {train_env.spaceship.score}')

		return total_reward_per_episode

	def choose_action(self, state, greedy=False):
		state = torch.as_tensor(state, dtype=torch.float32)
		with torch.inference_mode():
			action, action_log_prob, state_value = self.policy.act(state, greedy or not self.training_mode)

		if self.training_mode and not greedy:
			# Store data for rollout
			self.buffer.states.append(state)
			self.buffer.state_values.append(state_value.detach())
			self.buffer.actions.append(action.detach())
			self.buffer.action_log_probs.append(action_log_prob.detach())

		return action.item()

	def compute_gae_and_returns(self, batch_state_values, batch_rewards, batch_terminals, latest_state):
		"""
		Generalised Advantage Estimation: balances bias and variance in advantage estimation
		by combining multi-step returns using a discount factor gamma and a smoothing parameter lambda
		"""

		# Bootstrap value for last state
		if batch_terminals[-1]:
			next_state_value = 0
		else:
			with torch.inference_mode():
				next_state_value = self.policy.critic(torch.as_tensor(latest_state, dtype=torch.float32)).squeeze()
		gae = 0
		advantages = torch.zeros_like(batch_state_values)

		for t in reversed(range(BATCH_SIZE)):
			next_value = next_state_value if t == BATCH_SIZE - 1 else batch_state_values[t + 1]
			non_terminal = 1.0 - batch_terminals[t]

			# Temporal Difference error:
			# difference between the value of s_t and the actual reward + estimated value of s_(t+1)
			td_error = batch_rewards[t] + GAMMA * next_value * non_terminal - batch_state_values[t]

			gae = td_error + GAMMA * LAMBDA * non_terminal * gae
			advantages[t] = gae

		returns = advantages + batch_state_values

		return advantages, returns

	def checkpoint(self, env, num_runs=20):
		total_score = 0

		for seed in range(1, num_runs + 1):
			env.reset(seed)  # Test with the same set of random seeds each time
			state = env.get_state()

			for _ in range(MAX_EP_LENGTH):
				action = self.choose_action(state, True)  # Be greedy when testing
				_, state, terminal = env.step(action)
				if terminal:
					break

			total_score += env.spaceship.score

		mean_score = total_score // num_runs
		ts = datetime.now().strftime('%Y%m%d-%H%M%S')
		torch.save(self.policy.state_dict(), f'./ppo/model_{mean_score}_mean_score_{ts}.pth')

	def load_model(self, path='./ppo/model.pth'):
		self.policy.load_state_dict(torch.load(path))
