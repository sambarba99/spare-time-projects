# Agent class for reinforcement learning algorithms demo
# Author: Sam Barba
# Created 25/02/2022

import numpy as np

class Agent:
	def __init__(self, gamma=0.9, alpha=0.2, epsilon=0.1):
		self.gamma = gamma  # Discount factor
		self.alpha = alpha  # Learning rate (for SARSA, Q-learning)
		self.epsilon = epsilon  # For epsilon-greedy policy
		self.env = None
		self.q_table = None
		self.__reset_q_table()

	def __choose_action_epsilon_greedy(self, state):
		if np.random.random() < self.epsilon:
			# Random action (exploration)
			return np.random.choice(self.env.actions)
		else:
			# Choose best action available in current state (exploitation), breaking ties randomly
			return np.random.choice(np.where(self.q_table[state] == max(self.q_table[state]))[0])

	# Get discounted reward of a new state
	def __discounted_reward(self, new_state, v):
		return self.gamma * v[new_state]

	def policy_iteration(self, theta=10 ** -9):
		self.__reset_q_table()

		# Dicts and list initialised only with start state; new states are added as they're discovered
		# Initial arbitrary policy (random)
		policy = {self.env.start: np.random.choice(self.env.actions)}
		# Initial arbitrary value function (v(s) = 0 for each state)
		v = {self.env.start: 0}
		non_terminal_states = [self.env.start]

		policy_stable = False

		while not policy_stable:
			policy_stable = True

			# Policy evaluation (compute each v(s) under current policy)
			convergence = 1
			while convergence > theta:
				convergence = 0

				for state in non_terminal_states:
					current_v = v[state]
					new_state, reward, terminal = self.env.step(state, policy[state])

					# Update with any new states
					if new_state not in v:
						v[new_state] = 0
					if not terminal and new_state not in policy:
						policy[new_state] = np.random.choice(self.env.actions)
					if not terminal and new_state not in non_terminal_states:
						non_terminal_states.append(new_state)

					v[state] = reward + self.__discounted_reward(new_state, v)
					convergence = max(convergence, abs(current_v - v[state]))

			# Policy improvement (with updated state values, improve policy if needed)
			for state in non_terminal_states:
				old_action = policy[state]

				# Determine best action for this state, given (updated) v
				best_action, best_action_val = None, -10 ** 9
				for a in self.env.actions:
					new_state, reward, terminal = self.env.step(state, a)

					# Update with any new states
					if new_state not in v:
						v[new_state] = 0
					if not terminal and new_state not in policy:
						policy[new_state] = np.random.choice(self.env.actions)
					if not terminal and new_state not in non_terminal_states:
						non_terminal_states.append(new_state)

					action_val = reward + self.__discounted_reward(new_state, v)
					if action_val > best_action_val:
						best_action_val, best_action = action_val, a

				policy[state] = best_action

				if old_action != best_action:
					policy_stable = False

		# Update Q-table, as it is used in drawing grid at the end
		self.q_table = {state: np.zeros(4) for state in policy}  # 4 possible actions
		for state, action in policy.items():
			self.q_table[state][action] = v[state]

		self.__render_q_table(print_table=False)

	def value_iteration(self, theta=10 ** -9):
		self.__reset_q_table()

		# Dicts and list initialised only with start state; new states are added as they're discovered
		# Initial arbitrary value function (v(s) = 0 for each state)
		v = {self.env.start: 0}
		non_terminal_states = [self.env.start]
		policy = {}  # To keep track of best policy

		convergence = 1
		while convergence > theta:
			convergence = 0

			for state in non_terminal_states:
				current_v = v[state]
				best_v = -10 ** 9

				for a in self.env.actions:
					new_state, reward, terminal = self.env.step(state, a)

					# Update with any new states
					if new_state not in v:
						v[new_state] = 0
					if not terminal and new_state not in non_terminal_states:
						non_terminal_states.append(new_state)

					new_v = reward + self.__discounted_reward(new_state, v)
					if new_v > best_v:
						best_v = new_v
						policy[state] = a  # Update best policy for this state if needed

				v[state] = best_v
				convergence = max(convergence, abs(current_v - best_v))

		# Update Q-table, as it is used in drawing grid at the end
		self.q_table = {state: np.zeros(4) for state in policy}
		for state, action in policy.items():
			self.q_table[state][action] = v[state]

		self.__render_q_table(print_table=False)

	def on_policy_first_visit_mc_control(self, num_training_episodes=3000):
		self.__reset_q_table()

		rewards = {}

		for _ in range(num_training_episodes):
			# Generate trajectory by following epsilon-greedy policy
			state = self.env.start
			terminal = False
			trajectory = []

			while not terminal:
				if state not in self.q_table:  # Update with any new states
					self.q_table[state] = np.zeros(4)

				action = self.__choose_action_epsilon_greedy(state)
				new_state, reward, terminal = self.env.step(state, action)
				trajectory.append((state, action, reward))
				state = new_state

			trajectory.reverse()
			g = 0
			for idx, step in enumerate(trajectory):
				state, action, reward = step
				g = self.gamma * g + reward

				# First visit check
				check_list = [(state, action) for state, action, _ in trajectory[idx + 1:]]
				if (state, action) not in check_list:
					# Update with any new states
					if state not in rewards:
						rewards[state] = {a: [] for a in self.env.actions}
					if state not in self.q_table:
						self.q_table[state] = np.zeros(4)

					rewards[state][action].append(g)
					self.q_table[state][action] = np.mean(rewards[state][action])

		self.__render_q_table(print_table=True)

	def sarsa(self, num_training_episodes=5000):
		self.__reset_q_table()

		for _ in range(num_training_episodes):
			state = self.env.start
			terminal = False
			if state not in self.q_table:
				self.q_table[state] = np.zeros(4)

			# Choose action from current state, derived from Q-table (in this case epsilon-greedy)
			action = self.__choose_action_epsilon_greedy(state)

			while not terminal:
				new_state, reward, terminal = self.env.step(state, action)

				if new_state not in self.q_table:  # Update with any new states
					self.q_table[new_state] = np.zeros(4)

				new_action = self.__choose_action_epsilon_greedy(new_state)

				# Calculate new Q-value estimate of current state-action pair
				current_q = self.q_table[state][action]
				new_q = current_q + self.alpha * (reward + self.gamma * self.q_table[new_state][new_action] - current_q)
				self.q_table[state][action] = new_q

				state, action = new_state, new_action

		self.__render_q_table(print_table=True)

	def q_learning(self, num_training_episodes=5000):
		self.__reset_q_table()

		for _ in range(num_training_episodes):
			state = self.env.start
			terminal = False

			while not terminal:
				if state not in self.q_table:  # Update with any new states
					self.q_table[state] = np.zeros(4)

				# Choose action from current state, derived from Q-table (in this case epsilon-greedy)
				action = self.__choose_action_epsilon_greedy(state)

				# Observe new state and reward after taking chosen action
				new_state, reward, terminal = self.env.step(state, action)

				if new_state not in self.q_table:  # Update with any new states
					self.q_table[new_state] = np.zeros(4)

				# Calculate new Q-value estimate of current state-action pair
				current_q = self.q_table[state][action]
				new_q = current_q + self.alpha * (reward + self.gamma * max(self.q_table[new_state]) - current_q)
				self.q_table[state][action] = new_q

				# Move to new state
				state = new_state

		self.__render_q_table(print_table=True)

	def __render_q_table(self, print_table):
		if print_table:
			print("\n# --- Final Q-table ('state: NESW values') --- #\n")
			for state, action_vals in sorted(self.q_table.items()):
				print(f"{state}: ", np.round(action_vals, 3))

		# Draw environment grid depicting optimal policy and v(s) for each state
		self.env.render(self.q_table)

	def __reset_q_table(self):
		# Q-table is a dict containing numpy lists:
		# 	- All action values of a state are accessed via q_table[state]
		# 	- A specific value of a state-action is accessed via q_table[state][action]
		# Initialised empty; new states are added as they're discovered
		self.q_table = {}
