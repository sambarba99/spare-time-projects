## Dueling Double Deep Q Network (Dueling DDQN) for the Cart-Pole Swing-Up Problem

<p align="center">
	<img src="agent_gameplay.webp"/>
</p>

In regular Q-learning (see [this other project](../reinforcement_learning_basics)), the Q value update is:

$$Q(s_t,a_t|\theta_t)=r_t+\gamma \underset{a \in A}{\mathop{\mathrm{max}}}\bigg(Q(s_{t+1},a|\theta_t)\bigg)$$

Where:
- $Q(s_t,a_t|\theta_t)$ = estimated value at time $t$ of taking action $a$ in state $s$ given model parameters $\theta$
- $A$ = action space
- $s_t,a_t,r_t,s_{t+1}$ = state, action, reward, new state
- $\gamma$ = return discount factor (0-1), which controls how much future rewards matter compared to immediate rewards.

This method is prone to maximisation bias, as the same Q-function is used both to select and evaluate the next action when computing the target value. This can lead to a cumulative overestimation error. Double Q-learning mitigates this by decoupling action selection from action evaluation, using two separate models: an online policy model for selection (with parameters $\theta$), and an offline target model for evaluation (with parameters $\theta'$).

Hence, in Double Q-learning, the Q value update is:

$$Q(s_t,a_t|\theta_t)=r_t+\gamma Q\bigg(s_{t+1},\underset{a \in A}{\mathop{\mathrm{argmax}}}\bigg(Q(s_{t+1},a|\theta_t)\bigg)|\theta'_t\bigg)$$

The target model isn't directly optimised via gradient descent, but its parameters are gradually synchronised with those of the policy model after each training step, using a small value $\tau$:

$$\theta'=\tau\theta+(1-\tau)\theta'$$

Additionally, this DDQN implementation uses the _dueling_ variant, as it improves learning efficiency by decomposing the Q value into two separate functions: a state-value function and an advantage function. Instead of directly estimating $Q(s,a)$, the network learns:

- $V(s)$ = how good it is to be in state $s$, regardless of action
- $A(s,a)$ = how much better action $a$ is in state $s$ compared to the average action.

These are combined to produce the final Q value:

$$Q(s,a|\theta,\alpha,\beta)=V(s|\theta,\beta)+\bigg(A(s,a|\theta,\alpha)-\frac{1}{|A|} \sum_{a'}A(s,a'|\theta,\alpha)\bigg)$$

Where:
- $V(s|\theta,\beta)$ = value function output
- $A(s,a|\theta,\alpha)$ = advantage function output
- $\theta$ = parameters shared by both functions
- $\alpha,\beta$ = parameters specific to the advantage and value networks, respectively
- The mean advantage is subtracted to enforce _identifiability_, ensuring the value and advantage functions cannot arbitrarily shift while leaving the Q values unchanged.

See `dueling_ddqn/dueling_ddqn_agent.py` for this implementation.

PER is also implemented here (`dueling_ddqn/prioritised_replay_buffer.py`), as it's better than randomly sampling the replay buffer for learning. Instead, the agent favours transitions from which it can learn the most.

Sources:
- [Cart-Pole Gymnasium documentation](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf) (Hasselt, Guez, Silver 2015)
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/pdf/1511.06581) (Wang et. al. 2016)
- [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf) (Schaul et. al. 2016)
