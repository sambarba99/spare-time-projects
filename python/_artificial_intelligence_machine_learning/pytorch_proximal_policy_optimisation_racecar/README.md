## Deep Reinforcement Learning with PyTorch: Proximal Policy Optimisation

An implementation of a simulated self-driving car that learns to race around a track using Proximal Policy Optimisation (PPO). For comparison against another type of RL agent, a Double Deep Q Network (DDQN) agent with Prioritised Experience Replay (PER) is also implemented.

PPO agent's performance (clipped to 5 laps):

<p align="center">
	<img src="ppo/ppo_5_laps.webp"/>
</p>

For an implementation of PPO with Generalised Advantage Estimation (instead of returns-to-go), see [this other project](../pytorch_proximal_policy_optimisation_asteroids).

<h3 align="center">PPO</h3>

PPO is considered state-of-the-art in Deep RL. It is a policy gradient method, meaning it searches the space of policies rather than assigning values to state-action pairs like regular Q-learning methods. It uses 2 functions: a policy function to choose actions, and a value function to evaluate states. PPO is motivated by the same question as Trust Region Policy Optimisation (TRPO): how can we take the biggest possible improvement step on a policy without stepping so far as to accidentally cause performance collapse? TRPO tries to solve this problem with a complex second-order method; PPO is a family of first-order methods that use other tricks to keep new policies close to old. PPO methods are significantly simpler to implement, but empirically seem to perform at least as well as TRPO.

In `ppo/ppo_agent.py`, the PPO-Clip variant is implemented, as there is no need to use Kullback-Leibler divergence. Pseudocode:

<p align="center">
	<img src="ppo/ppo_clip_pseudocode.png"/>
</p>

Rewritten policy objective to maximise (step 6):

$$L^{CLIP}(\theta)=\hat{\mathbb{E}}_t\bigg[\mathrm{min}\bigg(r_t(\theta)\hat{A}_t,\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\bigg)\bigg]$$

Where:
- $\theta$ = policy function (actor network) weights
- $\hat{\mathbb{E}}_t$ denotes empirical expectation over timesteps

$$r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}=\frac{\text{prob. of choosing action } a_t \text{ in state } s_t \text{ under current policy } \pi_\theta}{\text{prob. of choosing } a_t \text{ in } s_t \text{ under } \pi_{\theta_{\text{old}}}}$$

- $\hat{A}_t=Q(s_t,a_t)-V(s_t)$ = expected advantage at time $t$
- $\epsilon$ is a hyperparameter (usually 0.1-0.3) for clipping, which penalises large policy updates
- The $\mathrm{min()}$ function means that the algorithm is [pessimistic](https://arxiv.org/pdf/2012.15085.pdf).

Sources:
- [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf) (Schulman et. al. 2017)
- [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/algorithms/ppo.html#exploration-vs-exploitation) (OpenAI 2018)

<h3 align="center">DDQN with PER</h3>

In regular Q-learning, the Q value update is:

$$Q(s_t,a_t|\theta_t)=r_t+\gamma \underset{a \in A}{\mathop{\mathrm{max}}}\bigg(Q(s_{t+1},a|\theta_t)\bigg)$$

Where:
- $Q(s_t,a_t|\theta_t)$ = estimated value at time $t$ of taking action $a$ in state $s$ given model parameters $\theta$
- $A$ = action space
- $s_t,a_t,r_t,s_{t+1}$ = state, action, return, new state
- $\gamma$ = return discount factor.

This method is prone to maximisation bias, as the future approximated action value is estimated using the same Q function as the current action selection policy. In noisy environments, this can lead to a cumulative overestimation error. To mitigate this, Double Q-learning uses 2 models: an online policy model for action selection (with weights $\theta$), and an offline target model for action evaluation (with weights $\theta'$).

Hence, in Double Q-learning, the Q value update is:

$$Q(s_t,a_t|\theta_t)=r_t+\gamma Q\bigg(s_{t+1},\underset{a \in A}{\mathop{\mathrm{argmax}}}\bigg(Q(s_{t+1},a|\theta_t)\bigg)|\theta'_t\bigg)$$

The target model isn't trained, but its weights are gradually synchronised with those of the policy model after each epoch, using a small parameter $\tau$ (e.g. `1e-3`):

$$\theta'=\tau\theta+(1-\tau)\theta'$$

See `ddqn/ddqn_agent.py` for the implementation.

PER is also implemented here (`ddqn/prioritised_replay_buffer.py`), as it's better than randomly sampling the replay buffer for learning. Instead, the agent favours transitions from which it can learn the most.

Sources:
- [Deep Reinforcement Learning with Double Q Learning](https://arxiv.org/pdf/1509.06461.pdf) (Hasselt, Guez, Silver 2015)
- [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf) (Schaul et. al. 2016)
