## Proximal Policy Optimisation with Generalised Advantage Estimation applied to Asteroids

<p align="center">
	<img src="agent_gameplay.webp"/>
</p>

For a full explanation of PPO, see [this other project](../pytorch_proximal_policy_optimisation). In `ppo/ppo_agent.py`, the PPO-Clip variant is implemented:

<p align="center">
	<img src="ppo/ppo_clip_pseudocode.png"/>
</p>

but with Generalised Advantage Estimation (GAE) in step 4 instead of returns-to-go.

Rewritten policy objective to maximise (step 6):

$$L^{CLIP}(\theta)=\hat{\mathbb{E}}_t\bigg[\mathrm{min}\bigg(r_t(\theta)\hat{A}_t,\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\bigg)\bigg]$$

Where:
- $\theta$ = policy function (actor network) weights
- $\hat{\mathbb{E}}_t$ denotes empirical expectation over timesteps

$$r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}=\frac{\text{prob. of choosing action } a_t \text{ in state } s_t \text{ under current policy } \pi_\theta}{\text{prob. of choosing } a_t \text{ in } s_t \text{ under } \pi_{\theta_{\text{old}}}}$$

- $\hat{A}_t$ = expected advantage at time $t$ (computed via GAE - see below)
- $\epsilon$ is a hyperparameter (usually 0.1-0.3) for clipping, which penalises large policy updates
- The $\mathrm{min()}$ function means that the algorithm is [pessimistic](https://arxiv.org/pdf/2012.15085.pdf).

As opposed to the other PPO project above, this uses GAE to compute advantages:

$$\hat{A}\_t = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V$$

- $\gamma$ = discount factor (0-1), which accounts for future returns
- $\lambda$ = GAE parameter (0-1), which trades off bias vs variance
- $\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$ = temporal difference error
- $r_t$ = return at timestep $t$
- $V(s_t)$ = value of state $s_t$.

Sources:
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/pdf/1506.02438) (Schulman et. al. 2016)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf) (Schulman et. al. 2017)
- [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/algorithms/ppo.html#exploration-vs-exploitation) (OpenAI 2018)
