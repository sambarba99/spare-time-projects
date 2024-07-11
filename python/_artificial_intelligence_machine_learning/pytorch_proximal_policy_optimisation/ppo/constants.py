"""
Constants for PPO agent

Author: Sam Barba
Created 16/02/2023
"""

# ----- TRAINING ----- #
TOTAL_TRAIN_TIMESTEPS = int(3e6)  # Simulate this many environment timesteps/transitions for training
MAX_EP_LENGTH = 5000              # Maximum no. timesteps of each training episode/trajectory
BATCH_SIZE = 8000                 # No. transition samples to use per PPO update
GRAD_STEPS_PER_EPOCH = 80         # No. gradient steps per PPO update (actor + critic networks)
ACTOR_LR = 3e-4                   # Learning rate for actor network
CRITIC_LR = 1e-3                  # Learning rate for critic network
GAMMA = 0.996                     # Return discount factor
EPSILON = 0.2                     # Used to clip the ratio during gradient update - see README/ppo_agent.py
VALUE_FUNC_COEFF = 0.5            # Value (critic) function coefficient
ENTROPY_COEFF = 0.01              # Coefficient for entropy regularisation term

# ----- MODEL/AGENT ----- #
NUM_INPUTS = 13   # 10 distances to walls, car's vel and drift vel, direction to next reward gate
NUM_ACTIONS = 9   # Do nothing, accelerate, decelerate/reverse, turn left/right, combinations of these
LAYER_SIZE = 16   # Nodes per hidden layer
