"""
Constants for PPO agent

Author: Sam Barba
Created 16/02/2023
"""

# ----- TRAINING ----- #
TOTAL_TRAIN_TIMESTEPS = int(4e6)  # Simulate this many environment timesteps for training
MAX_EP_LENGTH = 5000              # Maximum no. timesteps of each training episode
BATCH_SIZE = 8192                 # No. transition samples to use per PPO update (actor + critic networks)
MINIBATCH_SIZE = 512              # Use minibatches of this size per PPO update
NUM_EPOCHS = 15                   # No. training epochs per PPO update
ACTOR_LR = 1.5e-4                 # Learning rate for actor network
CRITIC_LR = 1e-4                  # Learning rate for critic network
GAMMA = 0.996                     # Return discount factor
LAMBDA = 0.95                     # Generalised Advantage Estimation parameter
EPSILON = 0.2                     # Used to clip the ratio during PPO updates - see README
VALUE_FUNC_COEFF = 0.4            # Value (critic) function coefficient
START_ENTROPY_COEFF = 0.02        # Max coefficient for entropy regularisation
END_ENTROPY_COEFF = 1e-3          # Min coefficient for entropy regularisation
ENTROPY_ANNEAL_STEPS = int(3e6)   # Anneal entropy coefficient from START to END over this many steps
KL_DIVERGENCE_THRESHOLD = 0.03    # Threshold for KL divergence
GRAD_NORM_THRESHOLD = 1           # Threshold to clip model gradients

# ----- MODEL/AGENT ----- #
NUM_INPUTS = 14         # 10 distances to walls; car's vel, drift vel and steering; direction to next reward gate
NUM_ACTIONS = 9         # Do nothing, accelerate, decelerate/reverse, turn left/right, combinations of these
LAYER_SIZE_ACTOR = 64   # Nodes per hidden layer (actor)
LAYER_SIZE_CRITIC = 64  # Nodes per hidden layer (critic)
