"""
Constants for PPO agent

Author: Sam Barba
Created 11/07/2024
"""

# ----- TRAINING ----- #
TOTAL_TRAIN_TIMESTEPS = int(1e7)  # Simulate this many environment timesteps/transitions for training
MAX_EP_LENGTH = 8000              # Maximum no. timesteps of each training episode/trajectory
BATCH_SIZE = 8192                 # No. transition samples to use per epoch
# MINIBATCH_SIZE = 32             # Use minibatches of this size for a single PPO update (actor + critic networks)
GRAD_STEPS_PER_UPDATE = 100       # Max. gradient steps per PPO update
ACTOR_LR = 1e-4                   # Learning rate for actor network
CRITIC_LR = 4e-4                  # Learning rate for critic network
GAMMA = 0.96                      # Return discount factor
LAMBDA = 0.95                     # Generalised Advantage Estimation parameter
EPSILON = 0.2                     # Used to clip the ratio during gradient update - see README
VALUE_FUNC_COEFF = 0.5            # Value (critic) function coefficient
ENTROPY_COEFF = 0.01              # Coefficient for entropy regularisation term
KL_THRESHOLD = 0.02               # KL divergence threshold

# ----- MODEL/AGENT ----- #
NUM_INPUTS = 65   # Asteroid info (see game_env.get_state)
NUM_ACTIONS = 12  # Do nothing, boost, turn left/right, shoot, combinations of these
LAYER_SIZE = 64   # Nodes per hidden layer
