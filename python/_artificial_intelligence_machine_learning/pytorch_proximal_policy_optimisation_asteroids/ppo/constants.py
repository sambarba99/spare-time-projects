"""
Constants for PPO agent

Author: Sam Barba
Created 11/07/2024
"""

# ----- TRAINING ----- #
TOTAL_TRAIN_TIMESTEPS = int(2e7)  # Simulate this many environment timesteps for training
MAX_EP_LENGTH = 8000              # Maximum no. timesteps of each training episode
BATCH_SIZE = 32768                # No. transition samples to use per PPO update (actor + critic networks)
MINIBATCH_SIZE = 512              # Use minibatches of this size per PPO update
NUM_EPOCHS = 5                    # No. training epochs per PPO update
ACTOR_LR = 1e-4                   # Learning rate for actor network
CRITIC_LR = 3e-4                  # Learning rate for critic network
GAMMA = 0.99                      # Return discount factor
LAMBDA = 0.95                     # Generalised Advantage Estimation parameter
EPSILON = 0.2                     # Used to clip the ratio during PPO updates - see README
VALUE_FUNC_COEFF = 0.5            # Value (critic) function coefficient
START_ENTROPY_COEFF = 0.02        # Max coefficient for entropy regularisation
END_ENTROPY_COEFF = 1e-3          # Min coefficient for entropy regularisation
ENTROPY_ANNEAL_STEPS = int(5e6)   # Anneal entropy coefficient from START to END over this many steps
KL_DIVERGENCE_THRESHOLD = 0.01    # Threshold for KL divergence
GRAD_NORM_THRESHOLD = 1           # Threshold to clip model gradients

# ----- MODEL/AGENT ----- #
MAX_ASTEROIDS_DETECT = 8   # Agent knows info about the nearest 8 asteroids (max)
NUM_INPUTS = 85            # Info about the spaceship and asteroids (see game_env.get_state)
NUM_ACTIONS = 12           # Do nothing, boost, turn left/right, shoot, combinations of these
LAYER_SIZE_ACTOR = 128     # Nodes per hidden layer (actor)
LAYER_SIZE_CRITIC = 128    # Nodes per hidden layer (critic)
