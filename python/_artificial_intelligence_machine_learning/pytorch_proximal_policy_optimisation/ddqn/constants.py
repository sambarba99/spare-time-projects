"""
Constants/hyperparameters for DDQN agent

Author: Sam Barba
Created 16/02/2023
"""

# ----- TRAINING ----- #
N_EPOCHS = 20_000      # No. training epochs
MAX_EP_LENGTH = 5000   # Maximum length (timesteps/transitions) of each training episode/trajectory
NO_REWARD_LIMIT = 150  # If no positive return is gained after this many timesteps, skip to next episode
BATCH_SIZE = 64        # No. transitions with which to do experience replay (learning)

# ----- MODEL/AGENT ----- #
N_INPUTS = 13             # 10 distances to walls, car's vel and drift vel, direction to next reward gate
N_ACTIONS = 9             # Do nothing, accelerate, decelerate/reverse, turn left/right, combinations of these
LAYER_SIZE = 16           # Nodes per hidden layer
TAU = 1e-3                # Target network update rate
GAMMA = 0.9               # Return discount factor
MIN_EPSILON = 0.1         # Minimum epsilon (exploration rate, which starts at 1 and is decayed after each epoch)
EPSILON_DECAY = 0.999884  # Epsilon decay rate
LEARNING_RATE = 2.1e-4

# ----- PRIORITISED EXPERIENCE REPLAY BUFFER ----- #
PER_CAPACITY = 125_000  # Maximum no. transitions to store
PER_EPSILON = 0.01      # Minimum priority (prevents probabilities of 0)
PER_ALPHA = 0.6         # Prioritisation rate (0 = uniform random sampling)
PER_BETA = 0.4          # Determines the amount of importance sampling correction (1 = fully compensate)
PER_BETA_INC = 1e-3     # For annealing beta towards 1
