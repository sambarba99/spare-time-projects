"""
Constants for DDQN agent

Author: Sam Barba
Created 16/02/2023
"""

# ----- TRAINING ----- #
NUM_EPOCHS = 30_000       # No. training epochs
MAX_EP_LENGTH = 5000      # Maximum no. timesteps of each training episode/trajectory
NO_REWARD_LIMIT = 200     # If no positive return is gained after this many timesteps, skip to next episode
BATCH_SIZE = 128          # No. transitions with which to do experience replay (learning)
TAU = 1e-3                # Target network update rate
GAMMA = 0.99              # Return discount factor
MIN_EPSILON = 0.01        # Minimum epsilon (exploration rate, which starts at 1 and is decayed after each epoch)
EPSILON_DECAY = 0.999846  # Epsilon decay rate (exploration rate is multiplied by this after every epoch)
LEARNING_RATE = 2.5e-4

# ----- MODEL/AGENT ----- #
NUM_INPUTS = 13   # 10 distances to walls, car's vel and drift vel, direction to next reward gate
NUM_ACTIONS = 9   # Do nothing, accelerate, decelerate/reverse, turn left/right, combinations of these
LAYER_SIZE = 128  # Nodes per hidden layer

# ----- PRIORITISED EXPERIENCE REPLAY BUFFER ----- #
WARMUP_SAMPLES = 10_000  # No. samples to add to buffer before training
PER_CAPACITY = 125_000   # Maximum no. transitions to store
PER_EPSILON = 0.01       # Minimum priority (prevents probabilities of 0)
PER_ALPHA = 0.6          # Prioritisation rate (0 = uniform random sampling)
PER_BETA = 0.4           # Determines the amount of importance sampling correction (1 = fully compensate)
PER_BETA_INC = 1e-4      # For annealing beta towards 1
