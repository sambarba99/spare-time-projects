"""
Constants for DDQN agent

Author: Sam Barba
Created 2026-06-05
"""

# ----- TRAINING ----- #
TOTAL_TRAIN_TIMESTEPS = int(1e6)  # Simulate this many environment timesteps for training
MAX_EP_LENGTH = 2000              # Maximum no. timesteps of each training episode
BATCH_SIZE = 64                   # Do experience replay (learning) with this many transitions
TRAIN_FREQ = 5                    # Do experience replay every 5 timesteps
CHECKPOINT_FREQ = 50              # Checkpoint model every 50 timesteps
LEARNING_RATE = 1e-4              # Learning rate for online action selection and offline evaluation (target) networks
GAMMA = 0.999                     # Return discount factor
TAU = 1e-3                        # Target network update rate
MIN_EPSILON = 0.05                # Minimum epsilon (exploration rate; starts at 1 and is decayed after each episode)
GRAD_NORM_THRESHOLD = 1           # Threshold to clip model gradients

# ----- MODEL/AGENT ----- #
NUM_INPUTS = 5    # Cart position, cart velocity, sin/cos of pole angle, pole angular velocity
NUM_ACTIONS = 3   # Do nothing, apply force to the left, apply force to the right
LAYER_SIZE = 128  # Nodes per hidden layer

# ----- PRIORITISED EXPERIENCE REPLAY BUFFER ----- #
WARMUP_SAMPLES = 10_000  # No. transition samples to add to buffer before training
PER_CAPACITY = 100_000   # Maximum no. transitions to store
PER_EPSILON = 0.01       # Minimum priority (prevents probabilities of 0)
PER_ALPHA = 0.4          # Prioritisation rate (0 = uniform random sampling)
PER_BETA = 0.4           # Determines the amount of importance sampling correction (1 = fully compensate)
