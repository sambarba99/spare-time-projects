"""
Constants for game environment

Author: Sam Barba
Created 11/07/2024
"""

# ----- PHYSICS ----- #
ACCELERATION_FORCE = 0.1
TURN_RATE = 0.07
MAX_VEL = 7
LASER_VEL = 12
MAX_LASERS = 4
LASER_LIFESPAN = 52  # Num. timesteps before a laser shot disappears
ASTEROID_RADII = {'small': 16, 'medium': 32, 'large': 64}
ASTEROID_VELS = {'small': 3, 'medium': 2, 'large': 1}
MAX_ASTEROIDS_DETECT = 8  # Agent state contains info about these nearest asteroids
MAX_ASTEROID_DIST = (1100 ** 2 + 760 ** 2) / 4  # Max. possible distance to an asteroid (considering wrap-around)

# ----- RETURNS ----- #
HUMAN_ASTEROID_DESTROY_REWARDS = {'small': 20, 'medium': 10, 'large': 5}
AGENT_ASTEROID_DESTROY_REWARD = 10
# TIMESTEP_REWARD = -0.1
COLLISION_PENALTY = -100

# ----- RENDERING ----- #
SPACESHIP_SCALE = 30
SCENE_WIDTH = 1100
SCENE_HEIGHT = 760
FPS = 90
