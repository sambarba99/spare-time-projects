"""
Constants for game environment

Author: Sam Barba
Created 11/07/2024
"""

# ----- PHYSICS ----- #
ACCELERATION_FORCE = 0.1
TURN_RATE = 0.07
MAX_VEL = 7
BULLET_SPEED = 12
MAX_BULLETS = 4
# Num. timesteps before a bullet disappears
BULLET_LIFESPAN = 52
ASTEROID_RADII = {'small': 16, 'medium': 32, 'large': 64}
ASTEROID_VELS = {'small': 4, 'medium': 2.5, 'large': 1}
# Agent knows info about these nearest asteroids
MAX_ASTEROIDS_DETECT = 16
# Max. possible distance to an asteroid (considering wrap-around and object sizes)
MAX_ASTEROID_DIST = ((1100 ** 2 + 760 ** 2) ** 0.5) / 2 - 46

# ----- RETURNS ----- #
HUMAN_ASTEROID_DESTROY_REWARDS = {'small': 20, 'medium': 10, 'large': 5}
AGENT_ASTEROID_DESTROY_REWARD = 50
COLLISION_PENALTY = -50

# ----- RENDERING ----- #
SPACESHIP_SCALE = 30
SCENE_WIDTH = 1100
SCENE_HEIGHT = 760
FPS = 90
