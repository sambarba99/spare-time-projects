"""
Constants for environment

Author: Sam Barba
Created 2024-07-11
"""

# ----- PHYSICS ----- #
ACCELERATION_FORCE = 0.15
TURN_RATE = 0.1
MAX_VEL = 10
BULLET_SPEED = 16
MAX_BULLETS = 4
BULLET_LIFESPAN = 40  # No. timesteps before a bullet disappears
SPACESHIP_SCALE = 3.2
ASTEROID_SCALES = {'small': 4.5, 'medium': 9, 'large': 18}
ASTEROID_VELS = {'small': 6, 'medium': 4, 'large': 2}
MAX_OBJECT_DIST = ((1100 ** 2 + 760 ** 2) ** 0.5) / 2  # Max possible distance between 2 objects (considering wrap-around)

# ----- REWARDS ----- #
HUMAN_ASTEROID_DESTROY_REWARDS = {'small': 100, 'medium': 50, 'large': 20}
AGENT_ASTEROID_DESTROY_REWARD = 10
MISS_PENALTY = -2
COLLISION_PENALTY = -20

# ----- RENDERING ----- #
SCENE_WIDTH = 1100
SCENE_HEIGHT = 760
FPS = 60
