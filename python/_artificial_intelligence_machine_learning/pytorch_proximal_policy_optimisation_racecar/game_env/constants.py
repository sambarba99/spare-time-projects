"""
Constants for environment

Author: Sam Barba
Created 16/02/2023
"""

# ----- PHYSICS ----- #
START_X = 78
START_Y = 782
FORCE = 0.11             # Force of acceleration/deceleration
FRICTION = 0.01          # Higher means faster deceleration
DRIFT_FRICTION = 0.03
TURN_RATE = 0.008
VEL_DRIFT_THRESHOLD = 3  # Min velocity required for drifitng
MAX_GRIP_VEL = 10.89     # Approximate max grip velocity (found by testing)
MAX_DRIFT_VEL = 17.86    # Approximate max drift velocity (found by testing)
DRIFT_FACTOR = 0.07      # Used to calculate change in drift speed
MAX_RAY_LENGTH = 350     # Max distance at which car can 'see'

# ----- RETURNS ----- #
GATE_REWARD = 10
CRASH_PENALTY = -100
TIMESTEP_PENALTY = -0.1

# ----- RENDERING ----- #
DRIFT_RENDER_THRESHOLD = 4  # Min drift speed for rendering drift marks
MIN_FOLLOW_SPEED = 0.02     # Min relative speed at which camera follows car
MAX_FOLLOW_SPEED = 0.12     # Max relative speed at which camera follows car
LOOKAHEAD_FACTOR = 12       # This x car speed = camera lookahead distance
CAR_WIDTH = 33
CAR_HEIGHT = 67
SCENE_WIDTH = 1500
SCENE_HEIGHT = 900
FPS = 75
