"""
Constants for game environment

Author: Sam Barba
Created 16/02/2023
"""

# ----- PHYSICS ----- #
START_X = 89
START_Y = 456
FORCE = 0.11             # Force of acceleration/deceleration
FRICTION = 0.01          # Higher means faster deceleration
DRIFT_FRICTION = 0.03
TURN_RATE = 0.012
VEL_DRIFT_THRESHOLD = 3  # Min velocity required for drifitng
MAX_GRIP_VEL = 10.89     # Approximate max grip velocity (found by testing)
MAX_DRIFT_VEL = 17.86    # Approximate max drift velocity (found by testing)
DRIFT_FACTOR = 0.07      # Used to calculate change in drift speed
MAX_RAY_LENGTH = 300     # Max distance at which car can 'see'

# ----- RETURNS ----- #
GATE_REWARD = 10
CRASH_PENALTY = -100
TIMESTEP_PENALTY = -0.1

# ----- RENDERING ----- #
DRIFT_RENDER_THRESHOLD = 4  # Min. drift speed for rendering drift marks
CAR_WIDTH = 21
CAR_HEIGHT = 42
TRACK_WIDTH = 1361
TRACK_HEIGHT = 869
FPS = 90
