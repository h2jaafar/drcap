# MotionCommand.msg
# Definition of a motion command for a robot

# Current position of the robot [x, y, theta]
float64[] current_position

# Target position for the robot to move to [x, y, theta]
float64[] target_position

# Desired velocity to apply
float64 velocity

# Duration to apply the velocity
float64 duration