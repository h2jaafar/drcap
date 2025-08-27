cmake_minimum_required(VERSION 3.5)
project(robot_gbp_msgs)

# Default to C99 for C and C++14 for C++
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)  # Ensure this is added to generate message files

# Generate message files
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/GraphState.msg"
  "msg/FactorState.msg"
  "msg/NodeState.msg"
  "msg/MotionCommand.msg"
  "msg/RobotStatus.msg"
  "action/MoveToPosition.action"
  "action/ContactServer.action"
  DEPENDENCIES std_msgs
)

# Testing setup
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
