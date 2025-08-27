# DR.CAP: Distributed Multi-Robot Control and Planning with Centroid Estimation using Gaussian Belief Propagation

```
 ___    ___       ___    _____  ___   
(  _`\ |  _`\    (  _`\ (  _  )(  _`\ 
| | ) || (_) )   | ( (_)| (_) || |_) )
| | | )| ,  /    | |  _ |  _  || ,__/'
| |_) || |\ \  _ | (_( )| | | || |    
(____/'(_) (_)(_)(____/'(_) (_)(_)   
```
## Overview
This package implements a distributed Gaussian Belief Propagation (GBP) algorithm for multi-robot control, planning and estimation with obstacle avoidance in ROS2.

## Installation

Clone the repository and move to the `src` directory:
```bash
cd src
```

## Usage

### Running Nodes
To run the GBP nodes, specify the robot ID, the number of robots, and the radius from the centroid for each robot:
```bash
ros2 run robot_gbp gbp_node 0 4 0.5
ros2 run robot_gbp gbp_node 1 4 0.5
```

To run the centroid node which manages the shared centroid belief, pass the number of robots as an argument:
```bash
ros2 run robot_gbp centroid_node 2
```

### Using Launch Files
Alternatively, you can use the provided launch files to start the nodes:
```bash
colcon build && ros2 launch robot_gbp pure-sim.launch.py
```

## Simulate four robots:
```bash
cd ~/dev_ws/src
git clone https://github.com/h2jaafar/turtlebot3_multi_robot
ros2 launch turtlebot3_multi_robot gazebo-sim.launch.py enable_drive:=False

```

## Features
- Each node manages a local factor graph with asynchronous message passing.
- Nodes are designed to publish their state at regular intervals and subscribe to updates from other nodes.
- Upon receiving updates, nodes recalculate their beliefs and publish the updated beliefs to a central control topic.

## Todo:
- [x] Implement node-specific factor graphs.
- [x] Ensure robust message handling between nodes.
- [x] Enhance belief updates based on incoming messages.
- [x] Implement advanced control topic interactions for centralized coordination.

### Setuptools Bug
There is a known issue with `setuptools` versions. Ensure you use a compatible version of `packaging` and `setuptools` by reinstalling them as follows:
```bash
pip install --force-reinstall packaging==24.0
pip install --force-reinstall setuptools==70.0
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request with your suggested changes.