#!/bin/bash

trap "kill 0" EXIT

cd /opt/carla-simulator

./CarlaUE4.sh -RenderOffScreen -prefernvidia &

cd /home/conecta/carla-ros-bridge/catkin_ws

colcon build
source install/setup.bash

ros2 launch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch.py &

wait