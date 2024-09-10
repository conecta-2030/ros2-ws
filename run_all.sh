#!/bin/bash

colcon build
source install/setup.bash

trap "kill 0" EXIT

# ros2 launch velodyne velodyne-all-nodes-VLP16-launch.py &

ros2 launch mmdet3d_ros2 mmdet3d_infer_launch.py &

ros2 run object_visualization object3d_visualizer_node &

ros2 run rviz2 rviz2 &

wait