#!/bin/bash

colcon build
source install/setup.bash

# publishes velodyne data on /velodyne_points topic
ros2 launch velodyne velodyne-all-nodes-VLP16-launch.py &

# read from velodyne_points topic infer and publishes the result on object_detection_visualization topic
ros2 run lidar_object_detection lidar_object_detector_node &
ros2 run object_visualization object3d_visualizer_node &

# to play a record. the line that publishes velodyne data should'nt be used in this case
#ros2 bag play rosbag2_2024_08_29-13_01_21 &

ros2 run rviz2 rviz2
