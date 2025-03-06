from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mmdet3d_ros2',
            executable='infer_node',
            name='mmdet3d_infer_node',
            parameters=[
                {'config_file': '/home/conecta/Projects/mmdetection3d/bevfusion_lidar_nondeterministic.py'},
                {'checkpoint_file': '/home/conecta/Projects/mmdetection3d/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth'},
                {'point_cloud_topic': '/carla/lidar'},
                {'point_cloud_frame', 'map'}
            ]
        )
    ])