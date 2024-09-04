from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mmdet3d_ros2',
            executable='infer_node',
            name='mmdet3d_infer_node',
            parameters=[
                {'config_file': '/home/conecta/ros2_ws/src/mmdet3d_ros2/mmdet3d_ros2/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'},
                {'checkpoint_file': '/home/conecta/ros2_ws/src/mmdet3d_ros2/mmdet3d_ros2/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'},
                {'score_threshold': 0.3},
                {'infer_device': 'cuda:0'},
                {'nms_interval': 0.5},
                {'point_cloud_qos': 'best_effort'},
                {'point_cloud_topic': 'velodyne_points'}
            ]
        )
    ])