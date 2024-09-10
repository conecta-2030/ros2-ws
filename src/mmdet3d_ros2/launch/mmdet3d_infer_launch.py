from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mmdet3d_ros2',
            executable='infer_node',
            name='mmdet3d_infer_node',
            parameters=[
                {'config_file': '/home/conecta/Projects/mmdetection3d/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py'},
                {'checkpoint_file': '/home/conecta/Projects/mmdetection3d/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth'},
                # {'config_file': '/home/conecta/Projects/mmdetection3d/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py'},
                # {'checkpoint_file': '/home/conecta/Projects/mmdetection3d/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth'},
                {'score_threshold': 0},
                {'infer_device': 'cuda:0'},
                {'nms_interval': 0.5},
                {'point_cloud_qos': 'best_effort'},
                {'point_cloud_topic': 'velodyne_points'}
            ]
        )
    ])