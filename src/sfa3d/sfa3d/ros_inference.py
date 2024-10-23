#!/usr/bin/env python3
from sensor_msgs_py import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, TransformStamped

import rclpy
from rclpy.node import Node
import numpy as np
import timeit
import os
import warnings
import torch

from sfa.models.model_utils import create_model
from sfa.utils.evaluation_utils import draw_predictions, convert_det_to_real_values
import sfa.config.kitti_config as cnf
from sfa.data_process.transformation import lidar_to_camera_box
from sfa.utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes
from sfa.data_process.kitti_data_utils import Calibration
from sfa.utils.demo_utils import parse_demo_configs, do_detect, download_and_unzip, write_credit
from sfa.data_process.kitti_bev_utils import makeBEVMap
from sfa.data_process.kitti_data_utils import get_filtered_lidar

warnings.filterwarnings("ignore", category=UserWarning)

ID_TO_CLASS_NAME = {
    0: 'pedestrian',
    1: 'car',
    2: 'cyclist',
    -3: 'truck',
    -99: 'tram',
    -1: 'unknown'
}

def euler_to_quaternion(yaw, pitch, roll):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return [qx, qy, qz, qw]

class SuperFastObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('super_fast_object_detection')
        self.pub = self.create_publisher(Detection3DArray, 'detect_bbox3d', 10)
        self.sub = self.create_subscription(PointCloud2, 'velodyne_points', self.on_scan, 10)
        self.configs = parse_demo_configs()
        self.load_model()
        self.point_cloud_frame = 'velodyne'

        self.map_ids = {
            "car": "0",
            "cyclist": "5",
            "pedestrian": "7"
        }

    def load_model(self):
        self.configs.pretrained_path = '/home/conecta/ros2_ws/src/sfa3d/checkpoints/fpn_resnet_18/fpn_resnet_18_epoch_300.pth'
        self.model = create_model(self.configs)
        assert os.path.isfile(self.configs.pretrained_path), f"No file at {self.configs.pretrained_path}"
        self.model.load_state_dict(torch.load(self.configs.pretrained_path, map_location='cuda:0'))
        self.configs.device = torch.device('cpu' if self.configs.no_cuda else f'cuda:{self.configs.gpu_idx}')
        self.model = self.model.to(device=self.configs.device)
        self.model.eval()

    def on_scan(self, scan):
        start = timeit.default_timer()
        self.get_logger().info("Got scan")
        gen = []
        for p in pc2.read_points(scan, field_names=("x", "y", "z", "intensity"), skip_nans=True):
            gen.append(np.array([p[0], p[1], p[2], p[3] / 100.0]))
        gen_numpy = np.array(gen, dtype=np.float32)

        front_lidar = get_filtered_lidar(gen_numpy, cnf.boundary)
        bev_map = makeBEVMap(front_lidar, cnf.boundary)
        bev_map = torch.from_numpy(bev_map)

        with torch.no_grad():
            detections, bev_map, fps = do_detect(self.configs, self.model, bev_map, is_front=True)
        self.get_logger().info(f"FPS: {fps}")
        objects_msg = Detection3DArray()
        objects_msg.header.stamp = self.get_clock().now().to_msg()
        objects_msg.header.frame_id = self.point_cloud_frame

        flag = False
        for j in range(self.configs.num_classes):
            class_name = ID_TO_CLASS_NAME[j]

            if len(detections[j]) > 0:
                flag = True
                for det in detections[j]:
                    _score, _x, _y, _z, _h, _w, _l, _yaw = det
                    yaw = -_yaw
                    x = _y / cnf.BEV_HEIGHT * cnf.bound_size_x + cnf.boundary['minX']
                    y = _x / cnf.BEV_WIDTH * cnf.bound_size_y + cnf.boundary['minY']
                    z = _z + cnf.boundary['minZ']
                    w = _w / cnf.BEV_WIDTH * cnf.bound_size_y
                    l = _l / cnf.BEV_HEIGHT * cnf.bound_size_x

                    obj = Detection3D()
                    obj.header.stamp = self.get_clock().now().to_msg()
                    obj.header.frame_id = self.point_cloud_frame

                    obj.results.append(ObjectHypothesisWithPose())
                    obj.results[0].hypothesis.score = float(_score)

                    obj.results[0].hypothesis.class_id = self.map_ids.get(class_name, '1')

                    pose = Pose()
                    pose.position.x = x
                    pose.position.y = y
                    pose.position.z = z
                    
                    [qx, qy, qz, qw] = euler_to_quaternion(yaw, 0, 0)

                    quat = Quaternion()
                    quat.x = qx 
                    quat.y = qy
                    quat.z = qz
                    quat.w = qw
                    pose.orientation = quat

                    dimensions = Vector3()
                    dimensions.x = l
                    dimensions.y = w
                    dimensions.z = float(_h)

                    obj.bbox.center = pose
                    obj.bbox.size = dimensions

                    objects_msg.detections.append(obj)
                
        if flag:
            self.pub.publish(objects_msg)

        stop = timeit.default_timer()
        self.get_logger().info(f'Time: {stop - start}')

def main(args=None):
    rclpy.init(args=args)
    node = SuperFastObjectDetectionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
