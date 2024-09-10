import os
import ctypes
import struct
import time

import numpy as np
import torch
import mmcv

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import ReliabilityPolicy, QoSProfile
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, TransformStamped
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
import tf2_ros
from object_detection_msgs.msg import Object3d, Object3dArray
import tf_transformations

from mmdet3d.apis import inference_detector, init_model
from mmdet3d.models.layers import aligned_3d_nms
from mmdet3d.structures import DepthInstance3DBoxes

def transform_point(trans, pt):
    # https://answers.ros.org/question/249433/tf2_ros-buffer-transform-pointstamped/
    quat = [
        trans.transform.rotation.x,
        trans.transform.rotation.y,
        trans.transform.rotation.z,
        trans.transform.rotation.w
    ]
    mat = tf_transformations.quaternion_matrix(quat)
    pt_np = [pt.x, pt.y, pt.z, 1.0]
    pt_in_map_np = np.dot(mat, pt_np)

    pt_in_map = Point()
    pt_in_map.x = pt_in_map_np[0] + trans.transform.translation.x
    pt_in_map.y = pt_in_map_np[1] + trans.transform.translation.y
    pt_in_map.z = pt_in_map_np[2] + trans.transform.translation.z

    return pt_in_map

class InferNode(Node):
    def __init__(self, device=torch.device('cuda:0')):
        super().__init__('infer_node')
        self.logger = self.get_logger()

        cache_time = Duration(seconds=2.0) 
        self.tf_buffer = tf2_ros.Buffer(cache_time)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.declare_parameter('config_file', '/path/to/py')
        self.declare_parameter('checkpoint_file', 'path/to/pth')
        self.declare_parameter('point_cloud_frame', 'velodyne')
        self.declare_parameter('point_cloud_topic', 'velodyne_points')
        self.declare_parameter('infer_device', 'cuda:0')
        self.declare_parameter('nms_interval', 0.5)
        self.declare_parameter('point_cloud_qos', 'best_effort')

        config_file_path = self.get_parameter('config_file').get_parameter_value().string_value
        self.checkpoint_file_path = self.get_parameter('checkpoint_file').get_parameter_value().string_value
        infer_device = self.get_parameter('infer_device').get_parameter_value().string_value
        nms_interval = self.get_parameter('nms_interval').get_parameter_value().double_value
        self.point_cloud_frame = self.get_parameter('point_cloud_frame').get_parameter_value().string_value
        point_cloud_qos = self.get_parameter('point_cloud_qos').get_parameter_value().string_value
        point_cloud_topic = self.get_parameter('point_cloud_topic').get_parameter_value().string_value

        self.score_thrs = {0: 0.50, 7: 0.1}

        qos = QoSProfile(depth=5)
        if point_cloud_qos == 'best_effort':
            qos.reliability = ReliabilityPolicy.BEST_EFFORT
        elif point_cloud_qos == 'reliable':
            qos.reliability = ReliabilityPolicy.RELIABLE
        else:
            self.logger.error('Invalid value for point_cloud_qos parameter')
            return

        self.device = device
        self.transform_stamped = TransformStamped()
        self.det3d_array = Detection3DArray()
        self.det3d_array.header.frame_id = self.point_cloud_frame

        if 'kitti' in self.checkpoint_file_path:
            self.logger.info("Running with Kitti")

            self.filtered_bboxes_tensor = torch.zeros(0, 7).cuda()
            self.field_names = ("x", "y", "z", "intensity")
        elif 'nus' in self.checkpoint_file_path:
            self.logger.info("Running with Nuscenes")
            
            self.filtered_bboxes_tensor = torch.zeros(0, 9).cuda()
            self.field_names = ("x", "y", "z", "intensity", "ring")
        else:
            self.logger.error('Unknown weight, path of weight should contain "kitti" or "nus"')

        self.get_logger().info('full_config_file: "%s"' % config_file_path)
        self.get_logger().info('checkpoint_file: "%s"' % self.checkpoint_file_path)
        self.model = init_model(config_file_path, self.checkpoint_file_path, device=self.device)

        self.subscription = self.create_subscription(
            PointCloud2,
            point_cloud_topic,
            self.listener_callback,
            qos)
        
        self.marker_pub = self.create_publisher(Detection3DArray, 'detect_bbox3d', 10)
        
        self.timer = self.create_timer(nms_interval, self.detections_callback)

        self.filtered_bboxes_nms = torch.zeros(0, 6).cuda()
        self.filtered_scores = torch.zeros(0).cuda()
        self.filtered_labels = torch.zeros(0).cuda()


    def listener_callback(self, msg):

        points = pc2.read_points(msg, field_names=self.field_names, skip_nans=True)
        x = points['x'].reshape(-1)
        y = points['y'].reshape(-1)
        z = points['z'].reshape(-1)
        i = (points['intensity'] / 255.0).reshape(-1)

        if 'kitti' in self.checkpoint_file_path:
            pc_np = np.stack((x, y, z, i)).T
        elif 'nus' in self.checkpoint_file_path:
            ring = points['ring'].reshape(-1)
            pc_np = np.stack((x, y, z, i, ring)).T
        else:
            self.logger.error('Unknown weight, path of weight should contain "kitti" or "nus"')

        point_cloud_tensor = torch.from_numpy(pc_np)
        point_cloud_tensor = point_cloud_tensor.contiguous()

        start_time = time.time()  # get current time

        model_result, data_afterprocess = inference_detector(self.model, point_cloud_tensor)
        
        end_time = time.time()  # get current time after inference
        # Calculate elapsed time in milliseconds
        elapsed_time_ms = (end_time - start_time) * 1000
        self.logger.debug("Inference time: {:.2f} ms".format(elapsed_time_ms))

        bboxes = model_result.pred_instances_3d.bboxes_3d
        scores = model_result.pred_instances_3d.scores_3d
        labels = model_result.pred_instances_3d.labels_3d
        
        masks = [scores > self.score_thrs.get(label, 0.7) for label in labels.unique()]
        mask = torch.stack(masks).any(dim=0)

        filtered_bboxes = bboxes[mask]
        filtered_scores = scores[mask]
        filtered_labels = labels[mask]
        
        if filtered_bboxes.shape[0] != 0:
            filtered_bboxes_x0 = filtered_bboxes.center[:,0]-0.5*filtered_bboxes.dims[:,0]
            filtered_bboxes_y0 = filtered_bboxes.center[:,1]-0.5*filtered_bboxes.dims[:,1]
            filtered_bboxes_z0 = filtered_bboxes.center[:,2]-0.5*filtered_bboxes.dims[:,2]
            filtered_bboxes_x1 = filtered_bboxes.center[:,0]+0.5*filtered_bboxes.dims[:,0]
            filtered_bboxes_y1 = filtered_bboxes.center[:,1]+0.5*filtered_bboxes.dims[:,1]
            filtered_bboxes_z1 = filtered_bboxes.center[:,2]+0.5*filtered_bboxes.dims[:,2]
            filtered_bboxes_nms = torch.stack((filtered_bboxes_x0, filtered_bboxes_y0,
                                               filtered_bboxes_z0, filtered_bboxes_x1,
                                               filtered_bboxes_y1, filtered_bboxes_z1), dim=1)
            self.filtered_bboxes_nms = torch.cat((self.filtered_bboxes_nms, filtered_bboxes_nms), dim=0)
            self.filtered_bboxes_tensor = torch.cat((self.filtered_bboxes_tensor, filtered_bboxes.tensor), dim=0)
            self.filtered_scores = torch.cat((self.filtered_scores, filtered_scores), dim=0)
            self.filtered_labels = torch.cat((self.filtered_labels, filtered_labels), dim=0)

    def detections_callback(self):
        pick_ind = aligned_3d_nms(self.filtered_bboxes_nms, self.filtered_scores, self.filtered_labels, 0.25)
        self.logger.info("[NMS] detections {} -> {}".format(self.filtered_bboxes_nms.shape[0], pick_ind.shape[0]))
        self.filtered_bboxes_nms = self.filtered_bboxes_nms[pick_ind]
        self.filtered_labels = self.filtered_labels[pick_ind]
        self.filtered_scores = self.filtered_scores[pick_ind]
        self.filtered_bboxes_tensor = self.filtered_bboxes_tensor[pick_ind]
        self.draw_bbox(self.filtered_bboxes_tensor.cpu(), self.filtered_labels.cpu().numpy(), self.filtered_scores.cpu().numpy())
    
    def draw_bbox(self, bboxes, labels, scores, timestamp=None):
        det3d_array = Detection3DArray()
        det3d_array.header.frame_id = self.point_cloud_frame
        if len(bboxes) > 0:
            for ind in range(len(bboxes)):
                bbox = bboxes[ind]
                score = scores[ind]
                label = int(labels[ind])
                det3d = Detection3D()
                det3d.header.frame_id = self.point_cloud_frame

                pose = Pose()
                pose.position.x = bbox[0].item()
                pose.position.y = bbox[1].item()
                pose.position.z = bbox[2].item()

                quat = Quaternion()
                if 'nus' in self.checkpoint_file_path:
                    if label in [1,3,4,6,8,9]: # skip unwanted labels (nuscenes case)
                        continue
                    q = tf_transformations.quaternion_from_euler(0, 0, bbox[6].item())
                else:
                    q = tf_transformations.quaternion_from_euler(0, 0, bbox[-1].item())
                quat.x = q[0]
                quat.y = q[1]
                quat.z = q[2]
                quat.w = q[3]
                pose.orientation = quat

                dimensions = Vector3()
                dimensions.x = bbox[3].item()
                dimensions.y = bbox[4].item()
                dimensions.z = bbox[5].item()

                det3d.bbox.center = pose
                det3d.bbox.size = dimensions
                object_hypothesis = ObjectHypothesisWithPose()
                object_hypothesis.hypothesis.class_id = str(label)
                object_hypothesis.hypothesis.score = score.item()
                det3d.results.append(object_hypothesis)
                
                det3d_array.detections.append(det3d)
                
                self.logger.debug(f"Found class_id: {label}, score: {score.item()}")

            self.marker_pub.publish(det3d_array)
        
def main(args=None):
    rclpy.init(args=args)
    infer_node = InferNode()
    rclpy.spin(infer_node)
    infer_node.destroy_node()
    rclpy.shutdown()
