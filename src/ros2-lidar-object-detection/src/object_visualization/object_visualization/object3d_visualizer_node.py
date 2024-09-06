import rclpy
from rclpy.node import Node
from object_detection_msgs.msg import Object3d, Object3dArray
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.duration import Duration
from geometry_msgs.msg import Point
from tf_transformations import quaternion_matrix
import numpy as np

# label to color mappings, RGB
LABEL_TO_COLOR = {
    0: [1.0, 0.0, 0.0],     # Pedestrian
    1: [0.0, 1.0, 0.0],     # Cyclist
    2: [0.0, 0.0, 1.0]      # Car
}

class Object3dVisualizerNode(Node):

    def __init__(self):
        super().__init__('object3d_visualizer_node')

        self.subscription = self.create_subscription(
            msg_type = Detection3DArray,
            topic = 'detect_bbox3d',
            callback = self.visualize_detection3d,
            qos_profile = 1
        )

        # self.subscription = self.create_subscription(
        #     msg_type = Object3dArray,
        #     topic = 'object_detections_3d',
        #     callback = self.visual_objects3d,
        #     qos_profile = 1
        # )

        self.visualization_publisher = self.create_publisher(MarkerArray, 'object_detection_visualization', 10)

    def visualize_detection3d(self, msg: Detection3DArray):
        
        marker_array = MarkerArray()
        for det3d in msg.detections:
            marker = Marker()
            marker.header.frame_id = msg.header.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = marker.header.stamp.nanosec  # Replace with a unique id if available
            marker.type = 5
            marker.color.r, marker.color.g, marker.color.b = LABEL_TO_COLOR[int(det3d.results[0].hypothesis.class_id)]
            marker.color.a = 1.0
            marker.scale.x = 0.1
            marker.lifetime = Duration(seconds=5.0).to_msg()
            marker.ns = "object_visualization"

            # Extract pose and dimensions
            pose = det3d.bbox.center
            size = det3d.bbox.size
            
            # Compute the rotation matrix from the quaternion
            quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            rotation_matrix = quaternion_matrix(quat)[:3, :3]
            
            # Compute the 8 corners of the bounding box
            half_size = np.array([size.x / 2, size.y / 2, size.z / 2])
            corner_offsets = np.array([
                [-half_size[0], -half_size[1], -half_size[2]],
                [half_size[0], -half_size[1], -half_size[2]],
                [half_size[0], half_size[1], -half_size[2]],
                [-half_size[0], half_size[1], -half_size[2]],
                [-half_size[0], -half_size[1], half_size[2]],
                [half_size[0], -half_size[1], half_size[2]],
                [half_size[0], half_size[1], half_size[2]],
                [-half_size[0], half_size[1], half_size[2]]
            ])
            
            rotated_corners = (rotation_matrix @ corner_offsets.T).T + np.array([
                pose.position.x, pose.position.y, pose.position.z
            ])
            
            # Add lines to marker.points to create the bounding box
            # Connect corners (0-1, 1-2, 2-3, 3-0) - bottom face
            for i in range(4):
                src = Point(x=rotated_corners[i][0], y=rotated_corners[i][1], z=rotated_corners[i][2])
                dst = Point(x=rotated_corners[(i + 1) % 4][0], y=rotated_corners[(i + 1) % 4][1], z=rotated_corners[(i + 1) % 4][2])
                marker.points.append(src)
                marker.points.append(dst)
            
            # Connect corners (4-5, 5-6, 6-7, 7-4) - top face
            for i in range(4, 8):
                src = Point(x=rotated_corners[i][0], y=rotated_corners[i][1], z=rotated_corners[i][2])
                dst = Point(x=rotated_corners[(i + 1) % 4 + 4][0], y=rotated_corners[(i + 1) % 4 + 4][1], z=rotated_corners[(i + 1) % 4 + 4][2])
                marker.points.append(src)
                marker.points.append(dst)
            
            # Connect vertical lines (0-4, 1-5, 2-6, 3-7)
            for i in range(4):
                src = Point(x=rotated_corners[i][0], y=rotated_corners[i][1], z=rotated_corners[i][2])
                dst = Point(x=rotated_corners[i + 4][0], y=rotated_corners[i + 4][1], z=rotated_corners[i + 4][2])
                marker.points.append(src)
                marker.points.append(dst)

            marker_array.markers.append(marker)

            # Create marker for score
            score_marker = Marker()
            score_marker.header.frame_id = msg.header.frame_id
            score_marker.header.stamp = self.get_clock().now().to_msg()
            score_marker.id = marker.header.stamp.nanosec + 1
            score_marker.type = Marker.TEXT_VIEW_FACING
            score_marker.color.r, score_marker.color.g, score_marker.color.b = [1.0, 1.0, 1.0]
            score_marker.color.a = 1.0
            score_marker.scale.z = 0.5
            score_marker.lifetime = Duration(seconds=5.0).to_msg()
            score_marker.ns = "object_visualization"

            score_marker.pose.position = Point(x=pose.position.x, y=pose.position.y, z=pose.position.z + size.z / 2 + 0.5)
            score_marker.pose.orientation.w = 1.0
            
            score = det3d.results[0].hypothesis.score
            score_marker.text = f"{score:.2f}"

            marker_array.markers.append(score_marker)
        
        self.visualization_publisher.publish(marker_array)


    def visual_objects3d(self, msg: Object3dArray):

        marker_array = MarkerArray()
        for object in msg.objects:
            
            marker = Marker()
            marker.header.frame_id = msg.header.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = marker.header.stamp.nanosec
            marker.type = 5
            marker.color.r, marker.color.g, marker.color.b = LABEL_TO_COLOR[object.label]
            marker.color.a = 1.0
            marker.scale.x = 0.10
            marker.lifetime = Duration(seconds=5.0).to_msg()
            marker.ns = "object_visualization"

            for i in range(4):
                # this should do 0-1, 1-2, 2-3, 3-4
                src = object.bounding_box.corners[i]
                dst = object.bounding_box.corners[(i+1) % 4]
                marker.points.append(src)
                marker.points.append(dst)

                # this should do 4-5, 5-6, 6-7, 7-4
                src = object.bounding_box.corners[i+4]
                dst = object.bounding_box.corners[((i+1) % 4) + 4]
                marker.points.append(src)
                marker.points.append(dst)

                # this should do 0-4, 1-5, 2-6, 3-7
                src = object.bounding_box.corners[i]
                dst = object.bounding_box.corners[i+4]
                marker.points.append(src)
                marker.points.append(dst)

            marker_array.markers.append(marker)

        self.visualization_publisher.publish(marker_array)
        self.get_logger().info("Published visualization")
        


def main(args=None):
    rclpy.init(args=args)


    object3d_visualizer_node = Object3dVisualizerNode()

    rclpy.spin(object3d_visualizer_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    object3d_visualizer_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
