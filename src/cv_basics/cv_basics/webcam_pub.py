import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO

class ImagePublisher(Node):

    def __init__(self):

        super().__init__('image_publisher')
        self.get_logger().info("OpenCV version: " + cv2.__version__)

        self.publisher_ = self.create_publisher(Image, 'video_frames', 10)
      
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.cap = cv2.VideoCapture("rtsp://192.168.1.120/LiveMedia/ch1/Media1")

        self.br = CvBridge()

        self.model = YOLO('./checkpoint.pt')

    def timer_callback(self):
        ret, frame = self.cap.read()

        if ret:
            results = self.model(frame, verbose=False)

            boxes = results[0].boxes.xywh.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()

            filtered_boxes = []
            filtered_labels = []
            box_frame = frame.copy()

            class_names = self.model.names

            for i, (x, y, w, h, conf, cls) in enumerate(zip(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], confidences, class_ids)):
                label = class_names[int(cls)]
                if label not in ['pedestrian', 'vehicle']:
                    continue
                
                if conf > 0.5:
                    x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
                    filtered_boxes.append([x1, y1, x2 - x1, y2 - y1])  # [x, y, w, h]
                    filtered_labels.append(label)

                    color = (0, 255, 0) if label == 'person' else (0, 0, 255)
                    cv2.rectangle(box_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(box_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            ros_image = self.br.cv2_to_imgmsg(box_frame, encoding='bgr8')
            self.publisher_.publish(ros_image)

def main(args=None):
    rclpy.init(args=args)

    image_publisher = ImagePublisher()
    rclpy.spin(image_publisher)

    image_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
