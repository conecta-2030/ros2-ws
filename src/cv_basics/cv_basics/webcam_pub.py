import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np
import time
import json

data = {}

class ImagePublisher(Node):

  def __init__(self):

    super().__init__('image_publisher')


    self.get_logger().info(cv2.__version__)

      
    self.publisher_ = self.create_publisher(Image, 'video_frames', 10)
      
    timer_period = 0.1
      
    self.timer = self.create_timer(timer_period, self.timer_callback)
  
    self.cap = cv2.VideoCapture("rtsp://192.168.1.120/LiveMedia/ch1/Media1")
         
    self.br = CvBridge()

    self.net = cv2.dnn.readNet(
      '/home/conecta-orin-01/Desktop/yolov3.weights',
      '/home/conecta-orin-01/Desktop/yolov3.cfg'
    )

    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

    self.classes = ['person', '', 'car']

  def get_output_layers(self):

    layer_names = self.net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
    return output_layers
   
  def timer_callback(self):

    ret, frame = self.cap.read()
          
    if ret:
      blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
      self.net.setInput(blob)
      layer_outputs = self.net.forward(self.get_output_layers())

      class_ids = []
      confidences = []
      boxes = []
      height, width, _ = frame.shape
      for output in layer_outputs:
          for detection in output:
              scores = detection[5:]
              class_id = np.argmax(scores)
              if class_id not in [0, 2]: continue
              confidence = scores[class_id]
              if confidence > 0.5 and self.classes[class_id] in ['person', '', 'car']:
                  center_x = int(detection[0] * width)
                  center_y = int(detection[1] * height)
                  w = int(detection[2] * width)
                  h = int(detection[3] * height)

                  x = int(center_x - w / 2)
                  y = int(center_y - h / 2)
                  boxes.append([x, y, w, h])
                  confidences.append(float(confidence))
                  class_ids.append(class_id)


      indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
      filtered_boxes = []
      filtered_labels = []
      box_frame = frame.copy()

      if len(indices) > 0:
          for i in indices.flatten():
              x, y, w, h = boxes[i]
              label = self.classes[class_ids[i]]
              filtered_boxes.append([x, y, w, h])
              filtered_labels.append(label)
              confidence = confidences[i]
              color = (0, 255, 0) if label == 'person' else (0, 0, 255)

              cv2.rectangle(box_frame, (x, y), (x + w, y + h), color, 2)
              cv2.putText(box_frame, f"{label} {confidence:.2f}", (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

      ros_image = self.br.cv2_to_imgmsg(box_frame, encoding='bgr8')
      self.publisher_.publish(ros_image)

      # if len(filtered_labels) > 0:
        
        # frame_id = time.time()
        # data[frame_id] = {
        #   "boxes": filtered_boxes,
        #   "labels": filtered_labels
        # }

        # with open("dataset/yolo_detections.json", "w") as f:

        #   json.dump(data, f, indent=4)

        # cv2.imwrite(f"dataset/lidar/frames/{frame_id}.jpg", frame)
        #cv2.imwrite(f"dataset/out/{frame_id}.jpg", box_frame)

  
def main(args=None):
  
  rclpy.init(args=args)

  image_publisher = ImagePublisher()
  rclpy.spin(image_publisher)
  image_publisher.destroy_node()

  rclpy.shutdown()
  
if __name__ == '__main__':
  main()