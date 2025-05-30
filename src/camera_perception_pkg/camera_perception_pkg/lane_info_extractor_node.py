import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from std_msgs.msg import String
from interfaces_pkg.msg import TargetPoint, LaneInfo, DetectionArray, BoundingBox2D, Detection
from .lib import camera_perception_func_lib as CPFL

#---------------Variable Setting---------------
SUB_TOPIC_NAME = "detections"
PUB_TOPIC_NAME = "yolov8_lane_info"
ROI_IMAGE_TOPIC_NAME = "roi_image"
LANE_CLASS_TOPIC_NAME = "current_lane_class"
SHOW_IMAGE = True
#----------------------------------------------

class Yolov8InfoExtractor(Node):
    def __init__(self):
        super().__init__('lane_info_extractor_node')

        self.sub_topic = self.declare_parameter('sub_detection_topic', SUB_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.show_image = self.declare_parameter('show_image', SHOW_IMAGE).value

        self.cv_bridge = CvBridge()
        self.current_lane_class = 'lane2'

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self.subscriber = self.create_subscription(DetectionArray, self.sub_topic, self.yolov8_detections_callback, self.qos_profile)
        self.publisher = self.create_publisher(LaneInfo, self.pub_topic, self.qos_profile)
        self.roi_image_publisher = self.create_publisher(Image, ROI_IMAGE_TOPIC_NAME, self.qos_profile)
        self.lane_class_subscriber = self.create_subscription(String, LANE_CLASS_TOPIC_NAME, self.lane_class_callback, self.qos_profile)

    def lane_class_callback(self, msg: String):
        self.current_lane_class = msg.data
        self.get_logger().info(f"[INFO] Current lane class updated to: {self.current_lane_class}")

    def yolov8_detections_callback(self, detection_msg: DetectionArray):
        if len(detection_msg.detections) == 0:
            return

        for detection in detection_msg.detections:
            self.get_logger().info(f"[DEBUG] class_name: '{detection.class_name}'")
            lane_edge_image = CPFL.draw_edges(detection_msg, cls_name=self.current_lane_class, color=255)

        

        (h, w) = lane_edge_image.shape[0], lane_edge_image.shape[1]
        dst_mat = [[round(w * 0.3), round(h * 0.0)], [round(w * 0.7), round(h * 0.0)], [round(w * 0.7), h], [round(w * 0.3), h]]
        src_mat = [[238, 316],[402, 313], [501, 476], [155, 476]]

        bird_image = CPFL.bird_convert(lane_edge_image, srcmat=src_mat, dstmat=dst_mat)
        roi_image = CPFL.roi_rectangle_below(bird_image, cutting_idx=300)

        if self.show_image:
            cv2.imshow('lane_edge_image', lane_edge_image)
            cv2.imshow('bird_img', bird_image)
            cv2.imshow('roi_img', roi_image)
            cv2.waitKey(1)

        roi_image = cv2.convertScaleAbs(roi_image)

        try:
            roi_image_msg = self.cv_bridge.cv2_to_imgmsg(roi_image, encoding="mono8")
            self.roi_image_publisher.publish(roi_image_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to convert and publish ROI image: {e}")

        grad = CPFL.dominant_gradient(roi_image, theta_limit=70)
        target_points = []
        for target_point_y in range(5, 155, 50):
            target_point_x = CPFL.get_lane_center(roi_image, detection_height=target_point_y,
                                                  detection_thickness=10, road_gradient=grad, lane_width=300)
            target_point = TargetPoint()
            target_point.target_x = round(target_point_x)
            target_point.target_y = round(target_point_y)
            target_points.append(target_point)

        lane = LaneInfo()
        lane.slope = grad
        lane.target_points = target_points

        self.publisher.publish(lane)

def main(args=None):
    rclpy.init(args=args)
    node = Yolov8InfoExtractor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()