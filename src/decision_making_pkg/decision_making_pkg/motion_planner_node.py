import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from std_msgs.msg import String, Bool
from interfaces_pkg.msg import PathPlanningResult, DetectionArray, MotionCommand
from .lib import decision_making_func_lib as DMFL

#---------------Variable Setting---------------
SUB_DETECTION_TOPIC_NAME = "detections"
SUB_PATH_TOPIC_NAME = "path_planning_result"
SUB_TRAFFIC_LIGHT_TOPIC_NAME = "yolov8_traffic_light_info"
SUB_LIDAR_OBSTACLE_TOPIC_NAME = "lidar_obstacle_info"
PUB_TOPIC_NAME = "topic_control_signal"
#----------------------------------------------

# 모션 플랜 발행 주기 (초)
TIMER = 0.1

class MotionPlanningNode(Node):
    def __init__(self):
        super().__init__('motion_planner_node')

        self.sub_detection_topic = self.declare_parameter('sub_detection_topic', SUB_DETECTION_TOPIC_NAME).value
        self.sub_path_topic = self.declare_parameter('sub_lane_topic', SUB_PATH_TOPIC_NAME).value
        self.sub_traffic_light_topic = self.declare_parameter('sub_traffic_light_topic', SUB_TRAFFIC_LIGHT_TOPIC_NAME).value
        self.sub_lidar_obstacle_topic = self.declare_parameter('sub_lidar_obstacle_topic', SUB_LIDAR_OBSTACLE_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.timer_period = self.declare_parameter('timer', TIMER).value

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self.detection_data = None
        self.path_data = None
        self.traffic_light_data = None
        self.lidar_data = None

        self.steering_command = 0
        self.left_speed_command = 0
        self.right_speed_command = 0

        self.current_lane_class = 'lane2'
        self.last_lane_change_time = self.get_clock().now()
        self.lane_change_cooldown = 20.0

        self.detection_sub = self.create_subscription(DetectionArray, self.sub_detection_topic, self.detection_callback, self.qos_profile)
        self.path_sub = self.create_subscription(PathPlanningResult, self.sub_path_topic, self.path_callback, self.qos_profile)
        self.traffic_light_sub = self.create_subscription(String, self.sub_traffic_light_topic, self.traffic_light_callback, self.qos_profile)
        self.lidar_sub = self.create_subscription(Bool, self.sub_lidar_obstacle_topic, self.lidar_callback, self.qos_profile)

        self.publisher = self.create_publisher(MotionCommand, self.pub_topic, self.qos_profile)
        self.lane_class_publisher = self.create_publisher(String, "current_lane_class", self.qos_profile)

        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def detection_callback(self, msg: DetectionArray):
        self.detection_data = msg

    def path_callback(self, msg: PathPlanningResult):
        self.path_data = list(zip(msg.x_points, msg.y_points))

    def traffic_light_callback(self, msg: String):
        self.traffic_light_data = msg

    def lidar_callback(self, msg: Bool):
        self.lidar_data = msg

    def toggle_lane(self):
        self.current_lane_class = 'lane2' if self.current_lane_class == 'lane1' else 'lane1'
        self.get_logger().info(f"[INFO] Changed lane to: {self.current_lane_class}")
        msg = String()
        msg.data = self.current_lane_class
        self.lane_class_publisher.publish(msg)

    def timer_callback(self):
        target_slope = 0
        if self.lidar_data is not None and self.lidar_data.data is True:
            self.steering_command = 0
            self.left_speed_command = 0
            self.right_speed_command = 0
        elif self.detection_data is not None : #and self.traffic_light_data.data == 'Red'
            istraffic=False
            for detection in self.detection_data.detections:
                if detection.class_name == 'traffic_light':
                    istraffic=True
                    print("if detection.class_name == 'traffic_light':\n")
                    y_max = int(detection.bbox.center.position.y + detection.bbox.size.y / 2)
                    # if y_max > 130:
                    #     self.left_speed_command = 0
                    #     self.right_speed_command = 0
                    height = detection.bbox.size.y
                    now = self.get_clock().now()
                    elapsed = (now - self.last_lane_change_time).nanoseconds * 1e-9
                    if height > 45 and elapsed > self.lane_change_cooldown:
                        print("toggled!':\n")
                        self.toggle_lane()
                        self.last_lane_change_time = now

            if self.path_data is None:
                self.steering_command = 0
            else:
                target_slope = DMFL.calculate_slope_between_points(self.path_data[-10], self.path_data[-1])

                # 차선 변경 후 10초 이내인지 확인
                now = self.get_clock().now()
                elapsed = (now - self.last_lane_change_time).nanoseconds * 1e-9
                delta = 3 if elapsed < self.lane_change_cooldown else 1  # 증가/감소량
                st_max = 10 if elapsed < self.lane_change_cooldown else 8  # 증가/감소량
                


                if -13 < target_slope < 13:
                    self.steering_command = 0
                elif target_slope > 0 and self.steering_command < st_max:
                    self.steering_command += delta
                elif target_slope < 0 and self.steering_command > -(st_max):
                    self.steering_command -= delta

                if elapsed < self.lane_change_cooldown:
                    self.left_speed_command = 150
                    self.right_speed_command = 150
                else:
                    self.left_speed_command = 240
                    self.right_speed_command = 240
            


        self.get_logger().info(f"steering: {self.steering_command}, gradient: {target_slope}, left_speed: {self.left_speed_command}, right_speed: {self.right_speed_command}")

        motion_command_msg = MotionCommand()
        motion_command_msg.steering = self.steering_command
        motion_command_msg.left_speed = self.left_speed_command
        motion_command_msg.right_speed = self.right_speed_command
        self.publisher.publish(motion_command_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MotionPlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()