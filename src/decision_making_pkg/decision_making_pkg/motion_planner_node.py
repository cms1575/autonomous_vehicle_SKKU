import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
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

TIMER = 0.1

# 🚗 사용자 주행 모드 설정: "driving" or "parking"
MODE_SELECT = "parking"    # <<< 여기만 바꾸면 됨!

class MotionPlanningNode(Node):
    def __init__(self):
        super().__init__('motion_planner_node')

        self.mode_select = MODE_SELECT
        self.get_logger().info(f"=== MODE SELECTED: {self.mode_select} ===")

        self.sub_detection_topic = self.declare_parameter('sub_detection_topic', SUB_DETECTION_TOPIC_NAME).value
        self.sub_path_topic = self.declare_parameter('sub_lane_topic', SUB_PATH_TOPIC_NAME).value
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
        self.lidar_data = None
        self.traffic_light_data = None  # 항상 선언

        self.steering_command = 0
        self.left_speed_command = 0
        self.right_speed_command = 0

        # Common
        self.detection_sub = self.create_subscription(DetectionArray, self.sub_detection_topic, self.detection_callback, self.qos_profile)
        self.path_sub = self.create_subscription(PathPlanningResult, self.sub_path_topic, self.path_callback, self.qos_profile)
        self.lidar_sub = self.create_subscription(Bool, self.sub_lidar_obstacle_topic, self.lidar_callback, self.qos_profile)

        self.publisher = self.create_publisher(MotionCommand, self.pub_topic, self.qos_profile)
        self.lane_class_publisher = self.create_publisher(String, "current_lane_class", self.qos_profile)

        # For "normal" mode (ver1)
        if self.mode_select == "driving":
            self.sub_traffic_light_topic = self.declare_parameter('sub_traffic_light_topic', SUB_TRAFFIC_LIGHT_TOPIC_NAME).value
            self.traffic_light_sub = self.create_subscription(String, self.sub_traffic_light_topic, self.traffic_light_callback, self.qos_profile)

            self.current_lane_class = 'lane2'
            self.last_lane_change_time = self.get_clock().now()
            self.lane_change_cooldown = 20.0

        # For "parking" mode (ver2)
        if self.mode_select == "parking":
            self.prev_lidar_state = None
            self.lidar_state_change_count = 0
            self.mode = "normal"
            self.exit_parking_start_time = self.get_clock().now()
            self.exit_parking_phase = 0

        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    # === CALLBACKS ===
    def detection_callback(self, msg: DetectionArray):
        self.detection_data = msg

    def path_callback(self, msg: PathPlanningResult):
        self.path_data = list(zip(msg.x_points, msg.y_points))

    def traffic_light_callback(self, msg: String):
        if self.mode_select == "driving":
            self.traffic_light_data = msg

    def lidar_callback(self, msg: Bool):
        self.lidar_data = msg
        if self.mode_select == "parking":
            current_state = msg.data
            if self.prev_lidar_state is None:
                self.prev_lidar_state = current_state
            elif self.prev_lidar_state != current_state:
                self.lidar_state_change_count += 1
                self.get_logger().info(
                    f"[변화 감지] Lidar 상태 변화: {self.prev_lidar_state} -> {current_state} | 변화 횟수: {self.lidar_state_change_count}"
                )
                self.prev_lidar_state = current_state

    def toggle_lane(self):
        self.current_lane_class = 'lane2' if self.current_lane_class == 'lane1' else 'lane1'
        self.get_logger().info(f"[INFO] Changed lane to: {self.current_lane_class}")
        msg = String()
        msg.data = self.current_lane_class
        self.lane_class_publisher.publish(msg)

    # === TIMER CALLBACK ===
    def timer_callback(self):
        if self.mode_select == "driving":
            # === ver1 주행 로직 ===
            target_slope = 0
            if self.lidar_data is not None and self.lidar_data.data is True:
                self.steering_command = 0
                self.left_speed_command = 0
                self.right_speed_command = 0
            elif self.detection_data is not None:
                for detection in self.detection_data.detections:
                    if detection.class_name == 'traffic_light':
                        y_max = int(detection.bbox.center.position.y + detection.bbox.size.y / 2)
                        height = detection.bbox.size.y
                        now = self.get_clock().now()
                        elapsed = (now - self.last_lane_change_time).nanoseconds * 1e-9
                        if height > 45 and elapsed > self.lane_change_cooldown:
                            self.toggle_lane()
                            self.last_lane_change_time = now

                if self.path_data is None:
                    self.steering_command = 0
                else:
                    target_slope = DMFL.calculate_slope_between_points(self.path_data[-10], self.path_data[-1])
                    now = self.get_clock().now()
                    elapsed = (now - self.last_lane_change_time).nanoseconds * 1e-9
                    delta = 3 if elapsed < self.lane_change_cooldown else 1
                    st_max = 10 if elapsed < self.lane_change_cooldown else 8

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

            self.get_logger().info(f"[NORMAL MODE] steering: {self.steering_command}, gradient: {target_slope}, left_speed: {self.left_speed_command}, right_speed: {self.right_speed_command}")

        elif self.mode_select == "parking":
            # === ver2 주차 FSM ===
            if self.mode == "normal":
                self.left_speed_command = 50
                self.right_speed_command = 50
                self.steering_command = 0
                if self.lidar_state_change_count == 2:
                    self.mode = "parking_align"
                    self.left_speed_command = 0
                    self.right_speed_command = 0
                    self.steering_command = -7
                    return

            elif self.mode == "parking_align":
                self.left_speed_command = 50
                self.right_speed_command = 50
                self.steering_command = -7
                if self.lidar_state_change_count == 4:
                    self.mode = "parking_reverse"
                    self.left_speed_command = 0
                    self.right_speed_command = 0
                    self.steering_command = 5
                    return

            elif self.mode == "parking_reverse":
                self.left_speed_command = -50
                self.right_speed_command = -50
                if self.    lidar_state_change_count == 10:
                    self.current_lane_class = 'jucha'
                    msg = String()
                    msg.data = self.current_lane_class
                    self.lane_class_publisher.publish(msg)
                    self.mode = "parking_follow"

            elif self.mode == "parking_follow":
                if self.path_data is not None:
                    slope = DMFL.calculate_slope_between_points(self.path_data[-10], self.path_data[-1])
                    slope *= -1
                    if -7 < slope < 7:
                        self.steering_command = 0
                    elif slope > 0 and self.steering_command < 6:
                        self.steering_command += 1
                    elif slope < 0 and self.steering_command > -6:
                        self.steering_command -= 1
                self.left_speed_command = -50
                self.right_speed_command = -50
                if self.lidar_state_change_count == 6:
                    self.left_speed_command = 0
                    self.right_speed_command = 0
                    self.steering_command = 0
                    self.mode = "exit_parking"
                    self.exit_parking_start_time = self.get_clock().now()

            elif self.mode == "exit_parking":
                elapsed = (self.get_clock().now() - self.exit_parking_start_time).nanoseconds * 1e-9
                if self.exit_parking_phase == 0:
                    self.left_speed_command = 0
                    self.right_speed_command = 0
                    self.steering_command = 0
                    if elapsed >= 3.0:
                        self.exit_parking_phase = 1
                        self.exit_parking_start_time = self.get_clock().now()
                elif self.exit_parking_phase == 1:
                    self.left_speed_command = 50
                    self.right_speed_command = 50
                    self.steering_command = 0
                    if elapsed >= 5.0:
                        self.exit_parking_phase = 2
                        self.exit_parking_start_time = self.get_clock().now()
                elif self.exit_parking_phase == 2:
                    self.left_speed_command = 50
                    self.right_speed_command = 50
                    self.steering_command = 7
                    if elapsed >= 10.0:
                        self.exit_parking_phase = 3
                        self.exit_parking_start_time = self.get_clock().now()
                elif self.exit_parking_phase == 3:
                    self.left_speed_command = 50
                    self.right_speed_command = 50
                    self.steering_command = 0

            self.get_logger().info(f"[PARKING MODE] mode: {self.mode}, lidar_count: {self.lidar_state_change_count}, steering: {self.steering_command}, left: {self.left_speed_command}, right: {self.right_speed_command}")

        # === Common MotionCommand Publish ===
        msg = MotionCommand()
        msg.steering = self.steering_command
        msg.left_speed = self.left_speed_command
        msg.right_speed = self.right_speed_command
        self.publisher.publish(msg)

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
