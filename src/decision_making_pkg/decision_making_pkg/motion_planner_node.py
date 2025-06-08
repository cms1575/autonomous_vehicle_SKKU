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

# 🚗 사용자 주행 모드 설정: "driving", "parking", "osbstacle"
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
        self.parking_align_start_time = None # 주차 시간 텀
        self.lidar_detected_msg_data=None
        self.steering_command = 0
        self.left_speed_command = 0
        self.right_speed_command = 0

        #obstacle mode전용
        # __init__ 안에 넣기
        self.obstacle_lane1_drive_start_time = None
        self.obstacle_lane2_car_detected_once = False

        self.lane1_wait_phase = 0  # 0: 대기, 1: 감지됨, 2: 넓이 통과, 3: 평균감소 추적
        self.lane1_car_width_history = []
        self.prev_lane1_car_avg = None
        self.lane1_width_decrease_count = 0
        self.lane1_width_decrease_threshold = 30  # 연속 감소 횟수 조건

        self.lane_change_to_lane2_start_time = None


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
            self.lane_change_cooldown = 16.0

        # For "parking" mode (ver2)
        if self.mode_select == "parking":
            self.prev_lidar_state = None
            self.lidar_state_change_count = 0
            self.mode = "normal"
            self.exit_parking_start_time = self.get_clock().now()
            self.exit_parking_phase = 0

        # For "obstacle_driving" mode
        if self.mode_select == "obstacle":
            self.sub_traffic_light_topic = self.declare_parameter('sub_traffic_light_topic', SUB_TRAFFIC_LIGHT_TOPIC_NAME).value
            self.traffic_light_sub = self.create_subscription(String, self.sub_traffic_light_topic, self.traffic_light_callback, self.qos_profile)

            self.current_lane_class = 'lane2'
            self.mode = "lane2_drive"
            self.lane1_moving_vehicle_threshold = 40
            self.lane2_static_vehicle_threshold = 41
            self.traffic_light_stop_threshold = 65
            self.waiting_for_green = False

            self.prev_lane1_car_height = None
            self.lane1_motion_state = None  # forward, backward
            self.lane1_forward_start_time = None
        
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    # === CALLBACKS ===
    def detection_callback(self, msg: DetectionArray):
        self.detection_data = msg

    def path_callback(self, msg: PathPlanningResult):
        self.path_data = list(zip(msg.x_points, msg.y_points))

    def traffic_light_callback(self, msg: String):
        if self.mode_select in ["driving", "obstacle"]:  # obstacle도 포함
            self.traffic_light_data = msg


    def lidar_callback(self, msg: Bool):
        self.lidar_data = msg
        if self.mode_select == "parking":
            current_state = msg.data
            self.lidar_detected_msg_data=current_state

             # 디버깅용 로그
            #self.get_logger().info(f"[디버깅] 수신된 Lidar 상태: current_state = {current_state}")

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
                    delta = 2 if elapsed < self.lane_change_cooldown else 2
                    st_max = 7 if elapsed < self.lane_change_cooldown else 8

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
                        self.left_speed_command = 255
                        self.right_speed_command = 255

            self.get_logger().info(f"[NORMAL MODE] steering: {self.steering_command}, gradient: {target_slope}, left_speed: {self.left_speed_command}, right_speed: {self.right_speed_command}")

        elif self.mode_select == "parking":
            # === ver2 주차 FSM ===
            if self.mode == "normal":
                self.left_speed_command = 50
                self.right_speed_command = 50
                self.steering_command = 0
                if self.lidar_detected_msg_data is True:
                    self.mode = "parking_align"
                    self.left_speed_command = 0
                    self.right_speed_command = 0
                    self.steering_command = -7
                    self.parking_align_start_time = self.get_clock().now()   # ⭐ 추가
                    return

            elif self.mode == "parking_align":
                self.left_speed_command = 50
                self.right_speed_command = 50
                self.steering_command = -6
                elapsed = (self.get_clock().now() - self.parking_align_start_time).nanoseconds * 1e-9
                if elapsed >= 13:    # ⭐ 5초 경과 시 parking_follow로 전환
                    self.mode = "parking_follow"
                    self.get_logger().info(f"[INFO] parking_align → parking_follow 전환됨 (elapsed {elapsed:.2f}s)")

            #     if self.lidar_state_change_count == 4:
            #         self.mode = "parking_reverse"
            #         self.left_speed_command = 0
            #         self.right_speed_command = 0
            #         self.steering_command = 5
            #         return

            # elif self.mode == "parking_reverse":
            #     self.left_speed_command = -50
            #     self.right_speed_command = -50
            #     if self.    lidar_state_change_count == 10:
            #         self.current_lane_class = 'jucha'
            #         msg = String()
            #         msg.data = self.current_lane_class
            #         self.lane_class_publisher.publish(msg)
            #         self.mode = "parking_follow"

            elif self.mode == "parking_follow":
                self.current_lane_class = 'jucha'
                msg = String()
                msg.data = self.current_lane_class
                self.lane_class_publisher.publish(msg)

                jucha_found = False
                jucha_center_y = 0

                if self.detection_data is not None:
                    for detection in self.detection_data.detections:
                        if detection.class_name == 'jucha':
                            jucha_found = True
                            jucha_center_y = detection.bbox.center.position.y
                            break  # 첫 jucha만 사용

                Y_THRESHOLD = 450  # <<< 원하는 y 기준값 설정 (예: 380px)

                if jucha_found and jucha_center_y > Y_THRESHOLD:
                    # jucha 있음 + bbox 중심 y가 너무 밑에 있음 → 직후진
                    self.get_logger().info(f"[INFO] jucha center_y={jucha_center_y:.2f}px > threshold {Y_THRESHOLD} → 직후진 (steering=0)")
                    self.steering_command = 0
                    self.left_speed_command = -40
                    self.right_speed_command = -40
                    self.mode="parking_back_straight"
                elif jucha_found:
                    # 정상 상황 → slope 기반 조향
                    if self.path_data is not None:
                        slope = DMFL.calculate_slope_between_points(self.path_data[-10], self.path_data[-8])
                        slope *= -1
                        if slope==0:
                            self.steering_command = self.steering_command
                        elif slope > 0 and self.steering_command < 7:
                            self.steering_command += 3
                        elif slope < 0 and self.steering_command > -7:
                            self.steering_command -= 3
                        self.left_speed_command = -50
                        self.right_speed_command = -50
                else:
                    # jucha 자체가 없으면 직후진으로 유지
                    self.get_logger().info(f"[INFO] jucha 없음 → 직후진 (steering=0)")
                    self.steering_command = 0
                    self.left_speed_command = -50
                    self.right_speed_command = -50


            elif self.mode=="parking_back_straight":
                self.steering_command =0
                self.left_speed_command = -40
                self.right_speed_command = -40
                if self.lidar_detected_msg_data is False:
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
                    self.left_speed_command = 100
                    self.right_speed_command = 100
                    self.steering_command = 0
                    if elapsed >= 0.7:
                        self.exit_parking_phase = 2
                        self.exit_parking_start_time = self.get_clock().now()
                elif self.exit_parking_phase == 2:
                    self.left_speed_command = 200
                    self.right_speed_command = 200
                    self.steering_command = 7
                    if elapsed >= 9.0:
                        self.exit_parking_phase = 3
                        self.exit_parking_start_time = self.get_clock().now()
                elif self.exit_parking_phase == 3:
                    self.left_speed_command = 200
                    self.right_speed_command = 200
                    self.steering_command = 1

            self.get_logger().info(f"[PARKING MODE] mode: {self.mode}, lidar_count: {self.lidar_state_change_count}, steering: {self.steering_command}, left: {self.left_speed_command}, right: {self.right_speed_command}")

        
        elif self.mode_select == "obstacle":
            # === obstacle_driving FSM ===
            if self.detection_data is None:
                return

            if self.mode == "lane2_drive":
                self.current_lane_class = 'lane2'

                # lane2_car 탐색 → block 여부 확인
                lane2_blocked = False
                for detection in self.detection_data.detections:
                    if detection.class_name == "lane2_car":
                        height = detection.bbox.size.y
                        if height > self.lane2_static_vehicle_threshold:
                            lane2_blocked = True
                            break

                # 상태 전이 조건
                if lane2_blocked:
                    self.get_logger().info("[FSM] lane2 blocked → lane_change_to_lane1")
                    self.mode = "lane_change_to_lane1"
                    return

                # 기본 동작 (신호등 관련 없음)
                self.left_speed_command = 100
                self.right_speed_command = 100

                if self.path_data and len(self.path_data) >= 10:
                    slope = DMFL.calculate_slope_between_points(self.path_data[-10], self.path_data[-1])
                    if -7 < slope < 7:
                        self.steering_command = 0
                    elif slope > 0 and self.steering_command < 6:
                        self.steering_command += 1
                    elif slope < 0 and self.steering_command > -6:
                        self.steering_command -= 1

            elif self.mode == "lane_change_to_lane1":
                self.toggle_lane()
                self.mode = "lane1_drive"
                self.get_logger().info("[FSM] lane_change_to_lane1 → lane1_drive")


            elif self.mode == "lane1_drive":

                # lane1_drive 진입 시간 기록 (처음에만)
                if self.obstacle_lane1_drive_start_time is None:
                    self.obstacle_lane1_drive_start_time = self.get_clock().now()
                    self.obstacle_lane2_car_detected_once = False
                    self.get_logger().info("[FSM] lane1_drive START TIMER")

                # 경과 시간 계산
                elapsed_time_sec = (self.get_clock().now() - self.obstacle_lane1_drive_start_time).nanoseconds / 1e9

                # 12초 이내 → 그냥 유지 (아무 판단 X)
                if elapsed_time_sec < 12.0:
                    self.left_speed_command = 100
                    self.right_speed_command = 100

                    if self.path_data and len(self.path_data) >= 10:
                        slope = DMFL.calculate_slope_between_points(self.path_data[-10], self.path_data[-1])
                        if -7 < slope < 7:
                            self.steering_command = 0
                        elif slope > 0 and self.steering_command < 7:
                            self.steering_command += 3
                        elif slope < 0 and self.steering_command > -7:
                            self.steering_command -= 3

                else: #12초이후

                    # 12초 이후부터는 lane2_car 인식 로직 활성화
                    lane2_car_detected_now = False
                    if self.detection_data is not None:
                        for detection in self.detection_data.detections:
                            if detection.class_name == "lane2_car":
                                width = detection.bbox.size.x
                                self.get_logger().info(f"[DEBUG] lane2_car detected!!!")
                                lane2_car_detected_now = True
                                self.obstacle_lane2_car_detected_once = True  # 감지한 적 있음 기록

                    # lane2_car가 한번이라도 감지 → 현재 사라졌으면 lane1_wait 전이
                    if self.obstacle_lane2_car_detected_once and not lane2_car_detected_now:
                        self.get_logger().info("[FSM] lane2_car 사라짐 → lane1_wait 전이")
                        self.mode = "lane1_wait"
                        self.obstacle_lane1_drive_start_time = None  # 초기화
                        self.obstacle_lane2_car_detected_once = False
                        return  # 현재 주행 상태 종료 (다음 콜백에서 lane1_wait 처리)

                    # 기존 lane1_drive 주행 명령 (12초 이후에도 유지)
                    self.left_speed_command = 100
                    self.right_speed_command = 100

                    if self.path_data and len(self.path_data) >= 10:
                        slope = DMFL.calculate_slope_between_points(self.path_data[-10], self.path_data[-1])
                        if -7 < slope < 7:
                            self.steering_command = 0
                        elif slope > 0 and self.steering_command < 6:
                            self.steering_command += 1
                        elif slope < 0 and self.steering_command > -6:
                            self.steering_command -= 1


            elif self.mode == "lane1_wait":
                self.left_speed_command = 0
                self.right_speed_command = 0
                self.steering_command = 0

                if self.detection_data is not None:
                    for detection in self.detection_data.detections:
                        if detection.class_name == "lane1_car":
                            width = detection.bbox.size.x
                            self.get_logger().info(f"[FSM] [lane1_wait] phase={self.lane1_wait_phase}, width={width:.2f}")

                            # Phase 0 → lane1_car 인식됨 → Phase 1
                            if self.lane1_wait_phase == 0:
                                self.get_logger().info("[FSM] lane1_car 감지됨 → Phase 1 진입")
                                self.lane1_wait_phase = 1

                            # Phase 1 → 넓이 160 이상이면 Phase 2
                            if self.lane1_wait_phase == 1 and width >= 150:
                                self.get_logger().info("[FSM] lane1_car width >= 160 → Phase 2 진입")
                                self.lane1_wait_phase = 2
                                self.lane1_car_width_history.clear()
                                self.prev_lane1_car_avg = None
                                self.lane1_width_decrease_count = 0

                            # Phase 2 → 평균 감소 추적
                            if self.lane1_wait_phase == 2:
                                self.lane1_car_width_history.append(width)
                                N = 5
                                if len(self.lane1_car_width_history) > N:
                                    self.lane1_car_width_history.pop(0)

                                if len(self.lane1_car_width_history) == N:
                                    current_avg = sum(self.lane1_car_width_history) / N
                                    if self.prev_lane1_car_avg is not None:
                                        if current_avg < self.prev_lane1_car_avg - 0.5:
                                            self.lane1_width_decrease_count += 1
                                            self.get_logger().info(f"[FSM] 평균 감소 감지! count={self.lane1_width_decrease_count}")
                                        else:
                                            self.lane1_width_decrease_count = 0  # 감소 아니면 초기화

                                        if self.lane1_width_decrease_count >= self.lane1_width_decrease_threshold:
                                            self.get_logger().info("[FSM] 평균 감소 100회 연속 → lane1_forward 전이")
                                            self.mode = "lane1_forward"
                                            self.lane1_forward_start_time = self.get_clock().now()
                                            self.lane1_wait_phase = 0
                                            return

                                    self.prev_lane1_car_avg = current_avg


            elif self.mode == "lane1_forward":
                self.left_speed_command = 40
                self.right_speed_command = 40

                if self.path_data and len(self.path_data) >= 10:
                    slope = DMFL.calculate_slope_between_points(self.path_data[-10], self.path_data[-1])
                    if -7 < slope < 7:
                        self.steering_command = 0
                    elif slope > 0 and self.steering_command < 6:
                        self.steering_command += 1
                    elif slope < 0 and self.steering_command > -6:
                        self.steering_command -= 1

            # ✅ Lidar가 True 되면 바로 전이
                if self.lidar_data and self.lidar_data.data is True:
                    self.get_logger().info("[FSM] Lidar 감지됨 → lane_change_to_lane2_phase1 전이 (2초동안 우측)")
                    self.mode = "lane_change_to_lane2_phase1"
                    self.lane_change_to_lane2_start_time = self.get_clock().now()
                    return
                
            elif self.mode == "lane_change_to_lane2_phase1":
                elapsed = (self.get_clock().now() - self.lane_change_to_lane2_start_time).nanoseconds / 1e9

                # 우측 조향 → 오른쪽으로 2초동안 가기
                self.left_speed_command = 100
                self.right_speed_command = 100
                self.steering_command = 7  # 우측 최대값 설정

                self.get_logger().info(f"[FSM] lane_change_to_lane2_phase1 진행중 ({elapsed:.2f}/2.0초)")

                if elapsed >= 2.0:
                    self.get_logger().info("[FSM] lane_change_to_lane2_phase1 완료 → lane2_drive_final_phase 전이")
                    self.toggle_lane()  # 여기서 토글 적용
                    self.mode = "lane2_drive_final_phase"
                    return

            

            elif self.mode == "lane2_drive_final_phase":
                self.current_lane_class = 'lane2'

                # traffic_light 상태 확인 (기존 lane2_drive의 코드 이동)
                traffic_light_close = False
                traffic_light_state = self.traffic_light_data.data.lower() if self.traffic_light_data and self.traffic_light_data.data else "unknown"
                for detection in self.detection_data.detections:
                    if detection.class_name == "traffic_light":
                        
                        height = detection.bbox.size.y

                        self.get_logger().info(f"height {height:.2f}")
                        self.get_logger().info(f"\n traffic light state {traffic_light_state}")


                        if height > self.traffic_light_stop_threshold:
                            traffic_light_close = True
                            break

                # 상태 전이 조건 (신호등 stop 처리)
                if traffic_light_close and traffic_light_state == "red":
                    self.get_logger().info("[FSM] traffic_light RED → traffic_stop")
                    self.mode = "traffic_stop"
                    return

                # 기본 주행
                self.left_speed_command = 120
                self.right_speed_command = 120

                if self.path_data and len(self.path_data) >= 10:
                    slope = DMFL.calculate_slope_between_points(self.path_data[-10], self.path_data[-1])
                    if -7 < slope < 7:
                        self.steering_command = 0
                    elif slope > 0 and self.steering_command < 6:
                        self.steering_command += 2
                    elif slope < 0 and self.steering_command > -6:
                        self.steering_command -= 2


            elif self.mode == "traffic_stop":
                self.left_speed_command = 0
                self.right_speed_command = 0
                self.steering_command = 0

                traffic_light_state = self.traffic_light_data.data.lower() if self.traffic_light_data and self.traffic_light_data.data else "unknown"
                if traffic_light_state == "green":
                    self.get_logger().info("[FSM] traffic_light GREEN → lane2_drive")
                    self.mode = "lane2_drive"


            # 로그 출력
            self.get_logger().info(f"[OBSTACLE FSM] mode={self.mode} lane={self.current_lane_class}")


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
