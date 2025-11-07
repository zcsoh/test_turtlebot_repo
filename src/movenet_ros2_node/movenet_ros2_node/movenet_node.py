import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
import os
import time
import ament_index_python.packages
import cv2
import numpy as np
from openvino.runtime import Core
from collections import namedtuple
import mediapipe as mp
import threading
import queue
import math

# --- MEDIAPIPE IMPORTS for Gesture Recognizer ---
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# ----------------------------------------------------

# --- LOCAL MODULE IMPORT (MOCK for standalone code) ---
try:
    from .tracker import TrackerIoU, TrackerOKS, TRACK_COLORS
except ImportError:
    print("Warning: Could not import tracker module. Using dummy classes/variables.")
    class DummyTracker:
        def __init__(self): pass
        def apply(self, bodies, timestamp): return bodies
    TrackerIoU = DummyTracker
    TrackerOKS = DummyTracker
    TRACK_COLORS = [(0, 255, 255), (255, 0, 0), (0, 0, 255), (255, 255, 0)] # Varied colors

# --- GLOBAL HELPER FUNCTION ---
def draw_outlined_text(image, text, org, font, font_scale, color, thickness):
    """Draws text with a black outline."""
    cv2.putText(image, text, org, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

# --- HELPER CLASSES AND DEFINITIONS ---

# Keypoint indices (MoveNet MultiPose)
KEYPOINT_DICT = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3,
    'right_ear': 4, 'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8, 'left_wrist': 9,
    'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15,
    'right_ankle': 16
}

# Lines to draw the skeleton
_POSE_LINES = [
    [4, 2], [2, 0], [0, 1], [1, 3],
    [10, 8], [8, 6], [6, 5], [5, 7], [7, 9],
    [6, 12], [12, 11], [11, 5],
    [12, 14], [14, 16], [11, 13], [13, 15]
]

# Padding (w, h: padding amount; padded_w, padded_h: new dimensions)
Padding = namedtuple('Padding', ['w', 'h', 'padded_w', 'padded_h'])

class Body:
    """Class to hold pose detection results, including bounding box, tracking info, and distance."""
    def __init__(self, score, xmin, ymin, xmax, ymax, keypoints_score, keypoints, keypoints_norm):
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.keypoints_score = keypoints_score
        self.keypoints = keypoints
        self.keypoints_norm = keypoints_norm
        self.track_id = -1
        self.distance_m = 0.0 # This will hold the "raw" distance for this frame
# =========================================================================
# 1. OpenVINO Inference and Processing Class
# =========================================================================
class MovenetProcessor:
    """Handles the OpenVINO MoveNet MultiPose inference and tracking."""
    def __init__(self, node_logger, model_xml, device='CPU', tracking_method=None):
        self.get_logger = node_logger

        try:
            self.ie = Core()
            device = 'CPU'
            self.model = self.ie.read_model(model=model_xml)
            self.compiled_model = self.ie.compile_model(model=self.model, device_name=device)
        except Exception as e:
            self.get_logger().error(f"Failed to initialize OpenVINO or load model: {e}")
            raise FileNotFoundError(f"Model file not accessible or OpenVINO error: {model_xml}")

        self.input_layer = self.compiled_model.input(0)
        _, _, H, W = self.input_layer.shape
        self.input_height, self.input_width = H, W
        self.output_layer = self.compiled_model.output(0)

        self.score_threshold = 0.15
        self.padding = None

        self.tracking = False
        if tracking_method in ["iou", "oks"]:
            self.tracking = True
            self.tracker = TrackerIoU() if tracking_method == "iou" else TrackerOKS()
            self.get_logger().info(f"Using {tracking_method.upper()} tracking.")
        else:
            self.get_logger().info("Tracking is disabled.")

    def calculate_padding(self, original_h, original_w):
        model_aspect = self.input_width / self.input_height
        frame_aspect = original_w / original_h

        if frame_aspect > model_aspect:
            pad_h = int(original_w / model_aspect - original_h)
            self.padding = Padding(0, pad_h, original_w, original_h + pad_h)
        else:
            pad_w = int(original_h * model_aspect - original_w)
            self.padding = Padding(pad_w, 0, original_w + pad_w, original_h)

    def pad_and_resize(self, frame):
        if self.padding is None:
            self.calculate_padding(frame.shape[0], frame.shape[1])

        padded = cv2.copyMakeBorder(frame, 0, self.padding.h, 0, self.padding.w, cv2.BORDER_CONSTANT)
        padded = cv2.resize(padded, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
        return padded

    def preprocess(self, padded_frame):
        frame_nn = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32)
        input_tensor = frame_nn[None,]
        return input_tensor

    def postprocess(self, output_data, original_h, original_w):
        output = output_data.squeeze()
        bodies = []
        for person_idx in range(output.shape[0]):
            person_score = output[person_idx, 55]
            if person_score < self.score_threshold:
                continue
            kps_yxs = output[person_idx, :51].reshape(17, -1)
            kp_scores = kps_yxs[:, 2]
            kp_yx_norm = kps_yxs[:, :2]
            bbox_norm_yx = output[person_idx, 51:55]
            bbox_scaled_yx = bbox_norm_yx * np.array([self.padding.padded_h, self.padding.padded_w, self.padding.padded_h, self.padding.padded_w])
            ymin, xmin, ymax, xmax = bbox_scaled_yx.astype(np.int32)
            xmin = np.clip(xmin, 0, original_w)
            ymin = np.clip(ymin, 0, original_h)
            xmax = np.clip(xmax, 0, original_w)
            ymax = np.clip(ymax, 0, original_h)
            kp_xy_norm = kp_yx_norm[:, [1, 0]]
            keypoints_px = kp_xy_norm * np.array([self.padding.padded_w, self.padding.padded_h])
            body = Body(
                score=person_score,
                xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
                keypoints_score=kp_scores,
                keypoints=keypoints_px.astype(np.int32),
                keypoints_norm=kp_xy_norm
            )
            bodies.append(body)
        return bodies

    def draw_poses(self, frame, bodies, active_target_id):
        DEFAULT_COLOR = (0, 255, 255) # Yellow (for untracked bodies)
        ACTIVE_TARGET_COLOR = (0, 255, 0) # Green (for the active target)

        for body in bodies:
            # 1. Determine the color for this body
            track_color = DEFAULT_COLOR 
            is_active_target = False

            if self.tracking and body.track_id != -1:
                # This is a tracked body.
                if body.track_id == active_target_id:
                    # It's the *active* target.
                    track_color = ACTIVE_TARGET_COLOR
                    is_active_target = True
                else:
                    # It's a *different* tracked body. Use the "various colors" logic.
                    track_color = TRACK_COLORS[body.track_id % len(TRACK_COLORS)]
            
            # 2. Draw skeleton (for everyone, with their specific color)
            for start_idx, end_idx in _POSE_LINES:
                start_pt_xy = body.keypoints[start_idx]
                end_pt_xy = body.keypoints[end_idx]
                start_score = body.keypoints_score[start_idx]
                end_score = body.keypoints_score[end_idx]
                if start_score > self.score_threshold and end_score > self.score_threshold:
                    p1 = (int(start_pt_xy[0]), int(start_pt_xy[1]))
                    p2 = (int(end_pt_xy[0]), int(end_pt_xy[1]))
                    cv2.line(frame, p1, p2, track_color, 2)

            for (x, y), score in zip(body.keypoints, body.keypoints_score):
                if score > self.score_threshold:
                    center = (int(x), int(y))
                    cv2.circle(frame, center, 4, track_color, -1)

            # 3. Draw Box and Label
            if self.tracking and body.track_id != -1:
                # This will draw a box for any person with a valid ID
                # It will be green for the active target and a varied color for others
                cv2.rectangle(frame, (body.xmin, body.ymin), (body.xmax, body.ymax), track_color, 2)
                id_text = f"ID: {body.track_id} ({body.score:.2f})"
                text_pt = (body.xmin, body.ymin - 10)
                draw_outlined_text(frame, id_text, text_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, track_color, 2)
        
        return frame

    def process_frame(self, frame):
        original_h, original_w = frame.shape[:2]
        padded_frame = self.pad_and_resize(frame.copy())
        input_tensor = self.preprocess(padded_frame)
        results = self.compiled_model(input_tensor)[self.output_layer]
        bodies = self.postprocess(results, original_h, original_w)
        if self.tracking:
            current_time_ms = int(time.time() * 1000)
            bodies = self.tracker.apply(bodies, current_time_ms)
        return frame.copy(), bodies


# =========================================================================
# 2. ROS 2 Node Class
# =========================================================================

class MovenetROS2Node(Node):
    """ROS2 Node for running MoveNet and MediaPipe hand detection with robot control."""

    def __init__(self):
        super().__init__('movenet_detector')
        self.get_logger().info("MoveNet Detector Node Initializing...")

        # --- Parameter and Model Setup ---
        self.declare_parameter('tracking_method', 'oks')
        self.declare_parameter('input_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw') 
        self.declare_parameter('camera_info_topic', '/camera/aligned_depth_to_color/camera_info')
        self.declare_parameter('output_topic', 'movenet/image_out')
        self.declare_parameter('diy_scan_topic', '/diy_scan') 
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')
        self.declare_parameter('reacquire_duration_per_step', 1.0) 
        self.declare_parameter('gesture_model_name', 'gesture_recognizer.task') 
        self.declare_parameter('odom_topic', '/odom')

        tracking_method = self.get_parameter('tracking_method').get_parameter_value().string_value
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value 
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.diy_scan_topic = self.get_parameter('diy_scan_topic').get_parameter_value().string_value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        self.reacquire_duration_per_step = self.get_parameter('reacquire_duration_per_step').get_parameter_value().double_value 
        gesture_model_name = self.get_parameter('gesture_model_name').get_parameter_value().string_value
        odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value

        try:
            package_share_directory = ament_index_python.packages.get_package_share_directory('movenet_ros2_node')
        except ament_index_python.packages.PackageNotFoundError:
            self.get_logger().warn("ROS package 'movenet_ros2_node' not found. Using local directory for models.")
            package_share_directory = os.path.dirname(os.path.abspath(__file__))

        # --- Model Paths ---
        model_name = 'movenet_multipose_lightning_256x256_FP32.xml'
        model_xml_path = os.path.join(package_share_directory, 'models', model_name)
        gesture_task_path = os.path.join(package_share_directory, 'models', gesture_model_name)
        if not os.path.exists(model_xml_path):
            self.get_logger().error(f"MoveNet Model file not found at: {model_xml_path}")
            pass
        if not os.path.exists(gesture_task_path):
            self.get_logger().error(f"MediaPipe Gesture Model not found at: {gesture_task_path}.")
            pass

        # --- OpenVINO Processor Setup ---
        self.processor = MovenetProcessor(
            node_logger=self.get_logger,
            model_xml=model_xml_path,
            device='CPU',
            tracking_method=tracking_method
        )
        self.bridge = CvBridge()

        # --- MediaPipe Hands Setup ---
        self.mp_hands = mp.solutions.hands
        self.hands_landmarker = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1,
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # --- MediaPipe Gesture Recognizer Setup ---
        BaseOptions = python.BaseOptions
        GestureRecognizer = vision.GestureRecognizer
        GestureRecognizerOptions = vision.GestureRecognizerOptions
        VisionRunningMode = vision.RunningMode
        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=gesture_task_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.1,
            min_hand_presence_confidence=0.1,
            min_tracking_confidence=0.1,
        )
        try:
            self.gesture_recognizer = GestureRecognizer.create_from_options(options)
            self.get_logger().info(f"MediaPipe Gesture Recognizer initialized with model: {gesture_model_name}")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize Gesture Recognizer: {e}")
            self.gesture_recognizer = None

        # --- Robot Control State & Data ---
        self.robot_state = "waiting_for_gesture" 
        self.active_gesture = "none"
        self.target_track_id = -1
        self.last_target_track_id = -1
        self.reacquire_step = 0 
        self.reacquire_start_time = 0.0
        self.reacquire_elapsed_in_step = 0.0 # For pausing/resuming
        self.verify_elapsed_in_step = 0.0 # For pausing/resuming
        self.pre_verify_reacquire_step = 0 
        self.reacquire_elapsed_time_before_verify = 0.0 
        self.last_target_direction = 'center'
        self.last_all_bodies = []
        self.last_seen_bbox = None 
        self.last_target_bbox = None 
        self.previous_robot_state = "waiting_for_gesture" 
        self.gesture_clear_start_time = 0.0
        self.GESTURE_CLEAR_HOLD_TIME = 0.5 
        self.thumbs_up_start_time = 0.0
        self.THUMBS_UP_HOLD_TIME = 0.00001
        self.iloveyou_start_time = 0.0
        self.ILOVEYOU_HOLD_TIME = 2.0 

        self.tracking_grace_period = False
        self.GRACE_PERIOD_ANGULAR_THRESHOLD = 0.1 # <-- Align within 10% of center
        
        self.SIZE_THRESHOLD_CLOSE = 0.80 
        self.SIZE_THRESHOLD_FAR = 0.50 
        self.MAX_EXPECTED_NORMALIZED_SIZE = 0.7
        self.OBSTACLE_THRESHOLD = 0.25 
        self.wait_for_clear_start_time = 0.0 
        self.WAIT_TIMEOUT = 10.0 
        self.verify_start_time = 0.0
        self.VERIFY_TIMEOUT = 10.0 
        
        self.last_target_position_angle_rad = 0.0
        self.last_target_position_range_m = 0.0
        
        self.last_known_target_position = None # Stores (X_robot, Y_robot) of target
        
        # --- ODOMETRY VARIABLES ---
        self.odom_lock = threading.Lock()
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0 # (in radians)
        self.world_frame_goal = None # (x, y) goal in the odom frame
        # --- END ODOMETRY ---

        self.obstacle_in_left = False
        self.obstacle_in_center = False
        self.obstacle_in_right = False
        self.SAFETY_DISTANCE = 0.3
        self.SLOW_DISTANCE = 1.0
        self.TARGET_DISTANCE_SETPOINT = 1.4
        self.KP_LINEAR_DEPTH = 0.8
        self.MAX_LINEAR_X = 0.45
        
        # --- Smoothing & Decay Parameters ---
        self.ANGULAR_SMOOTHING_FACTOR = 0.7
        self.LINEAR_SMOOTHING_FACTOR = 0.7
        self.DISTANCE_SMOOTHING_FACTOR = 0.6 
        self.ANGULAR_DECAY_FACTOR = 0.95
        self.LINEAR_DECAY_FACTOR = 0.95
        
        # --- Obstacle Debouncing Parameters ---
        self.OBSTACLE_CLEAR_FRAMES = 2
        self.obstacle_clear_counter_L = 0
        self.obstacle_clear_counter_C = 0
        self.obstacle_clear_counter_R = 0
        
        self.ANGULAR_GAIN_P = 1.5
        self.GESTURE_TURN_SPEED = 0.15
        self.AVOIDANCE_LINEAR_SPEED = 0.15
        self.AVOIDANCE_TURN_SPEED = 0.4
        
        # Shared data variables
        self.hip_midpoint_x = 0.5
        self.person_normalized_size = 0.0
        self.person_detected = False
        self.target_detected = False
        self.is_obstacle_close = False 
        self.target_distance_m = 0.0 # This will hold the SMOOTHED value

        # Locks for thread-safe access
        self.gesture_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.data_lock = threading.Lock()
        self.depth_lock = threading.Lock() 
        self.camera_info_lock = threading.Lock() 

        # --- Camera Intrinsics ---
        self.camera_info = None
        self.scan_config = {} 
        self.scan_bin_indices = None 

        # FPS Tracking
        self.last_time = time.time()
        self.current_fps = 0.0
        self.gesture_frame_skip = 0
        self.skip_frames_interval = 1

        # Timer for movement logic
        self.timer = self.create_timer(0.05, self.publish_cmd_vel)

        # Multithreading Setup
        self.image_queue = queue.Queue(maxsize=1)
        self.depth_queue = queue.Queue(maxsize=1) 
        self.processor_thread = threading.Thread(target=self.process_images_thread, daemon=True)
        self.processor_thread.start()

        # ROS Communication Setup
        qos_profile = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.subscription = self.create_subscription(Image, input_topic, self.image_callback, qos_profile)
        self.depth_subscription = self.create_subscription( 
            Image, 
            depth_topic, 
            self.depth_callback, 
            qos_profile
        )
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            qos_profile
        )

        # --- ODOMETRY SUBSCRIBER ---
        self.odom_subscription = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            qos_profile
        )
        # --- END SUBSCRIBER ---
        
        self.publisher_ = self.create_publisher(Image, output_topic, 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.scan_publisher_ = self.create_publisher(LaserScan, self.diy_scan_topic, 10)

        # =========================================================================
        # --- Bird's-Eye-View (BEV) Map Setup ---
        # =========================================================================
        # Real-world height of the camera sensor from the ground (in meters)
        self.CAMERA_HEIGHT_M = 0.3
        # Filter: Only map points between this height range (relative to ground)
        self.OBSTACLE_MIN_HEIGHT_M = 0.05 # 5cm (to ignore floor noise)
        self.OBSTACLE_MAX_HEIGHT_M = 1.0  # 1.0m (to ignore ceilings, etc.)
        
        # --- MAP DEFINITION ---
        self.MAP_SIZE_METERS = 10.0 # Map will be 10m x 10m
        self.MAP_RESOLUTION = 0.05  # 5cm per pixel
        self.MAP_PIXEL_SIZE = int(self.MAP_SIZE_METERS / self.MAP_RESOLUTION)
        self.MAP_CENTER_PIXEL = self.MAP_PIXEL_SIZE // 2
        
        # New publisher for our top-down map
        self.bev_map_publisher = self.create_publisher(Image, 'movenet/bev_map', 10)
        self.get_logger().info(f"BEV Map initialized ({self.MAP_PIXEL_SIZE}x{self.MAP_PIXEL_SIZE} @ {self.MAP_RESOLUTION*100:.1f} cm/px)")
        # =========================================================================


    def image_callback(self, msg):
        if not self.image_queue.empty():
            try:
                self.image_queue.get_nowait()
            except queue.Empty:
                pass
        self.image_queue.put(msg)


    def depth_callback(self, msg):
        if not self.depth_queue.empty():
            try:
                self.depth_queue.get_nowait()
            except queue.Empty:
                pass
        self.depth_queue.put(msg)


    def camera_info_callback(self, msg):
        """Stores the camera intrinsic parameters and pre-calculates scan bins."""
        with self.camera_info_lock:
            if self.camera_info is None:
                # Make sure the K matrix is valid before storing
                if msg.k[0] == 0.0 or msg.k[4] == 0.0:
                    self.get_logger().warn("Waiting for valid camera_info (K matrix is all zeros)...", throttle_duration_sec=2)
                    return

                self.camera_info = {
                    'fx': msg.k[0],
                    'fy': msg.k[4],
                    'cx': msg.k[2],
                    'cy': msg.k[5],
                    'width': msg.width,
                    'height': msg.height
                }
                
                SCAN_FOV_DEGREES = 70.0
                SCAN_BINS = 90
                
                u_coords = np.arange(self.camera_info['width'])
                x_over_z = (u_coords - self.camera_info['cx']) / self.camera_info['fx']
                
                self.scan_angles_rad = -np.arctan(x_over_z) 
                
                total_fov_rad = np.deg2rad(SCAN_FOV_DEGREES)
                min_angle_rad = -total_fov_rad / 2.0
                
                bin_indices = ((self.scan_angles_rad - min_angle_rad) / (total_fov_rad)) * (SCAN_BINS - 1)
                self.scan_bin_indices = np.round(bin_indices).astype(int)
                
                self.scan_config = {
                    'scan_bins': SCAN_BINS,
                    'total_fov_rad': total_fov_rad,
                    'min_angle_rad': min_angle_rad,
                    'angle_increment': total_fov_rad / (SCAN_BINS - 1),
                    'range_min': 0.1,
                    'range_max': 10.0
                }
                self.get_logger().info(f"Camera intrinsics received: fx={self.camera_info['fx']}, cx={self.camera_info['cx']}. DIY scan pre-calculated.")

        # Once we have valid info, we don't need this callback anymore.
        self.destroy_subscription(self.camera_info_subscription)

    # --- HELPER FUNCTION ---
    def quaternion_to_yaw(self, q):
        """Converts a geometry_msgs/Quaternion to a 2D yaw angle in radians."""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw
    # --- END HELPER ---

    # --- CALLBACK ---
    def odom_callback(self, msg):
        """Updates the robot's current position and orientation."""
        with self.odom_lock:
            self.robot_x = msg.pose.pose.position.x
            self.robot_y = msg.pose.pose.position.y
            self.robot_yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
    # --- END CALLBACK ---


    # =========================================================================
    # --- Bird's-Eye-View (BEV) Map Generation ---
    # =========================================================================
    def create_bev_map(self, cv_depth, camera_info, bodies, active_target_id, smoothed_target_distance):
        """
        Generates a 2D BEV map of the static environment AND all tracked people.
        """
        if cv_depth is None or camera_info is None:
            return None

        map_size = self.MAP_PIXEL_SIZE
        
        # 1. Create an empty black map
        bev_map = np.zeros((map_size, map_size, 3), dtype=np.uint8)
        
        # --- PART 1: Draw Static Environment (The white pixels) ---
        h, w = cv_depth.shape
        fx, fy = camera_info['fx'], camera_info['fy']
        cx, cy = camera_info['cx'], camera_info['cy']
        
        # Create coordinate grids
        U, V = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

        # De-project to 3D Camera frame
        d = cv_depth.astype(np.float32) / 1000.0
        Z_cam = d
        X_cam = (U - cx) * d / fx
        Y_cam = (V - cy) * d / fy
        
        # Transform to 3D Robot frame
        X_robot = Z_cam
        Y_robot = -X_cam
        Z_robot = self.CAMERA_HEIGHT_M - Y_cam # Height from ground

        # Filter for valid static obstacles
        valid_mask = (Z_cam > 0.1) & \
                     (Z_robot > self.OBSTACLE_MIN_HEIGHT_M) & \
                     (Z_robot < self.OBSTACLE_MAX_HEIGHT_M)
        
        X_map_points = X_robot[valid_mask]
        Y_map_points = Y_robot[valid_mask]
        
        # Convert to map pixels
        u_map = (-Y_map_points / self.MAP_RESOLUTION + self.MAP_CENTER_PIXEL).astype(np.int32)
        v_map = (-X_map_points / self.MAP_RESOLUTION + self.MAP_CENTER_PIXEL).astype(np.int32)
        
        valid_map_indices = (u_map >= 0) & (u_map < map_size) & (v_map >= 0) & (v_map < map_size)
        u_map_valid = u_map[valid_map_indices]
        v_map_valid = v_map[valid_map_indices]
        
        # Draw static obstacles (white)
        bev_map[v_map_valid, u_map_valid] = [255, 255, 255]

        # --- PART 2: Draw Detected People (NEW LKP LOGIC) ---
        left_hip_idx = KEYPOINT_DICT['left_hip']
        right_hip_idx = KEYPOINT_DICT['right_hip']
        
        target_seen_in_this_frame = False
        
        # --- First, find and process the ACTIVE target ---
        active_target_body = None
        if active_target_id != -1:
            for b in bodies:
                if b.track_id == active_target_id:
                    active_target_body = b
                    break
        
        if active_target_body:
            body = active_target_body
            try:
                if body.distance_m > 0 and \
                   (body.keypoints_score[left_hip_idx] > self.processor.score_threshold and
                    body.keypoints_score[right_hip_idx] > self.processor.score_threshold):
                    
                    # Use smoothed distance for the active target to reduce flicker
                    X_person = smoothed_target_distance if smoothed_target_distance > 0 else body.distance_m
                    
                    # Get person's Y (left/right) coordinate
                    hip_midpoint_x_norm = (body.keypoints_norm[left_hip_idx][0] + body.keypoints_norm[right_hip_idx][0]) / 2
                    u_pixel = hip_midpoint_x_norm * camera_info['width']
                    Y_person = -(u_pixel - camera_info['cx']) * X_person / camera_info['fx']
                    
                    # --- THIS IS THE LKP UPDATE ---
                    with self.data_lock:
                        self.last_known_target_position = (X_person, Y_person)
                    target_seen_in_this_frame = True
                    
                    # Convert person's (X, Y) to map pixels
                    u_map_person = int(-Y_person / self.MAP_RESOLUTION + self.MAP_CENTER_PIXEL)
                    v_map_person = int(-X_person / self.MAP_RESOLUTION + self.MAP_CENTER_PIXEL)
                    
                    if (0 <= u_map_person < map_size) and (0 <= v_map_person < map_size):
                        color = (0, 255, 0) # GREEN for active target
                        
                        # Draw the person's dot
                        cv2.circle(bev_map, (u_map_person, v_map_person), 4, color, -1)
                        
                        # Draw the label
                        id_text = f"ID: {body.track_id} (LIVE)"
                        text_pos = (u_map_person + 8, v_map_person + 8)
                        draw_outlined_text(bev_map, id_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            except Exception as e:
                self.get_logger().warn(f"Failed to draw ACTIVE target ID {active_target_id} on BEV map: {e}", throttle_duration_sec=5)
        
        # --- If active target NOT seen, use LKP memory ---
        if not target_seen_in_this_frame and self.last_known_target_position is not None and active_target_id != -1:
            try:
                # Get (X, Y) from memory
                with self.data_lock:
                    X_person, Y_person = self.last_known_target_position
                
                # Convert to map pixels
                u_map_person = int(-Y_person / self.MAP_RESOLUTION + self.MAP_CENTER_PIXEL)
                v_map_person = int(-X_person / self.MAP_RESOLUTION + self.MAP_CENTER_PIXEL)
                
                if (0 <= u_map_person < map_size) and (0 <= v_map_person < map_size):
                    # --- Draw a "ghost" dot ---
                    color = (0, 255, 255) # YELLOW
                    cv2.circle(bev_map, (u_map_person, v_map_person), 4, color, -1)
                    
                    # Draw the label
                    id_text = f"ID: {active_target_id} (LKP)"
                    text_pos = (u_map_person + 8, v_map_person + 8)
                    draw_outlined_text(bev_map, id_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            except Exception as e:
                self.get_logger().warn(f"Failed to draw LKP for target {active_target_id} on BEV map: {e}", throttle_duration_sec=5)

        # --- Now, draw all OTHER bodies (as yellow) ---
        for body in bodies:
            # Skip the active target (we already drew it or its LKP)
            if body.track_id != -1 and body.track_id == active_target_id:
                continue
                
            try:
                if body.distance_m > 0 and \
                   (body.keypoints_score[left_hip_idx] > self.processor.score_threshold and
                    body.keypoints_score[right_hip_idx] > self.processor.score_threshold):
                    
                    X_person = body.distance_m
                    
                    hip_midpoint_x_norm = (body.keypoints_norm[left_hip_idx][0] + body.keypoints_norm[right_hip_idx][0]) / 2
                    u_pixel = hip_midpoint_x_norm * camera_info['width']
                    Y_person = -(u_pixel - camera_info['cx']) * X_person / camera_info['fx']
                    
                    u_map_person = int(-Y_person / self.MAP_RESOLUTION + self.MAP_CENTER_PIXEL)
                    v_map_person = int(-X_person / self.MAP_RESOLUTION + self.MAP_CENTER_PIXEL)
                    
                    if (0 <= u_map_person < map_size) and (0 <= v_map_person < map_size):
                        color = (0, 255, 255) # YELLOW for other
                        cv2.circle(bev_map, (u_map_person, v_map_person), 4, color, -1)
                        if body.track_id != -1:
                           id_text = f"ID: {body.track_id}"
                           text_pos = (u_map_person + 8, v_map_person + 8)
                           draw_outlined_text(bev_map, id_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            except Exception as e:
                self.get_logger().warn(f"Failed to draw OTHER body ID {body.track_id} on BEV map: {e}", throttle_duration_sec=5)


        # --- PART 3: Draw Robot (on top of everything) ---
        cv2.circle(bev_map, (self.MAP_CENTER_PIXEL, self.MAP_CENTER_PIXEL), 5, (0, 0, 255), -1) # Red
        cv2.line(bev_map, (self.MAP_CENTER_PIXEL, self.MAP_CENTER_PIXEL), 
                         (self.MAP_CENTER_PIXEL, self.MAP_CENTER_PIXEL - 10), (0, 0, 255), 2)

        return bev_map

    # =========================================================================
    # --- process_images_thread function ---
    # =========================================================================
    def process_images_thread(self,):
        while rclpy.ok():
            current_time = time.time()
            try:
                msg = self.image_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # --- Get Depth Image ---
            depth_msg = None
            try:
                depth_msg = self.depth_queue.get_nowait()
            except queue.Empty:
                pass
            
            cv_depth = None
            if depth_msg:
                try:
                    cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
                except CvBridgeError as e:
                    self.get_logger().error(f'[Thread] CvBridge depth error: {e}')
                finally:
                    try:
                        self.depth_queue.task_done()
                    except ValueError:
                        pass 

            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except CvBridgeError as e:
                self.get_logger().error(f'[Thread] CvBridge error: {e}')
                self.image_queue.task_done()
                continue
            
            annotated_image, bodies = self.processor.process_frame(cv_image) 
            h, w = cv_image.shape[:2]

            # --- ZONED OBSTACLE & DISTANCE CHECK ---
            
            # 1. --- Zoned Obstacle Check (DIY LaserScan from Depth) ---
            obstacle_in_left_frame = False
            obstacle_in_center_frame = False
            obstacle_in_right_frame = False
            is_obstacle_close_in_frame = False
            scan_bins = None 
            scan_intensities = None

            with self.camera_info_lock:
                camera_info_data = self.camera_info
                bin_indices = self.scan_bin_indices
                scan_config = self.scan_config

            if cv_depth is not None and camera_info_data is not None and bin_indices is not None:
                SCAN_BINS = scan_config['scan_bins']
                SCAN_THRESHOLD_M = self.SLOW_DISTANCE
                
                scan_bins = np.full(SCAN_BINS, np.inf)
                
                self.INTENSITY_DEFAULT = 0.1
                self.INTENSITY_OTHER_HUMAN = 50.0
                self.INTENSITY_TARGET = 100.0
                
                scan_intensities = np.full(SCAN_BINS, self.INTENSITY_DEFAULT)
                
                # --- Use the robust band scan from the last attempt ---
                scan_height_pixels = 10
                scan_row_center = int(camera_info_data['cy'])
                scan_row_start = max(0, scan_row_center - (scan_height_pixels // 2))
                scan_row_end = min(camera_info_data['height'], scan_row_center + (scan_height_pixels // 2))
                
                depth_band_mm = cv_depth[scan_row_start:scan_row_end, :]
                # Use a copy to avoid modifying the original cv_depth which BEV map needs
                depth_band_mm_safe = depth_band_mm.copy() 
                depth_band_mm_safe[depth_band_mm_safe == 0] = 99999
                depth_row_mm = np.min(depth_band_mm_safe, axis=0) 
                depth_row_mm[depth_row_mm == 99999] = 0
                
                valid_mask = (depth_row_mm > 0) & \
                             (depth_row_mm < (scan_config['range_max'] * 1000.0)) & \
                             (bin_indices >= 0) & \
                             (bin_indices < SCAN_BINS)
                
                valid_bins = bin_indices[valid_mask]
                valid_depths_m = depth_row_mm[valid_mask] / 1000.0
                np.minimum.at(scan_bins, valid_bins, valid_depths_m)
                scan_bins[scan_bins < scan_config['range_min']] = np.inf 
                
                center_start_bin = SCAN_BINS // 3
                center_end_bin = 2 * (SCAN_BINS // 3)
                
                right_zone_scan = scan_bins[:center_start_bin]
                center_zone_scan = scan_bins[center_start_bin:center_end_bin]
                left_zone_scan = scan_bins[center_end_bin:]
                
                if np.any(center_zone_scan < SCAN_THRESHOLD_M):
                    obstacle_in_center_frame = True
                if np.any(left_zone_scan < SCAN_THRESHOLD_M):
                    obstacle_in_left_frame = True
                if np.any(right_zone_scan < SCAN_THRESHOLD_M):
                    obstacle_in_right_frame = True
                    
                is_obstacle_close_in_frame = obstacle_in_left_frame or \
                                             obstacle_in_center_frame or \
                                             obstacle_in_right_frame
            elif camera_info_data is None:
                self.get_logger().warn("Waiting for CameraInfo to perform obstacle/BEV check...", throttle_duration_sec=5)

            # 2. Calculate distance for every detected person (Body.distance_m)
            target_distance_m_in_frame = 0.0
            left_hip_idx = KEYPOINT_DICT['left_hip']
            right_hip_idx = KEYPOINT_DICT['right_hip']
            left_shoulder_idx = KEYPOINT_DICT['left_shoulder']
            right_shoulder_idx = KEYPOINT_DICT['right_shoulder']
            nose_idx = KEYPOINT_DICT['nose'] 

            if cv_depth is not None:
                depth_h, depth_w = cv_depth.shape[:2]
                
                for body in bodies:
                    body.distance_m = 0.0
                    if (body.keypoints_score[left_hip_idx] > self.processor.score_threshold and
                        body.keypoints_score[right_hip_idx] > self.processor.score_threshold):
                        
                        hip_midpoint_x_norm = (body.keypoints_norm[left_hip_idx][0] + body.keypoints_norm[right_hip_idx][0]) / 2
                        hip_midpoint_y_norm = (body.keypoints_norm[left_hip_idx][1] + body.keypoints_norm[right_hip_idx][1]) / 2
                        hip_px_x = int(hip_midpoint_x_norm * depth_w)
                        hip_px_y = int(hip_midpoint_y_norm * depth_h)
                        
                        roi_size = 20 
                        half_roi = roi_size // 2

                        y_min_roi = max(0, hip_px_y - half_roi)
                        y_max_roi = min(depth_h - 1, hip_px_y + half_roi)
                        x_min_roi = max(0, hip_px_x - half_roi)
                        x_max_roi = min(depth_w - 1, hip_px_x + half_roi)

                        if y_min_roi < y_max_roi and x_min_roi < x_max_roi:
                            depth_roi = cv_depth[y_min_roi:y_max_roi, x_min_roi:x_max_roi]
                            valid_depths = depth_roi[depth_roi > 0]
                            
                            if valid_depths.size > 0:
                                raw_depth_mm = np.percentile(valid_depths, 25) # Using 25th percentile
                                MAX_VALID_DEPTH_MM = 5000.0
                                if raw_depth_mm < MAX_VALID_DEPTH_MM:
                                    distance_m = raw_depth_mm / 1000.0
                                    body.distance_m = distance_m
            
            with self.data_lock:
                if obstacle_in_left_frame:
                    self.obstacle_in_left = True
                    self.obstacle_clear_counter_L = 0
                else:
                    self.obstacle_clear_counter_L += 1
                    if self.obstacle_clear_counter_L > self.OBSTACLE_CLEAR_FRAMES:
                        self.obstacle_in_left = False

                if obstacle_in_center_frame:
                    self.obstacle_in_center = True
                    self.obstacle_clear_counter_C = 0
                else:
                    self.obstacle_clear_counter_C += 1
                    if self.obstacle_clear_counter_C > self.OBSTACLE_CLEAR_FRAMES:
                        self.obstacle_in_center = False
                        
                if obstacle_in_right_frame:
                    self.obstacle_in_right = True
                    self.obstacle_clear_counter_R = 0
                else:
                    self.obstacle_clear_counter_R += 1
                    if self.obstacle_clear_counter_R > self.OBSTACLE_CLEAR_FRAMES:
                        self.obstacle_in_right = False
                        
                self.is_obstacle_close = self.obstacle_in_left or \
                                         self.obstacle_in_center or \
                                         self.obstacle_in_right
                
                self.last_all_bodies = bodies

            # --- State Machine: Identify Target ---
            with self.state_lock:
                robot_state = self.robot_state
            with self.data_lock:
                target_track_id = self.target_track_id
                last_target_track_id = self.last_target_track_id 
            person_detected_in_frame = len(bodies) > 0
            target_body = None
            id_to_check = -1
            
            if robot_state in ["tracking", "open_palm_stop", "gesture_turn_left", "gesture_turn_right"]:
                id_to_check = target_track_id
            elif robot_state in ["reacquire_target", "reacquire_wait_for_clear", "reacquire_verify_target"]: 
                id_to_check = last_target_track_id
            elif robot_state == "waiting_for_gesture" and person_detected_in_frame:
                pass
            
            target_detected_in_frame = False
            if id_to_check != -1: 
                target_bodies = [b for b in bodies if b.track_id == id_to_check]
                if target_bodies:
                    target_body = target_bodies[0]
                    target_detected_in_frame = True
            elif robot_state == "waiting_for_gesture" and person_detected_in_frame:
                # If waiting, just pick the highest-score body to show gestures for
                target_body = max(bodies, key=lambda b: b.score)
            
            
            # --- Publish the DIY Scan (with Target "Painted" on) ---
            if scan_bins is not None:
                
                active_target_id = id_to_check

                for body in bodies:
                    if (body.keypoints_score[left_hip_idx] > self.processor.score_threshold and
                        body.keypoints_score[right_hip_idx] > self.processor.score_threshold):
                        
                        hip_x_norm = (body.keypoints_norm[left_hip_idx][0] + 
                                      body.keypoints_norm[right_hip_idx][0]) / 2
                        u_pixel = int(hip_x_norm * camera_info_data['width'])
                        
                        if 0 <= u_pixel < camera_info_data['width']:
                            person_bin = bin_indices[u_pixel]
                            
                            if 0 <= person_bin < scan_config['scan_bins']:
                                paint_intensity = 0.0
                                
                                if body.track_id != -1 and body.track_id == active_target_id:
                                    paint_intensity = self.INTENSITY_TARGET

                                    # --- Store Last Known Position ---
                                    target_range_at_bin = scan_bins[person_bin]
                                    if target_range_at_bin != np.inf:
                                        with self.data_lock:
                                            self.last_target_position_range_m = target_range_at_bin
                                            self.last_target_position_angle_rad = scan_config['min_angle_rad'] + person_bin * scan_config['angle_increment']
                                    # --- END ---

                                else:
                                    paint_intensity = self.INTENSITY_OTHER_HUMAN

                                if paint_intensity > self.INTENSITY_DEFAULT:
                                    bin_width = 2
                                    min_bin = max(0, person_bin - bin_width)
                                    max_bin = min(scan_config['scan_bins'] - 1, person_bin + bin_width)

                                    person_bins_idx = np.arange(min_bin, max_bin + 1)
                                    valid_range_mask = (scan_bins[person_bins_idx] != np.inf)
                                    bins_to_paint = person_bins_idx[valid_range_mask]
                                    
                                    if bins_to_paint.size > 0:
                                        np.maximum.at(scan_intensities, bins_to_paint, paint_intensity)
                
                # Publish the message
                scan_msg = LaserScan()
                scan_msg.header.stamp = msg.header.stamp 
                scan_msg.header.frame_id = "camera_link"
                scan_msg.angle_min = scan_config['min_angle_rad']
                scan_msg.angle_max = scan_config['min_angle_rad'] + scan_config['total_fov_rad']
                scan_msg.angle_increment = scan_config['angle_increment']
                scan_msg.time_increment = 0.0
                scan_msg.scan_time = 0.0
                scan_msg.range_min = scan_config['range_min']
                scan_msg.range_max = scan_config['range_max']
                scan_msg.ranges = scan_bins.tolist()
                scan_msg.intensities = scan_intensities.tolist()
                
                self.scan_publisher_.publish(scan_msg)

            
            # --- State Machine and Data Update Logic ---
            if target_body:
                # --- 1. We have a valid body to look at ---
                with self.data_lock:
                    self.last_seen_bbox = (target_body.xmin, target_body.ymin, target_body.xmax, target_body.ymax)
                
                if robot_state not in ["reacquire_target", "reacquire_wait_for_clear", "open_palm_stop", "reacquire_verify_target", "gesture_turn_left", "gesture_turn_right"]:
                    with self.data_lock:
                        self.last_target_bbox = None
                
                nose_conf_ok = target_body.keypoints_score[nose_idx] > self.processor.score_threshold
                hip_conf_ok = (target_body.keypoints_score[left_hip_idx] > self.processor.score_threshold and
                                   target_body.keypoints_score[right_hip_idx] > self.processor.score_threshold)
                
                if target_body.score > 0.1 and nose_conf_ok and hip_conf_ok:
                    hip_midpoint_x_norm = (target_body.keypoints_norm[left_hip_idx][0] + target_body.keypoints_norm[right_hip_idx][0]) / 2
                    hip_y_norm = (target_body.keypoints_norm[left_hip_idx][1] + target_body.keypoints_norm[right_hip_idx][1]) / 2
                    nose_y_norm = target_body.keypoints_norm[nose_idx][1]
                    raw_vertical_size = hip_y_norm - nose_y_norm
                    person_normalized_size = max(0.0, min(1.0, raw_vertical_size / self.MAX_EXPECTED_NORMALIZED_SIZE))
                    with self.data_lock:
                        self.person_detected = person_detected_in_frame
                        current_hip_x = self.hip_midpoint_x
                        smoothed_hip_x = (current_hip_x * self.ANGULAR_SMOOTHING_FACTOR) + (hip_midpoint_x_norm * (1.0 - self.ANGULAR_SMOOTHING_FACTOR))
                        self.hip_midpoint_x = smoothed_hip_x
                        current_size = self.person_normalized_size
                        smoothed_size = (current_size * self.LINEAR_SMOOTHING_FACTOR) + (person_normalized_size * (1.0 - self.LINEAR_SMOOTHING_FACTOR))
                        self.person_normalized_size = smoothed_size
                        # Store the *direction* as well as the angle
                        if hip_midpoint_x_norm < 0.4: self.last_target_direction = 'left'
                        elif hip_midpoint_x_norm > 0.6: self.last_target_direction = 'right'
                        else: self.last_target_direction = 'center'
                else:
                    with self.data_lock:
                        self.person_detected = person_detected_in_frame
                        current_hip_x = self.hip_midpoint_x
                        self.hip_midpoint_x = (current_hip_x * self.ANGULAR_DECAY_FACTOR) + (0.5 * (1.0 - self.ANGULAR_DECAY_FACTOR))
                        current_size = self.person_normalized_size
                        self.person_normalized_size = current_size * self.LINEAR_DECAY_FACTOR
                
                target_distance_m_in_frame = target_body.distance_m
                
                with self.data_lock:
                    self.target_detected = True
                
                if robot_state in ["reacquire_target", "reacquire_wait_for_clear"] and target_detected_in_frame:
                    self.get_logger().info(f"Target ID {last_target_track_id} RE-ACQUIRED. Entering VERIFY state.")
                    self.reacquire_elapsed_in_step = 0.0 # RESET pause timer
                    if robot_state == "reacquire_target" and self.reacquire_step > 0:
                        time_elapsed_in_step = current_time - self.reacquire_start_time
                        self.reacquire_elapsed_time_before_verify = time_elapsed_in_step
                    else:
                        self.reacquire_elapsed_time_before_verify = 0.0
                    self.pre_verify_reacquire_step = self.reacquire_step 
                    with self.state_lock:
                        self.robot_state = "reacquire_verify_target"
                        robot_state = "reacquire_verify_target" 
                    with self.data_lock:
                        self.last_target_bbox = None 
                        self.world_frame_goal = None 
                    self.verify_start_time = current_time 
                    self.verify_elapsed_in_step = 0.0 # RESET pause timer
                    self.thumbs_up_start_time = 0.0 
                    self.iloveyou_start_time = 0.0 
                
            else:
                # --- 2. We do NOT see a valid target body ---
                
                with self.data_lock:
                    self.person_detected = person_detected_in_frame
                    current_hip_x = self.hip_midpoint_x
                    self.hip_midpoint_x = (current_hip_x * self.ANGULAR_DECAY_FACTOR) + (0.5 * (1.0 - self.ANGULAR_DECAY_FACTOR))
                    current_size = self.person_normalized_size
                    self.person_normalized_size = current_size * self.LINEAR_DECAY_FACTOR
                
                with self.data_lock:
                    self.target_detected = False
                
                if robot_state == "tracking" and not target_detected_in_frame:
                    self.get_logger().warn(f"Target ID {target_track_id} lost. Entering REACQUIRE state.")
                    
                    # --- *** CALCULATE WORLD GOAL *** ---
                    # We store the goal *before* clearing the target ID
                    with self.data_lock:
                        # Get the ghost's (X_robot, Y_robot) from self.last_known_target_position
                        if self.last_known_target_position:
                            X_robot, Y_robot = self.last_known_target_position
                            
                            with self.odom_lock:
                                rx, ry, ryaw = self.robot_x, self.robot_y, self.robot_yaw
                            
                            # Convert robot-centric (X-fwd, Y-left) to world-centric (X, Y)
                            # This is a 2D rotation
                            world_goal_x = rx + X_robot * math.cos(ryaw) - Y_robot * math.sin(ryaw)
                            world_goal_y = ry + X_robot * math.sin(ryaw) + Y_robot * math.cos(ryaw)

                            self.world_frame_goal = (world_goal_x, world_goal_y)
                            self.get_logger().info(f"Target lost! Storing world goal at: ({world_goal_x:.2f}, {world_goal_y:.2f})")
                        else:
                            self.world_frame_goal = None # No LKP, can't set a goal
                            self.get_logger().warn("Target lost but no LKP was ever stored. Cannot set world goal.")
                    
                    with self.data_lock:
                        self.last_target_bbox = self.last_seen_bbox 
                        self.target_track_id = -1
                        self.last_target_track_id = target_track_id 
                        self.tracking_grace_period = False
                    with self.state_lock:
                        self.robot_state = "reacquire_target" # <-- BACK TO reacquire_target
                        robot_state = "reacquire_target"
                    
                    self.reacquire_start_time = time.time() 
                    self.reacquire_step = 1 # <-- Start at Step 1 (Face Goal)
                    self.reacquire_elapsed_in_step = 0.0 # RESET pause timer
                    self.iloveyou_start_time = 0.0 
                    self.verify_start_time = 0.0 
                
                elif robot_state == "reacquire_verify_target" and not target_detected_in_frame:
                    self.get_logger().warn(f"Target {last_target_track_id} lost during VERIFY. Returning to REACQUIRE.")
                    
                    with self.state_lock:
                        # We return to the *same* step we were on before verifying
                        self.robot_state = "reacquire_target"
                        robot_state = "reacquire_target" # Update local copy for this frame
                        self.reacquire_step = self.pre_verify_reacquire_step 

                    # Resume the reacquire timer from where it left off
                    self.reacquire_start_time = current_time 
                    # We use the 'reacquire_elapsed_time_before_verify' that we saved
                    self.reacquire_elapsed_in_step = self.reacquire_elapsed_time_before_verify

                    # Reset verification variables
                    self.verify_start_time = 0.0
                    self.verify_elapsed_in_step = 0.0
            

            # --- Apply temporal smoothing to the distance ---
            with self.depth_lock:
                current_smoothed_dist = self.target_distance_m
                
                if target_distance_m_in_frame > 0.0:
                    self.target_distance_m = (current_smoothed_dist * self.DISTANCE_SMOOTHING_FACTOR) + \
                                             (target_distance_m_in_frame * (1.0 - self.DISTANCE_SMOOTHING_FACTOR))
                elif self.target_detected:
                    pass 
                else:
                    self.target_distance_m = current_smoothed_dist * self.LINEAR_DECAY_FACTOR

            # --- Gesture detection ---
            run_gesture_recognition = (self.gesture_frame_skip % self.skip_frames_interval == 0)
            self.gesture_frame_skip += 1
            image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB) 
            current_gesture = "none"
            thumbs_up_active = False
            iloveyou_active = False
            if self.gesture_recognizer is not None and run_gesture_recognition:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                gesture_result = self.gesture_recognizer.recognize(mp_image)
                results_hands_draw = self.hands_landmarker.process(image_rgb) 
                if results_hands_draw.multi_hand_landmarks and target_body and gesture_result.hand_landmarks:
                    target_xmin, target_ymin = target_body.xmin, target_body.ymin
                    target_xmax, target_ymax = target_body.xmax, target_body.ymax
                    for idx, hand_landmarks in enumerate(results_hands_draw.multi_hand_landmarks):
                        if idx < len(gesture_result.gestures) and gesture_result.gestures[idx]:
                            top_gesture = gesture_result.gestures[idx][0]
                            gesture_name = top_gesture.category_name
                            gesture_score = top_gesture.score
                            wrist_norm_x = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x
                            wrist_norm_y = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y
                            wrist_px_x, wrist_px_y = int(wrist_norm_x * w), int(wrist_norm_y * h)
                            is_hand_on_target = (target_xmin <= wrist_px_x <= target_xmax and
                                                 target_ymin <= wrist_px_y <= target_ymax)
                            if is_hand_on_target and gesture_score > 0.1:
                                if gesture_name == "Thumb_Up":
                                    thumbs_up_active = True
                                    current_gesture = gesture_name
                                elif gesture_name == "ILoveYou": 
                                    iloveyou_active = True
                                    current_gesture = gesture_name
                                elif gesture_name == "Open_Palm":
                                    if robot_state in ["tracking", "reacquire_target", "reacquire_wait_for_clear", "reacquire_verify_target", "open_palm_stop", "gesture_turn_left", "gesture_turn_right"]:
                                        current_gesture = "Open_Palm"
                                elif robot_state in ["tracking", "open_palm_stop", "gesture_turn_left", "gesture_turn_right"]:
                                    if gesture_name == "Pointing_Up":
                                        current_gesture = "Pointing_Up"
                                    elif gesture_name == "Victory":
                                        current_gesture = "Victory"
                                self.mp_drawing.draw_landmarks(
                                    image_rgb, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                                    self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2),
                                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                                )
                                display_gesture_name = current_gesture.upper() if current_gesture != "none" else gesture_name.upper()
                                gesture_text = f"{display_gesture_name} ({gesture_score:.2f})"
                                text_org = (wrist_px_x, wrist_px_y - 20)
                                draw_outlined_text(image_rgb, gesture_text, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                                break 
            with self.gesture_lock:
                self.active_gesture = current_gesture
            
            # --- Gesture debounce logic ---
            if target_body and (target_body.track_id != -1 or robot_state == "waiting_for_gesture"): 
                if robot_state == "waiting_for_gesture" or robot_state == "reacquire_verify_target":
                    target_hold_time = self.THUMBS_UP_HOLD_TIME
                    if thumbs_up_active:
                        if self.thumbs_up_start_time == 0.0:
                            self.thumbs_up_start_time = current_time
                        elif (current_time - self.thumbs_up_start_time) >= target_hold_time:
                            with self.data_lock:
                                self.target_track_id = target_body.track_id
                                self.last_target_track_id = target_body.track_id
                                self.tracking_grace_period = True # <-- Enable grace period
                                self.world_frame_goal = None # <-- Clear goal, we are tracking
                            with self.state_lock:
                                self.robot_state = "tracking"
                                robot_state = "tracking"
                            
                            self.get_logger().info(f"Tracking ID {target_body.track_id} acquired. Entering alignment grace period.") # <-- Modified log
                            
                            self.thumbs_up_start_time = 0.0
                            self.verify_start_time = 0.0
                            self.verify_elapsed_in_step = 0.0 # RESET pause timer
                    elif not thumbs_up_active and self.thumbs_up_start_time != 0.0:
                        self.thumbs_up_start_time = 0.0
                        if robot_state == "reacquire_verify_target":
                            self.verify_start_time = current_time # Resume timer
                elif robot_state in ["tracking", "reacquire_target", "reacquire_wait_for_clear", "reacquire_verify_target", "open_palm_stop", "gesture_turn_left", "gesture_turn_right"]:
                    if iloveyou_active:
                        if self.iloveyou_start_time == 0.0:
                            self.iloveyou_start_time = current_time
                        elif (current_time - self.iloveyou_start_time) >= self.ILOVEYOU_HOLD_TIME:
                            with self.data_lock:
                                self.target_track_id = -1
                                self.last_target_track_id = -1
                                self.last_target_bbox = None
                                self.tracking_grace_period = False
                                self.last_known_target_position = None # Clear LKP
                                self.world_frame_goal = None # <-- Clear world goal
                            with self.state_lock:
                                self.robot_state = "waiting_for_gesture"
                                robot_state = "waiting_for_gesture"
                            self.iloveyou_start_time = 0.0 
                            self.verify_start_time = 0.0
                            self.verify_elapsed_in_step = 0.0 # RESET pause timer
                    elif not iloveyou_active and self.iloveyou_start_time != 0.0:
                        self.iloveyou_start_time = 0.0
            if robot_state != "waiting_for_gesture" and robot_state != "reacquire_verify_target":
                self.thumbs_up_start_time = 0.0
            if robot_state not in ["tracking", "reacquire_target", "reacquire_wait_for_clear", "reacquire_verify_target", "open_palm_stop", "gesture_turn_left", "gesture_turn_right"]:
                self.iloveyou_start_time = 0.0

            # =========================================================================
            # --- Generate and Publish BEV Map ---
            # =========================================================================
            
            # Get the final smoothed distance for the target
            with self.depth_lock:
                smoothed_dist = self.target_distance_m

            # Create the map, passing all bodies, the active ID, and the smoothed distance
            bev_map_image = self.create_bev_map(cv_depth, camera_info_data, bodies, id_to_check, smoothed_dist)
            
            if bev_map_image is not None:
                try:
                    # We use the original color image's timestamp
                    map_msg = self.bridge.cv2_to_imgmsg(bev_map_image, "bgr8")
                    map_msg.header = msg.header
                    map_msg.header.frame_id = "base_link" # Set frame for RViz
                    self.bev_map_publisher.publish(map_msg)
                except CvBridgeError as e:
                    self.get_logger().error(f'[Thread] CvBridge error on BEV map publish: {e}')
            # =========================================================================


            # --- Draw MoveNet Poses and UI for Display ---
            final_annotated_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            final_annotated_image = self.processor.draw_poses(final_annotated_image, bodies, id_to_check) 
            
            for body in bodies:
                if body.distance_m > 0.0:
                    hip_midpoint_px = (body.keypoints[left_hip_idx] + body.keypoints[right_hip_idx]) // 2
                    distance_text = f"{body.distance_m:.2f}m"
                    text_pt = (hip_midpoint_px[0] - 50, hip_midpoint_px[1] + 40)
                    draw_color = (0, 0, 255) if body.distance_m < self.SLOW_DISTANCE else (0, 255, 255)
                    # Only draw distance text if not the active target (to avoid clutter)
                    if not (body.track_id != -1 and body.track_id == id_to_check):
                        draw_outlined_text(final_annotated_image, distance_text, text_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color, 2)
            
            with self.state_lock:
                current_state_for_drawing = self.robot_state
            with self.data_lock:
                bbox_to_draw = self.last_target_bbox 
            if current_state_for_drawing in ["reacquire_target", "reacquire_wait_for_clear", "reacquire_verify_target"] and bbox_to_draw is not None:
                xmin, ymin, xmax, ymax = bbox_to_draw
                draw_color = (0, 165, 255) if current_state_for_drawing == "reacquire_verify_target" else (0, 0, 255)
                cv2.rectangle(final_annotated_image, (xmin, ymin), (xmax, ymax), draw_color, 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                thickness = 2
                draw_outlined_text(final_annotated_image, "LAST SEEN", (xmin, ymin - 10), font, 0.6, draw_color, thickness)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            current_time_frame = time.time()
            if (current_time_frame - self.last_time) > 0:
                self.current_fps = 1.0 / (current_time_frame - self.last_time)
            self.last_time = current_time_frame
            draw_outlined_text(final_annotated_image, f"FPS: {self.current_fps:.1f}", (10, 30), font, font_scale, (0, 255, 0), thickness)
            draw_outlined_text(final_annotated_image, f"People Detected: {len(bodies)}", (10, 60), font, font_scale, (255, 255, 0), thickness)
            
            robot_state_display = current_state_for_drawing.upper()
            state_color = (0, 255, 255) if current_state_for_drawing == "tracking" else (0, 165, 255)
            progress_bar_active = False
            
            if current_state_for_drawing == "reacquire_target":
                robot_state_display += f" (Step {self.reacquire_step})"
                state_color = (0, 165, 255)
            elif current_state_for_drawing == "reacquire_wait_for_clear":
                elapsed_wait = current_time - self.wait_for_clear_start_time
                robot_state_display = f"OBSTACLE: {max(0.0, self.WAIT_TIMEOUT - elapsed_wait):.1f}s"
                state_color = (0, 0, 255) 
            elif current_state_for_drawing == "reacquire_verify_target":
                elapsed_verify = current_time - self.verify_start_time
                total_elapsed_verify = self.verify_elapsed_in_step + elapsed_verify
                time_remaining_verify = self.VERIFY_TIMEOUT - total_elapsed_verify
                robot_state_display = f"VERIFY: {max(0.0, time_remaining_verify):.1f}s"
                state_color = (255, 165, 0) 
            elif current_state_for_drawing == "open_palm_stop": 
                robot_state_display = "HALTED (OPEN PALM)"
                state_color = (0, 255, 255)
            elif current_state_for_drawing == "gesture_turn_left": 
                robot_state_display = "GESTURE TURN LEFT"
                state_color = (255, 255, 0)
            elif current_state_for_drawing == "gesture_turn_right": 
                robot_state_display = "GESTURE TURN RIGHT"
                state_color = (255, 255, 0)
            
            if self.thumbs_up_start_time != 0.0:
                progress_bar_active = True
                progress_time = current_time - self.thumbs_up_start_time
                hold_target_time = self.THUMBS_UP_HOLD_TIME
                progress_label = "VERIFY/ACQUIRE IN"
                if current_state_for_drawing == "reacquire_verify_target":
                    robot_state_display = "VERIFYING (HOLD)"
                    state_color = (255, 0, 255) 
                elif current_state_for_drawing == "waiting_for_gesture":
                    robot_state_display = "WAITING (HOLD)"
                    state_color = (0, 255, 0)
            elif self.iloveyou_start_time != 0.0:
                progress_bar_active = True
                progress_time = current_time - self.iloveyou_start_time
                hold_target_time = self.ILOVEYOU_HOLD_TIME
                progress_label = "STOPPING IN"
                robot_state_display = "TRACKING (HOLD STOP)"
                state_color = (0, 165, 255)
            
            if progress_bar_active:
                time_left = max(0.0, hold_target_time - progress_time)
                h, w = final_annotated_image.shape[:2]
                bar_width = int(w * 0.4)
                bar_height = 15
                bar_x = (w - bar_width) // 2
                bar_y = h - 30
                progress_ratio = min(1.0, progress_time / hold_target_time)
                progress_fill = int(bar_width * progress_ratio)
                cv2.rectangle(final_annotated_image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 0), 2)
                fill_color = (0, 255, 0) if progress_label == "VERIFY/ACQUIRE IN" else (0, 165, 255)
                cv2.rectangle(final_annotated_image, (bar_x, bar_y), (bar_x + progress_fill, bar_y + bar_height), fill_color, -1)
                text_progress = f"{progress_label}: {time_left:.1f}s"
                text_pt = (bar_x + 10, bar_y + bar_height - 3)
                cv2.putText(final_annotated_image, text_progress, text_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
            draw_outlined_text(final_annotated_image, f"State: {robot_state_display}", (10, 90), font, font_scale, state_color, thickness)
            with self.data_lock:
                current_id = self.target_track_id if self.target_track_id != -1 else self.last_target_track_id
            id_color = (0, 255, 255) if current_id != -1 else (0, 0, 255)
            draw_outlined_text(final_annotated_image, f"ID Tracked: {current_id}", (10, 120), font, font_scale, id_color, thickness)
            draw_outlined_text(final_annotated_image, f"Gesture: {self.active_gesture}", (10, 150), font, font_scale, (255, 0, 255), thickness)
            
            with self.data_lock:
                left_s = "L" if self.obstacle_in_left else "_"
                center_s = "C" if self.obstacle_in_center else "_"
                right_s = "R" if self.obstacle_in_right else "_"
                obstacle_status = f"{left_s}{center_s}{right_s}"
                is_any_obstacle = self.is_obstacle_close
            obstacle_color = (0, 0, 255) if is_any_obstacle else (255, 255, 255)
            draw_outlined_text(final_annotated_image, f"Obstacle (Depth < {self.SLOW_DISTANCE}m): {obstacle_status}", (10, 180), font, font_scale, obstacle_color, thickness)
            
            with self.depth_lock:
                distance_to_draw = self.target_distance_m
            dist_color = (0, 255, 255) if distance_to_draw > self.SAFETY_DISTANCE else (0, 0, 255)
            draw_outlined_text(final_annotated_image, f"Target Dist: {distance_to_draw:.2f}m", (10, 210), font, font_scale, dist_color, thickness)
            
            try:
                ros_image = self.bridge.cv2_to_imgmsg(final_annotated_image, "bgr8")
                ros_image.header = msg.header
                self.publisher_.publish(ros_image)
            except CvBridgeError as e:
                self.get_logger().error(f'[Thread] CvBridge error on publish: {e}')

            self.image_queue.task_done()
    # =========================================================================
    # --- publish_cmd_vel function ---
    # =========================================================================
    def publish_cmd_vel(self):
        twist_msg = Twist()
        current_time = time.time()

        with self.gesture_lock:
            active_gesture = self.active_gesture
        with self.state_lock:
            robot_state = self.robot_state
        
        with self.data_lock:
            is_any_obstacle = self.is_obstacle_close
            obs_left = self.obstacle_in_left
            obs_center = self.obstacle_in_center
            obs_right = self.obstacle_in_right
            current_target_id = self.target_track_id if self.target_track_id != -1 else self.last_target_track_id
            target_detected = self.target_detected 
            hip_midpoint_x = self.hip_midpoint_x  
            person_normalized_size = self.person_normalized_size

        with self.depth_lock:
            target_distance = self.target_distance_m 

        # --- 1. Handle "Waiting" or "Interrrupt" States (These all return) ---
        if active_gesture == "Open_Palm" or robot_state == "open_palm_stop":
            if active_gesture == "Open_Palm":
                if robot_state != "open_palm_stop":
                    if robot_state == "reacquire_target":
                        self.reacquire_elapsed_in_step = current_time - self.reacquire_start_time
                    elif robot_state == "reacquire_verify_target":
                        self.verify_elapsed_in_step = current_time - self.verify_start_time
                    with self.state_lock:
                        self.previous_robot_state = robot_state 
                        self.robot_state = "open_palm_stop"
                    self.get_logger().info(f"[GESTURE INTERRUPT] Open_Palm detected. Transitioning from {self.previous_robot_state.upper()} to OPEN_PALM_STOP (Reversing).")
                    with self.data_lock:
                        self.tracking_grace_period = False
                self.gesture_clear_start_time = 0.0
                twist_msg.linear.x = -0.15
                twist_msg.angular.z = 0.0
                self.get_logger().info(f"[OPEN_PALM ID: {current_target_id}] Reversing: L_X: {twist_msg.linear.x:.2f}, A_Z: {twist_msg.angular.z:.2f}")
            elif robot_state == "open_palm_stop" and active_gesture != "Open_Palm":
                if self.gesture_clear_start_time == 0.0:
                    self.gesture_clear_start_time = current_time
                    twist_msg.linear.x = 0.0
                    twist_msg.angular.z = 0.0
                    self.get_logger().info(f"[OPEN_PALM ID: {current_target_id}] Hand cleared. Debouncing start.")
                elif (current_time - self.gesture_clear_start_time) >= self.GESTURE_CLEAR_HOLD_TIME:
                    with self.state_lock:
                        self.robot_state = self.previous_robot_state
                        self.previous_robot_state = "waiting_for_gesture"
                    if self.robot_state == "reacquire_target":
                        self.reacquire_start_time = current_time 
                    elif self.robot_state == "reacquire_verify_target":
                        self.verify_start_time = current_time 
                    self.gesture_clear_start_time = 0.0 
                    twist_msg.linear.x = 0.0
                    twist_msg.angular.z = 0.0
                    self.get_logger().info(f"[GESTURE CLEAR] Debounce complete. Resuming previous state: {self.robot_state.upper()}.")
                else:
                    twist_msg.linear.x = 0.0
                    twist_msg.angular.z = 0.0
                    self.get_logger().info(f"[OPEN_PALM ID: {current_target_id}] Waiting for clear debounce...")
            self.cmd_vel_publisher.publish(twist_msg)
            return

        if active_gesture == "Pointing_Up" or robot_state == "gesture_turn_left":
            if active_gesture == "Pointing_Up":
                if robot_state != "gesture_turn_left":
                    if robot_state == "reacquire_target":
                        self.reacquire_elapsed_in_step = current_time - self.reacquire_start_time
                    elif robot_state == "reacquire_verify_target":
                        self.verify_elapsed_in_step = current_time - self.verify_start_time
                    with self.state_lock:
                        self.previous_robot_state = robot_state 
                        self.robot_state = "gesture_turn_left"
                    self.get_logger().info(f"[GESTURE INTERRUPT] Pointing_Up. Transitioning from {self.previous_robot_state.upper()} to GESTURE_TURN_LEFT.")
                    with self.data_lock:
                        self.tracking_grace_period = False
                self.gesture_clear_start_time = 0.0
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = self.GESTURE_TURN_SPEED
                self.get_logger().info(f"[GESTURE_TURN_LEFT ID: {current_target_id}] Turning Left: A_Z: {twist_msg.angular.z:.2f}")
            elif robot_state == "gesture_turn_left" and active_gesture != "Pointing_Up":
                if self.gesture_clear_start_time == 0.0:
                    self.gesture_clear_start_time = current_time
                    twist_msg.linear.x = 0.0
                    twist_msg.angular.z = 0.0
                    self.get_logger().info(f"[GESTURE_TURN_LEFT ID: {current_target_id}] Hand cleared. Debouncing start.")
                elif (current_time - self.gesture_clear_start_time) >= self.GESTURE_CLEAR_HOLD_TIME:
                    with self.state_lock:
                        self.robot_state = self.previous_robot_state
                        self.previous_robot_state = "waiting_for_gesture"
                    if self.robot_state == "reacquire_target":
                        self.reacquire_start_time = current_time 
                    elif self.robot_state == "reacquire_verify_target":
                        self.verify_start_time = current_time 
                    self.gesture_clear_start_time = 0.0 
                    twist_msg.linear.x = 0.0
                    twist_msg.angular.z = 0.0
                    self.get_logger().info(f"[GESTURE CLEAR] Debounce complete. Resuming previous state: {self.robot_state.upper()}.")
                else:
                    twist_msg.linear.x = 0.0
                    twist_msg.angular.z = 0.0
                    self.get_logger().info(f"[GESTURE_TURN_LEFT ID: {current_target_id}] Waiting for clear debounce...")
            self.cmd_vel_publisher.publish(twist_msg)
            return

        if active_gesture == "Victory" or robot_state == "gesture_turn_right":
            if active_gesture == "Victory":
                if robot_state != "gesture_turn_right":
                    if robot_state == "reacquire_target":
                        self.reacquire_elapsed_in_step = current_time - self.reacquire_start_time
                    elif robot_state == "reacquire_verify_target":
                        self.verify_elapsed_in_step = current_time - self.verify_start_time
                    with self.state_lock:
                        self.previous_robot_state = robot_state 
                        self.robot_state = "gesture_turn_right"
                    self.get_logger().info(f"[GESTURE INTERRUPT] Victory. Transitioning from {self.previous_robot_state.upper()} to GESTURE_TURN_RIGHT.")
                    with self.data_lock:
                        self.tracking_grace_period = False
                self.gesture_clear_start_time = 0.0
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = -self.GESTURE_TURN_SPEED
                self.get_logger().info(f"[GESTURE_TURN_RIGHT ID: {current_target_id}] Turning Right: A_Z: {twist_msg.angular.z:.2f}")
            elif robot_state == "gesture_turn_right" and active_gesture != "Victory":
                if self.gesture_clear_start_time == 0.0:
                    self.gesture_clear_start_time = current_time
                    twist_msg.linear.x = 0.0
                    twist_msg.angular.z = 0.0
                    self.get_logger().info(f"[GESTURE_TURN_RIGHT ID: {current_target_id}] Hand cleared. Debouncing start.")
                elif (current_time - self.gesture_clear_start_time) >= self.GESTURE_CLEAR_HOLD_TIME:
                    with self.state_lock:
                        self.robot_state = self.previous_robot_state
                        self.previous_robot_state = "waiting_for_gesture"
                    if self.robot_state == "reacquire_target":
                        self.reacquire_start_time = current_time 
                    elif self.robot_state == "reacquire_verify_target":
                        self.verify_start_time = current_time 
                    self.gesture_clear_start_time = 0.0 
                    twist_msg.linear.x = 0.0
                    twist_msg.angular.z = 0.0
                    self.get_logger().info(f"[GESTURE CLEAR] Debounce complete. Resuming previous state: {self.robot_state.upper()}.")
                else:
                    twist_msg.linear.x = 0.0
                    twist_msg.angular.z = 0.0
                    self.get_logger().info(f"[GESTURE_TURN_RIGHT ID: {current_target_id}] Waiting for clear debounce...")
            self.cmd_vel_publisher.publish(twist_msg)
            return

        if robot_state == "reacquire_verify_target":
            elapsed_time = current_time - self.verify_start_time
            total_elapsed = self.verify_elapsed_in_step + elapsed_time
            time_remaining = self.VERIFY_TIMEOUT - total_elapsed
            if total_elapsed >= self.VERIFY_TIMEOUT:
                self.get_logger().warn(f"[VERIFY TIMEOUT] Verification failed... Abandoning target ID {self.last_target_track_id}.")
                with self.state_lock:
                    self.robot_state = "waiting_for_gesture"
                with self.data_lock:
                    self.target_track_id = -1
                    self.last_target_track_id = -1
                    self.last_known_target_position = None 
                    self.world_frame_goal = None
                self.reacquire_step = 0
                self.reacquire_elapsed_in_step = 0.0 
                self.verify_elapsed_in_step = 0.0 
                self.verify_start_time = 0.0 
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
            self.get_logger().info(f"[VERIFY ID: {current_target_id}] Stopping, waiting for THUMBS UP. Remaining: {time_remaining:.1f}s")
            self.cmd_vel_publisher.publish(twist_msg)
            return

        # --- 2. Handle "Active Navigation" States ---
        
        final_linear_x = 0.0
        final_angular_z = 0.0
        control_mode = "NONE"
        is_in_grace_period = False # Local flag

        if robot_state == "tracking":
            if target_detected:
                
                if target_distance > 0.0 and target_distance < self.SAFETY_DISTANCE:
                    twist_msg.linear.x = 0.0
                    twist_msg.angular.z = 0.0
                    self.get_logger().warn(f"[TRACKING ID: {current_target_id}] EMERGENCY STOP: Depth {target_distance:.2f}m is TOO CLOSE!")
                    self.cmd_vel_publisher.publish(twist_msg)
                    return

                angular_error = 0.5 - hip_midpoint_x
                final_angular_z = angular_error * self.ANGULAR_GAIN_P
                
                with self.data_lock:
                    is_in_grace_period = self.tracking_grace_period
                
                if is_in_grace_period:
                    if abs(angular_error) > self.GRACE_PERIOD_ANGULAR_THRESHOLD:
                        control_mode = "GRACE_PERIOD (TURNING)"
                        final_linear_x = 0.0 
                    else:
                        with self.data_lock:
                            self.tracking_grace_period = False
                        self.get_logger().info("Grace period ended (aligned). Engaging full tracking.")
                        is_in_grace_period = False 
                
                if not is_in_grace_period:
                    if target_distance > 0.0:
                        control_mode = "DEPTH"
                        if target_distance < self.TARGET_DISTANCE_SETPOINT and target_distance > self.SAFETY_DISTANCE:
                            linear_error = 0.0
                        else:
                            linear_error = target_distance - self.TARGET_DISTANCE_SETPOINT
                        desired_linear_x = linear_error * self.KP_LINEAR_DEPTH
                        final_linear_x = np.clip(desired_linear_x, -self.MAX_LINEAR_X, self.MAX_LINEAR_X)
                    else:
                        control_mode = "SIZE"
                        if person_normalized_size > self.SIZE_THRESHOLD_CLOSE:
                            final_linear_x = -0.1
                        elif person_normalized_size < self.SIZE_THRESHOLD_FAR:
                            final_linear_x = 0.1
                        else:
                            final_linear_x = 0.0
            
            else:
                self.get_logger().warn(f"[TRACKING] Target lost, but state not changed. Sending STOP.")
                final_linear_x = 0.0
                final_angular_z = 0.0

        elif robot_state == "reacquire_target":
            control_mode = "REACQUIRE"
            with self.data_lock:
                goal = self.world_frame_goal
            
            if goal is None:
                self.get_logger().error("[REACQUIRE] In reacquire state but have no world_frame_goal. Giving up.")
                with self.state_lock: self.robot_state = "waiting_for_gesture"
                with self.data_lock:
                    self.target_track_id = -1
                    self.last_target_track_id = -1
                    self.last_target_bbox = None 
                    self.last_known_target_position = None
                    self.world_frame_goal = None
                self.reacquire_step = 0
                self.reacquire_elapsed_in_step = 0.0 
                self.cmd_vel_publisher.publish(twist_msg) 
                return

            with self.odom_lock:
                rx, ry, ryaw = self.robot_x, self.robot_y, self.robot_yaw
            
            delta_x = goal[0] - rx
            delta_y = goal[1] - ry
            
            distance_to_goal = math.sqrt(delta_x**2 + delta_y**2)
            angle_to_goal = math.atan2(delta_y, delta_x)
            
            angle_error = angle_to_goal - ryaw
            
            if angle_error > math.pi: angle_error -= 2 * math.pi
            if angle_error < -math.pi: angle_error += 2 * math.pi

            KP_ANGULAR = 1.0
            LINEAR_SPEED = 0.2
            MIN_ANGLE_ERROR = 0.1 
            MIN_DIST_ERROR = 0.2 
            
            if self.reacquire_step == 1:
                if abs(angle_error) > MIN_ANGLE_ERROR:
                    final_linear_x = 0.0
                    final_angular_z = KP_ANGULAR * angle_error
                else:
                    self.get_logger().info("[REACQUIRE Step 1] Facing goal complete. Moving to Step 2.")
                    with self.state_lock: self.reacquire_step = 2
                    final_linear_x = 0.0
                    final_angular_z = 0.0
            
            elif self.reacquire_step == 2:
                if distance_to_goal > MIN_DIST_ERROR:
                    final_linear_x = LINEAR_SPEED
                    final_angular_z = KP_ANGULAR * angle_error
                else:
                    self.get_logger().info("[REACQUIRE Step 2] Arrived at LKP. Moving to Step 3.")
                    with self.state_lock: self.reacquire_step = 3
                    final_linear_x = 0.0
                    final_angular_z = 0.0
            
            elif self.reacquire_step == 3:
                self.get_logger().warn("[REACQUIRE Step 3] Reached LKP but target not found. Giving up.")
                with self.state_lock: self.robot_state = "waiting_for_gesture"
                with self.data_lock:
                    self.target_track_id = -1
                    self.last_target_track_id = -1
                    self.last_target_bbox = None 
                    self.last_known_target_position = None
                    self.world_frame_goal = None
                self.reacquire_step = 0
                self.reacquire_elapsed_in_step = 0.0
                final_linear_x = 0.0
                final_angular_z = 0.0

        elif robot_state == "waiting_for_gesture":
            with self.data_lock:
                self.last_target_bbox = None
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
            self.cmd_vel_publisher.publish(twist_msg)
            return

        else: # Unknown state
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
            with self.state_lock:
                self.robot_state = "waiting_for_gesture"
            self.cmd_vel_publisher.publish(twist_msg)
            return

        # --- 3. NEW SHARED AVOIDANCE LOGIC (Applies to 'tracking' and 'reacquire_target') ---
        if is_any_obstacle:
            
            run_avoidance = True
            
            # Don't avoid if we are in the alignment grace period
            if is_in_grace_period: 
                self.get_logger().info(f"[TRACKING] Grace period active. Disabling avoidance.")
                run_avoidance = False
            
            # Don't avoid if we are intentionally moving backward or just turning in place
            elif final_linear_x <= 0.0: 
                self.get_logger().info(f"[{control_mode}] Moving backward or turning. Disabling avoidance.")
                run_avoidance = False
            
            if run_avoidance:
                if obs_center:
                    self.get_logger().warn(f"[AVOIDANCE] Center obstacle. Forcing arc maneuver.")
                    # Keep moving forward, but at a controlled avoidance speed
                    if final_linear_x > 0.0:
                        final_linear_x = self.AVOIDANCE_LINEAR_SPEED
                    
                    target_is_left = False
                    target_is_right = False

                    # Decide which way to turn based on the goal
                    if robot_state == 'tracking':
                        target_is_left = hip_midpoint_x < 0.4
                        target_is_right = hip_midpoint_x > 0.6
                    elif robot_state == 'reacquire_target':
                        # We already calculated angle_error, but it's out of scope. Recalculate.
                        with self.odom_lock: ryaw = self.robot_yaw
                        with self.data_lock: goal = self.world_frame_goal
                        if goal:
                            delta_x = goal[0] - self.robot_x
                            delta_y = goal[1] - self.robot_y
                            angle_to_goal = math.atan2(delta_y, delta_x)
                            angle_error = angle_to_goal - ryaw
                            if angle_error > math.pi: angle_error -= 2 * math.pi
                            if angle_error < -math.pi: angle_error += 2 * math.pi
                            
                            if angle_error > 0.1: target_is_left = True # Goal is to the left
                            if angle_error < -0.1: target_is_right = True # Goal is to the right

                    # Now, execute the turn logic
                    if target_is_left and not obs_left:
                        final_angular_z = self.AVOIDANCE_TURN_SPEED
                    elif target_is_right and not obs_right:
                        final_angular_z = -self.AVOIDANCE_TURN_SPEED
                    elif not obs_left: # Failsafe: if goal is center, try left
                        final_angular_z = self.AVOIDANCE_TURN_SPEED
                    elif not obs_right: # Failsafe: try right
                        final_angular_z = -self.AVOIDANCE_TURN_SPEED
                    else:
                        self.get_logger().error(f"[AVOIDANCE] TRAPPED! L, C, R blocked. Stopping.")
                        final_linear_x = 0.0
                        final_angular_z = 0.0
                
                elif final_linear_x > 0.0 and (obs_left and final_angular_z > 0):
                    self.get_logger().warn(f"[AVOIDANCE] Left obstacle. Reducing left turn.")
                    final_angular_z *= 0.5
                elif final_linear_x > 0.0 and (obs_right and final_angular_z < 0):
                    self.get_logger().warn(f"[AVOIDANCE] Right obstacle. Reducing right turn.")
                    final_angular_z *= 0.5

        # --- 4. Final Publish ---
        twist_msg.linear.x = final_linear_x
        twist_msg.angular.z = final_angular_z
        
        if twist_msg.linear.x != 0.0 or twist_msg.angular.z != 0.0 or control_mode == "GRACE_PERIOD (TURNING)":
             self.get_logger().info(f"[{control_mode} ID: {current_target_id}] Moving L_X: {twist_msg.linear.x:.2f}, A_Z: {twist_msg.angular.z:.2f} (Dist: {target_distance:.2f}m)")
        
        self.cmd_vel_publisher.publish(twist_msg)
# =========================================================================


def main(args=None):
    rclpy.init(args=args)
    node = MovenetROS2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped cleanly.')
    except Exception as e:
        node.get_logger().error(f'Caught exception: {e}')
    finally:
        node.get_logger().info('Shutting down threads...')
        final_stop = Twist()
        node.cmd_vel_publisher.publish(final_stop)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()