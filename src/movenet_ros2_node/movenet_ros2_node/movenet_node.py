import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
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

# --- NEW MEDIAPIPE IMPORTS for Gesture Recognizer ---
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# ----------------------------------------------------

# --- LOCAL MODULE IMPORT (MOCK for standalone code) ---
# NOTE: This section is kept for completeness but requires 'tracker.py' in a real ROS package.
try:
    from .tracker import TrackerIoU, TrackerOKS, TRACK_COLORS
except ImportError:
    print("Warning: Could not import tracker module. Using dummy classes/variables.")
    class DummyTracker:
        def __init__(self): pass
        def apply(self, bodies, timestamp): return bodies
    TrackerIoU = DummyTracker
    TrackerOKS = DummyTracker
    # Define TRACK_COLORS if the import fails
    TRACK_COLORS = [(0, 255, 255), (255, 0, 0), (0, 0, 255), (255, 255, 0)]

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
    """Class to hold pose detection results, including bounding box and tracking info."""
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


# =========================================================================
# 1. OpenVINO Inference and Processing Class
# =========================================================================

class MovenetProcessor:
    """Handles the OpenVINO MoveNet MultiPose inference and tracking."""

    def __init__(self, node_logger, model_xml, device='CPU', tracking_method=None):
        self.get_logger = node_logger

        try:
            self.ie = Core()
            # Set preferred device based on the user's explicit request for CPU
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

        self.score_threshold = 0.1 # Adjust this for min confidence
        self.padding = None

        # --- Tracking Initialization ---
        self.tracking = False
        if tracking_method in ["iou", "oks"]:
            self.tracking = True
            self.tracker = TrackerIoU() if tracking_method == "iou" else TrackerOKS()
            self.get_logger().info(f"Using {tracking_method.upper()} tracking.")
        else:
            self.get_logger().info("Tracking is disabled.")

    def calculate_padding(self, original_h, original_w):
        """Calculates padding to match the model's aspect ratio."""
        model_aspect = self.input_width / self.input_height
        frame_aspect = original_w / original_h

        if frame_aspect > model_aspect:
            pad_h = int(original_w / model_aspect - original_h)
            self.padding = Padding(0, pad_h, original_w, original_h + pad_h)
        else:
            pad_w = int(original_h * model_aspect - original_w)
            self.padding = Padding(pad_w, 0, original_w + pad_w, original_h)

    def pad_and_resize(self, frame):
        """Pad and resize the image to prepare for the model input."""
        if self.padding is None:
              self.calculate_padding(frame.shape[0], frame.shape[1])

        padded = cv2.copyMakeBorder(frame, 0, self.padding.h, 0, self.padding.w, cv2.BORDER_CONSTANT)
        padded = cv2.resize(padded, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)

        return padded

    def preprocess(self, padded_frame):
        """Converts to RGB, CHW, float32, and adds batch dimension."""
        frame_nn = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32)
        input_tensor = frame_nn[None,]

        return input_tensor

    def postprocess(self, output_data, original_h, original_w):
        """Decodes model output into Body objects and scales keypoints."""
        output = output_data.squeeze()

        bodies = []
        for person_idx in range(output.shape[0]):

            person_score = output[person_idx, 55]

            if person_score < self.score_threshold:
                continue

            kps_yxs = output[person_idx, :51].reshape(17, -1)
            kp_scores = kps_yxs[:, 2]
            kp_yx_norm = kps_yxs[:, :2]

            # --- Bounding Box extraction and scaling ---
            bbox_norm_yx = output[person_idx, 51:55]
            bbox_scaled_yx = bbox_norm_yx * np.array([self.padding.padded_h, self.padding.padded_w, self.padding.padded_h, self.padding.padded_w])

            ymin, xmin, ymax, xmax = bbox_scaled_yx.astype(np.int32)

            xmin = np.clip(xmin, 0, original_w)
            ymin = np.clip(ymin, 0, original_h)
            xmax = np.clip(xmax, 0, original_w)
            ymax = np.clip(ymax, 0, original_h)

            # --- Keypoint Scaling ---
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

    def draw_poses(self, frame, bodies):
        """Draws keypoints, skeletons, and track IDs on the frame."""
        for body in bodies:

            track_color = (0, 255, 255)
            if self.tracking and body.track_id != -1:
                track_color = TRACK_COLORS[body.track_id % len(TRACK_COLORS)]

            # 1. Draw Skeleton
            for start_idx, end_idx in _POSE_LINES:
                start_pt_xy = body.keypoints[start_idx]
                end_pt_xy = body.keypoints[end_idx]
                start_score = body.keypoints_score[start_idx]
                end_score = body.keypoints_score[end_idx]

                if start_score > self.score_threshold and end_score > self.score_threshold:
                    p1 = (int(start_pt_xy[0]), int(start_pt_xy[1]))
                    p2 = (int(end_pt_xy[0]), int(end_pt_xy[1]))
                    cv2.line(frame, p1, p2, track_color, 2)

            # 2. Draw Keypoints
            for (x, y), score in zip(body.keypoints, body.keypoints_score):
                if score > self.score_threshold:
                    center = (int(x), int(y))
                    cv2.circle(frame, center, 4, track_color, -1)

            # 3. Draw Track ID and Bounding Box
            if self.tracking and body.track_id != -1:
                cv2.rectangle(frame, (body.xmin, body.ymin), (body.xmax, body.ymax), track_color, 2)
                id_text = f"ID: {body.track_id} ({body.score:.2f})"
                text_pt = (body.xmin, body.ymin - 10)
                # Draw black background/border
                cv2.putText(frame, id_text, text_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                # Draw main text
                cv2.putText(frame, id_text, text_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, track_color, 2)

        return frame


    def process_frame(self, frame):
        """Full inference pipeline: pad -> pre-process -> infer -> post-process -> track -> draw."""

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

# Helper function to draw text with a black outline (border)
def draw_outlined_text(image, text, org, font, font_scale, color, thickness):
    """Draws text with a black outline for better visibility."""
    # Draw black outline
    cv2.putText(image, text, org, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # Draw main text
    cv2.putText(image, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

class MovenetROS2Node(Node):
    """ROS2 Node for running MoveNet and MediaPipe hand detection with robot control."""

    def __init__(self):
        super().__init__('movenet_detector')
        self.get_logger().info("MoveNet Detector Node Initializing...")

        # --- Parameter and Model Setup ---
        self.declare_parameter('tracking_method', 'oks')
        self.declare_parameter('input_topic', '/image_raw')
        self.declare_parameter('output_topic', 'movenet/image_out')
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')
        self.declare_parameter('reacquire_duration_per_step', 1.0) # RESTORED
        self.declare_parameter('gesture_model_name', 'gesture_recognizer.task') # Parameter for .task file

        tracking_method = self.get_parameter('tracking_method').get_parameter_value().string_value
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        self.reacquire_duration_per_step = self.get_parameter('reacquire_duration_per_step').get_parameter_value().double_value # RESTORED
        gesture_model_name = self.get_parameter('gesture_model_name').get_parameter_value().string_value

        try:
            # Assumes the node is inside a package named 'movenet_ros2_node'
            package_share_directory = ament_index_python.packages.get_package_share_directory('movenet_ros2_node')
        except ament_index_python.packages.PackageNotFoundError:
            # Fallback for local/non-ROS environment testing
            self.get_logger().warn("ROS package 'movenet_ros2_node' not found. Using local directory for models.")
            package_share_directory = os.path.dirname(os.path.abspath(__file__))

        # --- Model Paths ---
        # NOTE: Forcing smallest model for CPU performance
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
            device='CPU', # Locked to CPU
            tracking_method=tracking_method
        )
        self.bridge = CvBridge()

        # --- MediaPipe Hands Setup (for Drawing and Legacy Compatibility) ---
        self.mp_hands = mp.solutions.hands
        self.hands_landmarker = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1,
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # --- MediaPipe Gesture Recognizer Setup (NEW VISION API) ---
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


        # --- Robot Control State & Data (SHARED) ---
        self.robot_state = "waiting_for_gesture"
        self.active_gesture = "none"

        self.target_track_id = -1
        self.last_target_track_id = -1
        self.reacquire_step = 0 # RESTORED
        self.reacquire_start_time = 0.0 # RESTORED

        self.last_target_direction = 'center'
        self.last_all_bodies = []
        
        # --- Bounding Box variables for Reacquire logic ---
        self.last_seen_bbox = None # (xmin, ymin, xmax, ymax) from image thread
        self.last_target_bbox = None # (xmin, ymin, xmax, ymax) static for drawing during reacquire (RESTORED USAGE)

        # Stores the state before entering open_palm_stop
        self.previous_robot_state = "waiting_for_gesture" 
        
        # --- DEBOUNCE VARIABLES (Open_Palm Clear) ---
        self.gesture_clear_start_time = 0.0
        self.GESTURE_CLEAR_HOLD_TIME = 0.5 # 0.5 seconds required before resuming state
        # ------------------------------------------

        # --- THUMBS_UP DEBOUNCE VARIABLES (Acquire Target) ---
        self.thumbs_up_start_time = 0.0
        self.THUMBS_UP_HOLD_TIME = 2.0 # 2.0 seconds required to trigger tracking
        # ---------------------------------------------
        
        # --- ILOVEYOU DEBOUNCE VARIABLES (Stop Tracking) ---
        self.iloveyou_start_time = 0.0
        self.ILOVEYOU_HOLD_TIME = 2.0 # 2.0 seconds required to stop tracking
        # ----------------------------------------------------

        # Control thresholds
        self.SIZE_THRESHOLD_CLOSE = 0.80
        self.SIZE_THRESHOLD_FAR = 0.50
        self.MAX_EXPECTED_NORMALIZED_SIZE = 0.7

        # Shared data variables
        self.hip_midpoint_x = 0.5
        self.person_normalized_size = 0.0
        self.person_detected = False
        self.target_detected = False

        # Locks for thread-safe access
        self.gesture_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.data_lock = threading.Lock()

        # FPS Tracking Variables
        self.last_time = time.time()
        self.current_fps = 0.0

        # Frame Skip Counters for Gesture Recognition
        self.gesture_frame_skip = 0
        self.skip_frames_interval = 1


        # Timer for movement logic (20 Hz)
        self.timer = self.create_timer(0.05, self.publish_cmd_vel)

        # Multithreading Setup
        self.image_queue = queue.Queue(maxsize=1)
        self.processor_thread = threading.Thread(target=self.process_images_thread, daemon=True)
        self.processor_thread.start()

        # ROS Communication Setup
        qos_profile = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.subscription = self.create_subscription(Image, input_topic, self.image_callback, qos_profile)
        self.publisher_ = self.create_publisher(Image, output_topic, 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, cmd_vel_topic, 10)


    def image_callback(self, msg):
        if not self.image_queue.empty():
            try:
                self.image_queue.get_nowait()
            except queue.Empty:
                pass
        self.image_queue.put(msg)


    def process_images_thread(self):
        while rclpy.ok():
            current_time = time.time()
            try:
                msg = self.image_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except CvBridgeError as e:
                self.get_logger().error(f'[Thread] CvBridge error: {e}')
                self.image_queue.task_done()
                continue

            annotated_image, bodies = self.processor.process_frame(cv_image) 

            if len(bodies) > 0:
                with self.data_lock:
                    self.last_all_bodies = bodies

            with self.state_lock:
                robot_state = self.robot_state
            with self.data_lock:
                target_track_id = self.target_track_id
                last_target_track_id = self.last_target_track_id # Used in reacquire

            
            person_detected_in_frame = len(bodies) > 0
            hip_midpoint_x_norm = 0.5
            person_normalized_size = 0.0
            target_detected_in_frame = False
            target_body = None

            # 1. Determine which ID to check
            id_to_check = -1
            if robot_state == "tracking" or robot_state == "open_palm_stop":
                id_to_check = target_track_id
            elif robot_state == "reacquire_target":
                id_to_check = last_target_track_id
            elif robot_state == "waiting_for_gesture" and person_detected_in_frame:
                # In waiting, use the highest scoring body for gesture checks
                pass
            
            # 2. Find the target body based on ID or score
            if id_to_check != -1: 
                target_bodies = [b for b in bodies if b.track_id == id_to_check]
                if target_bodies:
                    target_body = target_bodies[0]
                    target_detected_in_frame = True
            elif robot_state == "waiting_for_gesture" and person_detected_in_frame:
                target_body = max(bodies, key=lambda b: b.score)

            # 3. Handle Target Loss (Transition to REACQUIRE)
            if robot_state == "tracking" and not target_detected_in_frame:
                self.get_logger().warn(f"Target ID {target_track_id} lost. Entering REACQUIRE state.")
                with self.data_lock:
                    self.last_target_bbox = self.last_seen_bbox # Save LKBBox
                    self.target_track_id = -1
                    self.last_target_track_id = target_track_id # Save lost ID
                with self.state_lock:
                    self.robot_state = "reacquire_target"
                    self.reacquire_start_time = time.time()
                    self.reacquire_step = 1 # Start reacquire sequence
                self.iloveyou_start_time = 0.0 # Reset ILY timer
            
            # 4. Handle Target Reacquired (Transition from REACQUIRE to TRACKING)
            if robot_state == "reacquire_target" and target_detected_in_frame:
                 self.get_logger().info(f"Target ID {last_target_track_id} reacquired! Returning to TRACKING.")
                 with self.data_lock:
                    self.last_target_bbox = None 
                    self.target_track_id = last_target_track_id 
                 with self.state_lock:
                    self.robot_state = "tracking"
                    self.reacquire_step = 0
            
            # 5. Update data if a target body was found
            if target_body:
                # --- Update the last_seen_bbox on successful detection ---
                with self.data_lock:
                    self.last_seen_bbox = (target_body.xmin, target_body.ymin, target_body.xmax, target_body.ymax)
                
                # If a target body is detected and is NOT in reacquire state, clear the LKBBox
                if robot_state != "reacquire_target" and robot_state != "open_palm_stop":
                    with self.data_lock:
                        self.last_target_bbox = None
                
                # ... (Midpoint and Size calculation remains the same)
                nose_idx, left_hip_idx, right_hip_idx = KEYPOINT_DICT['nose'], KEYPOINT_DICT['left_hip'], KEYPOINT_DICT['right_hip']
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
                self.target_detected = target_detected_in_frame
                self.hip_midpoint_x = hip_midpoint_x_norm
                self.person_normalized_size = person_normalized_size
                if self.target_detected:
                    if hip_midpoint_x_norm < 0.4: self.last_target_direction = 'left'
                    elif hip_midpoint_x_norm > 0.6: self.last_target_direction = 'right'
                    else: self.last_target_direction = 'center'

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
                    h, w = cv_image.shape[:2]

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
                                
                                # Priority: Thumbs Up (Acquire)
                                if gesture_name == "Thumb_Up":
                                    thumbs_up_active = True
                                    current_gesture = gesture_name
                                
                                # Priority: I Love You (Stop Tracking)
                                elif gesture_name == "ILoveYou": 
                                    iloveyou_active = True
                                    current_gesture = gesture_name
                                
                                # Open Palm: ONLY active if robot is in a TRACKING/REACQUIRE/STOPPED state
                                elif gesture_name == "Open_Palm":
                                    if robot_state in ["tracking", "reacquire_target", "open_palm_stop"]:
                                        current_gesture = gesture_name
                                
                                # --- DRAWING LOGIC (Restored for any hand on target) ---
                                # Draw hand skeleton 
                                self.mp_drawing.draw_landmarks(
                                    image_rgb, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                                    self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2),
                                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                                )
                                
                                # Draw gesture label 
                                display_gesture_name = current_gesture.upper() if current_gesture != "none" else gesture_name.upper()
                                gesture_text = f"{display_gesture_name} ({gesture_score:.2f})"
                                text_org = (wrist_px_x, wrist_px_y - 20)
                                draw_outlined_text(image_rgb, gesture_text, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                                
                                # Break immediately after processing the first hand detected on the target body.
                                break 
                                # --- END OF DRAWING LOGIC ---

            with self.gesture_lock:
                self.active_gesture = current_gesture
            
            # ----------------------------------------------------
            # --- GESTURE DEBOUNCE LOGIC (Executed on detected person) ---
            # ----------------------------------------------------
            # Ensure target_body is valid and has a track ID for tracking/stop logic
            if target_body and (target_body.track_id != -1 or robot_state == "waiting_for_gesture"): 

                # --- 1. THUMBS UP DEBOUNCE (Acquire Target) ---
                if robot_state == "waiting_for_gesture":
                    if thumbs_up_active:
                        if self.thumbs_up_start_time == 0.0:
                            self.thumbs_up_start_time = current_time
                            self.get_logger().info(f"[THUMBS UP HOLD] Start timer for ID {target_body.track_id}. Hold for {self.THUMBS_UP_HOLD_TIME:.1f}s.")
                        
                        elif (current_time - self.thumbs_up_start_time) >= self.THUMBS_UP_HOLD_TIME:
                            with self.data_lock:
                                self.target_track_id = target_body.track_id
                                self.last_target_track_id = target_body.track_id
                            with self.state_lock:
                                self.robot_state = "tracking"
                                self.reacquire_step = 0
                            self.thumbs_up_start_time = 0.0
                            self.get_logger().info(f"State transition: WAITING -> TRACKING for ID {self.target_track_id}. (Hold complete).")

                    elif self.thumbs_up_start_time != 0.0:
                        self.get_logger().info(f"[THUMBS UP HOLD] Resetting timer (Gesture lost or flickered).")
                        self.thumbs_up_start_time = 0.0
                
                # --- 2. I LOVE YOU DEBOUNCE (Stop Tracking) ---
                elif robot_state == "tracking" or robot_state == "reacquire_target" or robot_state == "open_palm_stop":
                    if iloveyou_active:
                        if self.iloveyou_start_time == 0.0:
                            with self.data_lock:
                                current_target_id = self.target_track_id if self.target_track_id != -1 else self.last_target_track_id
                            self.iloveyou_start_time = current_time
                            self.get_logger().info(f"[I LOVE YOU HOLD] Start timer for ID {current_target_id}. Hold for {self.ILOVEYOU_HOLD_TIME:.1f}s.")
                        
                        elif (current_time - self.iloveyou_start_time) >= self.ILOVEYOU_HOLD_TIME:
                            # Transition back to waiting
                            with self.data_lock:
                                self.target_track_id = -1
                                self.last_target_track_id = -1
                                self.last_target_bbox = None
                            with self.state_lock:
                                self.robot_state = "waiting_for_gesture"
                                self.reacquire_step = 0
                            self.iloveyou_start_time = 0.0 # Reset timer
                            self.get_logger().info(f"State transition: STOP TRACKING via 'I Love You'. -> WAITING.")
                    
                    elif self.iloveyou_start_time != 0.0:
                        self.get_logger().info(f"[I LOVE YOU HOLD] Resetting timer (Gesture lost or flickered).")
                        self.iloveyou_start_time = 0.0

            # If not in relevant state, ensure timers are reset
            if robot_state != "waiting_for_gesture":
                self.thumbs_up_start_time = 0.0
            if robot_state != "tracking" and robot_state != "reacquire_target" and robot_state != "open_palm_stop":
                self.iloveyou_start_time = 0.0
            
            # ----------------------------------------------------


            # --- Draw MoveNet Poses and UI for Display ---
            final_annotated_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) # BGR for OpenCV drawing
            final_annotated_image = self.processor.draw_poses(final_annotated_image, bodies) # Draw current poses

            with self.state_lock:
                current_state_for_drawing = self.robot_state
            with self.data_lock:
                bbox_to_draw = self.last_target_bbox # Fetches the saved LKBBox
            
            # --- LKBBox Drawing Logic (RESTORED) ---
            # Draw the static red "last known location" bounding box
            if current_state_for_drawing == "reacquire_target" and bbox_to_draw is not None:
                xmin, ymin, xmax, ymax = bbox_to_draw
                # Draw the red box
                cv2.rectangle(final_annotated_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                # Draw a label for the box
                font = cv2.FONT_HERSHEY_SIMPLEX
                thickness = 2
                draw_outlined_text(final_annotated_image, "LAST SEEN", (xmin, ymin - 10), font, 0.6, (0, 0, 255), thickness)
            # --- End LKBBox Drawing Logic ---


            # --- Existing UI Drawing ---
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            current_time_frame = time.time()
            if (current_time_frame - self.last_time) > 0:
                self.current_fps = 1.0 / (current_time_frame - self.last_time)
            self.last_time = current_time_frame
            draw_outlined_text(final_annotated_image, f"FPS: {self.current_fps:.1f}", (10, 30), font, font_scale, (0, 255, 0), thickness)
            draw_outlined_text(final_annotated_image, f"People Detected: {len(bodies)}", (10, 60), font, font_scale, (255, 255, 0), thickness)

            robot_state_display = robot_state.upper()
            state_color = (0, 255, 255) if robot_state == "tracking" else (0, 165, 255)
            
            # --- Progress Bar and State Overlays ---
            progress_bar_active = False
            progress_time = 0.0
            hold_target_time = 0.0
            progress_label = ""
            
            if self.thumbs_up_start_time != 0.0:
                progress_bar_active = True
                progress_time = current_time - self.thumbs_up_start_time
                hold_target_time = self.THUMBS_UP_HOLD_TIME
                progress_label = "TRACKING IN"
                robot_state_display = "WAITING (HOLD)"
                state_color = (0, 255, 0) # Green for countdown
            
            elif self.iloveyou_start_time != 0.0:
                progress_bar_active = True
                progress_time = current_time - self.iloveyou_start_time
                hold_target_time = self.ILOVEYOU_HOLD_TIME
                progress_label = "STOPPING IN"
                robot_state_display = "TRACKING (HOLD STOP)"
                state_color = (0, 165, 255) # Orange/Yellow for countdown
            
            # Apply general state colors/labels if no gesture timer is running
            elif robot_state == "reacquire_target": # RESTORED
                 robot_state_display += f" (Step {self.reacquire_step})"
                 state_color = (0, 165, 255)
            elif robot_state == "open_palm_stop": 
                robot_state_display = "HALTED (OPEN PALM)"
                state_color = (0, 255, 255) # Yellow

            
            # Draw Progress Bar if active
            if progress_bar_active:
                time_left = max(0.0, hold_target_time - progress_time)
                
                h, w = final_annotated_image.shape[:2]
                bar_width = int(w * 0.4)
                bar_height = 15
                bar_x = (w - bar_width) // 2
                bar_y = h - 30
                progress_ratio = min(1.0, progress_time / hold_target_time)
                progress_fill = int(bar_width * progress_ratio)
                
                # Draw outline (black)
                cv2.rectangle(final_annotated_image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 0), 2)
                # Draw fill (color depends on state)
                fill_color = (0, 255, 0) if progress_label == "TRACKING IN" else (0, 165, 255)
                cv2.rectangle(final_annotated_image, (bar_x, bar_y), (bar_x + progress_fill, bar_y + bar_height), fill_color, -1)
                
                # Draw text overlay
                text_progress = f"{progress_label}: {time_left:.1f}s"
                text_pt = (bar_x + 10, bar_y + bar_height - 3)
                cv2.putText(final_annotated_image, text_progress, text_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
            # --- End Progress Bar ---


            draw_outlined_text(final_annotated_image, f"State: {robot_state_display}", (10, 90), font, font_scale, state_color, thickness)

            with self.data_lock:
                current_id = self.target_track_id if self.target_track_id != -1 else self.last_target_track_id
            id_color = (0, 255, 255) if current_id != -1 else (0, 0, 255)
            draw_outlined_text(final_annotated_image, f"ID Tracked: {current_id}", (10, 120), font, font_scale, id_color, thickness)
            draw_outlined_text(final_annotated_image, f"Gesture: {self.active_gesture} ({'Run' if run_gesture_recognition else 'Skip'})", (10, 150), font, font_scale, (255, 0, 255), thickness)

            try:
                ros_image = self.bridge.cv2_to_imgmsg(final_annotated_image, "bgr8")
                ros_image.header = msg.header
                self.publisher_.publish(ros_image)
            except CvBridgeError as e:
                self.get_logger().error(f'[Thread] CvBridge error on publish: {e}')

            self.image_queue.task_done()


    def publish_cmd_vel(self):
        twist_msg = Twist()
        current_time = time.time() # Use consistent time for debouncing

        with self.gesture_lock:
            active_gesture = self.active_gesture
        with self.state_lock:
            robot_state = self.robot_state
            previous_robot_state = self.previous_robot_state 

        # ---------------------------------------------------------------------
        # STATE MACHINE ENTRY CHECK: OPEN_PALM_STOP Priority
        # ---------------------------------------------------------------------

        # If Open_Palm is detected (by the tracked person), or if we are already in the stop state, handle the stop logic.
        if active_gesture == "Open_Palm" or robot_state == "open_palm_stop":
            
            # If Open_Palm is detected, transition or remain in open_palm_stop
            if active_gesture == "Open_Palm":
                if robot_state != "open_palm_stop":
                    with self.state_lock:
                        self.previous_robot_state = robot_state # Save current state
                        self.robot_state = "open_palm_stop"
                    self.get_logger().info(f"[GESTURE INTERRUPT] Open_Palm detected. Transitioning from {previous_robot_state.upper()} to OPEN_PALM_STOP (Reversing).")
                
                self.gesture_clear_start_time = 0.0
                twist_msg.linear.x = -0.15
                twist_msg.angular.z = 0.0
                self.get_logger().info(f"[OPEN_PALM_STOP] Reversing L_X: {twist_msg.linear.x:.2f}, A_Z: {twist_msg.angular.z:.2f}")

            elif robot_state == "open_palm_stop" and active_gesture != "Open_Palm":
                # Gesture cleared: Start or check the debounce timer
                if self.gesture_clear_start_time == 0.0:
                    self.gesture_clear_start_time = current_time
                    self.get_logger().info(f"[GESTURE CLEAR DEBOUNCE] Open_Palm cleared. Starting {self.GESTURE_CLEAR_HOLD_TIME}s debounce before resuming {previous_robot_state.upper()}.")
                    twist_msg.linear.x = 0.0
                    twist_msg.angular.z = 0.0
                
                elif (current_time - self.gesture_clear_start_time) >= self.GESTURE_CLEAR_HOLD_TIME:
                    # Debounce time met, transition back
                    with self.state_lock:
                        self.robot_state = previous_robot_state
                        # Reset previous state to avoid recursion
                        self.previous_robot_state = "waiting_for_gesture" if previous_robot_state == "open_palm_stop" else previous_robot_state
                    
                    self.gesture_clear_start_time = 0.0 
                    self.get_logger().info(f"[GESTURE CLEAR] Debounce complete. Resuming previous state: {self.robot_state.upper()}.")
                    
                    twist_msg.linear.x = 0.0
                    twist_msg.angular.z = 0.0
                    
                else:
                    # Debouncing in progress, keep sending stop
                    twist_msg.linear.x = 0.0
                    twist_msg.angular.z = 0.0
                    self.get_logger().info(f"[OPEN_PALM_STOP] Debouncing... {self.GESTURE_CLEAR_HOLD_TIME - (current_time - self.gesture_clear_start_time):.2f}s left. L_X: 0.00, A_Z: 0.00")

            self.cmd_vel_publisher.publish(twist_msg)
            return


        # ---------------------------------------------------------------------
        # REGULAR STATE MACHINE LOGIC
        # ---------------------------------------------------------------------
        
        # --- TRACKING STATE ---
        if robot_state == "tracking":
            with self.data_lock:
                target_detected = self.target_detected
                hip_midpoint_x = self.hip_midpoint_x
                person_normalized_size = self.person_normalized_size
                current_target_id = self.target_track_id

            if target_detected:
                # Normal Tracking Logic
                angular_error = 0.5 - hip_midpoint_x
                angular_gain = 0.7
                twist_msg.angular.z = angular_error * angular_gain

                linear_x = 0.0
                if person_normalized_size > self.SIZE_THRESHOLD_CLOSE: linear_x = -0.15
                elif person_normalized_size < self.SIZE_THRESHOLD_FAR and person_normalized_size > 0: linear_x = 0.45
                
                if linear_x != 0.0:
                    reduction = abs(angular_error) * 0.8
                    twist_msg.linear.x = max(0.0, linear_x - reduction) if linear_x > 0 else min(0.0, linear_x + reduction)
                
                if twist_msg.linear.x != 0.0 or twist_msg.angular.z != 0.0:
                    self.get_logger().info(f"[TRACKING ID: {current_target_id}] Moving L_X: {twist_msg.linear.x:.2f}, A_Z: {twist_msg.angular.z:.2f}")
                
            else:
                # Target lost -> State change to reacquire already handled by image thread. Stop here.
                self.get_logger().info(f"[TRACKING] Target lost. Sending STOP.")
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 0.0

        # --- REACQUIRE TARGET STATE (RESTORED) ---
        elif robot_state == "reacquire_target":
            
            with self.data_lock:
                last_target_id = self.last_target_track_id
            
            # Check if target was reacquired is handled in image thread.
            
            elapsed_time = time.time() - self.reacquire_start_time
            linear_speed_default, angular_speed_default = 0.15, 0.5
            
            # Define movements and durations
            step_actions = {
                1: ("Moving Backward", 'linear', -linear_speed_default, 1.0),
                2: ("Turning Left", 'angular', angular_speed_default, self.reacquire_duration_per_step),
                3: ("Moving Forward", 'linear', 0.5, 2.0),
                4: ("Turning Right", 'angular', -angular_speed_default, self.reacquire_duration_per_step * 2),
                5: ("Stopping/Waiting", 'stop', 0.0, 3.0)
            }
            
            current_action_tuple = step_actions.get(self.reacquire_step, ("Resetting", 'stop', 0.0, 0.0))
            current_duration = current_action_tuple[3]
            
            if current_duration > 0 and elapsed_time >= current_duration:
                with self.state_lock:
                    self.reacquire_step += 1
                    self.reacquire_start_time = time.time()
                
                next_action_tuple = step_actions.get(self.reacquire_step)
                if next_action_tuple:
                    self.get_logger().info(f"Reacquire Step {self.reacquire_step-1} complete. Moving to Step {self.reacquire_step}.")
                
            # Execute the current step's action
            current_action_tuple = step_actions.get(self.reacquire_step)
            if current_action_tuple:
                action_desc, axis, speed, _ = current_action_tuple
                if axis == 'linear':
                    twist_msg.linear.x = speed
                    twist_msg.angular.z = 0.0
                elif axis == 'angular':
                    twist_msg.linear.x = 0.0
                    twist_msg.angular.z = speed
                elif axis == 'stop':
                    twist_msg.linear.x = 0.0
                    twist_msg.angular.z = 0.0
                
                if self.reacquire_step <= 5:
                    self.get_logger().info(f"[REACQUIRE STEP {self.reacquire_step}/5] {action_desc} (ID: {last_target_id}) L_X: {twist_msg.linear.x:.2f}, A_Z: {twist_msg.angular.z:.2f}")
            else:
                # Sequence complete (self.reacquire_step >= 6)
                self.get_logger().info("Reacquire sequence complete. Target not found. Entering waiting state.")
                with self.state_lock: self.robot_state = "waiting_for_gesture"
                with self.data_lock:
                    self.target_track_id = -1
                    self.last_target_track_id = -1
                    self.last_target_bbox = None 
                self.reacquire_step = 0

        # --- WAITING FOR GESTURE STATE ---
        elif robot_state == "waiting_for_gesture":
            with self.data_lock:
                self.last_target_bbox = None
            # Stop/slow scan in waiting state
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0

        # --- CATCH-ALL/STOP STATE ---
        else:
            # Ensures a stop command is sent for any unexpected/non-movement state
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
            with self.state_lock:
                self.robot_state = "waiting_for_gesture"

        self.cmd_vel_publisher.publish(twist_msg)


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
