#!/usr/bin/env python3
"""
Storage Area Hand-Eye Calibration Tool (4-Point Teaching Method) - ROS2 Version

This tool calculates the transformation matrix ^{b}T_s (storage frame to robot base frame)
using 4 teaching points with real robot control for the storage area.

Features:
1. Load storage area corners from storage_camera_calibration.yaml
2. Use arrow keys to jog robot to each storage corner
3. Record robot position at each point
4. Calculate transformation matrix using SVD
5. Save as storage_hand_eye_calibration.yaml

Usage:
1. First run: calibrate_storage_camera.py to get storage area perspective calibration
2. Launch the robot controller: ros2 launch episode_controller robot_controller.launch.py
3. Run this node: python3 calibrate_storage_hand_eye.py
4. For each storage corner:
   - Use arrow keys to move robot (X/Y), Page Up/Down for Z
   - Press Enter to record the point
5. After 4 points, press 'c' to calculate transformation
6. Press 's' to save calibration

Keys:
- Arrow keys: Move X/Y
- Page Up/Down: Move Z
- +/-: Adjust step size
- Enter: Record current point
- R: Reset all points
- C: Calculate transformation
- S: Save calibration
- Q: Quit
"""

import sys
import numpy as np
import threading
import time
import yaml
import os
from pathlib import Path

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from robot_arm_interfaces.action import MoveXyzRotation
from robot_arm_interfaces.srv import ReadMotorAngles

from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QHBoxLayout, QWidget, QMessageBox, QGroupBox,
                             QGridLayout, QPushButton, QDoubleSpinBox, QFrame,
                             QSpinBox, QTabWidget, QCheckBox)
from PyQt5.QtGui import QFont, QColor, QPalette, QImage, QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer, QPoint, pyqtSignal

# Camera and OpenCV for live feed
import cv2

# Import YOLO
try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("  YOLO not available. Install with: pip install ultralytics torch")

# Set OpenCV backend to not conflict with Qt
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''




# Configuration file paths
STORAGE_CAMERA_CONFIG_PATH = "storage_camera_calibration.yaml"
STORAGE_HAND_EYE_CONFIG_PATH = "storage_hand_eye_calibration.yaml"


def load_storage_camera_config(config_path: str) -> dict:
    """
    Load storage area camera calibration from YAML file.
    
    Returns:
        dict with keys: storage_corners, storage_size_mm, transformation_matrix, etc.
    """
    if not os.path.exists(config_path):
        print(f" Error: Storage camera calibration file not found: {config_path}")
        print(f"   Please run calibrate_storage_camera.py first to create this file.")
        return None
    
    try:
        with open(config_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        if not yaml_data or 'storage_corners' not in yaml_data:
            print(f" Error: Invalid storage camera calibration file: {config_path}")
            return None
        
        print(f" Storage camera calibration loaded from: {config_path}")
        return yaml_data
        
    except Exception as e:
        print(f" Error loading storage camera config from {config_path}: {e}")
        return None


def load_storage_hand_eye_config(config_path: str) -> dict:
    """
    Load storage hand-eye calibration configuration from YAML file.
    
    Returns:
        dict with keys: storage_points, T_matrix, robot_points (optional)
    """
    config = {
        'storage_points': None,
        'T_matrix': None,
        'robot_points': None,
        'storage_size_mm': None,
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
            
            if yaml_data:
                # Load storage points (required)
                if 'storage_points' in yaml_data:
                    config['storage_points'] = np.array(yaml_data['storage_points'], dtype=np.float32)
                
                # Load T matrix if exists (optional)
                if 'T_matrix' in yaml_data and yaml_data['T_matrix'] is not None:
                    config['T_matrix'] = np.array(yaml_data['T_matrix'], dtype=np.float64)
                
                # Load robot points if exists (optional, for reference)
                if 'robot_points' in yaml_data and yaml_data['robot_points'] is not None:
                    config['robot_points'] = np.array(yaml_data['robot_points'], dtype=np.float32)
                
                # Load storage size
                if 'storage_size_mm' in yaml_data:
                    config['storage_size_mm'] = yaml_data['storage_size_mm']
                
                print(f" Storage hand-eye configuration loaded from: {config_path}")
        except Exception as e:
            print(f" Error loading storage hand-eye config from {config_path}: {e}")
    else:
        print(f"ℹ Storage hand-eye config file not found: {config_path}, will create new")
    
    return config


def save_storage_hand_eye_config(config_path: str, storage_points: np.ndarray, 
                                 storage_size_mm: dict, T_matrix: np.ndarray = None, 
                                 robot_points: np.ndarray = None):
    """
    Save storage hand-eye calibration configuration to YAML file.
    """
    data = {
        'storage_points': storage_points.tolist() if storage_points is not None else None,
        'storage_size_mm': storage_size_mm,
        'T_matrix': T_matrix.tolist() if T_matrix is not None else None,
        'robot_points': robot_points.tolist() if robot_points is not None else None,
        'calibration_type': 'storage_hand_eye',
        'description': 'Hand-eye calibration for storage area - transformation from storage coordinates to robot base frame',
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    print(f" Storage hand-eye configuration saved to: {config_path}")


class RobotController(Node):
    """
    Real robot controller interface using episode_controller API
    """
    
    def __init__(self):
        super().__init__('storage_hand_eye_calibrator_robot_controller')
        
        # Action client for position control
        self._move_xyz_rotation_client = ActionClient(self, MoveXyzRotation, 'move_xyz_rotation')
        
        # Service client for reading motor angles
        self._service_client = self.create_client(ReadMotorAngles, 'read_motor_angles')
        
        # Current robot position [x, y, z, rx, ry, rz] (initial guess)
        self.current_position = [260.0, 0.0, 200.0, 180.0, 0.0, 90.0]
        self.current_rotation = [180.0, 0.0, 90.0]
        
        # Movement state tracking
        self._move_completed = True
        self._last_move_success = True
        
        # Initialize to home position
        self.get_logger().info('Initializing robot to home position for storage area calibration...')
        self.move_to(0.0, -200.0, 150.0)
    
    def move_relative(self, dx: float, dy: float, dz: float):
        """
        Move robot relative to current position
        """
        new_x = self.current_position[0] + dx
        new_y = self.current_position[1] + dy
        new_z = self.current_position[2] + dz
        
        self.get_logger().info(f"Move relative: dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}")
        self.move_to(new_x, new_y, new_z)
    
    def move_to(self, x: float, y: float, z: float, speed_ratio: float = 1.0):
        """
        Move robot to absolute position
        """
        # Update target position
        self.current_position[0] = x
        self.current_position[1] = y
        self.current_position[2] = z
        
        # Create goal
        goal_msg = MoveXyzRotation.Goal()
        goal_msg.position = [x, y, z]
        goal_msg.rotation = self.current_rotation
        goal_msg.ik_mode = "xyz"
        goal_msg.speed_ratio = speed_ratio
        
        self.get_logger().info(f"Moving to: x={x:.2f}, y={y:.2f}, z={z:.2f}")
        
        # Wait for action server
        if not self._move_xyz_rotation_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('Action server not available!')
            return
        
        # Send goal
        self._move_completed = False
        send_goal_future = self._move_xyz_rotation_client.send_goal_async(
            goal_msg, feedback_callback=self._move_feedback_callback
        )
        send_goal_future.add_done_callback(self._move_response_callback)
    
    def _move_feedback_callback(self, feedback_msg):
        """Movement feedback callback"""
        fb = feedback_msg.feedback
        # Update current angles if needed
        pass
    
    def _move_response_callback(self, future):
        """Movement response callback"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warning('Move goal rejected')
            self._move_completed = True
            self._last_move_success = False
            return
        
        # Get result
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._move_result_callback)
    
    def _move_result_callback(self, future):
        """Movement result callback"""
        result = future.result().result
        self._last_move_success = result.success
        self._move_completed = True
        
        if result.success:
            self.get_logger().info(f'Move completed successfully')
            # Update actual position from result
            if len(result.final_position) == 3:
                self.current_position[0] = result.final_position[0]
                self.current_position[1] = result.final_position[1]
                self.current_position[2] = result.final_position[2]
        else:
            self.get_logger().error(f'Move failed: {result.message}')
    
    def get_current_position(self) -> list:
        """
        Get current robot position [x, y, z, rx, ry, rz]
        """
        return self.current_position.copy()
    
    def is_moving(self) -> bool:
        """Check if robot is currently moving"""
        return not self._move_completed
    
    def wait_for_move_complete(self, timeout: float = 30.0):
        """Wait for current move to complete"""
        start_time = time.time()
        while not self._move_completed and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self._move_completed


def compute_transformation_matrix(storage_points: np.ndarray, robot_points: np.ndarray) -> np.ndarray:
    """
    Compute rigid transformation matrix from storage frame to robot base frame
    using SVD (Singular Value Decomposition)
    
    Args:
        storage_points: Nx3 array of points in storage frame [x_s, y_s, z_s]
        robot_points: Nx3 array of corresponding points in robot frame [x_b, y_b, z_b]
    
    Returns:
        4x4 homogeneous transformation matrix ^{b}T_s
    """
    assert storage_points.shape == robot_points.shape
    assert storage_points.shape[0] >= 3, "Need at least 3 points"
    
    n = storage_points.shape[0]
    
    # Compute centroids
    centroid_storage = np.mean(storage_points, axis=0)
    centroid_robot = np.mean(robot_points, axis=0)
    
    # Center the points
    storage_centered = storage_points - centroid_storage
    robot_centered = robot_points - centroid_robot
    
    # Compute covariance matrix H
    H = storage_centered.T @ robot_centered
    
    # SVD decomposition
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation matrix
    R = Vt.T @ U.T
    
    # Handle reflection case (ensure proper rotation)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    t = centroid_robot - R @ centroid_storage
    
    # Build 4x4 homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T


def compute_calibration_error(T: np.ndarray, storage_points: np.ndarray, robot_points: np.ndarray) -> tuple:
    """
    Compute calibration error (reprojection error)
    
    Returns:
        (mean_error, max_error, errors_per_point)
    """
    errors = []
    for i in range(len(storage_points)):
        # Transform storage point to robot frame
        pt_storage_h = np.append(storage_points[i], 1)  # homogeneous
        pt_robot_predicted = (T @ pt_storage_h)[:3]
        
        # Compute error
        error = np.linalg.norm(pt_robot_predicted - robot_points[i])
        errors.append(error)
    
    return np.mean(errors), np.max(errors), errors


class StorageHandEyeCalibrator(QMainWindow):
    """Storage Hand-Eye Calibration Tool with PyQt5 UI - ROS2 Version"""
    
    def __init__(self, robot_node, storage_camera_config: dict, hand_eye_config: dict, 
                 camera_config_path: str, hand_eye_config_path: str):
        super().__init__()
        
        # ROS node for robot control
        self.robot = robot_node
        
        # Configuration file paths
        self.camera_config_path = camera_config_path
        self.hand_eye_config_path = hand_eye_config_path
        
        # Storage area configuration from camera calibration
        self.storage_camera_config = storage_camera_config
        
        # Storage area size - load first before using in storage points
        self.storage_size_mm = storage_camera_config['storage_size_mm']
        self.storage_width = self.storage_size_mm['width']
        self.storage_height = self.storage_size_mm['height']
        
        # Storage corners in physical coordinates (mm) - these are the teaching points
        # The 4 corners of the storage area in storage coordinate system
        self.storage_points = np.array([
            [0.0, 0.0, 0.0],                                      # Bottom-Left (origin)
            [self.storage_width, 0.0, 0.0],                      # Bottom-Right  
            [self.storage_width, self.storage_height, 0.0],      # Top-Right
            [0.0, self.storage_height, 0.0],                     # Top-Left
        ], dtype=np.float32)
        self.num_points = len(self.storage_points)
        
        # Recorded robot positions
        self.robot_points = [None] * self.num_points
        
        # Load previous robot points if available
        if (hand_eye_config['robot_points'] is not None and 
            len(hand_eye_config['robot_points']) == self.num_points):
            for i, pt in enumerate(hand_eye_config['robot_points']):
                self.robot_points[i] = list(pt)
            self.current_point_index = self.num_points  # All points already recorded
            print(f" Loaded {self.num_points} previous robot points from config")
        else:
            self.current_point_index = 0
        
        # Track which point is being edited (for modification mode)
        self.editing_point_index = None
        
        # Jog step size (mm)
        self.step_size = 5.0
        
        # Calibration result - load from config if exists
        self.T_matrix = hand_eye_config['T_matrix']
        if self.T_matrix is not None:
            print(f" Loaded previous T matrix from config")
        
        # Point colors
        self.colors = [
            QColor(255, 100, 100),  # Red
            QColor(100, 255, 100),  # Green
            QColor(100, 100, 255),  # Blue
            QColor(255, 128, 0),    # Orange
        ]
        
        self.point_names = [
            "P1 (Storage BL)", 
            "P2 (Storage BR)", 
            "P3 (Storage TR)", 
            "P4 (Storage TL)"
        ]
        
        # Interactive storage area display
        self.camera_enabled = False
        self.cap = None
        self.current_frame = None
        self.clicked_points = []  # Store clicked points for display
        
        # YOLO detection variables
        self.yolo_model = None
        self.detected_pieces = []  # Store detected pieces
        self.selected_piece_center = None  # Currently selected piece center (x, y) for tracking
        self.detection_enabled = False
        
        # Try to initialize camera for interactive display
        self.init_camera_display()
        
        # Load YOLO model if available
        self.load_yolo_model("train_yolo/best.pt")
        
        # ROS spin timer
        self.ros_timer = QTimer()
        self.ros_timer.timeout.connect(self.spin_ros)
        self.ros_timer.start(50)  # 20 Hz
        
        # Camera update timer for live feed
        if self.camera_enabled:
            self.camera_timer = QTimer()
            self.camera_timer.timeout.connect(self.update_camera_frame)
            self.camera_timer.start(33)  # ~30 FPS
        
        self.init_ui()
        self.update_display()
        self.print_instructions()
    
    def init_camera_display(self):
        """Initialize camera for interactive storage area display"""
        try:
            self.cap = cv2.VideoCapture(0)  # Default camera
            if not self.cap.isOpened():
                print(" Camera not available - interactive display disabled")
                return
            
            # Load transformation matrix from storage camera config
            if 'transformation_matrix' in self.storage_camera_config:
                self.transformation_matrix = np.array(
                    self.storage_camera_config['transformation_matrix'], dtype=np.float32)
                
                # Get output dimensions
                self.output_width = self.storage_camera_config['output_size']['width']
                self.output_height = self.storage_camera_config['output_size']['height']
                
                # Get camera resolution
                camera_width = self.storage_camera_config['camera_resolution']['width']
                camera_height = self.storage_camera_config['camera_resolution']['height']
                
                # Set camera resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
                
                self.camera_enabled = True
                print(f" Interactive storage area display enabled ({self.output_width}x{self.output_height})")
            else:
                print(" No transformation matrix found - interactive display disabled")
                
        except Exception as e:
            print(f" Camera initialization failed: {e} - interactive display disabled")
            self.camera_enabled = False
    
    def load_yolo_model(self, weights_file):
        """Load YOLO model for piece detection"""
        if not YOLO_AVAILABLE:
            print("  YOLO not available - piece detection disabled")
            return
            
        weights_path = Path(weights_file)
        if not weights_path.exists():
            # Try relative path from current script location
            script_dir = Path(__file__).parent
            weights_path = script_dir / weights_file
            
        if weights_path.exists():
            try:
                self.yolo_model = YOLO(str(weights_path))
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.detection_enabled = True
                print(f" Loaded YOLO model: {weights_path}")
                print(f"  Using device: {self.device}")
            except Exception as e:
                print(f" Error loading YOLO model: {e}")
                self.yolo_model = None
                self.detection_enabled = False
        else:
            print(f"  YOLO weights not found: {weights_path}")
            print("   Piece detection will be disabled")
            self.yolo_model = None
            self.detection_enabled = False
    
    def update_camera_frame(self):
        """Update camera frame for interactive display with YOLO detection"""
        if not self.camera_enabled or not self.cap:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Detect pieces on raw frame if YOLO is available
        if self.detection_enabled and self.yolo_model:
            self.detected_pieces = self.detect_pieces_on_raw_frame(frame)
        
        # Apply perspective transformation
        try:
            transformed = cv2.warpPerspective(
                frame, 
                self.transformation_matrix, 
                (self.output_width, self.output_height)
            )
            
            # Draw detected pieces and interaction overlay
            transformed = self.draw_interaction_overlay(transformed)
            
            # Convert to Qt image format
            h, w = transformed.shape[:2]
            rgb = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
            
            # Update display if widget exists
            if hasattr(self, 'storage_display_label'):
                pixmap = QPixmap.fromImage(qimg)
                self.storage_display_label.setPixmap(
                    pixmap.scaled(self.storage_display_label.size(), 
                                Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
            
            # Store current frame for click processing
            self.current_frame = transformed
            
        except Exception as e:
            print(f"Camera frame update error: {e}")
    
    def detect_pieces_on_raw_frame(self, frame):
        """Detect pieces using YOLO on raw camera frame and return detection data"""
        detected_pieces = []
        try:
            # Run YOLO inference on raw frame
            results = self.yolo_model(frame, conf=0.1, device=self.device)
            
            # Process detections
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls_id = int(box.cls[0].cpu().numpy())
                        
                        # Convert to integers
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Calculate center point in raw frame
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        # Set colors and labels by class
                        if cls_id == 0:  # blackstone
                            color = (0, 0, 255)  # Red for black stones
                            label = f"Black: {conf:.2f}"
                            piece_type = "black"
                        elif cls_id == 1:  # whitestone
                            color = (255, 0, 0)  # Blue for white stones
                            label = f"White: {conf:.2f}"
                            piece_type = "white"
                        else:
                            color = (0, 255, 0)  # Green for unknown
                            label = f"Unknown: {conf:.2f}"
                            piece_type = "unknown"
                        
                        # Store detection data for transformation
                        detected_pieces.append({
                            'bbox': (x1, y1, x2, y2),
                            'center': (center_x, center_y),
                            'conf': conf,
                            'cls_id': cls_id,
                            'color': color,
                            'label': label,
                            'piece_type': piece_type
                        })
            
        except Exception as e:
            print(f"Detection error: {e}")
        
        return detected_pieces
    
    def draw_interaction_overlay(self, frame):
        """Draw detected pieces, clicked points and robot targets on the frame"""
        try:
            # Draw coordinate axes in corners for reference
            self.draw_coordinate_reference(frame)
            
            # Draw detected pieces first
            if self.detected_pieces:
                self.draw_detected_pieces(frame)
            
            # Draw only the latest clicked point
            if self.clicked_points:  # Only if there are clicked points
                point_data = self.clicked_points[-1]  # Get the most recent click
                pixel_x, pixel_y = point_data['pixel']
                storage_coord = point_data['storage']
                robot_coord = point_data['robot']
                
                # Use bright cyan color for the latest click
                color = (0, 255, 255)  # Bright cyan
                
                # Draw larger, more visible click point
                cv2.circle(frame, (int(pixel_x), int(pixel_y)), 6, color, -1)           # Filled circle
                cv2.circle(frame, (int(pixel_x), int(pixel_y)), 10, (255, 255, 255), 2) # White border
                cv2.circle(frame, (int(pixel_x), int(pixel_y)), 12, (0, 0, 0), 1)       # Black outer border
                
                # Draw precision crosshair
                cv2.line(frame, (int(pixel_x) - 20, int(pixel_y)), (int(pixel_x) + 20, int(pixel_y)), (255, 255, 255), 3)
                cv2.line(frame, (int(pixel_x) - 20, int(pixel_y)), (int(pixel_x) + 20, int(pixel_y)), (0, 0, 0), 1)
                cv2.line(frame, (int(pixel_x), int(pixel_y) - 20), (int(pixel_x), int(pixel_y) + 20), (255, 255, 255), 3)
                cv2.line(frame, (int(pixel_x), int(pixel_y) - 20), (int(pixel_x), int(pixel_y) + 20), (0, 0, 0), 1)
                
                # Draw precise pixel coordinate
                cv2.putText(frame, f"P({pixel_x:.0f},{pixel_y:.0f})", (int(pixel_x) + 15, int(pixel_y) - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
                cv2.putText(frame, f"P({pixel_x:.0f},{pixel_y:.0f})", (int(pixel_x) + 15, int(pixel_y) - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Draw coordinate text with better positioning
                text_bg_color = (0, 0, 0)
                text_color = (255, 255, 255)
                
                # Storage coordinates
                text = f"S:[{storage_coord[0]:.0f},{storage_coord[1]:.0f}]"
                text_pos = (int(pixel_x) + 15, int(pixel_y) - 15)
                (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame, (text_pos[0] - 2, text_pos[1] - text_h - 2), 
                             (text_pos[0] + text_w + 2, text_pos[1] + 2), text_bg_color, -1)
                cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
                           
                # Robot coordinates
                text2 = f"R:[{robot_coord[0]:.0f},{robot_coord[1]:.0f},{robot_coord[2]:.0f}]"
                text_pos2 = (int(pixel_x) + 15, int(pixel_y) + 10)
                (text_w2, text_h2), baseline2 = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame, (text_pos2[0] - 2, text_pos2[1] - text_h2 - 2), 
                             (text_pos2[0] + text_w2 + 2, text_pos2[1] + 2), text_bg_color, -1)
                cv2.putText(frame, text2, text_pos2, cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
                
                # Draw click number (total number of clicks)
                click_num = len(self.clicked_points)
                cv2.putText(frame, str(click_num), (int(pixel_x) - 5, int(pixel_y) + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(frame, str(click_num), (int(pixel_x) - 5, int(pixel_y) + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        except Exception as e:
            print(f"Overlay drawing error: {e}")
        
        return frame
    
    def draw_coordinate_reference(self, frame):
        """Draw coordinate reference indicators in corners"""
        try:
            # Draw coordinate system reference
            h, w = frame.shape[:2]
            
            # Coordinate reference markers with corrected positions
            # Bottom-left origin indicator (0,0 in storage coordinates) - should match pixel (0, h-1)
            cv2.circle(frame, (0, h - 1), 8, (0, 255, 0), -1)  # Green circle at pixel (0, h-1) = storage (0,0)
            cv2.putText(frame, "Origin (0,0)", (12, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Bottom-right (max X, 0 Y) - should match pixel (w-1, h-1)
            cv2.circle(frame, (w - 1, h - 1), 8, (255, 255, 0), -1)  # Yellow circle at pixel (w-1, h-1) = storage (width,0)
            cv2.putText(frame, f"({self.storage_width},0)", (w - 80, h - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Top-left (0 X, max Y) - should match pixel (0, 0)
            cv2.circle(frame, (0, 0), 8, (255, 0, 255), -1)  # Magenta circle at pixel (0, 0) = storage (0,height)
            cv2.putText(frame, f"(0,{self.storage_height})", (12, 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Top-right max coordinates (max X, max Y) - should match pixel (w-1, 0)
            cv2.circle(frame, (w - 1, 0), 8, (255, 0, 0), -1)  # Red circle at pixel (w-1, 0) = storage (width,height)
            cv2.putText(frame, f"Max ({self.storage_width},{self.storage_height})", 
                       (w - 150, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw center reference points
            center_x = w // 2
            center_y = h // 2
            cv2.circle(frame, (center_x, center_y), 4, (128, 128, 128), -1)  # Gray center
            cv2.putText(frame, f"Center ({self.storage_width/2:.0f},{self.storage_height/2:.0f})", 
                       (center_x + 8, center_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Draw pixel coordinate reference
            cv2.putText(frame, f"Image: {w}x{h} pixels", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(frame, f"Storage: {self.storage_width}x{self.storage_height}mm", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Draw coordinate system axes (for debugging)
            if hasattr(self, 'debug_mode_cb') and self.debug_mode_cb.isChecked():
                # X-axis (horizontal line)
                cv2.line(frame, (0, h//2), (w-1, h//2), (0, 255, 255), 1)  # Cyan X-axis
                cv2.putText(frame, "X-axis", (w//2 - 20, h//2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                
                # Y-axis (vertical line)  
                cv2.line(frame, (w//2, 0), (w//2, h-1), (255, 128, 0), 1)  # Orange Y-axis
                cv2.putText(frame, "Y-axis", (w//2 + 5, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 128, 0), 1)
            
        except Exception as e:
            print(f"Reference drawing error: {e}")
    
    def draw_detected_pieces(self, frame):
        """Draw detected pieces on the transformed frame with selection capability"""
        try:
            for i, piece in enumerate(self.detected_pieces):
                x1, y1, x2, y2 = piece['bbox']
                center_x, center_y = piece['center']
                color = piece['color']
                label = piece['label']
                piece_type = piece['piece_type']
                
                # Transform piece center from raw camera coordinates to storage coordinates
                center_raw = np.array([[center_x, center_y]], dtype=np.float32)
                center_transformed = cv2.perspectiveTransform(
                    center_raw.reshape(-1, 1, 2), self.transformation_matrix
                ).reshape(-1, 2)
                
                tx, ty = center_transformed[0]
                
                # Check if transformed center is within storage area bounds
                if 0 <= tx < self.output_width and 0 <= ty < self.output_height:
                    # Determine if this piece is selected (based on center proximity)
                    is_selected = False
                    if self.selected_piece_center is not None:
                        # Check if this piece's center matches the selected piece (within 20 pixels)
                        selected_tx, selected_ty = self.selected_piece_center
                        distance = np.sqrt((tx - selected_tx)**2 + (ty - selected_ty)**2)
                        is_selected = (distance < 20)
                    
                    # Draw bounding box (similar to chess piece controller)
                    # Transform bbox corners from raw camera to storage coordinates
                    bbox_corners = np.array([
                        [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                    ], dtype=np.float32)
                    
                    transformed_corners = cv2.perspectiveTransform(
                        bbox_corners.reshape(-1, 1, 2), self.transformation_matrix
                    ).reshape(-1, 2)
                    
                    # Get transformed bbox bounds
                    min_x = max(0, int(np.min(transformed_corners[:, 0])))
                    max_x = min(self.output_width-1, int(np.max(transformed_corners[:, 0])))
                    min_y = max(0, int(np.min(transformed_corners[:, 1])))
                    max_y = min(self.output_height-1, int(np.max(transformed_corners[:, 1])))
                    
                    # Draw piece marker (matching chess piece controller style)
                    if is_selected:
                        # Selected piece - thick yellow border around bounding box
                        cv2.rectangle(frame, (min_x-3, min_y-3), (max_x+3, max_y+3), (0, 255, 255), 4)
                        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), color, 2)
                    else:
                        # Regular piece - thin colored bounding box
                        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), color, 2)
                    
                    # Draw center point (small circle like chess controller)
                    cv2.circle(frame, (int(tx), int(ty)), 3, (0, 255, 255), -1)
                    
                    # Draw piece label with background
                    label_text = f"{piece_type.title()}: {piece['conf']:.2f}"
                    
                    # Position label above bounding box
                    label_x = min_x
                    label_y = min_y - 5
                    
                    # Background rectangle for label
                    (text_w, text_h), baseline = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                    )
                    cv2.rectangle(frame, 
                                 (label_x, label_y - text_h - 5),
                                 (label_x + text_w, label_y), 
                                 color, -1)
                    
                    # Draw label text (white text on colored background)
                    cv2.putText(frame, label_text, (label_x, label_y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    # If selected, show coordinates
                    if is_selected:
                        # Calculate storage coordinates
                        storage_x = tx * self.storage_width / self.output_width
                        storage_y = (self.output_height - ty) * self.storage_height / self.output_height
                        
                        # Calculate robot coordinates if transformation is available
                        if self.T_matrix is not None:
                            storage_point = np.array([storage_x, storage_y, 0.0, 1.0])
                            robot_point = self.T_matrix @ storage_point
                            robot_coord_text = f"R:[{robot_point[0]:.0f},{robot_point[1]:.0f},{robot_point[2]:.0f}]"
                        else:
                            robot_coord_text = "R:[Not calibrated]"
                        
                        # Draw coordinate information
                        coord_y_offset = 25
                        storage_coord_text = f"S:[{storage_x:.0f},{storage_y:.0f}]"
                        
                        # Storage coordinates
                        cv2.putText(frame, storage_coord_text, (int(tx) + 15, int(ty) + coord_y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
                        cv2.putText(frame, storage_coord_text, (int(tx) + 15, int(ty) + coord_y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                        
                        # Robot coordinates
                        cv2.putText(frame, robot_coord_text, (int(tx) + 15, int(ty) + coord_y_offset + 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
                        cv2.putText(frame, robot_coord_text, (int(tx) + 15, int(ty) + coord_y_offset + 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                        
        except Exception as e:
            print(f"Error drawing detected pieces: {e}")

    def spin_ros(self):
        """Spin ROS node to process callbacks"""
        rclpy.spin_once(self.robot, timeout_sec=0.0)
    
    def init_ui(self):
        """Initialize UI"""
        detection_status = "with YOLO Detection" if self.detection_enabled else "(Detection Disabled)"
        self.setWindowTitle(f" Storage Area Hand-Eye Calibration {detection_status} - ROS2")
        self.setMinimumSize(1400, 700)  # Wide layout for side-by-side display
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        
        # Title
        title = QLabel(" Storage Area Hand-Eye Calibration Tool")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #FF6600; padding: 10px; background-color: #FFF3E0; border: 2px solid #FF6600; border-radius: 5px;")
        main_layout.addWidget(title)
        
        # Storage info
        info_text = QLabel(f" Storage Area: {self.storage_width}x{self.storage_height}mm | "
                          f" Teaching {self.num_points} corner points")
        info_text.setAlignment(Qt.AlignCenter)
        info_text.setStyleSheet("padding: 5px; font-size: 12px; color: #666; background-color: #F5F5F5;")
        main_layout.addWidget(info_text)
        
        # Main content: Left side (calibration) and Right side (verification)
        content_layout = QHBoxLayout()
        
        # LEFT SIDE: Calibration controls (Points and Robot Control)
        left_side_layout = QVBoxLayout()
        
        # Left side: Point status
        points_group = QGroupBox(" Storage Corner Points")
        points_group.setStyleSheet("QGroupBox { font-weight: bold; color: #FF6600; }")
        points_layout = QGridLayout()
        
        self.point_labels = []
        self.storage_coord_labels = []
        self.robot_coord_labels = []
        self.status_indicators = []
        self.update_point_buttons = []
        
        # Header
        points_layout.addWidget(QLabel("Point"), 0, 0)
        points_layout.addWidget(QLabel("Storage Coord (mm)"), 0, 1)
        points_layout.addWidget(QLabel("Robot Coord (mm)"), 0, 2)
        points_layout.addWidget(QLabel("Status"), 0, 3)
        points_layout.addWidget(QLabel("Action"), 0, 4)
        
        for i in range(self.num_points):
            row = i + 1
            
            # Point name
            name_label = QLabel(self.point_names[i])
            name_label.setStyleSheet(f"color: {self.colors[i % len(self.colors)].name()}; font-weight: bold;")
            points_layout.addWidget(name_label, row, 0)
            self.point_labels.append(name_label)
            
            # Storage coordinates
            pt = self.storage_points[i]
            storage_label = QLabel(f"[{pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f}]")
            storage_label.setFont(QFont("Courier", 9))
            points_layout.addWidget(storage_label, row, 1)
            self.storage_coord_labels.append(storage_label)
            
            # Robot coordinates (to be filled)
            robot_label = QLabel("[ -- , -- , -- ]")
            robot_label.setFont(QFont("Courier", 9))
            points_layout.addWidget(robot_label, row, 2)
            self.robot_coord_labels.append(robot_label)
            
            # Status indicator
            status = QLabel(" Pending")
            points_layout.addWidget(status, row, 3)
            self.status_indicators.append(status)
            
            # Update button (for modifying saved points)
            update_btn = QPushButton("Update")
            update_btn.setEnabled(False)
            update_btn.setStyleSheet("padding: 3px; font-size: 10px;")
            update_btn.clicked.connect(lambda checked, idx=i: self.update_point(idx))
            points_layout.addWidget(update_btn, row, 4)
            self.update_point_buttons.append(update_btn)
        
        points_group.setLayout(points_layout)
        left_side_layout.addWidget(points_group)
        
        # Right side: Robot control
        control_group = QGroupBox(" Robot Control")
        control_group.setStyleSheet("QGroupBox { font-weight: bold; color: #2196F3; }")
        control_layout = QVBoxLayout()
        
        # Current robot position
        pos_layout = QGridLayout()
        pos_layout.addWidget(QLabel("Current Robot Position:"), 0, 0, 1, 2)
        
        self.pos_x_label = QLabel("X: 260.00")
        self.pos_y_label = QLabel("Y: 0.00")
        self.pos_z_label = QLabel("Z: 200.00")
        
        for label in [self.pos_x_label, self.pos_y_label, self.pos_z_label]:
            label.setFont(QFont("Courier", 12))
        
        pos_layout.addWidget(self.pos_x_label, 1, 0)
        pos_layout.addWidget(self.pos_y_label, 1, 1)
        pos_layout.addWidget(self.pos_z_label, 2, 0)
        
        control_layout.addLayout(pos_layout)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        control_layout.addWidget(line)
        
        # Direct XYZ input for moving to absolute position
        direct_move_layout = QGridLayout()
        direct_move_layout.addWidget(QLabel("Direct Move To (mm):"), 0, 0, 1, 4)
        
        direct_move_layout.addWidget(QLabel("X:"), 1, 0)
        self.input_x = QDoubleSpinBox()
        self.input_x.setRange(-500.0, 500.0)
        self.input_x.setValue(260.0)
        self.input_x.setSingleStep(1.0)
        self.input_x.setDecimals(2)
        direct_move_layout.addWidget(self.input_x, 1, 1)
        
        direct_move_layout.addWidget(QLabel("Y:"), 1, 2)
        self.input_y = QDoubleSpinBox()
        self.input_y.setRange(-500.0, 500.0)
        self.input_y.setValue(0.0)
        self.input_y.setSingleStep(1.0)
        self.input_y.setDecimals(2)
        direct_move_layout.addWidget(self.input_y, 1, 3)
        
        direct_move_layout.addWidget(QLabel("Z:"), 2, 0)
        self.input_z = QDoubleSpinBox()
        self.input_z.setRange(-500.0, 500.0)
        self.input_z.setValue(200.0)
        self.input_z.setSingleStep(1.0)
        self.input_z.setDecimals(2)
        direct_move_layout.addWidget(self.input_z, 2, 1)
        
        self.move_to_btn = QPushButton("Move To")
        self.move_to_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 5px;")
        self.move_to_btn.clicked.connect(self.move_to_position)
        direct_move_layout.addWidget(self.move_to_btn, 2, 2, 1, 2)
        
        control_layout.addLayout(direct_move_layout)
        
        # Separator
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        control_layout.addWidget(line2)
        
        # Step size control
        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("Step Size (mm):"))
        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(0.5, 50.0)
        self.step_spin.setValue(self.step_size)
        self.step_spin.setSingleStep(0.5)
        self.step_spin.valueChanged.connect(self.on_step_changed)
        step_layout.addWidget(self.step_spin)
        control_layout.addLayout(step_layout)
        
        # Jog buttons
        jog_layout = QGridLayout()
        
        btn_y_plus = QPushButton("Y+")
        btn_y_minus = QPushButton("Y-")
        btn_x_minus = QPushButton("X-")
        btn_x_plus = QPushButton("X+")
        btn_z_plus = QPushButton("Z+")
        btn_z_minus = QPushButton("Z-")
        
        # Set button size and style
        for btn in [btn_y_plus, btn_y_minus, btn_x_minus, btn_x_plus, btn_z_plus, btn_z_minus]:
            btn.setMinimumSize(80, 40)
            btn.setStyleSheet("QPushButton { background-color: #E3F2FD; border: 1px solid #2196F3; }")
        
        btn_y_plus.clicked.connect(lambda: self.jog_robot(0, 1, 0))
        btn_y_minus.clicked.connect(lambda: self.jog_robot(0, -1, 0))
        btn_x_minus.clicked.connect(lambda: self.jog_robot(-1, 0, 0))
        btn_x_plus.clicked.connect(lambda: self.jog_robot(1, 0, 0))
        btn_z_plus.clicked.connect(lambda: self.jog_robot(0, 0, 1))
        btn_z_minus.clicked.connect(lambda: self.jog_robot(0, 0, -1))
        
        jog_layout.addWidget(btn_y_minus, 0, 1)
        jog_layout.addWidget(btn_x_plus, 1, 0)
        jog_layout.addWidget(btn_x_minus, 1, 2)
        jog_layout.addWidget(btn_y_plus, 2, 1)
        jog_layout.addWidget(btn_z_plus, 0, 3)
        jog_layout.addWidget(btn_z_minus, 2, 3)
        
        control_layout.addLayout(jog_layout)
        
        # Action buttons
        action_layout = QVBoxLayout()
        
        self.record_btn = QPushButton(" Record Point (Enter)")
        self.record_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; font-size: 14px; font-weight: bold;")
        self.record_btn.clicked.connect(self.record_point)
        action_layout.addWidget(self.record_btn)
        
        self.save_modified_btn = QPushButton("Save Modified Point")
        self.save_modified_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 10px; font-size: 14px;")
        self.save_modified_btn.clicked.connect(self.save_modified_point)
        self.save_modified_btn.setVisible(False)
        action_layout.addWidget(self.save_modified_btn)
        
        self.calc_btn = QPushButton(" Calculate T Matrix (C)")
        self.calc_btn.setStyleSheet("background-color: #FF9800; color: white; padding: 8px; font-size: 12px;")
        self.calc_btn.clicked.connect(self.calculate_transformation)
        self.calc_btn.setEnabled(False)
        action_layout.addWidget(self.calc_btn)
        
        self.save_btn = QPushButton(" Save Storage Calibration (S)")
        self.save_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px; font-size: 12px; font-weight: bold;")
        self.save_btn.clicked.connect(self.save_calibration)
        self.save_btn.setEnabled(False)
        action_layout.addWidget(self.save_btn)
        
        reset_btn = QPushButton(" Reset All (R)")
        reset_btn.setStyleSheet("background-color: #f44336; color: white; padding: 8px; font-size: 12px;")
        reset_btn.clicked.connect(self.reset_all)
        action_layout.addWidget(reset_btn)
        
        home_btn = QPushButton(" Move to Home Position")
        home_btn.setStyleSheet("background-color: #9C27B0; color: white; padding: 8px; font-size: 12px;")
        home_btn.clicked.connect(self.move_to_home)
        action_layout.addWidget(home_btn)
        
        control_layout.addLayout(action_layout)
        
        control_group.setLayout(control_layout)
        left_side_layout.addWidget(control_group)
        
        # Result display in left side
        result_container = QGroupBox(" Calibration Results")
        result_container.setStyleSheet("QGroupBox { font-weight: bold; color: #4CAF50; }")
        result_container_layout = QVBoxLayout()
        
        self.result_label = QLabel()
        self.result_label.setFont(QFont("Courier", 9))
        self.result_label.setAlignment(Qt.AlignLeft)
        self.result_label.setStyleSheet("padding: 8px; background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 3px;")
        result_container_layout.addWidget(self.result_label)
        result_container.setLayout(result_container_layout)
        left_side_layout.addWidget(result_container)
        
        # Add left side to content layout
        content_layout.addLayout(left_side_layout, stretch=1)
        
        # RIGHT SIDE: Verification area (Interactive Storage Area Display)
        if self.camera_enabled:
            right_side_layout = QVBoxLayout()
            
            display_group = QGroupBox(" Interactive Storage Area with YOLO Detection - Click to Move Robot")
            display_group.setStyleSheet("QGroupBox { font-weight: bold; color: #9C27B0; }")
            display_layout = QVBoxLayout()
            
            # Detection status
            detection_status = " YOLO Detection Enabled" if self.detection_enabled else " YOLO Detection Disabled"
            detection_color = "#4CAF50" if self.detection_enabled else "#FF9800"
            status_label = QLabel(detection_status)
            status_label.setStyleSheet(f"padding: 5px; font-size: 11px; color: {detection_color}; font-weight: bold;")
            display_layout.addWidget(status_label)
            
            # Camera display with click interaction
            self.storage_display_label = ClickableLabel()
            self.storage_display_label.setMinimumSize(400, 300)
            self.storage_display_label.setMaximumSize(800, 600)  # Larger max size for right panel
            self.storage_display_label.setAlignment(Qt.AlignCenter)  # Center the image to match coordinate calculation
            self.storage_display_label.setStyleSheet(
                "border: 2px solid #9C27B0; background-color: black;"
            )
            self.storage_display_label.clicked.connect(self.on_storage_area_clicked)
            display_layout.addWidget(self.storage_display_label)
            
            # Display controls
            controls_layout = QVBoxLayout()  # Changed to vertical for better organization
            
            # First row of controls
            controls_row1 = QHBoxLayout()
            
            self.debug_mode_cb = QCheckBox("Debug Mode")
            self.debug_mode_cb.setChecked(False)
            controls_row1.addWidget(self.debug_mode_cb)
            
            # Z offset for click-to-move
            controls_row1.addWidget(QLabel("Click Z-offset (mm):"))
            self.click_z_offset = QDoubleSpinBox()
            self.click_z_offset.setRange(-50.0, 50.0)
            self.click_z_offset.setValue(5.0)  # 5mm above storage surface
            self.click_z_offset.setSingleStep(1.0)
            controls_row1.addWidget(self.click_z_offset)
            controls_row1.addStretch()
            
            controls_layout.addLayout(controls_row1)
            
            # Second row of controls (buttons)
            controls_row2 = QHBoxLayout()
            
            controls_layout.addLayout(controls_row1)
            
            # Second row of controls (buttons)
            controls_row2 = QHBoxLayout()
            
            # Move to selected piece button (if detection enabled)
            if self.detection_enabled:
                self.move_to_piece_btn = QPushButton(" Move to Selected Piece")
                self.move_to_piece_btn.clicked.connect(self.move_to_selected_piece)
                self.move_to_piece_btn.setEnabled(False)  # Enabled when piece is selected and calibrated
                self.move_to_piece_btn.setStyleSheet(
                    "background-color: #FF5722; color: white; padding: 8px; "
                    "font-weight: bold; border-radius: 3px;"
                )
                controls_row2.addWidget(self.move_to_piece_btn)
            
            clear_clicks_btn = QPushButton("Clear Clicked Points")
            clear_clicks_btn.clicked.connect(self.clear_clicked_points)
            controls_row2.addWidget(clear_clicks_btn)
            
            # Test coordinate button for debugging
            test_coords_btn = QPushButton("Test Corner Coords")
            test_coords_btn.setToolTip("Click to test coordinate mapping at corners")
            test_coords_btn.clicked.connect(self.test_corner_coordinates)
            controls_row2.addWidget(test_coords_btn)
            controls_row2.addStretch()
            
            controls_layout.addLayout(controls_row2)
            display_layout.addLayout(controls_layout)
            
            controls_layout.addLayout(controls_row2)
            display_layout.addLayout(controls_layout)
            
            # Click coordinates display
            self.click_coords_label = QLabel("Click on storage area to see coordinates and move robot")
            self.click_coords_label.setStyleSheet("font-size: 11px; padding: 5px; background-color: #F3E5F5;")
            display_layout.addWidget(self.click_coords_label)
            
            # Piece coordinates display (if detection enabled)
            if self.detection_enabled:
                self.piece_coords_label = QLabel(" Click on a detected piece to select it")
                self.piece_coords_label.setStyleSheet(
                    "padding: 6px; font-size: 10px; background-color: #FFF3E0; "
                    "border: 1px solid #FF9800; border-radius: 3px; color: #F57C00;"
                )
                display_layout.addWidget(self.piece_coords_label)
            
            display_group.setLayout(display_layout)
            right_side_layout.addWidget(display_group)
            
            # Add right side to content layout
            content_layout.addLayout(right_side_layout, stretch=1)
        
        # Add content layout to main layout
        main_layout.addLayout(content_layout)
        
        # Status bar at bottom
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 12px; padding: 8px; background-color: #FFF3E0; border: 1px solid #FF6600; border-radius: 3px;")
        main_layout.addWidget(self.status_label)
    
    def on_storage_area_clicked(self, pos):
        """Handle mouse click on storage area display - supports both piece selection and general clicking"""
        if not self.camera_enabled:
            return
            
        # Check for piece selection first if detection is enabled
        if self.detection_enabled and self.detected_pieces:
            piece_index = self.find_clicked_piece(pos)
            if piece_index is not None:
                self.select_piece(piece_index)
                return
        
        # Regular click processing (requires transformation matrix)
        if self.T_matrix is None:
            QMessageBox.warning(self, "Warning", "Please calculate transformation matrix first for coordinate transformation!")
            return
        
        try:
            # Get click coordinates relative to the display widget
            widget_size = self.storage_display_label.size()
            
            # Get the actual pixmap size (this accounts for KeepAspectRatio scaling)
            current_pixmap = self.storage_display_label.pixmap()
            if current_pixmap is None:
                print("No pixmap available")
                return
                
            pixmap_size = current_pixmap.size()
            
            # Calculate the actual display area within the widget (centered due to KeepAspectRatio)
            widget_w = widget_size.width()
            widget_h = widget_size.height()
            pixmap_w = pixmap_size.width()
            pixmap_h = pixmap_size.height()
            
            # Calculate scaling factor (same for both dimensions due to KeepAspectRatio)
            scale_factor = min(widget_w / pixmap_w, widget_h / pixmap_h)
            
            # Calculate actual displayed image size
            display_w = pixmap_w * scale_factor
            display_h = pixmap_h * scale_factor
            
            # Calculate offset to center the image in the widget
            offset_x = (widget_w - display_w) / 2
            offset_y = (widget_h - display_h) / 2
            
            # Convert click position to image coordinates
            click_x = pos.x() - offset_x
            click_y = pos.y() - offset_y
            
            # Check if click is within the actual image bounds
            if click_x < 0 or click_y < 0 or click_x > display_w or click_y > display_h:
                print(f"Click outside image bounds: ({click_x:.1f}, {click_y:.1f})")
                return
            
            # Scale to original image coordinates
            pixel_x = (click_x / display_w) * self.output_width
            pixel_y = (click_y / display_h) * self.output_height
            
            # Convert pixel coordinates to storage coordinates
            # Note: Y-axis is flipped - pixel (0,0) is top-left, but storage (0,0) is bottom-left
            # Also note: OpenCV image coordinates have origin at top-left, storage has origin at bottom-left
            storage_x = (pixel_x / self.output_width) * self.storage_width
            storage_y = ((self.output_height - pixel_y) / self.output_height) * self.storage_height
            z_offset = self.click_z_offset.value()

            # Debug output (only if debug mode enabled)
            if hasattr(self, 'debug_mode_cb') and self.debug_mode_cb.isChecked():
                print(f" Debug: Widget click ({pos.x()}, {pos.y()}) -> Image pixel ({pixel_x:.1f}, {pixel_y:.1f})")
                print(f" Debug: Widget size {widget_w}x{widget_h}, Pixmap size {pixmap_w}x{pixmap_h}")
                print(f" Debug: Scale factor {scale_factor:.3f}, Display size {display_w:.1f}x{display_h:.1f}")
                print(f" Debug: Offset ({offset_x:.1f}, {offset_y:.1f})")
                print(f" Debug: Click in display ({click_x:.1f}, {click_y:.1f})")
                print(f" Debug: Pixel Y flipped from {pixel_y:.1f} to {self.output_height - pixel_y:.1f}")
                print(f" Debug: Storage coordinates ({storage_x:.1f}, {storage_y:.1f})")
                print(f" Debug: Expected storage bounds: X[0-{self.storage_width}], Y[0-{self.storage_height}]")
            
            # Transform to robot coordinates using T_matrix
            storage_point = np.array([storage_x, storage_y, z_offset, 1.0])
            robot_coord = (self.T_matrix @ storage_point)[:3]
            
            # Store click data
            click_data = {
                'pixel': (pixel_x, pixel_y),
                'storage': (storage_x, storage_y, z_offset),
                'robot': robot_coord
            }
            self.clicked_points.append(click_data)
            
            # Update display
            self.click_coords_label.setText(
                f" Clicked: Pixel[{pixel_x:.0f},{pixel_y:.0f}] → "
                f"Storage[{storage_x:.1f},{storage_y:.1f},{z_offset:.1f}] → "
                f"Robot[{robot_coord[0]:.1f},{robot_coord[1]:.1f},{robot_coord[2]:.1f}]"
            )
            
            print(f"\n Storage area clicked:")
            print(f"   Pixel coordinates: [{pixel_x:.1f}, {pixel_y:.1f}]")
            print(f"   Storage coordinates: [{storage_x:.1f}, {storage_y:.1f}, {z_offset:.1f}] mm")
            print(f"   Robot coordinates: [{robot_coord[0]:.1f}, {robot_coord[1]:.1f}, {robot_coord[2]:.1f}] mm")
            
            # Ask user if they want to move robot to this position
            reply = QMessageBox.question(
                self, "Move Robot", 
                f"Move robot to clicked position?\n\n"
                f"Storage: [{storage_x:.1f}, {storage_y:.1f}, {z_offset:.1f}] mm\n"
                f"Robot: [{robot_coord[0]:.1f}, {robot_coord[1]:.1f}, {robot_coord[2]:.1f}] mm",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.move_robot_to_clicked_position(robot_coord)
                
        except Exception as e:
            print(f"Click processing error: {e}")
            QMessageBox.warning(self, "Error", f"Failed to process click: {e}")
    
    def move_robot_to_clicked_position(self, robot_coord):
        """Move robot to the clicked position"""
        if self.robot.is_moving():
            QMessageBox.warning(self, "Warning", "Robot is still moving, please wait...")
            return
        
        try:
            print(f"\n Moving robot to clicked position: [{robot_coord[0]:.1f}, {robot_coord[1]:.1f}, {robot_coord[2]:.1f}]")
            
            # Move to safe position first, then to target
            print("   Step 1: Moving to safe position...")
            self.robot.move_to(0.0, -200.0, 150.0)
            
            # Schedule target move after safety move completes
            QTimer.singleShot(2000, lambda: self._move_to_clicked_target(robot_coord))
            
        except Exception as e:
            print(f"Robot move error: {e}")
            QMessageBox.warning(self, "Error", f"Failed to move robot: {e}")
    
    def find_clicked_piece(self, pos):
        """Find which piece was clicked based on position"""
        try:
            # Get click coordinates relative to the display widget (same logic as regular clicks)
            widget_size = self.storage_display_label.size()
            current_pixmap = self.storage_display_label.pixmap()
            if current_pixmap is None:
                return None
                
            pixmap_size = current_pixmap.size()
            
            # Calculate scaling and offset
            widget_w = widget_size.width()
            widget_h = widget_size.height()
            pixmap_w = pixmap_size.width()
            pixmap_h = pixmap_size.height()
            
            scale_factor = min(widget_w / pixmap_w, widget_h / pixmap_h)
            display_w = pixmap_w * scale_factor
            display_h = pixmap_h * scale_factor
            offset_x = (widget_w - display_w) / 2
            offset_y = (widget_h - display_h) / 2
            
            # Convert click position to image coordinates
            click_x = pos.x() - offset_x
            click_y = pos.y() - offset_y
            
            if click_x < 0 or click_y < 0 or click_x > display_w or click_y > display_h:
                return None
            
            # Scale to original image coordinates
            pixel_x = (click_x / display_w) * self.output_width
            pixel_y = (click_y / display_h) * self.output_height
            
            # Check which piece was clicked (inside bounding box)
            for i, piece in enumerate(self.detected_pieces):
                x1, y1, x2, y2 = piece['bbox']
                
                # Transform bbox corners from raw camera to storage coordinates
                bbox_corners = np.array([
                    [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                ], dtype=np.float32)
                
                transformed_corners = cv2.perspectiveTransform(
                    bbox_corners.reshape(-1, 1, 2), self.transformation_matrix
                ).reshape(-1, 2)
                
                # Get transformed bbox bounds
                min_x = np.min(transformed_corners[:, 0])
                max_x = np.max(transformed_corners[:, 0])
                min_y = np.min(transformed_corners[:, 1])
                max_y = np.max(transformed_corners[:, 1])
                
                # Check if click is inside the bounding box
                if min_x <= pixel_x <= max_x and min_y <= pixel_y <= max_y:
                    return i
            
            return None
            
        except Exception as e:
            print(f"Error finding clicked piece: {e}")
            return None
    
    def select_piece(self, piece_index):
        """Select a detected piece and update UI"""
        if piece_index < 0 or piece_index >= len(self.detected_pieces):
            return
            
        piece = self.detected_pieces[piece_index]
        
        # Calculate coordinates for selected piece
        center_x, center_y = piece['center']
        center_raw = np.array([[center_x, center_y]], dtype=np.float32)
        center_transformed = cv2.perspectiveTransform(
            center_raw.reshape(-1, 1, 2), self.transformation_matrix
        ).reshape(-1, 2)
        
        tx, ty = center_transformed[0]
        
        # Store the transformed center coordinates for stable tracking across frames
        self.selected_piece_center = (tx, ty)
        
        # Convert to storage coordinates
        storage_x = tx * self.storage_width / self.output_width
        storage_y = (self.output_height - ty) * self.storage_height / self.output_height
        z_offset = self.click_z_offset.value()
        
        # Calculate robot coordinates if transformation is available
        if self.T_matrix is not None:
            storage_point = np.array([storage_x, storage_y, z_offset, 1.0])
            robot_coord = (self.T_matrix @ storage_point)[:3]
        else:
            robot_coord = [0, 0, 0]
        
        # Update display
        if hasattr(self, 'piece_coords_label'):
            if self.T_matrix is not None:
                self.piece_coords_label.setText(
                    f" Selected {piece['piece_type'].title()} Piece: "
                    f"Camera[{center_x},{center_y}] → Storage[{storage_x:.1f},{storage_y:.1f},{z_offset:.1f}] → "
                    f"Robot[{robot_coord[0]:.1f},{robot_coord[1]:.1f},{robot_coord[2]:.1f}]"
                )
            else:
                self.piece_coords_label.setText(
                    f" Selected {piece['piece_type'].title()} Piece: "
                    f"Camera[{center_x},{center_y}] → Storage[{storage_x:.1f},{storage_y:.1f}] (No robot transform)"
                )
        
        print(f"\n Selected {piece['piece_type']} piece (conf: {piece['conf']:.2f})")
        print(f"   Camera coordinates: [{center_x}, {center_y}]")
        print(f"   Storage coordinates: [{storage_x:.1f}, {storage_y:.1f}, {z_offset:.1f}] mm")
        if self.T_matrix is not None:
            print(f"   Robot coordinates: [{robot_coord[0]:.1f}, {robot_coord[1]:.1f}, {robot_coord[2]:.1f}] mm")
        
        # Update button states
        self.update_piece_button_state()
    
    def move_to_selected_piece(self):
        """Move robot to the currently selected piece"""
        if self.selected_piece_center is None or self.T_matrix is None:
            QMessageBox.warning(self, "Warning", "Please select a piece and ensure calibration is complete!")
            return
            
        if self.robot.is_moving():
            QMessageBox.warning(self, "Warning", "Robot is still moving, please wait...")
            return
        
        try:
            # Find the piece matching the selected center
            selected_piece = None
            selected_tx, selected_ty = self.selected_piece_center
            
            for piece in self.detected_pieces:
                center_x, center_y = piece['center']
                center_raw = np.array([[center_x, center_y]], dtype=np.float32)
                center_transformed = cv2.perspectiveTransform(
                    center_raw.reshape(-1, 1, 2), self.transformation_matrix
                ).reshape(-1, 2)
                
                tx, ty = center_transformed[0]
                distance = np.sqrt((tx - selected_tx)**2 + (ty - selected_ty)**2)
                
                if distance < 20:  # Same threshold as drawing
                    selected_piece = piece
                    break
            
            if selected_piece is None:
                QMessageBox.warning(self, "Warning", "Selected piece not found in current detection!")
                return
            
            # Use the stored center coordinates
            tx, ty = self.selected_piece_center
            storage_x = tx * self.storage_width / self.output_width
            storage_y = (self.output_height - ty) * self.storage_height / self.output_height
            z_offset = self.click_z_offset.value()
            
            # Transform to robot coordinates
            storage_point = np.array([storage_x, storage_y, z_offset, 1.0])
            robot_coord = (self.T_matrix @ storage_point)[:3]
            
            print(f"\n Moving robot to selected {selected_piece['piece_type']} piece:")
            print(f"   Storage: [{storage_x:.1f}, {storage_y:.1f}, {z_offset:.1f}] mm")
            print(f"   Robot: [{robot_coord[0]:.1f}, {robot_coord[1]:.1f}, {robot_coord[2]:.1f}] mm")
            
            # Move to safe position first, then to target
            print("   Step 1: Moving to safe position...")
            self.robot.move_to(0.0, -200.0, 150.0)
            
            # Schedule target move after safety move completes
            QTimer.singleShot(2000, lambda: self._move_to_piece_target(robot_coord))
            
        except Exception as e:
            print(f"Robot move to piece error: {e}")
            QMessageBox.warning(self, "Error", f"Failed to move robot to piece: {e}")
    
    def _move_to_piece_target(self, robot_coord):
        """Move to piece target position after safe position reached"""
        if self.robot.is_moving():
            # Still moving, wait more
            QTimer.singleShot(500, lambda: self._move_to_piece_target(robot_coord))
            return
        
        print(f"   Step 2: Moving to piece position...")
        self.robot.move_to(robot_coord[0], robot_coord[1], robot_coord[2])
    
    def _move_to_clicked_target(self, robot_coord):
        """Move to target position after safe position reached"""
        if self.robot.is_moving():
            # Still moving, wait more
            QTimer.singleShot(500, lambda: self._move_to_clicked_target(robot_coord))
            return
        
        print(f"   Step 2: Moving to target position...")
        self.robot.move_to(robot_coord[0], robot_coord[1], robot_coord[2])
        
        # Update position display
        QTimer.singleShot(100, self.update_robot_position_display)
    
    def clear_clicked_points(self):
        """Clear all clicked points and piece selections from display"""
        self.clicked_points.clear()
        self.selected_piece_center = None
        self.click_coords_label.setText("Click on storage area to see coordinates and move robot")
        if hasattr(self, 'piece_coords_label'):
            self.piece_coords_label.setText(" Click on a detected piece to select it")
        if hasattr(self, 'move_to_piece_btn'):
            self.move_to_piece_btn.setEnabled(False)
        print(" Cleared all clicked points and piece selections")
    
    def update_piece_button_state(self):
        """Update the state of piece-related buttons based on current conditions"""
        if hasattr(self, 'move_to_piece_btn'):
            # Enable button only if a piece is selected AND calibration is complete
            self.move_to_piece_btn.setEnabled(
                self.selected_piece_center is not None and self.T_matrix is not None
            )
    
    def test_corner_coordinates(self):
        """Test coordinate transformation by simulating clicks at known positions"""
        if not self.camera_enabled:
            return
            
        print("\n Testing corner coordinate transformations:")
        print("-" * 50)
        
        # Test corner positions in pixel coordinates
        test_corners = [
            (0, self.output_height - 1, "Bottom-Left Origin"),     # Should be (0, 0) in storage
            (self.output_width - 1, self.output_height - 1, "Bottom-Right"),  # Should be (width, 0)
            (self.output_width - 1, 0, "Top-Right Max"),          # Should be (width, height)
            (0, 0, "Top-Left"),                                    # Should be (0, height)
            (self.output_width // 2, self.output_height // 2, "Center")  # Should be (width/2, height/2)
        ]
        
        for pixel_x, pixel_y, corner_name in test_corners:
            # Convert to storage coordinates (same logic as click handler)
            # Note: Y-axis is flipped - pixel (0,0) is top-left, but storage (0,0) is bottom-left
            storage_x = (pixel_x / self.output_width) * self.storage_width
            storage_y = ((self.output_height - pixel_y) / self.output_height) * self.storage_height
            
            print(f"{corner_name:15} | Pixel: ({pixel_x:3.0f}, {pixel_y:3.0f}) -> Storage: ({storage_x:6.1f}, {storage_y:6.1f})")
        
        print("-" * 50)
        print("Expected storage coordinates:")
        print(f"{'Bottom-Left':15} | Storage: (   0.0,    0.0)")
        print(f"{'Bottom-Right':15} | Storage: ({self.storage_width:6.1f},    0.0)")  
        print(f"{'Top-Right':15} | Storage: ({self.storage_width:6.1f}, {self.storage_height:6.1f})")
        print(f"{'Top-Left':15} | Storage: (   0.0, {self.storage_height:6.1f})")
        print(f"{'Center':15} | Storage: ({self.storage_width/2:6.1f}, {self.storage_height/2:6.1f})")
        print("-" * 50)
    
    def print_instructions(self):
        """Print instructions to console"""
        print("=" * 70)
        print("  STORAGE AREA HAND-EYE CALIBRATION TOOL - ROS2")
        print("=" * 70)
        print(f" Storage area size: {self.storage_width}x{self.storage_height}mm")
        print(f" Teaching points: {self.num_points} storage corners")
        print("\n Storage corner coordinates to teach:")
        for i, pt in enumerate(self.storage_points):
            print(f"  {self.point_names[i]}: [{pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f}] mm")
        print("\n Use arrow keys or buttons to jog robot, then press Enter to record each point.")
        print(" This creates the transformation from storage coordinates to robot coordinates.")
        print("-" * 70)
    
    def on_step_changed(self, value):
        """Handle step size change"""
        self.step_size = value
    
    def move_to_position(self):
        """Move robot to the position specified in input fields"""
        if self.robot.is_moving():
            QMessageBox.warning(self, "Warning", "Robot is still moving, please wait...")
            return
        
        x = self.input_x.value()
        y = self.input_y.value()
        z = self.input_z.value()
        
        print(f"\n Moving to absolute position: X={x:.2f}, Y={y:.2f}, Z={z:.2f}")
        self.robot.move_to(x, y, z)
        
        # Update display after a short delay (force update even in editing mode)
        QTimer.singleShot(100, lambda: self.update_robot_position_display(force=True))
    
    def jog_robot(self, dx_dir: int, dy_dir: int, dz_dir: int):
        """Jog robot in specified direction"""
        if self.robot.is_moving():
            self.robot.get_logger().warning('Robot is still moving, please wait...')
            return
        
        dx = dx_dir * self.step_size
        dy = dy_dir * self.step_size
        dz = dz_dir * self.step_size
        
        self.robot.move_relative(dx, dy, dz)
        
        # Update display after a short delay (force update even in editing mode)
        QTimer.singleShot(100, lambda: self.update_robot_position_display(force=True))
    
    def update_robot_position_display(self, force=False):
        """Update robot position display"""
        pos = self.robot.get_current_position()
        self.pos_x_label.setText(f"X: {pos[0]:.2f}")
        self.pos_y_label.setText(f"Y: {pos[1]:.2f}")
        self.pos_z_label.setText(f"Z: {pos[2]:.2f}")
        
        # Update input fields to reflect current position only if not in editing mode or if forced
        if self.editing_point_index is None or force:
            self.input_x.setValue(pos[0])
            self.input_y.setValue(pos[1])
            self.input_z.setValue(pos[2])
    
    def update_display(self):
        """Update all displays"""
        self.update_robot_position_display()
        
        # Update point statuses
        recorded_count = sum(1 for p in self.robot_points if p is not None)
        
        for i in range(self.num_points):
            if self.robot_points[i] is not None:
                pt = self.robot_points[i]
                self.robot_coord_labels[i].setText(f"[{pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f}]")
                
                # Check if this point is being edited
                if self.editing_point_index == i:
                    self.status_indicators[i].setText(" Editing")
                    self.status_indicators[i].setStyleSheet("color: orange; font-weight: bold;")
                else:
                    self.status_indicators[i].setText(" Recorded")
                    self.status_indicators[i].setStyleSheet("color: green; font-weight: bold;")
                
                # Enable update button for recorded points
                self.update_point_buttons[i].setEnabled(True)
            elif i == self.current_point_index:
                self.robot_coord_labels[i].setText("[ -- , -- , -- ]")
                self.status_indicators[i].setText(" Current")
                self.status_indicators[i].setStyleSheet("color: blue; font-weight: bold;")
                self.update_point_buttons[i].setEnabled(False)
            else:
                self.robot_coord_labels[i].setText("[ -- , -- , -- ]")
                self.status_indicators[i].setText(" Pending")
                self.status_indicators[i].setStyleSheet("color: gray;")
                self.update_point_buttons[i].setEnabled(False)
        
        # Update status label
        if self.editing_point_index is not None:
            pt = self.storage_points[self.editing_point_index]
            self.status_label.setText(
                f" Editing {self.point_names[self.editing_point_index]} - "
                f"Adjust robot position for storage position [{pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f}] mm, then press 'Save Modified Point'"
            )
            self.record_btn.setVisible(False)
            self.save_modified_btn.setVisible(True)
        elif self.current_point_index < self.num_points:
            pt = self.storage_points[self.current_point_index]
            self.status_label.setText(
                f" Teaching {self.point_names[self.current_point_index]} - "
                f"Move robot to storage position [{pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f}] mm, then press Enter"
            )
            self.record_btn.setVisible(True)
            self.save_modified_btn.setVisible(False)
        else:
            self.status_label.setText(" All storage corners recorded! Press 'C' to calculate transformation matrix")
            self.record_btn.setVisible(True)
            self.save_modified_btn.setVisible(False)
        
        # Enable/disable buttons
        all_recorded = all(p is not None for p in self.robot_points)
        self.calc_btn.setEnabled(all_recorded)
        self.save_btn.setEnabled(self.T_matrix is not None)
    
    def update_point(self, point_index: int):
        """Load a recorded point into control panel for modification"""
        if self.robot_points[point_index] is None:
            return
        
        if self.robot.is_moving():
            QMessageBox.warning(self, "Warning", "Robot is still moving, please wait...")
            return
        
        # Enter editing mode
        self.editing_point_index = point_index
        
        # Load the saved position into input fields (without moving robot)
        pt = self.robot_points[point_index]
        name = self.point_names[point_index]
        print(f"\n Updating {name}: Loaded position [{pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f}]")
        
        # Update input fields with saved position
        self.input_x.setValue(pt[0])
        self.input_y.setValue(pt[1])
        self.input_z.setValue(pt[2])
        
        # Update display
        QTimer.singleShot(100, self.update_display)
    
    def save_modified_point(self):
        """Save the modified position for the current editing point"""
        if self.editing_point_index is None:
            return
        
        if self.robot.is_moving():
            QMessageBox.warning(self, "Warning", "Robot is still moving, please wait...")
            return
        
        # Get current robot position
        pos = self.robot.get_current_position()
        
        # Save to the editing point
        old_pos = self.robot_points[self.editing_point_index]
        self.robot_points[self.editing_point_index] = pos[:3]  # Only xyz
        
        name = self.point_names[self.editing_point_index]
        print(f"\n Updated {name}:")
        print(f"  Old: [{old_pos[0]:.2f}, {old_pos[1]:.2f}, {old_pos[2]:.2f}]")
        print(f"  New: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        
        # Exit editing mode
        self.editing_point_index = None
        
        # Auto-recalculate transformation if all points are recorded
        if all(p is not None for p in self.robot_points):
            print("\n Auto-recalculating transformation matrix...")
            self.calculate_transformation()
        
        self.update_display()
    
    def record_point(self):
        """Record current robot position for current point"""
        if self.current_point_index >= self.num_points:
            QMessageBox.information(self, "Info", "All storage corners already recorded!")
            return
        
        if self.robot.is_moving():
            QMessageBox.warning(self, "Warning", "Robot is still moving, please wait...")
            return
        
        pos = self.robot.get_current_position()
        self.robot_points[self.current_point_index] = pos[:3]  # Only xyz
        
        print(f"\n Recorded {self.point_names[self.current_point_index]}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        
        self.current_point_index += 1
        
        if self.current_point_index < self.num_points:
            next_pt = self.storage_points[self.current_point_index]
            print(f"  Next: {self.point_names[self.current_point_index]} at storage position [{next_pt[0]:.1f}, {next_pt[1]:.1f}, {next_pt[2]:.1f}]")
        else:
            print("\n All storage corners recorded! Press 'C' to calculate transformation matrix.")
        
        self.update_display()
    
    def calculate_transformation(self):
        """Calculate transformation matrix"""
        if not all(p is not None for p in self.robot_points):
            QMessageBox.warning(self, "Warning", "Please record all storage corner points first!")
            return
        
        storage_pts = np.array(self.storage_points)
        robot_pts = np.array(self.robot_points)
        
        print("\n" + "=" * 70)
        print(" Calculating storage area transformation matrix...")
        print("=" * 70)
        
        try:
            self.T_matrix = compute_transformation_matrix(storage_pts, robot_pts)
            
            # Compute error
            mean_err, max_err, errors = compute_calibration_error(self.T_matrix, storage_pts, robot_pts)
            
            # Display result
            result_text = "Storage Transformation Matrix ^{b}T_s:\n"
            result_text += "-" * 50 + "\n"
            for row in self.T_matrix:
                result_text += f"[{row[0]:10.4f} {row[1]:10.4f} {row[2]:10.4f} {row[3]:10.4f}]\n"
            result_text += "-" * 50 + "\n"
            result_text += f"Calibration Error: Mean={mean_err:.3f}mm, Max={max_err:.3f}mm\n"
            result_text += f"Per-point errors: {[f'{e:.3f}' for e in errors]}"
            
            self.result_label.setText(result_text)
            
            # Print to console
            print("\n ^{b}T_s (Storage to Robot Base) =")
            print(self.T_matrix)
            print(f"\n Calibration Error:")
            print(f"  Mean: {mean_err:.4f} mm")
            print(f"  Max:  {max_err:.4f} mm")
            print(f"  Per-point: {errors}")
            
            # Extract rotation and translation
            R = self.T_matrix[:3, :3]
            t = self.T_matrix[:3, 3]
            
            print(f"\n Rotation matrix R:")
            print(R)
            print(f"\n Translation vector t: {t}")
            
            self.update_display()
            
            # Update button states now that calibration is complete
            self.update_piece_button_state()
            
            # Auto-save to YAML after successful calculation
            self.auto_save_calibration()
            
            if max_err > 5.0:
                QMessageBox.warning(self, "Warning", 
                    f" Calibration error is high (max={max_err:.2f}mm).\n"
                    "Consider re-teaching the storage corner points more carefully.\n"
                    f"Results auto-saved to: {self.hand_eye_config_path}")
            else:
                QMessageBox.information(self, " Success", 
                    f" Storage area calibration complete!\n\n"
                    f" Mean error: {mean_err:.3f}mm\n"
                    f" Max error: {max_err:.3f}mm\n\n"
                    f" Results auto-saved to:\n{self.hand_eye_config_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f" Calculation failed: {str(e)}")
            print(f" Error: {e}")
    
    def auto_save_calibration(self):
        """Auto-save calibration result to YAML file after calculation"""
        if self.T_matrix is None:
            return
        
        # Prepare robot points array
        robot_pts = None
        if all(p is not None for p in self.robot_points):
            robot_pts = np.array(self.robot_points)
        
        # Save to YAML
        save_storage_hand_eye_config(
            self.hand_eye_config_path,
            storage_points=self.storage_points,
            storage_size_mm=self.storage_size_mm,
            T_matrix=self.T_matrix,
            robot_points=robot_pts
        )
        
        print(f"\n Auto-saved storage calibration to: {self.hand_eye_config_path}")
    
    def save_calibration(self):
        """Save calibration result to YAML file"""
        if self.T_matrix is None:
            QMessageBox.warning(self, "Warning", "Please calculate transformation first!")
            return
        
        # Prepare robot points array
        robot_pts = None
        if all(p is not None for p in self.robot_points):
            robot_pts = np.array(self.robot_points)
        
        # Save to YAML
        save_storage_hand_eye_config(
            self.hand_eye_config_path,
            storage_points=self.storage_points,
            storage_size_mm=self.storage_size_mm,
            T_matrix=self.T_matrix,
            robot_points=robot_pts
        )
        
        print(f"\n Storage calibration saved to: {self.hand_eye_config_path}")
        QMessageBox.information(self, " Saved", 
                               f" Storage calibration saved to:\n{self.hand_eye_config_path}")
    
    def reset_all(self):
        """Reset all recorded points and move robot to initial position"""
        reply = QMessageBox.question(self, " Confirm Reset", 
            "Reset all recorded storage corner points and move robot to initial position?",
            QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.robot_points = [None] * self.num_points
            self.current_point_index = 0
            self.editing_point_index = None
            self.T_matrix = None
            self.result_label.setText("")
            self.update_display()
            print("\n Reset all storage corner points")
    
    def move_to_home(self):
        """Move robot to home position to clear the view"""
        if self.robot.is_moving():
            QMessageBox.warning(self, "Warning", "Robot is still moving, please wait...")
            return
        
        # Move to a position that's out of the way but safe
        home_x = 0.0  # Default home X position
        home_y = -200.0    # Default home Y position  
        home_z = 150.0  # Higher Z to clear the view
        
        print(f"\n Moving robot to home position to clear view: [{home_x:.1f}, {home_y:.1f}, {home_z:.1f}]")
        self.robot.move_to(home_x, home_y, home_z)
        
        # Update position display
        QTimer.singleShot(100, self.update_robot_position_display)

    



    
    def keyPressEvent(self, event):
        """Handle keyboard events"""
        key = event.key()
        
        if key == Qt.Key_Up:
            self.jog_robot(0, 1, 0)
        elif key == Qt.Key_Down:
            self.jog_robot(0, -1, 0)
        elif key == Qt.Key_Left:
            self.jog_robot(-1, 0, 0)
        elif key == Qt.Key_Right:
            self.jog_robot(1, 0, 0)
        elif key == Qt.Key_PageUp:
            self.jog_robot(0, 0, 1)
        elif key == Qt.Key_PageDown:
            self.jog_robot(0, 0, -1)
        elif key == Qt.Key_Return or key == Qt.Key_Enter:
            self.record_point()
        elif key == Qt.Key_C:
            self.calculate_transformation()
        elif key == Qt.Key_S:
            self.save_calibration()
        elif key == Qt.Key_R:
            self.reset_all()
        elif key == Qt.Key_Q:
            self.close()
    
    def closeEvent(self, event):
        """Handle window close event"""
        self.ros_timer.stop()
        
        # Cleanup camera
        if hasattr(self, 'camera_timer'):
            self.camera_timer.stop()
        
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        
        event.accept()


class ClickableLabel(QLabel):
    """QLabel that emits clicked signal with mouse position"""
    
    clicked = pyqtSignal(QPoint)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(event.pos())
        super().mousePressEvent(event)


def main(args=None):
    """Main function"""
    
    print("  STORAGE AREA HAND-EYE CALIBRATION TOOL")
    print("=" * 70)
    
    # Load storage camera calibration (required)
    storage_camera_config = load_storage_camera_config(STORAGE_CAMERA_CONFIG_PATH)
    if storage_camera_config is None:
        print(f"\n Cannot proceed without storage camera calibration.")
        print(f"   Please run: python3 calibrate_storage_camera.py")
        print(f"   This will create: {STORAGE_CAMERA_CONFIG_PATH}")
        sys.exit(1)
    
    # Load storage hand-eye configuration (optional, will create if not exists)
    hand_eye_config = load_storage_hand_eye_config(STORAGE_HAND_EYE_CONFIG_PATH)
    
    print(f"\n Configuration:")
    print(f"   Storage camera config: {STORAGE_CAMERA_CONFIG_PATH} ")
    print(f"   Storage hand-eye config: {STORAGE_HAND_EYE_CONFIG_PATH}")
    print(f"   Storage size: {storage_camera_config['storage_size_mm']['width']}x{storage_camera_config['storage_size_mm']['height']}mm")
    print(f"   Storage corners: {len(storage_camera_config['storage_corners'])} points")
    print(f"   T matrix: {'Loaded' if hand_eye_config['T_matrix'] is not None else 'Not available'}")
    
    # Initialize ROS2
    rclpy.init(args=args)
    
    # Create robot controller node
    robot_node = RobotController()
    
    # Wait for robot to initialize
    print("⏳ Waiting for robot to initialize...")
    robot_node.wait_for_move_complete(timeout=10.0)
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create calibration window
    window = StorageHandEyeCalibrator(robot_node, storage_camera_config, hand_eye_config, 
                                      STORAGE_CAMERA_CONFIG_PATH, STORAGE_HAND_EYE_CONFIG_PATH)
    window.show()
    
    # Run Qt event loop
    exit_code = app.exec_()
    
    # Cleanup
    robot_node.destroy_node()
    rclpy.shutdown()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()