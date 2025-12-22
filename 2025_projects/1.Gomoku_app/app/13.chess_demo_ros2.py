#!/usr/bin/env python3
"""
Chessboard & Storage Display Demo - ROS2 Version

This tool displays both chessboard and storage areas with YOLO piece detection.
Uses ROS2 actions/services for robot control instead of TCP API.

Features:
- Live chessboard feed with perspective transformation
- Live storage area feed with perspective transformation
- YOLO-based piece detection (black/white) - single model for both views
- Real-time coordinate mapping to board grid
- ROS2-based robot control (MoveXyzRotation action, GripperControl service)
- Gomoku game with AI opponent

Prerequisites:
1. Camera calibration: python3 calibrate_camera.py
2. Storage camera calibration: python3 8.calibrate_storage_camera.py
3. YOLO model: ./calibration_scripts/train_yolo/best.pt
4. ROS2 robot controller running: ros2 run episode_controller controller

Usage:
    python3 13.chess_demo_ros2.py

Controls:
- Press 'q' to quit
- Press 'g' to toggle grid
- Press 'p' to print virtual board status
- Press 'o' to test auto-fill pick and place
"""

import cv2
import numpy as np
import yaml
import os
from pathlib import Path
import sys
import socket
import json
import time
import threading

# Import ROS2
try:
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from robot_arm_interfaces.action import MoveXyzRotation
    from robot_arm_interfaces.srv import GripperControl
    from visualization_msgs.msg import Marker, MarkerArray
    from std_msgs.msg import ColorRGBA
    from geometry_msgs.msg import Point, Pose, Vector3
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("WARNING: ROS2 not available. Robot control disabled.")

# Add Gomoku_app to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Gomoku_app'))
try:
    from core_api_demo import GomokuCoreAPI
    GOMOKU_AVAILABLE = True
except ImportError:
    GOMOKU_AVAILABLE = False
    print("WARNING: GomokuCoreAPI not available. Gomoku game disabled.")

# Import YOLO
try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("WARNING: YOLO not available. Install with: pip install ultralytics torch")

# Import Qt
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                                  QHBoxLayout, QWidget, QPushButton, QComboBox,
                                  QGroupBox, QFrame)
    from PyQt5.QtGui import QPixmap, QPainter, QColor, QBrush, QPen, QFont
    from PyQt5.QtCore import Qt, QTimer
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False
    print("WARNING: PyQt5 not available. Install with: pip install PyQt5")


class ChessboardDisplay:
    """Simple chessboard and storage display with YOLO detection"""
    
    # Display configuration (adjust this to change window size)
    display_width = 800
    virtual_board_size = 600  # Size of virtual board display
    
    def __init__(self, camera_config_path, storage_config_path, yolo_weights_path,
                 storage_hand_eye_path=None, chessboard_hand_eye_path=None):
        """
        Initialize chessboard and storage display
        
        Args:
            camera_config_path: Path to camera calibration YAML file (chessboard)
            storage_config_path: Path to storage camera calibration YAML file
            yolo_weights_path: Path to YOLO model weights
            storage_hand_eye_path: Path to storage hand-eye calibration YAML
            chessboard_hand_eye_path: Path to chessboard hand-eye calibration YAML
        """
        # Board configuration
        self.board_rows = 11
        self.board_cols = 13
        
        # Virtual board status (0=empty, 1=black, -1=white)
        # This is the controllable state - update via update_virtual_chessboard()
        self.virtual_board_status = [[0 for _ in range(self.board_cols)] for _ in range(self.board_rows)]
        
        # Last detected pieces from camera (for mirroring)
        self.detected_board_state = [[0 for _ in range(self.board_cols)] for _ in range(self.board_rows)]
        
        # Running flag for threading
        self.running = True
        
        # Auto-fill test position index (for 'o' key)
        self.next_place_index = 0
        
        # Robot execution thread control
        self.robot_busy = False
        
        # ============================================================
        # ROBOT CONFIGURATION - Modify these values as needed
        # ============================================================
        self.home_x = 0.0
        self.home_y = -200.0
        self.home_z = 150.0
        self.pickup_height = 30.0
        self.place_height = 30.0
        self.pickup_z_offset = 0.0
        self.place_z_offset = 0.0
        # ============================================================
        
        # Robot state
        self.robot_enabled = False
        self.current_position = [self.home_x, self.home_y, self.home_z]
        self.current_rotation = [180.0, 0.0, 90.0]
        
        # ROS2 robot controller (will be set externally)
        self.ros2_controller = None
        
        # Hand-eye calibration matrices
        self.storage_T_matrix = None
        self.chessboard_T_matrix = None
        self._load_hand_eye_calibration(storage_hand_eye_path, chessboard_hand_eye_path)
        
        # Storage detected pieces (for robot pickup)
        self.storage_detected_pieces = []
        
        # Latest camera frame (shared between display and detection)
        self.latest_frame = None
        
        # ============================================================
        # GOMOKU GAME STATE
        # ============================================================
        self.game_active = False
        self.human_first = True  # Human plays first (black)
        self.ai_depth = 4  # AI difficulty (2=easy, 4=normal, 6=hard)
        self.gomoku_api = None
        self.last_ai_move = None  # (row, col) of last AI move for highlight
        self.last_human_move = None  # (row, col) of last human move
        self.game_status_text = "Ready to play"
        self.waiting_for_human = False  # True when waiting for human move
        self.move_count = 0  # Total moves in current game
        self.move_order = {}  # Dict: (row, col) -> step number
        
        # Stability delay for human move detection
        self.stability_delay = 1.0  # Seconds to wait for stable detection
        self.last_detected_move = None  # Last detected (row, col) candidate
        self.detection_stable_since = 0  # Timestamp when detection became stable
        self.pending_human_move = None  # Confirmed stable move waiting to process
        # ============================================================
        
        # Load chessboard calibration
        self.camera_config = self._load_camera_config(camera_config_path)
        if self.camera_config is None:
            raise ValueError(f"Failed to load camera config: {camera_config_path}")
        
        # Load storage calibration
        self.storage_config = self._load_storage_config(storage_config_path)
        if self.storage_config is None:
            raise ValueError(f"Failed to load storage config: {storage_config_path}")
        
        # Setup transformation matrices
        self._setup_chessboard_transformation()
        self._setup_storage_transformation()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        
        # Set camera resolution (use storage camera resolution as reference)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.storage_camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.storage_camera_height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Load YOLO model (single instance for both views)
        self.yolo_model = None
        self.detection_enabled = False
        if YOLO_AVAILABLE:
            self._load_yolo_model(yolo_weights_path)
        
        print(f"Display Initialized:")
        print(f"  Chessboard: {self.board_rows}x{self.board_cols}, Output: {self.chess_output_width}x{self.chess_output_height}")
        print(f"  Storage: {self.storage_width}x{self.storage_height}mm, Output: {self.storage_output_width}x{self.storage_output_height}")
        print(f"  Display Width: {self.display_width}px")
        print(f"  YOLO: {'Enabled' if self.detection_enabled else 'Disabled'}")
    
    def _load_camera_config(self, config_path):
        """Load camera calibration from YAML file"""
        if not os.path.exists(config_path):
            print(f"ERROR: Camera calibration file not found: {config_path}")
            return None
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if not config or 'pixel_corners' not in config:
                print(f"ERROR: Invalid camera calibration file")
                return None
            
            return config
        except Exception as e:
            print(f"ERROR: Failed to load camera config: {e}")
            return None
    
    def _load_storage_config(self, config_path):
        """Load storage camera calibration from YAML file"""
        if not os.path.exists(config_path):
            print(f"ERROR: Storage calibration file not found: {config_path}")
            return None
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if not config or 'transformation_matrix' not in config:
                print(f"ERROR: Invalid storage calibration file")
                return None
            
            return config
        except Exception as e:
            print(f"ERROR: Failed to load storage config: {e}")
            return None
    
    def _setup_chessboard_transformation(self):
        """Setup perspective transformation matrix for chessboard"""
        # Get configuration parameters
        pixel_corners = np.array(self.camera_config['pixel_corners'], dtype=np.float32)
        self.chess_output_width = self.camera_config['output_size']['width']
        self.chess_output_height = self.camera_config['output_size']['height']
        
        # Define output corners (top-down view)
        output_corners = np.array([
            [0, self.chess_output_height],                      # Bottom-left
            [self.chess_output_width, self.chess_output_height],      # Bottom-right
            [self.chess_output_width, 0],                       # Top-right
            [0, 0]                                        # Top-left
        ], dtype=np.float32)
        
        # Calculate transformation matrix
        self.chess_transformation_matrix = cv2.getPerspectiveTransform(
            pixel_corners, output_corners
        )
        
        # Calculate grid cell sizes (intersection-based)
        self.grid_width = self.chess_output_width / (self.board_cols - 1)
        self.grid_height = self.chess_output_height / (self.board_rows - 1)
    
    def _setup_storage_transformation(self):
        """Setup perspective transformation matrix for storage area"""
        self.storage_transformation_matrix = np.array(
            self.storage_config['transformation_matrix'], dtype=np.float32
        )
        self.storage_output_width = self.storage_config['output_size']['width']
        self.storage_output_height = self.storage_config['output_size']['height']
        self.storage_camera_width = self.storage_config['camera_resolution']['width']
        self.storage_camera_height = self.storage_config['camera_resolution']['height']
        self.storage_width = self.storage_config['storage_size_mm']['width']
        self.storage_height = self.storage_config['storage_size_mm']['height']
    
    def _load_yolo_model(self, weights_path):
        """Load YOLO model for piece detection"""
        weights_file = Path(weights_path)
        if not weights_file.exists():
            # Try relative to script directory
            script_dir = Path(__file__).parent
            weights_file = script_dir / weights_path
        
        if weights_file.exists():
            try:
                self.yolo_model = YOLO(str(weights_file))
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.detection_enabled = True
                print(f"  YOLO model loaded: {weights_file} (device: {self.device})")
            except Exception as e:
                print(f"  YOLO model loading failed: {e}")
        else:
            print(f"  YOLO weights not found: {weights_file}")
    
    def _load_hand_eye_calibration(self, storage_path, chessboard_path):
        """Load hand-eye calibration matrices"""
        # Load storage hand-eye calibration
        if storage_path and os.path.exists(storage_path):
            try:
                with open(storage_path, 'r') as f:
                    config = yaml.safe_load(f)
                if config and 'T_matrix' in config:
                    self.storage_T_matrix = np.array(config['T_matrix'])
                    print(f"  Storage hand-eye loaded: {storage_path}")
            except Exception as e:
                print(f"  Failed to load storage hand-eye: {e}")
        
        # Load chessboard hand-eye calibration
        if chessboard_path and os.path.exists(chessboard_path):
            try:
                with open(chessboard_path, 'r') as f:
                    config = yaml.safe_load(f)
                if config and 'T_matrix' in config:
                    self.chessboard_T_matrix = np.array(config['T_matrix'])
                    print(f"  Chessboard hand-eye loaded: {chessboard_path}")
            except Exception as e:
                print(f"  Failed to load chessboard hand-eye: {e}")
        
        # Enable robot if both matrices are loaded
        self.robot_enabled = (self.storage_T_matrix is not None and 
                              self.chessboard_T_matrix is not None)
        print(f"  Robot control: {'Enabled' if self.robot_enabled else 'Disabled (missing hand-eye calibration)'}")
    
    # ========== Robot Control Methods (ROS2) ==========
    
    def set_ros2_controller(self, controller):
        """Set ROS2 controller for robot communication"""
        self.ros2_controller = controller
        self.robot_enabled = (controller is not None and 
                              self.storage_T_matrix is not None and 
                              self.chessboard_T_matrix is not None)
        print(f"ROS2 Robot control: {'Enabled' if self.robot_enabled else 'Disabled'}")
        
        # Set board config for RViz visualization
        if controller is not None and self.camera_config is not None and self.chessboard_T_matrix is not None:
            board_width_mm = self.camera_config['board_size_mm']['width']
            board_height_mm = self.camera_config['board_size_mm']['height']
            controller.set_board_config(
                self.board_rows,
                self.board_cols,
                board_width_mm,
                board_height_mm,
                self.chessboard_T_matrix
            )
            print(f"RViz board config set: {self.board_rows}x{self.board_cols}, {board_width_mm}x{board_height_mm}mm")
        
        # Set storage config for RViz visualization
        if controller is not None and self.storage_T_matrix is not None:
            controller.set_storage_config(
                self.storage_width,
                self.storage_height,
                self.storage_T_matrix,
                self.storage_output_width,
                self.storage_output_height,
                self.storage_transformation_matrix
            )
            print(f"RViz storage config set: {self.storage_width}x{self.storage_height}mm")
        
        # Publish initial empty board to RViz
        if controller is not None:
            self._publish_rviz_markers()
            print("Initial RViz markers published")
    
    def robot_move_to(self, x, y, z, speed_ratio=1.0):
        """Move robot to position using ROS2 action"""
        if self.ros2_controller is None:
            print("ERROR: ROS2 controller not set!")
            return False
        
        print(f"Moving to: x={x:.2f}, y={y:.2f}, z={z:.2f}")
        
        position = [float(x), float(y), float(z)]
        rotation = [float(r) for r in self.current_rotation]
        
        success = self.ros2_controller.move_xyz_sync(position, rotation, "xyz", speed_ratio)
        
        if success:
            self.current_position = [x, y, z]
        return success
    
    def robot_gripper(self, enable):
        """Control gripper using ROS2 service"""
        if self.ros2_controller is None:
            print("ERROR: ROS2 controller not set!")
            return False
        
        return self.ros2_controller.gripper_control_sync(enable)
    
    def pick_and_place_piece(self, pickup_x, pickup_y, pickup_z,
                             place_x, place_y, place_z,
                             pickup_height=50.0, place_height=50.0, speed_ratio=0.5):
        """Pick and place a piece
        
        Args:
            pickup_x, pickup_y, pickup_z: Pickup position
            place_x, place_y, place_z: Place position
            pickup_height: Height above pickup
            place_height: Height above place
            speed_ratio: Movement speed (0.0-1.0)
        
        Returns:
            bool: True if successful
        """
        print(f'Pick and place: ({pickup_x:.1f},{pickup_y:.1f},{pickup_z:.1f}) -> ({place_x:.1f},{place_y:.1f},{place_z:.1f})')
        
        try:
            # Move above pickup
            if not self.robot_move_to(pickup_x, pickup_y, pickup_z + pickup_height, speed_ratio):
                return False
            # time.sleep(0.3)
            
            # Lower to pickup
            if not self.robot_move_to(pickup_x, pickup_y, pickup_z, speed_ratio * 0.5):
                return False
            # time.sleep(0.3)
            
            # Activate vacuum
            self.robot_gripper(True)
            # time.sleep(0.5)
            
            # Lift piece
            if not self.robot_move_to(pickup_x, pickup_y, pickup_z + pickup_height, speed_ratio * 0.5):
                return False
            # time.sleep(0.3)
            
            # Move above place
            if not self.robot_move_to(place_x, place_y, place_z + place_height, speed_ratio):
                return False
            # time.sleep(0.3)
            
            # Lower to place
            if not self.robot_move_to(place_x, place_y, place_z, speed_ratio * 0.5):
                return False
            # time.sleep(0.3)
            
            # Release vacuum
            self.robot_gripper(False)
            # time.sleep(0.5)
            
            # Move back up
            if not self.robot_move_to(place_x, place_y, place_z + place_height, speed_ratio * 0.5):
                return False
            # time.sleep(0.3)
            
            # Return home
            if not self.robot_move_to(self.home_x, self.home_y, self.home_z, speed_ratio):
                return False
            
            print('Pick and place completed!')
            return True
            
        except Exception as e:
            print(f'Pick and place failed: {e}')
            return False
    
    # ========== Coordinate Transformation Methods ==========
    
    def get_storage_piece_position(self, color_char):
        """Get robot position for a storage piece
        
        Args:
            color_char: 'B' for black, 'W' for white
            
        Returns:
            tuple: (x, y, z) in robot coordinates, or None if not found
        """
        if self.storage_T_matrix is None:
            print("No storage hand-eye calibration!")
            return None
        
        if not self.storage_detected_pieces:
            print("No pieces detected in storage")
            return None
        
        # Map color to class ID (0=black, 1=white)
        target_cls = 0 if color_char.upper() == 'B' else 1
        
        # Find matching pieces (only those within storage area bounds)
        matching_pieces = []
        for piece_data in self.storage_detected_pieces:
            x1, y1, x2, y2, cls_id, conf = piece_data
            if cls_id == target_cls:
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Transform to storage view
                center_raw = np.array([[center_x, center_y]], dtype=np.float32)
                center_transformed = cv2.perspectiveTransform(
                    center_raw.reshape(-1, 1, 2), self.storage_transformation_matrix
                ).reshape(-1, 2)
                tx, ty = center_transformed[0]
                
                # IMPORTANT: Only include pieces within storage area bounds
                # This filters out chessboard pieces that are outside storage view
                if not (0 <= tx < self.storage_output_width and 0 <= ty < self.storage_output_height):
                    continue
                
                # Convert to storage mm coordinates
                storage_x = (tx / self.storage_output_width) * self.storage_width
                storage_y = ((self.storage_output_height - ty) / self.storage_output_height) * self.storage_height
                
                matching_pieces.append((storage_x, storage_y, conf))
        
        if not matching_pieces:
            color_name = "Black" if target_cls == 0 else "White"
            print(f"No {color_name} piece found in storage")
            return None
        
        # Select highest confidence piece
        matching_pieces.sort(key=lambda p: p[2], reverse=True)
        sel_x, sel_y, sel_conf = matching_pieces[0]
        
        # Transform to robot coordinates
        storage_point = np.array([sel_x, sel_y, 0.0, 1.0])
        robot_coord = (self.storage_T_matrix @ storage_point)[:3]
        
        color_name = "Black" if target_cls == 0 else "White"
        print(f"Selected {color_name} piece (conf: {sel_conf:.2f}) -> Robot: ({robot_coord[0]:.1f}, {robot_coord[1]:.1f}, {robot_coord[2]:.1f})")
        return tuple(robot_coord)
    
    def get_chessboard_position(self, row, col, z_offset=0.0):
        """Get robot position for a chessboard grid location
        
        Args:
            row: Board row (0 to board_rows-1)
            col: Board column (0 to board_cols-1)
            z_offset: Z offset in mm
            
        Returns:
            tuple: (x, y, z) in robot coordinates, or None
        """
        if self.chessboard_T_matrix is None:
            print("No chessboard hand-eye calibration!")
            return None
        
        # Get board dimensions
        board_width_mm = self.camera_config['board_size_mm']['width']
        board_height_mm = self.camera_config['board_size_mm']['height']
        
        # Calculate cell size
        cell_width = board_width_mm / (self.board_cols - 1)
        cell_height = board_height_mm / (self.board_rows - 1)
        
        # Calculate board coordinates
        board_x = col * cell_width
        board_y = row * cell_height
        
        # Transform to robot coordinates
        board_point = np.array([board_x, board_y, z_offset, 1.0])
        robot_coord = (self.chessboard_T_matrix @ board_point)[:3]
        
        print(f"Chessboard ({row}, {col}) -> Robot: ({robot_coord[0]:.1f}, {robot_coord[1]:.1f}, {robot_coord[2]:.1f})")
        return tuple(robot_coord)
    
    def execute_sequence(self, target_row, target_col, piece_color,
                         pickup_height=None, place_height=None,
                         pickup_z_offset=None, place_z_offset=None):
        """Execute pick and place sequence: pick from storage, place on chessboard
        
        Args:
            target_row: Target row on chessboard (0 to board_rows-1)
            target_col: Target column on chessboard (0 to board_cols-1)
            piece_color: 'B' for black, 'W' for white
            pickup_height: Height above pickup position
            place_height: Height above place position
            pickup_z_offset: Z offset for pickup
            place_z_offset: Z offset for place
            
        Returns:
            bool: True if successful
        """
        if not self.robot_enabled:
            print("Robot not enabled! Missing hand-eye calibration.")
            return False
        
        # Use instance defaults if not specified
        if pickup_height is None:
            pickup_height = self.pickup_height
        if place_height is None:
            place_height = self.place_height
        if pickup_z_offset is None:
            pickup_z_offset = self.pickup_z_offset
        if place_z_offset is None:
            place_z_offset = self.place_z_offset
        
        color_name = "Black" if piece_color.upper() == 'B' else "White"
        print(f"\n{'='*50}")
        print(f"Execute sequence: {color_name} piece -> ({target_row}, {target_col})")
        print(f"{'='*50}")
        
        # Step A: Get pickup position from storage
        print(f"\nStep A: Finding {color_name} piece in storage...")
        pickup_pos = self.get_storage_piece_position(piece_color)
        if pickup_pos is None:
            print(f"Failed: No {color_name} piece available in storage!")
            return False
        
        # Step B: Get place position on chessboard
        print(f"\nStep B: Calculating chessboard position ({target_row}, {target_col})...")
        place_pos = self.get_chessboard_position(target_row, target_col, place_z_offset)
        if place_pos is None:
            print("Failed: Cannot calculate chessboard position!")
            return False
        
        # Execute pick and place
        print(f"\nStep C: Executing pick and place...")
        success = self.pick_and_place_piece(
            pickup_x=pickup_pos[0],
            pickup_y=pickup_pos[1],
            pickup_z=pickup_pos[2] + pickup_z_offset,
            place_x=place_pos[0],
            place_y=place_pos[1],
            place_z=place_pos[2],
            pickup_height=pickup_height,
            place_height=place_height,
            speed_ratio=1
        )
        
        if success:
            print(f"\n✓ Successfully placed {color_name} piece at ({target_row}, {target_col})")
            # Update virtual board
            piece_value = 1 if piece_color.upper() == 'B' else -1
            self.virtual_board_status[target_row][target_col] = piece_value
            self._publish_rviz_markers()
        else:
            print(f"\n✗ Failed to place piece!")
        
        return success
    
    def detect_pieces(self, frame):
        """
        Detect pieces using YOLO on raw frame
        
        Returns:
            List of tuples: (x1, y1, x2, y2, cls_id, conf)
        """
        if not self.detection_enabled or self.yolo_model is None:
            return []
        
        detected_pieces = []
        try:
            results = self.yolo_model(frame, conf=0.8, device=self.device, verbose=False)
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    detected_pieces.append((
                        int(x1), int(y1), int(x2), int(y2), cls, conf
                    ))
        except Exception as e:
            print(f"Detection error: {e}")
        
        return detected_pieces
    
    def pixel_to_board_coords(self, x, y):
        """
        Convert pixel coordinates to board grid coordinates
        
        Args:
            x, y: Pixel coordinates in transformed view
            
        Returns:
            (row, col): Board grid coordinates
        """
        col_float = x / self.grid_width
        # Flip Y-axis: bottom-left origin
        row_float = (self.chess_output_height - y) / self.grid_height
        
        # Round to nearest intersection
        col = max(0, min(round(col_float), self.board_cols - 1))
        row = max(0, min(round(row_float), self.board_rows - 1))
        
        return row, col
    
    def draw_detections_chessboard(self, frame, detected_pieces):
        """
        Draw detected pieces on transformed chessboard frame
        
        Args:
            frame: Transformed chessboard image
            detected_pieces: List of detected pieces from raw frame
        """
        for piece_data in detected_pieces:
            x1, y1, x2, y2, cls_id, conf = piece_data
            
            # Calculate center in raw frame
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Transform center to chessboard view
            center_raw = np.array([[center_x, center_y]], dtype=np.float32)
            center_transformed = cv2.perspectiveTransform(
                center_raw.reshape(-1, 1, 2), 
                self.chess_transformation_matrix
            ).reshape(-1, 2)
            tx, ty = center_transformed[0]
            
            # Check if within bounds
            if not (0 <= tx < self.chess_output_width and 0 <= ty < self.chess_output_height):
                continue
            
            # Calculate board coordinates
            board_row, board_col = self.pixel_to_board_coords(tx, ty)
            
            # Draw bounding box (approximate)
            box_size = 30
            x1_draw = int(tx - box_size / 2)
            y1_draw = int(ty - box_size / 2)
            x2_draw = int(tx + box_size / 2)
            y2_draw = int(ty + box_size / 2)
            
            # Color based on class
            if cls_id == 0:  # Black
                color = (0, 0, 255)  # Red
                label = "Black"
            else:  # White
                color = (255, 0, 0)  # Blue
                label = "White"
            
            # Draw bounding box
            cv2.rectangle(frame, (x1_draw, y1_draw), (x2_draw, y2_draw), color, 2)
            
            # Draw center point
            cv2.circle(frame, (int(tx), int(ty)), 3, (0, 255, 255), -1)
            
            # Draw label with board coordinates
            text = f"{label} ({board_row},{board_col})"
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
            )
            cv2.rectangle(
                frame,
                (x1_draw, y1_draw - text_height - 10),
                (x1_draw + text_width, y1_draw),
                color,
                -1
            )
            cv2.putText(
                frame,
                text,
                (x1_draw, y1_draw - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )
    
    def count_pieces_in_chessboard(self, detected_pieces):
        """
        Count pieces that fall within the chessboard area
        
        Args:
            detected_pieces: List of detected pieces from raw frame
            
        Returns:
            tuple: (black_count, white_count, total_count)
        """
        black_count = 0
        white_count = 0
        
        for piece_data in detected_pieces:
            x1, y1, x2, y2, cls_id, conf = piece_data
            
            # Calculate center in raw frame
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Transform center to chessboard view
            center_raw = np.array([[center_x, center_y]], dtype=np.float32)
            center_transformed = cv2.perspectiveTransform(
                center_raw.reshape(-1, 1, 2), 
                self.chess_transformation_matrix
            ).reshape(-1, 2)
            tx, ty = center_transformed[0]
            
            # Check if within bounds
            if 0 <= tx < self.chess_output_width and 0 <= ty < self.chess_output_height:
                if cls_id == 0:  # Black
                    black_count += 1
                else:  # White
                    white_count += 1
        
        return black_count, white_count, black_count + white_count
    
    def count_pieces_in_storage(self, detected_pieces):
        """
        Count pieces that fall within the storage area
        
        Args:
            detected_pieces: List of detected pieces from raw frame
            
        Returns:
            tuple: (black_count, white_count, total_count)
        """
        black_count = 0
        white_count = 0
        
        for piece_data in detected_pieces:
            x1, y1, x2, y2, cls_id, conf = piece_data
            
            # Calculate center in raw frame
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Transform center to storage view
            center_raw = np.array([[center_x, center_y]], dtype=np.float32)
            center_transformed = cv2.perspectiveTransform(
                center_raw.reshape(-1, 1, 2), 
                self.storage_transformation_matrix
            ).reshape(-1, 2)
            tx, ty = center_transformed[0]
            
            # Check if within bounds
            if 0 <= tx < self.storage_output_width and 0 <= ty < self.storage_output_height:
                if cls_id == 0:  # Black
                    black_count += 1
                else:  # White
                    white_count += 1
        
        return black_count, white_count, black_count + white_count
    
    def draw_detections_storage(self, frame, detected_pieces):
        """
        Draw detected pieces on transformed storage frame
        
        Args:
            frame: Transformed storage image
            detected_pieces: List of detected pieces from raw frame
        """
        for piece_data in detected_pieces:
            x1, y1, x2, y2, cls_id, conf = piece_data
            
            # Calculate center in raw frame
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Transform center to storage view
            center_raw = np.array([[center_x, center_y]], dtype=np.float32)
            center_transformed = cv2.perspectiveTransform(
                center_raw.reshape(-1, 1, 2), 
                self.storage_transformation_matrix
            ).reshape(-1, 2)
            tx, ty = center_transformed[0]
            
            # Check if within bounds
            if not (0 <= tx < self.storage_output_width and 0 <= ty < self.storage_output_height):
                continue
            
            # Draw bounding box (approximate)
            box_size = 30
            x1_draw = int(tx - box_size / 2)
            y1_draw = int(ty - box_size / 2)
            x2_draw = int(tx + box_size / 2)
            y2_draw = int(ty + box_size / 2)
            
            # Color based on class
            if cls_id == 0:  # Black
                color = (0, 0, 255)  # Red
                label = "Black"
            else:  # White
                color = (255, 0, 0)  # Blue
                label = "White"
            
            # Draw bounding box
            cv2.rectangle(frame, (x1_draw, y1_draw), (x2_draw, y2_draw), color, 2)
            
            # Draw center point
            cv2.circle(frame, (int(tx), int(ty)), 3, (0, 255, 255), -1)
            
            # Draw label
            text = f"{label} ({conf:.2f})"
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
            )
            cv2.rectangle(
                frame,
                (x1_draw, y1_draw - text_height - 10),
                (x1_draw + text_width, y1_draw),
                color,
                -1
            )
            cv2.putText(
                frame,
                text,
                (x1_draw, y1_draw - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )
    
    def draw_grid_overlay(self, frame):
        """Draw grid overlay on chessboard frame"""
        # Draw vertical lines
        for col in range(self.board_cols):
            x = int(col * self.grid_width)
            cv2.line(frame, (x, 0), (x, self.chess_output_height), (0, 255, 0), 1)
        
        # Draw horizontal lines
        for row in range(self.board_rows):
            y = int(row * self.grid_height)
            cv2.line(frame, (0, y), (self.chess_output_width, y), (0, 255, 0), 1)
        
        # Draw row/col labels
        for col in range(self.board_cols):
            x = int(col * self.grid_width)
            cv2.putText(frame, str(col), (x - 5, 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        for row in range(self.board_rows):
            y = int((self.board_rows - 1 - row) * self.grid_height)
            cv2.putText(frame, str(row), (5, y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    def process_frame(self, show_grid=False):
        """
        Process single camera frame for both views
        
        Args:
            show_grid: Whether to show grid overlay on chessboard
            
        Returns:
            Tuple of (chessboard_frame, storage_frame) or (None, None) if failed
        """
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        
        # Store latest frame for use by other methods
        self.latest_frame = frame
        
        # Detect pieces on raw frame (single detection for both views)
        detected_pieces = self.detect_pieces(frame)
        
        # Update detected board state (for mirroring when 'M' pressed)
        self.update_detected_board_state(detected_pieces)
        
        # Store detected pieces for robot pickup
        self.storage_detected_pieces = detected_pieces
        
        # Process chessboard view
        chess_transformed = cv2.warpPerspective(
            frame,
            self.chess_transformation_matrix,
            (self.chess_output_width, self.chess_output_height)
        )
        self.draw_detections_chessboard(chess_transformed, detected_pieces)
        if show_grid:
            self.draw_grid_overlay(chess_transformed)
        
        # Count pieces in chessboard area
        chess_black, chess_white, chess_total = self.count_pieces_in_chessboard(detected_pieces)
        
        # Add status text to chessboard
        status = f"Chessboard | YOLO: {'ON' if self.detection_enabled else 'OFF'} | B:{chess_black} W:{chess_white} Total:{chess_total}"
        cv2.putText(
            chess_transformed,
            status,
            (10, self.chess_output_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Process storage view
        storage_transformed = cv2.warpPerspective(
            frame,
            self.storage_transformation_matrix,
            (self.storage_output_width, self.storage_output_height)
        )
        self.draw_detections_storage(storage_transformed, detected_pieces)
        
        # Count pieces in storage area
        storage_black, storage_white, storage_total = self.count_pieces_in_storage(detected_pieces)
        
        # Add status text to storage
        status = f"Storage | B:{storage_black} W:{storage_white} Total:{storage_total}"
        cv2.putText(
            storage_transformed,
            status,
            (10, self.storage_output_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        return chess_transformed, storage_transformed
    
    def refresh_detection(self):
        """Refresh piece detection by reading fresh frames from camera"""
        # Read multiple frames to clear camera buffer and get fresh image
        for _ in range(3):
            ret, frame = self.cap.read()
        
        if not ret or frame is None:
            return False
        
        # Store as latest frame
        self.latest_frame = frame
        
        # Detect pieces on fresh frame
        detected_pieces = self.detect_pieces(frame)
        
        # Update storage detected pieces
        self.storage_detected_pieces = detected_pieces
        
        return True
    
    def _auto_fill_thread(self):
        """Background thread for auto-fill operation"""
        self.robot_busy = True
        print("\n[AUTO-FILL] Starting continuous fill (background thread)...")
        
        try:
            while self.running and self.next_place_index < self.board_rows * self.board_cols:
                row = self.next_place_index // self.board_cols
                col = self.next_place_index % self.board_cols
                
                # Wait a moment for main thread to update detection
                time.sleep(0.5)
                
                print(f"\n[AUTO-FILL] Placing black piece at ({row}, {col}) - index {self.next_place_index}")
                if self.execute_sequence(row, col, 'B'):
                    self.next_place_index += 1
                else:
                    print("[AUTO-FILL] Stopped: No more pieces in storage or placement failed!")
                    break
            else:
                if self.next_place_index >= self.board_rows * self.board_cols:
                    print("\n[AUTO-FILL] Board is full!")
        except Exception as e:
            print(f"[AUTO-FILL] Error: {e}")
        finally:
            self.robot_busy = False
            print("[AUTO-FILL] Thread finished.")
    
    def run(self):
        """Main display loop (OpenCV only)"""
        print("\nStarting OpenCV display...")
        print("Press 'q' to quit")
        print("Press 'g' to toggle grid overlay on chessboard")
        
        show_grid = False
        
        # Create named windows
        cv2.namedWindow('Chessboard', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Storage', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Chessboard', self.display_width, self.display_width)
        cv2.resizeWindow('Storage', self.display_width, self.display_width)
        
        # Position windows side by side
        cv2.moveWindow('Chessboard', 50, 50)
        cv2.moveWindow('Storage', self.display_width + 70, 50)
        
        while True:
            chess_frame, storage_frame = self.process_frame(show_grid=show_grid)
            
            if chess_frame is None or storage_frame is None:
                print("Failed to capture frame")
                break
            
            # Display frames
            cv2.imshow('Chessboard', chess_frame)
            cv2.imshow('Storage', storage_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('g'):
                show_grid = not show_grid
                print(f"Grid overlay: {'ON' if show_grid else 'OFF'}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
    
    def run_opencv_loop(self):
        """OpenCV display loop for threading (no camera release)"""
        print("\nStarting OpenCV windows...")
        print("Press 'q' in OpenCV window to quit")
        print("Press 'g' to toggle grid overlay")
        
        show_grid = False
        
        # Create named windows
        cv2.namedWindow('Chessboard', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Storage', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Chessboard', self.display_width, self.display_width)
        cv2.resizeWindow('Storage', self.display_width, self.display_width)
        
        # Position windows
        cv2.moveWindow('Chessboard', 50, 50)
        cv2.moveWindow('Storage', self.display_width + 70, 50)
        
        while self.running:
            chess_frame, storage_frame = self.process_frame(show_grid=show_grid)
            
            if chess_frame is None or storage_frame is None:
                continue
            
            cv2.imshow('Chessboard', chess_frame)
            cv2.imshow('Storage', storage_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                break
            elif key == ord('g'):
                show_grid = not show_grid
                print(f"Grid overlay: {'ON' if show_grid else 'OFF'}")
        
        cv2.destroyAllWindows()
    
    def update_detected_board_state(self, detected_pieces):
        """Update detected board state from camera pieces (internal use)
        
        Args:
            detected_pieces: List of detected pieces from YOLO
        """
        # Reset detected state
        self.detected_board_state = [[0 for _ in range(self.board_cols)] for _ in range(self.board_rows)]
        
        # Map detected pieces to board
        for piece_data in detected_pieces:
            x1, y1, x2, y2, cls_id, conf = piece_data
            
            # Calculate center
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Transform to chessboard view
            center_raw = np.array([[center_x, center_y]], dtype=np.float32)
            center_transformed = cv2.perspectiveTransform(
                center_raw.reshape(-1, 1, 2), 
                self.chess_transformation_matrix
            ).reshape(-1, 2)
            tx, ty = center_transformed[0]
            
            # Check if within bounds
            if not (0 <= tx < self.chess_output_width and 0 <= ty < self.chess_output_height):
                continue
            
            # Get board coordinates
            board_row, board_col = self.pixel_to_board_coords(tx, ty)
            
            # Update detected state (1=black, -1=white)
            piece_value = 1 if cls_id == 0 else -1
            self.detected_board_state[board_row][board_col] = piece_value
    
    def update_virtual_chessboard(self, status=None):
        """Update virtual chessboard with given status
        
        Args:
            status: 2D list [row][col] with values:
                    0 = empty
                    1 = black piece
                   -1 = white piece
                   If None, keeps current status unchanged.
        """
        if status is not None:
            # Validate and copy status
            if len(status) == self.board_rows and len(status[0]) == self.board_cols:
                self.virtual_board_status = [row[:] for row in status]
            else:
                print(f"ERROR: Invalid status size. Expected {self.board_rows}x{self.board_cols}")
    
    def mirror_to_virtual_chessboard(self):
        """Mirror detected pieces from camera to virtual chessboard"""
        self.virtual_board_status = [row[:] for row in self.detected_board_state]
        self._publish_rviz_markers()
        print("Mirrored camera detection to virtual chessboard")
    
    def get_virtual_board_status(self):
        """Get current virtual board status
        
        Returns:
            2D list of board status (copy)
        """
        return [row[:] for row in self.virtual_board_status]
    
    def set_piece(self, row, col, piece_type):
        """Set a single piece on virtual board
        
        Args:
            row: Board row (0-10)
            col: Board column (0-12)
            piece_type: 0=empty, 1=black, -1=white
        """
        if 0 <= row < self.board_rows and 0 <= col < self.board_cols:
            self.virtual_board_status[row][col] = piece_type
            self._publish_rviz_markers()
    
    def _publish_rviz_markers(self):
        """Publish current board state to RViz"""
        if self.ros2_controller is not None and ROS2_AVAILABLE:
            try:
                self.ros2_controller.publish_chessboard_markers(
                    self.virtual_board_status,
                    self.last_ai_move,
                    self.last_human_move,
                    self.storage_detected_pieces  # Pass detected pieces for storage visualization
                )
            except Exception as e:
                print(f"RViz marker publish error: {e}")
    
    def clear_virtual_board(self):
        """Clear all pieces from virtual board"""
        self.virtual_board_status = [[0 for _ in range(self.board_cols)] for _ in range(self.board_rows)]
        self.next_place_index = 0  # Reset auto-fill position
        self._publish_rviz_markers()
        print("Virtual chessboard cleared, auto-fill reset to (0,0)")
    
    # ========== Gomoku Game Methods ==========
    
    def start_game(self):
        """Start a new Gomoku game"""
        if not GOMOKU_AVAILABLE:
            print("ERROR: GomokuCoreAPI not available!")
            return False
        
        # Initialize Gomoku API with board size
        self.gomoku_api = GomokuCoreAPI(
            rows=self.board_rows,
            cols=self.board_cols,
            ai_depth=self.ai_depth
        )
        
        # Set player roles based on who goes first
        if self.human_first:
            self.gomoku_api.human_role = 1   # BLACK
            self.gomoku_api.ai_role = -1     # WHITE
        else:
            self.gomoku_api.human_role = -1  # WHITE
            self.gomoku_api.ai_role = 1      # BLACK
        
        # Clear virtual board
        self.clear_virtual_board()
        
        # Reset game state
        self.game_active = True
        self.last_ai_move = None
        self.last_human_move = None
        self.waiting_for_human = False
        self.move_count = 0  # Reset move counter
        self.move_order = {}  # Reset move order tracking
        
        # Reset stability detection
        self.last_detected_move = None
        self.detection_stable_since = 0
        self.pending_human_move = None
        
        print(f"\n{'='*50}")
        print(f"GOMOKU GAME STARTED!")
        print(f"Human: {'BLACK (first)' if self.human_first else 'WHITE (second)'}")
        print(f"AI Depth: {self.ai_depth}")
        print(f"Board: {self.board_rows}x{self.board_cols}")
        print(f"Stability Delay: {self.stability_delay}s")
        print(f"{'='*50}\\n")
        
        if self.human_first:
            # Human goes first - wait for human move
            self.waiting_for_human = True
            self.game_status_text = "Your turn! Place a black piece"
            print("Waiting for human to place first piece...")
        else:
            # AI goes first - start AI turn
            self.game_status_text = "AI thinking..."
            threading.Thread(target=self._ai_turn_thread, daemon=True).start()
        
        return True
    
    def stop_game(self):
        """Stop the current game"""
        self.game_active = False
        self.waiting_for_human = False
        self.game_status_text = "Game stopped"
        print("Game stopped.")
    
    def _detect_human_move(self):
        """Detect if human placed a new piece by comparing detected vs virtual board
        
        Returns:
            (row, col, piece_value) if new piece found, None otherwise
        """
        if not self.game_active:
            return None
        
        expected_piece = 1 if self.human_first else -1  # Human's piece value
        
        for row in range(self.board_rows):
            for col in range(self.board_cols):
                detected = self.detected_board_state[row][col]
                virtual = self.virtual_board_status[row][col]
                
                # Found new piece that matches human's color
                if detected == expected_piece and virtual == 0:
                    return (row, col, detected)
        
        return None
    
    def _process_human_move(self, row, col):
        """Process detected human move
        
        Args:
            row, col: Position of human's move
        """
        if not self.game_active or self.gomoku_api is None:
            return
        
        print(f"\n[GAME] Human placed at ({row}, {col})")
        
        # Tell Gomoku API about human move
        if not self.gomoku_api.human_move(row, col):
            print("[GAME] Invalid move! Ignoring...")
            return
        
        # Update virtual board from Gomoku API (single source of truth)
        self.virtual_board_status = self.gomoku_api.get_board_state()
        self.last_human_move = (row, col)
        self.move_count += 1  # Increment move counter
        self.move_order[(row, col)] = self.move_count  # Track move order
        
        # Publish updated board to RViz
        self._publish_rviz_markers()
        
        # Check for winner
        winner = self.gomoku_api.check_winner()
        if winner != 0:
            self._handle_game_end(winner)
            return
        
        # Check if game over (draw)
        if self.gomoku_api.is_game_over():
            self._handle_game_end(0)
            return
        
        # Start AI turn
        self.waiting_for_human = False
        self.game_status_text = "AI thinking..."
        threading.Thread(target=self._ai_turn_thread, daemon=True).start()
    
    def _ai_turn_thread(self):
        """AI turn execution in background thread"""
        if not self.game_active or self.gomoku_api is None:
            return
        
        self.robot_busy = True
        
        try:
            # Get AI move
            ai_move = self.gomoku_api.get_ai_move()
            if ai_move is None:
                print("[GAME] AI has no valid move!")
                self.robot_busy = False
                return
            
            ai_row, ai_col = ai_move
            print(f"[GAME] AI chose ({ai_row}, {ai_col})")
            
            # Determine piece color (AI's color)
            piece_color = 'W' if self.human_first else 'B'
            
            # Execute robot movement
            self.game_status_text = f"Robot placing {piece_color} at ({ai_row},{ai_col})..."
            success = self.execute_sequence(ai_row, ai_col, piece_color)
            
            if success:
                # Apply AI move to Gomoku API
                self.gomoku_api.apply_ai_move(ai_row, ai_col)
                
                # Update virtual board from Gomoku API
                self.virtual_board_status = self.gomoku_api.get_board_state()
                self.last_ai_move = (ai_row, ai_col)
                self.move_count += 1  # Increment move counter
                self.move_order[(ai_row, ai_col)] = self.move_count  # Track move order
                
                # Publish updated board to RViz
                self._publish_rviz_markers()
                
                # Check for winner
                winner = self.gomoku_api.check_winner()
                if winner != 0:
                    self._handle_game_end(winner)
                    return
                
                # Check if game over (draw)
                if self.gomoku_api.is_game_over():
                    self._handle_game_end(0)
                    return
                
                # Now wait for human move
                self.waiting_for_human = True
                self.game_status_text = "Your turn! Place a piece"
                print("[GAME] Waiting for human move...")
            else:
                self.game_status_text = "Robot failed! Press 's' to restart"
                self.game_active = False
                print("[GAME] Robot execution failed!")
        
        except Exception as e:
            print(f"[GAME] AI turn error: {e}")
            self.game_status_text = f"Error: {e}"
        finally:
            self.robot_busy = False
    
    def _handle_game_end(self, winner):
        """Handle game end
        
        Args:
            winner: 1=BLACK wins, -1=WHITE wins, 0=draw
        """
        self.game_active = False
        self.waiting_for_human = False
        
        if winner == 0:
            self.game_status_text = "Game Over: DRAW!"
            print("\n" + "="*50)
            print("GAME OVER: DRAW!")
            print("="*50 + "\n")
        else:
            human_piece = 1 if self.human_first else -1
            if winner == human_piece:
                self.game_status_text = "Game Over: YOU WIN!"
                print("\n" + "="*50)
                print("GAME OVER: HUMAN WINS!")
                print("="*50 + "\n")
            else:
                self.game_status_text = "Game Over: AI WINS!"
                print("\n" + "="*50)
                print("GAME OVER: AI WINS!")
                print("="*50 + "\n")
    
    def _check_human_move_after_robot(self):
        """Check for human move with stability delay - only accepts move after stable detection"""
        if not self.waiting_for_human or self.robot_busy:
            return
        
        current_time = time.time()
        new_move = self._detect_human_move()
        
        if new_move:
            row, col, _ = new_move
            candidate = (row, col)
            
            # Check if this is the same position as last detection
            if candidate == self.last_detected_move:
                # Same position - check if stable long enough
                time_stable = current_time - self.detection_stable_since
                
                if time_stable >= self.stability_delay:
                    # Stable for long enough - accept the move
                    if self.pending_human_move != candidate:
                        self.pending_human_move = candidate
                        print(f"[GAME] Move confirmed at ({row}, {col}) after {time_stable:.1f}s stability")
                        self._process_human_move(row, col)
                        # Reset detection state
                        self.last_detected_move = None
                        self.pending_human_move = None
                else:
                    # Still waiting for stability
                    remaining = self.stability_delay - time_stable
                    self.game_status_text = f"Confirming move at ({row},{col})... {remaining:.1f}s"
            else:
                # New/different position detected - reset stability timer
                self.last_detected_move = candidate
                self.detection_stable_since = current_time
                self.pending_human_move = None
                print(f"[GAME] Detected piece at ({row}, {col}) - waiting {self.stability_delay}s for stability...")
                self.game_status_text = f"Detected ({row},{col}) - hold steady..."
        else:
            # No new move detected
            if self.last_detected_move is not None:
                # Had a candidate but now it's gone - reset
                print(f"[GAME] Detection lost - resetting stability timer")
                self.last_detected_move = None
                self.detection_stable_since = 0
                self.pending_human_move = None
                self.game_status_text = "Your turn! Place a piece"
    
    def print_virtual_board_status(self):
        """Print virtual_board_status for debugging"""
        print("\n" + "=" * 50)
        print("virtual_board_status (row 0 at bottom):")
        print("=" * 50)
        # Print from top row to bottom (visual order)
        for row in range(self.board_rows - 1, -1, -1):
            row_str = f"Row {row:2d}: ["
            for col in range(self.board_cols):
                val = self.virtual_board_status[row][col]
                if val == 0:
                    row_str += " . "
                elif val == 1:
                    row_str += " B "  # Black
                else:
                    row_str += " W "  # White
            row_str += "]"
            print(row_str)
        print("Col:     " + " ".join(f"{i:2d}" for i in range(self.board_cols)))
        print("=" * 50)
        
        # Also print raw data
        print("\nRaw data:")
        print(f"virtual_board_status = {self.virtual_board_status}")
        print("=" * 50 + "\n")
    
    def draw_virtual_board_qt(self):
        """Draw virtual board state and return QPixmap"""
        size = self.virtual_board_size
        margin = 20
        pixmap = QPixmap(size + margin * 2, size + margin * 2)
        pixmap.fill(QColor(240, 240, 240))  # Light gray background
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw board background
        painter.setBrush(QBrush(QColor(222, 184, 135)))  # Wood color
        painter.setPen(QPen(QColor(139, 90, 43), 2))  # Brown border
        painter.drawRect(margin, margin, size, size)
        
        # Calculate cell sizes (intersection-based)
        cell_width = size / (self.board_cols - 1)
        cell_height = size / (self.board_rows - 1)
        
        # Draw grid lines
        painter.setPen(QPen(QColor(139, 90, 43), 1))  # Brown lines
        for col in range(self.board_cols):
            x = margin + col * cell_width
            painter.drawLine(int(x), margin, int(x), margin + size)
        
        for row in range(self.board_rows):
            y = margin + row * cell_height
            painter.drawLine(margin, int(y), margin + size, int(y))
        
        # Draw pieces from virtual_board_status
        piece_radius = int(max(min(cell_width, cell_height) // 3, 8))
        
        for row in range(self.board_rows):
            for col in range(self.board_cols):
                piece_value = self.virtual_board_status[row][col]
                
                if piece_value != 0:
                    # Calculate position (flip Y-axis for bottom-left origin)
                    center_x = margin + col * cell_width
                    center_y = margin + (self.board_rows - 1 - row) * cell_height
                    
                    # Check if this is last move
                    is_last_move = (self.last_ai_move == (row, col) or 
                                   self.last_human_move == (row, col))
                    
                    if piece_value == 1:  # Black
                        painter.setBrush(QBrush(Qt.black))
                        painter.setPen(QPen(Qt.darkGray, 1))
                        text_color = Qt.white
                    else:  # White
                        painter.setBrush(QBrush(Qt.white))
                        painter.setPen(QPen(Qt.black, 1))
                        text_color = Qt.black
                    
                    # Draw piece
                    painter.drawEllipse(
                        int(center_x - piece_radius), 
                        int(center_y - piece_radius), 
                        int(piece_radius * 2), 
                        int(piece_radius * 2)
                    )
                    
                    # Draw step number on piece
                    step_num = self.move_order.get((row, col), 0)
                    if step_num > 0:
                        font = painter.font()
                        # Adjust font size based on number of digits
                        if step_num < 10:
                            font.setPointSize(max(piece_radius - 2, 6))
                        elif step_num < 100:
                            font.setPointSize(max(piece_radius - 4, 5))
                        else:
                            font.setPointSize(max(piece_radius - 6, 4))
                        font.setBold(True)
                        painter.setFont(font)
                        painter.setPen(QPen(text_color, 1))
                        
                        # Center the text on the piece
                        text = str(step_num)
                        text_rect = painter.fontMetrics().boundingRect(text)
                        text_x = int(center_x - text_rect.width() / 2)
                        text_y = int(center_y + text_rect.height() / 4)
                        painter.drawText(text_x, text_y, text)
                    
                    # Draw red circle highlight on last move
                    if is_last_move:
                        painter.setBrush(Qt.NoBrush)
                        painter.setPen(QPen(QColor(255, 50, 50), 3))  # Red border
                        painter.drawEllipse(
                            int(center_x - piece_radius - 2),
                            int(center_y - piece_radius - 2),
                            int(piece_radius * 2 + 4),
                            int(piece_radius * 2 + 4)
                        )
        
        # Draw "AI thinking" indicator if robot is busy
        if self.game_active and self.robot_busy:
            painter.setPen(QPen(QColor(255, 152, 0), 2))
            font = painter.font()
            font.setPointSize(14)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(margin, size + margin - 5, "\ud83e\udd14 AI thinking...")
        
        painter.end()
        return pixmap
    
    def run_qt_display(self):
        """Run Qt display for virtual board only"""
        if not QT_AVAILABLE:
            print("ERROR: PyQt5 not available. Cannot run virtual board display.")
            return
        
        app = QApplication(sys.argv)
        
        # Create main window
        window = QMainWindow()
        window.setWindowTitle("Virtual Chessboard")
        
        # Create central widget
        central = QWidget()
        window.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Create label for board display
        board_label = QLabel()
        board_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(board_label)
        
        # Update function
        def update_display():
            if not self.cap or not self.cap.isOpened():
                return
            
            ret, frame = self.cap.read()
            if ret:
                # Detect pieces
                detected_pieces = self.detect_pieces(frame)
                
                # Update virtual board
                self.update_virtual_board(detected_pieces)
                
                # Draw and display
                pixmap = self.draw_virtual_board_qt()
                board_label.setPixmap(pixmap)
        
        # Setup timer for updates
        timer = QTimer()
        timer.timeout.connect(update_display)
        timer.start(100)  # Update every 100ms
        
        # Show window
        window.resize(self.virtual_board_size + 100, self.virtual_board_size + 100)
        window.show()
        
        # Run Qt event loop
        sys.exit(app.exec_())
    
    def run_all_displays(self):
        """Run OpenCV and Qt displays simultaneously with Qt UI controls"""
        if not QT_AVAILABLE:
            print("WARNING: PyQt5 not available. Running OpenCV only.")
            self.run()
            return
        
        print("\nStarting all displays (OpenCV + Qt)...")
        print("Press 'q' in OpenCV window to quit")
        print("Press 'g' to toggle grid overlay")
        print("Press 'p' to print virtual board status")
        print("Press 'o' to test auto-fill pick and place")
        print("Use Qt window buttons to control the game")
        
        # Initialize Qt app
        app = QApplication(sys.argv)
        
        # Create Qt window
        window = QMainWindow()
        window.setWindowTitle("Gomoku Game - Virtual Chessboard")
        
        central = QWidget()
        window.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # ========== Left Panel: Virtual Board ==========
        board_label = QLabel()
        board_label.setAlignment(Qt.AlignCenter)
        board_label.setMinimumSize(self.virtual_board_size + 50, self.virtual_board_size + 120)
        main_layout.addWidget(board_label, stretch=1)
        
        # ========== Right Panel: Controls ==========
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(220)
        
        # --- Game Control Group ---
        game_group = QGroupBox("Game Control")
        game_layout = QVBoxLayout(game_group)
        
        # First Mover selector
        first_label = QLabel("First Mover:")
        first_combo = QComboBox()
        first_combo.addItems(["Human (Black)", "AI (Black)"])
        first_combo.setCurrentIndex(0 if self.human_first else 1)
        game_layout.addWidget(first_label)
        game_layout.addWidget(first_combo)
        
        # Difficulty selector
        diff_label = QLabel("AI Difficulty:")
        diff_combo = QComboBox()
        diff_combo.addItems(["Easy (2)", "Normal (4)", "Hard (6)", "Expert (8)"])
        diff_map = {2: 0, 4: 1, 6: 2, 8: 3}
        diff_combo.setCurrentIndex(diff_map.get(self.ai_depth, 1))
        game_layout.addWidget(diff_label)
        game_layout.addWidget(diff_combo)
        
        # Start/Stop button
        start_btn = QPushButton("▶ Start Game")
        start_btn.setStyleSheet("QPushButton { font-size: 14px; padding: 10px; background-color: #4CAF50; color: white; }")
        game_layout.addWidget(start_btn)
        
        # Reset button
        reset_btn = QPushButton("↻ Reset Game")
        reset_btn.setStyleSheet("QPushButton { font-size: 12px; padding: 8px; }")
        game_layout.addWidget(reset_btn)
        
        control_layout.addWidget(game_group)
        
        # --- Status Group ---
        status_group = QGroupBox("Game Status")
        status_layout = QVBoxLayout(status_group)
        
        # Turn indicator
        turn_label = QLabel("Turn:")
        turn_label.setStyleSheet("font-weight: bold;")
        turn_value = QLabel("--")
        turn_value.setStyleSheet("font-size: 16px; color: #333;")
        status_layout.addWidget(turn_label)
        status_layout.addWidget(turn_value)
        
        # Separator
        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        status_layout.addWidget(line1)
        
        # Status text
        status_label = QLabel("Status:")
        status_label.setStyleSheet("font-weight: bold;")
        status_value = QLabel("Ready to play")
        status_value.setWordWrap(True)
        status_value.setStyleSheet("font-size: 12px; color: #666;")
        status_layout.addWidget(status_label)
        status_layout.addWidget(status_value)
        
        # Separator
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        status_layout.addWidget(line2)
        
        # Move counter
        move_label = QLabel("Moves:")
        move_label.setStyleSheet("font-weight: bold;")
        move_value = QLabel("0")
        move_value.setStyleSheet("font-size: 18px; color: #2196F3;")
        status_layout.addWidget(move_label)
        status_layout.addWidget(move_value)
        
        # Winner display
        winner_label = QLabel("")
        winner_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #FF5722; text-align: center;")
        winner_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(winner_label)
        
        control_layout.addWidget(status_group)
        
        # --- OpenCV Control Group ---
        cv_group = QGroupBox("Camera View")
        cv_layout = QVBoxLayout(cv_group)
        
        # Grid toggle button
        grid_btn = QPushButton("Toggle Grid")
        cv_layout.addWidget(grid_btn)
        
        # Mirror button
        mirror_btn = QPushButton("Mirror Detection")
        cv_layout.addWidget(mirror_btn)
        
        control_layout.addWidget(cv_group)
        
        # Spacer
        control_layout.addStretch()
        
        main_layout.addWidget(control_panel)
        
        # Create OpenCV windows
        cv2.namedWindow('Chessboard', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Storage', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Chessboard', self.display_width, self.display_width)
        cv2.resizeWindow('Storage', self.display_width, self.display_width)
        cv2.moveWindow('Chessboard', 50, 50)
        cv2.moveWindow('Storage', self.display_width + 70, 50)
        
        # State
        show_grid = [False]
        
        # ========== Button Callbacks ==========
        def on_start_stop():
            if self.robot_busy:
                print("Robot is busy, please wait...")
                return
            if self.game_active:
                self.stop_game()
                start_btn.setText("▶ Start Game")
                start_btn.setStyleSheet("QPushButton { font-size: 14px; padding: 10px; background-color: #4CAF50; color: white; }")
                winner_label.setText("")
            else:
                # Read settings from combos
                self.human_first = (first_combo.currentIndex() == 0)
                depth_values = [2, 4, 6, 8]
                self.ai_depth = depth_values[diff_combo.currentIndex()]
                self.start_game()
                start_btn.setText("■ Stop Game")
                start_btn.setStyleSheet("QPushButton { font-size: 14px; padding: 10px; background-color: #f44336; color: white; }")
                winner_label.setText("")
        
        def on_reset():
            if self.robot_busy:
                print("Robot is busy, please wait...")
                return
            self.clear_virtual_board()
            self.stop_game()
            self.move_count = 0
            self.move_order = {}  # Clear move order tracking
            start_btn.setText("▶ Start Game")
            start_btn.setStyleSheet("QPushButton { font-size: 14px; padding: 10px; background-color: #4CAF50; color: white; }")
            winner_label.setText("")
            print("Game reset.")
        
        def on_grid_toggle():
            show_grid[0] = not show_grid[0]
            print(f"Grid overlay: {'ON' if show_grid[0] else 'OFF'}")
        
        def on_mirror():
            if not self.game_active:
                self.mirror_to_virtual_chessboard()
            else:
                print("Cannot mirror during game!")
        
        # Connect buttons
        start_btn.clicked.connect(on_start_stop)
        reset_btn.clicked.connect(on_reset)
        grid_btn.clicked.connect(on_grid_toggle)
        mirror_btn.clicked.connect(on_mirror)
        
        # Disable combos during game
        def update_combo_state():
            enabled = not self.game_active
            first_combo.setEnabled(enabled)
            diff_combo.setEnabled(enabled)
            mirror_btn.setEnabled(enabled)
        
        # ========== Update Function ==========
        def update_all():
            """Update both OpenCV and Qt displays"""
            if not self.running:
                timer.stop()
                cv2.destroyAllWindows()
                window.close()
                return
            
            # Process frame
            chess_frame, storage_frame = self.process_frame(show_grid=show_grid[0])
            
            if chess_frame is not None and storage_frame is not None:
                # Update OpenCV windows
                cv2.imshow('Chessboard', chess_frame)
                cv2.imshow('Storage', storage_frame)
                
                # Update Qt virtual board
                pixmap = self.draw_virtual_board_qt()
                board_label.setPixmap(pixmap)
            
            # Update status display
            status_value.setText(self.game_status_text)
            move_value.setText(str(self.move_count))
            update_combo_state()
            
            # Update turn indicator
            if self.game_active:
                if self.waiting_for_human:
                    human_color = "Black ●" if self.human_first else "White ○"
                    turn_value.setText(f"Human ({human_color})")
                    turn_value.setStyleSheet("font-size: 16px; color: #4CAF50; font-weight: bold;")
                else:
                    ai_color = "White ○" if self.human_first else "Black ●"
                    turn_value.setText(f"AI ({ai_color})")
                    turn_value.setStyleSheet("font-size: 16px; color: #FF9800; font-weight: bold;")
            else:
                turn_value.setText("--")
                turn_value.setStyleSheet("font-size: 16px; color: #333;")
            
            # Update winner display
            if not self.game_active and "Game Over" in self.game_status_text:
                if "YOU WIN" in self.game_status_text:
                    winner_label.setText("🎉 YOU WIN! 🎉")
                    winner_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #4CAF50;")
                elif "AI WINS" in self.game_status_text:
                    winner_label.setText("🤖 AI WINS 🤖")
                    winner_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #f44336;")
                elif "DRAW" in self.game_status_text:
                    winner_label.setText("🤝 DRAW 🤝")
                    winner_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #FF9800;")
                # Reset button style when game ends
                start_btn.setText("▶ Start Game")
                start_btn.setStyleSheet("QPushButton { font-size: 14px; padding: 10px; background-color: #4CAF50; color: white; }")
            
            # Check for human move if game is active and waiting
            if self.game_active and self.waiting_for_human and not self.robot_busy:
                self._check_human_move_after_robot()
            
            # Periodically publish RViz markers (every ~10 frames)
            if hasattr(self, '_rviz_counter'):
                self._rviz_counter += 1
            else:
                self._rviz_counter = 0
            if self._rviz_counter >= 10:
                self._rviz_counter = 0
                self._publish_rviz_markers()
            
            # Handle OpenCV keyboard input (minimal)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('g'):
                show_grid[0] = not show_grid[0]
                print(f"Grid overlay: {'ON' if show_grid[0] else 'OFF'}")
            elif key == ord('p'):
                self.print_virtual_board_status()
            elif key == ord('o'):
                # Auto-fill test (only when no game active)
                if self.game_active:
                    print("Cannot auto-fill during game!")
                elif self.robot_busy:
                    print("[AUTO-FILL] Robot is busy, please wait...")
                else:
                    threading.Thread(target=self._auto_fill_thread, daemon=True).start()
        
        # Setup timer for unified updates
        timer = QTimer()
        timer.timeout.connect(update_all)
        timer.start(30)  # ~33 FPS
        
        # Position and show Qt window
        window.move(self.display_width * 2 + 100, 50)
        window.resize(self.virtual_board_size + 280, self.virtual_board_size + 150)
        window.show()
        
        # Handle Qt window close
        def on_close():
            self.running = False
        
        app.aboutToQuit.connect(on_close)
        
        # Run Qt event loop
        app.exec_()
        
        # Cleanup
        self.running = False
        cv2.destroyAllWindows()
        self.cap.release()
        print("All displays closed.")


# ============================================================
# ROS2 Robot Controller Class
# ============================================================
class ROS2RobotController(Node):
    """ROS2 Node for robot control - simple synchronous wrapper"""
    
    def __init__(self):
        super().__init__('chess_demo_robot_controller')
        
        # Action client for move_xyz_rotation
        self._move_xyz_client = ActionClient(self, MoveXyzRotation, 'move_xyz_rotation')
        
        # Service client for gripper control
        self._gripper_client = self.create_client(GripperControl, 'gripper_control')
        
        # Publisher for RViz visualization markers
        self._marker_pub = self.create_publisher(MarkerArray, '/chessboard/markers', 10)
        
        # Chessboard configuration (will be set by ChessboardDisplay)
        self._board_config = None
        self._chessboard_T_matrix = None
        
        # Storage configuration (will be set by ChessboardDisplay)
        self._storage_config = None
        self._storage_T_matrix = None
        self._storage_transformation_matrix = None  # Perspective transform for pixel->output coords
        
        # Completion flags
        self._move_completed = False
        self._move_success = False
        
        self.get_logger().info('ROS2 Robot Controller initialized')
    
    def wait_for_servers(self, timeout=5.0):
        """Wait for action/service servers to be available"""
        self.get_logger().info('Waiting for robot servers...')
        
        # Wait for move_xyz_rotation action server
        if not self._move_xyz_client.wait_for_server(timeout_sec=timeout):
            self.get_logger().error('move_xyz_rotation action server not available!')
            return False
        
        # Wait for gripper service
        if not self._gripper_client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error('gripper_control service not available!')
            return False
        
        self.get_logger().info('All robot servers available!')
        return True
    
    def move_xyz_sync(self, position, rotation, ik_mode="xyz", speed_ratio=1.0):
        """Synchronous move to position - blocks until complete"""
        self._move_completed = False
        self._move_success = False
        
        # Create goal - ensure all values are native Python floats (not numpy types)
        goal_msg = MoveXyzRotation.Goal()
        goal_msg.position = [float(p) for p in position]
        goal_msg.rotation = [float(r) for r in rotation]
        goal_msg.ik_mode = str(ik_mode)
        goal_msg.speed_ratio = float(speed_ratio)
        
        self.get_logger().info(f'Moving to: pos={goal_msg.position}, rot={goal_msg.rotation}')
        
        # Send goal async
        send_future = self._move_xyz_client.send_goal_async(
            goal_msg, 
            feedback_callback=self._move_feedback_callback
        )
        send_future.add_done_callback(self._move_response_callback)
        
        # Wait for completion
        while rclpy.ok() and not self._move_completed:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        return self._move_success
    
    def _move_feedback_callback(self, feedback_msg):
        """Handle move feedback"""
        fb = feedback_msg.feedback
        self.get_logger().debug(f'Move progress: {fb.progress:.1%}')
    
    def _move_response_callback(self, future):
        """Handle goal accepted/rejected"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Move goal rejected!')
            self._move_completed = True
            self._move_success = False
            return
        
        # Get result
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._move_result_callback)
    
    def _move_result_callback(self, future):
        """Handle move result"""
        result = future.result().result
        self._move_success = result.success
        self._move_completed = True
        
        if result.success:
            self.get_logger().info(f'Move completed in {result.execution_time:.2f}s')
        else:
            self.get_logger().error(f'Move failed: {result.message}')
    
    def gripper_control_sync(self, enable):
        """Synchronous gripper control - blocks until complete"""
        request = GripperControl.Request()
        request.enable = enable
        
        action_str = "ON" if enable else "OFF"
        self.get_logger().info(f'Gripper {action_str}...')
        
        future = self._gripper_client.call_async(request)
        
        # Wait for response
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if future.result() is not None:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Gripper {action_str} success')
                return True
            else:
                self.get_logger().error(f'Gripper failed: {response.message}')
                return False
        else:
            self.get_logger().error('Gripper service call failed')
            return False
    
    def spin_background(self):
        """Spin in background for callbacks (call from main thread periodically)"""
        rclpy.spin_once(self, timeout_sec=0.01)
    
    def set_board_config(self, board_rows, board_cols, board_width_mm, board_height_mm, T_matrix):
        """Set chessboard configuration for RViz visualization
        
        Args:
            board_rows: Number of rows (intersections)
            board_cols: Number of columns (intersections)
            board_width_mm: Board width in mm
            board_height_mm: Board height in mm
            T_matrix: 4x4 transformation matrix (board coords -> robot coords)
        """
        self._board_config = {
            'rows': board_rows,
            'cols': board_cols,
            'width_mm': board_width_mm,
            'height_mm': board_height_mm
        }
        self._chessboard_T_matrix = T_matrix
        self.get_logger().info(f'Board config set: {board_rows}x{board_cols}, {board_width_mm}x{board_height_mm}mm')
        
        # Publish empty board immediately so it's visible in RViz
        empty_board = [[0 for _ in range(board_cols)] for _ in range(board_rows)]
        self.publish_chessboard_markers(empty_board)
        self.get_logger().info('Initial empty board markers published to RViz')
    
    def set_storage_config(self, storage_width_mm, storage_height_mm, T_matrix, 
                           storage_output_width=None, storage_output_height=None,
                           storage_transformation_matrix=None):
        """Set storage area configuration for RViz visualization
        
        Args:
            storage_width_mm: Storage area width in mm
            storage_height_mm: Storage area height in mm
            T_matrix: 4x4 transformation matrix (storage coords -> robot coords)
            storage_output_width: Output width in pixels (after perspective transform)
            storage_output_height: Output height in pixels (after perspective transform)
            storage_transformation_matrix: 3x3 perspective transformation matrix
        """
        self._storage_config = {
            'width_mm': storage_width_mm,
            'height_mm': storage_height_mm,
            'output_width': storage_output_width,
            'output_height': storage_output_height
        }
        self._storage_T_matrix = T_matrix
        self._storage_transformation_matrix = storage_transformation_matrix
        self.get_logger().info(f'Storage config set: {storage_width_mm}x{storage_height_mm}mm')
    
    def publish_chessboard_markers(self, virtual_board_status, last_ai_move=None, last_human_move=None, storage_detected_pieces=None):
        """Publish chessboard, storage, and pieces as RViz markers
        
        Args:
            virtual_board_status: 2D list [row][col] with 0=empty, 1=black, -1=white
            last_ai_move: (row, col) tuple of last AI move for highlight
            last_human_move: (row, col) tuple of last human move for highlight
            storage_detected_pieces: List of detected pieces in storage area
        """
        if self._board_config is None or self._chessboard_T_matrix is None:
            return
        
        marker_array = MarkerArray()
        marker_id = 0
        
        # ========== 0. Delete all previous dynamic markers ==========
        for ns in ["pieces", "highlights", "storage_pieces"]:
            delete_marker = Marker()
            delete_marker.header.frame_id = "base_link"
            delete_marker.header.stamp = self.get_clock().now().to_msg()
            delete_marker.ns = ns
            delete_marker.id = 0
            delete_marker.action = Marker.DELETEALL
            marker_array.markers.append(delete_marker)

        rows = self._board_config['rows']
        cols = self._board_config['cols']
        width_mm = self._board_config['width_mm']
        height_mm = self._board_config['height_mm']
        
        # Tool length offset: vacuum sucker is 60mm long, mounted vertically
        # The hand-eye calibration was done at the sucker tip, but RViz shows the flange
        # So we need to offset Z by -60mm to show the board at the correct position
        tool_length_offset = -0.060  # -60mm in meters
        
        # Convert mm to meters for RViz
        width_m = width_mm / 1000.0
        height_m = height_mm / 1000.0
        cell_width = width_mm / (cols - 1)
        cell_height = height_mm / (rows - 1)
        
        # Calculate board center in board coordinates
        board_center_x = width_mm / 2.0
        board_center_y = height_mm / 2.0
        
        # Transform board center to robot coordinates
        center_point = np.array([board_center_x, board_center_y, 0.0, 1.0])
        robot_center = (self._chessboard_T_matrix @ center_point)[:3]
        
        # Board position in meters (robot frame uses mm, RViz uses m)
        board_x = robot_center[0] / 1000.0
        board_y = robot_center[1] / 1000.0
        board_z = robot_center[2] / 1000.0 + tool_length_offset
        
        # ========== 1. Board Surface with Thickness (using TRIANGLE_LIST) ==========
        # Calculate the four corners of the board in robot coordinates
        margin_mm = 10.0  # 10mm margin around grid
        board_thickness = 0.008  # 8mm thick board
        
        corners_board = [
            np.array([-margin_mm, -margin_mm, 0.0, 1.0]),                    # 0: Bottom-left
            np.array([width_mm + margin_mm, -margin_mm, 0.0, 1.0]),          # 1: Bottom-right
            np.array([width_mm + margin_mm, height_mm + margin_mm, 0.0, 1.0]), # 2: Top-right
            np.array([-margin_mm, height_mm + margin_mm, 0.0, 1.0])          # 3: Top-left
        ]
        # Apply tool length offset to Z coordinate
        corners_robot = [(self._chessboard_T_matrix @ c)[:3] / 1000.0 for c in corners_board]
        corners_robot = [(c[0], c[1], c[2] + tool_length_offset) for c in corners_robot]
        
        # Top surface z offset (slightly below grid lines)
        top_z_offset = -0.001
        bottom_z_offset = top_z_offset - board_thickness
        
        board_marker = Marker()
        board_marker.header.frame_id = "base_link"
        board_marker.header.stamp = self.get_clock().now().to_msg()
        board_marker.ns = "chessboard"
        board_marker.id = marker_id
        marker_id += 1
        board_marker.type = Marker.TRIANGLE_LIST
        board_marker.action = Marker.ADD
        board_marker.pose.orientation.w = 1.0
        board_marker.scale.x = 1.0
        board_marker.scale.y = 1.0
        board_marker.scale.z = 1.0
        board_marker.color = ColorRGBA(r=0.87, g=0.72, b=0.53, a=1.0)  # Wood color
        board_marker.lifetime.sec = 0
        
        # Helper to add a quad as two triangles
        def add_quad(points_list, p0, p1, p2, p3):
            # Triangle 1: p0, p1, p2
            points_list.append(p0)
            points_list.append(p1)
            points_list.append(p2)
            # Triangle 2: p0, p2, p3
            points_list.append(p0)
            points_list.append(p2)
            points_list.append(p3)
        
        # Top face (at top_z_offset)
        top_corners = [Point(x=c[0], y=c[1], z=c[2] + top_z_offset) for c in corners_robot]
        add_quad(board_marker.points, top_corners[0], top_corners[1], top_corners[2], top_corners[3])
        
        # Bottom face (at bottom_z_offset) - reverse winding for correct normal
        bottom_corners = [Point(x=c[0], y=c[1], z=c[2] + bottom_z_offset) for c in corners_robot]
        add_quad(board_marker.points, bottom_corners[0], bottom_corners[3], bottom_corners[2], bottom_corners[1])
        
        # Side faces (darker color will be added via separate marker for simplicity, or use colors array)
        # Front side (bottom edge): corners 0, 1
        add_quad(board_marker.points, bottom_corners[0], bottom_corners[1], top_corners[1], top_corners[0])
        # Right side: corners 1, 2
        add_quad(board_marker.points, bottom_corners[1], bottom_corners[2], top_corners[2], top_corners[1])
        # Back side (top edge): corners 2, 3
        add_quad(board_marker.points, bottom_corners[2], bottom_corners[3], top_corners[3], top_corners[2])
        # Left side: corners 3, 0
        add_quad(board_marker.points, bottom_corners[3], bottom_corners[0], top_corners[0], top_corners[3])
        
        marker_array.markers.append(board_marker)
        
        # ========== 2. Grid Lines ==========
        grid_marker = Marker()
        grid_marker.header.frame_id = "base_link"
        grid_marker.header.stamp = self.get_clock().now().to_msg()
        grid_marker.ns = "chessboard"
        grid_marker.id = marker_id
        marker_id += 1
        grid_marker.type = Marker.LINE_LIST
        grid_marker.action = Marker.ADD
        grid_marker.pose.orientation.w = 1.0
        grid_marker.scale.x = 0.001  # Line width 1mm
        grid_marker.color = ColorRGBA(r=0.4, g=0.26, b=0.13, a=1.0)  # Dark brown
        grid_marker.lifetime.sec = 0
        
        # Vertical lines
        for c in range(cols):
            bx = c * cell_width
            # Start point (row 0)
            p1_board = np.array([bx, 0.0, 0.0, 1.0])
            p1_robot = (self._chessboard_T_matrix @ p1_board)[:3] / 1000.0
            # End point (row max)
            p2_board = np.array([bx, height_mm, 0.0, 1.0])
            p2_robot = (self._chessboard_T_matrix @ p2_board)[:3] / 1000.0
            
            grid_marker.points.append(Point(x=p1_robot[0], y=p1_robot[1], z=p1_robot[2] + tool_length_offset))
            grid_marker.points.append(Point(x=p2_robot[0], y=p2_robot[1], z=p2_robot[2] + tool_length_offset))
        
        # Horizontal lines
        for r in range(rows):
            by = r * cell_height
            # Start point (col 0)
            p1_board = np.array([0.0, by, 0.0, 1.0])
            p1_robot = (self._chessboard_T_matrix @ p1_board)[:3] / 1000.0
            # End point (col max)
            p2_board = np.array([width_mm, by, 0.0, 1.0])
            p2_robot = (self._chessboard_T_matrix @ p2_board)[:3] / 1000.0
            
            grid_marker.points.append(Point(x=p1_robot[0], y=p1_robot[1], z=p1_robot[2] + tool_length_offset))
            grid_marker.points.append(Point(x=p2_robot[0], y=p2_robot[1], z=p2_robot[2] + tool_length_offset))
        
        marker_array.markers.append(grid_marker)
        
        # ========== 3. Chess Pieces ==========
        piece_radius = min(cell_width, cell_height) / 3.0 / 1000.0  # in meters
        piece_height = 0.012  # 12mm tall pieces
        
        for row in range(rows):
            for col in range(cols):
                piece_value = virtual_board_status[row][col]
                if piece_value == 0:
                    continue  # Empty cell
                
                # Calculate piece position in board coords
                bx = col * cell_width
                by = row * cell_height
                piece_board = np.array([bx, by, 0.0, 1.0])
                piece_robot = (self._chessboard_T_matrix @ piece_board)[:3] / 1000.0
                
                piece_marker = Marker()
                piece_marker.header.frame_id = "base_link"
                piece_marker.header.stamp = self.get_clock().now().to_msg()
                piece_marker.ns = "pieces"
                piece_marker.id = marker_id
                marker_id += 1
                piece_marker.type = Marker.CYLINDER
                piece_marker.action = Marker.ADD
                piece_marker.pose.position.x = piece_robot[0]
                piece_marker.pose.position.y = piece_robot[1]
                piece_marker.pose.position.z = piece_robot[2] + tool_length_offset + piece_height / 2
                piece_marker.pose.orientation.w = 1.0
                piece_marker.scale.x = piece_radius * 2
                piece_marker.scale.y = piece_radius * 2
                piece_marker.scale.z = piece_height
                
                if piece_value == 1:  # Black piece
                    piece_marker.color = ColorRGBA(r=0.1, g=0.1, b=0.1, a=1.0)
                else:  # White piece (piece_value == -1)
                    piece_marker.color = ColorRGBA(r=0.95, g=0.95, b=0.95, a=1.0)
                
                piece_marker.lifetime.sec = 0
                marker_array.markers.append(piece_marker)
        
        # ========== 4. Last Move Highlights ==========
        highlight_positions = []
        if last_ai_move:
            highlight_positions.append((last_ai_move, ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)))  # Red for AI
        if last_human_move:
            highlight_positions.append((last_human_move, ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)))  # Green for human
        
        for (move_pos, color) in highlight_positions:
            row, col = move_pos
            bx = col * cell_width
            by = row * cell_height
            pos_board = np.array([bx, by, 0.0, 1.0])
            pos_robot = (self._chessboard_T_matrix @ pos_board)[:3] / 1000.0
            
            highlight = Marker()
            highlight.header.frame_id = "base_link"
            highlight.header.stamp = self.get_clock().now().to_msg()
            highlight.ns = "highlights"
            highlight.id = marker_id
            marker_id += 1
            highlight.type = Marker.CYLINDER
            highlight.action = Marker.ADD
            highlight.pose.position.x = pos_robot[0]
            highlight.pose.position.y = pos_robot[1]
            highlight.pose.position.z = pos_robot[2] + tool_length_offset + 0.015  # Above piece
            highlight.pose.orientation.w = 1.0
            highlight.scale.x = piece_radius * 2.5  # Slightly larger than piece
            highlight.scale.y = piece_radius * 2.5
            highlight.scale.z = 0.002  # Thin ring
            highlight.color = color
            highlight.lifetime.sec = 0
            marker_array.markers.append(highlight)
        
        # ========== 5. Storage Area ==========
        if self._storage_config is not None and self._storage_T_matrix is not None:
            storage_width_mm = self._storage_config['width_mm']
            storage_height_mm = self._storage_config['height_mm']
            storage_margin_mm = 5.0
            storage_thickness = 0.005  # 5mm thick
            
            # Calculate storage corners
            storage_corners_board = [
                np.array([-storage_margin_mm, -storage_margin_mm, 0.0, 1.0]),
                np.array([storage_width_mm + storage_margin_mm, -storage_margin_mm, 0.0, 1.0]),
                np.array([storage_width_mm + storage_margin_mm, storage_height_mm + storage_margin_mm, 0.0, 1.0]),
                np.array([-storage_margin_mm, storage_height_mm + storage_margin_mm, 0.0, 1.0])
            ]
            storage_corners_robot = [(self._storage_T_matrix @ c)[:3] / 1000.0 for c in storage_corners_board]
            # Apply tool length offset to storage corners
            storage_corners_robot = [(c[0], c[1], c[2] + tool_length_offset) for c in storage_corners_robot]
            
            storage_top_z = -0.001
            storage_bottom_z = storage_top_z - storage_thickness
            
            storage_marker = Marker()
            storage_marker.header.frame_id = "base_link"
            storage_marker.header.stamp = self.get_clock().now().to_msg()
            storage_marker.ns = "storage"
            storage_marker.id = marker_id
            marker_id += 1
            storage_marker.type = Marker.TRIANGLE_LIST
            storage_marker.action = Marker.ADD
            storage_marker.pose.orientation.w = 1.0
            storage_marker.scale.x = 1.0
            storage_marker.scale.y = 1.0
            storage_marker.scale.z = 1.0
            storage_marker.color = ColorRGBA(r=0.6, g=0.6, b=0.65, a=1.0)  # Gray color
            storage_marker.lifetime.sec = 0
            
            # Top face
            s_top = [Point(x=c[0], y=c[1], z=c[2] + storage_top_z) for c in storage_corners_robot]
            add_quad(storage_marker.points, s_top[0], s_top[1], s_top[2], s_top[3])
            
            # Bottom face
            s_bottom = [Point(x=c[0], y=c[1], z=c[2] + storage_bottom_z) for c in storage_corners_robot]
            add_quad(storage_marker.points, s_bottom[0], s_bottom[3], s_bottom[2], s_bottom[1])
            
            # Side faces
            add_quad(storage_marker.points, s_bottom[0], s_bottom[1], s_top[1], s_top[0])
            add_quad(storage_marker.points, s_bottom[1], s_bottom[2], s_top[2], s_top[1])
            add_quad(storage_marker.points, s_bottom[2], s_bottom[3], s_top[3], s_top[2])
            add_quad(storage_marker.points, s_bottom[3], s_bottom[0], s_top[0], s_top[3])
            
            marker_array.markers.append(storage_marker)
            
            # ========== 6. Detected Pieces in Storage ==========
            if storage_detected_pieces and self._storage_transformation_matrix is not None:
                storage_piece_radius = 0.008  # 8mm radius
                storage_piece_height = 0.010  # 10mm tall
                output_width = self._storage_config.get('output_width', 640)
                output_height = self._storage_config.get('output_height', 480)
                
                for piece_data in storage_detected_pieces:
                    x1, y1, x2, y2, cls_id, conf = piece_data
                    
                    # Calculate center in raw frame
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Transform pixel coordinates using perspective transform
                    center_raw = np.array([[center_x, center_y]], dtype=np.float32)
                    center_transformed = cv2.perspectiveTransform(
                        center_raw.reshape(-1, 1, 2), 
                        self._storage_transformation_matrix
                    ).reshape(-1, 2)
                    tx, ty = center_transformed[0]
                    
                    # Skip pieces outside storage area bounds
                    if not (0 <= tx < output_width and 0 <= ty < output_height):
                        continue
                    
                    # Convert to storage mm coordinates
                    storage_x_mm = (tx / output_width) * storage_width_mm
                    storage_y_mm = ((output_height - ty) / output_height) * storage_height_mm
                    
                    # Transform to robot coordinates
                    piece_board = np.array([storage_x_mm, storage_y_mm, 0.0, 1.0])
                    piece_robot = (self._storage_T_matrix @ piece_board)[:3] / 1000.0
                    
                    piece_marker = Marker()
                    piece_marker.header.frame_id = "base_link"
                    piece_marker.header.stamp = self.get_clock().now().to_msg()
                    piece_marker.ns = "storage_pieces"
                    piece_marker.id = marker_id
                    marker_id += 1
                    piece_marker.type = Marker.CYLINDER
                    piece_marker.action = Marker.ADD
                    piece_marker.pose.position.x = piece_robot[0]
                    piece_marker.pose.position.y = piece_robot[1]
                    piece_marker.pose.position.z = piece_robot[2] + tool_length_offset + storage_piece_height / 2
                    piece_marker.pose.orientation.w = 1.0
                    piece_marker.scale.x = storage_piece_radius * 2
                    piece_marker.scale.y = storage_piece_radius * 2
                    piece_marker.scale.z = storage_piece_height
                    
                    if cls_id == 0:  # Black piece
                        piece_marker.color = ColorRGBA(r=0.15, g=0.15, b=0.15, a=0.9)
                    else:  # White piece
                        piece_marker.color = ColorRGBA(r=0.9, g=0.9, b=0.9, a=0.9)
                    
                    piece_marker.lifetime.sec = 0
                    marker_array.markers.append(piece_marker)
        
        # Publish all markers
        self._marker_pub.publish(marker_array)


def main():
    """Main entry point with ROS2 support"""
    # Configuration paths
    CAMERA_CONFIG_PATH = "./calibration_scripts/camera_calibration.yaml"
    STORAGE_CONFIG_PATH = "./calibration_scripts/storage_camera_calibration.yaml"
    YOLO_WEIGHTS_PATH = "./calibration_scripts/train_yolo/best.pt"
    STORAGE_HAND_EYE_PATH = "./calibration_scripts/storage_hand_eye_calibration.yaml"
    CHESSBOARD_HAND_EYE_PATH = "./calibration_scripts/hand_eye_calibration.yaml"
    
    print("=" * 70)
    print(" CHESSBOARD & STORAGE DISPLAY DEMO (ROS2 VERSION)")
    print("=" * 70)
    print()
    
    # Initialize ROS2
    ros2_controller = None
    if ROS2_AVAILABLE:
        try:
            rclpy.init()
            ros2_controller = ROS2RobotController()
            
            # Wait for robot servers
            if not ros2_controller.wait_for_servers(timeout=5.0):
                print("WARNING: Robot servers not available. Robot control disabled.")
                ros2_controller = None
        except Exception as e:
            print(f"WARNING: ROS2 initialization failed: {e}")
            ros2_controller = None
    else:
        print("WARNING: ROS2 not available. Robot control disabled.")
    
    try:
        # Create display instance
        display = ChessboardDisplay(
            CAMERA_CONFIG_PATH, 
            STORAGE_CONFIG_PATH, 
            YOLO_WEIGHTS_PATH,
            STORAGE_HAND_EYE_PATH,
            CHESSBOARD_HAND_EYE_PATH
        )
        
        # Set ROS2 controller
        if ros2_controller:
            display.set_ros2_controller(ros2_controller)
        
        # Run all displays (OpenCV + Qt) simultaneously
        display.run_all_displays()
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup ROS2
        if ros2_controller:
            ros2_controller.destroy_node()
        if ROS2_AVAILABLE:
            try:
                rclpy.shutdown()
            except:
                pass
    
    return 0


if __name__ == "__main__":
    exit(main())
