#!/usr/bin/env python3
"""
Hand-Eye Calibration Tool (4-Point Teaching Method) - ROS2 Version

This tool calculates the transformation matrix ^{b}T_t (table frame to robot base frame)
using 4 teaching points with real robot control.

Features:
1. Input 4 known table coordinates (in mm)
2. Use arrow keys to jog robot to each point
3. Record robot position at each point
4. Calculate transformation matrix using SVD

Usage:
1. Launch the robot controller: ros2 launch episode_controller robot_controller.launch.py
2. Run this node: ros2 run episode_apps calibrate_hand_eye_qt
3. For each point:
   - Use arrow keys to move robot (X/Y), Page Up/Down for Z
   - Press Enter to record the point
4. After 4 points, press 'c' to calculate transformation
5. Press 's' to save calibration

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

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from robot_arm_interfaces.action import MoveXyzRotation
from robot_arm_interfaces.srv import ReadMotorAngles

from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QHBoxLayout, QWidget, QMessageBox, QGroupBox,
                             QGridLayout, QPushButton, QDoubleSpinBox, QFrame,
                             QSpinBox, QTabWidget)
from PyQt5.QtGui import QFont, QColor, QPalette
from PyQt5.QtCore import Qt, QTimer


# Default configuration file path
DEFAULT_CONFIG_PATH = "hand_eye_calibration.yaml"


def load_config_from_yaml(config_path: str) -> dict:
    """
    Load calibration configuration from YAML file.
    
    Returns:
        dict with keys: table_points, total_cols, total_rows, T_matrix (optional)
    """
    config = {
        'table_points': None,
        'total_cols': 8,
        'total_rows': 6,
        'T_matrix': None,
        'robot_points': None,
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
            
            if yaml_data:
                # Load table points (required)
                if 'table_points' in yaml_data:
                    config['table_points'] = np.array(yaml_data['table_points'], dtype=np.float32)
                
                # Load grid configuration (optional)
                if 'total_cols' in yaml_data:
                    config['total_cols'] = int(yaml_data['total_cols'])
                if 'total_rows' in yaml_data:
                    config['total_rows'] = int(yaml_data['total_rows'])
                
                # Load T matrix if exists (optional)
                if 'T_matrix' in yaml_data and yaml_data['T_matrix'] is not None:
                    config['T_matrix'] = np.array(yaml_data['T_matrix'], dtype=np.float64)
                
                # Load robot points if exists (optional, for reference)
                if 'robot_points' in yaml_data and yaml_data['robot_points'] is not None:
                    config['robot_points'] = np.array(yaml_data['robot_points'], dtype=np.float32)
                
                print(f" Configuration loaded from: {config_path}")
        except Exception as e:
            print(f" Error loading config from {config_path}: {e}")
    else:
        print(f"ℹ Config file not found: {config_path}, using defaults")
    
    return config


def save_config_to_yaml(config_path: str, table_points: np.ndarray, total_cols: int, 
                        total_rows: int, T_matrix: np.ndarray = None, 
                        robot_points: np.ndarray = None):
    """
    Save calibration configuration to YAML file.
    """
    data = {
        'table_points': table_points.tolist() if table_points is not None else None,
        'total_cols': total_cols,
        'total_rows': total_rows,
        'T_matrix': T_matrix.tolist() if T_matrix is not None else None,
        'robot_points': robot_points.tolist() if robot_points is not None else None,
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    print(f" Configuration saved to: {config_path}")


class RobotController(Node):
    """
    Real robot controller interface using episode_controller API
    """
    
    def __init__(self):
        super().__init__('hand_eye_calibrator_robot_controller')
        
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
        self.get_logger().info('Initializing robot to home position...')
        self.move_to(260.0, 0.0, 200.0)
    
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


def compute_transformation_matrix(table_points: np.ndarray, robot_points: np.ndarray) -> np.ndarray:
    """
    Compute rigid transformation matrix from table frame to robot base frame
    using SVD (Singular Value Decomposition)
    
    Args:
        table_points: Nx3 array of points in table frame [x_t, y_t, z_t]
        robot_points: Nx3 array of corresponding points in robot frame [x_b, y_b, z_b]
    
    Returns:
        4x4 homogeneous transformation matrix ^{b}T_t
    """
    assert table_points.shape == robot_points.shape
    assert table_points.shape[0] >= 3, "Need at least 3 points"
    
    n = table_points.shape[0]
    
    # Compute centroids
    centroid_table = np.mean(table_points, axis=0)
    centroid_robot = np.mean(robot_points, axis=0)
    
    # Center the points
    table_centered = table_points - centroid_table
    robot_centered = robot_points - centroid_robot
    
    # Compute covariance matrix H
    H = table_centered.T @ robot_centered
    
    # SVD decomposition
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation matrix
    R = Vt.T @ U.T
    
    # Handle reflection case (ensure proper rotation)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    t = centroid_robot - R @ centroid_table
    
    # Build 4x4 homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T


def compute_calibration_error(T: np.ndarray, table_points: np.ndarray, robot_points: np.ndarray) -> tuple:
    """
    Compute calibration error (reprojection error)
    
    Returns:
        (mean_error, max_error, errors_per_point)
    """
    errors = []
    for i in range(len(table_points)):
        # Transform table point to robot frame
        pt_table_h = np.append(table_points[i], 1)  # homogeneous
        pt_robot_predicted = (T @ pt_table_h)[:3]
        
        # Compute error
        error = np.linalg.norm(pt_robot_predicted - robot_points[i])
        errors.append(error)
    
    return np.mean(errors), np.max(errors), errors


class HandEyeCalibrator(QMainWindow):
    """Hand-Eye Calibration Tool with PyQt5 UI - ROS2 Version"""
    
    def __init__(self, robot_node, config: dict, config_path: str):
        super().__init__()
        
        # ROS node for robot control
        self.robot = robot_node
        
        # Configuration file path
        self.config_path = config_path
        
        # Table points (known positions in table frame, mm) - loaded from YAML, cannot change
        self.table_points = config['table_points']
        self.num_points = len(self.table_points)
        
        # Recorded robot positions
        self.robot_points = [None] * self.num_points
        
        # Load previous robot points if available
        if config['robot_points'] is not None and len(config['robot_points']) == self.num_points:
            for i, pt in enumerate(config['robot_points']):
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
        self.T_matrix = config['T_matrix']
        if self.T_matrix is not None:
            print(f" Loaded previous T matrix from config")
        
        # Point colors
        self.colors = [
            QColor(255, 100, 100),  # Red
            QColor(100, 255, 100),  # Green
            QColor(100, 100, 255),  # Blue
            QColor(255, 128, 0),    # Orange
        ]
        
        self.point_names = ["P1 (Origin)", "P2 (X-axis)", "P3 (Y-axis)", "P4 (Validation)"]
        
        # Board dimensions (calculated from calibration points)
        # X direction: P1 to P2, Y direction: P1 to P3
        self.board_width = self.table_points[1][0] - self.table_points[0][0]   
        self.board_height = self.table_points[2][1] - self.table_points[0][1]  
        
        # Grid configuration (loaded from YAML, can be changed in UI)
        self.total_cols = config['total_cols']
        self.total_rows = config['total_rows']
        
        # ROS spin timer
        self.ros_timer = QTimer()
        self.ros_timer.timeout.connect(self.spin_ros)
        self.ros_timer.start(50)  # 20 Hz
        
        self.init_ui()
        self.update_display()
        self.print_instructions()
    
    def spin_ros(self):
        """Spin ROS node to process callbacks"""
        rclpy.spin_once(self.robot, timeout_sec=0.0)
    
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("Hand-Eye Calibration (4-Point Teaching) - ROS2")
        self.setMinimumSize(900, 700)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        
        # Title
        title = QLabel("Hand-Eye Calibration Tool - ROS2")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 18, QFont.Bold))
        main_layout.addWidget(title)
        
        # Horizontal layout for points and controls
        h_layout = QHBoxLayout()
        
        # Left side: Point status
        points_group = QGroupBox("Calibration Points")
        points_layout = QGridLayout()
        
        self.point_labels = []
        self.table_coord_labels = []
        self.robot_coord_labels = []
        self.status_indicators = []
        self.update_point_buttons = []
        
        # Header
        points_layout.addWidget(QLabel("Point"), 0, 0)
        points_layout.addWidget(QLabel("Table Coord (mm)"), 0, 1)
        points_layout.addWidget(QLabel("Robot Coord (mm)"), 0, 2)
        points_layout.addWidget(QLabel("Status"), 0, 3)
        points_layout.addWidget(QLabel("Action"), 0, 4)
        
        for i in range(self.num_points):
            row = i + 1
            
            # Point name
            name_label = QLabel(self.point_names[i] if i < len(self.point_names) else f"P{i+1}")
            name_label.setStyleSheet(f"color: {self.colors[i % len(self.colors)].name()};")
            name_label.setFont(QFont("Arial", 10, QFont.Bold))
            points_layout.addWidget(name_label, row, 0)
            self.point_labels.append(name_label)
            
            # Table coordinates
            pt = self.table_points[i]
            table_label = QLabel(f"[{pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f}]")
            points_layout.addWidget(table_label, row, 1)
            self.table_coord_labels.append(table_label)
            
            # Robot coordinates (to be filled)
            robot_label = QLabel("[ -- , -- , -- ]")
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
        h_layout.addWidget(points_group)
        
        # Right side: Robot control
        control_group = QGroupBox("Robot Control")
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
        
        # Set button size
        for btn in [btn_y_plus, btn_y_minus, btn_x_minus, btn_x_plus, btn_z_plus, btn_z_minus]:
            btn.setMinimumSize(80, 40)
        
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
        
        self.record_btn = QPushButton("Record Point (Enter)")
        self.record_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; font-size: 14px;")
        self.record_btn.clicked.connect(self.record_point)
        action_layout.addWidget(self.record_btn)
        
        self.save_modified_btn = QPushButton("Save Modified Point")
        self.save_modified_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 10px; font-size: 14px;")
        self.save_modified_btn.clicked.connect(self.save_modified_point)
        self.save_modified_btn.setVisible(False)
        action_layout.addWidget(self.save_modified_btn)
        
        self.calc_btn = QPushButton("Calculate T Matrix (C)")
        self.calc_btn.setStyleSheet("padding: 8px; font-size: 12px;")
        self.calc_btn.clicked.connect(self.calculate_transformation)
        self.calc_btn.setEnabled(False)
        action_layout.addWidget(self.calc_btn)
        
        self.save_btn = QPushButton("Save Calibration (S)")
        self.save_btn.setStyleSheet("padding: 8px; font-size: 12px;")
        self.save_btn.clicked.connect(self.save_calibration)
        self.save_btn.setEnabled(False)
        action_layout.addWidget(self.save_btn)
        
        reset_btn = QPushButton("Reset All (R)")
        reset_btn.setStyleSheet("padding: 8px; font-size: 12px;")
        reset_btn.clicked.connect(self.reset_all)
        action_layout.addWidget(reset_btn)
        
        home_btn = QPushButton("Move to Home")
        home_btn.setStyleSheet("background-color: #9C27B0; color: white; padding: 8px; font-size: 12px;")
        home_btn.clicked.connect(self.move_to_home)
        action_layout.addWidget(home_btn)
        
        control_layout.addLayout(action_layout)
        
        control_group.setLayout(control_layout)
        h_layout.addWidget(control_group)
        
        main_layout.addLayout(h_layout)
        
        # Status bar
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14px; padding: 10px; background-color: #f0f0f0;")
        main_layout.addWidget(self.status_label)
        
        # Result display
        self.result_label = QLabel()
        self.result_label.setFont(QFont("Courier", 10))
        self.result_label.setAlignment(Qt.AlignLeft)
        self.result_label.setStyleSheet("padding: 10px; background-color: #e8e8e8;")
        main_layout.addWidget(self.result_label)
        
        # Verification section
        verify_group = QGroupBox("Verification - Test Grid Position")
        verify_layout = QGridLayout()
        
        # Grid configuration
        verify_layout.addWidget(QLabel("Grid Config:"), 0, 0)
        verify_layout.addWidget(QLabel("Total Cols:"), 0, 1)
        self.total_cols_spin = QSpinBox()
        self.total_cols_spin.setRange(2, 20)
        self.total_cols_spin.setValue(self.total_cols)
        self.total_cols_spin.valueChanged.connect(self.on_grid_config_changed)
        verify_layout.addWidget(self.total_cols_spin, 0, 2)
        
        verify_layout.addWidget(QLabel("Total Rows:"), 0, 3)
        self.total_rows_spin = QSpinBox()
        self.total_rows_spin.setRange(2, 20)
        self.total_rows_spin.setValue(self.total_rows)
        self.total_rows_spin.valueChanged.connect(self.on_grid_config_changed)
        verify_layout.addWidget(self.total_rows_spin, 0, 4)
        
        # Cell size display (calculated)
        self.cell_size_label = QLabel()
        self.update_cell_size_display()
        verify_layout.addWidget(self.cell_size_label, 0, 5, 1, 2)
        
        # Test position input
        verify_layout.addWidget(QLabel("Test Position:"), 1, 0)
        verify_layout.addWidget(QLabel("Col (0-based):"), 1, 1)
        self.test_col_spin = QSpinBox()
        self.test_col_spin.setRange(0, 999)
        self.test_col_spin.setValue(0)
        self.test_col_spin.valueChanged.connect(self.on_test_position_changed)
        verify_layout.addWidget(self.test_col_spin, 1, 2)
        
        verify_layout.addWidget(QLabel("Row (0-based):"), 1, 3)
        self.test_row_spin = QSpinBox()
        self.test_row_spin.setRange(0, 999)
        self.test_row_spin.setValue(0)
        self.test_row_spin.valueChanged.connect(self.on_test_position_changed)
        verify_layout.addWidget(self.test_row_spin, 1, 4)
        
        # Z offset for verification
        verify_layout.addWidget(QLabel("Z offset:"), 1, 5)
        self.verify_z_spin = QDoubleSpinBox()
        self.verify_z_spin.setRange(-100.0, 100.0)
        self.verify_z_spin.setValue(0.0)
        self.verify_z_spin.setSingleStep(1.0)
        self.verify_z_spin.valueChanged.connect(self.on_test_position_changed)
        verify_layout.addWidget(self.verify_z_spin, 1, 6)
        
        # Calculated coordinates display
        verify_layout.addWidget(QLabel("Table Coord:"), 2, 0)
        self.verify_table_coord_label = QLabel("[0.00, 0.00, 0.00]")
        self.verify_table_coord_label.setFont(QFont("Courier", 10))
        self.verify_table_coord_label.setStyleSheet("color: blue;")
        verify_layout.addWidget(self.verify_table_coord_label, 2, 1, 1, 3)
        
        verify_layout.addWidget(QLabel("Robot Coord:"), 2, 4)
        self.verify_robot_coord_label = QLabel("[---, ---, ---]")
        self.verify_robot_coord_label.setFont(QFont("Courier", 10))
        self.verify_robot_coord_label.setStyleSheet("color: green;")
        verify_layout.addWidget(self.verify_robot_coord_label, 2, 5, 1, 2)
        
        # Move button
        self.verify_move_btn = QPushButton("Calculate && Move to Grid Position")
        self.verify_move_btn.setStyleSheet("background-color: #FF9800; color: white; padding: 10px; font-size: 14px;")
        self.verify_move_btn.clicked.connect(self.verify_move_to_grid)
        self.verify_move_btn.setEnabled(False)
        verify_layout.addWidget(self.verify_move_btn, 3, 0, 1, 7)
        
        verify_group.setLayout(verify_layout)
        main_layout.addWidget(verify_group)
    
    def print_instructions(self):
        """Print instructions to console"""
        print("=" * 60)
        print("Hand-Eye Calibration Tool (4-Point Teaching) - ROS2")
        print("=" * 60)
        print("\nTable coordinates to teach:")
        for i, pt in enumerate(self.table_points):
            name = self.point_names[i] if i < len(self.point_names) else f"P{i+1}"
            print(f"  {name}: [{pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f}] mm")
        print("\nUse arrow keys or buttons to jog robot, then press Enter to record each point.")
        print("-" * 60)
    
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
                    self.status_indicators[i].setStyleSheet("color: green;")
                
                # Enable update button for recorded points
                self.update_point_buttons[i].setEnabled(True)
            elif i == self.current_point_index:
                self.robot_coord_labels[i].setText("[ -- , -- , -- ]")
                self.status_indicators[i].setText(" Current")
                self.status_indicators[i].setStyleSheet("color: blue;")
                self.update_point_buttons[i].setEnabled(False)
            else:
                self.robot_coord_labels[i].setText("[ -- , -- , -- ]")
                self.status_indicators[i].setText(" Pending")
                self.status_indicators[i].setStyleSheet("color: gray;")
                self.update_point_buttons[i].setEnabled(False)
        
        # Update status label
        if self.editing_point_index is not None:
            name = self.point_names[self.editing_point_index] if self.editing_point_index < len(self.point_names) else f"P{self.editing_point_index+1}"
            pt = self.table_points[self.editing_point_index]
            self.status_label.setText(
                f"Editing {name} - Adjust robot position for table position [{pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f}] mm, then press 'Save Modified Point'"
            )
            self.record_btn.setVisible(False)
            self.save_modified_btn.setVisible(True)
        elif self.current_point_index < self.num_points:
            name = self.point_names[self.current_point_index] if self.current_point_index < len(self.point_names) else f"P{self.current_point_index+1}"
            pt = self.table_points[self.current_point_index]
            self.status_label.setText(
                f"Teaching {name} - Move robot to table position [{pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f}] mm, then press Enter"
            )
            self.record_btn.setVisible(True)
            self.save_modified_btn.setVisible(False)
        else:
            self.status_label.setText("All points recorded! Press 'C' to calculate transformation matrix")
            self.record_btn.setVisible(True)
            self.save_modified_btn.setVisible(False)
        
        # Enable/disable buttons
        all_recorded = all(p is not None for p in self.robot_points)
        self.calc_btn.setEnabled(all_recorded)
        self.save_btn.setEnabled(self.T_matrix is not None)
        self.verify_move_btn.setEnabled(self.T_matrix is not None)
    
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
        name = self.point_names[point_index] if point_index < len(self.point_names) else f"P{point_index+1}"
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
        
        name = self.point_names[self.editing_point_index] if self.editing_point_index < len(self.point_names) else f"P{self.editing_point_index+1}"
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
            QMessageBox.information(self, "Info", "All points already recorded!")
            return
        
        if self.robot.is_moving():
            QMessageBox.warning(self, "Warning", "Robot is still moving, please wait...")
            return
        
        pos = self.robot.get_current_position()
        self.robot_points[self.current_point_index] = pos[:3]  # Only xyz
        
        name = self.point_names[self.current_point_index] if self.current_point_index < len(self.point_names) else f"P{self.current_point_index+1}"
        print(f"\n Recorded {name}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        
        self.current_point_index += 1
        
        if self.current_point_index < self.num_points:
            next_name = self.point_names[self.current_point_index] if self.current_point_index < len(self.point_names) else f"P{self.current_point_index+1}"
            next_pt = self.table_points[self.current_point_index]
            print(f"  Next: {next_name} at table position [{next_pt[0]:.1f}, {next_pt[1]:.1f}, {next_pt[2]:.1f}]")
        else:
            print("\n All points recorded! Press 'C' to calculate transformation matrix.")
        
        self.update_display()
    
    def calculate_transformation(self):
        """Calculate transformation matrix"""
        if not all(p is not None for p in self.robot_points):
            QMessageBox.warning(self, "Warning", "Please record all points first!")
            return
        
        table_pts = np.array(self.table_points)
        robot_pts = np.array(self.robot_points)
        
        print("\n" + "=" * 60)
        print("Calculating transformation matrix...")
        print("=" * 60)
        
        try:
            self.T_matrix = compute_transformation_matrix(table_pts, robot_pts)
            
            # Compute error
            mean_err, max_err, errors = compute_calibration_error(self.T_matrix, table_pts, robot_pts)
            
            # Display result
            result_text = "Transformation Matrix ^{b}T_t:\n"
            result_text += "-" * 40 + "\n"
            for row in self.T_matrix:
                result_text += f"[{row[0]:10.4f} {row[1]:10.4f} {row[2]:10.4f} {row[3]:10.4f}]\n"
            result_text += "-" * 40 + "\n"
            result_text += f"Calibration Error: Mean={mean_err:.3f}mm, Max={max_err:.3f}mm\n"
            result_text += f"Per-point errors: {[f'{e:.3f}' for e in errors]}"
            
            self.result_label.setText(result_text)
            
            # Print to console
            print("\n^{b}T_t =")
            print(self.T_matrix)
            print(f"\nCalibration Error:")
            print(f"  Mean: {mean_err:.4f} mm")
            print(f"  Max:  {max_err:.4f} mm")
            print(f"  Per-point: {errors}")
            
            # Extract rotation and translation
            R = self.T_matrix[:3, :3]
            t = self.T_matrix[:3, 3]
            
            print(f"\nRotation matrix R:")
            print(R)
            print(f"\nTranslation vector t: {t}")
            
            self.update_display()
            
            # Auto-save to YAML after successful calculation
            self.auto_save_calibration()
            
            if max_err > 5.0:
                QMessageBox.warning(self, "Warning", 
                    f"Calibration error is high (max={max_err:.2f}mm).\n"
                    "Consider re-teaching the points more carefully.\n"
                    f"Results auto-saved to: {self.config_path}")
            else:
                QMessageBox.information(self, "Success", 
                    f"Calibration complete!\nMean error: {mean_err:.3f}mm\nMax error: {max_err:.3f}mm\n"
                    f"Results auto-saved to: {self.config_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Calculation failed: {str(e)}")
            print(f"Error: {e}")
    
    def auto_save_calibration(self):
        """Auto-save calibration result to YAML file after calculation"""
        if self.T_matrix is None:
            return
        
        # Prepare robot points array
        robot_pts = None
        if all(p is not None for p in self.robot_points):
            robot_pts = np.array(self.robot_points)
        
        # Save to YAML
        save_config_to_yaml(
            self.config_path,
            table_points=np.array(self.table_points),
            total_cols=self.total_cols,
            total_rows=self.total_rows,
            T_matrix=self.T_matrix,
            robot_points=robot_pts
        )
        
        print(f"\n Auto-saved calibration to: {self.config_path}")
    
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
        save_config_to_yaml(
            self.config_path,
            table_points=np.array(self.table_points),
            total_cols=self.total_cols,
            total_rows=self.total_rows,
            T_matrix=self.T_matrix,
            robot_points=robot_pts
        )
        
        print(f"\n Calibration saved to: {self.config_path}")
        QMessageBox.information(self, "Saved", f"Calibration saved to:\n{self.config_path}")
    
    def reset_all(self):
        """Reset all recorded points and move robot to initial position"""
        reply = QMessageBox.question(self, "Confirm Reset", 
            "Reset all recorded points and move robot to initial position?",
            QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.robot_points = [None] * self.num_points
            self.current_point_index = 0
            self.editing_point_index = None
            self.T_matrix = None
            self.result_label.setText("")
            self.update_display()
            print("\n Reset all points")
            
            # Move robot to initial position
            print(" Moving robot to initial position...")
            self.robot.move_to(260.0, 0.0, 200.0)
            self.update_robot_position_display()
    
    def move_to_home(self):
        """Move robot to home position"""
        if self.robot.is_moving():
            QMessageBox.warning(self, "Warning", "Robot is still moving, please wait...")
            return
        
        print("\n Moving robot to home position [260.0, 0.0, 200.0]...")
        self.robot.move_to(260.0, 0.0, 200.0)
        
        # Update display after a short delay
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
    
    def on_grid_config_changed(self):
        """Handle grid configuration change"""
        self.total_cols = self.total_cols_spin.value()
        self.total_rows = self.total_rows_spin.value()
        
        # Update cell size display
        self.update_cell_size_display()
        
        # Update coordinates display
        self.on_test_position_changed()
    
    def update_cell_size_display(self):
        """Update cell size display based on board dimensions and grid config"""
        cell_width = self.board_width / self.total_cols
        cell_height = self.board_height / self.total_rows
        self.cell_size_label.setText(f"Cell: {cell_width:.2f} x {cell_height:.2f} mm")
    
    def on_test_position_changed(self):
        """Handle test position change - update coordinate displays"""
        col = self.test_col_spin.value()
        row = self.test_row_spin.value()
        z_offset = self.verify_z_spin.value()
        
        # Calculate cell size
        cell_width = self.board_width / self.total_cols
        cell_height = self.board_height / self.total_rows
        
        # Calculate table coordinates (grid intersection, not cell center)
        x_table = col * cell_width
        y_table = row * cell_height
        z_table = z_offset
        
        self.verify_table_coord_label.setText(f"[{x_table:.2f}, {y_table:.2f}, {z_table:.2f}]")
        
        # Calculate robot coordinates if T matrix is available
        if self.T_matrix is not None:
            pt_table_h = np.array([x_table, y_table, z_table, 1.0])
            pt_robot = (self.T_matrix @ pt_table_h)[:3]
            self.verify_robot_coord_label.setText(f"[{pt_robot[0]:.2f}, {pt_robot[1]:.2f}, {pt_robot[2]:.2f}]")
        else:
            self.verify_robot_coord_label.setText("[---, ---, ---]")
    
    def verify_move_to_grid(self):
        """Move robot to the calculated grid position for verification"""
        if self.T_matrix is None:
            QMessageBox.warning(self, "Warning", "Please calculate transformation matrix first!")
            return
        
        if self.robot.is_moving():
            QMessageBox.warning(self, "Warning", "Robot is still moving, please wait...")
            return
        
        col = self.test_col_spin.value()
        row = self.test_row_spin.value()
        z_offset = self.verify_z_spin.value()
        
        # Calculate cell size
        cell_width = self.board_width / self.total_cols
        cell_height = self.board_height / self.total_rows
        
        # Calculate table coordinates (grid intersection, not cell center)
        x_table = col * cell_width
        y_table = row * cell_height
        z_table = z_offset
        
        # Transform to robot coordinates
        pt_table_h = np.array([x_table, y_table, z_table, 1.0])
        pt_robot = (self.T_matrix @ pt_table_h)[:3]
        
        print(f"\n Verification: Moving to grid intersection (col={col}, row={row})")
        print(f"   Table coord: [{x_table:.2f}, {y_table:.2f}, {z_table:.2f}]")
        print(f"   Robot coord: [{pt_robot[0]:.2f}, {pt_robot[1]:.2f}, {pt_robot[2]:.2f}]")
        
        # First move to safe home position to avoid collision
        print(f"   Step 1: Moving to safe position [260.0, 0.0, 200.0]...")
        self.robot.move_to(260.0, 0.0, 200.0)
        
        # Wait for first move to complete, then move to target
        QTimer.singleShot(2000, lambda: self._move_to_target(pt_robot))
    
    def _move_to_target(self, pt_robot):
        """Move to target position after safe position reached"""
        if self.robot.is_moving():
            # Still moving, wait more
            QTimer.singleShot(500, lambda: self._move_to_target(pt_robot))
            return
        
        print(f"   Step 2: Moving to target [{pt_robot[0]:.2f}, {pt_robot[1]:.2f}, {pt_robot[2]:.2f}]...")
        self.robot.move_to(pt_robot[0], pt_robot[1], pt_robot[2])
        
        # Update display after a short delay
        QTimer.singleShot(100, self.update_robot_position_display)
    
    def closeEvent(self, event):
        """Handle window close event"""
        self.ros_timer.stop()
        event.accept()


def main(args=None):
    """Main function"""
    
    # Determine config file path
    config_path = DEFAULT_CONFIG_PATH
    
    # Check if custom config path provided via command line
    if len(sys.argv) > 1 and sys.argv[1].endswith('.yaml'):
        config_path = sys.argv[1]
    
    # Load configuration from YAML
    config = load_config_from_yaml(config_path)
    
    # table_points must be loaded from YAML - no hardcoded defaults
    if config['table_points'] is None:
        print(f"\n Error: table_points not found in config file: {config_path}")
        print(f"   Please create the config file with table_points defined.")
        print(f"\n   Example YAML format:")
        print(f"   table_points:")
        print(f"     - [0.0, 0.0, 0.0]      # P1: Origin")
        print(f"     - [272.0, 0.0, 0.0]    # P2: X-axis direction")
        print(f"     - [0.0, 240.0, 0.0]    # P3: Y-axis direction")
        print(f"     - [272.0, 240.0, 0.0]  # P4: Validation point")
        print(f"   total_cols: 8")
        print(f"   total_rows: 6")
        sys.exit(1)
    
    print(f"\n Configuration:")
    print(f"   Config file: {config_path}")
    print(f"   Table points: {len(config['table_points'])} points")
    print(f"   Grid: {config['total_cols']} cols x {config['total_rows']} rows")
    print(f"   T matrix: {'Loaded' if config['T_matrix'] is not None else 'Not available'}")
    
    # Initialize ROS2
    rclpy.init(args=args)
    
    # Create robot controller node
    robot_node = RobotController()
    
    # Wait for robot to initialize
    print("Waiting for robot to initialize...")
    robot_node.wait_for_move_complete(timeout=10.0)
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create calibration window
    window = HandEyeCalibrator(robot_node, config, config_path)
    window.show()
    
    # Run Qt event loop
    exit_code = app.exec_()
    
    # Cleanup
    robot_node.destroy_node()
    rclpy.shutdown()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
