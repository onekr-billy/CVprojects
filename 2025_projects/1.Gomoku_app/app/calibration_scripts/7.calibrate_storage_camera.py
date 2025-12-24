#!/usr/bin/env python3
"""
Storage Area Perspective Transform Calibration Tool

Features:
1. Continuously capture video from USB camera 
2. Click to select 4 corners of the storage area (paper/workspace) on live feed
3. Press Enter to display perspective-transformed video in real-time
4. Save calibration as storage_camera_calibration.yaml

Usage:
1. Place a rectangular paper or workspace where you want to store pieces
2. Run the program, camera feed will start
3. Click the 4 corners of the storage area in order: Bottom-Left -> Bottom-Right -> Top-Right -> Top-Left
4. Press 'r' to reset, press 'Enter' to confirm and show transformed view, press 'q' to quit
5. Press 's' to save as storage_camera_calibration.yaml

This creates a separate calibrated region for piece storage outside the chessboard area.
"""

import sys
import cv2
import numpy as np
import yaml
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QHBoxLayout, QWidget, QMessageBox, QPushButton,
                             QLineEdit, QGroupBox, QScrollArea, QSpinBox)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QPoint, QTimer


class CameraImageLabel(QLabel):
    """Clickable image label for camera feed"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setMouseTracking(True)
        self.original_size = None
        self.scaled_pixmap = None
        self.zoom_factor = 1.0
        self.original_pixmap = None
    
    def set_image(self, pixmap, original_w, original_h):
        """Set image with original size tracking"""
        self.original_size = (original_w, original_h)
        self.original_pixmap = pixmap
        self.update_zoom_display()
    
    def update_zoom_display(self):
        """Update display with current zoom factor"""
        if self.original_pixmap is None:
            return
        
        if self.zoom_factor == 1.0:
            # Use original scaling behavior for zoom = 1.0
            self.scaled_pixmap = self.original_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        else:
            # Apply zoom factor
            zoomed_size = self.original_pixmap.size() * self.zoom_factor
            self.scaled_pixmap = self.original_pixmap.scaled(zoomed_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.setPixmap(self.scaled_pixmap)
        # Update scroll area size
        self.resize(self.scaled_pixmap.size())
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.original_size and self.scaled_pixmap:
            label_w, label_h = self.width(), self.height()
            scaled_w, scaled_h = self.scaled_pixmap.width(), self.scaled_pixmap.height()
            orig_w, orig_h = self.original_size
            
            offset_x = (label_w - scaled_w) // 2
            offset_y = (label_h - scaled_h) // 2
            
            click_x = event.x() - offset_x
            click_y = event.y() - offset_y
            
            if 0 <= click_x < scaled_w and 0 <= click_y < scaled_h:
                x = int(click_x * orig_w / scaled_w)
                y = int(click_y * orig_h / scaled_h)
                
                if 0 <= x < orig_w and 0 <= y < orig_h:
                    self.parent_window.on_click(x, y)
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming around cursor"""
        if event.modifiers() == Qt.ControlModifier or True:  # Always allow zoom
            # Get cursor position relative to the image
            cursor_pos = event.pos()
            
            # Get scroll area (parent of this label)
            scroll_area = self.parent_window.scroll_area
            
            # Get current scroll position
            h_scroll = scroll_area.horizontalScrollBar()
            v_scroll = scroll_area.verticalScrollBar()
            old_h_value = h_scroll.value()
            old_v_value = v_scroll.value()
            
            # Calculate cursor position in the scaled image coordinates
            old_zoom = self.zoom_factor
            
            # Convert cursor position to image coordinates
            if self.scaled_pixmap and self.original_size:
                # Get the cursor position relative to the scaled image
                label_rect = self.geometry()
                pixmap_rect = self.scaled_pixmap.rect()
                
                # Calculate offset if image is centered in label
                offset_x = max(0, (label_rect.width() - pixmap_rect.width()) // 2)
                offset_y = max(0, (label_rect.height() - pixmap_rect.height()) // 2)
                
                # Cursor position in scaled image coordinates
                img_cursor_x = cursor_pos.x() - offset_x
                img_cursor_y = cursor_pos.y() - offset_y
                
                # Clamp to image bounds
                img_cursor_x = max(0, min(pixmap_rect.width(), img_cursor_x))
                img_cursor_y = max(0, min(pixmap_rect.height(), img_cursor_y))
                
                # Convert to normalized coordinates (0-1)
                if pixmap_rect.width() > 0 and pixmap_rect.height() > 0:
                    norm_x = img_cursor_x / pixmap_rect.width()
                    norm_y = img_cursor_y / pixmap_rect.height()
                else:
                    norm_x = norm_y = 0.5
            else:
                norm_x = norm_y = 0.5
            
            # Zoom in/out
            zoom_in = event.angleDelta().y() > 0
            zoom_step = 0.1
            
            if zoom_in:
                self.zoom_factor = min(5.0, self.zoom_factor + zoom_step)
            else:
                self.zoom_factor = max(0.2, self.zoom_factor - zoom_step)
            
            if old_zoom != self.zoom_factor:
                self.update_zoom_display()
                self.parent_window.update_zoom_label()
                
                # Adjust scroll position to keep cursor position fixed
                # Wait for the display to update
                QTimer.singleShot(1, lambda: self.adjust_scroll_to_cursor(scroll_area, norm_x, norm_y))
                
            event.accept()
        else:
            event.ignore()
    
    def adjust_scroll_to_cursor(self, scroll_area, norm_x, norm_y):
        """Adjust scroll area to keep the cursor position centered during zoom"""
        if not self.scaled_pixmap:
            return
            
        # Get new image dimensions
        new_width = self.scaled_pixmap.width()
        new_height = self.scaled_pixmap.height()
        
        # Get viewport dimensions
        viewport = scroll_area.viewport()
        viewport_width = viewport.width()
        viewport_height = viewport.height()
        
        # Calculate target position for the cursor in the new scaled image
        target_x = norm_x * new_width
        target_y = norm_y * new_height
        
        # Calculate scroll position to center the target position in viewport
        scroll_x = int(target_x - viewport_width / 2)
        scroll_y = int(target_y - viewport_height / 2)
        
        # Apply scroll position
        h_scroll = scroll_area.horizontalScrollBar()
        v_scroll = scroll_area.verticalScrollBar()
        
        h_scroll.setValue(scroll_x)
        v_scroll.setValue(scroll_y)


class StorageAreaCalibrator(QMainWindow):
    """Storage area camera calibration tool"""
    
    def __init__(self, camera_index: int = 0, storage_size_mm: tuple = (200, 150)):
        super().__init__()
        
        self.camera_index = camera_index
        self.storage_width, self.storage_height = storage_size_mm
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {camera_index}")
        
        # Set camera resolution to 1280x720
        print("Setting camera resolution to 1280x720...")
        
        # Try MJPG format for better HD support
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Set resolution
        width_success = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        height_success = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print(f"Resolution setting - Width: {width_success}, Height: {height_success}")
        
        # Reduce buffer size to minimize latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Get actual resolution from camera properties
        prop_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        prop_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera reports: {prop_width}x{prop_height}")
        
        # Verify with actual frame capture
        ret, test_frame = self.cap.read()
        if ret and test_frame is not None:
            actual_height, actual_width = test_frame.shape[:2]
            self.frame_width = actual_width
            self.frame_height = actual_height
            print(f"Actual frame size: {self.frame_width} x {self.frame_height}")
            
            if self.frame_width != 1280 or self.frame_height != 720:
                print(f"  WARNING: Could not set 1280x720, using {self.frame_width}x{self.frame_height}")
                print("This may be due to camera hardware limitations or driver issues.")
            else:
                print(" Successfully set to 1280x720")
        else:
            # Fallback to camera properties if frame read fails
            self.frame_width = prop_width
            self.frame_height = prop_height
            print(f"Using camera properties: {self.frame_width} x {self.frame_height}")
        
        # Current frame
        self.current_frame = None
        
        # Clicked pixel coordinates
        self.clicked_points = []
        
        # Storage area corner coordinates (Bottom-Left, Bottom-Right, Top-Right, Top-Left)
        self.storage_corners = np.array([
            [0, 0],
            [self.storage_width, 0],
            [self.storage_width, self.storage_height],
            [0, self.storage_height],
        ], dtype=np.float32)
        
        self.corner_names = ["Bottom-Left", "Bottom-Right", "Top-Right", "Top-Left"]
        self.colors = [
            (0, 0, 255),    # Red (BGR)
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
        ]
        self.qt_colors = [
            QColor(255, 0, 0),
            QColor(0, 255, 0),
            QColor(0, 0, 255),
            QColor(255, 255, 0),
        ]
        
        # Transform mode flag
        self.show_transformed = False
        self.H_warp = None  # Perspective transform matrix
        self.output_width = int(self.storage_width * 2)  # 2 pixels per mm
        self.output_height = int(self.storage_height * 2)
        
        # Circle display settings
        self.circle_radius = 5  # Smaller default radius
        self.circle_border_width = 2
        
        self.init_ui()
        self.print_instructions()
        
        # Timer for camera capture
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # ~30 FPS
    
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("Storage Area Camera Calibration - Click 4 corners (BL→BR→TR→TL)")
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Add instruction text
        instruction_label = QLabel(" STORAGE AREA CALIBRATION")
        instruction_label.setAlignment(Qt.AlignCenter)
        instruction_label.setFont(QFont("Arial", 16, QFont.Bold))
        instruction_label.setStyleSheet("color: #FF6600; padding: 10px; background-color: #FFF3E0; border: 2px solid #FF6600; border-radius: 5px;")
        layout.addWidget(instruction_label)
        
        instruction_text = QLabel("Place a rectangular paper/workspace for piece storage, then click its 4 corners")
        instruction_text.setAlignment(Qt.AlignCenter)
        instruction_text.setStyleSheet("padding: 5px; font-size: 12px; color: #333;")
        layout.addWidget(instruction_text)
        
        # Create horizontal layout for two views
        self.views_layout = QHBoxLayout()
        
        # Original camera feed label with scroll area for zoom
        self.camera_label = CameraImageLabel(self)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("border: 2px solid #FF6600;")
        
        # Scroll area for zoom functionality
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.camera_label)
        self.scroll_area.setWidgetResizable(False)  # Don't resize widget, allow scrolling
        self.scroll_area.setMinimumSize(640, 480)
        self.views_layout.addWidget(self.scroll_area)
        
        # Transformed view label (hidden initially)
        self.transformed_label = QLabel()
        self.transformed_label.setAlignment(Qt.AlignCenter)
        self.transformed_label.setMinimumSize(300, 300)
        self.transformed_label.setStyleSheet("border: 2px solid #0066cc;")
        self.transformed_label.hide()
        self.views_layout.addWidget(self.transformed_label)
        
        layout.addLayout(self.views_layout)
        
        # Storage area size input group
        storage_group = QGroupBox("Storage Area Size (mm)")
        storage_layout = QHBoxLayout()
        
        width_label = QLabel("Width:")
        storage_layout.addWidget(width_label)
        
        self.width_input = QLineEdit(str(self.storage_width))
        self.width_input.setMaximumWidth(80)
        self.width_input.textChanged.connect(self.on_storage_size_changed)
        storage_layout.addWidget(self.width_input)
        
        height_label = QLabel("Height:")
        storage_layout.addWidget(height_label)
        
        self.height_input = QLineEdit(str(self.storage_height))
        self.height_input.setMaximumWidth(80)
        self.height_input.textChanged.connect(self.on_storage_size_changed)
        storage_layout.addWidget(self.height_input)
        
        # Common storage sizes
        preset_label = QLabel("Presets:")
        storage_layout.addWidget(preset_label)
        
        preset_a4_btn = QPushButton("A4 Paper (210x297)")
        preset_a4_btn.clicked.connect(lambda: self.set_storage_size(210, 297))
        storage_layout.addWidget(preset_a4_btn)
        
        preset_letter_btn = QPushButton("Letter (216x279)")
        preset_letter_btn.clicked.connect(lambda: self.set_storage_size(216, 279))
        storage_layout.addWidget(preset_letter_btn)
        
        preset_small_btn = QPushButton("Small (150x200)")
        preset_small_btn.clicked.connect(lambda: self.set_storage_size(150, 200))
        storage_layout.addWidget(preset_small_btn)
        
        storage_layout.addStretch()
        storage_group.setLayout(storage_layout)
        layout.addWidget(storage_group)
        
        # Circle size control group
        circle_group = QGroupBox("Circle Display")
        circle_layout = QHBoxLayout()
        
        circle_label = QLabel("Radius:")
        circle_layout.addWidget(circle_label)
        
        self.circle_radius_spin = QSpinBox()
        self.circle_radius_spin.setRange(2, 20)
        self.circle_radius_spin.setValue(self.circle_radius)
        self.circle_radius_spin.setSuffix(" px")
        self.circle_radius_spin.valueChanged.connect(self.on_circle_radius_changed)
        circle_layout.addWidget(self.circle_radius_spin)
        
        circle_layout.addStretch()
        circle_group.setLayout(circle_layout)
        layout.addWidget(circle_group)
        
        # Zoom controls
        zoom_layout = QHBoxLayout()
        
        zoom_out_btn = QPushButton("Zoom Out (-)")
        zoom_out_btn.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(zoom_out_btn)
        
        self.zoom_label = QLabel("Zoom: 100%")
        self.zoom_label.setMinimumWidth(80)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        zoom_layout.addWidget(self.zoom_label)
        
        zoom_in_btn = QPushButton("Zoom In (+)")
        zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(zoom_in_btn)
        
        zoom_reset_btn = QPushButton("Fit (0)")
        zoom_reset_btn.clicked.connect(self.zoom_reset)
        zoom_layout.addWidget(zoom_reset_btn)
        
        zoom_layout.addStretch()
        layout.addLayout(zoom_layout)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("Reset (R)")
        self.reset_btn.clicked.connect(self.reset_points)
        self.reset_btn.setStyleSheet("background-color: #f44336; color: white; padding: 8px;")
        button_layout.addWidget(self.reset_btn)
        
        self.undo_btn = QPushButton("Undo Last (U)")
        self.undo_btn.clicked.connect(self.undo_last_point)
        self.undo_btn.setEnabled(False)
        button_layout.addWidget(self.undo_btn)
        
        self.transform_btn = QPushButton("Preview Transform (Enter)")
        self.transform_btn.clicked.connect(self.toggle_transform)
        self.transform_btn.setEnabled(False)
        self.transform_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 8px;")
        button_layout.addWidget(self.transform_btn)
        
        self.save_btn = QPushButton(" Save Storage Calibration")
        self.save_btn.clicked.connect(self.save_calibration)
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px; font-weight: bold;")
        button_layout.addWidget(self.save_btn)
        
        self.quit_btn = QPushButton("Quit (Q)")
        self.quit_btn.clicked.connect(self.close)
        button_layout.addWidget(self.quit_btn)
        
        layout.addLayout(button_layout)
        
        # Status label
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14px; padding: 10px; background-color: #FFF3E0; border: 1px solid #FF6600; border-radius: 3px;")
        layout.addWidget(self.status_label)
        
        # Window size
        self.resize(900, 800)
    
    def print_instructions(self):
        """Print usage instructions"""
        print("=" * 60)
        print("  STORAGE AREA CAMERA CALIBRATION")
        print("=" * 60)
        print(f"Camera: Index {self.camera_index}")
        print(f"Resolution: {self.frame_width} x {self.frame_height}")
        print(f"Storage area size: {self.storage_width} x {self.storage_height} mm")
        print("\n SETUP INSTRUCTIONS:")
        print("1. Place a rectangular paper or workspace where you want to store pieces")
        print("2. Make sure the storage area is well-lit and clearly visible")
        print("3. The storage area should be separate from your chessboard")
        print("\n CALIBRATION STEPS:")
        print("Click the 4 corners of the storage area in this order:")
        print("  1. Bottom-Left (origin)")
        print("  2. Bottom-Right") 
        print("  3. Top-Right")
        print("  4. Top-Left")
        print("-" * 60)
        print(f" Please click {self.corner_names[0]}...")
    
    def update_frame(self):
        """Capture and update camera frame"""
        ret, frame = self.cap.read()
        if not ret:
            return
        
        self.current_frame = frame
        
        # Draw clicked points on frame
        display = frame.copy()
        for i, pt in enumerate(self.clicked_points):
            color = self.colors[i]
            cv2.circle(display, tuple(pt), self.circle_radius, color, -1)
            cv2.circle(display, tuple(pt), self.circle_radius + self.circle_border_width, (255, 255, 255), self.circle_border_width)
            
            label = f"{i+1}.{self.corner_names[i]}"
            # Position label based on circle size
            label_offset_x = self.circle_radius + 8
            label_offset_y = -5
            cv2.putText(display, label, (pt[0] + label_offset_x, pt[1] + label_offset_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw connecting lines
        if len(self.clicked_points) >= 2:
            pts = np.array(self.clicked_points, dtype=np.int32)
            for i in range(len(pts) - 1):
                cv2.line(display, tuple(pts[i]), tuple(pts[i+1]), (255, 255, 255), 3)
            if len(pts) == 4:
                cv2.line(display, tuple(pts[3]), tuple(pts[0]), (255, 255, 255), 3)
                # Fill the area with semi-transparent overlay
                overlay = display.copy()
                cv2.fillPoly(overlay, [pts], (255, 165, 0))  # Orange fill
                display = cv2.addWeighted(display, 0.8, overlay, 0.2, 0)
        
        # Convert to Qt image and display - use contiguous array to avoid memory issues
        h, w = display.shape[:2]
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        self.camera_label.set_image(pixmap, w, h)
        
        # Update transformed view if enabled
        if self.show_transformed and self.H_warp is not None:
            warped = cv2.warpPerspective(
                frame, 
                self.H_warp, 
                (self.output_width, self.output_height)
            )
            h_w, w_w = warped.shape[:2]
            rgb_w = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
            rgb_w = np.ascontiguousarray(rgb_w)
            qimg_w = QImage(rgb_w.data, w_w, h_w, rgb_w.strides[0], QImage.Format_RGB888).copy()
            pixmap_w = QPixmap.fromImage(qimg_w)
            scaled_pixmap = pixmap_w.scaled(self.transformed_label.size(), 
                                            Qt.KeepAspectRatio, Qt.FastTransformation)
            self.transformed_label.setPixmap(scaled_pixmap)
        
        # Update status
        self.update_status()
    
    def update_status(self):
        """Update status label"""
        if len(self.clicked_points) < 4:
            self.status_label.setText(
                f" Click {self.corner_names[len(self.clicked_points)]} corner ({len(self.clicked_points)+1}/4) | "
                f"R: Reset | Q: Quit"
            )
            self.transform_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
        else:
            if self.show_transformed:
                self.status_label.setText(
                    " Showing transformed storage area view | Enter: Toggle | R: Reset |  S: Save | Q: Quit"
                )
            else:
                self.status_label.setText(
                    " All corners selected! | Enter: Preview Transform | R: Reset | Q: Quit"
                )
            self.transform_btn.setEnabled(True)
            self.save_btn.setEnabled(self.show_transformed)
    
    def on_click(self, x, y):
        """Handle click event"""
        if len(self.clicked_points) < 4 and not self.show_transformed:
            self.clicked_points.append([x, y])
            
            idx = len(self.clicked_points)
            print(f" Selected {self.corner_names[idx-1]}: ({x}, {y})")
            
            # Update button states
            self.undo_btn.setEnabled(True)
            
            if idx < 4:
                print(f" Please click {self.corner_names[idx]}...")
            else:
                print("\n All 4 corners selected!")
                print(" Press Enter to preview transformed view")
                print(" Press 'r' to reset if needed")
    
    def reset_points(self):
        """Reset all clicked points"""
        self.clicked_points = []
        self.show_transformed = False
        self.H_warp = None
        self.transformed_label.hide()
        self.resize(900, 800)
        
        # Reset button states
        self.undo_btn.setEnabled(False)
        self.transform_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        
        print(f"\n Reset! Storage area size: {self.storage_width}x{self.storage_height}mm")
        print(" Please select corners again...")
        print(f" Please click {self.corner_names[0]}...")
    
    def set_storage_size(self, width, height):
        """Set storage area size from preset"""
        self.storage_width = width
        self.storage_height = height
        self.width_input.setText(str(width))
        self.height_input.setText(str(height))
        self.on_storage_size_changed()
        print(f" Storage area size set to: {width}x{height}mm")
    
    def on_storage_size_changed(self):
        """Handle storage size input changes"""
        try:
            width = int(self.width_input.text())
            height = int(self.height_input.text())
            
            if width > 0 and height > 0:
                self.storage_width = width
                self.storage_height = height
                
                # Update storage corners
                self.storage_corners = np.array([
                    [0, 0],
                    [self.storage_width, 0],
                    [self.storage_width, self.storage_height],
                    [0, self.storage_height],
                ], dtype=np.float32)
                
                # Update output size
                self.output_width = int(self.storage_width * 2)
                self.output_height = int(self.storage_height * 2)
                
                # Reset points when storage size changes
                if len(self.clicked_points) > 0:
                    self.reset_points()
                    
        except ValueError:
            pass  # Ignore invalid input while typing
    
    def zoom_in(self):
        """Zoom in on camera feed"""
        self.camera_label.zoom_factor = min(5.0, self.camera_label.zoom_factor + 0.2)
        self.camera_label.update_zoom_display()
        self.update_zoom_label()
    
    def zoom_out(self):
        """Zoom out on camera feed"""
        self.camera_label.zoom_factor = max(0.2, self.camera_label.zoom_factor - 0.2)
        self.camera_label.update_zoom_display()
        self.update_zoom_label()
    
    def zoom_reset(self):
        """Reset zoom to fit"""
        self.camera_label.zoom_factor = 1.0
        self.camera_label.update_zoom_display()
        self.update_zoom_label()
    
    def update_zoom_label(self):
        """Update zoom percentage label"""
        zoom_percent = int(self.camera_label.zoom_factor * 100)
        self.zoom_label.setText(f"Zoom: {zoom_percent}%")
    
    def on_circle_radius_changed(self, value):
        """Handle circle radius change"""
        self.circle_radius = value
        print(f"Circle radius changed to: {value}px")
    
    def undo_last_point(self):
        """Remove the last clicked point"""
        if len(self.clicked_points) > 0:
            removed_point = self.clicked_points.pop()
            corner_name = self.corner_names[len(self.clicked_points)]
            print(f" Undid: {self.corner_names[len(self.clicked_points)]} at {removed_point}")
            if len(self.clicked_points) < 4:
                print(f" Please click {corner_name}...")
        
        # Update button states
        self.undo_btn.setEnabled(len(self.clicked_points) > 0)
        self.transform_btn.setEnabled(len(self.clicked_points) == 4)
        self.save_btn.setEnabled(False)
    
    
    def toggle_transform(self):
        """Toggle perspective transform view"""
        if len(self.clicked_points) == 4:
            if not self.show_transformed:
                self.do_transform()
            else:
                self.show_transformed = False
                self.transformed_label.hide()
                self.resize(900, 800)
                print("\n Transformed view hidden")
    
    def do_transform(self):
        """Calculate and show perspective transform"""
        print("\n" + "=" * 60)
        print(" Starting storage area perspective transform...")
        print("=" * 60)
        
        pixel_corners = np.array(self.clicked_points, dtype=np.float32)
        
        # Calculate transform matrix
        output_corners = np.array([
            [0, self.output_height],
            [self.output_width, self.output_height],
            [self.output_width, 0],
            [0, 0],
        ], dtype=np.float32)
        
        self.H_warp = cv2.getPerspectiveTransform(pixel_corners, output_corners)
        
        self.show_transformed = True
        self.transformed_label.show()
        self.resize(1300, 800)
        
        print(" Storage area transformed view enabled!")
        print(" Press Enter again to hide, or 'r' to reset")
        print(" Press 's' to save this calibration")
    
    def save_calibration(self):
        """Save storage area calibration data"""
        if not self.show_transformed:
            return
        
        # Save transformation matrix and calibration data in YAML format
        calib_path = "storage_camera_calibration.yaml"
        pixel_corners = np.array(self.clicked_points, dtype=np.float32)
        
        # Prepare calibration data
        calib_data = {
            'transformation_matrix': self.H_warp.tolist(),
            'pixel_corners': pixel_corners.tolist(),
            'storage_corners': self.storage_corners.tolist(),
            'storage_size_mm': {'width': self.storage_width, 'height': self.storage_height},
            'output_size': {'width': self.output_width, 'height': self.output_height},
            'camera_resolution': {'width': self.frame_width, 'height': self.frame_height},
            'corner_names': self.corner_names,
            'calibration_type': 'storage_area',
            'description': 'Storage area perspective transformation for piece handling'
        }
        
        # Save to YAML file
        with open(calib_path, 'w') as f:
            yaml.dump(calib_data, f, default_flow_style=False, indent=2)
        
        print(f" Storage calibration saved: {calib_path}")
        
        # Save current warped frame
        if self.current_frame is not None and self.H_warp is not None:
            warped = cv2.warpPerspective(
                self.current_frame, 
                self.H_warp, 
                (self.output_width, self.output_height)
            )
            snapshot_path = "storage_area_snapshot.png"
            cv2.imwrite(snapshot_path, warped)
            print(f" Storage snapshot saved: {snapshot_path}")
        
        QMessageBox.information(self, " Storage Area Calibration Saved", 
            f"Storage area calibration saved successfully!\n\n"
            f" Files created:\n"
            f"- {calib_path}\n"
            f"- storage_area_snapshot.png\n\n"
            f" Storage area: {self.storage_width}x{self.storage_height}mm\n"
            f" Output: {self.output_width}x{self.output_height}px\n\n"
            f" Next step: Run hand-eye calibration for this storage area!")
    
    def keyPressEvent(self, event):
        """Keyboard event handler"""
        key = event.key()
        
        if key == Qt.Key_Q:
            print("\n Exiting storage area calibration...")
            self.close()
        
        elif key == Qt.Key_R:
            self.reset_points()
        
        elif key == Qt.Key_Return or key == Qt.Key_Enter:
            self.toggle_transform()
        
        elif key == Qt.Key_S:
            self.save_calibration()
        
        elif key == Qt.Key_U:
            self.undo_last_point()
        
        elif key == Qt.Key_Plus or key == Qt.Key_Equal:
            self.zoom_in()
        
        elif key == Qt.Key_Minus:
            self.zoom_out()
        
        elif key == Qt.Key_0:
            self.zoom_reset()
    
    def closeEvent(self, event):
        """Clean up on close"""
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        print(" Camera released. Storage area calibration complete!")
        event.accept()


def main():
    # Camera index (0 is usually the first USB camera)
    camera_index = 0
    
    # Default storage area size (unit: mm) - can be changed in UI
    storage_size = (200, 150)  # Smaller than chessboard, suitable for piece storage
    
    # Command line arguments
    if len(sys.argv) > 1:
        try:
            camera_index = int(sys.argv[1])
        except ValueError:
            print(f"Invalid camera index: {sys.argv[1]}")
            print("Usage: python calibrate_storage_camera.py [camera_index]")
            sys.exit(1)
    
    print("  STORAGE AREA CAMERA CALIBRATION")
    print("=" * 60)
    print("This tool creates a separate calibrated region for piece storage.")
    print("You'll use this with hand-eye calibration to handle pieces outside the chessboard.")
    print("=" * 60)
    
    app = QApplication(sys.argv)
    try:
        window = StorageAreaCalibrator(camera_index, storage_size)
        window.show()
        sys.exit(app.exec_())
    except RuntimeError as e:
        print(f" Error: {e}")
        print("\n Tips:")
        print("  - Make sure your camera is connected")
        print("  - Try different camera indices: 0, 1, 2, etc.")
        print("  - Usage: python calibrate_storage_camera.py [camera_index]")
        sys.exit(1)


if __name__ == "__main__":
    main()