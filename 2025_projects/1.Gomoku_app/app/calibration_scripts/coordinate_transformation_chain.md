# Coordinate Transformation Chain: Camera Pixels to Robot Pose

This document outlines the complete coordinate transformation pipeline used in the chess piece robot controller system, showing how detected piece positions are converted from camera pixels to robot coordinates for precise manipulation.

## Overview

The system performs a multi-stage coordinate transformation:

```
Camera Pixels → Perspective-Corrected Board → Chessboard Coordinates → Robot Base Coordinates
```

## 1. Camera Coordinate System

**Origin:** Top-left corner of camera image  
**Units:** Pixels  
**Axes:** X (right), Y (down)  

### Source
- Raw camera feed from USB camera
- Typical resolution: 640×480 or 1920×1080 pixels
- YOLO detection provides piece centers in these coordinates

### Example Values
```
Camera coordinates: (x_cam, y_cam) = (320, 240) pixels
```

## 2. Perspective Transformation: Camera → Board View

**Purpose:** Correct camera perspective distortion to get top-down view of chessboard  
**Method:** OpenCV perspective transformation using 4 corner points

### Transformation Matrix
```python
# From camera calibration (camera_calibration.yaml)
pixel_corners = [
    [x1, y1],  # Bottom-left corner (user clicked)
    [x2, y2],  # Bottom-right corner  
    [x3, y3],  # Top-right corner
    [x4, y4]   # Top-left corner
]

# Target corners for rectified view
output_corners = [
    [0, output_height],           # Bottom-left → (0, height)
    [output_width, output_height], # Bottom-right → (width, height)  
    [output_width, 0],            # Top-right → (width, 0)
    [0, 0]                        # Top-left → (0, 0)
]

# OpenCV calculates transformation matrix
transform_matrix = cv2.getPerspectiveTransform(pixel_corners, output_corners)
```

### Coordinate Conversion
```python
# Apply perspective transformation
transformed_point = cv2.perspectiveTransform(camera_point, transform_matrix)
```

**Result:** Rectified board view coordinates  
**Origin:** Bottom-left corner of board  
**Units:** Pixels (in rectified view)  
**Axes:** X (right), Y (up)  

## 3. Board Coordinate System

**Purpose:** Convert rectified pixels to physical board coordinates  
**Origin:** Bottom-left corner of physical chessboard  
**Units:** Millimeters  
**Axes:** X (right), Y (up), Z (up from board surface)

### Transformation Logic
```python
def camera_to_chessboard(self, camera_x, camera_y):
    """Convert camera pixel coordinates to chessboard coordinates (mm)"""
    # Convert normalized coordinates to board coordinates
    # Y coordinate is flipped because camera origin is top-left, board origin is bottom-left
    board_x = (camera_x / self.output_width) * self.board_width
    board_y = ((self.output_height - camera_y) / self.output_height) * self.board_height
    return board_x, board_y
```

### Board Dimensions (from calibration data)
```python
# Calculated from table_points in hand_eye_calibration.yaml
self.board_width = self.table_points[1][0] - self.table_points[0][0]   # P1 to P2 (X direction)
self.board_height = self.table_points[2][1] - self.table_points[0][1]  # P1 to P3 (Y direction)

# Typical values:
# board_width = 272.0 mm
# board_height = 240.0 mm
```

### Example Calculation
```python
# Input: Rectified camera coordinates
camera_x, camera_y = 320, 240  # pixels in rectified view
output_width, output_height = 640, 480  # rectified image size
board_width, board_height = 272.0, 240.0  # physical board size in mm

# Calculate board coordinates
board_x = (320 / 640) * 272.0 = 136.0 mm
board_y = ((480 - 240) / 480) * 240.0 = 120.0 mm
```

## 4. Hand-Eye Calibration: Board → Robot Base

**Purpose:** Transform from board coordinate system to robot base coordinate system  
**Method:** 4×4 homogeneous transformation matrix from hand-eye calibration

### Transformation Matrix Structure
```python
# From hand_eye_calibration.yaml
T_matrix = [
    [R11, R12, R13, tx],   # Rotation matrix R and translation vector t
    [R21, R22, R23, ty],   # ^{robot}T_{board} transformation
    [R31, R32, R33, tz],
    [0,   0,   0,   1 ]
]
```

### Calibration Process
The hand-eye calibration establishes the relationship between:
- **Table Points:** Known positions in board frame (mm)
- **Robot Points:** Corresponding positions in robot base frame (mm)

#### Table Points (Board Frame)
```python
table_points = [
    [0.0,   0.0,   0.0],    # P1: Origin (bottom-left)
    [272.0, 0.0,   0.0],    # P2: X-axis direction (bottom-right)  
    [0.0,   240.0, 0.0],    # P3: Y-axis direction (top-left)
    [272.0, 240.0, 0.0]     # P4: Validation point (top-right)
]
```

#### Robot Points (Robot Base Frame)
```python
# Example robot points (taught during calibration)
robot_points = [
    [260.0, -136.0, 95.0],  # Robot position for P1
    [532.0, -136.0, 95.0],  # Robot position for P2  
    [260.0,  104.0, 95.0],  # Robot position for P3
    [532.0,  104.0, 95.0]   # Robot position for P4
]
```

### Coordinate Transformation
```python
def chessboard_to_robot(self, board_x, board_y, board_z):
    """Transform chessboard coordinates to robot base coordinates"""
    # Create homogeneous coordinate
    point_board = np.array([board_x, board_y, board_z, 1.0])
    
    # Apply transformation matrix
    point_robot = self.T_matrix @ point_board
    
    # Return 3D robot coordinates
    return point_robot[0], point_robot[1], point_robot[2]
```

## 5. Robot Coordinate System

**Origin:** Robot base center  
**Units:** Millimeters  
**Axes:** X (forward), Y (left), Z (up)  

### Final Robot Pose
```python
# Robot coordinates for chess piece manipulation
robot_x = 365.0  # mm from robot base
robot_y = -92.0  # mm from robot base  
robot_z = 95.0   # mm from robot base (with Z-offset)
```

## Complete Transformation Chain Example

### Input: YOLO Detection
```python
# YOLO detects piece at camera pixel coordinates
camera_pixel = (238, 483)  # pixels in original camera image
```

### Step 1: Perspective Correction
```python
# Apply perspective transformation (handled by OpenCV internally)
rectified_pixel = apply_perspective_transform(camera_pixel)
# Result: (x_rect, y_rect) in rectified view
```

### Step 2: Pixel to Board Coordinates  
```python
board_x = (x_rect / output_width) * board_width
board_y = ((output_height - y_rect) / output_height) * board_height  
board_z = z_offset  # Height above board (e.g., 20mm)

# Example result: (136.0, 60.0, 20.0) mm
```

### Step 3: Board to Robot Coordinates
```python
# Apply hand-eye calibration transformation
point_board = [136.0, 60.0, 20.0, 1.0]  # homogeneous coordinates
point_robot = T_matrix @ point_board

# Example result: (365.0, -92.0, 95.0) mm in robot base frame
```

### Step 4: Robot Movement
```python
# Send movement command to robot
robot.move_to(365.0, -92.0, 95.0)
```

## Calibration Files

### 1. Camera Calibration (camera_calibration.yaml)
```yaml
pixel_corners:
  - [x1, y1]  # User-clicked corner points
  - [x2, y2] 
  - [x3, y3]
  - [x4, y4]
camera_resolution:
  width: 640
  height: 480
output_size:
  width: 640
  height: 480
board_size_mm:
  width: 272.0
  height: 240.0
```

### 2. Hand-Eye Calibration (hand_eye_calibration.yaml)
```yaml
table_points:
  - [0.0, 0.0, 0.0]      # Board frame coordinates
  - [272.0, 0.0, 0.0]
  - [0.0, 240.0, 0.0] 
  - [272.0, 240.0, 0.0]
T_matrix:               # 4x4 transformation matrix
  - [R11, R12, R13, tx]
  - [R21, R22, R23, ty]
  - [R31, R32, R33, tz]
  - [0.0, 0.0, 0.0, 1.0]
total_cols: 8           # Grid configuration
total_rows: 6
```

## Error Sources and Considerations

### 1. Camera Calibration Accuracy
- Corner point selection precision
- Camera distortion (not corrected in current implementation)
- Lighting conditions affecting detection

### 2. Hand-Eye Calibration Accuracy  
- Robot positioning repeatability during teaching
- Measurement accuracy of table points
- Number and distribution of calibration points

### 3. YOLO Detection Accuracy
- Detection confidence threshold
- Piece occlusion or overlap
- Lighting and contrast conditions

### 4. Coordinate Transformation Precision
- Floating-point arithmetic precision
- Matrix condition number
- Calibration point distribution

## Validation and Testing

### Grid Position Verification
The system includes a grid position verification feature:
```python
# Test movement to specific grid positions
def move_to_grid_position(col, row, z_offset):
    # Calculate board coordinates
    cell_width = board_width / total_cols
    cell_height = board_height / total_rows
    board_x = col * cell_width
    board_y = row * cell_height
    
    # Transform to robot coordinates
    robot_x, robot_y, robot_z = chessboard_to_robot(board_x, board_y, z_offset)
    
    # Move robot
    robot.move_to(robot_x, robot_y, robot_z)
```

### Calibration Error Metrics
```python
# Reprojection error calculation
def compute_calibration_error(T_matrix, table_points, robot_points):
    errors = []
    for i in range(len(table_points)):
        # Transform table point to robot frame
        predicted = T_matrix @ np.append(table_points[i], 1)
        actual = robot_points[i]
        error = np.linalg.norm(predicted[:3] - actual)
        errors.append(error)
    
    return np.mean(errors), np.max(errors), errors
```

## Summary

The coordinate transformation chain provides precise mapping from camera-detected piece positions to robot manipulation coordinates through:

1. **Camera pixels** → **Rectified board view** (perspective correction)
2. **Rectified pixels** → **Board millimeters** (scaling and origin shift)  
3. **Board coordinates** → **Robot coordinates** (hand-eye transformation)

This enables accurate robotic manipulation of chess pieces detected by computer vision, with typical accuracy of 1-3mm depending on calibration quality.