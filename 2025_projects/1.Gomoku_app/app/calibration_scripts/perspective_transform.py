"""
Perspective Transform Module: Camera Pixel Coordinates → Board Coordinates

Used in Gomoku robotic arm project to convert pixel coordinates 
from camera images to actual physical coordinates on the board plane (unit: millimeters)
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


class PerspectiveTransformer:
    """Perspective Transformer: Pixel Coordinates ↔ Board Coordinates"""
    
    def __init__(self):
        self.H: Optional[np.ndarray] = None  # Transformation matrix: pixel → board
        self.H_inv: Optional[np.ndarray] = None  # Inverse transformation matrix: board → pixel
        self.is_calibrated: bool = False
    
    def calibrate_with_4_points(
        self,
        pixel_points: np.ndarray,
        board_points: np.ndarray
    ) -> bool:
        """
        Calibrate using 4 corresponding points
        
        Args:
            pixel_points: 4 pixel coordinates [[u1,v1], [u2,v2], [u3,v3], [u4,v4]]
            board_points: 4 corresponding board coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        
        Returns:
            bool: Whether calibration was successful
        """
        if len(pixel_points) != 4 or len(board_points) != 4:
            print("Error: Exactly 4 points are required")
            return False
        
        pixel_pts = np.float32(pixel_points)
        board_pts = np.float32(board_points)
        
        # Calculate perspective transformation matrix: pixel → board
        self.H = cv2.getPerspectiveTransform(pixel_pts, board_pts)
        # Calculate inverse transformation matrix: board → pixel
        self.H_inv = cv2.getPerspectiveTransform(board_pts, pixel_pts)
        
        self.is_calibrated = True
        print("Calibration successful!")
        print(f"Transformation matrix H:\n{self.H}")
        return True
    
    def calibrate_with_n_points(
        self,
        pixel_points: np.ndarray,
        board_points: np.ndarray,
        method: int = cv2.RANSAC,
        reproj_threshold: float = 5.0
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Calibrate using multiple points (least squares fitting, more robust)
        
        Args:
            pixel_points: N pixel coordinates (N>=4)
            board_points: N corresponding board coordinates
            method: Solution method (0, cv2.RANSAC, cv2.LMEDS, cv2.RHO)
            reproj_threshold: Reprojection error threshold for RANSAC
        
        Returns:
            Tuple[bool, mask]: (success status, inlier mask)
        """
        if len(pixel_points) < 4 or len(board_points) < 4:
            print("Error: At least 4 points are required")
            return False, None
        
        pixel_pts = np.float32(pixel_points)
        board_pts = np.float32(board_points)
        
        # Use findHomography to solve (supports multi-point least squares)
        self.H, mask = cv2.findHomography(pixel_pts, board_pts, method, reproj_threshold)
        
        if self.H is None:
            print("Error: Unable to calculate transformation matrix")
            return False, None
        
        self.H_inv = np.linalg.inv(self.H)
        self.is_calibrated = True
        
        inliers = np.sum(mask) if mask is not None else len(pixel_points)
        print(f"Calibration successful! Using {inliers}/{len(pixel_points)} inliers")
        print(f"Transformation matrix H:\n{self.H}")
        return True, mask
    
    def pixel_to_board(self, pixel_point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Pixel coordinates → Board coordinates
        
        Args:
            pixel_point: (u, v) pixel coordinates
        
        Returns:
            (x, y) board coordinates (unit: millimeters)
        """
        if not self.is_calibrated:
            raise RuntimeError("Please calibrate first!")
        
        u, v = pixel_point
        pt = np.float32([[[u, v]]])
        result = cv2.perspectiveTransform(pt, self.H)
        x, y = result[0][0]
        return float(x), float(y)
    
    def board_to_pixel(self, board_point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Board coordinates → Pixel coordinates (inverse transformation)
        
        Args:
            board_point: (x, y) board coordinates
        
        Returns:
            (u, v) pixel coordinates
        """
        if not self.is_calibrated:
            raise RuntimeError("Please calibrate first!")
        
        x, y = board_point
        pt = np.float32([[[x, y]]])
        result = cv2.perspectiveTransform(pt, self.H_inv)
        u, v = result[0][0]
        return float(u), float(v)
    
    def pixel_to_board_batch(self, pixel_points: np.ndarray) -> np.ndarray:
        """
        Batch conversion: Pixel coordinates → Board coordinates
        
        Args:
            pixel_points: Nx2 array, N pixel coordinates
        
        Returns:
            Nx2 array, N board coordinates
        """
        if not self.is_calibrated:
            raise RuntimeError("Please calibrate first!")
        
        pts = np.float32(pixel_points).reshape(-1, 1, 2)
        result = cv2.perspectiveTransform(pts, self.H)
        return result.reshape(-1, 2)
    
    def compute_reprojection_error(
        self,
        pixel_points: np.ndarray,
        board_points: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Calculate reprojection error to evaluate calibration accuracy
        
        Args:
            pixel_points: Original pixel coordinates
            board_points: Corresponding actual board coordinates
        
        Returns:
            (mean error, error for each point)
        """
        if not self.is_calibrated:
            raise RuntimeError("Please calibrate first!")
        
        # Convert pixel points to board coordinates
        predicted = self.pixel_to_board_batch(pixel_points)
        
        # Calculate errors
        errors = np.linalg.norm(predicted - board_points, axis=1)
        mean_error = np.mean(errors)
        
        return mean_error, errors
    
    def save_calibration(self, filepath: str):
        """Save calibration results"""
        if not self.is_calibrated:
            raise RuntimeError("Please calibrate first!")
        np.savez(filepath, H=self.H, H_inv=self.H_inv)
        print(f"Calibration results saved to: {filepath}")
    
    def load_calibration(self, filepath: str):
        """Load calibration results"""
        data = np.load(filepath)
        self.H = data['H']
        self.H_inv = data['H_inv']
        self.is_calibrated = True
        print(f"Calibration results loaded: {filepath}")


# ==================== Test Code ====================

def test_perspective_transform():
    """Test perspective transformation functionality"""
    print("=" * 60)
    print("Perspective Transformation Test")
    print("=" * 60)
    
    transformer = PerspectiveTransformer()
    
    # Simulation scenario:
    # Board actual size: 300mm x 300mm
    # Pixel coordinates of the four corner points seen by camera (simulating perspective distortion)
    
    # Pixel coordinates of board corners (simulating camera captured image)
    pixel_corners = np.array([
        [100, 80],    # Bottom-left corner
        [540, 100],   # Bottom-right corner
        [500, 400],   # Top-right corner
        [120, 380],   # Top-left corner
    ], dtype=np.float32)
    
    # Actual physical coordinates of board corners (unit: mm)
    board_corners = np.array([
        [0, 0],       # Bottom-left corner
        [300, 0],     # Bottom-right corner
        [300, 300],   # Top-right corner
        [0, 300],     # Top-left corner
    ], dtype=np.float32)
    
    # ===== Test 1: 4-point calibration =====
    print("\n[Test 1] Calibration using 4 corner points")
    print("-" * 40)
    success = transformer.calibrate_with_4_points(pixel_corners, board_corners)
    assert success, "Calibration failed!"
    
    # ===== Test 2: Single point conversion =====
    print("\n[Test 2] Single point conversion test")
    print("-" * 40)
    
    # Test corner point conversion (should be accurate)
    for i, (pixel_pt, board_pt) in enumerate(zip(pixel_corners, board_corners)):
        result = transformer.pixel_to_board(tuple(pixel_pt))
        print(f"Corner {i+1}: Pixel{tuple(pixel_pt)} → Board{result}")
        print(f"        Expected: {tuple(board_pt)}")
        # Verify error
        error = np.sqrt((result[0] - board_pt[0])**2 + (result[1] - board_pt[1])**2)
        assert error < 0.01, f"Corner conversion error too large: {error}"
    
    print("\n✓ Corner conversion accuracy verification passed!")
    
    # ===== Test 3: Board center point =====
    print("\n[Test 3] Board center point conversion")
    print("-" * 40)
    
    # Calculate "center" in pixel coordinate system (quadrilateral center)
    pixel_center = np.mean(pixel_corners, axis=0)
    board_result = transformer.pixel_to_board(tuple(pixel_center))
    print(f"Pixel center: {tuple(pixel_center)}")
    print(f"Conversion result: {board_result}")
    print(f"Expected approx: (150, 150)")
    
    # ===== Test 4: Inverse transformation =====
    print("\n[Test 4] Inverse transformation test (Board → Pixel)")
    print("-" * 40)
    
    test_board_point = (150, 150)  # Board center
    pixel_result = transformer.board_to_pixel(test_board_point)
    print(f"Board coordinates: {test_board_point}")
    print(f"Convert to pixel: {pixel_result}")
    
    # Convert back to verify
    back_to_board = transformer.pixel_to_board(pixel_result)
    print(f"Convert back to board: {back_to_board}")
    error = np.sqrt((back_to_board[0] - test_board_point[0])**2 + 
                    (back_to_board[1] - test_board_point[1])**2)
    assert error < 0.01, f"Round-trip conversion error too large: {error}"
    print("✓ Round-trip conversion accuracy verification passed!")
    
    # ===== Test 5: Batch conversion =====
    print("\n[Test 5] Batch conversion test")
    print("-" * 40)
    
    # Simulate detected pixel coordinates of 5 chess pieces
    chess_pixels = np.array([
        [150, 130],
        [320, 150],
        [250, 240],
        [180, 300],
        [400, 320],
    ], dtype=np.float32)
    
    chess_boards = transformer.pixel_to_board_batch(chess_pixels)
    print("Chess piece position conversion results:")
    for i, (px, bd) in enumerate(zip(chess_pixels, chess_boards)):
        print(f"  Piece {i+1}: Pixel{tuple(px)} → Board({bd[0]:.1f}, {bd[1]:.1f}) mm")
    
    # ===== Test 6: Multi-point calibration (using more points) =====
    print("\n[Test 6] Multi-point calibration test (8 points)")
    print("-" * 40)
    
    transformer2 = PerspectiveTransformer()
    
    # Add more calibration points (4 corners + 4 edge midpoints)
    pixel_points_8 = np.array([
        [100, 80],    # Bottom-left corner
        [540, 100],   # Bottom-right corner
        [500, 400],   # Top-right corner
        [120, 380],   # Top-left corner
        [320, 90],    # Bottom edge midpoint
        [520, 250],   # Right edge midpoint
        [310, 390],   # Top edge midpoint
        [110, 230],   # Left edge midpoint
    ], dtype=np.float32)
    
    board_points_8 = np.array([
        [0, 0],
        [300, 0],
        [300, 300],
        [0, 300],
        [150, 0],
        [300, 150],
        [150, 300],
        [0, 150],
    ], dtype=np.float32)
    
    success, mask = transformer2.calibrate_with_n_points(pixel_points_8, board_points_8)
    assert success, "Multi-point calibration failed!"
    
    # Calculate reprojection error
    mean_error, errors = transformer2.compute_reprojection_error(pixel_points_8, board_points_8)
    print(f"\nReprojection error:")
    print(f"  Mean error: {mean_error:.4f} mm")
    print(f"  Max error: {np.max(errors):.4f} mm")
    print(f"  Min error: {np.min(errors):.4f} mm")
    
    # ===== Test 7: Save and load =====
    print("\n[Test 7] Save and load calibration results")
    print("-" * 40)
    
    save_path = "/home/box/Documents/Episode1_robot_arm_controller/ros2/src/episode_apps/calibration.npz"
    transformer.save_calibration(save_path)
    
    transformer3 = PerspectiveTransformer()
    transformer3.load_calibration(save_path)
    
    # Verify conversion results after loading
    test_result = transformer3.pixel_to_board((100, 80))
    assert abs(test_result[0]) < 0.01 and abs(test_result[1]) < 0.01
    print("✓ Save/load verification passed!")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_perspective_transform()
