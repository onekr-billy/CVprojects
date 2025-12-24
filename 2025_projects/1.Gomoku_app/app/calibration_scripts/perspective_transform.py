"""
透视变换模块：相机像素坐标 → 棋盘坐标

用于五子棋机械臂项目，将相机拍摄的图像中的像素坐标
转换为棋盘平面上的实际物理坐标（单位：毫米）
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


class PerspectiveTransformer:
    """透视变换器：像素坐标 ↔ 棋盘坐标"""
    
    def __init__(self):
        self.H: Optional[np.ndarray] = None  # 像素→棋盘 的变换矩阵
        self.H_inv: Optional[np.ndarray] = None  # 棋盘→像素 的逆变换矩阵
        self.is_calibrated: bool = False
    
    def calibrate_with_4_points(
        self,
        pixel_points: np.ndarray,
        board_points: np.ndarray
    ) -> bool:
        """
        使用4个对应点进行标定
        
        Args:
            pixel_points: 4个像素坐标 [[u1,v1], [u2,v2], [u3,v3], [u4,v4]]
            board_points: 4个对应的棋盘坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        
        Returns:
            bool: 标定是否成功
        """
        if len(pixel_points) != 4 or len(board_points) != 4:
            print("错误：需要恰好4个点")
            return False
        
        pixel_pts = np.float32(pixel_points)
        board_pts = np.float32(board_points)
        
        # 计算透视变换矩阵：像素 → 棋盘
        self.H = cv2.getPerspectiveTransform(pixel_pts, board_pts)
        # 计算逆变换矩阵：棋盘 → 像素
        self.H_inv = cv2.getPerspectiveTransform(board_pts, pixel_pts)
        
        self.is_calibrated = True
        print("标定成功！")
        print(f"变换矩阵 H:\n{self.H}")
        return True
    
    def calibrate_with_n_points(
        self,
        pixel_points: np.ndarray,
        board_points: np.ndarray,
        method: int = cv2.RANSAC,
        reproj_threshold: float = 5.0
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        使用多个点进行标定（最小二乘拟合，更鲁棒）
        
        Args:
            pixel_points: N个像素坐标 (N>=4)
            board_points: N个对应的棋盘坐标
            method: 求解方法 (0, cv2.RANSAC, cv2.LMEDS, cv2.RHO)
            reproj_threshold: RANSAC的重投影误差阈值
        
        Returns:
            Tuple[bool, mask]: (是否成功, 内点掩码)
        """
        if len(pixel_points) < 4 or len(board_points) < 4:
            print("错误：至少需要4个点")
            return False, None
        
        pixel_pts = np.float32(pixel_points)
        board_pts = np.float32(board_points)
        
        # 使用 findHomography 求解（支持多点最小二乘）
        self.H, mask = cv2.findHomography(pixel_pts, board_pts, method, reproj_threshold)
        
        if self.H is None:
            print("错误：无法计算变换矩阵")
            return False, None
        
        self.H_inv = np.linalg.inv(self.H)
        self.is_calibrated = True
        
        inliers = np.sum(mask) if mask is not None else len(pixel_points)
        print(f"标定成功！使用 {inliers}/{len(pixel_points)} 个内点")
        print(f"变换矩阵 H:\n{self.H}")
        return True, mask
    
    def pixel_to_board(self, pixel_point: Tuple[float, float]) -> Tuple[float, float]:
        """
        像素坐标 → 棋盘坐标
        
        Args:
            pixel_point: (u, v) 像素坐标
        
        Returns:
            (x, y) 棋盘坐标（单位：毫米）
        """
        if not self.is_calibrated:
            raise RuntimeError("请先进行标定！")
        
        u, v = pixel_point
        pt = np.float32([[[u, v]]])
        result = cv2.perspectiveTransform(pt, self.H)
        x, y = result[0][0]
        return float(x), float(y)
    
    def board_to_pixel(self, board_point: Tuple[float, float]) -> Tuple[float, float]:
        """
        棋盘坐标 → 像素坐标（逆变换）
        
        Args:
            board_point: (x, y) 棋盘坐标
        
        Returns:
            (u, v) 像素坐标
        """
        if not self.is_calibrated:
            raise RuntimeError("请先进行标定！")
        
        x, y = board_point
        pt = np.float32([[[x, y]]])
        result = cv2.perspectiveTransform(pt, self.H_inv)
        u, v = result[0][0]
        return float(u), float(v)
    
    def pixel_to_board_batch(self, pixel_points: np.ndarray) -> np.ndarray:
        """
        批量转换：像素坐标 → 棋盘坐标
        
        Args:
            pixel_points: Nx2 数组，N个像素坐标
        
        Returns:
            Nx2 数组，N个棋盘坐标
        """
        if not self.is_calibrated:
            raise RuntimeError("请先进行标定！")
        
        pts = np.float32(pixel_points).reshape(-1, 1, 2)
        result = cv2.perspectiveTransform(pts, self.H)
        return result.reshape(-1, 2)
    
    def compute_reprojection_error(
        self,
        pixel_points: np.ndarray,
        board_points: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        计算重投影误差，用于评估标定精度
        
        Args:
            pixel_points: 原始像素坐标
            board_points: 对应的真实棋盘坐标
        
        Returns:
            (平均误差, 每个点的误差)
        """
        if not self.is_calibrated:
            raise RuntimeError("请先进行标定！")
        
        # 将像素点转换到棋盘坐标
        predicted = self.pixel_to_board_batch(pixel_points)
        
        # 计算误差
        errors = np.linalg.norm(predicted - board_points, axis=1)
        mean_error = np.mean(errors)
        
        return mean_error, errors
    
    def save_calibration(self, filepath: str):
        """保存标定结果"""
        if not self.is_calibrated:
            raise RuntimeError("请先进行标定！")
        np.savez(filepath, H=self.H, H_inv=self.H_inv)
        print(f"标定结果已保存到: {filepath}")
    
    def load_calibration(self, filepath: str):
        """加载标定结果"""
        data = np.load(filepath)
        self.H = data['H']
        self.H_inv = data['H_inv']
        self.is_calibrated = True
        print(f"已加载标定结果: {filepath}")


# ==================== 测试代码 ====================

def test_perspective_transform():
    """测试透视变换功能"""
    print("=" * 60)
    print("透视变换测试")
    print("=" * 60)
    
    transformer = PerspectiveTransformer()
    
    # 模拟场景：
    # 棋盘实际尺寸：300mm x 300mm
    # 相机看到的棋盘四个角点的像素坐标（模拟透视畸变）
    
    # 棋盘四角的像素坐标（模拟相机拍摄的图像）
    pixel_corners = np.array([
        [100, 80],    # 左下角
        [540, 100],   # 右下角
        [500, 400],   # 右上角
        [120, 380],   # 左上角
    ], dtype=np.float32)
    
    # 棋盘四角的实际物理坐标（单位：mm）
    board_corners = np.array([
        [0, 0],       # 左下角
        [300, 0],     # 右下角
        [300, 300],   # 右上角
        [0, 300],     # 左上角
    ], dtype=np.float32)
    
    # ===== 测试1：4点标定 =====
    print("\n【测试1】使用4个角点进行标定")
    print("-" * 40)
    success = transformer.calibrate_with_4_points(pixel_corners, board_corners)
    assert success, "标定失败！"
    
    # ===== 测试2：单点转换 =====
    print("\n【测试2】单点转换测试")
    print("-" * 40)
    
    # 测试角点转换（应该精确）
    for i, (pixel_pt, board_pt) in enumerate(zip(pixel_corners, board_corners)):
        result = transformer.pixel_to_board(tuple(pixel_pt))
        print(f"角点{i+1}: 像素{tuple(pixel_pt)} → 棋盘{result}")
        print(f"        期望值: {tuple(board_pt)}")
        # 验证误差
        error = np.sqrt((result[0] - board_pt[0])**2 + (result[1] - board_pt[1])**2)
        assert error < 0.01, f"角点转换误差过大: {error}"
    
    print("\n✓ 角点转换精度验证通过！")
    
    # ===== 测试3：棋盘中心点 =====
    print("\n【测试3】棋盘中心点转换")
    print("-" * 40)
    
    # 计算像素坐标系中的"中心"（四边形中心）
    pixel_center = np.mean(pixel_corners, axis=0)
    board_result = transformer.pixel_to_board(tuple(pixel_center))
    print(f"像素中心: {tuple(pixel_center)}")
    print(f"转换结果: {board_result}")
    print(f"期望值约: (150, 150)")
    
    # ===== 测试4：逆变换 =====
    print("\n【测试4】逆变换测试（棋盘→像素）")
    print("-" * 40)
    
    test_board_point = (150, 150)  # 棋盘中心
    pixel_result = transformer.board_to_pixel(test_board_point)
    print(f"棋盘坐标: {test_board_point}")
    print(f"转换到像素: {pixel_result}")
    
    # 再转回去验证
    back_to_board = transformer.pixel_to_board(pixel_result)
    print(f"再转回棋盘: {back_to_board}")
    error = np.sqrt((back_to_board[0] - test_board_point[0])**2 + 
                    (back_to_board[1] - test_board_point[1])**2)
    assert error < 0.01, f"往返转换误差过大: {error}"
    print("✓ 往返转换精度验证通过！")
    
    # ===== 测试5：批量转换 =====
    print("\n【测试5】批量转换测试")
    print("-" * 40)
    
    # 模拟检测到5个棋子的像素坐标
    chess_pixels = np.array([
        [150, 130],
        [320, 150],
        [250, 240],
        [180, 300],
        [400, 320],
    ], dtype=np.float32)
    
    chess_boards = transformer.pixel_to_board_batch(chess_pixels)
    print("棋子位置转换结果：")
    for i, (px, bd) in enumerate(zip(chess_pixels, chess_boards)):
        print(f"  棋子{i+1}: 像素{tuple(px)} → 棋盘({bd[0]:.1f}, {bd[1]:.1f}) mm")
    
    # ===== 测试6：多点标定（使用更多点） =====
    print("\n【测试6】多点标定测试（8个点）")
    print("-" * 40)
    
    transformer2 = PerspectiveTransformer()
    
    # 添加更多标定点（4角 + 4边中点）
    pixel_points_8 = np.array([
        [100, 80],    # 左下角
        [540, 100],   # 右下角
        [500, 400],   # 右上角
        [120, 380],   # 左上角
        [320, 90],    # 下边中点
        [520, 250],   # 右边中点
        [310, 390],   # 上边中点
        [110, 230],   # 左边中点
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
    assert success, "多点标定失败！"
    
    # 计算重投影误差
    mean_error, errors = transformer2.compute_reprojection_error(pixel_points_8, board_points_8)
    print(f"\n重投影误差：")
    print(f"  平均误差: {mean_error:.4f} mm")
    print(f"  最大误差: {np.max(errors):.4f} mm")
    print(f"  最小误差: {np.min(errors):.4f} mm")
    
    # ===== 测试7：保存和加载 =====
    print("\n【测试7】保存和加载标定结果")
    print("-" * 40)
    
    save_path = "/home/box/Documents/Episode1_robot_arm_controller/ros2/src/episode_apps/calibration.npz"
    transformer.save_calibration(save_path)
    
    transformer3 = PerspectiveTransformer()
    transformer3.load_calibration(save_path)
    
    # 验证加载后的转换结果
    test_result = transformer3.pixel_to_board((100, 80))
    assert abs(test_result[0]) < 0.01 and abs(test_result[1]) < 0.01
    print("✓ 保存/加载验证通过！")
    
    print("\n" + "=" * 60)
    print("所有测试通过！ ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_perspective_transform()
