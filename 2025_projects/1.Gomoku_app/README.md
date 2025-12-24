## Gomoku_app

> If you have any questions, please submit issues or email me: enpeicv@outlook.com, have fun with it!
>
> 扫码加入微信WeChat交流群：
>
> <img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202404161532376.png?x-oss-process=style/wp" style="width:200px;" />

## 1.Demo

|![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202512221602572.png?x-oss-process=style/resize) | ![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202512221604327.png?x-oss-process=style/resize) | ![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202512221604912.png?x-oss-process=style/resize) |
|---|---|---|

![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202512221558156.png?x-oss-process=style/wp)

## 2.Requirements
* Ros2 jazzy
* ubuntu 24.04
* Robot arm
* chessboard and pieces
* 1208x720 usb webcam
* depth camera (optional)

## 3.Usage
* Set your Robot arm to suport ROS2 first, tested on my Episode1 robot arm. 
  * [check the robot ros2 implementation demo](https://enpeicv.com/forum.php?mod=viewthread&tid=1217&extra=page%3D1)

* `colcon build --symlink-install --packages-select robot_arm_interfaces` to build robot arm interfaces package first
* Calibration:
  * `python 2.calibrate_perspective_camera_qt.py` to calibrate perspective transform of chessboard
  * `python 4.calibrate_hand_eye_qt_ros2.py` to calibrate hand-eye of robot arm and chessboard
  * `python 7.calibrate_storage_camera.py` to calibrate storage area (for robot arm to pick pieces)
  * `python 9.calibrate_storage_hand_eye.py` to calibrate hand-eye of robot arm and storage area
  * you will get four yaml files in calibration_scripts
* `python 13.chess_demo_ros2.py` to start main program
* Gomoku related files are in Gomoku_app folder
