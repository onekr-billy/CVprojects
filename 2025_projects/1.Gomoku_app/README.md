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
* set your Robot arm to suport ROS2 first, tested on my Episode1 robot arm. 
  * [check the robot link](https://enpeicv.com/forum.php?mod=viewthread&tid=1217&extra=page%3D1)
  * [check the Episode1 ros2 demo](https://pan.baidu.com/s/1eDa-Bre2ruY9CT_rWhFDVw?pwd=fak8&_at_=1766390971553#list/path=%2F)
* colcon build --symlink-install --packages-select robot_arm_interfaces
* python 13.chess_demo_ros2.py to start main program
