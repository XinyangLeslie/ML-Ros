First of all

- `Camera.py`: windows接收原相机主题
- `Camera2.py`: windows接收压缩的相机主题
- `Keyboard_control_velocity.py`：键盘控制ubuntu中的turtlebot机器人移动(W A S D)

## rosbridge

我研究这个rosbridge的原因是：我FasterRCNN 模型训练在实验室windows系统的电脑上，但是由于我们要训练强化学习，turtlebot跑在windows 的VMware中的Ubuntu 22.04+Humble的gazebo环境中，我需要从运行在gazebo世界中的机器人上安装RGB相机或者深度相机，通过订阅 `ros2 topic list` 中的`/camera_sensor/image_raw` 主题获取图片，但是订阅一般只能在ubuntu系统中订阅，我该如何从windows获取到这个实时的视频帧流呢？（因为我要取视频流帧，传给FasterRCNN网络，然后得到预测结果：目标框的x,y, conference, width, height, distance）我网上搜索了一下，发现rosbridge符合我的要求

### Introduction

[`rosbridge`](https://github.com/RobotWebTools/rosbridge_suite) 是 ROS 官方支持的桥接工具，它允许 **非 ROS 系统（如 Windows）通过 WebSocket 与 ROS 节点通信**，可以发布、订阅话题、调用服务，完全适配你这个“ROS 2 → Windows 图像传输”的场景。

### Install

#### 步骤 1：安装 rosbridge 套件

```bash
sudo apt update
sudo apt install ros-humble-rosbridge-server
```

#### 步骤 2：启动 rosbridge WebSocket 服务

```bash
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```

result：

```bash
[INFO] [launch]: All log files can be found below /home/xinyang/.ros/log/2025-04-14-14-31-06-229714-xinyang-virtual-machine-37225
[INFO] [launch]: Default logging verbosity is set to INFO
[INFO] [rosbridge_websocket-1]: process started with pid [37226]
[INFO] [rosapi_node-2]: process started with pid [37228]
[rosbridge_websocket-1] [INFO] [1744655466.585805888] [rosbridge_websocket]: Rosbridge WebSocket server started on port 9090
```

#### 步骤 3：检查 rosbridge 是否启动成功

```bash
ros2 topic list | grep rosbridge
```



#### 步骤 4：Windows 上使用 `roslibpy` 连接并订阅图像话题

前面已经测试过 `roslibpy` 成功接收图像。你只需确认：

- IP 地址是你的 Ubuntu 虚拟机的 IP（例如 `192.168.1.xxx`，不是 localhost）
- 端口是 `9090`
- 话题名是 `/camera_sensor/image_raw` 或 `/camera/image_raw/compressed`



```less
[ Gazebo 模拟相机 ] → /camera_sensor/image_raw
                          |
                          ↓
        [ rosbridge WebSocket @ Ubuntu 虚拟机 (9090) ]
                          |
                          ↓
        [ Windows 上 Python 程序 (roslibpy) ]
                          |
                          ↓
       [ Faster R-CNN 模型进行实时图像预测 ]
                          |
                          ↓
          [ 输出 bbox: x, y, conf, w, h, distance ]

```

#### 步骤五 图像压缩

我测试了使用/camera_sensor/image_raw主题获得图像并传送到windows中，这时我想要测试一下是否有延迟，于是就先在windows中安装了库： `keyboard` 和  `roslibpy` 

```bash
pip install roslibpy
pip install keyboard
```

> - 📚 官方文档：https://roslibpy.readthedocs.io/
> - 🧠 GitHub 源码：https://github.com/gramaziokohler/roslibpy

并使用`Keyboard_control_velocity.py` 代码在外部来控制内部的机器人移动，通过wasd 四个按键。

但是我发现我运行`Keyboard_control_velocity.py`代码之后，并查看获取到的图像，发现卡顿，比如我控制机器人左转，但是windows中监控的视频流过了15s后，相机内容才发生应有的变化，这很不符合我们的要求，因为高延迟代表着图像识别后我们给他发送指令也会有滞后性，因此我想找到问题所在，并如何解决问题。

网上建议使用 `/compressed` 话题订阅，说减少网络负担，我了解了一下，这个`/compressed` 实际是一个`/camera_sensor/image_raw/compressed`，是基于原有的`/camera_sensor/image_raw`主题，使用一个插件，压缩原有的主题，并发布为新主题。

```bash
# 1. Install the image_transport plugin
#This plugin automatically generates /compressed topics for all sensor_msgs/Image image topics.

sudo apt update
sudo apt install ros-humble-image-transport-plugins

# 2. Run Gazebo emulation + start camera
#Make sure your robot emulation is started and that the camera 
#is started (i.e. publish /camera/image_raw). Then run it again:

ros2 topic list | grep compressed

# 3. If the front process is not working, try reboot/restart the ubuntu

# 4.If you still don't see /compressed, you can start the forwarding node manually:

ros2 run image_transport republish raw --ros-args -r in:=/camera/image_raw -r out:=/camera/image_raw/compressed

```
![vmware_0KCy19S5xt](https://github.com/user-attachments/assets/eafe45a2-7fb8-4b5f-be21-460b72fbda5c)

<img src="./resources/images/vmware_0KCy19S5xt.png" alt="vmware_0KCy19S5xt" style="zoom:50%;" />

After completing the above steps, you can subscribe like this on Windows:

```python
listener = roslibpy.Topic(client, '/camera/image_raw/compressed', 'sensor_msgs/CompressedImage')
```

最后，我新建了一个代码`Camera2.py` ，来展示效果，发现是实时性的

后面，我会使用PPO agent的代码来替代这个手动键入的控制方式
