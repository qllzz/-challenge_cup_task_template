# 机器人控制与感知 API 速查手册

封装了所有底层 ROS 话题/服务，你只需要 `import` + 调方法，不用关心中间细节。

## 目录

- [快速上手](#快速上手)
- [RobotMover — 走路](#robotmover--走路)
- [ArmController — 手臂](#armcontroller--手臂)
- [ClawController — 夹爪](#clawcontroller--夹爪)
- [HeadController — 头部](#headcontroller--头部)
- [CameraReader — 相机](#camerareader--相机)
- [LidarReader — 激光雷达](#lidarreader--激光雷达)
- [SensorReader — 传感器](#sensorreader--传感器)
- [TFReader — 坐标系查询](#tfreader--坐标系查询)
- [完整示例一：用视觉 + 点云定位物体](#完整示例一用视觉--点云定位物体)
- [完整示例二：快递抓取流程](#完整示例二快递抓取流程)

---

## 快速上手

你的场景文件 `sceneX_task.py` 已经收到了 4 个控制器对象，直接用：

```python
def run_scene1(robot, arm, claw, head, log):
    # robot — 走路
    # arm   — 手臂 + IK
    # claw  — 夹爪
    # head  — 头部转动
    # log   — 打日志（用法同 print）
```

如果需要感知模块，自己在文件顶部 import：

```python
import sys, os
_pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_pkg, "src"))
from perception_api import CameraReader, LidarReader, SensorReader, TFReader
```

---

## RobotMover — 走路

**封装话题**：`/cmd_vel`

### 方法表

| 方法 | 参数 | 效果 |
|------|------|------|
| `move_forward(speed, duration=None)` | speed: 速度 m/s | 前进，有 duration 到时自动停 |
| `move_backward(speed, duration=None)` | speed: 正数即可 | 后退 |
| `move_left(speed, duration=None)` | speed: 速度 m/s | 左移（⚠ 双足危险，≤0.05） |
| `move_right(speed, duration=None)` | speed: 速度 m/s | 右移（⚠ 双足危险，≤0.05） |
| `turn_left(speed, duration=None)` | speed: rad/s（≈0.2~0.5） | 左转 |
| `turn_right(speed, duration=None)` | speed: rad/s | 右转 |
| `move(forward, left, turn, duration)` | 三个方向速度 + 可选时长 | 组合移动 |
| `stop()` | 无 | 立即停，切回站立 |
| `switch_to_walk()` | 无 | 手动切行走（通常自动，无需调） |
| `switch_to_stance()` | 无 | 手动切站立（精细操作前建议调） |

### 示例

```python
# 走两步后自动停
robot.move_forward(0.1, duration=2.0)

# 边前进边微调方向
robot.move(forward=0.08, turn=0.2, duration=3.0)

# 精细操作前切站立，身体更稳
robot.switch_to_stance()
```

---

## ArmController — 手臂

**封装话题/服务**：`/kuavo_arm_traj`、`/humanoid_change_arm_ctrl_mode`、`/ik/two_arm_hand_pose_cmd_srv`

### 方法表

| 方法 | 参数 | 效果 |
|------|------|------|
| `switch_to_external_control()` | 无 | **操作前必须调！** 切到外部控制模式 |
| `go_home()` | 无 | 手臂归零，自然下垂 |
| `go_ready()` | 无 | 双手前伸（肩 30° 肘 60°），准备抓取 |
| `go_to_joints([14个角度])` | 14 个 float，单位**度** | 直接控制 14 个关节 |
| `left_arm_to([7个角度])` | 7 个 float | 只动左臂 |
| `right_arm_to([7个角度])` | 7 个 float | 只动右臂 |
| `solve_ik(...)` | 见下方 | IK 逆解，输入末端位姿 → 输出关节角度 |

### 14 关节顺序

```
左臂 (索引 0~6): 肩俯仰, 肩侧摆, 肩旋转, 肘, 腕旋转, 腕俯仰, 腕侧摆
右臂 (索引 7~13): 同上
```

### IK 求解

```python
ok, q_arm = arm.solve_ik(
    left_pose_xyz=[x, y, z],       # 左手末端目标位置 (米)
    left_quat_xyzw=[qx, qy, qz, qw],  # 左手末端姿态四元数
    right_pose_xyz=[x, y, z],      # 右手末端目标位置 (米)
    right_quat_xyzw=[qx, qy, qz, qw], # 右手末端姿态四元数
    frame=2,                        # 坐标系：2=局部(默认)，1=odom
    use_current_as_q0=True          # 用当前关节角做初值（通常 True）
)
if ok:
    arm.go_to_joints(q_arm)         # 执行 IK 结果
```

### 示例

```python
# 操作前必须
arm.switch_to_external_control()

# 预设姿势
arm.go_ready()          # 快速到准备抓取位置
arm.go_home()           # 回到自然下垂

# 精确控制
arm.go_to_joints([30, -10, 0, -60, 0, 0, 0,      # 左臂
                   30, 10, 0, -60, 0, 0, 0])     # 右臂

# 用 IK 自动算关节角
ok, q = arm.solve_ik(
    [0.5, 0.15, 0.7], [0, 0, 0, 1],
    [0.5, -0.15, 0.7], [0, 0, 0, 1],
)
if ok: arm.go_to_joints(q)
```

---

## ClawController — 夹爪

**封装服务/话题**：`/control_robot_leju_claw`、`/leju_claw_state`

### 方法表

| 方法 | 参数 | 效果 |
|------|------|------|
| `open()` | 无 | 双手张开（默认 10%） |
| `close()` | 无 | 双手闭合（默认 90%） |
| `set_position(左%, 右%)` | 0=全开, 100=全闭 | 分别控制左右 |
| `left_open()` / `left_close()` | 无 | 只动左手 |
| `right_open()` / `right_close()` | 无 | 只动右手 |
| `is_grabbed()` | 无 | 返回 `True`/`False`，是否抓住物体 |
| `is_moving()` | 无 | 返回 `True`/`False`，夹爪运动是否完成 |
| `wait_until_done(timeout=秒)` | timeout 默认 5 | 阻塞等待夹爪运动完成，超时返回 False |

### 状态值对照

| 值 | 含义 |
|:--:|------|
| -1 | 出错 |
| 0 | 未知（刚初始化） |
| 1 | 移动中 |
| 2 | 已到达目标位置 |
| 3 | **已抓住物体** ← 抓取成功的信号 |

### 示例

```python
claw.open()                      # 先张开
# ... 手臂移到抓取位置 ...
claw.close()                     # 闭合抓取
claw.wait_until_done(timeout=3.0)

if claw.is_grabbed():
    log("抓住了！")
else:
    claw.open()                  # 没抓住，张开重来
```

---

## HeadController — 头部

**封装话题**：`/robot_head_motion_data`

### 方法表

| 方法 | 参数 | 效果 |
|------|------|------|
| `look_at(yaw, pitch)` | yaw: 偏航角 (-30~30°) 正=右看<br>pitch: 俯仰角 (-25~25°) 正=抬头 | 精确控制 |
| `look_forward()` | 无 | 直视前方 |
| `look_left(angle=15)` | angle: 角度 | 向左看 |
| `look_right(angle=15)` | angle: 角度 | 向右看 |
| `look_up(angle=15)` | angle: 角度 | 抬头 |
| `look_down(angle=15)` | angle: 角度 | 低头 |

### 示例

```python
head.look_down(20)    # 低头看桌面
head.look_left(15)    # 左看看
head.look_right(15)   # 右看看
head.look_forward()   # 正视前方
```

---

## CameraReader — 相机

**封装话题**：头部/左腕/右腕 RGB 压缩图 + 深度图

⚠ 需要 `opencv-python`，容器内：`pip install opencv-python`

| 方法 | 返回 |
|------|------|
| `get_head_rgb()` | 头部 RGB，numpy (H, W, 3)，BGR |
| `get_left_wrist_rgb()` | 左腕 RGB |
| `get_right_wrist_rgb()` | 右腕 RGB |
| `get_head_depth()` | 头部深度 (H, W)，float32，单位**米** |
| `get_left_wrist_depth()` | 左腕深度 |
| `get_right_wrist_depth()` | 右腕深度 |
| `has_new(key)` | 指定相机是否有新帧 |

### 示例

```python
import cv2

rgb = cam.get_head_rgb()           # numpy (720, 1280, 3)
depth = cam.get_head_depth()       # numpy (720, 1280)，值=几米

# 画面中心点的距离
h, w = depth.shape
dist = depth[h//2, w//2]           # 单位 米

# 保存画面调试
cv2.imwrite("/tmp/head.png", rgb)

# 按深度找最近的物体区域
mask = (depth > 0.3) & (depth < 1.5)  # 0.3~1.5m 范围
close_pixels = depth[mask]
if len(close_pixels) > 0:
    print("最近物体距离: %.2f 米" % close_pixels.min())
```

---

## LidarReader — 激光雷达

**封装话题**：`/lidar/points` (PointCloud2)

初始化等待 2 秒。10Hz，约 24000 个点/帧。

| 方法 | 返回 |
|------|------|
| `get_points()` | 全部点云 N×3 (x, y, z)，单位 米 |
| `get_points_2d()` | N×2 只看 xy 平面 |
| `get_points_in_region(x_range, y_range, z_range=None)` | 框选区域内点 N×3 |

### 示例

```python
lidar = LidarReader()

# 只看桌面高度（0.5~1.2m）面前 0.3~2m 的东西
pts = lidar.get_points_in_region(
    x_range=(0.3, 2.0),     # 前方
    y_range=(-0.6, 0.6),    # 左右
    z_range=(0.4, 1.2)      # 高度（桌面区域）
)

if len(pts) > 50:
    center = pts.mean(axis=0)       # 物体中心
    x_size = pts[:,0].ptp()         # x 方向尺寸
    y_size = pts[:,1].ptp()         # y 方向尺寸
    log("物体中心: x=%.2f y=%.2f z=%.2f 尺寸: %.2f×%.2f",
        center[0], center[1], center[2], x_size, y_size)
```

---

## SensorReader — 传感器

**封装话题**：`/sensors_data_raw`

| 方法 | 返回 | 单位 |
|------|------|------|
| `get_joint_q()` | 28 关节位置 | rad |
| `get_joint_v()` | 28 关节速度 | rad/s |
| `get_joint_degrees()` | 28 关节位置 | **度** |
| `get_arm_joint_degrees()` | 双臂 14 关节位置 | **度** |
| `get_imu_quat()` | 姿态四元数 (x, y, z, w) | — |
| `get_imu_acc()` | 加速度 (x, y, z) | m/s² |
| `get_imu_gyro()` | 角速度 (x, y, z) | rad/s |
| `get_claw_position()` | 夹爪 [左%, 右%] | 0~100 |

### 28 关节索引

```
 0~5   左腿 (髋侧摆, 髋偏航, 髋俯仰, 膝, 踝俯仰, 踝侧摆)
 6~11  右腿 (同上)
12~18  左臂 (肩俯仰, 肩侧摆, 肩旋转, 肘, 腕旋转, 腕俯仰, 腕侧摆)
19~25  右臂 (同上)
26     头 yaw
27     头 pitch
```

### 示例

```python
sensor = SensorReader()

# 检查手臂是否到位
current = sensor.get_arm_joint_degrees()
target  = [30, -10, 0, -60, 0, 0, 0, 30, 10, 0, -60, 0, 0, 0]
diff = sum(abs(c - t) for c, t in zip(current, target))
if diff < 5:                                    # < 5° 就认为到位
    log("手臂已到位，可以抓取")

# 机器人摔倒检测
_, _, z_acc = sensor.get_imu_acc()
if abs(z_acc) < 5.0:
    log("⚠ 机器人可能摔倒！")
```

---

## TFReader — 坐标系查询

**封装话题**：`/tf`

⚠ 场景物体（快递、零件、料盘）**不在 TF 树中**，只能用 TF 查机器人自身部件。

| 方法 | 返回 |
|------|------|
| `lookup(from, to, timeout=1.0)` | `(pos, quat)` 或 `(None, None)` |
| `get_distance(from, to)` | 直线距离(米) 或 `None` |

| 常用坐标系 | 含义 |
|-----------|------|
| `base_link` | 机器人底盘中点 |
| `left_claw` / `right_claw` | 左/右夹爪 |
| `eef_left` / `eef_right` | 左/右手末端 |
| `zarm_l7_end_effector` | 左腕坐标系 |

### 示例

```python
tf = TFReader()

# 左夹爪在底盘坐标系的位置
pos, _ = tf.lookup("base_link", "left_claw")
if pos:
    log("左夹爪: x=%.2f y=%.2f z=%.2f", pos[0], pos[1], pos[2])

# 两爪之间距离
dist = tf.get_distance("left_claw", "right_claw")
```

---

## 完整示例一：用视觉 + 点云定位物体

```python
import cv2, numpy as np, rospy
from perception_api import CameraReader, LidarReader

cam = CameraReader()
lidar = LidarReader()
rospy.sleep(0.5)

# 方法一：点云找桌面上物体
pts = lidar.get_points_in_region(x_range=(0.3, 2.0), y_range=(-1.0, 1.0), z_range=(0.5, 1.2))
if len(pts) > 100:
    cx, cy, cz = pts.mean(axis=0)
    log("物体中心: x=%.2f y=%.2f z=%.2f 尺寸: %.2f×%.2f",
        cx, cy, cz, pts[:,0].ptp(), pts[:,1].ptp())

# 方法二：深度图找物体
depth = cam.get_head_depth()
mask = (depth > 0.3) & (depth < 2.0)
if mask.sum() > 1000:
    ys, xs = np.where(mask)
    offset_x = (xs.mean() - depth.shape[1]/2) / depth.shape[1]   # 正=偏右
    avg_dist = depth[mask].mean()
    log("深度图物体: 距离%.2fm 偏移%.2f", avg_dist, offset_x)

# 方法三：腕部深度精定位（抓取前用）
left_depth = cam.get_left_wrist_depth()
if left_depth is not None:
    h, w = left_depth.shape
    log("左腕前方: %.2fm", left_depth[h//2, w//2])
```

---

## 完整示例二：快递抓取流程

```python
import cv2, numpy as np, rospy

cam = CameraReader()
lidar = LidarReader()
sensor = SensorReader()

# 0. 手臂初始化（操作前必须）
arm.switch_to_external_control()
rospy.sleep(0.5)

# 1. 低头看桌面
head.look_down(20)
rospy.sleep(0.5)

# 2. 用点云找面前物体的中心
pts = lidar.get_points_in_region(
    (0.3, 1.5), (-0.6, 0.6), (0.5, 1.2)
)
if len(pts) < 100:
    log("没找到物体")
    return
cx, cy, cz = pts.mean(axis=0)
log("物体中心: x=%.2f y=%.2f z=%.2f", cx, cy, cz)

# 3. 走到物体前
robot.move_forward(0.05, duration=max(0, (cx - 0.5) / 0.05))
rospy.sleep(0.5)

# 4. 通过 IK 计算预抓取位姿（物体上方 10cm）
ok, q_above = arm.solve_ik(
    [cx, cy + 0.05, cz + 0.1], [0, 0, 0, 1],     # 左手
    [cx, cy - 0.05, cz + 0.1], [0, 0, 0, 1],     # 右手
)
if not ok:
    log("IK 求解失败")
    return
arm.go_to_joints(q_above)
rospy.sleep(1.5)

# 5. 下降到抓取高度
ok, q_grasp = arm.solve_ik(
    [cx, cy + 0.02, cz], [0, 0, 0, 1],
    [cx, cy - 0.02, cz], [0, 0, 0, 1],
)
if ok: arm.go_to_joints(q_grasp)
rospy.sleep(1.0)

# 6. 抓取
claw.open()
rospy.sleep(0.3)
claw.close()
claw.wait_until_done(timeout=3.0)

# 7. 抬起来
arm.go_to_joints(q_above)
rospy.sleep(1.0)

# 8. 检查是否抓住
if claw.is_grabbed():
    log("抓取成功！搬往目的地")
    robot.move_backward(0.05, duration=1.5)
else:
    log("抓取失败，张开重试")
    claw.open()
```
