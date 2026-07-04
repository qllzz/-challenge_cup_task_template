#!/usr/bin/env python3
"""
机器人控制接口封装模块。

将底层的 ROS 话题/服务包装成简单的方法调用，让你在写任务逻辑时
不用关心 Twist、JointState 等消息格式。

用法示例:
    from robot_api import RobotMover, ArmController, ClawController

    robot = RobotMover()
    arm = ArmController()
    claw = ClawController()

    robot.move_forward(0.2, duration=2.0)   # 前进 2 秒
    arm.switch_to_external_control()         # 切到外部控制模式
    claw.close()                             # 闭合夹爪
"""

import time
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from kuavo_msgs.srv import controlLejuClaw, changeArmCtrlMode
from kuavo_msgs.msg import lejuClawState, endEffectorData


# ============================================================
#  RobotMover —— 底盘行走控制
# ============================================================
class RobotMover:
    """
    控制机器人底盘移动（前后/左右/转身）。

    用法:
        robot = RobotMover()
        robot.move_forward(0.2)        # 以 0.2 m/s 前进（不停）
        robot.move_forward(0.2, 3.0)   # 以 0.2 m/s 前进 3 秒后自动停下
        robot.turn_left(0.5)           # 以 0.5 rad/s 左转
        robot.stop()                   # 立即停下
    """

    def __init__(self):
        self._pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        # 等 publisher 注册完成
        rospy.sleep(0.1)

    # ---- 基础移动 ----

    def move_forward(self, speed, duration=None):
        """
        前进。
        speed   — 速度 (m/s)，例如 0.2
        duration — 持续秒数，不传则一直走直到调用 stop()
        """
        self._publish(speed, 0.0, 0.0, 0.0, duration)

    def move_backward(self, speed, duration=None):
        """后退。speed 填正数即可，内部自动取负。"""
        self._publish(-abs(speed), 0.0, 0.0, 0.0, duration)

    def move_left(self, speed, duration=None):
        """左移。"""
        self._publish(0.0, speed, 0.0, 0.0, duration)

    def move_right(self, speed, duration=None):
        """右移。"""
        self._publish(0.0, -abs(speed), 0.0, 0.0, duration)

    def turn_left(self, angular_speed, duration=None):
        """左转。angular_speed 单位 rad/s（约 0.5 ≈ 慢转，1.0 ≈ 快转）。"""
        self._publish(0.0, 0.0, 0.0, angular_speed, duration)

    def turn_right(self, angular_speed, duration=None):
        """右转。"""
        self._publish(0.0, 0.0, 0.0, -abs(angular_speed), duration)

    def stop(self):
        """立刻停止，切回站立步态。连续发多次零指令确保控制器收到。"""
        twist = self._make_twist(0, 0, 0, 0)
        for _ in range(10):
            self._pub.publish(twist)
            rospy.sleep(0.05)
 

    # ---- 组合移动 ----

    def move(self, forward=0.0, left=0.0, turn=0.0, duration=None):
        """
        同时控制前进、横移、转身。
        例如: robot.move(forward=0.2, turn=0.3) → 边前进边左转
        """
        self._publish(forward, left, 0.0, turn, duration)

    # ---- 内部方法 ----

    @staticmethod
    def _make_twist(linear_x, linear_y, linear_z, angular_z):
        """构造 Twist 消息。"""
        t = Twist()
        t.linear.x = linear_x
        t.linear.y = linear_y
        t.linear.z = linear_z
        t.angular.z = angular_z
        return t

    def _publish(self, lx, ly, lz, az, duration):
        """发布速度指令，如果指定了 duration(>0) 则到时自动停止。"""
        twist = self._make_twist(lx, ly, lz, az)
        self._pub.publish(twist)

        if duration and duration > 0:
            start = time.time()
            rate = rospy.Rate(20)  # 20 Hz = 每 0.05 秒发一次
            while time.time() - start < duration:
                self._pub.publish(twist)
                rate.sleep()
            self.stop()


# ============================================================
#  ArmController —— 手臂控制
# ============================================================
class ArmController:
    """
    控制双臂运动。

    用法:
        arm = ArmController()
        arm.switch_to_external_control()    # 操作前必须先切换模式
        arm.go_to_joints([0,0,0,0,0,0,0, 0,0,0,0,0,0,0])  # 14 个关节角度
        arm.go_home()                       # 回到初始位置
    """

    # 14 个关节的标准名称（左臂 7 + 右臂 7）
    JOINT_NAMES = [
        "l_arm_pitch", "l_arm_roll", "l_arm_yaw", "l_forearm_pitch",
        "l_hand_yaw", "l_hand_pitch", "l_hand_roll",
        "r_arm_pitch", "r_arm_roll", "r_arm_yaw", "r_forearm_pitch",
        "r_hand_yaw", "r_hand_pitch", "r_hand_roll",
    ]

    # 常用的预设姿势（单位：度 degree）
    PRESETS = {
        "home": [0.0] * 14,                                 # 全部归零，自然下垂
        "ready": [30.0, -10.0, 0.0, -60.0, 0.0, 0.0, 0.0,   # 双手前伸准备
                  30.0, 10.0, 0.0, -60.0, 0.0, 0.0, 0.0],
    }

    def __init__(self):
        self._pub = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10)
        rospy.sleep(0.1)

    # ---- 模式切换 ----

    def switch_to_external_control(self):
        """
        将手臂切到"外部控制"模式（control_mode=2）。
        操作手臂前**必须先调用这个方法**，否则发的关节指令不会生效。
        """
        rospy.wait_for_service("/humanoid_change_arm_ctrl_mode", timeout=5.0)
        try:
            srv = rospy.ServiceProxy("/humanoid_change_arm_ctrl_mode", changeArmCtrlMode)
            resp = srv(control_mode=2)
            rospy.loginfo("手臂已切换到外部控制模式: %s", resp.message)
            return resp.result
        except rospy.ServiceException as e:
            rospy.logerr("切换手臂模式失败: %s", e)
            return False

    # ---- 关节控制 ----

    def go_to_joints(self, positions, duration=0.0):
        """
        让手臂运动到指定的 14 个关节角度。

        positions — 长度为 14 的列表，单位 度(degree)，
                    前 7 个是左臂，后 7 个是右臂。
        duration  — 保留参数，当前直接发送目标位置。
        """
        if len(positions) != 14:
            rospy.logerr("go_to_joints 需要 14 个关节值，收到 %d 个", len(positions))
            return

        msg = JointState()
        msg.name = self.JOINT_NAMES
        msg.position = list(positions)
        self._pub.publish(msg)

    def go_home(self):
        """手臂回到初始位置（所有关节归零）。"""
        self.go_to_joints(self.PRESETS["home"])
        rospy.loginfo("手臂已回到初始位置")

    def go_ready(self):
        """手臂到准备抓取姿势。"""
        self.go_to_joints(self.PRESETS["ready"])
        rospy.loginfo("手臂已到准备姿势")

    # ---- 单臂快捷方法 ----

    def left_arm_to(self, joints):
        """
        只控制左臂（7 个关节），右臂保持不动。
        joints — 左臂 7 个关节角度，单位 度。
        """
        if len(joints) != 7:
            rospy.logerr("left_arm_to 需要 7 个关节值")
            return
        full = [0.0] * 14
        full[0:7] = joints
        self.go_to_joints(full)

    def right_arm_to(self, joints):
        """
        只控制右臂（7 个关节），左臂保持不动。
        joints — 右臂 7 个关节角度，单位 度。
        """
        if len(joints) != 7:
            rospy.logerr("right_arm_to 需要 7 个关节值")
            return
        full = [0.0] * 14
        full[7:14] = joints
        self.go_to_joints(full)


# ============================================================
#  ClawController —— 二指夹爪控制
# ============================================================
class ClawController:
    """
    控制二指夹爪（左右两个）。

    用法:
        claw = ClawController()
        claw.open()               # 双手同时张开
        claw.close()              # 双手同时闭合
        claw.close([80, 80])      # 闭合到 80%
        claw.left_open()          # 只张开左手

        if claw.is_grabbed():     # 检查是否抓住东西
            print("抓住了!")
    """

    def __init__(self):
        # 服务客户端
        rospy.wait_for_service("/control_robot_leju_claw", timeout=5.0)
        self._srv = rospy.ServiceProxy("/control_robot_leju_claw", controlLejuClaw)

        # 状态缓存（订阅 /leju_claw_state 更新）
        self._left_state = 0
        self._right_state = 0
        rospy.Subscriber("/leju_claw_state", lejuClawState, self._state_callback)
        rospy.sleep(0.1)

    # ---- 基础操作 ----

    def open(self, position=None):
        """双手张开。position 默认 [10, 10]（张开 10%）。"""
        pos = position if position is not None else [10, 10]
        return self._command(pos)

    def close(self, position=None):
        """双手闭合。position 默认 [90, 90]（闭合 90%）。"""
        pos = position if position is not None else [90, 90]
        return self._command(pos)

    def set_position(self, left_percent, right_percent):
        """分别设置左右夹爪开合百分比。0=全开, 100=全闭。"""
        return self._command([left_percent, right_percent])

    # ---- 单手快捷方法 ----

    def left_open(self):
        """只张开左夹爪。"""
        return self._command([10, None], single_side="left")

    def left_close(self):
        """只闭合左夹爪。"""
        return self._command([90, None], single_side="left")

    def right_open(self):
        """只张开右夹爪。"""
        return self._command([None, 10], single_side="right")

    def right_close(self):
        """只闭合右夹爪。"""
        return self._command([None, 90], single_side="right")

    # ---- 状态查询 ----

    def is_grabbed(self):
        """
        判断是否抓住了物体。
        只要任一侧夹爪状态为 3 (Grabbed) 就返回 True。
        """
        return self._left_state == 3 or self._right_state == 3

    def is_moving(self):
        """判断夹爪是否正在运动中（任一侧状态为 1）。"""
        return self._left_state == 1 or self._right_state == 1

    def wait_until_done(self, timeout=5.0):
        """
        阻塞等待直到夹爪运动完成（到达目标或抓住物体）。
        超时返回 False。
        """
        start = time.time()
        rate = rospy.Rate(20)
        while time.time() - start < timeout:
            if not self.is_moving():
                return True
            rate.sleep()
        rospy.logwarn("夹爪等待超时 (%.1f 秒)", timeout)
        return False

    # ---- 内部方法 ----

    def _command(self, position, single_side=None):
        """
        发送夹爪指令。
        position    — [left, right]，None 表示不控制该侧
        single_side — "left"/"right"/None
        """
        names = ["left_claw", "right_claw"]
        pos = [0, 0]
        vel = [50, 50]
        eff = [1.0, 1.0]

        # 处理单手控制
        if single_side == "left":
            pos[0] = position[0] if position[0] is not None else 50
            names = ["left_claw"]
            pos = [pos[0]]
            vel = [50]
            eff = [1.0]
        elif single_side == "right":
            pos[0] = position[1] if position[1] is not None else 50
            names = ["right_claw"]
            pos = [pos[0]]
            vel = [50]
            eff = [1.0]
        else:
            pos = [p if p is not None else 50 for p in position]

        try:
            data = endEffectorData()
            data.name = names
            data.position = pos
            data.velocity = vel
            data.effort = eff
            resp = self._srv(data=data)
            return resp.success
        except rospy.ServiceException as e:
            rospy.logerr("夹爪指令失败: %s", e)
            return False

    def _state_callback(self, msg):
        """接收 /leju_claw_state 的状态更新。"""
        if len(msg.state) >= 2:
            self._left_state = msg.state[0]
            self._right_state = msg.state[1]
