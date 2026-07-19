#!/usr/bin/env python3
"""
机器人控制接口封装模块。

将底层的 ROS 话题/服务包装成简单的方法调用，让你在写任务逻辑时
不用关心 Twist、JointState 等消息格式。

用法示例:
    from robot_api import (RobotMover, ArmController, ClawController,
                           HeadController, WaistController)

    robot = RobotMover()
    arm = ArmController()
    claw = ClawController()
    head = HeadController()
    waist = WaistController()

    robot.move_forward(0.2, duration=2.0)   # 前进 2 秒
    arm.switch_to_external_control()         # 切到外部控制模式
    q_arm = arm.solve_ik([0.5,0,0.8], [1,0,0,0], ...)  # IK 求解
    claw.close()                             # 闭合夹爪
    head.look_at(0, -10)                    # 平视、低头 10°
"""

import math
import time
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String
from sensor_msgs.msg import JointState
from kuavo_msgs.srv import controlLejuClaw, changeArmCtrlMode, twoArmHandPoseCmdSrv, fkSrv
from kuavo_msgs.msg import (lejuClawState, endEffectorData,
                            twoArmHandPoseCmd, twoArmHandPose,
                            armHandPose, ikSolveParam,
                            robotHeadMotionData, robotWaistControl, sensorsData)


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
        self._pose_pub = rospy.Publisher("/cmd_pose", Twist, queue_size=10)
        self._gait_pub = rospy.Publisher("/humanoid_mpc_gait_change", String, queue_size=10)
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
        self._publish(0.0, -speed, 0.0, 0.0, duration)

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

    # ---- 步态切换 ----

    def switch_to_walk(self):
        """手动切到行走步态。通常 /cmd_vel 非零时自动切换，无需手动调用。"""
        self._gait_pub.publish(String(data="walk"))

    def switch_to_stance(self):
        """手动切到站步步态。通常 /cmd_vel 全零时自动切换，无需手动调用。"""
        self._gait_pub.publish(String(data="stance"))
 

    # ---- 组合移动 ----

    def move(self, forward=0.0, left=0.0, turn=0.0, duration=None):
        """
        同时控制前进、横移、转身。
        例如: robot.move(forward=0.2, turn=0.3) → 边前进边左转
        """
        self._publish(forward, left, 0.0, turn, duration)

    # ---- 躯干高度控制 ----

    def squat(self, height_delta, repeat=10, interval=0.05):
        """
        调整躯干高度。height_delta 是相对标称高度 /com_height 的增量，单位 m。

        例如: robot.squat(-0.10) → 下蹲 10cm。
        不做高度限位，调用者需要自己保证输入安全。
        """
        self._publish_pose_height(height_delta, repeat, interval)

    def stand_height(self, height_delta=0.0, repeat=10, interval=0.05):
        """
        设置躯干高度增量。默认 0.0 表示恢复到标称高度 /com_height。

        例如: robot.stand_height(0.0) → 恢复标称高度。
        不做高度限位，调用者需要自己保证输入安全。
        """
        self._publish_pose_height(height_delta, repeat, interval)

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

    def _publish_pose_height(self, height_delta, repeat, interval):
        """发布 /cmd_pose 高度增量指令。"""
        twist = self._make_twist(0.0, 0.0, height_delta, 0.0)
        for _ in range(repeat):
            self._pose_pub.publish(twist)
            rospy.sleep(interval)


# ============================================================
#  WaistController —— 腰部 yaw 控制
# ============================================================
class WaistController:
    """控制单自由度腰部 yaw。

    腰部控制器默认由 RL 接管。调用 :meth:`enable_external_control` 后，
    再通过 :meth:`set_yaw` 发布目标角度；控制器端会将指令限幅到 ±30°。

    用法::

        waist = WaistController()
        waist.enable_external_control()
        waist.set_yaw(12.0)       # 单位：度，正负遵循腰关节自身约定
        ...
        waist.reset_and_release() # 回零并交回 RL
    """

    MIN_YAW_DEG = -30.0
    MAX_YAW_DEG = 30.0

    def __init__(self):
        self._enable_pub = rospy.Publisher(
            "/humanoid_controller/enable_waist_control", Bool, queue_size=1)
        self._motion_pub = rospy.Publisher(
            "/robot_waist_motion_data", robotWaistControl, queue_size=1)
        self._enabled = False
        self._last_yaw_deg = 0.0
        rospy.sleep(0.1)

    def enable_external_control(self):
        """切换到外部腰部控制模式（mode 2）。"""
        self._enable_pub.publish(Bool(data=True))
        self._enabled = True

    def disable_external_control(self):
        """交回 RL 腰部控制模式（mode 1）。"""
        self._enable_pub.publish(Bool(data=False))
        self._enabled = False

    def set_yaw(self, yaw_deg):
        """发布腰部 yaw 目标，单位为度；返回实际发布的限幅后角度。"""
        yaw_deg = max(self.MIN_YAW_DEG, min(self.MAX_YAW_DEG, float(yaw_deg)))
        msg = robotWaistControl()
        msg.header.stamp = rospy.Time.now()
        msg.data.data = [yaw_deg]
        self._motion_pub.publish(msg)
        self._last_yaw_deg = yaw_deg
        return yaw_deg

    def reset_and_release(self):
        """将腰部回零，然后交回 RL 控制。"""
        self.set_yaw(0.0)
        self.disable_external_control()


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
        self._last_joints_deg = [0.0] * 14  # 追踪最后发出的关节角
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
            return False

        msg = JointState()
        msg.name = self.JOINT_NAMES
        msg.position = list(positions)
        self._last_joints_deg = list(positions)
        self._pub.publish(msg)
        return True

    def go_home(self):
        """手臂回到初始位置（所有关节归零）。"""
        self.go_to_joints(self.PRESETS["home"])
        rospy.loginfo("手臂已回到初始位置")

    def go_ready(self):
        """手臂到准备抓取姿势。"""
        self.go_to_joints(self.PRESETS["ready"])
        rospy.loginfo("手臂已到准备姿势")

    # ---- 传感器关节读取 & FK ----

    def _read_arm_joints_rad(self, timeout=2.0):
        """从 /sensors_data_raw 读取 14 个手臂关节角（弧度）。"""
        try:
            msg = rospy.wait_for_message(
                "/sensors_data_raw", sensorsData, timeout=timeout)
        except Exception as exc:
            rospy.logwarn("_read_arm_joints_rad: %s", exc)
            return None
        joint_q = list(msg.joint_data.joint_q)
        if len(joint_q) >= 27:
            return joint_q[13:27]
        if len(joint_q) >= 26:
            return joint_q[12:26]
        rospy.logerr("_read_arm_joints_rad: joint_q 长度 %d 不足", len(joint_q))
        return None

    def call_fk(self, joint_angles_rad, timeout=5.0):
        """FK 服务调用，输入 14 个关节角（弧度），返回 hand_poses。"""
        rospy.wait_for_service("/ik/fk_srv", timeout=timeout)
        resp = rospy.ServiceProxy("/ik/fk_srv", fkSrv)(list(joint_angles_rad))
        if not resp.success:
            rospy.logerr("call_fk: FK 服务返回 success=false")
            return None
        return resp.hand_poses

    def get_endpoint_pose(self, hand, timeout=5.0):
        """返回当前末端在 base_link 下的 (pos_xyz, quat_xyzw)，失败返回 None。"""
        if hand not in ("left", "right"):
            raise ValueError("get_endpoint_pose: hand 必须为 left 或 right")
        q0 = self._read_arm_joints_rad()
        if q0 is None or len(q0) != 14:
            return None
        fk = self.call_fk(q0, timeout=timeout)
        if fk is None:
            return None
        pose = fk.left_pose if hand == "left" else fk.right_pose
        return list(pose.pos_xyz), list(pose.quat_xyzw)

    def get_endpoint_quat(self, hand, timeout=5.0):
        """返回当前末端在 base_link 下的姿态四元数 [x,y,z,w]，失败返回 None。"""
        pose = self.get_endpoint_pose(hand, timeout=timeout)
        return None if pose is None else pose[1]

    @staticmethod
    def _ik_param(constraint_mode=None, pos_cost_weight=0.0,
                  major_iterations_limit=500):
        """构造自定义 IK 参数。"""
        param = ikSolveParam()
        param.major_optimality_tol = 1e-3
        param.major_feasibility_tol = 1e-3
        param.minor_feasibility_tol = 1e-3
        param.major_iterations_limit = int(major_iterations_limit or 500)
        param.oritation_constraint_tol = 1e-3
        param.pos_constraint_tol = 1e-3
        param.pos_cost_weight = float(pos_cost_weight or 0.0)
        if constraint_mode is not None:
            param.constraint_mode = int(constraint_mode)
        return param

    # ---- 带 constraint_mode 的单臂 IK ----

    def solve_ik_one_hand(self, side, pos_xyz, quat_xyzw, frame=2,
                          constraint_mode=None, pos_cost_weight=0.0,
                          major_iterations_limit=500, timeout=5.0):
        """
        单臂 IK：只解指定手，另一只手用传感器实际关节角锁定。

        参数:
            side      — "left" 或 "right"
            pos_xyz   — 目标位置 [x, y, z]，单位 米
            quat_xyzw — 目标姿态 [x, y, z, w] 四元数
            frame     — 目标位姿坐标系；2=local/base_link（默认）
            constraint_mode — IK 约束模式 (0x03=位置、姿态均硬约束)；
                              精密移动时不得使用姿态软约束 0x02
            pos_cost_weight / major_iterations_limit — IK 精度参数

        返回:
            (success, q_arm_deg) — 14 个关节角度(度)
        """
        if side not in ("left", "right"):
            raise ValueError("solve_ik_one_hand: side 必须为 left 或 right")

        q0 = self._read_arm_joints_rad(timeout=timeout)
        if q0 is None or len(q0) != 14:
            rospy.logerr("solve_ik_one_hand: 无法从传感器读取当前关节角")
            return False, []

        fk = self.call_fk(q0, timeout=timeout)
        if fk is None:
            rospy.logerr("solve_ik_one_hand: FK 调用失败")
            return False, []

        req = twoArmHandPoseCmd()
        # FK 位姿及 delta_xyz 都是 move_relative 的 local/base_link 语义，
        # 必须显式设为 frame=2；不能依赖服务端默认 frame。
        req.frame = frame
        req.use_custom_ik_param = constraint_mode is not None
        req.joint_angles_as_q0 = True
        if req.use_custom_ik_param:
            req.ik_param = self._ik_param(
                constraint_mode, pos_cost_weight, major_iterations_limit)

        req.hand_poses.left_pose.joint_angles = list(q0[:7])
        req.hand_poses.right_pose.joint_angles = list(q0[7:])
        req.hand_poses.left_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
        req.hand_poses.right_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
        req.hand_poses.left_pose.pos_xyz = list(fk.left_pose.pos_xyz)
        req.hand_poses.left_pose.quat_xyzw = list(fk.left_pose.quat_xyzw)
        req.hand_poses.right_pose.pos_xyz = list(fk.right_pose.pos_xyz)
        req.hand_poses.right_pose.quat_xyzw = list(fk.right_pose.quat_xyzw)

        target = (req.hand_poses.left_pose if side == "left"
                  else req.hand_poses.right_pose)
        if pos_xyz is not None:
            target.pos_xyz = list(pos_xyz)
        if quat_xyzw is not None:
            target.quat_xyzw = list(quat_xyzw)

        rospy.wait_for_service("/ik/two_arm_hand_pose_cmd_srv", timeout=timeout)
        try:
            resp = rospy.ServiceProxy(
                "/ik/two_arm_hand_pose_cmd_srv", twoArmHandPoseCmdSrv)(req)
        except rospy.ServiceException as e:
            rospy.logerr("solve_ik_one_hand: IK 服务调用失败: %s", e)
            return False, []

        if not resp.success:
            rospy.logerr("solve_ik_one_hand: IK 求解失败: %s",
                         getattr(resp, "error_reason", ""))
            return False, []

        q_rad = list(q0[:14])
        if side == "left":
            jr = list(resp.hand_poses.left_pose.joint_angles)
            if len(jr) != 7 and len(resp.q_arm) >= 7:
                jr = list(resp.q_arm[:7])
            if len(jr) == 7:
                q_rad = jr + q_rad[7:]
        else:
            jr = list(resp.hand_poses.right_pose.joint_angles)
            if len(jr) != 7 and len(resp.q_arm) >= 14:
                jr = list(resp.q_arm[7:14])
            if len(jr) == 7:
                q_rad = q_rad[:7] + jr
        return True, [math.degrees(q) for q in q_rad]

    # ---- 单臂快捷方法 ----

    def left_arm_to(self, joints):
        """
        只控制左臂（7 个关节），右臂保持不动。
        joints — 左臂 7 个关节角度，单位 度。
        """
        if len(joints) != 7:
            rospy.logerr("left_arm_to 需要 7 个关节值")
            return False
        # 保留右臂最后一次发送的目标，避免单臂指令把右臂重置为零位。
        full = list(self._last_joints_deg)
        full[0:7] = joints
        return self.go_to_joints(full)

    def right_arm_to(self, joints):
        """
        只控制右臂（7 个关节），左臂保持不动。
        joints — 右臂 7 个关节角度，单位 度。
        """
        if len(joints) != 7:
            rospy.logerr("right_arm_to 需要 7 个关节值")
            return False
        # 保留左臂最后一次发送的目标，避免单臂指令把左臂重置为零位。
        full = list(self._last_joints_deg)
        full[7:14] = joints
        return self.go_to_joints(full)

    def apply_ik_single_arm_solution(self, hand, q_arm_deg):
        """
        执行单臂 IK 结果，仅把目标侧的 7 个关节写入控制目标。

        IK 服务仍需要双手位姿作为约束；本方法会丢弃另一侧的求解结果，
        并保持另一侧最后一次发送的关节目标不变。
        """
        if hand not in ("left", "right"):
            rospy.logerr("apply_ik_single_arm_solution: hand 必须为 left 或 right")
            return False
        if len(q_arm_deg) != 14:
            rospy.logerr("apply_ik_single_arm_solution 需要 14 个 IK 关节值，收到 %d 个", len(q_arm_deg))
            return False
        if hand == "left":
            return self.left_arm_to(q_arm_deg[:7])
        return self.right_arm_to(q_arm_deg[7:14])

    # ---- IK 求解 ----

    def solve_ik(self, left_pose_xyz, left_quat_xyzw,
                 right_pose_xyz, right_quat_xyzw,
                 frame=2, use_current_as_q0=True,
                 use_custom_ik_param=True, pos_cost_weight=0.0,
                 use_multiple_references=False):
        """
        调用 IK 服务，输入双手末端位姿，返回 14 个关节角度。

        参数:
            left_pose_xyz   — 左手末端位置 [x, y, z]，单位 米
            left_quat_xyzw  — 左手末端姿态 [x, y, z, w] 四元数
            right_pose_xyz  — 右手末端位置 [x, y, z]，单位 米
            right_quat_xyzw — 右手末端姿态 [x, y, z, w] 四元数
            frame           — 坐标系: 0=当前 1=odom 2=局部(默认) 3=VR 4=操作世界 5=关节空间
            use_current_as_q0 — 用当前关节角作为初值（通常 True）
            use_custom_ik_param — 使用高精度 IK 参数
            pos_cost_weight — 位置成本权重，0.0=最高精度
            use_multiple_references — 使用多参考点 IK 服务；会额外尝试关节
                限位中点、伪逆和解析种子，适合从预备姿态抓取目标

        返回:
            (success, q_arm) — success 为 True 时 q_arm 是 14 个关节角度(度)
        """
        service_name = ("/ik/two_arm_hand_pose_cmd_srv_muli_refer"
                        if use_multiple_references
                        else "/ik/two_arm_hand_pose_cmd_srv")
        try:
            rospy.wait_for_service(service_name, timeout=5.0)
        except rospy.ROSException:
            if not use_multiple_references:
                rospy.logerr("IK 服务不可用: %s", service_name)
                return False, []
            rospy.logwarn("多参考点 IK 服务不可用，回退到普通 IK 服务")
            service_name = "/ik/two_arm_hand_pose_cmd_srv"
            try:
                rospy.wait_for_service(service_name, timeout=5.0)
            except rospy.ROSException:
                rospy.logerr("IK 服务不可用: %s", service_name)
                return False, []
        try:
            srv = rospy.ServiceProxy(service_name, twoArmHandPoseCmdSrv)

            left_hp = armHandPose()
            left_hp.pos_xyz = list(left_pose_xyz)
            left_hp.quat_xyzw = list(left_quat_xyzw)

            right_hp = armHandPose()
            right_hp.pos_xyz = list(right_pose_xyz)
            right_hp.quat_xyzw = list(right_quat_xyzw)

            if use_current_as_q0:
                import math
                left_hp.joint_angles = [math.radians(q) for q in self._last_joints_deg[:7]]
                right_hp.joint_angles = [math.radians(q) for q in self._last_joints_deg[7:14]]

            hand_poses = twoArmHandPose()
            hand_poses.left_pose = left_hp
            hand_poses.right_pose = right_hp

            req = twoArmHandPoseCmd()
            req.hand_poses = hand_poses
            req.frame = frame
            req.joint_angles_as_q0 = use_current_as_q0
            req.use_custom_ik_param = use_custom_ik_param
            if use_custom_ik_param:
                req.ik_param.major_optimality_tol = 1e-3
                req.ik_param.major_feasibility_tol = 1e-3
                req.ik_param.minor_feasibility_tol = 1e-3
                req.ik_param.major_iterations_limit = 100
                req.ik_param.oritation_constraint_tol = 1e-3
                req.ik_param.pos_constraint_tol = 1e-3
                req.ik_param.pos_cost_weight = pos_cost_weight

            resp = srv(req)

            if resp.success:
                import math
                q_deg = [math.degrees(q) for q in resp.q_arm]
                rospy.loginfo("IK 求解成功，耗时 %.1f ms", resp.time_cost)
                return True, q_deg
            else:
                rospy.logerr("IK 求解失败")
                return False, []

        except rospy.ServiceException as e:
            rospy.logerr("IK 服务调用失败: %s", e)
            return False, []

    def move_relative(self, hand, delta_xyz, max_error_m=0.05, sleep=1.5):
        """
        在当前末端位姿基础上叠加一个相对位移，保持姿态不变。

        坐标系: base_link (frame=2), 原点 = 机器人基座中心

            X+ : 机器人正前方  (前进方向)
            Y+ : 机器人左侧    (左手方向)
            Z+ : 机器人上方    (竖直向上)

        参数:
            hand        — "left" 或 "right"
            delta_xyz   — 相对位移 [dx, dy, dz]，单位 米，base_link 坐标系
            max_error_m — IK 残差上限 (m)，超过则拒绝执行
            sleep       — 到位后等待秒数

        返回:
            True / False

        示例:
            arm.move_relative("right", [ 0.05, 0.0,  0.0])  # 右手前伸 5cm
            arm.move_relative("right", [-0.03, 0.0,  0.0])  # 右手后收 3cm
            arm.move_relative("left",  [ 0.0,  0.05, 0.0])  # 左手左移 5cm
            arm.move_relative("right", [ 0.0,  0.0,  0.02]) # 右手上抬 2cm
        """
        if hand not in ("left", "right"):
            rospy.logerr("move_relative: hand 必须为 left 或 right")
            return False

        # 1. FK 获取当前末端位姿
        lp, lq, rp, rq = self.fk()
        if lp is None:
            rospy.logerr("move_relative: FK 失败")
            return False

        if hand == "left":
            current_pos = lp
            current_quat = lq
        else:
            current_pos = rp
            current_quat = rq

        # 2. 叠加位移
        target_pos = [
            current_pos[i] + delta_xyz[i]
            for i in range(3)
        ]

        # 3. IK 求解并执行
        ok, q_arm = self.solve_ik_single_arm(
            hand, target_pos, current_quat,
            frame=2, max_error_m=max_error_m)
        if not ok:
            rospy.logerr("move_relative: IK 求解失败")
            return False

        self.apply_ik_single_arm_solution(hand, q_arm)
        if sleep > 0:
            rospy.sleep(sleep)
        return True

    def move_relative_cartesian(self, hand, delta_xyz,
                                 step_m=0.005, settle_s=0.35,
                                 max_error_m=0.005,
                                 max_orientation_error_rad=math.radians(3.0),
                                 constraint_modes=None,
                                 quat_xyzw=None, target_z_m=None):
        """
        笛卡尔直线相对位移：FK 当前末端 → 在起终点间插值 N 个 waypoint →
        逐点 IK → 依次下发，避免关节空间插补产生的横向漂移。

        坐标系: base_link (frame=2), 原点 = 机器人基座中心

            X+ : 机器人正前方  (前进方向)
            Y+ : 机器人左侧    (左手方向)
            Z+ : 机器人上方    (竖直向上)

        参数:
            hand        — "left" 或 "right"
            delta_xyz   — 相对位移 [dx, dy, dz]，单位 米，base_link 系
            step_m      — 每段最大步长（默认 5 mm）
            settle_s    — 每段到位后等待秒数
            max_error_m — 单步 IK 位置残差上限（默认 5 mm）
            max_orientation_error_rad — 单步 FK 姿态误差上限（默认 3°）
            constraint_modes — IK 模式列表，默认仅 [0x03]（位置、姿态硬约束）。
                              为防夹爪倾斜，默认不回退到姿态软约束 0x02。
            quat_xyzw   — 可选，手动指定末端目标姿态 [x,y,z,w]（base_link 系）。
                          传入则锁定此姿态不动；不传则从传感器 FK 读取当前姿态。
            target_z_m  — 可选，base_link 下的末端绝对目标高度（米）。传入后，
                          无论当前传感器 FK 高度或 delta_xyz 的 Z 分量为何，终点
                          都强制为该高度；用于横向/前向移动期间防止整臂下垂。

        返回:
            True / False

        示例:
            arm.move_relative_cartesian("right", [0.10, 0.0, 0.0])   # 前伸 10cm
            arm.move_relative_cartesian("right", [-0.05, 0.0, 0.0])  # 后收 5cm
            arm.move_relative_cartesian("right", [0.0, 0.0, 0.03])   # 上抬 3cm
        """
        if hand not in ("left", "right"):
            rospy.logerr("move_relative_cartesian: hand 必须为 left 或 right")
            return False

        if constraint_modes is None:
            # 安全优先：姿态硬约束不能满足就中止，不能以夹爪下垂换取位置可达。
            constraint_modes = [0x03]

        dt = settle_s

        # 从传感器读取实际关节角并 FK 得到当前末端位置
        q0 = self._read_arm_joints_rad()
        if q0 is None or len(q0) != 14:
            rospy.logerr("move_relative_cartesian: 无法读取传感器关节角")
            return False

        fk = self.call_fk(q0, timeout=5.0)
        if fk is None:
            rospy.logerr("move_relative_cartesian: FK 失败")
            return False

        if hand == "left":
            start = list(fk.left_pose.pos_xyz)
            quat = list(fk.left_pose.quat_xyzw) if quat_xyzw is None else list(quat_xyzw)
        else:
            start = list(fk.right_pose.pos_xyz)
            quat = list(fk.right_pose.quat_xyzw) if quat_xyzw is None else list(quat_xyzw)

        end = [start[j] + delta_xyz[j] for j in range(3)]
        if target_z_m is None:
            total_dist = math.sqrt(sum((end[j] - start[j]) ** 2 for j in range(3)))
            n = max(1, int(math.ceil(total_dist / step_m)))
            waypoints = [
                [start[j] + (end[j] - start[j]) * float(i) / float(n)
                 for j in range(3)]
                for i in range(1, n + 1)
            ]
            path_desc = ""
        else:
            # 高度锁定轨迹不能把下垂恢复与 XY 移动混成斜线：先在原 XY
            # 原地回到参考高度，再在固定 Z 平面内完成横向/前向运动。
            end[2] = float(target_z_m)
            vertical_dist = abs(end[2] - start[2])
            planar_dist = math.sqrt(
                (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            vertical_n = int(math.ceil(vertical_dist / step_m))
            planar_n = int(math.ceil(planar_dist / step_m))
            waypoints = []
            for i in range(1, vertical_n + 1):
                a = float(i) / float(vertical_n)
                waypoints.append([start[0], start[1],
                                  start[2] + (end[2] - start[2]) * a])
            for i in range(1, planar_n + 1):
                a = float(i) / float(planar_n)
                waypoints.append([
                    start[0] + (end[0] - start[0]) * a,
                    start[1] + (end[1] - start[1]) * a,
                    end[2],
                ])
            if not waypoints:
                waypoints = [end]
            n = len(waypoints)
            path_desc = " (Z locked: vertical %d + planar %d)" % (
                vertical_n, planar_n)

        rospy.loginfo("move_relative_cartesian: %s %d段 %.1fmm/段 "
                      "start=%s end=%s%s",
                      hand, n, step_m * 1000.0, start, end, path_desc)

        for i, pt in enumerate(waypoints, 1):

            ok = False
            joints = []
            used_mode = None
            for mode in constraint_modes:
                try:
                    ok, joints = self.solve_ik_one_hand(
                        hand, pt, quat, frame=2,
                        constraint_mode=mode,
                        pos_cost_weight=0.0,
                        major_iterations_limit=500)
                except Exception as exc:
                    rospy.logwarn(
                        "move_relative_cartesian: IK mode=%s 异常: %s",
                        mode, exc)
                    continue
                used_mode = mode
                if ok:
                    break

            if not ok:
                rospy.logerr(
                    "move_relative_cartesian: 第 %d/%d 步 IK 失败 (mode=%s)",
                    i, n, used_mode)
                return False

            # FK 验证位置误差
            solved_fk = self.call_fk(
                [math.radians(q) for q in joints], timeout=5.0)
            if solved_fk is None:
                rospy.logerr(
                    "move_relative_cartesian: 第 %d/%d 步 FK 验证失败", i, n)
                return False

            solved_pos = (list(solved_fk.left_pose.pos_xyz) if hand == "left"
                          else list(solved_fk.right_pose.pos_xyz))
            error_m = math.sqrt(
                sum((solved_pos[j] - pt[j]) ** 2 for j in range(3)))
            if error_m > max_error_m:
                rospy.logerr(
                    "move_relative_cartesian: 第 %d/%d 步 IK 残差 "
                    "%.1f mm > %.0f mm，拒绝执行",
                    i, n, error_m * 1000.0, max_error_m * 1000.0)
                return False

            solved_quat = (list(solved_fk.left_pose.quat_xyzw)
                           if hand == "left"
                           else list(solved_fk.right_pose.quat_xyzw))
            target_norm = math.sqrt(sum(value * value for value in quat))
            solved_norm = math.sqrt(sum(value * value for value in solved_quat))
            if target_norm <= 1e-9 or solved_norm <= 1e-9:
                rospy.logerr("move_relative_cartesian: 第 %d/%d 步四元数无效", i, n)
                return False
            # q 与 -q 表示同一旋转，所以取 dot 的绝对值。
            dot = abs(sum(a * b for a, b in zip(quat, solved_quat)) /
                      (target_norm * solved_norm))
            orientation_error = 2.0 * math.acos(min(1.0, max(-1.0, dot)))
            if orientation_error > max_orientation_error_rad:
                rospy.logerr(
                    "move_relative_cartesian: 第 %d/%d 步末端姿态偏差 %.1f° > %.1f°，"
                    "拒绝执行以防夹爪倾斜",
                    i, n, math.degrees(orientation_error),
                    math.degrees(max_orientation_error_rad))
                return False

            self.apply_ik_single_arm_solution(hand, joints)
            rospy.sleep(dt)

        return True

    def fk(self, joints_deg=None):
        """
        正运动学：给定 14 个关节角，返回双手末端位姿。

        参数:
            joints_deg — 14 个关节角度(度)，默认用最后一次 go_to_joints 的值

        返回:
            (left_pos, left_quat, right_pos, right_quat), 单位 m
            失败返回 (None, None, None, None)
        """
        if joints_deg is None:
            joints_deg = self._last_joints_deg
        import math
        joints_rad = [math.radians(q) for q in joints_deg]

        rospy.wait_for_service("/ik/fk_srv", timeout=5.0)
        try:
            from kuavo_msgs.srv import fkSrv
            fk_srv = rospy.ServiceProxy("/ik/fk_srv", fkSrv)
            resp = fk_srv(joints_rad)
            if resp.success:
                lp = resp.hand_poses.left_pose.pos_xyz
                lq = resp.hand_poses.left_pose.quat_xyzw
                rp = resp.hand_poses.right_pose.pos_xyz
                rq = resp.hand_poses.right_pose.quat_xyzw
                return list(lp), list(lq), list(rp), list(rq)
        except Exception as e:
            rospy.logerr("FK 调用失败: %s", e)
        return None, None, None, None

    def solve_ik_single_arm(self, hand, pose_xyz, quat_xyzw, frame=2,
                            max_error_m=0.03):
        """
        单臂 IK：只解指定手，另一只手用 FK 锁定当前位置不动。

        成功后请用 `apply_ik_single_arm_solution()` 执行结果；不要直接将
        返回的 14 关节结果传给 `go_to_joints()`，以免另一只手被 IK 解改变。

        参数:
            hand      — "left" 或 "right"
            pose_xyz  — 目标位置 [x, y, z]，单位 米
            quat_xyzw — 目标姿态 [x, y, z, w] 四元数
            frame     — 坐标系 (默认 2=局部/base_link)

        返回:
            (success, q_arm) — 14 个关节角度(度)
        """
        if hand not in ("left", "right"):
            rospy.logerr("solve_ik_single_arm: hand 必须为 left 或 right")
            return False, []

        # FK 获取两只手的当前位置，并将非目标手作为硬约束锁定。
        lp, lq, rp, rq = self.fk()
        if lp is None:
            rospy.logerr("solve_ik_single_arm: FK 失败，无法安全锁定非目标手")
            return False, []

        if hand == "left":
            # 右手锁定当前位姿。
            ok, q_arm = self.solve_ik(
                pose_xyz, quat_xyzw, rp, rq, frame=frame,
                use_current_as_q0=True, pos_cost_weight=0.0,
                use_multiple_references=True)
        else:
            # 左手锁定当前位姿。
            ok, q_arm = self.solve_ik(
                lp, lq, pose_xyz, quat_xyzw, frame=frame,
                use_current_as_q0=True, pos_cost_weight=0.0,
                use_multiple_references=True)

        if not ok:
            return False, []

        # IK 服务在目标超出工作空间时仍可能返回 success=True 和一组饱和
        # 关节角。必须用 FK 验证目标侧的位置误差和姿态误差，避免执行明显偏离的解。
        solved_lp, solved_lq, solved_rp, solved_rq = self.fk(q_arm)
        solved_pos = solved_lp if hand == "left" else solved_rp
        solved_quat = solved_lq if hand == "left" else solved_rq
        if solved_pos is None or solved_quat is None:
            rospy.logerr("solve_ik_single_arm: 无法验证 IK 解的 FK 结果")
            return False, []

        target_xyz = [float(value) for value in pose_xyz]
        solved_xyz = [float(value) for value in solved_pos]
        error_xyz = [actual - target for actual, target in zip(solved_xyz, target_xyz)]
        error_m = math.sqrt(sum(component ** 2 for component in error_xyz))
        rospy.loginfo(
            "[DEBUG] solve_ik_single_arm: hand=%s frame=%s target_m=%s solved_m=%s "
            "error_m=%s norm_mm=%.1f q0_deg=%s q_solution_deg=%s",
            hand, frame, target_xyz, solved_xyz, error_xyz, error_m * 1000.0,
            self._last_joints_deg, q_arm)
        if error_m > max_error_m:
            rospy.logerr("solve_ik_single_arm: IK 残差 %.1f mm > %.0f mm，拒绝执行",
                         error_m * 1000.0, max_error_m * 1000.0)
            return False, []

        # 检查姿态偏差（与 move_relative_cartesian 保持一致）
        target_quat = [float(v) for v in quat_xyzw]
        target_norm = math.sqrt(sum(v * v for v in target_quat))
        solved_norm = math.sqrt(sum(v * v for v in solved_quat))
        if target_norm > 1e-9 and solved_norm > 1e-9:
            dot = abs(sum(a * b for a, b in zip(target_quat, solved_quat))
                      / (target_norm * solved_norm))
            orientation_error_rad = 2.0 * math.acos(min(1.0, max(-1.0, dot)))
            max_orientation_error_rad = math.radians(3.0)
            if orientation_error_rad > max_orientation_error_rad:
                rospy.logerr(
                    "solve_ik_single_arm: FK 姿态偏差 %.1f° > %.1f°，拒绝执行",
                    math.degrees(orientation_error_rad),
                    math.degrees(max_orientation_error_rad))
                return False, []

        return True, q_arm


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

    def is_grabbed(self, hand=None):
        """
        判断夹爪是否抓住了物体。

        hand 为 "left" 或 "right" 时仅检查指定侧；默认只要任一侧状态为
        3 (Grabbed) 就返回 True。
        """
        if hand == "left":
            return self._left_state == 3
        if hand == "right":
            return self._right_state == 3
        if hand is not None:
            rospy.logerr("is_grabbed: hand 必须为 left、right 或 None")
            return False
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


# ============================================================
#  HeadController —— 头部控制
# ============================================================
class HeadController:
    """
    控制机器人头部（云台）运动。

    用法:
        head = HeadController()
        head.look_at(0, 0)     # 直视前方
        head.look_at(20, -10)  # 右看 20°，低头 10°
        head.look_left(15)     # 只看左边
        head.look_down(20)     # 只看下面
    """

    def __init__(self):
        self._pub = rospy.Publisher("/robot_head_motion_data",
                                    robotHeadMotionData, queue_size=10)
        rospy.sleep(0.1)

    def look_at(self, yaw=0.0, pitch=0.0):
        """
        控制头部角度。

        参数:
            yaw   — 偏航角 度，范围 [-30, 30]，正=右看，负=左看
            pitch — 俯仰角 度，范围 [-25, 25]，正=抬头，负=低头
        """
        yaw = max(-30.0, min(30.0, yaw))
        pitch = max(-25.0, min(25.0, pitch))
        msg = robotHeadMotionData()
        msg.joint_data = [float(yaw), float(pitch)]
        self._pub.publish(msg)
        rospy.loginfo("头部: yaw=%.1f° pitch=%.1f°", yaw, pitch)

    # ---- 快捷方法 ----

    def look_forward(self):
        """直视前方。"""
        self.look_at(0, 0)

    def look_left(self, angle=15.0):
        """向左看。angle 为正数。"""
        self.look_at(-abs(angle), 0)

    def look_right(self, angle=15.0):
        """向右看。"""
        self.look_at(abs(angle), 0)

    def look_up(self, angle=15.0):
        """抬头。"""
        self.look_at(0, abs(angle))

    def look_down(self, angle=15.0):
        """低头。"""
        self.look_at(0, -abs(angle))
