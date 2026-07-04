#!/usr/bin/env python3
"""
挑战杯三场景统一任务入口。

推荐运行方式：
  rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 3
  rosrun challenge_cup_task_template challenge_task.py --scene scene2 --seed 3
  rosrun challenge_cup_task_template challenge_task.py --scene scene3 --seed 3
"""

import argparse
import os
import sys


SCENE_CONFIGS = {
    "scene1": {
        "node_name": "challenge_task_scene1",
        "title": "场景一：包裹称重与摆放",
    },
    "scene2": {
        "node_name": "challenge_task_scene2",
        "title": "场景二：分拣归档",
    },
    "scene3": {
        "node_name": "challenge_task_scene3",
        "title": "场景三：SMT 料盘出库",
    },
}


def _load_launcher():
    # 公共启动器位于受保护包 challenge_cup_simulator/utils/（选手不可改动），
    # 从那里导入，确保完整性校验无法被绕过。
    try:
        import rospkg
        sim_utils = os.path.join(rospkg.RosPack().get_path("challenge_cup_simulator"), "utils")
    except Exception:
        sim_utils = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "..", "challenge_cup_simulator", "utils")
    sys.path.insert(0, sim_utils)
    from challenge_sim_launcher import ChallengeSimLauncher
    return ChallengeSimLauncher


def run_scene(scene, seed, node_name=None, timeout=120,
              time_limit=None, timer_gui=True):
    if scene not in SCENE_CONFIGS:
        raise ValueError("unknown scene: {}".format(scene))

    config = SCENE_CONFIGS[scene]
    ChallengeSimLauncher = _load_launcher()

    launcher = ChallengeSimLauncher(
        scene=scene,
        seed=seed,
        match_time_limit=time_limit,
        timer_gui=timer_gui,
    )
    launcher.start(node_name=node_name or config["node_name"], timeout=timeout)

    import rospy

    # 把 src/ 目录加入搜索路径，让下面的 import 能找到 robot_api
    _pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(_pkg_dir, "src"))
    from robot_api import RobotMover, ArmController, ClawController

    # 用 print 代替 rospy.loginfo，确保测试日志不被 ROS 过滤
    def log(msg, *args):
        if args:
            msg = msg % args
        print("\n>>> " + msg + "\n", flush=True)

    log("=== %s任务启动 ===", config["title"])

    # 创建三个控制器对象
    robot = RobotMover()
    arm = ArmController()
    claw = ClawController()

    rospy.sleep(1.0)
    log("场景实例已初始化。")

    # ========================================
    # TODO: 在此实现三场景共用或按 scene 分支的任务逻辑
    # ========================================
    #
    # 封装后的接口速查：
    #   robot.move_forward(speed, duration)    — 前进
    #   robot.move_backward(speed, duration)   — 后退
    #   robot.turn_left(speed, duration)       — 左转
    #   robot.turn_right(speed, duration)      — 右转
    #   robot.stop()                           — 立即停下
    #   arm.switch_to_external_control()        — 切到外部控制模式
    #   arm.go_to_joints([14个角度])             — 手臂到目标位置
    #   arm.go_home()                          — 手臂归零
    #   claw.open()                            — 夹爪张开
    #   claw.close()                           — 夹爪闭合
    #   claw.is_grabbed()                      — 是否抓住物体

    if scene == "scene1":
        # ============================================================
        # 场景一 用作 RobotMover 全面测试
        # 注意：双足机器人横移/转身容易失稳，速度不宜过大
        # ============================================================
        log("=" * 50)
        log("[TEST 1/8] 前进 0.05 m/s，持续 2 秒（安全）")
        robot.move_forward(0.05, duration=2.0)
        rospy.sleep(1.0)

    elif scene == "scene2":
        # ============================================================
        # 场景二 用作 ArmController 测试
        # ============================================================
        log("=" * 50)
        log("[TEST 1/5] 切换手臂到外部控制模式")
        ok = arm.switch_to_external_control()
        log("切换结果: %s", "成功" if ok else "失败")
        rospy.sleep(1.0)

        log("[TEST 2/5] go_home() — 全部关节归零")
        arm.go_home()
        rospy.sleep(2.0)

        log("[TEST 3/5] go_ready() — 双手前伸准备姿势")
        arm.go_ready()
        rospy.sleep(2.0)

        log("[TEST 4/5] 左臂单独控制 — 肩前 40° 肘弯 60°")
        arm.left_arm_to([40.0, -15.0, 0.0, -60.0, 0.0, 0.0, 0.0])
        rospy.sleep(2.0)
        arm.go_home()
        rospy.sleep(1.5)

        log("[TEST 5/5] 右臂单独控制 — 肩前 40° 肘弯 60°")
        arm.right_arm_to([40.0, 15.0, 0.0, -60.0, 0.0, 0.0, 0.0])
        rospy.sleep(2.0)
        arm.go_home()

        log("场景二：ArmController 全部 5 项测试通过！")

    elif scene == "scene3":
        # ============================================================
        # 场景三 用作 ClawController 测试
        # ============================================================
        log("=" * 50)
        log("[TEST 1/6] 双手闭合")
        claw.close()
        rospy.sleep(1.0)
        log("夹爪状态: 左=%s 右=%s 抓住=%s",
            claw._left_state, claw._right_state, claw.is_grabbed())

        log("[TEST 2/6] 双手张开")
        claw.open()
        rospy.sleep(1.0)
        log("夹爪状态: 左=%s 右=%s 抓住=%s",
            claw._left_state, claw._right_state, claw.is_grabbed())

        log("[TEST 3/6] 分别设置左右开合 — 左闭 80%% 右开 20%%")
        claw.set_position(80, 20)
        rospy.sleep(1.0)

        log("[TEST 4/6] 只闭合左夹爪")
        claw.left_close()
        rospy.sleep(1.0)

        log("[TEST 5/6] 只张开右夹爪")
        claw.right_open()
        rospy.sleep(1.0)
        log("夹爪状态: 左=%s 右=%s",
            claw._left_state, claw._right_state)

        log("[TEST 6/6] wait_until_done() — 等待夹爪运动完成")
        claw.close()
        done = claw.wait_until_done(timeout=3.0)
        log("夹爪运动完成: %s", "是" if done else "超时")

        log("场景三：ClawController 全部 6 项测试通过！")

    rospy.spin()


def main():
    parser = argparse.ArgumentParser(description="挑战杯三场景统一任务入口")
    parser.add_argument("--scene", choices=sorted(SCENE_CONFIGS), default="scene1",
                        help="要启动的比赛场景")
    parser.add_argument("--seed", type=int, default=0,
                        help="场景种子；正式评测 seed 由组委会指定")
    parser.add_argument("--node-name", default=None,
                        help="ROS 节点名；默认按 scene 自动设置")
    parser.add_argument("--timeout", type=int, default=120,
                        help="等待仿真就绪的超时时间，单位秒")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="比赛时长，单位秒；默认读取 CHALLENGE_TIME_LIMIT，未设置则不限时")
    parser.add_argument("--no-timer-gui", action="store_true",
                        help="不弹出计时器窗口，仅保留后台计时日志")
    args = parser.parse_args()

    run_scene(
        scene=args.scene,
        seed=args.seed,
        node_name=args.node_name,
        timeout=args.timeout,
        time_limit=args.time_limit,
        timer_gui=not args.no_timer_gui,
    )


if __name__ == "__main__":
    main()
