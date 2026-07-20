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


def _load_launcher_module():
    # 公共启动器位于受保护包 challenge_cup_simulator/utils/（选手不可改动），
    # 从那里导入，确保完整性校验无法被绕过。
    try:
        import rospkg
        sim_utils = os.path.join(rospkg.RosPack().get_path("challenge_cup_simulator"), "utils")
    except Exception:
        sim_utils = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "..", "challenge_cup_simulator", "utils")
    if sim_utils not in sys.path:
        sys.path.insert(0, sim_utils)
    import challenge_sim_launcher
    return challenge_sim_launcher


def _load_launcher():
    return _load_launcher_module().ChallengeSimLauncher


def _configure_camera_rendering(render_cameras):
    challenge_sim_launcher = _load_launcher_module()
    value = "true" if render_cameras else "false"
    challenge_sim_launcher.STABLE_CONTROL_ARGS["render_cameras"] = value
    print("[INFO] challenge_task: render_cameras:={}".format(value))


def _add_task_module_paths():
    # rosrun 执行的通常是 devel/lib 下的入口副本；不能据此推断源码根目录。
    # 优先由 rospkg 定位功能包，直接运行源码脚本时再使用相对路径回退。
    try:
        import rospkg
        pkg_dir = rospkg.RosPack().get_path("challenge_cup_task_template")
    except Exception:
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    src_dir = os.path.join(pkg_dir, "src")
    scripts_dir = os.path.join(pkg_dir, "scripts")
    for path in (src_dir, scripts_dir):
        if path not in sys.path:
            sys.path.insert(0, path)


def _run_teammate_scene(scene, seed=0, scene2_perception_only=False,
                        scene2_baseline_only=False,
                        scene2_perception_camera="head",
                        scene2_perception_output_dir="scene2_perception",
                        scene2_perception_target_frame="base_link",
                        display=False):
    _add_task_module_paths()

    import rospy

    # 场景二的视觉分拣使用其独立的控制管线；它不接收本入口创建的
    # RobotMover/ArmController 对象。
    if scene == "scene2":
        from scene2_task import (
            _run_scene2_baseline,
            _run_scene2_perception_debug,
            _run_scene2_visual_sorting,
        )

        if scene2_perception_only:
            _run_scene2_perception_debug(
                scene2_perception_camera,
                scene2_perception_output_dir,
                scene2_perception_target_frame,
                display=display,
            )
            rospy.loginfo("=== 场景二感知调试结束 ===")
        elif scene2_baseline_only:
            _run_scene2_baseline()
        else:
            _run_scene2_visual_sorting(
                camera=scene2_perception_camera,
                output_dir=scene2_perception_output_dir,
                target_frame=scene2_perception_target_frame,
                display=display,
            )
        return

    # 场景一保留组员扩展过的控制 API，避免覆盖场景二的 robot_api
    # 和场景三的 robot_api3。其任务动作自行创建官方 hold 控制器，故不在
    # 统一入口预先等待非必需的夹爪服务。
    if scene == "scene1":
        from robot_api1 import RobotMover, ArmController, HeadController
        from scene1_task import run_scene1

        robot = RobotMover()
        arm = ArmController()
        head = HeadController()

        def log(message, *args):
            rospy.loginfo(message, *args)

        run_scene1(robot, arm, None, head, log, seed=seed)
        return

    # 场景三依赖本分支扩展过的 robot_api3（腰部控制、单臂 IK 等）。
    from robot_api3 import RobotMover, ArmController, ClawController, HeadController
    from scene3_task import run_scene3

    robot = RobotMover()
    arm = ArmController()
    claw = ClawController()
    head = HeadController()

    def log(message, *args):
        rospy.loginfo(message, *args)

    run_scene3(robot, arm, claw, head, log)


def run_scene(scene, seed, node_name=None, timeout=120,
              time_limit=None, timer_gui=True, render_cameras=None,
              scene2_perception_only=False, scene2_baseline_only=False,
              scene2_perception_camera="head",
              scene2_perception_output_dir="scene2_perception",
              scene2_perception_target_frame="base_link", display=False):
    if scene not in SCENE_CONFIGS:
        raise ValueError("unknown scene: {}".format(scene))

    config = SCENE_CONFIGS[scene]
    if render_cameras is not None:
        _configure_camera_rendering(render_cameras)
    ChallengeSimLauncher = _load_launcher()

    launcher = ChallengeSimLauncher(
        scene=scene,
        seed=seed,
        match_time_limit=time_limit,
        timer_gui=timer_gui,
    )
    launcher.start(node_name=node_name or config["node_name"], timeout=timeout)

    import rospy

    rospy.loginfo("=== %s任务启动 ===", config["title"])

    rospy.sleep(1.0)
    rospy.loginfo("场景实例已初始化。")

    _run_teammate_scene(
        scene,
        seed=seed,
        scene2_perception_only=scene2_perception_only,
        scene2_baseline_only=scene2_baseline_only,
        scene2_perception_camera=scene2_perception_camera,
        scene2_perception_output_dir=scene2_perception_output_dir,
        scene2_perception_target_frame=scene2_perception_target_frame,
        display=display,
    )

    # 场景二的独立任务管线完成后应直接退出；场景一、三保留原有节点。
    if scene != "scene2":
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
    parser.add_argument("--render-cameras", action="store_true",
                        help="保留 MuJoCo 三路相机渲染与压缩发布；默认关闭以降低 CPU 压力")
    parser.add_argument("--scene2-perception-only", action="store_true",
                        help="仅用于开发调试：启动场景二并执行一次 RGB-D 识别，不执行抓取")
    parser.add_argument("--scene2-baseline-only", action="store_true",
                        help="仅用于回归对比：场景二不使用视觉识别，直接运行固定目标 baseline")
    parser.add_argument("--scene2-perception-camera", choices=["head", "left", "right"],
                        default="head", help="场景二感知使用的相机")
    parser.add_argument("--scene2-perception-output-dir", default="scene2_perception",
                        help="场景二感知图像和 JSON 输出目录（默认相对当前工作目录）")
    parser.add_argument("--scene2-perception-target-frame", default="base_link",
                        help="场景二感知的 TF 目标坐标系")
    parser.add_argument("--display", action="store_true",
                        help="弹出 imshow 窗口实时显示场景二检测结果（需要 X11 环境）")
    args = parser.parse_args()

    run_scene(
        scene=args.scene,
        seed=args.seed,
        node_name=args.node_name,
        timeout=args.timeout,
        time_limit=args.time_limit,
        timer_gui=not args.no_timer_gui,
        render_cameras=args.render_cameras,
        scene2_perception_only=args.scene2_perception_only,
        scene2_baseline_only=args.scene2_baseline_only,
        scene2_perception_camera=args.scene2_perception_camera,
        scene2_perception_output_dir=args.scene2_perception_output_dir,
        scene2_perception_target_frame=args.scene2_perception_target_frame,
        display=args.display,
    )


if __name__ == "__main__":
    main()
