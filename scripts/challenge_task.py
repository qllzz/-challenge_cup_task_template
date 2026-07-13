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
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(pkg_dir, "src")
    scripts_dir = os.path.join(pkg_dir, "scripts")
    for path in (src_dir, scripts_dir):
        if path not in sys.path:
            sys.path.insert(0, path)


def _run_teammate_scene(scene):
    _add_task_module_paths()

    import rospy
    from robot_api import RobotMover, ArmController, ClawController, HeadController
    from scene1_task import run_scene1
    from scene2_task import run_scene2
    from scene3_task import run_scene3

    robot = RobotMover()
    arm = ArmController()
    claw = ClawController()
    head = HeadController()

    def log(message, *args):
        rospy.loginfo(message, *args)

    scene_handlers = {
        "scene1": run_scene1,
        "scene2": run_scene2,
        "scene3": run_scene3,
    }
    scene_handlers[scene](robot, arm, claw, head, log)


def run_scene(scene, seed, node_name=None, timeout=120,
              time_limit=None, timer_gui=True, render_cameras=None):
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

    _run_teammate_scene(scene)

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
    args = parser.parse_args()

    run_scene(
        scene=args.scene,
        seed=args.seed,
        node_name=args.node_name,
        timeout=args.timeout,
        time_limit=args.time_limit,
        timer_gui=not args.no_timer_gui,
        render_cameras=args.render_cameras,
    )


if __name__ == "__main__":
    main()
