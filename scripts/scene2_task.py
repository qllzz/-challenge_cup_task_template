#!/usr/bin/env python3
"""场景二：分拣归档。

包含感知、视觉分拣、抓取、放置、重试恢复等完整逻辑。
由 challenge_task.py 统一入口调用。
"""

import importlib.util
import json
import math
import os
import sys
import time
from types import SimpleNamespace


def _sim_pkg_path():
    """获取 challenge_cup_simulator 包的路径。"""
    import rospkg
    return rospkg.RosPack().get_path("challenge_cup_simulator")


def _add_helper_path(*parts):
    """将辅助脚本路径添加到 sys.path。
    
    参数:
        *parts: 相对于 challenge_cup_simulator 包路径的子路径
    
    返回:
        str: 添加的路径
    """
    path = os.path.join(_sim_pkg_path(), *parts)
    if path not in sys.path:
        sys.path.insert(0, path)
    return path


def _run_scene2_perception_debug(camera, output_dir, target_frame, print_result=True, display=False):
    """运行场景二 RGB-D 感知模块。
    
    加载 scene2_perception.py 并调用 capture_once 进行一次感知。
    
    参数:
        camera:       相机名称 ("head"/"left"/"right")
        output_dir:   输出目录
        target_frame: 目标坐标系
        print_result: 是否打印结果
        display:      是否弹出 imshow 窗口显示检测结果
    
    返回:
        dict: {"detections": [...], "candidates": [...]}
    """
    import rospy

    try:
        _add_helper_path("test", "collect_scene2_dataset")
        import scene2_data_collection_pipeline as sc2
        if camera == "head":
            sc2._publish_head_target(sc2.TOPIC_TIMEOUT)
            rospy.sleep(max(0.5, float(getattr(sc2, "HEAD_SETTLE_TIME", 0.5))))
    except Exception as exc:
        rospy.logwarn("scene2 perception: failed to preset head target: %s", exc)

    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene2_perception.py")
    if not os.path.isfile(script_path) or "/devel/lib/" in os.path.abspath(script_path):
        import rospkg
        script_path = os.path.join(
            rospkg.RosPack().get_path("challenge_cup_task_template"),
            "scripts",
            "scene2_perception.py",
        )
    spec = importlib.util.spec_from_file_location("scene2_perception_source", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load scene2 perception script: {}".format(script_path))
    perception = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(perception)

    args = SimpleNamespace(
        camera=camera,
        target_frame=target_frame,
        output_dir=output_dir,
        timeout=20.0,
        tf_timeout=1.0,
        min_area=80.0,
        max_area=60000.0,
        min_depth=0.05 if camera in ("left", "right") else 0.30,
        max_depth=0.80 if camera in ("left", "right") else 0.90,
        display=display,
    )
    try:
        result = perception.capture_once(args)
        detections_count = len(result.get("detections") or [])
        candidates_count = len(result.get("candidates") or [])
        rospy.loginfo(
            "scene2 perception: %s camera capture succeeded, detections=%d candidates=%d",
            camera,
            detections_count,
            candidates_count,
        )
    except Exception as exc:
        rospy.logerr("scene2 perception: %s camera capture failed: %s", camera, exc)
        raise
    if print_result:
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return result


def _scene2_class_name(object_name):
    """从对象名称提取类别名称。
    
    参数:
        object_name: 如 "part_type_a_1"
    
    返回:
        str: 类别名，如 "part_type_a"
    """
    for class_name in ("part_type_a", "part_type_b", "part_type_c"):
        if object_name.startswith(class_name):
            return class_name
    raise ValueError("unknown scene2 object class: {}".format(object_name))


def _scene2_bin_for_object(object_name):
    """根据对象名称确定目标分拣槽。
    
    参数:
        object_name: 对象名称
    
    返回:
        str: 分拣槽名称 ("sorting_bin_a"/"sorting_bin_b"/"sorting_bin_c")
    """
    class_name = _scene2_class_name(object_name)
    return {
        "part_type_a": "sorting_bin_a",
        "part_type_b": "sorting_bin_b",
        "part_type_c": "sorting_bin_c",
    }[class_name]


def _scene2_fixed_place_target(sc2, object_name):
    """获取对象在对应分拣槽中的固定放置目标位置。
    
    参数:
        sc2:         scene2_data_collection_pipeline 模块
        object_name: 对象名称
    
    返回:
        list: [x, y, z] 放置目标位置
    """
    return list(sc2._place_target_xyz(_scene2_bin_for_object(object_name), 0))


def _scene2_enforce_fixed_bin(sc2, job):
    """强制任务使用固定分拣槽，忽略感知输出的放置区域。
    
    参数:
        sc2: scene2_data_collection_pipeline 模块
        job: 任务字典
    
    返回:
        dict: 更新后的任务字典
    """
    fixed_bin = _scene2_bin_for_object(job["object"])
    updated = dict(job)
    updated["bin"] = fixed_bin
    updated["place"] = _scene2_fixed_place_target(sc2, job["object"])
    if "perception" in updated:
        perception = dict(updated["perception"])
        perception["place_area"] = fixed_bin
        updated["perception"] = perception
    return updated


def _scene2_detection_grasp_offsets(det):
    """返回检测到的场景二零件的抓取偏移量列表。

    RGB-D 检测器估算物体在 base_link 中的可见中心。
    夹爪需要一个附近的但不完全相同的点：位于每种几何形状的有用夹取区域上方/内部。
    B 零件是 T 型的，容易被推开，因此它们使用较小的重试集，从不同方向接近杆部。
    """
    class_name = det["class"]
    if class_name == "part_type_a":
        # 2026-07-15 下午运行的稳定基线：保持在检测器中心附近，
        # 避免过深的向下偏移，防止在夹爪闭合前将小 A 零件推开。
        return [
            [0.030, -0.006, 0.010],
            [0.020, 0.030, 0.010],
            [0.015, 0.000, -0.010],
            [0.024, 0.018, 0.024],
            [0.015, -0.020, -0.014],
            [0.030, 0.000, -0.018],
        ]
    if class_name == "part_type_b":
        # T 型 B 零件用浅层公开样本抓取点重复性最好。
        # 两个 B 实例使用相同的候选集；机械臂根据观察到的源桌面位置单独选择。
        return [
            [0.035, 0.000, 0.011],
            [0.030, 0.000, 0.000],
            [0.020, 0.015, -0.012],
            [0.020, -0.015, -0.012],
            [0.005, 0.030, 0.000],
            [0.035, 0.020, -0.018],
            [0.000, 0.040, -0.012],
        ]
    if class_name == "part_type_c":
        # C 掩码是红色手柄本身，其深度质心已经是有效的夹取中心。
        # 之前的 4 cm 横向偏移可能让夹爪落在细杆上或完全错过手柄。
        # 重试仅保留小范围局部搜索；z 偏移保持夹爪接近几何校准的补偿值。
        if det.get("name") == "part_type_c_2":
            # C2 位于远端正 y 边缘。最近的合法运行显示
            # 左手 x 方向有约 2 cm 的欠冲，而旧的负 y 偏移
            # 将目标移离了观察到的手柄位置。在 RGB-D 手柄中心附近搜索并局部补偿 x。
            return [
                [0.020, 0.000, -0.020],
                [0.025, 0.000, -0.020],
                [0.020, 0.015, -0.020],
                [0.030, 0.015, -0.020],
                [0.015, -0.010, -0.020],
                [0.020, -0.015, -0.020],
                [0.020, 0.000, -0.035],
                [0.020, 0.000, -0.005],
            ]
        return [
            [0.000, 0.000, -0.020],
            [0.010, 0.000, -0.020],
            [-0.010, 0.000, -0.020],
            [0.000, 0.010, -0.020],
            [0.000, -0.010, -0.020],
            [0.000, 0.000, -0.035],
            [0.000, 0.000, -0.005],
        ]
    raise ValueError("unknown scene2 detection class: {}".format(class_name))


def _scene2_detection_grasp_offset(det):
    """获取检测结果的首选抓取偏移量。
    
    参数:
        det: 检测结果字典
    
    返回:
        list: [dx, dy, dz] 抓取偏移量
    """
    return _scene2_detection_grasp_offsets(det)[0]


def _scene2_job_grasp_offsets(job, det):
    """根据任务状态选择重试抓取偏移量（含恢复策略）。"""
    if job.get("is_recovery") and det.get("class") == "part_type_a":
        # A 零件在放置失败后被推离后，最初的 +2~3 cm x 补偿不再有效。
        # 在 2026-07-17 恢复日志中，两次重试后 A2 从 x=0.390 移到了 x=0.426。
        # 保留校准的 z 抬升补偿，但在当前 RGB-D 中心附近局部搜索，
        # 避免重试将其推得更远。
        return [
            [0.000, 0.000, 0.010],
            [-0.005, 0.000, 0.010],
            [0.005, -0.008, 0.010],
            [-0.010, -0.010, 0.010],
            [0.010, 0.000, 0.010],
            [0.000, 0.010, 0.010],
            [0.000, -0.015, 0.000],
            [-0.015, -0.008, 0.015],
        ]
    if job.get("is_recovery") and det.get("class") == "part_type_c":
        # C 抓取失败后可能将红色手柄推向正 y 边缘。
        # 最初的 C2 x 补偿是针对原始姿态校准的；
        # 位移后再次应用会超出左手可靠 IK 区域。
        # 因此恢复时在新观察到的中心附近搜索，并向内偏移远离边缘。
        return [
            [0.005, 0.000, -0.020],
            [0.000, 0.000, -0.020],
            [-0.005, 0.000, -0.020],
            [0.010, -0.008, -0.020],
            [0.015, -0.015, -0.020],
            [0.000, -0.015, -0.020],
            [-0.010, -0.010, -0.020],
            [0.010, 0.000, -0.035],
        ]
    return _scene2_detection_grasp_offsets(det)


def _scene2_rotate_detection_offset(det, offset_xyz):
    """对检测偏移量进行旋转变换（当前为恒等变换）。
    
    参数:
        det:        检测结果字典
        offset_xyz: 偏移量 [dx, dy, dz]
    
    返回:
        list: [dx, dy, dz]
    """
    return [float(v) for v in offset_xyz]


def _scene2_target_to_world_equivalent(sc2, target_xyz, arm):
    """将目标坐标系下的坐标转换为世界坐标系等效坐标。
    
    参数:
        sc2:        scene2_data_collection_pipeline 模块
        target_xyz: 目标坐标系下的 [x, y, z]
        arm:        机械臂 ("left"/"right")
    
    返回:
        list: [x, y, z] 世界坐标系等效坐标
    """
    y_offset = sc2.WORLD_TO_EE_OFFSET_Y_LEFT if arm == "left" else sc2.WORLD_TO_EE_OFFSET_Y_RIGHT
    return [
        float(target_xyz[0]) - float(sc2.WORLD_TO_EE_OFFSET_X),
        float(target_xyz[1]) - float(y_offset),
        float(target_xyz[2]) - float(sc2.WORLD_TO_EE_OFFSET_Z),
    ]


def _scene2_detection_pose_quat(object_name, arm, angle_deg, pose_key):
    """根据检测角度计算抓取/抬升/放置姿态四元数。
    
    参数:
        object_name: 对象名称
        arm:         机械臂 ("left"/"right")
        angle_deg:   检测到的物体角度（度）
        pose_key:    姿态类型 ("grasp_pose"/"lift_pose"/"place_pose")
    
    返回:
        list 或 None: [x, y, z, w] 四元数
    """
    if angle_deg is None:
        return None
    try:
        import scene2_part_grasp_ik as grasp_ik

        object_yaw_z = math.radians(float(angle_deg))
        pose_cfg = grasp_ik.OBJECT_PART_CONFIG[object_name][pose_key]
        adaptive_yaw_offset = 0.0
        if pose_key == "grasp_pose":
            adaptive_yaw_offset = grasp_ik._narrow_edge_yaw_offset_for_object(
                object_name,
                pose_cfg,
                object_yaw_z,
                arm,
            )
        elif pose_cfg.get("follow_grasp_narrow_edge", False):
            grasp_pose_cfg = grasp_ik.OBJECT_PART_CONFIG[object_name]["grasp_pose"]
            adaptive_yaw_offset = grasp_ik._narrow_edge_yaw_offset_for_object(
                object_name,
                grasp_pose_cfg,
                object_yaw_z,
                arm,
            )
        return list(grasp_ik._pose_config_to_quat_xyzw(
            pose_cfg,
            object_yaw_z,
            active_arm=arm,
            adaptive_yaw_offset=adaptive_yaw_offset,
        ))
    except Exception:
        return None


def _scene2_place_preferred_arm(bin_name):
    """返回放置到指定分拣槽的首选机械臂。
    
    参数:
        bin_name: 分拣槽名称
    
    返回:
        str 或 None: "left"/"right"/None
    """
    if bin_name == "sorting_bin_a":
        return "right"
    if bin_name == "sorting_bin_c":
        return "left"
    return None


def _scene2_arm_for_detection(object_name, bin_name, target_xyz):
    """根据检测位置选择抓取用机械臂。
    
    基于目标 Y 坐标判断物体在源桌面的左右半区，选择对应的机械臂。
    
    参数:
        object_name: 对象名称
        bin_name:    分拣槽名称
        target_xyz:  目标位置 [x, y, z]
    
    返回:
        str: "left" 或 "right"
    """
    target_y = float(target_xyz[1])
    # 用能到达当前源桌面半区的机械臂抓取。分拣槽侧后续由现有的 handoff 路径处理，
    # 因此被推到左半区的 A 零件先由左手抓取，然后交接给右手放置。
    if target_y > 0.08:
        return "left"
    if target_y < -0.08:
        return "right"
    preferred_arm = _scene2_place_preferred_arm(bin_name)
    if preferred_arm is not None:
        return preferred_arm
    if bin_name == "sorting_bin_b":
        # 历史稳定运行中，当 B 零件在中央源桌面区域时用左手抓取两个 B 零件。
        # 如果随机布局将 B 零件明显移到一侧，则使用观察到的该侧。
        if -0.10 <= target_y <= 0.10:
            return "left"
        return "right" if target_y < 0.0 else "left"
    return "left" if target_y >= 0.0 else "right"


def _scene2_apply_detection_to_job(sc2, job, det, grasp_offset=None, arm=None):
    """将检测结果应用到任务中，计算抓取目标位置。
    
    参数:
        sc2:          scene2_data_collection_pipeline 模块
        job:          任务字典
        det:          检测结果字典
        grasp_offset: 抓取偏移量 [dx, dy, dz]
        arm:          指定机械臂，None 则自动选择
    
    返回:
        dict: 更新后的任务字典
    """
    object_name = job["object"]
    base_xyz = det.get("base_link_xyz_m")
    if base_xyz is None:
        raise RuntimeError("scene2 detection has no base_link_xyz_m: {}".format(object_name))
    if grasp_offset is None:
        grasp_offset = _scene2_detection_grasp_offset(det)
    local_grasp_offset = [float(v) for v in grasp_offset]
    grasp_offset = _scene2_rotate_detection_offset(det, local_grasp_offset)
    grasp_xyz = [
        float(base_xyz[0]) + float(grasp_offset[0]),
        float(base_xyz[1]) + float(grasp_offset[1]),
        float(base_xyz[2]) + float(grasp_offset[2]),
    ]
    bin_name = _scene2_bin_for_object(object_name)
    if arm is None:
        arm = _scene2_arm_for_detection(object_name, bin_name, grasp_xyz)

    updated = dict(job)
    updated["bin"] = bin_name
    updated["place"] = _scene2_fixed_place_target(sc2, object_name)
    updated["arm"] = arm
    updated["world_xyz"] = _scene2_target_to_world_equivalent(sc2, grasp_xyz, arm)
    updated["grasp"] = grasp_xyz
    updated["perception"] = {
        "pixel": det.get("pixel"),
        "base_link_xyz_m": base_xyz,
        "area_px": det.get("area_px"),
        "angle_deg": det.get("angle_deg"),
        "depth_m": det.get("depth_m"),
        "place_area": bin_name,
        "selected_arm": arm,
        "grasp_offset": [float(v) for v in grasp_offset],
        "local_grasp_offset": local_grasp_offset,
    }
    updated.pop("grasp_quat", None)
    updated.pop("lift_quat", None)
    return updated


def _scene2_add_detection_pose_quats(job, det):
    """为任务添加基于检测角度计算的抓取/抬升/放置姿态四元数。
    
    仅对 type_a 和 type_c 零件计算姿态。
    
    参数:
        job: 任务字典
        det: 检测结果字典
    
    返回:
        dict: 更新后的任务字典（含 grasp_quat, lift_quat, place_quat）
    """
    if not job["object"].startswith(("part_type_a", "part_type_c")):
        return job
    # A 零件的宽扁轮廓在图像角度仅变化几度时就可能跨越窄边决策边界。
    # 其夹取不需要物体偏航跟踪，因此抓取/抬升姿态保持校准的零角度。
    grasp_lift_angle_deg = 0.0 if job["object"].startswith("part_type_a") else det.get("angle_deg")
    grasp_quat = _scene2_detection_pose_quat(
        job["object"],
        job["arm"],
        grasp_lift_angle_deg,
        "grasp_pose",
    )
    lift_quat = _scene2_detection_pose_quat(
        job["object"],
        job["arm"],
        grasp_lift_angle_deg,
        "lift_pose",
    )
    place_quat = _scene2_detection_pose_quat(
        job["object"],
        job["arm"],
        det.get("angle_deg"),
        "place_pose",
    )
    updated = dict(job)
    if grasp_quat is not None:
        updated["grasp_quat"] = grasp_quat
    if lift_quat is not None:
        updated["lift_quat"] = lift_quat
    if place_quat is not None:
        updated["place_quat"] = place_quat
    return updated


def _scene2_build_visual_jobs(sc2, detections):
    """根据检测结果构建分拣任务列表。
    
    将 6 个检测结果按类别排序，生成完整的抓取-放置任务。
    
    参数:
        sc2:        scene2_data_collection_pipeline 模块
        detections: 检测结果列表
    
    返回:
        list: 排序后的任务列表，按类别优先级排序
    """
    detections_by_name = {det.get("name"): det for det in detections}
    required = set(sc2.SORTING_OBJECT_ORDER)
    missing = sorted(required - set(detections_by_name))
    if missing:
        raise RuntimeError("scene2 visual detections missing objects: {}".format(", ".join(missing)))

    import rospy

    jobs = []
    slot_count_by_bin = {}
    for object_name in sc2.SORTING_OBJECT_ORDER:
        det = detections_by_name[object_name]
        bin_name = _scene2_bin_for_object(object_name)
        slot_index = slot_count_by_bin.get(bin_name, 0)
        slot_count_by_bin[bin_name] = slot_index + 1
        job = {
            "object": object_name,
            "bin": bin_name,
            "place": sc2._place_target_xyz(bin_name, slot_index),
        }
        job = _scene2_apply_detection_to_job(sc2, job, det)
        if object_name.startswith(("part_type_a", "part_type_b", "part_type_c")):
            job = _scene2_add_detection_pose_quats(job, det)
        jobs.append(job)
        rospy.loginfo(
            "scene2 visual sorting: built job for %s -> %s arm=%s grasp=%s",
            object_name,
            bin_name,
            job["arm"],
            [round(v, 4) for v in job["grasp"]],
        )
    class_priority = {"part_type_a": 0, "part_type_b": 1, "part_type_c": 2}
    return sorted(
        jobs,
        key=lambda item: (
            class_priority.get(_scene2_class_name(item["object"]), 99),
            item["object"],
        ),
    )


def _scene2_same_spot_detection(job, detections, xy_tolerance=0.075, z_tolerance=0.080):
    """检查物体是否仍在原位置（抓取失败检测）。
    
    参数:
        job:          任务字典
        detections:   当前检测结果列表
        xy_tolerance: 水平容差（米）
        z_tolerance:  垂直容差（米）
    
    返回:
        dict 或 None: 匹配的检测结果，或 None
    """
    expected = job.get("perception", {}).get("base_link_xyz_m")
    if expected is None:
        return None
    expected_class = _scene2_class_name(job["object"])
    for det in detections:
        if det.get("class") != expected_class:
            continue
        xyz = det.get("base_link_xyz_m")
        if xyz is None:
            continue
        dx = float(xyz[0]) - float(expected[0])
        dy = float(xyz[1]) - float(expected[1])
        dz = float(xyz[2]) - float(expected[2])
        if (dx * dx + dy * dy) ** 0.5 <= xy_tolerance and abs(dz) <= z_tolerance:
            return det
    return None


def _scene2_elevated_detection(job, detections, xy_tolerance=None, min_lift=0.120):
    """检查物体是否已被抓取抬升（抓取成功验证）。
    
    参数:
        job:          任务字典
        detections:   当前检测结果列表
        xy_tolerance: 水平容差（米），None 则按类别自动选择
        min_lift:     最小抬升高度（米）
    
    返回:
        dict 或 None: 匹配的抬升检测结果，或 None
    """
    expected = job.get("perception", {}).get("base_link_xyz_m")
    if expected is None:
        return None
    expected_class = _scene2_class_name(job["object"])
    if xy_tolerance is None:
        if job["object"].startswith("part_type_a"):
            # A1 在 2026-07-17 运行中 0.107 m 的边界检测被认定为已抬升，
            # 但在放置后返回了桌面。更早的稳定 A1 重试距离源中心 0.091 m。
            xy_tolerance = 0.10
        elif job["object"].startswith("part_type_b"):
            xy_tolerance = 0.14
        elif job["object"].startswith("part_type_c"):
            xy_tolerance = 0.18
        else:
            xy_tolerance = 0.20
    for det in detections:
        if det.get("class") != expected_class:
            continue
        xyz = det.get("base_link_xyz_m")
        if xyz is None:
            continue
        dx = float(xyz[0]) - float(expected[0])
        dy = float(xyz[1]) - float(expected[1])
        dz = float(xyz[2]) - float(expected[2])
        area = det.get("area_px")
        original_area = job.get("perception", {}).get("area_px")
        if job["object"].startswith("part_type_b"):
            if area is None or float(area) > 2500.0:
                continue
            if original_area is not None and float(area) > float(original_area) * 3.0:
                continue
        if (dx * dx + dy * dy) ** 0.5 <= xy_tolerance and dz >= min_lift:
            return det
    return None


def _scene2_wrist_camera_for_arm(arm):
    """返回指定机械臂对应的腕部相机。
    
    参数:
        arm: 机械臂 ("left"/"right")
    
    返回:
        str: "left" 或 "right"
    """
    return "left" if arm == "left" else "right"


def _scene2_nearest_class_detection(job, detections, max_xy_distance=None,
                                    source_table_only=True):
    """查找距离任务预期位置最近的同类检测结果。
    
    参数:
        job:              任务字典
        detections:       检测结果列表
        max_xy_distance:  最大水平距离
        source_table_only: 仅考虑源桌面检测
    
    返回:
        dict 或 None: 最近的同类检测结果
    """
    expected_class = _scene2_class_name(job["object"])
    reference = job.get("perception", {}).get("base_link_xyz_m") or job.get("grasp")
    matches = []
    for det in detections:
        if det.get("class") != expected_class:
            continue
        xyz = det.get("base_link_xyz_m")
        if xyz is None:
            continue
        if source_table_only and not _scene2_is_source_table_detection(det):
            continue
        if reference is not None and max_xy_distance is not None:
            if _scene2_xy_distance(xyz, reference) > float(max_xy_distance):
                continue
        matches.append(det)
    if not matches:
        return None
    if reference is None:
        return matches[0]
    return min(matches, key=lambda det: _scene2_xy_distance(det["base_link_xyz_m"], reference))


def _scene2_refine_job_with_wrist(
        sc2, job, locked_other_arm_joints, grasp_runtime,
        output_dir, target_frame):
    import rospy

    if os.environ.get("SCENE2_ENABLE_WRIST_REFINE", "0") != "1":
        return job

    if not job["object"].startswith(("part_type_a", "part_type_c")):
        return job

    active_arm = job["arm"]
    wrist_camera = _scene2_wrist_camera_for_arm(active_arm)
    look_target = list(job["grasp"])
    look_target[2] += 0.145
    look_quat = job.get("grasp_quat") or job.get("lift_quat")
    if look_quat is None:
        try:
            import scene2_part_grasp_ik as grasp_ik
            look_quat = grasp_ik.get_object_lift_quat_xyzw(job["object"], active_arm=active_arm)
        except Exception:
            look_quat = None

    try:
        sc2.move_arm_ik_once(
            runtime=grasp_runtime,
            active_arm=active_arm,
            active_pos=look_target,
            locked_other_arm_joints=locked_other_arm_joints,
            active_quat=look_quat,
            label="{}_wrist_look".format(job["object"]),
            constraint_mode=sc2.IK_MODE_THREE_POINT_MIXED,
            pos_cost_weight=2.0,
            move_time=max(float(sc2.ARM_MOVE_TIME), 1.6),
            settle_time=max(float(sc2.ARM_SETTLE_TIME), 0.45),
        )
        rospy.sleep(0.25)
        result = _run_scene2_perception_debug(
            wrist_camera,
            output_dir,
            target_frame,
            print_result=False,
        )
    except Exception as exc:
        rospy.logwarn("scene2 wrist refine: %s %s camera failed: %s", job["object"], wrist_camera, exc)
        return job

    max_xy = 0.13 if job["object"].startswith("part_type_a") else 0.20
    det = _scene2_nearest_class_detection(
        job,
        result.get("detections") or [],
        max_xy_distance=max_xy,
        source_table_only=True,
    )
    if det is None:
        rospy.logwarn(
            "scene2 wrist refine: %s no same-class source-table detection from %s camera",
            job["object"],
            wrist_camera,
        )
        return job

    refined = _scene2_apply_detection_to_job(sc2, job, det, arm=active_arm)
    refined = _scene2_add_detection_pose_quats(refined, det)
    rospy.loginfo(
        "scene2 wrist refine: %s %s camera=%s old_grasp=%s refined_camera=%s refined_grasp=%s",
        job["object"],
        active_arm,
        wrist_camera,
        [round(v, 4) for v in job["grasp"]],
        det.get("base_link_xyz_m"),
        [round(v, 4) for v in refined["grasp"]],
    )
    return refined


def _scene2_verify_pick_lifted(job, camera, output_dir, target_frame):
    """视觉验证抓取是否成功：检查物体是否被抬离桌面。
    
    参数:
        job:         任务字典
        camera:      相机名称
        output_dir:  输出目录
        target_frame: 目标坐标系
    
    返回:
        bool: 是否成功抬升
    """
    import rospy

    rospy.sleep(0.25)
    result = _run_scene2_perception_debug(camera, output_dir, target_frame, print_result=False)
    detections = result.get("detections") or []
    lifted = _scene2_elevated_detection(job, detections)
    if lifted is not None:
        rospy.loginfo(
            "scene2 visual verify: %s lifted, expected=%s observed=%s",
            job["object"],
            job.get("perception", {}).get("base_link_xyz_m"),
            lifted.get("base_link_xyz_m"),
        )
        return True
    still_there = _scene2_same_spot_detection(job, detections)
    if still_there is None:
        if job["object"].startswith("part_type_b"):
            if bool(job.get("last_grasp_ok", False)):
                rospy.logwarn(
                    "scene2 visual verify: %s disappeared after IK grasp; accepting provisional B verification and requiring bin/final inspection",
                    job["object"],
                )
                return True
            rospy.logwarn(
                "scene2 visual verify: %s disappeared without elevated evidence; retrying B part instead of weak-accepting early",
                job["object"],
            )
            return False
        if job["object"].startswith(("part_type_a", "part_type_c")):
            rospy.logwarn(
                "scene2 visual verify: %s disappeared but no elevated part was detected; retrying instead of weak-accepting A/C",
                job["object"],
            )
            return False
        rospy.logwarn(
            "scene2 visual verify: %s disappeared but no elevated part was detected; treating as failed/occluded",
            job["object"],
        )
        return False
    rospy.logwarn(
        "scene2 visual verify: %s still visible near original spot, expected=%s observed=%s",
        job["object"],
        job.get("perception", {}).get("base_link_xyz_m"),
        still_there.get("base_link_xyz_m"),
    )
    return False


def _scene2_xy_distance(a, b):
    """计算两个 3D 点之间的水平欧氏距离。
    
    参数:
        a, b: [x, y, z] 列表
    
    返回:
        float: sqrt((ax-bx)^2 + (ay-by)^2)
    """
    return (
        (float(a[0]) - float(b[0])) ** 2
        + (float(a[1]) - float(b[1])) ** 2
    ) ** 0.5


def _scene2_detection_matches_retry_region(job, det, max_xy_distance=None,
                                           require_same_source_side=False,
                                           source_table_only=False):
    """检查检测结果是否落在重试区域内。
    
    参数:
        job:                    任务字典
        det:                    检测结果
        max_xy_distance:        最大水平距离
        require_same_source_side: 要求与原始检测在同一侧
        source_table_only:      仅考虑源桌面区域
    
    返回:
        bool: 是否匹配
    """
    xyz = det.get("base_link_xyz_m")
    if xyz is None:
        return False
    reference = job.get("perception", {}).get("base_link_xyz_m")
    if reference is not None and max_xy_distance is not None:
        if _scene2_xy_distance(xyz, reference) > float(max_xy_distance):
            return False
    if reference is not None and require_same_source_side:
        ref_y = float(reference[1])
        det_y = float(xyz[1])
        if abs(ref_y) > 0.08 and (ref_y * det_y) < 0.0 and abs(det_y - ref_y) > 0.12:
            return False
    if source_table_only:
        x, y, z = [float(value) for value in xyz[:3]]
        if x < 0.14 or x > 0.48 or abs(y) > 0.52 or z > 0.10:
            return False
    object_name = job.get("object", "")
    if object_name.startswith("part_type_b_1") and float(xyz[1]) > 0.060:
        return False
    if object_name.startswith("part_type_b_2") and float(xyz[1]) < -0.020:
        return False
    return True


def _scene2_detection_for_job(job, detections, max_xy_distance=None,
                              require_same_source_side=False,
                              source_table_only=False):
    """在检测列表中查找与任务匹配的检测结果。
    
    优先按名称匹配，回退到同类最近距离匹配。
    
    参数:
        job:                    任务字典
        detections:             检测结果列表
        max_xy_distance:        最大水平距离
        require_same_source_side: 要求与原始检测在同一侧
        source_table_only:      仅考虑源桌面
    
    返回:
        dict 或 None: 匹配的检测结果
    """
    object_name = job["object"]
    for det in detections:
        if det.get("name") == object_name and _scene2_detection_matches_retry_region(
                job,
                det,
                max_xy_distance=max_xy_distance,
                require_same_source_side=require_same_source_side,
                source_table_only=source_table_only):
            return det

    expected_class = _scene2_class_name(object_name)
    reference = job.get("perception", {}).get("base_link_xyz_m")
    class_detections = [
        det for det in detections
        if det.get("class") == expected_class
        and _scene2_detection_matches_retry_region(
            job,
            det,
            max_xy_distance=max_xy_distance,
            require_same_source_side=require_same_source_side,
            source_table_only=source_table_only)
    ]
    if not class_detections:
        return None
    if reference is None:
        return class_detections[0]
    return min(
        class_detections,
        key=lambda det: (
            (float(det["base_link_xyz_m"][0]) - float(reference[0])) ** 2
            + (float(det["base_link_xyz_m"][1]) - float(reference[1])) ** 2
        ),
    )


def _scene2_retry_specs(job, det):
    """生成重试抓取规格列表（不含首选偏移量）。
    
    参数:
        job: 任务字典
        det: 检测结果
    
    返回:
        list: [{"arm", "offset", "label"}, ...]
    """
    if not job["object"].startswith(("part_type_a", "part_type_b", "part_type_c")):
        return []

    offsets = _scene2_detection_grasp_offsets(det)[1:]
    specs = []
    if job["object"].startswith("part_type_b"):
        for offset in offsets:
            specs.append({"arm": job["arm"], "offset": offset, "label": "same-arm"})
        return specs

    if job["object"].startswith("part_type_a"):
        for offset in offsets:
            specs.append({"arm": job["arm"], "offset": offset, "label": "same-arm"})
        return specs

    for offset in offsets:
        specs.append({"arm": job["arm"], "offset": offset, "label": "same-arm"})
    return specs


def _scene2_retry_limit(job):
    """获取各类零件的重试上限。
    
    参数:
        job: 任务字典
    
    返回:
        int: 最大重试次数
    """
    if job["object"].startswith("part_type_a"):
        return 8
    if job["object"].startswith("part_type_c"):
        return 6
    if job["object"].startswith("part_type_b"):
        return 9
    return 3


def _scene2_retry_search_distance(job):
    """获取重试时的搜索范围。
    
    参数:
        job: 任务字典
    
    返回:
        float: 最大搜索距离（米）
    """
    if job["object"].startswith("part_type_a"):
        return 0.45
    if job["object"].startswith("part_type_c"):
        return 0.35
    return 0.22


def _scene2_relocalize_retry_job(sc2, job, camera, output_dir, target_frame,
                                 attempt_index):
    """重新定位并生成重试任务。
    
    参数:
        sc2:           scene2_data_collection_pipeline 模块
        job:           任务字典
        camera:        相机名称
        output_dir:    输出目录
        target_frame:  目标坐标系
        attempt_index: 当前重试次数
    
    返回:
        dict 或 None: 重试任务，或 None
    """
    import rospy

    result = _run_scene2_perception_debug(camera, output_dir, target_frame, print_result=False)
    detections = result.get("detections") or []
    source_table_only = not bool(job.get("allow_bin_recovery", False))
    det = _scene2_detection_for_job(
        job,
        detections,
        max_xy_distance=_scene2_retry_search_distance(job),
        require_same_source_side=False,
        source_table_only=source_table_only,
    )
    if det is None:
        det = _scene2_nearest_class_detection(
            job,
            detections,
            max_xy_distance=_scene2_retry_search_distance(job),
            source_table_only=source_table_only,
        )
    if det is None:
        rospy.logwarn("scene2 retry: %s no source-table same-class target found for re-grasp", job["object"])
        return None

    offsets = _scene2_job_grasp_offsets(job, det)
    offset = offsets[min(max(int(attempt_index) - 1, 0), len(offsets) - 1)]
    retry_job = _scene2_apply_detection_to_job(
        sc2,
        job,
        det,
        grasp_offset=offset,
        arm=None,
    )
    retry_job = _scene2_add_detection_pose_quats(retry_job, det)
    rospy.logwarn(
        "scene2 retry: %s re-localized attempt=%d camera=%s offset=%s arm=%s grasp=%s",
        job["object"],
        attempt_index,
        det.get("base_link_xyz_m"),
        [round(float(v), 4) for v in offset],
        retry_job["arm"],
        [round(v, 4) for v in retry_job["grasp"]],
    )
    return retry_job


def _scene2_refresh_retry_job(sc2, job, camera, output_dir, target_frame, retry_spec):
    """刷新重试任务的抓取目标（物体未移动时的快速重试）。
    
    参数:
        sc2:           scene2_data_collection_pipeline 模块
        job:           任务字典
        camera:        相机名称
        output_dir:    输出目录
        target_frame:  目标坐标系
        retry_spec:    重试规格 {"offset": [...], "arm": ...}
    
    返回:
        dict 或 None: 更新后的任务
    """
    import rospy

    max_xy_distance = 0.14 if job["object"].startswith("part_type_a") else 0.20
    result = _run_scene2_perception_debug(camera, output_dir, target_frame, print_result=False)
    det = _scene2_detection_for_job(
        job,
        result.get("detections") or [],
        max_xy_distance=max_xy_distance,
        require_same_source_side=True,
        source_table_only=True,
    )
    if det is None:
        rospy.logwarn("scene2 retry: %s not visible, cannot refresh grasp", job["object"])
        return None
    return _scene2_apply_detection_to_job(
        sc2,
        job,
        det,
        grasp_offset=retry_spec["offset"],
        arm=retry_spec["arm"],
    )


def _scene2_gripper_close_time(sc2, job):
    """获取夹爪闭合时间。
    
    参数:
        sc2: scene2_data_collection_pipeline 模块
        job: 任务字典
    
    返回:
        float: 夹爪闭合时间（秒）
    """
    # 保持历史稳定配置中的闭合保持时间。几何形状由抓取偏移量处理；
    # 更长的保持时间只会增加抬升前带载臂的下垂。
    return sc2.GRIPPER_CLOSE_TIME


def _scene2_pick_and_verify_once(
        sc2, arm_pub, gripper_hold, job, grasp_runtime,
        camera, output_dir, target_frame):
    """执行单次抓取并视觉验证。
    
    参数:
        sc2:            scene2_data_collection_pipeline 模块
        arm_pub:        手臂控制 Publisher
        gripper_hold:   夹爪状态保持器
        job:            任务字典
        grasp_runtime:  GraspRuntime 实例
        camera:         相机名称
        output_dir:     输出目录
        target_frame:   目标坐标系
    
    返回:
        tuple: (verified, locked_other_arm_joints, job)
    """
    import rospy

    active_arm = job["arm"]
    locked_other_arm_joints = sc2._fixed_work_pose_other_arm_joints(active_arm)
    grasp_runtime.gripper_close_time = _scene2_gripper_close_time(sc2, job)
    job = _scene2_refine_job_with_wrist(
        sc2,
        job,
        locked_other_arm_joints,
        grasp_runtime,
        output_dir,
        target_frame,
    )
    try:
        rospy.loginfo(
            "scene2 sorting: %s attempting pick IK execution, arm=%s grasp=%s",
            job["object"],
            active_arm,
            [round(v, 4) for v in job["grasp"]],
        )
        grasp_ok = sc2._pick_part_absolute(
            arm_pub,
            None,
            gripper_hold,
            job,
            locked_other_arm_joints,
            sc2.FAST_GRASP_SETTLE_HOLD,
            grasp_runtime,
        )
        rospy.loginfo(
            "scene2 sorting: %s pick IK execution completed, ik_result=%s",
            job["object"],
            "success" if grasp_ok else "failed",
        )
    except Exception as exc:
        rospy.logerr("scene2 sorting: %s pick execution failed with exception: %s", job["object"], exc)
        return False, locked_other_arm_joints, job
    job = dict(job)
    job["last_grasp_ok"] = bool(grasp_ok)
    if not grasp_ok:
        rospy.logwarn("scene2 sorting: %s IK reported failed; checking visual lift evidence", job["object"])
    verified = _scene2_verify_pick_lifted(job, camera, output_dir, target_frame)
    return verified, locked_other_arm_joints, job


def _scene2_place_part_absolute(
        sc2, arm_pub, arm_hold, gripper_hold, job,
        locked_other_arm_joints, hold_time, grasp_runtime):
    """将零件放置到目标分拣槽。
    
    对 type_a 零件使用笛卡尔空间放置策略（避免碰撞），
    其他零件使用默认放置策略。
    
    参数:
        sc2:                     scene2_data_collection_pipeline 模块
        arm_pub:                 手臂控制 Publisher
        arm_hold:                手臂状态保持器
        gripper_hold:            夹爪状态保持器
        job:                     任务字典
        locked_other_arm_joints: 另一手臂的固定关节角度
        hold_time:               保持时间
        grasp_runtime:           GraspRuntime 实例
    """
    import rospy

    job = _scene2_enforce_fixed_bin(sc2, job)
    if job["object"].startswith("part_type_a"):
        # 旧的 A 槽关节姿态从侧面接近轻质空槽。
        # 改用已校准的笛卡尔放置目标：移动到槽上方，垂直下降，
        # 释放，然后垂直收回再返回工作姿态。
        active_arm = job["arm"]
        place_target = [float(value) for value in job["place"]]
        # 右手笛卡尔 IK 在上次运行中比标称投放高度低了 4-6 cm。
        # 将释放点提高 6 cm，确保实际 TCP 保持在槽底板/槽壁上方而非压入其中。
        drop_target = list(place_target)
        drop_target[2] += 0.06
        # 保持接近姿态在之前可达的高度。将其与投放目标一同提高
        # 导致了较大的姿态误差，持有的 A 零件在到达槽之前就丢失了。
        approach_target = list(place_target)
        approach_target[2] += 0.12
        held_quat = job.get("lift_quat")
        place_quat = job.get("place_quat") or held_quat
        rospy.loginfo(
            "scene2 sorting: %s %s-hand Cartesian clearance place target=%s drop=%s approach=%s",
            job["object"],
            active_arm,
            [round(value, 4) for value in place_target],
            [round(value, 4) for value in drop_target],
            [round(value, 4) for value in approach_target],
        )
        sc2.move_arm_ik_once(
            runtime=grasp_runtime,
            active_arm=active_arm,
            active_pos=approach_target,
            locked_other_arm_joints=locked_other_arm_joints,
            active_quat=held_quat,
            label="{}_place_above".format(job["object"]),
            constraint_mode=sc2.IK_MODE_THREE_POINT_MIXED,
            pos_cost_weight=2.0,
            move_time=max(float(sc2.ARM_MOVE_TIME), 1.6),
            settle_time=max(float(hold_time), 0.35),
        )
        drop_err, _quat_err, actual, _actual_quat, _cmd14 = sc2.move_arm_ik_once(
            runtime=grasp_runtime,
            active_arm=active_arm,
            active_pos=drop_target,
            locked_other_arm_joints=locked_other_arm_joints,
            active_quat=place_quat,
            label="{}_place_drop".format(job["object"]),
            constraint_mode=sc2.IK_MODE_THREE_POINT_MIXED,
            pos_cost_weight=2.0,
            move_time=max(float(sc2.ARM_MOVE_TIME), 1.6),
            settle_time=max(float(hold_time), 0.35),
        )
        rospy.loginfo(
            "scene2 sorting: %s Cartesian place actual=%s xyz_err=%.4f m; release %s gripper",
            job["object"],
            [round(value, 4) for value in actual],
            drop_err,
            active_arm,
        )
        # 持 A 零件时手臂可能下垂数厘米。FK 是在上述运动后测量的，
        # 因此在释放前做一次小修正，而不是仅依赖标称 IK 目标。
        residual = [
            float(drop_target[index]) - float(actual[index])
            for index in range(3)
        ]
        correction = [
            max(-0.025, min(0.025, residual[0])),
            max(-0.025, min(0.025, residual[1])),
            max(-0.035, min(0.035, residual[2])),
        ]
        if math.sqrt(sum(value * value for value in residual)) > 0.015:
            corrected_drop = [
                float(drop_target[index]) + correction[index]
                for index in range(3)
            ]
            rospy.logwarn(
                "scene2 sorting: %s A placement correction residual=%s correction=%s target=%s",
                job["object"],
                [round(value, 4) for value in residual],
                [round(value, 4) for value in correction],
                [round(value, 4) for value in corrected_drop],
            )
            _correction_err, _correction_quat_err, actual, _actual_quat, _cmd14 = sc2.move_arm_ik_once(
                runtime=grasp_runtime,
                active_arm=active_arm,
                active_pos=corrected_drop,
                locked_other_arm_joints=locked_other_arm_joints,
                active_quat=place_quat,
                label="{}_place_corrected".format(job["object"]),
                constraint_mode=sc2.IK_MODE_THREE_POINT_MIXED,
                pos_cost_weight=2.0,
                move_time=max(float(sc2.ARM_MOVE_TIME), 1.6),
                settle_time=max(float(hold_time), 0.35),
            )
        sc2._publish_gripper_open(gripper_hold)
        rospy.sleep(sc2.PLACE_DWELL)
        sc2.move_arm_ik_once(
            runtime=grasp_runtime,
            active_arm=active_arm,
            active_pos=approach_target,
            locked_other_arm_joints=locked_other_arm_joints,
            active_quat=place_quat,
            label="{}_place_retract".format(job["object"]),
            constraint_mode=sc2.IK_MODE_THREE_POINT_MIXED,
            pos_cost_weight=2.0,
            move_time=max(float(sc2.ARM_MOVE_TIME), 1.6),
            settle_time=max(float(hold_time), 0.35),
        )
        sc2._move_to_work_pose_joints(
            arm_pub,
            arm_hold,
            active_arm=active_arm,
            locked_other_arm_joints=locked_other_arm_joints,
        )
        rospy.loginfo(
            "scene2 sorting: %s Cartesian place completed, retracted to work pose",
            job["object"],
        )
        return
    rospy.loginfo(
        "scene2 sorting: %s using default place strategy for %s",
        job["object"],
        job["bin"],
    )
    sc2._place_part_absolute(
        arm_pub,
        arm_hold,
        gripper_hold,
        job,
        locked_other_arm_joints,
        hold_time,
        grasp_runtime,
    )


def _scene2_reset_for_retry(sc2, arm_pub, arm_hold, gripper_hold, job, locked_other_arm_joints):
    """重置机械臂状态以准备重试抓取。
    
    打开夹爪，回到工作姿态。
    
    参数:
        sc2:                     scene2_data_collection_pipeline 模块
        arm_pub:                 手臂控制 Publisher
        arm_hold:                手臂状态保持器
        gripper_hold:            夹爪状态保持器
        job:                     任务字典
        locked_other_arm_joints: 另一手臂的固定关节角度
    """
    import rospy

    sc2._publish_gripper_open(gripper_hold)
    rospy.sleep(0.25)
    sc2._move_to_work_pose_joints(
        arm_pub,
        arm_hold,
        active_arm=job["arm"],
        locked_other_arm_joints=locked_other_arm_joints,
    )
    rospy.sleep(0.25)


def _scene2_safe_logwarn(message, *args, **kwargs):
    """安全的 logwarn 包装，避免 TypeError。
    
    参数:
        message: 日志消息
        *args:   格式化参数
        **kwargs: 额外关键字参数
    """
    import rospy

    try:
        rospy.logwarn(message, *args, **kwargs)
    except TypeError:
        rendered = str(message)
        if args:
            rendered = "{} args={}".format(rendered, args)
        rospy.logwarn("%s", rendered)


def _scene2_slow_down_motion(sc2):
    """适度降低场景二运动速度以保证稳定性。
    
    参数:
        sc2: scene2_data_collection_pipeline 模块
    """
    # 基于 2026-07-15 17:01 运行的基线数据。这些速度对比赛控制器来说仍然
    # 足够慢，但避免了后续实验中使结果不可重复的额外轨迹稳定时间。
    sc2.ARM_MOVE_TIME = 1.4
    sc2.ARM_SETTLE_TIME = 0.15
    sc2.FAST_GRASP_SETTLE_HOLD = 0.8
    sc2.GRIPPER_CLOSE_TIME = 0.6
    sc2.PLACE_DWELL = 0.4


def _scene2_bin_for_xyz(xyz):
    """根据 3D 坐标判断物体位于哪个分拣槽。
    
    参数:
        xyz: [x, y, z] 3D 坐标
    
    返回:
        str 或 None: 分拣槽名称，或 None
    """
    if xyz is None:
        return None
    x, y, z = [float(value) for value in xyz[:3]]
    # RGB-D 投影会低估槽前边缘数厘米。日志中观察到的槽内物体 x=0.46..0.55；
    # A 零件在槽内仍可能投影到 z=-0.06，因此用 x/y 定义槽区域，
    # z 仅排除明显偏高/无效的观测。
    if x < 0.43 or x > 0.62 or z < -0.09:
        return None
    if y < -0.25:
        return "sorting_bin_a"
    if y > 0.24:
        return "sorting_bin_c"
    if -0.20 <= y <= 0.20:
        return "sorting_bin_b"
    return None


def _scene2_expected_class_for_bin(bin_name):
    """返回指定分拣槽应放置的零件类别。
    
    参数:
        bin_name: 分拣槽名称
    
    返回:
        str 或 None: 零件类别
    """
    return {
        "sorting_bin_a": "part_type_a",
        "sorting_bin_b": "part_type_b",
        "sorting_bin_c": "part_type_c",
    }.get(bin_name)


def _scene2_is_source_table_detection(det):
    """判断检测结果是否在源桌面区域（非分拣槽）。
    
    参数:
        det: 检测结果字典
    
    返回:
        bool: 是否在源桌面
    """
    xyz = det.get("base_link_xyz_m")
    if xyz is None:
        return False
    x, y, z = [float(value) for value in xyz[:3]]
    if _scene2_bin_for_xyz(xyz) is not None:
        return False
    # 源/工作桌面在机器人前方，三个分拣槽之前。
    # 偏高的检测通常是夹爪遮挡、槽边缘或被抬起的零件。
    return 0.14 <= x <= 0.48 and abs(y) <= 0.52 and z <= 0.10


def _scene2_detection_summary(det):
    """生成检测结果的摘要字符串。
    
    参数:
        det: 检测结果字典
    
    返回:
        str: "name:class@[x,y,z]"
    """
    xyz = det.get("base_link_xyz_m")
    if xyz is None:
        xyz_text = "None"
    else:
        xyz_text = "[{:.3f},{:.3f},{:.3f}]".format(float(xyz[0]), float(xyz[1]), float(xyz[2]))
    return "{}:{}@{}".format(det.get("name"), det.get("class"), xyz_text)


def _scene2_table_issues_from_detections(detections):
    """分析检测结果，识别源桌面遗留物和错误放置的零件。
    
    参数:
        detections: 检测结果列表
    
    返回:
        tuple: (leftovers, misplaced, bin_hits)
    """
    leftovers = []
    misplaced = []
    bin_hits = []
    for det in detections:
        bin_name = _scene2_bin_for_xyz(det.get("base_link_xyz_m"))
        if bin_name is not None:
            expected_class = _scene2_expected_class_for_bin(bin_name)
            if det.get("class") != expected_class:
                misplaced.append((bin_name, det))
            else:
                bin_hits.append((bin_name, det))
            continue
        if _scene2_is_source_table_detection(det):
            leftovers.append(det)
    return leftovers, misplaced, bin_hits


def _scene2_capture_table_issues(camera, output_dir, target_frame):
    """捕获当前桌面状态，识别遗留物和错误放置。
    
    参数:
        camera:       相机名称
        output_dir:   输出目录
        target_frame: 目标坐标系
    
    返回:
        tuple: (leftovers, misplaced, bin_hits)
    """
    result = _run_scene2_perception_debug(camera, output_dir, target_frame, print_result=False)
    return _scene2_table_issues_from_detections(result.get("detections") or [])


def _scene2_build_recovery_jobs(sc2, leftovers, misplaced, recovery_round):
    """根据遗留物和错误放置构建恢复任务。
    
    参数:
        sc2:            scene2_data_collection_pipeline 模块
        leftovers:      源桌面遗留物列表
        misplaced:      错误放置的零件列表
        recovery_round: 恢复轮次
    
    返回:
        list: 恢复任务列表
    """
    jobs = []
    seen_names = set()
    class_priority = {"part_type_a": 0, "part_type_b": 1, "part_type_c": 2}

    recovery_items = [("source_table", None, det) for det in leftovers]
    recovery_items.extend(("wrong_bin", bin_name, det) for bin_name, det in misplaced)

    for det_index, (reason, observed_bin, det) in enumerate(recovery_items, start=1):
        class_name = det.get("class")
        if class_name not in class_priority:
            continue
        object_name = det.get("name") or "{}_{}".format(class_name, det_index)
        if object_name in seen_names:
            object_name = "{}_recovery_{}_{}".format(class_name, recovery_round, det_index)
        seen_names.add(object_name)
        bin_name = _scene2_bin_for_object(object_name)
        job = {
            "object": object_name,
            "bin": bin_name,
            "place": _scene2_fixed_place_target(sc2, object_name),
            "is_recovery": True,
            "recovery_round": int(recovery_round),
            "recovery_reason": reason,
            "observed_bin": observed_bin,
            "allow_bin_recovery": reason == "wrong_bin",
        }
        job = _scene2_apply_detection_to_job(
            sc2,
            job,
            det,
            grasp_offset=_scene2_job_grasp_offsets(job, det)[0],
        )
        job["is_recovery"] = True
        job["recovery_round"] = int(recovery_round)
        job["recovery_reason"] = reason
        job["observed_bin"] = observed_bin
        job["allow_bin_recovery"] = reason == "wrong_bin"
        if object_name.startswith(("part_type_a", "part_type_b", "part_type_c")):
            job = _scene2_add_detection_pose_quats(job, det)
        jobs.append(job)
    return sorted(
        jobs,
        key=lambda item: (
            class_priority.get(_scene2_class_name(item["object"]), 99),
            float(item.get("perception", {}).get("base_link_xyz_m", [0.0, 0.0, 0.0])[1]),
            item["object"],
        ),
    )


def _scene2_final_table_inspection(sc2, camera, output_dir, target_frame,
                                   captures=8, required_clear=3):
    """最终检查：确保源桌面清空且无错误放置。
    
    参数:
        sc2:            scene2_data_collection_pipeline 模块
        camera:         相机名称
        output_dir:     输出目录
        target_frame:   目标坐标系
        captures:       最大检查次数
        required_clear: 需要连续清空的次数
    
    返回:
        bool: 检查通过
    
    异常:
        RuntimeError: 检查失败
    """
    import rospy

    consecutive_clear = 0
    last_leftovers = []
    last_misplaced = []
    for attempt in range(1, int(captures) + 1):
        result = _run_scene2_perception_debug(camera, output_dir, target_frame, print_result=False)
        leftovers, misplaced, bin_hits = _scene2_table_issues_from_detections(
            result.get("detections") or []
        )

        if leftovers or misplaced:
            consecutive_clear = 0
            last_leftovers = leftovers
            last_misplaced = misplaced
            rospy.logwarn(
                "scene2 final inspection %d/%d: not clear, leftovers=%s misplaced=%s",
                attempt,
                captures,
                [_scene2_detection_summary(det) for det in leftovers],
                ["{}<-{}".format(bin_name, _scene2_detection_summary(det)) for bin_name, det in misplaced],
            )
        else:
            consecutive_clear += 1
            rospy.loginfo(
                "scene2 final inspection %d/%d: source table clear (%d/%d consecutive), bin_hits=%s",
                attempt,
                captures,
                consecutive_clear,
                required_clear,
                ["{}<-{}".format(bin_name, det.get("name")) for bin_name, det in bin_hits],
            )
            if consecutive_clear >= int(required_clear):
                rospy.loginfo("scene2 final inspection: passed, source table clear and no wrong-bin detections")
                return True
        rospy.sleep(0.45)

    raise RuntimeError(
        "scene2 final inspection failed: leftovers={} misplaced={}".format(
            [_scene2_detection_summary(det) for det in last_leftovers],
            ["{}<-{}".format(bin_name, _scene2_detection_summary(det)) for bin_name, det in last_misplaced],
        )
    )


def _scene2_verify_recent_place(job, camera, output_dir, target_frame):
    """验证零件是否被正确放置到目标分拣槽。
    
    参数:
        job:          任务字典
        camera:       相机名称
        output_dir:   输出目录
        target_frame: 目标坐标系
    
    返回:
        bool: 放置是否成功
    """
    import rospy

    expected_bin = _scene2_bin_for_object(job["object"])
    expected_class = _scene2_class_name(job["object"])
    result = _run_scene2_perception_debug(camera, output_dir, target_frame, print_result=False)
    bin_hits = []
    wrong_hits = []
    for det in result.get("detections") or []:
        if det.get("class") != expected_class:
            continue
        bin_name = _scene2_bin_for_xyz(det.get("base_link_xyz_m"))
        if bin_name is None:
            continue
        if bin_name == expected_bin:
            bin_hits.append(det)
        else:
            wrong_hits.append((bin_name, det))

    if wrong_hits:
        rospy.logwarn(
            "scene2 place verification: %s expected %s, wrong_bin=%s; final recovery will re-sort",
            job["object"],
            expected_bin,
            ["{}<-{}".format(bin_name, _scene2_detection_summary(det))
             for bin_name, det in wrong_hits],
        )
        return False
    if bin_hits:
        rospy.loginfo(
            "scene2 place verification: %s confirmed in %s hits=%s",
            job["object"],
            expected_bin,
            [_scene2_detection_summary(det) for det in bin_hits],
        )
        return True

    rospy.logwarn(
        "scene2 place verification: %s expected %s but no same-class bin detection; final inspection will decide",
        job["object"],
        expected_bin,
    )
    return False


def _scene2_execute_sort_job(sc2, arm_pub, arm_hold, gripper_hold, job,
                             grasp_runtime, camera, output_dir, target_frame,
                             index=None, total=None):
    """执行单个分拣任务：抓取→验证→重试→放置→验证。
    
    参数:
        sc2:            scene2_data_collection_pipeline 模块
        arm_pub:        手臂控制 Publisher
        arm_hold:       手臂状态保持器
        gripper_hold:   夹爪状态保持器
        job:            任务字典
        grasp_runtime:  GraspRuntime 实例
        camera:         相机名称
        output_dir:     输出目录
        target_frame:   目标坐标系
        index:          任务序号
        total:          任务总数
    
    返回:
        dict: 执行后的任务（可能含 handoff 信息）
    """
    import rospy

    prefix = "scene2 recovery" if job.get("is_recovery") else "scene2 sorting"
    if index is not None and total is not None:
        rospy.loginfo(
            "%s: job %d/%d %s (%s hand) -> %s grasp=%s place=%s",
            prefix,
            index,
            total,
            job["object"],
            job["arm"],
            job["bin"],
            [round(v, 4) for v in job["grasp"]],
            [round(v, 4) for v in job["place"]],
        )
    else:
        rospy.loginfo(
            "%s: job %s (%s hand) -> %s grasp=%s place=%s",
            prefix,
            job["object"],
            job["arm"],
            job["bin"],
            [round(v, 4) for v in job["grasp"]],
            [round(v, 4) for v in job["place"]],
        )

    job = dict(job)
    job["attempt_index"] = 1
    sc2._publish_gripper_open(gripper_hold)
    verified, locked_other_arm_joints, job = _scene2_pick_and_verify_once(
        sc2,
        arm_pub,
        gripper_hold,
        job,
        grasp_runtime,
        camera,
        output_dir,
        target_frame,
    )
    attempt_index = 1
    attempt_limit = _scene2_retry_limit(job)
    while not verified and attempt_index < attempt_limit:
        attempt_index += 1
        rospy.logwarn(
            "%s: %s pick verification failed; re-grasp attempt %d/%d",
            prefix,
            job["object"],
            attempt_index,
            attempt_limit,
        )
        _scene2_reset_for_retry(
            sc2,
            arm_pub,
            arm_hold,
            gripper_hold,
            job,
            locked_other_arm_joints,
        )
        retry_job = _scene2_relocalize_retry_job(
            sc2,
            job,
            camera,
            output_dir,
            target_frame,
            attempt_index,
        )
        if retry_job is None:
            rospy.sleep(0.6)
            continue
        retry_job = dict(retry_job)
        retry_job["attempt_index"] = attempt_index
        verified, locked_other_arm_joints, retry_job = _scene2_pick_and_verify_once(
            sc2,
            arm_pub,
            gripper_hold,
            retry_job,
            grasp_runtime,
            camera,
            output_dir,
            target_frame,
        )
        job = retry_job
    if not verified:
        _scene2_reset_for_retry(
            sc2,
            arm_pub,
            arm_hold,
            gripper_hold,
            job,
            locked_other_arm_joints,
        )
        raise RuntimeError(
            "{}: {} failed grasp verification after {} attempts; stop for recovery".format(
                prefix,
                job["object"],
                attempt_limit,
            )
        )

    handoff_arm = sc2._handoff_target_arm(job)
    if handoff_arm is not None:
        rospy.loginfo(
            "%s: %s handoff required, transferring from %s to %s",
            prefix,
            job["object"],
            job["arm"],
            handoff_arm,
        )
        handoff_world_xyz = sc2._transfer_part_to_handoff(
            arm_pub,
            arm_hold,
            gripper_hold,
            job,
            handoff_arm,
            locked_other_arm_joints,
            sc2.FAST_GRASP_SETTLE_HOLD,
            grasp_runtime,
        )
        regrasp_job = sc2._make_regrasp_job(job, handoff_arm, handoff_world_xyz)
        rospy.loginfo(
            "%s: %s regrasp from handoff using %s hand",
            prefix,
            job["object"],
            handoff_arm,
        )
        locked_other_arm_joints = sc2._fixed_work_pose_other_arm_joints(handoff_arm)
        grasp_ok = sc2._pick_part_absolute(
            arm_pub,
            arm_hold,
            gripper_hold,
            regrasp_job,
            locked_other_arm_joints,
            sc2.FAST_GRASP_SETTLE_HOLD,
            grasp_runtime,
        )
        if not grasp_ok:
            rospy.logerr("%s: handoff regrasp IK failed for %s", prefix, job["object"])
            raise RuntimeError("handoff regrasp IK failed for {}".format(job["object"]))
        rospy.loginfo("%s: %s handoff regrasp succeeded, placing", prefix, job["object"])
        _scene2_place_part_absolute(
            sc2,
            arm_pub,
            arm_hold,
            gripper_hold,
            regrasp_job,
            locked_other_arm_joints,
            sc2.FAST_GRASP_SETTLE_HOLD,
            grasp_runtime,
        )
        place_ok = _scene2_verify_recent_place(regrasp_job, camera, output_dir, target_frame)
        rospy.loginfo(
            "%s: %s handoff+place completed, place_verified=%s",
            prefix,
            job["object"],
            place_ok,
        )
        return regrasp_job

    rospy.loginfo("%s: %s placing directly to %s", prefix, job["object"], job["bin"])
    _scene2_place_part_absolute(
        sc2,
        arm_pub,
        arm_hold,
        gripper_hold,
        job,
        locked_other_arm_joints,
        sc2.FAST_GRASP_SETTLE_HOLD,
        grasp_runtime,
    )
    place_ok = _scene2_verify_recent_place(job, camera, output_dir, target_frame)
    rospy.loginfo(
        "%s: %s direct place completed, place_verified=%s",
        prefix,
        job["object"],
        place_ok,
    )
    return job


def _scene2_recover_table_leftovers(sc2, arm_pub, arm_hold, gripper_hold,
                                    grasp_runtime, camera, output_dir, target_frame,
                                    max_rounds=3):
    """恢复处理：重新分拣源桌面遗留物和错误放置的零件。
    
    参数:
        sc2:            scene2_data_collection_pipeline 模块
        arm_pub:        手臂控制 Publisher
        arm_hold:       手臂状态保持器
        gripper_hold:   夹爪状态保持器
        grasp_runtime:  GraspRuntime 实例
        camera:         相机名称
        output_dir:     输出目录
        target_frame:   目标坐标系
        max_rounds:     最大恢复轮次
    
    返回:
        bool: 是否清空
    """
    import rospy

    for recovery_round in range(1, int(max_rounds) + 1):
        leftovers, misplaced, _bin_hits = _scene2_capture_table_issues(
            camera,
            output_dir,
            target_frame,
        )
        if not leftovers and not misplaced:
            rospy.loginfo("scene2 final recovery: no source-table leftovers or wrong-bin detections before round %d", recovery_round)
            return True

        recovery_jobs = _scene2_build_recovery_jobs(sc2, leftovers, misplaced, recovery_round)
        if not recovery_jobs:
            raise RuntimeError(
                "scene2 final recovery found issues but no recoverable jobs: leftovers={} misplaced={}".format(
                    [_scene2_detection_summary(det) for det in leftovers],
                    ["{}<-{}".format(bin_name, _scene2_detection_summary(det))
                     for bin_name, det in misplaced],
                )
            )
        rospy.logwarn(
            "scene2 final recovery round %d/%d: rebuilding jobs for leftovers=%s misplaced=%s",
            recovery_round,
            max_rounds,
            [_scene2_detection_summary(det) for det in leftovers],
            ["{}<-{}".format(bin_name, _scene2_detection_summary(det))
             for bin_name, det in misplaced],
        )
        sc2._move_to_work_pose_joints(arm_pub, arm_hold)
        round_had_error = False
        for index, job in enumerate(recovery_jobs, start=1):
            try:
                _scene2_execute_sort_job(
                    sc2,
                    arm_pub,
                    arm_hold,
                    gripper_hold,
                    job,
                    grasp_runtime,
                    camera,
                    output_dir,
                    target_frame,
                    index=index,
                    total=len(recovery_jobs),
                )
            except Exception as exc:
                round_had_error = True
                rospy.logwarn(
                    "scene2 final recovery round %d: %s failed (%s); will re-inspect and rebuild targets",
                    recovery_round,
                    job["object"],
                    exc,
                )
                try:
                    sc2._publish_gripper_open(gripper_hold)
                    sc2._move_to_work_pose_joints(arm_pub, arm_hold)
                except Exception as reset_exc:
                    rospy.logwarn("scene2 final recovery reset after failure also failed: %s", reset_exc)
                    break
                # 一个失败的恢复目标不能阻止本轮中剩余检测到的物体
                # 获得恢复尝试。下一轮会从相机观测重新定位所有物体，
                # 并移除已放置的物体。
                continue
        rospy.sleep(0.7)
        if round_had_error:
            continue

    return False


def _run_scene2_jobs(sc2, gripper_hold, arm_hold, jobs, camera, output_dir, target_frame):
    """运行场景二分拣任务列表（含最终检查和恢复）。
    
    参数:
        sc2:          scene2_data_collection_pipeline 模块
        gripper_hold: 夹爪状态保持器
        arm_hold:     手臂状态保持器
        jobs:         任务列表
        camera:       相机名称
        output_dir:   输出目录
        target_frame: 目标坐标系
    """
    import rospy
    from kuavo_msgs.msg import armTargetPoses

    arm_pub = rospy.Publisher(sc2.ARM_TARGET_POSES_TOPIC, armTargetPoses, queue_size=10)
    sc2._wait_for_connection(arm_pub, sc2.TOPIC_TIMEOUT)

    grasp_runtime = sc2.GraspRuntime(
        world_to_ee_offset_x=sc2.WORLD_TO_EE_OFFSET_X,
        world_to_ee_offset_y_left=sc2.WORLD_TO_EE_OFFSET_Y_LEFT,
        world_to_ee_offset_y_right=sc2.WORLD_TO_EE_OFFSET_Y_RIGHT,
        world_to_ee_offset_z=sc2.WORLD_TO_EE_OFFSET_Z,
        pre_grasp_z_offset=sc2.PRE_GRASP_APPROACH_Z_OFFSET,
        grasp_position_tolerance=sc2.GRASP_POSITION_TOLERANCE,
        orientation_tolerance_rad=sc2.ORIENTATION_TOLERANCE_RAD,
        gripper_close_time=sc2.GRIPPER_CLOSE_TIME,
        timeout=sc2.TOPIC_TIMEOUT,
        move_time=sc2.ARM_MOVE_TIME,
        settle_time=sc2.ARM_SETTLE_TIME,
        ik_mode_pos_hard_ori_hard=sc2.IK_MODE_THREE_POINT_MIXED,
        read_current_arm_joints_cb=lambda: sc2._read_current_arm_joints(sc2.TOPIC_TIMEOUT),
        execute_arm_motion_cb=lambda start_degrees, target_degrees, move_time, settle: sc2._execute_arm_motion(
            arm_pub,
            arm_hold,
            start_degrees,
            target_degrees,
            move_time,
            settle,
        ),
        publish_arm_gripper_close_cb=lambda arm: sc2._publish_arm_gripper_close(gripper_hold, arm),
        sleep_cb=rospy.sleep,
        loginfo_cb=rospy.loginfo,
        logwarn_cb=_scene2_safe_logwarn,
    )

    sc2._publish_gripper_open(gripper_hold)
    sc2._enter_work_pose(arm_pub, arm_hold, grasp_runtime)
    failed_jobs = []
    rospy.loginfo(
        "scene2 visual sorting: pick order=%s",
        " -> ".join(job["object"] for job in jobs),
    )
    for job in jobs:
        rospy.loginfo(
            "scene2 visual sorting: %s camera=%s angle=%.1f place_area=%s selected_arm=%s grasp=%s",
            job["object"],
            job.get("perception", {}).get("base_link_xyz_m"),
            float(job.get("perception", {}).get("angle_deg") or 0.0),
            job["bin"],
            job["arm"],
            [round(v, 4) for v in job["grasp"]],
        )
    for index, job in enumerate(jobs, start=1):
        try:
            _scene2_execute_sort_job(
                sc2,
                arm_pub,
                arm_hold,
                gripper_hold,
                job,
                grasp_runtime,
                camera,
                output_dir,
                target_frame,
                index=index,
                total=len(jobs),
            )
            rospy.loginfo(
                "scene2 visual sorting: job %d/%d %s completed successfully",
                index,
                len(jobs),
                job["object"],
            )
        except Exception as exc:
            rospy.logerr(
                "scene2 visual sorting: job %d/%d %s failed: %s",
                index,
                len(jobs),
                job["object"],
                exc,
            )
            failed_jobs.append(job["object"])
    rospy.loginfo("scene2 visual sorting: all jobs finished, return work -> side lift -> final inspection")
    sc2._move_to_work_pose_joints(arm_pub, arm_hold)
    sc2._move_arm_to(arm_pub, arm_hold, sc2.SIDE_LIFT_JOINTS_DEG)
    final_error = None
    try:
        _scene2_final_table_inspection(sc2, camera, output_dir, target_frame)
        rospy.loginfo("scene2 visual sorting: final inspection passed on first attempt")
    except Exception as exc:
        final_error = exc
        rospy.logwarn("scene2 visual sorting: final inspection failed before recovery: %s", exc)
        recovered = _scene2_recover_table_leftovers(
            sc2,
            arm_pub,
            arm_hold,
            gripper_hold,
            grasp_runtime,
            camera,
            output_dir,
            target_frame,
            max_rounds=5,
        )
        if recovered:
            rospy.loginfo("scene2 visual sorting: recovery completed, re-running final inspection")
        else:
            rospy.logwarn("scene2 visual sorting: recovery did not fully clear the table")
        try:
            _scene2_final_table_inspection(sc2, camera, output_dir, target_frame)
            final_error = None
            rospy.loginfo("scene2 visual sorting: final inspection passed after recovery")
        except Exception as recovery_exc:
            final_error = recovery_exc
            rospy.logerr("scene2 visual sorting: final inspection failed even after recovery: %s", recovery_exc)
    rospy.loginfo("scene2 visual sorting: retracting to home after final inspection")
    sc2._move_arm_to(arm_pub, arm_hold, sc2.HOME_JOINTS_DEG)
    if final_error is not None:
        raise final_error
    if failed_jobs:
        raise RuntimeError("scene2 sorting failed jobs without verified placement: {}".format(", ".join(failed_jobs)))
    rospy.loginfo("scene2 visual sorting: all jobs finished, retracted to home")


def _run_scene2_baseline():
    """Run the scene2 sorting pipeline using public fallback grasp targets."""
    import rospy

    _add_helper_path("test", "collect_scene2_dataset")
    import scene2_data_collection_pipeline as sc2

    sc2._publish_head_target(sc2.TOPIC_TIMEOUT)
    gripper_hold = sc2._start_gripper_hold(sc2.TOPIC_TIMEOUT)
    arm_hold = sc2._start_arm_traj_hold(sc2.TOPIC_TIMEOUT)
    arm_mode_changed = False
    try:
        sc2._set_arm_mode(sc2.ARM_MODE_EXTERNAL_CONTROL, timeout=sc2.TOPIC_TIMEOUT)
        arm_mode_changed = True
        rospy.loginfo("scene2 baseline: arm mode set to external control, running sorting pipeline")
        sc2._run_sorting_pipeline(gripper_hold, arm_hold, recorder=None)
        rospy.loginfo("scene2 baseline: sorting pipeline completed successfully")
    finally:
        if arm_mode_changed:
            try:
                sc2._set_arm_mode(sc2.ARM_MODE_AUTO_SWING, timeout=sc2.TOPIC_TIMEOUT)
                rospy.loginfo("scene2 baseline: arm mode restored to auto swing")
            except Exception as exc:
                rospy.logwarn("scene2: failed to restore arm mode: %s", exc)
        arm_hold.stop()
        gripper_hold.stop()


def _run_scene2_visual_sorting(camera="head", output_dir="scene2_perception", target_frame="base_link", display=False):
    """场景二视觉分拣主流程。
    
    1. 初始化相机和参数
    2. 检测零件并构建分拣任务
    3. 执行分拣并进行最终检查和恢复
    
    参数:
        camera:       相机名称
        output_dir:   输出目录
        target_frame: 目标坐标系
    """
    import rospy

    _add_helper_path("test", "collect_scene2_dataset")
    import scene2_data_collection_pipeline as sc2
    _scene2_slow_down_motion(sc2)

    detections = []
    valid_coordinate_count = 0
    for attempt in range(1, 6):
        perception = _run_scene2_perception_debug(
            camera,
            output_dir,
            target_frame,
            print_result=False,
            display=display,
        )
        detections = perception.get("detections") or []
        valid_coordinate_count = sum(
            isinstance(det.get("base_link_xyz_m"), (list, tuple))
            and len(det["base_link_xyz_m"]) == 3
            and all(math.isfinite(float(value)) for value in det["base_link_xyz_m"])
            for det in detections
        )
        if len(detections) == 6 and valid_coordinate_count == 6:
            rospy.loginfo("scene2 visual sorting: initial perception passed on attempt %d", attempt)
            break
        rospy.logwarn(
            "scene2 visual sorting: initial perception attempt %d expected 6 detections "
            "with base_link coordinates, got detections=%d coordinates=%d",
            attempt,
            len(detections),
            valid_coordinate_count,
        )
        rospy.sleep(0.4)
    if len(detections) != 6 or valid_coordinate_count != 6:
        raise RuntimeError(
            "scene2 visual sorting: initial perception requires 6 detections with "
            "base_link_xyz_m, got detections={} coordinates={}".format(
                len(detections), valid_coordinate_count))
    rospy.loginfo("scene2 visual sorting: initial perception succeeded with %d detections", len(detections))

    jobs = _scene2_build_visual_jobs(sc2, detections)
    rospy.loginfo("scene2 visual sorting: built %d visual jobs", len(jobs))
    gripper_hold = sc2._start_gripper_hold(sc2.TOPIC_TIMEOUT)
    arm_hold = sc2._start_arm_traj_hold(sc2.TOPIC_TIMEOUT)
    arm_mode_changed = False
    try:
        sc2._set_arm_mode(sc2.ARM_MODE_EXTERNAL_CONTROL, timeout=sc2.TOPIC_TIMEOUT)
        arm_mode_changed = True
        rospy.loginfo("scene2 visual sorting: arm mode set to external control, starting job execution")
        _run_scene2_jobs(sc2, gripper_hold, arm_hold, jobs, camera, output_dir, target_frame)
        rospy.loginfo("scene2 visual sorting: all jobs completed successfully")
    finally:
        if arm_mode_changed:
            try:
                sc2._set_arm_mode(sc2.ARM_MODE_AUTO_SWING, timeout=sc2.TOPIC_TIMEOUT)
                rospy.loginfo("scene2 visual sorting: arm mode restored to auto swing")
            except Exception as exc:
                rospy.logwarn("scene2: failed to restore arm mode: %s", exc)
        arm_hold.stop()
        gripper_hold.stop()


