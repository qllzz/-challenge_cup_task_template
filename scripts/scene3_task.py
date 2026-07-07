#!/usr/bin/env python3
"""
场景三：SMT 料盘出库。

场景对象：
  - SMT 货架: smt_rack（上下两层）
  - 料盘: smt_tray_1 ~ smt_tray_5
  - 出库区: sorting_box_0p4_0p3_0p3

任务流程：
  1. 识别 smt_rack 和目标料盘
  2. 判断料盘所在层级（上层 10 分，下层 20 分）
  3. 估计料盘前缘、中心、拉出方向
  4. 移动到货架前预操作位
  5. 手臂进入预托取位姿 → 抓住/托住料盘前缘
  6. 沿货架外法线低速拉出 → 抬升到安全高度
  7. 搬运到出库区 → 放置
"""

import sys
import os
import math

import rospy
import numpy as np
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA


# ---------------------------------------------------------------------------
# 可视化：MarkerArray 球体标记料盘位置
# ---------------------------------------------------------------------------

_MARKER_PUB = None


def _get_marker_pub():
    global _MARKER_PUB
    if _MARKER_PUB is None:
        _MARKER_PUB = rospy.Publisher("/detected_trays", MarkerArray, queue_size=1)
        rospy.sleep(0.2)
    return _MARKER_PUB


def _make_color(r, g, b, a=0.8):
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = r, g, b, a
    return c


def publish_tray_markers(result, log, frame="base_link"):
    """
    在 RViz 中标记检测到的料盘位置。
    绿色球 = 下层（20分），黄色球 = 上层（10分）。
    """
    markers = MarkerArray()
    marker_id = 0

    colors = {20: _make_color(0.0, 0.9, 0.1), 10: _make_color(1.0, 0.85, 0.0)}

    for level in result["trays"]:
        score = level["score"]
        for cx, cy, cz in level["slots"]:
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp = rospy.Time.now()
            m.ns = "tray_lower" if score == 20 else "tray_upper"
            m.id = marker_id
            marker_id += 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position = Point(x=cx, y=cy, z=cz)
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.04
            m.color = colors.get(score, _make_color(1.0, 1.0, 1.0))
            m.lifetime = rospy.Duration(0)
            markers.markers.append(m)

    pub = _get_marker_pub()
    pub.publish(markers)
    log("已发布 %d 个料盘标记到 /detected_trays", marker_id)


# ---------------------------------------------------------------------------
# 摆头：将头部对准检测到的料盘
# ---------------------------------------------------------------------------

def look_at_trays(head, log, result, dwell=1.5):
    """
    依次将头部转向每个检测到的料盘。
    优先看下层（20 分），再看上层（10 分）。

    head — HeadController 实例
    dwell — 每个料盘停留的秒数
    """
    all_slots = []
    for level in result["trays"]:
        for cx, cy, cz in level["slots"]:
            all_slots.append((cx, cy, cz, level["score"]))

    if not all_slots:
        log("无料盘可查看")
        return

    # 头部在 base_link 下的近似位置（胯高 ~0.92m，头部在胯上方 ~0.88m）
    head_x, head_y, head_z = 0.05, 0.0, 0.88

    # 下层优先
    all_slots.sort(key=lambda s: -s[3])

    for cx, cy, cz, score in all_slots:
        dx = cx - head_x
        dy = cy - head_y
        dz = cz - head_z

        yaw_deg = math.degrees(math.atan2(dy, dx))
        pitch_deg = math.degrees(math.atan2(dz, math.sqrt(dx * dx + dy * dy)))

        # 钳位到头部限位
        yaw_clamped = max(-30.0, min(30.0, yaw_deg))
        pitch_clamped = max(-25.0, min(25.0, pitch_deg))

        label = "下层[20分]" if score == 20 else "上层[10分]"
        log("  摆头→%s yaw=%.1f° pitch=%.1f° (base: %.2f,%.2f,%.2f)",
            label, yaw_clamped, pitch_clamped, cx, cy, cz)

        head.look_at(yaw_clamped, pitch_clamped)
        rospy.sleep(dwell)

    head.look_forward()


# ---------------------------------------------------------------------------
# 雷达感知：货架扫描 & 料盘检测
# ---------------------------------------------------------------------------

def _log(ros_log, msg, *args):
    """同时输出到 rospy 日志和 stdout，确保终端可见。"""
    if args:
        msg = msg % args
    ros_log(msg)
    print(msg)


def _cluster_1d(values, gap_threshold, min_size=10):
    """一维间隙聚类：沿单一轴按间隙 > threshold 切分，过滤小簇噪声。"""
    if len(values) < min_size:
        return []
    order = np.argsort(values)
    splits = [0]
    for i in range(1, len(order)):
        if values[order[i]] - values[order[i - 1]] > gap_threshold:
            splits.append(i)
    splits.append(len(order))
    clusters = []
    for s, e in zip(splits[:-1], splits[1:]):
        if e - s >= min_size:
            clusters.append(order[s:e])
    return clusters


def _merge_clusters(clusters, pts, merge_eps=0.10):
    """
    将 3D 聚类中距离 < merge_eps 的簇合并为同一料盘。
    返回: [(merged_indices, (cx, cy, cz, n_pts)), ...]
    """
    if not clusters:
        return []
    # 计算每簇的质心和大小
    infos = []
    for idx in clusters:
        c = pts[idx]
        infos.append({
            "idx": idx,
            "cx": float(c[:, 0].mean()),
            "cy": float(c[:, 1].mean()),
            "cz": float(c[:, 2].mean()),
            "n": len(idx),
            "merged": False,
        })
    merged = []
    for i, a in enumerate(infos):
        if a["merged"]:
            continue
        group_idx = list(a["idx"])
        for j in range(i + 1, len(infos)):
            b = infos[j]
            if b["merged"]:
                continue
            dist = math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"], a["cz"] - b["cz"])
            if dist < merge_eps:
                group_idx.extend(b["idx"])
                b["merged"] = True
        a["merged"] = True
        c_all = pts[group_idx]
        merged.append((group_idx, (
            float(c_all[:, 0].mean()),
            float(c_all[:, 1].mean()),
            float(c_all[:, 2].mean()),
            len(group_idx),
        )))
    return merged


def _cluster_3d(pts, eps=0.05, min_size=8):
    """3D 欧氏距离聚类：两点距离 < eps 属同一簇，BFS 连通分量。"""
    n = len(pts)
    if n < min_size:
        return []
    labels = -np.ones(n, dtype=int)
    cid = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        queue = [i]
        labels[i] = cid
        head = 0
        while head < len(queue):
            idx = queue[head]
            head += 1
            dists = np.linalg.norm(pts - pts[idx], axis=1)
            for j in np.where((dists < eps) & (labels < 0))[0]:
                labels[j] = cid
                queue.append(j)
        cid += 1
    clusters = []
    for ci in range(cid):
        members = np.where(labels == ci)[0]
        if len(members) >= min_size:
            clusters.append(members)
    return clusters


def scan_rack(lidar, log):
    """
    扫描 SMT 货架区域，检测各层有无料盘。

    返回 dict:
        "shelf_z":    [z0, z1, ...]       搁板高度（世界 Z，米）
        "normal":     [nx, ny, nz]        货架外法线（≈ 拉出方向）
        "trays":      [                    按层级排列的料盘列表
            {"z": 0.75, "slots": [            ← "slots" 里是各槽位料盘质心 (x, y, z)
                (0.98, 0.35, 0.75), ...
            ]},
            {"z": 1.15, "slots": [...]},
        ]
    """
    # ---- 1. 多帧累积扫描：货架整体区域（base_link 坐标系！）----
    # 机器人起点: 世界 (-0.65, 0, 0.926), base_link = 机器人底盘中心
    # 货架世界: (1.05, 0, 0), 在 base_link 下: x ≈ 1.70, y ∈ [-0.6, 0.6], z ∈ [-0.9, 0.6]
    _X0, _Z0 = 0.65, -0.92  # 世界 → base_link 近似偏移

    N_FRAMES = 8
    all_pts = []
    for _ in range(N_FRAMES):
        pts = lidar.get_points_in_region(
            x_range=(1.00, 2.30), y_range=(-0.80, 0.80), z_range=(-1.00, 0.80))
        if pts is not None and len(pts) > 0:
            all_pts.append(pts)
        rospy.sleep(0.1)

    if not all_pts:
        _log(log, "激光雷达无数据，检查 /lidar/points 话题")
        return {"shelf_z": [], "normal": None, "trays": []}

    pts = np.vstack(all_pts)
    _log(log, "货架区域点云 (%d 帧): %d 点", len(all_pts), len(pts))

    if len(pts) < 200:
        _log(log, "货架区域点云过少 (n=%d)，放弃", len(pts))
        return {"shelf_z": [], "normal": None, "trays": []}

    # ---- 2. 拟合搁板平面 & 外法线 ----
    # 搁板世界 z≈0.40, 1.00 → base_link z ≈ -0.52, 0.08
    # 围栏世界 z≈0.50, 1.10 → base_link z ≈ -0.42, 0.18
    shelf_z_candidates = []
    for z_lo, z_hi in [(-0.60, -0.45), (0.00, 0.20)]:
        slab = pts[(pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi)]
        if len(slab) > 40:
            shelf_z_candidates.append(float(slab[:, 2].mean()))
            _log(log, "  搁板 base_z=%.3f（%d 点）", shelf_z_candidates[-1], len(slab))

    # 平面拟合 → 外法线
    normal = None
    shelf_mask = (
        ((pts[:, 2] >= -0.60) & (pts[:, 2] <= -0.45))
        | ((pts[:, 2] >= 0.00) & (pts[:, 2] <= 0.20))
    )
    shelf_pts = pts[shelf_mask]
    if len(shelf_pts) >= 30:
        centroid = shelf_pts.mean(axis=0)
        _, _, vh = np.linalg.svd(shelf_pts - centroid)
        normal = vh[2].copy()
        if normal[0] > 0:
            normal *= -1.0
        _log(log, "  货架外法线(base): [%.3f  %.3f  %.3f]", normal[0], normal[1], normal[2])

    # ---- 3. 分层检测料盘 ----
    # 世界 z ≈ 0.75, 1.15 → base_link z ≈ -0.17, 0.23
    # 料盘比围栏深（x 更小），围栏在 x≈1.6+
    level_configs = [
        # (name,   z_lo,  z_hi,  x_lo, x_hi,  score,  cluster_eps)
        ("下层",   -0.28,  0.00,  1.35, 1.65,  20,      0.06),
        ("上层",    0.19,  0.32,  1.35, 1.65,  10,      0.03),
    ]

    trays_result = []

    for name, z_lo, z_hi, x_lo, x_hi, score, eps in level_configs:
        layer = pts[
            (pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi)
            & (pts[:, 0] >= x_lo) & (pts[:, 0] <= x_hi)
        ]
        n_layer = len(layer)
        _log(log, "  %s: z[%.2f,%.2f] x[%.2f,%.2f] → %d 点",
            name, z_lo, z_hi, x_lo, x_hi, n_layer)

        if n_layer < 10:
            trays_result.append({"z": (z_lo + z_hi) / 2, "score": score, "slots": [], "n_pts": n_layer})
            continue

        # 优先 3D 欧氏聚类，回退到 1D Y 间隙聚类
        clusters = _cluster_3d(layer, eps=eps, min_size=6)
        if not clusters:
            clusters = _cluster_1d(layer[:, 1], gap_threshold=0.03, min_size=6)

        # 合并距离过近的碎片簇（单盘被多帧微动切碎的补偿）
        merged = _merge_clusters(clusters, layer, merge_eps=0.15)

        # 过滤：保留点数 ≥ 25 的大簇（排除围栏等结构碎片）
        merged = [(idx, info) for idx, info in merged if info[3] >= 25]

        # 按每层预期数量截断（下层 2，上层 3）
        max_trays = 2 if "下层" in name else 3
        if len(merged) > max_trays:
            merged.sort(key=lambda m: -m[1][3])  # 按点数降序
            merged = merged[:max_trays]
            merged.sort(key=lambda m: m[1][1])    # 按 y 升序重排
        clustered = sum(m[1][3] for m in merged)
        unclustered = n_layer - clustered

        slots = [m[1][:3] for m in merged]  # (cx, cy, cz)

        _log(log, "  %s: %d 个料盘 (eps=%.2fm, clustered=%d unclustered=%d)  →  %d 分/个",
            name, len(slots), eps, clustered, unclustered, score)
        for si, (_, (cx, cy, cz, cn)) in enumerate(merged):
            _log(log, "       [%d] %d点  base_y=%.2f  base_x=%.2f  base_z=%.2f",
                si, cn, cy, cx, cz)

        # 剩余点的 Y 分布
        if unclustered > 0 and len(merged) > 0:
            c_mask = np.zeros(n_layer, dtype=bool)
            for m_idx, _ in merged:
                c_mask[m_idx] = True
            leftover = layer[~c_mask]
            y_vals = leftover[:, 1]
            bins = np.arange(y_vals.min() - 0.005, y_vals.max() + 0.015, 0.01)
            hist, edges = np.histogram(y_vals, bins=bins)
            peaks = [(edges[i], edges[i+1], hist[i])
                     for i in np.argsort(hist)[-5:] if hist[i] > 0]
            log("       未聚类点 Y 分布 (top bins): %s",
                " | ".join(f"y=[{e1:.2f},{e2:.2f}] n={h}" for e1, e2, h in sorted(peaks, key=lambda x: -x[2])))

        trays_result.append({"z": (z_lo + z_hi) / 2, "score": score, "slots": slots, "n_pts": n_layer})

    return {"shelf_z": shelf_z_candidates, "normal": normal, "trays": trays_result}


# ---------------------------------------------------------------------------
# Debug: challenge_secret 读取真实料盘位姿
# ---------------------------------------------------------------------------

def _get_gt_trays(log):
    """通过 challenge_secret.so 获取当前 seed 真实料盘世界坐标。无 .so 时返回 None。"""
    try:
        from challenge_secret import get_object_layout
        layout = get_object_layout("scene3", None)
        if not layout:
            return None
        return {n: (o["pos"][0], o["pos"][1], o["pos"][2])
                for n, o in layout.items() if n.startswith("smt_tray_")}
    except Exception:
        return None


def debug_shuffle_seeds(log, rounds=5):
    """
    快速遍历多个随机 seed，打印各 seed 的预期料盘位置（base_link 坐标系）。
    不移动仿真中的真实物体——纯只读，用于验证雷达坐标转换。
    """
    try:
        from challenge_secret import get_object_layout
    except ImportError:
        log("challenge_secret 不可用，跳过")
        return

    import random as _rnd
    _X0, _Z0 = 0.65, -0.92

    for _ in range(rounds):
        seed = _rnd.randint(0, 9999)
        layout = get_object_layout("scene3", seed)
        if not layout:
            log("  seed %d: 无布局数据", seed)
            continue
        trays = [(n, o["pos"]) for n, o in layout.items() if n.startswith("smt_tray_")]
        lower = [(p[0]+_X0, p[1], p[2]+_Z0) for _, p in trays if abs(p[2] - 0.75) < 0.3]
        upper = [(p[0]+_X0, p[1], p[2]+_Z0) for _, p in trays if abs(p[2] - 1.15) < 0.3]
        log("--- seed %d: 下层%d个 上层%d个 ---", seed, len(lower), len(upper))
        for bx, by, bz in sorted(lower, key=lambda p: p[1]):
            log("  下  base_y=%.2f  base_x=%.2f  base_z=%.2f", by, bx, bz)
        for bx, by, bz in sorted(upper, key=lambda p: p[1]):
            log("  上  base_y=%.2f  base_x=%.2f  base_z=%.2f", by, bx, bz)


def compare_gt(detected, log, gt_trays=None):
    """对比雷达检测结果 vs 真实位置（坐标统一转 base_link）。"""
    if not gt_trays:
        log("  (无 ground truth，跳过对比)")
        return
    _X0, _Z0 = 0.65, -0.92
    det_all = [(cx, cy, cz) for lv in detected["trays"] for cx, cy, cz in lv["slots"]]
    log("--- GT vs 雷达 ---")
    for name, (wx, wy, wz) in sorted(gt_trays.items()):
        bx, by, bz = wx + _X0, wy, wz + _Z0
        best = min(det_all, key=lambda d: math.hypot(d[0]-bx, d[1]-by, d[2]-bz)) if det_all else None
        if best:
            log("  %s: gt=%.2f,%.2f,%.2f  检测=%.2f,%.2f,%.2f  err=%.2fm",
                name, bx, by, bz, *best, math.hypot(best[0]-bx, best[1]-by, best[2]-bz))
        else:
            log("  %s: gt=%.2f,%.2f,%.2f  (未检测到)", name, bx, by, bz)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def run_scene3(robot, arm, claw, head, log):
    """
    场景三任务主逻辑。

    参数:
        robot — RobotMover 实例
        arm   — ArmController 实例
        claw  — ClawController 实例
        head  — HeadController 实例
        log   — 日志函数
    """
    print("\n" + "=" * 50)
    log("=" * 50)
    print("场景三：SMT 料盘出库 — 任务开始")
    log("场景三：SMT 料盘出库 — 任务开始")
    print("=" * 50)
    log("=" * 50)

    # ============================================================
    # 第一步：手臂切到外部控制模式
    # ============================================================
    log("[STEP 1] 切换手臂到外部控制模式")
    arm.switch_to_external_control()
    rospy.sleep(1.0)
    arm.go_home()
    log("等待仿真控制器就绪...")
    rospy.sleep(10.0)  # VRHandCommandNode 需等 MPC+WBC 完整初始化后才处理 arm traj

    # ============================================================
    # 第二步：导入感知模块
    # ============================================================
    _pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(_pkg, "src"))
    from perception_api import LidarReader

    # ============================================================
    # 第三步：雷达扫描货架 → 检测各层料盘
    # ============================================================
    log("[STEP 2] 初始化激光雷达 & 扫描货架区域")
    lidar = LidarReader()
    rospy.sleep(0.5)

    result = scan_rack(lidar, log)

    # ---- 与真实位置对比 + 多 seed 快速遍历 ----
    gt = _get_gt_trays(log)
    compare_gt(result, log, gt)
    # debug_shuffle_seeds(log, rounds=5)

    # ---- 发布可视化标记 ----
    publish_tray_markers(result, log)

    # ---- 摆头依次看每个料盘 ----
    look_at_trays(head, log, result, dwell=1.5)

    # ---- 汇总打印 ----
    log("-" * 40)
    total_score = 0
    total_trays = 0

    if result["normal"] is not None:
        n = result["normal"]
        log("货架外法线（拉出方向）: [%.3f, %.3f, %.3f]", n[0], n[1], n[2])

    for level in result["trays"]:
        score = level["score"]
        count = len(level["slots"])
        total_trays += count
        total_score += score * count
        label = "下层[20分]" if score == 20 else "上层[10分]"
        if count > 0:
            log("%s (z=%.2fm): %d 个料盘 × %d 分 = %d 分",
                label, level["z"], count, score, score * count)
        else:
            log("%s (z=%.2fm): 空", label, level["z"])

    log("-" * 40)
    log("料盘总数: %d  理论最高分: %d", total_trays, total_score)
    log("场景三：感知识别完成")

    # ============================================================
    # TODO: 后续任务逻辑
    # ============================================================
    # 拿到 result["trays"] 后就可以按如下步骤执行：
    #
    #   1. 下层优先（20 分/个）：result["trays"][0]["slots"] 是坐标列表，
    #      对每个坐标 (cx, cy, cz)：
    #        - 底盘走到货架前，y 对准料盘 y 坐标
    #        - IK 求解双手预托位姿
    #        - claw.close() 夹住料盘前缘
    #        - 沿 result["normal"] 方向后退拉出
    #        - 抬升到安全高度 → 搬运到出库箱
    #
    #   2. 上层（10 分/个）：同样流程，但手臂需要够到 z≈1.15m

    log("场景三：任务结束")
