#!/usr/bin/env python3
"""
场景二 RGB-D 感知模块。

本模块使用头部/腕部相机进行零件检测，通过颜色分割和深度投影
识别三种零件类型（type_a 银色管接头、type_b 暗色 T 型接头、type_c 红色螺丝刀手柄），
并输出它们在 base_link 坐标系下的 3D 位置。

核心流程：
    1. 从 ROS 话题获取 RGB 压缩图像、深度压缩图像和相机内参
    2. 解码图像并生成 HSV 颜色掩码
    3. 提取轮廓并过滤形状、面积、深度
    4. 使用相机内参和 TF 变换将像素坐标转换为 base_link 坐标
    5. 排除螺丝刀白色轴杆（误检为 type_a）的干扰
    6. 按类别选择前 k 个检测结果并输出
"""

import argparse
import json
import math
import os
import time

import cv2
import numpy as np
import rospy
import tf
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo, CompressedImage


CAMERA_TOPICS = {
    "head": {
        "color": "/cam_h/color/image_raw/compressed",
        "depth": "/cam_h/depth/image_raw/compressedDepth",
        "info": "/cam_h/color/camera_info",
    },
    "left": {
        "color": "/cam_l/color/image_raw/compressed",
        "depth": "/cam_l/depth/image_rect_raw/compressedDepth",
        "info": "/cam_l/color/camera_info",
    },
    "right": {
        "color": "/cam_r/color/image_raw/compressed",
        "depth": "/cam_r/depth/image_rect_raw/compressedDepth",
        "info": "/cam_r/color/camera_info",
    },
}

CLASS_LABELS = {
    "part_type_a": "silver pipe fastener",
    "part_type_b": "dark T junction",
    "part_type_c": "red screwdriver handle",
}

CLASS_MAX_COUNT = {
    "part_type_a": 2,
    "part_type_b": 2,
    "part_type_c": 2,
}


def _decode_color(msg):
    """解码压缩的 RGB 彩色图像。
    
    参数:
        msg: sensor_msgs/CompressedImage 消息
    
    返回:
        np.ndarray: BGR 格式的彩色图像 (H, W, 3)
    
    异常:
        RuntimeError: 解码失败
    """
    data = np.frombuffer(msg.data, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("failed to decode compressed color image")
    return image


def _find_png_payload(data):
    """在字节流中查找 PNG 签名并返回从签名开始的原始数据。
    
    参数:
        data: 原始字节数据
    
    返回:
        bytes: 从 PNG 签名开始的字节数据
    """
    signature = b"\x89PNG\r\n\x1a\n"
    offset = data.find(signature)
    if offset < 0:
        return data
    return data[offset:]


def _decode_depth(msg):
    """解码压缩的深度图像。
    
    处理 PNG 格式的压缩深度图，自动检测并转换单位：
    - 如果中位数 > 20（通常为毫米），则转换为米
    
    参数:
        msg: sensor_msgs/CompressedImage 消息
    
    返回:
        np.ndarray: float32 深度图像（米）
    
    异常:
        RuntimeError: 解码失败
    """
    payload = _find_png_payload(bytes(msg.data))
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError("failed to decode compressed depth image")

    depth = image.astype(np.float32)
    if image.dtype == np.uint16:
        finite = depth[depth > 0]
        if finite.size and float(np.nanmedian(finite)) > 20.0:
            depth *= 0.001
    return depth


def _camera_model(info):
    """从 CameraInfo 消息中提取相机内参模型。
    
    参数:
        info: sensor_msgs/CameraInfo 消息
    
    返回:
        dict: 包含 fx, fy, cx, cy, frame 的相机模型
    
    异常:
        RuntimeError: 内参矩阵 K 无效
    """
    if len(info.K) < 6:
        raise RuntimeError("camera_info K matrix is invalid")
    return {
        "fx": float(info.K[0]),
        "fy": float(info.K[4]),
        "cx": float(info.K[2]),
        "cy": float(info.K[5]),
        "frame": info.header.frame_id,
    }


def _depth_at(depth, u, v, radius=4):
    """在深度图中 (u,v) 像素附近取中位数深度值。
    
    参数:
        depth:  深度图像 (H, W)
        u, v:   像素坐标
        radius: 采样窗口半径（像素）
    
    返回:
        float 或 None: 该区域的中位数深度值
    """
    h, w = depth.shape[:2]
    x0 = max(0, int(round(u)) - radius)
    x1 = min(w, int(round(u)) + radius + 1)
    y0 = max(0, int(round(v)) - radius)
    y1 = min(h, int(round(v)) + radius + 1)
    patch = depth[y0:y1, x0:x1].astype(np.float32)
    finite = patch[np.isfinite(patch) & (patch > 0.02) & (patch < 10.0)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


def _depth_for_contour(depth, contour, u, v, min_depth, max_depth):
    """获取轮廓区域的深度值，使用三级回退策略。
    
    1. 优先使用轮廓中心点的深度
    2. 如果中心点深度不可用，使用轮廓内有效像素的中位数
    3. 如果轮廓内深度无效（如红色材质），使用轮廓外扩区域的深度
    
    参数:
        depth:     深度图像
        contour:   OpenCV 轮廓
        u, v:      轮廓中心像素坐标
        min_depth: 最小有效深度（米）
        max_depth: 最大有效深度（米）
    
    返回:
        tuple: (depth_m, sampling_method) 或 (None, None)
    """
    center_depth = _depth_at(depth, u, v)
    if center_depth is not None and min_depth <= center_depth <= max_depth:
        return center_depth, "center"

    if depth is None or depth.ndim != 2:
        return None, None
    contour_mask = np.zeros(depth.shape[:2], dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
    samples = depth[contour_mask > 0].astype(np.float32)
    samples = samples[np.isfinite(samples)]
    samples = samples[(samples >= float(min_depth)) & (samples <= float(max_depth))]
    if samples.size >= 8:
        return float(np.median(samples)), "contour"

    # Some red material pixels have invalid depth in the simulator.  A small
    # halo supplies the nearby tabletop depth while preserving the RGB center.
    halo_mask = cv2.dilate(contour_mask, np.ones((9, 9), np.uint8), iterations=1)
    halo_samples = depth[halo_mask > 0].astype(np.float32)
    halo_samples = halo_samples[np.isfinite(halo_samples)]
    halo_samples = halo_samples[
        (halo_samples >= float(min_depth)) & (halo_samples <= float(max_depth))
    ]
    if halo_samples.size < 8:
        return None, None
    return float(np.median(halo_samples)), "halo"


def _pixel_to_camera(u, v, z, model):
    """将像素坐标转换为相机坐标系下的 3D 点。
    
    使用针孔相机模型的反投影公式：
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
    
    参数:
        u, v:  像素坐标
        z:     深度值（米）
        model: 相机模型字典（含 fx, fy, cx, cy）
    
    返回:
        list: [x, y, z] 相机坐标系下的 3D 坐标
    """
    x = (float(u) - model["cx"]) * float(z) / model["fx"]
    y = (float(v) - model["cy"]) * float(z) / model["fy"]
    return [x, y, float(z)]


def _transform_point(listener, point_xyz, source_frame, target_frame, stamp, timeout):
    """使用 TF 变换将点从一个相机坐标系转换到目标坐标系。

    图像、深度和 CameraInfo 的时间戳可能早于刚启动的 TF listener 的缓存。
    对抓取规划而言应使用最新的相机→base_link 变换，而不是因历史图像时间戳
    缺少变换就静默返回空坐标。
    """
    if not source_frame:
        return None

    stamped = PointStamped()
    stamped.header.frame_id = source_frame
    # 先显式请求最新变换。场景二使用的是机器人固连相机，最新外参足以用于
    # 本次抓取规划，并避免刚创建 listener 时对旧图像时间戳的外推失败。
    stamped.header.stamp = rospy.Time(0)
    stamped.point.x = float(point_xyz[0])
    stamped.point.y = float(point_xyz[1])
    stamped.point.z = float(point_xyz[2])
    try:
        listener.waitForTransform(
            target_frame, source_frame, rospy.Time(0), rospy.Duration(timeout))
        transformed = listener.transformPoint(target_frame, stamped)
        return [transformed.point.x, transformed.point.y, transformed.point.z]
    except Exception as latest_error:
        # 若最新变换暂不可用，再尝试与 CameraInfo 同一时间戳的变换；这个
        # 回退对录包回放或严格时间同步的 TF 树仍然有效。
        if stamp is not None and stamp != rospy.Time(0):
            try:
                stamped.header.stamp = stamp
                transformed = listener.transformPoint(target_frame, stamped)
                return [transformed.point.x, transformed.point.y, transformed.point.z]
            except Exception:
                pass
        rospy.logwarn_throttle(
            2.0,
            "scene2 perception: TF transform %s -> %s unavailable: %s",
            source_frame, target_frame, latest_error,
        )
        return None


def _clean_mask(mask):
    """形态学清理掩码：开运算 + 闭运算去除噪点。
    
    参数:
        mask: 二值掩码图像
    
    返回:
        np.ndarray: 清理后的掩码
    """
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _clean_thin_mask(mask):
    """清理薄型零件掩码：闭运算 + 膨胀，保留细小特征。
    
    相比 _clean_mask 更保守，避免过度腐蚀薄型零件的检测区域。
    
    参数:
        mask: 二值掩码图像
    
    返回:
        np.ndarray: 清理后的掩码
    """
    close_kernel = np.ones((5, 5), np.uint8)
    dilate_kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.dilate(mask, dilate_kernel, iterations=1)
    return mask


def _make_masks(bgr):
    """生成三种零件的颜色分割掩码。
    
    使用 HSV 颜色空间进行分割：
    - part_type_c (红色螺丝刀手柄): 红色范围（两段合并覆盖 0° 和 180° 红色）
    - part_type_b (暗色 T 型接头): 低亮度区域，排除红色
    - part_type_a (银色管接头): 低饱和度 + 中高亮度，包含绿色反射区域
    
    参数:
        bgr: BGR 彩色图像
    
    返回:
        dict: {class_name: binary_mask} 三类零件的掩码字典
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    red1 = cv2.inRange(hsv, np.array([0, 70, 45]), np.array([8, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([165, 60, 40]), np.array([179, 255, 255]))
    red = red1 | red2
    # The black screwdriver tip is much smaller than a T junction.  Area and
    # top-k filtering later remove it, while this mask keeps B robust to glare.
    dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 115, 110]))
    neutral_silver = ((s < 90) & (v > 105) & (v < 250))
    green_reflective_silver = ((h > 25) & (h < 85) & (s < 190) & (v > 110))
    silver = (neutral_silver | green_reflective_silver).astype(np.uint8) * 255
    return {
        "part_type_c": _clean_mask(red),
        "part_type_b": _clean_mask(dark & cv2.bitwise_not(red)),
        "part_type_a": _clean_thin_mask(silver & cv2.bitwise_not(red)),
    }


def _contour_angle(contour):
    """计算轮廓的最小外接矩形角度。
    
    参数:
        contour: OpenCV 轮廓
    
    返回:
        float 或 None: 角度（度），宽度 < 高度时自动 +90°
    """
    if len(contour) < 5:
        return None
    rect = cv2.minAreaRect(contour)
    (width, height) = rect[1]
    angle = float(rect[2])
    if width < height:
        angle += 90.0
    return angle


def _contour_shape_metrics(contour, bbox):
    """计算轮廓的形状度量。
    
    参数:
        contour: OpenCV 轮廓
        bbox:    边界框 (x, y, w, h)
    
    返回:
        dict: {"extent": 面积/外接矩形面积, "solidity": 面积/凸包面积}
    """
    x, y, w, h = bbox
    area = float(cv2.contourArea(contour))
    rect_area = float(max(1, w * h))
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    return {
        "extent": area / rect_area,
        "solidity": area / hull_area if hull_area > 1e-6 else 0.0,
    }


def _passes_class_filter(class_name, area, aspect, metrics):
    """检查候选区域是否通过该类别的形状过滤。
    
    每个零件类型有特定的面积、宽高比、extent 和 solidity 阈值。
    
    参数:
        class_name: 零件类型 "part_type_a"/"part_type_b"/"part_type_c"
        area:       轮廓面积（像素）
        aspect:     宽高比
        metrics:    形状度量字典（含 extent, solidity）
    
    返回:
        bool: 是否通过过滤
    """
    extent = metrics["extent"]
    solidity = metrics["solidity"]
    if class_name == "part_type_c":
        # A diagonal screwdriver handle can project to an almost square box.
        return 70.0 <= area <= 18000.0 and 0.75 <= aspect <= 14.0 and extent >= 0.18
    if class_name == "part_type_b":
        return 120.0 <= area <= 24000.0 and aspect <= 5.5 and 0.16 <= extent <= 0.92 and solidity >= 0.28
    if class_name == "part_type_a":
        return 120.0 <= area <= 26000.0 and aspect <= 7.0 and 0.12 <= extent <= 0.96 and solidity >= 0.22
    return True


def _candidate_sort_key(item):
    """候选检测结果的排序键：先按 Y 坐标，再按 X 坐标。
    
    参数:
        item: 候选检测字典
    
    返回:
        tuple: (y, x) 排序键
    """
    base_key = item.get("base_link_xyz_m")
    if base_key is not None and len(base_key) >= 2:
        return (float(base_key[1]), float(base_key[0]))
    pixel = item.get("pixel") or [0.0, 0.0]
    return (float(pixel[0]), float(pixel[1]))


def _axis_angle_distance_deg(first, second):
    """计算两个无方向轴之间的锐角距离。
    
    参数:
        first, second: 角度值（度）
    
    返回:
        float: 锐角距离（度），范围 [0, 180]
    """
    if first is None or second is None:
        return 180.0
    delta = abs(float(first) - float(second)) % 180.0
    return min(delta, 180.0 - delta)


def _screwdriver_shaft_match(a_candidate, c_candidates):
    """查找红色螺丝刀手柄对应的白色轴杆候选。
    
    螺丝刀由红色手柄（type_c）和白色轴杆（被误检为 type_a）两部分组成。
    通过像素距离（35~120px）和角度共线性（<24°）判断轴杆是否属于某手柄。
    
    参数:
        a_candidate:   type_a 候选（可能是轴杆）
        c_candidates:  type_c 候选列表（红色手柄）
    
    返回:
        dict 或 None: 匹配到的 type_c 候选，或 None
    """
    a_pixel = a_candidate.get("pixel") or []
    if len(a_pixel) < 2:
        return None
    for c_candidate in c_candidates:
        c_pixel = c_candidate.get("pixel") or []
        if len(c_pixel) < 2:
            continue
        dx = float(a_pixel[0]) - float(c_pixel[0])
        dy = float(a_pixel[1]) - float(c_pixel[1])
        pixel_distance = math.hypot(dx, dy)
        if pixel_distance < 35.0 or pixel_distance > 120.0:
            continue
        c_angle = c_candidate.get("angle_deg")
        if c_angle is None:
            continue
        shaft_axis = math.degrees(math.atan2(dy, dx))
        if _axis_angle_distance_deg(shaft_axis, c_angle) > 24.0:
            continue

        a_xyz = a_candidate.get("base_link_xyz_m")
        c_xyz = c_candidate.get("base_link_xyz_m")
        if a_xyz is not None and c_xyz is not None:
            base_distance = math.hypot(
                float(a_xyz[0]) - float(c_xyz[0]),
                float(a_xyz[1]) - float(c_xyz[1]),
            )
            if base_distance > 0.18:
                continue
        return c_candidate
    return None


def _mark_screwdriver_shafts(candidates):
    """标记属于红色螺丝刀组件的银色轴杆候选。
    
    将误检为 type_a 的螺丝刀白色轴杆标记为 suppressed，
    避免在最终检测结果中重复计数。
    
    参数:
        candidates: 候选检测列表
    
    返回:
        list: 更新后的候选列表（含 suppressed_reason 标记）
    """
    c_candidates = [item for item in candidates if item.get("class") == "part_type_c"]
    if not c_candidates:
        return candidates
    for item in candidates:
        if item.get("class") != "part_type_a":
            continue
        matched_handle = _screwdriver_shaft_match(item, c_candidates)
        if matched_handle is None:
            continue
        item["suppressed_reason"] = "screwdriver_shaft"
        item["suppressed_by_handle_pixel"] = list(matched_handle.get("pixel") or [])
    return candidates


def _select_final_detections(candidates):
    """从候选列表中选出最终的检测结果。
    
    按类别分组，每类取面积最大的前 k 个未抑制候选，
    然后按空间位置排序并分配名称（如 part_type_a_1）。
    
    参数:
        candidates: 候选检测列表
    
    返回:
        list: 最终检测结果列表，每项含 name, label, class 等字段
    """
    detections = []
    for class_name in ("part_type_a", "part_type_b", "part_type_c"):
        class_items = [
            item for item in candidates
            if item["class"] == class_name and not item.get("suppressed_reason")
        ]
        # Larger connected components are normally the real part; small dark
        # components are usually screwdriver tips or shadows.
        class_items.sort(key=lambda item: item["area_px"], reverse=True)
        selected = class_items[:CLASS_MAX_COUNT[class_name]]
        selected.sort(key=_candidate_sort_key)
        for index, item in enumerate(selected, start=1):
            det = dict(item)
            det["name"] = "{}_{}".format(class_name, index)
            det["label"] = CLASS_LABELS[class_name]
            detections.append(det)
    detections.sort(key=lambda item: item["name"])
    return detections


def _extract_candidates(color, depth, info, target_frame, min_area, max_area,
                        min_depth, max_depth, tf_timeout):
    """从图像中提取候选检测区域。
    
    核心感知流程：
    1. 生成颜色掩码（_make_masks）
    2. 提取轮廓并过滤面积/形状/深度
    3. 计算像素坐标并转换为 base_link 坐标
    4. 排除螺丝刀轴杆干扰（_mark_screwdriver_shafts）
    5. 选择最终检测结果（_select_final_detections）
    
    参数:
        color:        BGR 彩色图像
        depth:        深度图像
        info:         CameraInfo 消息
        target_frame: 目标坐标系
        min_area:     最小面积阈值
        max_area:     最大面积阈值
        min_depth:    最小深度阈值
        max_depth:    最大深度阈值
        tf_timeout:   TF 变换超时
    
    返回:
        tuple: (candidates, detections, masks)
    """
    model = _camera_model(info)
    listener = tf.TransformListener()
    # 新建 listener 的 TF 缓存起初为空。先等待一次公共的相机→目标帧
    # 变换，避免轮廓逐个处理时出现“检测数正确但坐标全为 None”。
    try:
        listener.waitForTransform(
            target_frame, model["frame"], rospy.Time(0), rospy.Duration(tf_timeout))
    except Exception as exc:
        rospy.logwarn(
            "scene2 perception: waiting for TF %s -> %s failed: %s",
            model["frame"], target_frame, exc)
    masks = _make_masks(color)
    candidates = []

    for class_name, mask in masks.items():
        contours, _hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 2 or h <= 2:
                continue
            aspect = float(max(w, h)) / float(max(1, min(w, h)))
            metrics = _contour_shape_metrics(contour, (x, y, w, h))
            if not _passes_class_filter(class_name, area, aspect, metrics):
                continue
            moments = cv2.moments(contour)
            if abs(moments["m00"]) < 1e-6:
                continue
            u = float(moments["m10"] / moments["m00"])
            v = float(moments["m01"] / moments["m00"])
            z, depth_sampling = _depth_for_contour(
                depth,
                contour,
                u,
                v,
                min_depth,
                max_depth,
            )
            if z is None:
                continue
            camera_xyz = _pixel_to_camera(u, v, z, model)
            base_xyz = _transform_point(
                listener,
                camera_xyz,
                model["frame"],
                target_frame,
                info.header.stamp,
                tf_timeout,
            )
            if base_xyz is not None:
                bx, by, _bz = [float(value) for value in base_xyz]
                if bx < 0.15 or bx > 0.55 or abs(by) > 0.55:
                    continue
            candidates.append({
                "class": class_name,
                "pixel": [round(u, 2), round(v, 2)],
                "bbox": [int(x), int(y), int(w), int(h)],
                "area_px": round(area, 1),
                "aspect": round(aspect, 2),
                "extent": round(metrics["extent"], 3),
                "solidity": round(metrics["solidity"], 3),
                "angle_deg": None if _contour_angle(contour) is None else round(_contour_angle(contour), 1),
                "depth_m": None if z is None else round(z, 4),
                "depth_sampling": depth_sampling,
                "camera_frame": model["frame"],
                "camera_xyz_m": None if camera_xyz is None else [round(float(p), 4) for p in camera_xyz],
                target_frame + "_xyz_m": None if base_xyz is None else [round(float(p), 4) for p in base_xyz],
            })

    candidates.sort(key=lambda item: (item["class"], item["pixel"][1], item["pixel"][0]))
    _mark_screwdriver_shafts(candidates)
    detections = _select_final_detections(candidates)
    return candidates, detections, masks


def _draw_overlay(color, candidates, detections):
    """在彩色图像上绘制检测结果和候选区域并保存到文件。
    
    绘制所有候选区域（细线框）和最终检测结果（粗线框），
    标注类别名称和 3D 坐标。
    
    参数:
        color:       BGR 彩色图像
        candidates:  所有候选检测列表
        detections:  最终检测结果列表
    """
    overlay = color.copy()
    colors = {
        "part_type_a": (210, 210, 210),
        "part_type_b": (40, 40, 40),
        "part_type_c": (20, 20, 220),
    }
    for item in candidates:
        x, y, w, h = item["bbox"]
        color_bgr = colors.get(item["class"], (0, 255, 255))
        if item.get("suppressed_reason") == "screwdriver_shaft":
            color_bgr = (128, 128, 128)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color_bgr, 2)
        label_name = "shaft->c" if item.get("suppressed_reason") == "screwdriver_shaft" else item["class"].replace("part_type_", "")
        label = "{} {}".format(label_name, item["pixel"])
        cv2.putText(overlay, label, (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2)
    for item in detections:
        x, y, w, h = item["bbox"]
        color_bgr = colors.get(item["class"], (0, 255, 255))
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color_bgr, 3)
        cv2.putText(overlay, item["name"], (x, min(color.shape[0] - 8, y + h + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
    return overlay


def _write_outputs(output_dir, camera, color, depth, masks, overlay, candidates, detections):
    """将感知结果写入磁盘。
    
    保存内容：
    - 彩色图像、深度图像、叠加图像
    - 各颜色掩码图像
    - 候选和最终检测结果的 JSON 文件
    
    参数:
        output_dir:  输出目录
        camera:      相机名称
        color:       BGR 彩色图像
        depth:       深度图像
        masks:       颜色掩码字典
        overlay:     叠加图像
        candidates:  候选检测列表
        detections:  最终检测结果列表
    
    返回:
        str: 输出文件前缀
    """
    os.makedirs(output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = os.path.join(output_dir, "scene2_{}_{}".format(camera, stamp))
    cv2.imwrite(prefix + "_color.jpg", color)
    cv2.imwrite(prefix + "_overlay.jpg", overlay)
    if depth is not None:
        depth_vis = depth.copy()
        finite = depth_vis[np.isfinite(depth_vis) & (depth_vis > 0)]
        if finite.size:
            lo, hi = np.percentile(finite, [3, 97])
            if hi > lo:
                depth_vis = np.clip((depth_vis - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
                cv2.imwrite(prefix + "_depth.png", depth_vis)
    for name, mask in masks.items():
        cv2.imwrite(prefix + "_mask_{}.png".format(name), mask)
    with open(prefix + "_candidates.json", "w", encoding="utf-8") as handle:
        json.dump(candidates, handle, indent=2, ensure_ascii=False)
    with open(prefix + "_detections.json", "w", encoding="utf-8") as handle:
        json.dump(detections, handle, indent=2, ensure_ascii=False)
    return prefix


_DISPLAY_LAST_TIME = 0.0


def _display_results(color, depth, masks, overlay, detections, camera):
    """使用 cv2.imshow 实时显示检测结果（每秒最多刷新一次）。

    显示四个窗口：
    - 原始彩色图像
    - 检测叠加图像（带边界框和标签）
    - 颜色掩码合成图像
    - 深度图

    参数:
        color:      BGR 彩色图像
        depth:      深度图像
        masks:      颜色掩码字典
        overlay:    叠加图像
        detections: 最终检测结果列表
        camera:     相机名称
    """
    global _DISPLAY_LAST_TIME
    now = time.time()
    if now - _DISPLAY_LAST_TIME < 1.0:
        return
    _DISPLAY_LAST_TIME = now

    display_count = len(detections)
    status_text = "{}: {} objects detected".format(camera, display_count)
    status_color = (0, 255, 0) if display_count >= 6 else (0, 165, 255)

    status_overlay = overlay.copy()
    cv2.putText(
        status_overlay,
        status_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2,
        cv2.LINE_AA,
    )

    mask_vis = None
    mask_colors = {
        "part_type_a": (210, 210, 210),
        "part_type_b": (40, 40, 40),
        "part_type_c": (20, 20, 220),
    }
    if masks:
        mask_canvas = np.zeros_like(color)
        for name, mask in masks.items():
            mask_canvas[mask > 0] = mask_colors.get(name, (255, 255, 255))
        mask_vis = cv2.addWeighted(color, 0.3, mask_canvas, 0.7, 0)

    depth_vis = None
    if depth is not None:
        finite = depth[np.isfinite(depth) & (depth > 0)]
        if finite.size:
            lo, hi = np.percentile(finite, [3, 97])
            if hi > lo:
                depth_vis = np.clip((depth - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)

    cv2.imshow("scene2 {} - original".format(camera), color)
    cv2.imshow("scene2 {} - overlay".format(camera), status_overlay)

    if mask_vis is not None:
        cv2.imshow("scene2 {} - masks".format(camera), mask_vis)
    if depth_vis is not None:
        cv2.imshow("scene2 {} - depth".format(camera), depth_vis)

    cv2.waitKey(1)


def capture_once(args):
    """单次 RGB-D 感知采集。
    
    从指定相机获取一帧图像并运行完整感知流程。
    
    参数:
        args: argparse.Namespace，包含 camera, target_frame, min_area, max_area,
              min_depth, max_depth, tf_timeout, timeout, output_dir, print_result, display
    
    返回:
        dict: {"detections": [...], "candidates": [...], "prefix": str}
    """
    topics = CAMERA_TOPICS[args.camera]
    color_msg = rospy.wait_for_message(topics["color"], CompressedImage, timeout=args.timeout)
    depth_msg = rospy.wait_for_message(topics["depth"], CompressedImage, timeout=args.timeout)
    info_msg = rospy.wait_for_message(topics["info"], CameraInfo, timeout=args.timeout)
    color = _decode_color(color_msg)
    depth = _decode_depth(depth_msg)
    candidates, detections, masks = _extract_candidates(
        color,
        depth,
        info_msg,
        args.target_frame,
        args.min_area,
        args.max_area,
        args.min_depth,
        args.max_depth,
        args.tf_timeout,
    )
    overlay = _draw_overlay(color, candidates, detections)

    if getattr(args, "display", False):
        _display_results(color, depth, masks, overlay, detections, args.camera)

    prefix = None
    if args.output_dir:
        prefix = _write_outputs(args.output_dir, args.camera, color, depth, masks, overlay, candidates, detections)
    return {
        "camera": args.camera,
        "topics": topics,
        "output_prefix": prefix,
        "candidate_count": len(candidates),
        "detection_count": len(detections),
        "candidates": candidates,
        "detections": detections,
    }


def main():
    """场景二感知调试入口。
    
    解析命令行参数，初始化 ROS 节点，运行单次感知并输出 JSON 结果。
    可通过命令行参数调整检测阈值，方便调试不同场景。
    """
    parser = argparse.ArgumentParser(description="Scene2 RGB-D perception debug capture")
    parser.add_argument("--camera", choices=sorted(CAMERA_TOPICS), default="head")
    parser.add_argument("--target-frame", default="base_link")
    parser.add_argument("--output-dir", default="/tmp/scene2_perception")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--tf-timeout", type=float, default=0.8)
    parser.add_argument("--min-area", type=float, default=80.0)
    parser.add_argument("--max-area", type=float, default=60000.0)
    parser.add_argument("--min-depth", type=float, default=0.30,
                        help="Minimum accepted object depth in meters")
    parser.add_argument("--max-depth", type=float, default=0.90,
                        help="Maximum accepted object depth in meters")
    parser.add_argument("--display", action="store_true", default=False,
                        help="Show detection overlay with imshow")
    args = parser.parse_args()

    rospy.init_node("scene2_perception_debug", anonymous=True)
    result = capture_once(args)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()