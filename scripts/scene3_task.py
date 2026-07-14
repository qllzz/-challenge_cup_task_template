#!/usr/bin/env python3

import rospy
import numpy
import math

try:
    import cv2
except ImportError:
    cv2 = None
 
from perception_api import CameraReader, SensorReader, TFReader
from sensor_msgs.msg import CameraInfo

HEAD_CAMERA_FRAME = "Head Camera View"
LEFT_WRIST_CAMERA_FRAME = "Left Wrist Camera View"
RIGHT_WRIST_CAMERA_FRAME = "Right wrist Camera View"

SHOULDER_WIDTH = 505.4 # mm

def depthToPos(depth, fx, fy, cx, cy):
    h, w = depth.shape
    u, v = numpy.meshgrid(numpy.arange(w), numpy.arange(h))
    z = depth.astype(numpy.float32)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pos = numpy.stack((x, y, z), axis=-1)
    return pos

def quatToRotMatrix(quat):
    x, y, z, w = quat
    norm = numpy.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError("invalid zero-length quaternion")
    x /= norm
    y /= norm
    z /= norm
    w /= norm

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return numpy.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=numpy.float32)

def visualWorldVectorToBase(vectorWorldMm, worldBasis, tfReader, timeout=1.0):
    """将视觉世界系中的无原点位移向量（mm）变换为 base_link 位移（m）。"""
    _, headQuat = tfReader.lookup("base_link", HEAD_CAMERA_FRAME, timeout=timeout)
    if headQuat is None:
        return None
    try:
        rotationHeadToBase = quatToRotMatrix(headQuat)
    except ValueError:
        return None

    # 对行向量：world = head_camera @ basis，故逆变换乘 basis.T。
    vectorHeadMm = numpy.asarray(vectorWorldMm, dtype=numpy.float32) @ worldBasis.T
    return rotationHeadToBase @ (vectorHeadMm / 1000.0)

def wristCameraVectorToBase(vectorCameraMm, wristBasis, worldBasis, tfReader,
                            timeout=1.0):
    """腕相机光学系中的位移向量（mm）→ base_link 位移（m）。"""
    vectorWorldMm = numpy.asarray(vectorCameraMm, dtype=numpy.float32) @ wristBasis
    return visualWorldVectorToBase(vectorWorldMm, worldBasis, tfReader, timeout)

def getHandBasis(headBasis, tfReader, timeout=1.0):
    """
    将头部 optical 相机系下的 basis 方向向量转换到左右手 optical 相机系。

    headBasis 使用当前 scene3 约定：列向量为
    [groundTangent, groundBitangent, groundNormal]，且这些向量用
    `Head Camera View` 坐标表达。返回的左右 basis 保持同样列向量定义，
    但分别用 `Left Wrist Camera View` 和 `Right wrist Camera View` 坐标表达。

    返回:
        (leftBasis, rightBasis)，若 TF 查询失败则对应项为 None。
    """
    headBasis = numpy.asarray(headBasis, dtype=numpy.float32)

    def transformSingle(targetFrame):
        _, quat = tfReader.lookup(targetFrame, HEAD_CAMERA_FRAME, timeout=timeout)
        if quat is None:
            return None
        rotation_optical = quatToRotMatrix(quat)
        return rotation_optical @ headBasis

    leftHeadBasis = transformSingle(LEFT_WRIST_CAMERA_FRAME)
    rightHeadBasis = transformSingle(RIGHT_WRIST_CAMERA_FRAME)
    opticalFix = numpy.diag([1.0, -1.0, -1.0])
    leftBasis = None if leftHeadBasis is None else opticalFix @ leftHeadBasis
    rightBasis = None if rightHeadBasis is None else opticalFix @ rightHeadBasis
    return leftBasis, rightBasis

def getHandOrigin(headBasis, tfReader, timeout=1.0):
    """
    获取左右手相机光心在头部 optical 相机中心 basis 坐标系下的位置。

    headBasis 使用当前 scene3 约定：列向量为
    [groundTangent, groundBitangent, groundNormal]，这些轴用
    `Head Camera View` 坐标表达，坐标原点为头部相机光心。

    返回:
        (leftCenter, rightCenter)
        leftCenter/rightCenter 为 shape=(3,) 的 numpy 数组，分别表示左右手
        相机光心在该 headBasis 坐标系下的位置。若某侧 TF 查询失败，
        对应项为 None。
    """
    headBasis = numpy.asarray(headBasis, dtype=numpy.float32)

    def getSingleCenter(wristFrame):
        posHead, _ = tfReader.lookup(HEAD_CAMERA_FRAME, wristFrame, timeout=timeout)
        if posHead is None:
            return None
        posHead = numpy.asarray(posHead, dtype=numpy.float32) * 1000
        return posHead @ headBasis

    leftCenter = getSingleCenter(LEFT_WRIST_CAMERA_FRAME)
    rightCenter = getSingleCenter(RIGHT_WRIST_CAMERA_FRAME)
    return leftCenter, rightCenter

def planTrayApproach(tray_infos, basis, tf_reader, timeout=1.0):
    """
    给定所有检测到的料盘信息，计算最优站位和左右手分配，使机器人仅需
    蹲起 + 手臂微调即可同时够到上下层各一个料盘。

    参数:
        tray_infos : list of dict，每个元素来自 k-means 聚类后收集的数据：
            "layer"              : 0 (下层) 或 1 (上层)
            "y_center_world_mm"  : 视觉世界系 Y 坐标 (mm)
            "x_center_world_mm"  : 视觉世界系 X 坐标 (mm)
            "z_top_world_mm"     : 料盘顶面 Z 坐标 (mm)
        basis       : (3,3) 视觉世界系 basis 矩阵
        tf_reader   : TFReader 实例
        timeout     : TF 超时 (秒)

    返回:
        None (tf 失败或无可用配对时)
        dict:
            "trays": [
                {"layer": 0|1, "hand": "left"|"right",
                 "center_world_mm": (3,) numpy, 视觉世界系中心,
                 "aabb_world_mm": (min, max), 料盘在同一世界系下的 AABB,
            ]  共 2 个，分别对应左右手
    """
    if not tray_infos:
        return None

    # ---- 1. 按层分组，变换到 base_link ----
    trays_by_layer = {0: [], 1: []}
    for t in tray_infos:
        center_world = numpy.array([
            t["x_center_world_mm"],
            t["y_center_world_mm"],
            t["z_center_world_mm"],
        ], dtype=numpy.float32)
        trays_by_layer[t["layer"]].append({
            "layer": t["layer"],
            "center_world_mm": center_world,
            # 保留原始检测 AABB，供后续腕相机空间门限使用。
            "aabb_world_mm": t.get("aabb_world_mm"),
        })

    # 每层至少有一个料盘才可能配对
    if not trays_by_layer[0] or not trays_by_layer[1]:
        return None

    # ---- 2. 枚举跨层配对，找 |ΔY| 最接近肩宽的一对 ----
    best_pair = None
    best_score = float("inf")

    for lo in trays_by_layer[0]:
        for hi in trays_by_layer[1]:
            dy = abs(lo["center_world_mm"][1] - hi["center_world_mm"][1])
            score = abs(dy - SHOULDER_WIDTH)
            if score < best_score:
                best_score = score
                best_pair = (lo, hi)

    if best_pair is None:
        return None

    lo, hi = best_pair
    y_lo = lo["center_world_mm"][1]
    y_hi = hi["center_world_mm"][1]

    # 机器人站到两个料盘的 Y 中点
    # 分配: world 系 Y 较大的给左手，较小的给右手
    if y_lo > y_hi:
        left_tray, right_tray = hi, lo
    else:
        left_tray, right_tray = lo, hi

    return {
        "trays": [
            {**left_tray,  "hand": "left"},
            {**right_tray, "hand": "right"},
        ],
    }


def rasterizeSegment(canvas, point0, point1, basis, fx, fy, cx, cy, color=(0, 255, 0), thickness=2, near=1e-4):
    height, width = canvas.shape[:2]

    point0 = numpy.asarray(point0, dtype=numpy.float32)
    point1 = numpy.asarray(point1, dtype=numpy.float32)
    basisDelta = point1 - point0

    p0 = basis @ point0
    p1 = basis @ point1
    d = p1 - p0

    x_min = (0.0 - cx) / fx
    x_max = (width - 1.0 - cx) / fx
    y_min = (0.0 - cy) / fy
    y_max = (height - 1.0 - cy) / fy

    planes = (
        (lambda p: p[2] - near),
        (lambda p: p[0] - x_min * p[2]),
        (lambda p: x_max * p[2] - p[0]),
        (lambda p: p[1] - y_min * p[2]),
        (lambda p: y_max * p[2] - p[1]),
    )

    t0 = 0.0
    t1 = 1.0
    for plane in planes:
        f0 = plane(p0)
        f1 = plane(p1)
        df = f1 - f0

        if abs(df) < 1e-8:
            if f0 < 0.0:
                return False
            continue

        t = -f0 / df
        if df > 0.0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)

        if t0 > t1:
            return False

    p0 = p0 + t0 * d
    p1 = p0 + (t1 - t0) * d

    if p0[2] <= near or p1[2] <= near:
        return False

    q0 = (int(round(fx * p0[0] / p0[2] + cx)),
          int(round(fy * p0[1] / p0[2] + cy)))
    q1 = (int(round(fx * p1[0] / p1[2] + cx)),
          int(round(fy * p1[1] / p1[2] + cy)))

    cv2.line(canvas, q0, q1, color, thickness)
    return True

def mulFract(x, factor):
    x = x * factor
    return x - numpy.floor(x)

def getDiffBlendFactor(absDiff, s):
    return 1.0 / (1.0 + numpy.exp(absDiff * s))

def posToNormal(pos, depth):
    posWithPadding = numpy.pad(pos, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    depthWithPadding = numpy.pad(depth, ((1, 1), (1, 1)), mode='constant', constant_values=-100000)

    depthT = depthWithPadding[:-2, 1:-1]
    depthL = depthWithPadding[1:-1, :-2]
    depthR = depthWithPadding[1:-1, 2:]
    depthB = depthWithPadding[2:, 1:-1]

    depthAbsDiffT = numpy.abs(depthT - depth)
    depthAbsDiffL = numpy.abs(depthL - depth)
    depthAbsDiffR = numpy.abs(depthR - depth)
    depthAbsDiffB = numpy.abs(depthB - depth)

    horiFactor = getDiffBlendFactor(depthAbsDiffR - depthAbsDiffL, 0.1)[..., numpy.newaxis]
    vertFactor = getDiffBlendFactor(depthAbsDiffB - depthAbsDiffT, 0.1)[..., numpy.newaxis]

    posT  = posWithPadding[:-2, 1:-1] 
    posL  = posWithPadding[1:-1, :-2] 
    posR  = posWithPadding[1:-1, 2:]  
    posB  = posWithPadding[2:, 1:-1] 

    Tu = ( posR - pos ) * horiFactor + ( pos - posL ) * (1.0 - horiFactor)
    Tv = ( posB - pos ) * vertFactor + ( pos - posT ) * (1.0 - vertFactor)
    
    normal_raw = numpy.cross(Tv, Tu, axis=2)
    norm = numpy.linalg.norm(normal_raw, axis=2, keepdims=True)
    normal = normal_raw / norm
    
    return normal

def getAvgDir(vectors):
    mean = numpy.sum(vectors, axis=0)
    mean /= numpy.linalg.norm(mean)
    dots = vectors @ mean
    threshold = numpy.percentile(dots, 10)  
    mask = dots >= threshold
    final = numpy.sum(vectors[mask], axis=0)
    final /= numpy.linalg.norm(final)
    return final

def hsvKey(hsv_image, hsv_color, tolerances):
    H_MAX = 179
    S_MAX = 255
    V_MAX = 255

    hsv_color = numpy.array(hsv_color, dtype=numpy.int16)
    tol = numpy.array(tolerances, dtype=numpy.int16)

    h_center, s_center, v_center = hsv_color
    h_tol, s_tol, v_tol = tol

    s_low = numpy.clip(s_center - s_tol, 0, S_MAX).astype(numpy.uint8)
    s_high = numpy.clip(s_center + s_tol, 0, S_MAX).astype(numpy.uint8)
    v_low = numpy.clip(v_center - v_tol, 0, V_MAX).astype(numpy.uint8)
    v_high = numpy.clip(v_center + v_tol, 0, V_MAX).astype(numpy.uint8)

    mask = numpy.zeros(hsv_image.shape[:2], dtype=numpy.uint8)  
    h_low_raw = h_center - h_tol
    h_high_raw = h_center + h_tol

    if h_low_raw <= 0 and h_high_raw >= H_MAX:
        lower = numpy.array([0, s_low, v_low], dtype=numpy.uint8)
        upper = numpy.array([H_MAX, s_high, v_high], dtype=numpy.uint8)
        return cv2.inRange(hsv_image, lower, upper)

    intervals = []
    if h_low_raw >= 0 and h_high_raw <= H_MAX:
        intervals.append((h_low_raw, h_high_raw))
    else:
        if h_low_raw < 0:
            intervals.append((0, min(h_high_raw, H_MAX)))
            intervals.append((max(0, h_low_raw + H_MAX), H_MAX))
        if h_high_raw > H_MAX:
            intervals.append((max(0, h_low_raw), H_MAX))
            intervals.append((0, min(h_high_raw - H_MAX, H_MAX)))

    for h_low, h_high in intervals:
        h_low = int(numpy.clip(h_low, 0, H_MAX))
        h_high = int(numpy.clip(h_high, 0, H_MAX))
        if h_low > h_high:
            continue
        lower = numpy.array([h_low, s_low, v_low], dtype=numpy.uint8)
        upper = numpy.array([h_high, s_high, v_high], dtype=numpy.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv_image, lower, upper))

    return mask

from sklearn.cluster import DBSCAN

def maskedFuzzyMax(img, mask, channel, epsilon=0):
    valid = mask > 0
    ch_data = img[:, :, channel]
    masked_data = ch_data[valid]
    max_val = numpy.max(masked_data)
    threshold = max_val - epsilon
    condition = (ch_data >= threshold) & valid
    return numpy.where(condition)

def maskedFuzzyMin(img, mask, channel, epsilon=0):
    valid = mask > 0
    ch_data = img[:, :, channel]
    masked_data = ch_data[valid]
    min_val = numpy.min(masked_data)
    threshold = min_val + epsilon
    condition = (ch_data <= threshold) & valid
    return numpy.where(condition)

def classifyRect(a, b):
    tableArea = (0.9, 1.3)
    tableRatio = (1.45, 2.05)
    shelfArea = (0.3, 0.7)
    shelfRatio = (2.7, 3.3)
    area = a * b
    ratio = max(a, b) / min(a, b)
    if area >= tableArea[0] and area <= tableArea[1] and ratio >= tableRatio[0] and ratio <= tableRatio[1]:
        return 2 # table
    elif area >= shelfArea[0] and area <= shelfArea[1] and ratio >= shelfRatio[0] and ratio <= shelfRatio[1]:
        return 1 # shelf
    else:
        return 0 # unknown

def getAABBMask(minPoint, maxPoint, posImg):
    x = posImg[..., 0]
    y = posImg[..., 1]
    z = posImg[..., 2]
    return (x >= minPoint[0]) & (x <= maxPoint[0]) & \
           (y >= minPoint[1]) & (y <= maxPoint[1]) & \
           (z >= minPoint[2]) & (z <= maxPoint[2])

def expandAABBLeftRight(minPoint, maxPoint, yScale=5.0):
    """仅沿视觉世界系 Y（左右）轴，以中心为基准放大 AABB（单位 mm）。"""
    minPoint = numpy.asarray(minPoint, dtype=numpy.float32)
    maxPoint = numpy.asarray(maxPoint, dtype=numpy.float32)
    center = (minPoint + maxPoint) / 2.0
    halfExtent = (maxPoint - minPoint) / 2.0
    halfExtent[1] *= yScale
    return center - halfExtent, center + halfExtent

def detectTrayInWristAABB(depth, wristBasis, wristOrigin, fx, fy, cx, cy,
                           expectedCenterWorld, aabbMin, aabbMax,
                           minPixels=30):
    """
    以视觉世界系 AABB 作为 3D ROI，在腕部深度图中检测料盘。

    返回 (detection, componentMask)。detection 为空表示 ROI 中没有足够
    的有效料盘点；否则 offset_camera_mm 是 (X右, Y下, Z前)，单位 mm。
    """
    if depth is None:
        return None, None

    validDepth = numpy.isfinite(depth) & (depth > 0) & (depth < 10000)
    if numpy.count_nonzero(validDepth) < minPixels:
        return None, None

    # 对行向量：world = camera @ basis + origin。
    # wristBasis 的列是视觉世界轴在腕相机光学系中的表达。
    posCamera = depthToPos(depth, fx, fy, cx, cy)
    posWorld = posCamera @ wristBasis + wristOrigin
    roiMask = validDepth & getAABBMask(aabbMin, aabbMax, posWorld)
    if numpy.count_nonzero(roiMask) < minPixels:
        return None, roiMask

    # 架体会干扰投影 AABB 的下方区域；仅保留该 AABB 在腕部图像中
    # 上方 2/3（较小 v）的点。切线由投影后的 AABB 角点决定，不受
    # AABB 内架体深度点影响。
    aabbMin = numpy.asarray(aabbMin, dtype=numpy.float32)
    aabbMax = numpy.asarray(aabbMax, dtype=numpy.float32)
    aabbCorners = numpy.array([
        [x, y, z]
        for x in (aabbMin[0], aabbMax[0])
        for y in (aabbMin[1], aabbMax[1])
        for z in (aabbMin[2], aabbMax[2])
    ], dtype=numpy.float32)
    aabbCamera = (aabbCorners - wristOrigin) @ wristBasis.T
    visibleCorners = aabbCamera[aabbCamera[:, 2] > 1.0]
    if len(visibleCorners) == 0:
        return None, roiMask
    aabbRows = fy * visibleCorners[:, 1] / visibleCorners[:, 2] + cy
    topRow = float(numpy.min(aabbRows))
    bottomRow = float(numpy.max(aabbRows))
    upperCutRow = topRow + (bottomRow - topRow) * (1.0 / 2.0)
    pixelRows = numpy.arange(depth.shape[0], dtype=numpy.float32)[:, None]
    upperRoiMask = roiMask & (pixelRows <= upperCutRow)
    if numpy.count_nonzero(upperRoiMask) < minPixels:
        return None, upperRoiMask

    # 左右放大的 ROI 可能包含相邻料盘；按上方 2/3 的 2D 连通域拆分，
    # 再选择 3D 中心最接近头相机预测中心的目标。
    componentCount, labels, stats, _ = cv2.connectedComponentsWithStats(
        upperRoiMask.astype(numpy.uint8), connectivity=8)
    expectedCenterWorld = numpy.asarray(expectedCenterWorld, dtype=numpy.float32)
    bestLabel = -1
    bestScore = float("inf")
    for label in range(1, componentCount):
        if stats[label, cv2.CC_STAT_AREA] < minPixels:
            continue
        componentMask = labels == label
        componentCenterWorld = numpy.median(posWorld[componentMask], axis=0)
        score = numpy.linalg.norm(componentCenterWorld - expectedCenterWorld)
        if score < bestScore:
            bestScore = score
            bestLabel = label

    if bestLabel < 0:
        return None, upperRoiMask

    componentMask = labels == bestLabel
    componentCamera = posCamera[componentMask]
    centerDepthMm = float(numpy.median(componentCamera[:, 2]))
    if centerDepthMm <= 0:
        return None, componentMask

    # 上方 1/2 的点云本身不能直接取 y 中位数，否则会把"完整料盘中心"
    # 错误地偏向上方。假设筛到的是料盘上方 1/2：上段高度为完整高度的
    # 1/2，完整中心位于上段 top + 1.0 * upper_height = 上段底部。
    componentRows, componentCols = numpy.nonzero(componentMask)
    topComponentRow = float(numpy.percentile(componentRows, 2.0))
    bottomComponentRow = float(numpy.percentile(componentRows, 98.0))
    centerRow = topComponentRow + 1.0 * (bottomComponentRow - topComponentRow)
    centerCol = float(numpy.median(componentCols))
    centerCamera = numpy.array([
        (centerCol - cx) * centerDepthMm / fx,
        (centerRow - cy) * centerDepthMm / fy,
        centerDepthMm,
    ], dtype=numpy.float32)

    # 料盘面可能相对相机倾斜，中心深度不是机器人近边缘的深度。
    # 取最低 5% 深度的分位值，既贴近近边缘，又能剔除孤立错误深度点。
    nearEdgeDepthMm = float(numpy.percentile(componentCamera[:, 2], 5.0))
    u = centerCol
    v = centerRow
    return {
        "offset_camera_mm": centerCamera,
        "near_edge_depth_mm": nearEdgeDepthMm,
        "pixel_offset": numpy.array([u - cx, v - cy], dtype=numpy.float32),
        "pixel": numpy.array([u, v], dtype=numpy.float32),
        "point_count": int(numpy.count_nonzero(componentMask)),
        "upper_roi_cut_row": upperCutRow,
    }, componentMask

def markAABB(buffer, basis, fx, fy, cx, cy, minPoint, maxPoint, lineWeight = 1, color = (255, 255, 255), xColor = (0, 0, 255), yColor = (0, 255, 0), zColor = (255, 0, 0)):
    if maxPoint[0] != minPoint[0]:
        rasterizeSegment(buffer, [minPoint[0], maxPoint[1], maxPoint[2]], [maxPoint[0], maxPoint[1], maxPoint[2]], basis, fx, fy, cx, cy, color, lineWeight)
        rasterizeSegment(buffer, [minPoint[0], maxPoint[1], minPoint[2]], [maxPoint[0], maxPoint[1], minPoint[2]], basis, fx, fy, cx, cy, color, lineWeight)
        rasterizeSegment(buffer, [minPoint[0], minPoint[1], maxPoint[2]], [maxPoint[0], minPoint[1], maxPoint[2]], basis, fx, fy, cx, cy, color, lineWeight)
        rasterizeSegment(buffer, [minPoint[0], minPoint[1], minPoint[2]], [maxPoint[0], minPoint[1], minPoint[2]], basis, fx, fy, cx, cy, zColor, lineWeight)
    if maxPoint[1] != minPoint[1]:
        rasterizeSegment(buffer, [maxPoint[0], minPoint[1], maxPoint[2]], [maxPoint[0], maxPoint[1], maxPoint[2]], basis, fx, fy, cx, cy, color, lineWeight)
        rasterizeSegment(buffer, [maxPoint[0], minPoint[1], minPoint[2]], [maxPoint[0], maxPoint[1], minPoint[2]], basis, fx, fy, cx, cy, color, lineWeight)
        rasterizeSegment(buffer, [minPoint[0], minPoint[1], maxPoint[2]], [minPoint[0], maxPoint[1], maxPoint[2]], basis, fx, fy, cx, cy, color, lineWeight)
        rasterizeSegment(buffer, [minPoint[0], minPoint[1], minPoint[2]], [minPoint[0], maxPoint[1], minPoint[2]], basis, fx, fy, cx, cy, xColor, lineWeight)
    if maxPoint[2] != minPoint[2]:
        rasterizeSegment(buffer, [maxPoint[0], maxPoint[1], minPoint[2]], [maxPoint[0], maxPoint[1], maxPoint[2]], basis, fx, fy, cx, cy, color, lineWeight)
        rasterizeSegment(buffer, [maxPoint[0], minPoint[1], minPoint[2]], [maxPoint[0], minPoint[1], maxPoint[2]], basis, fx, fy, cx, cy, color, lineWeight)
        rasterizeSegment(buffer, [minPoint[0], maxPoint[1], minPoint[2]], [minPoint[0], maxPoint[1], maxPoint[2]], basis, fx, fy, cx, cy, color, lineWeight)
        rasterizeSegment(buffer, [minPoint[0], minPoint[1], minPoint[2]], [minPoint[0], minPoint[1], maxPoint[2]], basis, fx, fy, cx, cy, yColor, lineWeight)

def markTickedLine(canvas, point0, point1, basis, fx, fy, cx, cy,
                     tick_interval, long_tick_mod,
                     draw_start_tick, draw_end_tick,
                     short_tick_length, long_tick_length, longer_tick_length,
                     line_color, color_short, color_long, color_longer,
                     line_thickness=1, thick_short=1, thick_long=2, thick_longer=3,
                     near=1e-4):
    """
    在两点间绘制带刻度的线段（当刻度方向无法确定时，只画主线）。

    参数说明（基础参数与 rasterizeSegment 一致）：
        canvas      : 待绘制的图像画布 (numpy 数组)
        point0, point1 : 线段两端点的三维坐标 (list 或 numpy 数组)
        basis       : 3x3 变换矩阵（用于将点从世界坐标变换到相机坐标）
        fx, fy, cx, cy : 相机内参（焦距和主点）
        tick_interval (d)    : 短刻度间隔（从出发点起，每 d 出现一个刻度）
        long_tick_mod (x)    : 长刻度模数（每第 x 个短刻度变为长刻度）
        draw_start_tick      : 是否画出发点刻度（布尔值）
        draw_end_tick        : 是否画结束点刻度（布尔值）
        short_tick_length    : 短刻度的长度（三维空间中的实际长度）
        long_tick_length     : 长刻度的长度
        longer_tick_length   : 更长刻度的长度（用于起点和终点刻度）
        line_color           : 主线的颜色 (BGR 元组)
        color_short          : 短刻度的颜色
        color_long           : 长刻度的颜色
        color_longer         : 更长刻度的颜色
        line_thickness       : 主线的粗细
        thick_short          : 短刻度的粗细
        thick_long           : 长刻度的粗细
        thick_longer         : 更长刻度的粗细
        near                 : 近裁剪面距离，传递给 rasterizeSegment
    """
    # 转换为 numpy 数组方便计算
    p0 = numpy.asarray(point0, dtype=numpy.float32)
    p1 = numpy.asarray(point1, dtype=numpy.float32)
    delta = p1 - p0
    L = numpy.linalg.norm(delta)

    # 若线段长度几乎为零，无法绘制任何东西，直接返回
    if L < 1e-12:
        return

    # 1. 先绘制主线（无论刻度方向是否可确定，主线都要画）
    rasterizeSegment(canvas, point0, point1, basis, fx, fy, cx, cy,
                     line_color, line_thickness, near)

    # 2. 尝试计算刻度方向
    v = delta / L
    z_axis = numpy.asarray(basis[2, :], dtype=numpy.float32)
    t_dir = numpy.cross(v, [0, 0, 1])
    norm_t = numpy.linalg.norm(t_dir)

    # 若叉积失效（线方向与 Z 轴平行），无法确定刻度方向，则不画任何刻度，直接返回
    if norm_t < 1e-12:
        return

    t_dir = t_dir / norm_t  # 归一化

    # 3. 绘制内部刻度（从 tick_interval 开始，每隔 tick_interval 绘制一个）
    s = tick_interval
    idx = 1
    while s < L - 1e-9:
        p_on_line = p0 + s * v

        # 判断当前是长刻度还是短刻度
        if idx % long_tick_mod == 0:
            length = long_tick_length
            color = color_long
            thick = thick_long
        else:
            length = short_tick_length
            color = color_short
            thick = thick_short

        # 刻度两个端点：沿 t_dir 方向左右各延伸 length/2
        p_start = p_on_line - (length / 2.0) * t_dir
        p_end = p_on_line + (length / 2.0) * t_dir

        rasterizeSegment(canvas, p_start, p_end, basis, fx, fy, cx, cy,
                         color, thick, near)

        s += tick_interval
        idx += 1

    # 4. 绘制出发点刻度（仅当启用，使用更长刻度）
    if draw_start_tick:
        p_start = p0 - (longer_tick_length / 2.0) * t_dir
        p_end = p0 + (longer_tick_length / 2.0) * t_dir
        rasterizeSegment(canvas, p_start, p_end, basis, fx, fy, cx, cy,
                         color_longer, thick_longer, near)

    # 5. 绘制结束点刻度（仅当启用，使用更长刻度）
    if draw_end_tick:
        p_start = p1 - (longer_tick_length / 2.0) * t_dir
        p_end = p1 + (longer_tick_length / 2.0) * t_dir
        rasterizeSegment(canvas, p_start, p_end, basis, fx, fy, cx, cy,
                         color_longer, thick_longer, near)

def run_scene3(robot, arm, claw, head, log):
    log("=" * 50)
    log("场景三：SMT 料盘出库 — 任务开始")
    log("=" * 50)

    rospy.sleep(15.0)

    headCamInfo = rospy.wait_for_message("/cam_h/color/camera_info", CameraInfo)
    rightCamInfo = rospy.wait_for_message("/cam_r/color/camera_info", CameraInfo)
    leftCamInfo = rospy.wait_for_message("/cam_l/color/camera_info", CameraInfo)

    cam = CameraReader()
    sensor = SensorReader()
    tf = TFReader()

    headFX = headCamInfo.K[0]
    headFY = headCamInfo.K[4]
    headCX = headCamInfo.K[2]
    headCY = headCamInfo.K[5]
    rightFX = rightCamInfo.K[0]
    rightFY = rightCamInfo.K[4]
    rightCX = rightCamInfo.K[2]
    rightCY = rightCamInfo.K[5]
    leftFX = leftCamInfo.K[0]
    leftFY = leftCamInfo.K[4]
    leftCX = leftCamInfo.K[2]
    leftCY = leftCamInfo.K[5]

    rate = rospy.Rate(10.0)

    groundTangent = numpy.zeros((3,))

    startTime = sensor.get_sim_time()

    kernel33 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3), anchor=None)
    kernel55 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5), anchor=None)
    kernel77 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7), anchor=None)

    layerRemainCount = [2, 3]
    phase = "walk_to_shelf"
    initial_tangent = None
    # 料盘 AABB 与 basis 必须成对缓存。抬臂后头相机被 selfRadius 遮挡，
    # 腕相机的目标框仍应基于这份最后有效的检测快照来投影。
    approach_plan = None
    approach_basis = None
    approach_plan_locked = False
    active_grab_hand = None
    WRIST_ALIGN_TOLERANCE_MM = 8.0
    WRIST_DEPTH_TARGET_MM = 70.0
    WRIST_DEPTH_TOLERANCE_MM = 10.0
    WRIST_ALIGN_MAX_STEP_M = 0.04
    LIFT_DISTANCE_M = 0.05

    while not rospy.is_shutdown():
        objectBoundingBoxCorners = []

        depth = cam.get_head_depth()
        bgr = cam.get_head_rgb()
        Rbgr = cam.get_right_wrist_rgb()
        Lbgr = cam.get_left_wrist_rgb()
        Rdepth = cam.get_right_wrist_depth()
        Ldepth = cam.get_left_wrist_depth()
        # 本帧腕部深度实测的料盘偏移，key 为 "left" / "right"。
        wrist_tray_offsets = {}
        if bgr is None:
            log("[INFO] bgr is None")
            rate.sleep()
            continue
        if depth is None:
            log("[INFO] depth is None")
            rate.sleep()
            continue
        # farMask = ((depth <= 10000) & ~numpy.isnan(depth)).astype(numpy.uint8)
        depth[depth > 10000] = 0
        validMask = (depth != 0).astype(numpy.uint8)
        validMask = cv2.erode(validMask, kernel77, iterations=1)
        # cv2.imshow("validMask", validMask * 255)
        pos = depthToPos(depth, headFX, headFY, headCX, headCY)
        normal = posToNormal(pos, depth)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        groundMask = hsvKey(hsv, [73, 97, 110], [5, 10, 10])

        # cv2.imshow("depth", mulFract(depth, 0.01))
        # cv2.imshow("pos", mulFract(pos, 0.01))
        # cv2.imshow("normal", (normal + 1.0) / 2.0)

        groundVec = normal[groundMask > 0]
        if numpy.size(groundVec) != 0:
            groundNormal = getAvgDir(groundVec)
        groundImg = numpy.dot(normal, groundNormal)

        nonVertMask = (groundImg < 0.2).astype(numpy.uint8) * 255
        sideVec = normal[nonVertMask > 0]
        _, _, centers = cv2.kmeans(sideVec.astype(numpy.float32), 3, None, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001), 10, cv2.KMEANS_PP_CENTERS)

        alignmentThreshold = 0.95
        kernel77 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7), anchor=None)
        nonVertCount = cv2.countNonZero(nonVertMask)

        currentMaxRatio = -1
        referenceAxis = numpy.array([0.0, 0.0, 1.0])
        for i in range(3):
            center = centers[i]
            center = center / numpy.linalg.norm(center)
            vecImg = (numpy.dot(normal, center) > alignmentThreshold).astype(numpy.uint8) * 255
            vecImg = cv2.erode(vecImg, kernel77, iterations=1)
            # cv2.imshow(f"sideRef_{i}",  vecImg)
            correctCount = cv2.countNonZero(vecImg)
            ratio = correctCount / nonVertCount
            if ratio > currentMaxRatio:
                currentMaxRatio = ratio
                referenceAxis = center

        sideReferenceAxis = numpy.cross(groundNormal, referenceAxis)
        sideReferenceAxis = sideReferenceAxis / numpy.linalg.norm(sideReferenceAxis)
        reference = [referenceAxis, sideReferenceAxis, -referenceAxis, -sideReferenceAxis]
        maxDot = -2.0
        nextTangent = numpy.zeros((3,))
        for singleRef in reference:
            dot = numpy.dot(groundTangent, singleRef)
            if dot > maxDot:
                maxDot = dot
                nextTangent = singleRef
        groundTangent = nextTangent
        groundBitangent = numpy.cross(groundNormal, groundTangent)
        
        # worldNormal = numpy.stack([
        #     numpy.sum(normal * groundTangent, axis=-1),
        #     numpy.sum(normal * groundBitangent, axis=-1),
        #     numpy.sum(normal * groundNormal, axis=-1)
        # ], axis=-1)
        basis = numpy.stack([groundTangent, groundBitangent, groundNormal], axis=-1)  # shape (..., 3, 3)
        worldNormal = (normal[..., None, :] @ basis).squeeze(-2)
        worldPos = (pos[..., None, :] @ basis).squeeze(-2)
        # cv2.imshow("worldNormal", (worldNormal + 1.0) / 2.0)
        # cv2.imshow("worldPos", mulFract(worldPos, 0.01))


        # cv2.imshow("normalDotHori", normalHoriMask)
        # cv2.imshow("normalDotVert", normalVertMask)
        bgr = bgr.astype(numpy.float32)
        turnThreshold = 0.2
        
        ROI = validMask.copy()
        backFurthest = maskedFuzzyMax(worldPos, validMask, 0, epsilon=200)
        if numpy.dot(getAvgDir(normal[backFurthest]), groundNormal) < turnThreshold:
            bgr[backFurthest] = (bgr[backFurthest] * numpy.array([1, 2, 2]))
            ROI[backFurthest] = 0
        frontFurthest = maskedFuzzyMin(worldPos, validMask, 0, epsilon=200)
        if numpy.dot(getAvgDir(normal[frontFurthest]), groundNormal) < turnThreshold:
            bgr[frontFurthest] = (bgr[frontFurthest] * numpy.array([1, 2, 2]))
            ROI[frontFurthest] = 0
        leftFurthest = maskedFuzzyMin(worldPos, validMask, 1, epsilon=200)
        if numpy.dot(getAvgDir(normal[leftFurthest]), groundNormal) < turnThreshold:
            bgr[leftFurthest] = (bgr[leftFurthest] * numpy.array([1, 2, 2]))  
            ROI[leftFurthest] = 0
        rightFurthest = maskedFuzzyMax(worldPos, validMask, 1, epsilon=200)
        if numpy.dot(getAvgDir(normal[rightFurthest]), groundNormal) < turnThreshold:
            bgr[rightFurthest] = (bgr[rightFurthest] * numpy.array([1, 2, 2]))  
            ROI[rightFurthest] = 0
        downFurthest = maskedFuzzyMin(worldPos, validMask, 2, epsilon=120)
        bgr[downFurthest] = (bgr[downFurthest] * numpy.array([2, 1, 2]))  
        ROI[downFurthest] = 0

        selfRadius = 160
        selfRadiusSquare = selfRadius * selfRadius
        ROI[numpy.square(worldPos[:, :, 0]) + numpy.square(worldPos[:, :, 1]) < selfRadiusSquare] = 0

        # cv2.imshow("ROI", ROI * 255)
        bgr[ROI > 0] = (bgr[ROI > 0] * numpy.array([2, 2, 1]))
        # ROI = cv2.erode(ROI, kernel55, iterations=1)

        edgeEps = 5

        boxB = maskedFuzzyMax(worldPos, ROI, 0, epsilon=edgeEps)
        bgr[boxB] = (bgr[boxB] * numpy.array([0.5, 0.5, 2]))
        boxF = maskedFuzzyMin(worldPos, ROI, 0, epsilon=edgeEps)
        bgr[boxF] = (bgr[boxF] * numpy.array([0.5, 0.5, 2]))
        boxR = maskedFuzzyMax(worldPos, ROI, 1, epsilon=edgeEps)
        bgr[boxR] = (bgr[boxR] * numpy.array([0.5, 0.5, 2]))
        boxL = maskedFuzzyMin(worldPos, ROI, 1, epsilon=edgeEps)
        bgr[boxL] = (bgr[boxL] * numpy.array([0.5, 0.5, 2]))

        boxDPos = numpy.mean(worldPos[downFurthest][:, 2])
        boxBPos = numpy.mean(worldPos[boxB][:, 0])
        boxFPos = numpy.mean(worldPos[boxF][:, 0])
        boxRPos = numpy.mean(worldPos[boxR][:, 1])
        boxLPos = numpy.mean(worldPos[boxL][:, 1])

        boxHeight = [0, 1500, 830]
        boxClass = classifyRect((boxBPos - boxFPos) * 0.001, (boxRPos - boxLPos) * 0.001)
        # markAABB(bgr, basis, headFX, headFY, headCX, headCY, [boxFPos, boxLPos, boxDPos], [boxBPos, boxRPos, boxDPos + boxHeight[boxClass]])
        objectBoundingBoxCorners.append([[boxFPos, boxLPos, boxDPos], [boxBPos, boxRPos, boxDPos + boxHeight[boxClass]]])


        if boxClass == 1:
            cullDistThreshold = 20
            layerBottom = [500, 1100]
            layerTop = [900, 1400]
            layerMask = [None, None]
            tray_infos = []
            for i in range(2):
                # markAABB(bgr, basis, headFX, headFY, headCX, headCY, [boxFPos + 134, boxLPos + 100, boxDPos + layerBottom[i] + cullDistThreshold], [boxBPos, boxRPos - 100, boxDPos + layerTop[i]])
                objectBoundingBoxCorners.append([[boxFPos + 134, boxLPos + 100, boxDPos + layerBottom[i] + cullDistThreshold], [boxBPos, boxRPos - 100, boxDPos + layerTop[i]]])
                layerMask[i] = getAABBMask([boxFPos + 134, boxLPos + 100, boxDPos + layerBottom[i] + cullDistThreshold], [boxBPos, boxRPos - 100, boxDPos + layerTop[i]], worldPos).astype(numpy.uint8)
                u, v = numpy.where(layerMask[i])
                trayVecs = (worldPos[layerMask[i] != 0][:, 1:2]).astype(numpy.float32)
                _, labels, centers = cv2.kmeans(trayVecs, layerRemainCount[i], None, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001), 5, cv2.KMEANS_PP_CENTERS)
                markColors = [[2, 0.3, 0.3], [0.3, 2, 0.3], [0.3, 0.3, 2]]
                orderMap = numpy.argsort(centers[:, 0])
                for j in range(layerRemainCount[i]):
                    singleMask = labels[:, 0] == orderMap[j]
                    targetU = u[singleMask]
                    targetV = v[singleMask]
                    bgr[targetU, targetV] = bgr[targetU, targetV] * markColors[j]
                    LBoundary = numpy.min(worldPos[targetU, targetV][:, 1])
                    RBoundary = numpy.max(worldPos[targetU, targetV][:, 1])
                    FBoundary = numpy.min(worldPos[targetU, targetV][:, 0])
                    BBoundary = numpy.max(worldPos[targetU, targetV][:, 0])
                    UBoundary = numpy.max(worldPos[targetU, targetV][:, 2])
                    # markAABB(bgr, basis, headFX, headFY, headCX, headCY, [FBoundary, LBoundary, UBoundary + FBoundary - BBoundary], [BBoundary, RBoundary, UBoundary])
                    objectBoundingBoxCorners.append([[FBoundary, LBoundary, UBoundary + FBoundary - BBoundary], [BBoundary, RBoundary, UBoundary]])
                    tray_infos.append({
                        "layer": i,
                        "y_center_world_mm": float(centers[orderMap[j], 0]),
                        "x_center_world_mm": float((FBoundary + BBoundary) / 2.0),
                        "z_center_world_mm": float(UBoundary + (FBoundary - BBoundary) / 2.0),
                        "z_top_world_mm": float(UBoundary),
                        # 与 center_world_mm 同处当前 basis 定义的视觉世界系。
                        "aabb_world_mm": (
                            numpy.array([FBoundary, LBoundary, UBoundary + FBoundary - BBoundary], dtype=numpy.float32),
                            numpy.array([BBoundary, RBoundary, UBoundary], dtype=numpy.float32),
                        ),
                    })
        elif boxClass == 2:
            cullDistThreshold = 50
            targetMask = getAABBMask([boxFPos, boxLPos, boxDPos + boxHeight[boxClass] + cullDistThreshold], [boxBPos, boxRPos, boxDPos + boxHeight[boxClass] + 1000], worldPos).astype(numpy.uint8)
            bgr[targetMask > 0] = (bgr[targetMask > 0] * numpy.array([0.2, 2, 4]))
            coords = worldPos[targetMask > 0]
            LBoundary = numpy.min(coords[:, 1])
            RBoundary = numpy.max(coords[:, 1])
            FBoundary = numpy.min(coords[:, 0])
            BBoundary = numpy.max(coords[:, 0])
            UBoundary = numpy.max(coords[:, 2])
            # markAABB(bgr, basis, headFX, headFY, headCX, headCY, [FBoundary, LBoundary, boxDPos + boxHeight[boxClass]], [BBoundary, RBoundary, UBoundary])
            objectBoundingBoxCorners.append([[FBoundary, LBoundary, boxDPos + boxHeight[boxClass]], [BBoundary, RBoundary, UBoundary]])
            targetThickNess = 16
            goodHeight = [0, 100]
            # markAABB(bgr, basis, headFX, headFY, headCX, headCY, [FBoundary + targetThickNess, LBoundary + targetThickNess, UBoundary + goodHeight[0]], [BBoundary - targetThickNess, RBoundary - targetThickNess, UBoundary + goodHeight[1]])
            objectBoundingBoxCorners.append([[FBoundary + targetThickNess, LBoundary + targetThickNess, UBoundary + goodHeight[0]], [BBoundary - targetThickNess, RBoundary - targetThickNess, UBoundary + goodHeight[1]]])

        # ---- 最优站位规划（保持最后一次有效检测快照）----
        # 抬臂后头部画面会被 selfRadius 内的机械臂遮挡；此时绝不能用
        # 当前帧的错误 box/basis 覆盖供腕相机使用的目标坐标。
        if boxClass == 1 and not approach_plan_locked:
            candidate_plan = planTrayApproach(tray_infos, basis, tf)
            if candidate_plan is not None:
                approach_plan = candidate_plan
                approach_basis = basis.copy()

        leftBasis, rightBasis = getHandBasis(basis, tf)
        leftOrigin, rightOrigin = getHandOrigin(basis, tf)
        if approach_basis is not None:
            planLeftBasis, planRightBasis = getHandBasis(approach_basis, tf)
            planLeftOrigin, planRightOrigin = getHandOrigin(approach_basis, tf)
        else:
            planLeftBasis = planRightBasis = None
            planLeftOrigin = planRightOrigin = None

        for i in range(len(objectBoundingBoxCorners)):
            markAABB(bgr, basis, headFX, headFY, headCX, headCY, objectBoundingBoxCorners[i][0], objectBoundingBoxCorners[i][1])
            if Rbgr is not None and rightBasis is not None and rightOrigin is not None:
                markAABB(Rbgr, rightBasis, rightFX, rightFY, rightCX, rightCY, objectBoundingBoxCorners[i][0] - rightOrigin, objectBoundingBoxCorners[i][1] - rightOrigin)
            if Lbgr is not None and leftBasis is not None and leftOrigin is not None:
                markAABB(Lbgr, leftBasis, leftFX, leftFY, leftCX, leftCY, objectBoundingBoxCorners[i][0] - leftOrigin, objectBoundingBoxCorners[i][1] - leftOrigin)

        if boxClass != 0:
            markTickedLine(bgr, [0, 0, -100], [(boxBPos + boxFPos) / 2, (boxRPos + boxLPos) / 2, -100], basis, headFX, headFY, headCX, headCY, 100, 10, False, True, 10, 40, 50, [255, 255, 255], [255, 0, 0], [0, 0, 255], [0, 255, 0], 1, 1, 1, 1)
        camFrontPoint = [0, 0, 1] @ basis
        rasterizeSegment(bgr, camFrontPoint, camFrontPoint + [0, 0.1, 0], basis, headFX, headFY, headCX, headCY, [0, 0, 255], 2)
        rasterizeSegment(bgr, camFrontPoint, camFrontPoint + [0.1, 0, 0], basis, headFX, headFY, headCX, headCY, [255, 0, 0], 2)
        rasterizeSegment(bgr, camFrontPoint, camFrontPoint + [0, 0, 0.1], basis, headFX, headFY, headCX, headCY, [0, 255, 0], 2)
                
        
        # cv2.imshow("gNormalImg", numpy.dot(normal, groundNormal))
        # cv2.imshow("gTangentImg", numpy.dot(normal, groundTangent))
        # cv2.imshow("gBitangentImg", numpy.dot(normal, groundBitangent))
        # ---- 头相机红点：标记缓存检测快照中的料盘中心 ----
        if approach_plan is not None and approach_basis is not None:
            for tray in approach_plan["trays"]:
                center = numpy.asarray(tray["center_world_mm"], dtype=numpy.float32)
                p = approach_basis @ center
                if p[2] > 0:
                    u = int(round(headFX * p[0] / p[2] + headCX))
                    v = int(round(headFY * p[1] / p[2] + headCY))
                    cv2.circle(bgr, (u, v), 10, (0, 0, 255), -1)

        cv2.namedWindow("bgr", cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow("bgr", bgr / 255.0)

        wrist_cam_ok = (Rbgr is not None and rightBasis is not None
                        and Lbgr is not None and leftBasis is not None
                        and rightOrigin is not None and leftOrigin is not None)
        if wrist_cam_ok:
            RcamFrontPoint = [0, 0, 1] @ rightBasis
            rasterizeSegment(Rbgr, RcamFrontPoint, RcamFrontPoint + [0, 0.1, 0], rightBasis, rightFX, rightFY, rightCX, rightCY, [0, 0, 255], 2)
            rasterizeSegment(Rbgr, RcamFrontPoint, RcamFrontPoint + [0.1, 0, 0], rightBasis, rightFX, rightFY, rightCX, rightCY, [255, 0, 0], 2)
            rasterizeSegment(Rbgr, RcamFrontPoint, RcamFrontPoint + [0, 0, 0.1], rightBasis, rightFX, rightFY, rightCX, rightCY, [0, 255, 0], 2)

            LcamFrontPoint = [0, 0, 1] @ leftBasis
            rasterizeSegment(Lbgr, LcamFrontPoint, LcamFrontPoint + [0, 0.1, 0], leftBasis, leftFX, leftFY, leftCX, leftCY, [0, 0, 255], 2)
            rasterizeSegment(Lbgr, LcamFrontPoint, LcamFrontPoint + [0.1, 0, 0], leftBasis, leftFX, leftFY, leftCX, leftCY, [255, 0, 0], 2)
            rasterizeSegment(Lbgr, LcamFrontPoint, LcamFrontPoint + [0, 0, 0.1], leftBasis, leftFX, leftFY, leftCX, leftCY, [0, 255, 0], 2)

            # ---- 手腕相机：缓存 AABB + 深度 ROI 料盘实测 ----
            # 所有投影、ROI 均使用 approach_basis，不使用被抬起手臂干扰的当前 basis。
            if (approach_plan is not None and planLeftBasis is not None and
                    planRightBasis is not None and planLeftOrigin is not None and
                    planRightOrigin is not None):
                for tray in approach_plan["trays"]:
                    center = numpy.asarray(tray["center_world_mm"], dtype=numpy.float32)
                    aabb = tray.get("aabb_world_mm")
                    if aabb is None:
                        continue
                    aabbMin, aabbMax = aabb
                    # 使用与黄色框完全相同的左右放大 3D ROI。
                    expandedMin, expandedMax = expandAABBLeftRight(aabbMin, aabbMax)

                    if tray["hand"] == "left":
                        p = planLeftBasis @ (center - planLeftOrigin)
                        if p[2] > 0:
                            u = int(round(leftFX * p[0] / p[2] + leftCX))
                            v = int(round(leftFY * p[1] / p[2] + leftCY))
                            cv2.circle(Lbgr, (u, v), 10, (0, 0, 255), -1)
                        markAABB(Lbgr, planLeftBasis, leftFX, leftFY, leftCX, leftCY,
                                 expandedMin - planLeftOrigin, expandedMax - planLeftOrigin,
                                 lineWeight=3, color=(0, 255, 255),
                                 xColor=(0, 200, 255), yColor=(0, 255, 200), zColor=(0, 165, 255))
                        detection, componentMask = detectTrayInWristAABB(
                            Ldepth, planLeftBasis, planLeftOrigin,
                            leftFX, leftFY, leftCX, leftCY,
                            center, expandedMin, expandedMax)
                        if detection is not None:
                            wrist_tray_offsets["left"] = detection
                            Lbgr[componentMask] = Lbgr[componentMask] * 0.35 + numpy.array([0, 255, 0]) * 0.65
                            observedU, observedV = numpy.rint(detection["pixel"]).astype(int)
                            cv2.circle(Lbgr, (observedU, observedV), 7, (0, 255, 0), -1)
                    else:
                        p = planRightBasis @ (center - planRightOrigin)
                        if p[2] > 0:
                            u = int(round(rightFX * p[0] / p[2] + rightCX))
                            v = int(round(rightFY * p[1] / p[2] + rightCY))
                            cv2.circle(Rbgr, (u, v), 10, (0, 0, 255), -1)
                        markAABB(Rbgr, planRightBasis, rightFX, rightFY, rightCX, rightCY,
                                 expandedMin - planRightOrigin, expandedMax - planRightOrigin,
                                 lineWeight=3, color=(0, 255, 255),
                                 xColor=(0, 200, 255), yColor=(0, 255, 200), zColor=(0, 165, 255))
                        detection, componentMask = detectTrayInWristAABB(
                            Rdepth, planRightBasis, planRightOrigin,
                            rightFX, rightFY, rightCX, rightCY,
                            center, expandedMin, expandedMax)
                        if detection is not None:
                            wrist_tray_offsets["right"] = detection
                            Rbgr[componentMask] = Rbgr[componentMask] * 0.35 + numpy.array([0, 255, 0]) * 0.65
                            observedU, observedV = numpy.rint(detection["pixel"]).astype(int)
                            cv2.circle(Rbgr, (observedU, observedV), 7, (0, 255, 0), -1)

            cv2.namedWindow("leftWristImg", cv2.WINDOW_GUI_EXPANDED)
            cv2.imshow("leftWristImg", Lbgr / 255.0)
            cv2.namedWindow("rightWristImg", cv2.WINDOW_GUI_EXPANDED)
            cv2.imshow("rightWristImg", Rbgr / 255.0)


        if phase == "walk_to_shelf":
            if boxClass == 1:
                print(boxBPos)
                if abs(boxBPos) > 1000:
                    robot.move_forward(0.3, (boxFPos/10-70)/0.3)
                elif abs(boxBPos) > 600:
                    robot.move_forward(0.05)
                else:
                    robot.stop()
                    phase = "side_shift"
            else:
                robot.stop()
                log(f"[ERROR] {phase}: shelf not found")
        
        if phase == "side_shift":
            assert(approach_plan)

            upper_joints = None
            upper_tray_y = None
            for tray in approach_plan["trays"]:
                if tray["layer"] == 1:
                    upper_joints = tray["hand"]   # 记录"left"或"right"
                    upper_tray_y = tray["center_world_mm"][1]

            log(f"[INFO] {phase} y: {upper_tray_y}")

            # world Y+ is right
            if upper_joints == "left":
                off = upper_tray_y + SHOULDER_WIDTH/2
                log(f"[INFO] {phase} off: {off}")
                if abs(off) > 10:
                    robot.move_right(0.05 * abs(off) / off)
                elif abs(off) > 5:
                    robot.move_right(0.03 * abs(off) / off)
                elif boxClass == 1:
                    robot.stop()
                    phase = "post_side_shift"

        
        if phase == "post_side_shift":
            print(boxBPos)
            if boxClass == 1:
                if abs(boxBPos) > 300:
                    robot.move_forward(0.05)
                else:
                    robot.stop()
                    phase = "upper_hand_ready"

        if phase == "upper_hand_ready":
            assert(approach_plan)

            # 确定上下层各对应的手，上层抬、下层低位
            upper_joints = None
            lower_joints = [0, 0, 0, 0, 0, 0, 0]
            for tray in approach_plan["trays"]:
                if tray["layer"] == 1:
                    upper_joints = tray["hand"]   # 记录"left"或"right"

            if upper_joints is None:
                log("[ERROR] upper_hand_ready: no upper layer tray in plan")
                phase = "walk_to_shelf"
                continue

            active_grab_hand = upper_joints
            # 此后头相机被手臂遮挡：冻结最后一次有效检测的 plan/AABB/basis。
            approach_plan_locked = True

            arm.switch_to_external_control()
            rospy.sleep(3.0)

            left_arm  = [0, 0, 0, 0, 0, 0, 0]
            right_arm = [0, 0, 0, 0, 0, 0, 0]

            if upper_joints == "left":
                left_arm = [30, 0, 0, 0, 0, 0, 0]
            else:
                right_arm = [30, 0, 0, 0, 0, 0, 0]
            arm.go_to_joints(left_arm + right_arm)
            rospy.sleep(2.0)

            if upper_joints == "left":
                left_arm = [30, 0, 0, -150, -90, 0, 0]
            else:
                right_arm = [30, 0, 0, -150, 90, 0, 0]
            arm.go_to_joints(left_arm + right_arm)
            rospy.sleep(2.0)

            if upper_joints == "left":
                left_arm = [-10, 0, 0, -150, -90, 0, 0]
            else:
                right_arm = [-10, 0, 0, -150, 90, 0, 0]
            arm.go_to_joints(left_arm + right_arm)
            rospy.sleep(3.0)

            if upper_joints == "left":
                left_arm = [-20, 0, 0, -110, -90, -40, 0]
            else:
                right_arm = [-20, 0, 0, -110, 90, 40, 0]
            arm.go_to_joints(left_arm + right_arm)
            rospy.sleep(5.0)

            log(f"[INFO] upper_hand_ready: {upper_joints} arm raised for upper layer")
            phase = "grab_upper_smt"
        
        if phase == "grab_upper_smt":
            # 只校正当前抬起、准备抓取上层料盘的手，避免带动另一只手。
            detection = wrist_tray_offsets.get(active_grab_hand)
            if detection is None:
                log(f"[INFO] {active_grab_hand} wrist: AABB ROI 内未检测到料盘")
            else:
                offset = detection["offset_camera_mm"]
                pixelOffset = detection["pixel_offset"]
                lateralOffsetMm = float(offset[0])  # 腕相机光学帧 X：向右为正
                nearEdgeDepthMm = float(detection["near_edge_depth_mm"])
                forwardErrorMm = nearEdgeDepthMm - WRIST_DEPTH_TARGET_MM
                log(f"{active_grab_hand} 手腕实测→料盘: X={offset[0]:.0f}mm(右), "
                    f"Y={offset[1]:.0f}mm(下), Z中心={offset[2]:.0f}mm(前), "
                    f"Z近边缘={nearEdgeDepthMm:.0f}mm, "
                    f"像素偏移=({pixelOffset[0]:.1f}, {pixelOffset[1]:.1f}), "
                    f"点数={detection['point_count']}")

                if approach_basis is None:
                    log("[ERROR] wrist alignment: 缺少冻结的视觉 world basis")
                else:
                    handBasis = planLeftBasis if active_grab_hand == "left" else planRightBasis

                    if abs(lateralOffsetMm) > WRIST_ALIGN_TOLERANCE_MM:
                        deltaBaseM = wristCameraVectorToBase(
                            [lateralOffsetMm, 0.0, 0.0], handBasis, approach_basis, tf)
                        deltaNorm = float(numpy.linalg.norm(deltaBaseM))
                        if deltaNorm > WRIST_ALIGN_MAX_STEP_M:
                            deltaBaseM *= WRIST_ALIGN_MAX_STEP_M / deltaNorm
                        log(f"[INFO] {active_grab_hand} wrist: 横向校正 "
                            f"{lateralOffsetMm:.1f}mm -> base Δ={deltaBaseM.tolist()} m")
                        arm.move_relative(active_grab_hand, deltaBaseM.tolist(),
                                         max_error_m=0.03, sleep=1.5)

                    if abs(forwardErrorMm) > WRIST_DEPTH_TOLERANCE_MM:
                        deltaBaseM = wristCameraVectorToBase(
                            [0.0, 0.0, forwardErrorMm], handBasis, approach_basis, tf)
                        deltaNorm = float(numpy.linalg.norm(deltaBaseM))
                        if deltaNorm > WRIST_ALIGN_MAX_STEP_M:
                            deltaBaseM *= WRIST_ALIGN_MAX_STEP_M / deltaNorm
                        log(f"[INFO] {active_grab_hand} wrist: 前后校正 "
                            f"近边缘 {nearEdgeDepthMm:.1f}mm -> "
                            f"目标 {WRIST_DEPTH_TARGET_MM:.1f}mm, base Δ={deltaBaseM.tolist()} m")
                        arm.move_relative(active_grab_hand, deltaBaseM.tolist(),
                                         max_error_m=0.03, sleep=1.5)

                    log(f"[INFO] {active_grab_hand} wrist: 校正完成，开始夹取")
                    rospy.sleep(3.0)
                    phase = "close_upper_smt"
        if phase == "close_upper_smt":
            (claw.left_close() if active_grab_hand == "left" else claw.right_close())
            claw.wait_until_done(timeout=3.0)
            rospy.sleep(3.0)
            log(f"[INFO] {active_grab_hand} claw: 已闭合，开始上提")
            phase = "lift_upper_smt"

        if phase == "lift_upper_smt":
            left_arm  = [0, 0, 0, 0, 0, 0, 0]
            right_arm = [0, 0, 0, 0, 0, 0, 0]
            if active_grab_hand == "left":
                left_arm = [0, 0, 0, -150, -90, 0, 0]
            else:
                right_arm = [0, 0, 0, -150, 90, 0, 0]
            arm.go_to_joints(left_arm + right_arm)
            rospy.sleep(2.0)
            log(f"[INFO] {active_grab_hand} arm: 已上提")
            phase = "upper_smt_lifted"

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        rate.sleep()
    # cv2.destroyWindow("depth")
      
    # -------------------------------------------

    log("场景三：任务结束")
