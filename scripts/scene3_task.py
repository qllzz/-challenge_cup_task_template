#!/usr/bin/env python3

import rospy
import numpy

try:
    import cv2
except ImportError:
    cv2 = None
 
from perception_api import CameraReader
from sensor_msgs.msg import CameraInfo

def depthToPos(depth, fx, fy, cx, cy):
    h, w = depth.shape
    u, v = numpy.meshgrid(numpy.arange(w), numpy.arange(h))
    z = depth.astype(numpy.float32)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pos = numpy.stack((x, y, z), axis=-1)
    return pos

def mulFract(x, factor):
    x = x * factor
    return x - numpy.floor(x)

def getDiffBlendFactor(absDiff, s):
    return 1.0 / (1.0 + numpy.exp(absDiff * s))

def getNormalDot(normal):
    normalWithPadding = numpy.pad(normal, ((1, 1), (1, 1), (0, 0)), mode='edge')
    normalT = normalWithPadding[:-2, 1:-1]
    normalL = normalWithPadding[1:-1, :-2]
    normalR = normalWithPadding[1:-1, 2:]
    normalB = normalWithPadding[2:, 1:-1]

    normalDotT = 1.0 - numpy.sum(normalT * normal, axis=-1)
    normalDotL = 1.0 - numpy.sum(normalL * normal, axis=-1)
    normalDotR = 1.0 - numpy.sum(normalR * normal, axis=-1)
    normalDotB = 1.0 - numpy.sum(normalB * normal, axis=-1)

    return normalDotL * normalDotR, normalDotB * normalDotT


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
    
    normal_raw = numpy.cross(Tu, Tv, axis=2)
    norm = numpy.linalg.norm(normal_raw, axis=2, keepdims=True)
    normal = normal_raw / norm
    
    return normal

def getCylinderAxis(vectors, max_iter=50, tol=1e-6):
    """
    使用 Tukey's Biweight (M-estimator) 估计中轴方向
    Args:
        vectors: (N, 3) 单位向量
    Returns:
        axis: (3,) 中轴单位向量
    """
    # 1. 初始值：先用普通 PCA 给个初值（虽然怕噪，但方向大差不差）
    M = vectors.T @ vectors
    _, eig_vecs = numpy.linalg.eigh(M)
    axis = eig_vecs[:, 0].copy()  # 最小特征向量
    if axis[2] < 0:
        axis = -axis
    
    # 2. 计算初始投影中位数，确定平面常数 c
    proj = vectors @ axis
    c = numpy.median(proj)
    
    for _ in range(max_iter):
        # 计算残差（向量到拟合平面的距离）
        residuals = numpy.abs(vectors @ axis - c)
        
        # 计算缩放常数（Tukey 的调谐常数，通常取 4.685 对应 95% 效率）
        # 这里用中位数绝对偏差 (MAD) 做稳健缩放
        mad = numpy.median(residuals) / 0.6745  # 0.6745 是正态分布缩放因子
        if mad < 1e-12:
            break  # 已经完美拟合
        
        # 归一化残差
        u = residuals / (mad * 4.685)  # 4.685 是 Tukey 常用常数
        
        # 计算 Tukey 双平方权重：|u|<=1 时权重递减，|u|>1 时权重为0
        weights = numpy.ones_like(u)
        mask = numpy.abs(u) < 1.0
        weights[mask] = (1 - u[mask]**2) ** 2
        weights[~mask] = 0  # 完全剔除极端离群点
        
        # 如果大部分点被剔除，提前退出
        if numpy.sum(weights) < 3:
            break
            
        # 加权 PCA：计算加权协方差矩阵 M_w
        weighted_vectors = vectors * weights[:, numpy.newaxis]  # 逐行加权
        M_w = weighted_vectors.T @ vectors  # 注意这里不除以N，不影响特征向量
        
        # 特征分解取最小特征向量
        _, eig_vecs_new = numpy.linalg.eigh(M_w)
        axis_new = eig_vecs_new[:, 0]
        if axis_new[2] < 0:
            axis_new = -axis_new
            
        # 更新平面常数 c
        proj_new = vectors @ axis_new
        c_new = numpy.median(proj_new)  # 用中位数抗噪
        
        # 检查收敛
        if numpy.linalg.norm(axis - axis_new) < tol:
            axis = axis_new
            c = c_new
            break
            
        axis = axis_new
        c = c_new
        
    return axis

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

def getCylinderPos(points, normals, axis, radius, eps=0.05, min_samples=10):
    u = axis / numpy.linalg.norm(axis)
    ref = numpy.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else numpy.array([0.0, 1.0, 0.0])
    e1 = ref - numpy.dot(ref, u) * u
    e1 = e1 / numpy.linalg.norm(e1)
    e2 = numpy.cross(u, e1)

    pts_2d = numpy.stack([numpy.dot(points, e1), numpy.dot(points, e2)], axis=1)

    n_dot_axis = numpy.dot(normals, u)
    normals_perp = normals - n_dot_axis[:, numpy.newaxis] * u
    norm_perp = numpy.linalg.norm(normals_perp, axis=1)
    valid_mask = norm_perp > 1e-8

    if not numpy.any(valid_mask):
        return numpy.array([]), numpy.array([])

    pts_2d_valid = pts_2d[valid_mask]
    normals_perp_valid = normals_perp[valid_mask]
    radial_2d = normals_perp_valid / numpy.linalg.norm(normals_perp_valid, axis=1, keepdims=True)
    centers_2d_est = pts_2d_valid - radius * radial_2d

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pts_2d_valid)
    labels = clustering.labels_

    final_centers_2d = []
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:
            continue
        mask_cluster = (labels == label)
        cluster_pts = pts_2d_valid[mask_cluster]
        cluster_centers = centers_2d_est[mask_cluster]

        if len(cluster_pts) < min_samples:
            continue

        cov_pts = numpy.cov(cluster_pts.T)
        eigvals_pts = numpy.linalg.eigvalsh(cov_pts)
        aspect_ratio = eigvals_pts[1] / (eigvals_pts[0] + 1e-12)
        if aspect_ratio > 3.0:
            continue

        std_dev = numpy.std(cluster_centers, axis=0)
        if numpy.max(std_dev) > radius * 0.5:
            continue

        median_center = numpy.median(cluster_centers, axis=0)
        final_centers_2d.append(median_center)

    if not final_centers_2d:
        return numpy.array([]), numpy.array([])

    final_centers_2d = numpy.array(final_centers_2d)
    final_centers_3d = (final_centers_2d[:, 0][:, numpy.newaxis] * e1 +
                        final_centers_2d[:, 1][:, numpy.newaxis] * e2)
    return final_centers_2d, final_centers_3d

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

def run_scene3(robot, arm, claw, head, log):
    log("=" * 50)
    log("场景三：SMT 料盘出库 — 任务开始")
    log("=" * 50)

    cam = CameraReader()
    camInfo = rospy.wait_for_message("/cam_h/color/camera_info", CameraInfo)
    fx = camInfo.K[0]
    fy = camInfo.K[4]
    cx = camInfo.K[2]
    cy = camInfo.K[5]
    # fovx = numpy.atan(cx / fx) + numpy.atan((width - cx) / fx)
    # fovy = numpy.atan(cy / fy) + numpy.atan((height - cy) / fy)

    rate = rospy.Rate(15.0)

    groundTangent = numpy.zeros((3,))

    kernel33 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3), anchor=None)
    kernel55 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5), anchor=None)
    kernel77 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7), anchor=None)

    while not rospy.is_shutdown():
        depth = cam.get_head_depth()
        # farMask = ((depth <= 10000) & ~numpy.isnan(depth)).astype(numpy.uint8)
        depth[depth > 10000] = 0
        validMask = (depth != 0).astype(numpy.uint8)
        validMask = cv2.erode(validMask, kernel77, iterations=1)
        cv2.imshow("validMask", validMask * 255)

        bgr = cam.get_head_rgb()
        if bgr is None:
            log("[INFO] bgr is None")
            rate.sleep()
            continue
        if depth is None:
            log("[INFO] depth is None")
            rate.sleep()
            continue
        pos = depthToPos(depth, fx, fy, cx, cy)
        normal = posToNormal(pos, depth)
        normalDotHori, normalDotVert = getNormalDot(normal)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        groundMask = hsvKey(hsv, [73, 97, 110], [5, 10, 10])

        cv2.imshow("depth", mulFract(depth, 0.01))
        cv2.imshow("pos", mulFract(pos, 0.01))
        cv2.imshow("normal", (normal + 1.0) / 2.0)

        cv2.imshow("normalDotHori", normalDotHori * 500)
        # cv2.imshow("normalDotVert", normalDotVert * 500)

        detectThreshold = 0.0001
        normalHoriMask = (normalDotHori > detectThreshold).astype(numpy.uint8) * 255
        # normalVertMask = (normalDotVert > detectThreshold).astype(numpy.uint8) * 255

        normalHoriMask = cv2.erode(normalHoriMask, kernel33, iterations=1)
        # normalVertMask = cv2.erode(normalVertMask, kernel33, iterations=1)

        normalHoriMask = cv2.dilate(normalHoriMask, kernel77, iterations=1)
        # normalVertMask = cv2.dilate(normalVertMask, kernel77, iterations=1)
        normalHoriMask = cv2.erode(normalHoriMask, kernel55, iterations=1)
        # normalVertMask = cv2.erode(normalVertMask, kernel55, iterations=1)

        

        groundVec = normal[groundMask > 0]
        if numpy.size(groundVec) != 0:
            groundNormal = getAvgDir(groundVec)
        groundImg = numpy.dot(normal, groundNormal)

        nonVertMask = (groundImg < 0.2).astype(numpy.uint8) * 255
        sideVec = normal[nonVertMask > 0]
        _, _, centers = cv2.kmeans(sideVec.astype(numpy.float32), 3, None, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 10, 0.001), 10, cv2.KMEANS_PP_CENTERS)

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
        cv2.imshow("worldNormal", (worldNormal + 1.0) / 2.0)
        cv2.imshow("worldPos", mulFract(worldPos, 0.01))


        cv2.imshow("normalDotHori", normalHoriMask)
        # cv2.imshow("normalDotVert", normalVertMask)
        bgr = bgr.astype(numpy.float32)
        turnThreshold = 0.1
        
        ROI = validMask.copy()
        frontFurthest = maskedFuzzyMax(worldPos, validMask, 0, epsilon=200)
        if numpy.dot(getAvgDir(normal[frontFurthest]), groundNormal) < turnThreshold:
            # bgr[frontFurthest] = (bgr[frontFurthest] * numpy.array([2, 2, 1]))
            ROI[frontFurthest] = 0
        backFurthest = maskedFuzzyMin(worldPos, validMask, 0, epsilon=200)
        if numpy.dot(getAvgDir(normal[backFurthest]), groundNormal) < turnThreshold:
            # bgr[backFurthest] = (bgr[backFurthest] * numpy.array([2, 2, 1]))
            ROI[backFurthest] = 0
        leftFurthest = maskedFuzzyMin(worldPos, validMask, 1, epsilon=200)
        if numpy.dot(getAvgDir(normal[leftFurthest]), groundNormal) < turnThreshold:
            # bgr[leftFurthest] = (bgr[leftFurthest] * numpy.array([2, 2, 1]))  
            ROI[leftFurthest] = 0
        rightFurthest = maskedFuzzyMax(worldPos, validMask, 1, epsilon=200)
        if numpy.dot(getAvgDir(normal[rightFurthest]), groundNormal) < turnThreshold:
            # bgr[rightFurthest] = (bgr[rightFurthest] * numpy.array([2, 2, 1]))  
            ROI[rightFurthest] = 0
        bottomFurthest = maskedFuzzyMax(worldPos, validMask, 2, epsilon=120)
        # if numpy.dot(getAvgDir(normal[bottomFurthest]), groundNormal) < turnThreshold:
        # bgr[bottomFurthest] = (bgr[bottomFurthest] * numpy.array([2, 1, 2]))  
        ROI[bottomFurthest] = 0

        cv2.imshow("ROI", ROI * 255)

        bgr[ROI > 0] = (bgr[ROI > 0] * numpy.array([2, 2, 1]))
        cv2.imshow("bgr", bgr / 255.0)

        # cv2.imshow("maskedNormal", (normal + 1.0) / 2.0 * (normalHoriMask > 0).astype(numpy.float32)[:, :, numpy.newaxis])
        # arm.switch_to_external_control()
        robot.turn_left(angular_speed=0.6)
        # robot.squat(-0.4)
        # robot.move_forward(speed=0.4)



        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        rate.sleep()
    cv2.destroyWindow("depth")
      
    # -------------------------------------------

    log("场景三：任务结束")
