#!/usr/bin/env python3
"""Perception RGB-D Scene 1 — 4 colis (brown / yellow / orange / blue).

Caméra tête : LAB/HSV + profondeur → base_link. Pas de LiDAR.
"""
from __future__ import print_function

import argparse
import json
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

# parcel_id → couleur (mission scene 1)
CLASS_LABELS = {
    "parcel_1": "brown",
    "parcel_2": "yellow",
    "parcel_3": "orange",
    "parcel_4": "blue",
}

CLASS_MAX_COUNT = {
    "parcel_1": 1,
    "parcel_2": 1,
    "parcel_3": 1,
    "parcel_4": 1,
}

PARCEL_ORDER = ("parcel_1", "parcel_2", "parcel_3", "parcel_4")


def _default_output_dir(name):
    """Store optional perception diagnostics inside this task package."""
    package_dir = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    return os.path.join(package_dir, "artifacts", "scene1", name)


# HSV : bleu d'abord (hue distinct). Chauds = LAB only (trop de chevauchement HSV).
PARCEL_HSV_BLUE = ((88, 5, 70), (145, 200, 255))

# Réf. BGR matériaux scene1.yaml — distance LAB pour départager chauds
PARCEL_REF_BGR = {
    "parcel_2": (97, 199, 235),    # yellow
    "parcel_1": (186, 214, 224),   # brown
    "parcel_3": (71, 140, 230),    # orange
    "parcel_4": (235, 214, 199),   # blue
}

# Seuils LAB — chauds seulement (le bleu est pris en HSV prioritaire)
WARM_ORDER = ("parcel_1", "parcel_2", "parcel_3")
LAB_DIST_BY_NAME = {
    "parcel_1": 78.0,   # brown
    "parcel_2": 82.0,   # yellow
    "parcel_3": 90.0,   # orange
    "parcel_4": 72.0,   # blue (secours LAB si HSV rate)
}

TABLE_REF_BGR = (26, 77, 26)
TABLE_LAB_DIST_MAX = 34.0
LAB_WIN_MARGIN = 2.0

# Zone table base_link (IK-like), cf. scene1 config — un peu élargie pour le bleu loin
BASE_VALID_X = (0.15, 0.62)
BASE_VALID_Y = (-0.52, 0.05)
BASE_VALID_Z = (-0.22, 0.15)


def _decode_color(msg):
    data = np.frombuffer(msg.data, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("failed to decode compressed color image")
    return image


def _find_png_payload(data):
    signature = b"\x89PNG\r\n\x1a\n"
    offset = data.find(signature)
    if offset < 0:
        return data
    return data[offset:]


def _decode_depth(msg):
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
    x = (float(u) - model["cx"]) * float(z) / model["fx"]
    y = (float(v) - model["cy"]) * float(z) / model["fy"]
    return [x, y, float(z)]


def _transform_point(listener, point_xyz, source_frame, target_frame, stamp, timeout):
    if not source_frame:
        return None
    stamped = PointStamped()
    stamped.header.frame_id = source_frame
    stamped.header.stamp = stamp
    stamped.point.x = float(point_xyz[0])
    stamped.point.y = float(point_xyz[1])
    stamped.point.z = float(point_xyz[2])
    try:
        listener.waitForTransform(target_frame, source_frame, rospy.Time(0), rospy.Duration(timeout))
        transformed = listener.transformPoint(target_frame, stamped)
    except Exception:
        try:
            stamped.header.stamp = rospy.Time(0)
            transformed = listener.transformPoint(target_frame, stamped)
        except Exception:
            return None
    return [transformed.point.x, transformed.point.y, transformed.point.z]


def _clean_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _bgr_to_lab(bgr):
    swatch = np.uint8([[list(bgr)]])
    return cv2.cvtColor(swatch, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


def _make_masks(bgr):
    """
    1) Bleu HSV en premier (hue distinct) — réservé, les chauds ne peuvent pas le manger.
    2) LAB winner-takes-all sur brown/yellow/orange uniquement (pixels restants).
    3) Secours LAB bleu sur pixels encore libres.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    table_lab = _bgr_to_lab(TABLE_REF_BGR)
    table_dist = np.linalg.norm(lab - table_lab.reshape(1, 1, 3), axis=2)
    not_table = table_dist > float(TABLE_LAB_DIST_MAX)
    free = not_table.copy()

    lo, hi = PARCEL_HSV_BLUE
    blue_hsv = cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
    blue_hsv = (blue_hsv > 0) & free
    masks = {
        "parcel_1": np.zeros(bgr.shape[:2], dtype=np.uint8),
        "parcel_2": np.zeros(bgr.shape[:2], dtype=np.uint8),
        "parcel_3": np.zeros(bgr.shape[:2], dtype=np.uint8),
        "parcel_4": (blue_hsv.astype(np.uint8) * 255),
    }
    free = free & (~blue_hsv)

    # LAB WTA entre chauds seulement
    dist_stack = []
    thr_list = []
    for class_name in WARM_ORDER:
        ref = _bgr_to_lab(PARCEL_REF_BGR[class_name])
        dist_stack.append(np.linalg.norm(lab - ref.reshape(1, 1, 3), axis=2))
        thr_list.append(float(LAB_DIST_BY_NAME[class_name]))
    dist_stack = np.stack(dist_stack, axis=2)
    thr = np.array(thr_list, dtype=np.float32).reshape(1, 1, 3)

    order = np.argsort(dist_stack, axis=2)
    best_idx = order[:, :, 0]
    second_idx = order[:, :, 1]
    best_dist = np.take_along_axis(dist_stack, best_idx[:, :, None], axis=2)[:, :, 0]
    second_dist = np.take_along_axis(dist_stack, second_idx[:, :, None], axis=2)[:, :, 0]
    best_thr = thr[0, 0, best_idx]

    warm_win = (
        free
        & (best_dist < best_thr)
        & ((second_dist - best_dist) >= float(LAB_WIN_MARGIN))
    )
    for i, class_name in enumerate(WARM_ORDER):
        masks[class_name] = ((best_idx == i) & warm_win).astype(np.uint8) * 255
        free = free & ~((best_idx == i) & warm_win)

    # Secours LAB bleu (pâle) si HSV a raté
    blue_ref = _bgr_to_lab(PARCEL_REF_BGR["parcel_4"])
    blue_lab_dist = np.linalg.norm(lab - blue_ref.reshape(1, 1, 3), axis=2)
    blue_lab = free & (blue_lab_dist < float(LAB_DIST_BY_NAME["parcel_4"]))
    masks["parcel_4"] = cv2.bitwise_or(
        masks["parcel_4"], (blue_lab.astype(np.uint8) * 255),
    )

    for class_name in PARCEL_ORDER:
        masks[class_name] = _clean_mask(masks[class_name])
    return masks


def _contour_angle(contour):
    if len(contour) < 5:
        return None
    rect = cv2.minAreaRect(contour)
    (width, height) = rect[1]
    angle = float(rect[2])
    if width < height:
        angle += 90.0
    return angle


def _contour_shape_metrics(contour, bbox):
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
    """Filtres forme colis (boîtes plutôt carrées / rectangulaires)."""
    extent = metrics["extent"]
    solidity = metrics["solidity"]
    min_area = 100.0 if class_name in ("parcel_1", "parcel_3", "parcel_4") else 160.0
    if area < min_area or area > 80000.0:
        return False
    if aspect > 4.0:
        return False
    if extent < 0.15 or solidity < 0.25:
        return False
    return True


def _candidate_sort_key(item):
    base_key = item.get("base_link_xyz_m")
    if base_key is not None and len(base_key) >= 2:
        return (float(base_key[1]), float(base_key[0]))
    pixel = item.get("pixel") or [0.0, 0.0]
    return (float(pixel[0]), float(pixel[1]))


def _select_final_detections(candidates):
    detections = []
    for class_name in PARCEL_ORDER:
        class_items = [
            item for item in candidates
            if item["class"] == class_name and not item.get("suppressed_reason")
        ]
        class_items.sort(key=lambda item: item["area_px"], reverse=True)
        selected = class_items[:CLASS_MAX_COUNT[class_name]]
        selected.sort(key=_candidate_sort_key)
        for item in selected:
            det = dict(item)
            det["name"] = class_name
            det["label"] = CLASS_LABELS[class_name]
            detections.append(det)
    detections.sort(key=lambda item: item["name"])
    return detections


def _extract_candidates(color, depth, info, target_frame, min_area, max_area,
                        min_depth, max_depth, tf_timeout):
    model = _camera_model(info)
    listener = tf.TransformListener()
    rospy.sleep(0.2)
    masks = _make_masks(color)
    candidates = []
    xyz_key = "{}_xyz_m".format(target_frame)

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
                depth, contour, u, v, min_depth, max_depth,
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
            # Position base_link obligatoire (sinon inutilisable pour le pick)
            if base_xyz is None:
                continue
            bx, by, bz = [float(value) for value in base_xyz]
            if not (BASE_VALID_X[0] <= bx <= BASE_VALID_X[1]):
                continue
            if not (BASE_VALID_Y[0] <= by <= BASE_VALID_Y[1]):
                continue
            if not (BASE_VALID_Z[0] <= bz <= BASE_VALID_Z[1]):
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
                "camera_xyz_m": [round(float(p), 4) for p in camera_xyz],
                xyz_key: [round(float(p), 4) for p in base_xyz],
                "base_link_xyz_m": [round(float(p), 4) for p in base_xyz],
            })

    candidates.sort(key=lambda item: (item["class"], item["pixel"][1], item["pixel"][0]))
    detections = _select_final_detections(candidates)
    return candidates, detections, masks


def _draw_overlay(color, candidates, detections):
    overlay = color.copy()
    colors = {
        "parcel_1": (80, 120, 180),   # brown-ish BGR
        "parcel_2": (40, 220, 220),   # yellow
        "parcel_3": (40, 120, 255),   # orange
        "parcel_4": (220, 160, 80),   # blue
    }
    for item in candidates:
        x, y, w, h = item["bbox"]
        color_bgr = colors.get(item["class"], (0, 255, 255))
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color_bgr, 1)
        label = "{} {}".format(CLASS_LABELS.get(item["class"], "?"), item["pixel"])
        cv2.putText(overlay, label, (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)
    for item in detections:
        x, y, w, h = item["bbox"]
        color_bgr = colors.get(item["class"], (0, 255, 255))
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color_bgr, 3)
        name = "{} ({})".format(item["name"], item.get("label", ""))
        cv2.putText(overlay, name, (x, min(color.shape[0] - 8, y + h + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
        xyz = item.get("base_link_xyz_m")
        if xyz is not None:
            txt = "xyz=[{:.2f},{:.2f},{:.2f}]".format(xyz[0], xyz[1], xyz[2])
            cv2.putText(overlay, txt, (x, min(color.shape[0] - 8, y + h + 38)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1)
    return overlay


def _write_outputs(output_dir, camera, color, depth, masks, overlay, candidates, detections):
    os.makedirs(output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = os.path.join(output_dir, "scene1_{}_{}".format(camera, stamp))
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


def capture_once(args):
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
    prefix = None
    if args.output_dir:
        prefix = _write_outputs(
            args.output_dir, args.camera, color, depth, masks, overlay, candidates, detections,
        )
    # Positions base_link par colis (ce que la mission utilise pour le pick)
    positions = {}
    for det in detections:
        name = det.get("name")
        xyz = det.get("base_link_xyz_m")
        if name and xyz is not None:
            positions[name] = {
                "label": det.get("label"),
                "base_link_xyz_m": list(xyz),
                "pixel": det.get("pixel"),
                "depth_m": det.get("depth_m"),
            }
    return {
        "camera": args.camera,
        "topics": topics,
        "output_prefix": prefix,
        "candidate_count": len(candidates),
        "detection_count": len(detections),
        "expected_parcels": list(PARCEL_ORDER),
        "found_parcels": [d["name"] for d in detections],
        "missing_parcels": [n for n in PARCEL_ORDER if n not in {d["name"] for d in detections}],
        "positions": positions,
        "candidates": candidates,
        "detections": detections,
    }


def detect_parcels(lidar=None, cam=None, tf_reader=None, log=None):
    """
    Point d'entrée mission : RGB-D tête → liste {name, color, center, ...}.
    lidar/cam/tf_reader ignorés (API historique scene1_task).
    """
    def _log(msg, *args):
        if log is not None:
            log(msg, *args)
        else:
            rospy.loginfo(msg if not args else (msg % args))

    args = argparse.Namespace(
        camera="head",
        target_frame="base_link",
        output_dir=_default_output_dir("perception"),
        timeout=10.0,
        tf_timeout=0.8,
        min_area=100.0,
        max_area=80000.0,
        min_depth=0.30,
        max_depth=1.20,
    )
    _log("[DETECT] backend=rgb-d (tête)")
    result = capture_once(args)
    parcels = []
    try:
        from scene1.config import TABLE_PARCEL_Z
        z_table = float(TABLE_PARCEL_Z)
    except Exception:
        z_table = -0.04
    for det in result.get("detections") or []:
        name = det.get("name")
        xyz = det.get("base_link_xyz_m")
        if not name or xyz is None or len(xyz) < 3:
            continue
        cx, cy, cz = float(xyz[0]), float(xyz[1]), float(xyz[2])
        if cz > 0.10 or cz < -0.12:
            cz = z_table
        parcels.append({
            "name": name,
            "color": det.get("label") or "?",
            "center": (cx, cy, cz),
            "center_raw": (float(xyz[0]), float(xyz[1]), float(xyz[2])),
            "size_xy": (0.06, 0.06),
            "n_points": int(det.get("area_px") or 0),
            "source": "rgb-depth",
            "pixel": det.get("pixel"),
            "depth_m": det.get("depth_m"),
        })
    for p in parcels:
        cx, cy, cz = p["center"]
        _log("[DETECT] %s (%s): center=(%.3f, %.3f, %.3f)",
             p["name"], p.get("color", "?"), cx, cy, cz)
    _log("[DETECT] terminé, %d colis", len(parcels))
    return parcels


def run_scene1_perception_only(arm, head, log):
    """Mode debug : tête + détection, pas de bras."""
    import rospy
    from scene1.config import HEAD_LOOK_YAW, HEAD_LOOK_PITCH, HEAD_SETTLE_SEC
    try:
        head.look_at(HEAD_LOOK_YAW, HEAD_LOOK_PITCH)
        rospy.sleep(HEAD_SETTLE_SEC)
    except Exception as exc:
        log("[PERCEPT] head: %s", exc)
    parcels = detect_parcels(log=log)
    log("[PERCEPT] %d colis détectés", len(parcels))
    return parcels


def main():
    parser = argparse.ArgumentParser(description="Scene1 RGB-D parcel perception debug")
    parser.add_argument("--camera", choices=sorted(CAMERA_TOPICS), default="head")
    parser.add_argument("--target-frame", default="base_link")
    parser.add_argument("--output-dir", default=_default_output_dir("perception"))
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--tf-timeout", type=float, default=0.8)
    parser.add_argument("--min-area", type=float, default=100.0)
    parser.add_argument("--max-area", type=float, default=80000.0)
    parser.add_argument("--min-depth", type=float, default=0.30)
    parser.add_argument("--max-depth", type=float, default=1.20)
    args = parser.parse_args()

    rospy.init_node("scene1_perception_debug", anonymous=True)
    result = capture_once(args)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
