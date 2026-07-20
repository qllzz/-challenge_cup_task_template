#!/usr/bin/env python3
"""
Prise Scene1 via caméra MAIN droite (cam_r).

1) Descente par paliers + centrage image
2) Si depth trop petit (main bloquée sur le colis) → RECULER pour pouvoir redescendre
3) Avance = écart horizontal tip → centre colis (pas seulement depth)
   → corrige le cas « juste à côté » (ex. jaune) où depth dit OK mais tip court

Désactiver : SCENE1_ENABLE_WRIST_REFINE=0
"""
from __future__ import print_function

import importlib.util
import math
import os


def _default_output_dir(name):
    """Store optional wrist-camera diagnostics inside this task package."""
    package_dir = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    return os.path.join(package_dir, "artifacts", "scene1", name)


# --- descente / centrage (nouvelle perception RGB-D) ---
WRIST_LOOK_Z = 0.32
WRIST_Z_STEP = 0.045            # descente par paliers (m)
WRIST_SETTLE_SEC = 0.30
WRIST_CENTER_ITERS = 3          # boucles centrage par palier
WRIST_SERVO_GAIN = 0.70
WRIST_STEP_MAX_XY = 0.05        # max Δxy centrage image
WRIST_ACCEPT_PX = 40.0          # assez centré image pour descendre
WRIST_SANITY_XY_M = 0.28
# Trop près = tip/cam coincé sur le colis → reculer pour libérer la descente
WRIST_BLOCK_DEPTH_M = 0.10      # sous ce depth (m) pendant descente → RECUL
WRIST_MAX_RETREAT_M = 0.06      # recul max (m)
# Avance = distance tip→centre colis en XY (combien avancer pour le milieu)
WRIST_XY_GAP_OK_M = 0.012       # tip assez sur le centre (< 1.2 cm)
WRIST_FIT_GAIN = 0.90           # fraction du gap appliquée (presque tout le manque)
WRIST_MAX_ADVANCE_M = 0.08      # plafond sécurité par pas
WRIST_FIT_ITERS = 4
WRIST_INSERT_BIAS_X = 0.015     # +X tip souvent un peu court à la close


def _scene1_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _load_scene1_perception():
    try:
        from scene1 import perception as mod
        return mod
    except Exception:
        pass
    path = os.path.join(_scene1_dir(), "perception.py")
    if not os.path.isfile(path):
        raise RuntimeError("perception.py introuvable (wrist): %s" % path)
    spec = importlib.util.spec_from_file_location("scene1_perception_wrist", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _xy_dist(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _capture_wrist(output_dir=None, target_frame="base_link",
                   min_depth=0.08, max_depth=0.95):
    """RGB-D cam_r + centre optique (camera_info)."""
    import argparse
    import rospy
    from sensor_msgs.msg import CameraInfo

    if output_dir is None:
        output_dir = _default_output_dir("wrist")
    perception = _load_scene1_perception()
    args = argparse.Namespace(
        camera="right",
        target_frame=target_frame,
        output_dir=output_dir,
        timeout=8.0,
        tf_timeout=0.8,
        min_area=60.0,
        max_area=120000.0,
        min_depth=float(min_depth),
        max_depth=float(max_depth),
    )
    os.makedirs(output_dir, exist_ok=True)
    result = perception.capture_once(args)

    topics = perception.CAMERA_TOPICS["right"]
    try:
        info = rospy.wait_for_message(
            topics["info"], CameraInfo, timeout=float(args.timeout),
        )
        cx = float(info.K[2])
        cy = float(info.K[5])
        if cx <= 1.0 or cy <= 1.0:
            cx = 0.5 * float(info.width)
            cy = 0.5 * float(info.height)
        result["image_cx"] = cx
        result["image_cy"] = cy
        result["image_wh"] = [int(info.width), int(info.height)]
    except Exception as exc:
        rospy.logwarn("[WRIST] camera_info fail (%s) — centre approx", exc)
        result["image_cx"] = 320.0
        result["image_cy"] = 240.0
        result["image_wh"] = [640, 480]
    return result


def _pick_detection_for_center(detections, parcel_name, image_cx, image_cy,
                              ref_xy=None, sanity_xy=WRIST_SANITY_XY_M):
    best = None
    best_dpx = 1e9
    for det in detections or []:
        name = det.get("name") or det.get("class")
        if name != parcel_name:
            continue
        pix = det.get("pixel")
        xyz = det.get("base_link_xyz_m")
        if pix is None or len(pix) < 2 or xyz is None or len(xyz) < 2:
            continue
        if ref_xy is not None and _xy_dist(xyz, ref_xy) > float(sanity_xy):
            continue
        dpx = math.hypot(
            float(pix[0]) - float(image_cx),
            float(pix[1]) - float(image_cy),
        )
        if dpx < best_dpx:
            best_dpx = dpx
            best = det
    return best, best_dpx


def _ik_xy_from_det(sc1, name, det, job, final_z):
    """base_link det → (x,y) pick + z forcé (table ou pad)."""
    bx, by, bz = [float(v) for v in det["base_link_xyz_m"]]
    off = list(sc1.WORLD_TO_IK_OFFSET)
    source_world = [bx - float(off[0]), by - float(off[1]), bz - float(off[2])]
    if job.get("source_world") is not None:
        source_world[2] = float(job["source_world"][2])
    offset = list(sc1._right_pick_offset_for_parcel(name, source_world))
    return [
        bx + float(offset[0]),
        by + float(offset[1]),
        float(final_z),
    ], [bx, by, bz]


def _clamp_step(cur_xy, tgt_xy, gain, step_max):
    dx = (float(tgt_xy[0]) - float(cur_xy[0])) * float(gain)
    dy = (float(tgt_xy[1]) - float(cur_xy[1])) * float(gain)
    d = math.hypot(dx, dy)
    if d > float(step_max) and d > 1e-9:
        s = float(step_max) / d
        dx *= s
        dy *= s
    return [float(cur_xy[0]) + dx, float(cur_xy[1]) + dy]


def _current_right_xyz(sc1):
    cur = sc1._call_fk(
        sc1._read_current_arm_joints(sc1.TOPIC_TIMEOUT),
        sc1.TOPIC_TIMEOUT,
    ).right_pose.pos_xyz
    return [float(cur[0]), float(cur[1]), float(cur[2])]


def _center_loop_at_height(sc1, arm_pub, arm_hold, name, job, quat, xy, z,
                          ref_xy, final_z, output_dir, label_prefix):
    """
    Boucle cam_r à hauteur z fixe : corrige XY jusqu'au milieu image (ou max iters).
    Retourne (xy, last_det, last_dpx, centered).
    """
    import rospy

    q = list(quat)
    last_det = None
    last_dpx = None
    centered = False
    # profondeur : plus bas → objets plus proches cam
    min_d = 0.06 if float(z) < 0.18 else 0.10
    max_d = 0.70 if float(z) < 0.18 else 0.95

    for it in range(int(WRIST_CENTER_ITERS)):
        try:
            rospy.loginfo(
                "[WRIST] %s %s z=%.3f center_it=%d capture…",
                name, label_prefix, z, it,
            )
            result = _capture_wrist(
                output_dir=output_dir, min_depth=min_d, max_depth=max_d,
            )
        except Exception as exc:
            rospy.logwarn("[WRIST] %s capture fail: %s", name, exc)
            break

        cx = float(result.get("image_cx", 320.0))
        cy = float(result.get("image_cy", 240.0))
        det, dpx = _pick_detection_for_center(
            result.get("detections") or [], name, cx, cy,
            ref_xy=ref_xy, sanity_xy=WRIST_SANITY_XY_M,
        )
        if det is None:
            rospy.logwarn(
                "[WRIST] %s z=%.3f it=%d pas vu — found=%s",
                name, z, it, result.get("found_parcels"),
            )
            break

        last_det = det
        last_dpx = dpx
        pix = det.get("pixel") or [cx, cy]
        rospy.loginfo(
            "[WRIST] %s z=%.3f it=%d pixel=(%.0f,%.0f) mid=(%.0f,%.0f) dpx=%.1f",
            name, z, it, float(pix[0]), float(pix[1]), cx, cy, dpx,
        )

        tgt_ik, _ = _ik_xy_from_det(sc1, name, det, job, final_z)
        if dpx <= float(WRIST_ACCEPT_PX):
            centered = True
            xy = [float(tgt_ik[0]), float(tgt_ik[1])]
            rospy.loginfo(
                "[WRIST] %s CENTRÉ dpx=%.1f @ z=%.3f", name, dpx, z,
            )
            break

        try:
            cur = _current_right_xyz(sc1)
            cur_xy = cur[:2]
        except Exception:
            cur_xy = list(xy)

        step_xy = _clamp_step(
            cur_xy, tgt_ik[:2], WRIST_SERVO_GAIN, WRIST_STEP_MAX_XY,
        )
        xy = [float(step_xy[0]), float(step_xy[1])]
        pose = [xy[0], xy[1], float(z)]
        try:
            sc1._move_right_cartesian_to(
                arm_pub, arm_hold, pose, q,
                "%s_%s_c%d" % (name, label_prefix, it),
                n_points=6, seg_time=0.40,
            )
            rospy.sleep(float(WRIST_SETTLE_SEC))
        except Exception as exc:
            rospy.logwarn("[WRIST] %s move center fail: %s", name, exc)
            break

    return xy, last_det, last_dpx, centered


def _det_depth_m(det):
    depth = det.get("depth_m") if det else None
    if depth is None and det and det.get("camera_xyz_m"):
        depth = abs(float(det["camera_xyz_m"][2]))
    return None if depth is None else float(depth)


def _retreat_to_free_descent(sc1, arm_pub, arm_hold, name, quat, xy, z,
                            det, final_z, label):
    """
    Main venue du haut, coincée (depth trop petit) → RECULER pour pouvoir
    redescendre librement. Ne descend pas ici, seulement un recul XY.
    """
    import rospy

    depth = _det_depth_m(det)
    if depth is None or depth >= float(WRIST_BLOCK_DEPTH_M):
        return list(xy), False
    if float(z) <= float(final_z) + 0.02:
        return list(xy), False

    try:
        cur = _current_right_xyz(sc1)
    except Exception:
        cur = [float(xy[0]), float(xy[1]), float(z)]

    bx = float(det["base_link_xyz_m"][0])
    by = float(det["base_link_xyz_m"][1])
    # Reculer = s'éloigner du centre colis (opposé tip→colis)
    dx = float(cur[0]) - bx
    dy = float(cur[1]) - by
    n = math.hypot(dx, dy)
    if n < 0.01:
        ux, uy = -1.0, 0.0  # défaut : recul -X
    else:
        ux, uy = dx / n, dy / n

    # Plus on est collé (depth petit), plus on recule
    need = float(WRIST_BLOCK_DEPTH_M) - float(depth) + 0.02
    retreat = min(max(need, 0.025), float(WRIST_MAX_RETREAT_M))
    new_xy = [float(cur[0]) + ux * retreat, float(cur[1]) + uy * retreat]
    rospy.loginfo(
        "[WRIST] %s BLOQUÉ depth=%.3f < %.3f @ z=%.3f → RECUL %.1fcm pour redescendre",
        name, depth, WRIST_BLOCK_DEPTH_M, z, retreat * 100.0,
    )
    try:
        sc1._move_right_cartesian_to(
            arm_pub, arm_hold, [new_xy[0], new_xy[1], float(z)], list(quat),
            label, n_points=8, seg_time=0.45,
        )
        rospy.sleep(float(WRIST_SETTLE_SEC))
    except Exception as exc:
        rospy.logwarn("[WRIST] %s retreat fail: %s", name, exc)
        return list(xy), False
    return new_xy, True


def _align_tip_to_parcel_mid(sc1, arm_pub, arm_hold, name, job, quat,
                            pick_ik, ref_xy, final_z, output_dir):
    """
    Combien avancer ? = écart horizontal tip FK → centre colis (pick_ik).

    Cas jaune « juste à côté » : l'image peut être OK / depth OK, mais tip
    encore à 2–5 cm du centre → on avance de (presque) ce gap.

    Recul seulement si depth < BLOCK (coincé), pour libérer — pas pour
    viser une profondeur cible.
    """
    import rospy

    q = list(quat)
    ik = list(pick_ik)
    last_det = None
    last_depth = None
    gap_ok = float(WRIST_XY_GAP_OK_M)

    for it in range(int(WRIST_FIT_ITERS)):
        try:
            result = _capture_wrist(
                output_dir=output_dir, min_depth=0.05, max_depth=0.70,
            )
        except Exception as exc:
            rospy.logwarn("[WRIST] %s align capture fail: %s", name, exc)
            break

        cx = float(result.get("image_cx", 320.0))
        cy = float(result.get("image_cy", 240.0))
        det, dpx = _pick_detection_for_center(
            result.get("detections") or [], name, cx, cy,
            ref_xy=ref_xy, sanity_xy=WRIST_SANITY_XY_M,
        )
        if det is None:
            rospy.logwarn(
                "[WRIST] %s align it=%d pas vu — found=%s",
                name, it, result.get("found_parcels"),
            )
            break

        last_det = det
        last_depth = _det_depth_m(det)
        tgt_ik, _ = _ik_xy_from_det(sc1, name, det, job, final_z)
        try:
            cur = _current_right_xyz(sc1)
        except Exception:
            cur = list(ik)

        # Gap = combien il manque pour être au milieu (XY tip → cible prise)
        gap_x = float(tgt_ik[0]) - float(cur[0])
        gap_y = float(tgt_ik[1]) - float(cur[1])
        gap = math.hypot(gap_x, gap_y)

        blocked = (
            last_depth is not None
            and last_depth < float(WRIST_BLOCK_DEPTH_M)
        )

        rospy.loginfo(
            "[WRIST] %s ALIGN it=%d gap_xy=%.1fcm dpx=%.1f depth=%s → %s",
            name, it, gap * 100.0, dpx,
            ("%.3f" % last_depth) if last_depth is not None else "?",
            ("RECUL" if blocked else ("AVANCE" if gap > gap_ok else "OK")),
        )

        # 1) Coincé trop près → reculer pour ne pas rester sur le colis
        if blocked:
            new_xy, did = _retreat_to_free_descent(
                sc1, arm_pub, arm_hold, name, q, cur[:2], cur[2],
                det, final_z, "%s_align_retreat_%d" % (name, it),
            )
            if did:
                ik = [new_xy[0], new_xy[1], float(final_z)]
                ref_xy = list(new_xy)
                continue

        # 2) Tip pas au milieu → avancer du gap (presque tout)
        if gap <= gap_ok and dpx <= float(WRIST_ACCEPT_PX):
            ik = [
                float(tgt_ik[0]) + float(WRIST_INSERT_BIAS_X),
                float(tgt_ik[1]),
                float(final_z),
            ]
            rospy.loginfo(
                "[WRIST] %s ALIGN OK gap=%.1fcm — tip au milieu, prêt close",
                name, gap * 100.0,
            )
            break

        # Avance = fraction du gap réel (réponse à « de combien ? »)
        step = min(gap * float(WRIST_FIT_GAIN), float(WRIST_MAX_ADVANCE_M))
        if gap < 1e-6:
            ux, uy = 1.0, 0.0
        else:
            ux, uy = gap_x / gap, gap_y / gap

        ik = [
            float(cur[0]) + ux * step + float(WRIST_INSERT_BIAS_X),
            float(cur[1]) + uy * step,
            float(final_z),
        ]
        rospy.loginfo(
            "[WRIST] %s AVANCE %.1fcm / gap %.1fcm → ik=(%.3f,%.3f,%.3f)",
            name, step * 100.0, gap * 100.0, ik[0], ik[1], ik[2],
        )
        try:
            sc1._move_right_cartesian_to(
                arm_pub, arm_hold, ik, q,
                "%s_wrist_align_%d" % (name, it),
                n_points=8, seg_time=0.45,
            )
            rospy.sleep(float(WRIST_SETTLE_SEC))
        except Exception as exc:
            rospy.logwarn("[WRIST] %s align move fail: %s", name, exc)
            break

        ref_xy = [float(ik[0]), float(ik[1])]

    return ik, last_det, last_depth


def servo_descend_to_mid_parcel(sc1, arm_pub, arm_hold, job, pick_quat,
                                final_z=None, start_z=None,
                                output_dir=None):
    """
    Descente progressive : à chaque palier, boucle de centrage milieu colis.

    Returns job mis à jour (right_pick_ik = xy final + final_z).
    Ne ferme pas la pince — l'appelant close/lift.
    """
    import rospy

    if os.environ.get("SCENE1_ENABLE_WRIST_REFINE", "1") != "1":
        rospy.loginfo("[WRIST] désactivé — pas de servo descente")
        return job

    name = job.get("object")
    old_ik = list(job["right_pick_ik"])
    ref_xy = list(old_ik[:2])
    q = list(pick_quat)

    if final_z is None:
        final_z = float(old_ik[2])
    else:
        final_z = float(final_z)

    if start_z is None:
        start_z = max(
            float(WRIST_LOOK_Z),
            float(getattr(sc1, "RIGHT_PICK_TRANSIT_IK_Z", WRIST_LOOK_Z) or WRIST_LOOK_Z),
        )
    start_z = float(start_z)
    if start_z < final_z + 0.02:
        start_z = final_z + 0.12

    xy = [float(old_ik[0]), float(old_ik[1])]
    z = float(start_z)

    rospy.loginfo(
        "[WRIST] %s SERVO descente z=%.3f → %.3f pas=%.3f (centrage chaque palier)",
        name, z, final_z, WRIST_Z_STEP,
    )

    # Aller au premier palier
    try:
        sc1._move_right_cartesian_to(
            arm_pub, arm_hold, [xy[0], xy[1], z], q,
            "%s_wrist_start" % name,
            n_points=10, seg_time=0.50,
        )
        rospy.sleep(float(WRIST_SETTLE_SEC))
    except Exception as exc:
        rospy.logwarn("[WRIST] %s start fail: %s", name, exc)
        return job

    last_det = None
    last_dpx = None
    stage = 0

    while True:
        stage += 1
        xy, det, dpx, centered = _center_loop_at_height(
            sc1, arm_pub, arm_hold, name, job, q, xy, z,
            ref_xy, final_z, output_dir, "stg%d" % stage,
        )
        if det is not None:
            last_det = det
            last_dpx = dpx
            ref_xy = [float(xy[0]), float(xy[1])]
            # Coincé sur le colis en descendant → reculer puis continuer ↓
            xy, retreated = _retreat_to_free_descent(
                sc1, arm_pub, arm_hold, name, q, xy, z, det, final_z,
                "%s_retreat_%d" % (name, stage),
            )
            if retreated:
                ref_xy = list(xy)

        if z <= final_z + 0.012:
            rospy.loginfo(
                "[WRIST] %s bas atteint z=%.3f centered=%s dpx=%s",
                name, z, centered,
                ("%.1f" % last_dpx) if last_dpx is not None else "?",
            )
            break

        next_z = max(float(final_z), float(z) - float(WRIST_Z_STEP))
        rospy.loginfo(
            "[WRIST] %s palier ↓ z=%.3f → %.3f (xy=%.3f,%.3f)",
            name, z, next_z, xy[0], xy[1],
        )
        z = next_z
        try:
            sc1._move_right_cartesian_to(
                arm_pub, arm_hold, [xy[0], xy[1], z], q,
                "%s_wrist_down_%d" % (name, stage),
                n_points=8, seg_time=0.45,
            )
            rospy.sleep(float(WRIST_SETTLE_SEC))
        except Exception as exc:
            rospy.logwarn("[WRIST] %s down fail: %s — stop servo", name, exc)
            break

    # Dernier centrage tout en bas
    xy, det, dpx, centered = _center_loop_at_height(
        sc1, arm_pub, arm_hold, name, job, q, xy, z,
        ref_xy, final_z, output_dir, "final",
    )
    if det is not None:
        last_det = det
        last_dpx = dpx

    new_ik = [float(xy[0]), float(xy[1]), float(final_z)]
    if last_det is not None:
        tgt, bxyz = _ik_xy_from_det(sc1, name, last_det, job, final_z)
        new_ik = [float(tgt[0]), float(tgt[1]), float(final_z)]
    else:
        bxyz = [float(xy[0]), float(xy[1]), float(final_z)]

    # Avance = gap tip→centre (corrige « juste à côté » type jaune)
    insert_depth = None
    try:
        new_ik, ins_det, insert_depth = _align_tip_to_parcel_mid(
            sc1, arm_pub, arm_hold, name, job, q,
            new_ik, ref_xy, final_z, output_dir,
        )
        if ins_det is not None:
            last_det = ins_det
            bxyz = [float(v) for v in ins_det["base_link_xyz_m"]]
    except Exception as exc:
        rospy.logwarn("[WRIST] %s align skip: %s", name, exc)

    # Pose finale précise avant close
    try:
        sc1._move_right_cartesian_to(
            arm_pub, arm_hold, new_ik, q, "%s_wrist_grasp_pose" % name,
            n_points=8, seg_time=0.40,
        )
        rospy.sleep(0.25)
    except Exception as exc:
        rospy.logwarn("[WRIST] %s grasp pose: %s", name, exc)

    out = dict(job)
    out["right_pick_ik"] = list(new_ik)
    out["wrist_refined"] = True
    out["wrist_centered"] = bool(centered) if last_det is not None else False
    out["wrist_dpx"] = None if last_dpx is None else float(last_dpx)
    out["wrist_depth_m"] = insert_depth
    out["wrist_det_xy"] = [float(bxyz[0]), float(bxyz[1])]
    out["wrist_delta_xy"] = math.hypot(
        float(new_ik[0]) - float(old_ik[0]),
        float(new_ik[1]) - float(old_ik[1]),
    )
    perc = dict(out.get("perception") or {})
    perc["wrist_base_link_xyz_m"] = list(bxyz)
    if last_det is not None:
        perc["wrist_pixel"] = last_det.get("pixel")
        perc["wrist_depth_m"] = last_det.get("depth_m")
    if insert_depth is not None:
        perc["wrist_insert_depth_m"] = float(insert_depth)
    out["perception"] = perc

    rospy.loginfo(
        "[WRIST] %s FIN servo centered=%s dpx=%s depth=%s pick_ik %s → %s",
        name, out["wrist_centered"],
        ("%.1f" % last_dpx) if last_dpx is not None else "?",
        ("%.3f" % insert_depth) if insert_depth is not None else "?",
        [round(v, 3) for v in old_ik],
        [round(v, 3) for v in new_ik],
    )
    return out


def refine_job_pick_with_wrist(sc1, arm_pub, arm_hold, job, pick_quat,
                               output_dir=None):
    """Compat : délègue à la descente progressive (sans close)."""
    return servo_descend_to_mid_parcel(
        sc1, arm_pub, arm_hold, job, pick_quat,
        final_z=float(job["right_pick_ik"][2]),
        output_dir=output_dir,
    )
