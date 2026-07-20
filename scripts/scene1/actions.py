#!/usr/bin/env python3
"""
Actions Scene 1 — mission actuelle.

  1) Jobs depuis détections RGB-D
  2) Prise droite (helper orga, angles orga d'origine)
  3) Pesée + reprise
  4) Rotation taille → tendre D → dépose bac → preset
"""
from __future__ import print_function

import os
import sys

STATIC_SOURCES = {
    "parcel_1": [-0.26, -0.31, 0.880],
    "parcel_2": [-0.26, -0.09, 0.880],
    "parcel_3": [-0.11, -0.31, 0.880],
    "parcel_4": [-0.11, -0.09, 0.880],
}

# Debug : après 1er "main au-dessus du colis" → log angles pince + STOP mission
DEBUG_STOP_ABOVE_FIRST_PARCEL = False  # True = stop + log angles au-dessus du 1er colis

# Flux post-prise (copié de ton scene1/config + actions)
WAIST_BOX_YAW_DEG = 30.0
WAIST_BOX_SETTLE_SEC = 2.4
RIGHT_BOX_EXTEND_MID = [0.42, 0.05, 0.50]
RIGHT_BOX_EXTEND_TRIES = [
    [0.55, 0.12, 0.52],
    [0.60, 0.20, 0.52],
    [0.65, 0.24, 0.50],
]

# Au-dessus des colis (orga TRANSIT=0.12 rase la table → bouscule)
CLEARANCE_IK_Z = 0.42
# Retour origin : plus haut encore — corridor aérien à l'abri des colis
PRESET_CLEAR_Z = 0.55
# Approche prise : haut, puis descente verticale sur le centre
PICK_APPROACH_IK_Z = 0.32
# Hauteur prise (IK z) — orga=-0.03 trop haut → main vide (seed40 parcel_2).
PICK_IK_Z_BASE = -0.055
# Retry après MAIN VIDE : ouvrir + descendre encore (m)
PICK_RETRY_DZ = 0.018
PICK_IK_Z_MIN = -0.095  # ne pas aller trop bas (IK fail / table)
# Essais par colis (puis passage au suivant)
PICK_MAX_ATTEMPTS = 5
# Offset vision = offsets orga (FK ≠ tip TCP ~5–6 cm en x).
VISION_PICK_CENTER_OFFSET = None  # None → sc1._right_pick_offset_for_parcel

# Zone table réelle (seed17: det orange y=-0.44 = près balance → bras bouscule)
TABLE_VALID_X = (0.18, 0.55)
TABLE_VALID_Y = (-0.38, 0.02)
TABLE_VALID_Z = (-0.05, 0.12)

# Retract près corps — TOUJOURS à z haut (jamais raser la table)
SAFE_RETRACT_IK = [0.26, 0.12, 0.55]
# Cartésien lent (anti-jet / anti-secousse colis)
SLOW_CART_POINTS = 16
SLOW_CART_SEG = 0.75
SLOW_BOX_MOVE_TIME = 2.4
SLOW_BOX_SETTLE = 0.85
# Dépose calme (balance / bac) — pose puis attend, ne pas « jeter »
# Orga : WEIGH_RELEASE_SETTLE_BEFORE_OPEN=1.5, dwell≈1.0 — on reste ≥ orga
CALM_PLACE_SETTLE_BEFORE_OPEN = 1.5
CALM_PLACE_DWELL_AFTER_OPEN = 1.2
CALM_BOX_HOVER_BEFORE_OPEN = 1.2
# Reprise balance : essais + redétect vision si main vide (anti-/mujoco/qpos)
WEIGH_REGRASP_MAX_TRIES = 3
# Zone IK autour du pad orga [0.396, -0.574]
WEIGH_ZONE_IK_X = (0.28, 0.55)
WEIGH_ZONE_IK_Y = (-0.72, -0.42)
# Offsets XY si vision rate (m)
WEIGH_REGRASP_XY_OFFSETS = (
    (0.0, 0.0),
    (0.03, 0.0),
    (-0.03, 0.0),
    (0.0, 0.03),
    (0.0, -0.03),
)

_waist_enable_pub = None
_waist_cmd_pub = None


def _scene1_output_dir(name):
    """Keep optional Scene 1 diagnostics in the task workspace."""
    package_dir = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    return os.path.join(package_dir, "artifacts", "scene1", name)


def _add_helper_path(*parts):
    import rospkg
    path = os.path.join(rospkg.RosPack().get_path("challenge_cup_simulator"), *parts)
    if path not in sys.path:
        sys.path.insert(0, path)
    return path


def _ik_to_world(sc1, ik_xyz):
    offset = list(sc1.WORLD_TO_IK_OFFSET)
    return [float(ik_xyz[i]) - float(offset[i]) for i in range(3)]


def pick_offset_for_detection(sc1, name, base_xyz):
    """
    Vision XY + offset tip orga (même calibration que baseline).
    Sans offset: FK vise le centre mais le tip TCP est ~+5 cm en x → rate.
    """
    source_world = _ik_to_world(sc1, list(base_xyz))
    if VISION_PICK_CENTER_OFFSET is not None:
        return [float(VISION_PICK_CENTER_OFFSET[i]) for i in range(3)]
    return list(sc1._right_pick_offset_for_parcel(name, source_world))


def _det_on_table(xyz):
    """Rejette les faux positifs hors table (près balance / sous table)."""
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    return (
        TABLE_VALID_X[0] <= x <= TABLE_VALID_X[1]
        and TABLE_VALID_Y[0] <= y <= TABLE_VALID_Y[1]
        and TABLE_VALID_Z[0] <= z <= TABLE_VALID_Z[1]
    )


def _retract_safe(sc1, arm_pub, arm_hold, quat, label="safe_retract"):
    """
    Corridor aérien : monte HAUT d'abord, puis ramène près du corps
    à z constant (jamais de trajectoire basse au-dessus des colis).
    """
    import rospy

    q = list(quat) if quat is not None else _current_right_quat(sc1)
    z_air = float(PRESET_CLEAR_Z)
    target = [
        float(SAFE_RETRACT_IK[0]),
        float(SAFE_RETRACT_IK[1]),
        z_air,
    ]
    # 1) Monter d'abord où on est (au-dessus des colis)
    _raise_clear_above(
        sc1, arm_pub, arm_hold, quat=q,
        clear_z=z_air,
        label=label + "_raise",
    )
    # 2) Transfert horizontal à hauteur fixe (abri colis)
    rospy.loginfo(
        "[CLEAR] retract HAUTEUR z=%.2f → (%.2f,%.2f) — abri colis",
        z_air, target[0], target[1],
    )
    try:
        sc1._move_right_cartesian_to(
            arm_pub, arm_hold, target, q, label,
            n_points=SLOW_CART_POINTS, seg_time=SLOW_CART_SEG,
        )
        rospy.sleep(0.40)
        return True
    except Exception as exc:
        rospy.logwarn("[CLEAR] retract skip: %s", exc)
        return False


def jobs_from_detections(sc1, detections, fallback_static=None):
    import rospy

    if fallback_static is None:
        fallback_static = STATIC_SOURCES

    by_name = {}
    for det in detections:
        name = det.get("name") or det.get("class")
        xyz = det.get("base_link_xyz_m")
        if name and xyz is not None and len(xyz) >= 3:
            if not _det_on_table(xyz):
                rospy.logwarn(
                    "scene1 pick REJECT %s hors table det=(%.3f,%.3f,%.3f) → fallback",
                    name, float(xyz[0]), float(xyz[1]), float(xyz[2]),
                )
                continue
            by_name[name] = det

    jobs = []
    for name in sc1.PARCEL_NAMES:
        det = by_name.get(name)
        if det is not None:
            bx, by, bz = [float(v) for v in det["base_link_xyz_m"]]
            source_world = _ik_to_world(sc1, [bx, by, bz])
            source_world[2] = float(fallback_static[name][2])
            source_ik = [bx, by, float(sc1._world_to_ik(source_world)[2])]
            offset = pick_offset_for_detection(sc1, name, [bx, by, bz])
            # Viser le centre détection → les 2 pinces enjambent le colis
            right_pick_ik = [
                bx + offset[0],
                by + offset[1],
                float(sc1.RIGHT_PICK_IK_Z) + offset[2],
            ]
            rospy.loginfo(
                "scene1 pick VISION %s (%s) det=(%.3f,%.3f,%.3f) off=(%.3f,%.3f,%.3f) → pick_ik=(%.3f,%.3f,%.3f)",
                name, det.get("label"), bx, by, bz,
                offset[0], offset[1], offset[2],
                right_pick_ik[0], right_pick_ik[1], right_pick_ik[2],
            )
        else:
            source_world = list(fallback_static[name])
            source_ik = sc1._world_to_ik(source_world)
            offset = sc1._right_pick_offset_for_parcel(name, source_world)
            right_pick_ik = sc1._right_pick_ik_from_source_world(name, source_world)
            rospy.logwarn(
                "scene1 pick FALLBACK %s pick_ik=%s",
                name, [round(v, 3) for v in right_pick_ik],
            )

        jobs.append({
            "object": name,
            "source_world": list(source_world),
            "source_ik": list(source_ik),
            "right_pick_offset": list(offset),
            "right_pick_ik": list(right_pick_ik),
            "weigh_ik": list(sc1.WEIGH_RELEASE_IK),
            "box_drop_offset": sc1._box_drop_offset_for_parcel(name),
            "box_ik": sc1._box_drop_ik_for_parcel(name),
            "perception": None if det is None else {
                "label": det.get("label"),
                "base_link_xyz_m": list(det["base_link_xyz_m"]),
                "pixel": det.get("pixel"),
                "area_px": det.get("area_px"),
            },
        })

    jobs.sort(key=lambda j: float(j["right_pick_ik"][0]))
    return jobs


def jobs_from_static(sc1, fallback_static=None):
    if fallback_static is None:
        fallback_static = STATIC_SOURCES
    jobs = []
    for name in sc1.PARCEL_NAMES:
        source_world = fallback_static[name]
        jobs.append({
            "object": name,
            "source_world": list(source_world),
            "source_ik": sc1._world_to_ik(source_world),
            "right_pick_offset": sc1._right_pick_offset_for_parcel(name, source_world),
            "right_pick_ik": sc1._right_pick_ik_from_source_world(name, source_world),
            "weigh_ik": list(sc1.WEIGH_RELEASE_IK),
            "box_drop_offset": sc1._box_drop_offset_for_parcel(name),
            "box_ik": sc1._box_drop_ik_for_parcel(name),
        })
    return jobs


def _current_right_xyz(sc1):
    current = sc1._read_current_arm_joints(sc1.TOPIC_TIMEOUT)
    return list(sc1._call_fk(current, sc1.TOPIC_TIMEOUT).right_pose.pos_xyz)


def _current_right_quat(sc1):
    current = sc1._read_current_arm_joints(sc1.TOPIC_TIMEOUT)
    return list(sc1._call_fk(current, sc1.TOPIC_TIMEOUT).right_pose.quat_xyzw)


def _raise_clear_above(sc1, arm_pub, arm_hold, quat=None, clear_z=None, label="clear_raise"):
    """Monte la main D à clear_z (xy courant) — au-dessus des colis avant tout retour."""
    import rospy

    z_clear = float(CLEARANCE_IK_Z if clear_z is None else clear_z)
    try:
        cur = _current_right_xyz(sc1)
        q = list(quat) if quat is not None else _current_right_quat(sc1)
        if float(cur[2]) >= z_clear - 0.02:
            rospy.loginfo("[CLEAR] déjà haut z=%.3f", cur[2])
            return True
        high = [float(cur[0]), float(cur[1]), z_clear]
        rospy.loginfo(
            "[CLEAR] monte z=%.3f→%.3f (anti-bousculade colis)",
            cur[2], z_clear,
        )
        sc1._move_right_cartesian_to(
            arm_pub, arm_hold, high, q, label,
            n_points=SLOW_CART_POINTS, seg_time=SLOW_CART_SEG,
        )
        rospy.sleep(0.30)
        return True
    except Exception as exc:
        rospy.logwarn("[CLEAR] raise skip: %s", exc)
        return False


def _safe_return_preset(sc1, arm_pub, arm_hold, quat=None):
    """
    Retour origin À L'ABRI DES COLIS :
      1) monte z=PRESET_CLEAR_Z (où on est)
      2) ramène près du corps à z constant (pas de balayage table)
      3) joints → preset FINAL seulement (évite le pic d'abduction orga
         qui balaye la table si on part depuis la zone colis)
    """
    import rospy

    q = list(quat) if quat is not None else _current_right_quat(sc1)
    sc1.PRESET_SEGMENT_TIME = max(
        float(getattr(sc1, "PRESET_SEGMENT_TIME", 1.92) or 1.92), 2.2,
    )
    sc1.PRESET_SETTLE_TIME = max(
        float(getattr(sc1, "PRESET_SETTLE_TIME", 0.5) or 0.5), 0.6,
    )

    # Corridor aérien haut → corps
    _retract_safe(sc1, arm_pub, arm_hold, q, label="retract_before_preset")
    _raise_clear_above(
        sc1, arm_pub, arm_hold, quat=q,
        clear_z=PRESET_CLEAR_Z, label="clear_before_preset",
    )
    rospy.sleep(0.40)

    # Preset complet orga (= pics 90°) depuis la table = bousculade.
    # Depuis retract corps : aller direct au point FINAL lentement.
    home = list(sc1.PRESET_POINTS_DEG[-1])
    rospy.loginfo(
        "[PRESET] retour HAUTEUR puis joints final (pas de sweep abduction)",
    )
    try:
        sc1._move_arm_to(
            arm_pub,
            arm_hold,
            home,
            move_time=max(2.4, float(sc1.PRESET_SEGMENT_TIME)),
            settle=float(sc1.PRESET_SETTLE_TIME),
        )
    except Exception as exc:
        rospy.logwarn("[PRESET] move_arm_to final fail (%s) — fallback spline", exc)
        sc1._move_through_preset(arm_pub, arm_hold)
    rospy.sleep(float(sc1.PRESET_SETTLE_TIME))


def _quat_angle_error_deg(actual_xyzw, desired_xyzw):
    """Angle between two quats (deg), same convention as orga collect."""
    import math

    a = [float(v) for v in actual_xyzw]
    b = [float(v) for v in desired_xyzw]
    # q_err = a * conj(b) → angle = 2*acos(|w|)
    # conj(b) = (-x,-y,-z,w)
    ax, ay, az, aw = a
    bx, by, bz, bw = -b[0], -b[1], -b[2], b[3]
    w = aw * bw - ax * bx - ay * by - az * bz
    w = max(-1.0, min(1.0, abs(w)))
    return math.degrees(2.0 * math.acos(w))


def _debug_dump_gripper_angles(sc1, commanded_quat, pick_ik, label="above"):
    """Log quat commandé + FK réel + 7 joints bras D (deg) — pour lire l'orientation pince."""
    import math
    import rospy

    q_cmd = [float(v) for v in commanded_quat]
    q_fk = _current_right_quat(sc1)
    xyz = _current_right_xyz(sc1)
    joints_rad = sc1._read_current_arm_joints(sc1.TOPIC_TIMEOUT)
    right7 = [math.degrees(float(joints_rad[i])) for i in range(7, 14)]
    try:
        err_deg = _quat_angle_error_deg(q_fk, q_cmd)
    except Exception:
        err_deg = -1.0
    approach_z = max(
        float(PICK_APPROACH_IK_Z),
        float(getattr(sc1, "RIGHT_PICK_TRANSIT_IK_Z", PICK_APPROACH_IK_Z) or PICK_APPROACH_IK_Z),
    )
    rospy.loginfo("=" * 60)
    rospy.loginfo("[DEBUG-ANGLE] STOP au-dessus colis (%s)", label)
    rospy.loginfo(
        "[DEBUG-ANGLE] pick_ik cible (dessus)=[%.3f, %.3f, %.3f]",
        float(pick_ik[0]), float(pick_ik[1]), approach_z,
    )
    rospy.loginfo(
        "[DEBUG-ANGLE] xyz FK main D = [%.4f, %.4f, %.4f]",
        xyz[0], xyz[1], xyz[2],
    )
    rospy.loginfo(
        "[DEBUG-ANGLE] quat COMMANDÉ (orga pick) xyzw = [%.4f, %.4f, %.4f, %.4f]",
        q_cmd[0], q_cmd[1], q_cmd[2], q_cmd[3],
    )
    rospy.loginfo(
        "[DEBUG-ANGLE] quat FK RÉEL         xyzw = [%.4f, %.4f, %.4f, %.4f]",
        q_fk[0], q_fk[1], q_fk[2], q_fk[3],
    )
    rospy.loginfo("[DEBUG-ANGLE] écart ori cmd↔FK = %.1f deg", err_deg)
    rospy.loginfo(
        "[DEBUG-ANGLE] orga YPR commande = 1er[0, -90, 0]  2e[90, 0, 0]  (flat orga)",
    )
    rospy.loginfo(
        "[DEBUG-ANGLE] joints bras D (deg) j0..j6 = [%s]",
        ", ".join("%.2f" % v for v in right7),
    )
    rospy.loginfo(
        "[DEBUG-ANGLE] poignet D j4,j5,j6 (deg) = [%.2f, %.2f, %.2f]",
        right7[4], right7[5], right7[6],
    )
    rospy.loginfo("=" * 60)


def _go_above_parcel_centered(sc1, arm_pub, arm_hold, pick_ik, quat):
    """
    Avant prise orga : lever haut, puis se placer pile au-dessus du centre colis
    (pince ouverte, angles orga). Descente verticale ensuite.
    """
    import rospy

    approach_z = float(
        getattr(sc1, "RIGHT_PICK_TRANSIT_IK_Z", PICK_APPROACH_IK_Z) or PICK_APPROACH_IK_Z
    )
    approach_z = max(approach_z, float(PICK_APPROACH_IK_Z))
    q = list(quat)
    _raise_clear_above(
        sc1, arm_pub, arm_hold, quat=q,
        clear_z=max(approach_z, CLEARANCE_IK_Z), label="clear_before_approach",
    )
    above = [float(pick_ik[0]), float(pick_ik[1]), approach_z]
    rospy.loginfo(
        "[APPROACH] au-dessus centre colis (%.3f,%.3f,%.3f)",
        above[0], above[1], above[2],
    )
    sc1._move_right_cartesian_to(
        arm_pub, arm_hold, above, q, "above_parcel_center",
        n_points=SLOW_CART_POINTS, seg_time=SLOW_CART_SEG,
    )
    rospy.sleep(0.40)


def _waist_pubs():
    import rospy
    from std_msgs.msg import Bool
    from kuavo_msgs.msg import robotWaistControl

    global _waist_enable_pub, _waist_cmd_pub
    if _waist_enable_pub is None:
        _waist_enable_pub = rospy.Publisher(
            "/humanoid_controller/enable_waist_control", Bool,
            queue_size=1, latch=True)
        _waist_cmd_pub = rospy.Publisher(
            "/robot_waist_motion_data", robotWaistControl, queue_size=1)
        rospy.sleep(0.3)
    return _waist_enable_pub, _waist_cmd_pub


def _set_waist_yaw_deg(deg, settle=None):
    """Tourne la taille (yaw deg, + = gauche) — même API que ton actions.py."""
    import rospy
    from std_msgs.msg import Bool, Float64MultiArray
    from kuavo_msgs.msg import robotWaistControl

    enable_pub, waist_pub = _waist_pubs()
    enable_pub.publish(Bool(data=True))
    rospy.sleep(0.25)
    msg = robotWaistControl()
    msg.header.stamp = rospy.Time.now()
    msg.data = Float64MultiArray(data=[float(deg)])
    for _ in range(8):
        waist_pub.publish(msg)
        rospy.sleep(0.05)
    wait = float(WAIST_BOX_SETTLE_SEC if settle is None else settle)
    rospy.loginfo("[WAIST] yaw=%.1f deg (settle %.1fs)", deg, wait)
    rospy.sleep(wait)


def _move_right(sc1, arm_pub, arm_hold, xyz, quat, label, slow=False):
    """Une main droite via helper orga — True si OK. slow=True pour bac/extension."""
    import rospy
    try:
        move_t = float(
            SLOW_BOX_MOVE_TIME if slow
            else getattr(sc1, "PICK_ALIGN_MOVE_TIME", 1.5)
        )
        settle_t = float(
            SLOW_BOX_SETTLE if slow
            else getattr(sc1, "SCENE1_ARM_SETTLE_TIME", 0.5)
        )
        sc1._move_hand(
            "right",
            arm_pub,
            arm_hold,
            list(xyz),
            quat,
            label,
            constraint_mode=getattr(sc1, "IK_MODE_THREE_POINT_MIXED", sc1.SCENE1_IK_CONSTRAINT_MODE),
            move_time=move_t,
            settle_time=settle_t,
        )
        if slow:
            rospy.sleep(0.25)
        return True
    except Exception as exc:
        rospy.logwarn("[MOVE] %s FAIL: %s", label, exc)
        return False


def place_in_box_right(sc1, arm_pub, arm_hold, gripper_hold, parcel_name, pick_quat,
                       claw_mon=None):
    """
    Ton flux bac (copié) :
      lever D → tourner taille → tendre D → open → taille 0 → preset
    Refuse si main vide (pos>=85%).
    """
    import rospy

    if claw_mon is None:
        claw_mon = RightClawMonitor()
    # Grip-test avant bac : bloqué = colis → serre ; vide = refuse
    held = _grip_test_and_squeeze(
        sc1, gripper_hold, claw_mon, label="pre_box_%s" % parcel_name,
    )
    if not held and claw_mon._fully_closed_empty() and not claw_mon.likely_holding():
        rospy.logerr(
            "[BOX] %s MAIN VIDE — refuse bac — %s",
            parcel_name, claw_mon.describe_right(),
        )
        return False

    rq = list(pick_quat)
    yaw = float(WAIST_BOX_YAW_DEG)
    mid = list(RIGHT_BOX_EXTEND_MID)
    tries = [list(t) for t in RIGHT_BOX_EXTEND_TRIES]
    rospy.loginfo("[BOX] waist+RIGHT → bac %s (yaw=+%.0f)", parcel_name, yaw)

    # Serre encore AVANT de tourner / tendre
    _gripper_secure_close(sc1, gripper_hold, "right", hold_s=1.8, pulses=4)

    # 1) Lever LENT avant rotation
    try:
        cur = _current_right_xyz(sc1)
        raise_z = max(float(cur[2]) + 0.12, 0.42, float(mid[2]))
        _move_right(
            sc1, arm_pub, arm_hold,
            [float(cur[0]), float(cur[1]), raise_z], rq,
            "right_pre_waist_raise",
            slow=True,
        )
    except Exception as exc:
        rospy.logwarn("[BOX] raise skip: %s", exc)
    gripper_hold.set_right_closed()
    rospy.sleep(0.25)
    gripper_hold.set_right_closed()

    # 2) Tourner taille (settle long)
    _set_waist_yaw_deg(yaw, settle=max(WAIST_BOX_SETTLE_SEC, 2.6))
    gripper_hold.set_right_closed()
    rospy.sleep(0.25)
    gripper_hold.set_right_closed()

    # 3) Étendre D LENT — re-serre à chaque segment
    rospy.loginfo("[BOX] extension LENTE (anti-secousse)")
    _move_right(sc1, arm_pub, arm_hold, mid, rq, "right_box_mid", slow=True)
    gripper_hold.set_right_closed()
    last_ok = None
    for i, xyz in enumerate(tries):
        if _move_right(
            sc1, arm_pub, arm_hold, xyz, rq, "right_box_ext_%d" % i, slow=True,
        ):
            last_ok = list(xyz)
            gripper_hold.set_right_closed()
            rospy.sleep(0.20)
            gripper_hold.set_right_closed()
        else:
            rospy.logwarn("[BOX] extension stop @ try %d", i)
            break

    if last_ok is None:
        rospy.logerr("[BOX] RIGHT extend IK all failed")
        _retract_safe(sc1, arm_pub, arm_hold, rq, label="box_fail_retract")
        _set_waist_yaw_deg(0.0, settle=2.0)
        _safe_return_preset(sc1, arm_pub, arm_hold, quat=rq)
        return False

    # 4) Dépose CALME : hover → attendre → open → laisser tomber → retract lent
    _gripper_secure_close(sc1, gripper_hold, "right", hold_s=1.2, pulses=3)
    rospy.loginfo(
        "[BOX] POSE CALME hover %.1fs puis open @ (%.2f,%.2f,%.2f)",
        float(CALM_BOX_HOVER_BEFORE_OPEN), last_ok[0], last_ok[1], last_ok[2],
    )
    rospy.sleep(float(CALM_BOX_HOVER_BEFORE_OPEN))
    gripper_hold.set_right_open()
    try:
        sc1._wait_for_gripper_open("right", timeout=2.5)
    except Exception:
        pass
    rospy.sleep(float(CALM_PLACE_DWELL_AFTER_OPEN))
    rospy.sleep(float(getattr(sc1, "PLACE_DWELL", 0.5) or 0.5))

    # 5) Retract LENT AVANT waist=0 — sinon bras tendu balaye / jette
    rospy.loginfo("[BOX] retract LENT après dépose")
    _move_right(sc1, arm_pub, arm_hold, mid, rq, "right_box_retract_mid", slow=True)
    rospy.sleep(0.35)
    _retract_safe(sc1, arm_pub, arm_hold, rq, label="box_retract_body")
    _set_waist_yaw_deg(0.0, settle=2.4)
    _safe_return_preset(sc1, arm_pub, arm_hold, quat=rq)
    rospy.loginfo("[BOX] retour preset OK — %s", parcel_name)
    return True


# Lift doux (évite de pousser les autres colis) + détection main vide (ton code claw)
SOFT_LIFT_STAGE1_DZ = 0.07          # 1er stade : +7 cm seulement
SOFT_LIFT_STAGE1_POINTS = 12
SOFT_LIFT_STAGE1_SEG = 0.70
SOFT_LIFT_STAGE2_POINTS = 14
SOFT_LIFT_STAGE2_SEG = 0.75
HOLD_CONFIRM_TIMEOUT = 2.2
EMPTY_HAND_RETRIES = PICK_MAX_ATTEMPTS - 1  # total = PICK_MAX_ATTEMPTS (5)
# pos leju % : >=97 = fermée à fond = VIDE.
# 85 était trop bas → colis compressé = faux MAIN VIDE → open → LÂCHE après prise.
EMPTY_POS_PCT = 97.0
HOLD_POS_MIN = 15.0
HOLD_POS_MAX = 94.0


class RightClawMonitor(object):
    """
    Capteur main droite (/leju_claw_state) :
      - pos% : 0=ouvert … 100=fermé
      - pos partielle + effort → TIENT (colis bloque)
      - pos>=97% → VIDE
    Attention sim : state reste souvent MOVING — ne pas ignorer pos/effort.
    """

    def __init__(self):
        import rospy
        from sensor_msgs.msg import JointState

        self._right_state = -1
        self._right_pos = 0.0       # % 0=open .. 100=closed (leju)
        self._right_effort = 0.0
        self._grip_pos_255 = None   # JointState 0=open .. 255=close
        self._ok_leju = False
        try:
            from kuavo_msgs.msg import lejuClawState
            rospy.Subscriber("/leju_claw_state", lejuClawState, self._on_leju, queue_size=1)
            self._ok_leju = True
        except Exception:
            pass
        rospy.Subscriber("/gripper/state", JointState, self._on_grip, queue_size=1)
        rospy.sleep(0.15)

    def _on_leju(self, msg):
        try:
            if len(msg.state) >= 2:
                self._right_state = int(msg.state[1])
            data = getattr(msg, "data", None)
            if data is not None:
                pos = list(getattr(data, "position", []) or [])
                eff = list(getattr(data, "effort", []) or [])
                if len(pos) >= 2:
                    self._right_pos = float(pos[1])
                if len(eff) >= 2:
                    self._right_effort = float(eff[1])
        except Exception:
            pass

    def _on_grip(self, msg):
        try:
            if "right_gripper_joint" not in msg.name:
                return
            idx = msg.name.index("right_gripper_joint")
            if idx < len(msg.position):
                self._grip_pos_255 = float(msg.position[idx])
            if idx < len(getattr(msg, "effort", []) or []):
                self._right_effort = float(msg.effort[idx])
        except Exception:
            pass

    def _fully_closed_empty(self):
        """Vrai vide = mâchoires à fond. Pos partielle = pas vide."""
        # Priorité leju pos (plus fiable que grip 0/255 foireux en sim)
        if float(self._right_pos) >= float(EMPTY_POS_PCT):
            # Effort fort malgré fermeture → encore possible soft-hold
            if float(self._right_effort) >= 3.5 and float(self._right_pos) < 99.5:
                return False
            return True
        return False

    def likely_holding(self):
        """
        Preuve soft qu'un colis est entre les doigts — même si state=MOVING.
        Log reprise : pos=29% eff=5 = TIENT, pas vide.
        """
        pos = float(self._right_pos)
        eff = float(self._right_effort)
        if self._fully_closed_empty():
            return False
        if int(self._right_state) == 3:
            return True
        if HOLD_POS_MIN <= pos <= HOLD_POS_MAX:
            return True
        if eff >= 1.5 and pos < float(EMPTY_POS_PCT):
            return True
        return False

    def right_holding(self):
        """
        True = tient, False = vide clair, None = encore en mouvement / inconnu.
        """
        if self._fully_closed_empty():
            return False
        if self.likely_holding():
            return True
        rs = int(self._right_state)
        if rs == 1:  # MOVING sans preuve pos/effort → attendre
            return None
        return None

    def describe_right(self):
        names = {0: "UNKNOWN", 1: "MOVING", 2: "REACHED", 3: "GRABBED", -1: "NO_LEJU"}
        g = "?" if self._grip_pos_255 is None else ("%.0f/255" % self._grip_pos_255)
        h = self.right_holding()
        h_s = "?" if h is None else str(h)
        return "R=%s pos=%.0f%% eff=%.2f grip=%s hold=%s" % (
            names.get(int(self._right_state), str(self._right_state)),
            float(self._right_pos), float(self._right_effort), g, h_s,
        )


def _await_right_hold(claw_mon, label, timeout=HOLD_CONFIRM_TIMEOUT):
    """
    HOLD si pos partielle / effort / GRABBED.
    Ne traite PAS MOVING+pos partielle comme vide (bug reprise → lâcher).
    """
    import rospy
    import time

    hits = 0
    empty_hits = 0
    latched = False
    t0 = time.time()
    while time.time() - t0 < float(timeout):
        holding = claw_mon.right_holding()
        # Vide seulement si vraiment fermé à fond (pas si likely_holding)
        if holding is False and claw_mon._fully_closed_empty() and not claw_mon.likely_holding():
            empty_hits += 1
            hits = 0
            latched = False
            if empty_hits >= 4:
                rospy.logwarn("[GRASP] MAIN VIDE %s — %s", label, claw_mon.describe_right())
                return False, False
            rospy.sleep(0.08)
            continue
        if holding is True or claw_mon.likely_holding():
            latched = True
            hits += 1
            empty_hits = 0
            if hits >= 2:
                rospy.loginfo("[GRASP] hold confirm %s — %s", label, claw_mon.describe_right())
                return True, True
        else:
            # None = attendre un peu
            hits = max(0, hits - 1)
        rospy.sleep(0.08)
    # Timeout : si preuve soft (pos/effort) → HOLD, sinon vide
    if latched or claw_mon.likely_holding():
        rospy.loginfo(
            "[GRASP] hold latched/soft %s — %s",
            label, claw_mon.describe_right(),
        )
        return True, True
    if claw_mon._fully_closed_empty():
        rospy.logwarn(
            "[GRASP] MAIN VIDE %s — %s",
            label, claw_mon.describe_right(),
        )
        return False, False
    rospy.logwarn(
        "[GRASP] hold incertains %s — on GARDE fermé — %s",
        label, claw_mon.describe_right(),
    )
    # Ambigu après close → plutôt tenir (évite lâcher un colis saisi)
    return True, False


def _abort_empty_safe(sc1, arm_pub, arm_hold, gripper_hold, pick_quat, label):
    """Main vide : retract loin table PUIS open (jamais open au-dessus des colis)."""
    import rospy

    rospy.logerr("[GRASP] %s MAIN VIDE — retract puis open (pas d'action à vide)", label)
    try:
        _retract_safe(sc1, arm_pub, arm_hold, pick_quat, label="%s_empty_retract" % label)
    except Exception as exc:
        rospy.logwarn("[GRASP] empty retract: %s", exc)
    gripper_hold.set_right_open()
    try:
        sc1._wait_for_gripper_open("right", timeout=2.0)
    except Exception:
        pass
    rospy.sleep(0.3)


def _install_soft_lift_patch(sc1):
    """
    Remplace le lift orga (1 coup brutal → WEIGH_TRANSIT) par :
      stage1 +7cm lent → pause → stage2 transit lent
    Évite de bousculer les autres colis sur la table.
    """
    import rospy

    if getattr(sc1, "_team_soft_lift_patched", False):
        return
    orig = sc1._move_right_cartesian_to

    def _soft_cartesian(arm_pub, arm_hold, target_ik, quat, label,
                        n_points=6, seg_time=0.4, **kwargs):
        label_s = str(label)
        if "lift_straight_up" not in label_s:
            return orig(
                arm_pub, arm_hold, target_ik, quat, label,
                n_points=n_points, seg_time=seg_time, **kwargs
            )

        current = sc1._read_current_arm_joints(sc1.TOPIC_TIMEOUT)
        start = list(sc1._call_fk(current, sc1.TOPIC_TIMEOUT).right_pose.pos_xyz)
        end = [float(v) for v in target_ik]
        # Stage 1 : petit décollage vertical seulement
        mid_z = float(start[2]) + float(SOFT_LIFT_STAGE1_DZ)
        if mid_z > float(end[2]):
            mid_z = float(start[2]) + 0.5 * (float(end[2]) - float(start[2]))
        mid = [float(end[0]), float(end[1]), mid_z]
        rospy.loginfo(
            "[LIFT] soft stage1 %s z=%.3f→%.3f (lent, anti-bousculade)",
            label_s, start[2], mid_z,
        )
        orig(
            arm_pub, arm_hold, mid, quat, label_s + "_soft1",
            n_points=SOFT_LIFT_STAGE1_POINTS,
            seg_time=SOFT_LIFT_STAGE1_SEG,
            **kwargs
        )
        rospy.sleep(0.30)
        rospy.loginfo(
            "[LIFT] soft stage2 %s z=%.3f→%.3f",
            label_s, mid_z, end[2],
        )
        return orig(
            arm_pub, arm_hold, end, quat, label_s + "_soft2",
            n_points=SOFT_LIFT_STAGE2_POINTS,
            seg_time=SOFT_LIFT_STAGE2_SEG,
            **kwargs
        )

    sc1._move_right_cartesian_to = _soft_cartesian
    sc1._team_soft_lift_patched = True
    rospy.loginfo("[LIFT] patch soft 2-stades installé")


def _gripper_max_open(sc1, gripper_hold, side="right"):
    """Ouvre la pince (droite seule) avant prise / entre colis."""
    import rospy

    rospy.loginfo("[GRIP] ouverture MAX (%s) avant prise", side)
    if side == "right":
        gripper_hold.set_right_open()
    else:
        gripper_hold.set_left_open()
    try:
        sc1._wait_for_gripper_open(side, timeout=3.0)
    except Exception as exc:
        rospy.logwarn("[GRIP] wait open: %s — continue", exc)
    rospy.sleep(0.70)


def _gripper_secure_close(sc1, gripper_hold, side="right", hold_s=1.4, pulses=3):
    """
    Serre à fond : plusieurs close + pause (reprise → bac).
    Sans ça la pince reste molle et le colis glisse avant le bac.
    """
    import rospy

    n = max(2, int(pulses))
    slice_s = max(0.35, float(hold_s) / float(n))
    for i in range(n):
        if side == "right":
            gripper_hold.set_right_closed()
        else:
            gripper_hold.set_left_closed()
        rospy.sleep(slice_s)
    if side == "right":
        gripper_hold.set_right_closed()
    else:
        gripper_hold.set_left_closed()
    rospy.loginfo("[GRIP] serre fort %.1fs x%d (%s)", hold_s, n, side)


def _grip_test_and_squeeze(sc1, gripper_hold, claw_mon, label="grip_test"):
    """
    Test prise (avant ET après pesée) :
      1) commande close
      2) si la pince ne peut plus se fermer à fond (pos bloquée / effort)
         → c'est un COLIS → serre fort
      3) si fermée à 97%+ sans effort → VIDE

    Returns True si tient, False si vide.
    """
    import rospy

    # Phase 1 : close + sentir le blocage
    gripper_hold.set_right_closed()
    rospy.sleep(0.55)
    gripper_hold.set_right_closed()
    rospy.sleep(0.45)

    pos = float(claw_mon._right_pos)
    eff = float(claw_mon._right_effort)
    blocked = (
        claw_mon.likely_holding()
        or (HOLD_POS_MIN <= pos <= HOLD_POS_MAX)
        or (eff >= 1.5 and pos < float(EMPTY_POS_PCT))
    )
    empty = claw_mon._fully_closed_empty() and not claw_mon.likely_holding()

    if blocked and not empty:
        rospy.loginfo(
            "[GRIP-TEST] %s BLOQUÉ=colis pos=%.0f%% eff=%.2f → SERRE FORT — %s",
            label, pos, eff, claw_mon.describe_right(),
        )
        _gripper_secure_close(sc1, gripper_hold, "right", hold_s=1.6, pulses=4)
        # Re-check après serre
        if claw_mon.likely_holding() or not claw_mon._fully_closed_empty():
            rospy.loginfo(
                "[GRIP-TEST] %s TIENT OK — %s",
                label, claw_mon.describe_right(),
            )
            return True
        # Soft : on a senti un blocage → garder serré
        gripper_hold.set_right_closed()
        return True

    if empty:
        rospy.logwarn(
            "[GRIP-TEST] %s VIDE (ferme à fond) — %s",
            label, claw_mon.describe_right(),
        )
        return False

    # Ambigu : encore une passe close + juge
    _gripper_secure_close(sc1, gripper_hold, "right", hold_s=1.2, pulses=3)
    if claw_mon.likely_holding() or not claw_mon._fully_closed_empty():
        rospy.loginfo(
            "[GRIP-TEST] %s TIENT (2e passe) — %s",
            label, claw_mon.describe_right(),
        )
        return True
    rospy.logwarn(
        "[GRIP-TEST] %s incertains — %s",
        label, claw_mon.describe_right(),
    )
    return bool(claw_mon.likely_holding())


def _patch_high_pick_approach(sc1):
    """Force yz_align / pre à z haut — pas de trajectoire basse entre colis."""
    import rospy

    if getattr(sc1, "_team_high_approach_patched", False):
        return
    sc1.RIGHT_PICK_TRANSIT_IK_Z = float(PICK_APPROACH_IK_Z)
    orig_align = sc1._right_pick_yz_align_ik

    def _high_yz_align(current_right_ik, right_pick_pre_ik):
        pt = list(orig_align(current_right_ik, right_pick_pre_ik))
        z_min = max(
            float(PICK_APPROACH_IK_Z),
            float(getattr(sc1, "RIGHT_PICK_TRANSIT_IK_Z", PICK_APPROACH_IK_Z)),
            float(right_pick_pre_ik[2]),
        )
        pt[2] = max(float(pt[2]), z_min)
        return pt

    sc1._right_pick_yz_align_ik = _high_yz_align
    sc1._team_high_approach_patched = True
    rospy.loginfo(
        "[APPROACH] TRANSIT_Z=%.3f (au-dessus colis) + yz_align forcé haut",
        sc1.RIGHT_PICK_TRANSIT_IK_Z,
    )


def _confirm_good_grasp(sc1, arm_pub, arm_hold, gripper_hold, claw_mon, pick_quat, name):
    """
    Avant pesée : s'assurer que la prise EST BONNE.
      1) serre fort
      2) pause (colis ne doit pas glisser)
      3) grip-test (bloqué = tient)
      4) micro-lift + 2e grip-test
    Returns True seulement si tient clairement. Sinon False (pas de pesée).
    """
    import rospy

    rospy.loginfo("[PICK-OK] %s vérif prise avant pesée — %s", name, claw_mon.describe_right())
    _gripper_secure_close(sc1, gripper_hold, "right", hold_s=1.6, pulses=4)
    rospy.sleep(0.55)

    held = _grip_test_and_squeeze(
        sc1, gripper_hold, claw_mon, label="%s_pre_weigh" % name,
    )
    if not held:
        rospy.logwarn("[PICK-OK] %s échec 1er grip-test", name)
        return False

    # Micro-lift : si le colis tombe / pince se ferme à fond → mauvaise prise
    try:
        cur = _current_right_xyz(sc1)
        lift = [float(cur[0]), float(cur[1]), float(cur[2]) + 0.06]
        sc1._move_right_cartesian_to(
            arm_pub, arm_hold, lift, list(pick_quat), "%s_hold_check_lift" % name,
            n_points=8, seg_time=0.55,
        )
        rospy.sleep(0.45)
    except Exception as exc:
        rospy.logwarn("[PICK-OK] micro-lift skip: %s", exc)

    gripper_hold.set_right_closed()
    rospy.sleep(0.40)
    held2 = (
        claw_mon.likely_holding()
        or (
            HOLD_POS_MIN <= float(claw_mon._right_pos) <= HOLD_POS_MAX
            and not claw_mon._fully_closed_empty()
        )
    )
    if claw_mon._fully_closed_empty() and not claw_mon.likely_holding():
        rospy.logerr(
            "[PICK-OK] %s VIDE après micro-lift — PAS de pesée — %s",
            name, claw_mon.describe_right(),
        )
        return False
    if not held2:
        # 2e grip-test
        held2 = _grip_test_and_squeeze(
            sc1, gripper_hold, claw_mon, label="%s_pre_weigh2" % name,
        )
    if held2:
        _gripper_secure_close(sc1, gripper_hold, "right", hold_s=1.3, pulses=3)
        rospy.loginfo(
            "[PICK-OK] %s BONNE PRISE → autorise pesée — %s",
            name, claw_mon.describe_right(),
        )
        return True
    rospy.logerr(
        "[PICK-OK] %s prise NON confirmée — PAS de pesée — %s",
        name, claw_mon.describe_right(),
    )
    return False


def _close_and_lift_after_servo(sc1, arm_pub, arm_hold, gripper_hold,
                                pick_ik, pick_quat, label):
    """Pince déjà au-dessus / au contact : close + lift (pas de plongée)."""
    import rospy

    pick_ik = list(pick_ik)
    # Micro-ajustement final (déjà placé par le servo)
    try:
        sc1._move_hand_precise(
            "right", arm_pub, arm_hold, pick_ik, list(pick_quat),
            "%s_final_grasp" % label,
        )
    except Exception as exc:
        rospy.logwarn("[PICK] %s final precise: %s — close quand même", label, exc)

    gripper_hold.set_right_closed()
    rospy.sleep(float(getattr(sc1, "GRIPPER_CLOSE_HOLD_TIME", 1.0) or 1.0))
    _gripper_secure_close(sc1, gripper_hold, "right", hold_s=0.8, pulses=2)

    lift_z = float(getattr(sc1, "WEIGH_TRANSIT_IK_Z", CLEARANCE_IK_Z) or CLEARANCE_IK_Z)
    lift_ik = sc1._with_ik_z(pick_ik, lift_z)
    sc1._move_right_cartesian_to(
        arm_pub, arm_hold, lift_ik, list(pick_quat),
        "%s_soft_lift" % label,
        n_points=SLOW_CART_POINTS, seg_time=SLOW_CART_SEG,
    )


def _pick_parcel_secure(sc1, arm_pub, arm_hold, gripper_hold, job, pick_quat,
                        claw_mon=None):
    """
    PRISE table :
      open → au-dessus → descente PAR PALIERS + centrage cam_r chaque fois
      → close → lift → CONFIRM (sinon abort, pas de pesée).
    """
    import rospy

    name = job["object"]
    pick_ik = list(job["right_pick_ik"])
    pick_quat = list(pick_quat)
    if claw_mon is None:
        claw_mon = RightClawMonitor()
    _install_soft_lift_patch(sc1)
    _patch_high_pick_approach(sc1)

    _gripper_max_open(sc1, gripper_hold, "right")
    _go_above_parcel_centered(sc1, arm_pub, arm_hold, pick_ik, pick_quat)

    # Debug demandé : main au-dessus → lire angles pince → STOP (pas de descente)
    if DEBUG_STOP_ABOVE_FIRST_PARCEL:
        _debug_dump_gripper_angles(
            sc1, pick_quat, pick_ik, label=name,
        )
        raise RuntimeError("DEBUG_STOP_ABOVE_PARCEL (%s)" % name)

    servo_ok = False
    try:
        try:
            from scene1.wrist_refine import servo_descend_to_mid_parcel
        except ImportError:
            from wrist_refine import servo_descend_to_mid_parcel
        job = servo_descend_to_mid_parcel(
            sc1, arm_pub, arm_hold, job, pick_quat,
            final_z=float(pick_ik[2]),
            output_dir=_scene1_output_dir("wrist"),
        )
        pick_ik = list(job["right_pick_ik"])
        servo_ok = bool(job.get("wrist_refined"))
        rospy.loginfo(
            "[PICK] %s servo fin centered=%s dpx=%s ik=%s",
            name, job.get("wrist_centered"), job.get("wrist_dpx"),
            [round(v, 3) for v in pick_ik],
        )
    except Exception as exc:
        rospy.logwarn("[WRIST] servo skip: %s — fallback plongée orga", exc)

    if servo_ok:
        _close_and_lift_after_servo(
            sc1, arm_pub, arm_hold, gripper_hold,
            pick_ik, pick_quat, name,
        )
    else:
        # Fallback : ancienne plongée orga si cam_r totalement KO
        sc1._left_wait_and_right_pick_from(
            arm_pub, arm_hold, gripper_hold,
            pick_ik, pick_quat, name, carry_quat=pick_quat,
        )

    if _confirm_good_grasp(
        sc1, arm_pub, arm_hold, gripper_hold, claw_mon, pick_quat, name,
    ):
        return

    # Pas de « ambigu → pesée » : si pas confirmé = échec prise
    rospy.logerr(
        "[PICK] %s mauvaise prise — abort — %s",
        name, claw_mon.describe_right(),
    )
    _abort_empty_safe(
        sc1, arm_pub, arm_hold, gripper_hold, pick_quat, name,
    )
    raise RuntimeError("empty hand after grasp (%s)" % name)


def _ik_in_weigh_zone(xyz):
    x, y = float(xyz[0]), float(xyz[1])
    return (
        WEIGH_ZONE_IK_X[0] <= x <= WEIGH_ZONE_IK_X[1]
        and WEIGH_ZONE_IK_Y[0] <= y <= WEIGH_ZONE_IK_Y[1]
    )


def _detect_parcel_on_weigh_pad(sc1, parcel_name):
    """
    Redétecte le colis sur la balance (RGB-D tête) — comme la prise table.
    Retourne right_pick_ik [x,y,z] (z = WEIGH_REGRASP) ou None.
    """
    import math
    import rospy

    try:
        try:
            from scene1.perception import detect_parcels
        except ImportError:
            from perception import detect_parcels
        dets = detect_parcels(log=rospy.loginfo)
    except Exception as exc:
        rospy.logwarn("[WEIGH] detect balance fail: %s", exc)
        return None

    pad = list(sc1.WEIGH_RELEASE_IK)
    regrasp_z = float(sc1.WEIGH_REGRASP_IK[2])
    best_xyz = None
    best_name = None
    best_d = 1e9
    for det in dets or []:
        name = det.get("name")
        xyz = det.get("center") or det.get("base_link_xyz_m")
        if xyz is None or len(xyz) < 2:
            continue
        ix, iy = float(xyz[0]), float(xyz[1])
        if not _ik_in_weigh_zone([ix, iy, 0.0]):
            continue
        d = math.hypot(ix - float(pad[0]), iy - float(pad[1]))
        score = d - (0.15 if name == parcel_name else 0.0)
        if score < best_d:
            best_d = score
            best_xyz = [ix, iy, float(xyz[2]) if len(xyz) > 2 else float(pad[2])]
            best_name = name or parcel_name
    if best_xyz is None:
        rospy.logwarn("[WEIGH] aucun colis détecté sur pad")
        return None

    # Même offset tip que la prise table
    offset = pick_offset_for_detection(sc1, best_name, best_xyz)
    pick_ik = [
        float(best_xyz[0]) + float(offset[0]),
        float(best_xyz[1]) + float(offset[1]),
        regrasp_z,
    ]
    rospy.loginfo(
        "[WEIGH] vision pad → %s det=(%.3f,%.3f) off=(%.3f,%.3f) → ik=(%.3f,%.3f,%.3f)",
        best_name, best_xyz[0], best_xyz[1], offset[0], offset[1],
        pick_ik[0], pick_ik[1], pick_ik[2],
    )
    return pick_ik


def _regrasp_once(sc1, arm_pub, arm_hold, gripper_hold, claw_mon,
                  regrasp_ik, weigh_z, regrasp_quat, close_hold, label):
    """Une reprise : pre (z=release) → down → close → grip-test (comme descente prise)."""
    import rospy

    regrasp_ik = list(regrasp_ik)
    regrasp_pre_ik = sc1._with_ik_z(regrasp_ik, float(weigh_z))
    rospy.loginfo(
        "[WEIGH] reprise %s pre=%s down=%s",
        label,
        [round(v, 3) for v in regrasp_pre_ik],
        [round(v, 3) for v in regrasp_ik],
    )
    _gripper_max_open(sc1, gripper_hold, "right")
    regrasp_pre_cmd14 = sc1._move_hand(
        "right",
        arm_pub,
        arm_hold,
        regrasp_pre_ik,
        list(regrasp_quat),
        "%s_xy_ori" % label,
        settle_time=0.0,
    )
    sc1._move_hand_precise(
        "right",
        arm_pub,
        arm_hold,
        regrasp_ik,
        list(regrasp_quat),
        "%s_down" % label,
    )
    gripper_hold.set_right_closed()
    rospy.sleep(float(close_hold))
    held = _grip_test_and_squeeze(
        sc1, gripper_hold, claw_mon, label=label,
    )
    gripper_hold.set_right_closed()
    empty = (
        not held
        and claw_mon._fully_closed_empty()
        and not claw_mon.likely_holding()
    )
    return (not empty), regrasp_pre_cmd14


def _regrasp_like_pick(sc1, arm_pub, arm_hold, gripper_hold, claw_mon, job,
                       target_ik, weigh_z, regrasp_quat, close_hold, label):
    """
    Reprise balance = même logique que prise table :
      open → au-dessus → descente par paliers + centrage cam_r → close.
    """
    import rospy

    target_ik = list(target_ik)
    target_ik[2] = float(sc1.WEIGH_REGRASP_IK[2])
    mini_job = {
        "object": job.get("object"),
        "right_pick_ik": list(target_ik),
        "source_world": job.get("source_world"),
        "perception": dict(job.get("perception") or {}),
    }

    _gripper_max_open(sc1, gripper_hold, "right")
    _go_above_parcel_centered(
        sc1, arm_pub, arm_hold, target_ik, list(regrasp_quat),
    )

    servo_ok = False
    try:
        try:
            from scene1.wrist_refine import servo_descend_to_mid_parcel
        except ImportError:
            from wrist_refine import servo_descend_to_mid_parcel
        mini_job = servo_descend_to_mid_parcel(
            sc1, arm_pub, arm_hold, mini_job, list(regrasp_quat),
            final_z=float(target_ik[2]),
            output_dir=_scene1_output_dir("wrist_weigh"),
        )
        target_ik = list(mini_job["right_pick_ik"])
        target_ik[2] = float(sc1.WEIGH_REGRASP_IK[2])
        servo_ok = bool(mini_job.get("wrist_refined"))
        rospy.loginfo(
            "[WEIGH] %s servo centered=%s dpx=%s → %s",
            label,
            mini_job.get("wrist_centered"),
            mini_job.get("wrist_dpx"),
            [round(v, 3) for v in target_ik],
        )
    except Exception as exc:
        rospy.logwarn(
            "[WEIGH] %s wrist servo skip: %s — reprise orga", label, exc,
        )

    if servo_ok:
        # Déjà en bas centré : close + grip-test (sans re-plonger)
        try:
            sc1._move_hand_precise(
                "right", arm_pub, arm_hold, target_ik, list(regrasp_quat),
                "%s_final" % label,
            )
        except Exception as exc:
            rospy.logwarn("[WEIGH] %s final: %s", label, exc)
        gripper_hold.set_right_closed()
        rospy.sleep(float(close_hold))
        held = _grip_test_and_squeeze(
            sc1, gripper_hold, claw_mon, label=label,
        )
        gripper_hold.set_right_closed()
        empty = (
            not held
            and claw_mon._fully_closed_empty()
            and not claw_mon.likely_holding()
        )
        pre_cmd = None
        try:
            pre_ik = sc1._with_ik_z(target_ik, float(weigh_z))
            pre_cmd = sc1._move_hand(
                "right", arm_pub, arm_hold, pre_ik, list(regrasp_quat),
                "%s_lift_pre" % label, settle_time=0.0,
            )
        except Exception:
            pre_cmd = None
        return (not empty), pre_cmd

    return _regrasp_once(
        sc1, arm_pub, arm_hold, gripper_hold, claw_mon,
        target_ik, float(weigh_z), regrasp_quat, close_hold, label,
    )


def weigh_and_regrasp(sc1, arm_pub, arm_hold, gripper_hold, job, pick_quat, args,
                      claw_mon=None):
    """
    Pesée + reprise = flux ORGA (collect_scene1_handoff_dataset) :

      place :
        transit → descend WEIGH_RELEASE_IK + quat orga release
        → settle 1.5s → open → dwell (pose calme, pas de jet)
      regrasp (comme prise table) :
        detect pad (tête) → au-dessus → centrage cam_r → descente / close
      si main vide :
        redétecte + offsets XY → retry (max 3)
    """
    import rospy

    name = job["object"]
    # Poses + angles ORGA exacts
    weigh_ik = list(sc1.WEIGH_RELEASE_IK)
    regrasp_base = list(sc1.WEIGH_REGRASP_IK)
    release_quat = list(args.right_weigh_release_quat_xyzw)
    regrasp_quat = list(args.right_weigh_regrasp_quat_xyzw)
    carry_quat = list(pick_quat)
    dwell = float(getattr(args, "weigh_dwell", None) or getattr(sc1, "WEIGH_DWELL", 1.0) or 1.0)
    settle_before_open = float(
        getattr(sc1, "WEIGH_RELEASE_SETTLE_BEFORE_OPEN", None) or 1.5
    )
    close_hold = float(getattr(sc1, "GRIPPER_CLOSE_HOLD_TIME", 1.0) or 1.0)

    rospy.loginfo(
        "[WEIGH] %s dépôt=%s reprise=detect+wrist+down z=%.3f",
        name,
        [round(v, 3) for v in weigh_ik],
        float(regrasp_base[2]),
    )
    gripper_hold.set_right_closed()
    if claw_mon is None:
        claw_mon = RightClawMonitor()

    if claw_mon._fully_closed_empty() and not claw_mon.likely_holding():
        rospy.logerr(
            "[WEIGH] %s MAIN VIDE avant balance — abort — %s",
            name, claw_mon.describe_right(),
        )
        raise RuntimeError("empty hand before weigh (%s)" % name)
    _gripper_secure_close(sc1, gripper_hold, "right", hold_s=1.2, pulses=3)

    # --- ORGA place_on_weighing_area : descendre au pad, poser calmement ---
    sc1._move_right_to_weigh_release_pre(
        arm_pub, arm_hold, weigh_ik, release_quat, carry_quat=carry_quat,
    )
    rospy.sleep(0.30)
    # Descente verticale jusqu'au point orga (bas, sur le pad)
    sc1._move_hand_precise(
        "right", arm_pub, arm_hold, weigh_ik, release_quat, "right_weigh_down",
    )
    settle = max(float(settle_before_open), float(CALM_PLACE_SETTLE_BEFORE_OPEN))
    rospy.loginfo(
        "[WEIGH] ORGA POSE CALME — main au pad, settle %.2fs puis open", settle,
    )
    rospy.sleep(settle)
    gripper_hold.set_right_open()
    try:
        sc1._wait_for_gripper_open("right", timeout=2.5)
    except Exception as exc:
        rospy.logwarn("[WEIGH] wait open: %s", exc)
    dwell_calm = max(float(dwell), float(CALM_PLACE_DWELL_AFTER_OPEN))
    rospy.loginfo("[WEIGH] colis posé — dwell %.2fs (ne pas jeter / partir trop tôt)", dwell_calm)
    rospy.sleep(dwell_calm)

    # --- Reprise = même schéma que prise : detect → above → wrist → down ---
    regrasp_pre_cmd14 = None
    held = False
    max_tries = int(WEIGH_REGRASP_MAX_TRIES)
    for attempt in range(1, max_tries + 1):
        # Remonte / ouvre avant chaque essai (sauf si déjà haut après dépôt)
        try:
            above = sc1._with_ik_z(regrasp_base, float(weigh_ik[2]) + 0.10)
            sc1._move_hand(
                "right", arm_pub, arm_hold, above, regrasp_quat,
                "%s_regrasp_clear_%d" % (name, attempt),
            )
        except Exception as exc:
            rospy.logwarn("[WEIGH] clear before detect: %s", exc)
        _gripper_max_open(sc1, gripper_hold, "right")
        rospy.sleep(0.35)

        det_ik = _detect_parcel_on_weigh_pad(sc1, name)
        if det_ik is not None:
            target = list(det_ik)
            target[2] = float(regrasp_base[2])
            label = "%s_regrasp_%d_vision" % (name, attempt)
            rospy.loginfo(
                "[WEIGH] essai %d/%d DETECT+WRIST → %s",
                attempt, max_tries, [round(v, 3) for v in target],
            )
        else:
            ox, oy = WEIGH_REGRASP_XY_OFFSETS[
                (attempt - 1) % len(WEIGH_REGRASP_XY_OFFSETS)
            ]
            target = [
                float(regrasp_base[0]) + float(ox),
                float(regrasp_base[1]) + float(oy),
                float(regrasp_base[2]),
            ]
            label = "%s_regrasp_%d_off" % (name, attempt)
            rospy.logwarn(
                "[WEIGH] essai %d/%d pas de detect — OFFSET (%.2f,%.2f) + wrist → %s",
                attempt, max_tries, ox, oy,
                [round(v, 3) for v in target],
            )

        try:
            held, regrasp_pre_cmd14 = _regrasp_like_pick(
                sc1, arm_pub, arm_hold, gripper_hold, claw_mon, job,
                target, float(weigh_ik[2]), regrasp_quat, close_hold, label,
            )
        except Exception as exc:
            rospy.logwarn("[WEIGH] reprise %s fail: %s", label, exc)
            held = False
            regrasp_pre_cmd14 = None

        if held:
            rospy.loginfo(
                "[WEIGH] %s reprise OK essai %d — %s",
                name, attempt, claw_mon.describe_right(),
            )
            break
        rospy.logwarn(
            "[WEIGH] %s reprise VIDE essai %d/%d — %s",
            name, attempt, max_tries, claw_mon.describe_right(),
        )

    if not held:
        rospy.logerr(
            "[WEIGH] %s reprise VIDE après %d essais — STOP bac — %s",
            name, max_tries, claw_mon.describe_right(),
        )
        _abort_empty_safe(
            sc1, arm_pub, arm_hold, gripper_hold, pick_quat, "%s_regrasp" % name,
        )
        raise RuntimeError("empty hand after regrasp (%s)" % name)

    # Lift joints pre (orga) — pince reste fermée
    if regrasp_pre_cmd14 is not None:
        current = sc1._read_current_arm_joints(sc1.TOPIC_TIMEOUT)
        sc1._execute_joint_motion_chunked(
            arm_pub,
            arm_hold,
            sc1._rad_to_deg(current),
            regrasp_pre_cmd14,
            sc1.SCENE1_ARM_MOVE_TIME,
            sc1.SCENE1_ARM_SETTLE_TIME,
        )
    else:
        pre = sc1._with_ik_z(regrasp_base, float(weigh_ik[2]))
        sc1._move_hand(
            "right", arm_pub, arm_hold, pre, regrasp_quat, "%s_lift_pre" % name,
        )
    _gripper_secure_close(sc1, gripper_hold, "right", hold_s=1.2, pulses=3)
    rospy.loginfo(
        "[WEIGH] %s reprise OK (orga+retry) — %s",
        name, claw_mon.describe_right(),
    )
    return True


def run_pick_only_motion(sc1, arm_pub, arm_hold, gripper_hold, jobs, args):
    """Preset → prise soft + vérif hold → open pour le suivant."""
    import rospy

    claw_mon = RightClawMonitor()
    _install_soft_lift_patch(sc1)
    _patch_high_pick_approach(sc1)
    _gripper_max_open(sc1, gripper_hold, "right")
    pick_quat = list(args.right_pick_quat_xyzw)
    _safe_return_preset(sc1, arm_pub, arm_hold, quat=pick_quat)
    done = 0
    for index, job in enumerate(jobs, start=1):
        if not job.get("perception"):
            rospy.logwarn("scene1 pick-only: skip %s", job["object"])
            continue
        name = job["object"]
        rospy.loginfo("scene1 pick-only %d: %s pick_ik=%s", index, name,
                      [round(v, 3) for v in job["right_pick_ik"]])
        try:
            _pick_parcel_secure(
                sc1, arm_pub, arm_hold, gripper_hold, job, pick_quat,
                claw_mon=claw_mon,
            )
            done += 1
            rospy.loginfo("scene1 pick-only: %s HOLD", name)
            rospy.sleep(0.8)
            _raise_clear_above(sc1, arm_pub, arm_hold, quat=pick_quat)
        except Exception as exc:
            rospy.logerr("scene1 pick-only: %s FAIL: %s", name, exc)
            _raise_clear_above(sc1, arm_pub, arm_hold, quat=pick_quat)
        _gripper_max_open(sc1, gripper_hold, "right")
        rospy.sleep(0.4)
    rospy.loginfo("scene1 pick-only terminé: %d", done)
    return done


def run_full_mission_motion(sc1, arm_pub, arm_hold, gripper_hold, jobs, args):
    """
    Mission complète :
      open MAX → au-dessus centre → prise → HOLD → pesée → taille → bac
      Retours preset toujours via z haut (anti-bousculade).
    """
    import rospy

    claw_mon = RightClawMonitor()
    _install_soft_lift_patch(sc1)
    _patch_high_pick_approach(sc1)
    _gripper_max_open(sc1, gripper_hold, "right")
    pick_quat = list(args.right_pick_quat_xyzw)
    _safe_return_preset(sc1, arm_pub, arm_hold, quat=pick_quat)
    ok_n = 0
    fail_n = 0

    rospy.loginfo("=" * 50)
    rospy.loginfo("MISSION: center+highZ → prise → pesée → taille → bac")
    rospy.loginfo("=" * 50)

    for index, job in enumerate(jobs, start=1):
        name = job["object"]
        if not job.get("perception"):
            rospy.logwarn(
                "[MISSION] %s sans vision OK — fallback static pick_ik=%s",
                name, [round(v, 3) for v in job["right_pick_ik"]],
            )
        rospy.loginfo(
            "[MISSION] %d/%d %s pick_ik=%s",
            index, len(jobs), name,
            [round(v, 3) for v in job["right_pick_ik"]],
        )
        try:
            picked = False
            last_exc = None
            max_tries = int(PICK_MAX_ATTEMPTS)
            for attempt in range(1, max_tries + 1):
                try:
                    if attempt > 1:
                        # Après échec : ouvrir pince + descendre Z (plus près du colis)
                        _gripper_max_open(sc1, gripper_hold, "right")
                        old_z = float(job["right_pick_ik"][2])
                        new_z = max(
                            float(PICK_IK_Z_MIN),
                            old_z - float(PICK_RETRY_DZ),
                        )
                        job["right_pick_ik"] = [
                            float(job["right_pick_ik"][0]),
                            float(job["right_pick_ik"][1]),
                            new_z,
                        ]
                        rospy.logwarn(
                            "[MISSION] %s essai %d/%d OPEN + descend Z %.3f→%.3f",
                            name, attempt, max_tries, old_z, new_z,
                        )
                        rospy.sleep(0.20)
                    else:
                        rospy.loginfo(
                            "[MISSION] %s essai %d/%d", name, attempt, max_tries,
                        )
                    _pick_parcel_secure(
                        sc1, arm_pub, arm_hold, gripper_hold, job, pick_quat,
                        claw_mon=claw_mon,
                    )
                    picked = True
                    break
                except RuntimeError as exc:
                    last_exc = exc
                    if "empty hand" in str(exc).lower():
                        rospy.logwarn(
                            "[MISSION] %s MAIN VIDE essai %d/%d",
                            name, attempt, max_tries,
                        )
                        if attempt >= max_tries:
                            rospy.logerr(
                                "[MISSION] %s abandon après %d essais → colis suivant",
                                name, max_tries,
                            )
                        continue
                    raise
            if not picked:
                # 5 échecs → skip ce colis, passer au suivant (pas FATAL mission)
                fail_n += 1
                rospy.logerr(
                    "[MISSION] %s SKIP (prise KO x%d): %s",
                    name, max_tries, last_exc,
                )
                try:
                    _gripper_max_open(sc1, gripper_hold, "right")
                    _retract_safe(
                        sc1, arm_pub, arm_hold, pick_quat,
                        label="%s_skip_retract" % name,
                    )
                    _safe_return_preset(sc1, arm_pub, arm_hold, quat=pick_quat)
                except Exception:
                    pass
                continue

            weigh_and_regrasp(
                sc1, arm_pub, arm_hold, gripper_hold, job, pick_quat, args,
                claw_mon=claw_mon,
            )

            if not place_in_box_right(
                sc1, arm_pub, arm_hold, gripper_hold, name, pick_quat,
                claw_mon=claw_mon,
            ):
                raise RuntimeError("box deposit failed (empty or IK)")

            ok_n += 1
            rospy.loginfo("[MISSION] %s FULL OK (%d)", name, ok_n)
            _gripper_max_open(sc1, gripper_hold, "right")
        except Exception as exc:
            fail_n += 1
            rospy.logerr("[MISSION] %s FAIL: %s", name, exc)
            try:
                # Retract AVANT waist=0 — sinon jerk + balayage table
                _retract_safe(
                    sc1, arm_pub, arm_hold, pick_quat, label="%s_fail_retract" % name,
                )
                _set_waist_yaw_deg(0.0, settle=2.0)
                _abort_empty_safe(
                    sc1, arm_pub, arm_hold, gripper_hold, pick_quat, name,
                )
                _safe_return_preset(sc1, arm_pub, arm_hold, quat=pick_quat)
                _gripper_max_open(sc1, gripper_hold, "right")
            except Exception:
                pass

    rospy.loginfo("[DONE] mission OK=%d FAIL=%d", ok_n, fail_n)
    return ok_n


def _tune_sc1_timings(sc1):
    # Lent et stable — trop rapide = jet / lâcher colis
    sc1.PRESET_SEGMENT_TIME = 2.4
    sc1.PRESET_SETTLE_TIME = 0.70
    sc1.SCENE1_ARM_MOVE_TIME = 1.7
    sc1.SCENE1_ARM_SETTLE_TIME = 0.70
    sc1.HANDOFF_MOVE_TIME = 2.0
    sc1.PICK_ALIGN_MOVE_TIME = 1.9
    sc1.PICK_GRASP_MOVE_TIME = 2.1
    sc1.PLACE_MOVE_TIME = 2.4
    # Approche HAUTE — orga 0.12 rase / bouscule les autres colis
    sc1.RIGHT_PICK_TRANSIT_IK_Z = float(PICK_APPROACH_IK_Z)
    # Descente prise plus basse que orga (-0.03) — évite close dans le vide
    sc1.RIGHT_PICK_IK_Z = float(PICK_IK_Z_BASE)
    # Pesée / dépose calmes
    sc1.WEIGH_RELEASE_SETTLE_BEFORE_OPEN = float(CALM_PLACE_SETTLE_BEFORE_OPEN)
    sc1.WEIGH_DWELL = float(CALM_PLACE_DWELL_AFTER_OPEN)
    sc1.PLACE_DWELL = 1.0
    sc1.GRIPPER_CLOSE_HOLD_TIME = 1.2
    sc1.CONVERGE_TIMEOUT = 2.4
    sc1.CONVERGE_STABLE_HITS = 2


def run_scene1_actions(seed, detections=None, use_perception=True, pick_only=False):
    """
    pick_only=True  → prise seule (debug)
    pick_only=False → mission complète prise→pesée→waist→bac
    """
    import rospy
    from kuavo_msgs.msg import armTargetPoses

    _add_helper_path("test", "collect_scene1_dataset")
    import collect_scene1_handoff_dataset as sc1

    def _disabled_truth_pose_read(*_args, **_kwargs):
        raise RuntimeError("disabled in competition entry: no /mujoco/qpos truth read")

    sc1._mujoco_body_world_pos = _disabled_truth_pose_read
    _tune_sc1_timings(sc1)

    args = sc1.build_arg_parser().parse_args([])
    args.seed = int(seed)
    args.no_rosbag = True
    args.realtime_pick = False
    args.debug_verify_object_pose = False
    args.parcels = list(sc1.PARCEL_NAMES)
    args = sc1._normalize_args(args)
    # Angles prise = défaut orga (right_ypr / right_second_ypr)
    rospy.loginfo(
        "scene1 actions: pick_quat orga %s",
        [round(v, 4) for v in args.right_pick_quat_xyzw],
    )

    sc1._publish_head_target(sc1.TOPIC_TIMEOUT)

    if use_perception and detections is not None:
        jobs = jobs_from_detections(sc1, detections, STATIC_SOURCES)
        n_vis = sum(1 for j in jobs if j.get("perception"))
        if n_vis == 0:
            rospy.logerr("scene1 actions: 0 détection — abort")
            return False
        rospy.loginfo("scene1 actions: %d/%d colis vision", n_vis, len(jobs))
    else:
        rospy.loginfo("scene1 actions: baseline poses fixes")
        jobs = jobs_from_static(sc1, STATIC_SOURCES)

    gripper_hold = sc1._start_gripper_hold(sc1.TOPIC_TIMEOUT)
    arm_hold = sc1._start_arm_traj_hold(sc1.TOPIC_TIMEOUT)
    arm_mode_changed = False
    try:
        sc1._set_arm_mode(sc1.ARM_MODE_EXTERNAL_CONTROL, timeout=sc1.TOPIC_TIMEOUT)
        arm_mode_changed = True
        arm_pub = rospy.Publisher(sc1.ARM_TARGET_POSES_TOPIC, armTargetPoses, queue_size=10)
        sc1._wait_for_connection(arm_pub, sc1.TOPIC_TIMEOUT)
        if pick_only:
            run_pick_only_motion(sc1, arm_pub, arm_hold, gripper_hold, jobs, args)
        else:
            run_full_mission_motion(sc1, arm_pub, arm_hold, gripper_hold, jobs, args)
        return True
    finally:
        if arm_mode_changed:
            try:
                sc1._set_arm_mode(sc1.ARM_MODE_AUTO_SWING, timeout=sc1.TOPIC_TIMEOUT)
            except Exception as exc:
                rospy.logwarn("scene1 actions: restore arm mode: %s", exc)
        arm_hold.stop()
        gripper_hold.stop()
