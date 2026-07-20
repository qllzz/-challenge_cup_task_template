#!/usr/bin/env python3
"""
Scène 1 — Pesée et tri des colis.

  PERCEPTION_ONLY → détection RGB-D seule
  sinon           → perception.py + actions.py (mission complète)
"""
from __future__ import print_function
import os
import sys

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
_pkg = os.path.dirname(_scripts_dir)
sys.path.insert(0, os.path.join(_pkg, "src"))

from scene1.config import PERCEPTION_ONLY
from scene1.perception import run_scene1_perception_only, detect_parcels
from scene1.actions import run_scene1_actions


def _parcels_to_detections(parcels):
    """Format detect_parcels → jobs_from_detections."""
    out = []
    for p in parcels or []:
        name = p.get("name")
        center = p.get("center")
        if not name or center is None or len(center) < 3:
            continue
        out.append({
            "name": name,
            "class": name,
            "label": p.get("color"),
            "base_link_xyz_m": [float(center[0]), float(center[1]), float(center[2])],
            "pixel": p.get("pixel"),
            "source": p.get("source"),
        })
    return out


def _run_mission(robot, arm, claw, head, log, seed):
    """Mission : RGB-D → actions (prise → pesée → bac)."""
    import rospy
    from scene1.config import HEAD_LOOK_YAW, HEAD_LOOK_PITCH, HEAD_SETTLE_SEC

    log("[SCENE1] mission RGB-D + actions (seed=%s)", seed)
    try:
        arm.switch_to_external_control()
    except Exception as exc:
        log("[SCENE1] external control: %s", exc)

    head.look_at(HEAD_LOOK_YAW, HEAD_LOOK_PITCH)
    rospy.sleep(HEAD_SETTLE_SEC)

    parcels = []
    for try_i in range(1, 4):
        parcels = detect_parcels(log=log)
        if parcels:
            break
        log("[SCENE1] detect vide try %d/3", try_i)
        rospy.sleep(0.5)

    detections = _parcels_to_detections(parcels)
    if not detections:
        log("[SCENE1] 0 détection — abort")
        return

    log("[SCENE1] %d colis → actions", len(detections))
    run_scene1_actions(
        seed=int(seed),
        detections=detections,
        use_perception=True,
        pick_only=False,
    )


def run_scene1(robot, arm, claw, head, log, seed=0):
    """Point d'entrée appelé par challenge_task.py."""
    if PERCEPTION_ONLY:
        run_scene1_perception_only(arm, head, log)
        return
    _run_mission(robot, arm, claw, head, log, seed)
