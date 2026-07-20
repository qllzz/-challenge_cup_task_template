#!/usr/bin/env python3
"""Constantes Scene 1 — modes, perception, landmarks, bras."""
from __future__ import print_function
import math

# =============================================================================
# MODE
#   PERCEPTION_ONLY = True  → détection RGB-D seule (pas de bras)
#   PERCEPTION_ONLY = False → mission complète (défaut)
# =============================================================================
PERCEPTION_ONLY = False
DEBUG_STOP_IK_LABEL = ""  # ex. "right_x_to_pick_pre" = debug arrêt IK
DEBUG_STOP_AFTER_FIRST_IK = False
# Anti-triche : XY colis = detect_parcels ; points fixes scène = constantes orga.

# Tête — aligné script orga collect_scene1_handoff_dataset.py
# Orga: HEAD_TARGET = [0.0, 20.0], HEAD_SETTLE_TIME = 0.8
# Une fois baissée, on NE remonte PAS (pas de look_forward entre essais)
HEAD_LOOK_YAW = 0.0
HEAD_LOOK_PITCH = 20.0
HEAD_SETTLE_SEC = 0.8       # comme orga

# Debug labo UNIQUEMENT : comparer détection vs layout MuJoCo (world→IK).
# N'alimente JAMAIS le bras / la mission — log + CSV baseline seulement.
# Artefacts → labo/scene1/ (pas la racine repo / pas le code constructeur)
LABO_SCENE1_REL = "labo/scene1"
GT_COMPARE = False  # False en mission
GT_COMPARE_CSV = "labo/scene1/csv/scene1_gt_compare_fix_xbias.csv"
STUDY_GT_CSV = "labo/scene1/csv/scene1_study_gt_mujoco.csv"
STUDY_DET_CSV = "labo/scene1/csv/scene1_study_det_fix_xbias.csv"
COLOR_DEBUG_JPG = "labo/scene1/images/scene1_color_debug.jpg"

# Toucher le dessus du colis (pince ouverte, pas de fermeture)
TOUCH_Z_ABOVE_CENTER = 0.03       # juste au-dessus du centre (~dessus carton)
TOUCH_Z_ABOVE_BOX = 0.06
TOUCH_DWELL = 1.0
TOUCH_USE_FORWARD = False
TOUCH_ARM_SETTLE = 1.5
TOUCH_APPROACH_Z = 0.14           # approche encore haute, descente ensuite
TOUCH_TABLE_Z_MIN = -0.12
TOUCH_TABLE_Z_MAX = 0.02
TOUCH_MIN_X = 0.22
TOUCH_MAX_X = 0.52
TOUCH_LANDMARKS = False
TOUCH_MAX_PARCELS = 1             # 1er run : parcel_1 seulement
TOUCH_Y_OFFSET = 0.0              # pas de biais y (avant RIGHT_GRASP_Y_OFFSET=-0.02)

# Landmarks scène — couleur (scene1.yaml) + zones LiDAR IK
WEIGH_MARKER_REF_BGR = (66, 158, 26)       # weighing_area 0.10, 0.62, 0.26
BOX_MARKER_REF_BGR = (87, 158, 26)         # drop_box 0.10, 0.62, 0.34
WEIGH_MARKER_HSV = ((40, 100, 100), (85, 255, 255))
WEIGH_LIDAR_X_RANGE = (0.28, 0.52)
WEIGH_LIDAR_Y_RANGE = (-0.72, -0.42)
WEIGH_LIDAR_Z_RANGE = (-0.20, 0.25)
MIN_WEIGH_LIDAR_POINTS = 4
WEIGH_LAB_DIST_MAX = 42.0
WEIGH_COLOR_U_RANGE = (0.02, 0.42)
WEIGH_COLOR_V_RANGE = (0.35, 0.92)
WEIGH_COLOR_MAX_ERR_XY = 0.18
BOX_LIDAR_X_RANGE = (0.42, 0.88)
BOX_LIDAR_Y_RANGE = (0.02, 0.45)
BOX_LIDAR_Z_RANGE = (-0.15, 0.45)
BOX_FLOOR_LIDAR_Z_RANGE = (-0.11, 0.03)   # sol vert du bac (évite murs z≈0.09)
BOX_LIDAR_X_PERCENTILE = 82               # bord proche → intérieur dépose
BOX_LIDAR_Y_PERCENTILE = 72               # sol visible biaisé latéralement
BOX_COLOR_U_RANGE = (0.48, 0.98)
BOX_COLOR_V_RANGE = (0.35, 0.95)
MIN_LANDMARK_COLOR_PIXELS = 80
MAX_WEIGH_MARKER_AREA = 12000
MIN_BOX_LIDAR_POINTS = 6
LANDMARK_WORLD_POS = {
    "weighing_area": (-0.17, -0.56, 0.880),
    "sorting_box": (0.10, 0.29, 0.880),
}
# Ordre demandé : colis 1→4, puis bac, puis zone de pesée (truck)
TOUCH_ORDER = [
    "parcel_1", "parcel_2", "parcel_3", "parcel_4",
    "sorting_box",
    "weighing_area",
]

PARCEL_NAMES = ["parcel_1", "parcel_2", "parcel_3", "parcel_4"]

# Zone de filtrage LiDAR dans le repère base_link (IK) du robot :
#   x = devant, y = latéral, z = hauteur (table ~ z=-0.04, pas z=0.88 monde)
# Conversion world→IK : collect_scene1_handoff_dataset.py WORLD_TO_IK_OFFSET
WORLD_TO_IK_OFFSET = (0.565966, -0.013886, -0.923783)
PARCEL_WORLD_POS = {
    "parcel_1": (-0.26, -0.31, 0.880),
    "parcel_2": (-0.26, -0.09, 0.880),
    "parcel_3": (-0.11, -0.31, 0.880),
    "parcel_4": (-0.11, -0.09, 0.880),
}
TABLE_X_RANGE = (0.18, 0.58)     # seed30 : x≈0.25–0.48 (IK fail sous 0.18)
TABLE_Y_RANGE = (-0.42, 0.00)     # zone table seed30
PARCEL_Z_RANGE = (-0.15, 0.12)   # rejette amas flottants z≈0.22
PARCEL_RGB_Z_RANGE = (-0.12, 0.05)
TABLE_PARCEL_Z = -0.04
MIN_PICK_IK_X = 0.22               # sous ça : IK / workspace KO

# Contrat tête = ZONE stable (LiDAR←couleur UV).
# Labo seeds 0–9 : grid OFF → err_structure bloqué à 0.12 m (jaune sur colonne droite).
# grid-x ON + Δ max 18 cm : corrige association gauche ; row-lift reste OFF (bras/touch).
FUSE_ENABLE_GRID_SNAP = True
FUSE_ENABLE_ROW_LIFT = False
FUSE_MAX_RESHAPE_XY = 0.18         # jaune mal collé ~15 cm → 4 cm skippait toujours le snap
MAX_COLOR_AREA_RATIO = 0.18      # rejette masque > 18 % ROI (faux bleu ciel)
COLOR_ROI_V_START = 0.08         # pitch=20 : colis dès ~v=0.60 (plus haut qu'avec look_down)
COLOR_TABLE_BAND_V0 = 0.10       # fraction haute du ROI ignorée
COLOR_SOFT_DMIN = 65.0           # si refs trop loin → élargir recherche
COLOR_SOFT_TABLE_THR = 18.0      # seuil table assoupli en mode soft
COLOR_FULLROI_DMIN = 72.0        # si encore trop loin → chercher tout le ROI
COLOR_SOFT_THR_BOOST = 22.0      # +tolérance LAB en mode soft / full-ROI

# Clustering : regrouper les points LiDAR proches en "amas" (1 amas ≈ 1 colis)
CLUSTER_EPS_XY = 0.055           # plus serré → moins fusionner 2 colis voisins
MIN_CLUSTER_POINTS = 3             # peu de retours LiDAR par colis
MIN_PARCEL_SIZE_XY = 0.008         # filtre bruit très petit
MAX_PARCEL_SIZE_XY = 0.14          # au-dessus → découpe grille 2×2
LIDAR_GRID_X_MID = 0.381           # séparation parcel_1/2 vs 3/4 (IK x)
LIDAR_GRID_Y_MID = -0.214          # séparation parcel_1/3 vs 2/4 (IK y)
LIDAR_SEED_RADIUS = 0.14           # un peu plus large pour parcel_2 (jaune, peu de pts)
LIDAR_SEED_MIN_POINTS = 2          # grid_2 (jaune) souvent < 3 pts
MIN_COLOR_PIXELS = 80              # colis pâles / lointains (avant 120 trop strict)
MIN_COLOR_PIXELS_BY_NAME = {
    "parcel_2": 45,                # jaune : souvent peu de pixels LAB/HSV
    "parcel_1": 60,
    "parcel_3": 40,                # FOCUS orange : plus facile à voir
    "parcel_4": 90,
}
LAB_COLOR_DIST_BY_NAME = {
    "parcel_2": 78.0,              # jaune pâle (sim souvent saturée basse)
    "parcel_1": 72.0,
    "parcel_3": 85.0,              # FOCUS orange : plus tolérant
    "parcel_4": 66.0,              # bleu : 58 trop strict → colorish=3 (seeds 1/4/7)
}
MAX_FUSE_UV_DIST = 0.28            # pitch=20 : proj. relative + marge

# UV ↔ table IK — recalibré tête orga pitch=20 (COLOR u/v logs ↔ GT_ik seed2)
# Convention image : v↑ = rangée proche (x↑) ; u↑ = côté droit (y↑)
HEAD20_UV_U_LEFT = 0.28            # u min (gauche) sur plage relative
HEAD20_UV_U_SPAN = 0.50            # largeur u relative
HEAD20_UV_V_FAR = 0.58             # v min (rangée loin, x petit)
HEAD20_UV_V_SPAN = 0.40            # largeur v relative
# Secours absolu (1 amas / pas de peers) : x=a+b*v , y=c+d*u
HEAD20_X_FROM_V = (-0.01, 0.48)
HEAD20_Y_FROM_U = (-0.52, 0.55)
PERCEPTION_ATTEMPTS = 8            # validation multi-seed (assez, pas 20)
PERCEPTION_FORCE_ALL_ATTEMPTS = False  # stop tôt si géométrie+couleur OK
PERCEPTION_ERR_TARGET = 0.05       # N2 : err_structure_2x2 < 5 cm (anti-seed0)
LANDMARK_ERR_TARGET = 0.05         # N1b : balance + bac err_xy < 5 cm
SPATIAL_NAME_MAX_ERR = 0.12        # nommage LiDAR → parcel le plus proche (m)
MAX_BLUE_MASK_AREA = 25000         # blob bleu max après filtre profondeur
DEPTH_Z_MIN = 0.35                 # profondeur valide table (m)
DEPTH_Z_MAX = 1.2
# Profondeur typique des colis sur table (tête pitch=20) — coupe le ciel
COLOR_DEPTH_Z_MIN = 0.40
COLOR_DEPTH_Z_MAX = 0.95
RGB_DEPTH_XY_TOL = 0.10            # depth vs ray-plane : max Δxy pour accepter depth
LIDAR_O3D_PLANE_DIST = 0.015       # RANSAC distance au plan table (m)
LIDAR_O3D_ABOVE_MAX = 0.07         # points colis au-dessus du plan (m)
LIDAR_O3D_EPS = 0.045              # DBSCAN 3D plus serré (sépare colis ~15 cm)
LIDAR_O3D_MIN_POINTS = 3
# Biais systématique LiDAR→base_link (fix_rowy seeds 0/1/3/6 : dx≈+5.5 cm).
# Corrige l'ancre colonne droite avant grid-x (sinon toute la grille décale).
LIDAR_IK_CORR = (-0.050, 0.0, 0.0)
HUNGARIAN_BIG_COST = 1e4

# Couleurs nominales scene1.yaml (RGBA → BGR OpenCV) — distance LAB
TABLE_REF_BGR = (26, 77, 26)       # table_top 0.10, 0.30, 0.10
LAB_COLOR_DIST_MAX = 55.0          # un peu plus tolérant (éclairage sim)
TABLE_LAB_DIST_MAX = 36.0          # exclusion table (ne pas manger les colis pâles)
# pitch=20 : table + colis vers le bas/milieu (v≈0.60–0.99)
COLOR_V_NORM_MAX = 0.995
COLOR_V_NORM_MAX_BY_NAME = {}
COLOR_MAX_AREA_BY_NAME = {
    "parcel_2": 20000,
    "parcel_4": 25000,
}
# ROI image normalisée (u dans le ROI couleur)
COLOR_U_RANGE_BY_NAME = {
    "parcel_4": (0.30, 0.95),
    "parcel_2": (0.05, 0.75),
}
# Prefere blobs bas dans l'image (table) pour le bleu (évite ciel)
COLOR_PREFER_LOW_V = {"parcel_4"}
BASE_VALID_X = (0.18, 0.58)        # base_link IK valide pour rgb-depth
BASE_VALID_Y = (-0.48, -0.02)
BASE_VALID_Z = (-0.18, 0.08)
# Secours HSV si LAB insuffisant (scene1 colis pâles — S bas)
PARCEL_HSV_FALLBACK = [
    ("parcel_2", "yellow", (8, 20, 60), (55, 255, 255)),
    ("parcel_1", "brown", (4, 15, 70), (40, 200, 255)),
    ("parcel_3", "orange", (2, 30, 70), (30, 255, 255)),
    ("parcel_4", "blue", (90, 10, 120), (135, 130, 255)),
]
PARCEL_REF_COLORS = [
    ("parcel_2", "yellow", (97, 199, 235)),
    ("parcel_1", "brown", (186, 214, 224)),
    ("parcel_3", "orange", (71, 140, 230)),
    ("parcel_4", "blue", (235, 214, 199)),
]


# =============================================================================
# CONSTANTES — POINTS FIXES DE LA SCÈNE (ne changent PAS avec le seed)
# =============================================================================
# Coordonnées IK [x, y, z] en mètres dans le repère local du robot.
# Calibrées d'après le script organisateur collect_scene1_handoff_dataset.py.

WEIGH_TRANSIT_Z = 0.326217          # hauteur de transport vers la balance
# Centre balance orga (world ≈ -0.17, -0.56) → IK calibré
WEIGH_RELEASE_IK = [0.396, -0.574, 0.146217]   # MILIEU pad — orga exact
WEIGH_REGRASP_IK = [0.396, -0.574, -0.04]      # reprise même xy centre
LEFT_PRESET_2_IK = [0.313, 0.239, 0.282]       # main gauche en attente
RIGHT_HANDOFF_IK = [0.246, -0.044645, 0.3016983]  # point de passation
RIGHT_HANDOFF_TRANSIT_Z = 0.40     # hauteur intermédiaire avant passation
RIGHT_HANDOFF_TRANSIT_FALLBACK_ZS = [0.37, 0.35]  # orga: si 0.40 IK fail
# Gauche : même 1er YPR que droite ; 2e YPR = miroir (yaw/roll signés, pitch identique)
# pour coïncidence des pinces avant lâcher (orga avait pitch G=0 vs D=-20 → désaligné).
LEFT_HANDOFF_RECEIVE_XZ_READY_IK = [0.266, 0.139, 0.2816983]  # z aligné ~droite
LEFT_HANDOFF_RECEIVE_IK = [0.266, 0.04645, 0.2816983]
RIGHT_HANDOFF_RETRACT_Y = 0.0      # désactivé — plus de recul D après passation
BOX_DROP_BASE_IK = [0.58, 0.24, 0.556217]   # xy plus centré sur ouverture bac
BOX_DROP_HOVER_Z = 0.66            # au-dessus, assez bas pour viser le trou (sans toucher)
BOX_DROP_IK_X_FALLBACK_DELTAS = [-0.04, 0.0, 0.04, -0.08]

# Chaque colis a un petit décalage dans le bac pour ne pas se superposer (grille 2×2)
BOX_DROP_OFFSET_BY_PARCEL = {
    "parcel_1": [0.0, 0.0, 0.0],
    "parcel_2": [0.0, 0.02, 0.0],
    "parcel_3": [0.0, 0.0, 0.0],   # FOCUS: centre bac (pas décalé hors trou)
    "parcel_4": [0.02, 0.02, 0.0],
}

# Repères pour les logs (balance et bac) — utile pour le debug
SCENE_LANDMARKS = {
    "weighing_area": {
        "center": tuple(WEIGH_RELEASE_IK[:2]) + (0.88,),
        "release_z": WEIGH_RELEASE_IK[2],
    },
    "sorting_box": {
        "center": tuple(BOX_DROP_BASE_IK),
        "drop_z": BOX_DROP_BASE_IK[2],
    },
}


# =============================================================================
# CONSTANTES — MOUVEMENT DU BRAS ET DES PINCES
# =============================================================================

GRASP_QUAT = [0.0, 0.0, 0.0, 1.0]  # orientation neutre (quaternion x,y,z,w) pour la saisie
LEFT_WAIT_Y_OFFSET = 0.15           # main gauche décalée sur y pendant saisie droite
RIGHT_GRASP_Y_OFFSET = -0.02        # fine correction y pour la pince droite
APPROACH_Z_OFFSET = 0.10            # approche : 10 cm au-dessus du colis
GRASP_Z_OFFSET = -0.01              # descente légèrement sous le centre détecté
LIFT_Z_OFFSET = 0.30                # lever haut après saisie (orga) — évite la table
PLACE_APPROACH_Z = 0.06            # pesée seulement — bac = hover drop
ARM_SETTLE_TIME = 1.5
GRIPPER_SETTLE_TIME = 0.4           # pause après ouverture/fermeture pince
WEIGH_RELEASE_SETTLE = 1.5          # orga : stabilisation AVANT open sur balance
WEIGH_DWELL = 1.0                   # orga
PLACE_DWELL = 1.0                   # laisser le colis tomber dans le bac
MAX_PARCELS = 4
MAX_MISSION_FAILURES = 8
FORCE_PARCEL_NAME = None            # None = enchaîne les 4 colis (plus de FOCUS)

# Pipeline : prise → pesée → taille + droite → bac (plus de handoff D→G)
TRAIN_PICK_BOX = False
SKIP_WEIGH = False
SKIP_HANDOFF = True                 # True = pas de passation ; waist+main D
USE_WAIST_RIGHT_BOX = True          # tourner taille puis tendre D vers bac
WAIST_BOX_YAW_DEG = 30.0            # rotation taille gauche (deg)
WAIST_BOX_SETTLE_SEC = 2.5
# Extension droite vers bac (après taille) — z haut anti-table
RIGHT_BOX_EXTEND_MID = [0.42, 0.05, 0.50]
RIGHT_BOX_EXTEND_TRIES = [
    [0.55, 0.12, 0.52],
    [0.58, 0.16, 0.52],
    [0.60, 0.20, 0.52],
    [0.62, 0.22, 0.50],
    [0.65, 0.24, 0.50],
]
RIGHT_BOX_DROP_BASE_IK = [0.62, 0.22, 0.50]
RIGHT_BOX_DROP_HOVER_Z = 0.52
RIGHT_BOX_DROP_TRANSIT_Z = 0.48
RIGHT_BOX_DROP_IK_Y_TRIES = [0.0, 0.04, -0.04]
RIGHT_BOX_DROP_IK_X_TRIES = [0.0, -0.03, 0.03]
# Pause coïncidence : gauches fermée + angles alignés, avant ouverture droite
HANDOFF_COINCIDENCE_SETTLE = 0.8
HANDOFF_BOTH_HOLD_BEFORE_OPEN_R = 0.5

# Vérif saisie (LiDAR/RGB après lift) — Phase 1B : abort si encore sur table.
# /mujoco/qpos est INTERDIT (anti-triche).
GRASP_VERIFY_ENABLED = True
GRASP_VERIFY_ABORT_ON_EMPTY = False      # orga : pas d'abort custom
NEAR_ROW_USE_AXIS90 = False              # orga : un seul quat pour tous
GRASP_VERIFY_STILL_ON_TABLE_XY = 0.12

GRASP_VERIFY_TABLE_Z_MAX = 0.12


# =============================================================================
# MATH — conversion angles → quaternion (pour orienter les mains au bon angle)
# =============================================================================

def _matmul3(a, b):
    """Multiplication de deux matrices 3×3."""
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def _rotation_matrix_to_quat_xyzw(matrix):
    """Convertit une matrice de rotation 3×3 en quaternion [x, y, z, w]."""
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        w = math.sqrt(trace + 1.0) / 2.0
        x = (matrix[2][1] - matrix[1][2]) / (4.0 * w)
        y = (matrix[0][2] - matrix[2][0]) / (4.0 * w)
        z = (matrix[1][0] - matrix[0][1]) / (4.0 * w)
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        x = math.sqrt(max(0.0, 1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2])) / 2.0
        y = (matrix[0][1] + matrix[1][0]) / (4.0 * x)
        z = (matrix[0][2] + matrix[2][0]) / (4.0 * x)
        w = (matrix[2][1] - matrix[1][2]) / (4.0 * x)
    elif matrix[1][1] > matrix[2][2]:
        y = math.sqrt(max(0.0, 1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2])) / 2.0
        x = (matrix[0][1] + matrix[1][0]) / (4.0 * y)
        z = (matrix[1][2] + matrix[2][1]) / (4.0 * y)
        w = (matrix[0][2] - matrix[2][0]) / (4.0 * y)
    else:
        z = math.sqrt(max(0.0, 1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1])) / 2.0
        x = (matrix[0][2] + matrix[2][0]) / (4.0 * z)
        y = (matrix[1][2] + matrix[2][1]) / (4.0 * z)
        w = (matrix[1][0] - matrix[0][1]) / (4.0 * z)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        return [0.0, 0.0, 0.0, 1.0]
    return [x / norm, y / norm, z / norm, w / norm]


def _quat_from_ypr_deg(first_ypr_deg, second_ypr_deg=None):
    """
    Construit un quaternion à partir d'angles yaw/pitch/roll en degrés.
    Le robot Kuavo utilise deux jeux d'angles (comme dans arm_control.py).
    """
    yaw, pitch, _roll = [math.radians(float(v)) for v in first_ypr_deg]
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    matrix = [
        [cy * cp, -sy, cy * sp],
        [sy * cp, cy, sy * sp],
        [-sp, 0.0, cp],
    ]
    if second_ypr_deg is not None:
        manual_yaw, manual_pitch, manual_roll = [math.radians(float(v)) for v in second_ypr_deg]
        manual = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        if abs(manual_yaw) > 0.01:
            c, s = math.cos(manual_yaw), math.sin(manual_yaw)
            manual = _matmul3([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], manual)
        if abs(manual_pitch) > 0.01:
            c, s = math.cos(manual_pitch), math.sin(manual_pitch)
            manual = _matmul3([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], manual)
        if abs(manual_roll) > 0.01:
            c, s = math.cos(manual_roll), math.sin(manual_roll)
            manual = _matmul3([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], manual)
        matrix = _matmul3(matrix, manual)
    return _rotation_matrix_to_quat_xyzw(matrix)


# Quaternions pré-calculés pour chaque phase du mouvement (évite de recalculer à chaque fois)
LEFT_PRESET_2_QUAT = _quat_from_ypr_deg([-146.440, 4.966, 0.0], [0.0, 0.0, 96.580])
# Orientation prise = défaut orga (YPR [0,-90,0] + [90,0,0])
RIGHT_PICK_QUAT_ORGA_FLAT = _quat_from_ypr_deg([0, -90, 0.0], [90.0, 0.0, 0.0])
RIGHT_PICK_QUAT = RIGHT_PICK_QUAT_ORGA_FLAT
RIGHT_PICK_QUAT_AXIS90 = _quat_from_ypr_deg([0, -90, 0.0], [90.0, -90.0, 90.0])
RIGHT_WEIGH_RELEASE_QUAT = _quat_from_ypr_deg([0, -100, 0.0], [90.0, 0.0, 0.0])
RIGHT_WEIGH_REGRASP_QUAT = _quat_from_ypr_deg([0, -60, 0.0], [90.0, 0.0, 0.0])
RIGHT_HANDOFF_QUAT = _quat_from_ypr_deg([-0.839, -100.0, 0.0], [90.0, -20.0, 90.0])
# Miroir de la droite : same first YPR, second = [-yaw, pitch, -roll]
LEFT_HANDOFF_RECEIVE_QUAT = _quat_from_ypr_deg([-0.839, -100.0, 0.0], [-90.0, -20.0, -90.0])
LEFT_BOX_DROP_QUAT = _quat_from_ypr_deg([-0.328, -100.935, 0.0], [-90.0, 0.0, 0.369])

# Offset tip — DOC XY ; Z un peu plus bas (VISION OK Δpx=49 mais pince vide)
RIGHT_CLAW_TIP_OFFSET = [0.0, 0.0, 0.0]   # orga : pas de tip
RIGHT_PICK_IK_Z = -0.055                 # plus bas qu'orga (-0.03) — anti main vide
RIGHT_PICK_TRANSIT_IK_Z = 0.120           # orga
RIGHT_PICK_NEAR_FAR_Y_THRESHOLD = -0.20
RIGHT_PICK_OFFSET_FAR_ROW = [-0.03, 0.0, 0.0]       # orga
RIGHT_PICK_OFFSET_NEAR_ROW = [-0.03, 0.02, -0.02]   # orga exact
RIGHT_PICK_OFFSET_BY_PARCEL = {"parcel_4": [-0.01, 0.02, -0.02]}
RIGHT_PICK_YZ_ALIGN_SAFE_IK_X = 0.184
# Modes IK orga (pas de triche — paramètres de solve)
IK_MODE_POS_HARD_ORI_SOFT = 0x02
IK_MODE_POS_HARD_ORI_HARD = 0x03
IK_MODE_THREE_POINT_MIXED = 0x06
RIGHT_PICK_IK_MODE = IK_MODE_THREE_POINT_MIXED       # 0x06 approche
RIGHT_GRASP_FINAL_IK_MODE = IK_MODE_POS_HARD_ORI_HARD  # 0x03 descente
IK_MAJOR_ITERATIONS = 500
PICK_ALIGN_MOVE_SLEEP = 1.2   # orga PICK_ALIGN_MOVE_TIME
PICK_GRASP_MOVE_SLEEP = 1.4   # orga PICK_GRASP_MOVE_TIME
CARTESIAN_LIFT_POINTS = 4
CARTESIAN_LIFT_SEG_SLEEP = 0.35
GRIPPER_CLOSE_HOLD = 0.7      # orga GRIPPER_CLOSE_HOLD_TIME

# Levée "croix" orga PRESET_POINTS_DEG (5 waypoints) — évite de raser la table
ARM_RAISE_PRESET_DEG = [
    [20, 0, 0, -30, 0, 0, 0, 20, 0, 0, -30, 0, 0, 0],
    [20, 90, 0, -55, 0, 0, 0, 20, -90, 0, -55, 0, 0, 0],
    [20, 60, 0, -75, 0, 0, 0, 20, -60, 0, -75, 0, 0, 0],
    [29.89, 30.67, 29.889, -139.1, -59.33, 0, 0,
     29.89, -30.67, -29.889, -139.1, 59.33, 0, 0],
    [29.89, 10.67, 9.889, -139.1, -59.33, 0, 0,
     29.89, -10.67, -9.889, -139.1, 59.33, 0, 0],
]
ARM_RAISE_STEP_SLEEP = 1.2
ARM_CLEAR_TABLE_Z = 0.28   # hauteur mini avant tout déplacement horizontal

# =============================================================================
# VISION MAIN (priorité) — tête = zone ; main = peaufinage temps réel
# Pince (Grabbed) = après, une fois la vision stable
# =============================================================================
WRIST_ALLOW_RAY = True
WRIST_DEPTH_Z_MIN = 0.08
WRIST_DEPTH_Z_MAX = 0.55
WRIST_MIN_PIXELS = 50
# Log saturé area≈260k → faux centroid ; viser blobs colis (~15–80k)
WRIST_MAX_BLOB_FRAC = 0.16
WRIST_MASK_SAT_FRAC = 0.20         # si masque > 20% image → resserrer LAB
WRIST_ROI_FRAC = 0.85
WRIST_MAX_DELTA_XY = 0.02
WRIST_SERVO_GAIN = 0.45
WRIST_SERVO_ITERS = 4
WRIST_SERVO_SIGN_X = 1.0
WRIST_SERVO_SIGN_Y = -1.0
WRIST_CENTER_BIAS = 0.85
WRIST_ACCEPT_PX = 90.0
# Sweet spot run CENTRÉ Δpx=49 area≈40k
WRIST_LAB_BOOST = 8.0
WRIST_LAB_SOFT_EXTRA = 8.0
WRIST_USE_TABLE_EXCLUDE = True
WRIST_TABLE_EXCLUDE_SCALE = 1.15   # plus strict que tête (évite table « brun »)
WRIST_DEPTH_SOFT = True
WRIST_HSV_ONLY_IF_SPARSE = True    # HSV n'élargit pas un masque déjà plein
WRIST_SETTLE = 0.35
# Yaw // faces du colis carré : 1 mesure au-dessus → snap 0/90 → quat figé (pas en servo)
WRIST_YAW_ENABLE = False                 # orga pick : pas de yaw vision
WRIST_YAW_MAX_DEG = 90.0
WRIST_YAW_SNAP_SQUARE = True
WRIST_REQUIRE_SEE_BEFORE_CLOSE = False   # orga pick : close sans vision
WRIST_CLOSE_MAX_PIXEL_FRAC = 0.10
WRIST_LOCK_MIN_AREA = 3000
WRIST_UNDER_HAND_AREA = 12000
WRIST_UNDER_HAND_FRAC = 0.12
WRIST_UNDER_HAND_MAX_DPIX = 120
WRIST_POST_PLUNGE_MAX_DPIX = 170   # post-plongée Δpx monte (parallaxe tip)
WRIST_AIM_BIAS_U = 40.0
WRIST_AIM_BIAS_V = 70.0
WRIST_APPROACH_NUDGE = True
WRIST_APPROACH_NUDGE_MAX = 0.02
WRIST_SHALLOW_PLUNGE = 0.05
WRIST_CLOSE_EVEN_IF_IK_FAIL = True
WRIST_MID_DESCEND = False
WRIST_VISION_ONLY_GATE = False
WRIST_SKIP_CLAW_HOLD_CHECK = True        # orga : close + sleep

