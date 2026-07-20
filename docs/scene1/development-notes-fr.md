# DOC SCENE 1 — Vision & saisie (notes d’équipe)

**Auteur vision :** Williams  
**Dernière MAJ :** 19 juillet 2026  
**Rôle :** détection tête + refine caméra main ; mission saisie / pesée / bac via ce package `scene1/`.

Ce doc explique **ce qui a été fait**, **pourquoi**, **les pièges**, et **comment relancer**.  
Il vit **à côté du code** dans ce dossier `scene1/`.

**中文版 — 提交用技术方案（完整单文件）：**  
[`technical-plan-zh.md`](technical-plan-zh.md)

---

## 0. État au 19 juillet 2026

Entrée quotidienne :

```bash
rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 17
```

Config actuelle (`config.py`) : mission + détection RGB-D branchées  
→ `scene1_perception` + `scene1_actions` (copies locales), main droite jusqu’au bac (pas de handoff gauche dans ce flux).

Seed debug : **17**. Un seul run à la fois (`set_object_position is locked` sinon).

---

## 1. Objectif

Pipeline orga : détecter 4 colis → saisir à droite → peser → regrasp → passer à gauche → déposer dans le bac.  
*(Flux actuel branché : main droite seule jusqu’au bac — voir §0.)*

**Anti-triche :** le bras ne doit **jamais** être piloté par `/mujoco/qpos` ni `/ground_truth/state`.  
`GT_COMPARE = False` en mission. Le GT MuJoCo sert uniquement au labo (CSV compare).

---

## 2. Fichiers à connaître (ce dossier)

| Fichier | Rôle |
|---------|------|
| `perception.py` | Tête : RGB-D LAB/HSV + depth → 4 colis |
| `wrist_refine.py` | Main : peaufinage cam_r avant prise |
| `config.py` | Modes, seuils, tip offset, FORCE_PARCEL |
| `actions.py` | Grasp / weigh / bac (main droite) |
| `../scene1_task.py` | Entrée scène 1 |
| `../../src/perception_api.py` | Cams `cam_h` / `cam_l` / `cam_r` + TF |

**Ne pas push pour le travail runtime :** `labo/scene1/**` (csv, logs, reports), logs/CSV à la racine.

Artefacts labo → `labo/scene1/` (voir `labo/scene1/README.md` à la racine du repo).

---

## 3. Architecture (comment ça marche)

```
1. HEAD  detect_parcels()
   LiDAR clusters (OÙ) + couleur tête LAB/HSV (QUI) + depth
   → fusion → grille relative 2×2
   → [{name, color, center, source}, ...]

2. APPROCHE bras droit au-dessus du centre détecté

3. HAND  observe_hand() / refine_target_with_wrist()
   Blob couleur+depth caméra poignet
   Petit Δxy (max ~2 cm / pas, jusqu’à 4 itérations)
   Gate : « see-check OK » avant de fermer

4. GRASP  descente peu profonde + close
   Tip shallow (éviter IK fail trop bas)
```

**Stratégie vision-first :** on fait confiance à la caméra main pour autoriser la fermeture (`WRIST_VISION_ONLY_GATE`, `WRIST_SKIP_CLAW_HOLD_CHECK`).  
La simu dit souvent `REACHED` à ~88 % même pince vide — ne pas se fier au hold pour juger la vision.

---

## 4. Ce qui a été validé

| Seed / colis | Résultat vision / saisie |
|--------------|---------------------------|
| Seed **30**, `parcel_1` (gris/brun) | **Meilleur run** : VISION OK (frac≈0.10), `claw R=3` Grabbed, pesée + regrasp + handoff OK (`DONE 1/1`). |
| Seed 0, `parcel_4` bleu | Au-dessus OK ; pince souvent vide (problème grip / tip, pas head detect). |
| Seed 0 / 50, `parcel_2` jaune | Approche OK ; hold / tip séparés. |

Détection tête multi-seed 0–3 (labo plus tôt) : 4/4, err structure ~0.2–3.7 cm — voir rapports dans `labo/scene1/reports/`.

---

## 5. Problèmes rencontrés & solutions

| Problème | Symptôme | Fix / leçon |
|----------|----------|-------------|
| Plongée trop profonde | `z≈-0.057` → IK fail après see-check → **jamais close** | Tip shallow : `RIGHT_PICK_IK_Z ≈ -0.005`, `RIGHT_CLAW_TIP_OFFSET ≈ [0.02, 0.01, -0.005]` ; `WRIST_CLOSE_EVEN_IF_IK_FAIL = True` |
| Faux « holding » | `MOVING` + effort → croit tenir | `right_holding()` **ignore MOVING** |
| Servo wrist trop agressif | Δxy 6–29 cm → diverge | `WRIST_MAX_DELTA_XY = 0.02`, gain 0.5, 4 iters max |
| Yaw 0°/90° « axes carrés » | Dérègle le bras en pleine prise | **Retiré** ; `WRIST_YAW_ENABLE = False` ; garder `RIGHT_PICK_QUAT` orga fixe |
| YOLO | Torch Docker trop vieux | Abandonné → LAB/HSV + LiDAR |
| GT pour le bras | Anti-triche | Jamais ; `GT_COMPARE` labo only |

**Idée gardée pour plus tard (pas en mid-grasp) :** orienter la pince parallèle / perpendiculaire aux bords du colis — sans yaw pendant le servo.

---

## 6. Config actuelle (points clés)

Dans `config.py` (même dossier) :

```python
PERCEPTION_ONLY = False   # False = mission complète
TOUCH_TEST = False
GT_COMPARE = False        # True seulement en labo CSV

FORCE_PARCEL_NAME = "parcel_1"  # test unitaire ; None = tous les colis pour l’équipe

RIGHT_CLAW_TIP_OFFSET = [0.02, 0.01, -0.005]
RIGHT_PICK_IK_Z = -0.005

WRIST_SERVO_ITERS = 4
WRIST_MAX_DELTA_XY = 0.02
WRIST_YAW_ENABLE = False
WRIST_REQUIRE_SEE_BEFORE_CLOSE = True
WRIST_VISION_ONLY_GATE = True
WRIST_SKIP_CLAW_HOLD_CHECK = True   # focus vision ; le grip peut être ajusté par contrôle
```

**Noms colis ↔ couleurs (approx.) :**

| Name | Couleur |
|------|---------|
| `parcel_1` | brun / gris |
| `parcel_2` | jaune |
| `parcel_3` | orange |
| `parcel_4` | bleu |

Avant un push « pour toute l’équipe », envisager `FORCE_PARCEL_NAME = None`.

---

## 7. Comment relancer (reproduire)

```bash
cd ~/leju-kuavo-challenge-cup-2026

# Toujours stopper avant un nouveau run
bash docker/stop_scene1.sh

# Mission complète (seed, timeout s)
bash docker/run_scene1_mission.sh 30 900

# Perception seule (pas de bras)
# → PERCEPTION_ONLY = True dans config.py
bash docker/run_scene1_local.sh

# Multi-seed détection tête
bash docker/run_scene1_multiseed_perception.sh 0 1 2 3
```

**Logs utiles :**

```bash
# Mission
grep -E 'DETECT|COLOR|FUSE|WRIST|VISION|Grabbed|DONE|claw' scene1_mission_run.log

# Perception only
grep -E 'DETECT|COLOR|FUSE|REPORT|DONE' scene1_local_run.log
```

Succès vision saisie (exemple seed 30) : lignes du type `VISION OK`, `claw R=3`, Grabbed, puis pesée.  
Une chute au handoff **ne remet pas en cause** la détection tête/main.

**Ctrl+C** dans le terminal n’arrête pas Docker → utiliser `stop_scene1.sh`.

---

## 8. Techniques / couches perception

1. **LiDAR** — OÙ (clusters XY table, grille 2×2 de secours)  
2. **RGB tête (LAB/HSV)** — QUI (couleur → nom)  
3. **Depth tête** — 3D caméra / ray  
4. **Fusion** — associe couleur ↔ amas ; err_structure_2x2 = géométrie relative (pas seed0 absolu)  
5. **Caméra poignet** — peaufinage local avant close (pas de gros sauts)

Critères labo « détection OK » (tête) :

1. `named == 4/4`  
2. `err_structure_2x2 < 0.05` m  
3. `colorish == 4`  
4. landmarks `lm=2/2` (bonus)

---

## 9. Qui fait quoi (recommandé)

| Vision | Contrôle / mission |
|--------|---------------------|
| Qualité multi-seed, couleurs, wrist servo (frac ↓) | Hold réel, tip par couleur, handoff sans chute |
| `perception.py`, `wrist_vision.py`, seuils `config` | `actions.py` (grip / passation / bac) |
| Ne pas réintroduire GT → bras | Ne pas casser le gate vision sans raison |

---

## 10. Prochaines pistes (vision)

- Robustesse multi-seed sans `FORCE_PARCEL_NAME`  
- Calibrer signes servo wrist (`WRIST_SERVO_SIGN_*`) pour que `frac` / Δpx **améliore** toujours  
- Moins de faux blobs (ciel / doigts) — déjà filtrés par depth + ROI  
- Yaw / axes carrés : seulement hors boucle de descente, si besoin

---

## Liens

- Labo artefacts : `labo/scene1/` (racine repo)  
- Ancienne reprise 11–12 juil. : `docs/reprise_session/OU_JEN_ETAIS.md` (périmé pour la mission)
