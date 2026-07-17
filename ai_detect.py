# ai_detect.py — AI-powered marker detection for Easy Rigify.
# Body/fingers: encrypted ONNX (models/body_pose.rmodel) + BVH geometry.
# Face: not supported (MediaPipe removed).

import bpy
import os
import math
import random
import tempfile
import shutil

import numpy as np
from mathutils import Vector, Matrix

from .constants import ALL_MARKERS, BODY_SIZE, FINGER_SIZE, FACE_SIZE, CORE_FACE_LANDMARKS, FULL_FACE_LANDMARKS
from .constants import dbg, LITE_BUILD
from .utils import get_or_create_collection, make_empty, _mesh_bbox_world


# ── Dependency management ─────────────────────────────────────────────────────
#
# onnxruntime + Pillow ship as BUNDLED WHEELS (blender_manifest.toml `wheels =
# [...]`, ARP-style): Blender installs the platform/Python-matching wheel into
# a managed site-packages directory and puts it on sys.path automatically when
# the addon is enabled, before register() runs. So no pip, no install button,
# no manual sys.path management — a plain import is enough. If a user is on a
# platform/Python combo with no bundled wheel (rare; see the wheels= comment
# in the manifest for what's covered), the import fails and AI features report
# unavailable — the existing geometric-fallback paths handle that gracefully.

_BODY_MODEL_FILENAME = os.path.join("models", "body_pose.rmodel")

# Session-level caches — filesystem stat and import check done once per session,
# not on every panel draw (which fires 60+ times/second during viewport interaction).
_is_available_cache:      object = None   # True / False / None=unchecked
_body_onnx_avail_cache:   object = None   # True / False / None=uncheckedæ


def is_available():
    """Return True if the AI deps (onnxruntime + Pillow) are importable."""
    global _is_available_cache
    if _is_available_cache is None:
        try:
            import onnxruntime  # noqa: F401
            import PIL  # noqa: F401  (Pillow — image loading has no fallback)
            _is_available_cache = True
        except ImportError:
            _is_available_cache = False
    return _is_available_cache

def _scene_mesh_or_none(context, obj):
    """A saved mesh-picker pointer can reference a datablock that is no longer
    linked to the current scene (e.g. the file was saved with a picker set in a
    previous session, then the object was replaced/appended over). Detecting an
    invisible mesh produces markers that match nothing on screen — treat such
    pointers as unset."""
    if obj is None or obj.type != 'MESH':
        return None
    if obj.name not in context.scene.objects:
        dbg(f"[detect] Body Mesh picker points to '{obj.name}' which is not "
              f"in this scene — ignoring it")
        return None
    return obj


def _body_onnx_path():
    addon_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(addon_dir, _BODY_MODEL_FILENAME)

def is_body_onnx_available():
    global _body_onnx_avail_cache
    if _body_onnx_avail_cache is None:
        try:
            import onnxruntime  # noqa: F401
            import PIL  # noqa: F401
            _body_onnx_avail_cache = os.path.isfile(_body_onnx_path())
        except ImportError:
            _body_onnx_avail_cache = False
    return _body_onnx_avail_cache


def effective_finger_engine(context):
    """The engine Detect Fingers will actually run.

    Lite ships no neural models, so a stored AUTO/NEURAL/TEMPLATE choice has
    nothing to run — a .blend saved with the Full edition keeps its scene
    property when opened in Lite, so the stored value is corrected here rather
    than only in the UI (which a keymap or script can bypass entirely).
    """
    eng = getattr(context.scene, 'finger_detection_engine', 'AUTO')
    if LITE_BUILD and eng != 'GEOMETRIC':
        dbg(f"[fingers] Lite build — engine {eng} unavailable (no models), "
            f"using GEOMETRIC")
        return 'GEOMETRIC'
    return eng


# ── Orthographic rendering ────────────────────────────────────────────────────

_BODY_IMG_SIZE = 512

_BODY_MARKERS = [
    # Centre spine
    "PELVIS", "SPINE_001", "SPINE_002", "CHEST", "NECK", "HEAD",
    # Arms (L then R)
    "SHOULDER_L", "ARM_L", "ELBOW_L", "HAND_L",
    "SHOULDER_R", "ARM_R", "ELBOW_R", "HAND_R",
    # Legs (L then R)
    "THIGH_L", "SHIN_L", "FOOT_L", "HEEL_L", "TOES_L",
    "THIGH_R", "SHIN_R", "FOOT_R", "HEEL_R", "TOES_R",
]


# ── Orthographic rendering helpers ───────────────────────────────────────────


def _build_bvh(obj):
    """Build a world-space BVHTree from obj. Caller reuses it across multiple snap calls."""
    import bmesh as _bmesh
    from mathutils.bvhtree import BVHTree
    bm = _bmesh.new()
    bm.from_mesh(obj.data)
    bm.transform(obj.matrix_world)
    bvh = BVHTree.FromBMesh(bm)
    bm.free()
    return bvh


def _estimate_body_from_mesh(obj):
    """
    Estimate body landmark world positions from mesh bounding box + BVH raycasting.
    Used when ONNX body detection fails (grey renders, T-pose, stylized proportions).
    Returns {mp_landmark_name: Vector}.
    """
    import bmesh as _bmesh
    from mathutils.bvhtree import BVHTree

    mn, mx, cen = _mesh_bbox_world(obj)
    h  = mx.z - mn.z
    cx = cen.x

    bm = _bmesh.new()
    bm.from_mesh(obj.data)
    bm.transform(obj.matrix_world)
    bvh = BVHTree.FromBMesh(bm)
    bm.free()

    far_y = mn.y - 1.0
    fwd   = Vector((0, 1, 0))

    def snap_y(x, z):
        hit1, _, _, _ = bvh.ray_cast(Vector((x, far_y, z)), fwd)
        if hit1 is None:
            return cen.y
        last = hit1
        for _ in range(128):
            nxt, _, _, _ = bvh.ray_cast(last + fwd * 1e-4, fwd)
            if nxt is None:
                break
            last = nxt
        return hit1.y + (last.y - hit1.y) * 0.5

    def left_edge(z):
        for dz in (0.0, h * 0.01, -h * 0.01):
            hit, _, _, _ = bvh.ray_cast(Vector((mx.x + 1.0, cen.y, z + dz)), Vector((-1, 0, 0)))
            if hit:
                return hit.x
        return mx.x

    def right_edge(z):
        for dz in (0.0, h * 0.01, -h * 0.01):
            hit, _, _, _ = bvh.ray_cast(Vector((mn.x - 1.0, cen.y, z + dz)), Vector((1, 0, 0)))
            if hit:
                return hit.x
        return mn.x

    Z = lambda t: mn.z + h * t

    # ── Neck-anchored proportions ─────────────────────────────────────────────
    # Fixed fractions of TOTAL height assume adult proportions: on a chibi the
    # head is ~half the body, so Z(0.65-0.90) IS the head — shoulders, ears and
    # the spine top all landed on it. Find the neck from the width profile
    # (narrowest slab in the upper region with a wider head ABOVE and wider
    # shoulders BELOW — the face head-locator's rule) and anchor the body
    # proportions to NECK height. Adults: neck ≈ 0.87h → anchors ≈ unchanged.
    neck_z = None
    try:
        _nv = len(obj.data.vertices)
        _cf = np.empty(_nv * 3, dtype=np.float64)
        obj.data.vertices.foreach_get("co", _cf)
        _m4 = np.array(obj.matrix_world, dtype=np.float64)
        _cw = _cf.reshape(-1, 3) @ _m4[:3, :3].T + _m4[:3, 3]
        _zs, _xs = _cw[:, 2], _cw[:, 0]
        _nsl = 40
        _zlo, _zhi = mn.z + 0.30 * h, mx.z - 0.03 * h
        _wd, _zc = [], []
        for k in range(_nsl):
            z0 = _zlo + (_zhi - _zlo) * k / _nsl
            z1 = z0 + (_zhi - _zlo) / _nsl
            m = (_zs >= z0) & (_zs < z1)
            if int(m.sum()) >= 5:
                _wd.append(float(np.percentile(_xs[m], 95)
                                 - np.percentile(_xs[m], 5)))
            else:
                _wd.append(float('nan'))
            _zc.append((z0 + z1) * 0.5)
        best_k = None
        for k in range(2, _nsl - 2):
            w = _wd[k]
            if not (w == w):                 # NaN slab
                continue
            frac = (_zc[k] - mn.z) / h
            if not (0.45 <= frac <= 0.92):
                continue
            _above = [x for x in _wd[k + 1:] if x == x]
            _below = [x for x in _wd[:k] if x == x]
            if not _above or not _below:
                continue
            # Standard pinch: wider head above + wider shoulders below.
            # Relaxed pinch: a STRONGLY wider head above (chibi) only needs a
            # faint widening below — chibi shoulders are often NARROWER than
            # the head, which made the standard rule unsatisfiable.
            _wa, _wb = max(_above), max(_below)
            if ((_wa > w * 1.15 and _wb > w * 1.20)
                    or (_wa > w * 1.50 and _wb > w * 1.05)):
                if best_k is None or w < _wd[best_k]:
                    best_k = k
        if best_k is not None:
            neck_z = _zc[best_k]
            dbg(f"[body-est] neck at z={neck_z:.3f} "
                  f"({(neck_z - mn.z) / h * 100:.0f}% of height) "
                  f"-- proportions neck-anchored")
        else:
            # HEAD-BALL fallback: no pinch anywhere (baby chibi — the head
            # blends straight into the body). If the widest slab of the whole
            # scan sits in the upper region AND about one radius below the
            # crown (the ball signature; a T-pose arm span fails the crown
            # test), treat it as the head's equator: neck = one radius below.
            _valid = [(k, w) for k, w in enumerate(_wd) if w == w]
            if _valid:
                k_eq, w_eq = max(_valid, key=lambda t: t[1])
                frac_eq = (_zc[k_eq] - mn.z) / h
                _crown_gap = mx.z - _zc[k_eq]
                if (frac_eq >= 0.45
                        and abs(_crown_gap - w_eq * 0.5) < w_eq * 0.35):
                    neck_z = max(_zc[k_eq] - w_eq * 0.55, mn.z + h * 0.25)
                    dbg(f"[body-est] no neck pinch -- head-ball fallback: "
                          f"equator z={_zc[k_eq]:.3f} width={w_eq:.3f} -> "
                          f"neck at z={neck_z:.3f} "
                          f"({(neck_z - mn.z) / h * 100:.0f}% of height)")
    except Exception as _e:
        dbg(f"[body-est] neck scan skipped ({_e})")

    if neck_z is not None:
        _bh    = max(neck_z - mn.z, h * 0.2)     # ground -> neck
        z_sh   = neck_z - _bh * 0.06
        arm_z  = neck_z - _bh * 0.09
        z_hip  = mn.z + _bh * 0.55
        z_knee = mn.z + _bh * 0.29
        z_ank  = mn.z + _bh * 0.075
        z_foot = mn.z + _bh * 0.02
        z_ear  = (neck_z + mx.z) * 0.5           # mid-head
    else:
        z_sh, arm_z = Z(0.80), Z(0.78)
        z_hip, z_knee, z_ank, z_foot = Z(0.50), Z(0.27), Z(0.07), Z(0.02)
        z_ear = Z(0.90)

    wrist_l_x = left_edge(arm_z)
    wrist_r_x = right_edge(arm_z)
    sh_l_x    = left_edge(z_sh if neck_z is not None else Z(0.65))
    sh_r_x    = right_edge(z_sh if neck_z is not None else Z(0.65))
    elbow_l_x = (sh_l_x + wrist_l_x) * 0.5
    elbow_r_x = (sh_r_x + wrist_r_x) * 0.5

    out = {}
    def add(name, x, z):
        out[name] = Vector((x, snap_y(x, z), z))

    add("LEFT_SHOULDER",    sh_l_x,  z_sh)
    add("RIGHT_SHOULDER",   sh_r_x,  z_sh)
    add("LEFT_HIP",         cx + h * 0.12,  z_hip)
    add("RIGHT_HIP",        cx - h * 0.12,  z_hip)
    add("LEFT_EAR",         cx + h * 0.04,  z_ear)
    add("RIGHT_EAR",        cx - h * 0.04,  z_ear)
    add("LEFT_ELBOW",       elbow_l_x,      (z_sh + arm_z) * 0.5)
    add("RIGHT_ELBOW",      elbow_r_x,      (z_sh + arm_z) * 0.5)
    add("LEFT_WRIST",       wrist_l_x,      arm_z)
    add("RIGHT_WRIST",      wrist_r_x,      arm_z)
    add("LEFT_KNEE",        cx + h * 0.10,  z_knee)
    add("RIGHT_KNEE",       cx - h * 0.10,  z_knee)
    add("LEFT_ANKLE",       cx + h * 0.08,  z_ank)
    add("RIGHT_ANKLE",      cx - h * 0.08,  z_ank)
    add("LEFT_FOOT_INDEX",  cx + h * 0.08,  z_foot)
    add("RIGHT_FOOT_INDEX", cx - h * 0.08,  z_foot)

    # ── Geodesic extremity upgrade ────────────────────────────────────────────
    # The template's edge rays at FIXED heights assume a T-pose: on arms-down /
    # A-pose characters the lateral silhouette at 78% height is the SHOULDER,
    # so wrist=elbow=shoulder (the "arm markers on shoulders" graft failure).
    # Geodesic extremities find hands/feet for any pose and any proportions;
    # wrist/elbow/ankle come from surface-distance band centroids (inside the
    # limb). The template above stays as both the fallback and the provider of
    # everything the geodesic pass doesn't cover (shoulders, hips, ears...).
    try:
        from .ai_detect_geo import body_extremities_geodesic
        _geo = body_extremities_geodesic(obj)
    except Exception as _e:
        dbg(f"[body-geo] unavailable: {_e}")
        _geo = None
    if _geo:
        _map = {
            "LEFT_WRIST":       "wrist_l",  "RIGHT_WRIST":      "wrist_r",
            "LEFT_ELBOW":       "elbow_l",  "RIGHT_ELBOW":      "elbow_r",
            "LEFT_SHOULDER":    "shoulder_l", "RIGHT_SHOULDER": "shoulder_r",
            "LEFT_ANKLE":       "ankle_l",  "RIGHT_ANKLE":      "ankle_r",
            "LEFT_FOOT_INDEX":  "foot_l",   "RIGHT_FOOT_INDEX": "foot_r",
        }
        _n_up = 0
        for mp_name, gk in _map.items():
            if gk in _geo:
                # Shoulders above the DETECTED neck are anatomically
                # impossible — on a chibi the geodesic arm-chain's torso
                # explosion can fire at head level (the arm merges into the
                # head-blob), and its override then stomped the neck-anchored
                # template shoulders (arm markers ended up on the head sides).
                if (neck_z is not None and gk in ("shoulder_l", "shoulder_r")
                        and _geo[gk].z > neck_z):
                    continue
                out[mp_name] = _geo[gk].copy()
                _n_up += 1
        dbg(f"[body-geo] {_n_up} landmarks from geodesic extremities")

    # The DETECTED neck (when found) is authoritative for the NECK marker:
    # the default derivation lerps the shoulder midpoint 25% toward the EARS,
    # which on a big-headed chibi drags the neck — and the whole interpolated
    # spine chain with it — up into the head.
    if neck_z is not None:
        out["NECK_HINT"] = Vector((cx, snap_y(cx, neck_z), neck_z))

    return out


# ── Keypoint → rig marker positions ──────────────────────────────────────────

# Spine interpolation t-values derived from Rigify metarig A-pose proportions:
#   PELVIS(1.000) SPINE_001(1.160) SPINE_002(1.298) CHEST(1.434) NECK(1.582)
_SPINE_T = {
    "SPINE_001": 0.275,
    "SPINE_002": 0.512,
    "CHEST":     0.746,
}


def _build_marker_positions(kp3d, mesh_cen_x=None):
    """
    Map reconstructed 3D keypoints to rig body marker names.
    Returns {marker_name: Vector}.  Fingers and face not included.

    mesh_cen_x: bbox centre X of the body mesh.  When provided, midline markers
    are pinned to this X instead of being estimated from detection midpoints.
    Prevents detector asymmetry from drifting the spine off the centre line.
    """
    def get(n):
        return kp3d.get(n)

    def mid(a, b):
        va, vb = get(a), get(b)
        if va is not None and vb is not None:
            return (va + vb) * 0.5
        return va if va is not None else vb

    def lerp(a, b, t):
        return a + (b - a) * t if (a is not None and b is not None) else None

    pos = {}

    # ── Pelvis ───────────────────────────────────────────────────────────────
    pelvis = mid("LEFT_HIP", "RIGHT_HIP")
    if pelvis is not None:
        pos["PELVIS"] = pelvis

    # ── Neck ─────────────────────────────────────────────────────────────────
    sh_mid  = mid("LEFT_SHOULDER", "RIGHT_SHOULDER")
    ear_mid = mid("LEFT_EAR", "RIGHT_EAR")
    if get("NECK_HINT") is not None:
        # Geometric path with a width-profile-DETECTED neck: authoritative.
        # The ear-lerp below drags the neck into a big stylized head (chibi),
        # and the interpolated spine chain follows it up.
        pos["NECK"] = get("NECK_HINT").copy()
    elif sh_mid is not None:
        if ear_mid is not None:
            neck = lerp(sh_mid, ear_mid, 0.25)
        else:
            neck = Vector((sh_mid.x, sh_mid.y, sh_mid.z + 0.08))
        pos["NECK"] = neck

    # ── Head ─────────────────────────────────────────────────────────────────
    if ear_mid is not None:
        if pos.get("NECK") is not None:
            neck_z = pos["NECK"].z
            head_z = ear_mid.z + (ear_mid.z - neck_z) * 0.35
        else:
            head_z = ear_mid.z + 0.06
        pos["HEAD"] = Vector((ear_mid.x, ear_mid.y, head_z))

    # ── Spine chain (PELVIS → NECK) ───────────────────────────────────────────
    if pos.get("PELVIS") is not None and pos.get("NECK") is not None:
        for marker_name, t in _SPINE_T.items():
            result = lerp(pos["PELVIS"], pos["NECK"], t)
            if result is not None:
                pos[marker_name] = result

    # ── Arms ─────────────────────────────────────────────────────────────────
    for mp_s, arp_s in (("LEFT", "L"), ("RIGHT", "R")):
        sh  = get(f"{mp_s}_SHOULDER")
        elb = get(f"{mp_s}_ELBOW")
        wri = get(f"{mp_s}_WRIST")

        if sh  is not None: pos[f"SHOULDER_{arp_s}"] = sh
        if elb is not None: pos[f"ELBOW_{arp_s}"]    = elb
        if wri is not None: pos[f"HAND_{arp_s}"]     = wri
        arm = lerp(sh, elb, 0.2)
        if arm is not None: pos[f"ARM_{arp_s}"]      = arm

    # ── Legs ─────────────────────────────────────────────────────────────────
    for mp_s, arp_s in (("LEFT", "L"), ("RIGHT", "R")):
        hip   = get(f"{mp_s}_HIP")
        knee  = get(f"{mp_s}_KNEE")
        ankle = get(f"{mp_s}_ANKLE")
        fi    = get(f"{mp_s}_FOOT_INDEX")  # big-toe tip

        if hip   is not None: pos[f"THIGH_{arp_s}"] = hip
        if knee  is not None: pos[f"SHIN_{arp_s}"]  = knee
        if ankle is not None: pos[f"FOOT_{arp_s}"]  = ankle
        if fi    is not None: pos[f"TOES_{arp_s}"]  = fi

        # HEEL: ankle X, ground-level Z (same as toe tip), ankle depth Y
        if ankle is not None and fi is not None:
            pos[f"HEEL_{arp_s}"] = Vector((ankle.x, ankle.y, fi.z))

    # ── Symmetrize bilateral pairs around body center X ──────────────────────
    if mesh_cen_x is not None:
        # Prefer the exact mesh bbox centre — immune to detector asymmetry
        cx = mesh_cen_x
    else:
        _cx_samples = []
        for base in ("THIGH", "SHOULDER"):
            lv = pos.get(f"{base}_L"); rv = pos.get(f"{base}_R")
            if lv and rv:
                _cx_samples.append((lv.x + rv.x) * 0.5)
        cx = sum(_cx_samples) / len(_cx_samples) if _cx_samples else (pos["PELVIS"].x if "PELVIS" in pos else 0.0)

    # Force all midline markers to that X
    for mid_name in ("PELVIS", "SPINE_001", "SPINE_002", "CHEST", "NECK", "HEAD"):
        if mid_name in pos:
            v = pos[mid_name]
            pos[mid_name] = Vector((cx, v.y, v.z))
    for base in ("SHOULDER", "ARM", "ELBOW", "HAND", "THIGH", "SHIN", "FOOT", "HEEL", "TOES"):
        lk, rk = f"{base}_L", f"{base}_R"
        if lk not in pos or rk not in pos:
            continue
        lv, rv = pos[lk], pos[rk]
        avg_off = (abs(lv.x - cx) + abs(rv.x - cx)) * 0.5
        avg_z   = (lv.z + rv.z) * 0.5
        avg_y   = (lv.y + rv.y) * 0.5
        l_sign  = 1 if lv.x >= cx else -1
        pos[lk] = Vector((cx + l_sign * avg_off, avg_y, avg_z))
        pos[rk] = Vector((cx - l_sign * avg_off, avg_y, avg_z))

    return pos


def _snap_heels_back(obj, marker_pos, bounds, bvh=None):
    """Snap HEEL markers to the back surface of the mesh (raycast from +Y toward -Y)."""
    if bvh is None:
        bvh = _build_bvh(obj)

    max_y = bounds["cen_y"] + bounds["ortho_scale"] * 2.0
    bwd   = Vector((0, -1, 0))

    for key in ("HEEL_L", "HEEL_R"):
        if key not in marker_pos:
            continue
        p = marker_pos[key]
        hit, _, _, _ = bvh.ray_cast(Vector((p.x, max_y, p.z)), bwd)
        if hit is not None:
            marker_pos[key] = Vector((p.x, hit.y, p.z))

    return marker_pos


def _center_in_limb(bvh, pt, mx_y, min_depth=0.012,
                    axis_origin=None, axis_dir=None, max_perp=None):
    """
    Move `pt` to the interior midpoint of the limb it sits in, casting along -Y.

    Casts a ray through the mesh at (pt.x, pt.z), collects every surface crossing,
    groups them into solid spans (entry→exit pairs), then returns pt placed at the
    midpoint of the span nearest pt.y. Unlike a simple front+back midpoint, this
    ignores other geometry in the same X/Z column (a torso/thigh behind a hand)
    and always lands *inside* the correct limb.

    axis_origin / axis_dir / max_perp: optional limb-axis lateral gate. When given,
    any candidate interior point (and the find_nearest fallback) farther than
    max_perp perpendicular from the line (axis_origin + t*axis_dir) is rejected as
    off-limb geometry (e.g. torso behind an arm hugging the body).

    Falls back to find_nearest + inward-normal push (by min_depth) when the ray
    finds no clean span. Returns a new Vector; never returns a point on the
    surface (always at least min_depth deep when possible).
    """
    EPS  = 5e-4   # 0.5 mm — step past each hit so the next cast can't re-hit it
    down = Vector((0.0, -1.0, 0.0))

    _ax_o = Vector(axis_origin) if axis_origin is not None else None
    _ax_d = Vector(axis_dir).normalized() if axis_dir is not None else None

    def _on_axis(p):
        if _ax_o is None or _ax_d is None or max_perp is None:
            return True
        rel  = Vector(p) - _ax_o
        perp = (rel - _ax_d * rel.dot(_ax_d)).length
        return perp <= max_perp

    # Collect all surface Y-crossings along the column, marching +Y → -Y.
    ys = []
    o  = Vector((pt.x, mx_y + 0.5, pt.z))
    for _ in range(32):
        hit, _, _, _ = bvh.ray_cast(o, down)
        if hit is None:
            break
        # Deduplicate near-identical crossings (numerical re-hits of one face).
        if not ys or abs(ys[-1] - hit.y) > EPS:
            ys.append(hit.y)
        o = Vector((pt.x, hit.y - EPS, pt.z))

    # Build candidate solid spans from every consecutive crossing pair, then keep
    # only the ones whose midpoint is actually INSIDE the mesh — verified by a
    # short downward cast that hits a back-face (normal points up, same hemisphere
    # as the cast origin). This is robust to non-manifold / odd-parity meshes
    # where strict entry/exit pairing breaks.
    best_mid = None
    best_d   = float('inf')
    for i in range(len(ys) - 1):
        y0, y1 = ys[i + 1], ys[i]   # ys is descending (we marched down); y1 > y0
        if (y1 - y0) < 1e-3:
            continue
        mid = (y0 + y1) * 0.5
        if not _is_inside_y(bvh, pt.x, mid, pt.z):
            continue
        if not _on_axis(Vector((pt.x, mid, pt.z))):
            continue   # off the limb axis → torso/other geometry, skip
        d = abs(mid - pt.y)
        if d < best_d:
            best_d, best_mid = d, mid
    if best_mid is not None:
        return Vector((pt.x, best_mid, pt.z))

    # Fallback: nearest surface point, pushed inward along the inward normal —
    # but only if it's on the limb axis (avoid snapping into an adjacent torso).
    loc, nrm, _, _ = bvh.find_nearest(pt)
    if loc is not None and _on_axis(loc):
        inward = -Vector(nrm).normalized()
        return Vector(loc) + inward * min_depth
    return pt.copy()


def _is_inside_y(bvh, x, y, z):
    """
    True if (x, y, z) is inside the mesh, tested via the back-face rule along Y:
    cast downward; if the first hit's normal points the SAME way as the ray
    (i.e. we exit through a back-face), the origin was inside the solid.
    """
    hit, nrm, _, _ = bvh.ray_cast(Vector((x, y, z)), Vector((0.0, -1.0, 0.0)))
    if hit is None or nrm is None:
        return False
    return nrm.y < 0.0   # exiting surface faces downward → origin was inside


def _center_in_limb_cross(bvh, pt, axis_dir, radius):
    """
    Center `pt` in the limb CROSS-SECTION perpendicular to `axis_dir` — i.e. in ALL
    directions across the limb, not just front-back. `_center_in_limb` only centers
    along Y, so a joint sitting on TOP of the wrist (high Z) or off to the side (X)
    stays on the surface. This casts 6 paired rays in the plane perpendicular to the
    limb axis and returns the average of the opposite-wall midpoints, putting the
    marker truly inside the limb regardless of which way it was off-surface.

    `radius` caps each ray so the nearest wall (the limb itself) is found before any
    far geometry (the torso behind an arm). Returns a new Vector, or None if no
    clean hits.
    """
    import math
    a = Vector(axis_dir)
    if a.length < 1e-6:
        return None
    a = a.normalized()
    u = Vector((0.0, 0.0, 1.0)).cross(a)
    if u.length < 1e-4:
        u = Vector((0.0, 1.0, 0.0)).cross(a)
    u = u.normalized()
    v = a.cross(u).normalized()
    p = Vector(pt)
    mids = []
    for j in range(6):
        th = j * math.pi / 3.0
        d  = (u * math.cos(th) + v * math.sin(th)).normalized()
        hp, _, _, _ = bvh.ray_cast(p + d * 5e-4,  d, radius)
        hn, _, _, _ = bvh.ray_cast(p - d * 5e-4, -d, radius)
        if hp is not None and hn is not None:
            mids.append((Vector(hp) + Vector(hn)) * 0.5)
        elif hp is not None:               # pt on the -d wall → centre toward +d hit
            mids.append((p + Vector(hp)) * 0.5)
        elif hn is not None:               # pt on the +d wall → centre toward -d hit
            mids.append((p + Vector(hn)) * 0.5)
    if len(mids) < 2:
        return None
    return sum(mids, Vector()) / len(mids)


# ── Body anatomy post-processing ─────────────────────────────────────────────

def _fix_body_anatomy(mesh_obj, marker_pos, bvh=None, bounds=None):
    """
    Single consolidated anatomical cleanup pass run after _detect_body_onnx
    (which now returns RAW triangulated landmarks). All proportional ratios are
    named module constants (_ARM_T, _ELBOW_T, _SHIN_BLEND, …) so there is one
    authoritative place to tune them.

    v5 changes vs v4:
      - Detector cleanups 1–7 removed (they were overwritten here anyway). The
        two that mattered are now folded in: wrist clamp (step 0a) and spine
        straightening (step 4c).

    Bilateral symmetry is always enforced (Rigify metarigs are symmetric).

    Order: 0a wrist clamp · 0 shoulder socket · 1 ARM · 2 ELBOW · 3 SHIN ·
    3b FOOT · 4 foot chain · 4c spine straighten · 5 spine/head X · 5b thigh
    anchor · 6 symmetry · 7 elbow/hand Y-center.
    Character assumed to face -Y (Rigify standard): heel at +Y back, toes at -Y front.
    """
    dbg("[Anatomy] _fix_body_anatomy v5 running (single consolidated pass)")
    _mn, _mx, cen = _mesh_bbox_world(mesh_obj)
    cx = cen.x

    # ── 0a. Wrist coherence: clamp HAND to a realistic forearm length ─────────
    # Heatmap triangulation can place the wrist far down the hand or off-mesh.
    # If HAND is more than _max_forearm from ELBOW, re-derive it by extrapolating
    # the SHOULDER→ELBOW upper-arm direction. (Formerly cleanup 1 in the detector,
    # which was the only detector cleanup not overwritten here — moved in.)
    #
    # _MAX_FOREARM is tuned for a ~1.75 m human. The scale normalizer keeps most
    # characters near that, but an in-band figure with unusually long arms (heroic
    # proportions) can exceed 0.55 m legitimately, so scale the limit by the
    # character's own height — never BELOW the tuned value, so short/chibi
    # characters aren't over-clamped.
    _max_forearm = max(_MAX_FOREARM, (_mx.z - _mn.z) * (_MAX_FOREARM / _DETECT_TARGET_H))
    for _wn, _en, _sn in (("HAND_L", "ELBOW_L", "SHOULDER_L"),
                          ("HAND_R", "ELBOW_R", "SHOULDER_R")):
        _w = marker_pos.get(_wn); _e = marker_pos.get(_en); _s = marker_pos.get(_sn)
        if not (_w and _e and _s):
            continue
        if (_w - _e).length > _max_forearm:
            _adir = _e - _s
            _alen = _adir.length
            if _alen > 1e-6:
                _adir = _adir / _alen
            _wnew = _e + _adir * min(_max_forearm * _WRIST_CLAMP_FRAC,
                                     _alen * 0.8)
            # Guard against a mis-detected upper arm: this re-derivation extends the
            # SHOULDER→ELBOW direction, which is only valid when the elbow sits OUTBOARD
            # of the shoulder. On an extreme bent pose the model can place the elbow
            # INBOARD of the shoulder (toward the torso), so that direction points ACROSS
            # the body and the "clamped" wrist lands on the wrong side of the midline —
            # then bilateral symmetry averages the damage onto the good arm (the golem
            # arms-collapse-to-torso bug). Only accept the clamp if it keeps the wrist on
            # its own side; otherwise the raw triangulated hand is the safer signal.
            if (_wnew.x - cx) * (_w.x - cx) >= 0:
                marker_pos[_wn] = _wnew

    # ── 0. SHOULDER X: trust the AI, apply a surface guard only ───────────────────
    # Every geometric re-placement tried (chest-socket cast, elbow cap, armpit
    # cast) mis-fired on some body type — pushing onto the torso (wide chests) or
    # collapsing to the neck (narrow shoulders / off-centre torsos), because no
    # single fixed Z is the "armpit" across all proportions. The AI's detected
    # shoulder X has been the most reliable signal in every log, so we keep it and
    # only correct a shoulder that is actually OUTSIDE the mesh: cast inward from
    # beyond it at its own Y/Z; if it's more lateral than the surface there, inset
    # it just inside. A shoulder already inside the body is left untouched.
    _s0_body_h = _mx.z - _mn.z
    _s0_inset  = max(0.015, _s0_body_h * 0.012)
    for side in ('L', 'R'):
        _sh0_key = f"SHOULDER_{side}"
        _sh0 = marker_pos.get(_sh0_key)
        if _sh0 is None or bvh is None:
            continue
        _s0_sign = 1.0 if side == 'L' else -1.0
        _far_x   = (_mx.x + 0.5) if side == 'L' else (_mn.x - 0.5)
        _sg_hit, _, _, _ = bvh.ray_cast(
            Vector((_far_x, _sh0.y, _sh0.z)),
            Vector((-_s0_sign, 0, 0)))
        if _sg_hit is None:
            continue
        if abs(_sh0.x - cx) > abs(_sg_hit.x - cx) - _s0_inset:
            _new_x = _sg_hit.x - _s0_sign * _s0_inset
            dbg(f"[Anatomy] {_sh0_key}: outside mesh, inset "
                  f"{abs(_sh0.x - cx)*100:.1f}cm → {abs(_new_x - cx)*100:.1f}cm")
            marker_pos[_sh0_key] = Vector((_new_x, _sh0.y, _sh0.z))

    # ── 1. ARM: _ARM_T along SHOULDER→HAND ───────────────────────────────────
    for side in ('L', 'R'):
        sh = marker_pos.get(f"SHOULDER_{side}")
        hw = marker_pos.get(f"HAND_{side}")
        if sh and hw:
            marker_pos[f"ARM_{side}"] = sh + (hw - sh) * _ARM_T

    # ── 1b. Surface guard: inset SHOULDER/ARM only if touching or outside mesh ─
    _sg_inset = max(0.015, min(0.020, _s0_body_h * 0.01667 + 0.00667))
    for side in ('L', 'R'):
        _sg_sign = 1.0 if side == 'L' else -1.0
        for _sg_key in (f"SHOULDER_{side}", f"ARM_{side}"):
            _sg_pt = marker_pos.get(_sg_key)
            if _sg_pt is None or bvh is None:
                continue
            _sg_hit_out, _, _, _sg_dist = bvh.ray_cast(
                Vector((_sg_pt.x, _sg_pt.y, _sg_pt.z)), Vector((_sg_sign, 0, 0)))
            if _sg_hit_out is not None and _sg_dist >= _sg_inset:
                continue  # well inside the mesh — skip
            _sg_hit_in, _, _, _ = bvh.ray_cast(
                Vector((_sg_pt.x + _sg_sign * 0.5, _sg_pt.y, _sg_pt.z)), Vector((-_sg_sign, 0, 0)))
            if _sg_hit_in is None:
                continue
            _sg_new_x = _sg_hit_in.x - _sg_sign * _sg_inset
            # Inward-only: a surface-inset guard pulls a marker that pokes OUT of the
            # skin back just inside — the inner surface is always closer to the body
            # centre than the marker was. If the candidate is MORE lateral than the
            # marker, the return ray hit the wrong surface: in a T-pose the ray fired
            # from 0.5 m outside starts BEYOND the outstretched hand, so its first hit
            # is the outer arm/wrist, which would snap the shoulder onto the hand.
            # Reject any outward move (this was the "shoulder into the arm" bug).
            if abs(_sg_new_x - cx) >= abs(_sg_pt.x - cx):
                continue
            marker_pos[_sg_key] = Vector((_sg_new_x, _sg_pt.y, _sg_pt.z))

    # ── 2. ELBOW: arm-axis blend then BVH Y midpoint inside the arm mesh ───────
    for side in ('L', 'R'):
        sh = marker_pos.get(f"SHOULDER_{side}")
        hw = marker_pos.get(f"HAND_{side}")
        el = marker_pos.get(f"ELBOW_{side}")
        if not (sh and hw and el):
            continue
        line_pos = sh + (hw - sh) * _ELBOW_T
        _hb_w = _ELBOW_BLEND          # weight of anatomical line
        _hm_w = 1.0 - _ELBOW_BLEND    # weight of heatmap
        new_el = Vector((
            el.x * _hm_w + line_pos.x * _hb_w,
            el.y * _hm_w + line_pos.y * _hb_w,
            el.z * _hm_w + line_pos.z * _hb_w,
        ))
        if bvh is not None:
            _hf, _, _, _ = bvh.ray_cast(Vector((new_el.x, _mx.y + 2.0, new_el.z)), Vector((0, -1, 0)))
            _hb, _, _, _ = bvh.ray_cast(Vector((new_el.x, _mn.y - 2.0, new_el.z)), Vector((0,  1, 0)))
            if _hf is not None and _hb is not None:
                new_el.y = (_hf.y + _hb.y) * 0.5
            elif _hf is not None:
                new_el.y = _hf.y - 0.01
            elif _hb is not None:
                new_el.y = _hb.y + 0.01
            else:
                pass  # both raycasts missed — keep the blended axis position
        marker_pos[f"ELBOW_{side}"] = new_el

    # ── 2b. SHOULDER: arm-angle diagnostic ───────────────────────────────────────
    # Compute the arm angle from the SHOULDER→ELBOW upper-arm vector and log it —
    # useful for diagnosing shoulder placement errors. Position correction is
    # deferred to the BVH surface-snap and bilateral symmetry passes below.
    for side in ('L', 'R'):
        _sh  = marker_pos.get(f"SHOULDER_{side}")
        _el  = marker_pos.get(f"ELBOW_{side}")
        if not (_sh and _el):
            continue
        _upper = _el - _sh
        _arm_angle_deg = math.degrees(
            math.atan2(_upper.z, math.sqrt(_upper.x ** 2 + _upper.y ** 2))
        )
        dbg(f"[Anatomy] SHOULDER_{side} arm angle: {_arm_angle_deg:.1f}°")

    # ── 3. SHIN: midpoint of THIGH–FOOT axis, Y-centered inside the mesh ─────────
    # The knee sits at the exact midpoint between the thigh and ankle.
    # A single Y-centering cast at that Z places the marker inside the leg volume
    # without narrowness scanning (which drifts to the widest or narrowest point
    # depending on character proportions).
    for side in ('L', 'R'):
        th = marker_pos.get(f"THIGH_{side}")
        sh = marker_pos.get(f"SHIN_{side}")
        ft = marker_pos.get(f"FOOT_{side}")
        if not (th and sh and ft):
            continue
        mid = th + (ft - th) * 0.5
        _sb_w = _SHIN_BLEND; _sm_w = 1.0 - _SHIN_BLEND
        new_shin = Vector((
            sh.x * _sm_w + mid.x * _sb_w,
            sh.y * _sm_w + mid.y * _sb_w,
            sh.z * _sm_w + mid.z * _sb_w,
        ))
        if bvh is not None:
            new_shin = _center_in_limb(bvh, new_shin, _mx.y)
        marker_pos[f"SHIN_{side}"] = new_shin

    # ── 3b. FOOT: proportional ankle placement ───────────────────────────────────
    # Y-narrowness scanning fails for wide-foot characters: the flat sole spans a
    # large heel-to-toe Y range at low heights, so the scan finds the calf (whose
    # cross-section is smaller than the foot-length Y-span) as the "narrowest"
    # point.  Instead, use the anatomical ratio: ankle ≈ 30% of the floor-to-knee
    # distance.  This is consistent across realistic and cartoon bipedal characters.
    # A tight clamp [floor+4%, floor+15%] guards against extreme proportions.
    _h = _mx.z - _mn.z
    for side in ('L', 'R'):
        th = marker_pos.get(f"THIGH_{side}")
        sh = marker_pos.get(f"SHIN_{side}")   # already corrected by step 3
        ft = marker_pos.get(f"FOOT_{side}")
        if not (th and sh and ft):
            continue
        seg = sh - th
        seg_len = seg.length
        if seg_len < 1e-6:
            continue
        axis_foot = sh + (seg / seg_len) * seg_len * _FOOT_AXIS_EXT
        _fb_w = _FOOT_BLEND; _fm_w = 1.0 - _FOOT_BLEND
        new_foot = Vector((
            ft.x * _fm_w + axis_foot.x * _fb_w,
            ft.y * _fm_w + axis_foot.y * _fb_w,
            ft.z * _fm_w + axis_foot.z * _fb_w,
        ))
        # Z: ankle sits at _ANKLE_FLOOR_FRAC of the floor-to-knee distance from the floor.
        _ankle_z = _mn.z + (sh.z - _mn.z) * _ANKLE_FLOOR_FRAC
        _ankle_z = max(_mn.z + _h * _ANKLE_Z_MIN_FRAC,
                       min(_ankle_z, _mn.z + _h * _ANKLE_Z_MAX_FRAC))
        new_foot.z = _ankle_z
        # Y-center at this Z with a single cast pair (no Z movement — Z is fixed
        # by the proportion above, not by narrowness).
        _ffound = False
        if bvh is not None:
            _fc = _center_in_limb(bvh, Vector((sh.x, new_foot.y, new_foot.z)), _mx.y)
            new_foot.y = _fc.y
            _ffound    = True
        # X-center the ankle: outer ray from bbox far side (first hit = outer ankle face),
        # inner ray from cx toward the leg (first hit = inner ankle face).
        # Bbox far side is safe at ankle Z — arms are never at ground level.
        if bvh is not None:
            _fx_sign = 1.0 if side == 'L' else -1.0
            _far_x   = _mx.x + 2.0 if side == 'L' else _mn.x - 2.0
            _fx_y    = new_foot.y if _ffound else sh.y
            _fxo, _, _, _ = bvh.ray_cast(
                Vector((_far_x, _fx_y, new_foot.z)),
                Vector((-_fx_sign, 0, 0)))
            _fxi, _, _, _ = bvh.ray_cast(
                Vector((cx, _fx_y, new_foot.z)),
                Vector((_fx_sign, 0, 0)))
            if _fxo is not None and _fxi is not None:
                new_foot.x = (_fxo.x + _fxi.x) * 0.5
        marker_pos[f"FOOT_{side}"] = new_foot

    # ── 4. FOOT CHAIN: enforce spatial ordering, snap HEEL to back surface ────
    # Character faces -Y: heel is at back (+Y side), toes at front (-Y side).
    # Minimum offsets scale with character height so short/stylised characters
    # still get usable bone lengths in the metarig.
    _foot_min = max(0.05, _h * 0.05)   # min heel-behind and toe-forward offset
    for side in ('L', 'R'):
        ft = marker_pos.get(f"FOOT_{side}")
        he = marker_pos.get(f"HEEL_{side}")
        to = marker_pos.get(f"TOES_{side}")
        if not (ft and he and to):
            continue
        he = he.copy()
        to = to.copy()
        if he.y <= ft.y + _foot_min * 0.3:   # heel must be behind foot (+Y)
            he.y = ft.y + _foot_min
        he.x = ft.x * 0.9 + he.x * 0.1
        # TOES = ball of foot (Rigify toe bone starts here, not at the tip).
        # Pull _TOES_BALL_FRAC of the way from ankle toward the toe tip = metatarsal joint.
        to.y = ft.y + (to.y - ft.y) * _TOES_BALL_FRAC
        if to.y >= ft.y - _foot_min:   # enforce minimum forward distance from ankle
            to.y = ft.y - _foot_min
        to.x = ft.x * 0.9 + to.x * 0.1
        to.z = _mn.z + 0.02       # floor level, same anchor as HEEL
        marker_pos[f"HEEL_{side}"] = he
        marker_pos[f"TOES_{side}"] = to

    # X-center TOES: same outer/inner bbox ray approach as FOOT.
    # At floor Z the foot is wider than the ankle, so we use the actual geometry.
    if bvh is not None:
        for side in ('L', 'R'):
            to = marker_pos.get(f"TOES_{side}")
            if to is None:
                continue
            _tx_sign = 1.0 if side == 'L' else -1.0
            _t_far_x = _mx.x + 2.0 if side == 'L' else _mn.x - 2.0
            to = to.copy()
            _txo, _, _, _ = bvh.ray_cast(
                Vector((_t_far_x, to.y, to.z)),
                Vector((-_tx_sign, 0, 0)))
            _txi, _, _, _ = bvh.ray_cast(
                Vector((cx, to.y, to.z)),
                Vector((_tx_sign, 0, 0)))
            if _txo is not None and _txi is not None:
                to.x = (_txo.x + _txi.x) * 0.5
            marker_pos[f"TOES_{side}"] = to

    # ── 4b-heel. Snap HEEL to back surface of mesh (mirrors _snap_heels_back) ─
    # Scan Z is always _mn.z + 0.02 (just above mesh floor) so the cast hits
    # the back of the heel geometry regardless of what the heatmap predicted for Z.
    if bvh is not None:
        _scan_y   = _mx.y + 2.0
        _bwd      = Vector((0, -1, 0))
        _heel_z   = _mn.z + 0.02
        for side in ('L', 'R'):
            he = marker_pos.get(f"HEEL_{side}")
            if he is None:
                continue
            hloc, _, _, _ = bvh.ray_cast(Vector((he.x, _scan_y, _heel_z)), _bwd)
            if hloc is not None:
                marker_pos[f"HEEL_{side}"] = Vector((he.x, hloc.y, _heel_z))

    # ── 4b. Tail rejection ────────────────────────────────────────────────────
    _chest_v  = marker_pos.get("CHEST")
    _pelvis_v = marker_pos.get("PELVIS")
    if _chest_v is not None and _pelvis_v is not None:
        if _pelvis_v.y - _chest_v.y > 0.15:
            for _sp in ("PELVIS", "SPINE_001", "SPINE_002"):
                _sv = marker_pos.get(_sp)
                if _sv is not None and _sv.y > _chest_v.y + 0.05:
                    marker_pos[_sp] = Vector((_sv.x, _chest_v.y + 0.05, _sv.z))

    # ── 4c. Spine chain straightening (formerly detector cleanup 6) ───────────
    # Lerp SPINE_001/002/CHEST onto the straight PELVIS→NECK line. Humanoid
    # spines are already nearly straight (no harm); on tailed characters the tail
    # pulls these markers posteriorly and this removes the drift entirely.
    _sp_pelv = marker_pos.get("PELVIS")
    _sp_neck = marker_pos.get("NECK")
    if _sp_pelv is not None and _sp_neck is not None:
        for _sp_name, _sp_t in zip(("SPINE_001", "SPINE_002", "CHEST"), _SPINE_TS):
            if _sp_name in marker_pos:
                marker_pos[_sp_name] = _sp_pelv * (1.0 - _sp_t) + _sp_neck * _sp_t

    # ── 5. Spine/head X: always forced to mesh centre ─────────────────────────
    for name in ("PELVIS", "SPINE_001", "SPINE_002", "CHEST", "NECK", "HEAD"):
        p = marker_pos.get(name)
        if p:
            marker_pos[name] = Vector((cx, p.y, p.z))

    # ── 5b. THIGH Y/Z: anchor to PELVIS depth and height ─────────────────────
    _pelv = marker_pos.get("PELVIS")
    if _pelv is not None:
        for side in ('L', 'R'):
            th = marker_pos.get(f"THIGH_{side}")
            if th is not None:
                marker_pos[f"THIGH_{side}"] = Vector((th.x, _pelv.y, _pelv.z))

    # ── 6. Bilateral symmetry (default ON, "Symmetrical Detect" toggle) ───────
    # Rigify metarigs are symmetric, so L/R are mirrored about the body centre
    # X: lateral offset, depth (Y) and height (Z) are averaged, then X is
    # mirrored from cx. This corrects any per-side detection error regardless
    # of pose. Toggle OFF (scene.autorig_detect_symmetry) keeps each side's raw
    # detection — for deliberately asymmetric characters.
    _BILATERAL = ("SHOULDER", "ARM", "ELBOW", "HAND",
                  "THIGH", "SHIN", "FOOT", "HEEL", "TOES")
    if not getattr(bpy.context.scene, "autorig_detect_symmetry", True):
        _BILATERAL = ()
        dbg("[symmetry] Symmetrical Detect OFF — keeping raw per-side body markers")
    for base in _BILATERAL:
        lv = marker_pos.get(f"{base}_L")
        rv = marker_pos.get(f"{base}_R")
        if lv is None or rv is None:
            continue
        avg_off = (abs(lv.x - cx) + abs(rv.x - cx)) * 0.5
        avg_y   = (lv.y + rv.y) * 0.5
        avg_z   = (lv.z + rv.z) * 0.5
        l_sign  = 1.0 if (lv.x - cx) >= 0 else -1.0
        marker_pos[f"{base}_L"] = Vector((cx + l_sign * avg_off, avg_y, avg_z))
        marker_pos[f"{base}_R"] = Vector((cx - l_sign * avg_off, avg_y, avg_z))

    # ── 7. Interior depth: pull ELBOW/HAND/SHIN/FOOT inside their limb ─────────
    # Runs LAST (after symmetry, which averages Y/Z and could otherwise push a
    # centered marker back toward the surface). _center_in_limb selects the solid
    # span the marker sits in — so a hand posed near the torso is centered in the
    # HAND, not averaged across the gap to the body. Rigify needs bone heads
    # inside the mesh, never on the surface.
    if bvh is not None:
        for side in ('L', 'R'):
            for _key in (f"ELBOW_{side}", f"HAND_{side}",
                         f"SHIN_{side}",  f"FOOT_{side}"):
                _pt = marker_pos.get(_key)
                if _pt is None:
                    continue
                marker_pos[_key] = _center_in_limb(bvh, _pt, _mx.y)

    return marker_pos


# ── ONNX body detector ───────────────────────────────────────────────────────

_BODY_ONNX_SESSIONS: dict = {}
_BODY_HMAP_SIZE           = 64

# ── Anatomical ratios (single source of truth for the cleanup pass) ───────────
# All proportional placements used by _fix_body_anatomy live here so tuning is
# centralized and the two old (conflicting) copies can never drift again.
_ARM_T            = 0.10        # ARM head = this fraction along SHOULDER→HAND
_ELBOW_T          = 0.57        # anatomical elbow = this fraction along SHOULDER→HAND
_ELBOW_BLEND      = 0.70        # weight of anatomical line vs heatmap for ELBOW
_SHIN_BLEND       = 0.70        # weight of THIGH–FOOT midpoint vs heatmap for SHIN
_FOOT_BLEND       = 0.70        # weight of leg-axis extension vs heatmap for FOOT
_FOOT_AXIS_EXT    = 1.05        # ankle = SHIN + leg_axis * seg_len * this
_ANKLE_FLOOR_FRAC = 0.25        # ankle Z = floor + (knee_z - floor) * this
_ANKLE_Z_MIN_FRAC = 0.04        # ankle Z clamp lower bound (fraction of height)
_ANKLE_Z_MAX_FRAC = 0.15        # ankle Z clamp upper bound (fraction of height)
_TOES_BALL_FRAC   = 0.60        # TOES = ankle + (toe_tip - ankle) * this (ball of foot)
_SPINE_TS         = (0.275, 0.512, 0.746)   # SPINE_001/002/CHEST lerp PELVIS→NECK
_MAX_FOREARM      = 0.55        # m; wrist-clamp threshold (HAND too far from ELBOW)
_WRIST_CLAMP_FRAC = 0.80        # clamped wrist = ELBOW + arm_dir * upper_len * this


def _detect_body_onnx(front_path, side_path, top_path, bounds, bvh):
    """
    Run the body pose model (models/body_pose.rmodel, heatmap-primary).
    Per-view heatmaps are peak-detected and triangulated via the stored ortho
    camera matrices, then a BVH Y-depth midpoint pass places each non-foot
    landmark inside the mesh volume. Returns these RAW landmarks; ALL anatomical
    cleanup (proportional placement, symmetry, foot derivation) is done by
    _fix_body_anatomy so there is a single authoritative cleanup pass.

    Projection inverses (from _project_body_front/side/top):
      front: wx = (hx - H/2) / H * fs + cen_x   wz = -((hy - H/2) / H * fs) + cen_z
      side:  wy = (hx - H/2) / H * ss + cen_y   wz = -((hy - H/2) / H * ss) + cen_z
      top:   wx = (hx - H/2) / H * ts + cen_x   wy = -((hy - H/2) / H * ts) + cen_y
    """
    import onnxruntime as ort
    from PIL import Image as _PIL

    _path = _body_onnx_path()
    if _path not in _BODY_ONNX_SESSIONS:
        from . import model_crypto
        _BODY_ONNX_SESSIONS[_path] = ort.InferenceSession(
            model_crypto.load_model_bytes(_path), providers=['CPUExecutionProvider'])
    sess = _BODY_ONNX_SESSIONS[_path]

    def _load(p):
        img = _PIL.open(p).convert("RGB").resize((_BODY_IMG_SIZE, _BODY_IMG_SIZE))
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.ascontiguousarray(arr.transpose(2, 0, 1)[None])

    hm_f_raw, hm_s_raw, hm_t_raw, _ = sess.run(
        None,
        {"front": _load(front_path), "side": _load(side_path), "top": _load(top_path)}
    )
    hm_f = hm_f_raw[0]   # [24, 64, 64]
    hm_s = hm_s_raw[0]
    hm_t = hm_t_raw[0]

    H   = float(_BODY_HMAP_SIZE)   # 64
    H2  = H / 2.0                   # 32
    ACT = 0.3                       # activation gate: dead channel if peak < this
    ACT_LO = 0.18                   # rescue gate: only for landmarks ACT would drop
                                    # entirely (extreme proportions fire at 0.2-0.29)

    def _peak(ch):
        """50%-threshold weighted centroid — prevents noise spikes from hijacking argmax."""
        mx = float(ch.max())
        if mx < 1e-6:
            return H2, H2
        w = ch * (ch >= mx * 0.5)
        mass = float(w.sum())
        if mass > 1e-6:
            yy, xx = np.mgrid[0:ch.shape[0], 0:ch.shape[1]]
            return float((xx * w).sum() / mass), float((yy * w).sum() / mass)
        flat = int(ch.argmax())
        return float(flat % int(H)), float(flat // int(H))

    # ── Ortho triangulation ───────────────────────────────────────────────────
    cx  = bounds["cen_x"];  cy_b = bounds["cen_y"];  cz = bounds["cen_z"]
    fs  = bounds.get("front_scale", bounds.get("ortho_scale", 1.0))
    ss  = bounds.get("side_scale",  bounds.get("ortho_scale", 1.0))
    ts  = bounds.get("top_scale",   bounds.get("ortho_scale", 1.0))

    N   = len(_BODY_MARKERS)
    lm  = np.zeros((N, 3), dtype=np.float64)
    # Joints whose X or Z has no supporting view: that axis defaults to the body
    # centre (cx/cz) and the joint lands on the torso ("hand/elbow on the body").
    # Track them and leave them OUT of the result so the geometric graft
    # (_estimate_body_from_mesh) fills them on the actual limb. Finger "graft" pattern.
    undet = [False] * N

    for i in range(N):
        af  = float(hm_f[i].max())
        as_ = float(hm_s[i].max())
        at  = float(hm_t[i].max())

        fx, fy = _peak(hm_f[i])
        sx, sy = _peak(hm_s[i])
        tx, ty = _peak(hm_t[i])

        wx_f = (fx - H2) / H * fs + cx
        wz_f = -((fy - H2) / H * fs) + cz
        wy_s = (sx - H2) / H * ss + cy_b
        wz_s = -((sy - H2) / H * ss) + cz
        wx_t = (tx - H2) / H * ts + cx
        wy_t = -((ty - H2) / H * ts) + cy_b   # top py is also negated

        xs = ([wx_f] if af > ACT else []) + ([wx_t] if at > ACT else [])
        ys = ([wy_s] if as_ > ACT else []) + ([wy_t] if at > ACT else [])
        zs = ([wz_f] if af > ACT else []) + ([wz_s] if as_ > ACT else [])

        # Unreliable if X or Z has NO supporting view: X comes from front/top, Z from
        # front/side. With neither, that axis defaults to the body CENTRE (cx/cz) and
        # the joint lands on the torso (e.g. a hand seen only in the side view: front &
        # top below ACT -> X = cx). Leave it out so the geometric graft fills it. Y is
        # exempt — the BVH depth pass re-derives it regardless.
        undet[i] = (not xs) or (not zs)
        if undet[i]:
            # Rescue tier: on extreme proportions (Hulk-like) the model fires
            # at 0.2-0.29 on limb landmarks — just under the trusted gate. A
            # borderline peak is still the model's best guess and lands near
            # the true joint (the anatomy pass cleans it up), which beats
            # dropping it and letting the geometric graft park a hand on the
            # shoulder. Only consulted when the landmark would otherwise be
            # dropped entirely; anything under ACT_LO stays a graft.
            xs = ([wx_f] if af > ACT_LO else []) + ([wx_t] if at > ACT_LO else [])
            ys = ([wy_s] if as_ > ACT_LO else []) + ([wy_t] if at > ACT_LO else [])
            zs = ([wz_f] if af > ACT_LO else []) + ([wz_s] if as_ > ACT_LO else [])
            if xs and zs:
                undet[i] = False
                dbg(f"  [ONNX-v2] {_BODY_MARKERS[i]} rescued at low gate "
                      f"(front={af:.2f} side={as_:.2f} top={at:.2f})")

        lm[i, 0] = sum(xs) / len(xs) if xs else cx
        lm[i, 1] = sum(ys) / len(ys) if ys else cy_b
        lm[i, 2] = sum(zs) / len(zs) if zs else cz

    # ── Anatomical cleanup moved to _fix_body_anatomy ────────────────────────
    # This function now returns *raw* triangulated landmarks (plus the BVH
    # Y-depth pass below). All proportional / symmetry / foot-derivation cleanup
    # lives in _fix_body_anatomy so there is a single authoritative pass and no
    # duplicated, conflicting constants. (Previously cleanups 1–7 here were
    # almost entirely overwritten by _fix_body_anatomy seconds later.)

    FOOT_LM_NAMES = frozenset(("HEEL_L", "HEEL_R", "TOES_L", "TOES_R"))

    # ── BVH Y-depth midpoint for non-foot landmarks ────────────────────────────
    ray_top = bounds.get("mx_y", cy_b + 1.0) + 2.0
    ray_bot = bounds.get("mn_y", cy_b - 1.0) - 2.0
    up_v  = Vector((0, -1, 0))
    fwd_v = Vector((0,  1, 0))

    marker_pos = {}
    for i, name in enumerate(_BODY_MARKERS):
        if undet[i]:
            continue   # leave MISSING -> filled by the geometric graft on the real limb

        wx     = float(lm[i, 0])
        wz     = float(lm[i, 2])
        wy_tri = float(lm[i, 1])

        if name in FOOT_LM_NAMES:
            marker_pos[name] = Vector((wx, wy_tri, wz))
            continue

        hf, _, _, _ = bvh.ray_cast(Vector((wx, ray_top, wz)), up_v)
        hb, _, _, _ = bvh.ray_cast(Vector((wx, ray_bot, wz)), fwd_v)
        if hf is not None and hb is not None:
            wy = (hf.y + hb.y) * 0.5
        elif hf is not None:
            wy = hf.y - 0.01
        elif hb is not None:
            wy = hb.y + 0.01
        else:
            # Both casts missed — landmark is in empty space (classic tail confusion).
            # Full 3D snap to nearest mesh surface is better than the wrong triangulated Y.
            _pt_nn = Vector((wx, wy_tri, wz))
            _loc_nn, _, _, _dist_nn = bvh.find_nearest(_pt_nn)
            if _loc_nn is not None and _dist_nn < 0.5:
                marker_pos[name] = Vector(_loc_nn)
                continue
            wy = wy_tri
        marker_pos[name] = Vector((wx, wy, wz))

    # Collapse indicator (anomaly-only): a healthy detection spans most of the
    # body — warn when the landmark spread collapses toward the centre.
    if marker_pos:
        _xs = [p.x for p in marker_pos.values()]
        _zs = [p.z for p in marker_pos.values()]
        _bw = max(bounds.get("ortho_scale", 1.0), 1e-6)
        if (max(_xs) - min(_xs)) / _bw < 0.20 or (max(_zs) - min(_zs)) / _bw < 0.35:
            dbg(f"[ONNX-v2] WARNING low spread: X={max(_xs)-min(_xs):.2f}m "
                  f"Z={max(_zs)-min(_zs):.2f}m  (frame {_bw:.2f}m) "
                  f"-- detection may have collapsed")

    _n_undet = sum(undet)
    undet_names = {n for i, n in enumerate(_BODY_MARKERS) if undet[i]}
    if _n_undet:
        # Peak diagnostics: a peak just under ACT (0.3) means the gate is too
        # strict for this character (rescuable); ~0 means the model is blind
        # to it (training-data problem). Essential for triaging hands/elbows
        # grafted onto the torso on extreme proportions (Hulk-like).
        for i, name in enumerate(_BODY_MARKERS):
            if undet[i]:
                dbg(f"  [ONNX-v2] {name:12s} peaks: front={float(hm_f[i].max()):.2f} "
                      f"side={float(hm_s[i].max()):.2f} top={float(hm_t[i].max()):.2f} "
                      f"(gate {ACT})")
    dbg(f"[ONNX-v2] {len(marker_pos)}/{len(_BODY_MARKERS)} raw landmarks via heatmap triangulation"
          + (f"  ({_n_undet} undetected -> geometric graft: "
             + ", ".join(sorted(undet_names)) + ")"
             if _n_undet else ""))
    return marker_pos, undet_names


# ── Hand / finger detection ───────────────────────────────────────────────────

_HAND_IMG_SIZE        = 256
_HAND_SCALE           = 0.35   # fallback constant (used by front/top renders that lack ew)
_ORBIT_N_VIEWS        = 8      # number of evenly-spaced orbit views per hand
_DEBUG_SAVE_ORBIT     = False  # DEBUG: copy live orbit renders to ./debug_orbit for inspection


def _hand_scale(arm_len):
    """Ortho scale proportional to forearm length. Adult 30 cm forearm → ~21 cm view."""
    return max(0.10, arm_len * 0.70)


def _hand_finger_offset(arm_len):
    """Orbit-centre offset proportional to forearm. Adult 30 cm → ~10 cm (palm midpoint)."""
    return arm_len * 0.33



# ── Multi-view orbit rendering and triangulation ──────────────────────────────

def _orbit_cam_basis(hw, ew):
    """
    Build the coordinate frame for arm-axis orbit renders.
    Returns (cen, fwd, ref, perp, hand_scale, dist).
    - cen        : orbit centre (wrist shifted toward fingertips)
    - fwd        : arm direction = camera up in every orbit view
    - ref        : first perp axis (camera position at θ=0)
    - perp       : second perp axis (camera position at θ=90°)
    - hand_scale : ortho world-units for this character's hand
    - dist       : camera distance from orbit centre
    fwd, ref, perp form a right-handed frame: perp = fwd × ref.
    """
    if ew is not None:
        _av  = hw - ew
        _al  = _av.length
        fwd  = _av / _al if _al > 0.001 else Vector((1.0, 0.0, 0.0))
    else:
        _al  = _HAND_SCALE / 1.20   # canonical adult fallback when ew unknown
        fwd  = Vector((1.0, 0.0, 0.0))

    hs       = _hand_scale(_al)
    dist     = hs * 1.5
    cen      = hw + fwd * _hand_finger_offset(_al)
    # Orbit starts from directly above the hand (12 o'clock).
    # Project world_up perpendicular to the arm so ref always points "as upward as
    # possible" while staying ⊥ to fwd.  fwd.cross(old_world_up) gave the side view
    # (9 o'clock); Gram-Schmidt gives the top view instead.
    world_up = Vector((0.0, 0.0, 1.0)) if abs(fwd.z) < 0.9 else Vector((1.0, 0.0, 0.0))
    ref      = (world_up - fwd * fwd.dot(world_up)).normalized()
    perp     = fwd.cross(ref).normalized()
    return cen, fwd, ref, perp, hs, dist


def _wrist_from_mesh(mesh_obj, elbow, tip):
    """Locate the WRIST as the narrowest cross-section between the forearm and the
    palm, driven by the ELBOW->HAND_TIP axis.

    Why this exists: the HAND (wrist) marker frequently lands up the forearm or on
    the palm, and that single bad position corrupts finger framing, hand isolation
    and the geodesic field (the user had to drag it onto the wrist by hand every
    time). Body detection's wrist finder scanned relative to the HAND marker itself
    - so a bad HAND poisoned its own correction. Here the scan axis is ELBOW->TIP:
    both are far more stable than HAND (the fingertip is a geodesic extremity), so
    the narrowest-slice wrist is found even when HAND is badly placed.

    Returns a world-space Vector (the wrist centre) or None if it can't measure.
    """
    if elbow is None or tip is None:
        return None
    _dir = tip - elbow
    _dl  = _dir.length
    if _dl < 1e-4:
        return None
    _dir = _dir / _dl
    dg = bpy.context.evaluated_depsgraph_get()
    me = mesh_obj.evaluated_get(dg)
    mw = mesh_obj.matrix_world
    arm = []                                   # (world_vert, along-axis distance)
    for v in me.data.vertices:
        wv  = mw @ v.co
        rel = wv - elbow
        al  = rel.dot(_dir)
        if al <= 0:                            # behind the elbow
            continue
        if (rel - _dir * al).length > 0.22 * _dl:   # off the arm/hand cylinder
            continue
        arm.append((wv, al))
    if len(arm) < 8:
        return None
    reach = max(al for _, al in arm)           # elbow -> fingertip
    if reach < 1e-4:
        return None

    def _ctr(pts):
        s = Vector((0.0, 0.0, 0.0))
        for p in pts:
            s += p
        return s / len(pts)

    _N = 22
    prof = []
    for _i in range(_N):
        _frac = 0.30 + _i * (1.00 - 0.30) / (_N - 1)
        _tt   = _frac * reach
        slab  = [v for v, al in arm if abs(al - _tt) < 0.045 * reach]
        if len(slab) < 4:
            prof.append(None)
            continue
        c0 = _ctr(slab)
        r  = sum((v - c0 - _dir * ((v - c0).dot(_dir))).length for v in slab) / len(slab)
        prof.append((r, c0, _frac))
    vp = [(_i, p) for _i, p in enumerate(prof) if p is not None]
    if not vp:
        return None
    # PALM = widest cross-section in the hand region (frac 0.55-0.95).
    palm = [(_i, p) for _i, p in vp if 0.55 <= p[2] <= 0.95]
    if not palm:
        return min(vp, key=lambda ip: ip[1][0])[1][1]
    palm_i = max(palm, key=lambda ip: ip[1][0])[0]
    # WRIST = narrowest cross-section BEFORE the palm (never into the fingers past it).
    pre = [p for _i, p in vp if _i < palm_i and p[2] >= 0.40]
    if not pre:
        pre = [p for _i, p in vp if _i <= palm_i]
    return min(pre, key=lambda p: p[0])[1] if pre else None


def _estimate_hand_tip(mesh_obj, hw, fwd, radius=0.40, fallback_dist=0.18,
                       max_perp=None):
    """
    Estimate the longest-fingertip world position.

    The fingertip is the vertex that reaches farthest forward (along the
    elbow->wrist direction `fwd`) from the wrist `hw`.  We take the cluster of
    vertices nearest that farthest-reaching point and average them, so a single
    stray spike or a small gap in the topology doesn't define the tip.  This
    matches the geometric arm detection (farthest vertex = longest finger) and,
    unlike the previous "closest to arm axis" heuristic, it works for spread or
    curved hands where the longest finger is off the arm axis.

    max_perp: if set, reject vertices farther than this perpendicular distance
    from the wrist→fwd axis. This keeps the tip on the HAND when the arm hugs
    the torso (without it, a nearby torso vertex can be "farther forward" than
    the real fingertip and hijack the result). Defaults to radius * 0.6.
    """
    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        obj_eval  = mesh_obj.evaluated_get(depsgraph)
        hw_v  = Vector(hw)
        fwd_v = Vector(fwd).normalized()
        _perp_lim = max_perp if max_perp is not None else radius * 0.6

        candidates = []        # (proj, world_pos) for vertices forward of wrist
        max_proj   = 0.0
        for v in obj_eval.data.vertices:
            wp   = obj_eval.matrix_world @ v.co
            diff = wp - hw_v
            if diff.length > radius:
                continue
            proj = diff.dot(fwd_v)
            if proj < 0.02:
                continue
            # Lateral gate: drop anything too far off the arm axis (= torso).
            perp = (diff - fwd_v * proj).length
            if perp > _perp_lim:
                continue
            candidates.append((proj, wp))
            if proj > max_proj:
                max_proj = proj

        if not candidates or max_proj < 0.03:
            return Vector(hw) + Vector(fwd) * fallback_dist

        # Farthest-reaching vertex = longest fingertip.
        tip_v = max(candidates, key=lambda c: c[0])[1]
        # Average the small cluster around it for stability.
        cluster = sorted(candidates, key=lambda c: (c[1] - tip_v).length_squared
                         )[:max(1, len(candidates) // 40)]
        s = Vector((0.0, 0.0, 0.0))
        for _proj, wp in cluster:
            s += wp
        return s / len(cluster)
    except Exception:
        pass
    return Vector(hw) + Vector(fwd) * fallback_dist


def _mesh_hand_extent(mesh_obj, hw_v, fwd, radius):
    """Measure the hand's length from MESH geometry: the span of vertices within
    `radius` of the wrist, projected onto the forward axis `fwd`. Used to sanity-
    check the HAND_TIP marker (a marker dropped off the hand inflates the frame
    scale and poisons the whole pipeline). Returns extent in world units, or None."""
    try:
        _eval = mesh_obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mw    = _eval.matrix_world
        proj  = [(mw @ v.co - hw_v).dot(fwd)
                 for v in _eval.data.vertices
                 if (mw @ v.co - hw_v).length < radius]
        if proj:
            return max(proj) - min(proj)
    except Exception:
        pass
    return None


def _mesh_hand_spread(mesh_obj, center, fwd, reach):
    """Radius from `center` to the farthest hand-region MESH vertex, mirroring the
    training framing recipe (gen_hand_training_data: scale = 2.5 * max dist from the
    finger centroid). Lets the orbit frame adapt to the ACTUAL hand size (big / widely
    spread hand) instead of an arm-ratio guess.

    Restricts to a TUBE around the finger axis: vertices within a lateral radius of
    the axis, spanning from the proximal palm/thumb base (a little behind the centre)
    out to the fingertips. The torso sits OFF to the side (large lateral distance) and
    the forearm sits far behind on-axis, so both are excluded -- without the tube, a
    hand raised next to the body grabbed body geometry and blew the frame up. The
    proximal allowance lets the thumb base / palm be measured so the frame covers them
    (matching training, whose radius includes the thumb-base marker). Returns radius or
    None."""
    try:
        _eval   = mesh_obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mw      = _eval.matrix_world
        c       = Vector(center)
        f       = Vector(fwd).normalized()
        lat_max = reach * 0.60          # lateral tube radius (keeps fingers, drops body)
        best    = 0.0
        n       = 0
        for v in _eval.data.vertices:
            d     = mw @ v.co - c
            along = d.dot(f)
            # Include the proximal palm/thumb base (a little behind the centre); only
            # the far-behind forearm is dropped here, the torso by the lateral cap.
            if along < -reach * 0.45 or along > reach:   # forearm behind / past fingers
                continue
            if (d - f * along).length > lat_max:         # off to the side -> arm/body
                continue
            dl = d.length
            if dl > best:
                best = dl
            n += 1
        if n >= 8 and best > 0.01:
            return best
    except Exception:
        pass
    return None


def _render_hand_orbit(mesh_obj, hw, ew, temp_dir, prefix, tip=None, center=None, scale=None, orbit_fwd=None):
    """
    Render _ORBIT_N_VIEWS ortho crops of the hand, orbiting around the forearm axis.
    Camera up = arm direction (fwd) in every view, so image-Y always encodes
    finger depth and image-X encodes the lateral spread at that angle.
    tip: world-space fingertip position (MARKER_HAND_TIP).  When provided the
         camera centre and scale are derived exactly from hw→tip, with no vertex
         measurement needed.
    Returns [(img_path, right_vec, up_vec), ...].
    """
    scene   = bpy.context.scene
    display = scene.display
    cen, fwd, ref, perp, hand_scale, dist = _orbit_cam_basis(hw, ew)

    if tip is not None:
        # Exact framing from the two explicit markers — no heuristics needed.
        _hw      = Vector(hw)
        _tip     = Vector(tip)
        _fwd     = (_tip - _hw)
        _hlen    = _fwd.length
        # Sanity-check the HAND_TIP marker against the mesh. A marker dropped off
        # the hand inflates the frame scale (e.g. a 1.2m "hand") and poisons the
        # whole pipeline. If the marker is implausibly far vs the mesh extent near
        # the wrist, ignore it and fall back to the geometry path below.
        if _hlen > 0.05:
            _mext = _mesh_hand_extent(mesh_obj, _hw, _fwd / _hlen,
                                      radius=max(0.50, _hlen * 1.5))
            if _mext is not None and _mext > 0.03 and _hlen > 2.5 * _mext:
                dbg(f"  [orbit] HAND_TIP marker looks misplaced: "
                      f"{_hlen*1000:.0f}mm vs mesh ~{_mext*1000:.0f}mm — using geometry")
                tip = None
        if tip is not None and _hlen > 0.05:
            fwd        = _fwd / _hlen
            hand_scale = _hlen * 1.0
            cen        = _hw + _fwd * 0.6
            dist       = hand_scale * 1.5
            dbg(f"  [orbit] tip-marker  hand_length={_hlen*1000:.0f}mm  scale={hand_scale:.3f}m")
            # Rebuild ref/perp with the corrected fwd direction
            world_up = Vector((0.0, 0.0, 1.0)) if abs(fwd.z) < 0.9 else Vector((1.0, 0.0, 0.0))
            ref  = (world_up - fwd * fwd.dot(world_up)).normalized()
            perp = fwd.cross(ref).normalized()
    if tip is None:
        # Fallback: measure hand extent from isolated mesh geometry.
        try:
            _eval  = mesh_obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
            _hw    = Vector(hw)
            _fwd   = (Vector(hw) - Vector(ew)).normalized() if ew is not None else Vector((1.0, 0.0, 0.0))
            _all_proj = [(_eval.matrix_world @ v.co - _hw).dot(_fwd)
                         for v in _eval.data.vertices
                         if (_eval.matrix_world @ v.co - _hw).length < 0.50]
            if _all_proj:
                _min_p = min(_all_proj)
                _max_p = max(_all_proj)
                _hlen  = _max_p - _min_p
                if _hlen > 0.05:
                    hand_scale = _hlen * 1.1
                    cen        = _hw + _fwd * (_min_p + _max_p) * 0.5
                    dist       = hand_scale * 1.5
                    dbg(f"  [orbit] geometry    hand_length={_hlen*1000:.0f}mm  scale={hand_scale:.3f}m")
        except Exception:
            pass

    # Allow caller to pin the orbit centre, scale, and axis to values pre-computed from
    # LVT tip detection so the crop is stable regardless of wrist marker position.
    if center is not None:
        cen = center
    if scale is not None:
        hand_scale = scale
        dist       = scale * 1.5
    if orbit_fwd is not None:
        fwd  = Vector(orbit_fwd).normalized()
        world_up = Vector((0.0, 0.0, 1.0)) if abs(fwd.z) < 0.9 else Vector((1.0, 0.0, 0.0))
        ref  = (world_up - fwd * fwd.dot(world_up)).normalized()
        perp = fwd.cross(ref).normalized()
        dbg(f"  [orbit] anchor={tuple(round(x, 3) for x in cen)}  scale={hand_scale:.3f}  fwd={tuple(round(x, 3) for x in fwd)}  (LVT)")
    elif center is not None or scale is not None:
        dbg(f"  [orbit] anchor={tuple(round(x, 3) for x in cen)}  scale={hand_scale:.3f}  (LVT)")

    # NOTE: the crop is fitted to the ACTUAL hand extent AFTER isolation below (so the
    # measurement runs on the body-free isolated mesh) — see "[orbit] fit crop to hand".

    orig_cam = scene.camera;  orig_fp = scene.render.filepath
    orig_rx  = scene.render.resolution_x
    orig_ry  = scene.render.resolution_y
    orig_pct = scene.render.resolution_percentage
    orig_eng = scene.render.engine
    orig_fmt = scene.render.image_settings.file_format
    orig_bgt        = display.shading.background_type
    orig_bgc        = tuple(display.shading.background_color)
    orig_light      = display.shading.light
    orig_cavity     = display.shading.show_cavity
    orig_cav_type   = display.shading.cavity_type
    orig_cav_ridge  = display.shading.cavity_ridge_factor
    orig_cav_valley = display.shading.cavity_valley_factor
    orig_color_type   = display.shading.color_type
    orig_outline      = display.shading.show_object_outline
    orig_outline_col  = tuple(display.shading.object_outline_color)
    vs = scene.view_settings
    orig_vt  = vs.view_transform
    orig_lk  = vs.look

    from mathutils import Matrix
    cam_data = bpy.data.cameras.new("_er_orbit_cam")
    cam_data.type         = 'ORTHO'
    cam_data.ortho_scale  = hand_scale
    cam_data.display_size = 0.01
    cam_obj = bpy.data.objects.new("_er_orbit_cam", cam_data)
    scene.collection.objects.link(cam_obj)

    # Render an ARM-ISOLATED copy: cut the mesh just behind the ELBOW and beyond the
    # fingertips, then keep only the island connected to the hand. This removes the
    # chest / head / legs — so a hand resting near a thigh no longer renders the leg
    # into the crop — while KEEPING THE FOREARM context the model trains on. Falls
    # back to the full mesh if isolation fails. The generator renders through this
    # SAME function, so training and inference stay identical.
    # Render the base cage: disable subdivision before isolation so the evaluated
    # island (and any full-mesh fallback) use the control mesh — subdivision only
    # slows the render and adds no landmark signal. Restored in the finally below.
    _restore_subdiv = _disable_subdiv_modifiers(mesh_obj)
    _iso_obj = None
    try:
        from .ai_detect_lvt import _isolate_hand_mesh
        _iso_obj = _isolate_hand_mesh(mesh_obj, hw, ew, tip=tip,
                                      wrist_island=True, keep_forearm=True)
    except Exception as _e:
        dbg(f"  [orbit] hand isolation failed ({_e}) — rendering full mesh")
        _iso_obj = None

    # Fit the crop to the ACTUAL hand extent, measured from the isolated mesh (body-
    # free, so nothing nearby contaminates it; the forearm is dropped by the proximal
    # gate). This frames the hand snugly for ANY finger length or thumb splay:
    #   - a laterally-SPLAYED THUMB is included — the crop covers the max radius from
    #     the arm axis, and the orbit sees every lateral direction;
    #   - SHORT fingers leave no dead space above the tips — the distal edge tracks the
    #     real fingertips, not the fingertip-spread heuristic.
    # Recentre along the arm axis to the hand's mid-extent; size to the larger of the
    # fwd half-extent and the lateral radius. Generator + inference share this code, so
    # training and inference stay matched. (Fitting the render crop changes framing vs
    # the old fingertip-spread crop — regenerate + retrain to benefit fully.)
    if _iso_obj is not None and hand_scale and hand_scale > 1e-6:
        try:
            _fwd_n = Vector(fwd).normalized()
            _fn = np.array([_fwd_n.x, _fwd_n.y, _fwd_n.z])
            _hn = (np.array([hw[0], hw[1], hw[2]]) if hw is not None
                   else np.array([cen[0], cen[1], cen[2]]))
            _me = _iso_obj.data                     # verts are world-space (mw = identity)
            _nv = len(_me.vertices)
            if _nv >= 10:
                _co = np.empty(_nv * 3, dtype=np.float64)
                _me.vertices.foreach_get('co', _co)
                _wp    = _co.reshape(_nv, 3)
                _along = (_wp - _hn) @ _fn           # distance along the arm from the wrist

                # Body-bleed trim. On bulky characters (Hulk) a hand resting near the
                # torso/thigh BRIDGES into the wrist island, so isolation keeps a slab
                # of body and this fit then EXPANDS the crop to frame it (lat 1725mm on
                # a ~350mm hand -> scale 1.93 -> body fills every view -> garbage tips).
                # A hand is never laterally wider than it is long, so when we have a
                # sanity-checked HAND_TIP the tip->wrist length bounds a tube around the
                # finger axis: anything far outside it is body — DELETE it from the
                # render copy so both the frame AND the pixels stay hand-only. Anchored
                # on the TIP (stable, user-placed), not the wrist marker, per the
                # wrist-independence rule. No proximal bound: the forearm context the
                # model trains on is kept. Fires only on GROSS overshoot (>1.25x the
                # cap), so normal hands — and the body-free training renders, keeping
                # training/inference framing matched — are never nibbled.
                if tip is not None:
                    try:
                        _tn   = np.array([tip[0], tip[1], tip[2]], dtype=np.float64)
                        _tlen = float(np.linalg.norm(_tn - _hn))
                        if _tlen > 0.05:
                            _dt    = _wp - _tn
                            _alt   = _dt @ _fn                       # 0 at tip, <0 toward wrist
                            _latt  = np.linalg.norm(_dt - np.outer(_alt, _fn), axis=1)
                            _lcap  = _tlen * 0.95                    # covers a fully splayed thumb
                            _dcap  = _tlen * 0.35                    # margin past the tip marker
                            _tube  = (_alt <= _dcap) & (_latt - np.maximum(_alt, 0.0) <= _lcap)
                            _n_in  = int(_tube.sum())
                            _gross = (float(_latt[(_alt > -_tlen) & (_alt <= _dcap)].max())
                                      > _lcap * 1.25) if ((_alt > -_tlen) & (_alt <= _dcap)).any() else False
                            if _gross and 50 <= _n_in < _nv:
                                import bmesh
                                _bm = bmesh.new()
                                _bm.from_mesh(_me)
                                _bm.verts.ensure_lookup_table()
                                bmesh.ops.delete(
                                    _bm,
                                    geom=[_bm.verts[int(i)] for i in np.nonzero(~_tube)[0]],
                                    context='VERTS')
                                _bm.to_mesh(_me)
                                _bm.free()
                                dbg(f"  [orbit] trimmed body bleed: {_nv - _n_in} verts "
                                      f"outside hand tube (lat<={_lcap*1000:.0f}mm, "
                                      f"tip+{_dcap*1000:.0f}mm)")
                                _nv = len(_me.vertices)
                                _co = np.empty(_nv * 3, dtype=np.float64)
                                _me.vertices.foreach_get('co', _co)
                                _wp    = _co.reshape(_nv, 3)
                                _along = (_wp - _hn) @ _fn
                    except Exception as _te:
                        dbg(f"  [orbit] body-bleed trim skipped ({_te})")

                _keep  = _along >= -hand_scale * 0.10   # drop the forearm (proximal of wrist)
                if int(_keep.sum()) >= 10:
                    _wpk = _wp[_keep]
                    _alk = _along[_keep]
                    _amin, _amax = float(_alk.min()), float(_alk.max())
                    _cenv = (Vector((float(_hn[0]), float(_hn[1]), float(_hn[2])))
                             + _fwd_n * ((_amin + _amax) * 0.5))
                    _cn   = np.array([_cenv.x, _cenv.y, _cenv.z])
                    _d    = _wpk - _cn
                    _dl   = _d @ _fn
                    _lat  = np.linalg.norm(_d - np.outer(_dl, _fn), axis=1)
                    _fit  = 2.0 * max((_amax - _amin) * 0.5, float(_lat.max())) * 1.12
                    if _fit > 1e-4 and _fit < hand_scale * 0.40:
                        # Isolated piece is bogus — a fragmented mesh or a mis-seeded
                        # HAND_TIP made the flood-fill keep a scrap, so the fit collapsed
                        # far below the tip-spread scale. Discard the isolation and render
                        # the FULL mesh with the caller's framing (a hand-with-context
                        # crop beats a tiny wrong fragment, and is what worked pre-isolation).
                        dbg(f"  [orbit] isolation collapsed (fit {_fit:.3f} << "
                              f"{hand_scale:.3f}) -- rendering full mesh")
                        try:
                            _bad = _iso_obj.data
                            bpy.data.objects.remove(_iso_obj, do_unlink=True)
                            if _bad is not None:
                                bpy.data.meshes.remove(_bad, do_unlink=True)
                        except Exception:
                            pass
                        _iso_obj = None
                    elif _fit > 1e-4:
                        cen        = _cenv
                        hand_scale = _fit
                        dist       = hand_scale * 1.5
                        cam_data.ortho_scale = hand_scale
                        dbg(f"  [orbit] fit crop to hand: fwd={(_amax-_amin)*1000:.0f}mm "
                              f"lat={float(_lat.max())*2000:.0f}mm -> scale={hand_scale:.3f}")
        except Exception as _e:
            dbg(f"  [orbit] hand-fit framing skipped ({_e})")

    render_obj = _iso_obj if _iso_obj is not None else mesh_obj

    orig_mesh_hide_render = mesh_obj.hide_render
    render_obj.hide_render = False
    hidden = []
    for o in scene.objects:
        if o is not render_obj and not o.hide_render:
            o.hide_render = True; hidden.append(o)

    cam_data.clip_start = max(0.01, dist - hand_scale * 0.8)
    cam_data.clip_end   = dist + hand_scale * 0.6
    results = []
    try:
        scene.render.engine                    = 'BLENDER_WORKBENCH'
        scene.render.resolution_x              = _HAND_IMG_SIZE
        scene.render.resolution_y              = _HAND_IMG_SIZE
        scene.render.resolution_percentage     = 100
        scene.render.image_settings.file_format = 'PNG'
        scene.camera = cam_obj

        display.shading.background_type   = 'VIEWPORT'
        display.shading.background_color  = (0.03, 0.03, 0.03)
        display.shading.light             = 'STUDIO'
        display.shading.color_type        = 'OBJECT'
        display.shading.show_cavity          = False
        display.shading.show_object_outline  = True
        display.shading.object_outline_color = (0.0, 0.0, 0.0)
        vs.view_transform = 'Standard'
        vs.look           = 'None'

        for i in range(_ORBIT_N_VIEWS):
            theta     = i * (2.0 * math.pi / _ORBIT_N_VIEWS)
            orbit_dir = ref * math.cos(theta) + perp * math.sin(theta)
            cam_pos   = cen + orbit_dir * dist

            # Camera-up = arm direction (fwd).  orbit_dir is always ⊥ fwd so
            # fwd.cross(orbit_dir) is never degenerate — no gimbal-lock case needed.
            # This keeps image-Y = arm-axis projection, which is invariant to orbit
            # angle and gives the model a consistent, learnable py target.
            right_vec = fwd.cross(orbit_dir).normalized()
            up_vec    = orbit_dir.cross(right_vec).normalized()

            # Sanity: right × up must equal orbit_dir (camera +Z = away from scene).
            # A sign error in any axis would corrupt the camera vectors stored in
            # training JSONs, making the entire dataset inconsistent with inference.
            assert (right_vec.cross(up_vec) - orbit_dir).length < 1e-4, (
                f"Orbit camera basis is not right-handed at view {i} — "
                "check signs in right_vec / up_vec computation"
            )

            # Camera matrix: columns = [right, up, orbit_dir(=+Z = -look), position]
            cam_obj.matrix_world = Matrix([
                [right_vec.x, up_vec.x, orbit_dir.x, cam_pos.x],
                [right_vec.y, up_vec.y, orbit_dir.y, cam_pos.y],
                [right_vec.z, up_vec.z, orbit_dir.z, cam_pos.z],
                [0.0,         0.0,      0.0,          1.0      ],
            ])

            img_path = os.path.join(temp_dir, f"{prefix}_orbit_{i:02d}.png")
            scene.render.filepath = img_path
            bpy.ops.render.render(write_still=True)
            results.append((img_path, right_vec.copy(), up_vec.copy()))

            # DEBUG: mirror live orbit renders to a permanent per-character folder
            # (named by hand_scale in mm) so different characters don't overwrite.
            if _DEBUG_SAVE_ORBIT:
                try:
                    _dbg = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "debug_orbit", f"s{int(round(hand_scale*1000))}")
                    os.makedirs(_dbg, exist_ok=True)
                    import shutil as _sh
                    _sh.copy2(img_path, os.path.join(_dbg, f"{prefix}_orbit_{i:02d}.png"))
                except Exception:
                    pass

    finally:
        scene.camera                            = orig_cam
        scene.render.filepath                   = orig_fp
        scene.render.resolution_x               = orig_rx
        scene.render.resolution_y               = orig_ry
        scene.render.resolution_percentage      = orig_pct
        scene.render.engine                     = orig_eng
        scene.render.image_settings.file_format = orig_fmt
        display.shading.background_type         = orig_bgt
        display.shading.background_color        = orig_bgc
        display.shading.light                   = orig_light
        display.shading.show_cavity             = orig_cavity
        display.shading.cavity_type             = orig_cav_type
        display.shading.cavity_ridge_factor     = orig_cav_ridge
        display.shading.cavity_valley_factor    = orig_cav_valley
        display.shading.color_type              = orig_color_type
        display.shading.show_object_outline     = orig_outline
        display.shading.object_outline_color    = orig_outline_col
        vs.view_transform = orig_vt
        vs.look           = orig_lk
        mesh_obj.hide_render = orig_mesh_hide_render
        _restore_subdiv()
        for o in hidden: o.hide_render = False
        bpy.data.objects.remove(cam_obj, do_unlink=True)
        bpy.data.cameras.remove(cam_data, do_unlink=True)
        if _iso_obj is not None:
            try:
                _iso_mesh = _iso_obj.data
                bpy.data.objects.remove(_iso_obj, do_unlink=True)
                if _iso_mesh is not None:
                    bpy.data.meshes.remove(_iso_mesh, do_unlink=True)
            except Exception:
                pass

    return cen, results, hand_scale, dist


def _orbit_world_to_px(p, cen, right_vec, up_vec,
                        scale=_HAND_SCALE, S=_HAND_IMG_SIZE):
    """
    Project world point p into pixel coords for an orbit view.
    right_vec / up_vec are the camera's local X/Y axes (world space).
    Returns (px, py) — values outside [0, S] mean the point is out of frame.
    """
    dp    = p - cen
    x_cam = dp.dot(right_vec)
    y_cam = dp.dot(up_vec)
    px    = ( x_cam / scale + 0.5) * S
    py    = (0.5 - y_cam / scale)  * S
    return (px, py)


# Hand landmark name → rig base marker name
_MP_TO_ARP = {
    "THUMB_CMC":  "THUMB_1",          "THUMB_MCP":  "THUMB_2",
    "THUMB_IP":   "THUMB_3",          "THUMB_TIP":  "THUMB_TIP",
    "INDEX_MCP":  "FINGER_INDEX_1",   "INDEX_PIP":  "FINGER_INDEX_2",
    "INDEX_DIP":  "FINGER_INDEX_3",   "INDEX_TIP":  "FINGER_INDEX_TIP",
    "MIDDLE_MCP": "FINGER_MIDDLE_1",  "MIDDLE_PIP": "FINGER_MIDDLE_2",
    "MIDDLE_DIP": "FINGER_MIDDLE_3",  "MIDDLE_TIP": "FINGER_MIDDLE_TIP",
    "RING_MCP":   "FINGER_RING_1",    "RING_PIP":   "FINGER_RING_2",
    "RING_DIP":   "FINGER_RING_3",    "RING_TIP":   "FINGER_RING_TIP",
    "PINKY_MCP":  "FINGER_PINKY_1",   "PINKY_PIP":  "FINGER_PINKY_2",
    "PINKY_DIP":  "FINGER_PINKY_3",   "PINKY_TIP":  "FINGER_PINKY_TIP",
}

# Thumb lateral X offset from wrist per joint (default metarig, L side).
# Thumb extends outward in +X for L, −X for R.

# ── Geometric finger placement ────────────────────────────────────────────────

# (arm_projection_m, world_y_offset_m) relative to the wrist.
# arm_projection: distance along the forearm→finger direction.
# world_y_offset: Y spread (index = −Y, pinky = +Y; same sign for L and R).
# Values derived from Rigify Human Metarig default proportions.
_FINGER_GEOM = {
    "THUMB_1":           (0.008, -0.0295),
    "THUMB_2":           (0.052, -0.0506),
    "THUMB_3":           (0.081, -0.0661),
    "THUMB_TIP":         (0.106, -0.0793),
    "FINGER_INDEX_1":    (0.039, -0.0236),
    "FINGER_INDEX_2":    (0.086, -0.0256),
    "FINGER_INDEX_3":    (0.122, -0.0271),
    "FINGER_INDEX_TIP":  (0.146, -0.0279),
    "FINGER_MIDDLE_1":   (0.046,  0.0000),
    "FINGER_MIDDLE_2":   (0.098,  0.0000),
    "FINGER_MIDDLE_3":   (0.140,  0.0000),
    "FINGER_MIDDLE_TIP": (0.169,  0.0000),
    "FINGER_RING_1":     (0.039,  0.0236),
    "FINGER_RING_2":     (0.086,  0.0256),
    "FINGER_RING_3":     (0.122,  0.0271),
    "FINGER_RING_TIP":   (0.146,  0.0279),
    "FINGER_PINKY_1":    (0.030,  0.0459),
    "FINGER_PINKY_2":    (0.070,  0.0498),
    "FINGER_PINKY_3":    (0.100,  0.0529),
    "FINGER_PINKY_TIP":  (0.122,  0.0552),
}

_DEFAULT_FOREARM_LEN = 0.291  # (HAND_L − ELBOW_L).length in the default metarig


def _place_fingers_geometric(hand_pos, elbow_pos, arp_side):
    """
    Estimate finger positions from wrist + elbow world positions.
    Projects each finger along the forearm direction and spreads them
    perpendicular to the arm using default Rigify metarig proportions,
    scaled to the character's actual arm length.
    Works for any arm angle. Returns {marker_name: Vector}.
    """
    arm_vec    = hand_pos - elbow_pos
    actual_len = arm_vec.length
    if actual_len > 0.001:
        arm_dir = arm_vec / actual_len
        scale   = actual_len / _DEFAULT_FOREARM_LEN
    else:
        arm_dir = Vector((1.0 if arp_side == 'L' else -1.0, 0.0, -0.5)).normalized()
        scale   = 1.0

    # Spread direction for the index→pinky fan.
    # World-Y (depth) works for any arm angle up to ~53° from horizontal.
    # Switch to world-X only for truly near-vertical arms (hanging straight
    # down) where world-Y spread collapses to a single visible line.
    if abs(arm_dir.z) < 0.8:
        spread_dir = Vector((0.0, 1.0, 0.0))
    else:
        x_sign = 1.0 if arp_side == 'L' else -1.0
        spread_dir = Vector((x_sign, 0.0, 0.0))

    pos = {}
    for base_name, (arm_proj, y_off) in _FINGER_GEOM.items():
        p = hand_pos + arm_dir * (arm_proj * scale) + spread_dir * (y_off * scale)
        pos[f"{base_name}_{arp_side}"] = p
    return pos



# ── Face detection ────────────────────────────────────────────────────────────

_FACE_IMG_SIZE = 512   # higher resolution for the many fine face landmarks
_FACE_SCALE    = 0.38  # ortho world-units: covers full face (chin→forehead) + margin

# Face landmark index → rig face marker name.
# Convention: image-left (low px) = world negative X = character's RIGHT = _R
#             image-right (high px) = world positive X = character's LEFT  = _L
_FACE_LM = {
    # ── Centre / single markers ───────────────────────────────────────────
    "FACE_NOSE_TIP":          4,
    "FACE_NOSE_BRIDGE":       6,
    "FACE_NOSE_BOT":          2,
    "FACE_LIP_T":            13,
    "FACE_LIP_B":            14,
    "FACE_LIP_BOT":          17,
    "FACE_CHIN":            152,
    "FACE_JAW":             175,
    "FACE_FOREHEAD":         10,
    "FACE_BROW":            151,
    # ── Mouth ─────────────────────────────────────────────────────────────
    "FACE_MOUTH_CORNER_R":   61,   "FACE_MOUTH_CORNER_L":  291,
    "FACE_MOUTH_TOP_R":      37,   "FACE_MOUTH_TOP_L":     267,
    "FACE_MOUTH_BOT_R":      84,   "FACE_MOUTH_BOT_L":     314,
    # ── Jaw / chin sides ──────────────────────────────────────────────────
    "FACE_CHIN_SIDE_R":     172,   "FACE_CHIN_SIDE_L":     397,
    "FACE_JAW_SIDE_R":      234,   "FACE_JAW_SIDE_L":      454,
    # ── Eyes ──────────────────────────────────────────────────────────────
    "FACE_EYE_CENTER_R":    468,   "FACE_EYE_CENTER_L":    473,
    "FACE_EYE_OUTER_R":      33,   "FACE_EYE_OUTER_L":     263,
    "FACE_EYE_INNER_R":     133,   "FACE_EYE_INNER_L":     362,
    "FACE_EYE_TOP_R":       159,   "FACE_EYE_TOP_L":       386,
    "FACE_EYE_BOT_R":       145,   "FACE_EYE_BOT_L":       374,
    # ── Brows ─────────────────────────────────────────────────────────────
    "FACE_BROW_OUTER_R":     70,   "FACE_BROW_OUTER_L":    300,
    "FACE_BROW_3_R":         46,   "FACE_BROW_3_L":        276,
    "FACE_BROW_2_R":         52,   "FACE_BROW_2_L":        282,
    "FACE_BROW_1_R":         65,   "FACE_BROW_1_L":        295,
    # Under-brow / crease (approximated with nearby landmarks)
    "FACE_CREASE_OUTER_R":  107,   "FACE_CREASE_OUTER_L":  336,
    "FACE_CREASE_INNER_R":   46,   "FACE_CREASE_INNER_L":  276,
    "FACE_BROW_BOT_OUTER_R": 33,   "FACE_BROW_BOT_OUTER_L":263,
    "FACE_LID_CREASE_T_R":  159,   "FACE_LID_CREASE_T_L":  386,
    # ── Cheeks ────────────────────────────────────────────────────────────
    "FACE_CHEEK_R":         205,   "FACE_CHEEK_L":         425,
    "FACE_CHEEK_TOP_R":     187,   "FACE_CHEEK_TOP_L":     411,
    # ── Nose sides ────────────────────────────────────────────────────────
    "FACE_NOSE_WING_R":      64,   "FACE_NOSE_WING_L":     294,
    "FACE_NOSE_BRIDGE_R":    55,   "FACE_NOSE_BRIDGE_L":   285,
    # ── Ear (jaw-angle landmark — closest visible point from front) ───────
    "FACE_EAR_R":           127,   "FACE_EAR_L":           356,
    # ── Temple ────────────────────────────────────────────────────────────
    "FACE_TEMPLE_R":        162,   "FACE_TEMPLE_L":        389,
    # ── Forehead sides ────────────────────────────────────────────────────
    "FACE_FOREHEAD_SIDE_R":   54,  "FACE_FOREHEAD_SIDE_L":  284,
    "FACE_FOREHEAD_SIDE_1_R": 67,  "FACE_FOREHEAD_SIDE_1_L":297,
    "FACE_FOREHEAD_SIDE_2_R":104,  "FACE_FOREHEAD_SIDE_2_L":333,
    "FACE_FOREHEAD_SIDE_3_R":108,  "FACE_FOREHEAD_SIDE_3_L":338,
    # FACE_TEETH_T/B and FACE_TONGUE_* are interior — not detectable from render
}


# ── Blender operators ─────────────────────────────────────────────────────────
#
# No install/uninstall operators: onnxruntime + Pillow ship as bundled wheels
# (see the "Dependency management" section above) — Blender installs them
# automatically when the addon is enabled, nothing for the user to trigger.


# ── Scale normalization ───────────────────────────────────────────────────────
# The detection pipeline (especially the finger/wrist cleanup in ai_detect_lvt)
# carries thresholds hardcoded in absolute metres — forearm-clamp length, finger
# cross-section radii, tip-merge distances — all tuned for a ~1.75 m human. A
# character modelled far larger or smaller makes those thresholds mismatch and the
# detect fails (users had to manually scale the mesh to ~human height first).
#
# Rather than convert dozens of tuned constants, we AUTOMATE that manual step:
# temporarily rescale the mesh AND every existing marker uniformly about the world
# origin to a canonical height, run the whole detect in that frame, then invert the
# transform on the mesh and on every marker (pre-existing + newly placed) so results
# land exactly on the original, unscaled mesh. Uniform scaling preserves proportions,
# so the detector sees the same character — just at the scale its constants expect.
_DETECT_TARGET_H = 1.75    # canonical working height (m)
_DETECT_SCALE_LO = 1.40    # heights within [LO, HI] are already fine — no rescale
_DETECT_SCALE_HI = 2.00

from contextlib import contextmanager


def _detect_markers():
    return [o for o in bpy.data.objects
            if o.type == 'EMPTY' and o.name.startswith("MARKER_")]


def _detect_refresh():
    """Push transform changes to the depsgraph so BVH builds and renders see them."""
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass


@contextmanager
def _normalized_detect_scale(mesh_obj, scale=None):
    """Temporarily normalize mesh + markers to ~human height about the world origin.
    Yields the scale factor applied (1.0 = untouched). Inverts everything on exit.
    scale: explicit factor to apply instead of the height-based one — used by
    finger detection, which anchors on FOREARM length (a baby normalized to
    adult HEIGHT still has a proportionally tiny hand)."""
    if scale is not None:
        if abs(scale - 1.0) < 1e-3:
            yield 1.0
            return
        s = float(scale)
        h = 0.0
    else:
        try:
            mn, mx, _ = _mesh_bbox_world(mesh_obj)
            h = float(mx.z - mn.z)
        except Exception:
            h = 0.0
        if h <= 1e-6 or (_DETECT_SCALE_LO <= h <= _DETECT_SCALE_HI):
            yield 1.0
            return
        s = _DETECT_TARGET_H / h
    inv = 1.0 / s
    S   = Matrix.Diagonal((s, s, s, 1.0))       # uniform scale about world origin
    orig_mw = mesh_obj.matrix_world.copy()
    if scale is not None:
        dbg(f"[scale] forearm-anchored normalization x{s:.4f} for detection")
    else:
        dbg(f"[scale] character height {h:.2f}m outside [{_DETECT_SCALE_LO:.1f}, "
              f"{_DETECT_SCALE_HI:.1f}]m — normalizing x{s:.4f} for detection")

    mesh_obj.matrix_world = S @ orig_mw
    for _o in _detect_markers():                # scale input markers (HAND/ELBOW/...) too
        _o.location = _o.location * s
    _detect_refresh()
    try:
        yield s
    finally:
        # Restore the mesh exactly; map every marker (old + new) back to true scale.
        mesh_obj.matrix_world = orig_mw
        for _o in _detect_markers():
            _o.location = _o.location * inv
        _detect_refresh()


# Marker auto-scale anchors: (character height in m, marker scale multiplier).
# Linear interpolation between anchors, linear extrapolation beyond the ends.
_MSCALE_ANCHORS = ((0.10, 0.2), (0.35, 0.55), (1.75, 1.5), (3.50, 2.5))


def _auto_marker_scale(context, mesh_obj):
    """Set the scene marker scale from the character's height and rescale any
    already-placed markers, so AI-detected markers fit tiny/chibi/adult/giant
    characters alike."""
    from .markers import _rescale_all_markers
    try:
        mn, mx, _ = _mesh_bbox_world(mesh_obj)
        h = float(mx.z - mn.z)
    except Exception:
        return
    if h <= 1e-6:
        return
    pts = _MSCALE_ANCHORS
    if h <= pts[0][0]:
        (h0, s0), (h1, s1) = pts[0], pts[1]
    elif h >= pts[-1][0]:
        (h0, s0), (h1, s1) = pts[-2], pts[-1]
    else:
        for i in range(len(pts) - 1):
            if pts[i][0] <= h <= pts[i + 1][0]:
                (h0, s0), (h1, s1) = pts[i], pts[i + 1]
                break
    scale = s0 + (s1 - s0) * (h - h0) / (h1 - h0)
    scale = round(min(max(scale, 0.05), 10.0), 3)
    context.scene.autorig_marker_scale = scale
    _rescale_all_markers(scale)


def _clip_rear_protrusions(mesh_obj):
    """
    Detect a large REAR protrusion (tail, rear wings/props — the character
    faces -Y, so the back is +Y) and return a temporary mesh object with
    everything behind the body clipped off, or None when the body has no
    significant rear overhang (the normal-character fast path: nothing is
    copied or modified).

    Why: the body detector frames its ortho renders on the mesh bbox and
    snaps markers to the mesh via BVH. A long tail inflates the side/top
    framing (the body shrinks to a corner of the crop) and gives the spine
    markers tail surface to snap onto — spine/chest/neck end up ON the tail.
    Detecting on a body-only copy fixes framing, model input, BVH snapping,
    and the geometric estimates all at once. Markers are placed in world
    space, so no back-transform is needed.

    Back-plane estimate: slice the character by height; each slice's max-Y is
    where the BODY ends at that height for most slices — a tail is thin, so
    it dominates only a few slices. The 75th percentile of slice max-Y is a
    robust "back of body" even with legs bent backward. Feet are protected
    explicitly: the clip plane never cuts behind the rear-most heel vertex
    (bottom 15% of the height), so heel/toe markers keep their geometry.

    Caller must remove the returned object + its mesh datablock.
    """
    import bmesh as _bm
    from mathutils import Matrix

    try:
        dg = bpy.context.evaluated_depsgraph_get()
        ev = mesh_obj.evaluated_get(dg)
        mw = ev.matrix_world
        verts = [mw @ v.co for v in ev.data.vertices]
    except Exception as e:
        dbg(f"[body-clip] mesh read failed ({e}) — skipping rear clip")
        return None
    if len(verts) < 100:
        return None

    zs = [v.z for v in verts]
    z_lo, z_hi = min(zs), max(zs)
    height = z_hi - z_lo
    if height < 1e-4:
        return None

    n_slices = 40
    slice_ymax = [None] * n_slices
    for v in verts:
        si = min(n_slices - 1, int((v.z - z_lo) / height * n_slices))
        if slice_ymax[si] is None or v.y > slice_ymax[si]:
            slice_ymax[si] = v.y

    filled = sorted(y for y in slice_ymax if y is not None)
    if len(filled) < 8:
        return None
    body_back = filled[int(len(filled) * 0.75)]          # P75 of slice max-Y
    mx_y = filled[-1]

    overhang = mx_y - body_back
    if overhang <= 0.12 * height:
        return None                                       # normal body — no clip

    # Clip plane: a margin behind the body back, but NEVER in front of the
    # rear-most foot/heel vertex (bottom 15% of the height) — feet stick out
    # backward legitimately and their markers need that geometry.
    heel_ymax = max((v.y for v in verts if v.z < z_lo + 0.15 * height),
                    default=body_back)
    clip_y = max(body_back + 0.04 * height, heel_ymax + 0.02 * height)
    if mx_y - clip_y <= 0.08 * height:
        return None            # the "protrusion" was mostly feet — leave alone
    dbg(f"[body-clip] rear protrusion detected: overhang {overhang:.2f}m "
          f"({overhang / height:.2f}x height) — clipping behind y={clip_y:.3f} "
          f"and detecting on the body-only copy")

    clip_data = bpy.data.meshes.new_from_object(ev, depsgraph=dg)
    clip_obj = bpy.data.objects.new("_er_body_clipped", clip_data)
    bpy.context.collection.objects.link(clip_obj)
    clip_obj.matrix_world = Matrix.Identity(4)
    clip_obj.color = mesh_obj.color

    bm = _bm.new()
    bm.from_mesh(clip_data)
    bm.transform(mw)   # bake world transform (same pattern as _isolate_hand_mesh)
    res = _bm.ops.bisect_plane(
        bm,
        geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
        dist=0.0001,
        plane_co=Vector((0.0, clip_y, 0.0)),
        plane_no=Vector((0.0, 1.0, 0.0)),
        clear_outer=True,        # +Y side (behind the body) removed
        clear_inner=False,
    )
    # Cap the cut so the stump is watertight (open loops read as holes in the
    # ortho silhouettes and give BVH rays an interior to slip into).
    try:
        cut_edges = [e for e in res["geom_cut"] if isinstance(e, _bm.types.BMEdge)]
        if cut_edges:
            _bm.ops.holes_fill(bm, edges=cut_edges, sides=0)
    except Exception as e:
        dbg(f"[body-clip] stump cap skipped ({e})")

    bm.to_mesh(clip_data)
    bm.free()

    if len(clip_data.vertices) < 100:
        # Clip ate the mesh (bad estimate) — fail closed to the full mesh.
        dbg("[body-clip] clip left too little geometry — using the full mesh")
        bpy.data.objects.remove(clip_obj, do_unlink=True)
        bpy.data.meshes.remove(clip_data, do_unlink=True)
        return None
    return clip_obj


class AUTORIG_OT_AIDetectBody(bpy.types.Operator):
    """EasyDetect: place body markers automatically.
Primary: bundled body pose model (models/body_pose.rmodel).
Fallback: geometric mesh estimate."""
    bl_idname  = "autorig.ai_detect_body"
    bl_label   = "EasyDetect Body"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if not is_body_onnx_available():
            # The geometric estimate path handles everything the ONNX model
            # would (that IS the runtime fallback when inference fails), so a
            # missing model/onnxruntime shouldn't hard-block body detection —
            # warn and run geometry-only.
            self.report(
                {'WARNING'},
                "Body pose model/onnxruntime unavailable — using the geometric "
                "estimate only (install AI dependencies for best results).",
            )

        # ── Resolve body mesh ─────────────────────────────────────────────────
        props    = getattr(context.scene, "autorig_face_objs", None)
        mesh_obj = _scene_mesh_or_none(context, props.detect_body_obj if props else None)
        if mesh_obj is None:
            meshes = [
                o for o in context.scene.objects
                if o.type == 'MESH' and not o.get("autorig_marker")
            ]
            if not meshes:
                self.report({'ERROR'}, "No mesh found. Set a Body Mesh in the panel.")
                return {'CANCELLED'}
            mesh_obj = max(meshes, key=lambda o: o.dimensions.z)

        _auto_marker_scale(context, mesh_obj)

        with _normalized_detect_scale(mesh_obj):
            # Characters with a tail / large rear props: detect on a temp copy
            # with the rear protrusion clipped off, so the framing, the model
            # input and the marker snapping only ever see the body. Fails
            # closed: None for normal characters (zero behaviour change).
            _clip_obj = _clip_rear_protrusions(mesh_obj)
            if _clip_obj is not None:
                _clip_data = _clip_obj.data
                try:
                    return self._detect_body(context, _clip_obj)
                finally:
                    try:
                        bpy.data.objects.remove(_clip_obj, do_unlink=True)
                        bpy.data.meshes.remove(_clip_data, do_unlink=True)
                    except Exception:
                        pass
            return self._detect_body(context, mesh_obj)

    def _detect_body(self, context, mesh_obj):
        shared_bvh  = _build_bvh(mesh_obj)
        marker_pos  = None
        body_bounds = None
        undet_names = set()   # joints the ONNX under-detected (filled by graft)

        # ── 1. ONNX body detector (skipped cleanly when unavailable) ─────────
        _onnx_ok = False
        if is_body_onnx_available():
            try:
                _mn_w, _mx_w, _ = _mesh_bbox_world(mesh_obj)
                _mesh_w = _mx_w.x - _mn_w.x
                _rf = _RENDER_TARGET_FACES
                onnx_pos = None
                for _attempt in range(2):
                    fp, sp, tp, body_bounds, body_td = _render_body_views(
                        mesh_obj, render_faces=_rf)
                    try:
                        onnx_pos, undet_names = _detect_body_onnx(
                            fp, sp, tp, body_bounds, shared_bvh)
                    finally:
                        shutil.rmtree(body_td, ignore_errors=True)
                    # Collapse guard: a healthy detection spreads its markers across
                    # most of the character's width (arms-out reach the sides; even
                    # arms-down hips/shoulders span the torso). A marker X-spread that
                    # is tiny next to the mesh width means the ortho render was
                    # degenerate — over-decimation rendered the body black, so every
                    # arm marker triangulated to the centre line. Re-render ONCE at a
                    # higher face target (the surface survives) and re-detect. The
                    # threshold self-scales with the mesh width, so narrow arms-down
                    # characters (small width, proportionally small spread) don't trip.
                    if (onnx_pos and _attempt == 0 and _mesh_w > 1e-4):
                        _xs = [p.x for p in onnx_pos.values()]
                        # Healthy detections span 0.69–0.87 of the mesh width across
                        # every tested body (arms-out and arms-down alike); a black-body
                        # collapse measures ~0.37. 0.55 sits in that gap with margin.
                        if (max(_xs) - min(_xs)) < 0.55 * _mesh_w:
                            dbg(f"[ONNX-v2] detection collapsed (marker X-spread "
                                  f"{(max(_xs) - min(_xs)):.2f}m vs mesh width "
                                  f"{_mesh_w:.2f}m) — re-rendering at "
                                  f"{_RENDER_RETRY_FACES} faces and retrying")
                            _rf = _RENDER_RETRY_FACES
                            continue
                    break
                if onnx_pos:
                    onnx_pos = _fix_body_anatomy(mesh_obj, onnx_pos, bvh=shared_bvh, bounds=body_bounds)
                    marker_pos = onnx_pos
                    _onnx_ok = True
            except Exception as e:
                self.report({'WARNING'}, f"ONNX body detection failed ({e}) — using geometry estimate.")

        # ── 2. Fill missing landmarks from geometric estimate ─────────────────
        if body_bounds is None:
            mn, mx, cen = _mesh_bbox_world(mesh_obj)
            body_bounds = {
                "cen_x": cen.x, "cen_y": cen.y, "cen_z": cen.z,
                "ortho_scale": max(mx.z - mn.z, mx.x - mn.x) * 1.15,
                "mn_y": mn.y, "mx_y": mx.y,
            }
        missing = [n for n in _BODY_MARKERS if n not in (marker_pos or {})]
        if missing:
            kp3d     = _estimate_body_from_mesh(mesh_obj)
            geom_pos = _build_marker_positions(kp3d, mesh_cen_x=body_bounds["cen_x"])
            if marker_pos is None:
                marker_pos = geom_pos
                marker_pos = _snap_heels_back(mesh_obj, marker_pos, body_bounds, bvh=shared_bvh)
            else:
                for name in missing:
                    # A missing HAND grafted from the bbox estimate lands at shoulder
                    # level (the estimate assumes arms-out). Derive it from the arm
                    # chain instead: extend ELBOW along SHOULDER->ELBOW by ~one forearm
                    # (≈ upper-arm length), so it lands on the arm for any pose.
                    if name in ("HAND_L", "HAND_R"):
                        _sd = name[-1]
                        _el = marker_pos.get(f"ELBOW_{_sd}")
                        _sh = marker_pos.get(f"SHOULDER_{_sd}")
                        if _el is not None and _sh is not None and (_el - _sh).length > 1e-4:
                            _ua = _el - _sh
                            marker_pos[name] = _el + _ua.normalized() * (_ua.length * 0.95)
                            continue
                    if name in geom_pos:
                        marker_pos[name] = geom_pos[name]

        # ── 2a. Geometric LEG/FOOT placement (standing body) ─────────────────────
        # The 3-ortho model can't resolve feet (top-view occlusion). For a standing
        # body the lower leg is simple geometry: isolate each leg by proximity to its
        # reliable hip (THIGH), then read knee/ankle/foot/heel/toes from VERTEX-SUBSET
        # CENTROIDS. A centroid of mesh verts is ALWAYS inside the volume — unlike a
        # ray-cast it cannot fly off to another limb (the bug in earlier attempts).
        # Toes = ball of foot (~25% back from the tip). Body faces -Y (front render).
        if marker_pos:
            try:
                _dg = bpy.context.evaluated_depsgraph_get()
                _ev = mesh_obj.evaluated_get(_dg)
                _mw = _ev.matrix_world
                _vw = [_mw @ v.co for v in _ev.data.vertices]
            except Exception as _e:
                _vw = None
                dbg(f"[legs] mesh read skipped ({_e})")
            if _vw and len(_vw) >= 20:
                _gz = min(v.z for v in _vw); _hh = max(v.z for v in _vw) - _gz

                def _ctr(_vs):
                    _n = len(_vs)
                    return Vector((sum(v.x for v in _vs) / _n, sum(v.y for v in _vs) / _n,
                                   sum(v.z for v in _vs) / _n))

                _lcx = body_bounds.get("cen_x", sum(v.x for v in _vw) / len(_vw))
                _legs_centroided = 0
                if _hh > 1e-3:
                    for _ls in ("L", "R"):
                        # The centroid re-derivation predates the retrained
                        # model (its leg channels were DEAD, this was life
                        # support). With legs now at ~1.7px it is the FALLBACK:
                        # keep the model's leg chain whenever it detected this
                        # side; centroids only for model-missed legs and for
                        # the no-onnx geometric path (undet covers both).
                        if _onnx_ok and not any(
                                f"{_b}_{_ls}" in undet_names
                                for _b in ("SHIN", "FOOT", "HEEL", "TOES")):
                            continue   # model leg chain kept; centroids are the fallback
                        _hip = marker_pos.get(f"THIGH_{_ls}")
                        if _hip is None:
                            continue
                        # Isolate the leg by BODY SIDE (not hip-X proximity) so a foot
                        # that is splayed/wide is still captured wherever it sits.
                        _side = [v for v in _vw if (v.x > _lcx if _ls == "L" else v.x < _lcx)]
                        _foot = [v for v in _side if v.z < _gz + 0.12 * _hh]
                        if len(_foot) < 4:
                            continue
                        _fx  = sorted(v.x for v in _foot)[len(_foot) // 2]   # the foot's actual X
                        _toy = min(v.y for v in _foot); _hey = max(v.y for v in _foot)
                        _flen = max(_hey - _toy, 1e-5)
                        # Ankle = lower-leg cross-section centroid near the FOOT's X
                        # (handles splay — the leg is not directly under the hip).
                        _ab = [v for v in _side if abs(v.z - (_gz + 0.07 * _hh)) < 0.04 * _hh
                               and abs(v.x - _fx) < 0.13 * _hh]
                        _ankle = _ctr(_ab) if len(_ab) >= 3 else (_ctr(_foot) + Vector((0, 0, 0.05 * _hh)))
                        marker_pos[f"FOOT_{_ls}"] = _ankle
                        # Toes = ball of foot (~25% back from the tip): foot centroid there.
                        _tb = [v for v in _foot if abs(v.y - (_toy + 0.25 * _flen)) < 0.15 * _flen]
                        marker_pos[f"TOES_{_ls}"] = _ctr(_tb) if len(_tb) >= 3 else _ctr(_foot)
                        # Heel = BACK END of the foot, on the floor (rigify heel-roll pivot).
                        _hb = [v for v in _foot if v.y > _hey - 0.18 * _flen]
                        if len(_hb) >= 3:
                            _hcx = sum(v.x for v in _hb) / len(_hb)
                            marker_pos[f"HEEL_{_ls}"] = Vector((_hcx, _hey - 0.03 * _flen,
                                                                min(v.z for v in _hb)))
                        else:
                            marker_pos[f"HEEL_{_ls}"] = _ctr(_foot)
                        # Knee = mid-leg cross-section centroid; X INTERPOLATED along the
                        # hip->ankle line so an angled (splayed) leg is followed, not
                        # assumed vertical under the hip.
                        _kz = (_hip.z + _ankle.z) * 0.5
                        _t  = ((_kz - _hip.z) / (_ankle.z - _hip.z)) if abs(_ankle.z - _hip.z) > 1e-4 else 0.5
                        _kx = _hip.x + _t * (_ankle.x - _hip.x)
                        _kb = [v for v in _side if abs(v.z - _kz) < 0.04 * _hh
                               and abs(v.x - _kx) < 0.14 * _hh]
                        if len(_kb) >= 3:
                            marker_pos[f"SHIN_{_ls}"] = _ctr(_kb)
                        _legs_centroided += 1
                    if _legs_centroided:
                        dbg(f"[legs] knee/ankle/foot/heel/toes from leg-mesh "
                              f"centroids ({_legs_centroided} side(s))")

                    # ── WRIST placement ───────────────────────────────────────────
                    # Two cases:
                    #  - Well-detected hand: just CENTRE it in the forearm (thin slab in
                    #    the wrist plane perpendicular to ELBOW->HAND -> centroid inside).
                    #  - UNDER-detected hand (e.g. seen only in the side view): the graft
                    #    placed it via a straight-arm guess that's wrong on a bent arm.
                    #    FIND it from the forearm mesh instead: a sphere-shell of arm
                    #    verts ~forearm-length from the reliable ELBOW (pose-robust).
                    def _mesh_wrist(_side):
                        _elb = marker_pos.get(f"ELBOW_{_side}")
                        _sho = marker_pos.get(f"SHOULDER_{_side}")
                        if _elb is None or _sho is None:
                            return None
                        _ua = _elb - _sho; _ul = _ua.length
                        if _ul < 1e-4:
                            return None
                        _ud = _ua / _ul
                        _cand = []
                        for v in _vw:
                            _rel = v - _elb
                            _al = _rel.dot(_ud)
                            if _al <= 0:                      # behind elbow (upper-arm side)
                                continue
                            if (_rel - _ud * _al).length > 0.5 * _ul:   # off the arm (torso)
                                continue
                            _d = _rel.length
                            if _d > 1.5 * _ul:                # beyond the hand
                                continue
                            _cand.append((v, _d))
                        if len(_cand) < 4:
                            return None
                        _dmax = max(_d for _, _d in _cand)
                        _shell = [v for v, _d in _cand if abs(_d - 0.82 * _dmax) < 0.15 * _dmax]
                        return _ctr(_shell) if len(_shell) >= 3 else None

                    def _wrist_narrow(_side):
                        """Wrist = the NARROWEST cross-section before the palm.

                        REACH-BASED, not anchored to the HAND marker's position: the
                        HAND marker often sits up the forearm, so scanning relative to it
                        (old code) just found the forearm taper and left the wrist high.
                        Instead, use only the ELBOW->HAND DIRECTION, measure the actual
                        fingertip REACH from the mesh (farthest arm vertex along it), and
                        take the narrowest cross-section in the forearm->palm span
                        (0.50-0.82 of reach). The forearm tapers to the wrist, the palm
                        widens after -> the minimum there is the wrist."""
                        _elb = marker_pos.get(f"ELBOW_{_side}")
                        _hnd = marker_pos.get(f"HAND_{_side}")
                        if _elb is None or _hnd is None:
                            return None
                        _dir = _hnd - _elb; _dl = _dir.length
                        if _dl < 1e-4:
                            return None
                        _dir = _dir / _dl
                        # arm verts forward of the elbow, near the arm axis (drop torso)
                        _arm = []
                        for v in _vw:
                            _rel = v - _elb
                            _al  = _rel.dot(_dir)
                            if _al <= 0:
                                continue
                            if (_rel - _dir * _al).length > 0.6 * _dl:
                                continue
                            _arm.append((v, _al))
                        if len(_arm) < 8:
                            return None
                        _reach = max(_al for _, _al in _arm)   # elbow->fingertip
                        if _reach < 1e-4:
                            return None
                        # Radius profile along the whole arm region (forearm -> fingers).
                        _N = 22
                        _prof = []
                        for _i in range(_N):
                            _frac = 0.30 + _i * (1.00 - 0.30) / (_N - 1)
                            _tt   = _frac * _reach
                            _slab = [v for v, _al in _arm if abs(_al - _tt) < 0.045 * _reach]
                            if len(_slab) < 4:
                                _prof.append(None)
                                continue
                            _c0 = _ctr(_slab)
                            _r  = sum((v - _c0 - _dir * ((v - _c0).dot(_dir))).length
                                      for v in _slab) / len(_slab)
                            _prof.append((_r, _c0, _frac))
                        _vp = [(_i, p) for _i, p in enumerate(_prof) if p is not None]
                        if not _vp:
                            return None
                        # PALM = widest cross-section in the hand region (frac 0.55-0.95):
                        # a clear, pose/length-robust anchor (the knuckle/palm bulge).
                        _palm = [(_i, p) for _i, p in _vp if 0.55 <= p[2] <= 0.95]
                        if not _palm:
                            return min(_vp, key=lambda ip: ip[1][0])[1][1]
                        _palm_i = max(_palm, key=lambda ip: ip[1][0])[0]
                        # WRIST = narrowest cross-section BETWEEN the forearm and the palm
                        # (search up to the palm index, never into the fingers past it).
                        _pre = [p for _i, p in _vp if _i < _palm_i and p[2] >= 0.40]
                        if not _pre:
                            _pre = [p for _i, p in _vp if _i <= _palm_i]
                        return min(_pre, key=lambda p: p[0])[1] if _pre else None

                    _wmoved = 0
                    for _ws in ("L", "R"):
                        if marker_pos.get(f"ELBOW_{_ws}") is None:
                            continue
                        # Under-detected hand: forearm-shell finder (then symmetry mirrors
                        # the reliable side onto it). Detected hand: narrowest-slice wrist.
                        _w = _mesh_wrist(_ws) if f"HAND_{_ws}" in undet_names else _wrist_narrow(_ws)
                        if _w is not None:
                            marker_pos[f"HAND_{_ws}"] = _w
                            _wmoved += 1
                    if _wmoved:
                        dbg(f"[wrist] {_wmoved} HAND marker(s) placed at the wrist (narrowest cross-section)")

                    # ── Spine depth-centring ──────────────────────────────────────
                    # Put each midline marker at the TORSO's Y-centre at its own
                    # height: X -> body centre, Y -> geometric midpoint of the central
                    # slab's front-back extent there ((min+max)/2, NOT the vertex mean
                    # -- a mean is density-weighted and a flat-back / rounded-front
                    # torso would pull it toward the back). This follows the body's
                    # natural curve, keeps markers off the skin, and is truly central in
                    # depth. Height (Z) is kept from the model.
                    _cx = body_bounds.get("cen_x", sum(v.x for v in _vw) / len(_vw))
                    _smoved = 0
                    for _sm in ("PELVIS", "SPINE_001", "SPINE_002", "CHEST", "NECK", "HEAD"):
                        _p = marker_pos.get(_sm)
                        if _p is None:
                            continue
                        _band = [v for v in _vw if abs(v.z - _p.z) < 0.04 * _hh
                                 and abs(v.x - _cx) < 0.13 * _hh]
                        if len(_band) >= 5:
                            _ys = [v.y for v in _band]
                            _ymid = (min(_ys) + max(_ys)) * 0.5
                            marker_pos[_sm] = Vector((_cx, _ymid, _p.z))
                            _smoved += 1
                    if _smoved:
                        dbg(f"[spine] {_smoved} midline markers depth-centred in the torso")

        # ── 2b. Interior depth — FINAL pass, runs for EVERY source path ───────────
        # _fix_body_anatomy only runs when ONNX succeeds, and step 2 fills missing
        # markers from geometry *after* it. So HAND/SHIN/FOOT/ELBOW can reach here
        # still on the surface (geometry fallback, or ONNX-missed + geom-filled).
        # Re-center each inside its limb here so every path lands inside the mesh,
        # which Rigify requires for bone heads.
        if marker_pos:
            _mx_y = body_bounds.get("mx_y")
            if _mx_y is None:
                _mx_y = _mesh_bbox_world(mesh_obj)[1].y
            # Each marker is centered along the axis of its own limb, with a lateral
            # gate so geometry off that axis (a torso behind an arm hugging the body)
            # can't capture it. Arm axis = SHOULDER→HAND, leg axis = THIGH→FOOT.
            # Gate = 30% of limb length ≈ generous limb radius, far tighter than the
            # distance to the torso.
            for _ic_name, _root_name, _tipm_name in (
                ("HAND_L", "SHOULDER_L", "HAND_L"), ("HAND_R", "SHOULDER_R", "HAND_R"),
                ("ELBOW_L", "SHOULDER_L", "HAND_L"), ("ELBOW_R", "SHOULDER_R", "HAND_R"),
                ("SHIN_L", "THIGH_L", "FOOT_L"),     ("SHIN_R", "THIGH_R", "FOOT_R"),
                ("FOOT_L", "THIGH_L", "FOOT_L"),     ("FOOT_R", "THIGH_R", "FOOT_R"),
            ):
                _ic_pt = marker_pos.get(_ic_name)
                if _ic_pt is None:
                    continue
                _root = marker_pos.get(_root_name)
                _tipm = marker_pos.get(_tipm_name)
                _ax_o = _ax_d = _perp = None
                if _root is not None and _tipm is not None:
                    _seg = _tipm - _root
                    _slen = _seg.length
                    if _slen > 1e-4:
                        _ax_o = _root
                        _ax_d = _seg / _slen
                        _perp = _slen * 0.30
                # Cross-section centring (perpendicular to the limb axis) puts the
                # marker inside in EVERY direction — fixes a wrist on TOP (Z) or a
                # knee/elbow off to the SIDE (X), which the Y-only centre can't. Accept
                # it only if it lands within the limb-radius gate of the axis (so it
                # can't jump to the torso); otherwise fall back to the Y-only centre.
                # Ray REACH is longer than the gate (0.75 vs 0.30 of limb length): on
                # massive limbs (Hulk forearms) a marker ON the surface is a full limb
                # DIAMETER from the far wall, and gate-length rays never found it — the
                # centring silently failed and the wrist/elbow stayed off-centre. And
                # ITERATE: one pass under-moves a surface point (same lesson as the
                # finger depth pass); re-centring from the moved point converges.
                _cc = None
                if _ax_d is not None:
                    _reach = _slen * 0.75
                    _cur = _ic_pt
                    for _it in range(3):
                        _nxt = _center_in_limb_cross(shared_bvh, _cur, _ax_d, _reach)
                        if _nxt is None:
                            break
                        _cc, _cur = _nxt, _nxt
                    if _cc is not None:
                        _rel  = _cc - _ax_o
                        _perp_d = (_rel - _ax_d * _rel.dot(_ax_d)).length
                        if _perp_d > _perp:
                            _cc = None          # drifted off-limb → reject
                if _cc is not None:
                    marker_pos[_ic_name] = _cc
                else:
                    marker_pos[_ic_name] = _center_in_limb(
                        shared_bvh, _ic_pt, _mx_y,
                        axis_origin=_ax_o, axis_dir=_ax_d, max_perp=_perp)

        # ── 2c. Enforce L/R symmetry (Rigify requirement) ─────────────────────────
        # Rigify needs symmetric marker placement. For each L/R pair: if ONE side was
        # under-detected (in undet_names) and the other is reliable, MIRROR the reliable
        # side across the body centre onto the weak side (this is what fixes the left
        # wrist — mirror the well-detected right). If both are reliable (or both weak),
        # average their offset/Y/Z and place them mirror-symmetric. Midline markers are
        # pinned to the centre X. Valid because rest-pose characters are symmetric.
        if marker_pos and body_bounds is not None:
            # Centerline X = the body's true mirror plane. Derive it from the
            # bilateral pairs (midpoint of L/R shoulders, hips, etc.) rather than
            # the mesh bbox center, which an A-pose arm / tail / prop / asymmetric
            # hair can skew. Median of pair-midpoints is robust to one bad pair.
            # Fall back to bbox center only when no pair is available.
            _mids = []
            for _b in ("SHOULDER", "THIGH", "HAND", "FOOT", "ELBOW", "SHIN", "HEEL", "TOES"):
                _L = marker_pos.get(f"{_b}_L"); _R = marker_pos.get(f"{_b}_R")
                if _L is not None and _R is not None:
                    _mids.append((_L.x + _R.x) * 0.5)
            if _mids:
                _mids.sort()
                _scx = _mids[len(_mids) // 2]
            else:
                _scx = body_bounds.get("cen_x")
                if _scx is None:
                    _scx = _mesh_bbox_world(mesh_obj)[2].x
            for _mid in ("PELVIS", "SPINE_001", "SPINE_002", "CHEST", "NECK", "HEAD"):
                _mp = marker_pos.get(_mid)
                if _mp is not None:
                    marker_pos[_mid] = Vector((_scx, _mp.y, _mp.z))
            _sym_on = getattr(context.scene, "autorig_detect_symmetry", True)
            for _b in ("SHOULDER", "ARM", "ELBOW", "HAND",
                       "THIGH", "SHIN", "FOOT", "HEEL", "TOES"):
                _L = marker_pos.get(f"{_b}_L"); _R = marker_pos.get(f"{_b}_R")
                if _L is None or _R is None:
                    continue
                _lu = f"{_b}_L" in undet_names
                _ru = f"{_b}_R" in undet_names
                # Mirror-FILL of an under-detected side stays even with the
                # Symmetrical Detect toggle OFF (it is detection completion,
                # not cosmetic symmetry — without it that side is garbage).
                if _lu and not _ru:                       # mirror reliable R -> weak L
                    marker_pos[f"{_b}_L"] = Vector((2 * _scx - _R.x, _R.y, _R.z))
                elif _ru and not _lu:                     # mirror reliable L -> weak R
                    marker_pos[f"{_b}_R"] = Vector((2 * _scx - _L.x, _L.y, _L.z))
                elif _sym_on:                             # both same status -> average
                    _off = (abs(_L.x - _scx) + abs(_R.x - _scx)) * 0.5
                    _ay  = (_L.y + _R.y) * 0.5
                    _az  = (_L.z + _R.z) * 0.5
                    _ls  = 1.0 if _L.x >= _scx else -1.0
                    marker_pos[f"{_b}_L"] = Vector((_scx + _ls * _off, _ay, _az))
                    marker_pos[f"{_b}_R"] = Vector((_scx - _ls * _off, _ay, _az))
            dbg("[symmetry] midline pinned to centre; L/R pairs "
                  + ("mirrored/averaged" if _sym_on
                     else "kept raw (Symmetrical Detect OFF; weak sides still mirror-filled)"))

        # ── 2d. THIGH ↔ PELVIS level (Rigify requirement) ─────────────────────
        # The Rigify metarig expects the thigh heads LEVEL with the pelvis head:
        # same height (Z) and depth (Y), only X differs. _fix_body_anatomy step
        # 5b anchors this, but LATER passes break it (the torso depth-centring
        # moves PELVIS.y after 5b, and the geometric fallback path never anchors
        # at all) — so re-assert it here as the last word before placement.
        if marker_pos:
            _pl_pelv = marker_pos.get("PELVIS")
            if _pl_pelv is not None:
                for _pl_s in ("L", "R"):
                    _pl_th = marker_pos.get(f"THIGH_{_pl_s}")
                    if _pl_th is not None:
                        marker_pos[f"THIGH_{_pl_s}"] = Vector(
                            (_pl_th.x, _pl_pelv.y, _pl_pelv.z))

        # ── 3. Place / move markers ───────────────────────────────────────────
        col      = get_or_create_collection("RigifyMarkers")
        col.hide_viewport = False
        dtype_map = {name: dtype for name, dtype, _ in ALL_MARKERS}

        moved = 0
        for marker_name, world_pos in marker_pos.items():
            obj_name = f"MARKER_{marker_name}"
            existing  = bpy.data.objects.get(obj_name)
            if existing is None:
                dtype = dtype_map.get(marker_name, "SPHERE")
                make_empty(col, obj_name, dtype, world_pos, BODY_SIZE, context)
            else:
                existing.location = world_pos
            moved += 1

        # ── 4. Auto-estimate HAND_TIP markers from mesh geometry ─────────────
        # Places a visible sphere at the estimated middle fingertip so the user
        # can drag it to the correct position before running finger detection.
        for arp_side in ('L', 'R'):
            hw_obj = bpy.data.objects.get(f"MARKER_HAND_{arp_side}")
            ew_obj = bpy.data.objects.get(f"MARKER_ELBOW_{arp_side}")
            if hw_obj is None:
                continue
            hw_v = hw_obj.location.copy()
            ew_v = ew_obj.location.copy() if ew_obj else hw_v - Vector((0.1, 0.0, 0.0))
            arm_vec  = hw_v - ew_v
            _arm_len = arm_vec.length if arm_vec.length > 0.001 else 0.28
            fwd      = arm_vec / _arm_len if _arm_len > 0.001 else Vector((1.0, 0.0, 0.0))
            # Geodesic estimate first: farthest point from the wrist ALONG THE
            # SURFACE lands on the true middle fingertip even on curled hands
            # (knuckles reach farther FORWARD than curled-back tips) and spread
            # hands (longest finger off the forearm axis) — the two cases where
            # the straight-line projection below parks the marker on the finger.
            tip_pos = None
            try:
                from .ai_detect_geo import estimate_hand_tip_geodesic
                tip_pos = estimate_hand_tip_geodesic(
                    mesh_obj, hw_v, ew_v if ew_obj else None, arp_side)
            except Exception:
                tip_pos = None
            if tip_pos is None:
                tip_pos = _estimate_hand_tip(mesh_obj, hw_v, fwd,
                                             radius=min(0.45, _arm_len * 1.10),
                                             fallback_dist=_arm_len * 0.65,
                                             max_perp=_arm_len * 0.6)
            tip_name = f"MARKER_HAND_TIP_{arp_side}"
            existing = bpy.data.objects.get(tip_name)
            if existing is None:
                make_empty(col, tip_name, "SPHERE", tip_pos, BODY_SIZE, context)
            else:
                existing.location = tip_pos
            moved += 1

        # Arm landmarks the model missed were graft/mirror-placed — those are
        # the ones worth a visual check (legs are re-derived from leg-mesh
        # centroids regardless, so flagging them would be constant noise).
        _check = sorted(n for n in undet_names
                        if n.startswith(("HAND", "ELBOW", "SHOULDER")))
        if _check:
            self.report({'WARNING'},
                        f"EasyDetect placed {moved} body markers — check arm markers "
                        f"(model missed: {', '.join(_check)}). Drag each "
                        f"HAND_TIP onto the middle fingertip before EasyDetect Fingers.")
        else:
            self.report({'INFO'},
                        f"EasyDetect placed {moved} body markers. Tip: drag each HAND_TIP onto the "
                        "middle fingertip before running EasyDetect Fingers.")
        return {'FINISHED'}


class AUTORIG_OT_AIDetectFingers(bpy.types.Operator):
    """EasyDetect: place finger markers automatically (hand landmark / tip heatmap models).
Requires HAND_L/R and ELBOW_L/R markers to be placed first.
Runs the same hybrid pipeline as EasyDetect Body but for fingers only."""
    bl_idname  = "autorig.ai_detect_fingers"
    bl_label   = "EasyDetect Fingers"
    bl_options = {'REGISTER', 'UNDO'}

    def _snap_wrists(self, context, mesh_obj):
        """Snap each HAND (wrist) marker onto the mesh wrist, at TRUE scale, before
        the forearm scale is measured. Uses the reliable ELBOW->HAND_TIP axis (both
        far more stable than the HAND marker). Records (side, mm) in self._snapped
        for the UI report. Only moves a marker that is clearly off (> 4% of the
        elbow->tip span), so a good marker is left alone. Skipped entirely when the
        user turns off Auto-snap Wrist (small/stylized hands where they place the
        wrist by hand and don't want it overridden).

        The sanity-checked estimate is ALWAYS recorded in self._wrist_est (even
        with auto-snap OFF, where the marker itself is left alone) so that pure
        MEASUREMENTS — the forearm detect scale — stay wrist-marker independent:
        a manual wrist nudge must tune marker-anchored placement, not re-scale
        the whole detection world."""
        self._snapped   = []
        self._wrist_est = {}
        # Marker-MOVING is Geometric-only: the geodesic engine walks the mesh
        # from the wrist marker, so it needs the marker ON the wrist; the
        # Auto/template pipeline measures from the recorded estimate instead
        # (below) and must not override the user's marker placement.
        _eng  = effective_finger_engine(context)
        _auto = (_eng == 'GEOMETRIC'
                 and getattr(context.scene, 'finger_wrist_autosnap', True))
        if not _auto:
            dbg("[fingers] wrist auto-snap "
                  + ("not used by this engine (Geometric-only)"
                     if _eng != 'GEOMETRIC' else "OFF (manual wrist placement)")
                  + " -- keeping HAND markers as placed")
        for arp_side in ('L', 'R'):
            _h  = bpy.data.objects.get(f"MARKER_HAND_{arp_side}")
            _e  = bpy.data.objects.get(f"MARKER_ELBOW_{arp_side}")
            _tm = bpy.data.objects.get(f"MARKER_HAND_TIP_{arp_side}")
            if _h is None:
                continue
            if _e is None:
                dbg(f"[fingers] {arp_side}: no ELBOW marker -- wrist auto-snap skipped")
                continue
            if _tm is None:
                dbg(f"[fingers] {arp_side}: no HAND_TIP marker -- wrist auto-snap skipped")
                continue
            _tip_p = _tm.location.copy()
            _wr = _wrist_from_mesh(mesh_obj, _e.location.copy(), _tip_p)
            if _wr is None:
                dbg(f"[fingers] {arp_side}: mesh wrist not found -- keeping HAND marker")
                continue
            # Snap onto the mesh-wrist estimate essentially ALWAYS (2 mm floor, to
            # skip float noise). The ELBOW->HAND_TIP estimate is more reliable than
            # where body-detect or a hand-drag leaves the marker, and this hand is
            # so wrist-sensitive that even the ~3 mm gap between body-detect's wrist
            # and the estimate flipped the result -- with the old ~11 mm threshold
            # the first click never closed that 3 mm, so it only worked after a
            # manual move pushed the marker past the threshold. Snapping to the
            # estimate every time makes the FIRST click deterministic. (Turn off
            # Auto-snap Wrist for hands where the estimate itself is wrong.)
            _off = (_wr - _h.location).length
            # SANITY CAP: an estimate FARTHER from the marker than half the
            # elbow->tip span is not a wrist correction, it's a failed estimate
            # (Hulk: a misplaced HAND_TIP sent _wrist_from_mesh 1141mm away and
            # the snap dragged the whole finger step with it). Real corrections
            # observed across the regression hands are 2-23mm. Keep the
            # body-detect marker in that case.
            _span = (_e.location - _tip_p).length
            if _span > 1e-6 and _off > 0.5 * _span:
                dbg(f"[fingers] {arp_side}: mesh-wrist estimate {_off*1000:.0f}mm "
                      f"from the HAND marker (> half the elbow->tip span "
                      f"{_span*1000:.0f}mm) -- estimate rejected, keeping HAND marker")
                continue
            self._wrist_est[arp_side] = _wr.copy()
            if not _auto:
                continue
            if _off > 0.002:
                _h.location = _wr
                # Only ANNOUNCE a move the user would actually notice (>15 mm);
                # small corrections snap silently so the report isn't noisy.
                if _off > 0.015:
                    self._snapped.append((arp_side, _off * 1000))
                dbg(f"[fingers] HAND_{arp_side} snapped to mesh wrist ({_off*1000:.0f}mm)")
            else:
                dbg(f"[fingers] {arp_side}: HAND marker already on wrist ({_off*1000:.0f}mm)")

    def execute(self, context):
        # ── Resolve body mesh ─────────────────────────────────────────────────
        props    = getattr(context.scene, "autorig_face_objs", None)
        mesh_obj = _scene_mesh_or_none(context, props.detect_body_obj if props else None)
        if mesh_obj is None:
            meshes = [
                o for o in context.scene.objects
                if o.type == 'MESH' and not o.get("autorig_marker")
            ]
            if not meshes:
                self.report({'ERROR'}, "No mesh found. Set a Body Mesh in the panel.")
                return {'CANCELLED'}
            mesh_obj = max(meshes, key=lambda o: o.dimensions.z)

        # Snap the HAND (wrist) markers onto the mesh wrist FIRST, at true scale,
        # BEFORE the forearm is measured below. A wrist marker dragged up the
        # forearm shrinks the measured forearm and inflates the detect scale
        # (x1.90 vs x2.36 on the same hand), which then poisons the whole
        # detection even though the snap later fixes the wrist. Correcting it here
        # keeps the scale stable regardless of where the marker sat.
        self._snap_wrists(context, mesh_obj)

        # Fingers care about HAND size, not body height. The height-based
        # normalization leaves stylized proportions (babies, chibis) with a
        # proportionally tiny hand and the body inside the hand crop; anchor
        # the detect scale to the FOREARM (canonical metarig 0.291 m) instead.
        # Adults land in the +/-20% dead band -> no rescale, behaviour unchanged.
        # Measure from the mesh-wrist ESTIMATE when one was accepted (recorded in
        # _snap_wrists even with auto-snap OFF): the detect scale must not change
        # when the user nudges the wrist marker — a 6mm marker move otherwise
        # re-scales the whole detection world by several % and shifts every
        # finger marker (the auto-snap framing leak).
        _flens = []
        _west  = getattr(self, '_wrist_est', {})
        for _fs in ('L', 'R'):
            _fh = bpy.data.objects.get(f"MARKER_HAND_{_fs}")
            _fe = bpy.data.objects.get(f"MARKER_ELBOW_{_fs}")
            if _fh is not None and _fe is not None:
                _fw = _west.get(_fs, _fh.location)
                _fl = (_fw - _fe.location).length
                if _fl > 1e-4:
                    _flens.append(_fl)
        _auto_marker_scale(context, mesh_obj)
        _fscale = None
        if _flens:
            _favg = sum(_flens) / len(_flens)
            if not (_DEFAULT_FOREARM_LEN * 0.8 <= _favg <= _DEFAULT_FOREARM_LEN * 1.2):
                _fscale = _DEFAULT_FOREARM_LEN / _favg
        with _normalized_detect_scale(mesh_obj, scale=_fscale):
            return self._detect_fingers(context, mesh_obj)

    def _detect_fingers(self, context, mesh_obj):
        import tempfile, shutil

        # ── Read HAND / ELBOW marker positions ────────────────────────────────
        hand_elbow = {}
        for side in ('L', 'R'):
            for key in (f"HAND_{side}", f"ELBOW_{side}"):
                obj = bpy.data.objects.get(f"MARKER_{key}")
                if obj is not None:
                    hand_elbow[key] = obj.location.copy()

        if not any(f"HAND_{s}" in hand_elbow for s in ('L', 'R')):
            self.report({'ERROR'}, "No HAND markers found. Run Auto Detect Arms or EasyDetect Body first.")
            return {'CANCELLED'}

        # ── Run finger detection for each side ────────────────────────────────
        from .ai_detect_lvt import (detect_fingers_hybrid,
                                     enforce_finger_symmetry, resnap_tips_to_finger_end,
                                     seat_finger_tips_on_pad,
                                     center_finger_joints_in_volume,
                                     enforce_finger_chain_order, finger_quality_report,
                                     finger_failure_count,
                                     straighten_fingers_lateral, fix_thumb_base,
                                     separate_collapsed_tips, snap_joints_inside_mesh,
                                     contain_finger_chains, reseat_offaxis_tips,
                                     recentre_thumb_interior)
        from .constants import FINGER_BASE_NAMES, ALL_MARKERS

        finger_td = tempfile.mkdtemp(prefix="er_fingers_")
        marker_pos = {}
        _quality   = []
        _snapped   = getattr(self, '_snapped', [])   # set in _snap_wrists (pre-scale)
        try:
            from .ai_detect_geo import _LAST_PLAUSIBILITY
            _LAST_PLAUSIBILITY.clear()               # avoid a stale score leaking a flag
        except Exception:
            pass
        dtype_map  = {name: dtype for name, dtype, _ in ALL_MARKERS}
        engine     = effective_finger_engine(context)
        if engine == 'AUTO':
            # AUTO = the template-constrained pipeline: neural evidence (geo
            # fallback when thin), validation-gated template rebuild, full
            # cleanup, and the existing per-side geometric quality takeover.
            # Every stage is already "use the best, rebuild only on failure".
            engine = 'TEMPLATE'
            dbg("[fingers] AUTO engine -> template-constrained pipeline "
                  "(neural evidence + geometric fallback + quality takeover)")

        try:
            for arp_side in ('L', 'R'):
                hw = hand_elbow.get(f"HAND_{arp_side}")
                ew = hand_elbow.get(f"ELBOW_{arp_side}")
                if hw is None:
                    self.report({'WARNING'}, f"HAND_{arp_side} marker missing — skipping {arp_side} hand.")
                    continue
                # (Wrist markers were already snapped to the mesh wrist in
                # _snap_wrists, BEFORE the forearm scale was measured, so hw/hand_elbow
                # here already hold the corrected wrist.)
                if engine == 'GEOMETRIC':
                    # Geodesic-tube engine: mesh-only, no renders/onnxruntime.
                    # Raw output (no neural cleanup passes) so the engine can be
                    # judged on its own. A failed side is NOT template-filled
                    # here: the post-loop step mirrors the good side instead.
                    try:
                        from .ai_detect_geo import detect_fingers_geo
                        result = detect_fingers_geo(
                            mesh_obj, hw, ew, arp_side,
                            knuckle_depth=getattr(context.scene, 'geo_knuckle_depth', 0.22),
                            thumb_depth=getattr(context.scene, 'geo_thumb_depth', 0.45),
                            min_finger=getattr(context.scene, 'geo_min_finger', 0.30))
                        if result:
                            marker_pos.update(result)
                    except Exception as e:
                        self.report({'WARNING'}, f"Geometric finger detection ({arp_side}): {e}")
                    continue
                if engine == 'TEMPLATE':
                    # Phase-1 constrained builder. Get evidence from the NEURAL
                    # detector (fallback: GEOMETRIC), then REBUILD the hand from an
                    # always-valid template. Falls through to the shared cleanup
                    # below (engine != 'GEOMETRIC') so the template is projected
                    # onto the mesh exactly like a neural result.
                    try:
                        from .ai_detect_template import detect_fingers_template
                        width_tol = getattr(context.scene, 'finger_width_tolerance',  1.0)
                        str_clamp = getattr(context.scene, 'finger_straighten_clamp', 0.15)
                        center_r  = getattr(context.scene, 'finger_center_radius',   0.015)
                        palm_d    = getattr(context.scene, 'finger_palm_depth',       0.25)
                        knuckle_r = getattr(context.scene, 'finger_knuckle_radius',   0.90)
                        evidence = None
                        try:
                            evidence = detect_fingers_hybrid(
                                mesh_obj, hw, ew, finger_td, arp_side,
                                center_radius=center_r, palm_depth=palm_d,
                                knuckle_radius=knuckle_r, width_tolerance=width_tol,
                                straighten_clamp=str_clamp)
                        except Exception as _ne:
                            dbg(f"[template {arp_side}] neural evidence failed: {_ne}")
                        if not evidence:
                            try:
                                from .ai_detect_geo import detect_fingers_geo
                                evidence = detect_fingers_geo(
                                    mesh_obj, hw, ew, arp_side,
                                    knuckle_depth=getattr(context.scene, 'geo_knuckle_depth', 0.22),
                                    thumb_depth=getattr(context.scene, 'geo_thumb_depth', 0.45),
                                    min_finger=getattr(context.scene, 'geo_min_finger', 0.30))
                            except Exception as _ge:
                                dbg(f"[template {arp_side}] geometric evidence failed: {_ge}")
                        result = detect_fingers_template(mesh_obj, hw, ew, arp_side, evidence)
                        if result:
                            marker_pos.update(result)
                        elif evidence:
                            # Template declined (thin evidence) -> keep the raw
                            # detector result so we still place something.
                            marker_pos.update(evidence)
                    except Exception as e:
                        self.report({'WARNING'}, f"Template finger detection ({arp_side}): {e}")
                    continue
                try:
                    width_tol = getattr(context.scene, 'finger_width_tolerance',  1.0)
                    str_clamp = getattr(context.scene, 'finger_straighten_clamp', 0.15)
                    center_r  = getattr(context.scene, 'finger_center_radius',   0.015)
                    palm_d    = getattr(context.scene, 'finger_palm_depth',       0.25)
                    knuckle_r = getattr(context.scene, 'finger_knuckle_radius',   0.90)
                    result = detect_fingers_hybrid(mesh_obj, hw, ew, finger_td, arp_side,
                                                   center_radius=center_r,
                                                   palm_depth=palm_d,
                                                   knuckle_radius=knuckle_r,
                                                   width_tolerance=width_tol,
                                                   straighten_clamp=str_clamp)
                    if result:
                        marker_pos.update(result)
                except Exception as e:
                    self.report({'WARNING'}, f"Finger detection ({arp_side}): {e}")
            # Pull apart adjacent tips the model collapsed (e.g. ring->pinky on
            # untrained finger styles) using each finger's own chain -- per side,
            # BEFORE symmetry so a fixed side can still be mirrored if the other fails.
            if engine == 'GEOMETRIC':
                # Geometric post-handling: mirror a wholly-failed side from the
                # good side; run per-finger symmetry (average agreeing fingers,
                # mirror the better chain on disagreement) when both detected;
                # proportions template only when BOTH sides failed.
                hw_l = hand_elbow.get("HAND_L")
                hw_r = hand_elbow.get("HAND_R")
                ok = {s: any(k.endswith(f"_{s}") for k in marker_pos)
                      for s in ('L', 'R')}
                if hw_l and hw_r and ok['L'] != ok['R']:
                    src, dst = ('L', 'R') if ok['L'] else ('R', 'L')
                    cx = (hw_l.x + hw_r.x) * 0.5
                    for k, v in list(marker_pos.items()):
                        if k.endswith(f"_{src}"):
                            marker_pos[f"{k[:-2]}_{dst}"] = Vector((2.0 * cx - v.x, v.y, v.z))
                    dbg(f"[geo] {dst} side failed -- mirrored {src} onto {dst}")
                    self.report({'WARNING'},
                                f"Geometric engine: {dst} hand failed — mirrored the {src} hand.")
                elif (hw_l and hw_r and ok['L'] and ok['R']
                      and getattr(context.scene, "autorig_detect_symmetry", True)):
                    enforce_finger_symmetry(marker_pos, hw_l, hw_r)
                for _gs in ('L', 'R'):
                    _ghw = hand_elbow.get(f"HAND_{_gs}")
                    if _ghw is not None and not any(k.endswith(f"_{_gs}") for k in marker_pos):
                        self.report({'WARNING'},
                                    f"Geometric engine ({_gs}): no result — using proportions template.")
                        marker_pos.update(_place_fingers_geometric(
                            _ghw, hand_elbow.get(f"ELBOW_{_gs}"), _gs))
            # GEOMETRIC engine skips decollapse/symmetry/cleanup: those passes are
            # tuned for neural output, and raw results make the engine testable.
            if engine != 'GEOMETRIC':
                for _ds_side in ('L', 'R'):
                    if any(k.endswith(f"_{_ds_side}") for k in marker_pos):
                        separate_collapsed_tips(marker_pos, _ds_side, mesh_obj)
                hw_l = hand_elbow.get("HAND_L")
                hw_r = hand_elbow.get("HAND_R")
                _sym_on = getattr(context.scene, "autorig_detect_symmetry", True)
                if not _sym_on:
                    dbg("[symmetry] Symmetrical Detect OFF — keeping raw per-side hands")
                if hw_l and hw_r and _sym_on:
                    enforce_finger_symmetry(marker_pos, hw_l, hw_r)
                # ── Post-ONNX finger cleanup passes ───────────────────────────
                # Conservative, robust passes: tip-to-end, tip seated on the finger
                # pad, depth-centring into the finger volume, along-axis joint
                # ordering, guarded thumb-base anti-wrist-drift, and lateral
                # PIP/DIP straighten.
                _CLEANUP = {
                    'resnap':   True,   # tips -> distal finger end
                    'pad':      True,   # tips -> fingertip pad
                    'depth':    True,   # joints -> finger volume
                    'order':    True,   # along-axis MCP<PIP<DIP<TIP ordering
                    'straight': True,   # planar chains (lateral zigzag = 0)
                    'thumb_base': True, # guarded THUMB_1 anti-wrist-drift
                    'inside':   True,   # joints OUTSIDE the mesh -> just inside
                    'contain':  True,   # whole chain SEGMENTS inside the mesh
                }
                _dbg_key = os.environ.get("ER_DEBUG_JOINT")

                def _dbg(stage):
                    # Pass-by-pass position trace for ONE marker (debugging
                    # which cleanup pass moves a joint; harmless when the env
                    # var is unset).
                    if _dbg_key and _dbg_key in marker_pos:
                        _v = marker_pos[_dbg_key]
                        print(f"  [dbg {_dbg_key}] after {stage}: "
                              f"({_v.x:.4f}, {_v.y:.4f}, {_v.z:.4f})")

                _dbg("detect+symmetry")
                for _rs_side in ('L', 'R'):
                    if any(k.endswith(f"_{_rs_side}") for k in marker_pos):
                        if _CLEANUP['resnap']:
                            resnap_tips_to_finger_end(marker_pos, mesh_obj, _rs_side)
                            _dbg(f"resnap {_rs_side}")
                        if _CLEANUP['pad']:
                            seat_finger_tips_on_pad(marker_pos, mesh_obj, _rs_side)
                            _dbg(f"pad {_rs_side}")
                        if _CLEANUP['depth']:
                            center_finger_joints_in_volume(marker_pos, mesh_obj, _rs_side)
                            _dbg(f"depth {_rs_side}")
                        if _CLEANUP['thumb_base']:
                            fix_thumb_base(marker_pos, _rs_side)
                        if _CLEANUP['order']:
                            enforce_finger_chain_order(marker_pos, _rs_side)
                            _dbg(f"order {_rs_side}")
                        if _CLEANUP['straight']:
                            straighten_fingers_lateral(marker_pos, _rs_side)
                            _dbg(f"straight {_rs_side}")
                        if _CLEANUP['inside']:
                            snap_joints_inside_mesh(marker_pos, mesh_obj, _rs_side)
                            _dbg(f"inside {_rs_side}")
                        if _CLEANUP['contain']:
                            contain_finger_chains(marker_pos, mesh_obj, _rs_side)
                            _dbg(f"contain {_rs_side}")
                        if _CLEANUP['depth'] and _CLEANUP['contain']:
                            # Second depth pass: [inside]/[contain] rescue
                            # joints to JUST INSIDE the skin (1.5-3mm), which
                            # leaves them ON the finger surface instead of
                            # centred like their neighbours. Depth-only and
                            # clamped, so it is lane-safe and a no-op for
                            # joints already centred.
                            center_finger_joints_in_volume(marker_pos, mesh_obj, _rs_side)
                            recentre_thumb_interior(marker_pos, mesh_obj, _rs_side)
                            _dbg(f"depth2 {_rs_side}")
                        # Off-axis TIP re-seat: gated, fires only when a tip
                        # hugs the side/top surface while the interior chain
                        # is centred (chubby round fingertips defeat the
                        # surface-extreme seating of resnap/pad).
                        reseat_offaxis_tips(marker_pos, mesh_obj, _rs_side)
                        _dbg(f"tip-axis {_rs_side}")
                        if _CLEANUP['straight']:
                            # FINAL planarization: [inside]/[contain]/tip-axis
                            # can nudge joints laterally after the mid-chain
                            # straighten; Rigify bones zigzag on any lateral
                            # wobble, so the chains are flattened onto their
                            # bend planes once more as the LAST word.
                            straighten_fingers_lateral(marker_pos, _rs_side)
                            _dbg(f"straight2 {_rs_side}")
                # ── Quality-gated geometric takeover ───────────────────────────
                # On untrained styles the neural pair can "succeed" with merged
                # or laterally-drifted fingers — errors the cleanup passes must
                # not touch (no lateral manipulation). The geometric engine
                # fails closed, so when a side's quality report raises flags
                # AND the geometric engine returns a full hand (passed its own
                # structural gates), the geometric result takes that side.
                try:
                    from .ai_detect_geo import detect_fingers_geo as _geo_detect
                    _geo_swapped = []
                    for _gs in ('L', 'R'):
                        _ghw = hand_elbow.get(f"HAND_{_gs}")
                        if (_ghw is None
                                or not any(k.endswith(f"_{_gs}") for k in marker_pos)):
                            continue
                        _warns = finger_quality_report(marker_pos, _gs)
                        if not _warns:
                            continue
                        dbg(f"[takeover {_gs}] neural quality flags "
                              f"({'; '.join(_warns)}) -- trying geometric engine")
                        _geo = None
                        try:
                            _geo = _geo_detect(
                                mesh_obj, _ghw, hand_elbow.get(f"ELBOW_{_gs}"), _gs,
                                knuckle_depth=getattr(context.scene, 'geo_knuckle_depth', 0.22),
                                thumb_depth=getattr(context.scene, 'geo_thumb_depth', 0.45),
                                min_finger=getattr(context.scene, 'geo_min_finger', 0.30))
                        except Exception as _ge:
                            dbg(f"[takeover {_gs}] geometric engine error: {_ge}")
                        if _geo and len(_geo) >= 20:
                            marker_pos.update(_geo)
                            _geo_swapped.append(_gs)
                            dbg(f"[takeover {_gs}] geometric result replaces neural")
                        else:
                            dbg(f"[takeover {_gs}] geometric engine declined "
                                  f"-- keeping neural result")
                    if _geo_swapped and _sym_on:
                        hw_l = hand_elbow.get("HAND_L")
                        hw_r = hand_elbow.get("HAND_R")
                        if hw_l and hw_r:
                            enforce_finger_symmetry(marker_pos, hw_l, hw_r)
                except Exception as _te:
                    dbg(f"[takeover] skipped: {_te}")
                # FINAL symmetry: per-side cleanup guards fire asymmetrically
                # (L thumb DIP floated while R sat inside), and Rigify wants
                # exactly mirrored hands. Containment-based: the side that
                # sits inside the mesh wins; equal sides are mirror-averaged.
                # Skipped when Symmetrical Detect is OFF.
                try:
                    hw_l = hand_elbow.get("HAND_L")
                    hw_r = hand_elbow.get("HAND_R")
                    if hw_l and hw_r and _sym_on:
                        from .ai_detect_lvt import enforce_final_symmetry
                        enforce_final_symmetry(marker_pos, mesh_obj, hw_l, hw_r)
                except Exception as _fse:
                    dbg(f"[symmetry-final] skipped: {_fse}")
            # Quality check: flag low-confidence hands so the user can spot-check
            # instead of silently shipping a bad result.
            _quality = []
            for _qs in ('L', 'R'):
                for _w in finger_quality_report(marker_pos, _qs):
                    _quality.append(f"{_qs} {_w}")
            # Per-hand independent-defect count for the "drop the hand if >=2
            # fingers failed" policy. LOGGED ONLY for now (no dropping yet) so we
            # can validate the threshold on real hands before it drops results.
            _fail_counts = {}
            _fail_severe = {}
            for _qs in ('L', 'R'):
                if any(k.endswith(f"_{_qs}") for k in marker_pos):
                    _n, _fw = finger_failure_count(marker_pos, _qs)
                    _sev = any('collapsed' in _w for _w in _fw)
                    _fail_counts[_qs] = _n
                    _fail_severe[_qs] = _sev
                    dbg(f"[finger-fail] {_qs}: {_n} independent defect(s)"
                          + ("  [SEVERE: collapsed finger]" if _sev else "")
                          + (f" -- {'; '.join(_fw)}" if _fw else ""))
            # Backstop: a geometric parse where NO 5-tube set looked like a hand
            # (low plausibility) is flagged even when no specific invariant tripped
            # -- protection for end-user characters we never tested.
            try:
                from .ai_detect_geo import _LAST_PLAUSIBILITY, _PLAUSIBILITY_FLOOR
                for _qs in ('L', 'R'):
                    _pl = _LAST_PLAUSIBILITY.get(_qs)
                    if _pl is not None and _pl < _PLAUSIBILITY_FLOOR:
                        _quality.append(f"{_qs} low confidence (plausibility {_pl:.2f})")
            except Exception:
                pass
        finally:
            shutil.rmtree(finger_td, ignore_errors=True)

        if not marker_pos:
            self.report({'WARNING'}, "No finger markers detected.")
            return {'CANCELLED'}

        # ── Place / update finger markers ─────────────────────────────────────
        col = get_or_create_collection("RigifyMarkers")
        col.hide_viewport = False
        moved = 0
        for marker_name, world_pos in marker_pos.items():
            obj_name = f"MARKER_{marker_name}"
            existing = bpy.data.objects.get(obj_name)
            if existing is None:
                dtype = dtype_map.get(marker_name, "SPHERE")
                make_empty(col, obj_name, dtype, world_pos, FINGER_SIZE, context)
            else:
                existing.location = world_pos
                existing.show_name = False       # clear retired suspect labels
            moved += 1
        try:
            _clear_finger_confidence_visuals()
        except Exception:
            pass

        # Tell the user when a HAND (wrist) marker was auto-moved to the wrist,
        # so the marker shifting during Detect Fingers is never a silent surprise.
        _snap_msg = ("Moved "
                     + ", ".join(f"HAND_{s} {mm:.0f}mm to the wrist" for s, mm in _snapped)
                     + ". ") if _snapped else ""

        if _quality:
            # Graceful-failure UX: a low-confidence hand is FLAGGED with an
            # actionable next step instead of silently shipping a bad result —
            # suggest the OTHER engine + a manual touch-up.
            _other = {'NEURAL': 'Geometric', 'GEOMETRIC': 'EasyDetect',
                      'TEMPLATE': 'Geometric'}.get(engine, 'the other')
            self.report({'WARNING'},
                        _snap_msg
                        + f"EasyDetect Fingers: placed {moved} markers — LOW CONFIDENCE on "
                        + "; ".join(_quality)
                        + f". Spot-check these; try the {_other} engine or nudge the "
                          f"markers by hand.")
        else:
            self.report({'INFO'},
                        _snap_msg
                        + f"EasyDetect Fingers: placed {moved} finger markers.")

        # Blocking popup when a hand's detection is unreliable enough to warrant
        # switching engine or placing by hand: either >=2 INDEPENDENT defects (one
        # merged/short finger alone is an easy manual fix), OR a SEVERE single
        # defect -- a whole finger collapsed onto its neighbour (landed on the wrong
        # finger), which needs more than a single-marker nudge. The MARKERS ARE LEFT
        # IN PLACE (not dropped) so they can be reviewed / used for troubleshooting.
        _fail_sides = [s for s in ('L', 'R')
                       if _fail_counts.get(s, 0) >= 2 or _fail_severe.get(s)]
        if _fail_sides:
            _other2 = {'NEURAL': 'Geometric', 'GEOMETRIC': 'EasyDetect',
                       'TEMPLATE': 'Geometric'}.get(engine, 'the other')
            _hw = {'L': 'Left', 'R': 'Right'}
            _lines = ["Finger detection looks unreliable:"]
            for s in _fail_sides:
                _why = ("a finger collapsed onto its neighbour"
                        if _fail_severe.get(s)
                        else f"{_fail_counts[s]} fingers unreliable")
                _lines.append(f"  {_hw[s]} hand: {_why}")
            _lines += ["",
                       f"Try the {_other2} engine, or adjust the",
                       "markers manually. (Markers left in place.)"]
            _popup_message(context, "\n".join(_lines),
                           title="Finger Detection Warning", icon='ERROR')
        return {'FINISHED'}


def _popup_message(context, message, title="Notice", icon='INFO'):
    """Popup message box (window_manager.popup_menu). Each newline in `message`
    becomes its own label line (popup labels don't wrap, so keep lines short).
    Fireable from an operator's execute(). Never raises."""
    def _draw(self, _ctx):
        for _line in message.split("\n"):
            self.layout.label(text=_line)
    try:
        (context or bpy.context).window_manager.popup_menu(
            _draw, title=title, icon=icon)
    except Exception as _e:
        dbg(f"[popup] {title}: {message!r}  (popup failed: {_e})")


_CONF_HALO_PREFIX = "ER_CONF_HALO_"


def _clear_finger_confidence_visuals():
    """Remove the retired confidence visuals (halo spheres + name labels) from
    scenes that were detected while the feature existed. The user preferred the
    LOW CONFIDENCE warning text alone — this only cleans, it never creates."""
    for o in [o for o in bpy.data.objects if o.name.startswith(_CONF_HALO_PREFIX)]:
        me = o.data
        bpy.data.objects.remove(o, do_unlink=True)
        if me is not None and me.users == 0:
            bpy.data.meshes.remove(me)


def _finger_from_marker_name(obj_name):
    """MARKER_FINGER_INDEX_2_L -> ("index", "L"); MARKER_THUMB_TIP_R ->
    ("thumb", "R"); None for non-finger markers."""
    if not obj_name.startswith("MARKER_"):
        return None
    base = obj_name[len("MARKER_"):]
    if base.endswith("_L"):
        side, base = "L", base[:-2]
    elif base.endswith("_R"):
        side, base = "R", base[:-2]
    else:
        return None
    from .ai_detect_lvt import _FINGER_TO_ARP
    for finger, arp in _FINGER_TO_ARP.items():
        if base in arp.values():
            return (finger, side)
    return None


class AUTORIG_OT_ResolveFinger(bpy.types.Operator):
    """Re-solve ONE finger from its markers, without re-running detection.
Select any marker of the finger (tip is the natural handle), drag it where it
belongs, then run this: the knuckle (MCP) and TIP markers are kept as anchors,
the two interior joints are re-laid between them and re-seated into the finger
volume from the mesh. Sub-second — turns a residual bad finger into a
drag-and-click fix instead of a full re-detect or four manual joint placements"""
    bl_idname  = "autorig.resolve_finger"
    bl_label   = "Re-solve Selected Finger"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and _finger_from_marker_name(ob.name) is not None

    def execute(self, context):
        ident = _finger_from_marker_name(context.active_object.name)
        if ident is None:
            self.report({'ERROR'}, "Select a finger marker first (any joint of "
                                   "the finger to re-solve).")
            return {'CANCELLED'}
        finger, side = ident

        props    = getattr(context.scene, "autorig_face_objs", None)
        mesh_obj = _scene_mesh_or_none(context, props.detect_body_obj if props else None)
        if mesh_obj is None:
            meshes = [o for o in context.scene.objects
                      if o.type == 'MESH' and not o.get("autorig_marker")]
            if not meshes:
                self.report({'ERROR'}, "No mesh found. Set a Body Mesh in the panel.")
                return {'CANCELLED'}
            mesh_obj = max(meshes, key=lambda o: o.dimensions.z)

        from .ai_detect_lvt import (_FINGER_TO_ARP, center_finger_joints_in_volume,
                                    snap_joints_inside_mesh, contain_finger_chains,
                                    straighten_fingers_lateral)

        # Full-hand context: the containment lane guards need every finger's
        # markers to know which tube owns which joint. Only the selected
        # finger's keys are written back.
        marker_pos = {}
        for f, arp in _FINGER_TO_ARP.items():
            for part in ("phal1", "phal2", "phal3", "tip"):
                key = f"{arp[part]}_{side}"
                ob  = bpy.data.objects.get(f"MARKER_{key}")
                if ob is not None:
                    marker_pos[key] = ob.location.copy()

        arp = _FINGER_TO_ARP[finger]
        own_keys = [f"{arp[part]}_{side}" for part in ("phal1", "phal2", "phal3", "tip")]
        mcp = marker_pos.get(own_keys[0])
        tip = marker_pos.get(own_keys[3])
        if mcp is None or tip is None or (tip - mcp).length < 1e-5:
            self.report({'ERROR'}, f"Need both the {finger} knuckle (MCP) and TIP "
                                   f"markers on side {side} to re-solve.")
            return {'CANCELLED'}

        # Interior joints at anatomical fractions of the (possibly just-dragged)
        # MCP->TIP chord, then mesh passes seat them into the finger volume.
        # Same fractions as the template engine (ai_detect_template._PHALANX_FRAC).
        chord = tip - mcp
        marker_pos[own_keys[1]] = mcp + chord * 0.44
        marker_pos[own_keys[2]] = mcp + chord * 0.73

        try:
            # Same order lesson as the main pipeline: the PLANARIZER runs LAST,
            # because every mesh pass after it re-nudges joints laterally and
            # brings the zigzag back (Rigify needs each chain in one bend plane).
            center_finger_joints_in_volume(marker_pos, mesh_obj, side)
            snap_joints_inside_mesh(marker_pos, mesh_obj, side)
            contain_finger_chains(marker_pos, mesh_obj, side)
            center_finger_joints_in_volume(marker_pos, mesh_obj, side)
            straighten_fingers_lateral(marker_pos, side)
            # The planarizer pulls PIP/DIP onto the MCP->TIP line. When the user has
            # dragged the TIP off the finger's own axis, that line leaves the finger
            # tube and the pull can leave a joint ON or BELOW the skin — with nothing
            # after it to re-seat them (a depth re-centre here would re-introduce the
            # lateral zigzag the planarizer just removed). snap_joints_inside_mesh is
            # safe as the final pass: it moves ONLY joints measurably outside the mesh
            # (good, planar joints are untouched, so the straightening is preserved),
            # pulling each just inside along the surface normal with its own wrong-
            # finger / wrong-chord guards so it can't cross onto a neighbour.
            snap_joints_inside_mesh(marker_pos, mesh_obj, side)
        except Exception as e:
            self.report({'WARNING'}, f"Mesh refinement partly failed ({e}) — "
                                     f"chain laid on the straight MCP->TIP chord.")

        moved = 0
        for part, key in zip(("phal1", "phal2", "phal3", "tip"), own_keys):
            if part == "tip":
                continue          # the dragged tip is the trusted anchor — never moved
            ob = bpy.data.objects.get(f"MARKER_{key}")
            if ob is not None and key in marker_pos:
                ob.location = marker_pos[key]
                moved += 1
        self.report({'INFO'}, f"Re-solved {finger} ({side}): {moved} joints "
                              f"re-seated between your MCP and TIP.")
        return {'FINISHED'}


def _face_eyebrow_geo_suspects(positions):
    """QUALITY GATE for the NEURAL eye/brow markers (mirrors the finger quality-gated
    geometric takeover). On an in-distribution face the model places eye+brow well,
    so this stays SILENT. On an UNTRAINED / stylized / shallow face the model can
    collapse adjacent brow points together or shrink an eye — those markers are then
    better served by the geometric detector (the same tool that LABELED the training
    data). Returns a set of FULL marker names (with _L/_R) to hand to the geometric
    fill. Only ever flags markers the EYEBROWS_EYE geometric section can re-place.

    Conservative by design — do NOT tune the thresholds on a single face; validate
    across several (a working face must stay 0 suspects). Prints its metrics so a
    borderline face can be diagnosed.

    Two independent, self-scaling checks, applied per side:
      • BROW CROWDING: the inner->outer brow chain (BROW_1,2,3,OUTER) should be
        roughly evenly spread; if the smallest gap collapses below 0.30x the widest,
        two brow points merged -> hand that side's whole brow group to geometric.
      • EYE-WIDTH ASYMMETRY: |EYE_INNER-EYE_OUTER| per side; if one eye is under
        0.55x the other, that eye collapsed -> hand that side's eye ring to geometric.
    """
    suspects = set()
    _brow_grp = ("FACE_BROW_1", "FACE_BROW_2", "FACE_BROW_3",
                 "FACE_BROW_OUTER", "FACE_BROW_BOT_OUTER")
    # FULL eyelid set (all 8 markers the EYELIDS / "Detect Eyelid" section places as
    # ONE coherent unit: ring + creases + bot-outer). Hand over the WHOLE set, not
    # just the 4-marker ring — leaving LID_CREASE_T / CREASE_INNER / CREASE_OUTER as
    # neural while the ring is geometric gives a mismatched eye; "Detect Eyelid"
    # re-places all 8 together and looks better (markers.py EYELIDS section).
    _eye_grp  = ("FACE_EYE_INNER", "FACE_EYE_OUTER", "FACE_EYE_TOP", "FACE_EYE_BOT",
                 "FACE_LID_CREASE_T", "FACE_CREASE_INNER", "FACE_CREASE_OUTER",
                 "FACE_BROW_BOT_OUTER")

    # --- Brow spacing uniformity, per side ---
    # A good brow chain is roughly EVENLY spread inner->outer, so min-gap/max-gap is
    # near 1. Both failure modes drop it: two markers COLLAPSED (tiny min gap) OR one
    # marker displaced leaving a HUGE gap (large max). Threshold 0.45 catches the
    # chibi (L=0.35 huge-gap, R=0.41 collapsed-pair) while a working face clears it
    # (Face1 was 0.60/0.83). If EITHER brow trips, hand BOTH brows to geometric so the
    # pair stays consistent (= what "Detect Brows & Eyes" produces).
    _brow_bad = False
    for side in ("L", "R"):
        chain = [positions.get(f"{b}_{side}")
                 for b in ("FACE_BROW_1", "FACE_BROW_2", "FACE_BROW_3", "FACE_BROW_OUTER")]
        if all(p is not None for p in chain):
            gaps = [(chain[i] - chain[i + 1]).length for i in range(len(chain) - 1)]
            gmax, gmin = max(gaps), min(gaps)
            uneven = gmax > 1e-6 and gmin < 0.45 * gmax
            dbg(f"  [face-gate] brow {side}: gaps(mm)="
                  f"{[round(g*1000,1) for g in gaps]} min/max={gmin/gmax:.2f}"
                  f"{'  -> UNEVEN, geo' if uneven else ''}")
            _brow_bad = _brow_bad or uneven
    if _brow_bad:
        for side in ("L", "R"):
            suspects.update(f"{b}_{side}" for b in _brow_grp)
        suspects.add("FACE_BROW")            # centre marker, derived from the inner brows

    # --- Eye-width L/R asymmetry ---
    # Neutral faces are ~symmetric, so a >~2x eye-width difference means one eye
    # collapsed. We can't reliably tell WHICH side is the bad one (the wide side can
    # be the error just as easily as the narrow one), so hand BOTH eyes to the EYELIDS
    # detector — it re-derives both from the eyeball mesh, exactly like clicking
    # "Detect Eyelid". Silent on a working face (Face1 ratio was 0.97).
    def _eye_w(side):
        i = positions.get(f"FACE_EYE_INNER_{side}")
        o = positions.get(f"FACE_EYE_OUTER_{side}")
        return (i - o).length if (i is not None and o is not None) else None
    wl, wr = _eye_w("L"), _eye_w("R")
    if wl and wr:
        lo, hi = min(wl, wr), max(wl, wr)
        collapsed = lo < 0.55 * hi
        dbg(f"  [face-gate] eye width(mm) L={wl*1000:.1f} R={wr*1000:.1f} "
              f"ratio={lo/hi:.2f}{'  -> asymmetric, both eyes -> geo' if collapsed else ''}")
        if collapsed:
            for side in ("L", "R"):
                suspects.update(f"{e}_{side}" for e in _eye_grp)

    return suspects


def symmetrize_face_markers(context):
    """Mirror-average the MARKER_FACE_* empties about the face midline so the
    face comes out perfectly symmetric — the main asymmetry source is a detect
    run without the eye meshes assigned (eye/lid/brow markers then come from
    per-side estimates that disagree a few mm L/R).

    Gated on scene.autorig_detect_symmetry (the shared "Symmetrical Detect"
    toggle used by body/finger/face detects). Bilateral pairs are averaged and
    mirrored; midline markers are pinned to the centreline. TEETH/TONGUE are
    skipped — they're placed from their own meshes, and dragging them to the
    face centreline could pull them off those meshes.

    Returns the number of markers moved.
    """
    if not getattr(context.scene, "autorig_detect_symmetry", True):
        dbg("[face-symmetry] Symmetrical Detect OFF — keeping raw face markers")
        return 0
    objs = {}
    for o in bpy.data.objects:
        if (o.name.startswith("MARKER_FACE_")
                and "TEETH" not in o.name and "TONGUE" not in o.name):
            objs[o.name] = o
    pairs = []
    for n, o in objs.items():
        if n.endswith("_L"):
            ro = objs.get(n[:-2] + "_R")
            if ro is not None:
                pairs.append((o, ro))
    if not pairs:
        return 0
    # Face centreline = median of pair midpoints (robust to a few bad pairs;
    # the bbox centre would be skewed by asymmetric hair/horns).
    mids = sorted((o.location.x + ro.location.x) * 0.5 for o, ro in pairs)
    cx = mids[len(mids) // 2]
    moved = 0
    for o, ro in pairs:
        off = (abs(o.location.x - cx) + abs(ro.location.x - cx)) * 0.5
        y = (o.location.y + ro.location.y) * 0.5
        z = (o.location.z + ro.location.z) * 0.5
        s = 1.0 if o.location.x >= cx else -1.0
        o.location = (cx + s * off, y, z)
        ro.location = (cx - s * off, y, z)
        moved += 2
    for n, o in objs.items():
        if not (n.endswith("_L") or n.endswith("_R")):
            o.location.x = cx
            moved += 1
    dbg(f"[face-symmetry] {moved} face markers symmetrized about x={cx:.4f} "
          f"({len(pairs)} L/R pairs averaged; midline pinned)")
    return moved


class AUTORIG_OT_AIDetectFace(bpy.types.Operator):
    """EasyDetect: place the core face markers with the neural face model:
render six ortho views, peak-detect the per-view heatmaps, triangulate to 3D and
snap to the mesh. Teeth/tongue are excluded (place those geometrically)."""
    bl_idname  = "autorig.ai_detect_face"
    bl_label   = "EasyDetect Face"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if not is_face_onnx_available():
            self.report(
                {'ERROR'},
                "Face model unavailable. Reinstall the addon and Install ONNX Runtime.")
            return {'CANCELLED'}

        props    = getattr(context.scene, "autorig_face_objs", None)
        mesh_obj = _scene_mesh_or_none(context, getattr(props, "body_obj", None) if props else None)
        if mesh_obj is None and props:
            mesh_obj = _scene_mesh_or_none(context, getattr(props, "detect_body_obj", None))
        if mesh_obj is None:
            meshes = [o for o in context.scene.objects
                      if o.type == 'MESH' and not o.get("autorig_marker")]
            if not meshes:
                self.report({'ERROR'}, "No head/face mesh found. Set the Body/Face mesh first.")
                return {'CANCELLED'}
            mesh_obj = max(meshes, key=lambda o: o.dimensions.z)

        # The geometric fill pass below (detect_face_objects) ray-casts against
        # props.body_obj to place CHEEK / JAW / CHEEK_TOP / JAW_SIDE / CHIN. This
        # operator resolves the head mesh from body_obj OR the AI panel's
        # detect_body_obj OR the scene — so a user who set only the AI picker left
        # body_obj None, the fill's BVH was None, and every ray-based cheek/jaw
        # marker silently failed to place ("jaw goes missing, cheek doesn't
        # detect"). Point body_obj at the resolved mesh so the fill has a surface.
        if props is not None:
            try:
                props.body_obj = mesh_obj
            except Exception:
                pass

        _auto_marker_scale(context, mesh_obj)

        bvh = _build_bvh(mesh_obj)
        cen, views, S = _render_face_views(mesh_obj)
        temp_dir = views[0].get("temp_dir") if views else None
        try:
            positions = _detect_face_onnx(cen, views, S, bvh)
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

        if not positions:
            self.report({'WARNING'}, "Face model produced no confident landmarks.")
            return {'CANCELLED'}

        # Match the geometric detector's face-marker sizing so AI-placed and
        # geometric-fill markers look identical: a FIXED empty_display_size plus
        # an object scale of 0.4 (0.2 for the small eyelid/crease markers) —
        # NOT make_empty's char-scaled display size with scale 1.0, which made
        # the AI markers noticeably bigger than the fill markers.
        _FACE_EYELID = {
            'FACE_EYE_CENTER', 'FACE_EYE_TOP', 'FACE_EYE_BOT',
            'FACE_EYE_INNER', 'FACE_EYE_OUTER', 'FACE_LID_CREASE_T',
            'FACE_CREASE_INNER', 'FACE_CREASE_OUTER', 'FACE_BROW_BOT_OUTER',
        }
        # HYBRID CONTOUR SET (2026-07-11, user Blender test of the 64-marker
        # model: cheek / jaw / forehead "not perfect", the rest 80-90% good).
        # These sit on smooth, feature-poor surfaces — the model's documented
        # weak tail (worst-5 every epoch: FOREHEAD / FOREHEAD_SIDE / TEMPLE).
        # DON'T place them from the neural output: delete any stale copies and
        # let the geometric fill pass below rewrite them from mesh curvature,
        # anchored on the AI core markers (the same tool that LABELED these
        # markers for training — teacher beats student on its own labels).
        _HYBRID_GEO_CONTOUR = {
            "FACE_FOREHEAD", "FACE_FOREHEAD_SIDE", "FACE_FOREHEAD_SIDE_1",
            "FACE_FOREHEAD_SIDE_2", "FACE_FOREHEAD_SIDE_3",
            "FACE_TEMPLE", "FACE_CHEEK", "FACE_CHEEK_TOP",
            "FACE_JAW", "FACE_JAW_SIDE",
        }
        # QUALITY GATE: neural eye/brow markers that collapsed on an untrained face
        # are handed to the geometric fill (like the contour set), but with a safety
        # net — the neural position is stashed and RESTORED if geometry also fails to
        # place it, so we never end up worse than neural. Silent on working faces.
        _geo_suspects = _face_eyebrow_geo_suspects(positions)
        _suspect_stash = {}

        col    = get_or_create_collection("RigifyFaceMarkers")
        placed = 0
        hybrid = 0
        gated  = 0
        for name, pos in positions.items():
            _base = name[:-2] if name.endswith(('_L', '_R')) else name
            if _base in _HYBRID_GEO_CONTOUR or name in _geo_suspects:
                _old = bpy.data.objects.get(f"MARKER_{name}")
                if _old is not None:
                    bpy.data.objects.remove(_old, do_unlink=True)
                if name in _geo_suspects:
                    _suspect_stash[name] = pos     # keep neural as last-resort fallback
                    gated += 1
                else:
                    hybrid += 1
                continue
            obj = bpy.data.objects.get(f"MARKER_{name}")
            if obj is None:
                obj = make_empty(col, f"MARKER_{name}", 'SPHERE', pos, FACE_SIZE, context)
            else:
                obj.location = pos
            _sc   = 0.2 if _base in _FACE_EYELID else 0.4
            obj.empty_display_size = FACE_SIZE
            obj.scale = (_sc, _sc, _sc)
            placed += 1

        # Geometric fill pass. Covers whatever the loaded model doesn't place:
        # with the 30-core model that's the ~34 secondary markers; with the
        # 64-marker model it's eye centres / teeth / tongue PLUS the hybrid
        # contour set skipped above (forehead / temple / cheek / jaw). Fill-only
        # mode: it runs its full detection so it can read the AI core markers
        # as anchors, but only WRITES markers that don't exist yet — the AI
        # placements stay put.
        # EYE section runs FIRST: the LIPS, EYELIDS and CHEEK sections all read
        # FACE_EYE_CENTER (which the AI doesn't output), and in a single 'ALL'
        # pass lips run BEFORE the eye section would create it -> lips were
        # skipped entirely. Placing eyes first (from the eye mesh) satisfies
        # that dependency before the rest run.
        filled = 0
        try:
            before = {o.name for o in bpy.data.objects if o.name.startswith("MARKER_FACE_")}
            bpy.ops.autorig.detect_face_objects(section='EYEBROWS_EYE', fill_missing_only=True)
            bpy.ops.autorig.detect_face_objects(section='ALL', fill_missing_only=True)
            filled = len({o.name for o in bpy.data.objects
                          if o.name.startswith("MARKER_FACE_")} - before)
        except Exception as _e:
            dbg(f"[face] secondary-marker fill failed: {_e}")

        # Safety net: if the geometric fill could not place a gated eye/brow suspect
        # (e.g. the geometric detector also failed on this odd mesh), restore its
        # neural position rather than leave the marker missing.
        _rescued = 0
        for name, pos in _suspect_stash.items():
            if bpy.data.objects.get(f"MARKER_{name}") is None:
                _base = name[:-2] if name.endswith(('_L', '_R')) else name
                obj = make_empty(col, f"MARKER_{name}", 'SPHERE', pos, FACE_SIZE, context)
                _sc = 0.2 if _base in _FACE_EYELID else 0.4
                obj.empty_display_size = FACE_SIZE
                obj.scale = (_sc, _sc, _sc)
                _rescued += 1
        if _geo_suspects:
            dbg(f"  [face-gate] {gated} eye/brow marker(s) handed to geometric; "
                  f"{gated - _rescued} re-placed, {_rescued} kept neural (geo also failed)")

        # Symmetrical Detect: mirror-average the face L/R (mainly for detects
        # run without the eye meshes, whose per-side estimates disagree).
        symmetrize_face_markers(context)

        _gate_msg = (f" + {gated} eye/brow quality-gated to geometric" if gated else "")
        self.report({'INFO'},
            f"EasyDetect Face: {placed} model landmarks + {filled} geometric fill "
            f"({hybrid} contour markers handed to geometric{_gate_msg}). Adjust as needed.")
        return {'FINISHED'}


_RENDER_TARGET_FACES = 8_000  # normalise mesh density before every body render
# Retry target when the 8k render collapses detection. Decimating some meshes
# (thin limbs / layered clothing / lots of loose geometry) to 8k collapses them
# into degenerate, self-overlapping slivers that WORKBENCH renders BLACK — the
# body then vanishes from the ortho views and every arm marker triangulates to
# the centre line. Re-rendering denser keeps the surface intact. Applied ONLY on
# a detected collapse, so healthy characters render at the trained 8k density and
# their results are unchanged (raising 8k globally shifted dense baselines badly).
_RENDER_RETRY_FACES  = 20_000


_DECIMATE_MIN_FACES = 100_000  # only decimate genuinely heavy meshes (see below)


def _disable_subdiv_modifiers(obj):
    """
    Temporarily disable SUBSURF / MULTIRES modifiers on *obj* so renders use the
    base (control-cage) mesh. Subdivision inflates the rendered poly count — which
    trips decimation and slows the render — while adding nothing the marker CNNs
    localise from. Returns a zero-arg restore() to call in a finally block.
    """
    saved = []
    for m in obj.modifiers:
        if m.type in {'SUBSURF', 'MULTIRES'}:
            saved.append((m, m.show_render, m.show_viewport))
            m.show_render   = False
            m.show_viewport = False

    def _restore():
        for m, sr, sv in saved:
            try:
                m.show_render   = sr
                m.show_viewport = sv
            except ReferenceError:
                pass  # modifier was removed under us
    return _restore


def _make_render_copy(obj, target=_RENDER_TARGET_FACES):
    """
    Return (render_obj, render_data): a temporary mesh copy of obj. On genuinely
    heavy meshes (>100k base faces) a DECIMATE modifier normalises density to
    ~target faces; lighter meshes render at full base resolution. All bounding-box
    / landmark computations still use the original obj.
    Caller must remove both: bpy.data.objects.remove(render_obj, do_unlink=True)
                             bpy.data.meshes.remove(render_data, do_unlink=True)
    """
    render_data = obj.data.copy()
    render_obj  = bpy.data.objects.new(obj.name + "_rnd_tmp", render_data)
    bpy.context.scene.collection.objects.link(render_obj)
    render_obj.matrix_world  = obj.matrix_world.copy()
    render_obj.hide_render   = False

    # The copy carries no modifiers, so any SUBSURF/MULTIRES on the original is
    # already dropped and n_faces is base topology. Only decimate when the base
    # mesh is truly dense: decimating moderate meshes visibly destroys/distorts
    # surface detail (thin limbs, folds), so leave anything <=100k untouched.
    n_faces = len(obj.data.polygons)
    if n_faces > _DECIMATE_MIN_FACES:
        mod       = render_obj.modifiers.new("_decimate", 'DECIMATE')
        mod.ratio = min(1.0, target / max(n_faces, 1))

    return render_obj, render_data


def _render_body_views(obj, margin=1.15, render_faces=_RENDER_TARGET_FACES):
    """
    Render front, side, and top orthographic views of *obj*.

    Front: camera at -Y looking +Y  (right=+X, up=+Z)
    Side:  camera at +X looking -X  (right=+Y, up=+Z)
    Top:   camera at +Z looking -Z  (right=+X, up=+Y)

    Returns (front_path, side_path, top_path, bounds, temp_dir).
    Caller owns temp_dir and must delete it.
    margin: ortho scale multiplier — use 1.30 for augmented renders to avoid clipping.
    """
    mn, mx, cen = _mesh_bbox_world(obj)
    span_x = mx.x - mn.x
    span_y = mx.y - mn.y
    span_z = mx.z - mn.z

    # Per-view scales: each camera frames only the dimensions it can see.
    # Side view previously used front_scale, compressing body depth into ~8px of the
    # 64px heatmap. Now side_scale uses actual depth so the model can learn from it.
    front_scale = max(span_x, span_z) * margin  # right=X, up=Z
    side_scale  = max(span_y, span_z) * margin  # right=Y, up=Z
    top_scale   = max(span_x, span_y) * margin  # right=X, up=Y

    scene   = bpy.context.scene
    display = scene.display

    orig_cam        = scene.camera
    orig_fp         = scene.render.filepath
    orig_rx         = scene.render.resolution_x
    orig_ry         = scene.render.resolution_y
    orig_pct        = scene.render.resolution_percentage
    orig_engine     = scene.render.engine
    orig_fmt        = scene.render.image_settings.file_format
    orig_bg_type    = display.shading.background_type
    orig_bg_color   = tuple(display.shading.background_color)
    orig_color_type = display.shading.color_type

    # Decimate a temporary copy so all renders have consistent visual density.
    # Bbox / landmark positions still come from the original obj.
    render_obj, render_data = _make_render_copy(obj, target=render_faces)

    hidden = []
    for o in scene.objects:
        if o is not render_obj and not o.hide_render:
            o.hide_render = True
            hidden.append(o)

    cam_data = bpy.data.cameras.new("_er_body_cam")
    cam_data.type = 'ORTHO'
    cam_obj = bpy.data.objects.new("_er_body_cam", cam_data)
    scene.collection.objects.link(cam_obj)

    temp_dir = tempfile.mkdtemp(prefix="er_body_")

    try:
        scene.render.engine                      = 'BLENDER_WORKBENCH'
        scene.render.resolution_x                = _BODY_IMG_SIZE
        scene.render.resolution_y                = _BODY_IMG_SIZE
        scene.render.resolution_percentage       = 100
        scene.render.image_settings.file_format  = 'PNG'
        scene.camera                             = cam_obj
        display.shading.background_type          = 'VIEWPORT'
        display.shading.background_color         = (0.03, 0.03, 0.03)
        display.shading.color_type               = 'OBJECT'

        # Front: camera at -Y looking +Y  (right=+X, up=+Z)
        cam_data.ortho_scale = front_scale
        dist_f = front_scale * 5.0 + 5.0
        cam_obj.location       = (cen.x, mn.y - dist_f, cen.z)
        cam_obj.rotation_euler = (math.pi / 2, 0.0, 0.0)
        front_path = os.path.join(temp_dir, "front.png")
        scene.render.filepath = front_path
        bpy.ops.render.render(write_still=True)

        # Side: camera at +X looking -X  (right=+Y, up=+Z)
        cam_data.ortho_scale = side_scale
        dist_s = side_scale * 5.0 + 5.0
        cam_obj.location       = (mx.x + dist_s, cen.y, cen.z)
        cam_obj.rotation_euler = (math.pi / 2, 0.0, math.pi / 2)
        side_path = os.path.join(temp_dir, "side.png")
        scene.render.filepath = side_path
        bpy.ops.render.render(write_still=True)

        # Top: camera at +Z looking -Z  (right=+X, up=+Y)
        cam_data.ortho_scale = top_scale
        dist_t = top_scale * 5.0 + 5.0
        cam_obj.location       = (cen.x, cen.y, mx.z + dist_t)
        cam_obj.rotation_euler = (0.0, 0.0, 0.0)
        top_path = os.path.join(temp_dir, "top.png")
        scene.render.filepath = top_path
        bpy.ops.render.render(write_still=True)

    finally:
        scene.camera                             = orig_cam
        scene.render.filepath                    = orig_fp
        scene.render.resolution_x                = orig_rx
        scene.render.resolution_y                = orig_ry
        scene.render.resolution_percentage       = orig_pct
        scene.render.engine                      = orig_engine
        scene.render.image_settings.file_format  = orig_fmt
        display.shading.background_type          = orig_bg_type
        display.shading.background_color         = orig_bg_color
        display.shading.color_type               = orig_color_type
        for o in hidden:
            o.hide_render = False
        bpy.data.objects.remove(cam_obj, do_unlink=True)
        bpy.data.cameras.remove(cam_data, do_unlink=True)
        bpy.data.objects.remove(render_obj, do_unlink=True)
        bpy.data.meshes.remove(render_data, do_unlink=True)

    bounds = {
        "cen_x": cen.x, "cen_y": cen.y, "cen_z": cen.z,
        "front_scale": front_scale,
        "side_scale":  side_scale,
        "top_scale":   top_scale,
        "ortho_scale": front_scale,  # backward compat for _snap_heels_back etc.
        "mn_y": mn.y, "mx_y": mx.y,
        "min": [mn.x, mn.y, mn.z],
        "max": [mx.x, mx.y, mx.z],
    }
    return front_path, side_path, top_path, bounds, temp_dir


# ── Neural face detector: 6-view rendering + general ortho projection ─────────
#
# Unlike the body (axis-aligned front/side/top), the face uses six fixed viewing
# directions including 3/4 angles, so projection is done with an explicit camera
# basis (right, up, view_dir) — the same math the hand orbit renderer uses.
# Character faces -Y; character LEFT is +X (matches the bilateral _L marker signs).

_FACE_MODEL_IMG_SIZE = 384
_FACE_SQ2 = 0.7071067811865476

# (name, view_dir = scene-center -> camera, up_hint)
_FACE_VIEWS = [
    ("front",      Vector(( 0.0,      -1.0,      0.0)), Vector((0.0,  0.0, 1.0))),
    ("q_left",     Vector(( _FACE_SQ2, -_FACE_SQ2, 0.0)), Vector((0.0, 0.0, 1.0))),
    ("q_right",    Vector((-_FACE_SQ2, -_FACE_SQ2, 0.0)), Vector((0.0, 0.0, 1.0))),
    ("side_left",  Vector(( 1.0,       0.0,      0.0)), Vector((0.0,  0.0, 1.0))),
    ("side_right", Vector((-1.0,       0.0,      0.0)), Vector((0.0,  0.0, 1.0))),
    ("top",        Vector(( 0.0,       0.0,      1.0)), Vector((0.0, -1.0, 0.0))),
]


def _face_cam_basis(view_dir, up_hint):
    """Right-handed camera basis: right x up = view_dir (camera +Z, away from scene)."""
    v     = view_dir.normalized()
    right = up_hint.cross(v).normalized()
    up    = v.cross(right).normalized()
    return right, up, v


def _face_world_to_px(p, cen, right, up, scale, S):
    """World point -> pixel in an ortho view with the given camera basis/scale.
    Exact inverse of the triangulation used at inference — training and inference
    MUST share this so pixel labels are byte-compatible with the model."""
    rel  = p - cen
    half = S * 0.5
    px = half + S * rel.dot(right) / scale
    py = half - S * rel.dot(up)    / scale
    return px, py


def _locate_head_bounds(mesh_obj):
    """Locate the HEAD on a (possibly full-body) mesh so the face views frame the
    head, not the whole body. Returns (center: Vector, span: float).

    The neck is the narrowest horizontal cross-section just below the head bulge;
    the head is everything above it. Falls back to the top slice of the mesh when no
    neck stands out (e.g. a head-only mesh). Uses geometry only (no markers), so
    training and inference frame identically.
    """
    mesh = mesh_obj.data
    nv   = len(mesh.vertices)
    if nv < 8:
        mn, mx, cen = _mesh_bbox_world(mesh_obj)
        return cen, max(mx.x - mn.x, mx.y - mn.y, mx.z - mn.z, 1e-4)

    # Fast vertex read (foreach_get) then transform to world with the 4x4 matrix.
    flat = np.empty(nv * 3, dtype=np.float64)
    mesh.vertices.foreach_get('co', flat)
    co = flat.reshape(nv, 3)
    mw = np.array(mesh_obj.matrix_world, dtype=np.float64)
    co = co @ mw[:3, :3].T + mw[:3, 3]

    z = co[:, 2]
    zmin, zmax = float(z.min()), float(z.max())
    H = zmax - zmin
    span_xy = float(max(co[:, 0].max() - co[:, 0].min(),
                        co[:, 1].max() - co[:, 1].min(), 1e-6))
    if H <= 1e-6:
        mn, mx, cen = _mesh_bbox_world(mesh_obj)
        return cen, max(mx.x - mn.x, mx.y - mn.y, mx.z - mn.z, 1e-4)

    # PREFERRED: frame the head from the placed body HEAD marker (already verified
    # by the user via body detection) — far more reliable than guessing the neck
    # from geometry. The HEAD marker sits at the skull base, so extend below it to
    # catch the chin, and never dip below the NECK marker.
    _head_m = bpy.data.objects.get("MARKER_HEAD")
    if _head_m is not None:
        hz    = float(_head_m.matrix_world.translation.z)
        crown = zmax
        up    = max(crown - hz, 1e-3)          # skull base -> crown
        # Head WIDTH & centre-XY from the skull-base..crown band ONLY. This is the
        # key: measuring verts below the head (shoulders/arms in a T-pose) would
        # blow the frame up to arm span and show the whole upper body.
        band = co[z >= hz]
        if len(band) >= 4:
            bmn = band.min(axis=0); bmx = band.max(axis=0)
            cx = float((bmn[0] + bmx[0]) * 0.5)
            cy = float((bmn[1] + bmx[1]) * 0.5)
            head_w = float(max(bmx[0] - bmn[0], bmx[1] - bmn[1]))
        else:
            cx = float(co[:, 0].mean()); cy = float(co[:, 1].mean()); head_w = up
        # Bottom of the frame: drop below the marker to include the chin, but never
        # below the NECK marker — keeps the shoulders out, even on short necks.
        base_z = hz - up * 0.5
        _neck_m = bpy.data.objects.get("MARKER_NECK")
        if _neck_m is not None:
            base_z = max(base_z, float(_neck_m.matrix_world.translation.z))
        base_z = max(base_z, zmin)
        cen  = Vector((cx, cy, (base_z + crown) * 0.5))
        span = float(max(head_w, crown - base_z, 1e-4))
        return cen, span

    # FALLBACK (no HEAD marker placed): guess the head from the width profile over
    # the FULL height (need shoulders + head context).
    n     = 60
    edges = np.linspace(zmin, zmax, n + 1)
    zc    = 0.5 * (edges[:-1] + edges[1:])
    width = np.full(n, np.nan)
    for i in range(n):
        m = (z >= edges[i]) & (z <= edges[i + 1])
        if int(m.sum()) >= 3:
            width[i] = max(co[m, 0].max() - co[m, 0].min(),
                           co[m, 1].max() - co[m, 1].min())
    valid = ~np.isnan(width)

    neck_z = None
    if int(valid.sum()) >= 8:
        width = np.interp(zc, zc[valid], width[valid])          # fill empty slabs
        ws    = np.convolve(width, np.ones(3) / 3.0, mode='same')
        win     = 0.30 * H
        R_ABOVE = 1.08   # head only SLIGHTLY wider than a thick, stylized neck
        R_BELOW = 1.28   # shoulders are reliably much wider than the neck
        # A neck is a width-minimum with a wider HEAD above AND wider SHOULDERS
        # below — that dual check ignores hair/ear/horn bumps and a shallow waist.
        # Among all such minima in the UPPER 55% (skips the waist), pick the
        # NARROWEST: the true neck is narrower than a hat/head junction.
        cands = []
        for i in range(2, n - 2):
            if not (ws[i] <= ws[i - 1] and ws[i] <= ws[i + 1]):
                continue
            if zc[i] > zmax - 0.06 * H or zc[i] < zmin + 0.45 * H:
                continue
            above = ws[(zc > zc[i]) & (zc <= zc[i] + win)]
            below = ws[(zc < zc[i]) & (zc >= zc[i] - win)]
            if (above.size and below.size
                    and float(above.max()) > R_ABOVE * ws[i]
                    and float(below.max()) > R_BELOW * ws[i]):
                cands.append(i)
        if cands:
            neck_z = float(zc[min(cands, key=lambda k: ws[k])])

    if neck_z is None:
        # No detectable neck. Decide head-only vs full-body from the HEAD-REGION
        # width (NOT arm-span — T-pose arms inflate that and made a body look
        # "not tall"). A round head-like mesh has a wide top relative to its
        # height; a body's top (the head) is small vs total height.
        top_mask = z >= zmax - 0.30 * H
        if int(top_mask.sum()) >= 3:
            tc = co[top_mask]
            top_w = float(max(tc[:, 0].max() - tc[:, 0].min(),
                              tc[:, 1].max() - tc[:, 1].min()))
        else:
            top_w = span_xy
        neck_z = zmin if top_w > 0.55 * H else (zmax - H * 0.28)

    head_mask = z >= neck_z
    if int(head_mask.sum()) < 4:
        head_mask = z >= (zmax - H * 0.20)
    hc  = co[head_mask]
    hmn = hc.min(axis=0); hmx = hc.max(axis=0)
    cen = Vector(((hmn + hmx) * 0.5).tolist())
    span = float(max(hmx[0] - hmn[0], hmx[1] - hmn[1], hmx[2] - hmn[2], 1e-4))
    return cen, span


def _render_face_views(mesh_obj, margin=1.30):
    """Render the six face views of *mesh_obj*, framed on the HEAD (see
    _locate_head_bounds) so a full-body mesh still gives a face-filling render.

    Returns (cen, views, S) where views is a list of dicts:
        {"name","path","right":[3],"up":[3],"view_dir":[3],"scale"}
    Caller owns and must delete the temp dir (views[0]["temp_dir"]).
    """
    cen, span = _locate_head_bounds(mesh_obj)
    scale = max(span * margin, 1e-4)          # uniform (head is ~round) across views
    S     = _FACE_MODEL_IMG_SIZE

    scene   = bpy.context.scene
    display = scene.display

    orig_cam        = scene.camera
    orig_fp         = scene.render.filepath
    orig_rx         = scene.render.resolution_x
    orig_ry         = scene.render.resolution_y
    orig_pct        = scene.render.resolution_percentage
    orig_engine     = scene.render.engine
    orig_fmt        = scene.render.image_settings.file_format
    orig_bg_type    = display.shading.background_type
    orig_bg_color   = tuple(display.shading.background_color)
    orig_color_type = display.shading.color_type
    orig_light      = display.shading.light
    orig_studio     = display.shading.studio_light
    orig_single     = tuple(display.shading.single_color)
    orig_shadows    = display.shading.show_shadows
    orig_cavity     = display.shading.show_cavity
    orig_cav_type   = display.shading.cavity_type
    orig_cav_ridge  = display.shading.cavity_ridge_factor
    orig_cav_valley = display.shading.cavity_valley_factor

    # Render the FULL-RES mesh — NOT decimated. Faces need smooth geometry; the
    # ~8k body decimation makes the face look faceted/bad. (The head is a small
    # part of the mesh, so full-res here is cheap.)
    orig_hide_render = mesh_obj.hide_render
    mesh_obj.hide_render = False
    # Render the base cage — subdivision only slows the render and the CNN reads
    # landmarks from the matcap shading, not the extra tessellation.
    _restore_subdiv = _disable_subdiv_modifiers(mesh_obj)
    hidden = []
    for o in scene.objects:
        if o is not mesh_obj and not o.hide_render:
            o.hide_render = True
            hidden.append(o)

    cam_data = bpy.data.cameras.new("_er_face_cam")
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = scale
    cam_obj = bpy.data.objects.new("_er_face_cam", cam_data)
    scene.collection.objects.link(cam_obj)

    temp_dir = tempfile.mkdtemp(prefix="er_face_")
    dist = scale * 5.0 + 5.0
    views = []
    try:
        scene.render.engine                     = 'BLENDER_WORKBENCH'
        scene.render.resolution_x               = S
        scene.render.resolution_y               = S
        scene.render.resolution_percentage      = 100
        scene.render.image_settings.file_format = 'PNG'
        scene.camera                            = cam_obj
        display.shading.background_type         = 'VIEWPORT'
        display.shading.background_color        = (0.03, 0.03, 0.03)
        # CLAY MATCAP render for smooth stylized faces. Flat grey Lambert gave
        # the CNN almost no shading gradient on the forehead/jaw/cheeks, so
        # those landmarks had nothing to localize. A clay matcap shades purely
        # from surface NORMALS — every curve of the face gets a light/dark
        # gradient like a sculpt — and SCREEN cavity + shadows add crease/edge/
        # contour darkening on top. A single neutral base colour keeps every
        # character's render consistent (their material colours can't tint it).
        # Both training (generator) and inference call this fn, so they match.
        display.shading.color_type           = 'SINGLE'
        display.shading.single_color         = (0.8, 0.8, 0.8)
        display.shading.light                = 'MATCAP'
        display.shading.studio_light         = 'clay_brown.exr'
        display.shading.show_shadows         = True
        display.shading.show_cavity          = True
        display.shading.cavity_type          = 'SCREEN'
        display.shading.cavity_ridge_factor  = 2.0
        display.shading.cavity_valley_factor = 2.0

        for name, view_dir, up_hint in _FACE_VIEWS:
            right, up, vdir = _face_cam_basis(view_dir, up_hint)
            pos = cen + vdir * dist
            cam_obj.matrix_world = Matrix([
                [right.x, up.x, vdir.x, pos.x],
                [right.y, up.y, vdir.y, pos.y],
                [right.z, up.z, vdir.z, pos.z],
                [0.0,     0.0,  0.0,    1.0  ],
            ])
            path = os.path.join(temp_dir, f"{name}.png")
            scene.render.filepath = path
            bpy.ops.render.render(write_still=True)
            views.append({
                "name": name, "path": path,
                "right": [right.x, right.y, right.z],
                "up":    [up.x, up.y, up.z],
                "view_dir": [vdir.x, vdir.y, vdir.z],
                "scale": scale,
            })
    finally:
        scene.camera                            = orig_cam
        scene.render.filepath                   = orig_fp
        scene.render.resolution_x               = orig_rx
        scene.render.resolution_y               = orig_ry
        scene.render.resolution_percentage      = orig_pct
        scene.render.engine                     = orig_engine
        scene.render.image_settings.file_format = orig_fmt
        display.shading.background_type         = orig_bg_type
        display.shading.background_color        = orig_bg_color
        display.shading.color_type              = orig_color_type
        display.shading.light                   = orig_light
        display.shading.studio_light            = orig_studio
        display.shading.single_color            = orig_single
        display.shading.show_shadows            = orig_shadows
        display.shading.show_cavity             = orig_cavity
        display.shading.cavity_type             = orig_cav_type
        display.shading.cavity_ridge_factor     = orig_cav_ridge
        display.shading.cavity_valley_factor    = orig_cav_valley
        mesh_obj.hide_render = orig_hide_render
        _restore_subdiv()
        for o in hidden:
            o.hide_render = False
        bpy.data.objects.remove(cam_obj, do_unlink=True)
        bpy.data.cameras.remove(cam_data, do_unlink=True)

    if views:
        views[0]["temp_dir"] = temp_dir
    return cen, views, S


# ── Neural face detector: inference (heatmap peaks -> ray triangulation) ──────

_FACE_MODEL_FILENAME = os.path.join("models", "face_pose.rmodel")
_FACE_HMAP_SIZE      = 96
_FACE_ONNX_SESSIONS  = {}
_DEBUG_FACE          = True   # dump inference renders + peak stats to the console


def _face_model_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), _FACE_MODEL_FILENAME)


def is_face_onnx_available():
    try:
        import onnxruntime  # noqa: F401
        import PIL          # noqa: F401
    except ImportError:
        return False
    return os.path.isfile(_face_model_path())


def _face_peak(ch):
    """50%-threshold weighted centroid in heatmap space -> ((hx,hy), peak_value)."""
    mx = float(ch.max())
    if mx < 1e-6:
        return None, 0.0
    w = ch * (ch >= mx * 0.5)
    mass = float(w.sum())
    if mass < 1e-6:
        return None, mx
    yy, xx = np.mgrid[0:ch.shape[0], 0:ch.shape[1]]
    return (float((xx * w).sum() / mass), float((yy * w).sum() / mass)), mx


def _triangulate_rays(rays):
    """Least-squares 3D point nearest to a set of (origin, unit-dir) ortho rays."""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for Q, d in rays:
        Pm = np.eye(3) - np.outer(d, d)   # project onto plane perpendicular to the ray
        A += Pm; b += Pm @ Q
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return sol


def _view_occludes(bvh, p, vdir, probe=0.5, margin=0.008):
    """True if point p is HIDDEN from the ortho camera that looks along -vdir
    (vdir points scene->camera). Casts from `probe` metres in front of p back toward
    it; if the mesh is hit before reaching p, something occludes it from that view.
    The model predicts confident-but-wrong peaks for occluded landmarks, so we use
    this to drop those views."""
    pv  = Vector((float(p[0]), float(p[1]), float(p[2])))
    vd  = Vector((float(vdir[0]), float(vdir[1]), float(vdir[2])))
    org = pv + vd * probe
    hit, _, _, dist = bvh.ray_cast(org, -vd, probe * 2.0)
    if hit is None:
        return False
    return dist < probe - margin


def _triangulate_occluded(rays, bvh, iters=3):
    """Triangulate using only the views from which the landmark is actually visible.
    Iterates: estimate -> drop views that occlude the estimate -> re-triangulate. The
    accurate visible-view rays pull the estimate toward the true position, so even a
    rough start converges (the collapsed occluded-view rays get rejected)."""
    if len(rays) < 2:
        return None
    p = _triangulate_rays(rays)
    for _ in range(iters):
        vis = [(Q, d) for (Q, d) in rays if not _view_occludes(bvh, p, d)]
        if len(vis) < 2:
            break
        np_new = _triangulate_rays(vis)
        if np.linalg.norm(np_new - p) < 1e-4:
            p = np_new
            break
        p = np_new
    return p


def _detect_face_onnx(cen, views, S, bvh, act=0.22):
    """Run the face pose model on the six rendered views. Each landmark is
    triangulated from the views where its heatmap activates above *act*, then
    snapped to the mesh surface. Returns {marker_name: Vector} for located landmarks."""
    import onnxruntime as ort
    from PIL import Image as _PIL
    from . import model_crypto

    path = _face_model_path()
    if path not in _FACE_ONNX_SESSIONS:
        _FACE_ONNX_SESSIONS[path] = ort.InferenceSession(
            model_crypto.load_model_bytes(path), providers=['CPUExecutionProvider'])
    sess = _FACE_ONNX_SESSIONS[path]

    def _load(p):
        img = _PIL.open(p).convert("RGB").resize((_FACE_MODEL_IMG_SIZE, _FACE_MODEL_IMG_SIZE))
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.ascontiguousarray(arr.transpose(2, 0, 1)[None])

    feed      = {v["name"]: _load(v["path"]) for v in views}
    out_names = [o.name for o in sess.get_outputs()]
    outs      = sess.run(None, feed)
    hm        = {nm[3:]: arr[0] for nm, arr in zip(out_names, outs)}   # "hm_front" -> [N,h,h]

    # DEBUG: dump the 6 inference renders so they can be compared to the training
    # renders (a framing/appearance mismatch is the usual cause of clustered markers).
    if _DEBUG_FACE:
        try:
            _dbg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_face_infer")
            os.makedirs(_dbg, exist_ok=True)
            for v in views:
                shutil.copy2(v["path"], os.path.join(_dbg, f"{v['name']}.png"))
            dbg(f"[face-infer] cen={tuple(round(float(c),3) for c in (cen.x,cen.y,cen.z))} "
                  f"scale={round(float(views[0]['scale']),4)}  renders -> {_dbg}")
        except Exception as _e:
            dbg("[face-infer] render dump failed:", _e)

    Hs   = float(_FACE_HMAP_SIZE)
    cenv = np.array([cen.x, cen.y, cen.z], dtype=np.float64)

    # Match the landmark list to the model's output channel count, so this works
    # with BOTH the 30-core model and the full-marker model (channel order is the
    # same constants ordering the generator + trainer use).
    _n_ch = int(next(iter(hm.values())).shape[0])
    if _n_ch == len(FULL_FACE_LANDMARKS):
        _lm_list = FULL_FACE_LANDMARKS
    elif _n_ch == len(CORE_FACE_LANDMARKS):
        _lm_list = CORE_FACE_LANDMARKS
    else:
        _lm_list = (FULL_FACE_LANDMARKS if _n_ch >= len(FULL_FACE_LANDMARKS)
                    else CORE_FACE_LANDMARKS)[:_n_ch]

    result = {}
    for i, name in enumerate(_lm_list):
        cand = []   # (peak_value, (ray_origin, ray_dir))
        _pk  = []
        for v in views:
            ch = hm[v["name"]][i]
            peak, mx = _face_peak(ch)
            _pk.append(round(mx, 2))
            if peak is None or mx < act:
                continue
            hx, hy = peak
            right = np.array(v["right"],    dtype=np.float64)
            up    = np.array(v["up"],       dtype=np.float64)
            vdir  = np.array(v["view_dir"], dtype=np.float64)
            scale = float(v["scale"])
            a =  (hx / Hs - 0.5) * scale
            b = -(hy / Hs - 0.5) * scale
            Q = cenv + a * right + b * up
            cand.append((mx, (Q, vdir)))
        if _DEBUG_FACE:
            dbg(f"  [face] {name:24s} n={len(cand)} peaks={_pk}")
        if not cand:
            continue
        # Relative-confidence gate: keep only the views whose peak is near this
        # landmark's strongest peak. A landmark fires ~0.85-0.95 in views where it's
        # visible but only ~0.5-0.75 where it's occluded (far side of the head); those
        # weaker, wrong peaks otherwise drag lateral landmarks toward centre.
        mxmax = max(c[0] for c in cand)
        rays  = [r for mx, r in cand if mx >= 0.85 * mxmax]
        if len(rays) < 2:
            rays = [r for _, r in cand]
        if len(rays) < 2:
            continue
        p3 = _triangulate_occluded(rays, bvh)
        loc, _n, _idx, _d = bvh.find_nearest(Vector(p3))
        result[name] = Vector(loc) if loc is not None else Vector(p3)
    return result

