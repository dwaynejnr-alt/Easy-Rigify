"""
Hybrid LVT finger detection for Blender.
Primary:    models/hand_pose.rmodel  -- 20-joint regression, multi-view triangulation.
Fallback:   models/hand_tips.rmodel  -- 5-tip heatmaps + geometric phalange chain.
Last resort: _place_fingers_geometric.
"""

import bpy
import math
import os
import numpy as np
from mathutils import Vector

from .ai_detect import (_render_hand_orbit, _orbit_cam_basis, _ORBIT_N_VIEWS,
                        _HAND_IMG_SIZE, _HAND_SCALE, _hand_scale,
                        _place_fingers_geometric,
                        _build_bvh, _mesh_hand_spread)


# -- Constants -----------------------------------------------------------------

_LANDMARK_MODEL_FILENAME = os.path.join("models", "hand_pose.rmodel")
_TIP_MODEL_FILENAME      = os.path.join("models", "hand_tips.rmodel")
_MODEL_INPUT             = 256
_HMAP_OUTPUT             = 128   # must match TipHeatmapDataset.HMAP_SIZE in train_tip_heatmaps.py
_CONF_THRESHOLD          = 0.10
_EDGE_MARGIN             = 4     # px in 256-space; must match SaveHandData margin=4
_MAX_TIP_DIST            = 0.22  # metres; real finger tips are 10-20cm from wrist
_TIP_MIN_SEP_PX          = 12.0  # px in 256-space; two tips closer than this in a
                                 # single view = channel bleed, drop the weaker one
                                 # (real adjacent tips sit ~40px apart in the crop)

_FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

# ONNX sessions are expensive to construct (model load from disk).
# Cache them at module level so each model is loaded only once per Blender session.
_onnx_sessions: dict = {}

# LVT-stage orbit renders cached per side for the landmark stage (popped on
# use). Saves 8 renders per side when the two stages' framings agree.
_ORBIT_REUSE: dict = {}

# Set True to save per-view landmark overlay PNGs to C:\Temp for visual inspection.
# Flip back to False once you're satisfied with prediction quality.
_DEBUG_LM = False
_DEBUG_TIP = False

# Short display labels for the debug overlay (sorted landmark index -> label)
_LM_LABELS = {
    "THUMB_CMC": "Tc", "THUMB_MCP": "T1", "THUMB_IP":  "T2", "THUMB_TIP":  "Tt",
    "INDEX_MCP": "I1", "INDEX_PIP": "I2", "INDEX_DIP": "I3", "INDEX_TIP":  "It",
    "MIDDLE_MCP":"M1", "MIDDLE_PIP":"M2", "MIDDLE_DIP":"M3", "MIDDLE_TIP": "Mt",
    "RING_MCP":  "R1", "RING_PIP":  "R2", "RING_DIP":  "R3", "RING_TIP":   "Rt",
    "PINKY_MCP": "P1", "PINKY_PIP": "P2", "PINKY_DIP": "P3", "PINKY_TIP":  "Pt",
}

# 20 joint names sorted alphabetically  -- must match train_hand_detector.py
_LANDMARK_NAMES = sorted([
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP",  "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP","MIDDLE_PIP","MIDDLE_DIP","MIDDLE_TIP",
    "RING_MCP",  "RING_PIP",  "RING_DIP",  "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
])

_MP_TO_ARP = {
    "THUMB_CMC":  "THUMB_1",         "THUMB_MCP":  "THUMB_2",
    "THUMB_IP":   "THUMB_3",         "THUMB_TIP":  "THUMB_TIP",
    "INDEX_MCP":  "FINGER_INDEX_1",  "INDEX_PIP":  "FINGER_INDEX_2",
    "INDEX_DIP":  "FINGER_INDEX_3",  "INDEX_TIP":  "FINGER_INDEX_TIP",
    "MIDDLE_MCP": "FINGER_MIDDLE_1", "MIDDLE_PIP": "FINGER_MIDDLE_2",
    "MIDDLE_DIP": "FINGER_MIDDLE_3", "MIDDLE_TIP": "FINGER_MIDDLE_TIP",
    "RING_MCP":   "FINGER_RING_1",   "RING_PIP":   "FINGER_RING_2",
    "RING_DIP":   "FINGER_RING_3",   "RING_TIP":   "FINGER_RING_TIP",
    "PINKY_MCP":  "FINGER_PINKY_1",  "PINKY_PIP":  "FINGER_PINKY_2",
    "PINKY_DIP":  "FINGER_PINKY_3",  "PINKY_TIP":  "FINGER_PINKY_TIP",
}

_FINGER_TO_ARP = {
    "thumb":  {"tip": "THUMB_TIP",         "phal3": "THUMB_3",
               "phal2": "THUMB_2",         "phal1": "THUMB_1"},
    "index":  {"tip": "FINGER_INDEX_TIP",  "phal3": "FINGER_INDEX_3",
               "phal2": "FINGER_INDEX_2",  "phal1": "FINGER_INDEX_1"},
    "middle": {"tip": "FINGER_MIDDLE_TIP", "phal3": "FINGER_MIDDLE_3",
               "phal2": "FINGER_MIDDLE_2", "phal1": "FINGER_MIDDLE_1"},
    "ring":   {"tip": "FINGER_RING_TIP",   "phal3": "FINGER_RING_3",
               "phal2": "FINGER_RING_2",   "phal1": "FINGER_RING_1"},
    "pinky":  {"tip": "FINGER_PINKY_TIP",  "phal3": "FINGER_PINKY_3",
               "phal2": "FINGER_PINKY_2",  "phal1": "FINGER_PINKY_1"},
}


# -- Model paths ---------------------------------------------------------------

def _landmark_model_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), _LANDMARK_MODEL_FILENAME)

def _tip_model_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), _TIP_MODEL_FILENAME)

def is_landmark_available():
    try:
        import onnxruntime  # noqa: F401
        return os.path.isfile(_landmark_model_path())
    except ImportError:
        return False


def is_lvt_available():
    try:
        import onnxruntime  # noqa: F401
        return os.path.isfile(_tip_model_path())
    except ImportError:
        return False


# -- Shared image loader -------------------------------------------------------

def _load_orbit_image(image_path):
    """Return [1, 3, 256, 256] float32 array from an orbit PNG."""
    try:
        from PIL import Image as _PIL
        img = _PIL.open(image_path).convert("RGB")
        if img.size != (_MODEL_INPUT, _MODEL_INPUT):
            img = img.resize((_MODEL_INPUT, _MODEL_INPUT), _PIL.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
    except ImportError:
        img_bpy = bpy.data.images.load(image_path, check_existing=False)
        try:
            w, h = img_bpy.size
            arr  = np.array(img_bpy.pixels[:], dtype=np.float32).reshape((h, w, 4))
            arr  = arr[::-1, :, :3]
            if h != _MODEL_INPUT or w != _MODEL_INPUT:
                from scipy.ndimage import zoom as _zoom
                arr = _zoom(arr, (_MODEL_INPUT / h, _MODEL_INPUT / w, 1), order=1)
        finally:
            bpy.data.images.remove(img_bpy)
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None])  # [1, 3, 256, 256]


# -- Step 1a: 20-landmark detection with TTA -----------------------------------

def _detect_landmarks_onnx(image_path, cam_feat):
    """
    Run HandLandmarkNet with 3-variant TTA (base, h-flip, brightness).
    cam_feat: 7-dim [right_vec(3), up_vec(3), side] where side=0.0 for L, 1.0 for R.
    Returns {landmark_name: (px, py, conf)} in 256-pixel coords.
    conf is TTA consistency in [0, 1]: low std across variants -> high confidence.
    """
    import onnxruntime as ort
    from PIL import Image as _PIL

    _path = _landmark_model_path()
    if _path not in _onnx_sessions:
        from . import model_crypto
        _onnx_sessions[_path] = ort.InferenceSession(
            model_crypto.load_model_bytes(_path), providers=['CPUExecutionProvider'])
    sess = _onnx_sessions[_path]
    img  = _PIL.open(image_path).convert("RGB").resize((_MODEL_INPUT, _MODEL_INPUT))
    arr  = np.array(img, dtype=np.float32) / 255.0

    def _chw(a):
        return np.ascontiguousarray(a.transpose(2, 0, 1)[None])

    rv     = np.array(cam_feat[:3], dtype=np.float32)
    uv     = np.array(cam_feat[3:6], dtype=np.float32)
    side_f = float(cam_feat[6])
    # sin/cos of orbit angle (indices 7-8); fall back to zeros for old 7-float callers
    sin_t  = float(cam_feat[7]) if len(cam_feat) > 7 else 0.0
    cos_t  = float(cam_feat[8]) if len(cam_feat) > 8 else 1.0

    n_cam  = sess.get_inputs()[1].shape[1]   # 7 or 9 depending on which ONNX is loaded

    def _run(img_t, rv_in, side_in, sin_in, cos_in):
        if n_cam == 9:
            cam = np.array([*rv_in, *uv, side_in, sin_in, cos_in], dtype=np.float32)
        else:
            cam = np.array([*rv_in, *uv, side_in], dtype=np.float32)
        return sess.run(None, {"images": img_t, "cam_feat": cam.reshape(1, n_cam)})[0][0]

    # Base pass
    p_base = _run(_chw(arr), rv, side_f, sin_t, cos_t).reshape(20, 2)

    # Horizontal flip: negate right_vec and sin(theta) (mirror view angle)
    rv_f     = -rv.copy()
    p_flip   = _run(_chw(np.flip(arr, axis=1).copy()), rv_f, 1.0 - side_f, -sin_t, cos_t).reshape(20, 2)
    p_flip[:, 0] = 1.0 - p_flip[:, 0]

    # Brightness jitter
    p_bright = _run(_chw(np.clip(arr * 1.15, 0.0, 1.0)), rv, side_f, sin_t, cos_t).reshape(20, 2)

    preds = np.stack([p_base, p_flip, p_bright], axis=0)  # [3, 20, 2]
    p_med = np.median(preds, axis=0)                       # [20, 2]
    conf  = np.clip(1.0 - preds.std(axis=0).mean(axis=1) * 8.0, 0.0, 1.0)  # [20]

    lo = _EDGE_MARGIN / _HAND_IMG_SIZE
    hi = 1.0 - lo

    joints = {}
    for i, lm_name in enumerate(_LANDMARK_NAMES):
        nx, ny = float(p_med[i, 0]), float(p_med[i, 1])
        if nx < lo or nx > hi or ny < lo or ny > hi:
            continue
        joints[lm_name] = (nx * _HAND_IMG_SIZE, ny * _HAND_IMG_SIZE, float(conf[i]))

    if _DEBUG_LM:
        try:
            from PIL import Image as _PILD, ImageDraw as _Draw
            dbg  = _PILD.open(image_path).convert("RGB")
            draw = _Draw.Draw(dbg)
            for lm_name, (px, py, c) in joints.items():
                r     = 5
                color = (0, 220, 0) if c >= 0.7 else (255, 200, 0) if c >= 0.4 else (220, 0, 0)
                draw.ellipse([px - r, py - r, px + r, py + r], fill=color)
                draw.text((px + r + 1, py - r), _LM_LABELS.get(lm_name, lm_name[:3]),
                          fill=(255, 255, 255))
            stem     = os.path.splitext(os.path.basename(image_path))[0]
            dbg_path = os.path.join(r"C:\Temp", f"{stem}_lm.png")
            dbg.save(dbg_path)
            print(f"  [debug lm] {dbg_path}  ({len(joints)}/20 joints)")
        except Exception as _e:
            print(f"  [debug lm] failed: {_e}")

    return joints


# -- Step 1b: 5-tip heatmap detection -----------------------------------------

def _soft_argmax_np(hm, temperature=8.0):
    """Weighted spatial average matching the training soft-argmax loss."""
    H, W = hm.shape
    yy, xx = np.mgrid[0:H, 0:W]
    flat = hm.flatten()
    flat = flat - flat.max()          # numerical stability
    w = np.exp(flat * temperature)
    w /= w.sum()
    px = float((xx.flatten() * w).sum())
    py = float((yy.flatten() * w).sum())
    return px, py


def _detect_tips_onnx(image_path, cam_feat=None):
    """
    Run TipHeatmapNet on one orbit view.
    cam_feat: 8-dim [right_vec(3), up_vec(3), side_L, side_R] one-hot side encoding.
    Returns {finger_name: (px, py)} in 256-pixel coords, edge/low-conf discarded.
    """
    import onnxruntime as ort

    rgb  = _load_orbit_image(image_path)
    _path = _tip_model_path()
    if _path not in _onnx_sessions:
        from . import model_crypto
        _onnx_sessions[_path] = ort.InferenceSession(
            model_crypto.load_model_bytes(_path), providers=['CPUExecutionProvider'])
    sess = _onnx_sessions[_path]

    feed = {sess.get_inputs()[0].name: rgb}
    if len(sess.get_inputs()) > 1:
        # The caller passes an 8-dim ONE-HOT side encoding [right(3), up(3), side_L, side_R].
        # Adapt it to whatever the LOADED model expects (like _detect_landmarks_onnx does):
        #   n_cam==7 -> current model, SCALAR side [right(3), up(3), side] (L=0.0, R=1.0)
        #   n_cam==8 -> old one-hot model (pass through)
        # Hardcoding 8 here fed a (1,8) tensor to the new 7-dim model -> onnxruntime
        # shape mismatch -> silent tip-detection failure.
        try:
            n_cam = int(sess.get_inputs()[1].shape[1])
        except (TypeError, ValueError):
            n_cam = 7
        src = list(cam_feat) if cam_feat is not None else [0.0] * 8
        if n_cam == 7:
            # one-hot -> scalar: side = side_R element (L=[1,0]->0.0, R=[0,1]->1.0)
            cf_list = (src[:6] + [src[7]]) if len(src) >= 8 else (src + [0.0] * 7)[:7]
        elif n_cam == 8 and len(src) == 7:
            # scalar -> one-hot (back-compat if a 7-dim caller ever appears)
            cf_list = src[:6] + [1.0 - src[6], src[6]]
        else:
            cf_list = (src + [0.0] * n_cam)[:n_cam]
        feed[sess.get_inputs()[1].name] = np.array(cf_list, dtype=np.float32).reshape(1, n_cam)

    raw_outputs  = sess.run(None, feed)
    output_names = [o.name for o in sess.get_outputs()]
    out_map      = dict(zip(output_names, raw_outputs))
    if "heatmaps" not in out_map:
        print(f"  [LVT] ONNX missing 'heatmaps'. Available: {output_names}")
        return {}
    heatmaps = out_map["heatmaps"][0]  # [5, _HMAP_OUTPUT, _HMAP_OUTPUT]

    img_scale = _HAND_IMG_SIZE / _HMAP_OUTPUT
    hi   = _HAND_IMG_SIZE - _EDGE_MARGIN
    tips = {}

    # Collect raw peaks before threshold for debug overlay
    raw_peaks = []  # [(fname, px64, py64, conf)]

    for f_idx, fname in enumerate(_FINGER_NAMES):
        hm        = heatmaps[f_idx]

        # Hard-argmax SEED + windowed centre-of-mass refine.
        # NOT a GLOBAL soft-argmax over the whole map: soft-argmax is a full-map
        # weighted average, so on a DIFFUSE sigmoid heatmap (peak only ~0.4, not a
        # crisp ~0.9) the 16k background pixels out-mass the peak and the average
        # drifts toward image centre -> 40-110 px error, tips land nowhere, and the
        # LVT gate rejects every view (see feedback_softargmax_logits: soft-argmax
        # belongs on LOGITS, but the shipped ONNX emits sigmoid). Hard-argmax finds
        # the true peak (background can't pull it), and the local COM below recovers
        # sub-pixel accuracy WITHIN the peak. Robust to both crisp and diffuse peaks.
        iy0, ix0  = np.unravel_index(int(hm.argmax()), hm.shape)
        px, py    = float(ix0), float(iy0)
        peak_conf = float(hm.max())

        raw_peaks.append((fname, int(round(px)), int(round(py)), peak_conf))

        if peak_conf < _CONF_THRESHOLD:
            continue

        # Fine-tune: local patch centre-of-mass around the hard-argmax peak.
        r = 4
        ix, iy = int(round(px)), int(round(py))
        if r <= iy < _HMAP_OUTPUT - r and r <= ix < _HMAP_OUTPUT - r:
            patch  = hm[iy - r:iy + r + 1, ix - r:ix + r + 1]
            ys, xs = np.mgrid[iy - r:iy + r + 1, ix - r:ix + r + 1]
            total  = patch.sum()
            if total > 0:
                py = float((ys * patch).sum() / total)
                px = float((xs * patch).sum() / total)

        px_256 = float(px) * img_scale
        py_256 = float(py) * img_scale

        if px_256 < _EDGE_MARGIN or px_256 > hi or py_256 < _EDGE_MARGIN or py_256 > hi:
            continue

        tips[fname] = (px_256, py_256, peak_conf)

    # -- Per-view channel-collision gate ----------------------------------------
    # Each finger is an INDEPENDENT heatmap channel with no separation constraint,
    # so an unconfident channel drifts its soft-argmax onto whatever bright region
    # a neighbour already claimed. Two channels then return near-identical pixels,
    # and triangulating identical 2D points across views converges the tips to one
    # 3D blob -> stable_scale collapses (see _mdst guard downstream). Catch it at
    # the source: when two tips in THIS view sit within _TIP_MIN_SEP_PX, the lower-
    # confidence one is a bled duplicate, not a real tip -> drop it for this view.
    # Triangulation tolerates a missing detection per view; it cannot recover two
    # coincident ones. This leaves genuinely-separated tips untouched.
    if len(tips) > 1:
        names = list(tips.keys())
        drop  = set()
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                na, nb = names[a], names[b]
                if na in drop or nb in drop:
                    continue
                pa, pb = tips[na], tips[nb]
                d = ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5
                if d < _TIP_MIN_SEP_PX:
                    weaker = na if pa[2] < pb[2] else nb
                    drop.add(weaker)
        for n in drop:
            del tips[n]
    # Strip the confidence element the gate needed -> {finger: (px, py)}.
    tips = {f: (v[0], v[1]) for f, v in tips.items()}

    if _DEBUG_TIP:
        print(f"  [tip peaks] " + "  ".join(f"{n}:{c:.3f}@({x},{y})" for n, x, y, c in raw_peaks))
        try:
            from PIL import Image as _PILD, ImageDraw as _Draw
            dbg  = _PILD.open(image_path).convert("RGB")
            draw = _Draw.Draw(dbg)
            for fname, px64, py64, conf in raw_peaks:
                px_d = px64 * img_scale
                py_d = py64 * img_scale
                r    = 5
                color = (0, 220, 0) if conf >= _CONF_THRESHOLD else (220, 80, 0)
                draw.ellipse([px_d - r, py_d - r, px_d + r, py_d + r], fill=color)
                draw.text((px_d + r + 1, py_d - r), f"{fname[:3]}:{conf:.2f}", fill=(255, 255, 255))
            stem     = os.path.splitext(os.path.basename(image_path))[0]
            dbg_path = os.path.join(r"C:\Temp", f"{stem}_tip.png")
            dbg.save(dbg_path)
        except Exception as _e:
            print(f"  [debug tip] failed: {_e}")

    return tips


# -- Step 2: RANSAC multi-view triangulation -----------------------------------

def _triangulate_ransac(per_view_joints, per_view_cameras, hw, hand_scale=None,
                        inlier_thresh=0.012, min_conf=0.35, max_dist=_MAX_TIP_DIST):
    """
    RANSAC triangulation for named 2D joints across orbit views.
    per_view_joints: list of {name: (px, py[, conf])}  -- conf optional, defaults 1.0.
    Finds the largest inlier set for each joint, then refines with median of
    inlier-pair midpoints.  Outlier views are excluded before the median, so a
    single hallucinated detection cannot pull the result by several centimetres.
    """
    from mathutils import geometry as _geo

    half_scale = (hand_scale if hand_scale is not None else _HAND_SCALE) / 2.0
    hw_v       = Vector(hw)
    all_names  = set(n for jd in per_view_joints for n in jd)

    rays_per_joint = {name: [] for name in all_names}
    for joints, (cam_pos, right_vec, up_vec, forward_vec) in zip(per_view_joints, per_view_cameras):
        ray_dir = (-forward_vec).normalized()
        for name, det in joints.items():
            px, py   = det[0], det[1]
            conf     = det[2] if len(det) > 2 else 1.0
            if conf < min_conf:
                continue
            ndc_x  = (px / _HAND_IMG_SIZE - 0.5) * 2.0
            ndc_y  = (0.5 - py / _HAND_IMG_SIZE) * 2.0
            origin = cam_pos + right_vec * ndc_x * half_scale + up_vec * ndc_y * half_scale
            rays_per_joint[name].append({'origin': origin, 'dir': ray_dir, 'conf': conf})

    joints_3d = {}
    for name, rays in rays_per_joint.items():
        if len(rays) < 2:
            continue

        best_inliers = []
        best_point   = None

        n = len(rays)
        for i in range(n):
            for j in range(i + 1, n):
                r1, r2 = rays[i], rays[j]
                if abs(r1['dir'].dot(r2['dir'])) > 0.95:
                    continue
                result = _geo.intersect_line_line(
                    r1['origin'], r1['origin'] + r1['dir'] * 20,
                    r2['origin'], r2['origin'] + r2['dir'] * 20,
                )
                if result is None:
                    continue
                p1, p2 = result
                mid    = (Vector(p1) + Vector(p2)) * 0.5

                inliers = []
                for r in rays:
                    to_pt   = mid - r['origin']
                    proj    = to_pt.dot(r['dir'])
                    closest = r['origin'] + r['dir'] * proj
                    if (mid - closest).length < inlier_thresh:
                        inliers.append(r)

                if len(inliers) > len(best_inliers):
                    best_inliers = inliers
                    best_point   = mid

        if len(best_inliers) < 2 or best_point is None:
            continue

        # Cap refinement set  -- keeps O(n2) bounded if _ORBIT_N_VIEWS is ever increased
        if len(best_inliers) > 8:
            best_inliers = sorted(best_inliers, key=lambda r: r['conf'], reverse=True)[:8]

        # Refine: median of all pairwise intersections among inliers only
        refined = []
        m = len(best_inliers)
        for i in range(m):
            for j in range(i + 1, m):
                r1, r2 = best_inliers[i], best_inliers[j]
                if abs(r1['dir'].dot(r2['dir'])) > 0.95:
                    continue
                result = _geo.intersect_line_line(
                    r1['origin'], r1['origin'] + r1['dir'] * 20,
                    r2['origin'], r2['origin'] + r2['dir'] * 20,
                )
                if result:
                    p1, p2 = result
                    refined.append((Vector(p1) + Vector(p2)) * 0.5)

        candidate  = Vector(np.median(np.array([p[:] for p in refined]), axis=0)) \
                     if refined else best_point
        # Wrist-tolerant sanity gate (matches _triangulate_tips_ls). A fixed
        # distance-from-wrist limit is wrist-DEPENDENT: a wrist placed up the
        # forearm pushes real joints past 0.22m and they get rejected/grafted,
        # which is what made finger detection change when the wrist marker moved.
        # Scale the limit with hand_scale so wrist position can't flip it.
        _eff_max = max(max_dist, (hand_scale or _HAND_SCALE) * 1.25 + 0.05)
        if (candidate - hw_v).length > _eff_max:
            continue

        joints_3d[name] = candidate

    return joints_3d


def _triangulate_tips_ls(per_view_tips, per_view_cameras, hw, ew, arp_side,
                         hand_scale=None, max_dist=_MAX_TIP_DIST, bvh=None):
    """
    Grid-search triangulation for 5 fingertip heatmaps.
    Coarse search +/-6 cm around the geometric template, fine search +/-8 mm.
    Tips that overshoot the finger are snapped to the nearest mesh surface
    rather than discarded (drop only if snap_dist > 15 cm = wrong body part).
    """
    half_scale = (hand_scale if hand_scale is not None else _HAND_SCALE) / 2.0
    hw_v = Vector(hw)
    # Scale factor vs the canonical training hand size. All the world-space search /
    # snap constants below are tuned for a ~0.35 m hand; on a mesh authored at a large
    # scale (1 unit != 1 m, or a giant character) a correct tip back-projects metres
    # off the surface and every absolute threshold rejects it. Scale UP only, so
    # normal hands (_sf == 1) keep their tuned constants -> no regression.
    _sf = max(1.0, (hand_scale if hand_scale is not None else _HAND_SCALE) / _HAND_SCALE)

    geom_all = _place_fingers_geometric(hw, ew, arp_side) if ew is not None else {}

    all_names = set(n for jd in per_view_tips for n in jd)
    rays_per_joint = {name: [] for name in all_names}

    for joints, (cam_pos, right_vec, up_vec, forward_vec) in zip(per_view_tips, per_view_cameras):
        ray_dir = (-forward_vec).normalized()
        for name, det in joints.items():
            px, py  = det[0], det[1]
            ndc_x   = (px / _HAND_IMG_SIZE - 0.5) * 2.0
            ndc_y   = (0.5 - py / _HAND_IMG_SIZE) * 2.0
            origin  = cam_pos + right_vec * ndc_x * half_scale + up_vec * ndc_y * half_scale
            rays_per_joint[name].append((np.array(origin), np.array(ray_dir)))

    joints_3d = {}
    for name, rays in rays_per_joint.items():
        if len(rays) < 2:
            continue

        origins = np.stack([r[0] for r in rays])
        dirs    = np.stack([r[1] for r in rays])

        # Grid-search center = least-squares intersection of the rays. This is
        # WRIST-INDEPENDENT: the rays come from the (now wrist-independent) render,
        # so the search re-centers on the actual tip convergence, not on the
        # wrist-anchored geometric template - which scaled/shifted with the wrist
        # marker and was the second wrist leak (identical render, different grid_err).
        _A = np.zeros((3, 3)); _b = np.zeros(3)
        for _o, _d in zip(origins, dirs):
            _P  = np.eye(3) - np.outer(_d, _d)
            _A += _P
            _b += _P @ _o
        try:
            center = np.linalg.solve(_A, _b)
        except np.linalg.LinAlgError:
            center = origins.mean(axis=0)
        if not np.all(np.isfinite(center)):
            arp_tip_name = f"{_FINGER_TO_ARP[name]['tip']}_{arp_side}"
            geom_tip = geom_all.get(arp_tip_name)
            center = (np.array(geom_tip) if geom_tip is not None
                      else origins.mean(axis=0))

        # -- Coarse grid: 13x13x13, +/-6 cm around template (scaled for big meshes) --
        steps = np.linspace(-0.06 * _sf, 0.06 * _sf, 13)
        grid  = np.array(np.meshgrid(steps, steps, steps, indexing='ij')).reshape(3, -1).T + center

        chunk = 128
        best_err = float('inf')
        best_pt  = None
        for i in range(0, len(grid), chunk):
            g    = grid[i:i + chunk]
            diff = g[:, None, :] - origins[None, :, :]
            proj = np.sum(diff * dirs[None, :, :], axis=2)
            cls  = origins[None, :, :] + dirs[None, :, :] * proj[:, :, None]
            errs = np.linalg.norm(g[:, None, :] - cls, axis=2)
            meds = np.median(errs, axis=1)
            j    = int(np.argmin(meds))
            if meds[j] < best_err:
                best_err = meds[j]
                best_pt  = g[j]

        if best_pt is None:
            continue

        # -- Fine grid: 9x9x9, +/-8 mm around coarse best (scaled for big meshes) --
        fine_steps = np.linspace(-0.008 * _sf, 0.008 * _sf, 9)
        fine_grid  = np.array(np.meshgrid(fine_steps, fine_steps, fine_steps,
                                          indexing='ij')).reshape(3, -1).T + best_pt

        best_err_f = float('inf')
        best_pt_f  = None
        for i in range(0, len(fine_grid), chunk):
            g    = fine_grid[i:i + chunk]
            diff = g[:, None, :] - origins[None, :, :]
            proj = np.sum(diff * dirs[None, :, :], axis=2)
            cls  = origins[None, :, :] + dirs[None, :, :] * proj[:, :, None]
            errs = np.linalg.norm(g[:, None, :] - cls, axis=2)
            meds = np.median(errs, axis=1)
            j    = int(np.argmin(meds))
            if meds[j] < best_err_f:
                best_err_f = meds[j]
                best_pt_f  = g[j]

        if best_pt_f is None:
            continue

        P = Vector(best_pt_f)

        # -- Mesh snap: overshoot tips snap to actual fingertip surface -------
        if bvh is not None:
            hit, _n, _idx, snap_dist = bvh.find_nearest(P)
            if hit is None or snap_dist > 0.15 * _sf:
                print(f"  [grid] {name}: dropped (snap_dist={snap_dist:.3f}m  -- wrong body region)")
                continue
            if snap_dist > 0.01 * _sf:
                print(f"  {name}: tip snapped {snap_dist:.3f}m -> mesh surface")
                P = Vector(hit)
                # find_nearest snaps to the CLOSEST surface point - a tip triangulated
                # off to one side lands on the finger's SIDE, not its distal midline.
                # Re-centre once in the local cross-section (perpendicular to the
                # wrist->tip axis). Distal position is preserved; nothing downstream
                # corrects this across drift.
                _ax = P - hw_v
                if _ax.length > 1e-4:
                    _cc = _local_cross_section_center(P, _ax.normalized(), bvh,
                                                      search_radius=0.015 * _sf)
                    if _cc is not None:
                        P = _cc

        # Wrist-tolerant sanity gate. Measuring from the wrist marker (hw) is
        # wrist-DEPENDENT: a wrist placed up the forearm inflates the distance and
        # wrongly rejects perfectly-triangulated tips (the second wrist leak). The
        # tip is at most ~one hand length from the wrist, plus slack for a wrist
        # sitting up the forearm. Scale the limit with hand_scale so wrist position
        # can't flip it. Gross errors (tips on the wrong body part) are already
        # caught by the mesh-snap >15cm drop above.
        _eff_max = max(max_dist, (hand_scale or _HAND_SCALE) * 1.25 + 0.05)
        if (P - hw_v).length > _eff_max:
            print(f"  [grid] {name}: too far from wrist {(P - hw_v).length:.3f}m "
                  f"(limit {_eff_max:.3f}m)")
            continue

        joints_3d[name] = P

    return joints_3d


# -- Hand mesh isolation -------------------------------------------------------

def _isolate_hand_mesh(mesh_obj, hw, ew, tip=None, wrist_island=False, keep_forearm=False):
    """
    Return a temporary mesh object containing only the hand/finger region.
    Two bisect planes trap the hand between wrist and fingertip:
      - Back cut  : 8 cm behind hw toward elbow (removes arm/body)
      - Front cut : 2 cm beyond tip toward fingers (removes legs/ground)
    tip: world-space fingertip (MARKER_HAND_TIP).  Falls back to hw + fwd*0.30.
    bisect_plane inserts clean vertices at cut edges (no holes).

    wrist_island: when True, after the cuts keep ONLY the connected mesh island that
    contains the wrist (flood-fill over edges from the wrist-nearest vertex). This
    drops legs/props that sit in the cut slab but are disconnected from the hand
    once the body is severed. Default False so the neural orbit path is unchanged.

    Caller is responsible for removing the returned object when done.
    """
    import bmesh as _bm
    from mathutils import Matrix

    hw_v    = Vector(hw)
    ew_v    = Vector(ew) if ew is not None else hw_v - Vector((0.1, 0.0, 0.0))
    arm_vec = hw_v - ew_v
    arm_len = arm_vec.length
    arm_dir = arm_vec.normalized() if arm_len > 0.001 else Vector((1.0, 0.0, 0.0))

    if keep_forearm:
        # Cut just BEHIND THE ELBOW (arm_dir points elbow->wrist, so back off from the
        # elbow) so the whole FOREARM is kept — only the chest / head / legs beyond the
        # elbow are removed. Used by the orbit render, which trains on forearm context.
        cut_back = ew_v - arm_dir * max(0.04, arm_len * 0.3)
    else:
        # Back cut: 8 cm minimum behind hw (removes arm/body behind wrist).
        wrist_back = max(0.08, arm_len * 0.15)
        cut_back   = hw_v - arm_dir * wrist_back

    # Front cut: beyond the fingertip (removes legs/body ahead of hand). Generous,
    # scale-relative margin (25% of hand length, min 3 cm) so a finger LONGER than the
    # HAND_TIP marker — or a marker placed on a shorter finger — is NOT clipped. Legs
    # ahead that fall inside this slab are dropped by the wrist-island flood-fill
    # anyway, so a roomy front cut is safe.
    if tip is not None:
        tip_dir  = (Vector(tip) - hw_v)
        tip_len  = tip_dir.length
        fwd_tip  = (tip_dir / tip_len) if tip_len > 0.01 else arm_dir
        cut_front = Vector(tip) + fwd_tip * max(0.03, tip_len * 0.25)
    else:
        cut_front = hw_v + arm_dir * max(0.10, arm_len * 1.5)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval  = mesh_obj.evaluated_get(depsgraph)
    mw        = obj_eval.matrix_world

    hand_data = bpy.data.meshes.new_from_object(obj_eval, depsgraph=depsgraph)
    hand_obj  = bpy.data.objects.new("_er_hand_isolated", hand_data)
    bpy.context.collection.objects.link(hand_obj)
    hand_obj.matrix_world = Matrix.Identity(4)
    hand_obj.color        = mesh_obj.color

    bm = _bm.new()
    bm.from_mesh(hand_data)
    bm.transform(mw)  # bake world transform: verts -> world space, handles normals + negative scale

    # MERGE first: weld coincident / near-coincident verts so a hand modelled as
    # SEPARATE geometry (fingers or the whole hand not welded to the arm) fuses into
    # one connected island. Otherwise the island flood-fill below keeps only whichever
    # scrap the tip seed happened to land on (the R-hand-stub failure). Small, scale-
    # relative threshold welds unwelded seams without merging distinct fingers. NOTE:
    # this only joins pieces whose vertices actually coincide; truly gapped or purely
    # interpenetrating pieces can't be welded, and the caller's collapse-guard then
    # falls back to the full mesh so nothing is silently lost.
    # PINCHED-FINGER CAVEAT: at ~1.45mm this can also weld two DIFFERENT finger surfaces
    # that are pinched together. Every attempt to change it (tighten / boundary-only /
    # region-limit) shifts detection on NORMAL hands (regression-proven) because this
    # welded mesh also feeds the geodesic HAND_TIP estimate + framing — so the weld is
    # load-bearing and must stay full-strength. A pinched-finger fix belongs elsewhere
    # (render separation cue or post-process), NOT here. Diagnose the render first.
    if wrist_island:
        try:
            _bm.ops.remove_doubles(bm, verts=bm.verts[:], dist=max(1e-5, arm_len * 0.005))
        except Exception:
            pass

    # -- Back bisect: remove elbow/body side ----------------------------------
    # plane_no points TOWARD fingers -> clear_inner removes the elbow side.
    arm_dir_back = arm_dir.copy()
    if (hw_v - cut_back).dot(arm_dir_back) < 0:
        arm_dir_back = -arm_dir_back
    _bm.ops.bisect_plane(
        bm,
        geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
        dist=0.0001,
        plane_co=cut_back,
        plane_no=arm_dir_back,
        clear_inner=True,
        clear_outer=False,
    )

    # -- Front bisect: remove anything beyond the fingertip -------------------
    # plane_no points TOWARD fingers -> clear_outer removes the beyond-tip side.
    fwd_front = fwd_tip if tip is not None else arm_dir
    _bm.ops.bisect_plane(
        bm,
        geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
        dist=0.0001,
        plane_co=cut_front,
        plane_no=fwd_front,
        clear_inner=False,
        clear_outer=True,
    )

    # -- Keep only the hand-connected island (drops disconnected legs/props) ----
    # Seed at the FINGERTIP (HAND_TIP), not the wrist: on some characters the hand
    # is a separate mesh piece not welded to the arm, so a wrist-seeded flood-fill
    # grabs only the forearm stub. The fingertip is always on the hand island.
    if wrist_island:
        bm.verts.ensure_lookup_table()
        if bm.verts:
            def _flood_from(seed_co):
                start = min(bm.verts, key=lambda v: (v.co - seed_co).length_squared)
                reached = {start}
                stack = [start]
                while stack:
                    v = stack.pop()
                    for e in v.link_edges:
                        o = e.other_vert(v)
                        if o not in reached:
                            reached.add(o)
                            stack.append(o)
                return reached, (start.co - seed_co).length

            # Flood from BOTH the fingertip and the wrist; keep the LARGER island. The tip
            # seed wins when the hand is a separate mesh from the arm; the wrist seed wins
            # when HAND_TIP lands on a tiny disconnected island near the fingers (a 1cm blob,
            # not the hand). Picking the larger recovers the real hand either way.
            #
            # tip=None (HAND_TIP estimate, before any tip is known): a wrist seed can walk
            # UP the forearm when the hand is separate geometry from the arm (a topological
            # gap at the wrist), leaving the whole hand in a DIFFERENT island — the flood
            # then keeps only the forearm stub and the tip estimate collapses to the wrist
            # (the hanging-arm HAND_TIP-on-forearm bug). Substitute a geometric fingertip
            # seed: the vertex reaching FARTHEST FORWARD along the arm axis is on the hand
            # by construction (the front cut removed everything past the fingers). Gate it
            # near the axis so a nearby leg/torso vertex can't hijack it, then keep whichever
            # of {forward-seed island, wrist island} reaches farther toward the fingers.
            if tip is not None:
                r_tip, d_tip = _flood_from(Vector(tip))
                r_wr, d_wr = _flood_from(hw_v)
                reached = r_tip if len(r_tip) >= len(r_wr) else r_wr
            else:
                _axis_gate = max(arm_len, 0.05)
                _fwd_seed = None; _fwd_best = -1e9
                for v in bm.verts:
                    _rel = v.co - hw_v
                    _al  = _rel.dot(arm_dir)
                    if _al <= 0 or (_rel - arm_dir * _al).length > _axis_gate:
                        continue
                    if _al > _fwd_best:
                        _fwd_best = _al; _fwd_seed = v.co
                r_tip, d_tip = _flood_from(_fwd_seed) if _fwd_seed is not None else (set(), 1e9)
                r_wr, d_wr = _flood_from(hw_v)

                def _reach(_isl):
                    return max(((v.co - hw_v).dot(arm_dir) for v in _isl), default=-1e9)
                _cands = [_isl for _isl in (r_tip, r_wr) if len(_isl) >= 30]
                reached = max(_cands, key=_reach) if _cands else r_wr
            print(f"  [isolate] after cuts: {len(bm.verts)} verts, "
                  f"tip-island {len(r_tip)} (d{d_tip*1000:.0f}mm) "
                  f"wrist-island {len(r_wr)} (d{d_wr*1000:.0f}mm) -> kept {len(reached)}")
            if len(reached) >= 30:
                # Re-attach SEPARATE finger geometry — but ONLY when the fingers were
                # actually dropped. When the fingers are modelled as their own unwelded,
                # GAPPED pieces (fragmented avatar meshes), the flood-fill above keeps
                # only the palm island and drops every finger, so the rendered hand (and
                # the neural detection reading it) is a fingerless stub ("I don't get the
                # full hand image"). GATE on the HAND_TIP marker: if it sits well OUTSIDE
                # the kept island, the fingertip region is on a dropped piece → re-attach
                # any detached island within a tiny gap of the hand (the same touching /
                # near-touching threshold the geometric engine bridges with; legs, the
                # other hand and props stay far enough to be dropped). If the tip already
                # sits on the island the fingers are present, so the isolation is left
                # exactly as-is — a normal welded hand is never touched (no regression).
                if tip is not None:
                    _tv = Vector(tip)
                    _tip_gap = min((v.co - _tv).length for v in reached)
                    if _tip_gap > arm_len * 0.06:
                        _elens = [e.calc_length() for e in bm.edges]
                        _elens.sort()
                        _med = _elens[len(_elens) // 2] if _elens else 0.0
                        _gap = max(4.0 * _med, arm_len * 0.02)
                        from mathutils.kdtree import KDTree as _KDT
                        _rl = list(reached)
                        _kd = _KDT(len(_rl))
                        for _i, _rv in enumerate(_rl):
                            _kd.insert(_rv.co, _i)
                        _kd.balance()
                        _rest = set(v for v in bm.verts if v not in reached)
                        _added = 0
                        while _rest:
                            _s = next(iter(_rest))
                            _isl = [_s]; _rest.discard(_s); _stk = [_s]
                            while _stk:
                                _v = _stk.pop()
                                for _e in _v.link_edges:
                                    _o = _e.other_vert(_v)
                                    if _o in _rest:
                                        _rest.discard(_o); _isl.append(_o); _stk.append(_o)
                            if len(_isl) < 8:
                                continue               # noise scrap — leave it dropped
                            _step = max(1, len(_isl) // 200)
                            for _v in _isl[::_step]:
                                _c, _mi, _d = _kd.find(_v.co)
                                if _d is not None and _d <= _gap:
                                    reached.update(_isl); _added += len(_isl)
                                    break
                        if _added:
                            print(f"  [isolate] re-attached {_added} vert(s) of "
                                  f"separate finger geometry (tip was "
                                  f"{_tip_gap*1000:.0f}mm off the palm island)")
                dead = [v for v in bm.verts if v not in reached]
                if dead:
                    _bm.ops.delete(bm, geom=dead, context='VERTS')

    bm.to_mesh(hand_data)
    bm.free()
    hand_data.update()
    return hand_obj


# -- Finger lateral-plane straightening ---------------------------------------

def _straighten_fingers(marker_pos, hw, ew, arp_side):
    """
    Project index/middle/ring/pinky PIP/DIP onto each finger's anatomical plane
    (the plane spanned by arm direction and finger direction).  Removes lateral
    drift introduced by triangulation without touching MCP/TIP anchors or thumb.
    No mesh access  -- safe to call on raw triangulated positions.
    """
    hw_v    = Vector(hw)
    ew_v    = Vector(ew) if ew is not None else hw_v - Vector((0.1, 0.0, 0.0))
    arm_dir = (hw_v - ew_v).normalized()

    for finger in ("index", "middle", "ring", "pinky"):
        mcp_key = f"FINGER_{finger.upper()}_1_{arp_side}"
        tip_key = f"FINGER_{finger.upper()}_TIP_{arp_side}"
        mcp     = marker_pos.get(mcp_key)
        tip     = marker_pos.get(tip_key)
        if mcp is None or tip is None:
            continue
        finger_dir = (tip - mcp).normalized()
        if finger_dir.length < 0.001:
            continue
        plane_n = arm_dir.cross(finger_dir)
        if plane_n.length < 0.001:
            continue
        plane_n = plane_n.normalized()
        for j_suffix in ("_2_", "_3_"):
            j_key = f"FINGER_{finger.upper()}{j_suffix}{arp_side}"
            pt    = marker_pos.get(j_key)
            if pt is None:
                continue
            marker_pos[j_key] = pt - plane_n * (pt - mcp).dot(plane_n)


def _straighten_and_center_fingers(mesh_obj, marker_pos, arp_side):
    """
    Stage 1: Proportional PIP/DIP placement (MCP and TIP are the reliable anchors).
    Stage 2: lateral plane projection (removes lateral drift). Stores fn per finger.
    Stage 3: Elliptical cross-section centering  -- narrow laterally (LAT_R=5mm) to
             block adjacent-finger contamination, wide dorsal-palmarly (DP_R=12mm)
             to capture full finger thickness. MCP skipped (ONNX+Stage1 place it well).
             Thumb falls back to circular search (no neighbor contamination concern).
    """
    all_fingers = ("thumb", "index", "middle", "ring", "pinky")

    # Stage 1 disabled: _phalange_chain_geom's similarity transform already
    # places PIP/DIP with correct proportions and lateral spread from the template.
    # Re-interpolating them here collapses everything to a straight line.

    # -- Stage 2: lateral plane projection ------------------------------------
    idx_mcp = marker_pos.get(f"{_FINGER_TO_ARP['index']['phal1']}_{arp_side}")
    pnk_mcp = marker_pos.get(f"{_FINGER_TO_ARP['pinky']['phal1']}_{arp_side}")
    if idx_mcp is not None and pnk_mcp is not None:
        hand_axis = (idx_mcp - pnk_mcp).normalized()
    else:
        hand_axis = Vector((1.0 if arp_side == "L" else -1.0, 0.0, 0.0))

    finger_lateral = {}  # lateral normal (fn) per finger, reused in Stage 3

    for finger in ("index", "middle", "ring", "pinky"):
        arp_map = _FINGER_TO_ARP.get(finger)
        if not arp_map:
            continue
        mcp = marker_pos.get(f"{arp_map['phal1']}_{arp_side}")
        tip = marker_pos.get(f"{arp_map['tip']}_{arp_side}")
        if mcp is None or tip is None:
            continue
        fdir      = (tip - mcp).normalized()
        hand_norm = hand_axis.cross(fdir)
        if hand_norm.length < 1e-6:   # check before normalizing  -- correct guard
            continue
        fn = hand_norm.cross(fdir).normalized()
        finger_lateral[finger] = fn   # store for Stage 3 elliptical filter
        for phal_key in ("phal2", "phal3"):
            k  = f"{arp_map[phal_key]}_{arp_side}"
            pt = marker_pos.get(k)
            if pt is None:
                continue
            lateral = (pt - mcp).dot(fn)
            if abs(lateral) > 0.0005:
                marker_pos[k] = pt - lateral * fn

    # -- Stage 3: Elliptical cross-section centering ---------------------------
    # MCP intentionally skipped: knuckle cross-section includes adjacent palm
    # geometry, centroid pulls toward palm centre. ONNX + Stage 1 handle MCPs.
    mw        = mesh_obj.matrix_world
    verts_w   = [mw @ v.co for v in mesh_obj.data.vertices]
    SLAB      = 0.006   # was 0.005  -- slightly thicker search slab
    MAX_SHIFT = 0.015   # was 0.008  -- allow 1.5 cm correction for badly placed geometric joints
    LAT_R     = 0.004   # was 0.005  -- narrower laterally to block neighbor fingers harder
    DP_R      = 0.014   # was 0.012  -- deeper dorsal-palmar capture
    CIRC_R    = 0.012   # was 0.010  -- thumb gets more room

    def _vert_center_elliptical(pos, axis, lateral_dir):
        """Elliptical centroid: narrow laterally to prevent cross-finger bleed."""
        dp_dir = axis.cross(lateral_dir).normalized()
        proj = []
        for vw in verts_w:
            rel   = vw - pos
            along = rel.dot(axis)
            if abs(along) > SLAB:
                continue
            perp     = rel - axis * along
            lat_dist = abs(perp.dot(lateral_dir))
            dp_dist  = abs(perp.dot(dp_dir))
            if (lat_dist / LAT_R) ** 2 + (dp_dist / DP_R) ** 2 > 1.0:
                continue
            proj.append(perp)
        if len(proj) < 4:
            return pos
        cen_off = Vector((
            sum(v.x for v in proj) / len(proj),
            sum(v.y for v in proj) / len(proj),
            sum(v.z for v in proj) / len(proj),
        ))
        if cen_off.length > MAX_SHIFT:
            return pos
        return pos + cen_off

    def _vert_center_circular(pos, axis):
        """Circular centroid fallback for thumb (no neighbour contamination)."""
        proj = []
        for vw in verts_w:
            rel   = vw - pos
            along = rel.dot(axis)
            if abs(along) > SLAB:
                continue
            if (rel - axis * along).length < CIRC_R:
                proj.append(vw - axis * along)
        if len(proj) < 4:
            return pos
        cen = Vector((
            sum(v.x for v in proj) / len(proj),
            sum(v.y for v in proj) / len(proj),
            sum(v.z for v in proj) / len(proj),
        ))
        if (cen - pos).length > MAX_SHIFT:
            return pos
        return cen

    for finger in all_fingers:
        arp_map = _FINGER_TO_ARP.get(finger)
        if not arp_map:
            continue
        mcp_key = f"{arp_map['phal1']}_{arp_side}"
        pip_key = f"{arp_map['phal2']}_{arp_side}"
        dip_key = f"{arp_map.get('phal3', '')}_{arp_side}"
        tip_key = f"{arp_map['tip']}_{arp_side}"

        r_mcp = marker_pos.get(mcp_key)
        r_pip = marker_pos.get(pip_key)
        r_dip = marker_pos.get(dip_key)
        r_tip = marker_pos.get(tip_key)

        if not (r_mcp and r_tip):
            continue
        global_axis = (r_tip - r_mcp).normalized()
        fn = finger_lateral.get(finger)  # None for thumb

        for key, pos, axis in (
            (tip_key, r_tip, global_axis),
            (pip_key, r_pip, (r_dip - r_mcp).normalized() if r_dip and (r_dip - r_mcp).length > 1e-4 else global_axis),
            (dip_key, r_dip, (r_tip - r_pip).normalized() if r_pip and (r_tip - r_pip).length > 1e-4 else global_axis),
        ):
            if pos is None or not key:
                continue
            if fn is not None:
                marker_pos[key] = _vert_center_elliptical(pos, axis, fn)
            else:
                marker_pos[key] = _vert_center_circular(pos, axis)

    # Stage 4 (knuckle snap) removed  -- _refine_mcp_positions handles MCP
    # placement via the knuckle_radius slider. Keeping Stage 4 would override
    # the slider result.


# -- Step 3: Anatomically correct phalange chain -------------------------------

def _local_cross_section_center(pos, axis, bvh, search_radius=0.015):
    """
    Local finger-volume center. Casts 6 paired rays perpendicular to `axis`
    within `search_radius` of `pos`. Returns average of opposite-hit midpoints,
    or None if no hits.
    """
    import math
    from mathutils import Vector

    if axis.length < 0.001:
        return None

    perp_u = Vector((0, 0, 1)).cross(axis).normalized()
    if perp_u.length < 0.001:
        perp_u = Vector((0, 1, 0)).cross(axis).normalized()
    perp_v = axis.cross(perp_u).normalized()

    midpoints = []
    for j in range(6):
        theta   = j * math.pi / 3.0
        ray_dir = (perp_u * math.cos(theta) + perp_v * math.sin(theta)).normalized()

        hit_pos, _, _, _ = bvh.ray_cast(pos + ray_dir * 0.002,  ray_dir,  search_radius)
        hit_neg, _, _, _ = bvh.ray_cast(pos - ray_dir * 0.002, -ray_dir,  search_radius)

        if hit_pos is not None and hit_neg is not None:
            midpoints.append((Vector(hit_pos) + Vector(hit_neg)) * 0.5)
        elif hit_pos is not None:
            # Only the +dir wall was hit -> pos is on the OPPOSITE (-dir) wall,
            # so the cross-section centre is halfway between pos and that wall.
            midpoints.append((Vector(pos) + Vector(hit_pos)) * 0.5)
        elif hit_neg is not None:
            midpoints.append((Vector(pos) + Vector(hit_neg)) * 0.5)

    if not midpoints:
        return None
    return sum(midpoints, Vector()) / len(midpoints)


_DEFAULT_FOREARM_LEN_LVT = 0.291  # (HAND_L ^' ELBOW_L).length in the default metarig


_WRIST_EST_CACHE = {}   # (mesh name, n_verts, ew, tip) -> mesh-wrist Vector or None


def _mesh_wrist_for_framing(mesh_obj, hw, ew, tip):
    """Mesh-derived wrist for FRAMING only (elbow->HAND_TIP narrowest slice via
    _wrist_from_mesh — neither input is the HAND marker). Returns None when it
    can't measure or fails the same half-span sanity cap the auto-snap uses;
    the caller then keeps the marker. Cached: it is a full vertex scan and the
    LVT + landmark stages ask for the same inputs back to back."""
    if mesh_obj is None or ew is None or tip is None:
        return None
    try:
        key = (mesh_obj.name, len(mesh_obj.data.vertices),
               tuple(round(c, 6) for c in ew), tuple(round(c, 6) for c in tip))
        if key in _WRIST_EST_CACHE:
            est = _WRIST_EST_CACHE[key]
        else:
            from .ai_detect import _wrist_from_mesh
            est = _wrist_from_mesh(mesh_obj, Vector(ew), Vector(tip))
            _WRIST_EST_CACHE[key] = est.copy() if est is not None else None
            if len(_WRIST_EST_CACHE) > 32:
                _WRIST_EST_CACHE.clear()
        if est is None:
            return None
        span = (Vector(ew) - Vector(tip)).length
        if span > 1e-6 and (est - Vector(hw)).length > 0.5 * span:
            return None          # failed estimate (misplaced HAND_TIP) — keep marker
        if (est - Vector(hw)).length <= 0.005:
            # Marker already agrees with the mesh (auto-snapped, or placed well
            # by hand) — keep the marker so approved framings stay bit-stable.
            # The estimate only takes over for REAL divergence, where marker
            # framing is what made detection wrist-sensitive.
            return None
        return est.copy()
    except Exception:
        return None


def _compute_render_hw(hw, ew, arp_side, mesh_obj=None, tip=None):
    """
    Compute a wrist reference point for orbit rendering and mesh isolation.

    When the body-detection wrist marker sits far up the forearm the orbit
    camera centres on forearm geometry, not the hand.  This returns a point
    ~60 mm inside the hand from the geometric palm centre so that
    _render_hand_orbit and _isolate_hand_mesh always target actual fingers.
    Falls back to the original hw if ew is missing.

    When mesh_obj + tip are given, the wrist itself comes from the MESH
    (narrowest cross-section on the elbow->tip axis) instead of the HAND
    marker, so the render framing — and with it every neural detection —
    no longer shifts when the wrist marker moves a few mm (the auto-snap
    framing leak). The marker stays in charge of everything placement-side.
    """
    if ew is None:
        return Vector(hw)

    hw_v    = Vector(hw)
    _est = _mesh_wrist_for_framing(mesh_obj, hw, ew, tip)
    if _est is not None:
        if (_est - hw_v).length > 0.001:
            print(f"  [orbit] framing wrist = mesh estimate "
                  f"({(_est - hw_v).length * 1000:.0f}mm from HAND marker)")
        hw_v = _est
    ew_v    = Vector(ew)
    arm_vec = hw_v - ew_v
    arm_len = arm_vec.length
    if arm_len < 0.001:
        return hw_v

    arm_dir = arm_vec / arm_len
    scale   = arm_len / _DEFAULT_FOREARM_LEN_LVT

    if abs(arm_dir.z) < 0.8:
        spread_dir = Vector((0.0, 1.0, 0.0))
    else:
        x_sign = 1.0 if arp_side == 'L' else -1.0
        spread_dir = Vector((x_sign, 0.0, 0.0))

    # Geometric MCP positions for index/middle/ring/pinky (from Rigify metarig)
    mcp_offsets = [(0.039, -0.0236), (0.046, 0.0000),
                   (0.039,  0.0236), (0.030, 0.0459)]
    mcp_positions = [hw_v + arm_dir * (ap * scale) + spread_dir * (yo * scale)
                     for ap, yo in mcp_offsets]
    palm_center_geom = sum(mcp_positions, Vector()) / len(mcp_positions)

    # Pull back ~60 mm (at human scale) toward the elbow so _render_hand_orbit centres
    # on the palm. SCALE-RELATIVE: on a small character an absolute 60 mm lands up the
    # forearm, which then poisons the crop's forearm gate and fills the frame with arm.
    return palm_center_geom - arm_dir * (0.06 * scale)


def _estimate_palm_center(hw, tips_3d, ew=None, arp_side=None, palm_depth=0.25):
    """
    Compute palm center from detected tips using the arm direction (elbow->wrist).
    The arm line is stable even when the wrist marker sits a few cm too far up
    the forearm  -- only the direction matters, not the absolute hw position.
    palm_depth scales how far from the tip centroid toward the palm to shift.
    """
    if len(tips_3d) < 3:
        return Vector(hw)

    centroid = sum(tips_3d.values(), Vector((0, 0, 0))) / len(tips_3d)

    if ew is not None:
        arm_dir = (Vector(hw) - Vector(ew)).normalized()
    else:
        to_hw   = Vector(hw) - centroid
        arm_dir = to_hw.normalized() if to_hw.length > 0.001 else Vector((0.0, -1.0, 0.0))

    avg_tip_dist = sum((t - centroid).length for t in tips_3d.values()) / len(tips_3d)
    shift        = avg_tip_dist * 0.75 * (palm_depth / 0.25)
    return centroid + arm_dir * shift


def _project_onto_plane(point, plane_origin, plane_normal):
    """Project point onto the plane defined by plane_origin and plane_normal."""
    return point - plane_normal * (point - plane_origin).dot(plane_normal)


_FINGER_MCP_RATIOS = {
    'thumb':  0.38,
    'index':  0.33,
    'middle': 0.33,
    'ring':   0.35,
    'pinky':  0.37,
}


def _refine_mcp_positions(finger_mcps, tips_3d, palm_center, bvh=None, knuckle_radius=0.90):
    """
    Legacy circular projection for MCPs.

    Projects each raw chain-walk MCP onto a circle around palm_center at
    radius = avg_raw_dist * knuckle_radius.  The raw MCP direction from
    palm_center is reliable (chain-walk followed the finger); only the
    distance is wrong (stops at PIP instead of MCP on thick fingers).
    Returns corrected MCPs  -- same finger-name keys as input.
    """
    refined = {}
    four = ['index', 'middle', 'ring', 'pinky']

    avg_dist = 0.0
    count = 0
    for fname in four:
        if fname in finger_mcps:
            avg_dist += (finger_mcps[fname] - palm_center).length
            count += 1

    if count == 0:
        return {k: Vector(v) for k, v in finger_mcps.items()}

    avg_dist /= count
    target_radius = avg_dist * knuckle_radius

    for fname in four:
        if fname not in finger_mcps:
            continue
        raw       = Vector(finger_mcps[fname])
        direction = (raw - palm_center).normalized()
        ideal     = palm_center + direction * target_radius

        if bvh is not None:
            hit, _n, _idx, dist = bvh.find_nearest(ideal)
            if hit is not None and dist < 0.015:
                ideal = Vector(hit)

        refined[fname] = ideal

    # Thumb: same circular projection as long fingers, scaled by knuckle_radius
    if 'thumb' in finger_mcps:
        raw_t   = Vector(finger_mcps['thumb'])
        dir_t   = (raw_t - palm_center).normalized()
        dist_t  = (raw_t - palm_center).length
        ideal_t = palm_center + dir_t * (dist_t * knuckle_radius)
        if bvh is not None:
            hit, _n, _idx, dist = bvh.find_nearest(ideal_t)
            if hit is not None and dist < 0.015:
                ideal_t = Vector(hit)
        refined['thumb'] = ideal_t

    return refined


def _recompute_phalanges(refined_mcps, tips_3d, bvh=None):
    """
    Recompute PIP and DIP from corrected MCP to tip using anatomical fractions.

    Long fingers: PIP at 45%, DIP at 75% along the MCP->tip vector.
    Thumb: IP joint at 50% along the MCP->tip vector (maps to phal3/THUMB_3).
    This replaces chain-walk PIP/DIP that were computed from the wrong (raw) MCP.
    Returns {fname: {'pip': Vector, 'dip': Vector}}  -- only intermediate joints.
    """
    result = {}

    for fname in _FINGER_NAMES:
        if fname not in refined_mcps or fname not in tips_3d:
            continue

        mcp = Vector(refined_mcps[fname])
        tip = tips_3d[fname]
        vec = tip - mcp
        if vec.length < 0.001:
            continue

        if fname == 'thumb':
            ip = mcp + vec * 0.50
            if bvh is not None:
                hit, _n, _idx, dist = bvh.find_nearest(ip)
                if hit is not None and dist < 0.010:
                    ip = Vector(hit)
            result[fname] = {'pip': None, 'dip': ip}
        else:
            pip = mcp + vec * 0.45
            dip = mcp + vec * 0.75
            if bvh is not None:
                hit, _n, _idx, dist = bvh.find_nearest(pip)
                if hit is not None and dist < 0.010:
                    pip = Vector(hit)
                hit, _n, _idx, dist = bvh.find_nearest(dip)
                if hit is not None and dist < 0.010:
                    dip = Vector(hit)
            result[fname] = {'pip': pip, 'dip': dip}

    return result


def _straighten_finger_phalanges(marker_pos, tips_3d, palm_center, arp_side, straighten_clamp=0.15):
    """
    Anatomical plane straighten pass for PIP/DIP (phal2/phal3).
    Projects intermediate joints onto the plane defined by hand_normal x finger_dir,
    where hand_normal points from palm_center toward the tip centroid.
    Anchors (MCP = phal1, TIP) are never moved.
    Thumb is skipped  -- it has its own anatomical plane.
    A 15% finger-length clamp guards against over-correction on bent fingers.
    """
    if not tips_3d:
        return

    # Hand normal: palm-center -> tip centroid (perpendicular to the knuckle row)
    centroid = sum(tips_3d.values(), Vector((0, 0, 0))) / len(tips_3d)
    hand_normal = (centroid - palm_center).normalized()
    if hand_normal.length < 0.001:
        return

    for finger in ("index", "middle", "ring", "pinky"):
        arp      = _FINGER_TO_ARP[finger]
        mcp_key  = f"{arp['phal1']}_{arp_side}"
        pip_key  = f"{arp['phal2']}_{arp_side}"
        dip_key  = f"{arp['phal3']}_{arp_side}"
        tip_key  = f"{arp['tip']}_{arp_side}"

        mcp = marker_pos.get(mcp_key)
        tip = marker_pos.get(tip_key)
        if mcp is None or tip is None:
            continue

        finger_dir = (tip - mcp).normalized()
        if finger_dir.length < 0.001:
            continue

        plane_n = hand_normal.cross(finger_dir)
        if plane_n.length < 0.001:
            continue
        plane_n = plane_n.normalized()

        finger_len  = (tip - mcp).length
        max_shift   = finger_len * straighten_clamp

        for key in (pip_key, dip_key):
            pt = marker_pos.get(key)
            if pt is None:
                continue
            projected = _project_onto_plane(pt, mcp, plane_n)
            shift = (projected - pt).length
            if shift > max_shift:
                print(f"  [straighten] {finger}/{key}: shift {shift*1000:.1f}mm > 15%  -- skipped")
                continue
            marker_pos[key] = projected


def _phalange_chain_geom(tip_3d, hw, ew, arp_side, finger_name,
                         bvh=None, hand_obj=None, width_tolerance=1.0):
    """
    Finds MCP by ray-marching from the detected tip toward the wrist, measuring
    cross-section width at each step. The MCP is where the finger tube widens
    into the palm (cross-section jumps >=50%). PIP/DIP are then placed at fixed
    fractions of the MCP->tip line (inside the finger), then centered locally.

    This avoids both the geometric-template MCP error and the palm-bleed problem:
    we always start inside the finger (at the tip) and stop at the palm boundary.

    Returns [tip, phal3(DIP/IP), phal2(PIP/MCP), phal1(MCP/CMC)] as Vectors.
    """
    from mathutils import Vector, Matrix
    import math

    geom    = _place_fingers_geometric(hw, ew, arp_side)
    arp_map = _FINGER_TO_ARP[finger_name]

    tip_key   = f"{arp_map['tip']}_{arp_side}"
    phal1_key = f"{arp_map['phal1']}_{arp_side}"
    phal2_key = f"{arp_map['phal2']}_{arp_side}"
    phal3_key = f"{arp_map['phal3']}_{arp_side}"

    tip   = Vector(tip_3d)
    wrist = Vector(hw)
    mcp_template = Vector(geom.get(phal1_key, hw))

    # -- Fallback: similarity transform when no mesh ------------------------
    if bvh is None or hand_obj is None or (tip - wrist).length < 0.02:
        default_tip   = geom.get(tip_key)
        default_mcp   = Vector(geom.get(phal1_key, hw))
        default_phal2 = Vector(geom.get(phal2_key, hw))
        default_phal3 = Vector(geom.get(phal3_key, hw))
        default_tip_v = Vector(default_tip) if default_tip else tip

        tip_vec  = tip - default_mcp
        tmpl_vec = default_tip_v - default_mcp
        t_len, d_len = tmpl_vec.length, tip_vec.length
        if t_len > 1e-6 and d_len > 1e-6:
            scale    = d_len / t_len
            rot_axis = tmpl_vec.cross(tip_vec)
            rot_matrix = (Matrix.Rotation(tmpl_vec.angle(tip_vec), 3, rot_axis.normalized())
                          if rot_axis.length > 1e-6 else Matrix.Identity(3))
        else:
            scale, rot_matrix = 1.0, Matrix.Identity(3)
        return [
            tip,
            default_mcp + rot_matrix @ ((default_phal3 - default_mcp) * scale),
            default_mcp + rot_matrix @ ((default_phal2 - default_mcp) * scale),
            default_mcp,
        ]

    # -- Walk tip -> wrist, measuring cross-section width at each step -------
    march_dir = (wrist - tip).normalized()
    total_len = (tip - wrist).length

    world_up = Vector((0, 0, 1))
    if abs(march_dir.dot(world_up)) > 0.95:
        world_up = Vector((0, 1, 0))
    perp_u = world_up.cross(march_dir).normalized()
    perp_v = march_dir.cross(perp_u).normalized()

    step      = 0.004           # 4 mm
    n_steps   = max(0, int(total_len / step) - 1)
    widths    = []
    positions = []

    for i in range(n_steps):
        pos = tip + march_dir * (i * step)
        hits = []
        for j in range(4):
            theta   = j * math.pi / 4.0
            ray_dir = (perp_u * math.cos(theta) + perp_v * math.sin(theta)).normalized()
            hp, _, _, _ = bvh.ray_cast(pos + ray_dir * 0.003,  ray_dir, 0.06)
            hn, _, _, _ = bvh.ray_cast(pos - ray_dir * 0.003, -ray_dir, 0.06)
            if hp is not None and hn is not None:
                hits.append((Vector(hp) - Vector(hn)).length)
        widths.append(sum(hits) / len(hits) if hits else 0.0)
        positions.append(Vector(pos))

    # -- Detect MCP: step where cross-section jumps from finger -> palm -----
    mcp = mcp_template
    if len(widths) >= 6:
        # Baseline = median of first 40% of readings (inside finger shaft)
        n_base   = max(3, int(len(widths) * 0.4))
        baseline = sorted(w for w in widths[:n_base] if w > 0.005)
        baseline = baseline[len(baseline) // 2] if baseline else 0.016
        palm_threshold = max(baseline * 1.5 * width_tolerance, 0.022)

        for i in range(2, len(widths)):
            if widths[i] > palm_threshold and widths[i] > widths[i - 1] * 1.2:
                mcp = positions[max(0, i - 2)]
                print(f"  [chain] {finger_name}: MCP det step {i-2}/{n_steps}  "
                      f"w={widths[max(0,i-2)]:.3f}m base={baseline:.3f}m thr={palm_threshold:.3f}m")
                break
        else:
            # No clear palm transition  -- finger may be very short or palm flush
            # Fall back to template proportion from detected tip
            frac_from_tip = {"thumb": 0.77, "index": 0.69, "middle": 0.73,
                             "ring": 0.69, "pinky": 0.59}.get(finger_name, 0.69)
            mcp = tip + march_dir * total_len * frac_from_tip

    # -- PIP / DIP at fixed fractions along MCP->tip ------------------------
    # These fractions are of the MCP->tip segment (not wrist->tip).
    # 0.38 ~ PIP (middle of proximal phalange)
    # 0.72 ~ DIP (middle of middle phalange)
    pip_init = mcp.lerp(tip, 0.38)
    dip_init = mcp.lerp(tip, 0.72)

    finger_axis = (tip - mcp).normalized()
    if finger_axis.length < 0.001:
        finger_axis = (tip - wrist).normalized()

    pip_center = _local_cross_section_center(pip_init, finger_axis, bvh, search_radius=0.015)
    pip = pip_center * 0.7 + pip_init * 0.3 if pip_center is not None else pip_init

    dip_center = _local_cross_section_center(dip_init, finger_axis, bvh, search_radius=0.015)
    dip = dip_center * 0.7 + dip_init * 0.3 if dip_center is not None else dip_init

    if finger_name == "thumb":
        return [tip, dip, mcp, mcp]
    else:
        return [tip, dip, pip, mcp]


# -- Primary: 20-landmark model ------------------------------------------------

def _finger_ext_sum(marker_pos, arp_side):
    """Total MCP->TIP length over the 5 fingers - a collapse-sensitive quality score
    for comparing two framing passes (higher = fingers more extended)."""
    s = 0.0
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        arp = _FINGER_TO_ARP.get(finger)
        if not arp:
            continue
        tip = marker_pos.get(f"{arp['tip']}_{arp_side}")
        mcp = marker_pos.get(f"{arp['phal1']}_{arp_side}")
        if tip and mcp:
            s += (tip - mcp).length
    return s


def detect_fingers_landmark(mesh_obj, hw, ew, temp_dir, arp_side, orbit_center=None,
                            orbit_scale=None, orbit_fwd=None, _allow_two_pass=False):
    """
    Runs the hand pose model to predict all 20 hand joints per orbit view,
    then triangulates each joint independently. Per-joint geometric graft fills
    any joints the model missed.
    Returns {arp_marker_name: Vector} or None on failure.
    """
    if not is_landmark_available():
        return None

    _tip_obj = bpy.data.objects.get(f"MARKER_HAND_TIP_{arp_side}")
    if _tip_obj:
        _tip_pt = _tip_obj.location.copy()
    else:
        _fwd     = (Vector(hw) - Vector(ew)).normalized() if ew is not None else Vector((1.0, 0.0, 0.0))
        _arm_len = (Vector(hw) - Vector(ew)).length if ew is not None else 0.28
        _tip_pt  = Vector(hw) + _fwd * _arm_len * 0.65

    hw_render = _compute_render_hw(hw, ew, arp_side, mesh_obj=mesh_obj,
                                   tip=_tip_pt if _tip_obj else None)

    # Reuse the LVT stage's renders when the framing this stage WOULD use
    # (anchor/scale seeded from LVT's own results) agrees with how those
    # renders were actually framed — the second 8-view pass is then duplicate
    # work. Meaningful disagreement (LVT's detection moved the anchor) still
    # re-renders with the refined framing.
    _reuse = _ORBIT_REUSE.pop(arp_side, None)
    if _reuse is not None:
        _rc, _rv, _rs, _rd = _reuse
        _ok_cen = (orbit_center is None
                   or (Vector(orbit_center) - _rc).length <= _rs * 0.10)
        _ok_scl = (orbit_scale is None
                   or abs(orbit_scale - _rs) <= _rs * 0.15)
        if not (_ok_cen and _ok_scl and _rv):
            _reuse = None
    if _reuse is not None:
        cen, orbit_views, hand_scale, dist = _reuse
    else:
        cen, orbit_views, hand_scale, dist = _render_hand_orbit(
            mesh_obj, hw_render, ew, temp_dir, f"hand_{arp_side}", tip=_tip_pt,
            center=orbit_center, scale=orbit_scale, orbit_fwd=orbit_fwd)

    print(f"Landmark [{arp_side}]: hw={hw!r}  hw_render={tuple(round(x,3) for x in hw_render)}  "
          f"cen={cen!r}  hand_scale={hand_scale:.3f}")

    # 9-dim cam_feat: [right_vec(3), up_vec(3), side, sin(theta), cos(theta)]
    import math as _math
    side_f = 0.0 if arp_side == 'L' else 1.0

    per_view_joints  = []
    per_view_cameras = []

    for view_idx, (img_path, right_vec, up_vec) in enumerate(orbit_views):
        theta    = view_idx * (2.0 * _math.pi / _ORBIT_N_VIEWS)
        cam_feat = list(right_vec) + list(up_vec) + [side_f, _math.sin(theta), _math.cos(theta)]
        joints   = _detect_landmarks_onnx(img_path, cam_feat)
        n_good   = sum(1 for det in joints.values() if det[2] >= 0.4)
        if n_good < 14:
            continue
        forward_vec = right_vec.cross(up_vec).normalized()
        cam_pos     = cen + forward_vec * dist
        per_view_joints.append(joints)
        per_view_cameras.append((cam_pos, right_vec, up_vec, forward_vec))

    print(f"Landmark [{arp_side}]: {len(per_view_joints)}/{_ORBIT_N_VIEWS} views passed (14+ joints conf>=0.4)")
    if len(per_view_joints) < 2:
        return None

    joints_3d = _triangulate_ransac(per_view_joints, per_view_cameras, hw_render, hand_scale=hand_scale)

    # Convert landmark names to rig marker names
    marker_pos = {}
    for lm_name, pos in joints_3d.items():
        arp_base = _MP_TO_ARP.get(lm_name)
        if arp_base:
            marker_pos[f"{arp_base}_{arp_side}"] = pos

    # Graft missing joints from geometric fallback
    if ew is not None:
        geom_all = _place_fingers_geometric(hw, ew, arp_side)
        for lm_name, arp_base in _MP_TO_ARP.items():
            arp_key = f"{arp_base}_{arp_side}"
            if arp_key not in marker_pos and arp_key in geom_all:
                marker_pos[arp_key] = geom_all[arp_key]
                print(f"  {lm_name}: grafted from geometric")

    n = len(marker_pos)
    print(f"Landmark [{arp_side}]: {n}/20 markers placed")
    if n < 12:
        return None

    # Collapse check: require 3/5 fingers with plausible MCP->TIP length (>=20 mm).
    # LVT uses a stricter 30mm/4-of-5 gate; ONNX is more robust so we allow a
    # lower floor. 20mm still rejects true palm-bias collapse (~0-16mm) while
    # accepting borderline predictions on short fingers (e.g. index 27mm on A-pose).
    good_len = 0
    print(f"Landmark [{arp_side}]: MCP->TIP lengths:")
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        arp = _FINGER_TO_ARP.get(finger)
        if arp is None:
            continue
        tip_k = f"{arp['tip']}_{arp_side}"
        mcp_k = f"{arp['phal1']}_{arp_side}"
        tip = marker_pos.get(tip_k)
        mcp = marker_pos.get(mcp_k)
        flen = (tip - mcp).length if (tip and mcp) else 0.0
        print(f"  {finger}: {flen * 1000:.1f}mm")
        if flen >= 0.020:
            good_len += 1
    if good_len < 3:
        print(f"Landmark [{arp_side}]: collapsed ({good_len}/5 fingers >=2cm)  -- returning None")
        return None

    # -- Two-pass framing refinement ------------------------------------------
    # Pass 1 framed the orbit on the LVT tips (unreliable on hard poses -> wrong
    # crop -> index/middle collapse). Recompute the orbit centre as the centroid of
    # ALL 20 predicted joints - the SAME definition SaveHandData used at training
    # (and robust: ~16 good joints anchor it even if index/middle collapsed) - then
    # re-render once at that centre (same fwd & scale) so the model sees the framing
    # it was trained on. Keep whichever pass extends the fingers more (defensive:
    # a garbage pass-1 centroid can't win, and a collapsed pass-2 returns None).
    if _allow_two_pass and ew is not None and len(marker_pos) >= 16:
        # C20: centroid of ALL 20 markers (same as SaveHandData)
        c20 = sum(marker_pos.values(), Vector()) / len(marker_pos)
        
        # Fwd: from C20 toward elbow (same as SaveHandData)
        fwd_pass2 = (c20 - Vector(ew)).normalized()
        
        # Scale: max distance from C20 to ANY of the 20 markers (same as SaveHandData)
        max_dist = max((pos - c20).length for pos in marker_pos.values())
        scale_pass2 = max_dist * 2.5
        
        if orbit_center is None or (Vector(orbit_center) - c20).length > 0.01:
            res2 = detect_fingers_landmark(
                mesh_obj, hw, ew, temp_dir, arp_side,
                orbit_center=c20, orbit_scale=scale_pass2, orbit_fwd=fwd_pass2,
                _allow_two_pass=False)
            if res2:
                s1 = _finger_ext_sum(marker_pos, arp_side)
                s2 = _finger_ext_sum(res2, arp_side)
                print(f"Landmark [{arp_side}]: two-pass ext sum  pass1={s1*1000:.0f}mm  "
                      f"pass2={s2*1000:.0f}mm  -> using {'pass2' if s2 >= s1 else 'pass1'}")
                if s2 >= s1:
                    return res2   # already straightened/centered by the inner call

    _straighten_fingers(marker_pos, hw, ew, arp_side)
    _straighten_and_center_fingers(mesh_obj, marker_pos, arp_side)
    return marker_pos


# -- Fallback: tip heatmap model -----------------------------------------------

def detect_fingers_lvt(mesh_obj, hw, ew, temp_dir, arp_side,
                       palm_depth=0.25, knuckle_radius=0.90,
                       width_tolerance=1.0, straighten_clamp=0.15):
    """
    Runs the hand tips model to predict fingertip positions per orbit view,
    triangulates tips, then places phalange chains using geometric proportions.
    Returns {arp_marker_name: Vector} or None on failure.
    """
    if not is_lvt_available():
        return None

    _tip_obj = bpy.data.objects.get(f"MARKER_HAND_TIP_{arp_side}")
    if _tip_obj:
        _tip_pt = _tip_obj.location.copy()
    else:
        _fwd     = (Vector(hw) - Vector(ew)).normalized() if ew is not None else Vector((1.0, 0.0, 0.0))
        _arm_len = (Vector(hw) - Vector(ew)).length if ew is not None else 0.28
        _tip_pt  = Vector(hw) + _fwd * _arm_len * 0.65

    hw_render = _compute_render_hw(hw, ew, arp_side, mesh_obj=mesh_obj,
                                   tip=_tip_pt if _tip_obj else None)

    # Guard: a HAND_TIP marker dropped off the hand inflates the LVT frame scale
    # (a 1.2m "hand") and collapses detection. Validate it against the mesh extent
    # near the wrist (measured along the reliable forearm direction); if implausibly
    # far, replace it with a geometry estimate so framing degrades gracefully.
    if _tip_obj is not None and ew is not None:
        _arm_n = (Vector(hw) - Vector(ew)).normalized()
        _tlen  = (_tip_pt - Vector(hw)).length
        try:
            _ev  = mesh_obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
            _mw  = _ev.matrix_world
            _hwv = Vector(hw)
            _rad = max(0.50, _tlen * 1.5)
            _pr  = [(_mw @ v.co - _hwv).dot(_arm_n) for v in _ev.data.vertices
                    if (_mw @ v.co - _hwv).length < _rad]
            _mext = (max(_pr) - min(_pr)) if _pr else None
        except Exception:
            _mext = None
        if _mext is not None and _mext > 0.03 and _tlen > 2.5 * _mext:
            print(f"  LVT [{arp_side}]: HAND_TIP marker looks misplaced "
                  f"{_tlen*1000:.0f}mm vs mesh ~{_mext*1000:.0f}mm - using geometry")
            _tip_pt  = _hwv + _arm_n * _mext
            _tip_obj = None

    # -- Wrist-independent LVT orbit framing ---------------------------------
    # The default _render_hand_orbit derives cen/scale/fwd from hw_render + tip,
    # so the crop shifts when the wrist marker slides up/down the forearm - which
    # changes the detected tips, the stable params, and ultimately collapses the
    # landmark fingers. Anchor the framing on HAND_TIP + elbow ONLY (both stable),
    # so the wrist marker's position no longer affects detection.
    _ew_v = Vector(ew) if ew is not None else None
    if _tip_obj is not None and _ew_v is not None and (_tip_pt - _ew_v).length > 1e-4:
        _arm_dir   = (_tip_pt - _ew_v).normalized()
        # Hand length ~ 45% of elbow->fingertip (humanoid ratio ~1:2.2).
        _hand_len  = (_tip_pt - _ew_v).length * 0.45
        # Centre near the hand CENTROID (~45% back from the fingertip) to match the
        # training framing (which centres on the finger-marker centroid) -> correct
        # zoom. The thumb base / proximal palm are kept in frame not by pushing the
        # centre back (that over-zooms: scale = spread*2.5 grows with a proximal centre)
        # but by letting the spread MEASURE the proximal hand (see _mesh_hand_spread).
        _cen_lvt   = _tip_pt - _arm_dir * (_hand_len * 0.45)
        # Frame width. The arm-ratio length above is a reliable BASE, but it does not
        # know the LATERAL spread, so a wide/big hand clips its edge fingers (pinky)
        # out of frame. Measure the actual hand spread from the mesh and widen the
        # frame to contain it. We never go BELOW the arm-ratio base, so normal hands
        # (where the mesh measure is small) keep their previously-correct framing.
        _scale_lvt = _hand_len * 1.1
        _spread = _mesh_hand_spread(mesh_obj, _cen_lvt, _arm_dir,
                                    reach=max(_hand_len * 1.6, 0.05))
        if _spread is not None:
            # training recipe: scale = 2.5 * radius-from-centroid. _spread is that
            # radius measured from the mesh -> match the trained framing for big hands.
            _scale_lvt = max(_scale_lvt, _spread * 2.5)
        print(f"  [framing {arp_side}] arm_ratio={_hand_len*1.1:.3f} "
              f"spread={'%.3f' % _spread if _spread else 'None'} "
              f"-> scale={_scale_lvt:.3f}")
        cen, orbit_views, hand_scale, dist = _render_hand_orbit(
            mesh_obj, hw_render, ew, temp_dir, f"hand_{arp_side}", tip=_tip_pt,
            center=_cen_lvt, scale=_scale_lvt, orbit_fwd=_arm_dir)
    else:
        # Fallback: HAND_TIP or elbow missing - keep the original wrist-based framing.
        cen, orbit_views, hand_scale, dist = _render_hand_orbit(
            mesh_obj, hw_render, ew, temp_dir, f"hand_{arp_side}", tip=_tip_pt)
    # up_vec in each orbit view equals fwd (the arm direction used for the LVT renders).
    # Passing this to ONNX ensures both pipelines orbit around the same axis.
    lvt_fwd = orbit_views[0][2] if orbit_views else None

    # Cache the render set for the landmark stage: its framing is seeded from
    # THIS stage's results, so when the two framings agree the second 8-view
    # render pass is pure duplicate work (detection is render-bound).
    if orbit_views:
        _ORBIT_REUSE[arp_side] = (cen, orbit_views, hand_scale, dist)

    print(f"LVT [{arp_side}]: hw={hw!r}  hw_render={tuple(round(x,3) for x in hw_render)}  "
          f"cen={cen!r}  hand_scale={hand_scale:.3f}")

    # 8-dim cam_feat: [right_vec(3), up_vec(3), side_L, side_R] one-hot
    side_feat = [1.0, 0.0] if arp_side == 'L' else [0.0, 1.0]

    per_view_tips    = []
    per_view_cameras = []

    for img_path, right_vec, up_vec in orbit_views:
        cam_feat    = list(right_vec) + list(up_vec) + side_feat
        tips        = _detect_tips_onnx(img_path, cam_feat)
        if len(tips) < 3:
            continue
        forward_vec = right_vec.cross(up_vec).normalized()
        cam_pos     = cen + forward_vec * dist
        per_view_tips.append(tips)
        per_view_cameras.append((cam_pos, right_vec, up_vec, forward_vec))

    print(f"LVT [{arp_side}]: {len(per_view_tips)}/{_ORBIT_N_VIEWS} views had 3+ tips")
    if len(per_view_tips) < 2:
        return None

    # Build BVH before triangulation so snap-to-mesh happens inside the grid search.
    _hand_bvh = _build_bvh(mesh_obj)

    tips_3d = _triangulate_tips_ls(per_view_tips, per_view_cameras, hw_render, ew, arp_side,
                                    hand_scale=hand_scale, bvh=_hand_bvh)

    # Count raw 2D detections per finger (for diagnostics).
    raw_counts = {fname: 0 for fname in _FINGER_NAMES}
    for tips in per_view_tips:
        for fname in tips:
            raw_counts[fname] += 1

    # Graft geometric template tip for any finger whose triangulation failed.
    # Snap already ran inside _triangulate_tips_ls so grafted tips need another
    # snap pass below to pull them to the actual surface.
    if ew is not None:
        geom_all = _place_fingers_geometric(hw, ew, arp_side)
        for fname in _FINGER_NAMES:
            if fname not in tips_3d:
                arp_tip  = f"{_FINGER_TO_ARP[fname]['tip']}_{arp_side}"
                geom_tip = geom_all.get(arp_tip)
                if geom_tip is not None and (geom_tip - Vector(hw)).length < _MAX_TIP_DIST:
                    n2d = raw_counts.get(fname, 0)
                    reason = f"{n2d} views but grid failed" if n2d >= 3 else f"only {n2d} detections"
                    # Snap the geometric tip to the mesh surface.
                    hit, _n, _idx, snap_dist = _hand_bvh.find_nearest(geom_tip)
                    if hit is not None and snap_dist < 0.15:
                        tips_3d[fname] = Vector(hit)
                        print(f"  {fname}: grafted+snapped ({reason}, snap={snap_dist:.3f}m)")
                    else:
                        tips_3d[fname] = geom_tip
                        print(f"  {fname}: grafted tip from geometric ({reason})")

    n_detected = len(tips_3d)
    hw_v = Vector(hw)
    print(f"LVT [{arp_side}]: {n_detected}/5 tips detected")
    if n_detected < 3:
        return None

    # Deduplicate snapped tips.
    # Adjacent fingers (index/middle, middle/ring, ring/pinky) that collapse onto
    # the same mesh vertex are pushed apart to 12 mm so both survive chain-walk.
    # Non-adjacent collapses are genuine duplicates  -- keep the one with the better
    # spread score (farther from its neighbours) and delete the other.
    _ADJACENT_PAIRS = {
        frozenset(('thumb', 'index')),
        frozenset(('index', 'middle')),
        frozenset(('middle', 'ring')),
        frozenset(('ring', 'pinky')),
    }
    _COLLAPSE_DIST = 0.005   # < 5 mm = collapsed
    _PUSH_TARGET   = 0.012   # push to 12 mm separation

    tip_names = list(tips_3d.keys())
    for i in range(len(tip_names)):
        for j in range(i + 1, len(tip_names)):
            fa, fb = tip_names[i], tip_names[j]
            if fa not in tips_3d or fb not in tips_3d:
                continue
            dist = (tips_3d[fa] - tips_3d[fb]).length
            if dist >= _COLLAPSE_DIST:
                continue

            if frozenset((fa, fb)) in _ADJACENT_PAIRS:
                # Push apart: keep midpoint, spread along the collapse vector
                mid  = (tips_3d[fa] + tips_3d[fb]) * 0.5
                diff = tips_3d[fa] - tips_3d[fb]
                axis = diff.normalized() if diff.length > 1e-6 else Vector((0.0, 1.0, 0.0))
                tips_3d[fa] = mid + axis * (_PUSH_TARGET * 0.5)
                tips_3d[fb] = mid - axis * (_PUSH_TARGET * 0.5)
                print(f"  LVT [{arp_side}]: {fa}/{fb} collapsed  -- pushed apart to {_PUSH_TARGET*1000:.0f}mm")
            else:
                # Non-adjacent: genuine duplicate  -- remove the more isolated tip
                def _min_dist(name):
                    others = [v for k, v in tips_3d.items() if k != name]
                    return min((tips_3d[name] - o).length for o in others) if others else 0
                if _min_dist(fa) < _min_dist(fb):
                    del tips_3d[fa]
                    print(f"  LVT [{arp_side}]: {fa} tip removed (collapsed onto {fb})")
                else:
                    del tips_3d[fb]
                    print(f"  LVT [{arp_side}]: {fb} tip removed (collapsed onto {fa})")
                break

    # Shift walk target 25% toward tip centroid  -- makes palm transition detection
    # work regardless of where body-detection placed the wrist marker.
    chain_hw = _estimate_palm_center(hw, tips_3d, ew=ew, arp_side=arp_side, palm_depth=palm_depth)
    print(f"  LVT [{arp_side}]: palm_center={tuple(round(x,3) for x in chain_hw)}")

    marker_pos  = {}
    joint_names = ['tip', 'phal3', 'phal2', 'phal1']

    for fname in _FINGER_NAMES:
        if fname not in tips_3d:
            continue
        chain     = _phalange_chain_geom(tips_3d[fname], chain_hw, ew, arp_side, fname,
                                         bvh=_hand_bvh, hand_obj=mesh_obj,
                                         width_tolerance=width_tolerance)
        arp_names = _FINGER_TO_ARP[fname]
        for j_idx, j_name in enumerate(joint_names):
            marker_pos[f"{arp_names[j_name]}_{arp_side}"] = chain[j_idx]

    # Geometric knuckle idealization: project MCPs onto anatomical arc, snap to mesh.
    raw_mcps = {}
    for fname in _FINGER_NAMES:
        if fname not in tips_3d:
            continue
        mcp_phal = "phal2" if fname == "thumb" else "phal1"
        mcp_key  = f"{_FINGER_TO_ARP[fname][mcp_phal]}_{arp_side}"
        if mcp_key in marker_pos:
            raw_mcps[fname] = marker_pos[mcp_key]
    if raw_mcps:
        refined_mcps = _refine_mcp_positions(raw_mcps, tips_3d, chain_hw,
                                             bvh=_hand_bvh, knuckle_radius=knuckle_radius)
        for fname, mcp_pos in refined_mcps.items():
            mcp_phal = "phal2" if fname == "thumb" else "phal1"
            marker_pos[f"{_FINGER_TO_ARP[fname][mcp_phal]}_{arp_side}"] = mcp_pos

        # Recompute PIP/DIP from the corrected MCP positions
        phalanges = _recompute_phalanges(refined_mcps, tips_3d, bvh=_hand_bvh)
        for fname, pts in phalanges.items():
            arp = _FINGER_TO_ARP[fname]
            if fname == 'thumb':
                # IP joint -> THUMB_3 (phal3)
                if pts['dip'] is not None:
                    marker_pos[f"{arp['phal3']}_{arp_side}"] = pts['dip']
            else:
                if pts['pip'] is not None:
                    marker_pos[f"{arp['phal2']}_{arp_side}"] = pts['pip']
                if pts['dip'] is not None:
                    marker_pos[f"{arp['phal3']}_{arp_side}"] = pts['dip']

    # Hand-normal plane straighten for PIP/DIP (runs before arm-dir pass)
    _straighten_finger_phalanges(marker_pos, tips_3d, chain_hw, arp_side,
                                  straighten_clamp=straighten_clamp)
    _straighten_fingers(marker_pos, hw, ew, arp_side)
    _straighten_and_center_fingers(mesh_obj, marker_pos, arp_side)

    # Stable orbit parameters from tip geometry - no hw_render dependency.
    # fwd = ew->tip_centroid (same formula used in SaveHandData so training and
    # inference always share identical camera conditioning regardless of where
    # the user placed the wrist marker).
    # Set True when the tip geometry has genuinely collapsed (all tips onto one
    # blob). A collapsed chain is unrecoverable - the push-apart/scale-fallback
    # only makes it LOOK plausible to _has_spread, then it can get accepted as
    # primary or leaked into anchoring (see project_lvt_tip_collapse). When this
    # is set we null the returned chain so the caller fails closed to ONNX.
    _tip_collapsed = False
    if len(tips_3d) >= 3:
        _tip_c = sum(tips_3d.values(), Vector()) / len(tips_3d)
        if ew is not None:
            _sfwd = (_tip_c - Vector(ew)).normalized()
        else:
            _sfwd = Vector(lvt_fwd).normalized() if lvt_fwd is not None else None
        if _sfwd is not None:
            # Match SaveHandData's orbit framing EXACTLY so the ONNX inference crop
            # matches the training crop. SaveHandData uses: center = centroid of the
            # points (NO backward shift), scale = max(dist from centroid) * 2.5.
            # The previous inference code shifted the center backward by 2*_mfwd,
            # which inflated _mdst and over-shrank the hand in frame (scale 0.272 vs
            # true ~0.229) -> the model saw a smaller hand than it trained on and
            # missed tips (grafted geometric -> collapsed fingers), even on trained
            # characters. (Inputs still differ - 5 detected tips here vs 20 corrected
            # markers at save - but the formula/center now match, which is the fix.)
            _scen        = _tip_c
            _mdst        = max((Vector(t) - _scen).length for t in tips_3d.values())
            stable_cen   = _scen
            stable_scale = _mdst * 2.5
            stable_fwd   = _sfwd

            # Collapse guard - LOWER bound only. Training crops were rendered at
            # SaveHandData's stable_scale, so inference must trust stable_scale too
            # (NOT render_scale). A LARGE stable_scale is legitimate - e.g. a spread
            # thumb pulls _mdst up - and clamping it down to render_scale shrinks the
            # hand below the trained size and the model collapses. So there is NO
            # upper clamp. The only failure mode worth guarding is genuine tip
            # COLLAPSE (triangulation merges tips), which makes stable_scale fall far
            # below the geometric hand length - then fall back to the reliable render
            # scale. Plus a wide absolute sanity bound for absurd values.
            if hand_scale and stable_scale < hand_scale * 0.70:
                print(f"  LVT [{arp_side}]: stable_scale {stable_scale:.3f}m collapsed "
                      f"(<70% of render {hand_scale:.3f}m) - tips merged, rejecting LVT chain")
                stable_scale   = hand_scale
                _tip_collapsed = True
            elif stable_scale < 0.03:
                print(f"  LVT [{arp_side}]: stable_scale {stable_scale:.3f}m below floor "
                      f"- tips merged, rejecting LVT chain")
                stable_scale   = hand_scale if hand_scale else stable_scale
                _tip_collapsed = True
            elif stable_scale > 0.80:
                print(f"  LVT [{arp_side}]: stable_scale {stable_scale:.3f}m out of bounds "
                      f"- using render scale {hand_scale:.3f}m")
                stable_scale = hand_scale if hand_scale else stable_scale

            print(f"  LVT [{arp_side}]: stable_cen={tuple(round(x,3) for x in stable_cen)}  "
                  f"stable_scale={stable_scale:.3f}m  (geometry-based)")
        else:
            stable_cen, stable_scale, stable_fwd = cen, hand_scale, lvt_fwd
    else:
        stable_cen, stable_scale, stable_fwd = cen, hand_scale, lvt_fwd

    # Fail closed on collapse: discard the marker chain (so it is neither accepted
    # as primary nor kept as lvt_raw for tip anchoring) but KEEP the raw per-view
    # tips so the mesh fallback still has something to work with if ONNX also fails.
    if _tip_collapsed:
        print(f"  LVT [{arp_side}]: tip geometry collapsed - failing closed to ONNX (chain discarded)")
        marker_pos = None

    return marker_pos, per_view_tips, per_view_cameras, stable_cen, stable_scale, stable_fwd


# -- Mesh-aware finger fallback ------------------------------------------------

def _place_fingers_from_mesh(mesh_obj, hw, ew, arp_side,
                             per_view_tips=None, per_view_cameras=None):
    """
    Find 5 fingertips from mesh vertices, using 2D tip detections to establish
    finger identity. Falls back to geometric if no 2D data is available.
    """
    hw_v = Vector(hw)
    if ew is not None and (hw_v - Vector(ew)).length > 1e-4:
        arm_dir = (hw_v - Vector(ew)).normalized()
    else:
        arm_dir = Vector((1.0 if arp_side == 'L' else -1.0, 0.0, 0.0))

    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj  = depsgraph.objects.get(mesh_obj.name)
    if eval_obj:
        src_mesh = eval_obj.data
        mw       = eval_obj.matrix_world
    else:
        src_mesh = mesh_obj.data
        mw       = mesh_obj.matrix_world

    # -- 2D-guided identity assignment (primary) -----------------------------
    if per_view_tips and per_view_cameras and len(per_view_tips) >= 2:
        cen, _fwd, _ref, _perp, hand_scale, _dist = _orbit_cam_basis(hw, ew)

        finger_votes = {fname: [] for fname in _FINGER_NAMES}

        for v in src_mesh.vertices:
            v_world = mw @ v.co
            vec     = v_world - hw_v
            along   = vec.dot(arm_dir)
            if along < 0.02 or along > 0.35:
                continue
            if (vec - arm_dir * along).length > 0.18:
                continue

            view_votes = {fname: 0 for fname in _FINGER_NAMES}
            for tips, (cam_pos, right_vec, up_vec, forward_vec) in zip(per_view_tips, per_view_cameras):
                dp    = v_world - cen
                x_cam = dp.dot(right_vec)
                y_cam = dp.dot(up_vec)
                px    = (x_cam / hand_scale + 0.5) * _HAND_IMG_SIZE
                py    = (0.5 - y_cam / hand_scale) * _HAND_IMG_SIZE

                best_dist = float('inf')
                best_name = None
                for fname, (tip_px, tip_py) in tips.items():
                    d = ((px - tip_px) ** 2 + (py - tip_py) ** 2) ** 0.5
                    if d < best_dist and d < 25:
                        best_dist = d
                        best_name = fname
                if best_name:
                    view_votes[best_name] += 1

            total_votes = sum(view_votes.values())
            if total_votes < 2:
                continue
            best_finger = max(view_votes, key=view_votes.get)
            if view_votes[best_finger] >= total_votes * 0.6:
                finger_votes[best_finger].append(v_world)

        tips_3d = {}
        for fname in _FINGER_NAMES:
            cluster = finger_votes[fname]
            if len(cluster) < 3:
                continue
            xs = sorted(v.x for v in cluster)
            ys = sorted(v.y for v in cluster)
            zs = sorted(v.z for v in cluster)
            tips_3d[fname] = Vector((xs[len(xs) // 2], ys[len(ys) // 2], zs[len(zs) // 2]))

        if len(tips_3d) >= 3:
            _mesh_bvh2 = _build_bvh(mesh_obj)
            chain_hw   = _estimate_palm_center(hw, tips_3d, ew=ew, arp_side=arp_side)
            marker_pos  = {}
            joint_names = ['tip', 'phal3', 'phal2', 'phal1']
            for fname, tip in tips_3d.items():
                chain = _phalange_chain_geom(tip, chain_hw, ew, arp_side, fname,
                                             bvh=_mesh_bvh2, hand_obj=mesh_obj)
                arp   = _FINGER_TO_ARP[fname]
                for j_idx, j_name in enumerate(joint_names):
                    marker_pos[f"{arp[j_name]}_{arp_side}"] = chain[j_idx]
            _straighten_fingers(marker_pos, hw, ew, arp_side)
            _straighten_and_center_fingers(mesh_obj, marker_pos, arp_side)
            return marker_pos

    # -- Fallback: geometric -------------------------------------------------
    return _place_fingers_geometric(hw, ew, arp_side)


# -- Post-detection symmetry enforcement --------------------------------------

def resnap_tips_to_finger_end(marker_pos, mesh_obj, arp_side, radius=0.020):
    """
    Move each fingertip marker from the fingernail to the actual distal END of the
    finger. The straight-ray tip triangulation overshoots a curled fingertip, and
    the mesh-snap then lands on the NEAREST surface (the nail), short of the end.

    The finger end is the mesh vertex (within `radius` of the current tip) that is
    FARTHEST from the DIP joint - the distal extreme of the last phalange. This is
    direction-free, so it follows the curl automatically (nail points sit between
    the DIP and the end; the true tip is the farthest distal point).

    Thin, closely-spaced fingers: a fixed-radius sphere around the tip reaches into
    the NEIGHBORING finger, and a neighbor vertex can be farther from this finger's
    DIP than this finger's own end -> the tip jumps sideways onto the wrong finger.
    Guard: reject candidates whose lateral (perpendicular) offset from this finger's
    own DIP->tip axis exceeds half the spacing to the nearest other finger's DIP.
    A neighbor-finger vertex has a large lateral offset; a distal/curl vertex on the
    correct finger stays near the axis, so it survives the gate.
    """
    try:
        from mathutils.kdtree import KDTree
        depsgraph = bpy.context.evaluated_depsgraph_get()
        obj_eval  = mesh_obj.evaluated_get(depsgraph)
        mw        = obj_eval.matrix_world
        verts     = obj_eval.data.vertices
        kd = KDTree(len(verts))
        for i, v in enumerate(verts):
            kd.insert(mw @ v.co, i)
        kd.balance()
    except Exception as _e:
        print(f"  [tip-end] skipped ({_e})")
        return

    # DIP positions per finger, to derive adaptive lateral gates from spacing.
    dips = {}
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        arp = _FINGER_TO_ARP.get(finger)
        if arp:
            dips[finger] = marker_pos.get(f"{arp['phal3']}_{arp_side}")

    # Across-fingers direction: the distal snap may only extend the tip forward/into
    # depth, never sideways toward a touching neighbour. We strip this component.
    _imcp = marker_pos.get(f"{_FINGER_TO_ARP['index']['phal1']}_{arp_side}")
    _pmcp = marker_pos.get(f"{_FINGER_TO_ARP['pinky']['phal1']}_{arp_side}")
    across = None
    if _imcp is not None and _pmcp is not None and (_imcp - _pmcp).length > 1e-5:
        across = (_imcp - _pmcp).normalized()

    moved = 0
    # Thumb skipped: its tip comes from the reliable LVT tip model, and on a curled
    # thumb the tip sits physically near the knuckle/palm, so the distal-vertex snap
    # grabs a knuckle vertex (the thumb's perp gate is loose - no close neighbour).
    for finger in ("index", "middle", "ring", "pinky"):
        arp = _FINGER_TO_ARP.get(finger)
        if not arp:
            continue
        tip = marker_pos.get(f"{arp['tip']}_{arp_side}")
        dip = dips.get(finger)
        if tip is None or dip is None:
            continue

        axis = tip - dip
        if axis.length < 1e-6:
            continue
        axis = axis.normalized()

        # Adaptive lateral gate: half the distance to the nearest OTHER finger's DIP,
        # clamped to a sane finger-width range. Tightens automatically for thin hands.
        nearest = min((dip - d).length
                      for f2, d in dips.items() if f2 != finger and d is not None)
        max_perp = max(0.005, min(0.012, nearest * 0.45))

        best = None
        best_along = (tip - dip).dot(axis)   # require the end to be at least this distal
        for co, _idx, _dist in kd.find_range(tip, radius):
            vec   = Vector(co) - dip
            along = vec.dot(axis)
            perp  = (vec - axis * along).length
            if perp > max_perp:
                continue                      # on a neighbouring finger - reject
            if along > best_along:
                best_along = along
                best = Vector(co)
        if best is not None:
            move = best - tip
            # Extend distally but no sideways drift. Thumb excluded: its "sideways"
            # is not the index->pinky axis (it's rotated ~90deg), so stripping that
            # would cancel the thumb's real placement.
            if across is not None and finger != "thumb":
                move = move - across * move.dot(across)
            marker_pos[f"{arp['tip']}_{arp_side}"] = tip + move
            moved += 1
    if moved:
        print(f"  [tip-end {arp_side}]: re-snapped {moved}/5 tips to finger end")


def snap_joints_inside_mesh(marker_pos, mesh_obj, arp_side, max_pull=0.035):
    """
    Pull finger joints that ended OUTSIDE the mesh back just inside the skin.
    Bent/curved fingers can leave a silhouette-placed DIP floating off the
    finger; the depth pass can't always seat those (known hard case — its
    cross-section ray misses on sharp curls), but a marker OUTSIDE the volume
    is always wrong regardless of pose. Only joints measurably outside the
    surface (and within max_pull of it) are touched, so correctly-seated
    joints are never moved.
    """
    bvh = _build_bvh(mesh_obj)
    if bvh is None:
        return

    # Wrong-finger guard: find_nearest snaps to the nearest surface, which on a
    # small/thin hand (fingers sitting close together) can pull an outside joint
    # onto a NEIGHBOURING finger - a lateral cross-finger drift no other pass
    # introduces. Build the knuckle-row across-axis and each finger's MCP coord on
    # it, then reject any snap whose destination lands in a different finger's lane.
    across = None
    mcp_coord = {}
    i_mcp = marker_pos.get(f"{_FINGER_TO_ARP['index']['phal1']}_{arp_side}")
    p_mcp = marker_pos.get(f"{_FINGER_TO_ARP['pinky']['phal1']}_{arp_side}")
    if i_mcp is not None and p_mcp is not None and (i_mcp - p_mcp).length > 1e-4:
        across = (i_mcp - p_mcp).normalized()
        for f in ("index", "middle", "ring", "pinky"):
            m = marker_pos.get(f"{_FINGER_TO_ARP[f]['phal1']}_{arp_side}")
            if m is not None:
                mcp_coord[f] = m.dot(across)

    def _crosses_finger(finger, cand):
        # True when `cand` sits nearer a DIFFERENT finger's MCP lane than its own.
        if across is None or finger not in mcp_coord:
            return False
        c   = cand.dot(across)
        own = abs(c - mcp_coord[finger])
        return any(g != finger and abs(c - gc) < own for g, gc in mcp_coord.items())

    # CHORD ownership: the MCP-lane test alone misses convergent fingers - at
    # PIP level the fingers sit at different across-positions than their
    # knuckles, so a PIP in the inter-finger crease snapped to the NEIGHBOUR's
    # side wall while still passing the MCP-lane test (Hand_A: index PIP
    # jumped 17mm onto the middle finger in this very pass). The destination
    # must also stay nearest THIS finger's own MCP->TIP chord.
    chords = {}
    for f2 in ("index", "middle", "ring", "pinky"):
        a2 = _FINGER_TO_ARP[f2]
        m2 = marker_pos.get(f"{a2['phal1']}_{arp_side}")
        t2 = marker_pos.get(f"{a2['tip']}_{arp_side}")
        if m2 is not None and t2 is not None and (t2 - m2).length > 1e-5:
            chords[f2] = (m2, t2)

    def _seg_dist(p, seg):
        a, b = seg
        ab = b - a
        t = max(0.0, min(1.0, (p - a).dot(ab) / ab.length_squared))
        return (p - (a + ab * t)).length

    def _wrong_chord(finger, cand):
        if finger not in chords:
            return False
        own = _seg_dist(cand, chords[finger])
        return any(g != finger and _seg_dist(cand, c2) < own
                   for g, c2 in chords.items())

    moved = 0
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        arp = _FINGER_TO_ARP.get(finger)
        if not arp:
            continue
        for part in ("phal1", "phal2", "phal3", "tip"):
            k = f"{arp[part]}_{arp_side}"
            p = marker_pos.get(k)
            if p is None:
                continue
            loc, nrm, _i, d = bvh.find_nearest(p)
            if loc is None or nrm is None or d is None:
                continue
            outside = (p - Vector(loc)).dot(Vector(nrm)) > 0.0005
            if outside and d <= max_pull:
                cand = Vector(loc) - Vector(nrm) * min(0.003, d)
                # Don't let the snap cross into a neighbouring finger. Thumb's lane
                # overlaps the index on the across-axis, so it is exempt.
                if finger != "thumb" and (_crosses_finger(finger, cand)
                                          or _wrong_chord(finger, cand)):
                    continue
                marker_pos[k] = cand
                moved += 1
    if moved:
        print(f"  [inside {arp_side}]: pulled {moved} joint(s) back inside the mesh")


# Interior-chain bend (angle between MCP->PIP and PIP->DIP) above which a
# finger counts as CURLED and its tip is left on the model's trained placement.
# Calibrated on the regression scenes: straight hands top out ~10deg
# (A-pose ring 9.6, Hand_B <=7), genuinely curled hands start ~16deg
# (UntrainedHand 16-35, Infinity 21-38). 13deg sits cleanly between, so the
# straighten-reseat keeps firing UNCHANGED on every straight hand.
_CURL_BEND_DEG = 13.0


def reseat_offaxis_tips(marker_pos, mesh_obj, arp_side, bvh=None):
    """
    Snap a fingertip that sits on the SIDE/TOP surface of the fingertip back
    onto the finger's own chain axis (ARP-voxel-style centreline, done with
    one ray). When MCP/PIP/DIP are centred and collinear but the TIP hugs the
    dorsal or side surface (Hand_B: middle/ring/thumb tips on the top of the
    fingertip - same on both engines), the chain's own distal direction is
    more reliable than any surface extreme: ray-cast from the DIP along
    (DIP - PIP) to find where the finger tube ENDS along the centreline and
    seat the tip just inside that apex. Gated on the tip being clearly
    OFF-AXIS (perpendicular offset > 26% of the distal phalange) so
    well-placed tips - every approved hand - are untouched, and on the ray
    exiting at a plausible distal distance so curled fingers (where the
    straight extension leaves through the dorsal wall early) are left alone.

    CURVED-FINGER GUARD: the straight-centreline reseat is only correct when the
    interior chain is itself roughly collinear. On a genuinely CURLED finger the
    distal phalange should CONTINUE the bend, so the tip is legitimately off the
    straight PIP->DIP line; straightening it flattens the curl out of the
    fingertip. Fingers whose interior chain is bent past `_CURL_BEND_DEG` keep
    the detector's trained tip.
    """
    if not marker_pos:
        return
    if bvh is None:
        bvh = _build_bvh(mesh_obj)
    if bvh is None:
        return
    moved = []
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        arp = _FINGER_TO_ARP.get(finger)
        if not arp:
            continue
        pip = marker_pos.get(f"{arp['phal2']}_{arp_side}")
        dip = marker_pos.get(f"{arp['phal3']}_{arp_side}")
        tip = marker_pos.get(f"{arp['tip']}_{arp_side}")
        if pip is None or dip is None or tip is None:
            continue
        d = dip - pip
        if d.length < 1e-5:
            continue
        axis = d.normalized()            # CHAIN axis (PIP->DIP): OFF-AXIS DETECTION only
        seg = tip - dip
        seglen = seg.length
        if seglen < 1e-5:
            continue
        # Curved-finger guard: if the interior chain (MCP->PIP vs PIP->DIP) is
        # already bent, the finger is curled and its tip belongs off the straight
        # centreline (it follows the curl into the fingertip). Trust the detector.
        mcp = marker_pos.get(f"{arp['phal1']}_{arp_side}")
        if mcp is not None:
            pm = pip - mcp
            if pm.length > 1e-5:
                bend = math.degrees(pm.angle(d, 0.0))
                if bend > _CURL_BEND_DEG:
                    print(f"  [tip-axis {arp_side}] {finger}: interior chain "
                          f"curled ({bend:.0f}deg > {_CURL_BEND_DEG:.0f}) -- tip "
                          f"follows the curl, left as detected")
                    continue
        perp = seg - axis * seg.dot(axis)
        _off = perp.length / seglen
        # 0.26, not higher: a tip resting on the TOP surface of a chubby
        # fingertip is offset by ~one finger radius =~ 0.3x the distal
        # phalange, right at the old 0.35 gate (Hand_B ring/thumb slipped
        # under it). The frozen regression scenes guard the other side: their
        # pad-seated tips must stay under this gate.
        if _off < 0.26:
            continue                     # tip already on the chain axis
        hit, _n, _i, _dh = bvh.ray_cast(dip + axis * 0.002, axis, 2.0 * seglen)
        if hit is None:
            print(f"  [tip-axis {arp_side}] {finger}: off-axis ({_off:.2f}) "
                  f"but centreline ray found no finger end -- left alone")
            continue
        L = (Vector(hit) - dip).length
        if not (0.75 * seglen <= L <= 1.6 * seglen):
            print(f"  [tip-axis {arp_side}] {finger}: off-axis ({_off:.2f}) "
                  f"but exit at {L/seglen:.2f}x of the distal phalange -- "
                  f"left alone (curl / stray hit)")
            continue
        _ndot = _n.normalized().dot(axis) if _n is not None and _n.length > 1e-6 else 1.0
        _seg_ratio = d.length / seglen
        print(f"  [tip-axis {arp_side}] {finger}: off={_off:.2f} exit={L/seglen:.2f}x "
              f"cap_ndot={_ndot:.2f} pipdip/seg={_seg_ratio:.2f} "
              f"tip_to_exit={(Vector(hit) - tip).length / seglen:.2f}x")
        marker_pos[f"{arp['tip']}_{arp_side}"] = dip + axis * (L * 0.88)
        moved.append(finger)
    if moved:
        print(f"  [tip-axis {arp_side}]: re-seated {len(moved)} off-axis "
              f"tip(s) onto the finger centreline ({', '.join(moved)})")


def contain_finger_chains(marker_pos, mesh_obj, arp_side, samples=12,
                          rounds=8, bvh=None):
    """
    Keep each finger POLYLINE inside the mesh - not just the joints. Every
    earlier pass validates joint POSITIONS, but a straight bone SEGMENT
    between two inside joints can still exit the mesh: a curled finger's
    DIP->TIP chord cuts across the crease, and a laterally-drifted DIP sends
    its segment across the inter-finger gap (the "DIP on the middle finger" /
    Kid-hand class - measured 8mm-deep outside runs on every finger).

    For each chain, repeatedly find the DEEPEST outside sample on any segment
    and pull that point back just inside the skin by moving the segment's
    endpoints, weighted by mobility: TIP never moves (trusted evidence), MCP
    moves at half rate (knuckle-row anchored), PIP/DIP move freely. Per-joint
    move budget caps runaway corrections, and the knuckle-lane guard rejects
    any correction that would push a joint into a NEIGHBOUR finger's lane, so
    the pass can fix depth/curl violations but never create cross-finger
    drift. Chains already fully inside are untouched.
    """
    if not marker_pos:
        return
    if bvh is None:
        bvh = _build_bvh(mesh_obj)
    if bvh is None:
        return

    across = None
    mcp_coord = {}
    i_mcp = marker_pos.get(f"{_FINGER_TO_ARP['index']['phal1']}_{arp_side}")
    p_mcp = marker_pos.get(f"{_FINGER_TO_ARP['pinky']['phal1']}_{arp_side}")
    if i_mcp is not None and p_mcp is not None and (i_mcp - p_mcp).length > 1e-4:
        across = (i_mcp - p_mcp).normalized()
        for f in ("index", "middle", "ring", "pinky"):
            m = marker_pos.get(f"{_FINGER_TO_ARP[f]['phal1']}_{arp_side}")
            if m is not None:
                mcp_coord[f] = m.dot(across)

    def _crosses(finger, cand):
        if across is None or finger not in mcp_coord:
            return False
        c   = cand.dot(across)
        own = abs(c - mcp_coord[finger])
        return any(g != finger and abs(c - gc) < own
                   for g, gc in mcp_coord.items())

    # Chord ownership (see snap_joints_inside_mesh): MCP lanes miss convergent
    # fingers at PIP/DIP level; a correction must also stay nearest THIS
    # finger's own MCP->TIP chord.
    _chords = {}
    for f2 in ("index", "middle", "ring", "pinky"):
        a2 = _FINGER_TO_ARP[f2]
        m2 = marker_pos.get(f"{a2['phal1']}_{arp_side}")
        t2 = marker_pos.get(f"{a2['tip']}_{arp_side}")
        if m2 is not None and t2 is not None and (t2 - m2).length > 1e-5:
            _chords[f2] = (m2, t2)

    def _seg_dist(p, seg):
        a, b = seg
        ab = b - a
        t = max(0.0, min(1.0, (p - a).dot(ab) / ab.length_squared))
        return (p - (a + ab * t)).length

    def _wrong_chord(finger, cand):
        if finger not in _chords:
            return False
        own = _seg_dist(cand, _chords[finger])
        return any(g != finger and _seg_dist(cand, c2) < own
                   for g, c2 in _chords.items())

    _PARTS    = ("phal1", "phal2", "phal3", "tip")
    # Only PIP/DIP move. TIP is trusted evidence; MCP is knuckle-row anchored
    # and letting it drift here CROWDED the row (3mm MCP gaps -> the quality
    # takeover replaced the whole hand with the geometric engine's result,
    # silently discarding this pass's work). Chain containment = bending the
    # interior of the chain, never its anchors.
    _MOBILITY = (0.0, 1.0, 1.0, 0.0)      # MCP, PIP, DIP, TIP

    def _out_depth(p):
        loc, nrm, _i, _d = bvh.find_nearest(p)
        if loc is None or nrm is None:
            return 0.0
        return max(0.0, (p - Vector(loc)).dot(Vector(nrm)))
    n_chains = 0
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        arp = _FINGER_TO_ARP.get(finger)
        if not arp:
            continue
        keys = [f"{arp[p]}_{arp_side}" for p in _PARTS]
        pts = [marker_pos.get(k) for k in keys]
        if any(p is None for p in pts):
            continue
        flen = (pts[3] - pts[0]).length
        if flen < 1e-5:
            continue
        budget = [0.30 * flen] * 4
        changed = False
        # TIP containment first, so the segment relaxation below adapts to the
        # corrected tip. The tip's DIRECTION is trusted evidence, but a tip
        # OUTSIDE the mesh is objectively wrong (nail-plate hands: the [pad]
        # seating misses and the tip floats 6-7mm off the nail, dragging the
        # whole DIP->TIP segment out with it). Pull it just inside via the
        # nearest surface; when that would cross into a neighbour's lane,
        # retreat along the finger's own DIP->TIP axis instead.
        tip_out = _out_depth(pts[3])
        if tip_out > 0.0015:
            loc, nrm, _i, _d = bvh.find_nearest(pts[3])
            cand = (Vector(loc) - Vector(nrm) * 0.0015) if loc is not None else None
            if cand is not None and (finger == "thumb"
                                     or not _crosses(finger, cand)):
                if _out_depth(cand) < tip_out:
                    pts[3] = cand
                    changed = True
            else:
                back = pts[2] - pts[3]
                if back.length > 1e-6:
                    cand = pts[3] + back.normalized() * min(tip_out + 0.0015,
                                                            0.25 * flen)
                    if _out_depth(cand) < tip_out:
                        pts[3] = cand
                        changed = True
        # ALONG-CHORD anchor: contain's job is DEPTH (a curled chain bows
        # dorsal-palmar back inside the volume). On a fat/stubby finger the
        # nearest-surface pull also has a large ALONG-finger component that
        # slides the DIP toward the tip, collapsing the distal phalange in one
        # pass (chibi middle dist/mid 0.66 -> 0.26; long-nail index DIP moved
        # 19mm). Record each mobile joint's position along the MCP->TIP chord
        # (both chord ends are immobile during the rounds) and restore it
        # afterwards, keeping only the perpendicular depth correction.
        _chord = pts[3] - pts[0]
        _chord_axis = _chord.normalized() if _chord.length > 1e-6 else None
        _along0 = ([(pts[i] - pts[0]).dot(_chord_axis) for i in range(4)]
                   if _chord_axis is not None else None)
        for _r in range(rounds):
            # Accumulate a correction per JOINT from ALL violating samples at
            # once (relaxation), instead of nudging one worst sample per round
            # - the greedy version just chased the violation from one segment
            # to the next (fixing DIP->TIP re-broke PIP->DIP; the chain has to
            # CURL as a whole).
            acc = [Vector((0.0, 0.0, 0.0)) for _ in range(4)]
            cnt = [0, 0, 0, 0]
            for si in range(3):
                a, b = pts[si], pts[si + 1]
                seg = (b - a).length
                if seg < 1e-6:
                    continue
                eps = max(0.0008, 0.04 * seg)
                for k in range(1, samples + 1):
                    t = k / (samples + 1.0)
                    s = a.lerp(b, t)
                    loc, nrm, _i, _d = bvh.find_nearest(s)
                    if loc is None or nrm is None:
                        continue
                    out = (s - Vector(loc)).dot(Vector(nrm))
                    if out <= eps:
                        continue
                    tgt = Vector(loc) - Vector(nrm) * min(0.0015, 0.4 * out)
                    v = tgt - s
                    iA, iB = si, si + 1
                    mA, mB = _MOBILITY[iA], _MOBILITY[iB]
                    D = mA * (1.0 - t) ** 2 + mB * t ** 2
                    if D < 1e-9:
                        continue
                    acc[iA] += v * (mA * (1.0 - t) / D)
                    cnt[iA] += 1
                    acc[iB] += v * (mB * t / D)
                    cnt[iB] += 1
            if not any(cnt):
                break                       # chain fully inside
            for i in range(4):
                if not cnt[i]:
                    continue
                d = acc[i] / cnt[i]
                if d.length > budget[i]:
                    d = d * (budget[i] / d.length)
                if d.length < 1e-6:
                    continue                # budget exhausted for this joint
                cand = pts[i] + d
                # Lane guard per JOINT: a correction must not land this joint
                # nearer another finger's lane. For a LATERALLY-drifted joint
                # the nearest surface IS the neighbour's wall, so the
                # nearest-surface pull always crosses lanes and every guarded
                # pass just leaves the joint frozen outside (the Hand_A DIPs).
                # The lane-SAFE correction for that case: pull the joint
                # toward its OWN chain's MCP->TIP chord - both chord ends are
                # anchored in the own finger, so moving toward it can never
                # cross into a neighbour.
                if finger != "thumb" and (_crosses(finger, cand)
                                          or _wrong_chord(finger, cand)):
                    axis = pts[3] - pts[0]
                    if axis.length < 1e-6 or i in (0, 3):
                        continue            # chord ends can't chord-pull
                    axis = axis.normalized()
                    rel  = pts[i] - pts[0]
                    perp = rel - axis * rel.dot(axis)
                    d = -perp * 0.5         # halfway toward the chord
                    if d.length > budget[i]:
                        d = d * (budget[i] / d.length)
                    if d.length < 1e-6:
                        continue
                    cand = pts[i] + d
                # MONOTONE guard: a correction may never leave THIS joint
                # deeper outside than it already is. On a curved finger the
                # straight MCP->TIP chord itself runs outside the volume, so
                # an unchecked chord-pull EJECTED joints (Kid-hand pinky went
                # from 2 to 3 joints outside).
                if _out_depth(cand) > _out_depth(pts[i]) + 0.0003:
                    continue
                pts[i] = cand
                budget[i] -= d.length
                changed = True
        # Restore the along-chord split (undo the along-slide, keep depth).
        if changed and _chord_axis is not None:
            _tol = 0.03 * flen
            for i in (1, 2):
                _drift = (pts[i] - pts[0]).dot(_chord_axis) - _along0[i]
                if abs(_drift) > _tol:
                    _corr = _drift - math.copysign(_tol, _drift)
                    pts[i] = pts[i] - _chord_axis * _corr
        if changed:
            for i, k in enumerate(keys):
                marker_pos[k] = pts[i]
            n_chains += 1
    if n_chains:
        print(f"  [contain {arp_side}]: pulled {n_chains} finger chain(s) "
              f"back inside the mesh volume")


def _out_depth_at(p, bvh):
    """Signed outside distance: >0 = `p` is this many metres beyond the mesh
    surface (nearest-normal convention), 0 = on/inside the surface."""
    loc, nrm, _i, _d = bvh.find_nearest(p)
    if loc is None or nrm is None:
        return 0.0
    return max(0.0, (p - Vector(loc)).dot(Vector(nrm)))


def seat_finger_tips_on_pad(marker_pos, mesh_obj, arp_side,
                            pullback=0.30, bvh=None, max_shift=0.022):
    """
    Move each fingertip marker OFF the nail tip (the distal extreme resnap finds)
    back to the middle of the fingertip pad: pull it PROXIMALLY along the finger's
    own axis by `pullback` of the distal phalange (DIP->tip), then centre it in the
    finger cross-section there. Both moves stay inside the finger's own tube - the
    axial pull is toward this finger's DIP, never toward a neighbour, and the radial
    centring keeps the across-finger lateral clamp - so it cannot drift sideways.
    """
    if not marker_pos:
        return
    if bvh is None:
        bvh = _build_bvh(mesh_obj)
    if bvh is None:
        return

    idx_mcp = marker_pos.get(f"{_FINGER_TO_ARP['index']['phal1']}_{arp_side}")
    pnk_mcp = marker_pos.get(f"{_FINGER_TO_ARP['pinky']['phal1']}_{arp_side}")
    across = None
    if idx_mcp is not None and pnk_mcp is not None and (idx_mcp - pnk_mcp).length > 1e-5:
        across = (idx_mcp - pnk_mcp).normalized()

    # DIP positions per finger -> adaptive across clamp from neighbour spacing, so the
    # left-right centring below can pull a side-drifted tip back to the midline without
    # ever jumping onto a touching neighbour.
    dips = {}
    for _f in ("thumb", "index", "middle", "ring", "pinky"):
        _a = _FINGER_TO_ARP.get(_f)
        if _a:
            dips[_f] = marker_pos.get(f"{_a['phal3']}_{arp_side}")

    moved = 0
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        arp = _FINGER_TO_ARP.get(finger)
        if not arp:
            continue
        tip = marker_pos.get(f"{arp['tip']}_{arp_side}")
        dip = marker_pos.get(f"{arp['phal3']}_{arp_side}")
        if tip is None or dip is None:
            continue
        seg = tip - dip
        seglen = seg.length
        if seglen < 1e-4:
            continue
        axis = seg / seglen
        # The 30% pull-back moves a tip OFF the nail onto the pad — correct for the
        # four fingers, whose tips resnap first pushed OUT to the nail extreme. The
        # THUMB is skipped by resnap (its distal-vertex snap grabs a knuckle on a
        # curl), so its tip is already ~on the pad; a full 30% pull then overshoots
        # toward the DIP -> "thumb tip always short" (Hand_A: 40mm -> 29mm). Give the
        # thumb a much smaller pull; its depth (nail-normal) seat below is unchanged.
        _pb = 0.10 if finger == "thumb" else pullback
        pulled = tip - axis * (seglen * _pb)        # back toward the DIP (off the nail)

        if finger == "thumb":
            # Thumb: pull straight back along its own axis - the full radial
            # cross-section centring is skipped (its tip is near the palm/
            # index, so a 6-ray centring collapses it onto the DIP) and no
            # across-strip (index->pinky is not the thumb's lateral). But a
            # NAIL-NORMAL depth seat is still needed: without it the tip stays
            # ON the nail plane while every other joint sits centred ("thumb
            # tip moves to the nail"). Find the surface the tip rides (the
            # nail) via nearest-point, ray through the thumb to the opposite
            # (pad) wall along the inward normal, and seat the tip midway -
            # one axis only, so it cannot collapse axially or drift sideways.
            loc, nrm, _i, _d = bvh.find_nearest(pulled)
            if loc is not None and nrm is not None:
                n = Vector(nrm)
                hit, _hn, _hi, _hd = bvh.ray_cast(Vector(loc) - n * 0.001,
                                                  -n, 1.2 * seglen)
                if hit is not None:
                    mid = (Vector(loc) + Vector(hit)) * 0.5
                    delta = n * (mid - pulled).dot(n)
                    if delta.length > 0.5 * seglen:
                        delta = delta * (0.5 * seglen / delta.length)
                    pulled = pulled + delta
            marker_pos[f"{arp['tip']}_{arp_side}"] = pulled
            moved += 1
            continue

        # Centre the pulled-back point in the finger cross-section (dorsal-palmar).
        cur = pulled
        for _it in range(3):
            cen = _local_cross_section_center(cur, axis, bvh, search_radius=0.024)
            if cen is None:
                break
            stp = cen - cur
            cur = cen
            if stp.length < 0.0005:
                break
        off = cur - pulled
        if off.length > max_shift:
            off = off.normalized() * max_shift
        # CLAMP (not strip) the across-finger component. Border fingers (index/pinky)
        # drift sideways to the open outer edge; the cross-section centring above
        # computes the correct left-right pull, so we must KEEP it - only bound it to
        # within ~half the gap to the nearest neighbour DIP so a neighbour-contaminated
        # slice can't drag the tip onto a touching finger. The proximal + depth move is
        # unbounded (it stays in this finger's own tube).
        move = (pulled + off) - tip
        if across is not None:
            a_comp = move.dot(across)
            # Neighbour gap measured ALONG the across axis, not as a raw 3D DIP
            # distance. A short finger (pinky) sits well behind its neighbours
            # along the finger axis, so the 3D distance over-reads the true
            # side gap and inflates the clamp - letting the tip slide across
            # the inter-finger gap onto the neighbour (Hand_Drift pinky->ring).
            _own = dips.get(finger)
            nbr = [ abs((_own - d).dot(across))
                    for f2, d in dips.items()
                    if f2 != finger and d is not None and _own is not None ]
            a_max = max(0.004, min(0.012, (min(nbr) * 0.45) if nbr else 0.012))
            a_clamped = max(-a_max, min(a_max, a_comp))
            move = move - across * (a_comp - a_clamped)
        new_tip = tip + move
        # Fail-closed guard: if the pad seating pushed the tip OUT of the mesh
        # (a tapering fingertip's cross-section rays catch a neighbour wall and
        # drag the centre into the inter-finger gap), strip the across-finger
        # component so the tip stays in its own tube. The proximal + depth move
        # is kept - it never leaves this finger's volume.
        if across is not None:
            _oo = _out_depth_at(new_tip, bvh)
            if _oo > 0.0015:
                a_now = move.dot(across)
                cand_tip = tip + (move - across * a_now)
                if _out_depth_at(cand_tip, bvh) < _oo:
                    new_tip = cand_tip
        marker_pos[f"{arp['tip']}_{arp_side}"] = new_tip
        moved += 1
    if moved:
        print(f"  [pad {arp_side}]: seated {moved} tips on the finger pad")


def center_finger_joints_in_volume(marker_pos, mesh_obj, arp_side,
                                   bvh=None, max_shift=0.022):
    """
    Push each finger joint (MCP/PIP/DIP) OFF the dorsal surface and INTO the centre
    of the finger volume. The landmark model predicts joints on the silhouette, so
    after triangulation they sit on the back surface - most visible on curled
    fingers, where the dorsal surface bulges far from the bone.

    For each joint, cast paired perpendicular rays (via _local_cross_section_center)
    using the LOCAL chain direction as the slice axis, then move the joint toward the
    cross-section midpoint. TIPS are skipped - resnap already placed them at the
    distal end and centring drifts them off it. The move is clamped so a bad
    cross-section (e.g. catching an adjacent finger) can't fling a joint away.
    """
    if not marker_pos:
        return
    if bvh is None:
        bvh = _build_bvh(mesh_obj)
    if bvh is None:
        return

    # Across-fingers (knuckle-row) direction, passed to the centring so it can reject
    # LATERAL hits on a neighbouring finger while still centring fully (dorsal-palmar
    # AND laterally) within this finger's own walls.
    idx_mcp = marker_pos.get(f"{_FINGER_TO_ARP['index']['phal1']}_{arp_side}")
    pnk_mcp = marker_pos.get(f"{_FINGER_TO_ARP['pinky']['phal1']}_{arp_side}")
    across = None
    if idx_mcp is not None and pnk_mcp is not None and (idx_mcp - pnk_mcp).length > 1e-5:
        across = (idx_mcp - pnk_mcp).normalized()

    moved = 0
    # Thumb skipped: its base sits in the thenar eminence (palm mound), not a clean
    # finger tube, so cross-section centring drags the thumb knuckle into the palm.
    for finger in ("index", "middle", "ring", "pinky"):
        arp = _FINGER_TO_ARP.get(finger)
        if not arp:
            continue
        # chain in tip->base order; tip first so we can derive per-joint axes
        chain_keys = [f"{arp['tip']}_{arp_side}",  f"{arp['phal3']}_{arp_side}",
                      f"{arp['phal2']}_{arp_side}", f"{arp['phal1']}_{arp_side}"]
        pts = [marker_pos.get(k) for k in chain_keys]
        for i in (1, 2, 3):            # phal3(DIP), phal2(PIP), phal1(MCP) - skip tip
            pt = pts[i]
            if pt is None:
                continue
            # Local BONE direction at the joint = average of the two adjacent segment
            # directions, so the slice is taken straight across the finger here rather
            # than slanted along the PIP->tip chord (which cuts through the curl).
            nxt = pts[i - 1]                                 # toward tip
            prv = pts[i + 1] if i + 1 < len(pts) else None   # toward base
            axis = Vector((0.0, 0.0, 0.0))
            if nxt is not None and (nxt - pt).length > 1e-5:
                axis += (nxt - pt).normalized()
            if prv is not None and (pt - prv).length > 1e-5:
                axis += (pt - prv).normalized()
            if axis.length < 1e-4:
                continue
            axis = axis.normalized()
            # Push the joint off the dorsal skin toward the finger's dorsal-palmar
            # midline ALONG THE EXPLICIT DEPTH AXIS (finger axis x knuckle-row), with a
            # single ray pair - NOT a 6-ray radial average. At the knuckle the lateral
            # rays of a radial average hit the ADJACENT knuckles (middle/ring have a
            # neighbour on both sides), diluting the average and leaving the MCP stuck
            # on the surface. A depth-only ray pair can't be contaminated sideways.
            depth_dir = axis.cross(across) if across is not None else Vector((0.0, 0.0, 0.0))
            if depth_dir.length < 1e-4:
                # No across reference - fall back to the radial cross-section.
                cur = pt
                for _it in range(3):
                    cen = _local_cross_section_center(cur, axis, bvh, search_radius=0.024)
                    if cen is None:
                        break
                    step = cen - cur
                    cur = cen
                    if step.length < 0.0005:
                        break
                off = cur - pt
                if across is not None:
                    off = off - across * off.dot(across)
            else:
                depth_dir = depth_dir.normalized()
                _hp, _, _, _ = bvh.ray_cast(pt + depth_dir * 0.002,  depth_dir,  0.05)
                _hn, _, _, _ = bvh.ray_cast(pt - depth_dir * 0.002, -depth_dir,  0.05)
                if _hp is not None and _hn is not None:
                    mid = (Vector(_hp) + Vector(_hn)) * 0.5
                elif _hp is not None:
                    mid = (pt + Vector(_hp)) * 0.5
                elif _hn is not None:
                    mid = (pt + Vector(_hn)) * 0.5
                else:
                    mid = pt
                off = depth_dir * (mid - pt).dot(depth_dir)   # depth component only
            if off.length < 1e-5:
                continue
            if off.length > max_shift:
                off = off.normalized() * max_shift
            marker_pos[chain_keys[i]] = pt + off
            pts[i] = marker_pos[chain_keys[i]]
            moved += 1
    if moved:
        print(f"  [depth {arp_side}]: centred {moved} joints into finger volume")


def enforce_finger_chain_order(marker_pos, arp_side, gap_frac=0.06, min_gap=0.003):
    """
    Guarantee the joints run in anatomical order ALONG the finger: MCP < PIP < DIP
    < TIP measured by distance projected onto the MCP->TIP axis. The detector
    sometimes places the PIP farther out than the DIP (or vice-versa); enforcing
    world-Z order can't fix that when the finger isn't vertical, so we work in the
    finger's own direction instead - orientation-independent.

    MCP (t=0) and TIP (t=flen) are anchors. Each intermediate joint is projected to
    its along-axis distance t and slid along the axis (only) so that t is at least
    `step` past the previous joint and at least `step` short of the tip, where
    step = max(min_gap, gap_frac * finger_length). The perpendicular curl offset is
    preserved - only the along-axis position is corrected.
    """
    if not marker_pos:
        return
    moved = 0
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        arp = _FINGER_TO_ARP.get(finger)
        if not arp:
            continue
        mcp = marker_pos.get(f"{arp['phal1']}_{arp_side}")
        tip = marker_pos.get(f"{arp['tip']}_{arp_side}")
        if mcp is None or tip is None:
            continue
        axis = tip - mcp
        flen = axis.length
        if flen < 1e-4:
            continue
        axis = axis / flen
        step = max(min_gap, flen * gap_frac)

        inter = [f"{arp['phal2']}_{arp_side}", f"{arp['phal3']}_{arp_side}"]  # PIP, DIP
        prev_t = 0.0                          # MCP
        for i, k in enumerate(inter):
            pt = marker_pos.get(k)
            if pt is None:
                continue
            t = (pt - mcp).dot(axis)
            remaining = len(inter) - i        # joints left incl. this one, before tip
            lo = prev_t + step
            hi = flen - step * remaining      # leave room for this + the tip
            new_t = min(max(t, lo), max(lo, hi))
            if abs(new_t - t) > 1e-6:
                marker_pos[k] = pt + axis * (new_t - t)
                moved += 1
            prev_t = new_t
    if moved:
        print(f"  [order {arp_side}]: re-ordered {moved} joints to MCP<PIP<DIP<TIP")


def fix_thumb_base(marker_pos, arp_side, max_ratio=1.7, target_ratio=1.3):
    """
    GUARDED correction for the neural thumb base (THUMB_1 / CMC), which the model
    sometimes over-predicts toward the wrist on certain hands. The base is the
    highest-variance thumb landmark; the MCP (THUMB_2) and IP (THUMB_3) are placed
    reliably, so we use THEM as the anchor.

    Anatomy: the thumb metacarpal (THUMB_1->THUMB_2) is only ~1.1-1.4x the proximal
    phalanx (THUMB_2->THUMB_3). If the predicted metacarpal is implausibly long
    (> max_ratio x phalanx) the base has drifted proximally - clamp its length back
    to target_ratio x phalanx ALONG ITS OWN PREDICTED DIRECTION (pull it back toward
    the MCP, not sideways). Hands whose base is already plausible are left untouched.
    Wrist-independent by construction: nothing here reads the HAND/wrist marker, so
    the base can't track wrist error. See [[feedback_wrist_independence]].
    """
    if not marker_pos:
        return
    arp  = _FINGER_TO_ARP["thumb"]
    base = marker_pos.get(f"{arp['phal1']}_{arp_side}")   # THUMB_1 (CMC)
    mcp  = marker_pos.get(f"{arp['phal2']}_{arp_side}")   # THUMB_2 (MCP, stable)
    ip   = marker_pos.get(f"{arp['phal3']}_{arp_side}")   # THUMB_3 (IP,  stable)
    if base is None or mcp is None or ip is None:
        return
    phal_len = (mcp - ip).length
    if phal_len < 1e-4:
        return
    cur     = base - mcp
    cur_len = cur.length
    if cur_len < 1e-4:
        return
    if cur_len > max_ratio * phal_len:
        marker_pos[f"{arp['phal1']}_{arp_side}"] = mcp + (cur / cur_len) * (target_ratio * phal_len)
        print(f"  [thumb {arp_side}]: base clamped {cur_len/phal_len:.2f}->{target_ratio:.2f}x phalanx "
              f"(was drifting toward wrist)")


def straighten_fingers_lateral(marker_pos, arp_side, clamp=0.30):
    """
    Remove side-to-side (across-finger) drift so each finger reads as a straight,
    clean chain - WITHOUT flattening its bend or its natural fan.

    Why this is safe where earlier lateral passes were not: the finger's forward is
    built from the STABLE proximal joints (MCP->PIP, MCP->DIP averaged), NOT MCP->TIP.
    The tip is the joint that drifts, so anchoring on it tilts the reference toward the
    drift and fixes nothing. With a stable forward, the lateral direction is the
    across-fingers axis projected perpendicular to it, and each of PIP/DIP/TIP is
    pulled onto the finger's OWN line in that lateral direction only. The bend
    (dorsal-palmar) and the along-finger position (incl. fan) are untouched, and
    because every move is relative to this finger's own joints, nothing can drift
    toward a neighbour. Move clamped to `clamp` x finger length.
    """
    if not marker_pos:
        return
    imcp = marker_pos.get(f"{_FINGER_TO_ARP['index']['phal1']}_{arp_side}")
    pmcp = marker_pos.get(f"{_FINGER_TO_ARP['pinky']['phal1']}_{arp_side}")
    if imcp is None or pmcp is None or (imcp - pmcp).length < 1e-5:
        return
    across = (imcp - pmcp).normalized()

    moved = 0
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        arp = _FINGER_TO_ARP[finger]
        mcp = marker_pos.get(f"{arp['phal1']}_{arp_side}")
        pip = marker_pos.get(f"{arp['phal2']}_{arp_side}")
        dip = marker_pos.get(f"{arp['phal3']}_{arp_side}")
        tip = marker_pos.get(f"{arp['tip']}_{arp_side}")
        if mcp is None:
            continue

        # NOTE: the TIP is deliberately NOT straightened laterally anymore. With the
        # retrained landmark model + LVT tip detection the tip lands accurately on the
        # pad centre (and resnap/pad already strip any across drift). Forcing it onto
        # the MCP line here de-fanned it and pushed an already-correct tip sideways off
        # the pad (worst on ring/pinky, which fan most). We only straighten PIP/DIP.
        if finger == "thumb":
            # Thumb: planarize onto its OWN bend plane (the knuckle-row across
            # is not the thumb's lateral). Rigify bones zigzag whenever the
            # chain leaves a single bend plane, so: bend dir = mean
            # perpendicular offset of the intermediate joints from the
            # CMC->tip line; the OUT-OF-PLANE component of each joint is the
            # zigzag - remove only that, keeping the in-plane curl intact.
            if tip is None or pip is None or dip is None:
                continue
            fwd_t = tip - mcp
            if fwd_t.length < 1e-4:
                continue
            fwd_t = fwd_t.normalized()
            offs = []
            for J in (pip, dip):
                r = J - mcp
                offs.append(r - fwd_t * r.dot(fwd_t))
            bend = offs[0] + offs[1]
            if bend.length < 1e-5:
                continue                    # chain already straight
            n = fwd_t.cross(bend.normalized())
            if n.length < 1e-5:
                continue
            n = n.normalized()
            flen_t = (tip - mcp).length
            for key, J in ((f"{arp['phal2']}_{arp_side}", pip),
                           (f"{arp['phal3']}_{arp_side}", dip)):
                out_c = (J - mcp).dot(n)
                if abs(out_c) < 1e-5:
                    continue
                if abs(out_c) > flen_t * clamp:
                    out_c = math.copysign(flen_t * clamp, out_c)
                marker_pos[key] = J - n * out_c
                moved += 1
            continue
        else:
            # Finger forward = MCP->TIP. The tip is now reliably centred (pad pass), so
            # it is the correct distal anchor: PIP and DIP are pulled laterally onto the
            # straight MCP->TIP line. (Earlier the tip drifted, so we anchored on the
            # proximal joints instead; that's no longer needed.)
            if tip is not None and (tip - mcp).length > 1e-4:
                fwd = (tip - mcp).normalized()
            else:
                fwd = Vector((0.0, 0.0, 0.0))
                for j in (pip, dip):
                    if j is not None and (j - mcp).length > 1e-4:
                        fwd += (j - mcp).normalized()
                if fwd.length < 1e-4:
                    continue
                fwd = fwd.normalized()
            lat_n = across - fwd * across.dot(fwd)   # across, perpendicular to the finger
            if lat_n.length < 1e-4:
                continue
            lat_n = lat_n.normalized()
            targets = ((f"{arp['phal2']}_{arp_side}", pip),
                       (f"{arp['phal3']}_{arp_side}", dip))

        flen = (tip - mcp).length if tip is not None else (dip - mcp).length
        max_shift = flen * clamp
        for key, J in targets:
            if J is None:
                continue
            lateral = (J - mcp).dot(lat_n)        # side-to-side offset from the line
            if abs(lateral) < 1e-5:
                continue
            if abs(lateral) > max_shift:
                lateral = math.copysign(max_shift, lateral)
            marker_pos[key] = J - lat_n * lateral
            moved += 1
    if moved:
        print(f"  [straight {arp_side}]: aligned {moved} joints laterally")


def finger_quality_report(marker_pos, arp_side):
    """
    Cheap post-placement sanity check for one hand. Returns a list of human-readable
    warnings (empty = looks fine) so the operator can flag low-confidence results to
    the user instead of silently shipping a bad hand. Read-only - never moves markers.

    Flags: a long finger that came out implausibly short (collapsed chain), and a
    pair of adjacent fingertips that collapsed onto each other.
    """
    warns = []
    lens, tips, mcps = {}, {}, {}
    for f in ("index", "middle", "ring", "pinky"):
        arp = _FINGER_TO_ARP.get(f)
        if not arp:
            continue
        mcp = marker_pos.get(f"{arp['phal1']}_{arp_side}")
        tip = marker_pos.get(f"{arp['tip']}_{arp_side}")
        if mcp is not None:
            mcps[f] = mcp
        if mcp is not None and tip is not None:
            lens[f] = (tip - mcp).length
            tips[f] = tip
    if not lens:
        return warns

    ref = max(lens.values())                      # longest finger = scale reference
    for f, L in lens.items():
        if L < 0.45 * ref:
            warns.append(f"{f} short ({L*1000:.0f}mm)")
    for a, b in (("index", "middle"), ("middle", "ring"), ("ring", "pinky")):
        if a in tips and b in tips and (tips[a] - tips[b]).length < 0.012:
            warns.append(f"{a}/{b} tips merged")

    # MCP knuckle-row sanity: the four MCPs must sit in monotonic index->pinky order
    # across the palm with no finger crowded onto its neighbour. A middle MCP drifted
    # onto the ring (or ring onto pinky) shows up here as an OUT-OF-ORDER or NEAR-ZERO
    # gap along the across-axis - a lateral error the length/tip-merge checks are
    # blind to, and which NO cleanup pass corrects (none move an MCP sideways). This
    # is the signal that should hand the side to the geometric engine.
    row = [f for f in ("index", "middle", "ring", "pinky") if f in mcps]
    if len(row) == 4:
        across = mcps["index"] - mcps["pinky"]
        if across.length > 1e-4:
            across = across.normalized()
            coord = {f: mcps[f].dot(across) for f in row}   # index highest -> pinky lowest
            gaps = {("index", "middle"): coord["index"]  - coord["middle"],
                    ("middle", "ring"):  coord["middle"] - coord["ring"],
                    ("ring", "pinky"):   coord["ring"]   - coord["pinky"]}
            ref_gap = max(gaps.values())            # widest gap = robust spacing reference

            for (a, b), g in gaps.items():
                # Two distinct finger KNUCKLES are never this close: a gap that is a
                # tiny fraction of normal finger spacing (or out of order) means a
                # finger's MCP landed on its neighbour's -> a COLLAPSED finger (the
                # whole chain is on the wrong finger, even if its tip fans out
                # elsewhere). A hard failure needing more than a single-marker nudge.
                # A normal hand's smallest knuckle gap is ~0.6-0.8x the widest, so
                # this never fires on fingers that are merely a bit close at the base.
                if g <= 0.0 or (ref_gap > 1e-4 and g < 0.20 * ref_gap):
                    warns.append(f"{a}/{b} collapsed")
                elif ref_gap > 1e-4 and g < 0.30 * ref_gap:
                    warns.append(f"{a}/{b} MCP crowded ({g*1000:.0f}mm)")

    # Thumb collapse: the thumb is offset/opposable so it can't be range-checked
    # against the knuckle row -- it gets its own checks, against `ref` (longest
    # non-thumb finger = scale reference). Signals are chosen to NOT trip on a
    # valid tucked/opposed thumb (which legitimately sits near the index):
    #   1. a piled-up thumb CHAIN (segment sum, pose-invariant -- bone lengths
    #      don't change with curl) means the markers collapsed to a blob;
    #   2. the thumb BASE (THUMB_1, normally at the wrist/thenar) landing up on the
    #      index knuckle means the whole thumb landed on the index finger -- a
    #      tucked thumb keeps its base near the wrist, far from the index MCP.
    _ta     = _FINGER_TO_ARP["thumb"]
    t1      = marker_pos.get(f"{_ta['phal1']}_{arp_side}")   # base (near wrist)
    t2      = marker_pos.get(f"{_ta['phal2']}_{arp_side}")
    t3      = marker_pos.get(f"{_ta['phal3']}_{arp_side}")
    tt      = marker_pos.get(f"{_ta['tip']}_{arp_side}")
    idx_mcp = mcps.get("index")
    chain   = ((t2 - t1).length + (t3 - t2).length + (tt - t3).length
               if (t1 and t2 and t3 and tt) else None)
    if chain is not None and chain < 0.30 * ref:
        warns.append("thumb collapsed")
    elif t1 is not None and idx_mcp is not None and (t1 - idx_mcp).length < 0.25 * ref:
        warns.append("thumb collapsed")
    return warns


def finger_failure_count(marker_pos, arp_side):
    """
    Count INDEPENDENT finger defects for one hand from finger_quality_report's
    warnings, for the "drop the hand if too many fingers failed" policy. Returns
    (n_defects, warns).

    A defect involving a PAIR (merged tips, crowded/out-of-order MCPs) counts as
    ONE problem, not two — one drifted finger is an easy manual fix. Defects that
    share a finger are grouped (a middle finger merged onto BOTH neighbours is one
    bad finger, not two), via connected components over the flagged finger-sets.
    So: a single merge or one short finger -> 1 (keepable); two independent
    problems -> 2 (drop candidate).
    """
    warns = finger_quality_report(marker_pos, arp_side)
    if not warns:
        return 0, warns

    # Each warning starts with the finger(s) it concerns: "ring short (...)" or
    # "index/middle tips merged" / "middle/ring MCP crowded (...)".
    groups = []  # list of sets of finger names
    for w in warns:
        token = w.split(" ", 1)[0]
        fingers = set(token.split("/")) if "/" in token else {token}
        # union into an existing group that shares a finger, else start a new one
        hit = None
        for g in groups:
            if g & fingers:
                g |= fingers
                hit = g
                break
        if hit is None:
            groups.append(set(fingers))
    # a second pass merges groups that became connected via the growth above
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if groups[i] & groups[j]:
                    groups[i] |= groups[j]
                    del groups[j]
                    merged = True
                    break
            if merged:
                break
    return len(groups), warns


def separate_collapsed_tips(marker_pos, arp_side, mesh_obj=None, collapse_dist=0.024):
    """Pull apart adjacent fingers that collapsed onto each other.

    On untrained finger styles the model can't separate neighbouring fingers
    (classically the RING finger drifting onto the PINKY). The whole ring CHAIN can
    drift -- not just the tip -- so an own-chain extension can't separate it (the
    chain itself points at the pinky). The existing symmetry-mirror repair also
    fails when BOTH sides drift the same way.

    Robust fix, anchored to geometry that the model DOES get right:
      * The MCP knuckles are reliable and well separated (ring MCP != pinky MCP).
      * A non-collapsed finger (the middle, normally) gives a trusted finger
        DIRECTION; relaxed fingers are roughly parallel.
    When two adjacent fingers' tips are within `collapse_dist`, rebuild any finger
    whose tip has wandered far from where its OWN MCP + the trusted direction puts
    it: lay its phalanges straight out from its own MCP along that direction,
    preserving the finger's own segment lengths. Ring and pinky then separate by
    their knuckle spacing. A finger already sitting above its MCP is left untouched.
    No wrist/hand-marker input is used.
    """
    order = ("index", "middle", "ring", "pinky")

    def _key(f, p):
        arp = _FINGER_TO_ARP[f]
        return f"{arp[p]}_{arp_side}" if arp.get(p) else None

    def _get(f, p):
        k = _key(f, p)
        return marker_pos.get(k) if k else None

    def _chain(f):                     # [MCP, PIP, DIP, TIP] or None
        pts = [_get(f, p) for p in ("phal1", "phal2", "phal3", "tip")]
        return pts if all(p is not None for p in pts) else None

    # Which adjacent pairs collapsed?
    collapsed = set()
    pairs = (("index", "middle"), ("middle", "ring"), ("ring", "pinky"))
    for a, b in pairs:
        ta, tb = _get(a, "tip"), _get(b, "tip")
        if ta is not None and tb is not None and (ta - tb).length < collapse_dist:
            collapsed.add(a); collapsed.add(b)
    if not collapsed:
        return 0

    # Trusted direction = a NON-collapsed finger's MCP->TIP (prefer middle, index).
    ref_dir = None
    for f in ("middle", "index", "ring", "pinky"):
        if f in collapsed:
            continue
        ch = _chain(f)
        if ch and (ch[3] - ch[0]).length > 0.02:
            ref_dir = (ch[3] - ch[0]).normalized()
            break
    if ref_dir is None:
        return 0                       # no trusted finger to align to

    # BVH to seat the rebuilt (straight) joints into the REAL finger volume: the
    # straight chain follows the trusted direction, but ring/pinky angle/curl a bit
    # differently, so the synthesized joints sit off-axis. A local cross-section
    # centre pulls each onto the finger (lateral + depth). Allowed laterally here
    # because these joints are SYNTHESIZED, not model placements.
    _bvh = _build_bvh(mesh_obj) if mesh_obj is not None else None

    moved = 0
    for f in collapsed:
        ch = _chain(f)
        if not ch:
            continue
        mcp = ch[0]
        segs = [(ch[i + 1] - ch[i]).length for i in range(3)]
        chain_len = sum(segs)
        expected_tip = mcp + ref_dir * chain_len
        dev = (ch[3] - expected_tip).length
        # Rebuild only if the tip wandered off its own MCP's line (the drifted
        # finger). A correctly-placed finger sits ~above its MCP (small dev). The
        # drift onto a neighbour is ~ the knuckle spacing.
        if dev < max(0.012, chain_len * 0.12):
            continue
        # GARBAGE GUARD: a real drift can't exceed the finger's own length. dev far
        # beyond chain_len means the MCP or the trusted direction is itself bad (the
        # whole-hand detection failed, e.g. the camera clipped the hand) -- rebuilding
        # would only move the tip somewhere else wrong, so leave it for the user.
        if dev > chain_len * 1.3:
            print(f"  [decollapse {arp_side}] {f}: dev={dev*1000:.0f}mm >> "
                  f"len={chain_len*1000:.0f}mm -- detection unreliable, skipping")
            continue
        cum = 0.0
        new_pts = [mcp]
        for s in segs:
            cum += s
            new_pts.append(mcp + ref_dir * cum)
        # Seat the synthesized joints onto the real finger volume. The search radius
        # scales with finger length (a big hand's finger is wider AND its parallel-
        # assumption straight guess sits further from the splayed real finger); a
        # second, wider try recovers a pinky that the tight radius missed.
        _sr = max(0.012, chain_len * 0.10)
        seated  = {}
        tip_ok  = (_bvh is None)               # nothing to verify against -> trust it
        for p, np_ in zip(("phal2", "phal3", "tip"), new_pts[1:]):
            pos = np_
            if _bvh is not None:
                c = _local_cross_section_center(pos, ref_dir, _bvh, search_radius=_sr)
                if c is None:
                    c = _local_cross_section_center(pos, ref_dir, _bvh, search_radius=_sr * 2.0)
                if c is not None:
                    pos = c                    # seat onto the real finger (lat + depth)
                    if p == "tip":
                        tip_ok = True
            seated[p] = pos
        # If the rebuilt TIP found no finger volume even at the wider radius, the
        # straight-line guess is floating in the gap/air (the parallel assumption
        # failed for a splayed pinky) -- don't apply it; leave the finger as-is.
        if not tip_ok:
            print(f"  [decollapse {arp_side}] {f}: rebuilt tip found no finger volume "
                  f"(<= {_sr*2000:.0f}mm) -- leaving finger as-is")
            continue
        for p, pos in seated.items():
            marker_pos[_key(f, p)] = pos
        moved += 1
        print(f"  [decollapse {arp_side}] rebuilt {f}: dev={dev*1000:.0f}mm "
              f"len={chain_len*1000:.0f}mm along trusted dir")

    if moved:
        _g = {f: marker_pos.get(_key(f, "tip")) for f in order}
        gaps = " ".join(
            f"{a[0]}{b[0]}={(_g[a]-_g[b]).length*1000:.0f}mm"
            for a, b in (("middle", "ring"), ("ring", "pinky"))
            if _g[a] and _g[b])
        print(f"  [decollapse {arp_side}]: rebuilt {moved} collapsed finger(s)  ({gaps})")
    return moved


def _recentre_mirrored(marker_pos, bvh, finger, side):
    """Re-centre a just-MIRRORED chain's interior joints in their local
    cross-section. The mirror plane comes from the HAND markers and real
    meshes aren't perfectly X-symmetric, so a joint centred on the source
    side lands ON the target side's surface instead of inside it ("touching
    the thumb, not inside") - and every centring pass has already run by the
    time the final symmetry fires. Same math as the [depth] pass, applied
    only to the joints the mirror just wrote (incl. the thumb: its PIP/DIP
    are tube joints; only its CMC has the thenar concern - left untouched)."""
    arp = _FINGER_TO_ARP.get(finger)
    if not arp:
        return
    keys = [f"{arp[p]}_{side}" for p in ("phal1", "phal2", "phal3", "tip")]
    pts = [marker_pos.get(k) for k in keys]
    if any(p is None for p in pts):
        return
    flen = (pts[3] - pts[0]).length
    if flen < 1e-5:
        return
    sr = min(0.035, max(0.010, 0.30 * flen))
    for i in (1, 2):                         # PIP, DIP only
        axis = Vector((0.0, 0.0, 0.0))
        if (pts[i + 1] - pts[i]).length > 1e-5:
            axis += (pts[i + 1] - pts[i]).normalized()
        if (pts[i] - pts[i - 1]).length > 1e-5:
            axis += (pts[i] - pts[i - 1]).normalized()
        if axis.length < 1e-4:
            continue
        axis = axis.normalized()
        cur = pts[i]
        for _it in range(3):
            cen = _local_cross_section_center(cur, axis, bvh, search_radius=sr)
            if cen is None:
                break
            stp = cen - cur
            # 0.15, not 0.35: a joint at skin depth needs ~half a finger
            # radius of correction (6-12mm); anything larger means the rays
            # caught a NEIGHBOUR finger or the palm (thumbs sit next to the
            # index) and would drag the joint out of its own volume.
            if stp.length > 0.15 * flen:     # neighbour-contaminated slice
                break
            cur = cen
            if stp.length < 0.0005:
                break
        if (cur - pts[i]).length > 1e-5:
            marker_pos[keys[i]] = cur
            pts[i] = cur


def recentre_thumb_interior(marker_pos, mesh_obj, arp_side, bvh=None):
    """Depth-centre the THUMB's PIP/DIP. Every depth pass excludes the thumb
    (the thenar-mound rule - valid for its CMC, not its tube joints), so a
    thumb joint rescued by [inside]/[contain] stays at rescue depth: 1.5-3mm
    under the skin, reading as "on the surface" while every finger joint gets
    re-centred by depth2. Same cross-section recentring as the mirrored-chain
    polish, thumb only."""
    if not marker_pos:
        return
    if bvh is None:
        bvh = _build_bvh(mesh_obj)
    if bvh is None:
        return
    _recentre_mirrored(marker_pos, bvh, "thumb", arp_side)


def enforce_final_symmetry(marker_pos, mesh_obj, hw_l, hw_r):
    """FINAL L/R reconciliation, run AFTER all per-side cleanup passes.

    The pre-cleanup symmetry pass can't guarantee symmetric hands because the
    PER-SIDE passes diverge them afterwards - their guards fire on one side
    only (seen as the L thumb DIP floating in air while the R one sat
    correctly inside the thumb: [inside]/[contain] rescued R but rejected the
    L corrections). Rigify wants exactly mirrored hands, so per FINGER:
      * whichever side's chain sits better INSIDE the mesh (lower summed
        outside-depth) wins and is MIRRORED onto the other - containment is
        ground truth the length/collapse heuristics of the earlier pass can't
        see (a floating joint has a normal chain length);
      * when both sides are equally contained, the joints are mirror-AVERAGED
        so the hands end exactly symmetric.
    """
    if hw_l is None or hw_r is None or not marker_pos:
        return
    bvh = _build_bvh(mesh_obj)
    if bvh is None:
        return
    cx = (hw_l.x + hw_r.x) * 0.5

    def _mir(p):
        return Vector((2.0 * cx - p.x, p.y, p.z))

    def _out(p):
        loc, nrm, _i, _d = bvh.find_nearest(p)
        if loc is None or nrm is None:
            return 0.0
        return max(0.0, (p - Vector(loc)).dot(Vector(nrm)))

    n_avg = n_copy = 0
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        pairs = []
        for pt in ("phal1", "phal2", "phal3", "tip"):
            arp = _FINGER_TO_ARP[finger].get(pt)
            if not arp:
                continue
            kl, kr = f"{arp}_L", f"{arp}_R"
            if kl in marker_pos and kr in marker_pos:
                pairs.append((kl, kr))
        if not pairs:
            continue
        bad_l = sum(_out(marker_pos[kl]) for kl, _ in pairs)
        bad_r = sum(_out(marker_pos[kr]) for _, kr in pairs)
        if bad_l - bad_r > 0.004:
            for kl, kr in pairs:
                marker_pos[kl] = _mir(marker_pos[kr])
            n_copy += 1
            _recentre_mirrored(marker_pos, bvh, finger, "L")
            print(f"  [symmetry-final] {finger}: L {bad_l*1000:.0f}mm outside "
                  f"vs R {bad_r*1000:.0f}mm -- mirrored R->L")
        elif bad_r - bad_l > 0.004:
            for kl, kr in pairs:
                marker_pos[kr] = _mir(marker_pos[kl])
            n_copy += 1
            _recentre_mirrored(marker_pos, bvh, finger, "R")
            print(f"  [symmetry-final] {finger}: R {bad_r*1000:.0f}mm outside "
                  f"vs L {bad_l*1000:.0f}mm -- mirrored L->R")
        else:
            for kl, kr in pairs:
                m = (marker_pos[kl] + _mir(marker_pos[kr])) * 0.5
                marker_pos[kl] = m
                marker_pos[kr] = _mir(m)
            n_avg += 1
    if n_avg or n_copy:
        print(f"  [symmetry-final]: {n_avg} finger(s) mirror-averaged, "
              f"{n_copy} mirrored from the better-contained side")


def enforce_finger_symmetry(marker_pos, hw_l, hw_r):
    """Make L and R finger markers symmetric for Rigify (always symmetric rigs).

    Decided per finger, since detection fails per finger (a whole chain
    collapses, not a single joint):
      * If the two sides agree (all joints' mirrored gap <= 5 cm), mirror-AVERAGE
        each joint - cancels model noise.
      * If they disagree (any joint > 5 cm apart, i.e. one side's chain is bad),
        MIRROR THE BETTER SIDE onto the other. "Better" = the longer, less-collapsed
        MCP->TIP chain (the failure mode is a collapsed finger). This guarantees
        symmetry without averaging a bad detection into the good side.
    """
    if hw_l is None or hw_r is None or not marker_pos:
        return
    cx = (hw_l.x + hw_r.x) * 0.5   # X symmetry plane

    def _mir(p):
        return Vector((2.0 * cx - p.x, p.y, p.z))

    def _collapsed_fingers(side):
        # Fingers whose tip has merged onto an adjacent finger's tip (lateral
        # collapse) on this side - a per-pose detection failure, length still normal.
        order = ("index", "middle", "ring", "pinky")
        tips = {f: marker_pos.get(f"{_FINGER_TO_ARP[f]['tip']}_{side}") for f in order}
        coll = set()
        for a, b in (("index", "middle"), ("middle", "ring"), ("ring", "pinky")):
            if tips[a] is not None and tips[b] is not None and (tips[a] - tips[b]).length < 0.014:
                coll.add(a); coll.add(b)
        return coll

    coll_L = _collapsed_fingers("L")
    coll_R = _collapsed_fingers("R")

    n_avg = n_copy = 0
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        parts = [pt for pt in ("phal1", "phal2", "phal3", "tip")
                 if _FINGER_TO_ARP[finger].get(pt)]
        # Joints present on BOTH sides for this finger.
        pairs = []
        for pt in parts:
            arp = _FINGER_TO_ARP[finger][pt]
            k_l, k_r = f"{arp}_L", f"{arp}_R"
            if k_l in marker_pos and k_r in marker_pos:
                pairs.append((k_l, k_r))
        if not pairs:
            continue

        # Lateral collapse on ONE side -> mirror the good side (don't average a
        # collapsed finger into the good one). Chain length is normal here, so the
        # length tiebreak below can't catch it.
        c_l, c_r = finger in coll_L, finger in coll_R
        if c_r and not c_l:
            for kl, kr in pairs:
                marker_pos[kr] = _mir(marker_pos[kl]); n_copy += 1
            print(f"  [symmetry] {finger}: R collapsed - mirrored L->R")
            continue
        if c_l and not c_r:
            for kl, kr in pairs:
                marker_pos[kl] = _mir(marker_pos[kr]); n_copy += 1
            print(f"  [symmetry] {finger}: L collapsed - mirrored R->L")
            continue

        max_gap = max((marker_pos[kl] - _mir(marker_pos[kr])).length
                      for kl, kr in pairs)

        if max_gap <= 0.05:
            # Sides agree - average each joint (noise cancellation).
            for kl, kr in pairs:
                avg_l = (marker_pos[kl] + _mir(marker_pos[kr])) * 0.5
                marker_pos[kl] = avg_l
                marker_pos[kr] = _mir(avg_l)
                n_avg += 1
        else:
            # Sides disagree - mirror the better (longer / less-collapsed) chain.
            mcp = _FINGER_TO_ARP[finger].get("phal1")
            tip = _FINGER_TO_ARP[finger].get("tip")
            len_l = len_r = 0.0
            if mcp and tip:
                ml, tl = marker_pos.get(f"{mcp}_L"), marker_pos.get(f"{tip}_L")
                mr, tr = marker_pos.get(f"{mcp}_R"), marker_pos.get(f"{tip}_R")
                if ml and tl:
                    len_l = (tl - ml).length
                if mr and tr:
                    len_r = (tr - mr).length
            use_left = len_l >= len_r   # longer chain wins (tie -> left)
            for kl, kr in pairs:
                if use_left:
                    marker_pos[kr] = _mir(marker_pos[kl])
                else:
                    marker_pos[kl] = _mir(marker_pos[kr])
                n_copy += 1
            print(f"  [symmetry] {finger}: gap {max_gap*1000:.0f}mm - "
                  f"mirrored {'L->R' if use_left else 'R->L'} "
                  f"(L={len_l*1000:.0f}mm R={len_r*1000:.0f}mm)")

    print(f"[symmetry] {n_avg} joints averaged, {n_copy} joints mirrored "
          f"around X={cx:.3f}m")


# -- Geometric finger extraction via flood-fill clustering ---------------------

# -- Geometric (no-ONNX) entry -------------------------------------------------

# How far to push the MCP proximally (toward the wrist) onto the knuckle row, as a fraction of
# the proximal-phalanx forward span. 0 = at the detected knuckle (the edge-ify ROOT already sits
# at the knuckle, so no extra push is needed; pushing made the MCPs sit inside the palm).
_MCP_PROXIMAL_PUSH = 0.0

# OUR-OWN MCP: the proximal phalanx (MCP->PIP) is ~1.33x the middle phalanx (PIP->DIP) in every
# human hand. We CONSTRUCT the MCP by extending the PIP->DIP axis past the PIP by this ratio,
# instead of DETECTING the knuckle from the converging palm geometry (where every detector
# collapses). Anchored to the two reliable distal joints; straight; cannot cluster.
_MCP_PHALANX_RATIO = 1.33


# -- Hybrid entry: landmark -> tip heatmap -> geometric --------------------------

def _finger_valid_mask(result, hw, ew, arp_side, for_onnx=False):
    """
    Per-finger validity for a detection result. A finger is valid when its
    MCP->TIP chain is long enough AND its tip is far enough from the wrist -
    the same thresholds _has_spread uses, but reported per finger instead of
    collapsed to a single pass/fail. Returns {finger_name: bool}.
    """
    hw_v       = Vector(hw)
    arm_len    = (hw_v - Vector(ew)).length if ew else 0.25
    flen_floor = 0.020 if for_onnx else 0.030
    flen_scale = 0.08  if for_onnx else 0.12
    min_flen   = max(flen_floor, arm_len * flen_scale)
    min_dist   = arm_len * 0.20

    mask = {}
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        arp = _FINGER_TO_ARP.get(finger, {})
        mcp = result.get(f"{arp.get('phal1', '')}_{arp_side}")
        tip = result.get(f"{arp.get('tip', '')}_{arp_side}")
        if not (mcp and tip):
            mask[finger] = False
            continue
        long_enough = (tip - mcp).length >= min_flen
        far_enough  = (tip - hw_v).length >= min_dist
        mask[finger] = bool(long_enough and far_enough)
    return mask


def _merge_finger_sources(sources, hw, ew, arp_side):
    """
    Build a 20-marker result by taking each finger from the first source (in
    priority order) where that finger is valid. `sources` is a list of
    (result_dict, for_onnx_bool) - earlier entries win ties. Fingers valid in
    no source are left for the caller to fill geometrically.
    Returns (merged_dict, set_of_filled_finger_names).
    """
    merged = {}
    filled = set()
    masks  = [(_finger_valid_mask(r, hw, ew, arp_side, fo), r) for r, fo in sources]
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        arp  = _FINGER_TO_ARP.get(finger, {})
        keys = [f"{arp[p]}_{arp_side}" for p in ("phal1", "phal2", "phal3", "tip")
                if p in arp]
        for mask, r in masks:
            if mask.get(finger) and all(k in r for k in keys):
                for k in keys:
                    merged[k] = r[k]
                filled.add(finger)
                break
    return merged, filled


def detect_fingers_hybrid(mesh_obj, hw, ew, temp_dir, arp_side, center_radius=0.015,
                          palm_depth=0.25, knuckle_radius=0.90,
                          width_tolerance=1.0, straighten_clamp=0.15):
    def _has_spread(result, for_onnx=False):
        # Check 1: tips must be spread in world space (not all at same point)
        tips = [v for k, v in result.items() if 'TIP' in k]
        if len(tips) < 3:
            return False
        tip_spread = max(
            max(p.x for p in tips) - min(p.x for p in tips),
            max(p.y for p in tips) - min(p.y for p in tips),
            max(p.z for p in tips) - min(p.z for p in tips),
        )
        if tip_spread < 0.015:
            return False
        # Check 2: MCP->TIP plausibility.  LVT uses strict thresholds (30mm, 4/5)
        # because it can fail silently with bad geometry.  ONNX is more robust so
        # we accept a lower floor (20mm, 3/5) - still rejects true collapse (<16mm).
        hw_v      = Vector(hw)
        arm_len   = (hw_v - Vector(ew)).length if ew else 0.25
        flen_floor = 0.020 if for_onnx else 0.030
        flen_scale = 0.08  if for_onnx else 0.12
        min_count  = 3     if for_onnx else 4
        min_flen   = max(flen_floor, arm_len * flen_scale)
        long_fingers = 0
        finger_lens  = {}
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            arp  = _FINGER_TO_ARP.get(finger, {})
            mcp  = result.get(f"{arp.get('phal1', '')}_{arp_side}")
            tip  = result.get(f"{arp.get('tip', '')}_{arp_side}")
            flen = (tip - mcp).length if (mcp and tip) else 0.0
            finger_lens[finger] = flen
            if flen >= min_flen:
                long_fingers += 1
        print(f"  _has_spread [{arp_side}]: {long_fingers}/5 long - "
              f"thumb={finger_lens.get('thumb',0)*1000:.1f}mm  "
              f"index={finger_lens.get('index',0)*1000:.1f}mm  "
              f"middle={finger_lens.get('middle',0)*1000:.1f}mm  "
              f"ring={finger_lens.get('ring',0)*1000:.1f}mm  "
              f"pinky={finger_lens.get('pinky',0)*1000:.1f}mm")
        if long_fingers < min_count:
            print(f"  [{arp_side}] _has_spread: only {long_fingers}/5 fingers long enough  -- palm bias")
            return False
        # Check 3: at least 4/5 tips must be >=20% arm_len from the wrist.
        min_dist = arm_len * 0.20
        far_tips = sum(1 for t in tips if (t - hw_v).length >= min_dist)
        if far_tips < 4:
            print(f"  [{arp_side}] _has_spread: only {far_tips}/5 tips far from wrist  -- rejecting")
            return False
        return True

    per_view_tips    = None
    per_view_cameras = None
    result           = None   # bound by LVT/landmark branches; guarded at merge
    if True:
        # 1. Tip heatmap model first  -- 5 outputs, does not collapse, triangulation works.
        lvt_accepted    = None
        lvt_raw         = None
        lvt_palm_center = None
        lvt_hand_scale  = None
        lvt_fwd         = None
        if is_lvt_available():
            lvt_result = detect_fingers_lvt(mesh_obj, hw, ew, temp_dir, arp_side,
                                            palm_depth=palm_depth,
                                            knuckle_radius=knuckle_radius,
                                            width_tolerance=width_tolerance,
                                            straighten_clamp=straighten_clamp)
            if isinstance(lvt_result, tuple) and len(lvt_result) == 6:
                result, per_view_tips, per_view_cameras, lvt_palm_center, lvt_hand_scale, lvt_fwd = lvt_result
            elif isinstance(lvt_result, tuple) and len(lvt_result) == 5:
                result, per_view_tips, per_view_cameras, lvt_palm_center, lvt_hand_scale = lvt_result
            elif isinstance(lvt_result, tuple):
                result, per_view_tips, per_view_cameras = lvt_result
            else:
                result, per_view_tips, per_view_cameras = lvt_result, None, None
            # LVT tips come from the dedicated tip model (~1.7px) and are reliable
            # even when LVT's MCP/knuckle CHAIN fails _has_spread. Keep the raw LVT
            # result so we can anchor grafted landmark tips from it later, regardless
            # of whether the chain was accepted as primary.
            lvt_raw = result if (result and len(result) >= 16) else None
            if result and len(result) >= 16 and _has_spread(result):
                print(f"Tip [{arp_side}]: accepted as primary ({len(result)} markers)  [pipeline=LVT]")
                lvt_accepted = result   # store but keep going so landmark diagnostics run
            else:
                print(f"Tip [{arp_side}]: rejected  -- trying 20-landmark fallback")

        # 2. 20-landmark model  -- always run for diagnostics so we can see MCP->TIP lengths
        if is_landmark_available():
            result = detect_fingers_landmark(mesh_obj, hw, ew, temp_dir, arp_side,
                                             orbit_center=lvt_palm_center,
                                             orbit_scale=lvt_hand_scale,
                                             orbit_fwd=lvt_fwd)
            if result and len(result) >= 16 and _has_spread(result, for_onnx=True):
                print(f"Landmark [{arp_side}]: accepted ({len(result)} markers)  [pipeline=ONNX]")
                # Anchor landmark TIPs with LVT's mesh-snapped positions when available.
                # LVT tips are more accurate for endpoints; landmark provides the
                # intermediate PIP/DIP joints. Skip any LVT tip that is degenerate
                # (within 15 mm of a neighbouring tip  -- triangulation collapsed).
                if lvt_accepted is not None:
                    _finger_order = ("thumb", "index", "middle", "ring", "pinky")
                    tip_keys = [f"{_FINGER_TO_ARP[f]['tip']}_{arp_side}"
                                for f in _finger_order]
                    lvt_tips = [lvt_accepted.get(k) for k in tip_keys]
                    lm_tips  = [result.get(k) for k in tip_keys]
                    good = [t is not None for t in lvt_tips]
                    good[0] = False   # thumb tip from ONNX (LVT tip unreliable on curls)

                    # -- Smart degenerate filter ---------------------------------
                    # LVT collapses adjacent tips onto one vertex, THEN pushes them
                    # apart to _PUSH_TARGET (~12 mm) — which sails past a naive <5 mm
                    # collapse test and lets both bogus tips overwrite good ONNX ones.
                    # So flag a pair as collapsed when its LVT tips sit suspiciously
                    # close (< 16 mm) WHILE the landmark model keeps them clearly
                    # farther apart (the "LVT collapsed but ONNX didn't" signal), and
                    # drop BOTH LVT tips — a real collapse makes both unreliable, and
                    # the landmark model already separated this pair. The d_lm guard
                    # leaves genuinely close-but-agreeing fingers untouched.
                    # Check EVERY pair, regardless of flags already set: skipping
                    # pairs whose first member was flagged let the PARTNER
                    # survive (ring/pinky never tested after index/ring flagged
                    # -> the pushed-apart bogus pinky got anchored over a good
                    # ONNX pinky and drifted into the palm). Thumb (0) is
                    # excluded from anchoring anyway.
                    for i in range(1, 5):
                        if lm_tips[i] is None or lvt_tips[i] is None:
                            continue
                        for j in range(i + 1, 5):
                            if lm_tips[j] is None or lvt_tips[j] is None:
                                continue
                            d_lvt = (lvt_tips[i] - lvt_tips[j]).length
                            d_lm  = (lm_tips[i] - lm_tips[j]).length
                            if d_lvt < 0.016 and d_lm > d_lvt + 0.010:
                                if good[i] or good[j]:
                                    print(f"  [{arp_side}] {_finger_order[i]}/{_finger_order[j]} "
                                          f"LVT tips collapsed ({d_lvt*1000:.0f}mm vs ONNX "
                                          f"{d_lm*1000:.0f}mm) -- keeping ONNX for both")
                                good[i] = False
                                good[j] = False

                    # -- Anchor tips ---------------------------------------------
                    replaced = 0
                    for k, t, g in zip(tip_keys, lvt_tips, good):
                        if g:
                            result[k] = t
                            replaced += 1
                    print(f"  [{arp_side}] anchored {replaced}/5 tips from LVT")

                    _post_bvh = _build_bvh(mesh_obj)

                    # -- Promote MCPs --------------------------------------------
                    for i, f in enumerate(_finger_order):
                        mcp_phal = "phal2" if f == "thumb" else "phal1"
                        mcp_key  = f"{_FINGER_TO_ARP[f][mcp_phal]}_{arp_side}"
                        lvt_mcp  = lvt_accepted.get(mcp_key)
                        if lvt_mcp is None or mcp_key not in result:
                            continue
                        tip_k = tip_keys[i]
                        if tip_k not in result:
                            continue
                        # Gate: only promote from a PLAUSIBLE LVT chain. We
                        # already distrust a collapsed LVT finger's TIP — its
                        # MCP is just as bad, and promoting it while the tip
                        # stays ONNX builds a MIXED chain (LVT thumb 36mm vs
                        # ONNX 135mm) whose axis points off the finger; the
                        # axis-projection pass then drags PIP/DIP out of it.
                        # 0.60, not 0.45: stylized hands run LVT chains at
                        # ~1/3 of ONNX (a 55.8mm LVT thumb slipped past a 45%
                        # gate of a 120mm ONNX chain by 2mm and the DIP went
                        # into the palm again); genuine LVT/ONNX agreement
                        # runs near 1:1, so 0.60 still promotes those.
                        lvt_tip  = lvt_accepted.get(tip_k)
                        onnx_len = (result[tip_k] - result[mcp_key]).length
                        lvt_len  = ((lvt_tip - lvt_mcp).length
                                    if lvt_tip is not None else 0.0)
                        if lvt_len < max(0.03, onnx_len * 0.60):
                            print(f"  [{arp_side}] {f} MCP: LVT chain "
                                  f"{lvt_len*1000:.0f}mm vs ONNX "
                                  f"{onnx_len*1000:.0f}mm -- keeping ONNX MCP")
                            continue
                        if (lvt_mcp - hw).length < (lvt_mcp - result[tip_k]).length:
                            result[mcp_key] = lvt_mcp
                            print(f"  [{arp_side}] {f} MCP -> LVT")

                    # -- BVH centre TIPs and MCPs --------------------------------
                    for i, (f, g) in enumerate(zip(_finger_order, good)):
                        mcp_phal = "phal2" if f == "thumb" else "phal1"
                        mcp_key  = f"{_FINGER_TO_ARP[f][mcp_phal]}_{arp_side}"
                        tip_k    = tip_keys[i]
                        mcp = result.get(mcp_key)
                        tip = result.get(tip_k)
                        if mcp is None or tip is None:
                            continue
                        axis = (tip - mcp).normalized()
                        if axis.length < 0.001:
                            continue
                        if g:
                            cen_tip = _local_cross_section_center(tip, axis, _post_bvh, search_radius=0.018)
                            if cen_tip is not None:
                                result[tip_k] = cen_tip * 0.75 + tip * 0.25
                                tip = result[tip_k]
                            else:
                                result[tip_k] = tip - axis * 0.003
                                tip = result[tip_k]
                            axis = (tip - mcp).normalized()
                        # Skip BVH centering if MCP was promoted from LVT
                        # (_refine_mcp_positions already placed it via the slider)
                        lvt_mcp = lvt_accepted.get(mcp_key)
                        if lvt_mcp is not None and (result[mcp_key] - lvt_mcp).length < 0.001:
                            continue
                        cen_mcp = _local_cross_section_center(mcp, axis, _post_bvh, search_radius=center_radius * 1.33)
                        if cen_mcp is not None:
                            result[mcp_key] = cen_mcp * 0.6 + mcp * 0.4

                    # -- Axis projection for PIP/DIP -----------------------------
                    for i, f in enumerate(_finger_order):
                        mcp_phal = "phal2" if f == "thumb" else "phal1"
                        mcp_key  = f"{_FINGER_TO_ARP[f][mcp_phal]}_{arp_side}"
                        tip_k    = tip_keys[i]
                        mcp = result.get(mcp_key)
                        tip = result.get(tip_k)
                        if mcp is None or tip is None:
                            continue
                        axis     = tip - mcp
                        axis_len = axis.length
                        if axis_len < 0.020:
                            continue
                        axis_n   = axis / axis_len

                        proj_phals = ["phal3"] if f == "thumb" else ["phal2", "phal3"]
                        prev_t = axis_len * 0.08
                        for proj_phal in proj_phals:
                            pk = f"{_FINGER_TO_ARP[f][proj_phal]}_{arp_side}"
                            if pk not in result:
                                continue
                            t = (result[pk] - mcp).dot(axis_n)
                            t = max(prev_t, min(axis_len * 0.90, t))
                            result[pk] = mcp + axis_n * t
                            prev_t = t + axis_len * 0.05

                    # -- BVH centre PIP/DIP --------------------------------------
                    for i, f in enumerate(_finger_order):
                        mcp_phal = "phal2" if f == "thumb" else "phal1"
                        mcp_key  = f"{_FINGER_TO_ARP[f][mcp_phal]}_{arp_side}"
                        tip_k    = tip_keys[i]
                        mcp = result.get(mcp_key)
                        tip = result.get(tip_k)
                        if mcp is None or tip is None:
                            continue
                        axis = (tip - mcp).normalized()
                        if axis.length < 0.001:
                            continue
                        proj_phals = ["phal3"] if f == "thumb" else ["phal2", "phal3"]
                        for proj_phal in proj_phals:
                            pk = f"{_FINGER_TO_ARP[f][proj_phal]}_{arp_side}"
                            if pk not in result:
                                continue
                            cen = _local_cross_section_center(result[pk], axis, _post_bvh, search_radius=center_radius)
                            if cen is not None:
                                result[pk] = cen * 0.7 + result[pk] * 0.3

                # LVT chain rejected, but its TIPS (from the 1.7px tip model) are
                # reliable. The landmark model grafts short/wrong tips on some poses
                # (e.g. "INDEX_TIP grafted from geometric" -> 18mm finger). Replace a
                # landmark tip with the LVT tip when LVT's tip is meaningfully farther
                # from the wrist (a longer, more plausible finger), and redistribute
                # that finger's PIP/DIP along the corrected MCP->tip axis.
                elif lvt_raw is not None:
                    _fo   = ("thumb", "index", "middle", "ring", "pinky")
                    _hw_v = Vector(hw)
                    _anch = 0
                    for f in _fo:
                        if f == "thumb":
                            continue          # thumb tip comes from ONNX (see rescue)
                        _arp = _FINGER_TO_ARP[f]
                        tk  = f"{_arp['tip']}_{arp_side}"
                        mk  = f"{_arp['phal1']}_{arp_side}"
                        lt  = lvt_raw.get(tk)
                        rt  = result.get(tk)
                        mcp = result.get(mk)
                        if lt is None or rt is None:
                            continue
                        if (lt - _hw_v).length > (rt - _hw_v).length + 0.010:
                            result[tk] = lt
                            _anch += 1
                            if mcp is not None and f != "thumb":
                                for _ph, _frac in (("phal2", 0.40), ("phal3", 0.72)):
                                    pk = f"{_arp[_ph]}_{arp_side}"
                                    if pk in result:
                                        result[pk] = mcp.lerp(lt, _frac)
                    if _anch:
                        print(f"  [{arp_side}] anchored {_anch}/5 tips from LVT (chain rejected)")

                # -- Drifted-tip rescue ----------------------------------------
                # The LVT tip-model (1.7px) is the most reliable tip detector. If the
                # ONNX tip has drifted toward a NEIGHBOUR while the LVT tip keeps it
                # more separated, take the LVT tip. Fires on partial drift too, not
                # only full collapse, but only when LVT is meaningfully better so a
                # correctly-placed ONNX tip is left alone.
                _lvt_src = lvt_accepted if lvt_accepted is not None else lvt_raw
                if _lvt_src is not None:
                    _fo = ("thumb", "index", "middle", "ring", "pinky")
                    _tk = {f: f"{_FINGER_TO_ARP[f]['tip']}_{arp_side}" for f in _fo}
                    _rescued = 0
                    for f in _fo:
                        if f == "thumb":
                            # Thumb has no close neighbour, so "more separated" is
                            # meaningless for it; on a curled thumb the LVT tip sits
                            # near the knuckle and this would pick it. Keep ONNX.
                            continue
                        rt = result.get(_tk[f])
                        lt = _lvt_src.get(_tk[f])
                        if rt is None or lt is None:
                            continue
                        d_onnx = min((( rt - result[_tk[o]]).length
                                      for o in _fo if o != f and result.get(_tk[o])),
                                     default=1.0)
                        d_lvt  = min((( lt - _lvt_src[_tk[o]]).length
                                      for o in _fo if o != f and _lvt_src.get(_tk[o])),
                                     default=1.0)
                        # LVT keeps the tip apart (>14mm) AND more separated than ONNX
                        # does (>=4mm better) -> the ONNX tip drifted; use the LVT tip.
                        take = d_lvt > 0.014 and (d_lvt - d_onnx) > 0.004
                        print(f"  [{arp_side}] {f} tip: onnx_gap={d_onnx*1000:.0f}mm "
                              f"lvt_gap={d_lvt*1000:.0f}mm{'  -> LVT' if take else ''}")
                        if take:
                            result[_tk[f]] = lt
                            _rescued += 1
                    if _rescued:
                        print(f"  [{arp_side}] rescued {_rescued} drifted tip(s) from LVT")
                return result
            print(f"Landmark [{arp_side}]: rejected  -- per-finger merge")

        # -- Per-finger merge -------------------------------------------------
        # Neither source passed _has_spread as a whole, but each may have several
        # good fingers. Rather than discard a mostly-correct hand, take each finger
        # from the first source that detected it validly. Priority (= current
        # accept order): LVT (stricter gate) first, then the rejected ONNX result.
        # `result` still holds the last landmark detection when it was rejected.
        _sources = []
        if lvt_accepted is not None:
            _sources.append((lvt_accepted, False))   # for_onnx=False (LVT thresholds)
        if isinstance(result, dict) and result is not lvt_accepted:
            _sources.append((result, True))          # rejected ONNX, looser thresholds

        if _sources:
            merged, filled = _merge_finger_sources(_sources, hw, ew, arp_side)
            missing = [f for f in ("thumb", "index", "middle", "ring", "pinky")
                       if f not in filled]
            if missing:
                # Geometric fallback supplies the fingers no AI source got right.
                mesh_res = _place_fingers_from_mesh(
                    mesh_obj, hw, ew, arp_side, per_view_tips, per_view_cameras)
                for f in missing:
                    arp  = _FINGER_TO_ARP.get(f, {})
                    keys = [f"{arp[p]}_{arp_side}"
                            for p in ("phal1", "phal2", "phal3", "tip") if p in arp]
                    for k in keys:
                        if k in mesh_res:
                            merged[k] = mesh_res[k]
            print(f"[{arp_side}]: per-finger merge - {len(filled)}/5 from AI "
                  f"({'+'.join(sorted(filled)) or 'none'}), "
                  f"{len(missing)}/5 geometric")
            if merged:
                return merged

    # 3. Mesh-aware fallback: find actual fingertip vertices, derive phalanges
    print(f"[{arp_side}]: mesh-aware fallback")
    return _place_fingers_from_mesh(mesh_obj, hw, ew, arp_side, per_view_tips, per_view_cameras)
