"""
Synthetic hand training-data generator (camera + shape randomization).
================================================================
Multiplies the CURRENT hand(s) into many training samples by randomizing the
CAMERA framing (orbit forward axis, scale, centre) and — when Warp Variants is
on — the HAND SHAPE itself: each sample pushes the mesh vertices AND the marker
labels through one identical smooth warp (finger length, hand width/thickness,
distal taper), the same domain-randomization recipe as Save Face Data xN.
Ground truth is read from the 20 MARKER_* empties — exactly the source
AUTORIG_OT_SaveHandData uses — so output is byte-compatible with
train_hand_detector.py (sample_XXXX/hand_{side}.json + .png).

WORKFLOW: pose the hand / swap the mesh -> AI Detect -> fix the 20 markers by
hand -> run this to emit N varied samples. Repeat per pose/mesh. Mesh
DIVERSITY is still the #1 driver of cross-style robustness; the warp multiplies
each real hand across the style envelope (chunky/slim, long/stubby, tapered);
camera jitter adds viewpoint invariance; PHOTOMETRIC augmentation belongs
in the TRAINER's data loader (rig-independent, benefits every sample).
"""

import bpy
import os
import json as _json
import math
import random
import tempfile
import shutil
import numpy as np
from mathutils import Vector, Matrix

from .ai_detect import (
    _render_hand_orbit, _orbit_world_to_px,
    _MP_TO_ARP, _HAND_IMG_SIZE, _ORBIT_N_VIEWS, _DEFAULT_FOREARM_LEN,
)
from .ai_detect_lvt import _compute_render_hw


def _all_finger_keys(side):
    return [
        f"THUMB_1_{side}", f"THUMB_2_{side}", f"THUMB_3_{side}", f"THUMB_TIP_{side}",
        f"FINGER_INDEX_1_{side}",  f"FINGER_INDEX_2_{side}",  f"FINGER_INDEX_3_{side}",  f"FINGER_INDEX_TIP_{side}",
        f"FINGER_MIDDLE_1_{side}", f"FINGER_MIDDLE_2_{side}", f"FINGER_MIDDLE_3_{side}", f"FINGER_MIDDLE_TIP_{side}",
        f"FINGER_RING_1_{side}",   f"FINGER_RING_2_{side}",   f"FINGER_RING_3_{side}",   f"FINGER_RING_TIP_{side}",
        f"FINGER_PINKY_1_{side}",  f"FINGER_PINKY_2_{side}",  f"FINGER_PINKY_3_{side}",  f"FINGER_PINKY_TIP_{side}",
    ]


def _read_markers_world(side):
    """GT joint positions from the MARKER_* empties (same set SaveHandData uses)."""
    out = {}
    for arp_base in set(_MP_TO_ARP.values()):
        obj = bpy.data.objects.get(f"MARKER_{arp_base}_{side}")
        if obj:
            out[f"{arp_base}_{side}"] = list(obj.matrix_world.translation)
    return out


# ── Procedural hand warp (domain randomization, mirrors gen_face_training_data) ──

def _hand_warp_params(rng):
    """One random smooth hand deformation. Modest ranges — the warp must stay
    inside the plausible-hand envelope or it dilutes the real signal."""
    return {
        "len":   rng.uniform(0.85, 1.25),   # wrist->tip stretch (long/stubby)
        "wide":  rng.uniform(0.80, 1.30),   # lateral spread axis (chunky/slim)
        "flat":  rng.uniform(0.80, 1.25),   # palm-normal thickness
        "taper": rng.uniform(-0.25, 0.30),  # cross-section growth toward the tips
    }


_HAND_WARP_ID = {"len": 1.0, "wide": 1.0, "flat": 1.0, "taper": 0.0}


def _hand_warp_frame(side):
    """Local warp frame for this side: (wrist origin, fwd/lat/nor axes, hand
    length). Built from the side's HAND marker and 5 TIP markers. None when the
    markers aren't there."""
    hand = bpy.data.objects.get(f"MARKER_HAND_{side}")
    if hand is None:
        return None
    hw = np.array(hand.matrix_world.translation, dtype=np.float64)
    tips = []
    for k in (f"THUMB_TIP_{side}", f"FINGER_INDEX_TIP_{side}",
              f"FINGER_MIDDLE_TIP_{side}", f"FINGER_RING_TIP_{side}",
              f"FINGER_PINKY_TIP_{side}"):
        o = bpy.data.objects.get(f"MARKER_{k}")
        if o:
            tips.append(np.array(o.matrix_world.translation, dtype=np.float64))
    if len(tips) < 3:
        return None
    fwd = np.mean(tips, axis=0) - hw
    L = float(np.linalg.norm(fwd))
    if L < 1e-5:
        return None
    fwd /= L
    up  = np.array((0.0, 0.0, 1.0)) if abs(fwd[2]) < 0.9 else np.array((1.0, 0.0, 0.0))
    lat = np.cross(fwd, up); lat /= np.linalg.norm(lat)
    nor = np.cross(fwd, lat)
    return hw, fwd, lat, nor, L


def _hand_warp_np(pts, frame, prm):
    """Apply the warp to an (N,3) world-space array. The SAME function
    transforms mesh vertices and marker labels, so they stay consistent.
    Along-axis: only the HAND (a>=0, wrist->tips) stretches by `len`; the
    FOREARM (a<0) keeps its original length — a stretched forearm is not a
    plausible hand-shape variation. Taper likewise applies only to a>=0, so
    the forearm keeps its base cross-section."""
    hw, fwd, lat, nor, L = frame
    d = pts - hw
    a = d @ fwd
    u = d @ lat
    v = d @ nor
    cs = 1.0 + prm["taper"] * np.clip(a / L, 0.0, 1.2)
    a2 = np.where(a >= 0.0, a * prm["len"], a)   # forearm (a<0) unchanged
    u2 = u * prm["wide"] * cs
    v2 = v * prm["flat"] * cs
    return hw + np.outer(a2, fwd) + np.outer(u2, lat) + np.outer(v2, nor)


# ── Per-finger ARTICULATION (curl + splay) — replaces the old length stretch ────
# Each finger is a 4-joint chain [_1, _2, _3, TIP]. We synthesize new POSES by
# hinge-rotating the finger: CURL (flexion toward the palm) at all three joints
# progressively (forward kinematics), plus SPLAY (abduction in the palm plane
# about the palm normal) at the base. Rotation is rigid per segment, so bone
# lengths are preserved (no stretch/tearing along the finger) and the base stays
# put (displacement ~0 at the hinge). The SAME transform moves mesh verts AND the
# marker labels, so joint markers land exactly on their FK positions.
_FINGER_CHAINS = ("THUMB", "FINGER_INDEX", "FINGER_MIDDLE", "FINGER_RING", "FINGER_PINKY")


def _smootherstep(a, aj, b):
    """0 proximal of (aj-b), 1 distal of (aj+b), C2-smooth between — feathers a
    hinge so the curl onset across a joint doesn't crease."""
    t = np.clip((a - (aj - b)) / (2.0 * b), 0.0, 1.0)
    return t * t * t * (t * (t * 6 - 15) + 10)


def _rot_vec(v, axis, ang):
    c, s = math.cos(ang), math.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * float(axis @ v) * (1 - c)


def _rot_point(p, center, axis, ang):
    return center + _rot_vec(p - center, axis, ang)


def _rot_points(pts, center, axis, ang_pp):
    """Rotate (N,3) about center/axis by a PER-POINT angle ang_pp (N,) — Rodrigues."""
    v = pts - center
    c = np.cos(ang_pp)[:, None]; s = np.sin(ang_pp)[:, None]
    kxv = np.cross(np.broadcast_to(axis, v.shape), v)
    kdv = (v @ axis)[:, None]
    return center + v * c + kxv * s + axis[None, :] * kdv * (1 - c)


def _articulate_one(pts, g):
    """FK-articulate every point as if it fully belonged to finger g. Hinge order:
    splay (about MCP), then curl at MCP -> PIP -> DIP with the distal joint centres
    updated by each preceding rotation. Along-finger param `a` is a material coord
    (from the original straight axis) so the feather weights are deformation-stable."""
    j0, j1, j2, j3 = g["joints"]
    dirv, L = g["dir"], g["L"]
    a  = (pts - j0) @ dirv
    a1 = float((j1 - j0) @ dirv)
    a2 = float((j2 - j0) @ dirv)
    sax, curl, splay = g["splay_axis"], g["curl"], g["splay"]

    # Feather band. Bending a bone of radius r through angle t stretches its outer
    # surface by ~t*r regardless of HOW the bend is distributed (that total is
    # fixed by the curl angle) — but a WIDE band spreads it into a gentle curve so
    # the local peak stretch stays low, whereas a narrow band concentrates it into
    # an ugly crease. So keep the band generous.
    b  = 0.35 * L
    q = _rot_points(pts, j0, sax, splay * _smootherstep(a, 0.0, b))
    fax = _rot_vec(g["flex"], sax, splay)          # flex axis rides the splay
    c1  = _rot_point(j1, j0, sax, splay)
    c2  = _rot_point(j2, j0, sax, splay)

    q  = _rot_points(q, j0, fax, curl[0] * _smootherstep(a, 0.0, b))
    c1 = _rot_point(c1, j0, fax, curl[0]); c2 = _rot_point(c2, j0, fax, curl[0])
    q  = _rot_points(q, c1, fax, curl[1] * _smootherstep(a, a1, b))
    c2 = _rot_point(c2, c1, fax, curl[1])
    q  = _rot_points(q, c2, fax, curl[2] * _smootherstep(a, a2, b))
    return q


def _finger_articulate_np(pts, geoms, sigma):
    """SOFT per-finger articulation. Each finger's rigid FK displacement is blended
    across points by a Gaussian of perpendicular distance to that finger's axis
    line (adjacent fingers blend, palm/forearm barely move). Same fn on mesh verts
    AND marker labels so they stay consistent."""
    disp = np.zeros_like(pts)
    wsum = np.zeros(len(pts))
    for g in geoms:
        rel  = pts - g["joints"][0]
        a    = rel @ g["dir"]
        perp = np.linalg.norm(rel - np.outer(a, g["dir"]), axis=1)
        w    = np.exp(-(perp / sigma) ** 2)
        disp += w[:, None] * (_articulate_one(pts, g) - pts)
        wsum += w
    return pts + disp / np.maximum(wsum, 1e-6)[:, None]


def _finger_joints(side):
    """Per-finger geometry from the placed markers: for each of the 5 fingers a
    dict with joints(4,3)=[_1,_2,_3,TIP], unit dir, flex_axis (curl hinge) and
    splay_axis (palm normal). Also returns the blend sigma. None if any marker is
    missing."""
    geoms, mcps, dirs = [], [], []
    for chain in _FINGER_CHAINS:
        names = [f"{chain}_1", f"{chain}_2", f"{chain}_3", f"{chain}_TIP"]
        pts = []
        for nm in names:
            o = bpy.data.objects.get(f"MARKER_{nm}_{side}")
            if o is None:
                return None
            pts.append(np.array(o.matrix_world.translation, dtype=np.float64))
        joints = np.array(pts)
        d = joints[3] - joints[0]
        L = float(np.linalg.norm(d))
        if L < 1e-4:
            return None
        geoms.append({"joints": joints, "dir": d / L, "L": L})
        mcps.append(joints[0]); dirs.append(d / L)

    mcps = np.array(mcps)
    # Palm normal from the four non-thumb knuckles: knuckle-row x mean finger dir.
    knuckle_row = mcps[4] - mcps[1]                       # index -> pinky
    mean_dir = np.mean(dirs[1:], axis=0)
    palm_n = np.cross(knuckle_row, mean_dir)
    nlen = np.linalg.norm(palm_n)
    if nlen < 1e-6:
        return None
    palm_n /= nlen
    # Orient palm_n to the PALMAR side (thumb tip sits on the palm side of the hand).
    thumb_tip = geoms[0]["joints"][3]
    if (thumb_tip - np.mean(mcps[1:], axis=0)) @ palm_n < 0.0:
        palm_n = -palm_n
    for g in geoms:
        f = np.cross(g["dir"], palm_n)
        fn = np.linalg.norm(f)
        g["flex"] = f / fn if fn > 1e-6 else palm_n   # +curl bends tip toward palm
        g["splay_axis"] = palm_n

    gaps = np.linalg.norm(np.diff(mcps[1:], axis=0), axis=1)   # non-thumb spacing
    sigma = float(max(np.mean(gaps) * 0.6, 1e-3)) if len(gaps) else 0.01
    return geoms, sigma


def _finger_rot_params(rng, joints_data, cfg):
    """Draw a random per-finger pose: independent curl (flexion, biased toward
    small so relaxed hands stay common, occasional strong curl/fist) split across
    the 3 joints, plus a splay per finger. Thumb gets less curl / more splay
    (opposition). Returns (geoms-with-angles, sigma)."""
    geoms, sigma = joints_data
    max_curl  = math.radians(cfg.finger_curl_deg)
    max_splay = math.radians(cfg.finger_splay_deg)
    for gi, g in enumerate(geoms):
        is_thumb = (gi == 0)
        total = max_curl * (rng.random() ** 1.6) * (0.6 if is_thumb else 1.0)
        g["curl"]  = np.array([0.35, 0.45, 0.20]) * total     # MCP, PIP, DIP
        g["splay"] = rng.uniform(-max_splay, max_splay) * (1.8 if is_thumb else 1.0)
    return geoms, sigma


def _rot_signature(geoms):
    """Compact rounded angle tuple for the sample's provenance/dedup tag."""
    return [[round(float(c), 3) for c in g["curl"]] + [round(float(g["splay"]), 3)]
            for g in geoms]


def _warped_hand_copy(src, frame, prm, artic=None):
    """Evaluated copy of src with world-baked, warped vertices. Caller removes.
    Only this copy is passed to the renderer — _render_hand_orbit hides every
    other object itself, so the original mesh never appears in the crops.
    artic=(geoms, sigma) applies the per-finger curl/splay articulation FIRST
    (using the original-marker joints), then the global warp."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = src.evaluated_get(dg)
    me = bpy.data.meshes.new_from_object(ev, depsgraph=dg)
    ob = bpy.data.objects.new(f"_er_handwarp_{src.name}", me)
    bpy.context.collection.objects.link(ob)
    ob.matrix_world = Matrix.Identity(4)
    n = len(me.vertices)
    if n:
        co = np.empty(n * 3, dtype=np.float64)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        mw = np.array(ev.matrix_world, dtype=np.float64)
        co = co @ mw[:3, :3].T + mw[:3, 3]
        if artic is not None:
            co = _finger_articulate_np(co, artic[0], artic[1])
        co = _hand_warp_np(co, frame, prm)
        me.vertices.foreach_set("co", co.astype(np.float32).ravel())
        me.update()
    return ob


def _side_marker_names(side):
    """Every empty _save_sample_side reads for this side: the 20 joints plus
    the framing anchors (HAND / ELBOW / HAND_TIP)."""
    return ([f"MARKER_{k}" for k in _all_finger_keys(side)]
            + [f"MARKER_HAND_{side}", f"MARKER_ELBOW_{side}",
               f"MARKER_HAND_TIP_{side}"])


def _jitter_orbit(cen, scale, fwd, cfg, rng):
    """Randomize camera framing: rotate fwd, scale, offset centre."""
    ang = math.radians(rng.uniform(0.0, cfg.cam_angle_deg))
    axis = Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)))
    if axis.length < 1e-6:
        axis = Vector((0, 0, 1))
    fwd_j = (Matrix.Rotation(ang, 4, axis.normalized()) @ Vector(fwd)).normalized()
    scale_j = scale * rng.uniform(1.0 - cfg.cam_scale_jit, 1.0 + cfg.cam_scale_jit)
    off = scale * cfg.cam_center_jit
    cen_j = Vector(cen) + Vector((rng.uniform(-off, off),
                                  rng.uniform(-off, off),
                                  rng.uniform(-off, off)))
    return cen_j, scale_j, fwd_j


def _save_sample_side(mesh_obj, side, sample_dir, cfg, rng, source_name="",
                      warp_sig=None):
    """Render + label + write one hand side into sample_dir. Returns True on success.
    source_name = the ORIGINAL character mesh name (not the warp copy), so a
    character's clean sample + all its warp variants share one group key and the
    trainer keeps them on the SAME side of the train/val split (a warp of a train
    character in val is a leak — the same failure the face trainer had).
    warp_sig = the warp params dict (or None). The warp is anchored at the WRIST,
    so hand_world_pos/arm_len don't change between warps — the trainer's dedup
    cap keys on THIS instead, so each distinct warp shape survives the cap while
    redundant camera-only copies (warp_sig=None) still collapse together."""
    hand_obj  = bpy.data.objects.get(f"MARKER_HAND_{side}")
    elbow_obj = bpy.data.objects.get(f"MARKER_ELBOW_{side}")
    if hand_obj is None:
        return False
    hw = hand_obj.matrix_world.translation.copy()
    ew = elbow_obj.matrix_world.translation.copy() if elbow_obj else None
    arm_len = (hw - ew).length if ew is not None else _DEFAULT_FOREARM_LEN
    tip_obj = bpy.data.objects.get(f"MARKER_HAND_TIP_{side}")
    tip = tip_obj.matrix_world.translation.copy() if tip_obj else None

    markers_world = _read_markers_world(side)
    if not markers_world:
        return False

    hw_render = _compute_render_hw(hw, ew, side)
    s = _HAND_IMG_SIZE
    margin = 4

    # Framing MUST match inference. The landmark model runs at detect time on a crop
    # framed from the 5 DETECTED FINGERTIPS (tips_3d centroid) — see ai_detect_lvt
    # stable_cen/stable_scale. So frame training on the 5 TIP markers too, NOT all 20
    # joints: the 20-joint centroid sits ~one finger-half toward the palm, so a
    # 20-joint crop is systematically offset from what the model sees at inference.
    # Recipe otherwise identical (centroid center, max-dist*2.5 scale, elbow->centroid fwd).
    tip_keys = [f"THUMB_TIP_{side}",        f"FINGER_INDEX_TIP_{side}",
                f"FINGER_MIDDLE_TIP_{side}", f"FINGER_RING_TIP_{side}",
                f"FINGER_PINKY_TIP_{side}"]
    tip_pts = [Vector(markers_world[k]) for k in tip_keys if k in markers_world]
    if len(tip_pts) < 4 or ew is None:
        return False
    tip_c = sum(tip_pts, Vector()) / len(tip_pts)
    base_fwd = (tip_c - Vector(ew)).normalized()
    base_cen = tip_c
    base_scale = max((t - base_cen).length for t in tip_pts) * 2.5

    cen0, scale0, fwd0 = _jitter_orbit(base_cen, base_scale, base_fwd, cfg, rng)

    temp_dir = tempfile.mkdtemp(prefix="er_gen_")
    try:
        cen, orbit_views, hand_scale, _ = _render_hand_orbit(
            mesh_obj, hw_render, ew, temp_dir, "orbit", tip=tip,
            center=cen0, scale=scale0, orbit_fwd=fwd0)
        scale = hand_scale
        for i, (img_path, *_) in enumerate(orbit_views):
            shutil.copy2(img_path, os.path.join(sample_dir, f"hand_{side}_orbit_{i:02d}.png"))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    orbit_cam_vectors = {
        str(i): {"right_vec": [rv.x, rv.y, rv.z], "up_vec": [uv.x, uv.y, uv.z]}
        for i, (_img, rv, uv) in enumerate(orbit_views)
    }
    markers_px = {}
    for key, (wx, wy, wz) in markers_world.items():
        p = Vector((wx, wy, wz))
        vlabels = {}
        for i, (_img, rv, uv) in enumerate(orbit_views):
            px, py = _orbit_world_to_px(p, cen, rv, uv, scale, s)
            if margin <= px <= s - margin and margin <= py <= s - margin:
                vlabels[str(i)] = [px, py]
        if vlabels:
            markers_px[key] = vlabels

    data = {
        "arp_side": side, "view_type": "orbit", "n_views": _ORBIT_N_VIEWS,
        "hand_world_pos": list(hw),
        "elbow_world_pos": list(ew) if ew is not None else None,
        "arm_len": arm_len, "hand_img_size": _HAND_IMG_SIZE,
        "hand_scale": hand_scale, "orbit_center": list(cen),
        "orbit_cam_vectors": orbit_cam_vectors,
        "markers_world": markers_world, "markers_px": markers_px,
        "source_mesh": source_name,
        "warp": (dict(warp_sig) if warp_sig else None),
        "synthetic": True,
    }
    with open(os.path.join(sample_dir, f"hand_{side}.json"), 'w') as f:
        _json.dump(data, f, indent=2)
    return True


class AUTORIG_OT_GenHandData(bpy.types.Operator):
    """Generate N camera-varied training samples from the current hand(s).
Fix the 20 markers by hand first; this multiplies that pose into N samples
with randomized camera framing."""
    bl_idname = "autorig.gen_hand_data"
    bl_label = "Generate Hand Training Data (Augmented)"
    bl_options = {'REGISTER'}

    def execute(self, context):
        cfg = context.scene.gen_hand_data
        rng = random.Random(cfg.seed)

        props = getattr(context.scene, "autorig_face_objs", None)
        mesh_obj = props.detect_body_obj if props else None
        if mesh_obj is not None and mesh_obj.name not in context.scene.objects:
            mesh_obj = None          # stale saved picker — object not in scene
        if mesh_obj is None:
            meshes = [o for o in context.scene.objects
                      if o.type == 'MESH' and not o.get("autorig_marker")]
            if not meshes:
                self.report({'ERROR'}, "No mesh found.")
                return {'CANCELLED'}
            mesh_obj = max(meshes, key=lambda o: o.dimensions.z)

        # This tool lives at the addon root (so `from .ai_detect` imports resolve),
        # but the trainer reads dev/hand_dataset — write new samples there so they
        # append to the existing training set.
        dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dev", "hand_dataset")
        os.makedirs(dataset_dir, exist_ok=True)

        made = 0
        for _n in range(cfg.n_samples):
            existing = [d for d in os.listdir(dataset_dir) if d.startswith("sample_")]
            idx = len(existing)
            sample_dir = os.path.join(dataset_dir, f"sample_{idx:04d}")
            os.makedirs(sample_dir, exist_ok=True)

            ok_any = False
            for side in ("L", "R"):
                # Sample 0 = the clean original pose+shape (identity); later samples
                # each get their own random warp AND per-finger articulation per side.
                _do_warp = cfg.warp and _n != 0
                _do_rot  = cfg.finger_rotate and _n != 0
                prm   = _hand_warp_params(rng) if _do_warp else _HAND_WARP_ID
                # PER-FINGER ARTICULATION: curl (flexion) + splay per finger, built
                # from the ORIGINAL marker joints, applied BEFORE the global warp.
                # Synthesizes bent/curled + splayed hands from the posed one.
                _artic = None
                if _do_rot:
                    _fj = _finger_joints(side)
                    if _fj is not None:
                        _artic = _finger_rot_params(rng, _fj, cfg)
                # A warp frame is needed whenever we bake a copy (warp OR rotation);
                # with warp off it carries identity params so only rotation applies.
                frame = (_hand_warp_frame(side)
                         if (_do_warp or _artic is not None) else None)
                warp_ob, moved = None, []
                try:
                    render_src = mesh_obj
                    if frame is not None:
                        warp_ob = _warped_hand_copy(mesh_obj, frame, prm, _artic)
                        render_src = warp_ob
                        # Move this side's marker empties through the SAME transform
                        # (labels + framing anchors must match the warped mesh),
                        # restored in finally.
                        for _mn in _side_marker_names(side):
                            _mo = bpy.data.objects.get(_mn)
                            if _mo is None:
                                continue
                            _orig = _mo.location.copy()
                            _pt = np.array([list(_mo.matrix_world.translation)],
                                           dtype=np.float64)
                            if _artic is not None:
                                _pt = _finger_articulate_np(_pt, _artic[0], _artic[1])
                            _w = _hand_warp_np(_pt, frame, prm)[0]
                            _mo.location = Vector((_w[0], _w[1], _w[2]))
                            moved.append((_mo, _orig))
                        bpy.context.view_layer.update()
                    _wsig = None
                    if prm is not _HAND_WARP_ID or _artic is not None:
                        _wsig = dict(prm)
                        if _artic is not None:
                            _wsig["rot"] = _rot_signature(_artic[0])
                    if _save_sample_side(render_src, side, sample_dir, cfg, rng,
                                         source_name=mesh_obj.name, warp_sig=_wsig):
                        ok_any = True
                except Exception as e:
                    print(f"  [gen {side}] sample {idx} failed: {e}")
                finally:
                    for _mo, _orig in moved:
                        _mo.location = _orig
                    if moved:
                        bpy.context.view_layer.update()
                    if warp_ob is not None:
                        try:
                            _wme = warp_ob.data
                            bpy.data.objects.remove(warp_ob, do_unlink=True)
                            bpy.data.meshes.remove(_wme)
                        except Exception:
                            pass

            if ok_any:
                made += 1
            else:
                shutil.rmtree(sample_dir, ignore_errors=True)

        total = len([d for d in os.listdir(dataset_dir) if d.startswith("sample_")])
        self.report({'INFO'}, f"Generated {made} camera-varied samples. Total dataset: {total}.")
        return {'FINISHED'}


class GenHandDataConfig(bpy.types.PropertyGroup):
    n_samples: bpy.props.IntProperty(
        name="Samples", default=20, min=1, max=5000,
        description="How many camera-varied samples to generate from the current hand(s)")
    cam_angle_deg: bpy.props.FloatProperty(
        name="Camera Angle Jitter", default=12.0, min=0.0, max=45.0,
        description="Max random rotation of the orbit forward axis (degrees)")
    cam_scale_jit: bpy.props.FloatProperty(
        name="Camera Scale Jitter", default=0.15, min=0.0, max=0.5,
        description="Fractional random scale of the crop framing")
    cam_center_jit: bpy.props.FloatProperty(
        name="Camera Center Jitter", default=0.06, min=0.0, max=0.3,
        description="Random centre offset as a fraction of crop scale")
    seed: bpy.props.IntProperty(
        name="Seed", default=0, min=0, max=10_000_000,
        description="RNG seed for reproducible generation")
    warp: bpy.props.BoolProperty(
        name="Warp Variants", default=True,
        description="Push mesh + labels through a random smooth hand warp per "
                    "sample (finger length, width, thickness, taper) — the "
                    "Save Face Data xN recipe. Sample 0 stays the clean "
                    "original shape")
    finger_rotate: bpy.props.BoolProperty(
        name="Per-Finger Rotation", default=True,
        description="Give each finger its own random CURL (flexion toward the "
                    "palm, at all 3 joints) + SPLAY (spread/abduction) per sample, "
                    "applied before the warp. Synthesizes bent/curled + splayed "
                    "hands from the one posed hand. Bones stay rigid (no stretch). "
                    "Sample 0 stays the clean original pose")
    finger_curl_deg: bpy.props.FloatProperty(
        name="Max Curl", default=45.0, min=0.0, max=170.0,
        description="Max total flexion per finger (deg), split across MCP/PIP/DIP. "
                    "Curl rotates un-skinned surface points, so the outer surface "
                    "stretches ~linearly with angle (~10% peak at 45deg, ~20% at "
                    "90deg). Keep <=~50 for clean results; strong curls/fists show "
                    "stretch — pose those with the rig instead. Biased toward small")
    finger_splay_deg: bpy.props.FloatProperty(
        name="Max Splay", default=12.0, min=0.0, max=40.0,
        description="Max abduction per finger (deg) in the palm plane; the thumb "
                    "gets ~1.8x for opposition/spread")


class AUTORIG_PT_GenHandData(bpy.types.Panel):
    bl_label = "Hand Data Generator"
    bl_idname = "AUTORIG_PT_GenHandData"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'AutoRig'

    def draw(self, context):
        layout = self.layout
        cfg = context.scene.gen_hand_data
        layout.prop(cfg, "n_samples")
        layout.prop(cfg, "seed")
        layout.prop(cfg, "warp")
        layout.prop(cfg, "finger_rotate")
        _fr = layout.column(align=True)
        _fr.enabled = cfg.finger_rotate
        _fr.prop(cfg, "finger_curl_deg")
        _fr.prop(cfg, "finger_splay_deg")
        box = layout.box()
        box.label(text="Camera jitter:")
        box.prop(cfg, "cam_angle_deg")
        box.prop(cfg, "cam_scale_jit")
        box.prop(cfg, "cam_center_jit")
        layout.operator("autorig.gen_hand_data", icon='RENDER_STILL')


_classes = (GenHandDataConfig, AUTORIG_OT_GenHandData, AUTORIG_PT_GenHandData)


def register():
    for c in _classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.gen_hand_data = bpy.props.PointerProperty(type=GenHandDataConfig)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
    if hasattr(bpy.types.Scene, "gen_hand_data"):
        del bpy.types.Scene.gen_hand_data


if __name__ == "__main__":
    register()
