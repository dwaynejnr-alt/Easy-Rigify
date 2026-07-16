"""
Face training-data generator.
==============================
Saves the CURRENT face (its CORE_FACE_LANDMARKS marker positions) as one training
sample for the neural face detector: renders the six face views via ai_detect's
_render_face_views and projects each marker to per-view pixel labels with the SAME
projection inference uses (_face_world_to_px) — so training data stays byte-
compatible with the model at run time.

WORKFLOW: pick the head/face mesh -> AI/geometric face detect -> correct the
markers by hand -> click "Save Face Data". Repeat across DIVERSE faces (chibi,
realistic, creature, wide/narrow, big/small features) — mesh diversity is the #1
driver of robustness.

AUGMENTED SAVE ("Save Face Data xN"): domain randomization for faces, the same
recipe that made the hand/body models work. Each labeled face is multiplied into
N procedurally WARPED variants — anisotropic width/depth/height, jaw taper,
forehead taper, cheek puff — with the mesh vertices AND the marker labels pushed
through the IDENTICAL smooth warp, then rendered through the same view pipeline.
This directly teaches the width variation whose absence caused the bilateral
X-collapse (markers collapsing to the face centre). Photometric/rotation
augmentation still belongs in the TRAINER (train_face_detector.py).

Output: face_dataset/sample_XXXX/{front,q_left,q_right,side_left,side_right,top}.png
        + meta.json  (landmark order, per-view camera basis, world + pixel labels).

NOTE: WIP tool — registered during face-model development. Move to dev/ and drop
its register() from __init__ before shipping (like gen_body/gen_hand were).
"""

import bpy
import os
import random
import shutil
import json as _json
import numpy as np
from mathutils import Vector, Matrix

from .constants import CORE_FACE_LANDMARKS, FULL_FACE_LANDMARKS
from .ai_detect import _render_face_views, _face_world_to_px

# The neural face model targets the FULL skin-marker set (everything the rig
# needs EXCEPT teeth, tongue and the eyeball centres, which come from the picked
# meshes, not the face surface). CORE_FACE_LANDMARKS was the original 30-anchor
# subset; capturing the full set lets AI Detect Face place every marker.
_SAVE_FACE_LANDMARKS = FULL_FACE_LANDMARKS


def _pick_face_mesh(context):
    props = getattr(context.scene, "autorig_face_objs", None)
    if props and getattr(props, "body_obj", None) and props.body_obj.type == 'MESH':
        return props.body_obj
    if props and getattr(props, "detect_body_obj", None) and props.detect_body_obj.type == 'MESH':
        return props.detect_body_obj
    meshes = [o for o in context.scene.objects
              if o.type == 'MESH' and not o.get("autorig_marker")]
    if not meshes:
        return None
    return max(meshes, key=lambda o: o.dimensions.z)


def _gather_face_markers():
    markers_world = {}
    for name in _SAVE_FACE_LANDMARKS:
        obj = bpy.data.objects.get(f"MARKER_{name}")
        if obj:
            markers_world[name] = list(obj.matrix_world.translation)
    return markers_world


def _save_face_sample(mesh_obj, markers_world):
    """Render the six views of mesh_obj and save one sample with the given
    world-space marker labels. Returns the sample index. Raises on failure."""
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    # max+1, not count: deleting bad samples leaves index gaps, and a count
    # collides with the highest surviving sample.
    _existing = [d for d in os.listdir(dataset_dir) if d.startswith("sample_")]
    idx = (max(int(d.split("_")[1]) for d in _existing) + 1) if _existing else 0
    sample_dir = os.path.join(dataset_dir, f"sample_{idx:04d}")
    os.makedirs(sample_dir)
    try:
        cen, views, S = _render_face_views(mesh_obj)
        temp_dir = views[0].get("temp_dir") if views else None
        try:
            for v in views:
                shutil.copy2(v["path"], os.path.join(sample_dir, f"{v['name']}.png"))
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

        margin = 4
        markers_px = {}
        for name, w in markers_world.items():
            p = Vector(w)
            vlabels = {}
            for v in views:
                px, py = _face_world_to_px(
                    p, cen, Vector(v["right"]), Vector(v["up"]), v["scale"], S)
                if margin <= px <= S - margin and margin <= py <= S - margin:
                    vlabels[v["name"]] = [px, py]
            if vlabels:
                markers_px[name] = vlabels

        # Framing quality gate: when the head-locator fails on a warped mesh
        # (chibi-like proportions) it frames the WHOLE BODY and the face ends
        # up a few dozen pixels wide (~9% of the frame) — useless supervision
        # that would teach the model tiny faces. Only reject that clear-garbage
        # case: a genuinely framed face (even a narrow one) spans ~30%+, so the
        # threshold sits well below any real face but above whole-body framing.
        _fx = [v2["front"][0] for v2 in markers_px.values() if "front" in v2]
        _fy = [v2["front"][1] for v2 in markers_px.values() if "front" in v2]
        _xspan = (max(_fx) - min(_fx)) if _fx else 0
        _yspan = (max(_fy) - min(_fy)) if _fy else 0
        if (len(_fx) < 5
                or _xspan < S * 0.18
                or _yspan < S * 0.15):
            raise ValueError(
                f"bad framing (face spread {_xspan:.0f}x{_yspan:.0f}px of {S} "
                f"-- looks like whole-body framing) -- sample discarded")

        meta = {
            "sample_id":     idx,
            "mesh_name":     mesh_obj.name,
            "landmarks":     list(_SAVE_FACE_LANDMARKS),
            "img_size":      S,
            "center":        [cen.x, cen.y, cen.z],
            "views":         [{k: v[k] for k in ("name", "right", "up", "view_dir", "scale")}
                              for v in views],
            "markers_world": markers_world,
            "markers_px":    markers_px,
        }
        with open(os.path.join(sample_dir, "meta.json"), 'w') as f:
            _json.dump(meta, f, indent=2)
    except Exception:
        shutil.rmtree(sample_dir, ignore_errors=True)
        raise
    return idx


# ── Procedural face warp (domain randomization) ───────────────────────────────

def _warp_params(rng):
    """One random smooth face deformation. Ranges follow the hand/body recipe:
    strong on WIDTH (the X-collapse axis), moderate elsewhere."""
    return {
        "sx":    rng.uniform(0.78, 1.30),   # width — the critical axis
        "sy":    rng.uniform(0.85, 1.18),   # depth
        "sz":    rng.uniform(0.85, 1.18),   # height
        "jaw":   rng.uniform(-0.20, 0.35),  # lower-face taper (xy vs depth below centre)
        "fore":  rng.uniform(-0.15, 0.25),  # upper-face taper
        "cheek": rng.uniform(-0.10, 0.22),  # radial puff at cheek height
    }


_WARP_ID = {"sx": 1.0, "sy": 1.0, "sz": 1.0, "jaw": 0.0, "fore": 0.0, "cheek": 0.0}


def _warp_np(pts, cen, span, prm):
    """Apply the smooth warp to an (N,3) world-space array. The SAME function
    transforms mesh vertices and marker labels, so they stay consistent."""
    n = (pts - cen) / span
    x = n[:, 0] * prm["sx"]
    y = n[:, 1] * prm["sy"]
    z = n[:, 2] * prm["sz"]
    t = np.minimum(np.maximum(0.0, -z), 1.2)          # below face centre
    f = 1.0 + prm["jaw"] * t
    x = x * f; y = y * f
    tt = np.minimum(np.maximum(0.0, z - 0.15), 1.2)   # above brow-ish
    ff = 1.0 + prm["fore"] * tt
    x = x * ff; y = y * ff
    g = np.exp(-((z + 0.15) ** 2) / 0.08)             # cheek-height bump
    puff = 1.0 + prm["cheek"] * g
    x = x * puff; y = y * puff
    return cen + np.stack([x, y, z], axis=1) * span


def _warped_copy(src, cen, span, prm):
    """Evaluated copy of src with world-baked, warped vertices. Caller removes."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = src.evaluated_get(dg)
    me = bpy.data.meshes.new_from_object(ev, depsgraph=dg)
    ob = bpy.data.objects.new(f"_er_facewarp_{src.name}", me)
    bpy.context.collection.objects.link(ob)
    ob.matrix_world = Matrix.Identity(4)
    n = len(me.vertices)
    if n:
        co = np.empty(n * 3, dtype=np.float64)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        mw = np.array(ev.matrix_world, dtype=np.float64)
        co = co @ mw[:3, :3].T + mw[:3, 3]
        co = _warp_np(co, cen, span, prm)
        me.vertices.foreach_set("co", co.astype(np.float32).ravel())
        me.update()
    return ob


class AUTORIG_OT_SaveFaceData(bpy.types.Operator):
    """Render the six face views of the current head mesh and save its core face
marker positions as a training sample. Detect the face, correct the markers, then
click this."""
    bl_idname  = "autorig.save_face_data"
    bl_label   = "Save Face Data"
    bl_options = {'REGISTER'}

    def execute(self, context):
        mesh_obj = _pick_face_mesh(context)
        if mesh_obj is None:
            self.report({'ERROR'}, "No face/head mesh found. Set the Body/Face mesh first.")
            return {'CANCELLED'}
        markers_world = _gather_face_markers()
        if len(markers_world) < len(_SAVE_FACE_LANDMARKS) // 2:
            self.report({'ERROR'},
                "Too few face markers. Detect the face, correct the markers, then save.")
            return {'CANCELLED'}
        try:
            idx = _save_face_sample(mesh_obj, markers_world)
        except Exception as e:
            self.report({'ERROR'}, f"Save failed: {e}")
            return {'CANCELLED'}
        self.report({'INFO'},
            f"Saved sample_{idx:04d} ({len(markers_world)}/{len(_SAVE_FACE_LANDMARKS)} markers). "
            f"Total in face_dataset/: {idx + 1}.")
        return {'FINISHED'}


class AUTORIG_OT_SaveFaceDataAug(bpy.types.Operator):
    """Save this labeled face plus N procedurally WARPED variants (width, jaw,
forehead, cheek deformation). Mesh and marker labels go through the identical
warp, rendered through the same pipeline — the face equivalent of the hand/body
domain randomization. One click per character; expect ~1-2s per variant"""
    bl_idname  = "autorig.save_face_data_aug"
    bl_label   = "Save Face Data xN (warped variants)"
    bl_options = {'REGISTER'}

    def execute(self, context):
        mesh_obj = _pick_face_mesh(context)
        if mesh_obj is None:
            self.report({'ERROR'}, "No face/head mesh found. Set the Body/Face mesh first.")
            return {'CANCELLED'}
        markers_world = _gather_face_markers()
        if len(markers_world) < len(_SAVE_FACE_LANDMARKS) // 2:
            self.report({'ERROR'},
                "Too few face markers. Detect the face, correct the markers, then save.")
            return {'CANCELLED'}

        n_var = max(1, int(getattr(context.scene, "face_aug_variants", 30)))

        # Warp frame: centre + half-extent of the labeled FACE markers.
        pts = np.array(list(markers_world.values()), dtype=np.float64)
        cen = pts.mean(axis=0)
        span = float(max((pts.max(axis=0) - pts.min(axis=0)).max() * 0.5, 1e-3))

        # Meshes that must move WITH the face: the head/body mesh itself plus
        # any visible mesh near the face (eyes, teeth, brows, lashes, hair) —
        # unwarped eyes inside a warped head would mislabel the eye landmarks.
        near = []
        for o in context.scene.objects:
            if o.type != 'MESH' or o.get("autorig_marker"):
                continue
            if o.hide_render and o.hide_get():
                continue
            if o is mesh_obj:
                continue
            oc = np.array(o.matrix_world.translation, dtype=np.float64)
            if np.linalg.norm(oc - cen) <= span * 6.0:
                near.append(o)
        co_meshes = [mesh_obj] + near

        # Framing markers (_locate_head_bounds reads them) must warp too.
        frame_markers = [m for m in (bpy.data.objects.get("MARKER_HEAD"),
                                     bpy.data.objects.get("MARKER_NECK"))
                         if m is not None]

        base_seed = random.randrange(1 << 30)
        saved, first_idx = 0, None
        for k in range(n_var):
            prm = _WARP_ID if k == 0 else _warp_params(random.Random(base_seed + k))
            if k == 0:
                # variant 0 = the clean original, straight through the plain path
                try:
                    idx = _save_face_sample(mesh_obj, markers_world)
                    first_idx = idx if first_idx is None else first_idx
                    saved += 1
                except Exception as e:
                    self.report({'ERROR'}, f"Save failed on original: {e}")
                    return {'CANCELLED'}
                continue

            copies, hidden, moved = [], [], []
            try:
                for src in co_meshes:
                    copies.append(_warped_copy(src, cen, span, prm))
                    if not src.hide_render:
                        src.hide_render = True
                        if src not in hidden:
                            hidden.append(src)
                    try:
                        if not src.hide_get():
                            src.hide_set(True)
                            if src not in hidden:
                                hidden.append(src)
                    except Exception:
                        pass
                for m in frame_markers:
                    _orig = m.location.copy()
                    _w = _warp_np(np.array([list(m.location)]), cen, span, prm)[0]
                    m.location = Vector((_w[0], _w[1], _w[2]))
                    moved.append((m, _orig))
                mk_w = {}
                _names = list(markers_world.keys())
                _wpts = _warp_np(np.array([markers_world[nm] for nm in _names],
                                          dtype=np.float64), cen, span, prm)
                for nm, wp in zip(_names, _wpts):
                    mk_w[nm] = [float(wp[0]), float(wp[1]), float(wp[2])]
                idx = _save_face_sample(copies[0], mk_w)
                first_idx = idx if first_idx is None else first_idx
                saved += 1
            except Exception as e:
                print(f"[face-aug] variant {k} failed: {e}")
            finally:
                for m, _orig in moved:
                    m.location = _orig
                for src in hidden:
                    try:
                        src.hide_render = False
                        src.hide_set(False)
                    except Exception:
                        pass
                for ob in copies:
                    try:
                        _me = ob.data
                        bpy.data.objects.remove(ob, do_unlink=True)
                        bpy.data.meshes.remove(_me)
                    except Exception:
                        pass

        if not saved:
            self.report({'ERROR'}, "No variants saved — see console.")
            return {'CANCELLED'}
        self.report({'INFO'},
            f"Saved {saved}/{n_var} samples (sample_{first_idx:04d}+) "
            f"with warped variants.")
        return {'FINISHED'}


class AUTORIG_PT_GenFaceData(bpy.types.Panel):
    bl_label = "Face Data Generator"
    bl_idname = "AUTORIG_PT_GenFaceData"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'AutoRig'

    def draw(self, context):
        layout = self.layout
        layout.label(text="Detect -> fix face markers -> save")
        layout.operator("autorig.save_face_data", icon='RENDER_STILL')
        col = layout.column(align=True)
        if hasattr(context.scene, "face_aug_variants"):
            col.prop(context.scene, "face_aug_variants")
        col.operator("autorig.save_face_data_aug", icon='MOD_LATTICE')


_classes = (AUTORIG_OT_SaveFaceData, AUTORIG_OT_SaveFaceDataAug,
            AUTORIG_PT_GenFaceData)


def register():
    for c in _classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.face_aug_variants = bpy.props.IntProperty(
        name="Variants", default=30, min=2, max=200,
        description="How many samples to save per character: the clean "
                    "original plus N-1 procedurally warped variants")


def unregister():
    if hasattr(bpy.types.Scene, "face_aug_variants"):
        del bpy.types.Scene.face_aug_variants
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
