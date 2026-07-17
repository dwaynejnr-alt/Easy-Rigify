# markers.py — Body/face marker placement, rig alignment, and UI panels for Easy Rigify.
import bpy
from bpy.app.handlers import persistent
import math
import mathutils
from mathutils import Vector, Matrix, Euler
import os
from contextlib import contextmanager
import numpy as np
try:
    import gpu
    import blf
    from gpu_extras.batch import batch_for_shader as _batch_for_shader
except ImportError:
    gpu = None
    blf = None
    _batch_for_shader = None

from .constants import (
    BODY_SIZE, ARM_SIZE, FINGER_SIZE, FACE_SIZE,
    TIP_EXTEND, FOOT_Y, HEEL_X, HEEL_Y,
    ARM_MARKER_BASES, FINGER_PREFIXES, FINGER_01_BONES, FINGER_BASE_NAMES,
    SINGLE_MARKERS, BILATERAL_MARKERS, ALL_MARKERS,
    FACE_SINGLE, FACE_BILATERAL, ALL_FACE_MARKERS,
    FACE_DIRECT_MAP, FACE_DIRECT_Y_MAP, FACE_TAIL_MAP,
    FACE_AIM_TAIL_MAP, FACE_EQUAL_PAIR_CHAINS,
    FACE_BONE_TAIL_FROM_BONE_HEAD, FACE_TRANSLATE_CHAINS,
    FACE_INTERP_CHAINS, FACE_CHAIN_MAP,
    FINGER_BONES_L, PALM_BONES_L, ROLL_RULES,
)
from .utils import (
    get_icon, get_roll_rule, get_or_create_collection,
    get_marker_col, get_face_col,
    _apply_marker_style, set_mesh_select, set_xray,
    _mesh_bbox_world, get_base, _ensure_marker_empty, make_empty,
)
from . import ai_detect as _ai

# Module-level handles for viewport draw callbacks
_marker_hint_handle      = None
_marker_billboard_handle = None
_marker_lines_handle     = None
_marker_images: dict       = {}    # icon key → bpy.types.Image
_marker_gpu_textures: dict = {}    # icon key → gpu.types.GPUTexture (cached, never rebuilt)
_icon_key_cache: dict      = {}    # obj.name → icon key (names never change, safe to cache forever)
_index_cache: dict         = {}    # n_markers → pre-built triangle index list (never changes shape)
_hint_desc_cache: dict     = {}    # desc string → wrapped line list (text never changes)
_billboard_shader          = None  # cached across frames
_hint_shader               = None  # cached across frames

# Pre-allocated per-frame projection buffers — grown on demand, never shrunk.
# Eliminates per-frame numpy object allocations that drive Python GC pressure and
# cause the intermittent 1-3 s freeze when gen2 GC walks bpy.data.
_prj_cap:  int = 0
_prj_pm   = np.empty((4, 4), dtype=np.float32)   # perspective matrix (reused each frame)
_prj_locs = None   # (cap, 4) float32 — world positions + w=1
_prj_clip = None   # (cap, 4) float32 — clip-space result of pm @ locs.T
_prj_ok   = None   # (cap,)   bool    — w > threshold AND on-screen
_prj_px_x = None   # (cap,)   float32 — screen x in pixels
_prj_px_y = None   # (cap,)   float32 — screen y in pixels
_prj_half = None   # (cap,)   int32   — pixel half-size per marker

# Pre-allocated per-group vertex/UV buffers — keyed by icon_key, grown as needed.
_grp_pos_buf: dict = {}   # icon_key → np.ndarray (cap*4, 2) float32
_grp_uv_buf:  dict = {}   # icon_key → np.ndarray (cap*4, 2) float32
_grp_cap:     dict = {}   # icon_key → allocated capacity (in marker count)

# Constant UV tile — allocated once, never changed.
_UV_TILE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)



def _load_marker_images():
    _marker_gpu_textures.clear()   # invalidate any cached GPU textures on reload
    icons_dir = os.path.join(os.path.dirname(__file__), "icons")
    for key in ("Marker_green", "Marker_orange", "Marker_purple", "Marker_yellow", "Marker_red", "Marker_blue", "Marker_active"):
        path = os.path.join(icons_dir, f"{key}.png")
        if not os.path.isfile(path):
            continue
        img = bpy.data.images.get(f"{key}.png")
        if img is None:
            img = bpy.data.images.load(path)
        img.use_fake_user = True
        # Force the pixel buffer to load from disk now, so a GPU texture built on
        # the first draw gets real data instead of a magenta "no data" placeholder.
        if not img.has_data:
            try:
                img.pixels[0]
            except Exception:
                pass
        _marker_images[key] = img


def _unload_marker_images():
    global _billboard_shader, _hint_shader
    for img in list(_marker_images.values()):
        try:
            bpy.data.images.remove(img)
        except Exception:
            pass
    _marker_images.clear()
    _marker_gpu_textures.clear()
    _icon_key_cache.clear()
    _index_cache.clear()
    _hint_desc_cache.clear()
    _grp_pos_buf.clear()
    _grp_uv_buf.clear()
    _grp_cap.clear()
    _billboard_shader = None
    _hint_shader      = None


@persistent
def _reset_marker_cache(*_args):
    """load_post handler. After a file load the cached image/GPU datablocks belong to
    the PREVIOUS file and are stale, so drop them AND reload the icon images — without
    the reload the billboard draw handler finds no images and the marker icons vanish
    until the addon is re-registered. @persistent so it survives every file load, and
    *_args so Blender can call it with the load-handler argument(s)."""
    _marker_gpu_textures.clear()
    _icon_key_cache.clear()
    _index_cache.clear()
    _hint_desc_cache.clear()
    _grp_pos_buf.clear()
    _grp_uv_buf.clear()
    _grp_cap.clear()
    _marker_images.clear()
    try:
        _load_marker_images()   # repopulate so icons reappear after the load
    except Exception as _e:
        print(f"[markers] icon reload after file load failed: {_e}")


def _marker_icon_key(obj_name: str) -> str:
    raw  = obj_name[len('MARKER_'):]
    base = raw[:-2] if raw.endswith(('_L', '_R')) else raw
    if base == 'FACE_EYE_CENTER':
        return 'Marker_red'
    if any(base.startswith(p) for p in ('FACE_EYE_', 'FACE_LID_', 'FACE_CREASE_')):
        return 'Marker_blue'
    if any(base.startswith(p) for p in ('FACE_TEETH', 'FACE_TONGUE')):
        return 'Marker_yellow'
    if any(base.startswith(p) for p in ('FACE_CHEEK', 'FACE_CHIN', 'FACE_EAR')):
        return 'Marker_purple'
    if base.startswith('FACE_'):
        return 'Marker_orange'
    if any(base.startswith(p) for p in ('THUMB', 'FINGER_')):
        return 'Marker_yellow'
    return 'Marker_green'


# ── Marker connection lines (ARP-style skeleton preview) ───────────────────
_MARKER_MIDLINE   = {"PELVIS", "SPINE_001", "SPINE_002", "CHEST", "NECK", "HEAD"}
_MARKER_MID_CHAIN = ["PELVIS", "SPINE_001", "SPINE_002", "CHEST", "NECK", "HEAD"]
# Per-side chains. Midline names (CHEST/PELVIS/…) are used as-is; everything else
# gets the _L / _R side suffix when resolved.
_MARKER_SIDE_CHAINS = [
    ["CHEST",  "SHOULDER", "ARM", "ELBOW", "HAND", "HAND_TIP"],
    ["PELVIS", "THIGH", "SHIN", "FOOT", "TOES"],
    ["FOOT",   "HEEL"],
    ["HAND", "FINGER_INDEX_1",  "FINGER_INDEX_2",  "FINGER_INDEX_3",  "FINGER_INDEX_TIP"],
    ["HAND", "FINGER_MIDDLE_1", "FINGER_MIDDLE_2", "FINGER_MIDDLE_3", "FINGER_MIDDLE_TIP"],
    ["HAND", "FINGER_RING_1",   "FINGER_RING_2",   "FINGER_RING_3",   "FINGER_RING_TIP"],
    ["HAND", "FINGER_PINKY_1",  "FINGER_PINKY_2",  "FINGER_PINKY_3",  "FINGER_PINKY_TIP"],
    ["HAND", "THUMB_1", "THUMB_2", "THUMB_3", "THUMB_TIP"],
]

# Face connectivity. FACE_SINGLE names are midline; FACE_BILATERAL names take a side
# suffix (note FACE_NOSE_BRIDGE exists in BOTH — in a side chain the side one wins).
_FACE_SINGLE_NAMES    = {n for n, *_ in FACE_SINGLE}
_FACE_BILATERAL_NAMES = {n for n, *_ in FACE_BILATERAL}
_FACE_MID_CHAIN = ["FACE_FOREHEAD", "FACE_BROW", "FACE_NOSE_BRIDGE", "FACE_NOSE_TIP",
                   "FACE_NOSE_BOT", "FACE_LIP_T", "FACE_LIP_B", "FACE_LIP_BOT",
                   "FACE_CHIN", "FACE_JAW"]
_FACE_SIDE_CHAINS = [
    ["FACE_EYE_INNER", "FACE_EYE_TOP", "FACE_EYE_OUTER", "FACE_EYE_BOT", "FACE_EYE_INNER"],
    ["FACE_BROW", "FACE_BROW_1", "FACE_BROW_2", "FACE_BROW_3", "FACE_BROW_OUTER"],
    ["FACE_LIP_T", "FACE_MOUTH_TOP", "FACE_MOUTH_CORNER"],
    ["FACE_LIP_B", "FACE_MOUTH_BOT", "FACE_MOUTH_CORNER"],
    ["FACE_NOSE_TIP", "FACE_NOSE_WING", "FACE_NOSE_BRIDGE"],
    ["FACE_CHIN", "FACE_CHIN_SIDE", "FACE_JAW_SIDE", "FACE_EAR"],
    ["FACE_CHEEK_TOP", "FACE_CHEEK"],
    ["FACE_FOREHEAD", "FACE_FOREHEAD_SIDE", "FACE_FOREHEAD_SIDE_1",
     "FACE_FOREHEAD_SIDE_2", "FACE_FOREHEAD_SIDE_3"],
    ["FACE_TEMPLE", "FACE_EAR"],
]
_line_shader = None


def _draw_marker_lines():
    """GPU draw callback (POST_VIEW) — draw bone-like lines connecting the markers,
    so the layout reads as a skeleton (like Auto-Rig Pro)."""
    try:
        if gpu is None or _batch_for_shader is None:
            return
        ctx = bpy.context
        if not ctx or not ctx.area or ctx.area.type != 'VIEW_3D':
            return
        region = ctx.region
        if region is None:
            return

        def _wp(nm):
            o = bpy.data.objects.get("MARKER_" + nm)
            return o.matrix_world.translation.copy() if o else None

        _body_col = bpy.data.collections.get("RigifyMarkers")
        _face_col = bpy.data.collections.get("RigifyFaceMarkers")
        _body_on  = bool(_body_col) and not _body_col.hide_viewport
        _face_on  = bool(_face_col) and not _face_col.hide_viewport

        segs = []
        if _body_on:
            for a, b in zip(_MARKER_MID_CHAIN, _MARKER_MID_CHAIN[1:]):
                pa, pb = _wp(a), _wp(b)
                if pa and pb:
                    segs += [pa, pb]
            for side in ("L", "R"):
                def _res(nm):
                    return nm if nm in _MARKER_MIDLINE else f"{nm}_{side}"
                for chain in _MARKER_SIDE_CHAINS:
                    for a, b in zip(chain, chain[1:]):
                        pa, pb = _wp(_res(a)), _wp(_res(b))
                        if pa and pb:
                            segs += [pa, pb]

        if _face_on:
            # Face midline
            for a, b in zip(_FACE_MID_CHAIN, _FACE_MID_CHAIN[1:]):
                pa, pb = _wp(a), _wp(b)
                if pa and pb:
                    segs += [pa, pb]
            # Face per-side (bilateral name wins over a same-named midline marker)
            for side in ("L", "R"):
                def _fres(nm):
                    return f"{nm}_{side}" if nm in _FACE_BILATERAL_NAMES else nm
                for chain in _FACE_SIDE_CHAINS:
                    for a, b in zip(chain, chain[1:]):
                        pa, pb = _wp(_fres(a)), _wp(_fres(b))
                        if pa and pb:
                            segs += [pa, pb]
        if not segs:
            return

        global _line_shader
        if _line_shader is None:
            _line_shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')

        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE')   # draw over the mesh, like the markers
        batch = _batch_for_shader(_line_shader, 'LINES', {"pos": segs})
        _line_shader.bind()
        _line_shader.uniform_float("viewportSize", (region.width, region.height))
        _line_shader.uniform_float("lineWidth", 2.0)
        _line_shader.uniform_float("color", (0.15, 0.5, 1.0, 0.9))
        batch.draw(_line_shader)
        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('LESS_EQUAL')
    except Exception:
        pass


def _draw_marker_billboards():
    """GPU draw callback — screen-space billboard image over every marker empty."""
    try:
        if gpu is None or _batch_for_shader is None:
            return

        ctx = bpy.context
        if not ctx or not ctx.area or ctx.area.type != 'VIEW_3D':
            return
        region = ctx.region
        rv3d   = ctx.region_data
        if not region or not rv3d:
            return

        markers = []
        for col_name in ("RigifyMarkers", "RigifyFaceMarkers"):
            col = bpy.data.collections.get(col_name)
            if col and not col.hide_viewport:
                markers.extend(o for o in col.objects if o.name.startswith('MARKER_'))
        if not markers:
            return

        if not _marker_images:
            _load_marker_images()
        if not _marker_images:
            return

        # Pixel-scale from window_matrix (pure projection — rotation-free).
        try:
            vc_view   = rv3d.view_matrix @ rv3d.view_location
            vz        = max(-vc_view[2], 0.01)
            wm11      = rv3d.window_matrix[1][1]
            px_per_wu = region.height * wm11 / (2.0 * vz) if rv3d.is_perspective \
                        else region.height * wm11 / 2.0
            px_per_wu = max(px_per_wu, 1.0)
        except Exception:
            px_per_wu = 50.0

        global _billboard_shader
        if _billboard_shader is None:
            _billboard_shader = gpu.shader.from_builtin('IMAGE')
        shader     = _billboard_shader
        active_obj = ctx.active_object

        visible = [obj for obj in markers if not obj.hide_viewport]
        if not visible:
            return

        # --- Phase 1: project all markers — using pre-allocated buffers to avoid
        # per-frame numpy object allocations that trigger Python gen2 GC freezes. ---
        global _prj_cap, _prj_locs, _prj_clip, _prj_ok, _prj_px_x, _prj_px_y, _prj_half
        n = len(visible)
        if n > _prj_cap:
            cap = max(n + 16, int(_prj_cap * 1.5 + 16))
            _prj_locs = np.empty((cap, 4), dtype=np.float32)
            _prj_clip = np.empty((cap, 4), dtype=np.float32)
            _prj_ok   = np.empty(cap, dtype=np.bool_)
            _prj_px_x = np.empty(cap, dtype=np.float32)
            _prj_px_y = np.empty(cap, dtype=np.float32)
            _prj_half = np.empty(cap, dtype=np.int32)
            _prj_cap  = cap

        # Fill positions in-place (no new array allocated)
        locs = _prj_locs[:n]
        for i, obj in enumerate(visible):
            t = obj.matrix_world.translation
            locs[i, 0] = t.x;  locs[i, 1] = t.y;  locs[i, 2] = t.z;  locs[i, 3] = 1.0

        # Reuse _prj_pm buffer for the perspective matrix (fills in-place, no new array)
        _prj_pm[:] = rv3d.perspective_matrix
        np.dot(locs, _prj_pm.T, out=_prj_clip[:n])
        clip = _prj_clip[:n]
        w    = clip[:, 3]

        # Compute ok mask, screen coords, and half-sizes — all in pre-allocated buffers
        ok   = _prj_ok[:n]
        px_x = _prj_px_x[:n]
        px_y = _prj_px_y[:n]
        half = _prj_half[:n]

        np.greater(w, 1e-6, out=ok)

        w2 = region.width  * 0.5
        h2 = region.height * 0.5
        np.divide(clip[:, 0], w, out=px_x, where=ok)
        np.divide(clip[:, 1], w, out=px_y, where=ok)
        np.multiply(px_x, w2, out=px_x, where=ok)
        np.add(px_x, w2, out=px_x, where=ok)
        np.multiply(px_y, h2, out=px_y, where=ok)
        np.add(px_y, h2, out=px_y, where=ok)

        # Off-screen cull — zero-cost: markers outside viewport skip vertex building
        M  = 64.0
        rw = region.width  + M
        rh = region.height + M
        ok &= (px_x >= -M) & (px_x <= rw) & (px_y >= -M) & (px_y <= rh)

        # Pixel half-sizes in-place
        for i, obj in enumerate(visible):
            s = int(obj.empty_display_size * max(obj.scale) * px_per_wu * 0.85)
            half[i] = max(3, min(s, 50))

        # --- Phase 2: group by icon key ---
        groups = {}
        for i, obj in enumerate(visible):
            if not ok[i]:
                continue
            if obj is active_obj:
                icon_key = 'Marker_active'
            else:
                icon_key = _icon_key_cache.get(obj.name)
                if icon_key is None:
                    icon_key = _marker_icon_key(obj.name)
                    _icon_key_cache[obj.name] = icon_key

            gputex = _marker_gpu_textures.get(icon_key)
            if gputex is None:
                img = _marker_images.get(icon_key)
                if img is None:
                    continue
                # Skip (don't cache) until the image's pixels are actually loaded
                # from disk — building a GPU texture from an empty image yields the
                # magenta "no data" placeholder, and caching it makes every marker
                # stay purple until an addon reload clears the cache.
                if not img.has_data:
                    try:
                        img.pixels[0]          # touch the buffer to trigger the lazy load
                    except Exception:
                        pass
                    if not img.has_data:
                        continue
                gputex = gpu.texture.from_image(img)
                _marker_gpu_textures[icon_key] = gputex

            if icon_key not in groups:
                groups[icon_key] = (gputex, [])
            groups[icon_key][1].append(i)

        if not groups:
            return

        # --- Phase 3: build vertex arrays from pre-allocated per-group buffers, draw ---
        gpu.state.blend_set('ALPHA')
        for icon_key, (gputex, idxs) in groups.items():
            ng = len(idxs)

            # Grow per-group buffers only when this group's marker count increases
            if _grp_cap.get(icon_key, 0) < ng:
                cap = max(ng + 4, int(_grp_cap.get(icon_key, 0) * 1.5 + 4))
                _grp_pos_buf[icon_key] = np.empty((cap * 4, 2), dtype=np.float32)
                _grp_uv_buf[icon_key]  = np.tile(_UV_TILE, (cap, 1))
                _grp_cap[icon_key]     = cap

            pos_arr = _grp_pos_buf[icon_key][:ng * 4]
            uv_arr  = _grp_uv_buf[icon_key][:ng * 4]

            # Fancy-index into projection buffers (copies a small slice, ng ≤ ~50)
            ia = np.array(idxs, dtype=np.intp)
            gx = px_x[ia]
            gy = px_y[ia]
            gh = half[ia].astype(np.float32)
            mh = 3.0 if icon_key == 'Marker_yellow' else 5.0
            gh[gh < mh] = mh

            # Fill pos_arr in-place — no new large array allocated
            pos_arr[0::4, 0] = gx - gh;  pos_arr[0::4, 1] = gy - gh
            pos_arr[1::4, 0] = gx + gh;  pos_arr[1::4, 1] = gy - gh
            pos_arr[2::4, 0] = gx + gh;  pos_arr[2::4, 1] = gy + gh
            pos_arr[3::4, 0] = gx - gh;  pos_arr[3::4, 1] = gy + gh

            tri_idx = _index_cache.get(ng)
            if tri_idx is None:
                base    = np.arange(ng, dtype=np.uint32) * 4
                tris    = np.empty((ng * 2, 3), dtype=np.uint32)
                tris[0::2] = np.column_stack([base, base+1, base+2])
                tris[1::2] = np.column_stack([base, base+2, base+3])
                tri_idx = tris.tolist()
                _index_cache[ng] = tri_idx

            batch = _batch_for_shader(shader, 'TRIS',
                                      {"pos": pos_arr, "texCoord": uv_arr},
                                      indices=tri_idx)
            shader.bind()
            shader.uniform_sampler('image', gputex)
            batch.draw(shader)
        gpu.state.blend_set('NONE')

    except Exception:
        pass

# ---------------------------------------------------------------------------
# Viewport hint overlay — palette + per-marker text descriptions
# ---------------------------------------------------------------------------

_HINT_PALETTE = {
    "body":    (0.10, 0.30, 0.65, 0.90),
    "arm":     (0.20, 0.55, 0.25, 0.90),
    "face":    (0.65, 0.15, 0.55, 0.90),
    "finger":  (0.65, 0.45, 0.10, 0.90),
    "default": (0.18, 0.18, 0.18, 0.85),
}

_MARKER_HINTS = {
    "PELVIS":          ("Pelvis",      "Place at the hip joint center", "body"),
    "SPINE_001":       ("Spine 001",   "Roughly navel height.", "body"),
    "SPINE_002":       ("Spine 002",   "Between navel and ribcage.", "body"),
    "CHEST":           ("Chest",       "Base of the ribcage", "body"),
    "NECK":            ("Neck",        "Base of the neck", "body"),
    "HEAD":            ("Head",        "Top of neck — where head meets spine.", "body"),
    "SHOULDER":        ("Shoulder",    "Tip of the shoulder", "arm"),
    "ARM":             ("Upper Arm",   "Mid-point of the upper arm.", "arm"),
    "ELBOW":           ("Elbow",       "Centre of the elbow joint.", "arm"),
    "HAND":            ("Wrist/Hand",  "Centre of the wrist joint.", "arm"),
    "THIGH":           ("Thigh",       "Hip socket / top of the leg.", "body"),
    "SHIN":            ("Shin/Knee",   "Centre of the knee joint.", "body"),
    "FOOT":            ("Foot/Ankle",  "Centre of the ankle joint.", "body"),
    "TOES":            ("Toes",        "Ball of the foot.", "body"),
    "HEEL":            ("Heel",        "Back of the heel bone.", "body"),
    "BREAST":          ("Breast",      "Breast bone apex.", "body"),
    "FACE_LIP":        ("Lips",        "Centre of the upper/lower lip surface.", "face"),
    "FACE_MOUTH":      ("Mouth",       "Mouth corner or secondary lip control point.", "face"),
    "FACE_BROW":       ("Brow",        "Eyebrow arc control point.", "face"),
    "FACE_EYE":        ("Eye",         "Eye centre or corner control point.", "face"),
    "FACE_LID":        ("Eyelid",      "Eyelid crease or corner point.", "face"),
    "FACE_CREASE":     ("Lid Crease",  "eyelid crease.", "face"),
    "FACE_NOSE":       ("Nose",        "Nose bridge, tip, or wing.", "face"),
    "FACE_CHIN":       ("Chin",        "Chin or lower jaw point.", "face"),
    "FACE_JAW":        ("Jaw",         "Jaw line control point.", "face"),
    "FACE_CHEEK":      ("Cheek",       "Cheek or malar control point.", "face"),
    "FACE_FOREHEAD":   ("Forehead",    "Forehead centre or side point.", "face"),
    "FACE_TEMPLE":     ("Temple",      "Temple bone area - Side of the face.", "face"),
    "FACE_EAR":        ("Ear",         "Ear attachment point.", "face"),
    "FACE_TONGUE":     ("Tongue",      "Tongue segment control point.", "face"),
    "FACE_TEETH":      ("Teeth",       "Teeth alignment point.", "face"),
    "THUMB":           ("Thumb",       "Thumb bone segment.", "finger"),
    "FINGER_INDEX":    ("Index",       "Index finger bone segment.", "finger"),
    "FINGER_MIDDLE":   ("Middle",      "Middle finger bone segment.", "finger"),
    "FINGER_RING":     ("Ring",        "Ring finger bone segment.", "finger"),
    "FINGER_PINKY":    ("Pinky",       "Pinky finger bone segment.", "finger"),
}


def _draw_marker_hint():
    """GPU draw callback — blf+shader quad hint overlay in the 3D viewport."""
    try:
        if gpu is None or blf is None or _batch_for_shader is None:
            return
        ctx = bpy.context
        if not ctx or not ctx.area or ctx.area.type != 'VIEW_3D':
            return
        if not getattr(ctx.scene, 'autorig_show_hints', True):
            return
        ao = ctx.active_object
        if not ao or not ao.name.startswith('MARKER_'):
            return
        raw  = ao.name[len('MARKER_'):]
        base = raw[:-2] if raw.endswith(('_L', '_R')) else raw
        entry = _MARKER_HINTS.get(base)
        if not entry:
            for key in _MARKER_HINTS:
                if base.startswith(key):
                    entry = _MARKER_HINTS[key]
                    break
        if not entry:
            return
        label, desc, col_key = entry
        hdr_rgba = _HINT_PALETTE.get(col_key, _HINT_PALETTE['default'])

        pad = 16; bw = 360; hdr_h = 30; dsc_h = 56; x0 = pad; y0 = pad
        global _hint_shader
        if _hint_shader is None:
            _hint_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        shader = _hint_shader
        gpu.state.blend_set('ALPHA')

        def _quad(rx, ry, rw, rh, color):
            verts = [(rx, ry), (rx + rw, ry), (rx + rw, ry + rh), (rx, ry + rh)]
            b = _batch_for_shader(shader, 'TRI_FAN', {"pos": verts})
            shader.uniform_float("color", color)
            b.draw(shader)

        _quad(x0, y0,         bw, dsc_h, (0.08, 0.08, 0.08, 0.84))
        _quad(x0, y0 + dsc_h, bw, hdr_h, hdr_rgba)
        gpu.state.blend_set('NONE')

        fid = 0

        def _size(pt):
            try:
                blf.size(fid, pt)
            except TypeError:
                blf.size(fid, pt, 72)

        _size(13); blf.color(fid, 1, 1, 1, 1)
        blf.position(fid, x0 + 10, y0 + dsc_h + 9, 0)
        blf.draw(fid, f"{label}   ·   {ao.name}")

        # Word-wrap cached — desc text never changes, no need to re-split every frame
        lines = _hint_desc_cache.get(desc)
        if lines is None:
            words = desc.split(); lines = []; cur = ""
            for w in words:
                test = (cur + " " + w).strip()
                if len(test) > 50:
                    lines.append(cur); cur = w
                else:
                    cur = test
            if cur:
                lines.append(cur)
            lines = lines[:3]
            _hint_desc_cache[desc] = lines

        _size(11); blf.color(fid, 0.82, 0.82, 0.82, 1)
        for i, ln in enumerate(lines):
            blf.position(fid, x0 + 10, y0 + dsc_h - 20 - i * 18, 0)
            blf.draw(fid, ln)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Property Groups
# ---------------------------------------------------------------------------

def _is_mesh_poll(self, obj):
    """Null-safe poll for object PointerProperties.

    Blender's collection-search widget can invoke poll with a None or freed
    object while the search list is rebuilding (e.g. right after operators
    create/remove temp objects + markers), which crashed on a bare `o.type`
    (EXCEPTION_ACCESS_VIOLATION in rna_collection_search_update_fn). Guard it.
    """
    return obj is not None and getattr(obj, 'type', None) == 'MESH'


class AutoRigFaceObjProps(bpy.types.PropertyGroup):
    """Mesh object references and face-section settings for face detection."""
    detect_body_obj: bpy.props.PointerProperty(
        name="Body Mesh",
        description="Character body mesh used for Auto Detect Body & Legs — pick once, reuse every time",
        type=bpy.types.Object,
        poll=_is_mesh_poll,
    )
    body_obj:      bpy.props.PointerProperty(type=bpy.types.Object, poll=_is_mesh_poll)
    show_facial:   bpy.props.BoolProperty(default=False)
    use_tongue:    bpy.props.BoolProperty(default=False)
    tongue_obj:    bpy.props.PointerProperty(type=bpy.types.Object, poll=_is_mesh_poll)
    use_teeth:     bpy.props.BoolProperty(default=False)
    teeth_count:   bpy.props.EnumProperty(
        items=[('SPLIT', 'Split', 'Separate top and bottom meshes'), ('COMBINED', 'Combined', 'One mesh for both')], default='SPLIT')
    teeth_top_obj: bpy.props.PointerProperty(type=bpy.types.Object, poll=_is_mesh_poll)
    teeth_bot_obj: bpy.props.PointerProperty(type=bpy.types.Object, poll=_is_mesh_poll)
    teeth_obj:     bpy.props.PointerProperty(type=bpy.types.Object, poll=_is_mesh_poll)
    use_eyes:      bpy.props.BoolProperty(default=False)
    eye_count:     bpy.props.EnumProperty(
        items=[('SPLIT', 'Split', 'Separate left and right eye meshes'), ('COMBINED', 'Combined', 'One mesh for both eyes')], default='SPLIT')
    eye_obj:       bpy.props.PointerProperty(type=bpy.types.Object, poll=_is_mesh_poll)
    eye_l_obj:     bpy.props.PointerProperty(type=bpy.types.Object, poll=_is_mesh_poll)
    eye_r_obj:     bpy.props.PointerProperty(type=bpy.types.Object, poll=_is_mesh_poll)
    use_brows:     bpy.props.BoolProperty(default=False)
    brow_count:    bpy.props.EnumProperty(
        items=[('SPLIT', 'Split', 'Separate left and right brow meshes'), ('COMBINED', 'Combined', 'One mesh for both brows')], default='SPLIT')
    brow_obj:      bpy.props.PointerProperty(type=bpy.types.Object, poll=_is_mesh_poll)
    brow_l_obj:    bpy.props.PointerProperty(type=bpy.types.Object, poll=_is_mesh_poll)
    brow_r_obj:    bpy.props.PointerProperty(type=bpy.types.Object, poll=_is_mesh_poll)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_marker_positions(col_name="RigifyMarkers"):
    pos = {}
    col = bpy.data.collections.get(col_name)
    if col is None:
        return pos
    for obj in col.objects:
        name = obj.name
        if name.startswith("MARKER_"):
            name = name[7:]
        pos[name] = obj.location.copy()
    return pos


def _find_metarig(context):
    """Return the first armature that looks like a Rigify metarig (no DEF- bones)."""
    ao = context.active_object
    if ao and ao.type == 'ARMATURE' and not any(
            b.name.startswith('DEF-') for b in ao.data.bones):
        return ao
    for obj in context.scene.objects:
        if obj.type == 'ARMATURE' and not any(
                b.name.startswith('DEF-') for b in obj.data.bones):
            return obj
    return None


# ---------------------------------------------------------------------------
# Simple toggle / utility operators
# ---------------------------------------------------------------------------

class AUTORIG_OT_ToggleMarkers(bpy.types.Operator):
    bl_idname = "autorig.toggle_markers"
    bl_label = "Toggle Markers"
    bl_description = "Show or hide all marker empties"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        for col_name in ("RigifyMarkers", "RigifyFaceMarkers"):
            col = bpy.data.collections.get(col_name)
            if col:
                col.hide_viewport = not col.hide_viewport
        return {'FINISHED'}


class AUTORIG_OT_ToggleXRay(bpy.types.Operator):
    bl_idname = "autorig.toggle_xray"
    bl_label = "Toggle X-Ray"
    bl_description = "Toggle X-Ray shading in the 3D viewport"
    bl_options = {'REGISTER'}

    def execute(self, context):
        set_xray(context, not any(
            sp.shading.show_xray
            for area in context.screen.areas if area.type == 'VIEW_3D'
            for sp in area.spaces if sp.type == 'VIEW_3D'
        ))
        return {'FINISHED'}


class AUTORIG_OT_ToggleMeshSel(bpy.types.Operator):
    """Toggle selection on all mesh objects"""
    bl_idname  = "autorig.toggle_mesh_selection"
    bl_label   = "Toggle Mesh Select"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        meshes = [o for o in context.scene.objects if o.type == 'MESH']
        if not meshes:
            self.report({'INFO'}, "No mesh objects.")
            return {'FINISHED'}
        new_state = not meshes[0].hide_select
        for obj in meshes:
            obj.hide_select = new_state
        return {'FINISHED'}


class AUTORIG_OT_SelectAllMarkers(bpy.types.Operator):
    """Select all body and face markers"""
    bl_idname  = "autorig.select_markers"
    bl_label   = "Select All Markers"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        bpy.ops.object.select_all(action='DESELECT')
        count = 0
        for cname in ("RigifyMarkers", "RigifyFaceMarkers"):
            c = bpy.data.collections.get(cname)
            if c:
                for obj in c.objects:
                    obj.select_set(True)
                    count += 1
        self.report({'INFO'}, f"Selected {count} markers")
        return {'FINISHED'}


class AUTORIG_OT_ApplyMarkerStyle(bpy.types.Operator):
    bl_idname = "autorig.apply_marker_style"
    bl_label = "Apply Marker Style"
    bl_description = "Apply Show-in-Front and display settings to all markers"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        for col_name in ("RigifyMarkers", "RigifyFaceMarkers"):
            col = bpy.data.collections.get(col_name)
            if col:
                for obj in col.objects:
                    _apply_marker_style(obj)
        return {'FINISHED'}


def _mirror_center_x():
    """Return the X coordinate of the character's symmetry plane.

    Checks (in order):
    1. Average X of placed centre-line (SINGLE) markers — pelvis/spine live on the
       character's exact midline regardless of where the character is in world space.
    2. Average of midpoints between any placed bilateral (L/R) pairs.
    3. Bounding-box centre X of the active mesh object.
    4. 0.0 fallback.
    """
    xs = []

    # 1. Centre-line markers
    for name, *_ in SINGLE_MARKERS:
        obj = bpy.data.objects.get(f"MARKER_{name}")
        if obj:
            xs.append(obj.location.x)
    if xs:
        return sum(xs) / len(xs)

    # 2. Midpoints of bilateral pairs
    for base, *_ in list(BILATERAL_MARKERS) + list(FACE_BILATERAL):
        l_obj = bpy.data.objects.get(f"MARKER_{base}_L")
        r_obj = bpy.data.objects.get(f"MARKER_{base}_R")
        if l_obj and r_obj:
            xs.append((l_obj.location.x + r_obj.location.x) / 2.0)
    if xs:
        return sum(xs) / len(xs)

    # 3. Active mesh bounding-box centre X
    ctx_obj = bpy.context.active_object
    if ctx_obj and ctx_obj.type == 'MESH':
        corners = [ctx_obj.matrix_world @ Vector(v) for v in ctx_obj.bound_box]
        xs_corners = [v.x for v in corners]
        return (min(xs_corners) + max(xs_corners)) / 2.0

    return 0.0


# ── Live marker symmetry (ARP-style: move one side, the other follows) ──────
_sym_busy = False
_sym_last = {"name": None, "loc": None}


@persistent
def _live_symmetry_handler(scene, depsgraph):
    """depsgraph_update_post handler: when the ACTIVE marker is a bilateral (_L/_R)
    marker and the user moved it, mirror it onto its counterpart across the symmetry
    plane. Re-entrancy guarded; only fires on real movement so selecting a marker does
    not snap the pair together."""
    global _sym_busy
    if _sym_busy or not getattr(scene, "autorig_live_symmetry", False):
        return
    try:
        vl  = bpy.context.view_layer
        obj = vl.objects.active if vl else None
    except Exception:
        return
    if obj is None or not obj.name.startswith("MARKER_"):
        _sym_last["name"] = None
        return
    raw = obj.name[len("MARKER_"):]
    if not (raw.endswith("_L") or raw.endswith("_R")):
        _sym_last["name"] = None
        return

    cur = obj.location.copy()
    # First time this marker becomes active: remember it, don't mirror yet.
    if _sym_last["name"] != obj.name:
        _sym_last["name"] = obj.name
        _sym_last["loc"]  = cur
        return
    if _sym_last["loc"] is not None and (cur - _sym_last["loc"]).length < 1e-7:
        return   # active but not moved

    other = "MARKER_" + raw[:-1] + ("R" if raw.endswith("_L") else "L")
    co = bpy.data.objects.get(other)
    if co is not None:
        cx     = _mirror_center_x()
        target = Vector((2.0 * cx - cur.x, cur.y, cur.z))
        if (co.location - target).length > 1e-6:
            _sym_busy = True
            try:
                co.location = target
            finally:
                _sym_busy = False
    _sym_last["loc"] = cur


def _remove_live_symmetry_handlers():
    """Remove EVERY live-symmetry handler by name. After a script reload the module
    is re-imported, so `_live_symmetry_handler` is a new function object while stale
    copies from previous reloads remain in depsgraph_update_post (identity no longer
    matches). Multiple copies with independent re-entrancy guards fight each other
    and the mirror stops working — so match by __name__ to clear them all."""
    hs = bpy.app.handlers.depsgraph_update_post
    for h in list(hs):
        if getattr(h, "__name__", "") == "_live_symmetry_handler":
            try:
                hs.remove(h)
            except Exception:
                pass


def register_live_symmetry():
    global _sym_busy
    _sym_busy = False                       # clear any stale re-entrancy lock
    _remove_live_symmetry_handlers()        # drop copies left by earlier reloads
    bpy.app.handlers.depsgraph_update_post.append(_live_symmetry_handler)


def unregister_live_symmetry():
    _remove_live_symmetry_handlers()


class AUTORIG_OT_MirrorMarkers(bpy.types.Operator):
    """Mirror Left markers to Right or Right to Left across the character's symmetry plane"""
    bl_idname  = "autorig.mirror_markers"
    bl_label   = "Mirror Markers"
    bl_options = {'REGISTER', 'UNDO'}
    source_side: bpy.props.EnumProperty(
        name="Source Side",
        items=[("L", "Left → Right", ""), ("R", "Right → Left", "")],
        default="L")

    def execute(self, context):
        dst = "R" if self.source_side == "L" else "L"
        cx  = _mirror_center_x()
        count = 0
        for base, *_ in (list(BILATERAL_MARKERS) + list(FACE_BILATERAL)):
            src_obj = bpy.data.objects.get(f"MARKER_{base}_{self.source_side}")
            dst_obj = bpy.data.objects.get(f"MARKER_{base}_{dst}")
            if src_obj and dst_obj:
                p   = src_obj.location.copy()
                p.x = 2.0 * cx - p.x   # reflect across cx instead of hard-coded 0
                dst_obj.location = p
                count += 1
        self.report({'INFO'}, f"Mirrored {count} marker pairs (centre X = {cx:.4f})")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Marker placement operators
# ---------------------------------------------------------------------------

class AUTORIG_OT_PlaceMarkers(bpy.types.Operator):
    """Place all body + finger markers (L and R) on the selected mesh.
    Disables mesh selection and enables X-Ray."""
    bl_idname  = "autorig.place_markers"
    bl_label   = "Place Body Markers"
    bl_options = {'REGISTER', 'UNDO'}
    replace_existing: bpy.props.BoolProperty(name="Replace Existing", default=False)

    def execute(self, context):
        col = get_or_create_collection("RigifyMarkers")
        created = skipped = 0
        for name, dtype, pos in ALL_MARKERS:
            obj_name = f"MARKER_{name}"
            if obj_name in bpy.data.objects:
                if self.replace_existing:
                    bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)
                else:
                    skipped += 1
                    continue
            base = get_base(name)
            if base in FINGER_BASE_NAMES:
                size = FINGER_SIZE
            elif base in ARM_MARKER_BASES:
                size = ARM_SIZE
            else:
                size = BODY_SIZE
            make_empty(col, obj_name, dtype, pos, size, context)
            created += 1
        col.hide_viewport = False
        set_mesh_select(context, False)
        set_xray(context, True)
        self.report({'INFO'}, f"Created {created} markers | X-Ray ON | Mesh select OFF"
                    + (f" | {skipped} skipped" if skipped else ""))
        return {'FINISHED'}

    def invoke(self, context, event):
        if any(f"MARKER_{n}" in bpy.data.objects for n, *_ in ALL_MARKERS) and not self.replace_existing:
            return context.window_manager.invoke_props_dialog(self)
        return self.execute(context)

    def draw(self, context):
        self.layout.prop(self, "replace_existing")
        self.layout.label(text="Some markers already exist.")


class AUTORIG_OT_PlaceFaceMarkers(bpy.types.Operator):
    """Place comprehensive face markers matching every Rigify face bone chain."""
    bl_idname  = "autorig.place_face_markers"
    bl_label   = "Add Face Markers"
    bl_options = {'REGISTER', 'UNDO'}
    replace_existing: bpy.props.BoolProperty(name="Replace Existing", default=False)

    def execute(self, context):
        col = get_or_create_collection("RigifyFaceMarkers")
        created = skipped = 0
        for name, dtype, pos in ALL_FACE_MARKERS:
            obj_name = f"MARKER_{name}"
            if obj_name in bpy.data.objects:
                if self.replace_existing:
                    bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)
                else:
                    skipped += 1
                    continue
            e = make_empty(col, obj_name, dtype, pos, FACE_SIZE, context)
            base = '_'.join(obj_name.rsplit('_', 1)[:-1]) if obj_name.endswith(('_L', '_R')) else obj_name
            e.scale = (0.2, 0.2, 0.2) if base in self._EYELID_BASES else (0.4, 0.4, 0.4)
            created += 1
        col.hide_viewport = False
        self.report({'INFO'}, f"Created {created} face markers" + (f" | {skipped} skipped" if skipped else ""))
        return {'FINISHED'}

    def invoke(self, context, event):
        if any(f"MARKER_{n}" in bpy.data.objects for n, *_ in ALL_FACE_MARKERS) and not self.replace_existing:
            return context.window_manager.invoke_props_dialog(self)
        return self.execute(context)

    def draw(self, context):
        self.layout.prop(self, "replace_existing")
        self.layout.label(text="Some face markers already exist.")


class AUTORIG_OT_RemoveFaceMarkers(bpy.types.Operator):
    """Delete all face markers and remove the RigifyFaceMarkers collection"""
    bl_idname  = "autorig.remove_face_markers"
    bl_label   = "Remove Face Markers"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        removed = 0
        col = bpy.data.collections.get("RigifyFaceMarkers")
        if col:
            for obj in list(col.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
                removed += 1
            bpy.data.collections.remove(col)
        for name, *_ in ALL_FACE_MARKERS:
            obj = bpy.data.objects.get(f"MARKER_{name}")
            if obj:
                bpy.data.objects.remove(obj, do_unlink=True)
                removed += 1
        self.report({'INFO'}, f"Removed {removed} face markers")
        return {'FINISHED'}

    def invoke(self, context, event):
        try:
            return context.window_manager.invoke_confirm(
                self, event, message="Remove all face markers?")
        except TypeError:
            return context.window_manager.invoke_confirm(self, event)


# ---------------------------------------------------------------------------
# Delete ALL markers (body + face)
# ---------------------------------------------------------------------------

class AUTORIG_OT_DeleteAllMarkers(bpy.types.Operator):
    """Delete every body and face marker and remove their collections"""
    bl_idname  = "autorig.delete_all_markers"
    bl_label   = "Delete All Markers"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        removed = 0
        for col_name in ("RigifyMarkers", "RigifyFaceMarkers"):
            col = bpy.data.collections.get(col_name)
            if col:
                for obj in list(col.objects):
                    bpy.data.objects.remove(obj, do_unlink=True)
                    removed += 1
                bpy.data.collections.remove(col)
        for obj in list(bpy.data.objects):
            if obj.get("autorig_marker"):
                bpy.data.objects.remove(obj, do_unlink=True)
                removed += 1
        self.report({'INFO'}, f"Deleted {removed} markers")
        return {'FINISHED'}

    def invoke(self, context, event):
        try:
            return context.window_manager.invoke_confirm(
                self, event, message="Delete ALL markers (body + face)?")
        except TypeError:
            return context.window_manager.invoke_confirm(self, event)


# ---------------------------------------------------------------------------
# Auto-detect body
# ---------------------------------------------------------------------------

class AUTORIG_OT_AutoDetectBody(bpy.types.Operator):
    """Automatically detect body marker positions from the stored body mesh.
    Pick the body mesh in the panel once; markers must already exist (Place Body Markers)."""
    bl_idname  = "autorig.auto_detect_body"
    bl_label   = "Auto Detect Body and Legs"
    bl_options = {'REGISTER', 'UNDO'}

    @staticmethod
    def _world_verts(mesh_obj):
        mw = mesh_obj.matrix_world
        return [mw @ v.co for v in mesh_obj.data.vertices]

    @staticmethod
    def _com(verts):
        if not verts:
            return None
        s = Vector((0, 0, 0))
        for v in verts:
            s += v
        return s / len(verts)

    @staticmethod
    def _top_pct(verts, pct):
        if not verts:
            return []
        s = sorted(verts, key=lambda v: v.z, reverse=True)
        return s[:max(1, int(len(s) * pct))]

    @staticmethod
    def _segment_by_height(verts, min_z, max_z):
        h = (max_z - min_z) or 1.0
        regions = {"head": [], "neck": [], "chest": [], "pelvis": [], "legs": []}
        for v in verts:
            t = (v.z - min_z) / h
            if   t > 0.85: regions["head"].append(v)
            elif t > 0.80: regions["neck"].append(v)
            elif t > 0.60: regions["chest"].append(v)
            elif t > 0.45: regions["pelvis"].append(v)
            else:          regions["legs"].append(v)
        return regions

    @staticmethod
    def _detect_forward_axis(verts):
        ys = [v.y for v in verts]
        xs = [v.x for v in verts]
        return "y" if (max(ys) - min(ys)) > (max(xs) - min(xs)) else "x"

    @staticmethod
    def _find_knee(verts, thigh_z, ankle_z):
        best = None
        min_w = None
        steps = 15
        dz = (thigh_z - ankle_z) / max(steps, 1)
        for i in range(steps + 1):
            z = thigh_z - i * dz
            band = [v for v in verts if abs(v.z - z) < 0.03]
            if len(band) < 5:
                continue
            xs = [v.x for v in band]
            w = max(xs) - min(xs)
            if min_w is None or w < min_w:
                min_w = w
                best = AUTORIG_OT_AutoDetectBody._com(band)
        return best

    @staticmethod
    def _scan_width_profile(verts, min_z, max_z, steps=200):
        """Width of the mesh X cross-section at each height step."""
        h    = (max_z - min_z) or 1.0
        step = h / steps
        out  = []
        for i in range(steps):
            z    = min_z + (i + 0.5) * step
            band = [v for v in verts if abs(v.z - z) < step * 0.75]
            if len(band) < 4:
                out.append(None)
            else:
                xs = [v.x for v in band]
                out.append({'z': z, 'width': max(xs) - min(xs)})
        return out

    @staticmethod
    def _profile_min_width_z(profile, lo_frac, hi_frac):
        """Z of the narrowest cross-section in the [lo_frac, hi_frac] height range."""
        n  = len(profile)
        lo = int(lo_frac * n)
        hi = max(lo + 1, int(hi_frac * n))
        best_z, best_w = None, float('inf')
        for p in profile[lo:hi]:
            if p and p['width'] < best_w:
                best_w, best_z = p['width'], p['z']
        return best_z

    @staticmethod
    def _profile_max_width_z(profile, lo_frac, hi_frac):
        """Z of the widest cross-section in the [lo_frac, hi_frac] height range."""
        n  = len(profile)
        lo = int(lo_frac * n)
        hi = max(lo + 1, int(hi_frac * n))
        best_z, best_w = None, 0.0
        for p in profile[lo:hi]:
            if p and p['width'] > best_w:
                best_w, best_z = p['width'], p['z']
        return best_z

    @staticmethod
    def _find_crotch_z(verts, min_z, height, cx):
        """Z where separate left/right leg columns merge into the pelvis.

        Scans upward from ~4% height.  While a large X-gap near the centre
        axis exists the legs are still separate.  When that gap closes the
        scan has reached the crotch — the anatomically correct hip-joint Z.
        """
        for i in range(8, 130):
            z    = min_z + height * i * 0.005
            band = [v for v in verts if abs(v.z - z) < height * 0.012]
            if len(band) < 8:
                continue
            xs      = sorted(v.x for v in band)
            total_w = xs[-1] - xs[0]
            if total_w < 1e-4:
                continue
            gaps = [(xs[j + 1] - xs[j], (xs[j] + xs[j + 1]) * 0.5)
                    for j in range(len(xs) - 1)]
            big_gap, gap_cx = max(gaps, key=lambda g: g[0])
            if big_gap / total_w > 0.20 and abs(gap_cx - cx) < total_w * 0.35:
                continue       # two legs still visible
            return z           # gap closed → crotch height
        return min_z + height * 0.50

    @staticmethod
    def _set_marker(name, pos, cx=None, cy=None, centre_xy=False):
        obj = bpy.data.objects.get(f"MARKER_{name}")
        if obj and pos:
            loc = pos.copy()
            if centre_xy:
                loc.x = cx
                loc.y = cy
            obj.location = loc
            return True
        return False

    @staticmethod
    def _detect_arms(verts, profile, neck_z, chest_z, pelvis_z,
                     cx, height, min_z, _bvh_y, _push_inside):
        """
        Arm detection using the body profile as the anatomical reference.

        Shoulder  = side endpoint of the body at neck_z (profile half-width).
        Hand      = farthest vertex from shoulder in 3-D (arm tip).
        Elbow     = cluster at 50 % of shoulder→hand.
        ARM       = cluster at 25 % of shoulder→hand.

        Works for any arm angle (horizontal, diagonal, hanging) because the
        shoulder joint is always at approximately chest_width * 1.15 from
        center and the arm tip is always the 3-D-farthest vertex from it.
        """
        def _com(vs):
            if not vs:
                return None
            s = Vector((0.0, 0.0, 0.0))
            for v in vs:
                s += v
            return s / len(vs)

        nz = neck_z if neck_z is not None else min_z + height * 0.82
        results = {}

        # Arm-FREE chest half-width. The profile width can't be used here: a
        # horizontal arm inflates the chest band to the ARM SPAN, which pushes the
        # shoulder estimate out to the hand (the chain then collapses onto the palm).
        # Two pose-orthogonal estimators, take the MIN per side (each one over-
        # estimates in exactly one pose and is correct in the other -> the smaller
        # is the true torso edge):
        #   GAP  : at the chest band the torso is the central contiguous cluster and
        #          a DIAGONAL arm (A-pose) is a lateral lobe BEYOND an air gap. Fails
        #          in T-pose (arm is contiguous with the torso at shoulder height).
        #   VERT : a torso X-column reaches DOWN the body; a horizontal arm (T-pose)
        #          exists only at shoulder height, so its column doesn't reach down.
        #          Fails in A-pose (the diagonal arm reaches down too).
        ref_z = chest_z if chest_z is not None else nz - height * 0.12

        def _torso_edge_gap(z_c, sgn):
            ds = sorted(abs(v.x - cx) for v in verts
                        if abs(v.z - z_c) < height * 0.06 and (v.x - cx) * sgn > 0)
            if len(ds) < 5:
                return None
            edge = ds[0]
            for d in ds[1:]:
                if d - edge > height * 0.06:   # gap -> lateral arm lobe begins
                    break
                edge = d
            return edge

        def _torso_edge_vert(sgn):
            # A column counts as torso only if it reaches below z_ref (well under the
            # arm). Walk X outward; stop where columns stop reaching down = arm only.
            z_ref = nz - height * 0.20
            step  = height * 0.02
            colmin = {}
            for v in verts:
                d = (v.x - cx) * sgn
                if d <= 0 or v.z > nz + height * 0.02:
                    continue
                k = int(d / step)
                if v.z < colmin.get(k, 1e9):
                    colmin[k] = v.z
            if not colmin:
                return None
            edge = 0.0
            for k in range(max(colmin) + 1):
                if k not in colmin:
                    if (k + 1) in colmin:   # tolerate one empty bin
                        continue
                    break
                if colmin[k] > z_ref:       # column doesn't reach down -> arm only
                    break
                edge = (k + 1) * step
            return edge if edge > 0 else None

        _hw = []
        for _sgn in (1, -1):
            _es = [e for e in (_torso_edge_gap(ref_z, _sgn),
                               _torso_edge_vert(_sgn)) if e]
            if _es:
                _hw.append(min(_es))
        chest_hw = (sum(_hw) / len(_hw)) if _hw else height * 0.18
        t_shoulder_hw = chest_hw * 1.15  # estimated shoulder joint X (any pose)

        for side in ('L', 'R'):
            sign = 1 if side == 'L' else -1

            arm_lo = min_z + height * 0.10
            arm_hi = nz + height * 0.05
            side_v = [v for v in verts
                      if arm_lo < v.z < arm_hi and (v.x - cx) * sign > 0]
            if not side_v:
                continue

            # ── SHOULDER ──────────────────────────────────────────────────────
            neck_band = [v for v in verts
                         if abs(v.z - nz) < height * 0.04
                         and (v.x - cx) * sign > 0]
            if not neck_band:
                continue

            # Shoulder sits at ~chest_width * 1.15 from center for any arm angle.
            # Targeting this X locates the joint whether the arm extends
            # horizontally, diagonally, or hangs straight down.
            target_x = cx + sign * t_shoulder_hw
            sh_vert  = min(neck_band, key=lambda v: abs(v.x - target_x))
            sh_dist  = (sh_vert.x - cx) * sign

            # ── ARM VERTEX SET ────────────────────────────────────────────────
            # Vertices lateral to the shoulder, but ABOVE the legs (z > 35% height) so
            # the arm chain can never grab a thigh/foot (the old full-height set let the
            # "farthest vertex" tip land on a leg). Captures the arm at any angle.
            leg_cut = min_z + height * 0.35
            arm_v = [v for v in side_v
                     if v.z > leg_cut and (v.x - cx) * sign > sh_dist * 0.90]
            if not arm_v:
                arm_v = [v for v in side_v if v.z > leg_cut] or side_v

            # ── SHOULDER + ARM: both at centroid of shoulder cross-section ────
            # Centroid of neck_band vertices within ± 8 % of height around
            # sh_dist = inside the cylinder at the joint, not on the skin.
            entry_v = [v for v in neck_band
                       if abs((v.x - cx) * sign - sh_dist) < height * 0.08]
            if not entry_v:
                entry_v = [sh_vert]
            joint_pos = _com(entry_v)
            if joint_pos is None:
                continue
            # SHOULDER: depth-centre via the BVH front/back midpoint, then push
            # inside. The raw entry_v centroid sat too close to the skin here.
            joint_pos.y = _bvh_y(joint_pos.x, joint_pos.z)
            sh_pos      = _push_inside(joint_pos)

            # Rough arm axis from the MOST-LATERAL vertex (both target poses reach
            # out along X). Independent of the shoulder, so it can't grab a
            # diagonally-distant hip/thigh vertex. Refined below.
            tip_v    = max(arm_v, key=lambda v: abs(v.x - cx))
            arm_axis = tip_v - sh_pos
            arm_len  = arm_axis.length
            hand_pos = None
            elbow_pos = None
            tip_pos   = None

            # ARM marker = shoulder centroid nudged toward the upper arm. The Z drop
            # only makes sense for a DOWN-angled (A-pose) arm; for a horizontal
            # (T-pose) arm it would push the marker off the arm, so skip it there.
            _t_pose = arm_len > 1e-5 and abs(arm_axis.z) < 0.30 * arm_len
            arm_mid_pos = joint_pos.copy()
            arm_mid_pos.x += sign * 0.023
            if not _t_pose:
                arm_mid_pos.z -= 0.035
            # ARM: depth-centre + push inside too (matches the shoulder).
            arm_mid_pos.y = _bvh_y(arm_mid_pos.x, arm_mid_pos.z)
            arm_mid_pos   = _push_inside(arm_mid_pos)

            if arm_len > 1e-5:
                arm_dir = arm_axis / arm_len

                # Strip off-axis BODY verts: in A-pose the torso/hip hug the arm and
                # the gate above lets them in. They project onto the axis with a huge
                # perpendicular radius (a 7x "bulge" the palm detector mistakes for
                # the hand, putting the wrist mid-forearm). Keep only a thin tube
                # around the shoulder->tip axis -- the arm is thin, the torso is not.
                _axis_tol = arm_len * 0.18
                _tube = [v for v in arm_v
                         if (v - sh_pos
                             - arm_dir * (v - sh_pos).dot(arm_dir)).length < _axis_tol]
                if len(_tube) >= 10:
                    arm_v = _tube

                # ── TIP: longest fingertip = vertex farthest ALONG the arm axis ──
                # (follows a fingertip that points slightly off pure-lateral). Use a
                # TIGHT cluster so it stays AT the tip, and keep the cluster centroid
                # (the finger's cross-section centre) -- no surface snap, which was
                # pulling it onto the skin / off the longest finger.
                tip_v    = max(arm_v, key=lambda v: (v - sh_pos).dot(arm_dir))
                tip_zone = sorted(arm_v, key=lambda v: (v - tip_v).length)[:5]
                tip_pos  = _com(tip_zone) or tip_v.copy()

                # Radius profile along shoulder->tip, in bins (robust to low-poly:
                # we read whole bins, never single thin slabs that can be empty).
                # For each bin store the MAX perpendicular radius and the vertices.
                N = 24
                bin_r  = [0.0] * N          # max perpendicular radius in the bin
                bin_v  = [[] for _ in range(N)]
                for v in arm_v:
                    proj = (v - sh_pos).dot(arm_dir)
                    tn   = proj / arm_len
                    if tn < 0.0 or tn > 1.0:
                        continue
                    b = min(N - 1, int(tn * N))
                    r = (v - sh_pos - arm_dir * proj).length
                    bin_v[b].append(v)
                    if r > bin_r[b]:
                        bin_r[b] = r

                def _bin_t(b):       # bin centre -> t along the axis (0..arm_len)
                    return arm_len * (b + 0.5) / N

                # PALM = the radius bulge near the tip: bin of max radius in the
                # outer arm (tn 0.60..0.96). Fingers are past it (thin again).
                palm_b = None; palm_r = -1.0
                for b in range(N):
                    tn = (b + 0.5) / N
                    if 0.60 <= tn <= 0.96 and bin_v[b] and bin_r[b] > palm_r:
                        palm_r, palm_b = bin_r[b], b
                if palm_b is None:
                    palm_b = int(0.85 * N)

                # WRIST = narrowest cross-section BEFORE the palm (tn 0.45..palm).
                # The forearm tapers to the wrist neck, then the palm widens. This
                # is exactly "the narrowest point before the palm".
                wrist_b = None; wrist_r = float('inf')
                for b in range(N):
                    tn = (b + 0.5) / N
                    if 0.45 <= tn and b < palm_b and bin_v[b] and bin_r[b] < wrist_r:
                        wrist_r, wrist_b = bin_r[b], b
                if wrist_b is None:
                    wrist_b = max(0, palm_b - 2)

                # HAND marker = centroid of the wrist bin (+/-1 bin for vert count).
                wrist_zone = bin_v[wrist_b][:]
                for nb in (wrist_b - 1, wrist_b + 1):
                    if 0 <= nb < N:
                        wrist_zone += bin_v[nb]
                if wrist_zone:
                    hand_pos = _com(wrist_zone)
                wrist_t = _bin_t(wrist_b)

                # ELBOW = midpoint between the ARM marker (~25 %) and the WRIST,
                # centred in the limb (centroid of the cross-section there).
                arm_t   = (arm_mid_pos - sh_pos).dot(arm_dir)
                arm_t   = min(max(arm_t, 0.0), arm_len)
                elbow_t = (arm_t + wrist_t) * 0.5
                eb_band = arm_len * 0.06
                elbow_zone = [v for v in arm_v
                              if abs((v - sh_pos).dot(arm_dir) - elbow_t) <= eb_band]
                if not elbow_zone:
                    elbow_ctr  = sh_pos + arm_dir * elbow_t
                    elbow_zone = sorted(arm_v, key=lambda v: (v - elbow_ctr).length
                                        )[:max(1, len(arm_v) // 6)]
                elbow_pos = _com(elbow_zone)

            # Fallbacks if the axis was degenerate.
            if hand_pos is None:
                max_dist   = max((v - sh_pos).length for v in arm_v)
                wrist_zone = [v for v in arm_v
                              if 0.68 * max_dist <= (v - sh_pos).length <= 0.90 * max_dist]
                if not wrist_zone:
                    wrist_zone = sorted(arm_v, key=lambda v: (v - sh_pos).length,
                                        reverse=True)[:max(1, len(arm_v) // 5)]
                hand_pos = _com(wrist_zone)
            if hand_pos is None:
                continue
            # Keep the wrist-ring centroid: it is already the wrist CENTRE (inside
            # the volume). Snapping Y to the surface would put it on the skin.
            hand_pos = hand_pos.copy()

            # TIP fallback if the axis was degenerate.
            if tip_pos is None:
                tip_v   = max(arm_v, key=lambda v: abs(v.x - cx))
                tip_pos = _com(sorted(arm_v, key=lambda v: (v - tip_v).length)[:5]) \
                          or tip_v.copy()
            tip_pos = tip_pos.copy()

            if elbow_pos is None:
                mid_eh    = arm_mid_pos.lerp(hand_pos, 0.5)
                elbow_pos = _com(sorted(arm_v, key=lambda v: (v - mid_eh).length
                                        )[:max(1, len(arm_v) // 6)])
            if elbow_pos:
                # elbow_pos is the cross-section centroid = elbow CENTRE, inside the
                # volume. Keep it (no surface snap).
                elbow_pos = elbow_pos.copy()

            results[side] = (sh_pos, arm_mid_pos, elbow_pos, hand_pos, tip_pos)

        return results

    def execute(self, context):
        # 1. Persistent picker (set once in the panel — preferred path)
        mesh_obj = None
        face_props = getattr(context.scene, 'autorig_face_objs', None)
        if face_props and face_props.detect_body_obj and face_props.detect_body_obj.type == 'MESH' \
                and face_props.detect_body_obj.name in context.scene.objects:
            mesh_obj = face_props.detect_body_obj
        # 2. Fallback: selected / active object
        if not mesh_obj:
            for o in context.selected_objects:
                if o.type == 'MESH':
                    mesh_obj = o
                    break
        if not mesh_obj and context.active_object and context.active_object.type == 'MESH':
            mesh_obj = context.active_object
        # 3. Last resort: tallest mesh in the scene
        if not mesh_obj:
            meshes = [o for o in context.scene.objects if o.type == 'MESH']
            if meshes:
                mesh_obj = max(meshes, key=lambda o: o.dimensions.z)
        if not mesh_obj:
            self.report({'ERROR'}, "No mesh found. Set a Body Mesh in the Body Markers panel.")
            return {'CANCELLED'}

        col = get_or_create_collection("RigifyMarkers")
        for name, dtype, pos in ALL_MARKERS:
            base = get_base(name)
            if base in FINGER_BASE_NAMES or base in ARM_MARKER_BASES:
                continue
            obj_name = f"MARKER_{name}"
            if obj_name not in bpy.data.objects:
                make_empty(col, obj_name, dtype, pos, BODY_SIZE, context)
        col.hide_viewport = False

        verts = self._world_verts(mesh_obj)
        if not verts:
            self.report({'ERROR'}, "Mesh has no vertices.")
            return {'CANCELLED'}

        min_z = min(v.z for v in verts)
        max_z = max(v.z for v in verts)
        min_x = min(v.x for v in verts)
        max_x = max(v.x for v in verts)
        min_y = min(v.y for v in verts)
        max_y = max(v.y for v in verts)
        cx    = (min_x + max_x) * 0.5
        cy    = (min_y + max_y) * 0.5
        height = max_z - min_z

        regions = self._segment_by_height(verts, min_z, max_z)
        moved = 0

        import bmesh as _bmesh
        from mathutils.bvhtree import BVHTree
        bm = _bmesh.new()
        bm.from_mesh(mesh_obj.data)
        bm.transform(mesh_obj.matrix_world)
        bvh = BVHTree.FromBMesh(bm)
        bm.free()
        depth  = max(max_y - min_y, max_x - min_x, max_z - min_z)
        margin = depth * 5.0

        def _bvh_y(x, z):
            ray_o = Vector((x, min_y - margin, z))
            ray_d = Vector((0, 1, 0))
            hit, _, _, _ = bvh.ray_cast(ray_o, ray_d)
            if not hit:
                return cy
            front_y = hit.y
            last = hit
            while True:
                nxt, _, _, _ = bvh.ray_cast(last + Vector((0, 0.0001, 0)), ray_d)
                if not nxt:
                    break
                last = nxt
            return front_y + (last.y - front_y) * 0.5

        def _bvh_x(z, y_probe, side):
            sign    = 1 if side == 'L' else -1
            start_x = (min_x - margin) if side == 'L' else (max_x + margin)
            ray_o   = Vector((start_x, y_probe, z))
            ray_d   = Vector((sign, 0, 0))
            hit, _, _, _ = bvh.ray_cast(ray_o, ray_d)
            if not hit:
                return None
            outer_x = hit.x
            last = hit
            while True:
                nxt, _, _, _ = bvh.ray_cast(last + Vector((sign * 0.0001, 0, 0)), ray_d)
                if not nxt:
                    break
                last = nxt
            return outer_x, last.x

        def com_z(region_verts, pct):
            pos = self._com(self._top_pct(region_verts, pct))
            return pos.z if pos else None

        def place_spine(name, z):
            nonlocal moved
            if z is None:
                return None
            # Y = geometric depth midpoint of the central torso slab at this height
            # (min+max)/2. This is robustly INSIDE the volume -- the BVH centre ray
            # can miss or hit multiple front/back pairs (e.g. at the pelvis) and
            # land the marker outside the mesh. Fall back to the ray only if the
            # slab is too sparse.
            band = [v for v in verts if abs(v.z - z) < height * 0.04
                    and abs(v.x - cx) < height * 0.13]
            if len(band) >= 5:
                ys = [v.y for v in band]
                y  = (min(ys) + max(ys)) * 0.5
            else:
                y = _bvh_y(cx, z)
            pos = Vector((cx, y, z))
            if self._set_marker(name, pos):
                moved += 1
            return pos

        # ── Geometric width profile — drives all spine/neck/head placement ──
        profile = self._scan_width_profile(verts, min_z, max_z, steps=200)

        # Find the narrowest neck cross-section (mid-cervical reference, not placed directly)
        min_neck_z = self._profile_min_width_z(profile, 0.72, 0.91)

        # The 72-91% window assumes ADULT proportions — on a baby chibi
        # (head ~half the body) it sits entirely inside the HEAD, and neck/
        # head/chest/spine/shoulders/arms all cascade from this one value.
        # Validate: a real neck minimum has a SKULL WIDENING above it (>=1.3x).
        # A mid-head minimum doesn't -> re-search with chibi-capable rules.
        def _pw(z):
            _b, _bw = float('inf'), None
            for p in profile:
                if p and abs(p['z'] - z) < _b:
                    _b, _bw = abs(p['z'] - z), p['width']
            return _bw
        if min_neck_z is not None:
            _w0 = _pw(min_neck_z)
            _above = [p['width'] for p in profile
                      if p and p['z'] > min_neck_z]
            if not (_w0 and _above and max(_above) > _w0 * 1.30):
                min_neck_z = None            # no skull above -> not a neck
        if min_neck_z is None:
            # Wide pinch search (40-93%): qualifying = head above (>=1.3x) and
            # ANY widening below (>=1.05x — chibi shoulders can be narrower
            # than the head). Highest qualifying pinch = the neck (excludes
            # the waist, which also pinches but sits lower).
            _n = len(profile)
            for _k in range(int(0.93 * _n) - 1, int(0.40 * _n), -1):
                p = profile[_k]
                if p is None:
                    continue
                _wa = [q['width'] for q in profile[_k + 1:] if q]
                _wb = [q['width'] for q in profile[:_k] if q]
                if not _wa or not _wb:
                    continue
                if (max(_wa) > p['width'] * 1.30
                        and max(_wb) > p['width'] * 1.05
                        and p['width'] == min(
                            q['width'] for q in profile[max(0, _k - 6):_k + 7] if q)):
                    min_neck_z = p['z']
                    print(f"[auto-body] adult neck window rejected -- pinch "
                          f"search: neck at "
                          f"{(min_neck_z - min_z) / height * 100:.0f}% of height")
                    break
        if min_neck_z is None:
            # HEAD-BALL fallback: no pinch at all (head blends into the body).
            # Widest slab in the upper half sitting ~one radius below the
            # crown = the head's equator; neck = one radius below it.
            _cands = [p for p in profile
                      if p and (p['z'] - min_z) / height >= 0.45]
            if _cands:
                _eq = max(_cands, key=lambda p: p['width'])
                if abs((max_z - _eq['z']) - _eq['width'] * 0.5) < _eq['width'] * 0.35:
                    min_neck_z = max(_eq['z'] - _eq['width'] * 0.55,
                                     min_z + height * 0.25)
                    print(f"[auto-body] head-ball fallback: neck at "
                          f"{(min_neck_z - min_z) / height * 100:.0f}% of height")
        if min_neck_z is None:
            neck_verts = regions["neck"]
            if neck_verts:
                neck_zs    = sorted(set(round(v.z, 3) for v in neck_verts))
                min_neck_z = neck_zs[len(neck_zs) // 2]
                best_w     = None
                for z in neck_zs:
                    band = [v for v in neck_verts if abs(v.z - z) < 0.015]
                    if len(band) < 3:
                        continue
                    w = max(v.x for v in band) - min(v.x for v in band)
                    if best_w is None or w < best_w:
                        best_w, min_neck_z = w, z

        # Width at the narrowest neck point — used as the widen-threshold reference
        min_neck_w = None
        if min_neck_z is not None:
            best_d = float('inf')
            for p in profile:
                if p is None:
                    continue
                d = abs(p['z'] - min_neck_z)
                if d < best_d:
                    best_d, min_neck_w = d, p['width']

        # NECK — collarbone / base of neck.
        # Scan downward from the narrowest-neck point; the first step where the
        # cross-section is noticeably wider (>=1.4x) is the chest-to-neck transition.
        neck_z = min_neck_z
        if min_neck_z is not None and min_neck_w is not None:
            threshold   = min_neck_w * 1.4
            gap         = height * 0.03       # skip 3% immediately below the narrowest
            low_limit_z = min_z + height * 0.60
            for p in reversed(profile):
                if p is None or p['z'] >= min_neck_z - gap:
                    continue
                if p['z'] < low_limit_z:
                    break
                if p['width'] > threshold:
                    neck_z = p['z']
                    break
        neck_com = place_spine("NECK", neck_z)

        # HEAD — top of neck / base of skull.
        # Scan upward from the narrowest-neck point; the first step where the
        # cross-section widens (>=1.4x) is the neck-to-skull transition.
        head_z = None
        if min_neck_z is not None and min_neck_w is not None:
            threshold    = min_neck_w * 1.4
            gap          = height * 0.02      # skip 2% immediately above the narrowest
            high_limit_z = min_z + height * 0.97
            for p in profile:
                if p is None or p['z'] <= min_neck_z + gap:
                    continue
                if p['z'] > high_limit_z:
                    break
                if p['width'] > threshold:
                    head_z = p['z']
                    break
        if head_z is None:
            head_verts = self._top_pct(verts, 0.08)
            head_z = self._com(head_verts).z if head_verts else com_z(regions["head"], 0.50)
        place_spine("HEAD", head_z)

        # PELVIS — widest cross-section between 38% and 56% OF NECK HEIGHT
        # (the fixed fractions assumed an adult neck at ~82%; on a chibi with
        # the neck at ~50% they pointed into the head). Scaling by the actual
        # neck keeps adults identical and adapts stylized proportions.
        _nk_scale = 1.0
        if neck_z is not None and height > 1e-6:
            _nk_scale = max(0.4, min(1.2, ((neck_z - min_z) / height) / 0.82))
        pelvis_z   = self._profile_max_width_z(
            profile, 0.38 * _nk_scale, 0.56 * _nk_scale)
        if pelvis_z is None:
            pelvis_z = com_z(regions["pelvis"], 0.50)
        if pelvis_z is not None:
            pelvis_z += height * 0.03
        # Lift the pelvis UP into the pelvic mass. Below the crotch the body
        # centreline (x=cx) is the air gap between the thighs, so a marker placed
        # there floats "under the pelvis", outside the mesh. The crotch (where the
        # leg columns merge) is the anatomical hip-joint Z; the pelvis centre sits
        # above it. Anchor to crotch + ~7% of height, never below that.
        _crotch_z = self._find_crotch_z(verts, min_z, height, cx)
        # Lift above the crotch by a fraction of the TORSO length (crotch->neck),
        # NOT of total height: tall characters have a short torso + long legs, so a
        # %-of-height offset overshoots into the belly. Torso-relative self-scales
        # across proportions. Never drop below the widest-cross-section pick.
        if neck_z is not None:
            pelvis_min = _crotch_z + (neck_z - _crotch_z) * 0.33
        else:
            pelvis_min = _crotch_z + height * 0.10
        if pelvis_z is None or pelvis_z < pelvis_min:
            pelvis_z = pelvis_min
        pelvis_com = place_spine("PELVIS", pelvis_z)

        # CHEST — base of the ribcage. The widest cross-section is UNRELIABLE here:
        # on wide-bellied / wide-chest characters the belly (or a broad chest) is
        # wider than the ribcage base and drags the marker down to the stomach. Use
        # the ANATOMICAL position (~70% from pelvis up to neck) instead — robust to
        # torso width. Falls back to width/region only if the anchors are missing.
        if neck_z is not None and pelvis_z is not None:
            chest_z = pelvis_z + (neck_z - pelvis_z) * 0.70
        else:
            chest_z = self._profile_max_width_z(profile, 0.65, 0.77)
        if chest_z is None:
            chest_z = com_z(regions["chest"], 0.35)

        # SPINE_001 / SPINE_002 — evenly spaced thirds between PELVIS and CHEST.
        if pelvis_z is not None and chest_z is not None:
            span       = chest_z - pelvis_z
            spine001_z = pelvis_z + span * (1 / 3)
            spine002_z = pelvis_z + span * (2 / 3)
        else:
            spine001_z = None
            spine002_z = None
        chest_com = place_spine("CHEST",     chest_z)
        place_spine("SPINE_002", spine002_z)
        place_spine("SPINE_001", spine001_z)

        # ── Arm detection — pose-agnostic (T-pose and A-pose) ─────────────────
        def _push_inside(pos, _md=height * 0.025):
            nearest, _, _, dist = bvh.find_nearest(pos)
            if nearest is None or dist >= _md:
                return pos
            inward = pos - nearest
            L = inward.length
            if L < 1e-7:
                return pos
            return nearest + inward * (_md / L)

        # Ensure arm marker empties exist before detection writes to them —
        # the create loop above skips ARM_MARKER_BASES, so HAND/HAND_TIP/etc.
        # may not exist yet on a fresh run.
        for aname in ("SHOULDER_L", "SHOULDER_R", "ARM_L", "ARM_R",
                      "ELBOW_L",    "ELBOW_R",    "HAND_L", "HAND_R",
                      "HAND_TIP_L", "HAND_TIP_R"):
            oname = f"MARKER_{aname}"
            if oname not in bpy.data.objects:
                make_empty(col, oname, 'SPHERE', Vector((0, 0, 0)), ARM_SIZE, context)

        sh_l_pos = sh_r_pos = None
        arm_results = self._detect_arms(
            verts, profile, neck_z, chest_z, pelvis_z, cx, height, min_z,
            _bvh_y, _push_inside)

        for side in ('L', 'R'):
            if side not in arm_results:
                continue
            sh_pos, arm_mid_pos, elbow_pos, hand_pos, tip_pos = arm_results[side]
            if sh_pos:
                if side == 'L': sh_l_pos = sh_pos
                else:           sh_r_pos = sh_pos
                if self._set_marker(f"SHOULDER_{side}", sh_pos):     moved += 1
            if arm_mid_pos:
                if self._set_marker(f"ARM_{side}",      arm_mid_pos): moved += 1
            if elbow_pos:
                if self._set_marker(f"ELBOW_{side}",    elbow_pos):  moved += 1
            if hand_pos:
                if self._set_marker(f"HAND_{side}",     hand_pos):   moved += 1
            if tip_pos:
                if self._set_marker(f"HAND_TIP_{side}", tip_pos):    moved += 1

        # ── Crotch detection — anatomically correct hip-joint Z ─────────────
        crotch_z = self._find_crotch_z(verts, min_z, height, cx)

        leg_verts = regions["legs"]
        left_leg  = [v for v in leg_verts if v.x < cx]
        right_leg = [v for v in leg_verts if v.x >= cx]
        if not left_leg:  left_leg  = leg_verts
        if not right_leg: right_leg = leg_verts

        def leg_markers(leg_v, side):
            nonlocal moved
            if not leg_v:
                return
            leg_min_z  = min(v.z for v in leg_v)
            leg_max_z  = max(v.z for v in leg_v)
            leg_height = leg_max_z - leg_min_z

            # Ground-contact vertices (lowest 10%) are guaranteed to be foot,
            # not arm — use their X centroid as the leg's lateral reference.
            ground_v = [v for v in leg_v
                        if v.z < leg_min_z + leg_height * 0.10]
            if not ground_v:
                ground_v = sorted(leg_v, key=lambda v: v.z)[
                    :max(1, len(leg_v) // 10)]
            foot_ref = self._com(ground_v)
            foot_x_ref = foot_ref.x if foot_ref else cx

            # Exclude arm/wrist vertices: arms are far from the leg's X centroid.
            # Tolerance = 35 % of body half-width, enough for any realistic leg
            # spread while rejecting all but the most extreme A-pose arms.
            half_width = (max_x - min_x) * 0.5
            leg_x_tol  = half_width * 0.35
            true_leg_v = [v for v in leg_v
                          if abs(v.x - foot_x_ref) < leg_x_tol]
            if len(true_leg_v) < 20:
                true_leg_v = leg_v   # fallback if filter was too aggressive

            # Foot region = lowest ~20 % of the true leg. Body faces -Y, so the toe
            # tip is the front (min Y) and the heel is the back (max Y).
            toes_sort = sorted(true_leg_v, key=lambda v: v.z)
            foot_v    = toes_sort[:max(3, len(toes_sort) // 5)]
            foot_toy  = min(v.y for v in foot_v)
            foot_hey  = max(v.y for v in foot_v)
            foot_flen = max(foot_hey - foot_toy, 1e-5)
            # TOES = ball of foot (~25 % back from the toe tip), centred — the rigify
            # toe bone starts here, not at the toe tip.
            ball_v   = [v for v in foot_v
                        if abs(v.y - (foot_toy + 0.25 * foot_flen)) < 0.15 * foot_flen]
            toes_pos = self._com(ball_v) if len(ball_v) >= 3 else self._com(foot_v)

            # FOOT (ankle): narrowest cross-section between ~3 % and ~28 %
            # of leg height above the floor.  _find_knee finds the minimum
            # X-width band, which is the ankle joint for this Z range.
            ankle_lo  = leg_min_z + leg_height * 0.03
            ankle_hi  = leg_min_z + leg_height * 0.18
            ankle_pos = self._find_knee(true_leg_v, ankle_hi, ankle_lo)
            if not ankle_pos:
                ankle_band = [v for v in true_leg_v
                              if ankle_lo < v.z < ankle_hi]
                ankle_pos  = self._com(ankle_band) if ankle_band else toes_pos

            # HEEL = back END of the foot, on the floor (rigify heel-roll pivot):
            # rearmost foot band, centred laterally, Y at the rear edge, Z on the floor.
            rear_v = [v for v in foot_v if v.y > foot_hey - 0.18 * foot_flen]
            if len(rear_v) >= 3:
                heel_pos = Vector((sum(v.x for v in rear_v) / len(rear_v),
                                   foot_hey - 0.03 * foot_flen,
                                   min(v.z for v in rear_v)))
            else:
                heel_pos = max(ground_v, key=lambda v: v.y).copy() if ground_v else toes_pos

            # THIGH: centroid of arm-filtered leg verts at PELVIS height (the thigh
            # markers share the pelvis marker's height by design).
            # BVH is avoided here — in A-pose the wrists hang at pelvis height so the
            # first lateral BVH hit lands on the hand, not the hip. true_leg_v
            # already excludes arm verts, so its centroid here gives the hip socket.
            thigh_z = pelvis_com.z if pelvis_com else crotch_z
            band = [v for v in true_leg_v
                    if abs(v.z - thigh_z) < height * 0.05]
            if not band:
                band = sorted(true_leg_v,
                              key=lambda v: abs(v.z - thigh_z))[:60]
            thigh_pos = self._com(band)
            if thigh_pos:
                thigh_pos = thigh_pos.copy()
                thigh_pos.z = thigh_z
                # The hip-mass centroid sits laterally (toward the outer hip / greater
                # trochanter surface); pull X ~35% toward the body centre so the thigh
                # seats at the hip SOCKET, inside the volume, not on the side surface.
                thigh_pos.x = thigh_pos.x + (cx - thigh_pos.x) * 0.35
                # Floor: never let the socket end up more medial than the leg's
                # CENTRE. The 35% pull above over-corrects on thin / close-together
                # legs and drags the thigh in toward the pelvis; the hip joint sits
                # around the middle of the leg cross-section, so clamp it there.
                _sgn  = -1 if side == 'L' else 1   # L leg sits at x<cx, R at x>cx
                _pre  = (thigh_pos.x - cx) * _sgn
                _offs = sorted((v.x - cx) * _sgn for v in band)
                _inner, _outer = _offs[0], _offs[-1]
                _min_off = _inner + (_outer - _inner) * 0.50
                if (thigh_pos.x - cx) * _sgn < _min_off:
                    thigh_pos.x = cx + _sgn * _min_off

            # SHIN (knee): place at the exact thigh-to-ankle midpoint, then scan
            # within ±5 % of the leg span for the narrowest cross-section.
            # The tight window stops the scan from drifting to the ankle on
            # thin-legged characters.  Falls back to the plain midpoint when no
            # band with ≥5 vertices is found inside the window.
            knee_pos = None
            if thigh_pos and ankle_pos:
                span   = thigh_pos.z - ankle_pos.z
                mid_z  = ankle_pos.z + span * 0.50
                radius = span * 0.05
                knee_pos = self._find_knee(true_leg_v, mid_z + radius, mid_z - radius)
                if not knee_pos:
                    knee_pos = thigh_pos.lerp(ankle_pos, 0.5)

            if self._set_marker(f"THIGH_{side}", thigh_pos): moved += 1
            if self._set_marker(f"SHIN_{side}",  knee_pos):  moved += 1
            if self._set_marker(f"FOOT_{side}",  ankle_pos): moved += 1
            if self._set_marker(f"TOES_{side}",  toes_pos):  moved += 1
            if heel_pos:
                heel_obj = bpy.data.objects.get(f"MARKER_HEEL_{side}")
                if heel_obj:
                    heel_obj.location = heel_pos
                    moved += 1

        leg_markers(left_leg,  'L')
        leg_markers(right_leg, 'R')

        if chest_com:
            br_x_l =  (max_x - cx) * 0.28
            br_x_r = -(max_x - cx) * 0.28
            front_ch = sorted(regions["chest"], key=lambda v: v.y)[:max(1, len(regions["chest"]) // 6)]
            br_y = self._com(front_ch).y if front_ch else min_y + 0.03
            br_z = chest_com.z - height * 0.03
            if self._set_marker("BREAST_L", Vector((br_x_l, br_y - 0.01, br_z))): moved += 1
            if self._set_marker("BREAST_R", Vector((br_x_r, br_y - 0.01, br_z))): moved += 1

        pairs = [
            ("SHOULDER_L", "SHOULDER_R"), ("ARM_L", "ARM_R"),
            ("ELBOW_L", "ELBOW_R"),       ("HAND_L", "HAND_R"),
            ("THIGH_L", "THIGH_R"),       ("SHIN_L", "SHIN_R"),
            ("FOOT_L", "FOOT_R"),         ("TOES_L", "TOES_R"),
            ("HEEL_L", "HEEL_R"),         ("BREAST_L", "BREAST_R"),
        ]
        for ln, rn in pairs:
            lo = bpy.data.objects.get(f"MARKER_{ln}")
            ro = bpy.data.objects.get(f"MARKER_{rn}")
            if lo and ro:
                ax = (abs(lo.location.x) + abs(ro.location.x)) * 0.5
                ay = (lo.location.y + ro.location.y) * 0.5
                az = (lo.location.z + ro.location.z) * 0.5
                lo.location = Vector(( ax, ay, az))
                ro.location = Vector((-ax, ay, az))

        self.report({'INFO'},
            f"Auto-detected {moved} markers on '{mesh_obj.name}'. "
            "Review and fine-tune before aligning.")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Auto-detect arms (pose-agnostic — T-pose and A-pose)
# ---------------------------------------------------------------------------

class AUTORIG_OT_AutoDetectArms(bpy.types.Operator):
    """Auto-detect arm marker positions from the body mesh.
    Works for both T-pose and A-pose. Run after Auto Detect Body & Legs
    so NECK and PELVIS anchor markers are already placed."""
    bl_idname  = "autorig.auto_detect_arms"
    bl_label   = "Auto Detect Arms"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # ── Mesh resolution ──────────────────────────────────────────────────
        mesh_obj = None
        face_props = getattr(context.scene, 'autorig_face_objs', None)
        if face_props and face_props.detect_body_obj \
                and face_props.detect_body_obj.type == 'MESH' \
                and face_props.detect_body_obj.name in context.scene.objects:
            mesh_obj = face_props.detect_body_obj
        if not mesh_obj:
            for o in context.selected_objects:
                if o.type == 'MESH':
                    mesh_obj = o; break
        if not mesh_obj and context.active_object \
                and context.active_object.type == 'MESH':
            mesh_obj = context.active_object
        if not mesh_obj:
            meshes = [o for o in context.scene.objects if o.type == 'MESH']
            if meshes:
                mesh_obj = max(meshes, key=lambda o: o.dimensions.z)
        if not mesh_obj:
            self.report({'ERROR'}, "No mesh found. Set a Body Mesh in the panel.")
            return {'CANCELLED'}

        # ── Require existing body markers as Z anchors ────────────────────────
        neck_m   = bpy.data.objects.get("MARKER_NECK")
        chest_m  = bpy.data.objects.get("MARKER_CHEST")
        pelvis_m = bpy.data.objects.get("MARKER_PELVIS")
        if not neck_m or not pelvis_m:
            self.report({'ERROR'},
                "Run 'Auto Detect Body & Legs' first — NECK and PELVIS markers "
                "are needed as anchors.")
            return {'CANCELLED'}
        neck_z   = neck_m.location.z
        chest_z  = chest_m.location.z  if chest_m  else None
        pelvis_z = pelvis_m.location.z

        # ── Ensure arm marker empties exist ───────────────────────────────────
        col = get_or_create_collection("RigifyMarkers")
        for aname in ("SHOULDER_L", "SHOULDER_R", "ARM_L", "ARM_R",
                      "ELBOW_L",    "ELBOW_R",    "HAND_L", "HAND_R",
                      "HAND_TIP_L", "HAND_TIP_R"):
            oname = f"MARKER_{aname}"
            if oname not in bpy.data.objects:
                make_empty(col, oname, 'SPHERE', Vector((0, 0, 0)), ARM_SIZE, context)
        col.hide_viewport = False

        # ── Geometry ─────────────────────────────────────────────────────────
        D      = AUTORIG_OT_AutoDetectBody
        verts  = D._world_verts(mesh_obj)
        if not verts:
            self.report({'ERROR'}, "Mesh has no vertices.")
            return {'CANCELLED'}

        min_z  = min(v.z for v in verts)
        max_z  = max(v.z for v in verts)
        min_x  = min(v.x for v in verts)
        max_x  = max(v.x for v in verts)
        min_y  = min(v.y for v in verts)
        cx     = (min_x + max_x) * 0.5
        cy     = (min_y + max(v.y for v in verts)) * 0.5
        height = max_z - min_z

        profile = D._scan_width_profile(verts, min_z, max_z, steps=200)

        # ── BVH for Y surface snap ────────────────────────────────────────────
        import bmesh as _bm_arm
        from mathutils.bvhtree import BVHTree as _BVH_arm
        _bm = _bm_arm.new()
        _bm.from_mesh(mesh_obj.data)
        _bm.transform(mesh_obj.matrix_world)
        bvh    = _BVH_arm.FromBMesh(_bm)
        _bm.free()
        depth  = max(max(v.y for v in verts) - min_y,
                     max_x - min_x, max_z - min_z)
        margin = depth * 5.0

        def _bvh_y(x, z):
            ray_o = Vector((x, min_y - margin, z))
            ray_d = Vector((0, 1, 0))
            hit, _, _, _ = bvh.ray_cast(ray_o, ray_d)
            if not hit:
                return cy
            front_y = hit.y
            last = hit
            while True:
                nxt, _, _, _ = bvh.ray_cast(last + Vector((0, 0.0001, 0)), ray_d)
                if not nxt:
                    break
                last = nxt
            return front_y + (last.y - front_y) * 0.5

        # ── Per-side detection via shared static method ───────────────────────
        def _push_inside(pos, _md=height * 0.025):
            nearest, _, _, dist = bvh.find_nearest(pos)
            if nearest is None or dist >= _md:
                return pos
            inward = pos - nearest
            L = inward.length
            if L < 1e-7:
                return pos
            return nearest + inward * (_md / L)

        moved       = 0
        arm_results = D._detect_arms(
            verts, profile, neck_z, chest_z, pelvis_z, cx, height, min_z,
            _bvh_y, _push_inside)

        for side in ('L', 'R'):
            if side not in arm_results:
                self.report({'WARNING'}, f"Could not detect arm on side {side}.")
                continue
            sh_pos, arm_mid_pos, elbow_pos, hand_pos, tip_pos = arm_results[side]
            if sh_pos      and D._set_marker(f"SHOULDER_{side}", sh_pos):     moved += 1
            if arm_mid_pos and D._set_marker(f"ARM_{side}",      arm_mid_pos): moved += 1
            if elbow_pos   and D._set_marker(f"ELBOW_{side}",    elbow_pos):  moved += 1
            if hand_pos    and D._set_marker(f"HAND_{side}",     hand_pos):   moved += 1
            if tip_pos     and D._set_marker(f"HAND_TIP_{side}", tip_pos):    moved += 1

        # ── Symmetry enforcement ──────────────────────────────────────────────
        for ln, rn in (("SHOULDER_L", "SHOULDER_R"), ("ARM_L", "ARM_R"),
                       ("ELBOW_L",    "ELBOW_R"),    ("HAND_L", "HAND_R"),
                       ("HAND_TIP_L", "HAND_TIP_R")):
            lo = bpy.data.objects.get(f"MARKER_{ln}")
            ro = bpy.data.objects.get(f"MARKER_{rn}")
            if lo and ro:
                ax = (abs(lo.location.x) + abs(ro.location.x)) * 0.5
                ay = (lo.location.y + ro.location.y) * 0.5
                az = (lo.location.z + ro.location.z) * 0.5
                lo.location = Vector(( ax, ay, az))
                ro.location = Vector((-ax, ay, az))

        # ── Show arm markers ──────────────────────────────────────────────────
        for name in ("SHOULDER_L", "SHOULDER_R", "ARM_L", "ARM_R",
                     "ELBOW_L",    "ELBOW_R",    "HAND_L", "HAND_R",
                     "HAND_TIP_L", "HAND_TIP_R"):
            obj = bpy.data.objects.get(f"MARKER_{name}")
            if obj:
                try:
                    obj.hide_set(False)
                except RuntimeError:
                    pass

        self.report({'INFO'},
            f"Detected {moved} arm markers on '{mesh_obj.name}'. "
            "Review and fine-tune if needed.")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Detect face objects
# ---------------------------------------------------------------------------

# ── Geometric face-detect scale normalization ─────────────────────────────────
# The geometric face detector mixes proportional placement (scale-safe) with
# hardcoded metre offsets — temple +40mm, lip/lid insets (_LIP_IN, _LID_IN),
# interior surface snaps, cheek/jaw/chin nudges. Those are tuned for a ~human head
# (~0.25 m); on a very small (or very large) character they dwarf the head, throw
# ray probes far off, and find_nearest grabs whatever's closest (a finger). Rather
# than convert dozens of offsets, normalize the whole thing to a canonical size —
# scale the face mesh(es) + all face markers about the mesh centre, run the tuned
# detector, then invert (mapping newly placed markers back to the true scale).
_FACE_MESH_ATTRS = ("body_obj", "eye_obj", "eye_l_obj", "eye_r_obj",
                    "tongue_obj", "teeth_obj", "teeth_top_obj", "teeth_bot_obj",
                    "brow_obj", "brow_l_obj", "brow_r_obj")


def _face_refresh():
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass


@contextmanager
def _face_detect_normalized(props):
    """Temporarily normalize the face mesh(es) + face markers to a canonical head
    size (about the mesh centre) so the detector's absolute-metre offsets work at
    any character scale. Yields the applied factor; inverts everything on exit."""
    body = getattr(props, "body_obj", None) if props else None
    if not (body and getattr(body, "type", None) == 'MESH'):
        yield 1.0
        return
    mn, mx, cen = _mesh_bbox_world(body)
    # Normalize by FRONT-BACK DEPTH (Y extent), not height/width. Depth is the most
    # robust head-scale proxy: it's ~the same for a head-only mesh and a full body
    # (~0.24 m either way, since a torso isn't much deeper than a head), and it's
    # POSE-INVARIANT — spread arms extend X, raised arms extend Z, but neither adds
    # much Y. (Height fails for head-only meshes; width fails for A/T-pose arms.)
    depth = float(mx.y - mn.y)
    if depth <= 1e-6:
        yield 1.0
        return
    _CANON_DEPTH = 0.24            # ~human head/torso front-back depth (m)
    s = _CANON_DEPTH / depth
    if 0.8 <= s <= 1.25:          # already ~canonical -> leave untouched
        yield 1.0
        return

    # Pivot the X scaling on x=0 — the symmetry plane the detector assumes (it forces
    # every midline marker to x=0 and mirrors pairs about it). Pivoting on the bbox
    # centre X instead would shift the two sides unequally vs x=0 and make the detected
    # markers asymmetric. Y/Z pivot on the mesh centre (no symmetry constraint there).
    C = Vector((0.0, cen.y, cen.z))

    def _about_c(fac):
        return (Matrix.Translation(C)
                @ Matrix.Diagonal((fac, fac, fac, 1.0))
                @ Matrix.Translation(-C))

    objs = []
    for attr in _FACE_MESH_ATTRS:
        o = getattr(props, attr, None)
        if o and getattr(o, "type", None) == 'MESH' and o not in objs:
            objs.append(o)
    orig = {o.name: o.matrix_world.copy() for o in objs}
    print(f"[face-scale] mesh depth {depth:.3f}m (canonical {_CANON_DEPTH}m) -> "
          f"normalizing x{s:.3f} for detection")

    Sm = _about_c(s)
    for o in objs:
        o.matrix_world = Sm @ o.matrix_world
    for o in bpy.data.objects:                     # scale existing face markers too
        if o.name.startswith("MARKER_FACE_"):
            o.location = C + (o.location - C) * s
    _face_refresh()
    try:
        yield s
    finally:
        inv = 1.0 / s
        for o in objs:
            if o.name in orig:
                o.matrix_world = orig[o.name]
        for o in bpy.data.objects:                 # old markers -> identity, new -> true
            if o.name.startswith("MARKER_FACE_"):
                o.location = C + (o.location - C) * inv
        _face_refresh()


class AUTORIG_OT_DetectFaceObjects(bpy.types.Operator):
    """Add any missing face markers then move them to detected positions."""
    bl_idname  = "autorig.detect_face_objects"
    bl_label   = "Detect Face Markers"
    bl_options = {'REGISTER', 'UNDO'}
    section:   bpy.props.StringProperty(default='ALL')
    left_only: bpy.props.BoolProperty(default=False)
    # Fill-only mode: place ONLY markers that don't exist yet, leaving any
    # already-placed markers untouched. Used after AI Detect Face (which places
    # the 30 core landmarks) to fill in the ~34 secondary/interpolation markers
    # the neural model doesn't output, without overwriting the AI's placements.
    fill_missing_only: bpy.props.BoolProperty(default=False)

    _EYELID_BASES = {
        'MARKER_FACE_EYE_CENTER', 'MARKER_FACE_EYE_TOP', 'MARKER_FACE_EYE_BOT',
        'MARKER_FACE_EYE_INNER',  'MARKER_FACE_EYE_OUTER',
        'MARKER_FACE_LID_CREASE_T',
        'MARKER_FACE_CREASE_INNER', 'MARKER_FACE_CREASE_OUTER',
        'MARKER_FACE_BROW_BOT_OUTER',
    }

    _CUBE_BASES = {
        'MARKER_FACE_TEETH_T', 'MARKER_FACE_TEETH_B',
        'MARKER_FACE_LID_CREASE_T',
        'MARKER_FACE_CREASE_INNER', 'MARKER_FACE_CREASE_OUTER',
        'MARKER_FACE_BROW_BOT_OUTER',
    }

    _PLAIN_AXES_BASES = {
        'MARKER_FACE_TONGUE_1', 'MARKER_FACE_TONGUE_2', 'MARKER_FACE_TONGUE_3',
    }

    def _snap(self, name, pos):
        if self.left_only and name.endswith('_R'):
            return True
        obj_name = f"MARKER_{name}"
        # Fill-only: never move a marker that already exists (e.g. the AI core
        # landmarks). The section's detection math still runs so it can read
        # those markers as anchors — only the write is suppressed.
        if self.fill_missing_only and bpy.data.objects.get(obj_name) is not None:
            return True
        obj = bpy.data.objects.get(obj_name)
        base = '_'.join(obj_name.rsplit('_', 1)[:-1]) if obj_name.endswith(('_L', '_R')) else obj_name
        if obj is None:
            col = get_or_create_collection("RigifyFaceMarkers")
            col.hide_viewport = False
            obj = bpy.data.objects.new(obj_name, None)
            obj.empty_display_size = FACE_SIZE
            obj["autorig_marker"] = True
            obj.show_in_front = True
            sc = 0.2 if base in self._EYELID_BASES else 0.4
            obj.scale = (sc, sc, sc)
            col.objects.link(obj)
        if base in self._CUBE_BASES:
            obj.empty_display_type = 'CUBE'
        elif base in self._PLAIN_AXES_BASES:
            obj.empty_display_type = 'PLAIN_AXES'
        else:
            obj.empty_display_type = 'SPHERE'
        obj.location = pos
        obj.hide_set(False)
        return True

    @staticmethod
    def _mesh_verts_world(obj):
        if not obj or obj.type != 'MESH':
            return []
        mw = obj.matrix_world
        return [(mw @ v.co).copy() for v in obj.data.vertices]

    @staticmethod
    def _nearest_vert(body_verts, estimate, force_x0=False):
        if not body_verts:
            return estimate.copy()
        pos = min(body_verts, key=lambda v, e=estimate: (v - e).length_squared).copy()
        if force_x0:
            pos.x = 0.0
        return pos

    def execute(self, context):
        props = context.scene.autorig_face_objs
        with _face_detect_normalized(props):
            return self._execute_core(context)

    def _execute_core(self, context):
        import bmesh as _bmesh_fdet
        from mathutils.bvhtree import BVHTree as _BVHTree_fdet

        props  = context.scene.autorig_face_objs
        sec    = (self.section or 'ALL').upper()
        moved  = 0
        miss   = []

        body_bvh   = None
        _body_mn_y = 0.0
        if props.body_obj and props.body_obj.type == 'MESH':
            _bm_f = _bmesh_fdet.new()
            _bm_f.from_mesh(props.body_obj.data)
            _bm_f.transform(props.body_obj.matrix_world)
            body_bvh   = _BVHTree_fdet.FromBMesh(_bm_f)
            _bm_f.free()
            _body_mn_y = _mesh_bbox_world(props.body_obj)[0].y

        def _fwd(x, z):
            if body_bvh is None:
                return None
            hit, _, _, _ = body_bvh.ray_cast(
                Vector((x, _body_mn_y - 1.0, z)), Vector((0, 1, 0)))
            return hit

        def _lat(y, z, side):
            if body_bvh is None:
                return None
            x_sign = 1 if side == 'L' else -1
            hit, _, _, _ = body_bvh.ray_cast(
                Vector((x_sign * 5.0, y, z)), Vector((-x_sign, 0, 0)))
            return hit

        # ── LIPS ──────────────────────────────────────────────────────────────
        if sec in ('ALL', 'LIPS'):
            _lip_miss = ['FACE_LIP_T', 'FACE_LIP_B', 'FACE_LIP_BOT',
                         'FACE_MOUTH_CORNER_L', 'FACE_MOUTH_CORNER_R',
                         'FACE_MOUTH_TOP_L', 'FACE_MOUTH_TOP_R',
                         'FACE_MOUTH_BOT_L', 'FACE_MOUTH_BOT_R']
            # Mouth band (tz=top, bz=bottom) from the eye/nose ESTIMATE only.
            # The teeth guide is deliberately NOT consulted (even when teeth
            # meshes are assigned): cartoon teeth spans are taller than the
            # lips and routinely dragged the band/scan window off the mouth —
            # the skin seam scan below locates the real lips on its own.
            tz = bz = None
            if body_bvh is not None and props.body_obj and props.body_obj.type == 'MESH':
                _eL = bpy.data.objects.get('MARKER_FACE_EYE_CENTER_L')
                _eR = bpy.data.objects.get('MARKER_FACE_EYE_CENTER_R')
                if _eL and _eR:
                    _eye_cz = (_eL.location.z + _eR.location.z) * 0.5
                    _bo  = props.body_obj
                    _nvv = len(_bo.data.vertices)
                    _ff  = np.empty(_nvv * 3, dtype=np.float64)
                    _bo.data.vertices.foreach_get('co', _ff)
                    _m4  = np.array(_bo.matrix_world, dtype=np.float64)
                    _crown = float((_ff.reshape(_nvv, 3) @ _m4[:3, :3].T + _m4[:3, 3])[:, 2].max())
                    _face_v = max(_crown - _eye_cz, 0.02)
                    # Anchor to the NOSE TIP: the most-forward (min-Y) front-centre point
                    # below the eyes. It holds a stable ratio to the mouth across styles,
                    # unlike the crown (a big forehead pushed the lips down to the chin).
                    _nt_y = float('inf')
                    _nt_z = _eye_cz - _face_v * 0.4
                    _zlo  = _eye_cz - _face_v * 0.70        # search the upper mid-face only
                    _z    = _eye_cz - _face_v * 0.15
                    _st   = max(_face_v / 30.0, 0.002)
                    while _z > _zlo:
                        _hh = _fwd(0.0, _z)
                        if _hh is not None and _hh.y < _nt_y:
                            _nt_y = _hh.y; _nt_z = _z
                        _z -= _st
                    _e2n = max(_eye_cz - _nt_z, 0.01)       # eye line -> nose tip
                    _mzc = _nt_z - _e2n * 0.80              # mouth ~0.8x below the nose tip
                    _msp = _e2n * 0.35                      # mouth vertical opening
                    tz = _mzc + _msp * 0.5
                    bz = _mzc - _msp * 0.5
            if tz is None or bz is None or body_bvh is None:
                miss.extend(_lip_miss)
            else:
                z_span = tz - bz
                mz     = (tz + bz) * 0.5

                # ── Seam refinement (auto-pick, band-bounded) ─────────────────
                # The eye/nose estimate gives the mouth REGION; the exact lip
                # edges come from the mesh when it has a real crease: profile
                # the centre column, seam = the most-recessed point between the
                # two lip bulges, lip edge = where the surface comes FORWARD
                # again off the seam. Smooth-bulge lips (no crease modelled —
                # depth below 8% of the band) keep the band placement, per the
                # auto-pick rule: a crease detector must not invent one.
                # Window: room for the LIP BULGES (they sit near/just outside
                # the band — a tight ±15% window missed them and no seam was
                # found) but still under the NOSE (a ±55% window reached its
                # recession and put the lips below the nose). Asymmetric:
                # +30% above (nose guard), -40% below (lower lip is bigger).
                _scan_t = tz + z_span * 0.30
                _scan_b = bz - z_span * 0.40
                _sstep  = max((_scan_t - _scan_b) / 18.0, 0.0015)
                _profc  = []
                _z2 = _scan_t
                while _z2 >= _scan_b:
                    _hh2 = _fwd(0.0, _z2)
                    if _hh2 is not None:
                        _profc.append((_z2, _hh2.y))
                    _z2 -= _sstep
                seam_z  = None
                lip_t_z = tz
                lip_b_z = bz
                _cdep   = 0.0
                if len(_profc) >= 5:
                    # Seam = the interior point with the deepest recession
                    # relative to the most-forward point on EACH side. No
                    # assumption about WHERE it sits in the window — the old
                    # band-midpoint split put both "bulges" on the same lip
                    # when the estimate band sat high (depth came out
                    # negative and no seam was ever found).
                    _best_d, _best_i = 0.0, None
                    for _i3 in range(1, len(_profc) - 1):
                        _ya = min(p[1] for p in _profc[:_i3])       # fwd above
                        _yb = min(p[1] for p in _profc[_i3 + 1:])   # fwd below
                        _d3 = _profc[_i3][1] - max(_ya, _yb)
                        if _d3 > _best_d:
                            _best_d, _best_i = _d3, _i3
                    if _best_i is not None and _best_d > z_span * 0.08:
                        seam_z = _profc[_best_i][0]
                        _cdep  = _best_d
                        # LIP_T/B on the LIP BODIES = the most-forward point
                        # on each side of the seam (the two lip bulges).
                        _bt = min(_profc[:_best_i], key=lambda p: p[1])
                        _bb = min(_profc[_best_i + 1:], key=lambda p: p[1])
                        lip_t_z = min(max(_bt[0], seam_z + _sstep),
                                      tz + z_span * 0.25)
                        lip_b_z = max(min(_bb[0], seam_z - _sstep),
                                      bz - z_span * 0.25)
                        print(f"[lips] crease seam z={seam_z:.4f} "
                              f"depth={_cdep*1000:.1f}mm -> lips at bulges "
                              f"t={lip_t_z:.4f} b={lip_b_z:.4f}")
                    else:
                        print(f"[lips] no crease in window "
                              f"(best {_best_d*1000:.1f}mm vs need "
                              f"{z_span*0.08*1000:.1f}mm) -- band placement")
                else:
                    print(f"[lips] profile too sparse ({len(_profc)} pts) "
                          f"-- band placement")

                h = _fwd(0.0, lip_t_z)
                if h:
                    p = h.copy(); p.x = 0.0
                    if self._snap('FACE_LIP_T', p): moved += 1
                    else: miss.append('FACE_LIP_T')
                else:
                    miss.append('FACE_LIP_T')

                h = _fwd(0.0, lip_b_z)
                if h:
                    p = h.copy(); p.x = 0.0
                    if self._snap('FACE_LIP_B', p): moved += 1
                    else: miss.append('FACE_LIP_B')
                else:
                    miss.append('FACE_LIP_B')

                # LIP_BOT: below the LOWER-LIP CREASE (the mentolabial fold —
                # the recession where the lip body ends above the chin boss)
                # when the mesh has one: first local recession below the lower
                # lip. Smooth mesh -> the old half-band offset.
                # LENGTH CAP: the fold sits close under the lip; the CHIN/JAW
                # crease is much further down. Scan (and clamp the result) only
                # within _bot_cap of the lower lip, so a low lip marker can't
                # let LIP_BOT run down to the jaw or below it.
                _bot_cap  = max(z_span * 1.25, 0.012)
                lip_bot_z = lip_b_z - max(z_span * 0.5, 0.010)
                _blprof = []
                _z4 = lip_b_z - _sstep * 0.5
                while _z4 >= lip_b_z - _bot_cap:
                    _h5 = _fwd(0.0, _z4)
                    if _h5 is None:
                        break
                    _blprof.append((_z4, _h5.y))
                    _z4 -= _sstep * 0.5
                _fold = None
                for _j4 in range(1, len(_blprof) - 1):
                    if (_blprof[_j4][1] >= _blprof[_j4 - 1][1]
                            and _blprof[_j4][1] > _blprof[_j4 + 1][1]):
                        _fold = _blprof[_j4]
                        break
                if _fold is None and len(_blprof) >= 4:
                    # No strict bump (scan may end before the chin boss turns
                    # the surface forward again): accept the deepest INTERIOR
                    # recession — interior means the profile came back forward
                    # after it, i.e. a real fold, not the throat run-off.
                    _gj = max(range(1, len(_blprof) - 1),
                              key=lambda j: _blprof[j][1])
                    if (_blprof[_gj][1] > _blprof[0][1]
                            and _blprof[_gj][1] > _blprof[-1][1]):
                        _fold = _blprof[_gj]
                if _fold is not None:
                    lip_bot_z = _fold[0]
                    print(f"[lips] lower-lip crease at z={lip_bot_z:.4f} "
                          f"-> LIP_BOT")
                else:
                    print(f"[lips] no lower-lip fold found "
                          f"({len(_blprof)} pts scanned) -- LIP_BOT at offset")
                # Hard clamp (insurance for the offset default too): never more
                # than _bot_cap below the lower lip.
                _bot_floor = lip_b_z - _bot_cap
                if lip_bot_z < _bot_floor:
                    lip_bot_z = _bot_floor
                    print(f"[lips] LIP_BOT capped at {_bot_cap*1000:.0f}mm "
                          f"below the lower lip (was reaching the jaw)")
                h = _fwd(0.0, lip_bot_z)
                if h:
                    p = h.copy(); p.x = 0.0
                    if self._snap('FACE_LIP_BOT', p): moved += 1
                    else: miss.append('FACE_LIP_BOT')
                else:
                    miss.append('FACE_LIP_BOT')

                # Corner = where the lip CREASE ends on X: the upper and lower lip
                # converge at the commissure, so the lip's vertical thickness (forward-
                # shell Z extent) is large at the centre and collapses to ~0 there.
                # Read it from the mesh vertices around the mouth band.
                _bd  = max(tz - bz, 1e-4)
                _mw  = props.body_obj.matrix_world
                _reg = [_mw @ v.co for v in props.body_obj.data.vertices]
                _reg = [v for v in _reg
                        if (bz - _bd) < v.z < (tz + _bd) and abs(v.x) < _bd * 8.0]
                _shell = []
                if _reg:
                    _miny  = min(v.y for v in _reg)
                    _shell = [v for v in _reg if v.y < _miny + _bd * 1.3]

                # Corner height = the DETECTED seam when we have one, not the
                # band midpoint: with the no-teeth estimate band the midpoint
                # sits above the true mouth line, which floated the corners
                # above the lip corners.
                _corner_z = seam_z if seam_z is not None else mz

                def _corner(x_sign):
                    # Returns None unless the thickness collapse actually
                    # CONVERGED. An unconverged walk (X-cap / vert gaps — e.g.
                    # cheek verts as forward as the lips keep the shell extent
                    # high past the commissure) used to return the last visited
                    # column, planting the corner far PAST the real lip corner.
                    if not _shell:
                        return None
                    step = _bd * 0.10; chw = _bd * 0.60; xcap = _bd * 6.0
                    base = None; corner = None; gaps = 0; why = "end"
                    for i in range(1, 80):
                        xc = x_sign * step * i
                        if abs(xc) > xcap:
                            why = "xcap"; break
                        col = [v for v in _shell if abs(v.x - xc) < chw]
                        if len(col) < 3:
                            gaps += 1
                            if gaps > 4:
                                why = "gaps"; break
                            continue
                        gaps = 0
                        ext = max(v.z for v in col) - min(v.z for v in col)
                        if base is None:
                            base = ext
                        h = _fwd(xc, _corner_z)
                        corner = Vector((xc, h.y if h else
                                         sum(v.y for v in col) / len(col),
                                         _corner_z))
                        if base and ext < base * 0.62:     # lips converged -> corner
                            print(f"[lips] corner shell converged at "
                                  f"x={xc*1000:+.0f}mm (i={i}, base ext "
                                  f"{base*1000:.1f}mm -> {ext*1000:.1f}mm)")
                            return corner
                    print(f"[lips] corner shell walk did not converge "
                          f"({why}) -> IOD fallback")
                    return None

                def _corner_crease(x_sign):
                    # User's lip model: the corner is where the CREASE ends on
                    # X. Per column the crease depth uses the SAME recession
                    # scoring that found the seam at the centre (interior point
                    # behind the most-forward point on EACH side) — the old raw
                    # min/max window measured whole-face curvature and drowned
                    # shallow seams, which forced a strong-seam gate and pushed
                    # shallow-lip characters onto the shell corner (which fades
                    # on CHEEK curvature, far past the real corner). The corner
                    # is the LAST column where the crease still holds ≥35% of
                    # the centre depth; the fade must persist 2 columns so one
                    # noisy column can't end the lips early.
                    step = _bd * 0.10
                    xcap = _bd * 6.0

                    def _col_depth(xc):
                        prof = []
                        _z3 = seam_z + z_span * 0.5
                        while _z3 >= seam_z - z_span * 0.5:
                            _h3 = _fwd(xc, _z3)
                            if _h3 is not None:
                                prof.append(_h3.y)
                            _z3 -= _sstep
                        if len(prof) < 5:
                            return None
                        best = 0.0
                        for i in range(1, len(prof) - 1):
                            _ya = min(prof[:i])
                            _yb = min(prof[i + 1:])
                            best = max(best, prof[i] - max(_ya, _yb))
                        return best

                    d0 = _col_depth(0.0)
                    if not d0 or d0 <= 1e-6:
                        return None
                    last = None
                    fade = gaps = 0
                    for i in range(1, 80):
                        xc = x_sign * step * i
                        if abs(xc) > xcap:
                            return None    # crease never ended — untrustworthy
                        d = _col_depth(xc)
                        if d is None:
                            gaps += 1
                            if gaps > 4:
                                return None
                            continue
                        gaps = 0
                        if d >= d0 * 0.35:
                            _h4 = _fwd(xc, seam_z)
                            if _h4 is not None:
                                last = Vector((xc, _h4.y, seam_z))
                            fade = 0
                        else:
                            fade += 1
                            if fade >= 2 and last is not None:
                                print(f"[lips] corner crease-end at "
                                      f"x={last.x*1000:+.0f}mm (centre depth "
                                      f"{d0*1000:.1f}mm)")
                                return last
                    return None

                # A real mouth is WIDER than the lips are thick. When the crease
                # is near noise-floor the crease/shell walk can converge a few mm
                # out (an 8mm-wide mouth), collapsing the markers. Require the
                # corner half-width to clear this floor; below it, use the IOD
                # proportion (mouth width ~ eye distance) which is reliable.
                _mouth_min_half = abs(lip_t_z - lip_b_z) * 0.7
                for side in ('L', 'R'):
                    x_sign = 1 if side == 'L' else -1
                    corner_pos = (_corner_crease(x_sign)
                                  if seam_z is not None else None)
                    if corner_pos is not None and abs(corner_pos.x) < _mouth_min_half:
                        # Crease converged into an implausibly narrow mouth
                        # (near-noise crease). Skip the shell walk — it fades on
                        # the CHEEK on these shallow lips (too WIDE) — go to IOD.
                        print(f"[lips] crease corner {side} too narrow "
                              f"({abs(corner_pos.x)*1000:.0f}mm < "
                              f"{_mouth_min_half*1000:.0f}mm floor) -- IOD")
                        corner_pos = None
                    elif corner_pos is None:
                        # Crease didn't run (no seam) or didn't converge:
                        # shell-thickness corner, floor-checked.
                        corner_pos = _corner(x_sign)
                        if corner_pos is not None and abs(corner_pos.x) < _mouth_min_half:
                            corner_pos = None
                    if corner_pos is None:
                        # Proportional: mouth width ~= eye distance, corner at
                        # half-IOD on the seam height, on the skin.
                        _ceL = bpy.data.objects.get('MARKER_FACE_EYE_CENTER_L')
                        _ceR = bpy.data.objects.get('MARKER_FACE_EYE_CENTER_R')
                        if _ceL and _ceR:
                            _iodx = abs(_ceL.location.x - _ceR.location.x)
                            if _iodx > 1e-5:
                                _hc = _fwd(x_sign * _iodx * 0.5, _corner_z)
                                if _hc is not None:
                                    corner_pos = _hc.copy()
                                    print(f"[lips] corner {side} from IOD "
                                          f"proportion (x={_iodx*0.5*1000:+.0f}mm)")
                    if corner_pos:
                        if self._snap(f'FACE_MOUTH_CORNER_{side}', corner_pos): moved += 1
                        else: miss.append(f'FACE_MOUTH_CORNER_{side}')
                    else:
                        miss.append(f'FACE_MOUTH_CORNER_{side}')

                for side in ('L', 'R'):
                    x_sign  = 1 if side == 'L' else -1
                    corn_m  = bpy.data.objects.get(f'MARKER_FACE_MOUTH_CORNER_{side}')
                    probe_x = corn_m.location.x * 0.5 if corn_m else 0.015 * x_sign
                    for mname, ref_z in (('MOUTH_TOP', lip_t_z), ('MOUTH_BOT', lip_b_z)):
                        h = _fwd(probe_x, ref_z)
                        if h:
                            if self._snap(f'FACE_{mname}_{side}', h): moved += 1
                            else: miss.append(f'FACE_{mname}_{side}')
                        else:
                            miss.append(f'FACE_{mname}_{side}')

        # ── NOSE ──────────────────────────────────────────────────────────────
        if sec in ('ALL', 'NOSE'):
            _nose_miss = ('FACE_NOSE_BRIDGE', 'FACE_NOSE_TIP', 'FACE_NOSE_BOT',
                          'FACE_NOSE_BRIDGE_L', 'FACE_NOSE_BRIDGE_R',
                          'FACE_NOSE_WING_L',   'FACE_NOSE_WING_R')
            lip_t_m = bpy.data.objects.get('MARKER_FACE_LIP_T')
            if not lip_t_m or body_bvh is None:
                for m in _nose_miss: miss.append(m)
            else:
                lip_z  = lip_t_m.location.z
                _b1L_m = bpy.data.objects.get('MARKER_FACE_BROW_1_L')
                brow_z = _b1L_m.location.z if _b1L_m else lip_z + 0.120

                tip_lo, tip_hi = lip_z + 0.015, lip_z + 0.070
                nose_tip, tip_y_best = None, float('inf')
                for k in range(13):
                    h = _fwd(0.0, tip_lo + (tip_hi - tip_lo) * k / 12)
                    if h and h.y < tip_y_best:
                        nose_tip = h.copy(); nose_tip.x = 0.0; tip_y_best = h.y
                if nose_tip:
                    if self._snap('FACE_NOSE_TIP', nose_tip): moved += 1
                    else: miss.append('FACE_NOSE_TIP')
                else:
                    miss.append('FACE_NOSE_TIP')

                # Anchor BOT and WING off the reliable TIP.
                tip_z  = nose_tip.z if nose_tip else (lip_z + 0.040)
                tip_y  = nose_tip.y if nose_tip else 0.0
                nose_h = max(tip_z - lip_z, 0.006)

                # NOSE_BOT: scan DOWN from the tip; the base/subnasale is where the
                # underside recedes back toward the philtrum (front-Y goes back past a
                # margin). Take that recession point.
                nbot = None
                _bot_found = False
                for k in range(1, 16):                  # scan only the top ~75% from the
                    z = tip_z - nose_h * 0.05 * k        # tip — never down to the lip
                    hh = _fwd(0.0, z)
                    if hh is None:
                        break
                    nbot = Vector((0.0, hh.y, z))
                    if hh.y - tip_y > nose_h * 0.5:     # underside receded -> base reached
                        _bot_found = True
                        break
                if not _bot_found:
                    # Smooth/cartoon nose with no clear recession: use a proportional
                    # subnasale (~45% from tip toward the lip) so NOSE_BOT can't slide
                    # onto the lips.
                    _fz = tip_z - nose_h * 0.45
                    _fh = _fwd(0.0, _fz)
                    nbot = Vector((0.0, _fh.y if _fh else tip_y, _fz))
                if nbot:
                    if self._snap('FACE_NOSE_BOT', nbot): moved += 1
                    else: miss.append('FACE_NOSE_BOT')
                else:
                    miss.append('FACE_NOSE_BOT')

                # NOSE_WING: SIDEWAYS from the tip. Scan X outward at the TIP's Z; the
                # wing is where the nose HEIGHT drops off (front-Y recedes sharply) at
                # the side of the nose. Keep the scan short so it stays on the nose.
                _wing_z = tip_z
                # Scan X outward at height z; return the point where the nose surface
                # DROPS OFF (front-Y recedes sharply) = the groove at the side of the
                # nose. Used for the wing (at the tip) and the bridge sides (at the
                # nasion). The marker lands IN the groove (the recessed point).
                def _nose_groove(x_sign, z, max_x):
                    """Returns (point, groove_found). groove_found is True ONLY when a
                    real side-of-nose recession was hit — not when the scan just ran out
                    to max_x (which, on a smooth nose, lands on the cheek)."""
                    step = max(nose_h * 0.10, 0.0015)
                    prev = None; out = None; found = False
                    for i in range(1, 20):
                        xc = x_sign * step * i
                        if abs(xc) > max_x:            # capped reach (keeps it on the nose)
                            break
                        hw = _fwd(xc, z)
                        if hw is None:
                            break
                        if prev is not None and (hw.y - prev) > nose_h * 0.35:
                            if (hw.y - prev) > nose_h * 1.0:
                                break        # huge recession = a hole/socket (eye) ->
                                             # stop BEFORE it, not inside
                            out = Vector((xc, hw.y, z)); found = True; break   # groove
                        out = Vector((xc, hw.y, z))
                        prev = hw.y
                    return out, found

                # NOSE_BRIDGE = nasion (top of the nose, between the eyes). Anchor to the
                # EYE line from the eye meshes -- brow-independent, and correct on wide
                # noses where the brow-proportion version hung mid-nose. Falls back to
                # the lip->brow proportion if no eye meshes are picked.
                _eye_cz = None
                if props.use_eyes:
                    if props.eye_count == 'COMBINED' and props.eye_obj:
                        _eye_cz = _mesh_bbox_world(props.eye_obj)[2].z
                    elif props.eye_l_obj and props.eye_r_obj:
                        _eye_cz = (_mesh_bbox_world(props.eye_l_obj)[2].z +
                                   _mesh_bbox_world(props.eye_r_obj)[2].z) * 0.5
                bridge_z = (_eye_cz - nose_h * 0.15) if _eye_cz is not None \
                    else lip_z + (brow_z - lip_z) * 0.60
                h = _fwd(0.0, bridge_z)
                if h:
                    p = h.copy(); p.x = 0.0
                    if self._snap('FACE_NOSE_BRIDGE', p): moved += 1
                    else: miss.append('FACE_NOSE_BRIDGE')
                else:
                    miss.append('FACE_NOSE_BRIDGE')

                for side in ('L', 'R'):
                    x_sign = 1 if side == 'L' else -1
                    # bridge side: short reach so it stays on the narrow nose bridge,
                    # not out at the eyelid.
                    b, _ = _nose_groove(x_sign, bridge_z, nose_h * 0.6)
                    if b:
                        if self._snap(f'FACE_NOSE_BRIDGE_{side}', b): moved += 1
                        else: miss.append(f'FACE_NOSE_BRIDGE_{side}')
                    else:
                        miss.append(f'FACE_NOSE_BRIDGE_{side}')
                    # WIDTH CAP: the nostril wing sits at the side of the NOSE,
                    # not the face. Cap the groove scan reach (and clamp the
                    # result) so a weak/absent groove can't walk the marker out
                    # onto the cheek. Nose half-width is ~0.4-0.7 nose_h even on
                    # wide/cartoon noses, so 0.85 nose_h stays on the nose.
                    _wing_cap = nose_h * 0.85
                    w, _wf = _nose_groove(x_sign, _wing_z, _wing_cap)
                    if not _wf:
                        # Smooth/cartoon nose with no nostril groove: the scan would run
                        # out onto the CHEEK. Place the wing at a proportional nose edge.
                        _wx = x_sign * nose_h * 0.55
                        _wh = _fwd(_wx, _wing_z)
                        if _wh is not None:
                            w = Vector((_wx, _wh.y, _wing_z))
                    if w is not None and abs(w.x) > _wing_cap:
                        # Hard clamp (insurance): pull back onto the nose edge.
                        _wx = x_sign * _wing_cap
                        _wh = _fwd(_wx, _wing_z)
                        w = Vector((_wx, _wh.y if _wh else w.y, _wing_z))
                        print(f"[nose] WING_{side} capped at {_wing_cap*1000:.0f}mm "
                              f"from centre (was reaching the face side)")
                    if w:
                        if self._snap(f'FACE_NOSE_WING_{side}', w): moved += 1
                        else: miss.append(f'FACE_NOSE_WING_{side}')
                    else:
                        miss.append(f'FACE_NOSE_WING_{side}')

        # ── EYEBROWS + EYE CENTER (vertex-based) ──────────────────────────────
        if sec in ('ALL', 'EYEBROWS_EYE'):
            brow_mesh_L = brow_mesh_R = None
            if props.use_brows:
                if props.brow_count == 'SPLIT':
                    brow_mesh_L = props.brow_l_obj
                    brow_mesh_R = props.brow_r_obj
                else:
                    brow_mesh_L = brow_mesh_R = props.brow_obj

            brow_inner = {}   # side → innermost picked vertex (for FACE_BROW center)
            for side, brow_mesh in (('L', brow_mesh_L), ('R', brow_mesh_R)):
                x_sign = 1 if side == 'L' else -1
                names = [f'FACE_BROW_1_{side}', f'FACE_BROW_2_{side}',
                         f'FACE_BROW_3_{side}', f'FACE_BROW_OUTER_{side}']
                if brow_mesh and brow_mesh.type == 'MESH':
                    mw  = brow_mesh.matrix_world
                    wvs = [mw @ v.co for v in brow_mesh.data.vertices]
                    if props.brow_count == 'COMBINED':
                        side_verts = [v for v in wvs if (v.x * x_sign) > 0]
                        if not side_verts:
                            # Mirror modifier not applied — mesh only has one side.
                            # Mirror the opposite verts to get this side's positions.
                            opp = [v for v in wvs if (v.x * -x_sign) > 0]
                            from mathutils import Vector as _V
                            side_verts = [_V((-v.x, v.y, v.z)) for v in opp]
                        wvs = side_verts
                    # The brow runs inner->outer along X. Place each marker at the
                    # CENTRE of the brow mesh (centroid of a thin X-slice), NOT on a
                    # surface vertex.
                    _off   = [v.x * x_sign for v in wvs]
                    _xin   = min(_off); _bspan = max(max(_off) - _xin, 1e-5)
                    picks  = []
                    for mname, f in zip(names, (0.07, 0.38, 0.68, 0.95)):
                        _xc   = _xin + _bspan * f
                        _slab = [v for v in wvs if abs(v.x * x_sign - _xc) < _bspan * 0.15]
                        if not _slab:
                            _slab = sorted(wvs, key=lambda v: abs(v.x * x_sign - _xc)
                                           )[:max(1, len(wvs) // 8)]
                        # Geometric MIDPOINT of the slice (min+max)/2 in all axes ->
                        # centre of the brow mesh at this point.
                        _xs = [v.x for v in _slab]; _ys = [v.y for v in _slab]
                        _zs = [v.z for v in _slab]
                        _c = Vector(((min(_xs) + max(_xs)) * 0.5,
                                     (min(_ys) + max(_ys)) * 0.5 + 0.002151,
                                     (min(_zs) + max(_zs)) * 0.5))
                        self._snap(mname, _c); moved += 1
                        picks.append(_c)
                    brow_inner[side] = picks[0]
                else:
                    # No brow mesh — DETECT the brow ridge from the face skin: above the
                    # eye, the brow protrudes FORWARD (the supraorbital ridge). At each
                    # brow X-column, scan Z upward from the eye top and take the most-
                    # forward (min-Y) skin hit -> the ridge. The eye mesh only supplies
                    # the X span (inner->outer corner) and the scan start height.
                    _eye_mesh = None
                    if props.use_eyes:
                        _eye_mesh = (props.eye_l_obj if side == 'L' else props.eye_r_obj) \
                                    if props.eye_count == 'SPLIT' else props.eye_obj
                    if _eye_mesh and _eye_mesh.type == 'MESH' and body_bvh:
                        _emw = _eye_mesh.matrix_world
                        _ev  = [_emw @ v.co for v in _eye_mesh.data.vertices]
                        if props.eye_count == 'COMBINED':
                            _ev = [v for v in _ev if (v.x * x_sign) > 0] or _ev
                        # Use the VISIBLE eye opening, not the whole eye MESH.
                        # A full eyeball SPHERE (common: 20mm) puts its top far
                        # above the small visible eye, so the brow scan anchored
                        # to the mesh top looks ABOVE the brow and finds no
                        # bulge (depth goes negative). Keep only the exposed
                        # cap — verts on the front hemisphere AND in front of
                        # the face skin — same isolation the eyelid section uses.
                        _ray_org_y = _body_mn_y - 1.0
                        _ray_dir   = Vector((0, 1, 0))
                        _eyf = min(v.y for v in _ev); _eyb = max(v.y for v in _ev)
                        _eymid = _eyf + 0.55 * (_eyb - _eyf)
                        _exp = []
                        for v in _ev:
                            if v.y > _eymid:
                                continue
                            _bh = body_bvh.ray_cast(
                                Vector((v.x, _ray_org_y, v.z)), _ray_dir)[0]
                            if _bh is None or v.y <= _bh.y + 0.003:
                                _exp.append(v)
                        _src_v = _exp if len(_exp) >= 6 else _ev
                        _ezs   = [v.z for v in _src_v]
                        _etz   = max(_ezs)                       # VISIBLE eye top (height)
                        _ehh   = max(max(_ezs) - min(_ezs), 0.004)
                        # Horizontal span for the 4 brow columns = the FULL eye
                        # width (inner->outer canthus), from ALL verts. The
                        # exposed cap alone is the central visible part (lids
                        # cover the corners), so it under-spans and the brow
                        # markers bunch together — only the Z anchor above needs
                        # the visible-eye subset.
                        _inx   = min(_ev, key=lambda v: abs(v.x)).x   # inner corner
                        _outx  = max(_ev, key=lambda v: abs(v.x)).x   # outer corner

                        def _scan_bulge(bx):
                            # The brow at this column, ANATOMICALLY, walking UP
                            # the skin from the eye:
                            #   lid (skin follows the eyeball, forward)
                            #   -> RECESS  (lid fold / socket line)
                            #   -> [eyelid puff — a SMALL forward bulge]
                            #   -> furrow
                            #   -> BROW RIDGE — the BIG forward bulge (this is
                            #      what we want; the user's "big forward bulge
                            #      after the eyelids")
                            #   -> forehead (recedes).
                            # The brow is the MOST PROMINENT forward crest above
                            # the fold — bigger than the small eyelid puff — so
                            # pick by PROMINENCE, not by "first crest" (which
                            # grabbed the eyelid) and not by a tight height cap
                            # (which stopped at the eyelid before reaching the
                            # brow). No template height. _etz = VISIBLE eye top.
                            z0   = _etz - _ehh * 0.35
                            z1   = _etz + _ehh * 1.90
                            step = max((z1 - z0) / 48.0, 0.001)
                            prof = []
                            z = z0
                            while z <= z1:
                                hh = _fwd(bx, z)
                                if hh is not None:
                                    prof.append((z, hh.y))
                                z += step
                            n = len(prof)
                            if n < 8:
                                return None
                            ys = [p[1] for p in prof]
                            # recess (fold) = deepest (max-Y) point in the lower
                            # part of the window
                            _lo_end = max(2, int(n * 0.55))
                            j_rec = max(range(_lo_end), key=lambda j: ys[j])
                            _gate = max(0.0008, _ehh * 0.02)
                            # All forward crests (local min-Y) above the fold,
                            # each scored by PROMINENCE = how far it stands
                            # forward of the deepest recession on either side
                            # (fold below, forehead/furrow above). The brow
                            # ridge has the largest prominence.
                            best_prom, j_b = 0.0, None
                            for j in range(j_rec + 1, n - 1):
                                if not (ys[j] <= ys[j - 1] and ys[j] < ys[j + 1]):
                                    continue
                                _left  = max(ys[j_rec:j])       # recession below
                                _right = max(ys[j + 1:])        # recession above
                                _prom  = min(_left, _right) - ys[j]
                                if _prom > best_prom:
                                    best_prom, j_b = _prom, j
                            if j_b is None:
                                # Monotonic climb (no interior crest): take the
                                # most-forward point above the fold.
                                j_b = min(range(j_rec + 1, n), key=lambda j: ys[j])
                                best_prom = ys[j_rec] - ys[j_b]
                            if best_prom < _gate:
                                return None       # no real bulge above the fold
                            depth = ys[j_rec] - ys[j_b]   # crest vs fold
                            return (prof[j_b][0], prof[j_b][1], depth,
                                    prof[j_rec][0], best_prom)

                        # Scan all columns first; use the MEDIAN detected height
                        # to reject per-column outliers (a socket-corner nick on
                        # one column must not kink the row) and to fill columns
                        # where no bulge was found.
                        _cols = []
                        for mname, f in zip(names, (0.18, 0.45, 0.72, 1.00)):
                            _bx = _inx + (_outx - _inx) * f
                            _cols.append((mname, f, _bx, _scan_bulge(_bx)))
                        _offs = sorted((r[0] - _etz) / _ehh
                                       for (_, _, _, r) in _cols if r)
                        _med  = _offs[len(_offs) // 2] if _offs else None
                        _first = None
                        for mname, f, _bx, r in _cols:
                            if r is not None and abs((r[0] - _etz) / _ehh
                                                     - _med) > 0.35:
                                r = None       # outlier column -> row height
                            if r is not None:
                                h = Vector((_bx, r[1], r[0]))
                            else:
                                if _med is not None:
                                    _az = _etz + _ehh * _med
                                else:
                                    _az = _etz + _ehh * (
                                        0.30 + 0.30 * math.sin(math.pi * f))
                                hh = _fwd(_bx, _az)
                                h  = Vector((_bx, hh.y, _az)) if hh else None
                            if h is not None:
                                h.y += 0.002151                  # nudge inside the skin
                                self._snap(mname, h); moved += 1
                                if _first is None:
                                    _first = h.copy()
                            else:
                                miss.append(mname)
                        if _first is not None:
                            brow_inner[side] = _first
                    elif body_bvh:
                        # Last resort (no brow OR eye mesh): proportional body-bbox arc.
                        b_mn, b_mx, _ = _mesh_bbox_world(props.body_obj)
                        body_h  = b_mx.z - b_mn.z
                        eye_z   = b_mn.z + body_h * 0.905
                        eye_x   = (b_mx.x - b_mn.x) * 0.15 * x_sign
                        eye_r   = 0.014
                        brow_z  = eye_z + eye_r * 1.5
                        for mname, bx_f in zip(names, [
                            eye_x - eye_r * 0.47 * x_sign,
                            eye_x - eye_r * 0.20 * x_sign,
                            eye_x + eye_r * 0.10 * x_sign,
                            eye_x + eye_r * 1.56 * x_sign,
                        ]):
                            h = _fwd(bx_f, brow_z)
                            if h:
                                self._snap(mname, h); moved += 1
                            else:
                                miss.append(mname)
                    else:
                        miss.extend(names)

            # Center brow marker (FACE_BROW, x=0, between inner brows)
            b1L = bpy.data.objects.get('MARKER_FACE_BROW_1_L')
            b1R = bpy.data.objects.get('MARKER_FACE_BROW_1_R')
            if b1L and b1R:
                c = (b1L.location + b1R.location) * 0.5; c.x = 0.0
                self._snap('FACE_BROW', c); moved += 1
            elif 'L' in brow_inner:
                c = brow_inner['L'].copy(); c.x = 0.0
                self._snap('FACE_BROW', c); moved += 1
            else:
                miss.append('FACE_BROW')

            # Eye center: eyeball mesh bbox centre → FACE_EYE_CENTER only
            for side in ('L', 'R'):
                x_sign = 1 if side == 'L' else -1
                eye_mesh = None
                if props.use_eyes:
                    eye_mesh = (props.eye_l_obj if side == 'L' else props.eye_r_obj) \
                               if props.eye_count == 'SPLIT' else props.eye_obj
                if eye_mesh and eye_mesh.type == 'MESH':
                    if props.eye_count == 'COMBINED':
                        # One mesh holds BOTH eyeballs — take THIS eye's own centre,
                        # not the mesh centre (which sits between the eyes). Split by
                        # side (character LEFT = +X), same rule the eyelid/brow code uses.
                        _emw = eye_mesh.matrix_world
                        _ev  = [_emw @ v.co for v in eye_mesh.data.vertices]
                        _sv  = [v for v in _ev if (v.x * x_sign) > 0] or _ev
                        _xs = [v.x for v in _sv]; _ys = [v.y for v in _sv]; _zs = [v.z for v in _sv]
                        cen = Vector(((min(_xs) + max(_xs)) * 0.5,
                                      (min(_ys) + max(_ys)) * 0.5,
                                      (min(_zs) + max(_zs)) * 0.5))
                    else:
                        _, _, cen = _mesh_bbox_world(eye_mesh)
                        cen.x = abs(cen.x) * x_sign
                    self._snap(f'FACE_EYE_CENTER_{side}', cen); moved += 1
                elif body_bvh:
                    b_mn, b_mx, _ = _mesh_bbox_world(props.body_obj)
                    body_h = b_mx.z - b_mn.z
                    eye_z  = b_mn.z + body_h * 0.905
                    eye_x  = (b_mx.x - b_mn.x) * 0.15 * x_sign
                    fwd_hit = _fwd(eye_x, eye_z)
                    if fwd_hit:
                        cen = fwd_hit.copy(); cen.x = eye_x
                        self._snap(f'FACE_EYE_CENTER_{side}', cen); moved += 1
                    else:
                        miss.append(f'FACE_EYE_CENTER_{side}')
                else:
                    miss.append(f'FACE_EYE_CENTER_{side}')

        # ── EYELIDS (requires FACE_EYE_CENTER) ─────────────────────────────────
        if sec in ('ALL', 'EYELIDS'):
            for side in ('L', 'R'):
                x_sign = 1 if side == 'L' else -1
                cen_m = bpy.data.objects.get(f'MARKER_FACE_EYE_CENTER_{side}')
                _lid_names = [
                    f'FACE_EYE_TOP_{side}',    f'FACE_EYE_BOT_{side}',
                    f'FACE_EYE_INNER_{side}',  f'FACE_EYE_OUTER_{side}',
                    f'FACE_LID_CREASE_T_{side}',
                    f'FACE_CREASE_INNER_{side}', f'FACE_CREASE_OUTER_{side}',
                    f'FACE_BROW_BOT_OUTER_{side}',
                ]
                if not cen_m:
                    miss.extend(_lid_names)
                    self.report({'WARNING'},
                        f"Run 'Detect Brows & Eyes' first to place FACE_EYE_CENTER_{side}.")
                    continue
                cen = cen_m.location.copy()

                _max_back = cen.y + 0.050
                _lid_y    = cen.y - 0.005

                _ray_dir   = Vector((0, 1, 0))
                _ray_org_y = _body_mn_y - 1.0

                def _skin_hit(sx, sz, nrm_thr=-0.3):
                    loc, nrm, _, _ = body_bvh.ray_cast(
                        Vector((sx, _ray_org_y, sz)), _ray_dir)
                    if loc is not None and loc.y <= _max_back and nrm.y < nrm_thr:
                        return loc
                    return None

                def _lid_pos(px, pz):
                    h = _skin_hit(px, pz, nrm_thr=-0.15)
                    if h:
                        return h
                    # Plain forward raycast onto the face so the marker TOUCHES the
                    # surface (the normal-restricted hit fails out at the temple).
                    h2 = body_bvh.ray_cast(Vector((px, _ray_org_y, pz)), _ray_dir)[0]
                    return h2 if h2 else Vector((px, _lid_y, pz))

                def _find_rim(dx, dz, step=0.005, steps=50, nrm_thr=-0.3, skip=0):
                    """Scan outward from eye centre; return first skin hit passing nrm_thr.
                    skip: number of steps to ignore near eye centre before accepting hits."""
                    last_x = cen.x + dx * step * skip
                    last_z = cen.z + dz * step * skip
                    for i in range(skip + 1, steps + 1):
                        sx = cen.x + dx * step * i
                        sz = cen.z + dz * step * i
                        h = _skin_hit(sx, sz, nrm_thr)
                        if h:
                            mx = (last_x + sx) * 0.5
                            mz = (last_z + sz) * 0.5
                            return Vector((mx, h.y, mz))
                        last_x, last_z = sx, sz
                    return Vector((cen.x + dx * step * 3, _lid_y, cen.z + dz * step * 3))

                _p_top = _find_rim( 0,  1)
                _max_back = _p_top.y + 0.015

                _p_bot = _find_rim( 0, -1, nrm_thr=-0.1)

                # Eye corners (canthi) = the lateral extremes of the EYEBALL mesh (the
                # eyeball fills the opening and ends at the corners), snapped to the
                # eyelid surface. Far more reliable than the rim-normal sweep.
                _eye_mesh = None
                if props.use_eyes:
                    _eye_mesh = ((props.eye_l_obj if side == 'L' else props.eye_r_obj)
                                 if props.eye_count == 'SPLIT' else props.eye_obj)
                _p_inn = _p_out = None
                if _eye_mesh and _eye_mesh.type == 'MESH':
                    _emw = _eye_mesh.matrix_world
                    _ev  = [_emw @ v.co for v in _eye_mesh.data.vertices]
                    if props.eye_count == 'COMBINED':
                        _ev = [v for v in _ev if (v.x * x_sign) > 0] or _ev
                    # Visible eye opening = eyeball verts on the FRONT of the eyeball AND
                    # in front of the face skin (not covered by lid/nose skin, and not
                    # the deep back of the eyeball). Corners = the lateral extremes of
                    # that exposed set -> the real canthi, for any eye shape.
                    _yf = min(v.y for v in _ev); _yb = max(v.y for v in _ev)
                    _edep = max(_yb - _yf, 1e-4)            # eyeball depth (size scale)
                    _ymid = _yf + 0.55 * (_yb - _yf)        # front-half cutoff
                    _exposed = []
                    _exposed_strict = []                    # clearly in FRONT of the skin
                    for v in _ev:
                        if v.y > _ymid:                     # back of eyeball -> not visible
                            continue
                        _bh = body_bvh.ray_cast(
                            Vector((v.x, _ray_org_y, v.z)), _ray_dir)[0]
                        if _bh is None or v.y <= _bh.y + 0.003:
                            _exposed.append(v)
                        # strict: eyeball ahead of skin by a real margin -> NOT under a lid
                        if _bh is None or v.y < _bh.y - _edep * 0.06:
                            _exposed_strict.append(v)
                    _src = _exposed if len(_exposed) >= 4 else _ev
                    _p_inn = min(_src, key=lambda v: abs(v.x)).copy()
                    _p_out = max(_src, key=lambda v: abs(v.x)).copy()
                    # TOP/BOT from the VERTICAL extremes of the exposed eyeball (the eye
                    # opening top/bottom), centered horizontally over the eye, then
                    # snapped onto the lid skin — same robust logic as INNER/OUTER. The
                    # old rim-ray sweep drifted on stylized/bent eyes. The BOTTOM uses the
                    # STRICT set so a lower lid riding over the eyeball doesn't drag the
                    # marker below the visible lid edge.
                    # Use the eyeball VERTS directly (the eye region is a hole in the
                    # body mesh, so a forward raycast there punches through to the back of
                    # the socket — exactly what INNER/OUTER avoid by using the verts).
                    _src_bot = _exposed_strict if len(_exposed_strict) >= 4 else _src
                    # TOP uses the STRICT set too: the loose _exposed set includes
                    # eyeball verts up to 3mm BEHIND the skin (tucked under the upper
                    # lid), so its z-max sat high under the lid and dragged EYE_TOP up.
                    # The strict set is where the eyeball is clearly IN FRONT of the
                    # skin = the real visible opening edge (the upper-lid margin).
                    _v_top = max(_src_bot, key=lambda v: v.z)
                    _v_bot = min(_src_bot, key=lambda v: v.z)
                    _xc    = (_p_inn.x + _p_out.x) * 0.5
                    _p_top = Vector((_xc, _v_top.y, _v_top.z))
                    _p_bot = Vector((_xc, _v_bot.y, _v_bot.z))
                    # Snap all four eyeball-derived points onto the LID SKIN:
                    # they are eyeball-vertex positions, so without this the
                    # markers sit ON THE EYEBALL, slightly inside the socket.
                    # The nearest body-mesh surface to an eye-opening extreme
                    # is the lid rim itself (a forward ray can't be used here —
                    # the eye hole punches through to the socket back). The
                    # snap cap uses the eyeball's LARGEST dimension: stylized
                    # eyes are tall/wide but SHALLOW, and the old one-depth cap
                    # rejected the snap (lid rim farther than one depth from
                    # the opening extreme) — leaving markers ON the eyeball.
                    _esize = max(
                        _edep,
                        max(v.x for v in _ev) - min(v.x for v in _ev),
                        max(v.z for v in _ev) - min(v.z for v in _ev),
                    )

                    def _to_lid(_pt):
                        _loc, _n2, _i2, _d2 = body_bvh.find_nearest(_pt)
                        if _loc is not None and _d2 <= _esize:
                            return Vector(_loc)
                        return _pt
                    # EYE_TOP = the top of the visible eye OPENING (upper-lid margin).
                    # This is the eyeball's exposed z-max (computed above from the STRICT
                    # set), snapped onto the lid skin — the SAME method as BOT/INNER/OUTER.
                    # A skin scan UP the face was wrong: it walked to the first forward-
                    # facing skin regardless of where the eye opening is, so on deep-set
                    # eyes it climbed onto the lid/brow. Drive it off the eyeball, where
                    # the opening actually is.
                    _p_top = _to_lid(_p_top)
                    _p_bot = _to_lid(_p_bot)
                    _p_inn = _to_lid(_p_inn)
                    _p_out = _to_lid(_p_out)
                # Rim sweep (24 rays) -> inner/outer rim points. This is the PRE-corner-
                # fix detection: used as the corner fallback AND as the reference for the
                # OUTER creases (so they sit exactly where they did before).
                _max_back = cen.y + 0.050
                _rim_pts = []
                for _ai in range(24):
                    _ang       = math.tau * _ai / 24
                    _rdx, _rdz = math.cos(_ang), math.sin(_ang)
                    _rp        = _find_rim(_rdx, _rdz, step=0.004, steps=50,
                                           nrm_thr=-0.1, skip=1)
                    if ((_rp.x - cen.x) ** 2 + (_rp.z - cen.z) ** 2) > 0.000025:
                        _rim_pts.append(_rp)
                if len(_rim_pts) >= 4:
                    _rim_inn = min(_rim_pts, key=lambda p: p.x * x_sign)
                    _rim_out = max(_rim_pts, key=lambda p: p.x * x_sign)
                else:
                    _rim_inn = _find_rim(-x_sign, 0, nrm_thr=-0.15, skip=1)
                    _rim_out = _find_rim( x_sign, 0, nrm_thr=-0.10)
                if _p_inn is None: _p_inn = _rim_inn
                if _p_out is None: _p_out = _rim_out

                _lid_y = min(_p_top.y, _p_bot.y, _p_inn.y, _p_out.y)

                _dx_inn    = _p_inn.x - cen.x
                _dz_top    = _p_top.z - cen.z
                _dz_bot    = _p_bot.z - cen.z
                _bot_mid_x = (_p_inn.x + _p_out.x) * 0.5
                _dx_out_c  = _rim_out.x - cen.x   # outer creases: pre-fix rim reference

                for mname, pos in (
                    (f'FACE_EYE_TOP_{side}',   _p_top),
                    (f'FACE_EYE_BOT_{side}',   Vector((_bot_mid_x, _p_bot.y, _p_bot.z))),
                    (f'FACE_EYE_INNER_{side}', _p_inn),
                    (f'FACE_EYE_OUTER_{side}', _p_out),
                ):
                    if self._snap(mname, pos): moved += 1
                    else: miss.append(mname)

                # Upper-lid crease = the skin fold above the eye. Scanning a vertical
                # column up from the top rim, the lid bulges FORWARD, then the surface
                # RECEDES at the crease (a local depth/Y maximum), then the brow bulges
                # forward again. The crease is that first recession, detected — not
                # placed by a fixed offset.
                def _lid_crease(px, z0):
                    step = max(abs(_dz_top) * 0.12, 0.0015)
                    prof = []
                    for i in range(1, 24):
                        sz = z0 + step * i
                        h = body_bvh.ray_cast(
                            Vector((px, _ray_org_y, sz)), _ray_dir)[0]
                        if h is None:
                            break
                        prof.append((sz, h.y, h))
                    if len(prof) < 3:
                        return None
                    # first local depth-maximum (most recessed) above the lid = fold
                    for j in range(1, len(prof) - 1):
                        if prof[j][1] >= prof[j-1][1] and prof[j][1] > prof[j+1][1]:
                            return prof[j][2].copy()
                    # no clear fold: most recessed point in the lower half of the scan
                    return max(prof[:max(3, len(prof)//2)],
                               key=lambda p: p[1])[2].copy()

                _cr_px = cen.x + _dx_inn*0.27
                _cr    = _lid_crease(_cr_px, _p_top.z)
                if _cr is None:                       # fallback: above the top rim
                    _cr = _lid_pos(_cr_px, cen.z + _dz_top*1.55)
                # The crease must sit ABOVE the EYE_TOP marker (it is the fold above the
                # lid). If detection/fallback landed at or below it, lift it on the skin.
                _cr_min_z = _p_top.z + max(abs(_dz_top) * 0.40, 0.003)
                if _cr.z < _cr_min_z:
                    _cr = _lid_pos(_cr_px, _cr_min_z)
                # ...and BELOW the brow: on lids without a pronounced fold the
                # recession fallback climbs until it hits the brow's own recess.
                # Cap at 65% of the way from the eye top to the brow row (nearest
                # brow marker by X on this side — placed by the brows section).
                _brow_z = None
                _bb_d   = None
                for _bn in (f'FACE_BROW_1_{side}', f'FACE_BROW_2_{side}',
                            f'FACE_BROW_3_{side}', f'FACE_BROW_OUTER_{side}'):
                    _bm = bpy.data.objects.get(f'MARKER_{_bn}')
                    if _bm is not None:
                        _d = abs(_bm.location.x - _cr_px)
                        if _bb_d is None or _d < _bb_d:
                            _bb_d, _brow_z = _d, _bm.location.z
                if _brow_z is not None and _brow_z > _p_top.z:
                    _cr_max_z = _p_top.z + (_brow_z - _p_top.z) * 0.65
                    if _cr.z > _cr_max_z:
                        _cr = _lid_pos(_cr_px, _cr_max_z)
                if self._snap(f'FACE_LID_CREASE_T_{side}', _cr): moved += 1
                else: miss.append(f'FACE_LID_CREASE_T_{side}')

                _xs  = 1 if side == 'L' else -1
                _ew  = max(abs(_p_out.x - _p_inn.x), 1e-4)   # eye width
                _es  = max(abs(_dz_top), 1e-4)               # eye half-height

                # CREASE_INNER sits ON the brow.B.004 bone, which runs from LID_CREASE_T
                # (center of the under-brow line) inward-down to NOSE_BRIDGE. The rigify
                # bone puts CREASE_INNER BETWEEN those two, so interpolate between the two
                # already-placed markers rather than nudging off the eye corner (which
                # floated). Blend toward the nose-bridge side, then SNAP to the surface so
                # it rides the skin of the bridge instead of floating in the valley.
                _nb = bpy.data.objects.get(f'MARKER_FACE_NOSE_BRIDGE_{side}') \
                    or bpy.data.objects.get('MARKER_FACE_NOSE_BRIDGE')
                _lc = bpy.data.objects.get(f'MARKER_FACE_LID_CREASE_T_{side}')
                if _nb is not None and _lc is not None:
                    # 55% of the way from the lid-crease toward the nose bridge
                    _ci_t = _lc.location.lerp(_nb.location, 0.55)
                else:
                    _ci_t = _p_inn + Vector((-_xs * _ew * 0.20, 0.0, -_es * 0.15))
                _cl, _cn, _ci2, _cd = body_bvh.find_nearest(_ci_t)
                _pp = Vector(_cl) if _cl is not None else _ci_t
                if self._snap(f'FACE_CREASE_INNER_{side}', _pp): moved += 1
                else: miss.append(f'FACE_CREASE_INNER_{side}')

                # The OUTER under-brow markers sit on the TEMPLE, lateral to and BEHIND
                # the outer eye corner, heading toward the EAR. A forward (+Y) raycast
                # misses (the temple faces sideways); a LATERAL ray (in along X) hits the
                # side of the head, and sampling FURTHER BACK in Y walks it toward the ear.
                # CREASE_OUTER = the outer end of the brow.B line, just behind EYE_OUTER.
                _cro = _lat(_p_out.y + _ew * 0.40, _p_out.z + _es * 0.10, side)
                if _cro is None:
                    _cro = Vector((_p_out.x + _xs * _ew * 0.5, _p_out.y, _p_out.z))
                if self._snap(f'FACE_CREASE_OUTER_{side}', _cro): moved += 1
                else: miss.append(f'FACE_CREASE_OUTER_{side}')

                # BROW_BOT_OUTER = back toward the EAR (maps to cheek.T, the temple/
                # upper cheek), lateral and slightly up from the outer corner. 0.85
                # eye-widths back (1.50 overshot past the temple toward the ear).
                _bbo = _lat(_p_out.y + _ew * 0.85, _p_out.z + _es * 0.25, side)
                if _bbo is None:
                    _bt = Vector((_p_out.x + _xs * _ew * 0.9,
                                  _p_out.y, _p_out.z + _es * 0.25))
                    _loc, _n, _i, _d = body_bvh.find_nearest(_bt)
                    _bbo = Vector(_loc) if _loc is not None else _bt
                if self._snap(f'FACE_BROW_BOT_OUTER_{side}', _bbo): moved += 1
                else: miss.append(f'FACE_CREASE_OUTER_{side}')

        # ── CHIN / CHEEK / JAW ────────────────────────────────────────────────
        if sec in ('ALL', 'CHIN_CHEEK_JAW'):
            lip_b_m = bpy.data.objects.get('MARKER_FACE_LIP_B')
            corn_L  = bpy.data.objects.get('MARKER_FACE_MOUTH_CORNER_L')
            corn_R  = bpy.data.objects.get('MARKER_FACE_MOUTH_CORNER_R')
            _ccj_miss = ['FACE_CHIN', 'FACE_JAW',
                         'FACE_CHEEK_L', 'FACE_CHEEK_R',
                         'FACE_CHEEK_TOP_L', 'FACE_CHEEK_TOP_R',
                         'FACE_JAW_SIDE_L', 'FACE_JAW_SIDE_R',
                         'FACE_CHIN_SIDE_L', 'FACE_CHIN_SIDE_R']
            if not lip_b_m or body_bvh is None:
                miss.extend(_ccj_miss)
            else:
                lb_z = lip_b_m.location.z

                # Face vertical scale = eye-line to lip-bottom (robust across sizes).
                _eL = bpy.data.objects.get('MARKER_FACE_EYE_CENTER_L')
                _eR = bpy.data.objects.get('MARKER_FACE_EYE_CENTER_R')
                if _eL and _eR:
                    _fv = max((_eL.location.z + _eR.location.z) * 0.5 - lb_z, 0.02)
                else:
                    _fv = 0.10

                # CHIN = the point of the chin. Scan the front profile DOWN from the
                # lip: the face protrudes forward to the chin tip, then recedes back
                # toward the neck. The chin is the lowest still-forward hit before
                # that recession.
                _cstep  = max(_fv * 0.06, 0.003)
                _prev_y = None
                chin_hit = None
                _cprof  = []
                _c_recessed = False
                for k in range(1, 30):
                    z  = lb_z - _cstep * k
                    if z < lb_z - _fv * 0.95:
                        break   # anatomical floor: the chin is never a full
                                # eye->lip span below the lip (smooth chins
                                # that blend into the throat gave the old
                                # unbounded scan a NECK hit)
                    hh = _fwd(0.0, z)
                    if hh is None:
                        break
                    if _prev_y is not None and (hh.y - _prev_y) > _fv * 0.18:
                        _c_recessed = True
                        break                       # receded to neck -> chin captured
                    _cprof.append(hh.copy())
                    _prev_y = hh.y
                if _cprof:
                    if _c_recessed:
                        # Sharp jaw: the recession break ended the profile AT
                        # the chin — the last forward hit (original behaviour,
                        # which was correct on these faces).
                        chin_hit = _cprof[-1].copy()
                    else:
                        # Smooth chin (profile hit the floor, no recession):
                        # prefer the LOWEST local forward bump (the chin boss).
                        # NOT "nearest the forward-most point" — the sub-lip
                        # area is often the most forward thing here, which
                        # dragged the chin up to the lips.
                        _loc = [_cprof[j] for j in range(1, len(_cprof) - 1)
                                if _cprof[j].y <= _cprof[j - 1].y
                                and _cprof[j].y < _cprof[j + 1].y]
                        if _loc:
                            chin_hit = min(_loc, key=lambda h: h.z).copy()
                        else:
                            # Featureless profile: anatomical expectation,
                            # ~half the eye->lip span below the lip.
                            _zt = lb_z - _fv * 0.50
                            chin_hit = min(_cprof,
                                           key=lambda h: abs(h.z - _zt)).copy()
                    chin_hit.x = 0.0
                if chin_hit:
                    if self._snap('FACE_CHIN', chin_hit): moved += 1
                    else: miss.append('FACE_CHIN')
                else:
                    miss.append('FACE_CHIN')

                # JAW = under the chin, heading toward the neck. Below the chin tip the
                # front surface recedes; a forward ray there lands on the underside.
                _chin_z = chin_hit.z if chin_hit else lb_z - _fv * 0.30
                h = _fwd(0.0, _chin_z - _fv * 0.08)
                if h:
                    p = h.copy(); p.x = 0.0
                    if self._snap('FACE_JAW', p): moved += 1
                    else: miss.append('FACE_JAW')
                else:
                    miss.append('FACE_JAW')

                for side, corn in (('L', corn_L), ('R', corn_R)):
                    x_sign = 1 if side == 'L' else -1
                    if not corn:
                        for m in (f'FACE_CHEEK_{side}', f'FACE_CHEEK_TOP_{side}',
                                  f'FACE_JAW_SIDE_{side}', f'FACE_CHIN_SIDE_{side}'):
                            miss.append(m)
                        continue
                    cx2 = corn.location.x
                    cy2 = corn.location.y
                    cz2 = corn.location.z

                    # CHEEK: lateral ray — _fwd(cx*2.0) landed near the nose on
                    # narrow-faced characters; _lat at front-cheek depth is reliable.
                    _cheek_eye_m = bpy.data.objects.get(f'MARKER_FACE_EYE_CENTER_{side}')
                    if _cheek_eye_m:
                        _cheek_z = lb_z + (_cheek_eye_m.location.z - lb_z) * 0.40
                    else:
                        _cheek_z = cz2 + 0.030
                    h = _lat(cy2 + 0.012, _cheek_z, side)
                    if h:
                        if self._snap(f'FACE_CHEEK_{side}', h): moved += 1
                        else: miss.append(f'FACE_CHEEK_{side}')
                    else:
                        miss.append(f'FACE_CHEEK_{side}')

                    cheek_placed = bpy.data.objects.get(f'MARKER_FACE_CHEEK_{side}')
                    if cheek_placed:
                        ct_x = cheek_placed.location.x * 0.80
                        ct_z = cheek_placed.location.z + 0.022
                    else:
                        ct_x = cx2 * 1.8
                        ct_z = cz2 + 0.055
                    # HEIGHT CAP: CHEEK_TOP must stay BELOW the bottom eyelid
                    # (EYE_BOT). Above it the forward ray climbs onto the eye or
                    # brow — or, worse, punches THROUGH the eye-socket hole and
                    # lands at the back of the head. Clamp just under EYE_BOT;
                    # only bites when the proportional offset overshoots, so
                    # characters where it already works keep their height.
                    _eb = bpy.data.objects.get(f'MARKER_FACE_EYE_BOT_{side}')
                    if _eb is not None:
                        _ct_max = _eb.location.z - max(_fv * 0.05, 0.004)
                        if ct_z > _ct_max:
                            ct_z = _ct_max
                    h = _fwd(ct_x, ct_z)
                    # Punch-through guard: a forward ray that entered the eye
                    # opening lands far behind the face. Pin to the CHEEK
                    # marker's depth if the hit is implausibly deep.
                    if (h is not None and cheek_placed is not None
                            and h.y > cheek_placed.location.y + max(_fv * 0.25, 0.010)):
                        h = Vector((ct_x, cheek_placed.location.y, ct_z))
                    if h:
                        if self._snap(f'FACE_CHEEK_TOP_{side}', h): moved += 1
                        else: miss.append(f'FACE_CHEEK_TOP_{side}')
                    else:
                        miss.append(f'FACE_CHEEK_TOP_{side}')

                    # JAW_SIDE = the jaw angle (gonion) at the side of the face, back
                    # toward the ear. Lateral ray well behind the mouth corner, at
                    # jaw height.
                    h = _lat(cy2 + _fv * 0.90, lb_z + _fv * 0.05, side)
                    if h:
                        if self._snap(f'FACE_JAW_SIDE_{side}', h): moved += 1
                        else: miss.append(f'FACE_JAW_SIDE_{side}')
                    else:
                        miss.append(f'FACE_JAW_SIDE_{side}')

                    # CHIN_SIDE: on the jaw line between the chin tip and the
                    # mouth corner. Face-scaled and anchored to the DETECTED
                    # chin — the old ABSOLUTE offsets (x=0.020, z=lip-0.028)
                    # were a neck hit on small/stylized faces. X follows the
                    # mouth corner (face width), Z sits 72% of the way from
                    # the lip down to the chin. A hit that lands well BEHIND
                    # the chin tip is throat -> retry higher, then give up.
                    _cs_x = cx2 * 0.55
                    _cs_ref_z = chin_hit.z if chin_hit else lb_z - _fv * 0.40
                    h = None
                    for _cs_t in (0.72, 0.45):
                        _cs_z = lb_z + (_cs_ref_z - lb_z) * _cs_t
                        _h_try = _fwd(_cs_x, _cs_z)
                        if _h_try is not None and (
                                chin_hit is None
                                or (_h_try.y - chin_hit.y) <= _fv * 0.25):
                            h = _h_try
                            break
                    if h:
                        if self._snap(f'FACE_CHIN_SIDE_{side}', h): moved += 1
                        else: miss.append(f'FACE_CHIN_SIDE_{side}')
                    else:
                        miss.append(f'FACE_CHIN_SIDE_{side}')

                    # EAR — height from the EYE and NOSE. The ear spans eye->nose
                    # vertically, so this brackets it reliably. (Jaw/temple could sit
                    # high and drag the scan up to the top of the head.)
                    _eyec_m  = bpy.data.objects.get(f'MARKER_FACE_EYE_CENTER_{side}')
                    _noseb_m = (bpy.data.objects.get('MARKER_FACE_NOSE_BOT')
                                or bpy.data.objects.get('MARKER_FACE_NOSE_TIP'))
                    if _eyec_m and _noseb_m:
                        ear_z = _eyec_m.location.z * 0.55 + _noseb_m.location.z * 0.45
                    elif _eyec_m:
                        ear_z = _eyec_m.location.z
                    else:
                        jaws_m = bpy.data.objects.get(f'MARKER_FACE_JAW_SIDE_{side}')
                        ear_z  = (jaws_m.location.z + 0.055) if jaws_m else None
                    if ear_z is not None:
                        # Scan front->back along Y at ear height. The head side is flat
                        # (baseline X), then the ear bulges out laterally. The START of
                        # the ear = the FRONT root, where the lateral X first rises above
                        # the baseline (not the most-lateral tip at the back).
                        # Scan only the REAR region (behind the mouth corner / cheek) so
                        # the cheek bulge is excluded — it sits forward of the ear.
                        _ey0 = cy2 + _fv * 0.50
                        _eprof = []
                        for _ky in range(20):
                            _probe_y = _ey0 + _ky * (_fv * 0.13)
                            _h = _lat(_probe_y, ear_z, side)
                            if _h:
                                _eprof.append((abs(_h.x), _h.copy()))
                        ear_hit = None
                        if _eprof:
                            # baseline = MEDIAN X (the flat side of the head); the global
                            # min is the back of the head curving in, which would drag the
                            # threshold down so the cheek triggers first.
                            _xs_s  = sorted(p[0] for p in _eprof)
                            _ns    = len(_xs_s)
                            _ebase = (_xs_s[_ns // 2] if _ns % 2
                                      else (_xs_s[_ns // 2 - 1] + _xs_s[_ns // 2]) * 0.5)
                            _epeak = _xs_s[-1]                    # ear tip (widest)
                            _ethr  = _ebase + (_epeak - _ebase) * 0.35
                            for _px, _hh in _eprof:              # front -> back
                                if _px >= _ethr:
                                    ear_hit = _hh; break
                            if ear_hit is None:
                                ear_hit = max(_eprof, key=lambda p: p[0])[1]
                        if ear_hit is None and props.body_obj and props.body_obj.type == 'MESH':
                            # Fallback: the lateral scan found nothing (unusual head
                            # shape / ear height). Take the widest head vertex near ear
                            # height on this side so the ear always lands ON the mesh.
                            _bo   = props.body_obj
                            _nv   = len(_bo.data.vertices)
                            _flat = np.empty(_nv * 3, dtype=np.float64)
                            _bo.data.vertices.foreach_get('co', _flat)
                            _vco = _flat.reshape(_nv, 3)
                            _mw  = np.array(_bo.matrix_world, dtype=np.float64)
                            _vco = _vco @ _mw[:3, :3].T + _mw[:3, 3]
                            _sel = (_vco[:, 0] * x_sign > 0) & \
                                   (np.abs(_vco[:, 2] - ear_z) < _fv * 0.6)
                            if _sel.any():
                                _cand = _vco[_sel]
                                ear_hit = Vector(_cand[int(np.argmax(np.abs(_cand[:, 0])))].tolist())
                        if ear_hit:
                            if self._snap(f'FACE_EAR_{side}', ear_hit): moved += 1
                            else: miss.append(f'FACE_EAR_{side}')
                        else:
                            miss.append(f'FACE_EAR_{side}')
                    else:
                        miss.append(f'FACE_EAR_{side}')

        # ── FOREHEAD ──────────────────────────────────────────────────────────
        if sec in ('ALL', 'FOREHEAD'):
            if body_bvh is None:
                miss.append('FACE_FOREHEAD')
                for side in ('L', 'R'):
                    for k in ('FACE_FOREHEAD_SIDE', 'FACE_FOREHEAD_SIDE_1',
                              'FACE_FOREHEAD_SIDE_2', 'FACE_FOREHEAD_SIDE_3',
                              'FACE_TEMPLE'):
                        miss.append(f'{k}_{side}')
            else:
                for side in ('L', 'R'):
                    x_sign = 1 if side == 'L' else -1
                    brow_outer_m = bpy.data.objects.get(f'MARKER_FACE_BROW_OUTER_{side}')
                    brow1_m      = bpy.data.objects.get(f'MARKER_FACE_BROW_1_{side}')
                    if not (brow_outer_m and brow1_m):
                        for k in ('FACE_FOREHEAD_SIDE', 'FACE_FOREHEAD_SIDE_1',
                                  'FACE_FOREHEAD_SIDE_2', 'FACE_FOREHEAD_SIDE_3',
                                  'FACE_TEMPLE'):
                            miss.append(f'{k}_{side}')
                        continue
                    bz_base = brow1_m.location.z

                    # Forehead HEIGHT = brow -> crown (head-mesh top), the real vertical
                    # extent. The old code scaled height by brow WIDTH, which made wide
                    # brows push the forehead markers way up. Fall back to brow width only
                    # if the head bbox is unusable.
                    span      = abs(brow_outer_m.location.x - brow1_m.location.x)
                    _head_top = _mesh_bbox_world(props.body_obj)[1].z
                    _fh_h     = _head_top - bz_base
                    if _fh_h < span:                 # bad/short bbox -> fall back
                        _fh_h = span * 3.0

                    # The forehead markers arc above the brows: higher over the inner
                    # brow, dropping toward the temple at the outer side. Z = fraction of
                    # the brow->crown height; X follows each brow point; Y from the skin.
                    brow2_m = bpy.data.objects.get(f'MARKER_FACE_BROW_2_{side}')
                    brow3_m = bpy.data.objects.get(f'MARKER_FACE_BROW_3_{side}')
                    for mname, brow_ref, dz_f in (
                        (f'FACE_FOREHEAD_SIDE_{side}',   brow1_m,      0.45),
                        (f'FACE_FOREHEAD_SIDE_1_{side}', brow2_m,      0.45),
                        (f'FACE_FOREHEAD_SIDE_2_{side}', brow3_m,      0.38),
                        (f'FACE_FOREHEAD_SIDE_3_{side}', brow_outer_m, 0.28),
                    ):
                        ref_x = brow_ref.location.x if brow_ref else None
                        _rx   = ref_x if ref_x is not None else brow_outer_m.location.x
                        _fz   = bz_base + _fh_h * dz_f
                        h = _fwd(_rx, _fz)
                        if h is None:
                            # Forward ray missed (round forehead / target above the
                            # front-facing skin). Snap to the nearest forehead surface.
                            _ref = brow_ref if brow_ref else brow_outer_m
                            _loc, _n, _i, _d = body_bvh.find_nearest(
                                Vector((_rx, _ref.location.y, _fz)))
                            if _loc is not None:
                                h = Vector(_loc)
                        if h:
                            if ref_x is not None:
                                h.x = ref_x
                            if self._snap(mname, h): moved += 1
                            else: miss.append(mname)
                        else:
                            miss.append(mname)

                    # Temple is on the SIDE of the head near the outer brow height.
                    # Lateral ray pushed toward the ear from the brow-outer front depth.
                    temple_z = brow_outer_m.location.z + _fh_h * 0.10
                    temple_y = brow_outer_m.location.y + 0.040
                    h = _lat(temple_y, temple_z, side)
                    if h is None:
                        # Lateral ray missed. Snap to the nearest side-of-head surface,
                        # probing JUST OUTSIDE the outer brow at temple height. Do NOT
                        # probe far out (x=±5): on an A/T-pose character the widest-X
                        # geometry is the HAND, so find_nearest would grab a finger.
                        _probe = Vector((brow_outer_m.location.x + x_sign * max(span, 1e-3),
                                         temple_y, temple_z))
                        _loc, _, _, _ = body_bvh.find_nearest(_probe)
                        if _loc is not None:
                            h = Vector(_loc)
                    if h:
                        if self._snap(f'FACE_TEMPLE_{side}', h): moved += 1
                        else: miss.append(f'FACE_TEMPLE_{side}')
                    else:
                        miss.append(f'FACE_TEMPLE_{side}')

                # Center forehead marker (x=0) — average the two side markers' z
                fh_L = bpy.data.objects.get('MARKER_FACE_FOREHEAD_SIDE_L')
                fh_R = bpy.data.objects.get('MARKER_FACE_FOREHEAD_SIDE_R')
                if fh_L or fh_R:
                    fh_z = ((fh_L.location.z if fh_L else 0.0) +
                            (fh_R.location.z if fh_R else 0.0)) / (2 if (fh_L and fh_R) else 1)
                    h = _fwd(0.0, fh_z)
                    if h is None:
                        _fy = (fh_L or fh_R).location.y
                        _loc, _n, _i, _d = body_bvh.find_nearest(Vector((0.0, _fy, fh_z)))
                        if _loc is not None:
                            h = Vector(_loc)
                    if h:
                        p = h.copy(); p.x = 0.0
                        if self._snap('FACE_FOREHEAD', p): moved += 1
                        else: miss.append('FACE_FOREHEAD')
                    else:
                        miss.append('FACE_FOREHEAD')
                else:
                    miss.append('FACE_FOREHEAD')

        # ── TONGUE ────────────────────────────────────────────────────────────
        if sec in ('ALL', 'TONGUE', 'TEETH_TONGUE') and props.use_tongue and props.tongue_obj:
            # Place each segment at the CENTROID of a thin Y-slice of the tongue
            # vertices -- this rides the tongue's real shape (curve/droop), unlike
            # the bbox centre which can sit in empty space off the mesh.
            _tmw = props.tongue_obj.matrix_world
            _tv  = [_tmw @ v.co for v in props.tongue_obj.data.vertices]
            if _tv:
                _ymn = min(v.y for v in _tv); _ymx = max(v.y for v in _tv)
                _syn = max(_ymx - _ymn, 1e-5)
                for mname, t in (("FACE_TONGUE_1", 0.15),
                                  ("FACE_TONGUE_2", 0.50),
                                  ("FACE_TONGUE_3", 0.85)):
                    _yc   = _ymn + _syn * t
                    _slab = [v for v in _tv if abs(v.y - _yc) < _syn * 0.12]
                    if _slab:
                        _cx = sum(v.x for v in _slab) / len(_slab)
                        _cz = sum(v.z for v in _slab) / len(_slab)
                        pos = Vector((_cx, _yc, _cz))
                    else:
                        pos = Vector((0.0, _yc, (min(v.z for v in _tv) +
                                                 max(v.z for v in _tv)) * 0.5))
                    if self._snap(mname, pos): moved += 1
                    else: miss.append(mname)
            else:
                miss.extend(["FACE_TONGUE_1", "FACE_TONGUE_2", "FACE_TONGUE_3"])

        # ── TEETH ─────────────────────────────────────────────────────────────
        if sec in ('ALL', 'TEETH', 'TEETH_TONGUE') and props.use_teeth:
            if props.teeth_count == 'SPLIT':
                for mname, t_obj in (("FACE_TEETH_T", props.teeth_top_obj),
                                     ("FACE_TEETH_B", props.teeth_bot_obj)):
                    if not t_obj: continue
                    _, _, cen = _mesh_bbox_world(t_obj)
                    cen.x = 0.0
                    if self._snap(mname, cen): moved += 1
                    else: miss.append(mname)
            else:
                if props.teeth_obj:
                    mn, mx, cen = _mesh_bbox_world(props.teeth_obj)
                    z_q = (mx.z - mn.z) * 0.25
                    for mname, z in (("FACE_TEETH_T", cen.z + z_q),
                                     ("FACE_TEETH_B", cen.z - z_q)):
                        if self._snap(mname, Vector((0.0, cen.y, z))): moved += 1
                        else: miss.append(mname)

        # Mirror left→right when left_only was used
        if self.left_only:
            for _base, *_ in FACE_BILATERAL:
                _src = bpy.data.objects.get(f"MARKER_{_base}_L")
                _dst = bpy.data.objects.get(f"MARKER_{_base}_R")
                if _src and _dst:
                    _p = _src.location.copy(); _p.x = -_p.x
                    _dst.location = _p
        else:
            # ── Symmetry: Rigify metarigs expect L/R-symmetric placement, but
            # per-side detection returns slightly different results (meshes are
            # rarely perfectly mirrored). Average every L/R pair about X=0 and
            # pin the midline markers to X=0 — same policy as the body detect's
            # [symmetry] pass.
            for _base, *_ in FACE_BILATERAL:
                _l = bpy.data.objects.get(f"MARKER_{_base}_L")
                _r = bpy.data.objects.get(f"MARKER_{_base}_R")
                if not (_l and _r):
                    continue
                _ax = (abs(_l.location.x) + abs(_r.location.x)) * 0.5
                _ay = (_l.location.y + _r.location.y) * 0.5
                _az = (_l.location.z + _r.location.z) * 0.5
                _l.location = Vector(( _ax, _ay, _az))
                _r.location = Vector((-_ax, _ay, _az))
            for _base, *_ in FACE_SINGLE:
                _m = bpy.data.objects.get(f"MARKER_{_base}")
                if _m is not None:
                    _m.location.x = 0.0

        # Enforce small scale on eyelid markers
        for _base in ('FACE_EYE_CENTER', 'FACE_EYE_TOP', 'FACE_EYE_BOT',
                      'FACE_EYE_INNER', 'FACE_EYE_OUTER',
                      'FACE_LID_CREASE_T',
                      'FACE_CREASE_INNER', 'FACE_CREASE_OUTER',
                      'FACE_BROW_BOT_OUTER'):
            for _sfx in ('_L', '_R'):
                _em = bpy.data.objects.get(f'MARKER_{_base}{_sfx}')
                if _em:
                    _em.scale = (0.2, 0.2, 0.2)

        if miss:
            self.report({'WARNING'},
                f"Moved {moved} marker(s). Not found: {', '.join(miss)}")
        else:
            self.report({'INFO'}, f"Moved {moved} face marker(s) from objects.")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Metarig generation operators
# ---------------------------------------------------------------------------

_FACE_BONE_PREFIXES = (
    "face", "nose", "lip", "lips", "tongue", "teeth", "jaw", "chin",
    "brow", "forehead", "ear", "cheek", "eye", "lid", "temple",
)


def _import_metarig_from_blend(context, blend_name):
    """Append the first armature object from a blend file in armature_presets/."""
    addon_dir  = os.path.dirname(os.path.abspath(__file__))
    blend_path = os.path.join(addon_dir, "armature_presets", blend_name)
    if not os.path.isfile(blend_path):
        return None

    with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
        data_to.objects = list(data_from.objects)

    arm_obj = None
    for obj in data_to.objects:
        if obj is None:
            continue
        context.collection.objects.link(obj)
        if obj.type == 'ARMATURE':
            arm_obj = obj

    if arm_obj:
        bpy.ops.object.select_all(action='DESELECT')
        arm_obj.select_set(True)
        context.view_layer.objects.active = arm_obj

    return arm_obj


def _add_human_metarig(context):
    """Import the full human metarig (with face) from the preset blend file.
    Used by MetarigFace (which strips non-face bones afterwards) and
    AddRigifySample (which keeps it as-is for reference)."""
    return _import_metarig_from_blend(context, "Human_Metarig_with_face.blend")


class AUTORIG_OT_MetarigNoFace(bpy.types.Operator):
    """Add a Rigify Human metarig without face bones (imported from preset)."""
    bl_idname = "autorig.generate_metarig_no_face"
    bl_label = "Human (No Face)"
    bl_description = "Import the Human metarig preset (no face bones)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        arm = _import_metarig_from_blend(context, "Human_Metarig.blend")
        if arm is None:
            self.report({'ERROR'}, "Human_Metarig.blend not found in armature_presets/")
            return {'CANCELLED'}
        arm.name = "metarig"
        self.report({'INFO'}, "Human metarig (no face) imported")
        return {'FINISHED'}


class AUTORIG_OT_MetarigWithFace(bpy.types.Operator):
    """Add a Rigify Human metarig with face bones (imported from preset)."""
    bl_idname = "autorig.generate_metarig_with_face"
    bl_label = "Human (With Face)"
    bl_description = "Import the Human metarig preset (with face bones)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        arm = _import_metarig_from_blend(context, "Human_Metarig_with_face.blend")
        if arm is None:
            self.report({'ERROR'}, "Human_Metarig_with_face.blend not found in armature_presets/")
            return {'CANCELLED'}
        arm.name = "metarig"
        self.report({'INFO'}, "Human metarig (with face) imported")
        return {'FINISHED'}


class AUTORIG_OT_MetarigFace(bpy.types.Operator):
    """Add a face-only Rigify metarig."""
    bl_idname = "autorig.generate_metarig_face"
    bl_label = "Face Metarig"
    bl_description = "Add a face-only Rigify metarig to the scene"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        arm = _add_human_metarig(context)
        if arm is None:
            self.report({'ERROR'}, "Could not add metarig — is Rigify addon enabled?")
            return {'CANCELLED'}
        bpy.ops.object.mode_set(mode='EDIT')
        ebs = arm.data.edit_bones
        keep = _FACE_BONE_PREFIXES + ("head", "spine.006")
        to_remove = [b for b in ebs
                     if not any(b.name.lower().startswith(p) for p in keep)]
        for b in to_remove:
            ebs.remove(b)
        bpy.ops.object.mode_set(mode='OBJECT')
        arm.name = "metarig_face"
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Align Rig
# ---------------------------------------------------------------------------


_SIDE_SUFFIX = [("_L", ".L"), ("_R", ".R")]


class AUTORIG_OT_AlignRig(bpy.types.Operator):
    """Align the active Rigify metarig to all placed markers (body + face)."""
    bl_idname  = "autorig.align_rig"
    bl_label   = "Align Rig to Markers"
    bl_options = {'REGISTER', 'UNDO'}
    apply_roll: bpy.props.BoolProperty(name="Apply Bone Roll", default=True)
    align_face: bpy.props.BoolProperty(name="Align Face Bones", default=True)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        self.layout.prop(self, "apply_roll")
        self.layout.prop(self, "align_face")

    def execute(self, context):
        arm_obj = context.active_object
        if not arm_obj or arm_obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Active object must be an Armature.")
            return {'CANCELLED'}

        inv = arm_obj.matrix_world.inverted()

        # Roll-debug: a non-identity armature transform is the usual cause of
        # "rolls wrong on THIS rig but fine on the default metarig" (roll
        # targets are global axes; edit bones live in armature space).
        _mw_e = arm_obj.matrix_world.to_euler()
        _mw_s = arm_obj.matrix_world.to_scale()
        if (any(abs(a) > 1e-4 for a in _mw_e)
                or any(abs(s - 1.0) > 1e-4 for s in _mw_s)):
            import math as _math
            print(f"[align] armature '{arm_obj.name}' has a non-identity "
                  f"transform: rot=("
                  f"{_math.degrees(_mw_e.x):.1f}, {_math.degrees(_mw_e.y):.1f}, "
                  f"{_math.degrees(_mw_e.z):.1f})°  scale=({_mw_s.x:.3f}, "
                  f"{_mw_s.y:.3f}, {_mw_s.z:.3f}) -- roll targets converted "
                  f"to armature space")

        mpos = {}
        for name, *_ in ALL_MARKERS:
            o = bpy.data.objects.get(f"MARKER_{name}")
            if o:
                mpos[name] = inv @ o.matrix_world.translation

        fpos = {}
        for name, *_ in ALL_FACE_MARKERS:
            o = bpy.data.objects.get(f"MARKER_{name}")
            if o:
                fpos[name] = inv @ o.matrix_world.translation

        if not mpos and not fpos:
            self.report({'ERROR'}, "No markers found.")
            return {'CANCELLED'}

        prev = arm_obj.mode
        bpy.ops.object.mode_set(mode='EDIT')
        ebs = arm_obj.data.edit_bones

        # X-Axis Mirror editing MUST be off while we align: with it saved ON
        # (rig-dependent — the default metarig ships with it off), every
        # head/tail/roll written to a .R bone is auto-mirrored onto its .L
        # counterpart inside Blender's setters, silently overwriting the left
        # hand's already-correct rolls (found via roll tracing: f_index.01.L
        # flipped the moment f_index.01.R was written). We manage both sides
        # explicitly, so implicit mirroring is never wanted here.
        _mirror_x_prev = arm_obj.data.use_mirror_x
        if _mirror_x_prev:
            arm_obj.data.use_mirror_x = False
            print("[align] X-Axis Mirror was ON — disabled during alignment "
                  "(restored after)")

        self._align_body(ebs, mpos, arm_obj.matrix_world.to_3x3())
        if self.align_face:
            self._align_face(ebs, fpos)
        if self.apply_roll:
            self._restore_rolls(ebs)

        # Rigify fails with "zero length vectors have no valid angle" if any
        # bone ends up with head == tail.  Give degenerate bones a safe minimum
        # length and warn the user which ones need their markers repositioned.
        _MIN = 0.001
        bad_bones = []
        for eb in ebs:
            if (eb.tail - eb.head).length < _MIN:
                bad_bones.append(eb.name)
                eb.tail = eb.head + Vector((0, 0, _MIN))
        if bad_bones:
            self.report(
                {'WARNING'},
                "Zero-length bones corrected (check marker positions for): "
                + ", ".join(bad_bones[:10])
                + (" …" if len(bad_bones) > 10 else ""),
            )

        if _mirror_x_prev:
            arm_obj.data.use_mirror_x = True

        bpy.ops.object.mode_set(mode=prev)

        bpy.ops.object.mode_set(mode='POSE')
        for bone_name in FINGER_01_BONES:
            pb = arm_obj.pose.bones.get(bone_name)
            if pb is None:
                continue
            params = getattr(pb, "rigify_parameters", None)
            if params and hasattr(params, "primary_rotation_axis"):
                try:
                    params.primary_rotation_axis = "X"
                except Exception:
                    pass
        bpy.ops.object.mode_set(mode=prev)

        # Markers are intentionally KEPT after alignment so they can be
        # re-used / re-aligned. Use "Delete All Markers" to remove them.
        set_mesh_select(context, True)

        self.report({'INFO'}, "Aligned rig to markers.")
        return {'FINISHED'}

    def _align_body(self, ebs, mpos, arm_rot3=None):
        def mget(name):
            return mpos.get(name)

        # edit_bones live in ARMATURE space, and align_roll() takes its vector
        # in that space too — but our roll targets are GLOBAL axes. Blender's
        # own Recalculate Roll menu converts the global axis through the
        # object's inverse matrix first; without this, a metarig object with
        # any world rotation rolls wrong (manual recalc fixed what ours
        # didn't — that was the tell). Identity rigs: exact no-op.
        _inv3 = arm_rot3.inverted() if arm_rot3 is not None else None

        def _arm_vec(v):
            if _inv3 is None:
                return v
            w = _inv3 @ v
            return w.normalized() if w.length > 1e-9 else v

        # ── PELVIS ────────────────────────────────────────────────────────
        b_spine  = ebs.get("spine")
        b_pelvis = ebs.get("pelvis")
        pelvis_p = mget("PELVIS")
        if pelvis_p:
            sp1 = mget("SPINE_001")
            # Use SPINE_001 marker Z for pelvis.L/R tails (scales with character)
            spine_tail_z = sp1.z if sp1 else (b_spine.tail.z if b_spine else pelvis_p.z)
            if b_pelvis:
                old_len       = max((b_pelvis.tail - b_pelvis.head).length, 0.10)
                b_pelvis.head = pelvis_p.copy()
                b_pelvis.tail = pelvis_p + Vector((0, 0, spine_tail_z - pelvis_p.z)) if sp1 else pelvis_p + Vector((0, 0, old_len))
            for side in ("L", "R"):
                bp = ebs.get(f"pelvis.{side}")
                if bp:
                    _sgn = 1.0 if side == "L" else -1.0
                    # Outward on X: fan the tail toward the hip (THIGH marker) so it
                    # spreads with the character, instead of the fixed metarig X.
                    _th = mget(f"THIGH_{side}")
                    _tx = (abs(_th.x) * 0.70) if _th is not None else abs(bp.tail.x) * 1.5
                    # Lower height: only part-way up toward the spine base, not the full
                    # SPINE_001 height (which sat too high).
                    _tz = pelvis_p.z + (spine_tail_z - pelvis_p.z) * 0.50
                    bp.head = pelvis_p.copy()
                    bp.tail = Vector((_sgn * _tx, bp.tail.y, _tz))
            if b_spine:
                b_spine.head = pelvis_p.copy()
                if sp1:
                    b_spine.tail = sp1.copy()

        # ── BREAST (no marker): capture its transform RELATIVE to the chest bone
        # (spine.003) before the spine moves, so we can re-apply the same relative
        # placement afterwards. There is no breast marker, so it must ride the chest
        # instead of being left at the old metarig position. ──
        _b_chest        = ebs.get("spine.003")
        _chest_old_mat  = _b_chest.matrix.copy() if _b_chest else None
        _breast_old     = {}
        for _bn in ("breast.L", "breast.R"):
            _bb = ebs.get(_bn)
            if _bb:
                _breast_old[_bn] = (_bb.head.copy(), _bb.tail.copy())

        # ── SPINE CHAIN ───────────────────────────────────────────────────
        for mh, bn, mt in [("SPINE_001", "spine.001", "SPINE_002"),
                            ("SPINE_002", "spine.002", "CHEST"),
                            ("CHEST",     "spine.003", "NECK")]:
            p  = mget(mh)
            b  = ebs.get(bn)
            pt = mget(mt)
            if b:
                if p:  b.head = p.copy()
                if pt: b.tail = pt.copy()

        # ── BREAST: re-apply the captured relative placement. The chest bone moved
        # from _chest_old_mat to its new matrix; transform each breast bone by that
        # same delta so it keeps the position/orientation it had between spine.002
        # and spine.003 in the metarig. ──
        if _b_chest is not None and _chest_old_mat is not None and _breast_old:
            _delta = _b_chest.matrix @ _chest_old_mat.inverted()
            for _bn, (_h, _t) in _breast_old.items():
                _bb = ebs.get(_bn)
                if _bb:
                    _bb.head = _delta @ _h
                    _bb.tail = _delta @ _t

        # ── NECK / HEAD ───────────────────────────────────────────────────
        neck_p = mget("NECK")
        head_p = mget("HEAD")
        if neck_p and head_p:
            mid = neck_p.lerp(head_p, 0.5)
            b4  = ebs.get("spine.004")
            if b4: b4.head = neck_p.copy(); b4.tail = mid.copy()
            b5  = ebs.get("spine.005")
            if b5: b5.head = mid.copy();    b5.tail = head_p.copy()
        if head_p:
            b6 = ebs.get("spine.006")
            if b6:
                b6.head = head_p.copy()
                b6.tail = head_p + Vector((0, 0, 0.2))

        # ── SHOULDER ─────────────────────────────────────────────────────
        sp3 = ebs.get("spine.003")
        for side in ("L", "R"):
            sh_p = mget(f"SHOULDER_{side}")
            b    = ebs.get(f"shoulder.{side}")
            if b and sh_p:
                xs = 1.0 if side == "L" else -1.0
                if sp3:
                    b.head = Vector((sp3.tail.x + xs * 0.02,
                                     sp3.tail.y,
                                     sp3.tail.z))
                b.tail = sh_p.copy()

        # ── UPPER ARM + FOREARM + HAND ────────────────────────────────────
        for m_suf, b_suf in _SIDE_SUFFIX:
            arm_p   = mget("ARM"   + m_suf)
            elbow_p = mget("ELBOW" + m_suf)
            hand_p  = mget("HAND"  + m_suf)

            ua = ebs.get("upper_arm" + b_suf)
            if ua:
                if arm_p:   ua.head = arm_p.copy()
                if elbow_p: ua.tail = elbow_p.copy()

            fa = ebs.get("forearm" + b_suf)
            if fa:
                if elbow_p: fa.head = elbow_p.copy()
                if hand_p:  fa.tail = hand_p.copy()

            h_bone = ebs.get("hand" + b_suf)
            if h_bone and hand_p:
                old_z    = h_bone.z_axis.copy()
                knuckles = [mpos.get(f"FINGER_{n}{m_suf}")
                            for n in ("INDEX_1", "MIDDLE_1", "RING_1", "PINKY_1")]
                knuckles = [k for k in knuckles if k is not None]
                h_bone.head = hand_p.copy()
                if knuckles:
                    knuckle_cen = sum(knuckles, Vector()) / len(knuckles)
                    h_bone.tail = hand_p.lerp(knuckle_cen, 0.7)
                else:
                    old_len = (h_bone.tail - h_bone.head).length or 0.05
                    old_dir = (h_bone.tail - h_bone.head).normalized()
                    h_bone.tail = hand_p + old_dir * old_len
                h_bone.align_roll(old_z)

            # ── LEG ───────────────────────────────────────────────────────
            thigh_p = mget("THIGH" + m_suf)
            shin_p  = mget("SHIN"  + m_suf)
            foot_p  = mget("FOOT"  + m_suf)
            toes_p  = mget("TOES"  + m_suf)
            heel_p  = mget("HEEL"  + m_suf)

            th = ebs.get("thigh" + b_suf)
            if th:
                if thigh_p: th.head = thigh_p.copy()
                if shin_p:  th.tail = shin_p.copy()

            sh_b = ebs.get("shin" + b_suf)
            if sh_b:
                if shin_p:  sh_b.head = shin_p.copy()
                if foot_p:  sh_b.tail = foot_p.copy()

            ft = ebs.get("foot" + b_suf)
            if ft:
                if foot_p:  ft.head = foot_p.copy()
                if toes_p:  ft.tail = toes_p.copy()

            toe_b = ebs.get("toe" + b_suf)
            if toe_b and toes_p:
                # Length/direction from the character's own foot markers, NOT the
                # metarig's default length (which overshoots past the mesh on feet of a
                # different scale). TOES is the ball (~25% back from the toe tip), so the
                # toe bone (ball→tip) ≈ 1/3 of HEEL→ball. Direction follows the foot's
                # horizontal forward (handles splayed feet); falls back to -Y.
                if heel_p:
                    _tfwd = (toes_p - heel_p); _tfwd.z = 0.0
                    _tfwd = _tfwd.normalized() if _tfwd.length > 1e-4 else Vector((0, -1, 0))
                    _toe_len = max((toes_p - heel_p).length * 0.33, 0.01)
                elif foot_p:
                    _tfwd = (toes_p - foot_p); _tfwd.z = 0.0
                    _tfwd = _tfwd.normalized() if _tfwd.length > 1e-4 else Vector((0, -1, 0))
                    _toe_len = max((toes_p - foot_p).length * 0.45, 0.01)
                else:
                    _tfwd    = Vector((0, -1, 0))
                    _toe_len = (toe_b.tail - toe_b.head).length or 0.04
                toe_b.head = toes_p.copy()
                toe_b.tail = toes_p + _tfwd * _toe_len

            heel_b = ebs.get("heel.02" + b_suf)
            if heel_b and heel_p:
                old_len    = (heel_b.tail - heel_b.head).length or 0.04
                old_dir    = (heel_b.tail - heel_b.head).normalized()
                heel_b.head = heel_p.copy()
                heel_b.tail = heel_p + old_dir * old_len

        # ── FINGERS ───────────────────────────────────────────────────────
        # Set head/tail from markers, then align roll. RESTORED to the shipped v1.0.0/
        # v1.0.1 convention (verified working in BOTH A-pose and T-pose):
        #   Thumb (both sides) → Global +Y  (0,  1, 0)
        #   Left  non-thumb    → Global -X  (-1, 0, 0)
        #   Right non-thumb    → Global +X  (+1, 0, 0)
        # Mirroring X for the two sides is critical: the L/R finger bones point along
        # -X / +X, so using the same vector on both makes one side's align_roll degenerate
        # (anti-parallel). The RIGHT is the source; the .R -> .L mirror below copies the
        # negated right roll onto the left. (My -Z axis + L->R-mirror rewrite broke both
        # poses — do not reintroduce it.)
        _THUMB_ROLL_VEC = _arm_vec(Vector((0.0, 1.0, 0.0)))
        _FING_ROLL_L    = _arm_vec(Vector((-1.0, 0.0, 0.0)))   # left non-thumb  → Global -X
        _FING_ROLL_R    = _arm_vec(Vector(( 1.0, 0.0, 0.0)))   # right non-thumb → Global +X
        # T-POSE branch: horizontal fingers point along ±X — the SAME axis as
        # the shipped ±X targets, so align_roll degenerates (parallel) and the
        # roll fails exactly on T-posed arms. Horizontal fingers use ∓Z
        # instead. AXES (user spec 2026-07-13): both sides use the SAME
        # targets — non-thumb L → -Z, R → -Z; thumbs L → +Y, R → +Y (no
        # L/R mirror). Hanging fingers (A-pose) keep the
        # shipped ±X / thumb +Y — the -Z-everywhere rewrite broke that pose
        # (see note above); this is per-pose, not a convention change.
        _FING_ROLL_L_T  = _arm_vec(Vector((0.0, 0.0, -1.0)))   # T-pose left  → Global -Z
        _FING_ROLL_R_T  = _arm_vec(Vector((0.0, 0.0, -1.0)))   # T-pose right → Global -Z (user 2026-07-13)
        _THUMB_ROLL_L_T = _arm_vec(Vector((0.0,  1.0, 0.0)))   # T-pose left  thumb → Global +Y
        _THUMB_ROLL_R_T = _arm_vec(Vector((0.0,  1.0, 0.0)))   # T-pose right thumb → Global +Y (user 2026-07-13)
        _horiz_side = {}
        _side_roll  = {}
        for m_suf, b_suf in _SIDE_SUFFIX:
            _mcp = mpos.get("FINGER_MIDDLE_1" + m_suf)
            _mtp = mpos.get("FINGER_MIDDLE_TIP" + m_suf)
            _horiz = False
            if _mcp and _mtp:
                _fd = _mtp - _mcp                     # armature-local delta
                if arm_rot3 is not None:
                    _fd = arm_rot3 @ _fd              # judge in WORLD axes
                _horiz = abs(_fd.x) > abs(_fd.z)
            _horiz_side[b_suf] = _horiz
            if _horiz:
                fing_roll_vec = _FING_ROLL_R_T if b_suf == ".R" else _FING_ROLL_L_T
            else:
                fing_roll_vec = _FING_ROLL_R if b_suf == ".R" else _FING_ROLL_L
                # A-POSE TILT FIX (additive): the global ±X target ignores how
                # the hand actually hangs (arms angle outward, palms rotate
                # with them) — the residual tilt. Follow the hand's OWN palm
                # plane instead: normal = finger_dir × knuckle_row, sign
                # anchored by "back of the hand faces AWAY from the body
                # midline" (robust in A-pose; avoids the thumb-sign trap of a
                # past attempt). GUARD: a tilt correction is small by
                # definition — if the computed target strays >60° from the
                # shipped ±X axis the markers are untrustworthy and the
                # shipped axis stands. Worst case = previous behaviour.
                _i1 = mpos.get("FINGER_INDEX_1" + m_suf)
                _p1 = mpos.get("FINGER_PINKY_1" + m_suf)
                if _i1 and _p1 and _mcp and _mtp:
                    _n = (_mtp - _mcp).cross(_p1 - _i1)
                    if _n.length > 1e-9:
                        _n.normalize()
                        _n_w = (arm_rot3 @ _n) if arm_rot3 is not None else _n
                        if (_n_w.x if b_suf == ".L" else -_n_w.x) < 0:
                            _n = -_n                  # make it dorsal
                        _tgt = -_n                    # Z toward the PALM side
                        _dev = _tgt.angle(fing_roll_vec, 3.15)
                        if _dev < 1.0472:             # 60°
                            fing_roll_vec = _tgt
                        else:
                            # anomaly-only: guard rejected the palm normal
                            print(f"[align{b_suf}] palm normal {math.degrees(_dev):.0f}° "
                                  f"off global X -- keeping shipped axis")
            _side_roll[b_suf] = fing_roll_vec
            _missing_fb = []
            for bone_name, head_m, tail_m in FINGER_BONES_L:
                actual_bone = bone_name.replace(".L", b_suf)
                b  = ebs.get(actual_bone)
                hp = mpos.get(head_m + m_suf)
                tp = mpos.get(tail_m + m_suf)
                if b:
                    if hp: b.head = hp.copy()
                    if tp: b.tail = tp.copy()
                    # Thumb: +Y both sides in A-pose (shipped). In the T-POSE
                    # (horizontal) branch the thumbs are also +Y both sides
                    # (user spec 2026-07-13). Non-thumb fingers use the side's
                    # -Z target.
                    if actual_bone.startswith("thumb."):
                        if _horiz:
                            roll_vec = _THUMB_ROLL_R_T if b_suf == ".R" else _THUMB_ROLL_L_T
                        else:
                            roll_vec = _THUMB_ROLL_VEC
                    else:
                        roll_vec = fing_roll_vec
                    b.align_roll(roll_vec)
                else:
                    # A missed bone keeps its SAVED roll — on a rig whose bones
                    # got .001-renamed (e.g. armatures joined at some point)
                    # this looks like "align roll fails" while the code never
                    # touched the bone at all.
                    _missing_fb.append(actual_bone)
            if _missing_fb:
                print(f"[align{b_suf}] finger bones NOT FOUND (roll untouched): "
                      f"{', '.join(_missing_fb)}")

        # ── FINGER ROLL MIRROR (.R → .L) ──────────────────────────────────
        # RIGHT is the source (aligned to Global +X above); overwrite each .L roll with
        # the negated .R roll — the exact YZ-plane mirror. Cancels marker asymmetry and
        # gives the symmetric roll on both hands. The .R bones are never touched here.
        # SKIPPED on the T-pose branch: the negated-roll mirror maps +X→-X
        # (mirror-consistent with the A-pose convention) but +Z→+Z — it would
        # overwrite the left hand's Global -Z roll with a +Z-equivalent. On
        # T-pose each side is already aligned to its own target directly.
        if not _horiz_side.get(".L"):
            for bone_name, head_m, tail_m in FINGER_BONES_L:
                b_L = ebs.get(bone_name)                    # bone_name ends in .L
                b_R = ebs.get(bone_name.replace(".L", ".R"))
                if b_L and b_R:
                    b_L.roll = -b_R.roll

        # ── PALM BONES ────────────────────────────────────────────────────
        # Each palm head = midpoint between HAND marker and its own knuckle.
        # Palms follow the FINGER roll convention of their side in BOTH poses
        # (user spec 2026-07-08; previously left at the metarig preset, which
        # no longer matches the repositioned bones): T-pose = ∓Z, A-pose =
        # palm-following (or shipped ±X when the guard rejected the normal).
        for m_suf, b_suf in _SIDE_SUFFIX:
            hand_p2 = mpos.get("HAND" + m_suf)
            _palm_roll = _side_roll.get(b_suf)
            for bone_name, tail_m in PALM_BONES_L:
                actual_bone = bone_name.replace(".L", b_suf)
                b  = ebs.get(actual_bone)
                tp = mpos.get(tail_m + m_suf)
                if b:
                    if hand_p2 and tp:
                        b.head = hand_p2.lerp(tp, 0.5)
                    elif hand_p2:
                        b.head = hand_p2.copy()
                    if tp: b.tail = tp.copy()
                    if _palm_roll is not None:
                        b.align_roll(_palm_roll)

    # ---- face alignment ----

    def _align_face(self, ebs, fpos):
        def fget(name):
            return fpos.get(name)

        def set_head(bname, pos):
            b = ebs.get(bname)
            if b and pos:
                old_len = (b.tail - b.head).length or 0.01
                old_dir = (b.tail - b.head).normalized() if old_len > 1e-6 else Vector((0, 0, 1))
                b.head = pos.copy()
                b.tail = pos + old_dir * old_len

        def set_tail(bname, pos):
            b = ebs.get(bname)
            if b and pos:
                b.tail = pos.copy()

        # Step 1: Direct head snaps
        for marker_name, bone_names in FACE_DIRECT_MAP.items():
            p = fget(marker_name)
            if p is None:
                continue
            for bname in bone_names:
                set_head(bname, p)

        # Step 2: Teeth (Y-facing)
        for marker_name, bone_names in FACE_DIRECT_Y_MAP.items():
            p = fget(marker_name)
            if p is None:
                continue
            for bname in bone_names:
                b = ebs.get(bname)
                if b:
                    old_len = (b.tail - b.head).length or 0.02
                    b.head = p.copy()
                    b.tail = p + Vector((0, 1, 0)) * old_len

        # Step 3: Chain map
        for entry in FACE_CHAIN_MAP:
            bones   = entry[0]
            markers = entry[1]
            even    = len(entry) > 2 and entry[2] == "even"
            self._place_face_chain(ebs, bones, markers, fpos, even)

        # Step 3b: Eyelid arcs — re-place each lid chain on a smooth curve through its
        # corner/apex/corner markers so the lid follows the eye's curve (the generic
        # chain placer flattens or mis-bows it). Upper lid: OUTER->TOP->INNER; lower
        # lid: INNER->BOT->OUTER (matches the bone order in FACE_CHAIN_MAP).
        for _side in ("L", "R"):
            self._place_lid_arc(
                ebs,
                [f"lid.T.{_side}", f"lid.T.{_side}.001",
                 f"lid.T.{_side}.002", f"lid.T.{_side}.003"],
                fget(f"FACE_EYE_OUTER_{_side}"), fget(f"FACE_EYE_TOP_{_side}"),
                fget(f"FACE_EYE_INNER_{_side}"))
            self._place_lid_arc(
                ebs,
                [f"lid.B.{_side}", f"lid.B.{_side}.001",
                 f"lid.B.{_side}.002", f"lid.B.{_side}.003"],
                fget(f"FACE_EYE_INNER_{_side}"), fget(f"FACE_EYE_BOT_{_side}"),
                fget(f"FACE_EYE_OUTER_{_side}"))

        # Step 3c: Jaw side — bow jaw.{S}/.001 OUTWARD to follow the jawbone instead of
        # chording straight (the "even" chain placement cuts inside the jaw). Apex =
        # chord midpoint pushed laterally away from the face midline.
        for _side, _osign in (("L", 1.0), ("R", -1.0)):
            _ja = fget(f"FACE_JAW_SIDE_{_side}")
            _jb = fget(f"FACE_CHIN_SIDE_{_side}")
            if _ja is not None and _jb is not None:
                _mid   = (_ja + _jb) * 0.5
                _chord = (_jb - _ja).length
                _apex  = _mid + Vector((_osign * _chord * 0.15, 0.0, 0.0))
                self._place_lid_arc(
                    ebs, [f"jaw.{_side}", f"jaw.{_side}.001"], _ja, _apex, _jb)

        # Step 4: Tail snaps
        for marker_name, bone_names in FACE_TAIL_MAP.items():
            p = fget(marker_name)
            if p is None:
                continue
            for bname in bone_names:
                set_tail(bname, p)

        # Step 5: Aimed-tail snaps
        for marker_name, bone_names in FACE_AIM_TAIL_MAP.items():
            p = fget(marker_name)
            if p is None:
                continue
            for bname in bone_names:
                b = ebs.get(bname)
                if b:
                    direction = p - b.head
                    if direction.length > 1e-6:
                        old_len = (b.tail - b.head).length or 0.01
                        b.tail = b.head + direction.normalized() * old_len

        # Step 6: Bone-to-bone tail snaps
        for src_name, tgt_name in FACE_BONE_TAIL_FROM_BONE_HEAD:
            src = ebs.get(src_name)
            tgt = ebs.get(tgt_name)
            if src and tgt:
                tgt.tail = src.head.copy()

        # Step 7: Rigid translate chains
        # Pre-compute all new positions first to avoid connected-bone cascade issues
        for bone_names, marker_name in FACE_TRANSLATE_CHAINS:
            p = fget(marker_name)
            if p is None:
                continue
            first = ebs.get(bone_names[0])
            if first is None:
                continue
            delta = p - first.head
            new_pos = {}
            for bname in bone_names:
                b = ebs.get(bname)
                if b:
                    new_pos[bname] = (b.head.copy() + delta, b.tail.copy() + delta)
            for bname, (nh, nt) in new_pos.items():
                b = ebs.get(bname)
                if b:
                    b.head = nh
                    b.tail = nt

        # jaw_master: head = midpoint of JAW_SIDE_L/R, tail = FACE_CHIN
        jm = ebs.get("jaw_master")
        if jm:
            jl = ebs.get("jaw.L")
            jr = ebs.get("jaw.R")
            if jl and jr:
                jaw_p = (jl.head + jr.head) * 0.5
            else:
                jl_p = fpos.get("FACE_JAW_SIDE_L")
                jr_p = fpos.get("FACE_JAW_SIDE_R")
                jaw_p = (jl_p + jr_p) * 0.5 if (jl_p and jr_p) else fpos.get("FACE_JAW")
            chin_p = fpos.get("FACE_CHIN")
            if jaw_p:
                jm.head = jaw_p.copy()
                if chin_p:
                    jm.tail = chin_p.copy()
                else:
                    old_len = max((jm.tail - jm.head).length, 0.01)
                    old_dir = (jm.tail - jm.head).normalized()
                    jm.tail = jaw_p + old_dir * old_len

        # nose_master: head = midpoint of nose wing L and R markers
        nm = ebs.get("nose_master")
        if nm:
            nwl = fpos.get("FACE_NOSE_WING_L")
            nwr = fpos.get("FACE_NOSE_WING_R")
            if nwl and nwr:
                nm_p = (nwl + nwr) * 0.5
                old_len = max((nm.tail - nm.head).length, 0.01)
                old_dir = (nm.tail - nm.head).normalized()
                nm.head = nm_p.copy()
                nm.tail = nm_p + old_dir * old_len

        # Step 8: face bone — matches spine.006 head (skull base position)
        face_b = ebs.get("face")
        sp6 = ebs.get("spine.006")
        if face_b and sp6:
            old_len = max((face_b.tail - face_b.head).length, 0.01)
            old_dir = (face_b.tail - face_b.head).normalized()
            face_b.head = sp6.head.copy()
            face_b.tail = face_b.head + old_dir * old_len

        # Step 9: brow.T upper-arc — conditional on custom (.004) vs default rig
        # brow.T.L.003 is the innermost nose-dive bone in BOTH variants (head=BROW_1, tail=NOSE_BRIDGE)
        nose_bridge_p = fpos.get("FACE_NOSE_BRIDGE")
        for sfx, side in ((".L", "_L"), (".R", "_R")):
            bot_outer = fpos.get("FACE_BROW_BOT_OUTER" + side)
            outer     = fpos.get("FACE_BROW_OUTER" + side)
            brow3     = fpos.get("FACE_BROW_3" + side)
            brow2     = fpos.get("FACE_BROW_2" + side)
            brow1     = fpos.get("FACE_BROW_1" + side)

            b000   = ebs.get("brow.T" + sfx)
            b001   = ebs.get("brow.T" + sfx + ".001")
            b002   = ebs.get("brow.T" + sfx + ".002")
            b003   = ebs.get("brow.T" + sfx + ".003")
            b004   = ebs.get("brow.T" + sfx + ".004")
            bglue  = ebs.get("brow_glue.B" + sfx + ".002")

            if b000 and bot_outer and outer:
                b000.head = bot_outer.copy()
                b000.tail = outer.copy()
            if b001 and outer and brow3:
                b001.head = outer.copy()
                b001.tail = brow3.copy()
            if b004:
                # Custom rig: .002=BROW_3→BROW_2, .004=BROW_2→BROW_1
                if b002 and brow3 and brow2:
                    b002.head = brow3.copy()
                    b002.tail = brow2.copy()
                if brow2 and brow1:
                    b004.head = brow2.copy()
                    b004.tail = brow1.copy()
                # brow_glue.B.L.002 tail = BROW_2 when .004 is present
                if bglue and brow2:
                    bglue.tail = brow2.copy()
            else:
                # Default rig (no .004): .002 spans BROW_3→BROW_1
                if b002 and brow3 and brow1:
                    b002.head = brow3.copy()
                    b002.tail = brow1.copy()
                # brow_glue.B.L.002 tail = BROW_3 on default rig
                if bglue and brow3:
                    bglue.tail = brow3.copy()
            # .003 is the nose-dive bone in both variants: head=BROW_1, tail=NOSE_BRIDGE
            if b003 and brow1 and nose_bridge_p:
                b003.head = brow1.copy()
                b003.tail = nose_bridge_p.copy()

        # Step 10: forehead .001/.002/.003 — conditional on custom (.003 exists) vs default
        for sfx, side in ((".L", "_L"), (".R", "_R")):
            outer = fpos.get("FACE_BROW_OUTER" + side)
            brow3 = fpos.get("FACE_BROW_3" + side)
            brow2 = fpos.get("FACE_BROW_2" + side)
            fh1   = fpos.get("FACE_FOREHEAD_SIDE_1" + side)
            fh2   = fpos.get("FACE_FOREHEAD_SIDE_2" + side)
            fh3   = fpos.get("FACE_FOREHEAD_SIDE_3" + side)

            f001 = ebs.get("forehead" + sfx + ".001")
            f002 = ebs.get("forehead" + sfx + ".002")
            f003 = ebs.get("forehead" + sfx + ".003")

            if f003:
                # Custom rig: .001=SIDE1/BROW2, .002=SIDE2/BROW3, .003=SIDE3/OUTER
                if f001:
                    if fh1:   f001.head = fh1.copy()
                    if brow2: f001.tail = brow2.copy()
                if f002:
                    if fh2:   f002.head = fh2.copy()
                    if brow3: f002.tail = brow3.copy()
                if fh3:   f003.head = fh3.copy()
                if outer: f003.tail = outer.copy()
            else:
                # Default rig (no .003): .001=SIDE2/BROW3, .002=SIDE3/OUTER
                if f001:
                    if fh2:   f001.head = fh2.copy()
                    if brow3: f001.tail = brow3.copy()
                if f002:
                    if fh3:   f002.head = fh3.copy()
                    if outer: f002.tail = outer.copy()

        # Step 11: Default rig fallbacks — applied when custom extra bones are absent
        # cheek.B.L.001 tail: use BROW_BOT_OUTER when cheek.B.L.003 doesn't exist
        for sfx, side in ((".L", "_L"), (".R", "_R")):
            if not ebs.get("cheek.B" + sfx + ".003"):
                cb1 = ebs.get("cheek.B" + sfx + ".001")
                bbot = fpos.get("FACE_BROW_BOT_OUTER" + side)
                if cb1 and bbot:
                    cb1.tail = bbot.copy()

    def _place_lid_arc(self, ebs, bone_names, p_a, p_apex, p_b):
        """Place an eyelid bone chain on a smooth arc through corner -> apex -> corner
        (the eye markers), so the lid keeps its CURVE following the character's eye. A
        straight/linear interpolation chords across and flattens it; the metarig arc-
        preserve rotates the metarig's curve onto the new endpoints with an unconstrained
        twist, so its bow can flatten or aim wrong on a differently-shaped eye.

        Quadratic Bezier with the control point chosen so the curve passes EXACTLY
        through the apex at its midpoint (B(0.5) == apex)."""
        if p_a is None or p_apex is None or p_b is None:
            return
        bones = [ebs.get(bn) for bn in bone_names]
        bones = [b for b in bones if b is not None]
        if not bones:
            return
        C = p_apex * 2.0 - (p_a + p_b) * 0.5     # control: B(0.5) == apex

        def _bez(t):
            u = 1.0 - t
            return p_a * (u * u) + C * (2.0 * u * t) + p_b * (t * t)

        n = len(bones)
        joints = [_bez(i / n) for i in range(n + 1)]
        for b, _h, _t in zip(bones, joints, joints[1:]):
            b.head = _h.copy()
            b.tail = _t.copy()

    def _place_face_chain(self, ebs, bone_names, marker_names, fpos, even):
        pts = [fpos.get(m) for m in marker_names]
        pts = [p for p in pts if p is not None]
        if len(pts) < 2:
            return
        bones = [ebs.get(bn) for bn in bone_names]
        bones = [b for b in bones if b is not None]
        if not bones:
            return

        if len(pts) == len(bones) + 1:
            # Exact: every joint explicitly given
            for i, b in enumerate(bones):
                b.head = pts[i].copy()
                b.tail = pts[i + 1].copy()
        else:
            # Use first and last available pts as start/end
            start, end = pts[0], pts[-1]
            n = len(bones)
            if even:
                for i, b in enumerate(bones):
                    b.head = start.lerp(end, i / n)
                    b.tail = start.lerp(end, (i + 1) / n)
            else:
                # Arc-preserve: rotate + scale original chain to new endpoints,
                # then redistribute joints for equal bone lengths along the arc.
                orig_start = bones[0].head.copy()
                orig_end   = bones[-1].tail.copy()
                orig_axis  = orig_end - orig_start
                orig_len   = orig_axis.length
                if orig_len < 1e-6:
                    return
                new_axis = end - start
                scale    = new_axis.length / orig_len
                rot_mat  = orig_axis.normalized().rotation_difference(
                               new_axis.normalized()).to_matrix()
                # Build arc-preserved joint list (n+1 points for n bones)
                arc_pts = [start + rot_mat @ (b.head - orig_start) * scale for b in bones]
                arc_pts.append(start + rot_mat @ (bones[-1].tail - orig_start) * scale)
                # Redistribute joints at equal arc-length intervals
                n = len(bones)
                cum = [0.0]
                for i in range(n):
                    cum.append(cum[-1] + (arc_pts[i + 1] - arc_pts[i]).length)
                total = cum[-1]
                if total < 1e-6:
                    return
                final_pts = [arc_pts[0]]
                for j in range(1, n):
                    s = total * j / n
                    for i in range(n):
                        if cum[i + 1] >= s:
                            t = (s - cum[i]) / max(cum[i + 1] - cum[i], 1e-9)
                            final_pts.append(arc_pts[i].lerp(arc_pts[i + 1], t))
                            break
                final_pts.append(arc_pts[-1])
                for b, nh, nt in zip(bones, final_pts, final_pts[1:]):
                    b.head = nh
                    b.tail = nt

    # ---- roll restoration ----

    def _restore_rolls(self, ebs):
        for eb in ebs:
            rule = get_roll_rule(eb.name)
            if rule is None:
                continue
            align_vec, extra_rad = rule
            eb.align_roll(align_vec)
            eb.roll += extra_rad


# ---------------------------------------------------------------------------
# Detect Face Landmarks
# ---------------------------------------------------------------------------

class AUTORIG_OT_DetectFaceLandmarks(bpy.types.Operator):
    """Auto-detect face landmark positions from the active head mesh using
    BVH raycasting. Place MARKER_HEAD first, then select the head mesh."""
    bl_idname  = "autorig.detect_face_landmarks"
    bl_label   = "Detect Face Landmarks"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        import bmesh
        from mathutils.bvhtree import BVHTree

        head_mobj = bpy.data.objects.get("MARKER_HEAD")
        if not head_mobj:
            self.report({'ERROR'}, "Place MARKER_HEAD first.")
            return {'CANCELLED'}
        mesh_obj = context.active_object
        if not mesh_obj or mesh_obj.type != 'MESH':
            self.report({'ERROR'}, "Select the head mesh first.")
            return {'CANCELLED'}

        depsgraph = context.evaluated_depsgraph_get()
        me_eval   = mesh_obj.evaluated_get(depsgraph)
        tmp_mesh  = me_eval.to_mesh()
        mw        = mesh_obj.matrix_world
        vws       = [mw @ v.co for v in tmp_mesh.vertices]

        bm = bmesh.new()
        bm.from_mesh(tmp_mesh)
        bm.transform(mw)
        bvh = BVHTree.FromBMesh(bm)
        bm.free()
        me_eval.to_mesh_clear()

        head_loc = head_mobj.location.copy()
        cx       = head_loc.x

        head_verts = [v for v in vws if v.z >= head_loc.z - 0.12]
        if not head_verts:
            head_verts = vws

        skull_top_z = max(v.z for v in head_verts)

        face_fwd = [v for v in head_verts
                    if v.y <= head_loc.y + 0.01 and abs(v.x - cx) < 0.08]
        if not face_fwd:
            face_fwd = head_verts

        chin_z = min(v.z for v in face_fwd)
        face_h = skull_top_z - chin_z

        nose_lo   = chin_z + face_h * 0.28
        nose_hi   = chin_z + face_h * 0.48
        nose_cands = [v for v in face_fwd if nose_lo < v.z < nose_hi]
        y_nose_tip = (min(v.y for v in nose_cands)
                      if nose_cands else min(v.y for v in face_fwd))

        ray_y = y_nose_tip - 0.05

        _RY  = Vector((0, 1, 0))
        _RLX = Vector((-1, 0, 0))
        _RRX = Vector((1, 0, 0))

        def fwd(x, z):
            loc, _, _, _ = bvh.ray_cast(Vector((x, ray_y, z)), _RY)
            return loc

        def lat(z, y, s):
            orig = Vector((s * 5.0, y, z))
            d    = _RLX if s > 0 else _RRX
            loc, _, _, _ = bvh.ray_cast(orig, d)
            return loc

        def fz(frac):
            return chin_z + face_h * frac

        ex_f = face_h * 0.131
        out = {}

        def snap(name, loc):
            if loc:
                out[name] = Vector(loc)

        snap("FACE_NOSE_TIP",    fwd(cx, fz(0.371)))
        snap("FACE_NOSE_BRIDGE", fwd(cx, fz(0.420)))
        snap("FACE_NOSE_BOT",    fwd(cx, fz(0.363)))
        snap("FACE_LIP_T",       fwd(cx, fz(0.224)))
        snap("FACE_LIP_B",       fwd(cx, fz(0.184)))
        snap("FACE_LIP_BOT",     fwd(cx, fz(0.192)))
        snap("FACE_JAW",         fwd(cx, fz(0.020)))
        snap("FACE_CHIN",        fwd(cx, fz(0.053)))
        snap("FACE_FOREHEAD",    fwd(cx, fz(0.788)))
        snap("FACE_BROW",        fwd(cx, fz(0.653)))

        lip_t = out.get("FACE_LIP_T")
        lip_b = out.get("FACE_LIP_B")
        teeth_y_t = (lip_t.y + 0.006) if lip_t else (ray_y + 0.15)
        teeth_y_b = (lip_b.y + 0.006) if lip_b else (ray_y + 0.15)
        snap("FACE_TEETH_T", Vector((cx, teeth_y_t, fz(0.204))))
        snap("FACE_TEETH_B", Vector((cx, teeth_y_b, fz(0.171))))

        tong_y = (lip_t.y + face_h * 0.278) if lip_t else (ray_y + face_h * 0.556)
        snap("FACE_TONGUE_1", Vector((cx, tong_y,                   fz(0.151))))
        snap("FACE_TONGUE_2", Vector((cx, tong_y + face_h * 0.069, fz(0.151))))
        snap("FACE_TONGUE_3", Vector((cx, tong_y + face_h * 0.138, fz(0.151))))

        ear_y = y_nose_tip + face_h * 0.625

        for s, sfx in ((+1, "_L"), (-1, "_R")):
            ex = cx + s * ex_f

            snap("FACE_EYE_CENTER" + sfx, fwd(ex,                    fz(0.592)))
            snap("FACE_EYE_TOP"    + sfx, fwd(ex,                    fz(0.637)))
            snap("FACE_EYE_BOT"    + sfx, fwd(ex,                    fz(0.555)))
            snap("FACE_EYE_INNER"  + sfx, fwd(cx + s * ex_f * 0.56, fz(0.592)))
            snap("FACE_EYE_OUTER"  + sfx, fwd(cx + s * ex_f * 1.50, fz(0.592)))

            snap("FACE_BROW_1"     + sfx, fwd(cx + s * ex_f * 0.47, fz(0.653)))
            snap("FACE_BROW_2"     + sfx, fwd(cx + s * ex_f * 0.72, fz(0.661)))
            snap("FACE_BROW_3"     + sfx, fwd(ex,                    fz(0.669)))
            snap("FACE_BROW_OUTER" + sfx, fwd(cx + s * ex_f * 1.56, fz(0.649)))

            snap("FACE_CREASE_INNER"   + sfx, fwd(cx + s * ex_f * 1.44, fz(0.592)))
            snap("FACE_CREASE_OUTER"   + sfx, fwd(cx + s * ex_f * 1.525, fz(0.580)))
            snap("FACE_LID_CREASE_T"   + sfx, fwd(cx + s * ex_f * 0.88, fz(0.649)))
            snap("FACE_BROW_BOT_OUTER" + sfx, fwd(cx + s * ex_f * 1.55, fz(0.567)))

            snap("FACE_CHEEK"     + sfx, fwd(cx + s * ex_f * 1.56, fz(0.420)))
            snap("FACE_CHEEK_TOP" + sfx, fwd(cx + s * ex_f * 1.31, fz(0.480)))

            snap("FACE_NOSE_BRIDGE" + sfx, fwd(cx + s * ex_f * 0.375, fz(0.461)))
            snap("FACE_NOSE_WING"   + sfx, fwd(cx + s * ex_f * 0.500, fz(0.388)))

            snap("FACE_MOUTH_CORNER" + sfx, fwd(cx + s * ex_f * 0.938, fz(0.212)))
            snap("FACE_MOUTH_TOP"    + sfx, fwd(cx + s * ex_f * 0.406, fz(0.237)))
            snap("FACE_MOUTH_BOT"    + sfx, fwd(cx + s * ex_f * 0.406, fz(0.196)))

            snap("FACE_JAW_SIDE"  + sfx, fwd(cx + s * ex_f * 1.56, fz(0.033)))
            snap("FACE_CHIN_SIDE" + sfx, fwd(cx + s * ex_f * 0.56, fz(0.020)))

            snap("FACE_EAR" + sfx, lat(fz(0.490), ear_y, s))

            snap("FACE_TEMPLE" + sfx, fwd(cx + s * ex_f * 1.94, fz(0.763)))

            snap("FACE_FOREHEAD_SIDE"   + sfx, fwd(cx + s * ex_f * 0.94, fz(0.788)))
            snap("FACE_FOREHEAD_SIDE_1" + sfx, fwd(cx + s * ex_f * 0.97, fz(0.755)))
            snap("FACE_FOREHEAD_SIDE_2" + sfx, fwd(cx + s * ex_f * 0.94, fz(0.718)))
            snap("FACE_FOREHEAD_SIDE_3" + sfx, fwd(cx + s * ex_f * 0.88, fz(0.673)))

        moved = 0
        for name, pos in out.items():
            mobj = bpy.data.objects.get("MARKER_" + name)
            if mobj:
                mobj.location = pos
                moved += 1

        self.report({'INFO'}, f"Detected {len(out)} landmarks, moved {moved} markers.")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Generate Rig
# ---------------------------------------------------------------------------

class AUTORIG_OT_GenerateRig(bpy.types.Operator):
    """Generate the final Rigify rig from the aligned metarig."""
    bl_idname = "autorig.generate_rig"
    bl_label = "Generate Rig"
    bl_description = "Run Rigify's generate step on the active metarig"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        arm = _find_metarig(context)
        if arm is None:
            self.report({'ERROR'}, "No metarig found")
            return {'CANCELLED'}

        context.view_layer.objects.active = arm

        # Set rotation_axis on finger .01 bones before generate
        for bname in FINGER_01_BONES:
            pb = arm.pose.bones.get(bname)
            if pb is None:
                continue
            try:
                pb.rigify_parameters.primary_rotation_axis = 'X'
            except AttributeError:
                pass

        try:
            bpy.ops.pose.rigify_generate()
        except Exception as e:
            self.report({'ERROR'}, f"Rigify generate failed: {e}")
            return {'CANCELLED'}

        # Hide the metarig after the rig is generated
        arm.hide_set(True)

        self.report({'INFO'}, "Rig generated")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Add Rigify Sample
# ---------------------------------------------------------------------------

class AUTORIG_OT_AddRigifySample(bpy.types.Operator):
    """Add a Rigify sample metarig (convenience wrapper)."""
    bl_idname = "autorig.add_rigify_sample"
    bl_label = "Add Rigify Sample"
    bl_description = "Add a Rigify sample rig to the scene"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        arm = _add_human_metarig(context)
        if arm is None:
            self.report({'ERROR'}, "Could not add sample — is Rigify enabled?")
            return {'CANCELLED'}
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Tab draw helpers — called from AUTORIG_PT_Main (in __init__.py)
# ---------------------------------------------------------------------------

def draw_rig_tab(layout, context):
    # ── Add Meta Rig ──────────────────────────────────────────────────
    add_box = layout.box()
    add_box.label(text="Add Meta Rig", icon='ARMATURE_DATA')
    col = add_box.column(align=True)
    col.operator("autorig.generate_metarig_no_face",  text="Human (No Face)",  icon='BONE_DATA')
    col.operator("autorig.generate_metarig_with_face", text="Human (With Face)", icon='BONE_DATA')

    # ── Align ─────────────────────────────────────────────────────────
    align_box = layout.box()
    align_box.label(text="Align", icon='BONE_DATA')
    align_box.operator("autorig.align_rig", icon='BONE_DATA')

    # ── Rigify ────────────────────────────────────────────────────────
    rig_box = layout.box()
    rig_box.label(text="Rigify", icon='PLAY')
    rig_box.operator("autorig.generate_rig", icon='PLAY')

    # ── Regenerate Rigify & Custom Preserve ───────────────────────────
    layout.separator()
    box = layout.box()
    box.label(text="Regenerate Rigify & Custom Preserve", icon='DECORATE_LINKED')

    props = context.scene.autorig_preserve_props
    box.prop(props, "metarig",     text="Metarig")
    box.prop(props, "target_rig",  text="Generated Rig")

    row = box.row(align=True)
    row.operator("autorig.preserve_backup",  text="Backup",  icon='FILE_BACKUP')
    row.operator("autorig.preserve_restore", text="Restore", icon='FILE_REFRESH')

    if props.backup_json:
        box.label(text="Backup ready", icon='CHECKMARK')
    else:
        box.label(text="No backup yet", icon='INFO')

    box.operator("autorig.preserve_generate",
                 text="Regenerate & Preserve", icon='PLAY')

    # ── Report a Bug ──────────────────────────────────────────────────
    layout.separator()
    layout.operator("autorig.report_bug", icon='URL')


_panel_cache: dict = {"placed": 0, "onnx": False, "t": 0.0}

# Key markers that must exist for a usable Align Rig to Markers.
_STATUS_KEY_MARKERS = ["PELVIS", "NECK", "HEAD",
                       "SHOULDER_L", "SHOULDER_R", "HAND_L", "HAND_R",
                       "THIGH_L", "THIGH_R", "FOOT_L", "FOOT_R"]


def _compute_marker_status(context):
    """Cheap workflow snapshot for the preflight box: body mesh, missing key markers,
    unapplied transforms, rig state, and the recommended next step."""
    scene = context.scene
    props = getattr(scene, "autorig_face_objs", None)
    body  = getattr(props, "detect_body_obj", None) if props else None
    if body is not None and body.name not in scene.objects:
        body = None      # saved picker referencing an object no longer in the scene
    if body is None or body.type != 'MESH':
        act  = context.active_object
        body = act if (act and act.type == 'MESH' and not act.get("autorig_marker")) else None

    missing = [m for m in _STATUS_KEY_MARKERS if f"MARKER_{m}" not in bpy.data.objects]

    # Unapplied transforms on the body mesh (scale especially poisons binding).
    xform = None
    if body is not None:
        if any(abs(s - 1.0) > 1e-3 for s in body.scale):
            xform = "scale"
        elif any(abs(v) > 1e-3 for v in body.rotation_euler):
            xform = "rotation"

    # Rig state: generated (has DEF- bones) vs metarig vs none.
    rig_state = "none"
    for o in bpy.data.objects:
        if o.type == 'ARMATURE':
            if any(b.name.startswith("DEF-") for b in o.data.bones):
                rig_state = "generated"
                break
            rig_state = "metarig"

    return {"body": body.name if body else None,
            "missing": missing, "xform": xform, "rig_state": rig_state}


def _draw_status_box(layout, context, placed, total):
    """Compact one-line preflight: the recommended next step, plus a single short
    problem summary only when something actually needs attention."""
    st   = _panel_cache.get("status") or {}
    miss = st.get("missing") or []
    rs   = st.get("rig_state", "none")

    problems = []
    if not st.get("body"):
        problems.append("no body mesh")
    if placed == 0:
        problems.append("no markers")
    elif miss:
        problems.append(f"{len(miss)} key marker(s) missing")
    if st.get("xform"):
        problems.append(f"apply {st['xform']}")

    if not st.get("body"):
        nxt = "Pick a Body Mesh"
    elif placed == 0:
        nxt = "Auto Detect Body"
    elif miss:
        nxt = "Place the missing markers"
    elif rs == "none":
        nxt = "Add a Rigify metarig"
    elif rs == "metarig":
        nxt = "Align Rig to Markers → Generate"
    else:
        nxt = "Rig ready — bind in the Skin tab"

    box = layout.box()
    box.row().label(text=f"Next: {nxt}",
                    icon='ERROR' if problems else 'FORWARD')
    if problems:
        p = box.row()
        p.scale_y = 0.7
        p.label(text="Fix: " + " · ".join(problems))


def draw_markers_tab(layout, context):
    import time as _time
    scene = context.scene

    # Recompute expensive values at most 4× per second.
    # Without this, every viewport orbit/move redraws the N-panel and triggers
    # 72 bpy.data.objects lookups + 2 os.path.isfile() calls per frame.
    now = _time.monotonic()
    if now - _panel_cache["t"] > 0.25:
        _panel_cache["placed"] = sum(1 for n, *_ in ALL_MARKERS
                                     if f"MARKER_{n}" in bpy.data.objects)
        _panel_cache["onnx"]   = _ai.is_body_onnx_available()
        _panel_cache["status"] = _compute_marker_status(context)
        _panel_cache["t"]      = now

    placed   = _panel_cache["placed"]
    onnx_ok  = _panel_cache["onnx"]
    total      = len(ALL_MARKERS)
    marker_col = get_marker_col()
    hidden     = bool(marker_col and marker_col.hide_viewport)

    # ── Preflight / status (reads scene state, flags problems, suggests next step) ──
    _draw_status_box(layout, context, placed, total)
    layout.separator()

    # ── ① Body markers ────────────────────────────────────────────────────
    body_box = layout.box()
    body_lbl_icon = get_icon("Body")
    if body_lbl_icon:
        body_box.label(text="Body Markers", icon_value=body_lbl_icon)
    else:
        body_box.label(text="Body Markers", icon='POSE_HLT')

    # Persistent body mesh picker — pick once, used every time
    _bprops = scene.autorig_face_objs
    if _bprops is not None:
        mesh_row = body_box.row(align=True)
        mesh_row.prop(_bprops, "detect_body_obj", text="Body Mesh", icon='MESH_DATA')

    body_col = body_box.column(align=True)
    body_col.scale_y = 1.2
    if onnx_ok:
        body_col.operator("autorig.ai_detect_body", text="✦ EasyDetect Body", icon='SHADERFX')
    body_col.operator("autorig.auto_detect_body", text="① Auto Detect Body", icon='POSE_HLT')

    layout.separator()

    # ── Arm markers (manual / optional — arms are placed by Auto Detect Body) ──
    arm_box  = layout.box()
    sh_icon  = get_icon("Sh")
    if sh_icon:
        arm_box.label(text="Arm Markers  (optional)", icon_value=sh_icon)
    else:
        arm_box.label(text="Arm Markers  (optional)", icon='BONE_DATA')
    arm_col = arm_box.column(align=True)
    arm_col.scale_y = 1.2
    hand_icon = get_icon("Hand")
    if hand_icon:
        arm_col.operator("autorig.place_arm_markers",
                         text="   Place Arm Markers  (3 clicks)", icon_value=hand_icon)
    else:
        arm_col.operator("autorig.place_arm_markers",
                         text="   Place Arm Markers  (3 clicks)", icon='BONE_DATA')

    layout.separator()

    # ── ② Finger markers ──────────────────────────────────────────────────
    fing_box      = layout.box()
    fing_lbl_icon = get_icon("Finger")
    if fing_lbl_icon:
        fing_box.label(text="Finger Markers", icon_value=fing_lbl_icon)
    else:
        fing_box.label(text="Finger Markers", icon='HAND')
    fing_col = fing_box.column(align=True)
    fing_col.scale_y = 1.15

    # Two user-facing choices: EasyDetect (the full pipeline — neural evidence
    # + template + geometric takeover; enum key 'AUTO') and Geometric (explicit
    # mesh-only override with its own sliders). Neural/Template remain in the
    # enum for dev/regression use but are not exposed in the UI.
    if hasattr(scene, "finger_detection_engine"):
        eng_row = fing_col.row(align=True)
        eng_row.prop_enum(scene, "finger_detection_engine", 'AUTO', text="EasyDetect")
        eng_row.prop_enum(scene, "finger_detection_engine", 'GEOMETRIC', text="Geometric")
    _fing_eng = getattr(scene, "finger_detection_engine", 'AUTO')
    if _fing_eng == 'GEOMETRIC':
        # Geometric-only options: the wrist auto-snap moves the HAND marker
        # onto the mesh wrist (the geodesic walk starts there); Auto/template
        # measures from the mesh-wrist estimate without touching the marker.
        if hasattr(scene, "finger_wrist_autosnap"):
            fing_col.prop(scene, "finger_wrist_autosnap")
        for _gp in ("geo_knuckle_depth", "geo_thumb_depth", "geo_min_finger"):
            if hasattr(scene, _gp):
                fing_col.prop(scene, _gp)
    btn_row = fing_col.row(align=True)
    btn_row.scale_y = 1.4
    _fing_btn_text = {'AUTO':      "✦ EasyDetect Fingers",
                      'NEURAL':    "✦ Detect Fingers (Neural)",
                      'GEOMETRIC': "✦ Detect Fingers (Geometric)",
                      'TEMPLATE':  "✦ Detect Fingers (Template)"}
    btn_row.operator("autorig.ai_detect_fingers",
                     text=_fing_btn_text.get(_fing_eng, "✦ EasyDetect Fingers"),
                     icon='SHADERFX')

    # Clear gap so the auto and manual buttons aren't confused for each other.
    fing_col.separator(factor=2.5)

    fing_row = fing_col.row(align=True)
    fing_row.operator("autorig.place_finger_click", text="② Place All Fingers").single_finger = False
    fing_row.popover(panel="AUTORIG_PT_FingerPicker", text="", icon='DOWNARROW_HLT')

    fing_col.operator("autorig.straighten_finger_markers", text="Straighten Finger Markers", icon='MOD_LINEART')
    fing_col.operator("autorig.resolve_finger",
                      text="Re-solve Selected Finger", icon='FILE_REFRESH')

    layout.separator()

    # ── Mirror ────────────────────────────────────────────────────────────
    mir_box = layout.box()
    mir_box.label(text="Mirror", icon='MOD_MIRROR')
    if hasattr(scene, "autorig_live_symmetry"):
        mir_box.prop(scene, "autorig_live_symmetry", toggle=True,
                     icon='MOD_MIRROR',
                     text="Live Symmetry (move one side, the other follows)")
    mir_row = mir_box.row(align=True)
    op_lr = mir_row.operator("autorig.mirror_markers", text="Mirror L→R", icon='MOD_MIRROR')
    op_lr.source_side = "L"   # L→R: source = L, copies onto R
    op_rl = mir_row.operator("autorig.mirror_markers", text="R→L")
    op_rl.source_side = "R"   # R→L: source = R, copies onto L

    layout.separator()

    # ── ⑥ Facial section detectors (step-by-step, mesh pickers first) ─────
    props = scene.autorig_face_objs
    if props:
        fbox = layout.box()
        face_lbl_icon = get_icon("Face")
        if face_lbl_icon:
            fbox.label(text="⑥ Facial Markers", icon_value=face_lbl_icon)
        else:
            fbox.label(text="⑥ Facial Markers", icon='FACE_MAPS')
        frow = fbox.row(align=True)
        frow.prop(props, "show_facial", text="Show Facial Markers", toggle=True, icon='HIDE_OFF')


        if props.show_facial:
            # ── Mesh object pickers (select all meshes first) ──────────────
            mesh_box = fbox.box()
            mesh_box.label(text="Select Meshes", icon='MESH_DATA')
            face_mc = mesh_box.column(align=True)

            face_mc.prop(props, "body_obj", text="Body Mesh")

            # Eyeball
            face_mc.prop(props, "use_eyes", text="Eyeball")
            if props.use_eyes:
                face_mc.row().prop(props, "eye_count", expand=True)
                if props.eye_count == 'SPLIT':
                    face_mc.prop(props, "eye_l_obj", text="Left Eye")
                    face_mc.prop(props, "eye_r_obj", text="Right Eye")
                else:
                    face_mc.prop(props, "eye_obj", text="Eye Mesh")

            # Tongue
            face_mc.prop(props, "use_tongue", text="Tongue")
            if props.use_tongue:
                face_mc.prop(props, "tongue_obj", text="")

            # Teeth
            face_mc.prop(props, "use_teeth", text="Teeth")
            if props.use_teeth:
                face_mc.row().prop(props, "teeth_count", expand=True)
                if props.teeth_count == 'SPLIT':
                    face_mc.prop(props, "teeth_top_obj", text="Top")
                    face_mc.prop(props, "teeth_bot_obj", text="Bottom")
                else:
                    face_mc.prop(props, "teeth_obj", text="")

            # Eyebrow
            face_mc.prop(props, "use_brows", text="Eyebrow")
            if props.use_brows:
                face_mc.row().prop(props, "brow_count", expand=True)
                if props.brow_count == 'SPLIT':
                    face_mc.prop(props, "brow_l_obj", text="Left Brow")
                    face_mc.prop(props, "brow_r_obj", text="Right Brow")
                else:
                    face_mc.prop(props, "brow_obj", text="Brow Mesh")

            fbox.separator()

            # ── EasyDetect Face (neural — shown once the face model is installed) ──
            if _ai.is_available() and _ai.is_face_onnx_available():
                ai_fbox = fbox.box()
                ai_fbox.label(text="EasyDetect — set Body Mesh above")
                ai_fbox.operator("autorig.ai_detect_face",
                                 text="✦ EasyDetect Face", icon='SHADERFX')
                fbox.separator()

            # ── Detect steps ───────────────────────────────────────────────
            s1 = fbox.box()
            s1.label(text="Step 1 — Teeth & Tongue")
            op_tt = s1.operator("autorig.detect_face_objects",
                                text="Detect Teeth & Tongue", icon='VIEWZOOM')
            op_tt.section   = 'TEETH_TONGUE'
            op_tt.left_only = False

            s2 = fbox.box()
            s2.label(text="Step 2 — Brow & Eyes")
            op_be = s2.operator("autorig.detect_face_objects",
                                text="Detect Brows & Eyes", icon='VIEWZOOM')
            op_be.section   = 'EYEBROWS_EYE'
            op_be.left_only = False

            for step, section, label in (
                (3, 'LIPS',           "Lips"),
                (4, 'NOSE',           "Nose"),
                (5, 'EYELIDS',        "Eyelids"),
                (6, 'CHIN_CHEEK_JAW', "Chin / Cheek / Jaw"),
                (7, 'FOREHEAD',       "Forehead"),
            ):
                sx = fbox.box()
                sx.label(text=f"Step {step} — {label}")
                op_s = sx.operator("autorig.detect_face_objects",
                                   text=f"Detect {label}", icon='VIEWZOOM')
                op_s.section   = section
                op_s.left_only = False

    # ── Check Markers ─────────────────────────────────────────────────────
    layout.separator()
    chk_box = layout.box()
    chk_box.label(text="Diagnostics", icon='VIEWZOOM')
    chk_box.operator("autorig.check_markers", text="Check All Markers", icon='CHECKMARK')

    # ── Delete All Markers ────────────────────────────────────────────────
    layout.separator()
    del_box = layout.box()
    del_box.operator("autorig.delete_all_markers", text="Delete All Markers", icon='TRASH')

    layout.separator()
    layout.label(text=f"{placed}/{total} body markers placed",
                 icon='CHECKMARK' if placed == total else 'INFO')

    # ── Viewport ──────────────────────────────────────────────────────────
    view_box = layout.box()
    view_box.label(text="Viewport", icon='HIDE_OFF')
    row = view_box.row(align=True)
    row.operator("autorig.toggle_markers",
                 text="Show Markers" if hidden else "Hide Markers",
                 icon='HIDE_ON' if hidden else 'HIDE_OFF')
    row.operator("autorig.toggle_xray", icon='XRAY')
    row.operator("autorig.toggle_mesh_selection", icon='MESH_DATA')

    if hasattr(scene, "autorig_show_hints"):
        view_box.prop(scene, "autorig_show_hints", toggle=True, icon='INFO')

    if hasattr(scene, "autorig_marker_scale"):
        row = view_box.row(align=True)
        row.prop(scene, "autorig_marker_scale", text="Marker Size")
        row.operator("autorig.rescale_markers", text="", icon='DRIVER_TRANSFORM')


def _rescale_all_markers(scale):
    """Apply scale to every marker empty in both marker collections."""
    from .constants import BODY_SIZE, FINGER_SIZE, FACE_SIZE
    for col_name in ("RigifyMarkers", "RigifyFaceMarkers"):
        col = bpy.data.collections.get(col_name)
        if not col:
            continue
        for obj in col.objects:
            if not (obj.name.startswith('MARKER_') and obj.type == 'EMPTY'):
                continue
            raw  = obj.name[len('MARKER_'):]
            base = raw[:-2] if raw.endswith(('_L', '_R')) else raw
            if base.startswith(('THUMB', 'FINGER_')):
                base_size = FINGER_SIZE
            elif base.startswith('FACE_'):
                base_size = FACE_SIZE
            else:
                base_size = BODY_SIZE
            obj.empty_display_size = base_size * scale


def marker_scale_update(scene, _context):
    """Property update callback — fires whenever the Marker Size slider changes."""
    _rescale_all_markers(scene.autorig_marker_scale)


class AUTORIG_OT_RescaleMarkers(bpy.types.Operator):
    """Resize all placed marker empties to match the current Marker Size setting"""
    bl_idname  = "autorig.rescale_markers"
    bl_label   = "Rescale Markers"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scale = getattr(context.scene, 'autorig_marker_scale', 1.0)
        _rescale_all_markers(scale)
        self.report({'INFO'}, f"Rescaled markers (×{scale:.2f})")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

class AUTORIG_PT_MetaRig(bpy.types.Panel):
    """Metarig generation — content served via AUTORIG_PT_Main Rig tab."""
    bl_label       = "Metarig"
    bl_idname      = "AUTORIG_PT_MetaRig"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = "Easy Rigify"
    bl_options     = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context): return False

    def draw(self, context):
        draw_rig_tab(self.layout, context)


class AUTORIG_PT_Markers(bpy.types.Panel):
    """Marker placement — content served via AUTORIG_PT_Main Markers tab."""
    bl_label       = "Markers"
    bl_idname      = "AUTORIG_PT_Markers"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = "Easy Rigify"
    bl_options     = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context): return False

    def draw(self, context):
        draw_markers_tab(self.layout, context)


# ---------------------------------------------------------------------------
# Update Check
# ---------------------------------------------------------------------------

# ── CONFIGURE THIS ──────────────────────────────────────────────────────────
# Host a version.json file at this URL (GitHub raw content is ideal).
# File format:
#   {
#       "version": [2, 1, 0],
#       "download_url": "https://your-download-or-release-page-url",
#       "changelog": "Brief description of what changed"
#   }
# Update this JSON whenever you release a new version so users are notified.
_UPDATE_MANIFEST_URL = (
    "https://raw.githubusercontent.com/dwaynejnr-alt/EasyRigifyRepo/main/version.json"
)

_BUG_REPORT_EMAIL = "dwayne.jnr@gmail.com"
# ────────────────────────────────────────────────────────────────────────────


class AUTORIG_OT_CheckMarkers(bpy.types.Operator):
    """Check all placed markers for missing or overlapping positions"""
    bl_idname = "autorig.check_markers"
    bl_label  = "Check All Markers"

    def execute(self, context):
        body_col = bpy.data.collections.get("RigifyMarkers")
        face_col = bpy.data.collections.get("RigifyFaceMarkers")

        if not body_col and not face_col:
            self.report({'WARNING'}, "No markers found — place markers first.")
            return {'FINISHED'}

        issues = []

        # ── 1. Missing markers ────────────────────────────────────────────
        if body_col:
            for name, *_ in ALL_MARKERS:
                if bpy.data.objects.get(f"MARKER_{name}") is None:
                    issues.append(('ERROR', f"MARKER_{name} is missing — re-run Auto Detect or re-place manually."))

        if face_col:
            for name, *_ in ALL_FACE_MARKERS:
                if bpy.data.objects.get(f"MARKER_{name}") is None:
                    issues.append(('ERROR', f"MARKER_{name} is missing — re-run Detect Face Landmarks."))

        # ── 2. Coincident (too close) markers ─────────────────────────────
        existing = {}
        for col_name in ("RigifyMarkers", "RigifyFaceMarkers"):
            col = bpy.data.collections.get(col_name)
            if col:
                for obj in col.objects:
                    if obj.name.startswith("MARKER_"):
                        existing[obj.name] = obj.matrix_world.translation.copy()

        COIN_THR = 0.005
        names_list = list(existing.keys())
        checked_pairs = set()
        for i, n1 in enumerate(names_list):
            for n2 in names_list[i + 1:]:
                if (n1, n2) in checked_pairs:
                    continue
                checked_pairs.add((n1, n2))
                dist = (existing[n1] - existing[n2]).length
                if dist < COIN_THR:
                    issues.append((
                        'WARNING',
                        f"{n1} and {n2} are {dist * 1000:.1f} mm apart — move them further apart."
                    ))

        # ── Report ────────────────────────────────────────────────────────
        if not issues:
            self.report({'INFO'}, "All markers OK — no issues found.")
        else:
            for level, msg in issues:
                self.report({level}, msg)
            self.report({'WARNING'}, f"{len(issues)} issue(s) found — see Info bar for details.")

        return {'FINISHED'}


class AUTORIG_OT_CheckUpdate(bpy.types.Operator):
    """Fetch the latest Easy Rigify version info from the web and compare with your installed version"""
    bl_idname = "autorig.check_update"
    bl_label  = "Check for Update"

    def execute(self, context):
        import urllib.request
        import json as _json
        import sys   as _sys

        prefs = context.preferences.addons[__package__].preferences

        pkg_module = _sys.modules.get(__package__)
        local_ver  = getattr(pkg_module, '_ADDON_VERSION', (0, 0, 0))

        try:
            req = urllib.request.Request(
                _UPDATE_MANIFEST_URL,
                headers={"User-Agent": "EasyRigify-UpdateCheck/1.0"},
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = _json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            prefs.update_status       = f"error: Could not reach server ({e})"
            prefs.update_latest_ver   = ""
            prefs.update_download_url = ""
            self.report({'WARNING'}, f"Update check failed: {e}")
            return {'FINISHED'}

        raw_ver    = data.get("version", [0, 0, 0])
        if isinstance(raw_ver, str):
            raw_ver = raw_ver.split(".")
        remote_ver = tuple(int(x) for x in raw_ver)
        changelog  = data.get("changelog", "")
        dl_url     = data.get("download_url", "")

        prefs.update_latest_ver   = ".".join(str(x) for x in remote_ver)
        prefs.update_download_url = dl_url

        if remote_ver > local_ver:
            msg = f"v{prefs.update_latest_ver} is available"
            if changelog:
                msg += f" — {changelog}"
            prefs.update_status = f"update: {msg}"
            self.report({'INFO'}, f"Easy Rigify update available: v{prefs.update_latest_ver}")
        elif remote_ver == local_ver:
            prefs.update_status = "ok: You are up to date."
            self.report({'INFO'}, "Easy Rigify is up to date.")
        else:
            prefs.update_status = (
                f"ok: Installed v{'.'.join(str(x) for x in local_ver)} "
                f"is newer than remote v{prefs.update_latest_ver}."
            )
            self.report({'INFO'}, "Your installed version is ahead of the remote.")

        return {'FINISHED'}


class AUTORIG_OT_OpenUpdateURL(bpy.types.Operator):
    """Open the Easy Rigify download / release page in your web browser"""
    bl_idname = "autorig.open_update_url"
    bl_label  = "Download Update"

    def execute(self, context):
        import webbrowser
        prefs = context.preferences.addons[__package__].preferences
        url   = prefs.update_download_url or _UPDATE_MANIFEST_URL
        webbrowser.open(url)
        return {'FINISHED'}


class AUTORIG_OT_ReportBug(bpy.types.Operator):
    """Open your email client with a pre-filled bug report"""
    bl_idname = "autorig.report_bug"
    bl_label  = "Report a Bug"

    def execute(self, context):
        import sys as _sys
        import platform
        import webbrowser
        import urllib.parse

        pkg       = _sys.modules.get(__package__)
        local_ver = getattr(pkg, '_ADDON_VERSION', (0, 0, 0))
        ver_str   = ".".join(str(x) for x in local_ver)

        gpu_info = "unknown"
        try:
            import gpu
            gpu_info = gpu.platform.renderer_get()
        except Exception:
            pass

        subject = f"Easy Rigify Bug Report - v{ver_str}"

        body = "\n".join([
            "=== Environment ===",
            f"Easy Rigify version: {ver_str}",
            f"Blender: {bpy.app.version_string}",
            f"OS: {platform.system()} {platform.release()} ({platform.machine()})",
            f"Python: {_sys.version.split()[0]}",
            f"GPU: {gpu_info}",
            "",
            "=== Steps to Reproduce ===",
            "1. ",
            "2. ",
            "",
            "=== Expected Behaviour ===",
            "",
            "=== Actual Behaviour ===",
            "",
            "=== Error Messages (Window > Toggle System Console) ===",
            "",
        ])

        mailto = (
            f"mailto:{_BUG_REPORT_EMAIL}"
            f"?subject={urllib.parse.quote(subject)}"
            f"&body={urllib.parse.quote(body)}"
        )
        webbrowser.open(mailto)
        self.report({'INFO'}, "Opening your email client with a pre-filled bug report.")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Addon Preferences
# ---------------------------------------------------------------------------

class AUTORIG_Prefs(bpy.types.AddonPreferences):
    bl_idname = __package__

    show_advanced: bpy.props.BoolProperty(name="Show Advanced Options", default=False)
    marker_scale:  bpy.props.FloatProperty(
        name="Marker Scale",
        description="Global scale multiplier for marker empties",
        default=1.0, min=0.1, max=5.0,
    )

    # ── Update-check state (session-only — not saved to disk) ────────────────
    update_status:       bpy.props.StringProperty(default="")
    update_latest_ver:   bpy.props.StringProperty(default="")
    update_download_url: bpy.props.StringProperty(default="")

    def draw(self, context):
        import sys as _sys
        layout    = self.layout
        pkg       = _sys.modules.get(__package__)
        local_ver = getattr(pkg, '_ADDON_VERSION', (0, 0, 0))
        local_str = ".".join(str(x) for x in local_ver)

        # ── Version / update section ─────────────────────────────────────────
        box = layout.box()
        row = box.row(align=True)
        row.label(text=f"Easy Rigify  v{local_str}", icon='INFO')
        row.operator("autorig.check_update", text="Check for Update", icon='FILE_REFRESH')

        if self.update_status:
            if self.update_status.startswith("update:"):
                # New version available — highlight in red
                msg_row = box.row()
                msg_row.alert = True
                msg_row.label(
                    text=self.update_status[len("update:"):].strip(),
                    icon='ERROR',
                )
                if self.update_download_url:
                    box.operator(
                        "autorig.open_update_url",
                        text=f"Download v{self.update_latest_ver}",
                        icon='URL',
                    )
            elif self.update_status.startswith("error:"):
                box.label(
                    text=self.update_status[len("error:"):].strip(),
                    icon='ERROR',
                )
            else:
                # "ok:" prefix — all good
                box.label(
                    text=self.update_status[len("ok:"):].strip(),
                    icon='CHECKMARK',
                )

        # ── AI dependencies ──────────────────────────────────────────────────
        # onnxruntime + Pillow ship as bundled wheels — Blender installs the
        # matching one automatically when the addon is enabled. Nothing for
        # the user to install/uninstall; this is a status line only.
        layout.separator()
        ai_box = layout.box()
        ai_box.label(text="EasyDetect (ONNX Runtime)", icon='SHADERFX')
        if _ai.is_available():
            ai_box.label(text="onnxruntime + Pillow: Ready", icon='CHECKMARK')
        else:
            ai_box.label(text="onnxruntime + Pillow: not available on this",
                         icon='ERROR')
            ai_box.label(text="platform/Blender version — using the")
            ai_box.label(text="geometric engines instead.")

        # ── Other preferences ────────────────────────────────────────────────
        layout.separator()
        layout.prop(self, "show_advanced")
        if self.show_advanced:
            layout.prop(self, "marker_scale")
        layout.label(text="Easy Rigify — Marker-based Rigify alignment tool")
