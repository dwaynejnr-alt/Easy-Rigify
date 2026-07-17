# fingers.py — Finger marker placement, hand detection, and finger picker panel.
import bpy
import numpy as np
from mathutils import Vector

from .constants import FINGER_SIZE
from .utils import get_or_create_collection

_NAV_EVENTS = frozenset({
    'MOUSEMOVE',
    'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE',
    'WHEELINMOUSE', 'WHEELOUTMOUSE',
    'NUMPAD_0', 'NUMPAD_1', 'NUMPAD_2', 'NUMPAD_3', 'NUMPAD_4',
    'NUMPAD_5', 'NUMPAD_6', 'NUMPAD_7', 'NUMPAD_8', 'NUMPAD_9',
    'NUMPAD_ENTER', 'NUMPAD_PERIOD', 'NUMPAD_PLUS', 'NUMPAD_MINUS',
    'TRACKPADPAN', 'TRACKPADZOOM',
})

def _bvh_from_mesh(mesh_obj):
    """Build a world-space BVH tree from mesh_obj."""
    import bmesh as _bmesh
    from mathutils.bvhtree import BVHTree
    bm = _bmesh.new()
    bm.from_mesh(mesh_obj.data)
    bm.transform(mesh_obj.matrix_world)
    bvh = BVHTree.FromBMesh(bm)
    bm.free()
    return bvh

def _interior_center_normal(bvh, surface_pt, inward_dir, max_dist=0.15):
    """Find the interior centreline of a mesh at a *surface* point.

    Casts along inward_dir to the far wall and returns the midpoint.
    Works for any mesh orientation — makes no global-axis assumption.
    Falls back to surface_pt when no far wall is found (open edge / thin mesh)."""
    origin = surface_pt + inward_dir * 0.0005   # nudge just past the surface
    hit, _, _, _ = bvh.ray_cast(origin, inward_dir, max_dist)
    if not hit:
        return surface_pt
    return (surface_pt + hit) * 0.5


def _interior_center_axis(bvh, point, axis_dir, max_dist=0.15):
    """Find the interior centreline at a non-surface point by casting
    perpendicular to axis_dir in both directions and returning the midpoint.
    Falls back to point when the mesh cannot be straddled."""
    world_up = Vector((0, 0, 1))
    if abs(axis_dir.dot(world_up)) > 0.9:
        world_up = Vector((0, 1, 0))
    perp = axis_dir.cross(world_up).normalized()
    h1, _, _, _ = bvh.ray_cast(point,  perp, max_dist)
    h2, _, _, _ = bvh.ray_cast(point, -perp, max_dist)
    if h1 and h2:
        return (h1 + h2) * 0.5
    # Try the orthogonal perpendicular
    perp2 = axis_dir.cross(perp).normalized()
    h3, _, _, _ = bvh.ray_cast(point,  perp2, max_dist)
    h4, _, _, _ = bvh.ray_cast(point, -perp2, max_dist)
    if h3 and h4:
        return (h3 + h4) * 0.5
    return point

def _snap_to_surface(context, event, mesh_obj):
    """Cast a viewport ray onto mesh_obj.
    Returns (world_location, world_outward_normal), or (None, None) on miss."""
    region = context.region
    rv3d   = context.region_data
    coord  = event.mouse_region_x, event.mouse_region_y
    from bpy_extras import view3d_utils
    ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
    ray_dir    = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
    mw_inv = mesh_obj.matrix_world.inverted()
    lo     = mw_inv @ ray_origin
    ld     = (mw_inv.to_3x3() @ ray_dir).normalized()
    ok, loc, normal, _ = mesh_obj.ray_cast(lo, ld)
    if not ok:
        return None, None
    mw           = mesh_obj.matrix_world
    world_loc    = mw @ loc
    world_normal = (mw.to_3x3().inverted().transposed() @ normal).normalized()
    return world_loc, world_normal


def _resolve_body_mesh(context):
    """Return the body mesh to use for marker placement, in priority order:
    1. Persistent picker stored in scene.autorig_face_objs.detect_body_obj
    2. Selected non-marker mesh
    3. Active non-marker mesh
    Returns None if nothing found."""
    face_props = getattr(context.scene, 'autorig_face_objs', None)
    if face_props and face_props.detect_body_obj \
            and face_props.detect_body_obj.type == 'MESH' \
            and face_props.detect_body_obj.name in context.scene.objects:
        return face_props.detect_body_obj
    for obj in context.selected_objects:
        if obj.type == 'MESH' and not obj.get("autorig_marker"):
            return obj
    ao = context.active_object
    if ao and ao.type == 'MESH' and not ao.get("autorig_marker"):
        return ao
    return None


class AUTORIG_OT_PlaceFingerMarkers(bpy.types.Operator):
    """Place finger markers with 2 clicks per finger: first the fingertip, then the knuckle.
    Intermediate joints (_2, _3) are evenly distributed between knuckle and tip.
    Both sides (L and R) are placed automatically."""
    bl_idname  = "autorig.place_finger_click"
    bl_label   = "Place Fingers (Click)"
    bl_options = {'REGISTER', 'UNDO'}

    finger_name: bpy.props.EnumProperty(
        name="Finger",
        items=[
            ("THUMB",         "Thumb",  ""),
            ("FINGER_INDEX",  "Index",  ""),
            ("FINGER_MIDDLE", "Middle", ""),
            ("FINGER_RING",   "Ring",   ""),
            ("FINGER_PINKY",  "Pinky",  ""),
        ],
        default="THUMB"
    )
    single_finger: bpy.props.BoolProperty(
        name="Single Finger",
        default=False
    )

    FINGERS = [
        ("THUMB",        "Thumb"),
        ("FINGER_INDEX", "Index"),
        ("FINGER_MIDDLE","Middle"),
        ("FINGER_RING",  "Ring"),
        ("FINGER_PINKY", "Pinky"),
    ]

    def _place_one_finger(self, fname, tip, tip_normal, knuckle, knuckle_normal):
        """Place all 4 markers for one finger on both sides.

        Tip and knuckle are inset 3mm along their surface normals so markers
        sit inside the mesh.  Intermediate joints use anatomical phalanx
        proportions: proximal 45%, middle 30%, distal 25% of finger length."""
        col   = get_or_create_collection("RigifyMarkers")
        fsize = FINGER_SIZE
        inset = 0.005  # 5mm inside surface

        for side in ("L", "R"):
            x_sign = 1 if side == "L" else -1

            tip_s     = tip.copy();     tip_s.x     = x_sign * abs(tip.x)
            knuckle_s = knuckle.copy(); knuckle_s.x = x_sign * abs(knuckle.x)

            # Mirror normals on X for the R side
            tn_s = tip_normal.copy();     tn_s.x = x_sign * abs(tip_normal.x)
            kn_s = knuckle_normal.copy(); kn_s.x = x_sign * abs(knuckle_normal.x)

            # Push inward along surface normals
            tip_s     -= tn_s * inset
            knuckle_s -= kn_s * inset

            fvec = tip_s - knuckle_s
            if fvec.length < 0.001:
                continue

            # Anatomical proportions: proximal(45%) middle(30%) distal(25%)
            p2 = knuckle_s + fvec * 0.45
            p3 = knuckle_s + fvec * 0.75

            positions = {
                f"{fname}_1_{side}":   knuckle_s,
                f"{fname}_2_{side}":   p2,
                f"{fname}_3_{side}":   p3,
                f"{fname}_TIP_{side}": tip_s,
            }

            for mname, pos in positions.items():
                full = f"MARKER_{mname}"
                obj  = bpy.data.objects.get(full)
                if obj is None:
                    obj = bpy.data.objects.new(full, None)
                    obj.empty_display_type = 'SPHERE'
                    obj.empty_display_size = fsize
                    obj["autorig_marker"]  = True
                    obj.show_in_front      = True
                    col.objects.link(obj)
                else:
                    obj.empty_display_size = fsize
                obj.location = pos.copy()
                obj.hide_set(False)

    def _header_text(self):
        fname, flabel = self.fingers[self.finger_idx]
        idx   = self.finger_idx + 1
        total = len(self.fingers)
        if self.click_phase == 'tip':
            return (f"Auto Rigify — Click {flabel} FINGERTIP  "
                    f"({idx}/{total})  | ESC to cancel")
        return (f"Auto Rigify — Click {flabel} KNUCKLE  "
                f"({idx}/{total})  | ESC to cancel")

    def cancel(self, context):
        context.area.header_text_set(None)
        self._bvh = None

    def modal(self, context, event):
        # Deferred cancel: Ctrl+Z was caught on a previous event and we swallowed
        # it (RUNNING_MODAL) to prevent undo from firing mid-cleanup.  Cancel now
        # that we're safely on a new event cycle.
        if getattr(self, '_cancel_next', False):
            # Eat additional Ctrl+Z presses — returning CANCELLED on a undo-key
            # event causes Blender to fire the undo a second time and crash.
            if event.type == 'Z' and event.ctrl:
                return {'RUNNING_MODAL'}
            context.area.header_text_set(None)
            self.report({'INFO'}, "Finger placement cancelled — press Ctrl+Z to undo placed markers.")
            return {'CANCELLED'}

        if event.type in _NAV_EVENTS:
            return {'PASS_THROUGH'}

        context.area.header_text_set(self._header_text())

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            mesh_obj = bpy.data.objects.get(
                context.scene.get("autorig_click_mesh", ""))
            if not mesh_obj:
                self.report({'WARNING'}, "No target mesh. Select mesh first.")
                return {'CANCELLED'}

            hit, normal = _snap_to_surface(context, event, mesh_obj)
            if hit is None:
                return {'PASS_THROUGH'}

            if self.click_phase == 'tip':
                self.pending_tip        = hit
                self.pending_tip_normal = normal
                self.click_phase        = 'knuckle'
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            # Second click — knuckle
            fname, _ = self.fingers[self.finger_idx]
            self._place_one_finger(
                fname,
                self.pending_tip, self.pending_tip_normal,
                hit, normal,
            )
            self.finger_idx  += 1
            self.click_phase  = 'tip'
            self.pending_tip  = None

            if self.finger_idx >= len(self.fingers):
                context.area.header_text_set(None)
                self._bvh = None   # release before undo can free the source mesh
                n = len(self.fingers)
                self.report({'INFO'},
                    f"{n} finger{'s' if n > 1 else ''} placed. Review and fine-tune.")
                return {'FINISHED'}

            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            context.area.header_text_set(None)
            self.report({'INFO'},
                f"Finger placement cancelled after {self.finger_idx}/{len(self.fingers)} fingers.")
            return {'CANCELLED'}

        # Intercept Ctrl+Z — return RUNNING_MODAL (not CANCELLED) to CONSUME the
        # event so Blender's undo system never fires while this modal is live.
        # Returning CANCELLED here passes the event through and undo fires immediately
        # after, which corrupts Blender state and crashes.  The _cancel_next flag
        # triggers a clean CANCELLED on the very next event (typically a mouse move).
        if event.type == 'Z' and event.ctrl:
            self._bvh = None
            self._cancel_next = True
            return {'RUNNING_MODAL'}

        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        self._bvh = None
        mesh_obj = _resolve_body_mesh(context)
        if mesh_obj:
            context.scene["autorig_click_mesh"] = mesh_obj.name
            self._bvh = _bvh_from_mesh(mesh_obj)

        if self.single_finger:
            self.fingers = [(f, l) for f, l in self.FINGERS if f == self.finger_name]
        else:
            self.fingers = list(self.FINGERS)

        self.finger_idx         = 0
        self.click_phase        = 'tip'
        self.pending_tip        = None
        self.pending_tip_normal = None
        self._cancel_next       = False

        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}


class AUTORIG_OT_StraightenFingerMarkers(bpy.types.Operator):
    """Redistribute intermediate finger markers (_2, _3) along the straight
    line between each finger's MCP (_1) and TIP so the bone chain is co-linear."""
    bl_idname  = "autorig.straighten_finger_markers"
    bl_label   = "Straighten Finger Markers"
    bl_options = {'REGISTER', 'UNDO'}

    _FINGERS = ["THUMB", "FINGER_INDEX", "FINGER_MIDDLE", "FINGER_RING", "FINGER_PINKY"]

    def execute(self, context):
        moved = 0
        for fname in self._FINGERS:
            for side in ("L", "R"):
                mcp_obj = bpy.data.objects.get(f"MARKER_{fname}_1_{side}")
                tip_obj = bpy.data.objects.get(f"MARKER_{fname}_TIP_{side}")
                p2_obj  = bpy.data.objects.get(f"MARKER_{fname}_2_{side}")
                p3_obj  = bpy.data.objects.get(f"MARKER_{fname}_3_{side}")
                if not (mcp_obj and tip_obj):
                    continue
                mcp  = mcp_obj.location.copy()
                tip  = tip_obj.location.copy()
                fvec = tip - mcp
                if fvec.length < 0.001:
                    continue
                fvec_sq = fvec.dot(fvec)
                for jobj, fallback_t in ((p2_obj, 0.45), (p3_obj, 0.75)):
                    if not jobj:
                        continue
                    t = (jobj.location - mcp).dot(fvec) / fvec_sq
                    t = max(0.05, min(0.95, t)) if 0.0 <= t <= 1.0 else fallback_t
                    jobj.location = mcp + fvec * t
                    moved += 1
        self.report({'INFO'}, f"Straightened {moved} intermediate markers.")
        return {'FINISHED'}


# ─────────────────────────────────────────────────────────────────────────────
# DETECT FINGERS — fan-ray from wrist (adapted from import.py)
# ─────────────────────────────────────────────────────────────────────────────

class AUTORIG_OT_DetectFingers(bpy.types.Operator):
    """Detect finger tip positions using fan rays from the HAND marker.
    Requires HAND_L and ELBOW_L markers to be placed first.
    Places FINGER_*_TIP and THUMB_TIP markers for both hands."""
    bl_idname  = "autorig.detect_fingers"
    bl_label   = "④ Detect Fingers"
    bl_options = {'REGISTER', 'UNDO'}

    ray_distance: bpy.props.FloatProperty(
        name="Ray Distance", default=0.4, min=0.1, max=1.0)
    hand_radius:  bpy.props.FloatProperty(
        name="Hand Radius",
        description="Radius around wrist to search for finger vertices",
        default=0.22, min=0.05, max=0.6)
    min_spacing:  bpy.props.FloatProperty(
        name="Min Finger Spacing", default=0.025, min=0.005, max=0.1)

    FINGER_NAMES = ["THUMB", "FINGER_INDEX", "FINGER_MIDDLE",
                    "FINGER_RING", "FINGER_PINKY"]

    def _build_bvh_eval(self, context, mesh_obj):
        """Build BVH from the evaluated mesh (includes modifiers)."""
        import bmesh as _bmesh
        from mathutils.bvhtree import BVHTree
        depsgraph = context.evaluated_depsgraph_get()
        eval_obj  = mesh_obj.evaluated_get(depsgraph)
        mesh      = eval_obj.to_mesh()
        bm        = _bmesh.new()
        bm.from_mesh(mesh)
        bm.transform(mesh_obj.matrix_world)
        bvh = BVHTree.FromBMesh(bm)
        bm.free()
        eval_obj.to_mesh_clear()
        return bvh

    def _hand_frame(self, elbow_loc, wrist_loc):
        """Compute forward/right/up axes from elbow→wrist direction."""
        forward  = (wrist_loc - elbow_loc).normalized()
        temp_up  = Vector((0, 0, 1))
        if abs(forward.dot(temp_up)) > 0.95:
            temp_up = Vector((0, 1, 0))
        right = forward.cross(temp_up).normalized()
        up    = right.cross(forward).normalized()
        return forward, right, up

    def _detect_side(self, context, mesh_obj, bvh, side):
        """
        Vertex score + BVH refinement approach.

        1. Isolate hand vertices: within HAND_RADIUS of wrist AND forward
        2. Score each vertex: forward_score + spread_score * 0.25
        3. Pick 5 unique tips with min spacing
        4. Refine with BVH ray from wrist → vertex direction
        """
        wrist_obj = bpy.data.objects.get(f"MARKER_HAND_{side}")
        elbow_obj = bpy.data.objects.get(f"MARKER_ELBOW_{side}")
        if not wrist_obj or not elbow_obj:
            self.report({'WARNING'},
                f"HAND_{side} and ELBOW_{side} markers must be placed first.")
            return []

        wrist   = wrist_obj.location.copy()
        elbow   = elbow_obj.location.copy()

        # arm_dir = wrist forward direction (elbow → wrist)
        arm_dir    = (wrist - elbow).normalized()
        world_up   = Vector((0, 0, 1))
        if abs(arm_dir.dot(world_up)) > 0.95:
            world_up = Vector((0, 1, 0))
        spread_dir = arm_dir.cross(world_up).normalized()
        if side == 'R':
            spread_dir = -spread_dir

        # ── Step 1: isolate hand region vertices ─────────────────────────
        all_verts  = [mesh_obj.matrix_world @ v.co
                      for v in mesh_obj.data.vertices]
        hand_verts = []
        for v in all_verts:
            dist        = (v - wrist).length
            forwardness = (v - wrist).dot(arm_dir)
            if dist < self.hand_radius and forwardness > 0:
                hand_verts.append(v)

        if len(hand_verts) < 10:
            self.report({'WARNING'},
                f"Side {side}: only {len(hand_verts)} hand verts found. "
                "Try increasing Hand Radius.")
            return []

        # ── Step 2: Project into spread/forward space, split into 5 lanes ─
        # Each lane = one finger column. Take farthest-forward vert per lane.
        # Guarantees one tip per finger regardless of finger proximity.
        projected = []
        for v in hand_verts:
            spread_val  = (v - wrist).dot(spread_dir)
            forward_val = (v - wrist).dot(arm_dir)
            projected.append((v, spread_val, forward_val))

        projected.sort(key=lambda x: x[1])   # left → right across palm

        # Divide into 5 lanes by spread *value* range, not array index count.
        # This prevents dense palm geometry from pushing all tips into the middle finger.
        min_s = projected[0][1]
        max_s = projected[-1][1]
        spread_range = max_s - min_s
        if spread_range < 0.001:
            self.report({'WARNING'},
                f"Side {side}: hand vertices are too tightly packed in spread direction.")
            return []

        tips = []
        for i in range(5):
            lo = min_s + i       * spread_range / 5.0
            hi = min_s + (i + 1) * spread_range / 5.0
            # Include right edge in last lane
            lane = [t for t in projected if lo <= t[1] < hi or (i == 4 and t[1] <= hi)]
            if not lane:
                continue
            best = max(lane, key=lambda x: x[2])   # farthest forward in this lane
            tips.append(best[0])

        if not tips:
            return []

        tips.sort(key=lambda p: (p - wrist).dot(spread_dir))

        # ── Step 3: BVH refinement — cast from beyond tip toward wrist ────
        # Casting from outside inward gives us both the surface hit and the
        # outward face normal, which we keep for pose-independent centering later.
        refined = []   # list of (tip_position, outward_normal)
        for tip in tips:
            direction  = (tip - wrist).normalized()
            origin_far = tip + direction * 0.05
            hit, normal, _, _ = bvh.ray_cast(origin_far, -direction, self.ray_distance)
            if hit:
                refined.append((hit.copy(), normal.copy()))
            else:
                # No surface found — use the finger-outward direction as fallback normal
                refined.append((tip, direction))

        refined.sort(key=lambda x: (x[0] - wrist).dot(spread_dir))
        return refined

    def _place_finger_markers(self, _context, mesh_obj, bvh, side, tip_pairs):
        """Place _1/_2/_3/TIP markers for each detected finger.

        Clusters hand verts by spread-lane (same partition as _detect_side), filters
        out inner-palm verts, then uses PCA to get the true finger axis.  The knuckle
        (_1) is placed at the 10th-percentile projection of the filtered finger verts
        — now in the finger region, not the deep palm.
        """
        col   = get_or_create_collection("RigifyMarkers")
        hand  = bpy.data.objects.get(f"MARKER_HAND_{side}")
        wrist = hand.location.copy() if hand else Vector((0, 0, 0))

        mw       = mesh_obj.matrix_world
        all_np   = np.array([list(mw @ v.co) for v in mesh_obj.data.vertices])
        wrist_np = np.array(list(wrist))

        elbow_obj = bpy.data.objects.get(f"MARKER_ELBOW_{side}")
        if elbow_obj:
            arm_np = np.array(list((wrist - elbow_obj.location).normalized()))
        else:
            arm_np = np.array([0.0, 1.0, 0.0])

        # Spread direction — identical to _detect_side so lanes match
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(arm_np @ world_up) > 0.95:
            world_up = np.array([0.0, 1.0, 0.0])
        spread_np = np.cross(arm_np, world_up)
        spread_np /= np.linalg.norm(spread_np) + 1e-9
        if side == 'R':
            spread_np = -spread_np

        # Isolate hand verts
        diffs        = all_np - wrist_np
        forward_proj = diffs @ arm_np
        radii        = np.linalg.norm(diffs, axis=1)
        hand_mask    = (forward_proj > 0) & (radii < self.hand_radius)
        hand_np      = all_np[hand_mask]
        hand_fwd     = forward_proj[hand_mask]   # forward distance from wrist per vert

        # Assign each vert to a spread lane (0–4), same equal-width lanes as _detect_side
        lanes      = None
        spread_vals = hand_np @ spread_np
        s_min, s_max = spread_vals.min(), spread_vals.max()
        s_range = s_max - s_min
        if len(hand_np) >= 10 and s_range > 0.001:
            raw_lanes = ((spread_vals - s_min) / s_range * 5).astype(int)
            lanes = np.clip(raw_lanes, 0, 4)

        for fi, (fname, (tip, outward_normal)) in enumerate(zip(self.FINGER_NAMES, tip_pairs)):
            tip_c = _interior_center_normal(bvh, tip, -outward_normal)

            # ── PCA knuckle detection ────────────────────────────────────────
            knuckle = None
            if lanes is not None:
                cluster_np  = hand_np[lanes == fi]
                cluster_fwd = hand_fwd[lanes == fi]

                # Filter out inner-palm verts: keep only verts in the outer 65 %
                # of the forward range.  This removes wrist/metacarpal verts so
                # PCA aligns with the finger phalanges, not the palm slab.
                if len(cluster_np) >= 6:
                    fwd_thresh  = cluster_fwd.min() + (cluster_fwd.max() - cluster_fwd.min()) * 0.35
                    finger_np   = cluster_np[cluster_fwd >= fwd_thresh]
                    if len(finger_np) < 4:
                        finger_np = cluster_np

                    center = finger_np.mean(axis=0)
                    _, _, Vt = np.linalg.svd(finger_np - center, full_matrices=False)
                    axis = Vt[0]

                    tip_np = np.array([tip.x, tip.y, tip.z])
                    if (tip_np - center) @ axis < 0:
                        axis = -axis

                    projs     = (finger_np - center) @ axis
                    proj_base = float(np.percentile(projs, 10))
                    knuckle   = Vector(center + axis * proj_base)

            if knuckle is None:
                knuckle = wrist.lerp(tip_c, 0.35)

            # ── Joint placement ──────────────────────────────────────────────
            fvec  = tip_c - knuckle
            flen  = fvec.length
            if flen < 0.001:
                continue
            fdir  = fvec / flen

            p2 = knuckle + fvec * (1.0 / 3.0)
            p3 = knuckle + fvec * (2.0 / 3.0)
            p2 = _interior_center_axis(bvh, p2, fdir)
            p3 = _interior_center_axis(bvh, p3, fdir)

            positions = {
                f"{fname}_1_{side}":   knuckle,
                f"{fname}_2_{side}":   p2,
                f"{fname}_3_{side}":   p3,
                f"{fname}_TIP_{side}": tip_c,
            }
            for mname, pos in positions.items():
                full = f"MARKER_{mname}"
                obj  = bpy.data.objects.get(full)
                if obj is None:
                    obj = bpy.data.objects.new(full, None)
                    obj.empty_display_type = 'SPHERE'
                    obj.empty_display_size = FINGER_SIZE
                    obj["autorig_marker"]  = True
                    obj.show_in_front      = True
                    col.objects.link(obj)
                else:
                    obj.empty_display_size = FINGER_SIZE
                obj.location = pos.copy()
                obj.hide_set(False)

    def execute(self, context):
        mesh_obj = _resolve_body_mesh(context)
        if not mesh_obj:
            self.report({'ERROR'},
                "Set a Body Mesh in the panel, or select the character mesh first.")
            return {'CANCELLED'}
        if not bpy.data.objects.get("MARKER_HAND_L"):
            self.report({'ERROR'},
                "HAND markers not found. Complete steps ①②③ first.")
            return {'CANCELLED'}

        bvh    = self._build_bvh_eval(context, mesh_obj)
        placed = 0
        for side in ('L', 'R'):
            tip_pairs = self._detect_side(context, mesh_obj, bvh, side)
            if len(tip_pairs) < 5:
                self.report({'WARNING'},
                    f"Only {len(tip_pairs)}/5 fingers on side {side}. "
                    "Try increasing Hand Radius.")
            if tip_pairs:
                self._place_finger_markers(context, mesh_obj, bvh, side, tip_pairs)
                placed += len(tip_pairs)
        self.report({'INFO'}, f"Finger detection complete — {placed} fingertips.")
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.label(text="HAND + ELBOW markers must be placed first.", icon='INFO')
        layout.prop(self, "hand_radius")
        layout.prop(self, "ray_distance")
        layout.prop(self, "min_spacing")


class AUTORIG_PT_FingerPicker(bpy.types.Panel):
    """Popover for picking individual fingers to place markers on."""
    bl_label       = "Pick Finger"
    bl_idname      = "AUTORIG_PT_FingerPicker"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_ui_units_x  = 8

    def draw(self, context):
        layout = self.layout
        _FINGER_ICONS = {
            "THUMB":         'COLORSET_03_VEC',
            "FINGER_INDEX":  'COLORSET_05_VEC',
            "FINGER_MIDDLE": 'COLORSET_07_VEC',
            "FINGER_RING":   'COLORSET_09_VEC',
            "FINGER_PINKY":  'COLORSET_11_VEC',
        }
        col = layout.column(align=True)
        col.scale_y = 1.4
        for fname, flabel in AUTORIG_OT_PlaceFingerMarkers.FINGERS:
            op = col.operator("autorig.place_finger_click",
                              text=flabel, icon=_FINGER_ICONS.get(fname, 'DOT'))
            op.finger_name   = fname
            op.single_finger = True
        col.separator(factor=0.5)
        all_row = col.row(align=True)
        all_row.scale_y = 1.2
        all_op = all_row.operator("autorig.place_finger_click",
                                  text="All Fingers", icon='HAND')
        all_op.single_finger = False
