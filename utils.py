# utils.py — Shared helpers, icon system, and vertex weight I/O for Easy Rigify.
import bpy
import os
import numpy as np
from mathutils import Vector
from typing import Dict

from .constants import ROLL_RULES

# ---------------------------------------------------------------------------
# Custom icon preview collection — set by __init__.register()
# ---------------------------------------------------------------------------

_preview_coll = None


def get_icon(name: str) -> int:
    """Return the icon_id for a loaded custom preview, or 0 if unavailable."""
    if _preview_coll and name in _preview_coll:
        return _preview_coll[name].icon_id
    return 0


# ---------------------------------------------------------------------------
# Bone roll lookup
# ---------------------------------------------------------------------------

def get_roll_rule(bone_name):
    """Return (roll_vector, extra_radians) for a bone, or None."""
    for prefixes, vec, extra in ROLL_RULES:
        for p in prefixes:
            if bone_name.startswith(p) or bone_name == p:
                return vec, extra
    return None


# ---------------------------------------------------------------------------
# Scene collection / object helpers
# ---------------------------------------------------------------------------

def get_or_create_collection(name):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def get_marker_col():
    return bpy.data.collections.get("RigifyMarkers")


def get_face_col():
    return bpy.data.collections.get("RigifyFaceMarkers")


def _apply_marker_style(obj):
    """Enable Show in Front on a marker empty."""
    obj.show_in_front = True


def set_mesh_select(context, selectable):
    for obj in context.scene.objects:
        if obj.type == 'MESH' and not obj.get("autorig_marker"):
            obj.hide_select = not selectable


def set_xray(context, on):
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            for sp in area.spaces:
                if sp.type == 'VIEW_3D':
                    sp.shading.show_xray = on


def _mesh_bbox_world(obj):
    """Return (min_co, max_co, center) from obj.bound_box in world space."""
    corners = [obj.matrix_world @ Vector(v) for v in obj.bound_box]
    xs = [v.x for v in corners]; ys = [v.y for v in corners]; zs = [v.z for v in corners]
    mn = Vector((min(xs), min(ys), min(zs)))
    mx = Vector((max(xs), max(ys), max(zs)))
    return mn, mx, (mn + mx) * 0.5


def get_base(name):
    for s in ("_L", "_R"):
        if name.endswith(s):
            return name[:-len(s)]
    return name


def _marker_scale():
    """Return the current scene marker scale multiplier (default 1.0)."""
    try:
        return bpy.context.scene.autorig_marker_scale
    except Exception:
        return 1.0


def _ensure_marker_empty(name, size):
    """Return a marker empty with the given name, creating or replacing it if needed.
    If an object with that name already exists but is not an empty, it is removed first."""
    col = get_or_create_collection("RigifyMarkers")
    obj = bpy.data.objects.get(name)
    if obj is not None and obj.type != 'EMPTY':
        bpy.data.objects.remove(obj, do_unlink=True)
        obj = None
    if obj is None:
        obj = bpy.data.objects.new(name, None)
        obj.empty_display_type = 'SPHERE'
        obj.empty_display_size = size * _marker_scale()
        obj["autorig_marker"]  = True
        obj.show_in_front      = True
        col.objects.link(obj)
    return obj


def make_empty(col, obj_name, dtype, pos, size, _context=None):
    e = bpy.data.objects.new(obj_name, None)
    e.empty_display_type = dtype
    e.empty_display_size = size * _marker_scale()
    e.location = Vector(pos)
    e["autorig_marker"] = True
    e.show_in_front = True
    col.objects.link(e)
    return e


# ---------------------------------------------------------------------------
# DEF bone visibility helpers
# ---------------------------------------------------------------------------

def _set_def_bones_visible(rig, visible):
    """Show or hide DEF bones — handles both Blender 3.x layers and 4.x collections."""
    arm = rig.data
    if hasattr(arm, 'collections'):
        for col in arm.collections:
            if col.name.upper().startswith('DEF'):
                col.is_visible = visible
        return
    if hasattr(arm, 'layers'):
        layers = list(arm.layers)
        layers[29] = visible
        arm.layers = layers


def _get_def_bones_visible(rig):
    """Return True if DEF bones are currently visible on this armature."""
    arm = rig.data
    if hasattr(arm, 'collections'):
        return any(c.is_visible for c in arm.collections
                   if c.name.upper().startswith('DEF'))
    if hasattr(arm, 'layers'):
        return arm.layers[29]
    return False


def _reorder_arm_modifier(context, obj):
    """Move Armature modifier to top, keeping Mirror (if any) above it."""
    arm_mod = next((m for m in obj.modifiers if m.type == 'ARMATURE'), None)
    if not arm_mod:
        return
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    context.view_layer.objects.active = obj
    # Armature to index 0 first
    bpy.ops.object.modifier_move_to_index(modifier=arm_mod.name, index=0)
    # If a Mirror exists and isn't already at 0, float it above Armature
    mir_mod = next((m for m in obj.modifiers if m.type == 'MIRROR'), None)
    if mir_mod and obj.modifiers.find(mir_mod.name) != 0:
        bpy.ops.object.modifier_move_to_index(modifier=mir_mod.name, index=0)


# ---------------------------------------------------------------------------
# Vertex weight I/O (numpy-safe, Weight Paint mode safe)
# ---------------------------------------------------------------------------

class WeightDataAccess:
    """Safe wrapper for reading/writing vertex weights without API crashes."""

    @staticmethod
    def get_vertex_weights(obj, vert_index):
        mesh = obj.data
        if vert_index >= len(mesh.vertices):
            return {}
        weights = {}
        for g in mesh.vertices[vert_index].groups:
            if g.group < len(obj.vertex_groups):
                weights[obj.vertex_groups[g.group].name] = g.weight
        return weights

    @staticmethod
    def set_vertex_weight(obj, vert_index, group_name, weight, mode='REPLACE'):
        if weight < 0.0001:
            WeightDataAccess.remove_vertex_weight(obj, vert_index, group_name)
            return
        group = obj.vertex_groups.get(group_name)
        if not group:
            group = obj.vertex_groups.new(name=group_name)
        group.add([vert_index], weight, mode)

    @staticmethod
    def remove_vertex_weight(obj, vert_index, group_name):
        group = obj.vertex_groups.get(group_name)
        if group:
            group.remove([vert_index])

    @staticmethod
    def get_all_influencing_groups(obj, vert_indices):
        all_groups = set()
        for idx in vert_indices:
            all_groups.update(WeightDataAccess.get_vertex_weights(obj, idx).keys())
        return all_groups

    @staticmethod
    def normalize_vertex(obj, vert_index):
        mesh = obj.data
        if vert_index >= len(mesh.vertices):
            return
        total = sum(g.weight for g in mesh.vertices[vert_index].groups)
        if total <= 0:
            return
        current = {}
        for g in mesh.vertices[vert_index].groups:
            if g.group < len(obj.vertex_groups):
                current[obj.vertex_groups[g.group].name] = g.weight
        for name in current:
            WeightDataAccess.remove_vertex_weight(obj, vert_index, name)
        for name, weight in current.items():
            n = weight / total
            if n >= 0.0001:
                obj.vertex_groups[name].add([vert_index], n, 'REPLACE')


def _write_weights_back(obj, group_names, w_old, w_new):
    """Write changed weight values back to vertex groups without clearing all data first.

    Preserves Blender's Weight Paint active-bone link by using REPLACE writes and
    targeted removes instead of the nuke-all approach that breaks mode state.
    Also saves and restores the active vertex group index.
    """
    saved_idx = obj.vertex_groups.active_index
    for gi, gname in enumerate(group_names):
        vg = obj.vertex_groups.get(gname)
        if vg is None:
            continue
        new_col = w_new[:, gi]
        old_col = w_old[:, gi]
        set_mask = new_col > 0.0001
        rem_mask = (old_col > 0.0001) & ~set_mask
        if set_mask.any():
            for vi in np.where(set_mask)[0]:
                vg.add([int(vi)], float(new_col[vi]), 'REPLACE')
        if rem_mask.any():
            vg.remove(np.where(rem_mask)[0].tolist())
    if 0 <= saved_idx < len(obj.vertex_groups):
        obj.vertex_groups.active_index = saved_idx
