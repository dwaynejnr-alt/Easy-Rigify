# weight_tools.py — Weight mirroring, paint workflow, transfer, editing, and advanced panel.
import bpy
from .constants import dbg
import bmesh
import numpy as np
from mathutils import Vector, kdtree
from collections import defaultdict
import math
import re
from typing import List, Dict, Optional, Any

from .utils import WeightDataAccess, _write_weights_back, get_icon, _get_def_bones_visible, _set_def_bones_visible
from .pipeline import _get_target_rig, _is_generated_rig, _REGION_BONES, _REGION_KEYWORDS


class AUTORIG_OT_SymmetrizeWeights(bpy.types.Operator):
    """Copy vertex weights from one side to the other (one-way, no vertex flipping)"""
    bl_idname  = "autorig.symmetrize_weights"
    bl_label   = "Symmetrize Weights"
    bl_options = {'REGISTER', 'UNDO'}

    direction: bpy.props.EnumProperty(
        items=[
            ('NEGATIVE_X', 'L → R', 'Copy left side weights onto the right side'),
            ('POSITIVE_X', 'R → L', 'Copy right side weights onto the left side'),
        ],
        default='NEGATIVE_X',
    )

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _flip_group_name(name):
        """Swap .L/.R or _L/_R suffix so weights land on the correct bone.
        Handles Rigify multi-segment names: DEF-upper_arm.L.001 → DEF-upper_arm.R.001"""
        import re
        # Multi-segment Rigify: ends with .L.NNN or .R.NNN
        m = re.match(r'^(.*)\.(L)(\.\d+)$', name)
        if m:
            return m.group(1) + '.R' + m.group(3)
        m = re.match(r'^(.*)\.(R)(\.\d+)$', name)
        if m:
            return m.group(1) + '.L' + m.group(3)
        # Standard single-segment
        if name.endswith('.L'):   return name[:-2] + '.R'
        if name.endswith('.R'):   return name[:-2] + '.L'
        if name.endswith('_L'):   return name[:-2] + '_R'
        if name.endswith('_R'):   return name[:-2] + '_L'
        return name

    def _mirror_weights_on(self, obj):
        mesh     = obj.data
        n_verts  = len(mesh.vertices)
        grp_names = [g.name for g in obj.vertex_groups]
        n_grps   = len(grp_names)
        if n_grps == 0:
            return

        # Read all weights into a numpy array
        weights = np.zeros((n_verts, n_grps), dtype=np.float32)
        for v in mesh.vertices:
            for g in v.groups:
                if g.group < n_grps:
                    weights[v.index, g.group] = g.weight

        # Symmetry plane (median X in local space)
        x_coords = np.array([v.co.x for v in mesh.vertices], dtype=np.float32)
        center_x = float(np.median(x_coords))

        # Precompute flip index: for each group index, the index of its .L/.R counterpart
        name_to_idx  = {n: i for i, n in enumerate(grp_names)}
        flip_idx_arr = np.array(
            [name_to_idx.get(self._flip_group_name(n), i) for i, n in enumerate(grp_names)],
            dtype=np.int32)

        # Build KD-tree over all vertex positions
        kd = kdtree.KDTree(n_verts)
        for i, v in enumerate(mesh.vertices):
            kd.insert(v.co, i)
        kd.balance()

        # direction NEGATIVE_X = L→R: source = LEFT side (+X), dest = RIGHT side (-X)
        # direction POSITIVE_X = R→L: source = RIGHT side (-X), dest = LEFT side (+X)
        source_is_left = (self.direction == 'NEGATIVE_X')

        new_weights = weights.copy()

        TOLERANCE = 0.010  # 10 mm — handles slightly asymmetric mesh topology

        for v in mesh.vertices:
            x        = v.co.x
            is_left  = x > center_x
            if is_left != source_is_left:
                continue  # only process source-side vertices

            # Mirror position across the symmetry plane
            mirror_co = (2.0 * center_x - x, v.co.y, v.co.z)
            _, dest_idx, dist = kd.find(mirror_co)
            if dist > TOLERANCE:
                continue  # no matching mirror vertex within tolerance

            # Copy source weights → dest with .L/.R name swap
            dst_row = np.zeros(n_grps, dtype=np.float32)
            dst_row[flip_idx_arr] = weights[v.index]
            new_weights[dest_idx] = dst_row

        # Write back — clear all then re-assign
        all_idx = list(range(n_verts))
        for vg in obj.vertex_groups:
            vg.remove(all_idx)
        for vi in range(n_verts):
            for gi in range(n_grps):
                w = float(new_weights[vi, gi])
                if w > 0.0001:
                    obj.vertex_groups[grp_names[gi]].add([vi], w, 'REPLACE')

    # ── execute ────────────────────────────────────────────────────────────────

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Select the mesh you want to symmetrize.")
            return {'CANCELLED'}
        self._mirror_weights_on(obj)
        self.report({'INFO'}, f"Symmetrized {self.direction.replace('_', ' ')} on '{obj.name}'")
        return {'FINISHED'}


class AUTORIG_OT_EnterWeightPaint(bpy.types.Operator):
    """Enter weight paint mode with the armature linked for visual bone feedback"""
    bl_idname  = "autorig.enter_weight_paint"
    bl_label   = "Enter Weight Paint"
    bl_options = {'REGISTER'}

    def execute(self, context):
        obj = context.active_object
        rig = _get_target_rig(context)
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Select the character mesh first")
            return {'CANCELLED'}
        if not rig:
            self.report({'ERROR'}, "No armature found in scene")
            return {'CANCELLED'}

        sp = context.scene.autorig_skin
        rig.hide_set(False)
        rig.hide_viewport = False

        # Save current DEF bone visibility so we can restore it on exit
        arm = rig.data
        if hasattr(arm, 'collections'):
            prev = any(c.is_visible for c in arm.collections
                       if c.name.upper().startswith('DEF'))
        elif hasattr(arm, 'layers'):
            prev = arm.layers[29]
        else:
            prev = False
        sp.def_bones_were_visible = prev

        # Enable DEF bones so weight painting shows the right influences
        _set_def_bones_visible(rig, True)

        # Must be in OBJECT mode before entering weight paint
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        # Select mesh (active) + rig (secondary) — Blender puts the rig into
        # Pose mode automatically when weight paint is entered this way,
        # which is what allows clicking bones to switch vertex groups.
        for o in context.view_layer.objects:
            o.select_set(False)
        rig.select_set(True)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='WEIGHT_PAINT')

        # Show zero-weight vertices highlighted in the viewport
        context.tool_settings.vertex_group_user = 'ACTIVE'

        self.report({'INFO'}, "Weight paint ready — click bones to switch vertex groups.")
        return {'FINISHED'}


class AUTORIG_OT_ExitWeightPaint(bpy.types.Operator):
    """Exit weight paint mode and restore DEF bone visibility to its previous state"""
    bl_idname  = "autorig.exit_weight_paint"
    bl_label   = "Exit Weight Paint"
    bl_options = {'REGISTER'}

    def execute(self, context):
        sp  = context.scene.autorig_skin
        rig = _get_target_rig(context)

        bpy.ops.object.mode_set(mode='OBJECT')

        if rig:
            _set_def_bones_visible(rig, sp.def_bones_were_visible)

        self.report({'INFO'}, "Weight paint exited — DEF bone visibility restored")
        return {'FINISHED'}


class AUTORIG_OT_ToggleDefBones(bpy.types.Operator):
    """Toggle DEF bone visibility on the generated rig"""
    bl_idname  = "autorig.toggle_def_bones"
    bl_label   = "Toggle DEF Bones"
    bl_options = {'REGISTER'}

    def execute(self, context):
        rig = _get_target_rig(context)
        if not rig:
            self.report({'ERROR'}, "No armature found in scene")
            return {'CANCELLED'}
        current = _get_def_bones_visible(rig)
        _set_def_bones_visible(rig, not current)
        state = "visible" if not current else "hidden"
        self.report({'INFO'}, f"DEF bones {state}")
        return {'FINISHED'}


class AUTORIG_OT_SelectBoneRegion(bpy.types.Operator):
    """Set the active vertex group to the first bone of the chosen body region"""
    bl_idname  = "autorig.select_bone_region"
    bl_label   = "Select Bone Region"
    bl_options = {'REGISTER'}

    region: bpy.props.StringProperty(default='SPINE')

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Mesh must be active")
            return {'CANCELLED'}

        rig = _get_target_rig(context)
        is_generated = rig and _is_generated_rig(rig)

        if is_generated:
            candidates = _REGION_BONES.get(self.region, [])
        else:
            keywords = _REGION_KEYWORDS.get(self.region, ())
            candidates = [vg.name for vg in obj.vertex_groups
                          if any(kw in vg.name.lower() for kw in keywords)]

        for bone_name in candidates:
            vg = obj.vertex_groups.get(bone_name)
            if vg:
                obj.vertex_groups.active_index = vg.index
                self.report({'INFO'}, f"Painting: {bone_name}")
                return {'FINISHED'}

        self.report({'WARNING'}, f"No {self.region.lower()} vertex groups found on this mesh")
        return {'FINISHED'}


class SmartWeightTransfer:
    """KD-tree accelerated weight transfer with advanced options."""

    def __init__(self, source_obj, target_obj):
        self.source = source_obj
        self.target = target_obj
        self.source_kd = None
        self.source_verts = None

    def build_source_kdtree(self):
        if self.source.type != 'MESH':
            raise TypeError("Source must be a mesh object")
        mesh = self.source.data
        self.source_verts = mesh.vertices
        size = len(self.source_verts)
        self.source_kd = kdtree.KDTree(size)
        for i, vert in enumerate(self.source_verts):
            self.source_kd.insert(self.source.matrix_world @ vert.co, i)
        self.source_kd.balance()

    def get_weight_array(self, obj):
        mesh = obj.data
        groups = obj.vertex_groups
        group_names = [g.name for g in groups]
        group_idx = {name: i for i, name in enumerate(group_names)}
        weights = np.zeros((len(mesh.vertices), len(groups)), dtype=np.float32)
        for vert in mesh.vertices:
            for g in vert.groups:
                if g.group < len(groups):
                    weights[vert.index, g.group] = g.weight
        return weights, group_names, group_idx

    def set_weights_from_array(self, obj, weights, group_names):
        """Write a (V × G) weight matrix using bmesh deform layer for speed."""
        for name in group_names:
            if name not in obj.vertex_groups:
                obj.vertex_groups.new(name=name)

        gi_map   = [obj.vertex_groups[name].index for name in group_names]
        clear_gi = set(gi_map)

        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        dl = bm.verts.layers.deform.verify()

        for bv in bm.verts:
            d   = bv[dl]
            row = weights[bv.index]
            for gi in clear_gi:
                if gi in d:
                    del d[gi]
            for local_i, gi in enumerate(gi_map):
                w = float(row[local_i])
                if w > 0.0001:
                    d[gi] = w

        bm.to_mesh(obj.data)
        bm.free()

    def _get_normals_np(self, obj):
        """Return world-space vertex normals as (V, 3) float32 array."""
        mesh   = obj.data
        n      = len(mesh.vertices)
        nrm    = np.empty(n * 3, dtype=np.float32)
        mesh.vertices.foreach_get("normal", nrm)
        nrm    = nrm.reshape(n, 3)
        nmat   = np.array(obj.matrix_world.to_3x3(), dtype=np.float32)
        world  = nrm @ nmat.T
        lens   = np.linalg.norm(world, axis=1, keepdims=True)
        lens   = np.where(lens > 0, lens, 1.0)
        return (world / lens).astype(np.float32)

    def transfer_nearest_vertex(self, num_nearest=1, max_influences=4):
        if not self.source_kd:
            self.build_source_kdtree()
        source_weights, source_group_names, _ = self.get_weight_array(self.source)
        n_target      = len(self.target.data.vertices)
        target_weights = np.zeros((n_target, len(source_group_names)), dtype=np.float32)
        target_world  = self.target.matrix_world

        for vert in self.target.data.vertices:
            nearest = self.source_kd.find_n(target_world @ vert.co, num_nearest)
            if not nearest:
                continue
            if num_nearest == 1 or nearest[0][2] < 0.0001:
                target_weights[vert.index] = source_weights[nearest[0][1]]
                continue
            total = 0.0
            for _co, src_i, dist in nearest:
                wf = 1.0 / (dist + 0.0001)
                target_weights[vert.index] += source_weights[src_i] * wf
                total += wf
            if total > 0:
                target_weights[vert.index] /= total

        if max_influences > 0:
            target_weights = self._limit_influences(target_weights, max_influences)
        self.set_weights_from_array(self.target, self._normalize(target_weights), source_group_names)
        return len(source_group_names)

    def transfer_surface_projection(self, search_radius=0.1, max_influences=4):
        if not self.source_kd:
            self.build_source_kdtree()
        source_weights, source_group_names, _ = self.get_weight_array(self.source)
        n_target       = len(self.target.data.vertices)
        target_weights = np.zeros((n_target, len(source_group_names)), dtype=np.float32)
        target_world   = self.target.matrix_world
        world_normals  = self._get_normals_np(self.target)

        for vert in self.target.data.vertices:
            vi       = vert.index
            world_co = target_world @ vert.co
            world_n  = Vector(world_normals[vi].tolist())
            nearest  = self.source_kd.find_range(world_co, search_radius)
            if not nearest:
                nearest = [self.source_kd.find(world_co)]
            total = 0.0
            for co, src_i, dist in nearest:
                dv = co - world_co
                na = abs(world_n.dot(dv.normalized())) if dv.length > 0.0001 else 1.0
                cw = na / (dist + 0.001)
                target_weights[vi] += source_weights[src_i] * cw
                total += cw
            if total > 0:
                target_weights[vi] /= total

        if max_influences > 0:
            target_weights = self._limit_influences(target_weights, max_influences)
        self.set_weights_from_array(self.target, self._normalize(target_weights), source_group_names)
        return len(source_group_names)

    def transfer_volume_sampling(self, num_samples=5, max_influences=4):
        if not self.source_kd:
            self.build_source_kdtree()
        source_weights, source_group_names, _ = self.get_weight_array(self.source)
        n_target       = len(self.target.data.vertices)
        target_weights = np.zeros((n_target, len(source_group_names)), dtype=np.float32)
        target_world   = self.target.matrix_world
        world_normals  = self._get_normals_np(self.target)
        depths         = np.linspace(-0.02, 0.02, num_samples)

        for vert in self.target.data.vertices:
            vi       = vert.index
            world_co = target_world @ vert.co
            world_n  = Vector(world_normals[vi].tolist())
            for depth in depths:
                found = self.source_kd.find(world_co + world_n * float(depth))
                target_weights[vi] += source_weights[found[1]] * (1.0 / (found[2] + 0.001))

        if max_influences > 0:
            target_weights = self._limit_influences(target_weights, max_influences)
        self.set_weights_from_array(self.target, self._normalize(target_weights), source_group_names)
        return len(source_group_names)

    def _limit_influences(self, weights, max_influences):
        if max_influences <= 0 or weights.shape[1] <= max_influences:
            return weights
        limited = np.zeros_like(weights)
        top_idx = np.argpartition(-weights, max_influences, axis=1)[:, :max_influences]
        rows    = np.arange(weights.shape[0])[:, np.newaxis]
        limited[rows, top_idx] = weights[rows, top_idx]
        return limited

    def _normalize(self, weights):
        sums = weights.sum(axis=1)
        mask = sums > 0
        weights[mask] = weights[mask] / sums[mask, np.newaxis]
        return weights


class SmartWeightProperties(bpy.types.PropertyGroup):
    method: bpy.props.EnumProperty(
        name="Transfer Method",
        items=[
            ('NEAREST', "Nearest Vertex",   "Simple nearest vertex transfer"),
            ('SURFACE', "Surface Projection","Normal-aware surface projection"),
            ('VOLUME',  "Volume Sampling",  "Sample multiple depths along normal"),
        ],
        default='NEAREST',
    )
    num_samples: bpy.props.IntProperty(
        name="Nearest Samples",
        description="Number of nearest source vertices to blend",
        default=1, min=1, max=10,
    )
    volume_samples: bpy.props.IntProperty(
        name="Volume Samples",
        description="Number of depth samples along the vertex normal",
        default=5, min=2, max=20,
    )
    search_radius: bpy.props.FloatProperty(
        name="Search Radius",
        description="Maximum distance for surface projection",
        default=0.1, min=0.001, max=10.0,
    )
    max_influences: bpy.props.IntProperty(
        name="Max Influences",
        description="Maximum bones per vertex (0 = unlimited)",
        default=4, min=0, max=32,
    )
    show_transfer: bpy.props.BoolProperty(name="Smart Weight Transfer", default=False)


class SMARTWEIGHT_OT_transfer(bpy.types.Operator):
    """Transfer weights from active mesh to selected mesh(es) using KD-tree"""
    bl_idname = "object.smart_weight_transfer"
    bl_label = "Smart Weight Transfer"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.smart_weight_props
        source = context.active_object
        if not source or source.type != 'MESH':
            self.report({'ERROR'}, "Active object must be a mesh (source)")
            return {'CANCELLED'}
        targets = [o for o in context.selected_objects if o != source and o.type == 'MESH']
        if not targets:
            self.report({'ERROR'}, "Select at least one target mesh")
            return {'CANCELLED'}
        transfer = SmartWeightTransfer(source, targets[0])
        try:
            if props.method == 'NEAREST':
                groups = transfer.transfer_nearest_vertex(
                    num_nearest=props.num_samples, max_influences=props.max_influences)
            elif props.method == 'SURFACE':
                groups = transfer.transfer_surface_projection(
                    search_radius=props.search_radius, max_influences=props.max_influences)
            elif props.method == 'VOLUME':
                groups = transfer.transfer_volume_sampling(
                    num_samples=props.volume_samples, max_influences=props.max_influences)
            else:
                self.report({'ERROR'}, f"Unknown transfer method: {props.method}")
                return {'CANCELLED'}
            self.report({'INFO'}, f"Transferred {groups} vertex groups")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}


class SMARTWEIGHT_OT_batch_transfer(bpy.types.Operator):
    """Transfer weights from active mesh to all other selected meshes"""
    bl_idname = "object.smart_weight_batch_transfer"
    bl_label = "Batch Weight Transfer"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.smart_weight_props
        source = context.active_object
        if not source or source.type != 'MESH':
            self.report({'ERROR'}, "Active object must be the source mesh")
            return {'CANCELLED'}
        targets = [o for o in context.selected_objects if o != source and o.type == 'MESH']
        if not targets:
            self.report({'ERROR'}, "Select at least one target mesh")
            return {'CANCELLED'}
        transfer = SmartWeightTransfer(source, targets[0])
        transfer.build_source_kdtree()
        for target in targets:
            transfer.target = target
            try:
                if props.method == 'NEAREST':
                    transfer.transfer_nearest_vertex(
                        num_nearest=props.num_samples, max_influences=props.max_influences)
                elif props.method == 'SURFACE':
                    transfer.transfer_surface_projection(
                        search_radius=props.search_radius, max_influences=props.max_influences)
                elif props.method == 'VOLUME':
                    transfer.transfer_volume_sampling(
                        num_samples=props.volume_samples, max_influences=props.max_influences)
                else:
                    self.report({'WARNING'}, f"Unknown method: {props.method}")
            except Exception as e:
                self.report({'WARNING'}, f"Failed on {target.name}: {e}")
        self.report({'INFO'}, f"Transferred weights to {len(targets)} objects")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Tab draw helpers — called from AUTORIG_PT_Main (Skin tab)
# ---------------------------------------------------------------------------

def draw_smart_transfer_section(layout, context):
    props = context.scene.smart_weight_props
    layout.prop(props, "method")
    layout.separator()
    if props.method == 'NEAREST':
        layout.prop(props, "num_samples")
    elif props.method == 'SURFACE':
        layout.prop(props, "search_radius")
    elif props.method == 'VOLUME':
        layout.prop(props, "volume_samples")
    layout.prop(props, "max_influences")
    layout.separator()
    col = layout.column(align=True)
    col.operator("object.smart_weight_transfer",       icon='MOD_VERTEX_WEIGHT')
    col.operator("object.smart_weight_batch_transfer", icon='DUPLICATE')


class SMARTWEIGHT_PT_panel(bpy.types.Panel):
    bl_label      = "Smart Weight Transfer"
    bl_idname     = "SMARTWEIGHT_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type= 'UI'
    bl_category   = "Easy Rigify"
    bl_order      = 2
    bl_options    = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context): return False

    def draw(self, context):
        draw_smart_transfer_section(self.layout, context)


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT EDITOR
# ─────────────────────────────────────────────────────────────────────────────

class WeightEditProperties(bpy.types.PropertyGroup):
    hammer_influences: bpy.props.IntProperty(
        name="Max Bones", default=4, min=1, max=8,
    )
    hammer_smooth_passes: bpy.props.IntProperty(
        name="Smooth Passes", default=3, min=1, max=10,
    )


class WEIGHTEDIT_OT_hammer(bpy.types.Operator):
    """One-click weight cleanup: smooth, normalize, limit influences"""
    bl_idname = "object.weight_hammer"
    bl_label = "Weight Hammer"
    bl_options = {'REGISTER', 'UNDO'}

    max_influences: bpy.props.IntProperty(name="Max Bones", default=4, min=1, max=8)
    smooth_iterations: bpy.props.IntProperty(name="Smooth Passes", default=3, min=1, max=10)
    keep_locked: bpy.props.BoolProperty(
        name="Keep Locked Groups", default=True,
        description="Don't modify locked vertex groups",
    )

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Active object must be a mesh")
            return {'CANCELLED'}

        original_mode = obj.mode
        if original_mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        mesh = obj.data
        access = WeightDataAccess()

        locked_groups = set()
        if self.keep_locked:
            for vg in obj.vertex_groups:
                if vg.lock_weight:
                    locked_groups.add(vg.name)

        # Build adjacency once — avoids O(V×E) edge scan inside the per-vertex loop
        adjacency = [set() for _ in range(len(mesh.vertices))]
        for edge in mesh.edges:
            v0, v1 = edge.vertices
            adjacency[v0].add(v1)
            adjacency[v1].add(v0)

        processed = 0
        for vert_idx in range(len(mesh.vertices)):
            if not mesh.vertices[vert_idx].groups:
                continue
            for _ in range(self.smooth_iterations):
                self._smooth_vertex_weights(obj, vert_idx, locked_groups, adjacency)
            self._limit_vertex_influences(obj, vert_idx, self.max_influences, locked_groups)
            vert_w = access.get_vertex_weights(obj, vert_idx)
            if not any(name in locked_groups for name in vert_w):
                access.normalize_vertex(obj, vert_idx)
            processed += 1

        self._remove_empty_groups(obj)

        if original_mode != 'OBJECT':
            bpy.ops.object.mode_set(mode=original_mode)

        self.report({'INFO'}, f"Hammered {processed} vertices")
        return {'FINISHED'}

    def _smooth_vertex_weights(self, obj, vert_idx, locked_groups, adjacency):
        connected = adjacency[vert_idx]
        if not connected:
            return
        access = WeightDataAccess()
        all_w = defaultdict(list)
        for idx in [vert_idx] + list(connected):
            for name, w in access.get_vertex_weights(obj, idx).items():
                if name not in locked_groups:
                    all_w[name].append(w)
        avg = {name: sum(ws) / len(ws) for name, ws in all_w.items() if ws}
        for name in access.get_vertex_weights(obj, vert_idx):
            if name not in locked_groups:
                access.remove_vertex_weight(obj, vert_idx, name)
        for name, w in avg.items():
            if w >= 0.0001:
                access.set_vertex_weight(obj, vert_idx, name, w)

    def _limit_vertex_influences(self, obj, vert_idx, max_count, locked_groups):
        mesh = obj.data
        unlocked_w = []
        for g in mesh.vertices[vert_idx].groups:
            if g.group >= len(obj.vertex_groups):
                continue
            name = obj.vertex_groups[g.group].name
            if name not in locked_groups:
                unlocked_w.append((name, g.weight))
        if len(unlocked_w) <= max_count:
            return
        unlocked_w.sort(key=lambda x: x[1], reverse=True)
        access = WeightDataAccess()
        for name, _ in unlocked_w[max_count:]:
            access.remove_vertex_weight(obj, vert_idx, name)

    def _remove_empty_groups(self, obj):
        # FIX #2: collect used group indices first, then remove any that are unused
        used = {g.group for v in obj.data.vertices for g in v.groups}
        for vg in reversed(list(obj.vertex_groups)):
            if vg.index not in used:
                obj.vertex_groups.remove(vg)


class WEIGHTEDIT_OT_mirror(bpy.types.Operator):
    """Mirror vertex weights with smart bone name matching"""
    bl_idname = "object.weight_mirror"
    bl_label = "Mirror Weights"
    bl_options = {'REGISTER', 'UNDO'}

    direction: bpy.props.EnumProperty(
        name="Direction",
        items=[
            ('L_TO_R', "Left to Right", "Mirror left-side weights to right"),
            ('R_TO_L', "Right to Left", "Mirror right-side weights to left"),
        ],
        default='L_TO_R',
    )

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Active object must be a mesh")
            return {'CANCELLED'}
        if obj.mode != 'EDIT':
            self.report({'ERROR'}, "Must be in Edit Mode")
            return {'CANCELLED'}
        bone_map = self._build_bone_map(obj)
        if not bone_map:
            self.report({'WARNING'}, "No mirrored bone pairs found")
            return {'CANCELLED'}
        bm = bmesh.from_edit_mesh(obj.data)
        mirror_pairs = self._find_mirror_vertices(bm)

        # VertexGroup.remove() is forbidden in Edit mode — switch out, write, return
        bpy.ops.object.mode_set(mode='OBJECT')
        access = WeightDataAccess()
        mirrored = 0
        for src_idx, dst_idx in mirror_pairs:
            src_w = access.get_vertex_weights(obj, src_idx)
            if not src_w:
                continue
            dst_w = {bone_map[n]: w for n, w in src_w.items() if n in bone_map}
            if dst_w:
                for name in access.get_vertex_weights(obj, dst_idx):
                    access.remove_vertex_weight(obj, dst_idx, name)
                for name, w in dst_w.items():
                    access.set_vertex_weight(obj, dst_idx, name, w)
                mirrored += 1
        bpy.ops.object.mode_set(mode='EDIT')
        self.report({'INFO'}, f"Mirrored weights on {mirrored} vertices")
        return {'FINISHED'}

    def _build_bone_map(self, obj):
        # FIX #6: regex suffix/word-boundary patterns — no greedy single-char replace
        group_names = {g.name for g in obj.vertex_groups}
        bone_map = {}
        if self.direction == 'L_TO_R':
            patterns = [
                (r'\.L$', '.R'), (r'_L$', '_R'),
                (r'\.l$', '.r'), (r'_l$', '_r'),
                (r'\bLeft\b', 'Right'), (r'\bleft\b', 'right'),
            ]
        else:
            patterns = [
                (r'\.R$', '.L'), (r'_R$', '_L'),
                (r'\.r$', '.l'), (r'_r$', '_l'),
                (r'\bRight\b', 'Left'), (r'\bright\b', 'left'),
            ]
        for name in group_names:
            for pat, repl in patterns:
                mirrored = re.sub(pat, repl, name)
                if mirrored != name and mirrored in group_names:
                    bone_map[name] = mirrored
                    break
        return bone_map

    def _find_mirror_vertices(self, bm):
        pairs = []
        threshold = 0.001
        neg_verts = {}
        for vert in bm.verts:
            if vert.co.x < -threshold:
                neg_verts[(round(vert.co.y, 3), round(vert.co.z, 3))] = vert.index
        for vert in bm.verts:
            if vert.co.x > threshold:
                key = (round(vert.co.y, 3), round(vert.co.z, 3))
                if key in neg_verts:
                    if self.direction == 'L_TO_R':
                        pairs.append((neg_verts[key], vert.index))
                    else:
                        pairs.append((vert.index, neg_verts[key]))
        return pairs


class WEIGHTEDIT_OT_multi_edit(bpy.types.Operator):
    """Edit weights for multiple bones simultaneously on selected vertices"""
    bl_idname = "object.weight_multi_edit"
    bl_label = "Multi-Bone Weight Edit"
    bl_options = {'REGISTER', 'UNDO'}

    weight_string: bpy.props.StringProperty(
        name="Weights",
        description="Comma-separated bone:weight pairs",
        default="",
    )
    operation: bpy.props.EnumProperty(
        name="Operation",
        items=[
            ('SET',    "Set Exact", "Set weights to exact values"),
            ('ADD',    "Add",       "Add to existing weights"),
            ('SCALE',  "Scale",     "Multiply existing weights"),
            ('NUDGE',  "Nudge",     "Smart weight redistribution"),
            ('SMOOTH', "Smooth",    "Average with neighbors"),
        ],
        default='SET',
    )
    nudge_amount: bpy.props.FloatProperty(name="Amount", default=0.1, min=-1.0, max=1.0)

    def parse_weight_string(self):
        weights = {}
        for pair in self.weight_string.split(','):
            if ':' not in pair:
                continue
            bone, ws = pair.split(':', 1)
            try:
                weights[bone.strip()] = max(0.0, min(1.0, float(ws.strip())))
            except ValueError:
                continue
        return weights

    def invoke(self, context, event):
        obj = context.active_object
        if obj and obj.type == 'MESH' and obj.mode == 'EDIT':
            bm = bmesh.from_edit_mesh(obj.data)
            selected = [v.index for v in bm.verts if v.select]
            if selected:
                access = WeightDataAccess()
                avg = defaultdict(float)
                for idx in selected:
                    for name, w in access.get_vertex_weights(obj, idx).items():
                        avg[name] += w
                count = len(selected)
                top4 = sorted(avg.items(), key=lambda x: x[1], reverse=True)[:4]
                self.weight_string = ','.join(f"{n}:{w/count:.3f}" for n, w in top4)
        return context.window_manager.invoke_props_dialog(self, width=400)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "weight_string")
        layout.prop(self, "operation")
        if self.operation == 'NUDGE':
            layout.prop(self, "nudge_amount")
        layout.label(text="Format: bone_name:weight, bone_name:weight")

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Active object must be a mesh")
            return {'CANCELLED'}
        if obj.mode != 'EDIT':
            self.report({'ERROR'}, "Must be in Edit Mode with vertices selected")
            return {'CANCELLED'}
        target_weights = self.parse_weight_string()
        if not target_weights and self.operation == 'SET':
            self.report({'WARNING'}, "No valid weights specified")
            return {'CANCELLED'}
        bm = bmesh.from_edit_mesh(obj.data)
        selected = [v.index for v in bm.verts if v.select]
        if not selected:
            self.report({'WARNING'}, "No vertices selected")
            return {'CANCELLED'}

        # Pre-collect neighbor map from bmesh while still in Edit mode.
        # VertexGroup.remove() is forbidden in Edit mode, so we switch out
        # before any weight writes happen.
        neighbor_map = {v.index: [e.other_vert(v).index for e in v.link_edges]
                        for v in bm.verts}

        bpy.ops.object.mode_set(mode='OBJECT')
        access = WeightDataAccess()
        for vert_idx in selected:
            current = access.get_vertex_weights(obj, vert_idx)
            if self.operation == 'SET':
                for name in current:
                    access.remove_vertex_weight(obj, vert_idx, name)
                for name, w in target_weights.items():
                    access.set_vertex_weight(obj, vert_idx, name, w)
            elif self.operation == 'ADD':
                for name, w in target_weights.items():
                    access.set_vertex_weight(obj, vert_idx, name,
                                             min(1.0, current.get(name, 0.0) + w))
                access.normalize_vertex(obj, vert_idx)
            elif self.operation == 'SCALE':
                for name, factor in target_weights.items():
                    access.set_vertex_weight(obj, vert_idx, name,
                                             current.get(name, 0.0) * factor)
                access.normalize_vertex(obj, vert_idx)
            elif self.operation == 'NUDGE':
                self._nudge_weights(obj, vert_idx, current, target_weights)
            elif self.operation == 'SMOOTH':
                self._smooth_with_neighbors(obj, vert_idx, neighbor_map)
        bpy.ops.object.mode_set(mode='EDIT')
        self.report({'INFO'}, f"Updated {len(selected)} vertices")
        return {'FINISHED'}

    def _nudge_weights(self, obj, vert_idx, current, target):
        # FIX #1: parameter is 'target', not 'target_weights'
        access = WeightDataAccess()
        total = sum(current.values())
        if total <= 0:
            return
        new_w = dict(current)
        adjustment = self.nudge_amount * total
        for name in target:
            new_w[name] = max(0.0, min(1.0, new_w.get(name, 0.0) + adjustment))
        row_total = sum(new_w.values())
        if row_total > 0:
            for n in new_w:
                new_w[n] /= row_total
        for name in current:
            access.remove_vertex_weight(obj, vert_idx, name)
        for name, w in new_w.items():
            if w >= 0.0001:
                access.set_vertex_weight(obj, vert_idx, name, w)

    def _smooth_with_neighbors(self, obj, vert_idx, neighbor_map):
        access = WeightDataAccess()
        neighbors = neighbor_map.get(vert_idx, [])
        if not neighbors:
            return
        all_groups = set()
        vert_weights = {}
        for idx in [vert_idx] + neighbors:
            w = access.get_vertex_weights(obj, idx)
            vert_weights[idx] = w
            all_groups.update(w.keys())
        count = len(neighbors) + 1
        avg = defaultdict(float)
        for idx in vert_weights:
            for name in all_groups:
                avg[name] += vert_weights[idx].get(name, 0.0)
        for name in avg:
            avg[name] /= count
        for name in access.get_vertex_weights(obj, vert_idx):
            access.remove_vertex_weight(obj, vert_idx, name)
        for name, w in avg.items():
            if w >= 0.0001:
                access.set_vertex_weight(obj, vert_idx, name, w)


class WEIGHTEDIT_PT_panel(bpy.types.Panel):
    bl_label      = "Weight Editor"
    bl_idname     = "WEIGHTEDIT_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type= 'UI'
    bl_category   = "Easy Rigify"

    bl_options    = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        props = context.scene.weight_edit_props

        box = layout.box()
        box.label(text="Multi-Bone Edit", icon='BONE_DATA')
        box.operator("object.weight_multi_edit", text="Edit Selected Verts")
        # FIX #4: guard bmesh call — only valid in Edit mode
        if obj and obj.type == 'MESH' and obj.mode == 'EDIT':
            bm = bmesh.from_edit_mesh(obj.data)
            box.label(text=f"Selected: {sum(1 for v in bm.verts if v.select)} vertices")

        box = layout.box()
        box.label(text="Mirror", icon='MOD_MIRROR')
        col = box.column(align=True)
        op = col.operator("object.weight_mirror", text="Left to Right")
        op.direction = 'L_TO_R'
        op = col.operator("object.weight_mirror", text="Right to Left")
        op.direction = 'R_TO_L'

        box = layout.box()
        box.label(text="Cleanup", icon='BRUSH_DATA')
        col = box.column(align=True)
        col.prop(props, "hammer_influences")
        col.prop(props, "hammer_smooth_passes")
        col.operator("object.weight_hammer", text="Weight Hammer")


# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

class SmartWeightSmoother:
    """Edge-flow-aware weight smoothing that respects mesh topology and boundaries."""

    def __init__(self, mesh_obj):
        if mesh_obj.type != 'MESH':
            raise TypeError("Expected a mesh object")
        self.obj = mesh_obj
        self.mesh = mesh_obj.data
        self.access = WeightDataAccess()

        self.edge_to_faces = defaultdict(list)
        self.vertex_neighbors = defaultdict(set)
        self.boundary_edges = set()
        self.seam_edges = set()
        self.material_boundary_edges = set()

        self._analyze_topology()

    def _analyze_topology(self):
        mesh = self.mesh

        for poly in mesh.polygons:
            for edge_key in poly.edge_keys:
                self.edge_to_faces[edge_key].append(poly.index)

        for edge in mesh.edges:
            edge_key = tuple(sorted(edge.vertices))
            if len(self.edge_to_faces.get(edge_key, [])) == 1:
                self.boundary_edges.add(edge_key)
            if edge.use_edge_sharp:
                self.seam_edges.add(edge_key)
            self.vertex_neighbors[edge.vertices[0]].add(edge.vertices[1])
            self.vertex_neighbors[edge.vertices[1]].add(edge.vertices[0])

        for edge_key, face_indices in self.edge_to_faces.items():
            if len(face_indices) == 2:
                f1, f2 = face_indices
                if mesh.polygons[f1].material_index != mesh.polygons[f2].material_index:
                    self.material_boundary_edges.add(edge_key)

    def smooth_weights(self, group_names=None, iterations=5, factor=0.5,
                       respect_boundaries=True, respect_seams=True,
                       respect_materials=True, edge_flow_weight=0.7):
        """Smooth weights with edge-flow awareness — fully vectorized."""
        obj  = self.obj
        mesh = self.mesh

        grp_list = [g.name for g in obj.vertex_groups]
        n_verts  = len(mesh.vertices)
        n_grps   = len(grp_list)
        if n_grps == 0:
            return

        smooth_names = group_names if group_names is not None else grp_list
        col_idx      = {n: i for i, n in enumerate(grp_list)}
        smooth_cols  = np.array([col_idx[n] for n in smooth_names if n in col_idx], dtype=np.int32)
        if len(smooth_cols) == 0:
            return

        # Read all weights into a matrix once
        w = np.zeros((n_verts, n_grps), dtype=np.float32)
        for v in mesh.vertices:
            for g in v.groups:
                if g.group < n_grps:
                    w[v.index, g.group] = g.weight
        w_old = w.copy()

        # Build blocked edge set from topology flags
        blocked = set()
        if respect_boundaries:
            blocked.update(self.boundary_edges)
        if respect_seams:
            blocked.update(self.seam_edges)
        if respect_materials:
            blocked.update(self.material_boundary_edges)

        # Build directed edge arrays excluding blocked edges
        src_list, dst_list = [], []
        for edge in mesh.edges:
            v0, v1 = edge.vertices[0], edge.vertices[1]
            if (min(v0, v1), max(v0, v1)) not in blocked:
                src_list.append(v0); dst_list.append(v1)
                src_list.append(v1); dst_list.append(v0)

        if not src_list:
            return

        src = np.array(src_list, dtype=np.int32)
        dst = np.array(dst_list, dtype=np.int32)

        # Precompute per-directed-edge flow factors from initial weight gradient
        grad      = np.abs(w[src] - w[dst]).sum(axis=1)
        flow_bin  = np.where(grad > 0.1, 1.0, 0.3).astype(np.float32)
        ef        = (edge_flow_weight * flow_bin + (1.0 - edge_flow_weight)).astype(np.float32)

        # Weighted degree per vertex
        degree   = np.zeros(n_verts, dtype=np.float32)
        np.add.at(degree, dst, ef)
        deg_safe = np.where(degree > 0, degree, 1.0)

        # Iterative neighbour-average blend — pure numpy, no API calls inside the loop
        for _ in range(iterations):
            nbr_sum = np.zeros_like(w)
            np.add.at(nbr_sum, dst, w[src] * ef[:, np.newaxis])
            nbr_avg = nbr_sum / deg_safe[:, np.newaxis]
            w[:, smooth_cols] = (w[:, smooth_cols] * (1.0 - factor)
                                 + nbr_avg[:, smooth_cols] * factor)

        # Renormalize
        totals = w.sum(axis=1)
        valid  = totals > 0.0
        w[valid] = w[valid] / totals[valid, np.newaxis]

        _write_weights_back(obj, grp_list, w_old, w)




def _set_viewport_vertex_color():
    """Switch the active 3D viewport's solid shading to Vertex Color so heatmaps are visible."""
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.color_type = 'VERTEX'
                    break
            break


def _reset_viewport_color():
    """Restore the active 3D viewport's solid shading to Material (default)."""
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.color_type = 'MATERIAL'
                    break
            break


class WEIGHTVIS_OT_reset_shading(bpy.types.Operator):
    """Restore viewport shading back to Material (default) after viewing a heatmap"""
    bl_idname  = "object.weight_vis_reset_shading"
    bl_label   = "Reset Viewport Color"
    bl_options = {'REGISTER'}

    def execute(self, context):
        _reset_viewport_color()
        self.report({'INFO'}, "Viewport shading restored to Material")
        return {'FINISHED'}


class WeightVisualizer:
    """Weight visualization: heatmaps, gradient display, island detection, and reports."""

    def __init__(self, mesh_obj):
        if mesh_obj.type != 'MESH':
            raise TypeError("Expected a mesh object")
        self.obj = mesh_obj
        self.mesh = mesh_obj.data

    # ── numpy helpers ─────────────────────────────────────────────────────────

    def _read_group_weights(self, group_index):
        """Return per-vertex weight array for one vertex group."""
        n = len(self.mesh.vertices)
        w = np.zeros(n, dtype=np.float32)
        for v in self.mesh.vertices:
            for g in v.groups:
                if g.group == group_index:
                    w[v.index] = g.weight
                    break
        return w

    def _weights_to_rgba(self, values, color_stops):
        """Map a float32 array in [0,1] to RGBA via a color ramp (numpy)."""
        stops  = np.array(color_stops, dtype=np.float32)   # (N, 4)
        n      = len(stops)
        pos    = np.clip(values * (n - 1), 0.0, n - 1 - 1e-6)
        lo     = np.floor(pos).astype(np.int32)
        t      = (pos - lo)[:, np.newaxis]
        return (stops[lo] + (stops[lo + 1] - stops[lo]) * t).astype(np.float32)

    def _write_color_layer(self, name, rgba_loops):
        """Write an RGBA (n_loops, 4) array to a vertex color layer and set it active."""
        vc = self.mesh.vertex_colors
        if name not in vc:
            vc.new(name=name)
        layer = vc[name]
        layer.data.foreach_set("color", rgba_loops.ravel())
        vc.active = layer
        return layer

    # ── public methods ────────────────────────────────────────────────────────

    def create_weight_heatmap(self, group_name, color_scheme='RED_BLUE'):
        """Create a vertex color layer showing weight distribution."""
        group = self.obj.vertex_groups.get(group_name)
        if not group:
            return False

        schemes = {
            'RED_BLUE': [(0, 0, 1, 1), (1, 1, 0, 1), (1, 0, 0, 1)],
            'HEAT':     [(0, 0, 0, 1), (1, 0, 0, 1), (1, 1, 0, 1), (1, 1, 1, 1)],
            'GRADIENT': [(0.2, 0.2, 0.8, 1), (0.8, 0.8, 1, 1)],
        }
        stops = schemes.get(color_scheme, schemes['RED_BLUE'])

        w_vert     = self._read_group_weights(group.index)
        n_loops    = len(self.mesh.loops)
        loop_verts = np.empty(n_loops, dtype=np.int32)
        self.mesh.loops.foreach_get("vertex_index", loop_verts)

        rgba = self._weights_to_rgba(w_vert[loop_verts], stops)
        self._write_color_layer("WeightHeatmap", rgba)
        return True

    def find_influence_islands(self, group_name, min_weight=0.1, min_island_size=10):
        """Find disconnected regions of weight influence."""
        group = self.obj.vertex_groups.get(group_name)
        if not group:
            return []

        influenced = {
            v.index for v in self.mesh.vertices
            if any(g.group == group.index and g.weight >= min_weight for g in v.groups)
        }
        if not influenced:
            return []

        adjacency = defaultdict(set)
        for edge in self.mesh.edges:
            v0, v1 = edge.vertices[0], edge.vertices[1]
            if v0 in influenced and v1 in influenced:
                adjacency[v0].add(v1)
                adjacency[v1].add(v0)

        visited = set()
        islands = []
        for start in influenced:
            if start in visited:
                continue
            island = set()
            queue = [start]
            while queue:
                cur = queue.pop(0)
                if cur in visited:
                    continue
                visited.add(cur)
                island.add(cur)
                queue.extend(n for n in adjacency[cur] if n not in visited)
            if len(island) >= min_island_size:
                islands.append(island)
        return islands

    def calculate_weight_gradient(self, group_name):
        """Calculate per-vertex weight gradient magnitude."""
        group = self.obj.vertex_groups.get(group_name)
        if not group:
            return {}

        weights = {}
        for vert in self.mesh.vertices:
            weights[vert.index] = next(
                (g.weight for g in vert.groups if g.group == group.index), 0.0)

        gradients = defaultdict(float)
        for edge in self.mesh.edges:
            v1, v2 = edge.vertices
            dist = (self.mesh.vertices[v2].co - self.mesh.vertices[v1].co).length
            if dist > 0.0001:
                grad = abs(weights.get(v2, 0.0) - weights.get(v1, 0.0)) / dist
                gradients[v1] = max(gradients[v1], grad)
                gradients[v2] = max(gradients[v2], grad)
        return dict(gradients)

    def create_gradient_display(self, group_name):
        """Create a vertex color layer showing weight gradient intensity."""
        group = self.obj.vertex_groups.get(group_name)
        if not group:
            return False

        n_verts = len(self.mesh.vertices)
        n_edges = len(self.mesh.edges)
        if n_edges == 0:
            return False

        w_vert = self._read_group_weights(group.index)

        # Per-edge gradient via foreach_get
        ev = np.empty(n_edges * 2, dtype=np.int32)
        self.mesh.edges.foreach_get("vertices", ev)
        ev = ev.reshape(n_edges, 2)
        v0, v1 = ev[:, 0], ev[:, 1]

        co = np.empty(n_verts * 3, dtype=np.float32)
        self.mesh.vertices.foreach_get("co", co)
        co = co.reshape(n_verts, 3)

        dist      = np.linalg.norm(co[v1] - co[v0], axis=1)
        diff      = np.abs(w_vert[v1] - w_vert[v0])
        valid     = dist > 0.0001
        grad_edge = np.where(valid, diff / np.where(valid, dist, 1.0), 0.0).astype(np.float32)

        grad_vert = np.zeros(n_verts, dtype=np.float32)
        np.maximum.at(grad_vert, v0, grad_edge)
        np.maximum.at(grad_vert, v1, grad_edge)

        max_g = grad_vert.max()
        if max_g > 0:
            grad_vert /= max_g

        n_loops    = len(self.mesh.loops)
        loop_verts = np.empty(n_loops, dtype=np.int32)
        self.mesh.loops.foreach_get("vertex_index", loop_verts)
        t    = grad_vert[loop_verts]
        rgba = np.stack([t, 1.0 - t, np.zeros_like(t), np.ones_like(t)], axis=1).astype(np.float32)

        self._write_color_layer("GradientDisplay", rgba)
        return True

    def generate_weight_report(self, group_name):
        """Generate a statistics report for a vertex group."""
        group = self.obj.vertex_groups.get(group_name)
        if not group:
            return None

        weights = []
        influenced_count = 0
        for vert in self.mesh.vertices:
            w = next((g.weight for g in vert.groups if g.group == group.index), 0.0)
            weights.append(w)
            if w > 0.0001:
                influenced_count += 1

        if not weights:
            return None

        arr = np.array(weights)
        nonzero = arr[arr > 0]
        islands = self.find_influence_islands(group_name)
        return {
            'group_name': group_name,
            'total_vertices': len(arr),
            'influenced_vertices': influenced_count,
            'influence_percentage': (influenced_count / len(arr)) * 100,
            'min_weight': float(np.min(arr)),
            'max_weight': float(np.max(arr)),
            'mean_weight': float(np.mean(nonzero)) if len(nonzero) else 0.0,
            'std_weight': float(np.std(nonzero)) if len(nonzero) else 0.0,
            'num_islands': len(islands),
            'island_sizes': [len(isl) for isl in islands],
        }

    def cleanup(self):
        for name in ("WeightHeatmap", "GradientDisplay"):
            if name in self.mesh.vertex_colors:
                self.mesh.vertex_colors.remove(self.mesh.vertex_colors[name])


class WeightAdvProperties(bpy.types.PropertyGroup):
    active_group: bpy.props.StringProperty(name="Active Group", default="")
    prune_threshold: bpy.props.FloatProperty(
        name="Threshold", default=0.01, min=0.001, max=0.10, step=0.1,
        description="Zero out bone weights below this value then renormalize",
    )
    clamp_max: bpy.props.FloatProperty(
        name="Max Weight", default=0.80, min=0.10, max=1.0, subtype='FACTOR',
        description="Cap the active vertex group to this maximum weight per vertex",
    )
    show_visualization: bpy.props.BoolProperty(name="Visualization", default=False)


class WEIGHTADV_OT_process(bpy.types.Operator):
    """Advanced weight processing: smooth and visualize"""
    bl_idname = "object.weight_advanced_process"
    bl_label = "Advanced Weight Processing"
    bl_options = {'REGISTER', 'UNDO'}

    process_type: bpy.props.EnumProperty(
        name="Process",
        items=[
            ('SMOOTH',   "Smart Smooth",    "Edge-flow-aware weight smoothing"),
            ('HEATMAP',  "Weight Heatmap",  "Visualize weight distribution"),
            ('GRADIENT', "Gradient Display","Show weight gradient intensity"),
            ('ISLANDS',  "Find Islands",    "Find disconnected weight regions"),
            ('REPORT',   "Weight Report",   "Generate weight distribution report"),
        ],
        default='SMOOTH'
    )
    group_name: bpy.props.StringProperty(
        name="Bone Group",
        description="Vertex group to process (leave empty for all)",
        default=""
    )
    smooth_iterations: bpy.props.IntProperty(default=5, min=1, max=20)
    smooth_factor: bpy.props.FloatProperty(default=0.5, min=0.0, max=1.0)
    edge_flow_weight: bpy.props.FloatProperty(default=0.7, min=0.0, max=1.0)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "process_type")
        if self.process_type != 'REPORT':
            layout.prop(self, "group_name")
        if self.process_type == 'SMOOTH':
            layout.prop(self, "smooth_iterations")
            layout.prop(self, "smooth_factor")
            layout.prop(self, "edge_flow_weight")

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh object")
            return {'CANCELLED'}

        armature = next(
            (mod.object for mod in obj.modifiers
             if mod.type == 'ARMATURE' and mod.object), None)

        group_names = [self.group_name] if self.group_name else None

        try:
            if self.process_type == 'SMOOTH':
                SmartWeightSmoother(obj).smooth_weights(
                    group_names=group_names,
                    iterations=self.smooth_iterations,
                    factor=self.smooth_factor,
                    edge_flow_weight=self.edge_flow_weight)
                self.report({'INFO'}, "Smart smoothing complete")

            elif self.process_type == 'HEATMAP':
                target = self.group_name or (obj.vertex_groups[0].name if obj.vertex_groups else "")
                if not target:
                    self.report({'ERROR'}, "No vertex groups on mesh")
                    return {'CANCELLED'}
                WeightVisualizer(obj).create_weight_heatmap(target)
                _set_viewport_vertex_color()
                self.report({'INFO'}, f"Heatmap created for '{target}' — viewport switched to Vertex Color")

            elif self.process_type == 'GRADIENT':
                target = self.group_name or (obj.vertex_groups[0].name if obj.vertex_groups else "")
                if not target:
                    self.report({'ERROR'}, "No vertex groups on mesh")
                    return {'CANCELLED'}
                WeightVisualizer(obj).create_gradient_display(target)
                _set_viewport_vertex_color()
                self.report({'INFO'}, f"Gradient created for '{target}' — viewport switched to Vertex Color")

            elif self.process_type == 'ISLANDS':
                target = self.group_name or (obj.vertex_groups[0].name if obj.vertex_groups else "")
                if not target:
                    self.report({'ERROR'}, "No vertex groups on mesh")
                    return {'CANCELLED'}
                islands = WeightVisualizer(obj).find_influence_islands(target)
                self.report({'INFO'}, f"Found {len(islands)} influence islands")
                for i, island in enumerate(islands):
                    dbg(f"Island {i + 1}: {len(island)} vertices")

            elif self.process_type == 'REPORT':
                viz = WeightVisualizer(obj)
                groups_to_report = ([self.group_name] if self.group_name
                                    else [vg.name for vg in obj.vertex_groups[:5]])
                for gname in groups_to_report:
                    rep = viz.generate_weight_report(gname)
                    if rep:
                        dbg(f"\n=== {rep['group_name']} ===")
                        dbg(f"Vertices: {rep['influenced_vertices']} ({rep['influence_percentage']:.1f}%)")
                        dbg(f"Weight range: {rep['min_weight']:.3f} - {rep['max_weight']:.3f}")
                        dbg(f"Mean (non-zero): {rep['mean_weight']:.3f}")
                        dbg(f"Islands: {rep['num_islands']}")
                if self.group_name:
                    rep = viz.generate_weight_report(self.group_name)
                    if rep:
                        self.report({'INFO'},
                            f"{rep['group_name']}: {rep['influenced_vertices']} verts, "
                            f"mean={rep['mean_weight']:.3f}, {rep['num_islands']} islands")
                else:
                    self.report({'INFO'}, f"Report printed to console ({len(groups_to_report)} groups)")

        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        return {'FINISHED'}


# ─────────────────────────────────────────────────────────────────────────────
# CLEANUP EXTRAS — Prune, Remove Unused, Select Unweighted, Clamp
# ─────────────────────────────────────────────────────────────────────────────

class WEIGHTCLEAN_OT_prune_small(bpy.types.Operator):
    """Zero out bone weights below the threshold then renormalize"""
    bl_idname  = "object.weight_prune_small"
    bl_label   = "Prune Small Weights"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and bool(obj.vertex_groups)

    def execute(self, context):
        obj       = context.active_object
        threshold = context.scene.weight_adv_props.prune_threshold
        mesh      = obj.data
        n_verts   = len(mesh.vertices)
        grp_names = [g.name for g in obj.vertex_groups]
        n_grps    = len(grp_names)

        w = np.zeros((n_verts, n_grps), dtype=np.float32)
        for v in mesh.vertices:
            for g in v.groups:
                if g.group < n_grps:
                    w[v.index, g.group] = g.weight

        w_old    = w.copy()
        w[w < threshold] = 0.0
        totals   = w.sum(axis=1)
        valid    = totals > 0.0
        w[valid] = w[valid] / totals[valid, np.newaxis]

        _write_weights_back(obj, grp_names, w_old, w)
        pruned = int(np.count_nonzero(w_old) - np.count_nonzero(w))
        self.report({'INFO'}, f"Pruned {pruned} weight entries below {threshold:.3f} on '{obj.name}'")
        return {'FINISHED'}


class WEIGHTCLEAN_OT_remove_unused_groups(bpy.types.Operator):
    """Delete vertex groups where every vertex has zero weight"""
    bl_idname  = "object.weight_remove_unused_groups"
    bl_label   = "Remove Unused Groups"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and bool(obj.vertex_groups)

    def execute(self, context):
        obj     = context.active_object
        used    = {g.group for v in obj.data.vertices for g in v.groups if g.weight > 0.0001}
        to_del  = [vg for vg in obj.vertex_groups if vg.index not in used]
        for vg in to_del:
            obj.vertex_groups.remove(vg)
        self.report({'INFO'}, f"Removed {len(to_del)} unused group(s) from '{obj.name}'")
        return {'FINISHED'}


class WEIGHTCLEAN_OT_select_unweighted(bpy.types.Operator):
    """Select vertices that have no bone influence (total weight = 0)"""
    bl_idname  = "object.weight_select_unweighted"
    bl_label   = "Select Unweighted Verts"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH'

    def execute(self, context):
        obj  = context.active_object
        mesh = obj.data

        orig_mode = context.mode
        if orig_mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        # vertices whose total weight across all groups is zero
        unweighted = []
        for v in mesh.vertices:
            total = sum(g.weight for g in v.groups)
            if total < 0.0001:
                unweighted.append(v.index)

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT')
        for vi in unweighted:
            mesh.vertices[vi].select = True

        if orig_mode == 'PAINT_WEIGHT':
            bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
        elif orig_mode in ('EDIT', 'EDIT_MESH'):
            bpy.ops.object.mode_set(mode='EDIT')

        self.report({'INFO'}, f"Selected {len(unweighted)} unweighted vertex/vertices on '{obj.name}'")
        return {'FINISHED'}


class WEIGHTCLEAN_OT_clamp_bone(bpy.types.Operator):
    """Cap the active vertex group to the max weight, redistributing the excess to other bones"""
    bl_idname  = "object.weight_clamp_bone"
    bl_label   = "Clamp Active Bone"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (obj and obj.type == 'MESH' and obj.vertex_groups
                and obj.vertex_groups.active is not None)

    def execute(self, context):
        obj      = context.active_object
        cap      = context.scene.weight_adv_props.clamp_max
        vg       = obj.vertex_groups.active
        if not vg:
            self.report({'ERROR'}, "No active vertex group")
            return {'CANCELLED'}

        mesh      = obj.data
        n_verts   = len(mesh.vertices)
        grp_names = [g.name for g in obj.vertex_groups]
        n_grps    = len(grp_names)
        bi        = vg.index

        w = np.zeros((n_verts, n_grps), dtype=np.float32)
        for v in mesh.vertices:
            for g in v.groups:
                if g.group < n_grps:
                    w[v.index, g.group] = g.weight

        w_old    = w.copy()
        over     = w[:, bi] > cap
        if not over.any():
            self.report({'INFO'}, f"No vertices exceed {cap:.2f} on '{vg.name}'")
            return {'FINISHED'}

        excess         = w[over, bi] - cap
        w[over, bi]    = cap
        others         = w[over].copy()
        others[:, bi]  = 0.0
        other_totals   = others.sum(axis=1)
        valid          = other_totals > 0.0001
        vi_over        = np.where(over)[0]
        for i, vi in enumerate(vi_over):
            if valid[i]:
                scale = (other_totals[i] + excess[i]) / other_totals[i]
                w[vi] = w[vi].copy()
                w[vi, bi] = cap
                other_cols = [c for c in range(n_grps) if c != bi]
                w[vi][other_cols] *= scale

        # Renormalize
        totals = w.sum(axis=1)
        nz     = totals > 0.0
        w[nz]  = w[nz] / totals[nz, np.newaxis]

        _write_weights_back(obj, grp_names, w_old, w)
        self.report({'INFO'}, f"Clamped '{vg.name}' to {cap:.2f} on {over.sum()} verts")
        return {'FINISHED'}


def draw_weights_tab(layout, context):
    obj = context.active_object
    sp = context.scene.autorig_skin
    rig = _get_target_rig(context)
    in_wp = bool(obj and obj.type == 'MESH' and context.mode == 'PAINT_WEIGHT')

    # ── WEIGHT PAINT ──────────────────────────────────────────────
    paint_box = layout.box()
    paint_hdr = paint_box.row(align=True)
    paint_hdr.prop(sp, "show_paint",
                   icon='DISCLOSURE_TRI_DOWN' if sp.show_paint else 'DISCLOSURE_TRI_RIGHT',
                   emboss=False, text="")
    paint_hdr.label(text="Weight Paint", icon='WPAINT_HLT')

    if sp.show_paint:
        btn_row = paint_box.row(align=True)
        btn_row.scale_y = 1.4
        if in_wp:
            btn_row.operator("autorig.exit_weight_paint", icon='BACK')
        else:
            btn_row.enabled = bool(obj and obj.type == 'MESH')
            btn_row.operator("autorig.enter_weight_paint", icon='WPAINT_HLT')

        paint_box.separator(factor=0.3)
        def_row = paint_box.row(align=True)
        def_row.enabled = bool(rig)
        if rig:
            def_visible = _get_def_bones_visible(rig)
            def_row.operator(
                "autorig.toggle_def_bones",
                text="Hide DEF Bones" if def_visible else "Show DEF Bones",
                icon='HIDE_OFF' if def_visible else 'HIDE_ON',
            )
        else:
            def_row.label(text="No rig found", icon='INFO')

    layout.separator(factor=0.4)

    # ── CLEANUP WEIGHTS ───────────────────────────────────────────
    clean_box = layout.box()
    clean_hdr = clean_box.row(align=True)
    clean_hdr.prop(sp, "show_cleanup",
                   icon='DISCLOSURE_TRI_DOWN' if sp.show_cleanup else 'DISCLOSURE_TRI_RIGHT',
                   emboss=False, text="")
    clean_hdr.label(text="Cleanup Weights", icon='BRUSH_DATA')

    if sp.show_cleanup:
        col = clean_box.column(align=True)
        col.scale_y = 1.15

        col.label(text="Weight Operations:", icon='NORMALIZE_FCURVES')
        op = col.operator("autorig.cleanup_weights", text="Full Cleanup", icon='SHADERFX')
        op.action = 'CLEAN'
        row3 = col.row(align=True)
        op = row3.operator("autorig.cleanup_weights", text="Normalize")
        op.action = 'NORMALIZE'
        op = row3.operator("autorig.cleanup_weights", text="Limit")
        op.action = 'LIMIT'
        op = row3.operator("autorig.cleanup_weights", text="Remove Zero")
        op.action = 'REMOVE_ZERO'
        smr = col.row(align=True)
        smr.prop(sp, "joint_smooth_factor", slider=True, text="Smooth")
        smr.prop(sp, "joint_smooth_iters", text="×")
        smr.operator("autorig.smooth_joint_weights", text="Apply", icon='MOD_SMOOTH')

        col.separator(factor=0.5)
        col.label(text="Advanced Cleanup:", icon='FILTER')

        adv = context.scene.weight_adv_props

        pr = col.row(align=True)
        pr.prop(adv, "prune_threshold", text="Threshold")
        pr.operator("object.weight_prune_small", text="Prune", icon='X')

        cl = col.row(align=True)
        cl.prop(adv, "clamp_max", text="Max Weight", slider=True)
        cl.operator("object.weight_clamp_bone", text="Clamp", icon='PINNED')

        ru = col.row(align=True)
        ru.operator("object.weight_remove_unused_groups", text="Remove Unused Groups", icon='TRASH')
        ru.operator("object.weight_select_unweighted",    text="Unweighted Verts",     icon='VERTEXSEL')

        col.separator(factor=0.5)

        col.label(text="Mirror / Symmetry:", icon='MOD_MIRROR')
        sym_row = col.row(align=True)
        s = sym_row.operator("autorig.symmetrize_weights", text="Sym L→R")
        s.direction = 'NEGATIVE_X'
        s = sym_row.operator("autorig.symmetrize_weights", text="Sym R→L")
        s.direction = 'POSITIVE_X'

        col.separator(factor=0.5)

        col.label(text="Joint Cleaner:", icon='BONE_DATA')
        jrow = col.row(align=True)
        jrow.operator("autorig.joint_fix_bleeding", text="Fix Bleeding", icon='ERROR')
        jrow.operator("autorig.joint_analyze", text="Analyze", icon='VIEWZOOM')
        col.label(text="Manual Clean — Strength:")
        mrow = col.row(align=True)
        op = mrow.operator("autorig.joint_clean", text="Light")
        op.mode = 'LIGHT'
        op = mrow.operator("autorig.joint_clean", text="Moderate")
        op.mode = 'MODERATE'
        op = mrow.operator("autorig.joint_clean", text="Aggressive")
        op.mode = 'AGGRESSIVE'

        col.separator(factor=0.5)

        col.label(text="Twist Weights:", icon='FORCE_WIND')
        col.operator("autorig.smooth_twist_weights",
                     text="Smooth Twist Weights", icon='MOD_SMOOTH')

    layout.separator(factor=0.4)

    # ── SMART SMOOTHING ───────────────────────────────────────────
    box = layout.box()
    box.label(text="Smart Smoothing", icon='MOD_SMOOTH')
    col = box.column(align=True)
    op = col.operator("object.weight_advanced_process", text="Smooth Weights")
    op.process_type = 'SMOOTH'

    layout.separator(factor=0.4)

    # ── SMART WEIGHT TRANSFER ─────────────────────────────────────
    st_box   = layout.box()
    st_props = context.scene.smart_weight_props
    st_hdr   = st_box.row(align=True)
    st_hdr.prop(st_props, "show_transfer",
                icon='DISCLOSURE_TRI_DOWN' if st_props.show_transfer else 'DISCLOSURE_TRI_RIGHT',
                emboss=False, text="")
    st_hdr.label(text="Smart Weight Transfer", icon='MOD_VERTEX_WEIGHT')
    if st_props.show_transfer:
        draw_smart_transfer_section(st_box, context)


def draw_visualization_section(layout, context):
    obj = context.active_object
    adv = context.scene.weight_adv_props

    vis_box = layout.box()
    vis_hdr = vis_box.row(align=True)
    vis_hdr.prop(adv, "show_visualization",
                 icon='DISCLOSURE_TRI_DOWN' if adv.show_visualization else 'DISCLOSURE_TRI_RIGHT',
                 emboss=False, text="")
    vis_hdr.label(text="Visualization", icon='SHADING_RENDERED')

    if adv.show_visualization:
        col = vis_box.column(align=True)
        if obj and obj.vertex_groups:
            col.prop_search(adv, "active_group", obj, "vertex_groups", text="Group")
        row = col.row(align=True)
        op = row.operator("object.weight_advanced_process", text="Heatmap")
        op.process_type = 'HEATMAP'
        op.group_name   = adv.active_group
        op = row.operator("object.weight_advanced_process", text="Gradient")
        op.process_type = 'GRADIENT'
        op.group_name   = adv.active_group
        col.operator("object.weight_vis_reset_shading", text="Reset Viewport Color", icon='SHADING_SOLID')
        row = col.row(align=True)
        op = row.operator("object.weight_advanced_process", text="Find Islands")
        op.process_type = 'ISLANDS'
        op.group_name   = adv.active_group
        op = row.operator("object.weight_advanced_process", text="Report")
        op.process_type = 'REPORT'
        op.group_name   = adv.active_group


class WEIGHTADV_PT_panel(bpy.types.Panel):
    bl_label = "Advanced Weights"
    bl_idname = "WEIGHTADV_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Easy Rigify"
    bl_order = 1
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context): return False

    def draw(self, context):
        draw_weights_tab(self.layout, context)
