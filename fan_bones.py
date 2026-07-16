# fan_bones.py — Joint fan bone generation for volume preservation.
import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix
from math import pi


class FanBoneGenerator:
    """Automatic fan bone creation for joint volume preservation."""

    def __init__(self, armature_obj):
        if armature_obj.type != 'ARMATURE':
            raise TypeError("Expected an armature object")
        self.armature = armature_obj
        self.fan_bone_prefix = "FAN_"
        self.fan_bones_created = []

    def find_joint_bones(self):
        """Find deform bone pairs that form limb joints.
        Uses two strategies so every joint type is found regardless of Rigify version:

        Strategy 1 – walk-up: for each deform bone walk up through non-deform MCH/ORG
          parents to the nearest deform ancestor and check if they form a joint pair.
          Reliably catches elbow, knee, wrist, ankle via Rigify's DEF twist chain.

        Strategy 2 – name scan: explicitly search for upper_arm and thigh root bones
          (armpit and buttock).  These bones are often siblings of the shoulder/spine
          DEF bone under a shared MCH root, so the walk-up never sees them."""
        def _strip(name):
            for pfx in ('DEF-', 'ORG-', 'MCH-'):
                if name.upper().startswith(pfx):
                    return name[len(pfx):]
            return name

        def _side(name):
            n = name.lower()
            if '.l' in n or '_l' in n or n.endswith('left'):  return 'L'
            if '.r' in n or '_r' in n or n.endswith('right'): return 'R'
            return None

        deform_bones = [b for b in self.armature.data.bones if b.use_deform]
        deform_set   = {b.name for b in deform_bones}
        deform_map   = {b.name: b for b in deform_bones}
        joints       = []
        seen         = set()

        # ── Strategy 1: walk-up parent chain ─────────────────────────────────
        for bone in deform_bones:
            lower_clean = _strip(bone.name).lower()
            p = bone.parent
            while p is not None:
                if p.name in deform_set:
                    upper_clean = _strip(p.name).lower()
                    key = (p.name, bone.name)
                    if key not in seen and self._names_form_joint(upper_clean, lower_clean):
                        joints.append((p, bone))
                        seen.add(key)
                    break
                p = p.parent

        # ── Strategy 2: name scan for armpit and buttock ─────────────────────
        # lower_patterns   – what the lower bone name must contain
        # upper_patterns   – preferred upper bone name keywords (same side)
        # exclude          – suffixes that mark a twist copy, not the root bone
        ATTACHMENT_SEARCHES = [
            (['upper_arm', 'upperarm'], ['shoulder'],               ['.001', '.002']),
            (['thigh'],                 ['spine', 'pelvis', 'hip'], ['.001', '.002']),
        ]

        for lower_pats, upper_pats, excludes in ATTACHMENT_SEARCHES:
            for bone in deform_bones:
                clean = _strip(bone.name).lower()
                if not any(lp in clean for lp in lower_pats):
                    continue
                if any(e in bone.name for e in excludes):
                    continue

                side = _side(bone.name)

                # Prefer a named upper bone on the same side
                upper = None
                for bname, bobj in deform_map.items():
                    bclean = _strip(bname).lower()
                    if any(up in bclean for up in upper_pats) and _side(bname) == side:
                        upper = bobj
                        break

                # Fall back: nearest deform ancestor regardless of name
                if not upper:
                    p = bone.parent
                    while p:
                        if p.name in deform_set:
                            upper = deform_map[p.name]
                            break
                        p = p.parent

                if upper:
                    key = (upper.name, bone.name)
                    if key not in seen:
                        joints.append((upper, bone))
                        seen.add(key)

        return joints

    def _names_form_joint(self, upper_clean, lower_clean):
        for kw in ('knee', 'elbow', 'ankle', 'wrist'):
            if kw in upper_clean or kw in lower_clean:
                return True
        pairs = [
            # Elbow / knee
            ('thigh', 'shin'), ('thigh', 'calf'),
            ('upperarm', 'lowerarm'), ('upper_arm', 'lower_arm'),
            # Underarm / shoulder (armpit area)
            ('shoulder', 'upper_arm'), ('shoulder', 'upperarm'),
            # Buttock / hip (where thigh meets pelvis)
            ('spine', 'thigh'), ('pelvis', 'thigh'), ('hip', 'thigh'),
        ]
        for u_kw, l_kw in pairs:
            if u_kw in upper_clean and l_kw in lower_clean:
                return True
        # 'arm' → 'forearm' guard: DEF-forearm.L.001 contains 'arm' but is not an upper arm
        if ('arm' in upper_clean and 'forearm' not in upper_clean
                and 'forearm' in lower_clean):
            return True
        return False

    def _joint_category(self, upper_clean, lower_clean):
        """Return 'armpit', 'hip', or 'standard' for a joint pair."""
        armpit_pairs = [
            ('shoulder', 'upper_arm'), ('shoulder', 'upperarm'),
        ]
        hip_pairs = [
            ('spine', 'thigh'), ('pelvis', 'thigh'), ('hip', 'thigh'),
        ]
        for u, l in armpit_pairs:
            if u in upper_clean and l in lower_clean:
                return 'armpit'
        for u, l in hip_pairs:
            if u in upper_clean and l in lower_clean:
                return 'hip'
        return 'standard'

    def _detect_joint_axis(self, bone):
        name = bone.name.lower()
        if 'ankle' in name or 'wrist' in name:
            return 'X'
        return 'Y'

    def _fan_bone_name(self, bone_name, k):
        """Build a fan bone name with the side suffix (.L/.R) at the end so
        Blender's weight-paint symmetry recognises the pair."""
        clean = bone_name
        for pfx in ('DEF-', 'ORG-', 'MCH-'):
            if clean.upper().startswith(pfx):
                clean = clean[len(pfx):]
                break

        side = ''
        low = clean.lower()
        if low.endswith('.l') or low.endswith('.r'):
            side  = clean[-2:]
            clean = clean[:-2]
        elif '.l.' in low or '.r.' in low:
            for marker in ('.L.', '.R.', '.l.', '.r.'):
                idx = clean.find(marker)
                if idx != -1:
                    side  = clean[idx:idx+2]
                    clean = clean[:idx] + clean[idx+2:]
                    break

        return f"{self.fan_bone_prefix}{clean}_{k + 1}{side}"

    def _build_fans(self, edit_bones, upper_bone_name, bone_name, bone_head, bone_tail,
                    axis, num_fans, spread_angle):
        """Add fan edit-bones for one joint. Must already be in Edit mode.
        Fans are parented to the upper bone so they stay anchored to the upper
        limb segment when the joint bends."""
        if bone_name not in edit_bones:
            return []
        parent_name = upper_bone_name if upper_bone_name in edit_bones else bone_name

        bone_direction = (bone_tail - bone_head).normalized()
        bone_length    = (bone_tail - bone_head).length
        if bone_length < 1e-5:
            return []

        # World-up (0,0,1) as the reference gives symmetric perp1 for both L and R
        # mirrored bones: projecting it against (+dx,0,-dz) and (-dx,0,-dz) yields
        # perp1 vectors that are YZ-plane mirrors of each other with matching Z sign.
        perp1 = Vector((0, 0, 1)) - Vector((0, 0, 1)).project(bone_direction)
        if perp1.length < 0.001:
            perp1 = Vector((0, 1, 0)) - Vector((0, 1, 0)).project(bone_direction)
        perp1.normalize()

        fan_length = bone_length * 0.35
        total_fans = num_fans * 2
        angle_step = (2 * pi) / total_fans

        # Mirror the winding direction for right-side bones so that fan_N.L and
        # fan_N.R end up as true reflections across the YZ plane.  Without this,
        # rotating by +angle_step around a rightward (+X) axis gives the opposite
        # result to rotating by +angle_step around a leftward (-X) axis.
        bone_name_lo = bone_name.lower()
        if '.r' in bone_name_lo or '_r.' in bone_name_lo or bone_name_lo.endswith('_r'):
            angle_step = -angle_step

        fan_names  = []

        for k in range(total_fans):
            fan_direction = Matrix.Rotation(angle_step * k, 4, bone_direction) @ perp1
            fan_name      = self._fan_bone_name(bone_name, k)
            eb             = edit_bones.new(fan_name)
            eb.head        = bone_head.copy()
            eb.tail        = bone_head + fan_direction * fan_length
            eb.roll        = 0
            eb.parent      = edit_bones[parent_name]
            eb.use_connect = False
            eb.use_deform  = True
            fan_names.append(fan_name)

        return fan_names

    def assign_fan_weights(self, mesh_obj, lower_bone_name, upper_bone_name, fan_bone_names,
                           falloff_distance=0.2, blend_strength=0.5):
        """Redistribute joint-area weights from upper/lower limb bones to fan bones."""
        if mesh_obj.type != 'MESH' or not fan_bone_names:
            return 0

        mesh       = mesh_obj.data
        arm_bones  = self.armature.data.bones
        lower_bone = arm_bones.get(lower_bone_name)

        if not lower_bone or lower_bone_name not in mesh_obj.vertex_groups:
            return 0

        for fan_name in fan_bone_names:
            if fan_name not in mesh_obj.vertex_groups:
                mesh_obj.vertex_groups.new(name=fan_name)

        all_groups = [g.name for g in mesh_obj.vertex_groups]
        n_verts    = len(mesh.vertices)
        n_groups   = len(all_groups)
        gi_map     = {g: i for i, g in enumerate(all_groups)}

        w = np.zeros((n_verts, n_groups), dtype=np.float32)
        for v in mesh.vertices:
            for g in v.groups:
                if g.group < n_groups:
                    w[v.index, g.group] = g.weight

        co = np.empty(n_verts * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", co)
        co = co.reshape(n_verts, 3)
        mat3     = np.array(mesh_obj.matrix_world, dtype=np.float32)[:3, :3]
        loc      = np.array(mesh_obj.matrix_world.translation, dtype=np.float32)
        co_world = co @ mat3.T + loc

        joint_pt = np.array(self.armature.matrix_world @ lower_bone.head_local,
                            dtype=np.float32)
        dists = np.linalg.norm(co_world - joint_pt, axis=1)

        lower_gi   = gi_map.get(lower_bone_name, -1)
        upper_gi   = gi_map.get(upper_bone_name, -1) if upper_bone_name else -1
        lower_w    = w[:, lower_gi] if lower_gi >= 0 else np.zeros(n_verts, dtype=np.float32)
        upper_w    = w[:, upper_gi] if upper_gi >= 0 else np.zeros(n_verts, dtype=np.float32)
        combined_w = lower_w + upper_w

        mask = (dists < falloff_distance) & (combined_w > 0.01)
        if not mask.any():
            return 0

        # Squared falloff — concentrated at joint crease, fades toward limb
        t         = 1.0 - dists[mask] / falloff_distance
        ratio     = blend_strength * (t * t)
        fan_total = ratio * combined_w[mask]
        per_fan   = fan_total / len(fan_bone_names)

        if lower_gi >= 0:
            frac = np.where(combined_w[mask] > 0, lower_w[mask] / combined_w[mask], 0)
            w[mask, lower_gi] = np.maximum(0.0, lower_w[mask] - fan_total * frac)
        if upper_gi >= 0:
            frac = np.where(combined_w[mask] > 0, upper_w[mask] / combined_w[mask], 0)
            w[mask, upper_gi] = np.maximum(0.0, upper_w[mask] - fan_total * frac)

        for fan_name in fan_bone_names:
            fan_gi = gi_map.get(fan_name, -1)
            if fan_gi >= 0:
                w[mask, fan_gi] += per_fan

        totals = w.sum(axis=1)
        valid  = totals > 0
        w[valid] = w[valid] / totals[valid, np.newaxis]

        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        dl = bm.verts.layers.deform.verify()
        for bv in bm.verts:
            d   = bv[dl]
            row = w[bv.index]
            for gi in range(n_groups):
                val = float(row[gi])
                if val > 0.0001:
                    d[gi] = val
                elif gi in d:
                    del d[gi]
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

        return int(mask.sum())

    def generate_all_fans(self, mesh_obj=None, axis='AUTO', num_fans=2, spread_angle=30.0,
                          falloff_distance=0.2, blend_strength=0.5,
                          include_armpit=False, include_hip=False):
        """Generate fan bones for all detected joints and optionally assign weights."""
        raw_joints = self.find_joint_bones()
        if not raw_joints:
            return []

        def _strip(name):
            for pfx in ('DEF-', 'ORG-', 'MCH-'):
                if name.upper().startswith(pfx):
                    return name[len(pfx):]
            return name

        filtered = []
        for upper, lower in raw_joints:
            cat = self._joint_category(_strip(upper.name).lower(), _strip(lower.name).lower())
            if cat == 'armpit' and not include_armpit:
                continue
            if cat == 'hip' and not include_hip:
                continue
            filtered.append((upper, lower))

        raw_joints = filtered
        if not raw_joints:
            return []

        # Capture bone data before any mode switch invalidates Bone references
        joint_data = []
        for upper, lower in raw_joints:
            joint_data.append({
                'upper_name': upper.name,
                'name':       lower.name,
                'head':       lower.head_local.copy(),
                'tail':       lower.tail_local.copy(),
                'axis':       self._detect_joint_axis(lower) if axis == 'AUTO' else axis,
            })

        # Single edit-mode session for all joints
        bpy.context.view_layer.objects.active = self.armature
        original_mode = self.armature.mode
        if original_mode != 'EDIT':
            bpy.ops.object.mode_set(mode='EDIT')

        edit_bones = self.armature.data.edit_bones
        created = []
        for jd in joint_data:
            fan_names = self._build_fans(
                edit_bones, jd['upper_name'], jd['name'], jd['head'], jd['tail'],
                jd['axis'], num_fans, spread_angle)
            created.append((jd['upper_name'], jd['name'], fan_names))
            self.fan_bones_created.extend(fan_names)

        if original_mode != 'EDIT':
            bpy.ops.object.mode_set(mode=original_mode)

        if mesh_obj:
            mesh_objects = self._find_all_deformed_meshes(self.armature)
            if not mesh_objects:
                mesh_objects = [mesh_obj]
            for m_obj in mesh_objects:
                for upper_name, lower_name, fan_names in created:
                    if fan_names:
                        self.assign_fan_weights(m_obj, lower_name, upper_name, fan_names,
                                                falloff_distance=falloff_distance,
                                                blend_strength=blend_strength)

        # Copy Rotation constraint — blends each fan halfway between upper and lower
        # limb rotation for natural volume preservation in the Blender viewport.
        bpy.context.view_layer.objects.active = self.armature
        bpy.ops.object.mode_set(mode='POSE')
        pb_map = self.armature.pose.bones
        for upper_name, lower_name, fan_names in created:
            for fan_name in fan_names:
                pb = pb_map.get(fan_name)
                if not pb:
                    continue
                c              = pb.constraints.new('COPY_ROTATION')
                c.name         = "Fan Joint Blend"
                c.target       = self.armature
                c.subtarget    = lower_name
                c.target_space = 'LOCAL'
                c.owner_space  = 'LOCAL'
                c.mix_mode     = 'ADD'
                c.influence    = 0.5
        bpy.ops.object.mode_set(mode='OBJECT')

        return created

    def _find_all_deformed_meshes(self, armature_obj):
        return [
            obj for obj in bpy.data.objects
            if obj.type == 'MESH'
            and any(m.type == 'ARMATURE' and m.object == armature_obj
                    for m in obj.modifiers)
        ]

    def _find_deformed_mesh(self, armature_obj):
        meshes = self._find_all_deformed_meshes(armature_obj)
        return meshes[0] if meshes else None


class FANBONE_OT_generate(bpy.types.Operator):
    """Generate fan bones for joint volume preservation"""
    bl_idname = "armature.generate_fan_bones"
    bl_label  = "Generate Fan Bones"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        armature = context.active_object
        if not armature or armature.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature first")
            return {'CANCELLED'}

        props     = context.scene.fan_bone_props
        generator = FanBoneGenerator(armature)
        mesh_obj  = None
        if props.auto_assign_weights:
            mesh_obj = generator._find_deformed_mesh(armature)
            if not mesh_obj:
                self.report({'WARNING'},
                            "No mesh found with this armature. Fan bones created without weights.")
        try:
            result = generator.generate_all_fans(
                mesh_obj=mesh_obj if props.auto_assign_weights else None,
                axis=props.joint_axis,
                num_fans=props.num_fans,
                falloff_distance=props.falloff_distance,
                blend_strength=props.blend_strength,
                include_armpit=props.include_armpit,
                include_hip=props.include_hip,
            )
            if not result:
                self.report({'WARNING'}, "No joint bones found to add fans to")
                return {'CANCELLED'}
            total_fans = sum(len(fans) for _u, _l, fans in result)
            msg = f"Created {total_fans} fan bones across {len(result)} joints"
            if props.auto_assign_weights and mesh_obj:
                all_meshes = generator._find_all_deformed_meshes(armature)
                mesh_names = ", ".join(o.name for o in all_meshes) if all_meshes else mesh_obj.name
                msg += f" — weights applied to: {mesh_names}"
            self.report({'INFO'}, msg)
        except Exception as e:
            self.report({'ERROR'}, f"Fan bone generation failed: {e}")
            return {'CANCELLED'}

        return {'FINISHED'}

class FANBONE_OT_remove(bpy.types.Operator):
    """Remove all generated fan bones"""
    bl_idname = "armature.remove_fan_bones"
    bl_label  = "Remove Fan Bones"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        armature = context.active_object
        if not armature or armature.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature first")
            return {'CANCELLED'}

        fan_bone_names = [b.name for b in armature.data.bones
                          if b.name.startswith("FAN_")]
        if not fan_bone_names:
            self.report({'WARNING'}, "No fan bones found")
            return {'CANCELLED'}

        original_mode = armature.mode
        bpy.ops.object.mode_set(mode='EDIT')
        edit_bones = armature.data.edit_bones
        removed = [n for n in fan_bone_names if n in edit_bones]
        for name in removed:
            edit_bones.remove(edit_bones[name])
        bpy.ops.object.mode_set(mode=original_mode)

        if context.scene.fan_bone_props.remove_weights:
            for obj in bpy.data.objects:
                if obj.type == 'MESH':
                    for mod in obj.modifiers:
                        if mod.type == 'ARMATURE' and mod.object == armature:
                            for name in fan_bone_names:
                                vg = obj.vertex_groups.get(name)
                                if vg:
                                    obj.vertex_groups.remove(vg)

        self.report({'INFO'}, f"Removed {len(removed)} fan bones")
        return {'FINISHED'}


class FanBoneProperties(bpy.types.PropertyGroup):
    joint_axis: bpy.props.EnumProperty(
        name="Joint Axis",
        items=[
            ('AUTO', "Auto Detect", "Automatically detect joint axis"),
            ('X', "X Axis", "Primary bend on X axis"),
            ('Y', "Y Axis", "Primary bend on Y axis"),
            ('Z', "Z Axis", "Primary bend on Z axis"),
        ],
        default='AUTO',
    )
    num_fans: bpy.props.IntProperty(
        name="Fans Per Side", default=2, min=1, max=4,
        description="Number of fan bones on each side of the joint",
    )
    include_armpit: bpy.props.BoolProperty(
        name="Armpit / Shoulder", default=False,
        description="Add fan bones at the shoulder/armpit joint (upper arm meets shoulder)",
    )
    include_hip: bpy.props.BoolProperty(
        name="Hip / Buttock", default=False,
        description="Add fan bones at the hip/buttock joint (thigh meets pelvis or spine)",
    )
    auto_assign_weights: bpy.props.BoolProperty(
        name="Auto-Assign Weights", default=True,
        description="Automatically transfer weights to fan bones",
    )
    falloff_distance: bpy.props.FloatProperty(
        name="Falloff Distance", default=0.2, min=0.05, max=5.0,
        description="Radius around the joint within which vertices are affected",
    )
    blend_strength: bpy.props.FloatProperty(
        name="Blend Strength", default=0.5, min=0.05, max=1.0, subtype='FACTOR',
        description="How much weight is transferred from limb bones to fan bones at the joint centre",
    )
    remove_weights: bpy.props.BoolProperty(
        name="Remove Weights on Delete", default=True,
        description="Also remove fan bone vertex groups when removing fan bones",
    )
    show_fan_bones: bpy.props.BoolProperty(name="Fan Bones", default=True)


# ---------------------------------------------------------------------------
# Tab draw helper — called from AUTORIG_PT_Main (Tools tab)
# ---------------------------------------------------------------------------

def draw_fan_bones_section(layout, context):
    props = context.scene.fan_bone_props

    box     = layout.box()
    box_hdr = box.row(align=True)
    box_hdr.prop(props, "show_fan_bones",
                 icon='DISCLOSURE_TRI_DOWN' if props.show_fan_bones else 'DISCLOSURE_TRI_RIGHT',
                 emboss=False, text="")
    box_hdr.label(text="Joint Fan Bones", icon='BONE_DATA')

    if props.show_fan_bones:
        col = box.column(align=True)
        col.prop(props, "joint_axis")
        col.prop(props, "num_fans")

        box.separator()
        box.label(text="Optional Joints:")
        col = box.column(align=True)
        col.prop(props, "include_armpit")
        col.prop(props, "include_hip")

        box.separator()
        box.prop(props, "auto_assign_weights")
        if props.auto_assign_weights:
            sub = box.column(align=True)
            sub.prop(props, "falloff_distance")
            sub.prop(props, "blend_strength")

        box.separator()
        row = box.row(align=True)
        row.scale_y = 1.3
        row.operator("armature.generate_fan_bones", text="Generate Fan Bones", icon='BONE_DATA')

        armature = context.active_object
        if armature and armature.type == 'ARMATURE':
            fan_count = sum(1 for b in armature.data.bones if b.name.startswith("FAN_"))
            if fan_count > 0:
                box.separator()
                box.label(text=f"{fan_count} fan bones active")
                box.prop(props, "remove_weights")
                box.operator("armature.remove_fan_bones", text="Remove All Fan Bones", icon='X')


class FANBONE_PT_panel(bpy.types.Panel):
    bl_label       = "Fan Bones"
    bl_idname      = "FANBONE_PT_panel"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = "Easy Rigify"
    bl_order       = 3
    bl_options     = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context): return False

    def draw(self, context):
        draw_fan_bones_section(self.layout, context)
