# pipeline.py — Deformation pipeline, SurgicalWeightFixer, SmartBind, and skinning panel.
import bpy
from .constants import dbg
import bmesh
import numpy as np
from mathutils import Vector, kdtree
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from .utils import (
    WeightDataAccess, _write_weights_back,
    _set_def_bones_visible, _get_def_bones_visible, _reorder_arm_modifier,
)


def _is_generated_rig(obj):
    """True if this armature is a Rigify-generated rig (not a metarig)."""
    if 'rig_id' in obj:
        return True
    # Fallback: generated rigs always have DEF- deform bones; metarigs never do
    return bool(obj.data and any(b.name.startswith('DEF-') for b in obj.data.bones))


def _get_target_rig(context):
    """Return the generated Rigify rig if present, otherwise any armature."""
    # Prefer a confirmed generated rig
    for obj in context.scene.objects:
        if obj.type == 'ARMATURE' and _is_generated_rig(obj):
            return obj
    # Fall back to any armature so the UI banner can still show a status
    for obj in context.scene.objects:
        if obj.type == 'ARMATURE':
            return obj
    return None


# Vertex group name keywords allowed per mesh type after auto-weight (None = keep all).
_BIND_BONE_FILTER = {
    'BODY':     None,
    'CLOTHING': None,
    'EYES':     ('eye', 'master_eye'),
    'TEETH':    ('teeth', 'tooth'),
    'TONGUE':   ('tongue',),
    # EYE_L/EYE_R/TEETH_T/TEETH_B use direct assignment — filter not consulted.
}

# Fallback bone priority when no allowed groups survive after stripping.
# DEF-jaw is intentionally excluded from TEETH and TONGUE — the jaw should
# never drive teeth or tongue geometry; they are controlled by their own bones.
_BIND_FALLBACK_BONES = {
    'EYES':   ['DEF-eye.L', 'DEF-eye.R', 'DEF-spine.006'],
    'TEETH':  ['DEF-teeth.T', 'DEF-teeth.B', 'DEF-spine.006'],
    'TONGUE': ['DEF-tongue', 'DEF-tongue.001', 'DEF-tongue.002', 'DEF-spine.006'],
}

# Rigid mesh types: skip auto-weight, assign 100% to this exact bone.
_BIND_DIRECT_BONE = {
    'EYE_L':   'DEF-eye.L',
    'EYE_R':   'DEF-eye.R',
    'TEETH_T': 'DEF-teeth.T',
    'TEETH_B': 'DEF-teeth.B',
}

# DEF-bone lists per region for generated Rigify rigs.
_REGION_BONES = {
    'SPINE': [
        'DEF-pelvis.L', 'DEF-pelvis.R',
        'DEF-spine', 'DEF-spine.001', 'DEF-spine.002',
        'DEF-spine.003', 'DEF-spine.004', 'DEF-spine.005', 'DEF-spine.006',
    ],
    'ARMS': [
        'DEF-shoulder.L', 'DEF-upper_arm.L', 'DEF-upper_arm.L.001',
        'DEF-forearm.L', 'DEF-forearm.L.001', 'DEF-hand.L',
        'DEF-shoulder.R', 'DEF-upper_arm.R', 'DEF-upper_arm.R.001',
        'DEF-forearm.R', 'DEF-forearm.R.001', 'DEF-hand.R',
    ],
    'LEGS': [
        'DEF-thigh.L', 'DEF-thigh.L.001', 'DEF-shin.L', 'DEF-shin.L.001',
        'DEF-foot.L', 'DEF-toe.L',
        'DEF-thigh.R', 'DEF-thigh.R.001', 'DEF-shin.R', 'DEF-shin.R.001',
        'DEF-foot.R', 'DEF-toe.R',
    ],
    'FINGERS': [
        'DEF-palm.01.L', 'DEF-palm.02.L', 'DEF-palm.03.L', 'DEF-palm.04.L',
        'DEF-f_index.01.L', 'DEF-f_index.02.L', 'DEF-f_index.03.L',
        'DEF-f_middle.01.L', 'DEF-f_middle.02.L', 'DEF-f_middle.03.L',
        'DEF-f_ring.01.L', 'DEF-f_ring.02.L', 'DEF-f_ring.03.L',
        'DEF-f_pinky.01.L', 'DEF-f_pinky.02.L', 'DEF-f_pinky.03.L',
        'DEF-thumb.01.L', 'DEF-thumb.02.L', 'DEF-thumb.03.L',
        'DEF-palm.01.R', 'DEF-palm.02.R', 'DEF-palm.03.R', 'DEF-palm.04.R',
        'DEF-f_index.01.R', 'DEF-f_index.02.R', 'DEF-f_index.03.R',
        'DEF-f_middle.01.R', 'DEF-f_middle.02.R', 'DEF-f_middle.03.R',
        'DEF-f_ring.01.R', 'DEF-f_ring.02.R', 'DEF-f_ring.03.R',
        'DEF-f_pinky.01.R', 'DEF-f_pinky.02.R', 'DEF-f_pinky.03.R',
        'DEF-thumb.01.R', 'DEF-thumb.02.R', 'DEF-thumb.03.R',
    ],
    'FACE': [
        'DEF-jaw', 'DEF-jaw.L', 'DEF-jaw.R', 'DEF-jaw.L.001', 'DEF-jaw.R.001',
        'DEF-chin', 'DEF-chin.L', 'DEF-chin.R',
        'DEF-cheek.T.L', 'DEF-cheek.T.L.001', 'DEF-cheek.T.R', 'DEF-cheek.T.R.001',
        'DEF-cheek.B.L', 'DEF-cheek.B.L.001', 'DEF-cheek.B.R', 'DEF-cheek.B.R.001',
        'DEF-brow.T.L', 'DEF-brow.T.L.001', 'DEF-brow.T.L.002', 'DEF-brow.T.L.003',
        'DEF-brow.T.R', 'DEF-brow.T.R.001', 'DEF-brow.T.R.002', 'DEF-brow.T.R.003',
        'DEF-lid.T.L', 'DEF-lid.T.L.001', 'DEF-lid.T.L.002', 'DEF-lid.T.L.003',
        'DEF-lid.T.R', 'DEF-lid.T.R.001', 'DEF-lid.T.R.002', 'DEF-lid.T.R.003',
        'DEF-lid.B.L', 'DEF-lid.B.L.001', 'DEF-lid.B.L.002', 'DEF-lid.B.L.003',
        'DEF-lid.B.R', 'DEF-lid.B.R.001', 'DEF-lid.B.R.002', 'DEF-lid.B.R.003',
        'DEF-nose', 'DEF-nose.001', 'DEF-nose.L', 'DEF-nose.L.001',
        'DEF-nose.R', 'DEF-nose.R.001',
        'DEF-lip.T.L', 'DEF-lip.T.L.001', 'DEF-lip.T.R', 'DEF-lip.T.R.001',
        'DEF-lip.B.L', 'DEF-lip.B.L.001', 'DEF-lip.B.R', 'DEF-lip.B.R.001',
        'DEF-forehead.L', 'DEF-forehead.L.001', 'DEF-forehead.L.002',
        'DEF-forehead.R', 'DEF-forehead.R.001', 'DEF-forehead.R.002',
        'DEF-temple.L', 'DEF-temple.R',
        'DEF-ear.L', 'DEF-ear.L.001', 'DEF-ear.R', 'DEF-ear.R.001',
        'DEF-teeth.T', 'DEF-teeth.B',
        'DEF-tongue', 'DEF-tongue.001', 'DEF-tongue.002',
    ],
}

# Keyword fallback for metarig (no DEF- prefix).
_REGION_KEYWORDS = {
    'SPINE':   ('spine', 'chest', 'neck', 'head', 'pelvis'),
    'ARMS':    ('shoulder', 'upper_arm', 'forearm', 'hand'),
    'LEGS':    ('thigh', 'shin', 'foot', 'toe', 'heel'),
    'FINGERS': ('thumb', 'f_index', 'f_middle', 'f_ring', 'f_pinky', 'palm'),
    'FACE':    ('jaw', 'chin', 'cheek', 'brow', 'lid', 'nose', 'lip',
                'forehead', 'temple', 'ear', 'tongue', 'teeth'),
}


class AutoRigMeshItem(bpy.types.PropertyGroup):
    obj:       bpy.props.PointerProperty(name="Object", type=bpy.types.Object)
    enabled:   bpy.props.BoolProperty(name="Include", default=True)
    mesh_type: bpy.props.EnumProperty(
        name="Type",
        items=[
            ('BODY',     'Body',          'Main body — full weight painting'),
            ('CLOTHING', 'Clothing',      'Clothing/hair — inherits full body weights'),
            ('EYE_L',    'Eye Left',      'Left eyeball — 100% weighted to DEF-eye.L'),
            ('EYE_R',    'Eye Right',     'Right eyeball — 100% weighted to DEF-eye.R'),
            ('EYES',     'Eyes (Both)',   'Both eyeballs in one mesh — weighted to eye bones'),
            ('TEETH_T',  'Teeth Top',     'Upper teeth — 100% weighted to DEF-teeth.T'),
            ('TEETH_B',  'Teeth Bottom',  'Lower teeth — 100% weighted to DEF-teeth.B'),
            ('TEETH',    'Teeth (Both)',  'Teeth in one mesh — weighted to teeth bones'),
            ('TONGUE',   'Tongue',        'Tongue — weighted to DEF-tongue bones'),
        ],
        default='BODY',
    )


class AutoRigSkinProps(bpy.types.PropertyGroup):
    meshes:           bpy.props.CollectionProperty(type=AutoRigMeshItem)
    show_mesh_setup:  bpy.props.BoolProperty(name="Mesh Setup", default=True)
    bind_method: bpy.props.EnumProperty(
        name="Bind Method",
        description="Algorithm used to compute automatic weights",
        items=[
            ('AUTO',        'Automatic Weight',
             'Automatic weights via heat diffusion — best all-round result'),
            ('ENVELOPE',    'Envelope',
             'Bone envelope volumes — fast, use when Automatic Weight fails'),
            ('PIPELINE',    'Heat Map',
             'Full multi-stage pipeline: heat diffusion → joint refinement → '
             'smooth → sharpen → normalize → limit influences'),
        ],
        default='AUTO',
    )
    smooth_iterations: bpy.props.IntProperty(
        name="Smooth Iterations", default=3, min=1, max=20,
        description="Number of weight smoothing passes in the pipeline")
    smooth_factor: bpy.props.FloatProperty(
        name="Smooth Factor", default=0.3, min=0.0, max=1.0, subtype='FACTOR',
        description="Blend amount toward neighbor average per smooth pass")
    split_parts: bpy.props.BoolProperty(
        name="Split Parts",
        default=False,
        description="Separate disconnected mesh islands and bind each independently — "
                    "essential for split teeth, accessories, multi-part meshes")
    fix_scale: bpy.props.BoolProperty(
        name="Fix Scale",
        default=False,
        description="Scale mesh and armature 10× before binding — fixes heat diffusion "
                    "failures on very small or imported characters (auto-reverts after bind)")
    max_influences: bpy.props.IntProperty(
        name="Max Influences", default=4, min=1, max=8,
        description="Maximum bones influencing each vertex")
    preserve_volume: bpy.props.BoolProperty(
        name="Preserve Volume",
        default=False,
        description="Use dual-quaternion blending on the Armature modifier — "
                    "prevents candy-wrapper twisting at elbows and wrists")
    bind_report:    bpy.props.StringProperty(default="")
    def_bones_were_visible: bpy.props.BoolProperty(default=False)
    show_cleanup:        bpy.props.BoolProperty(name="Cleanup", default=False)
    show_paint:          bpy.props.BoolProperty(name="Paint Tools", default=False)
    show_tighten:        bpy.props.BoolProperty(name="Joint Tightening", default=False)
    show_pipeline_opts:  bpy.props.BoolProperty(name="Weight Options", default=False)
    show_smart_bind:     bpy.props.BoolProperty(name="Smart Bind",     default=True)
    joint_smooth_factor: bpy.props.FloatProperty(
        name="Factor", default=0.3, min=0.0, max=1.0, subtype='FACTOR',
        description="Blend strength per pass — higher = more smoothing")
    joint_smooth_iters:  bpy.props.IntProperty(
        name="Passes", default=1, min=1, max=5,
        description="Number of smoothing passes")
    optimize_highres: bpy.props.BoolProperty(
        name="Optimize High Res",
        default=False,
        description="Temporarily decimate the mesh before binding, then transfer weights back. "
                    "Speeds up auto-weight on meshes with 100k+ vertices")


class AUTORIG_OT_DetectMeshes(bpy.types.Operator):
    """Scan the scene for mesh objects and list them ready for binding"""
    bl_idname  = "autorig.detect_meshes"
    bl_label   = "⑦ Detect Character Meshes"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        sp  = context.scene.autorig_skin
        sp.meshes.clear()
        sp.bind_report = ""
        rig = _get_target_rig(context)

        skip = set()
        if rig:
            skip.add(rig.name)
        for col_name in ("RigifyMarkers", "RigifyFaceMarkers"):
            col = bpy.data.collections.get(col_name)
            if col:
                for o in col.objects:
                    skip.add(o.name)
        # Skip Rigify widget collections (WGTS_*)
        for col in bpy.data.collections:
            if col.name.startswith("WGTS_"):
                for o in col.objects:
                    skip.add(o.name)

        for obj in sorted(context.scene.objects, key=lambda o: o.name):
            if obj.type != 'MESH' or obj.name in skip:
                continue
            item = sp.meshes.add()
            item.obj     = obj
            item.enabled = True
            n = obj.name.lower()
            if any(k in n for k in ('eye', 'eyeball', 'cornea', 'iris', 'pupil')):
                item.mesh_type = 'EYES'
            elif any(k in n for k in ('teeth', 'tooth', 'gum')):
                item.mesh_type = 'TEETH'
            elif 'tongue' in n:
                item.mesh_type = 'TONGUE'
            elif any(k in n for k in ('cloth', 'shirt', 'pant', 'jacket',
                                       'dress', 'hair', 'shoe', 'boot')):
                item.mesh_type = 'CLOTHING'
            else:
                item.mesh_type = 'BODY'

        msg = f"Found {len(sp.meshes)} meshes"
        if rig:
            is_gen = _is_generated_rig(rig)
            msg += f" — {'Generated rig' if is_gen else 'Metarig'}: {rig.name}"
        else:
            msg += " — WARNING: no armature in scene"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class SurgicalSectionProps(bpy.types.PropertyGroup):
    enabled: bpy.props.BoolProperty(name="Enabled", default=True)
    feather: bpy.props.FloatProperty(
        name="Feather", default=2.0, min=0.1, max=2.0,
        description="Controls bleed reduction: 2.0=no change, 1.0=reduce bleed to 50%, 0.1=reduce to 5%"
    )


def _fit_cone_radius(proj, perp, L, pct=0.90, floor_frac=0.05, cap_frac=0.6):
    """Fit a TAPERED (cone) radius to a limb from per-vertex axial position `proj`
    and perpendicular distance `perp` (both 1-D arrays, same length), bone length `L`.

    A limb is a truncated cone (wide at the shoulder/elbow, narrow at the wrist), so a
    single radius either lets bleed through at the narrow end or clips real weight at
    the wide end. We take a high percentile of `perp` in the proximal and distal halves,
    place each at its band centre (0.25L / 0.75L), and extrapolate a line to the bone
    ends -> (R_head, R_tail). Sparse bands fall back to a single radius.

    Returns (R_head, R_tail), each clamped to [L*floor_frac, L*cap_frac].
    """
    floor = L * floor_frac
    cap   = L * cap_frac

    def _band(lo, hi):
        m = (proj >= lo * L) & (proj < hi * L)
        if int(np.count_nonzero(m)) < 4:
            return None
        ps = np.sort(perp[m])
        return float(ps[min(int(len(ps) * pct), len(ps) - 1)])

    rp = _band(0.0, 0.5)
    rd = _band(0.5, 1.0)
    if rp is None and rd is None:
        if len(perp) == 0:
            return floor, floor
        ps = np.sort(perp)
        R  = float(ps[min(int(len(ps) * pct), len(ps) - 1)])
        R  = min(max(R, floor), cap)
        return R, R
    if rp is None:
        rp = rd
    if rd is None:
        rd = rp
    slope  = (rd - rp) / (0.5 * L)
    r_head = rp - slope * 0.25 * L
    r_tail = rd + slope * 0.25 * L
    return (min(max(r_head, floor), cap), min(max(r_tail, floor), cap))


class SurgicalWeightFixer:
    """Per-bone cylinder tightening. feather=2.0 → no change; 0.1 → 95% reduction."""

    def __init__(self, mesh_obj, armature_obj):
        self.mesh_obj     = mesh_obj
        self.armature_obj = armature_obj
        self.mesh         = mesh_obj.data
        self.armature     = armature_obj.data
        self._vert_pos    = None  # cached on first call to _get_vert_positions()

        self.shoulder_bones   = []
        self.upper_arm_bones  = []
        self.forearm_bones    = []
        self.thigh_bones      = []
        self.shin_bones       = []
        self.finger_bones     = []
        self.spine_bones      = []
        self.neck_bones       = []
        self._classify_bones()

    def _strip_prefix(self, name):
        for prefix in ['def-', 'org-', 'mch-', 'wgt-', 'mixamorig:', 'arp_', 'c_', 'x_']:
            if name.startswith(prefix):
                return name[len(prefix):]
        return name

    def _is_finger(self, name):
        return any(p in name for p in
                   ['thumb', 'index', 'middle', 'ring', 'pinky',
                    'f_index', 'f_middle', 'f_ring', 'f_pinky', 'finger', 'digit'])

    def _classify_bones(self):
        for bone in self.armature.bones:
            if not bone.use_deform:
                continue
            clean = self._strip_prefix(bone.name.lower())
            if 'shoulder' in clean or 'clavicle' in clean:
                self.shoulder_bones.append(bone.name)
            elif 'upper_arm' in clean or 'upperarm' in clean or 'biceps' in clean:
                self.upper_arm_bones.append(bone.name)
            elif 'forearm' in clean or 'lower_arm' in clean or 'lowerarm' in clean:
                self.forearm_bones.append(bone.name)
            elif 'thigh' in clean or 'upper_leg' in clean or 'upperleg' in clean:
                self.thigh_bones.append(bone.name)
            elif 'shin' in clean or 'lower_leg' in clean or 'lowerleg' in clean or 'calf' in clean:
                self.shin_bones.append(bone.name)
            elif 'spine' in clean or 'back' in clean:
                self.spine_bones.append(bone.name)
            elif 'neck' in clean or 'cervical' in clean:
                self.neck_bones.append(bone.name)
            elif self._is_finger(clean):
                self.finger_bones.append(bone.name)

    def _get_bone_side(self, bone_name):
        name = bone_name.lower()
        for p in ['.r', '_r', 'right']:
            if p in name:
                return 'RIGHT'
        for p in ['.l', '_l', 'left']:
            if p in name:
                return 'LEFT'
        if bone_name in self.armature.bones:
            bone = self.armature.bones[bone_name]
            head = self.armature_obj.matrix_world @ bone.head_local
            if head.x > 0.01:
                return 'LEFT'
            elif head.x < -0.01:
                return 'RIGHT'
        return None

    def _get_vert_positions(self):
        if self._vert_pos is None:
            self._vert_pos = np.array(
                [v.co for v in self.mesh.vertices], dtype=np.float32)
        return self._vert_pos

    def _get_bone_cylinder(self, bone_name):
        if bone_name not in self.armature.bones:
            return None
        bone  = self.armature.bones[bone_name]
        m_inv = self.mesh_obj.matrix_world.inverted()
        head  = m_inv @ (self.armature_obj.matrix_world @ bone.head_local)
        tail  = m_inv @ (self.armature_obj.matrix_world @ bone.tail_local)
        seg   = tail - head
        length = seg.length
        if length < 1e-6:
            return None
        direction = seg.normalized()
        head_np = np.array(head,      dtype=np.float32)
        dir_np  = np.array(direction, dtype=np.float32)
        r_head, r_tail = self._estimate_bone_radius_fast(head_np, dir_np, length)
        return {'head': head_np, 'direction': dir_np, 'length': length,
                'r_head': r_head, 'r_tail': r_tail,
                'radius': 0.5 * (r_head + r_tail)}  # back-compat for any 'radius' reader

    def _estimate_bone_radius_fast(self, head_np, dir_np, length):
        """TAPERED (cone) radius (r_head, r_tail) from sampled vertices along the
        whole bone. A limb tapers, so a single radius mis-fits one end."""
        vp   = self._get_vert_positions()
        step = max(1, len(self.mesh.vertices) // 500)   # finer sample than before
        vp_s = vp[::step]
        to_v = vp_s - head_np
        proj = (to_v * dir_np).sum(axis=1)

        # Vertices spanning the bone (exclude the very ends where caps flare).
        in_ax = (proj > 0.05 * length) & (proj < 0.95 * length)
        if not in_ax.any():
            return length * 0.20, length * 0.20

        p_ax     = proj[in_ax]
        cl       = np.clip(p_ax, 0.0, length)[:, np.newaxis]
        perp     = np.linalg.norm(vp_s[in_ax] - (head_np + dir_np * cl), axis=1)

        # Discard distant vertices (other limbs / torso at the same height).
        cap  = length * 0.40
        near = perp <= cap
        if not near.any():
            return length * 0.15, length * 0.15

        return _fit_cone_radius(p_ax[near], perp[near], length, pct=0.80)

    def tighten_section(self, weights, group_names, bone_list, feather, center_x):
        """Combined cylinder + dominance tightening.

        Three zones per bone:
          Core   (inside cylinder)     — bone's own territory, untouched.
          Feather ring (just outside)  — smooth gradient from 1.0 → multiplier
                                         so the cylinder edge is never a hard cut.
          Outer zone (beyond ring)     — dominance check: only reduce if this bone
                                         is NOT the dominant bone at that vertex.
                                         Protects joint transitions where adjacent
                                         bone cylinders overlap.

        feather=2.0 → no change; feather=0.1 → bleeding reduced to 5%.
        """
        if feather >= 2.0:
            return 0

        # feather [0.1, 2.0] → multiplier [0.05, 1.0]
        t          = max(0.0, min(1.0, (feather - 0.1) / 1.9))
        multiplier = float(0.05 + t * 0.95)

        vp    = self._get_vert_positions()
        fixed = 0

        # Snapshot the dominant bone per vertex ONCE, before any reduction, so the
        # result is order-independent — sibling bones that share a joint (finger
        # segments) won't see each other's reductions and flip the bleeding test.
        dominant_at = np.argmax(weights, axis=1)

        for bone_name in bone_list:
            if bone_name not in group_names:
                continue
            bi = group_names.index(bone_name)

            wi = np.where(weights[:, bi] > 0.001)[0]
            if len(wi) == 0:
                continue

            cylinder = self._get_bone_cylinder(bone_name)

            if cylinder is None:
                # No bone geometry — fall back to pure dominance check
                bleeding = dominant_at[wi] != bi
                if bleeding.any():
                    bidx  = wi[bleeding]
                    old_w = weights[bidx, bi].copy()
                    new_w = old_w * multiplier
                    new_w[new_w < 0.001] = 0.0
                    weights[bidx, bi] = new_w
                    fixed += int((new_w != old_w).sum())
                continue

            verts  = vp[wi]
            head   = cylinder['head']
            direc  = cylinder['direction']
            L      = cylinder['length']
            r_head = cylinder['r_head']
            r_tail = cylinder['r_tail']

            to_v  = verts - head
            proj  = (to_v * direc).sum(axis=1)
            cl    = np.clip(proj, 0.0, L)[:, np.newaxis]
            perp  = np.linalg.norm(verts - (head + direc * cl), axis=1)

            # Per-vertex tapered (cone) radius along the bone.
            tt    = np.clip(proj / L, 0.0, 1.0)
            R     = r_head + (r_tail - r_head) * tt
            Rmin  = min(r_head, r_tail)
            pad   = Rmin * 0.3                 # axial buffer at bone ends
            fend  = R * (1.0 + feather)        # outer edge of feather ring (per-vert)

            in_axl     = (proj >= -pad) & (proj <= L + pad)
            in_core    = in_axl & (perp <= R)
            in_feather = in_axl & (~in_core) & (perp <= fend)
            out_zone   = ~in_core & ~in_feather

            # ── Feather ring: gradient 1.0 → multiplier, dominance-gated ──
            # Dominance check here too: if this bone IS the winner even in the
            # ring (e.g. the forearm IS dominant at the very edge of the upper
            # arm cylinder), leave it alone — only reduce actual bleeding.
            if in_feather.any():
                fth_wi     = wi[in_feather]
                bleeding   = dominant_at[fth_wi] != bi
                if bleeding.any():
                    R_f        = R[in_feather][bleeding]
                    safe_denom = np.maximum(fend[in_feather][bleeding] - R_f, 1e-6)
                    fac        = np.clip((perp[in_feather][bleeding] - R_f) / safe_denom, 0.0, 1.0)
                    fth_mult   = (1.0 - fac * (1.0 - multiplier)).astype(np.float32)
                    bidx       = fth_wi[bleeding]
                    old_w      = weights[bidx, bi].copy()
                    new_w      = old_w * fth_mult
                    new_w[new_w < 0.001] = 0.0
                    weights[bidx, bi] = new_w
                    fixed += int((new_w != old_w).sum())

            # ── Outer zone: dominance check, full multiplier ──────────────
            if out_zone.any():
                out_wi   = wi[out_zone]
                bleeding = dominant_at[out_wi] != bi
                if bleeding.any():
                    bidx  = out_wi[bleeding]
                    old_w = weights[bidx, bi].copy()
                    new_w = old_w * multiplier
                    new_w[new_w < 0.001] = 0.0
                    weights[bidx, bi] = new_w
                    fixed += int((new_w != old_w).sum())

        return fixed


@dataclass
class _WeightData:
    weights:     np.ndarray
    group_names: List[str]
    metadata:    Dict[str, Any] = field(default_factory=dict)


class DeformationPipeline:
    """Enhanced skin binding pipeline: generates Blender's auto-weights then fixes
    cross-side bleeding, vertical chain bleeding, joint boundaries, smoothing and limits.

    Stages: generate base → fix cross-side → fix vertical bleeding →
            tighten joints → smooth → normalize → limit influences → validate
    """

    UNIVERSAL_BONE_KEYWORDS = {
        'UPPER_ARM': ['upper_arm', 'upperarm', 'biceps', 'humerus', 'arm_upper',
                      'brazo', 'bras', 'oberarm', 'ude'],
        'FOREARM':   ['forearm', 'lower_arm', 'lowerarm', 'arm_lower', 'radius', 'ulna',
                      'antebrazo', 'avant_bras', 'unterarm', 'maeude'],
        'THIGH':     ['thigh', 'upper_leg', 'upperleg', 'leg_upper', 'femur',
                      'muslo', 'cuisse', 'oberschenkel', 'futomomo'],
        'SHIN':      ['shin', 'lower_leg', 'lowerleg', 'calf', 'leg_lower', 'tibia',
                      'pantorrilla', 'jambe', 'unterschenkel', 'sune'],
        'SHOULDER':  ['shoulder', 'clavicle', 'collar', 'deltoid',
                      'hombro', 'epaule', 'schulter', 'kata'],
        'HAND':      ['hand', 'palm', 'wrist', 'carpal',
                      'mano', 'main', 'te'],
        'FOOT':      ['foot', 'ankle', 'heel', 'tarsal',
                      'pie', 'pied', 'fuss', 'ashi'],
        'SPINE':     ['spine', 'back', 'vertebra',
                      'columna', 'dos', 'wirbel', 'senaka'],
        'NECK':      ['neck', 'cervical',
                      'cuello', 'cou', 'hals', 'kubi'],
        'HEAD':      ['head', 'skull', 'cranium',
                      'cabeza', 'tete', 'kopf', 'atama'],
        'FINGER':    ['finger', 'index', 'middle', 'ring', 'pinky', 'thumb', 'digit', 'f_',
                      'dedo', 'doigt', 'yubi'],
        'TOE':       ['toe', 'toes', 'phalanx'],
        'JAW':       ['jaw', 'mandible', 'chin'],
        'EYE':       ['eye', 'eyeball', 'pupil'],
    }

    KNOWN_PREFIXES = [
        'DEF-', 'ORG-', 'MCH-', 'WGT-',
        'mixamorig:',
        'arp_', 'c_', 'x_',
        'bpy_', 'rig_',
    ]

    _FACE_KW = frozenset([
        'jaw', 'chin', 'cheek', 'brow', 'lid', 'nose', 'lip',
        'forehead', 'temple', 'ear', 'tongue', 'teeth', 'eye',
    ])

    def __init__(self, mesh_obj, armature_obj, mesh_type='BODY'):
        if mesh_obj.type != 'MESH':
            raise TypeError("mesh_obj must be a mesh")
        if armature_obj.type != 'ARMATURE':
            raise TypeError("armature_obj must be an armature")

        self.mesh_obj     = mesh_obj
        self.armature_obj = armature_obj
        self.mesh         = mesh_obj.data
        self.armature     = armature_obj.data
        self.mesh_type    = mesh_type

        self.current_stage   = None
        self.weight_data     = None
        self.analysis        = None
        self._sym_plane_cache = None   # cached once per pipeline run
        self.errors:   List[str] = []
        self.warnings: List[str] = []

        self.config = self._default_config()

    def _default_config(self) -> dict:
        return {
            'smooth': {'iterations': 3, 'factor': 0.3},
            'limit_influences': {'max_bones': 4},
            'optimize_highres': False,
            'fix_scale': False,
            'surgical': {
                'fix_fingers':    True,  'fingers_feather':    2.0,
                'fix_forearms':   True,  'forearms_feather':   2.0,
                'fix_upper_arms': True,  'upper_arms_feather': 2.0,
                'fix_shins':      True,  'shins_feather':      2.0,
                'fix_thighs':     True,  'thighs_feather':     2.0,
                'fix_shoulders':  True,  'shoulders_feather':  2.0,
                'fix_spine':      False, 'spine_feather':      2.0,
                'fix_neck':       False, 'neck_feather':       2.0,
            },
        }

    def run_full_pipeline(self, progress_callback=None) -> bool:
        stages = [
            ("Generating Base Weights",        self._stage_generate_base),
            ("Cylinder Weight Cleanup",        self._stage_cylinder_cleanup),
            ("Smoothing Cylinder Boundaries",  self._stage_smooth_post_cylinder),
            ("Fixing Cross-Side Bleeding",     self._stage_fix_cross_side),
            ("Tightening Joints (Surgical)",   self._stage_tighten_surgical),
            ("Distributing Twist Weights",     self._stage_distribute_twist),
            ("Smoothing Within Bone Zones",    self._stage_smooth_zones),
            ("Limiting Influences",            self._stage_limit_influences),
            ("Final Validation",               self._stage_validate),
        ]
        total = len(stages)
        for idx, (name, stage_func) in enumerate(stages):
            try:
                success = stage_func()
                if not success:
                    self.errors.append(f"Stage '{name}' failed")
                    return False
                if progress_callback:
                    progress_callback(name, (idx + 1) / total)
            except Exception as e:
                self.errors.append(f"Stage '{name}' crashed: {str(e)}")
                return False

        w      = self.weight_data.weights
        totals = w.sum(axis=1)
        valid  = totals > 0.001
        if valid.any():
            w[valid] = w[valid] / totals[valid, np.newaxis]
        dbg(f"  Final normalize: {int(valid.sum())} vertices normalized")

        self._apply_weights_to_mesh()
        return True

    def run_pipeline_from_existing_weights(self, progress_callback=None) -> bool:
        """Refine weights already on the mesh — skips base generation."""
        self._read_weights_from_mesh()
        if self.weight_data is None or not self.weight_data.group_names:
            self.errors.append("No weights found on mesh to refine")
            return False

        stages = [
            ("Cylinder Weight Cleanup",        self._stage_cylinder_cleanup),
            ("Smoothing Cylinder Boundaries",  self._stage_smooth_post_cylinder),
            ("Fixing Cross-Side Bleeding",     self._stage_fix_cross_side),
            ("Tightening Joints (Surgical)",   self._stage_tighten_surgical),
            ("Distributing Twist Weights",     self._stage_distribute_twist),
            ("Smoothing Within Bone Zones",    self._stage_smooth_zones),
            ("Limiting Influences",            self._stage_limit_influences),
            ("Final Validation",               self._stage_validate),
        ]
        total = len(stages)
        for idx, (name, stage_func) in enumerate(stages):
            try:
                success = stage_func()
                if not success:
                    self.errors.append(f"Stage '{name}' failed")
                    return False
                if progress_callback:
                    progress_callback(name, (idx + 1) / total)
            except Exception as e:
                self.errors.append(f"Stage '{name}' crashed: {str(e)}")
                return False

        w      = self.weight_data.weights
        totals = w.sum(axis=1)
        valid  = totals > 0.001
        if valid.any():
            w[valid] = w[valid] / totals[valid, np.newaxis]
        dbg(f"  Final normalize: {int(valid.sum())} vertices normalized")

        self._apply_weights_to_mesh()
        return True

    # ── Stage 1 ───────────────────────────────────────────────────────────────

    def _stage_generate_base(self) -> bool:
        """Use Blender's built-in automatic weights as the starting point.
        When optimize_highres is enabled, binds a decimated proxy first and
        transfers weights back — keeps binding fast on 100k+ meshes."""
        bpy.context.view_layer.objects.active = self.mesh_obj
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        if self.mesh_obj.parent:
            mw = self.mesh_obj.matrix_world.copy()
            self.mesh_obj.parent = None
            self.mesh_obj.matrix_world = mw

        for mod in list(self.mesh_obj.modifiers):
            if mod.type == 'ARMATURE':
                self.mesh_obj.modifiers.remove(mod)

        _fix_scale = self.config.get('fix_scale', False)
        _orig_mesh_world = _orig_arm_world = None
        if _fix_scale:
            from mathutils import Matrix
            _orig_mesh_world = self.mesh_obj.matrix_world.copy()
            _orig_arm_world  = self.armature_obj.matrix_world.copy()
            # Scale BOTH objects by ONE shared transform about a COMMON pivot (the
            # armature's world origin). Scaling each object's own .scale instead pivots
            # about its own origin, so a mesh and armature with different origins drift
            # apart at 10× and bind to misaligned geometry. A shared pivot keeps them
            # locked together regardless of their origins.
            _pivot = self.armature_obj.matrix_world.translation.copy()
            _S = (Matrix.Translation(_pivot) @ Matrix.Scale(10.0, 4)
                  @ Matrix.Translation(-_pivot))
            self.mesh_obj.matrix_world     = _S @ _orig_mesh_world
            self.armature_obj.matrix_world = _S @ _orig_arm_world
            dbg(f"  Scale fix ON: scaled up 10× (shared pivot) for binding")

        try:
            use_proxy = (self.config.get('optimize_highres', False)
                         and len(self.mesh.vertices) > 30_000)
            if use_proxy:
                self._generate_base_via_proxy()
            else:
                for o in bpy.context.view_layer.objects:
                    o.select_set(False)
                self.mesh_obj.select_set(True)
                self.armature_obj.select_set(True)
                bpy.context.view_layer.objects.active = self.armature_obj
                bpy.ops.object.parent_set(type='ARMATURE_AUTO')

            if not self._mesh_has_weights():
                # Heat diffusion requires polygon faces — it silently produces nothing
                # on wire/edge-only/point meshes.  Retry with Envelope weights, which
                # work on any geometry type.
                for mod in list(self.mesh_obj.modifiers):
                    if mod.type == 'ARMATURE':
                        self.mesh_obj.modifiers.remove(mod)
                self.mesh_obj.vertex_groups.clear()
                for o in bpy.context.view_layer.objects:
                    o.select_set(False)
                self.mesh_obj.select_set(True)
                self.armature_obj.select_set(True)
                bpy.context.view_layer.objects.active = self.armature_obj
                bpy.ops.object.parent_set(type='ARMATURE_ENVELOPE')
                self.warnings.append(
                    f"'{self.mesh_obj.name}': Heat Map requires polygon faces — "
                    "fell back to Envelope weights."
                )
        finally:
            # ALWAYS restore the original scale, even if binding raised — otherwise the
            # mesh and the (shared) armature are left stuck at 10×.
            if _fix_scale:
                # Unparent before restoring matrix_world — otherwise the stale
                # matrix_parent_inverse baked at 10× by parent_set fights the restore.
                _was_parented = (self.mesh_obj.parent == self.armature_obj)
                if _was_parented:
                    self.mesh_obj.parent = None
                self.mesh_obj.matrix_world     = _orig_mesh_world
                self.armature_obj.matrix_world = _orig_arm_world
                if _was_parented:
                    self.mesh_obj.parent = self.armature_obj
                    self.mesh_obj.matrix_parent_inverse = self.armature_obj.matrix_world.inverted()

        if not self._mesh_has_weights():
            raise RuntimeError(
                f"No weights generated for '{self.mesh_obj.name}'. "
                "Check the mesh has faces and the armature is visible."
            )

        self._read_weights_from_mesh()

        stats = self._analyze_weight_quality()
        dbg(f"\nBase weights quality:")
        dbg(f"  Avg influences per vertex: {stats['avg_influences']:.1f}")
        dbg(f"  Cross-side bleeders: {stats['cross_side_count']}")
        return True

    def _generate_base_via_proxy(self):
        """Decimate → bind proxy → transfer weights back → remove proxy."""
        ctx = bpy.context
        # Duplicate mesh as proxy
        proxy = self.mesh_obj.copy()
        proxy.data = self.mesh_obj.data.copy()
        proxy.name = self.mesh_obj.name + "_proxy_bind"
        ctx.collection.objects.link(proxy)

        # Add Decimate modifier to proxy and apply it
        dec = proxy.modifiers.new(name="DecimateProxy", type='DECIMATE')
        dec.ratio = 0.1
        for o in ctx.view_layer.objects:
            o.select_set(False)
        proxy.select_set(True)
        ctx.view_layer.objects.active = proxy
        bpy.ops.object.modifier_apply(modifier=dec.name)

        # Bind the proxy
        proxy.select_set(True)
        self.armature_obj.select_set(True)
        ctx.view_layer.objects.active = self.armature_obj
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')

        # Add Data Transfer modifier to original mesh to pull weights from proxy
        for o in ctx.view_layer.objects:
            o.select_set(False)
        self.mesh_obj.select_set(True)
        ctx.view_layer.objects.active = self.mesh_obj
        dt = self.mesh_obj.modifiers.new(name="WeightTransfer", type='DATA_TRANSFER')
        dt.object             = proxy
        dt.use_vert_data      = True
        dt.data_types_verts   = {'VGROUP_WEIGHTS'}
        dt.vert_mapping       = 'NEAREST'
        bpy.ops.object.modifier_apply(modifier=dt.name)

        # Re-parent original to rig (without weights — already transferred)
        self.mesh_obj.select_set(True)
        self.armature_obj.select_set(True)
        ctx.view_layer.objects.active = self.armature_obj
        bpy.ops.object.parent_set(type='ARMATURE_NAME')

        # Remove proxy
        bpy.data.objects.remove(proxy, do_unlink=True)
        dbg(f"  High-res proxy bind complete (original: {len(self.mesh.vertices)} verts)")

    # ── Stage 1.5 ─────────────────────────────────────────────────────────────

    def _stage_smooth_post_cylinder(self) -> bool:
        """Light Laplacian smooth over cylinder cut boundaries (no zone guard)."""
        if self.weight_data is None:
            return False

        weights  = self.weight_data.weights
        n_verts  = weights.shape[0]
        iters    = 2
        factor   = 0.35

        edges  = np.array([(e.vertices[0], e.vertices[1])
                           for e in self.mesh.edges], dtype=np.int32)
        src    = np.concatenate([edges[:, 0], edges[:, 1]])
        dst    = np.concatenate([edges[:, 1], edges[:, 0]])
        degree = np.zeros(n_verts, dtype=np.float32)
        np.add.at(degree, dst, 1.0)
        deg_safe = np.where(degree > 0, degree, 1.0)

        frozen = {gi: weights[:, gi].copy()
                  for gi, gn in enumerate(self.weight_data.group_names)
                  if self._is_face_bone(gn)}

        for _ in range(iters):
            nbr_sum = np.zeros_like(weights)
            np.add.at(nbr_sum, dst, weights[src])
            nbr_avg = nbr_sum / deg_safe[:, np.newaxis]
            weights = weights * (1.0 - factor) + nbr_avg * factor

        for gi, col in frozen.items():
            weights[:, gi] = col

        self.weight_data.weights = weights
        dbg(f"  Post-cylinder smooth: {iters} passes × factor={factor:.2f}")
        return True

    def _stage_cylinder_cleanup(self) -> bool:
        """Zero heat-diffusion bleed outside each bone's dominant-vertex cylinder."""
        if self.weight_data is None:
            return False

        import re as _re
        weights     = self.weight_data.weights
        group_names = self.weight_data.group_names
        m_inv       = self.mesh_obj.matrix_world.inverted()
        arm_mw      = self.armature_obj.matrix_world
        vp          = np.array([v.co for v in self.mesh.vertices], dtype=np.float32)

        zeroed = 0
        faded  = 0

        # Snapshot dominant bone per vertex once — used to measure each bone's
        # true cross-section from its own territory only.
        dominant_at = np.argmax(weights, axis=1)  # (n_verts,)

        for gi, name in enumerate(group_names):
            if self._is_face_bone(name):
                continue
            # Twist variants (.001/.002/.003) are redistributed by distribute_twist;
            # process only the main (non-numbered) bones here.
            if _re.search(r'\.\d{3}$', name):
                continue
            # Fingers ARE cleaned here now (tapered + dominance-gated cylinder): they
            # were skipped before because the old ungated zeroing was too aggressive on
            # tiny finger bones, leaving finger joints over-smooth/bleedy. The dominance
            # gate now protects each finger's own verts, so the cylinder crisps the
            # joints and trims cross-finger bleed without deleting real weight.
            # Shoulder (clavicle) is a transition bone — cylinder radius estimation
            # often fails on clothing because upper_arm dominates the region.
            if self._classify_bone_type(name) == 'SHOULDER':
                continue
            if name not in self.armature.bones:
                continue
            bone = self.armature.bones[name]
            if not bone.use_deform:
                continue

            # Fingers get a TIGHTER cone (smaller radius + narrower feather) so the
            # joints read crisply and cross-finger bleed is trimmed harder.
            is_finger = (self._classify_bone_type(name) == 'FINGER')

            wi = np.where(weights[:, gi] > 0.001)[0]
            if len(wi) == 0:
                continue

            # ── Bone axis in mesh-local space ─────────────────────────────
            head  = np.array(m_inv @ (arm_mw @ bone.head_local), dtype=np.float32)
            tail  = np.array(m_inv @ (arm_mw @ bone.tail_local), dtype=np.float32)
            seg   = tail - head
            L     = float(np.linalg.norm(seg))
            if L < 1e-6:
                continue
            direc = seg / L

            # ── Tapered (cone) radius from this bone's dominant vertices ──
            # A limb is a truncated cone; one radius clips the wide end (deltoid/
            # elbow) and leaks at the narrow end (wrist). Fit R_head/R_tail instead.
            dom_mask  = (dominant_at == gi) & (weights[:, gi] > 0.01)
            if not dom_mask.any():
                # Bone never dominates any vertex on this mesh → pure bleed
                weights[wi, gi] = 0.0
                zeroed += len(wi)
                continue

            dom_verts = vp[dom_mask]
            to_dom    = dom_verts - head
            proj_dom  = (to_dom * direc).sum(axis=1)
            cl_dom    = np.clip(proj_dom, 0.0, L)[:, np.newaxis]
            perp_dom  = np.linalg.norm(dom_verts - (head + direc * cl_dom), axis=1)

            r_head, r_tail = _fit_cone_radius(
                proj_dom, perp_dom, L, pct=(0.80 if is_finger else 0.90))

            # ── Zone check for all weighted vertices (per-vertex cone radius) ──
            verts = vp[wi]
            to_v  = verts - head
            proj  = (to_v * direc).sum(axis=1)
            cl    = np.clip(proj, 0.0, L)[:, np.newaxis]
            perp  = np.linalg.norm(verts - (head + direc * cl), axis=1)

            tt   = np.clip(proj / L, 0.0, 1.0)
            Rv   = r_head + (r_tail - r_head) * tt   # tapered radius per vertex
            Rmin = min(r_head, r_tail)
            pad  = max(Rmin * 0.5, L * 0.05)         # axial buffer at bone ends
            fend = Rv * (1.25 if is_finger else 1.6)  # tighter feather on fingers

            in_axl     = (proj >= -pad) & (proj <= L + pad)
            in_core    = in_axl & (perp <= Rv)
            in_feather = in_axl & ~in_core & (perp <= fend)
            out_zone   = ~in_core & ~in_feather

            # DOMINANCE GATE: never hard-zero (or fade) a vertex this bone actually
            # OWNS. A too-thin radius at a wide cone end would otherwise delete real
            # weight — the previous ungated zeroing was the main arm-bind problem.
            owns = (dominant_at[wi] == gi)

            if out_zone.any():
                kill = out_zone & ~owns
                if kill.any():
                    weights[wi[kill], gi] = 0.0
                    zeroed += int(kill.sum())

            fade_mask = in_feather & ~owns
            if fade_mask.any():
                fth_wi  = wi[fade_mask]
                safe_d  = np.maximum(fend[fade_mask] - Rv[fade_mask], 1e-6)
                t       = np.clip((perp[fade_mask] - Rv[fade_mask]) / safe_d, 0.0, 1.0)
                weights[fth_wi, gi] *= (1.0 - t)
                # Drop anything too small to matter
                tiny = weights[fth_wi, gi] < 0.001
                weights[fth_wi[tiny], gi] = 0.0
                faded += int(fade_mask.sum())

        dbg(f"  Cylinder cleanup: zeroed {zeroed}, faded {faded} weights")
        return True

    # ── Stage 2 ───────────────────────────────────────────────────────────────

    def _stage_fix_cross_side(self) -> bool:
        """Remove weights from left bones on right-side vertices and vice versa."""
        if self.weight_data is None:
            return False

        weights     = self.weight_data.weights
        group_names = self.weight_data.group_names
        num_verts   = weights.shape[0]
        num_groups  = weights.shape[1]

        center_x = self._find_symmetry_plane()
        if center_x is None:
            self.warnings.append("No symmetry plane found, skipping cross-side fix")
            return True

        bone_sides  = [self._get_bone_side(n) for n in group_names]
        vert_x      = np.array([self.mesh.vertices[i].co.x for i in range(num_verts)],
                               dtype=np.float32)
        is_left     = vert_x > center_x
        remove_mask = np.zeros((num_verts, num_groups), dtype=bool)
        for gidx, bside in enumerate(bone_sides):
            if bside is None:
                continue
            if self._is_face_bone(group_names[gidx]):
                continue
            bone_is_left         = (bside == 'LEFT')
            wrong_side           = (is_left != bone_is_left)
            has_weight           = (weights[:, gidx] > 0.001)
            remove_mask[:, gidx] = wrong_side & has_weight

        removed_count        = int(remove_mask.sum())
        weights[remove_mask] = 0.0

        dbg(f"  Cross-side: removed {removed_count} weights")
        return True

    def _find_symmetry_plane(self):
        if self._sym_plane_cache is not None:
            return self._sym_plane_cache
        # Project armature world origin into mesh local space — true midline even
        # for accessories that sit entirely on one side of the character.
        arm_world = self.armature_obj.matrix_world.translation
        local_x   = (self.mesh_obj.matrix_world.inverted() @ arm_world).x
        self._sym_plane_cache = float(local_x)
        return self._sym_plane_cache

    def _strip_known_prefixes(self, bone_name):
        clean = bone_name
        for prefix in self.KNOWN_PREFIXES:
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
                break
        return clean

    def _get_bone_side(self, bone_name):
        import re as _re
        clean = self._strip_known_prefixes(bone_name).lower()

        # 1. Standard Blender suffix: .L / .R / _L / _R at the very end.
        if clean.endswith('.r') or clean.endswith('_r'):
            return 'RIGHT'
        if clean.endswith('.l') or clean.endswith('_l'):
            return 'LEFT'

        # 2. Rigify multi-segment: DEF-upper_arm.L.001, DEF-upper_arm.R.002, etc.
        if _re.search(r'\.(r)\.\d+$', clean):
            return 'RIGHT'
        if _re.search(r'\.(l)\.\d+$', clean):
            return 'LEFT'

        tokens = _re.split(r'[^a-z]+', clean)
        if 'right' in tokens:
            return 'RIGHT'
        if 'left' in tokens:
            return 'LEFT'

        if clean.startswith('r_') or clean.startswith('r.'):
            return 'RIGHT'
        if clean.startswith('l_') or clean.startswith('l.'):
            return 'LEFT'

        if bone_name in self.armature.bones:
            bone  = self.armature.bones[bone_name]
            hw    = self.armature_obj.matrix_world @ bone.head_local
            cx_w  = float(self.armature_obj.matrix_world.translation.x)
            if hw.x > cx_w + 0.01:
                return 'LEFT'
            elif hw.x < cx_w - 0.01:
                return 'RIGHT'
        return None

    def _classify_bone_type(self, bone_name):
        clean = self._strip_known_prefixes(bone_name).lower()
        for bone_type, keywords in self.UNIVERSAL_BONE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in clean:
                    return bone_type
        return 'OTHER'

    def _is_face_bone(self, name):
        import re as _re
        clean = self._strip_known_prefixes(name).lower()
        # Use non-letter boundaries so "ear" doesn't match inside "forearm".
        return any(_re.search(r'(?<![a-z])' + kw + r'(?![a-z])', clean)
                   for kw in self._FACE_KW)

    # ── Stage 3 ───────────────────────────────────────────────────────────────

    def _stage_tighten_surgical(self) -> bool:
        """Cylinder-based per-section weight tightening using SurgicalWeightFixer."""
        if self.weight_data is None:
            return False

        weights     = self.weight_data.weights
        group_names = self.weight_data.group_names
        center_x    = self._find_symmetry_plane()
        scfg        = self.config.get('surgical', {})

        surgical = getattr(self, '_surgical_fixer', None) or SurgicalWeightFixer(self.mesh_obj, self.armature_obj)
        self._surgical_fixer = surgical

        section_defs = [
            ("Fingers",    surgical.finger_bones,    scfg.get('fix_fingers',    True),  scfg.get('fingers_feather',    2.0)),
            ("Forearms",   surgical.forearm_bones,   scfg.get('fix_forearms',   True),  scfg.get('forearms_feather',   2.0)),
            ("Upper Arms", surgical.upper_arm_bones, scfg.get('fix_upper_arms', True),  scfg.get('upper_arms_feather', 2.0)),
            ("Shins",      surgical.shin_bones,      scfg.get('fix_shins',      True),  scfg.get('shins_feather',      2.0)),
            ("Thighs",     surgical.thigh_bones,     scfg.get('fix_thighs',     True),  scfg.get('thighs_feather',     2.0)),
            ("Shoulders",  surgical.shoulder_bones,  scfg.get('fix_shoulders',  True),  scfg.get('shoulders_feather',  2.0)),
            ("Spine",      surgical.spine_bones,     scfg.get('fix_spine',      False), scfg.get('spine_feather',      2.0)),
            ("Neck",       surgical.neck_bones,      scfg.get('fix_neck',       False), scfg.get('neck_feather',       2.0)),
        ]

        total = 0
        for label, bone_list, enabled, feather in section_defs:
            if not enabled:
                dbg(f"  Surgical {label:<12}: OFF")
                continue
            n = surgical.tighten_section(weights, group_names, bone_list, feather, center_x)
            total += n
            dbg(f"  Surgical {label:<12}: feather={feather:.2f}  bones={len(bone_list)}  verts_adjusted={n}")

        dbg(f"  Surgical tighten: adjusted {total} weights")
        return True

    # ── Stage 5 ───────────────────────────────────────────────────────────────

    def _stage_smooth_zones(self) -> bool:
        """Laplacian smooth for non-face bones with a per-bone zone guard."""
        if self.weight_data is None:
            return False

        cfg    = self.config.get('smooth', {})
        iters  = max(0, int(cfg.get('iterations', 3)))
        factor = float(cfg.get('factor', 0.3))

        if iters == 0 or factor <= 0.0:
            dbg(f"  Smooth: skipped (iterations={iters}, factor={factor:.2f})")
            return True

        weights     = self.weight_data.weights
        group_names = self.weight_data.group_names
        n_verts     = weights.shape[0]

        # Exclude face AND finger bones: fingers are tightened by the cylinder stage
        # and smoothing here re-blends the crisp joints back to mush.
        smooth_cols = [gi for gi, gn in enumerate(group_names)
                       if not self._is_face_bone(gn)
                       and self._classify_bone_type(gn) != 'FINGER']
        if not smooth_cols:
            dbg("  Smooth: no non-face bones found, skipped")
            return True

        # Freeze face + finger bone columns
        smooth_set = set(smooth_cols)
        frozen = {gi: weights[:, gi].copy()
                  for gi in range(len(group_names)) if gi not in smooth_set}

        # Directed edge arrays
        edges = np.array([(e.vertices[0], e.vertices[1])
                          for e in self.mesh.edges], dtype=np.int32)
        src = np.concatenate([edges[:, 0], edges[:, 1]])
        dst = np.concatenate([edges[:, 1], edges[:, 0]])

        degree = np.zeros(n_verts, dtype=np.float32)
        np.add.at(degree, dst, 1.0)
        deg_safe = np.where(degree > 0, degree, 1.0)

        for _ in range(iters):
            nbr_sum  = np.zeros_like(weights)
            np.add.at(nbr_sum, dst, weights[src])
            nbr_avg  = nbr_sum / deg_safe[:, np.newaxis]
            new_w    = weights * (1.0 - factor) + nbr_avg * factor
            max_gain = np.maximum(weights * 2.0, 0.02)
            weights  = np.minimum(new_w, max_gain)

        for gi, col in frozen.items():
            weights[:, gi] = col

        # Fingers: ONE light pass so the crisp cylinder joints don't end up faceted,
        # without the full multi-pass blur the other limbs get.
        finger_cols = [gi for gi, gn in enumerate(group_names)
                       if self._classify_bone_type(gn) == 'FINGER'
                       and not self._is_face_bone(gn)]
        if finger_cols:
            f_factor = 0.25
            nbr_sum  = np.zeros_like(weights)
            np.add.at(nbr_sum, dst, weights[src])
            nbr_avg  = nbr_sum / deg_safe[:, np.newaxis]
            sm       = weights * (1.0 - f_factor) + nbr_avg * f_factor
            for gi in finger_cols:
                weights[:, gi] = sm[:, gi]

        self.weight_data.weights = weights
        dbg(f"  Smooth: {iters} passes × factor={factor:.2f}, "
              f"{len(smooth_cols)} / {len(group_names)} columns active; "
              f"fingers 1 light pass ({len(finger_cols)} cols)")
        return True

    # ── Stage 5b ──────────────────────────────────────────────────────────────

    def _stage_distribute_twist(self) -> bool:
        """Hand main-bone weight to its .001 twist twin along an AXIAL GRADIENT.

        Twist accumulates toward the DISTAL end of a limb (the wrist for the forearm),
        so the handover ramps from ~0 at the bone head to MAX_TWIST at the tail instead
        of a flat split everywhere (which over-twists the proximal end and under-drives
        the distal end). Falls back to a flat split if the bone axis is unavailable."""
        if self.weight_data is None:
            return False
        import re as _re
        weights     = self.weight_data.weights
        group_names = self.weight_data.group_names
        m_inv       = self.mesh_obj.matrix_world.inverted()
        arm_mw      = self.armature_obj.matrix_world
        vp          = np.array([v.co for v in self.mesh.vertices], dtype=np.float32)
        adjusted    = 0
        MAX_TWIST   = 0.5     # fraction handed to the twist twin at the distal tail
        for gi, name in enumerate(group_names):
            if _re.search(r'\.\d+$', name):
                continue                    # skip bones that are already twist variants
            if self._is_face_bone(name):
                continue                    # face bone segments are not twist pairs
            twist_name = name + '.001'
            if twist_name not in group_names:
                continue
            ti   = group_names.index(twist_name)
            has_w = np.where(weights[:, gi] > 0.001)[0]
            if len(has_w) == 0:
                continue

            # Axial fraction: ramp 0 (head) → MAX_TWIST (tail/distal).
            bone = self.armature.bones.get(name) if hasattr(self.armature.bones, 'get') \
                   else (self.armature.bones[name] if name in self.armature.bones else None)
            frac = None
            if bone is not None:
                head = np.array(m_inv @ (arm_mw @ bone.head_local), dtype=np.float32)
                tail = np.array(m_inv @ (arm_mw @ bone.tail_local), dtype=np.float32)
                seg  = tail - head
                L    = float(np.linalg.norm(seg))
                if L >= 1e-6:
                    direc = seg / L
                    proj  = ((vp[has_w] - head) * direc).sum(axis=1)
                    t     = np.clip(proj / L, 0.0, 1.0)
                    frac  = (MAX_TWIST * t).astype(np.float32)
            if frac is None:
                frac = np.full(len(has_w), MAX_TWIST * 0.8, dtype=np.float32)  # flat fallback

            main_w = weights[has_w, gi].copy()
            weights[has_w, gi]  = main_w * (1.0 - frac)
            weights[has_w, ti] += main_w * frac    # accumulate, don't overwrite
            adjusted += len(has_w)
        dbg(f"  Twist distribution (axial gradient): {adjusted} vertices adjusted")
        return True

    # ── Stage 6 ───────────────────────────────────────────────────────────────

    def _stage_limit_influences(self) -> bool:
        if self.weight_data is None:
            return False
        max_bones = self.config['limit_influences']['max_bones']
        weights   = self.weight_data.weights
        for vi in range(weights.shape[0]):
            vw = weights[vi]
            if np.count_nonzero(vw) > max_bones:
                top_idx        = np.argpartition(vw, -max_bones)[-max_bones:]
                new_vw         = np.zeros_like(vw)
                new_vw[top_idx] = vw[top_idx]
                total          = new_vw.sum()
                if total > 0:
                    new_vw /= total
                weights[vi] = new_vw
        self.weight_data.weights = weights
        avg = np.count_nonzero(weights > 0.001, axis=1).mean()
        dbg(f"  Limited to {max_bones} bones. Avg influences: {avg:.1f}")
        return True

    # ── Stage 8 ───────────────────────────────────────────────────────────────

    def _stage_validate(self) -> bool:
        if self.weight_data is None:
            return False
        weights     = self.weight_data.weights
        orphan_mask = weights.sum(axis=1) < 0.001
        n_orphans   = int(orphan_mask.sum())
        if n_orphans:
            self._fix_orphans_from_neighbors(orphan_mask)
            self.warnings.append(f"Fixed {n_orphans} orphan vertices")
        inf_counts = np.count_nonzero(weights > 0.001, axis=1)
        dbg(f"\n{'='*50}")
        dbg(f"FINAL WEIGHT QUALITY REPORT")
        dbg(f"{'='*50}")
        dbg(f"  Groups: {weights.shape[1]}")
        dbg(f"  Avg influences per vertex: {inf_counts.mean():.1f}")
        dbg(f"  Single-bone vertices: {int((inf_counts == 1).sum())}")
        dbg(f"  Orphans fixed: {n_orphans}")
        return True

    def _fix_orphans_from_neighbors(self, orphan_mask):
        """Accept either a numpy boolean mask or a list of indices."""
        weights = self.weight_data.weights
        if isinstance(orphan_mask, np.ndarray) and orphan_mask.dtype == bool:
            orphan_indices = np.where(orphan_mask)[0]
            valid_indices  = np.where(~orphan_mask)[0]
        else:
            orphan_indices = list(orphan_mask)
            valid_indices  = [i for i in range(weights.shape[0])
                              if weights[i].sum() > 0.001]
        if len(orphan_indices) == 0 or len(valid_indices) == 0:
            return
        valid_coords = [self.mesh.vertices[int(i)].co for i in valid_indices]
        kd = kdtree.KDTree(len(valid_coords))
        for i, co in enumerate(valid_coords):
            kd.insert(co, i)
        kd.balance()
        for orphan_idx in orphan_indices:
            _, nearest_i, _ = kd.find(self.mesh.vertices[int(orphan_idx)].co)
            weights[orphan_idx] = weights[valid_indices[nearest_i]].copy()

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _mesh_has_weights(self):
        return any(len(v.groups) > 0 for v in self.mesh.vertices)

    def _read_weights_from_mesh(self):
        group_names = [g.name for g in self.mesh_obj.vertex_groups]
        num_groups  = len(group_names)
        weights     = np.zeros((len(self.mesh.vertices), num_groups), dtype=np.float32)
        for vert in self.mesh.vertices:
            for g in vert.groups:
                if g.group < num_groups:
                    weights[vert.index, g.group] = g.weight
        self.weight_data = _WeightData(weights=weights, group_names=group_names)

    def _apply_weights_to_mesh(self):
        if self.weight_data is None:
            return
        weights     = self.weight_data.weights
        group_names = self.weight_data.group_names

        for name in group_names:
            if name not in self.mesh_obj.vertex_groups:
                self.mesh_obj.vertex_groups.new(name=name)

        # col_to_vg maps our numpy column index → mesh vertex group index
        col_to_vg = [self.mesh_obj.vertex_groups[name].index for name in group_names]

        # Face bones must never deform non-body meshes (clothing, hair etc.)
        face_cols = (frozenset(gi for gi, nm in enumerate(group_names)
                               if self._is_face_bone(nm))
                     if self.mesh_type != 'BODY' else frozenset())

        # BMesh deform layer write — much faster than per-vertex vg.add() calls
        bm = bmesh.new()
        bm.from_mesh(self.mesh)
        dl = bm.verts.layers.deform.verify()
        for vert in bm.verts:
            dv  = vert[dl]
            dv.clear()
            row = weights[vert.index]
            for gi, vg_i in enumerate(col_to_vg):
                if gi in face_cols:
                    continue
                w = float(row[gi])
                if w > 0.0001:
                    dv[vg_i] = w
        bm.to_mesh(self.mesh)
        bm.free()

    def _analyze_weight_quality(self):
        if self.weight_data is None:
            return {}
        weights     = self.weight_data.weights
        group_names = self.weight_data.group_names
        inf_counts  = np.count_nonzero(weights > 0.001, axis=1)
        avg_inf     = float(inf_counts.mean())
        single      = int((inf_counts == 1).sum())
        center_x    = self._find_symmetry_plane()
        cross       = 0
        if center_x is not None:
            # Vectorized cross-side count: for each group build a wrong-side
            # boolean column, OR them together, count affected vertices once.
            vert_x   = np.array([v.co.x for v in self.mesh.vertices], dtype=np.float32)
            is_left  = vert_x > center_x
            any_wrong = np.zeros(weights.shape[0], dtype=bool)
            for gi, gname in enumerate(group_names):
                bside = self._get_bone_side(gname)
                if bside is None:
                    continue
                bone_left = (bside == 'LEFT')
                any_wrong |= (is_left != bone_left) & (weights[:, gi] > 0.01)
            cross = int(any_wrong.sum())
        return {'avg_influences': avg_inf, 'single_bone_count': single, 'cross_side_count': cross}

    def get_stage_report(self) -> str:
        lines = ["=" * 50, "DEFORMATION PIPELINE REPORT", "=" * 50]
        if self.weight_data is not None:
            w = self.weight_data.weights
            lines.append(
                f"Weights: {w.shape[1]} groups | "
                f"avg influences={np.count_nonzero(w > 0.001, axis=1).mean():.1f}")
        for e in self.errors:
            lines.append(f"ERROR: {e}")
        for w in self.warnings:
            lines.append(f"WARN: {w}")
        return "\n".join(lines)


class AUTORIG_OT_SmartBind(bpy.types.Operator):
    """Bind enabled meshes to the rig using automatic weights with smart per-type bone filtering"""
    bl_idname  = "autorig.smart_bind"
    bl_label   = "⑧ Smart Bind"
    bl_options = {'REGISTER', 'UNDO'}

    def _direct_bind(self, context, mesh_obj, rig_obj, bone_name):
        """Assign all vertices 100% to bone_name — skips auto-weight entirely.
        Used for rigid meshes (eyeballs, split teeth) that need no deformation."""
        for mod in list(mesh_obj.modifiers):
            if mod.type == 'ARMATURE':
                mesh_obj.modifiers.remove(mod)
        mesh_obj.vertex_groups.clear()
        if bone_name not in {b.name for b in rig_obj.data.bones}:
            return False
        vg = mesh_obj.vertex_groups.new(name=bone_name)
        vg.add(list(range(len(mesh_obj.data.vertices))), 1.0, 'REPLACE')
        mod = mesh_obj.modifiers.new(name="Armature", type='ARMATURE')
        mod.object = rig_obj
        _reorder_arm_modifier(context, mesh_obj)
        return True

    def _strip_disallowed_groups(self, obj, allowed_keywords):
        to_remove = [vg for vg in obj.vertex_groups
                     if not any(kw in vg.name.lower() for kw in allowed_keywords)]
        for vg in to_remove:
            obj.vertex_groups.remove(vg)

    def _assign_fallback_group(self, obj, rig, mesh_type):
        """Assign all vertices weight 1.0 to the first available fallback bone.
        Called when no allowed vertex groups survived after stripping — e.g. eyes/teeth/tongue
        on a rig with no face bones."""
        candidates = _BIND_FALLBACK_BONES.get(mesh_type, [])
        bone_names = {b.name for b in rig.data.bones}
        fallback = next((b for b in candidates if b in bone_names), None)
        if not fallback:
            return
        vg = obj.vertex_groups.get(fallback) or obj.vertex_groups.new(name=fallback)
        all_indices = list(range(len(obj.data.vertices)))
        vg.add(all_indices, 1.0, 'REPLACE')

    def _bind_one(self, context, mesh_obj, rig_obj, method='AUTO', use_scale_fix=False):
        """Bind a single mesh. method: 'AUTO' or 'ENVELOPE'. Returns (ok, fallback)."""
        for mod in list(mesh_obj.modifiers):
            if mod.type == 'ARMATURE':
                mesh_obj.modifiers.remove(mod)
        mesh_obj.vertex_groups.clear()

        if use_scale_fix:
            temp_mesh = None
            temp_arm  = None
            ok       = False
            fallback = False
            try:
                temp_mesh      = mesh_obj.copy()
                temp_mesh.data = mesh_obj.data.copy()
                temp_mesh.name = mesh_obj.name + "_BIND_TEMP"
                context.collection.objects.link(temp_mesh)

                temp_arm      = rig_obj.copy()
                temp_arm.data = rig_obj.data.copy()
                temp_arm.name = rig_obj.name + "_BIND_TEMP"
                context.collection.objects.link(temp_arm)

                for obj in [temp_mesh, temp_arm]:
                    for o in context.view_layer.objects:
                        o.select_set(False)
                    obj.select_set(True)
                    context.view_layer.objects.active = obj
                    obj.scale = obj.scale.copy() * 10.0
                    bpy.ops.object.transform_apply(
                        location=False, rotation=False, scale=True)

                for o in context.view_layer.objects:
                    o.select_set(False)
                temp_mesh.select_set(True)
                temp_arm.select_set(True)
                context.view_layer.objects.active = temp_arm

                if method == 'ENVELOPE':
                    try:
                        bpy.ops.object.parent_set(type='ARMATURE_ENVELOPE')
                        ok = True
                    except Exception:
                        pass
                else:
                    _auto_ok = False
                    try:
                        bpy.ops.object.parent_set(type='ARMATURE_AUTO')
                        _auto_ok = any(
                            len(v.groups) > 0 for v in temp_mesh.data.vertices)
                    except Exception:
                        pass
                    if not _auto_ok:
                        for mod in list(temp_mesh.modifiers):
                            if mod.type == 'ARMATURE':
                                temp_mesh.modifiers.remove(mod)
                        temp_mesh.vertex_groups.clear()
                        for o in context.view_layer.objects:
                            o.select_set(False)
                        temp_mesh.select_set(True)
                        temp_arm.select_set(True)
                        context.view_layer.objects.active = temp_arm
                        try:
                            bpy.ops.object.parent_set(type='ARMATURE_ENVELOPE')
                            ok = True
                            fallback = True
                        except Exception:
                            pass
                    else:
                        ok = True

                if ok:
                    # Weights are pure 0-1 numbers — copy to original mesh
                    vg_lookup = {}
                    for src_vg in temp_mesh.vertex_groups:
                        vg_lookup[src_vg.index] = mesh_obj.vertex_groups.new(
                            name=src_vg.name)
                    for vert in temp_mesh.data.vertices:
                        for g in vert.groups:
                            if g.group in vg_lookup:
                                vg_lookup[g.group].add(
                                    [vert.index], g.weight, 'REPLACE')

                    arm_mod        = mesh_obj.modifiers.new(
                        name="Armature", type='ARMATURE')
                    arm_mod.object = rig_obj
                    mesh_obj.parent                = rig_obj
                    mesh_obj.matrix_parent_inverse = rig_obj.matrix_world.inverted()

            finally:
                for obj in [temp_mesh, temp_arm]:
                    if obj is not None:
                        try:
                            bpy.data.objects.remove(obj, do_unlink=True)
                        except Exception:
                            pass

            return ok, fallback

        # ── Direct path (no scale fix) ────────────────────────────────────────
        for o in context.view_layer.objects:
            o.select_set(False)
        mesh_obj.select_set(True)
        rig_obj.select_set(True)
        context.view_layer.objects.active = rig_obj

        if method == 'ENVELOPE':
            try:
                bpy.ops.object.parent_set(type='ARMATURE_ENVELOPE')
                return True, False
            except Exception:
                return False, False

        try:
            bpy.ops.object.parent_set(type='ARMATURE_AUTO')
            if any(len(v.groups) > 0 for v in mesh_obj.data.vertices):
                return True, False
            # AUTO produced no weights — clear and fall through to envelope
            for mod in list(mesh_obj.modifiers):
                if mod.type == 'ARMATURE':
                    mesh_obj.modifiers.remove(mod)
            mesh_obj.vertex_groups.clear()
        except Exception:
            pass
        try:
            bpy.ops.object.parent_set(type='ARMATURE_ENVELOPE')
            return True, True
        except Exception:
            return False, False

    def _bind_split_parts(self, context, mesh_obj, rig_obj, method='AUTO', use_scale_fix=False):
        """Separate by loose parts, bind each independently, rejoin. Returns (ok, fallback)."""
        for o in context.view_layer.objects:
            o.select_set(False)
        mesh_obj.select_set(True)
        context.view_layer.objects.active = mesh_obj
        with context.temp_override(active_object=mesh_obj, selected_objects=[mesh_obj]):
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.separate(type='LOOSE')
            bpy.ops.object.mode_set(mode='OBJECT')

        parts = [o for o in context.selected_objects if o.type == 'MESH']
        if len(parts) <= 1:
            return self._bind_one(context, mesh_obj, rig_obj, method=method, use_scale_fix=use_scale_fix)

        ok_all = True
        any_fallback = False
        for part in parts:
            for mod in list(part.modifiers):
                if mod.type == 'ARMATURE':
                    part.modifiers.remove(mod)
            part.vertex_groups.clear()
            ok, fallback = self._bind_one(context, part, rig_obj, method=method, use_scale_fix=use_scale_fix)
            if not ok:
                ok_all = False
            if fallback:
                any_fallback = True

        # Rejoin — keep mesh_obj as the survivor so item.obj stays valid
        for o in context.view_layer.objects:
            o.select_set(False)
        for part in parts:
            part.select_set(True)
        with context.temp_override(active_object=mesh_obj, selected_objects=list(parts)):
            bpy.ops.object.join()

        return ok_all, any_fallback

    def _bind_pipeline(self, context, mesh_obj, rig_obj, mesh_type='BODY') -> bool:
        """Run the DeformationPipeline on a single body/clothing mesh."""
        sp = context.scene.autorig_skin

        for mod in list(mesh_obj.modifiers):
            if mod.type == 'ARMATURE':
                mesh_obj.modifiers.remove(mod)
        mesh_obj.vertex_groups.clear()

        try:
            pipeline = DeformationPipeline(mesh_obj, rig_obj, mesh_type=mesh_type)
        except TypeError as e:
            self.report({'ERROR'}, str(e))
            return False

        pipeline.config['smooth']['iterations']          = sp.smooth_iterations
        pipeline.config['smooth']['factor']              = sp.smooth_factor
        pipeline.config['limit_influences']['max_bones'] = sp.max_influences
        pipeline.config['optimize_highres']              = sp.optimize_highres
        pipeline.config['fix_scale']                     = sp.fix_scale

        sc = context.scene
        for key, attr in [
            ('fix_fingers',    'surgical_fix_fingers'),
            ('fix_forearms',   'surgical_fix_forearms'),
            ('fix_upper_arms', 'surgical_fix_upper_arms'),
            ('fix_shins',      'surgical_fix_shins'),
            ('fix_thighs',     'surgical_fix_thighs'),
            ('fix_shoulders',  'surgical_fix_shoulders'),
            ('fix_spine',      'surgical_fix_spine'),
            ('fix_neck',       'surgical_fix_neck'),
        ]:
            if hasattr(sc, attr):
                sec = getattr(sc, attr)
                feather_key = key.replace('fix_', '') + '_feather'
                pipeline.config['surgical'][key]         = sec.enabled
                pipeline.config['surgical'][feather_key] = sec.feather

        scfg = pipeline.config.get('surgical', {})
        sm   = pipeline.config.get('smooth', {})
        li   = pipeline.config.get('limit_influences', {})
        dbg("\n=== PIPELINE SETTINGS ===")
        dbg(f"  Mesh  : {mesh_obj.name}")
        dbg(f"  Smooth: {sm.get('iterations', '?')} iterations × {sm.get('factor', '?')} factor")
        dbg(f"  Limit : {li.get('max_bones', '?')} max bones")
        for label, en_key, fth_key in [
            ("Fingers",    'fix_fingers',    'fingers_feather'),
            ("Forearms",   'fix_forearms',   'forearms_feather'),
            ("Upper Arms", 'fix_upper_arms', 'upper_arms_feather'),
            ("Shoulders",  'fix_shoulders',  'shoulders_feather'),
            ("Thighs",     'fix_thighs',     'thighs_feather'),
            ("Shins",      'fix_shins',      'shins_feather'),
            ("Spine",      'fix_spine',      'spine_feather'),
            ("Neck",       'fix_neck',       'neck_feather'),
        ]:
            state = 'ON ' if scfg.get(en_key, False) else 'OFF'
            dbg(f"  {label:<12}: {state}  feather={scfg.get(fth_key, '?')}")
        dbg("=========================\n")

        success = pipeline.run_full_pipeline()
        if success:
            dbg(pipeline.get_stage_report())

        for msg in pipeline.warnings[:3]:
            self.report({'WARNING'}, msg)
        for msg in pipeline.errors[:2]:
            self.report({'ERROR'}, msg)

        return success

    def _bind_pipeline_split_parts(self, context, mesh_obj, rig_obj, mesh_type='BODY') -> bool:
        """Two-step split-parts Heat Map binding.

        Step 1 — ARMATURE_AUTO per isolated island.
          Each island is separated before heat diffusion runs, so a lace eyelet
          or sole piece never inherits heat paths from distant bones (thigh,
          upper arm, etc.).  This produces clean, proximity-correct initial
          weights on every island no matter how small.

        Step 2 — Rejoin, then run pipeline refinement once on the full mesh.
          Cross-side fix, surgical tighten, smooth, twist distribution and
          equalization all work with the complete mesh context — correct bone-
          radius estimation, meaningful gradient smoothing, and proper edge
          topology.  Running these stages on 3-vertex micro-islands individually
          produced bad results; on the rejoined mesh they work correctly.
        """
        # ── Step 1: separate by loose islands, AUTO-bind each ────────────────
        for o in context.view_layer.objects:
            o.select_set(False)
        mesh_obj.select_set(True)
        context.view_layer.objects.active = mesh_obj

        # Use temp_override so the operator poll sees the correct active object.
        # Directly assigning context.view_layer.objects.active doesn't always
        # propagate to bpy.context before the operator's poll() runs, causing
        # "Context missing active object" at mode_set.
        with context.temp_override(active_object=mesh_obj, selected_objects=[mesh_obj]):
            bpy.ops.object.mode_set(mode='OBJECT')   # ensure clean state
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.separate(type='LOOSE')
            bpy.ops.object.mode_set(mode='OBJECT')

        parts = [o for o in context.selected_objects if o.type == 'MESH']
        if len(parts) <= 1:
            # Single island — use the normal full pipeline
            return self._bind_pipeline(context, mesh_obj, rig_obj, mesh_type=mesh_type)

        sp_fix = context.scene.autorig_skin.fix_scale

        ok_all = True
        for part in parts:
            ok, _ = self._bind_one(context, part, rig_obj, method='AUTO', use_scale_fix=sp_fix)
            if not ok:
                ok_all = False

        # ── Step 2: rejoin, then refine the combined mesh ─────────────────────
        for o in context.view_layer.objects:
            o.select_set(False)
        for part in parts:
            part.select_set(True)
        context.view_layer.objects.active = mesh_obj

        with context.temp_override(active_object=mesh_obj, selected_objects=list(parts)):
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.ops.object.join()

        # join() merges all modifiers — keep only the first ARMATURE mod
        seen_arm = False
        for mod in list(mesh_obj.modifiers):
            if mod.type == 'ARMATURE':
                if seen_arm:
                    mesh_obj.modifiers.remove(mod)
                else:
                    seen_arm = True

        # Build and configure the refinement pipeline
        try:
            pipeline = DeformationPipeline(mesh_obj, rig_obj, mesh_type=mesh_type)
        except TypeError as e:
            self.report({'ERROR'}, str(e))
            return False

        sp = context.scene.autorig_skin
        pipeline.config['smooth']['iterations']          = sp.smooth_iterations
        pipeline.config['smooth']['factor']              = sp.smooth_factor
        pipeline.config['limit_influences']['max_bones'] = sp.max_influences
        pipeline.config['optimize_highres']              = False  # proxy not useful post-join
        pipeline.config['fix_scale']                     = False  # scale fix only needed for initial bind

        sc = context.scene
        for key, attr in [
            ('fix_fingers',    'surgical_fix_fingers'),
            ('fix_forearms',   'surgical_fix_forearms'),
            ('fix_upper_arms', 'surgical_fix_upper_arms'),
            ('fix_shins',      'surgical_fix_shins'),
            ('fix_thighs',     'surgical_fix_thighs'),
            ('fix_shoulders',  'surgical_fix_shoulders'),
            ('fix_spine',      'surgical_fix_spine'),
            ('fix_neck',       'surgical_fix_neck'),
        ]:
            if hasattr(sc, attr):
                sec = getattr(sc, attr)
                feather_key = key.replace('fix_', '') + '_feather'
                pipeline.config['surgical'][key]         = sec.enabled
                pipeline.config['surgical'][feather_key] = sec.feather

        success = pipeline.run_pipeline_from_existing_weights()
        if success:
            dbg(pipeline.get_stage_report())

        for msg in pipeline.warnings[:3]:
            self.report({'WARNING'}, msg)
        for msg in pipeline.errors[:2]:
            self.report({'ERROR'}, msg)

        return ok_all and success

    def execute(self, context):
        sp  = context.scene.autorig_skin
        rig = _get_target_rig(context)

        if not rig:
            self.report({'ERROR'}, "No armature in scene. Generate the Rigify rig first.")
            return {'CANCELLED'}

        if not _is_generated_rig(rig):
            self.report({'ERROR'},
                        f"'{rig.name}' is a metarig — generate the Rigify rig first, "
                        "then run Smart Bind.")
            return {'CANCELLED'}

        enabled = [item for item in sp.meshes if item.enabled and item.obj]
        if not enabled:
            self.report({'ERROR'}, "No meshes enabled. Run Detect Meshes first.")
            return {'CANCELLED'}

        if sp.fix_scale:
            def _has_unapplied(obj):
                if any(abs(v) > 0.0001 for v in obj.location):
                    return True
                if any(abs(s - 1.0) > 0.0001 for s in obj.scale):
                    return True
                if obj.rotation_mode == 'QUATERNION':
                    q = obj.rotation_quaternion
                    if abs(q.w - 1.0) > 0.0001 or abs(q.x) > 0.0001 or abs(q.y) > 0.0001 or abs(q.z) > 0.0001:
                        return True
                elif obj.rotation_mode == 'AXIS_ANGLE':
                    if abs(obj.rotation_axis_angle[0]) > 0.0001:
                        return True
                else:
                    if any(abs(v) > 0.0001 for v in obj.rotation_euler):
                        return True
                return False

            bad = [item.obj.name for item in enabled if _has_unapplied(item.obj)]
            if bad:
                self.report(
                    {'WARNING'},
                    "Fix Scale: the following meshes have unapplied transforms (location, "
                    "rotation, or scale) and may stretch when bound — apply all transforms "
                    f"first (Ctrl+A → All Transforms): {', '.join(bad)}")

        # mode_set requires a non-None active object even when already in Object mode.
        if context.view_layer.objects.active is None:
            context.view_layer.objects.active = rig
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        ok_auto = ok_env = ok_fail = 0

        for item in enabled:
            obj    = item.obj
            method = sp.bind_method

            # Heat Map path: full pipeline + distance-based joint equalization
            if method == 'PIPELINE' and item.mesh_type in ('BODY', 'CLOTHING'):
                if sp.split_parts:
                    ok = self._bind_pipeline_split_parts(context, obj, rig, mesh_type=item.mesh_type)
                else:
                    ok = self._bind_pipeline(context, obj, rig, mesh_type=item.mesh_type)
                if not ok:
                    ok_fail += 1
                    continue
                arm_mod = next((m for m in obj.modifiers if m.type == 'ARMATURE'), None)
                if arm_mod:
                    arm_mod.use_deform_preserve_volume = sp.preserve_volume
                _reorder_arm_modifier(context, obj)
                ok_auto += 1
                continue

            # Direct-assign path: rigid meshes — skip auto-weight, one bone, 100% weight
            direct_bone = _BIND_DIRECT_BONE.get(item.mesh_type)
            if direct_bone:
                ok = self._direct_bind(context, obj, rig, direct_bone)
                if ok:
                    arm_mod = next((m for m in obj.modifiers if m.type == 'ARMATURE'), None)
                    if arm_mod:
                        arm_mod.use_deform_preserve_volume = sp.preserve_volume
                    ok_auto += 1
                else:
                    self.report({'WARNING'}, f"{obj.name}: bone '{direct_bone}' not found in rig — skipped.")
                    ok_fail += 1
                continue

            # Standard bind path (also used for eyes/teeth/tongue when method==PIPELINE)
            bind_method = 'AUTO' if method == 'PIPELINE' else method
            allowed     = _BIND_BONE_FILTER.get(item.mesh_type)

            if sp.split_parts:
                success, fallback = self._bind_split_parts(context, obj, rig, method=bind_method, use_scale_fix=sp.fix_scale)
            else:
                success, fallback = self._bind_one(context, obj, rig, method=bind_method, use_scale_fix=sp.fix_scale)

            if not success:
                ok_fail += 1
                continue

            if allowed:
                self._strip_disallowed_groups(obj, allowed)
                if not obj.vertex_groups:
                    self._assign_fallback_group(obj, rig, item.mesh_type)

            for o in context.view_layer.objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            if obj.vertex_groups:
                bpy.ops.object.vertex_group_limit_total(
                    group_select_mode='ALL', limit=sp.max_influences)
                bpy.ops.object.vertex_group_normalize_all(lock_active=False)

            arm_mod = next((m for m in obj.modifiers if m.type == 'ARMATURE'), None)
            if arm_mod:
                arm_mod.use_deform_preserve_volume = sp.preserve_volume
            _reorder_arm_modifier(context, obj)

            if fallback:
                ok_env += 1
            else:
                ok_auto += 1

        for o in context.view_layer.objects:
            o.select_set(False)
        rig.select_set(True)
        context.view_layer.objects.active = rig

        parts = []
        if ok_auto:  parts.append(f"{ok_auto} bound (auto)")
        if ok_env:   parts.append(f"{ok_env} bound (envelope fallback)")
        if ok_fail:  parts.append(f"{ok_fail} FAILED")
        sp.bind_report = " • ".join(parts) if parts else "Nothing bound"
        self.report({'INFO'}, sp.bind_report)
        return {'FINISHED'}


class AUTORIG_OT_Unbind(bpy.types.Operator):
    """Remove Armature modifier and vertex groups from the selected mesh"""
    bl_idname  = "autorig.unbind"
    bl_label   = "Unbind"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        sp = context.scene.autorig_skin

        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'WARNING'}, "Select a mesh object in the scene to unbind.")
            return {'CANCELLED'}

        changed = False
        for mod in list(obj.modifiers):
            if mod.type == 'ARMATURE':
                obj.modifiers.remove(mod)
                changed = True
        if obj.vertex_groups:
            obj.vertex_groups.clear()
            changed = True
        if obj.parent and obj.parent.type == 'ARMATURE':
            obj.parent = None
            changed = True

        msg = f"Unbound '{obj.name}'." if changed else f"'{obj.name}' was not bound."
        sp.bind_report = msg
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class AUTORIG_OT_CleanupWeights(bpy.types.Operator):
    """Run weight cleanup operations across all bound meshes"""
    bl_idname  = "autorig.cleanup_weights"
    bl_label   = "Cleanup Weights"
    bl_options = {'REGISTER', 'UNDO'}

    action: bpy.props.EnumProperty(
        items=[
            ('NORMALIZE',   'Normalize',       'Normalize all weights to sum 1.0'),
            ('LIMIT',       'Limit Influences', 'Limit max bone influences per vertex'),
            ('REMOVE_ZERO', 'Remove Zero',      'Strip zero-weight vertex group entries'),
            ('SMOOTH',      'Smooth Weights',   'Smooth weight gradients to reduce harsh transitions'),
            ('CLEAN',       'Full Cleanup',     'Normalize + Limit + Remove Zero + Smooth'),
        ],
        default='CLEAN',
    )

    @classmethod
    def poll(cls, context):
        # Allow in both Object and Weight Paint modes so buttons never gray out.
        return (context.active_object is not None
                and context.mode in {'OBJECT', 'PAINT_WEIGHT'})

    def _bound_meshes(self, context):
        sp  = context.scene.autorig_skin
        rig = _get_target_rig(context)
        return [item.obj for item in sp.meshes
                if item.obj and any(m.type == 'ARMATURE' and m.object == rig
                                    for m in item.obj.modifiers)]

    @staticmethod
    def _numpy_smooth(obj, iters=2, factor=0.3):
        """Gentle Laplacian smooth — freezes face, finger, and forearm columns."""
        _skip_kw = frozenset([
            'thumb', 'f_index', 'f_middle', 'f_ring', 'f_pinky',
            'finger', 'digit', 'palm', 'forearm', 'lower_arm', 'lowerarm',
            'jaw', 'chin', 'cheek', 'brow', 'lid', 'nose', 'lip',
            'forehead', 'temple', 'ear', 'tongue', 'teeth', 'eye',
        ])
        def _should_skip(n):
            for p in ('DEF-', 'ORG-', 'MCH-', 'WGT-'):
                if n.startswith(p): n = n[len(p):]; break
            return any(kw in n.lower() for kw in _skip_kw)

        mesh        = obj.data
        n_verts     = len(mesh.vertices)
        group_names = [g.name for g in obj.vertex_groups]
        n_groups    = len(group_names)
        if n_groups == 0:
            return

        ws = np.zeros((n_verts, n_groups), dtype=np.float32)
        for v in mesh.vertices:
            for g in v.groups:
                if g.group < n_groups:
                    ws[v.index, g.group] = g.weight
        ws_before = ws.copy()

        skip_cols = [gi for gi, gn in enumerate(group_names) if _should_skip(gn)]
        frozen    = {gi: ws[:, gi].copy() for gi in skip_cols}

        edges = np.array([(e.vertices[0], e.vertices[1])
                          for e in mesh.edges], dtype=np.int32)
        if len(edges) == 0:
            return
        src = np.concatenate([edges[:, 0], edges[:, 1]])
        dst = np.concatenate([edges[:, 1], edges[:, 0]])
        degree   = np.zeros(n_verts, dtype=np.float32)
        np.add.at(degree, dst, 1.0)
        deg_safe = np.where(degree > 0, degree, 1.0)

        for _ in range(iters):
            nbr_sum = np.zeros_like(ws)
            np.add.at(nbr_sum, dst, ws[src])
            nbr_avg = nbr_sum / deg_safe[:, np.newaxis]
            ws = ws * (1.0 - factor) + nbr_avg * factor

        for gi, col in frozen.items():
            ws[:, gi] = col

        totals = ws.sum(axis=1)
        valid  = totals > 0.001
        if valid.any():
            ws[valid] = ws[valid] / totals[valid, np.newaxis]

        _write_weights_back(obj, group_names, ws_before, ws)

    @staticmethod
    def _numpy_ops(obj, max_influences, do_normalize, do_limit, do_remove_zero):
        """Normalize / limit / remove-zero via numpy — works in any Blender mode."""
        mesh        = obj.data
        n_verts     = len(mesh.vertices)
        group_names = [g.name for g in obj.vertex_groups]
        n_groups    = len(group_names)
        if n_groups == 0:
            return
        w = np.zeros((n_verts, n_groups), dtype=np.float32)
        for v in mesh.vertices:
            for g in v.groups:
                if g.group < n_groups:
                    w[v.index, g.group] = g.weight
        w_old = w.copy()
        if do_remove_zero:
            w[w < 0.001] = 0.0
        if do_limit:
            for vi in range(n_verts):
                row = w[vi]
                if np.count_nonzero(row) > max_influences:
                    top     = np.argpartition(row, -max_influences)[-max_influences:]
                    new_row = np.zeros_like(row)
                    new_row[top] = row[top]
                    w[vi]   = new_row
        if do_normalize:
            totals = w.sum(axis=1)
            valid  = totals > 0.001
            if valid.any():
                w[valid] = w[valid] / totals[valid, np.newaxis]
        _write_weights_back(obj, group_names, w_old, w)

    def execute(self, context):
        sp          = context.scene.autorig_skin
        obj         = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'WARNING'}, "Select a mesh object in the viewport.")
            return {'CANCELLED'}
        if not obj.vertex_groups:
            self.report({'WARNING'}, f"'{obj.name}' has no vertex groups.")
            return {'CANCELLED'}

        orig_active = context.view_layer.objects.active
        orig_mode   = context.mode

        do_norm   = self.action in ('NORMALIZE', 'CLEAN')
        do_limit  = self.action in ('LIMIT',     'CLEAN')
        do_zero   = self.action in ('REMOVE_ZERO','CLEAN')
        do_smooth = self.action in ('SMOOTH',    'CLEAN')

        # Normalize / limit / remove-zero use pure numpy — no mode switch needed.
        if do_norm or do_limit or do_zero:
            self._numpy_ops(obj, sp.max_influences, do_norm, do_limit, do_zero)

        # Smooth — numpy-based, no mode switch, respects frozen bone columns.
        if do_smooth:
            self._numpy_smooth(obj)
            # Smoothing spreads weight to neighbouring bones, which re-introduces
            # influences above the limit. Re-enforce the limit (and re-normalize)
            # AFTER smoothing so "Full Cleanup" actually honours max_influences.
            if do_limit:
                self._numpy_ops(obj, sp.max_influences,
                                do_normalize=do_norm, do_limit=True, do_remove_zero=False)

        # Restore original active object and mode (including Weight Paint).
        if orig_active and orig_active.name in bpy.data.objects:
            for o in context.view_layer.objects:
                o.select_set(False)
            # When restoring Weight Paint, re-select the rig alongside the mesh
            # so Blender auto-enters Pose mode on the rig — same as EnterWeightPaint.
            # Without this the armature loses its Pose mode link and bones go gray.
            if orig_mode == 'PAINT_WEIGHT' and orig_active.type == 'MESH':
                rig = _get_target_rig(context)
                if rig:
                    rig.select_set(True)
            orig_active.select_set(True)
            context.view_layer.objects.active = orig_active
            if orig_mode == 'PAINT_WEIGHT' and orig_active.type == 'MESH':
                bpy.ops.object.mode_set(mode='WEIGHT_PAINT')

        self.report({'INFO'}, f"Cleanup applied to '{obj.name}'")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Tab draw helper — called from AUTORIG_PT_Main (Skin tab)
# ---------------------------------------------------------------------------

def draw_skinning_tab(layout, context):
    sp  = context.scene.autorig_skin
    rig = _get_target_rig(context)
    obj = context.active_object
    in_wp = bool(obj and obj.type == 'MESH' and context.mode == 'PAINT_WEIGHT')

    # ── MESH SETUP (collapsible) ──────────────────────────────────
    setup_box = layout.box()
    setup_hdr = setup_box.row(align=True)
    setup_hdr.prop(sp, "show_mesh_setup",
                   icon='DISCLOSURE_TRI_DOWN' if sp.show_mesh_setup else 'DISCLOSURE_TRI_RIGHT',
                   emboss=False, text="")
    setup_hdr.label(text="Mesh Setup", icon='OUTLINER_OB_MESH')

    if sp.show_mesh_setup:
        setup_box.operator("autorig.detect_meshes", icon='VIEWZOOM')
        if sp.meshes:
            for item in sp.meshes:
                if not item.obj:
                    continue
                row = setup_box.row(align=True)
                row.prop(item, "enabled", text="")
                has_bind = any(m.type == 'ARMATURE' for m in item.obj.modifiers)
                row.label(text=item.obj.name,
                          icon='CHECKMARK' if has_bind else 'LAYER_USED')
                row.prop(item, "mesh_type", text="")

    layout.separator(factor=0.4)

    # ── SMART BIND ────────────────────────────────────────────────
    bind_box = layout.box()
    bind_hdr = bind_box.row(align=True)
    bind_hdr.prop(sp, "show_smart_bind",
                  icon='DISCLOSURE_TRI_DOWN' if sp.show_smart_bind else 'DISCLOSURE_TRI_RIGHT',
                  emboss=False, text="")
    bind_hdr.label(text="Smart Bind", icon='SNAP_ON')

    if sp.show_smart_bind:
        mrow = bind_box.row(align=True)
        mrow.prop(sp, "bind_method", text="")
        mrow.prop(sp, "max_influences", text="Max", slider=True)

        crow = bind_box.row(align=True)
        crow.prop(sp, "preserve_volume")
        crow.prop(sp, "split_parts")
        crow.prop(sp, "fix_scale")

        if sp.fix_scale:
            warn_row = bind_box.row()
            warn_row.label(
                text="Apply all transforms before binding (Ctrl+A → All Transforms)",
                icon='ERROR')

        if sp.bind_method == 'PIPELINE':
            pbox = bind_box.box()
            phdr = pbox.row(align=True)
            phdr.prop(sp, "show_pipeline_opts",
                      icon='DISCLOSURE_TRI_DOWN' if sp.show_pipeline_opts else 'DISCLOSURE_TRI_RIGHT',
                      emboss=False, text="")
            phdr.label(text="Weight Options", icon='SETTINGS')
        if sp.bind_method == 'PIPELINE' and sp.show_pipeline_opts:
            pcol = pbox.column(align=True)
            srow = pcol.row(align=True)
            srow.prop(sp, "smooth_iterations", text="Smooth", slider=True)
            srow.prop(sp, "smooth_factor", text="×", slider=True)
            pcol.prop(sp, "optimize_highres", icon='MOD_DECIM')

            sbox = pbox.box()
            shdr = sbox.row(align=True)
            shdr.prop(sp, "show_tighten",
                      icon='DISCLOSURE_TRI_DOWN' if sp.show_tighten else 'DISCLOSURE_TRI_RIGHT',
                      emboss=False, text="")
            shdr.label(text="Joint Tightening", icon='SNAP_VERTEX')
            if sp.show_tighten:
                prow = sbox.row(align=True)
                for preset in ('DEFAULT', 'GENTLE', 'AGGRESSIVE'):
                    op = prow.operator("autorig.surgical_preset", text=preset.capitalize())
                    op.preset = preset
                _sec_attrs = [
                    ('surgical_fix_fingers',    "Fingers"),
                    ('surgical_fix_forearms',   "Forearms"),
                    ('surgical_fix_upper_arms', "Upper Arms"),
                    ('surgical_fix_shins',      "Shins"),
                    ('surgical_fix_thighs',     "Thighs"),
                    ('surgical_fix_shoulders',  "Shoulders"),
                    ('surgical_fix_spine',      "Spine"),
                    ('surgical_fix_neck',       "Neck"),
                ]
                scol = sbox.column(align=True)
                for attr, lbl in _sec_attrs:
                    sec = getattr(context.scene, attr, None)
                    if sec is None:
                        continue
                    row = scol.row(align=True)
                    row.prop(sec, "enabled", text="")
                    row.label(text=lbl)
                    sub = row.row(align=True)
                    sub.enabled = sec.enabled
                    sub.prop(sec, "feather", text="", slider=True)

        bind_btn = bind_box.row(align=True)
        bind_btn.scale_y = 1.6
        sub_bind = bind_btn.row(align=True)
        sub_bind.enabled = bool(rig and sp.meshes)
        sub_bind.operator("autorig.smart_bind", icon='MOD_ARMATURE')
        sub_unbind = bind_btn.row(align=True)
        sub_unbind.enabled = bool(obj and obj.type == 'MESH')
        sub_unbind.operator("autorig.unbind", text="Unbind", icon='X')

        if sp.bind_report:
            bind_box.label(text=sp.bind_report, icon='INFO')


class AUTORIG_PT_Skinning(bpy.types.Panel):
    bl_label       = "Skinning"
    bl_idname      = "AUTORIG_PT_Skinning"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = "Easy Rigify"
    bl_options     = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context): return False

    def draw_header(self, _context):
        self.layout.label(text="", icon='MOD_ARMATURE')

    def draw(self, context):
        draw_skinning_tab(self.layout, context)


class AUTORIG_OT_SurgicalPreset(bpy.types.Operator):
    bl_idname  = "autorig.surgical_preset"
    bl_label   = "Surgical Preset"
    bl_options = {'REGISTER', 'UNDO'}
    preset: bpy.props.EnumProperty(items=[
        ('DEFAULT',    "Default",    ""),
        ('GENTLE',     "Gentle",     ""),
        ('AGGRESSIVE', "Aggressive", ""),
    ])

    def execute(self, context):
        sc = context.scene
        # feather → bleeding-weight multiplier:  0.1→5%  0.5→25%  1.0→50%  1.5→75%  2.0→100%
        presets = {
            # DEFAULT: moderate reduction — bleeding weight kept at ~35-50%
            'DEFAULT':    {'fingers': (True,0.7),  'forearms': (True,0.8),  'upper_arms': (True,1.0),
                           'shins':   (True,0.8),  'thighs':   (True,1.0),  'shoulders':  (True,1.2),
                           'spine':   (False,2.0), 'neck':     (False,2.0)},
            # GENTLE: light clean-up — bleeding weight kept at ~65-75%
            'GENTLE':     {'fingers': (True,1.3),  'forearms': (True,1.4),  'upper_arms': (True,1.5),
                           'shins':   (True,1.4),  'thighs':   (True,1.5),  'shoulders':  (True,1.6),
                           'spine':   (False,2.0), 'neck':     (False,2.0)},
            # AGGRESSIVE: heavy reduction — bleeding weight kept at ~10-25%
            'AGGRESSIVE': {'fingers': (True,0.3),  'forearms': (True,0.4),  'upper_arms': (True,0.5),
                           'shins':   (True,0.4),  'thighs':   (True,0.5),  'shoulders':  (True,0.6),
                           'spine':   (False,2.0), 'neck':     (False,2.0)},
        }
        p = presets.get(self.preset, {})
        attr_map = {
            'fingers':    'surgical_fix_fingers',
            'forearms':   'surgical_fix_forearms',
            'upper_arms': 'surgical_fix_upper_arms',
            'shins':      'surgical_fix_shins',
            'thighs':     'surgical_fix_thighs',
            'shoulders':  'surgical_fix_shoulders',
            'spine':      'surgical_fix_spine',
            'neck':       'surgical_fix_neck',
        }
        for key, attr in attr_map.items():
            if key in p and hasattr(sc, attr):
                sec = getattr(sc, attr)
                sec.enabled, sec.feather = p[key]
        return {'FINISHED'}
