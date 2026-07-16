# joint_cleaner.py — RigifyJointCleaner, joint analysis/clean operators,
#                    DEF bone influence report, and smooth weight operators.
import bpy
import re
import numpy as np
from mathutils import Vector

from .utils import WeightDataAccess, _write_weights_back


class RigifyJointCleaner:
    """Joint weight cleaner for Rigify DEF- prefixed deform bones.

    Uses bone-segment distance with feathered falloff and redistributes
    removed weight to neighboring bones via inverse-distance weighting.
    """

    RIGIFY_DEF_PATTERNS = {
        'DEF-spine':     {'type': 'SPINE',       'radius_mult': 0.7},
        'DEF-spine.001': {'type': 'SPINE_LOWER',  'radius_mult': 0.7},
        'DEF-spine.002': {'type': 'SPINE_MID',    'radius_mult': 0.6},
        'DEF-spine.003': {'type': 'SPINE_UPPER',  'radius_mult': 0.6},
        'DEF-spine.004': {'type': 'NECK',         'radius_mult': 0.5},
        'DEF-spine.005': {'type': 'NECK',         'radius_mult': 0.4},
        'DEF-spine.006': {'type': 'NECK_TOP',     'radius_mult': 0.35},
        'DEF-shoulder.L':{'type': 'SHOULDER',     'radius_mult': 0.7},
        'DEF-shoulder.R':{'type': 'SHOULDER',     'radius_mult': 0.7},
        'DEF-upper_arm.L':    {'type': 'UPPER_ARM', 'radius_mult': 0.45},
        'DEF-upper_arm.R':    {'type': 'UPPER_ARM', 'radius_mult': 0.45},
        'DEF-upper_arm.L.001':{'type': 'TWIST',     'radius_mult': 0.40},
        'DEF-upper_arm.R.001':{'type': 'TWIST',     'radius_mult': 0.40},
        'DEF-forearm.L':      {'type': 'FOREARM',   'radius_mult': 0.35},
        'DEF-forearm.R':      {'type': 'FOREARM',   'radius_mult': 0.35},
        'DEF-forearm.L.001':  {'type': 'TWIST',     'radius_mult': 0.30},
        'DEF-forearm.R.001':  {'type': 'TWIST',     'radius_mult': 0.30},
        'DEF-hand.L':         {'type': 'HAND',      'radius_mult': 0.30},
        'DEF-hand.R':         {'type': 'HAND',      'radius_mult': 0.30},
        'DEF-thigh.L':        {'type': 'THIGH',     'radius_mult': 0.70},
        'DEF-thigh.R':        {'type': 'THIGH',     'radius_mult': 0.70},
        'DEF-thigh.L.001':    {'type': 'TWIST',     'radius_mult': 0.60},
        'DEF-thigh.R.001':    {'type': 'TWIST',     'radius_mult': 0.60},
        'DEF-shin.L':         {'type': 'SHIN',      'radius_mult': 0.40},
        'DEF-shin.R':         {'type': 'SHIN',      'radius_mult': 0.40},
        'DEF-shin.L.001':     {'type': 'TWIST',     'radius_mult': 0.35},
        'DEF-shin.R.001':     {'type': 'TWIST',     'radius_mult': 0.35},
        'DEF-foot.L':    {'type': 'FOOT',         'radius_mult': 0.35},
        'DEF-foot.R':    {'type': 'FOOT',         'radius_mult': 0.35},
        'DEF-toe.L':     {'type': 'TOE',          'radius_mult': 0.20},
        'DEF-toe.R':     {'type': 'TOE',          'radius_mult': 0.20},
        'DEF-thumb.01.L':{'type': 'THUMB_BASE',   'radius_mult': 0.18},
        'DEF-thumb.02.L':{'type': 'THUMB_MID',    'radius_mult': 0.15},
        'DEF-thumb.03.L':{'type': 'THUMB_TIP',    'radius_mult': 0.12},
        'DEF-thumb.01.R':{'type': 'THUMB_BASE',   'radius_mult': 0.18},
        'DEF-thumb.02.R':{'type': 'THUMB_MID',    'radius_mult': 0.15},
        'DEF-thumb.03.R':{'type': 'THUMB_TIP',    'radius_mult': 0.12},
        'DEF-palm.01.L': {'type': 'PALM_INDEX',   'radius_mult': 0.22},
        'DEF-palm.02.L': {'type': 'PALM_MIDDLE',  'radius_mult': 0.22},
        'DEF-palm.03.L': {'type': 'PALM_RING',    'radius_mult': 0.20},
        'DEF-palm.04.L': {'type': 'PALM_PINKY',   'radius_mult': 0.18},
        'DEF-palm.01.R': {'type': 'PALM_INDEX',   'radius_mult': 0.22},
        'DEF-palm.02.R': {'type': 'PALM_MIDDLE',  'radius_mult': 0.22},
        'DEF-palm.03.R': {'type': 'PALM_RING',    'radius_mult': 0.20},
        'DEF-palm.04.R': {'type': 'PALM_PINKY',   'radius_mult': 0.18},
        'DEF-f_index.01.L': {'type': 'FINGER',    'radius_mult': 0.15},
        'DEF-f_index.02.L': {'type': 'FINGER',    'radius_mult': 0.13},
        'DEF-f_index.03.L': {'type': 'FINGER_TIP','radius_mult': 0.10},
        'DEF-f_middle.01.L':{'type': 'FINGER',    'radius_mult': 0.15},
        'DEF-f_middle.02.L':{'type': 'FINGER',    'radius_mult': 0.13},
        'DEF-f_middle.03.L':{'type': 'FINGER_TIP','radius_mult': 0.10},
        'DEF-f_ring.01.L':  {'type': 'FINGER',    'radius_mult': 0.14},
        'DEF-f_ring.02.L':  {'type': 'FINGER',    'radius_mult': 0.12},
        'DEF-f_ring.03.L':  {'type': 'FINGER_TIP','radius_mult': 0.09},
        'DEF-f_pinky.01.L': {'type': 'FINGER',    'radius_mult': 0.13},
        'DEF-f_pinky.02.L': {'type': 'FINGER',    'radius_mult': 0.11},
        'DEF-f_pinky.03.L': {'type': 'FINGER_TIP','radius_mult': 0.08},
        'DEF-f_index.01.R': {'type': 'FINGER',    'radius_mult': 0.15},
        'DEF-f_index.02.R': {'type': 'FINGER',    'radius_mult': 0.13},
        'DEF-f_index.03.R': {'type': 'FINGER_TIP','radius_mult': 0.10},
        'DEF-f_middle.01.R':{'type': 'FINGER',    'radius_mult': 0.15},
        'DEF-f_middle.02.R':{'type': 'FINGER',    'radius_mult': 0.13},
        'DEF-f_middle.03.R':{'type': 'FINGER_TIP','radius_mult': 0.10},
        'DEF-f_ring.01.R':  {'type': 'FINGER',    'radius_mult': 0.14},
        'DEF-f_ring.02.R':  {'type': 'FINGER',    'radius_mult': 0.12},
        'DEF-f_ring.03.R':  {'type': 'FINGER_TIP','radius_mult': 0.09},
        'DEF-f_pinky.01.R': {'type': 'FINGER',    'radius_mult': 0.13},
        'DEF-f_pinky.02.R': {'type': 'FINGER',    'radius_mult': 0.11},
        'DEF-f_pinky.03.R': {'type': 'FINGER_TIP','radius_mult': 0.08},
        'DEF-jaw':           {'type': 'JAW',       'radius_mult': 0.30},
        'DEF-chin':          {'type': 'FACE',      'radius_mult': 0.25},
        'DEF-eye.L':         {'type': 'EYE',       'radius_mult': 0.20},
        'DEF-eye.R':         {'type': 'EYE',       'radius_mult': 0.20},
        'DEF-nose':          {'type': 'NOSE',      'radius_mult': 0.20},
    }

    # Joint types that bleed the most — use tighter feather + power curve
    _TIGHT_JOINTS = {'UPPER_ARM', 'THIGH'}

    # Bone types never touched by the cleaner — fine articulation that
    # needs its weights left exactly as painted.
    _NEVER_CLEAN = {
        'TWIST',                                        # limb .001 joint-transition bones
        'FOREARM', 'SHIN',
        'FINGER', 'FINGER_TIP',
        'THUMB_BASE', 'THUMB_MID', 'THUMB_TIP',
        'PALM_INDEX', 'PALM_MIDDLE', 'PALM_RING', 'PALM_PINKY',
        'FOOT', 'TOE',
        'FACE', 'JAW', 'EYE', 'NOSE',
    }

    # Auto-radius: use the Nth percentile of the bone's influenced-vertex distances.
    # 72 = protect the inner 72% of the influence spread; the outer 28% enters the
    # feather zone and is candidates for removal.  Falls back to static radius_mult
    # when fewer than _MIN_SAMPLES vertices are influenced.
    _RADIUS_PERCENTILE = 72
    _MIN_SAMPLES       = 8

    # Cleaning-strength scale applied to the auto-radius per level.
    # Light: 10% bigger safe zone → gentler removal (~10% less weight removed).
    # Aggressive: 10% smaller safe zone → stronger removal (~10-15% more removed).
    _LEVEL_RADIUS_SCALE = {
        'LIGHT':      1.10,
        'MODERATE':   1.00,
        'AGGRESSIVE': 0.90,
    }

    # Which types each mode cleans (all subject to _NEVER_CLEAN exclusion)
    _MODE_TYPES = {
        'LIGHT':      {'UPPER_ARM', 'HAND'},
        'MODERATE':   {'UPPER_ARM', 'HAND', 'SHOULDER'},
        'AGGRESSIVE': None,  # all types except _NEVER_CLEAN
    }

    def __init__(self, mesh_obj, armature_obj):
        self.mesh_obj     = mesh_obj
        self.armature_obj = armature_obj
        self.mesh         = mesh_obj.data
        self.armature     = armature_obj.data
        self.access       = WeightDataAccess()

    def get_all_rigify_joints(self) -> list:
        joints = []
        for bone in self.armature.bones:
            if not bone.use_deform:
                continue
            # Strip ORG- prefix some Rigify versions emit
            lookup = bone.name
            if lookup.startswith('ORG-'):
                lookup = lookup[4:]
            info = (self.RIGIFY_DEF_PATTERNS.get(bone.name)
                    or self.RIGIFY_DEF_PATTERNS.get(lookup)
                    or self._pattern_match_def_bone(bone.name))
            if not info:
                continue
            info = info.copy()
            info['bone']       = bone
            info['bone_name']  = bone.name
            info['head_world'] = self.armature_obj.matrix_world @ bone.head_local
            info['tail_world'] = self.armature_obj.matrix_world @ bone.tail_local
            bone_len           = (bone.tail_local - bone.head_local).length
            info['radius']     = bone_len * info['radius_mult']
            joints.append(info)
        return joints

    def _pattern_match_def_bone(self, bone_name: str):
        if not bone_name.startswith('DEF-'):
            return None
        base = bone_name[4:]
        # Numbered limb twist/transition bones (.L.001, .R.001, .001, etc.) —
        # never clean these; they carry the joint-blend weights.
        if re.search(r'\.\d{3}$', base):
            b = base.lower()
            for kw in ('upper_arm', 'forearm', 'thigh', 'shin', 'hand', 'foot', 'toe'):
                if kw in b:
                    return {'type': 'TWIST', 'radius_mult': 0.4}
        if base.startswith('f_') and any(f in base for f in ('index','middle','ring','pinky')):
            if '.01.' in base: return {'type': 'FINGER',     'radius_mult': 0.15}
            if '.02.' in base: return {'type': 'FINGER',     'radius_mult': 0.13}
            if '.03.' in base: return {'type': 'FINGER_TIP', 'radius_mult': 0.10}
        if base.startswith('thumb'):
            if '.01.' in base: return {'type': 'THUMB_BASE', 'radius_mult': 0.18}
            if '.02.' in base: return {'type': 'THUMB_MID',  'radius_mult': 0.15}
            if '.03.' in base: return {'type': 'THUMB_TIP',  'radius_mult': 0.12}
        if base.startswith('palm'):  return {'type': 'PALM_INDEX',  'radius_mult': 0.22}
        b = base.lower()
        limb_map = [
            ('upper_arm', 'UPPER_ARM', 0.45), ('forearm', 'FOREARM', 0.35),
            ('hand',      'HAND',      0.30), ('shoulder','SHOULDER',0.70),
            ('thigh',     'THIGH',     0.70), ('shin',    'SHIN',    0.40),
            ('foot',      'FOOT',      0.35), ('toe',     'TOE',     0.20),
            ('spine',     'SPINE',     0.60), ('neck',    'NECK',    0.40),
            ('jaw',       'JAW',       0.30), ('eye',     'EYE',     0.20),
            ('lip',       'FACE',      0.15), ('brow',    'FACE',    0.20),
            ('nose',      'NOSE',      0.20), ('cheek',   'FACE',    0.30),
            ('chin',      'FACE',      0.25), ('ear',     'FACE',    0.20),
            ('tongue',    'FACE',      0.15), ('temple',  'FACE',    0.25),
            ('breast',    'DEFORM',    0.50),
        ]
        for kw, jtype, mult in limb_map:
            if kw in b:
                return {'type': jtype, 'radius_mult': mult}
        return {'type': 'DEFORM', 'radius_mult': 0.5}

    @staticmethod
    def _seg_dist(point, a, b) -> float:
        ab  = b - a
        abl = ab.length
        if abl < 0.0001:
            return (point - a).length
        t = max(0.0, min(abl, (point - a).dot(ab / abl)))
        return (point - (a + ab / abl * t)).length

    @staticmethod
    def _seg_proj_perp(point, a, b):
        """Return (t, perp): t = normalized axial position [0,1] of `point` along the
        a→b segment (clamped), perp = distance to that clamped closest point (same as
        _seg_dist). Used for the tapered (cone) radius."""
        ab  = b - a
        ab2 = ab.dot(ab)
        if ab2 < 1e-8:
            return 0.0, (point - a).length
        t       = max(0.0, min(1.0, (point - a).dot(ab) / ab2))
        closest = a + ab * t
        return t, (point - closest).length

    def _adaptive_cone(self, group, head, tail, static_fallback):
        """Tapered radius (r_head, r_tail) from the bone's influenced vertices — a
        limb is a cone, so a single radius clips the wide end and leaks at the narrow
        end. Takes the _RADIUS_PERCENTILE of perp distances in the proximal/distal
        halves and extrapolates a line to the bone ends. Falls back to a single radius
        (both ends equal) when samples are sparse."""
        gi = group.index
        ts, ps = [], []
        for vert in self.mesh.vertices:
            w = next((g.weight for g in vert.groups if g.group == gi), 0.0)
            if w > 0.001:
                t, p = self._seg_proj_perp(vert.co, head, tail)
                ts.append(t); ps.append(p)
        if len(ps) < self._MIN_SAMPLES:
            return static_fallback, static_fallback
        ts = np.asarray(ts, dtype=np.float32)
        ps = np.asarray(ps, dtype=np.float32)

        def _band(lo, hi):
            m = (ts >= lo) & (ts < hi)
            if int(np.count_nonzero(m)) < 4:
                return None
            return float(np.percentile(ps[m], self._RADIUS_PERCENTILE))

        rp = _band(0.0, 0.5)
        rd = _band(0.5, 1.0)
        if rp is None and rd is None:
            R = float(np.percentile(ps, self._RADIUS_PERCENTILE))
            return R, R
        if rp is None: rp = rd
        if rd is None: rd = rp
        slope  = (rd - rp) / 0.5          # per normalized-t unit (band centres 0.25/0.75)
        r_head = rp - slope * 0.25
        r_tail = rd + slope * 0.25
        flo    = max(static_fallback * 0.3, 1e-4)
        return max(r_head, flo), max(r_tail, flo)


    def analyze_problem_joints(self) -> list:
        """Quick focused analysis on the joint types most prone to bleeding.

        Only reports types that are actually CLEANABLE (problem set minus _NEVER_CLEAN)
        so the analysis matches what Fix/Clean will act on — previously FOOT was flagged
        but silently skipped at clean time."""
        problem_types = {'UPPER_ARM', 'THIGH', 'SHOULDER', 'HAND'}
        matrix_inv = self.mesh_obj.matrix_world.inverted()
        report     = []
        for info in self.get_all_rigify_joints():
            if info['type'] not in problem_types or info['type'] in self._NEVER_CLEAN:
                continue
            group = self.mesh_obj.vertex_groups.get(info['bone_name'])
            if not group:
                continue
            head = matrix_inv @ info['head_world']
            tail = matrix_inv @ info['tail_world']
            # Same tapered (cone) radius the cleaner uses.
            r_head, r_tail = self._adaptive_cone(group, head, tail, info['radius'])
            total = bleeding = 0
            max_b = 0.0
            for vert in self.mesh.vertices:
                w = next((g.weight for g in vert.groups if g.group == group.index), 0.0)
                if w < 0.001:
                    continue
                total += 1
                t, dist = self._seg_proj_perp(vert.co, head, tail)
                rad     = r_head + (r_tail - r_head) * t
                if dist > rad:
                    bleeding += 1
                    max_b = max(max_b, dist - rad)
            if total == 0:
                continue
            pct = bleeding / total * 100
            report.append({
                'bone':       info['bone_name'],
                'type':       info['type'],
                'influenced': total,
                'bleeding':   bleeding,
                'bleed_pct':  round(pct, 1),
                'max_bleed':  round(max_b, 4),
                'radius':     round((r_head + r_tail) * 0.5, 4),
                'radius_src': 'auto' if total >= self._MIN_SAMPLES else 'static',
                'status':     'FIX'   if pct > 20 else
                              'CHECK' if pct > 10 else 'OK',
            })
        return report

    def clean_all_joints(self, mode='AGGRESSIVE', level='MODERATE', problem_bones=None) -> list:
        """Clean joints on the mesh.

        mode         — bone-type scope filter (pipeline use); 'AGGRESSIVE' = all bones.
        level        — cleaning strength: 'LIGHT', 'MODERATE', or 'AGGRESSIVE'.
        problem_bones — optional whitelist of bone names to clean (Fix Bleeding path).
        """
        allowed_types = self._MODE_TYPES.get(mode)  # None = all types except _NEVER_CLEAN
        results = []
        for info in self.get_all_rigify_joints():
            if info['type'] in self._NEVER_CLEAN:
                continue
            if allowed_types is not None and info['type'] not in allowed_types:
                continue
            if problem_bones is not None and info['bone_name'] not in problem_bones:
                continue
            ok = self._clean_single_joint(info, level=level)
            results.append({'bone': info['bone_name'], 'type': info['type'], 'success': ok})
        return results

    def _clean_single_joint(self, info, level='MODERATE') -> bool:
        group = self.mesh_obj.vertex_groups.get(info['bone_name'])
        if not group:
            return False

        matrix_inv = self.mesh_obj.matrix_world.inverted()
        head       = matrix_inv @ info['head_world']
        tail       = matrix_inv @ info['tail_world']
        scale      = self._LEVEL_RADIUS_SCALE.get(level, 1.0)
        r_head, r_tail = self._adaptive_cone(group, head, tail, info['radius'])
        r_head *= scale
        r_tail *= scale
        tight      = info['type'] in self._TIGHT_JOINTS
        gi         = group.index

        modified: list = []

        for vert in self.mesh.vertices:
            cur = next((g.weight for g in vert.groups if g.group == gi), 0.0)
            if cur < 0.001:
                continue
            # DOMINANCE GATE: never strip weight from a vertex this bone OWNS (a too-
            # tight radius at a wide cone end would otherwise delete real weight).
            max_w = max((g.weight for g in vert.groups), default=0.0)
            if cur >= max_w - 1e-6:
                continue
            t, dist = self._seg_proj_perp(vert.co, head, tail)
            rad     = r_head + (r_tail - r_head) * t      # tapered per-vertex radius
            if dist <= rad:
                continue
            feather = rad * (0.4 if tight else 0.6)
            excess  = dist - rad
            if excess >= feather:
                new_w = 0.0
            else:
                fade  = excess / feather
                if tight:
                    fade = fade ** 0.7
                new_w = cur * (1.0 - fade)

            if new_w < 0.0001:
                self.access.remove_vertex_weight(
                    self.mesh_obj, vert.index, info['bone_name'])
            else:
                self.access.set_vertex_weight(
                    self.mesh_obj, vert.index, info['bone_name'], new_w)
            modified.append(vert.index)

        # Normalize each touched vertex so its remaining bone weights
        # absorb the removed influence proportionally — no new bones are
        # introduced, only existing influences scale up.
        for vi in modified:
            self.access.normalize_vertex(self.mesh_obj, vi)

        return True


class RIGIFYJOINT_OT_analyze(bpy.types.Operator):
    """Analyze Rigify DEF- joint weight bleeding on the active mesh (results in console)"""
    bl_idname  = "autorig.joint_analyze"
    bl_label   = "Analyze Joint Bleeding"
    bl_options = {'REGISTER'}

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh object")
            return {'CANCELLED'}
        rig = next((m.object for m in obj.modifiers
                    if m.type == 'ARMATURE' and m.object), None)
        if not rig:
            self.report({'ERROR'}, "Mesh has no Armature modifier")
            return {'CANCELLED'}

        cleaner = RigifyJointCleaner(obj, rig)
        report  = cleaner.analyze_problem_joints()

        fix_count   = sum(1 for e in report if e['status'] == 'FIX')
        check_count = sum(1 for e in report if e['status'] == 'CHECK')

        print("\n" + "=" * 68)
        print("RIGIFY JOINT BLEEDING ANALYSIS  (DEF- bones, auto-radius)")
        print("=" * 68)
        for entry in sorted(report, key=lambda x: x['bleed_pct'], reverse=True):
            tag = '!!' if entry['status'] == 'FIX' else '? ' if entry['status'] == 'CHECK' else '  '
            src = entry.get('radius_src', '?')
            print(f"  {tag} {entry['bone']:32s}  "
                  f"bleed={entry['bleed_pct']:5.1f}%  "
                  f"r={entry['radius']:.3f}({src})  [{entry['type']}]")
        print(f"\n  {fix_count} need fixing, {check_count} to check, "
              f"{len(report) - fix_count - check_count} OK")
        print("=" * 68 + "\n")

        self.report({'INFO'},
                    f"{fix_count} need fixing, {check_count} to check — see console")
        return {'FINISHED'}


class RIGIFYJOINT_OT_fix_bleeding(bpy.types.Operator):
    """Fix the cleanable joints most prone to bleeding: upper arms, thighs, shoulders, hands"""
    bl_idname  = "autorig.joint_fix_bleeding"
    bl_label   = "Fix Bleeding Joints"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (context.active_object is not None
                and context.active_object.type == 'MESH'
                and context.mode in {'OBJECT', 'PAINT_WEIGHT'})

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh object")
            return {'CANCELLED'}
        rig = next((m.object for m in obj.modifiers
                    if m.type == 'ARMATURE' and m.object), None)
        if not rig:
            self.report({'ERROR'}, "Mesh has no Armature modifier")
            return {'CANCELLED'}

        cleaner = RigifyJointCleaner(obj, rig)
        report  = cleaner.analyze_problem_joints()

        problem_bones = [e['bone'] for e in report if e['status'] in ('FIX', 'CHECK')]

        if not problem_bones:
            self.report({'INFO'}, "No bleeding joints detected")
            return {'FINISHED'}

        results = cleaner.clean_all_joints(mode='AGGRESSIVE', level='MODERATE',
                                            problem_bones=problem_bones)
        cleaned = sum(1 for r in results if r['success'])
        # Count against bones actually PROCESSED (results), not the flagged list — some
        # flagged bones may be excluded by _NEVER_CLEAN, which made the old count wrong.
        self.report({'INFO'},
                    f"Fixed {cleaned}/{len(results)} bleeding joints (moderate)")
        return {'FINISHED'}


class RIGIFYJOINT_OT_clean(bpy.types.Operator):
    """Clean weight bleeding on all DEF- joints at the chosen strength level"""
    bl_idname  = "autorig.joint_clean"
    bl_label   = "Clean Joint Bleeding"
    bl_options = {'REGISTER', 'UNDO'}

    mode: bpy.props.EnumProperty(
        name="Strength",
        items=[
            ('LIGHT',      'Light',
             'Gentle clean — 10% larger safe zone, ~10% less weight removed from bleed areas'),
            ('MODERATE',   'Moderate',
             'Balanced clean — standard auto-radius; fingers ~10% bleed reduction, '
             'forearm/shin ~5%'),
            ('AGGRESSIVE', 'Aggressive',
             'Strong clean — 10% smaller safe zone, ~10-15% more weight removed'),
        ],
        default='MODERATE',
    )

    @classmethod
    def poll(cls, context):
        return (context.active_object is not None
                and context.active_object.type == 'MESH'
                and context.mode in {'OBJECT', 'PAINT_WEIGHT'})

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh object")
            return {'CANCELLED'}
        rig = next((m.object for m in obj.modifiers
                    if m.type == 'ARMATURE' and m.object), None)
        if not rig:
            self.report({'ERROR'}, "Mesh has no Armature modifier")
            return {'CANCELLED'}

        cleaner = RigifyJointCleaner(obj, rig)
        results = cleaner.clean_all_joints(mode='AGGRESSIVE', level=self.mode)
        ok      = sum(1 for r in results if r['success'])
        self.report({'INFO'}, f"Cleaned {ok}/{len(results)} joints ({self.mode.lower()})")
        return {'FINISHED'}


class AUTORIG_OT_DEFBoneReport(bpy.types.Operator):
    """Cross-bone weight sharing analysis for L-side DEF bones and spine.
    Shows how each bone shares weight with neighbours and which bones it bleeds into."""
    bl_idname  = "autorig.def_bone_report"
    bl_label   = "DEF Bone Influence Report"
    bl_options = {'REGISTER'}

    # Ordered list of focus bones — L side + spine only
    _SECTIONS = [
        ("SPINE", [
            'DEF-spine', 'DEF-spine.001', 'DEF-spine.002', 'DEF-spine.003',
            'DEF-spine.004', 'DEF-spine.005', 'DEF-spine.006',
        ]),
        ("ARM.L", [
            'DEF-shoulder.L',
            'DEF-upper_arm.L', 'DEF-upper_arm.L.001',
            'DEF-forearm.L',   'DEF-forearm.L.001',
            'DEF-hand.L',
        ]),
        ("LEG.L", [
            'DEF-thigh.L', 'DEF-thigh.L.001',
            'DEF-shin.L',  'DEF-shin.L.001',
            'DEF-foot.L',  'DEF-toe.L',
        ]),
        ("PALM.L", [
            'DEF-palm.01.L', 'DEF-palm.02.L',
            'DEF-palm.03.L', 'DEF-palm.04.L',
        ]),
        ("FINGERS.L", [
            'DEF-thumb.01.L',   'DEF-thumb.02.L',   'DEF-thumb.03.L',
            'DEF-f_index.01.L', 'DEF-f_index.02.L', 'DEF-f_index.03.L',
            'DEF-f_middle.01.L','DEF-f_middle.02.L','DEF-f_middle.03.L',
            'DEF-f_ring.01.L',  'DEF-f_ring.02.L',  'DEF-f_ring.03.L',
            'DEF-f_pinky.01.L', 'DEF-f_pinky.02.L', 'DEF-f_pinky.03.L',
        ]),
    ]

    # Expected immediate chain neighbours — sharing with these is normal at joints
    _NEIGHBORS = {
        'DEF-spine':            {'DEF-spine.001', 'DEF-thigh.L', 'DEF-thigh.R'},
        'DEF-spine.001':        {'DEF-spine', 'DEF-spine.002'},
        'DEF-spine.002':        {'DEF-spine.001', 'DEF-spine.003'},
        'DEF-spine.003':        {'DEF-spine.002', 'DEF-spine.004',
                                  'DEF-shoulder.L', 'DEF-shoulder.R'},
        'DEF-spine.004':        {'DEF-spine.003', 'DEF-spine.005'},
        'DEF-spine.005':        {'DEF-spine.004', 'DEF-spine.006'},
        'DEF-spine.006':        {'DEF-spine.005'},
        'DEF-shoulder.L':       {'DEF-upper_arm.L', 'DEF-spine.003', 'DEF-spine.004'},
        'DEF-upper_arm.L':      {'DEF-shoulder.L', 'DEF-upper_arm.L.001'},
        'DEF-upper_arm.L.001':  {'DEF-upper_arm.L', 'DEF-forearm.L'},
        'DEF-forearm.L':        {'DEF-upper_arm.L.001', 'DEF-forearm.L.001'},
        'DEF-forearm.L.001':    {'DEF-forearm.L', 'DEF-hand.L'},
        'DEF-hand.L':           {'DEF-forearm.L.001', 'DEF-palm.01.L', 'DEF-palm.02.L',
                                  'DEF-palm.03.L', 'DEF-palm.04.L', 'DEF-thumb.01.L'},
        'DEF-thigh.L':          {'DEF-spine', 'DEF-spine.001', 'DEF-thigh.L.001'},
        'DEF-thigh.L.001':      {'DEF-thigh.L', 'DEF-shin.L'},
        'DEF-shin.L':           {'DEF-thigh.L.001', 'DEF-shin.L.001'},
        'DEF-shin.L.001':       {'DEF-shin.L', 'DEF-foot.L'},
        'DEF-foot.L':           {'DEF-shin.L.001', 'DEF-toe.L'},
        'DEF-toe.L':            {'DEF-foot.L'},
        'DEF-palm.01.L':        {'DEF-hand.L', 'DEF-f_index.01.L',
                                  'DEF-palm.02.L'},
        'DEF-palm.02.L':        {'DEF-hand.L', 'DEF-f_middle.01.L',
                                  'DEF-palm.01.L', 'DEF-palm.03.L'},
        'DEF-palm.03.L':        {'DEF-hand.L', 'DEF-f_ring.01.L',
                                  'DEF-palm.02.L', 'DEF-palm.04.L'},
        'DEF-palm.04.L':        {'DEF-hand.L', 'DEF-f_pinky.01.L', 'DEF-palm.03.L'},
        'DEF-thumb.01.L':       {'DEF-hand.L', 'DEF-thumb.02.L'},
        'DEF-thumb.02.L':       {'DEF-thumb.01.L', 'DEF-thumb.03.L'},
        'DEF-thumb.03.L':       {'DEF-thumb.02.L'},
        'DEF-f_index.01.L':     {'DEF-palm.01.L', 'DEF-f_index.02.L'},
        'DEF-f_index.02.L':     {'DEF-f_index.01.L', 'DEF-f_index.03.L'},
        'DEF-f_index.03.L':     {'DEF-f_index.02.L'},
        'DEF-f_middle.01.L':    {'DEF-palm.02.L', 'DEF-f_middle.02.L'},
        'DEF-f_middle.02.L':    {'DEF-f_middle.01.L', 'DEF-f_middle.03.L'},
        'DEF-f_middle.03.L':    {'DEF-f_middle.02.L'},
        'DEF-f_ring.01.L':      {'DEF-palm.03.L', 'DEF-f_ring.02.L'},
        'DEF-f_ring.02.L':      {'DEF-f_ring.01.L', 'DEF-f_ring.03.L'},
        'DEF-f_ring.03.L':      {'DEF-f_ring.02.L'},
        'DEF-f_pinky.01.L':     {'DEF-palm.04.L', 'DEF-f_pinky.02.L'},
        'DEF-f_pinky.02.L':     {'DEF-f_pinky.01.L', 'DEF-f_pinky.03.L'},
        'DEF-f_pinky.03.L':     {'DEF-f_pinky.02.L'},
    }

    # Class-level cache so draw() can read what invoke() computed
    _cache: dict = {}

    @classmethod
    def poll(cls, context):
        return (context.active_object is not None
                and context.active_object.type == 'MESH'
                and context.mode in {'OBJECT', 'PAINT_WEIGHT'})

    def _build_report(self, obj):
        """Build cross-bone sharing matrix for all focus bones."""
        mesh        = obj.data
        n_verts     = len(mesh.vertices)
        vg_names    = [g.name for g in obj.vertex_groups]
        n_groups    = len(vg_names)
        name_to_idx = {n: i for i, n in enumerate(vg_names)}

        # Full weight matrix
        W = np.zeros((n_verts, n_groups), dtype=np.float32)
        for v in mesh.vertices:
            for g in v.groups:
                if g.group < n_groups:
                    W[v.index, g.group] = g.weight

        cache = {}
        all_focus = [b for _, bones in self._SECTIONS for b in bones]
        for bone in all_focus:
            gi = name_to_idx.get(bone)
            if gi is None:
                continue
            my_col  = W[:, gi]
            my_mask = my_col > 0.001
            n_mine  = int(my_mask.sum())
            if n_mine == 0:
                continue

            neighbors = self._NEIGHBORS.get(bone, set())
            sharing   = []
            for other_gi, other_name in enumerate(vg_names):
                if other_gi == gi:
                    continue
                other_col    = W[:, other_gi]
                overlap_mask = my_mask & (other_col > 0.001)
                n_overlap    = int(overlap_mask.sum())
                if n_overlap == 0:
                    continue
                sharing.append({
                    'name':      other_name,
                    'count':     n_overlap,
                    'pct':       n_overlap / n_mine * 100.0,
                    'avg_other': float(other_col[overlap_mask].mean()),
                    'avg_mine':  float(my_col[overlap_mask].mean()),
                    'bleed':     other_name not in neighbors,
                })
            sharing.sort(key=lambda x: x['count'], reverse=True)

            cache[bone] = {
                'total':   n_mine,
                'pct_mesh': n_mine / n_verts * 100.0,
                'avg_w':   float(my_col[my_mask].mean()),
                'sharing': sharing,
            }
        return cache

    @staticmethod
    def _print_report(cache, sections, mesh_name):
        W = 72
        print("\n" + "=" * W)
        print(f"  WEIGHT SHARING REPORT — L side + Spine   [{mesh_name}]")
        print("=" * W)
        for sec_name, bones in sections:
            print(f"\n  ── {sec_name} {'─' * (W - 6 - len(sec_name))}")
            for bone in bones:
                d = cache.get(bone)
                if d is None:
                    print(f"    {bone}  (not found / no influence)")
                    continue
                print(f"\n    {bone:<35}  {d['total']:>5}v  "
                      f"{d['pct_mesh']:>5.1f}%mesh  avg={d['avg_w']:.3f}")
                if not d['sharing']:
                    print("      (no overlap with other bones)")
                    continue
                for s in d['sharing'][:10]:
                    tag = "share" if not s['bleed'] else "BLEED"
                    marker = "  " if not s['bleed'] else "!!"
                    print(f"    {marker} {tag}  {s['name']:<35} "
                          f"{s['count']:>5}v ({s['pct']:>4.0f}%)  "
                          f"peer_avg={s['avg_other']:.3f}")
        print("\n" + "=" * W + "\n")

    def invoke(self, context, event):
        obj = context.active_object
        cache = self._build_report(obj)
        AUTORIG_OT_DEFBoneReport._cache = cache
        self._print_report(cache, self._SECTIONS, obj.data.name)
        return context.window_manager.invoke_popup(self, width=520)

    def draw(self, context):
        layout = self.layout
        cache  = AUTORIG_OT_DEFBoneReport._cache
        layout.label(text="Weight Sharing — L side + Spine  (full detail in console)",
                     icon='BONE_DATA')

        for sec_name, bones in self._SECTIONS:
            box = layout.box()
            row = box.row()
            row.label(text=sec_name, icon='GROUP_BONE')
            col = box.column(align=True)
            col.scale_y = 0.78

            for bone in bones:
                d = cache.get(bone)
                short = bone.replace('DEF-', '')
                if d is None:
                    col.label(text=f"  {short}  —  not found")
                    continue

                # Bone header
                col.label(text=f"  {short:<28}  {d['total']:>4}v  "
                               f"{d['pct_mesh']:>4.1f}%  avg {d['avg_w']:.2f}")

                # Show up to 6 sharing partners
                shown_bleed = 0
                for s in d['sharing'][:8]:
                    sname = s['name'].replace('DEF-', '')
                    if s['bleed']:
                        if shown_bleed >= 4:
                            continue
                        shown_bleed += 1
                        col.label(text=f"    !! BLEED  {sname:<26} "
                                       f"{s['count']:>4}v ({s['pct']:>3.0f}%)"
                                       f"  {s['avg_other']:.3f}")
                    else:
                        col.label(text=f"       share  {sname:<26} "
                                       f"{s['count']:>4}v ({s['pct']:>3.0f}%)"
                                       f"  {s['avg_other']:.3f}")

    def execute(self, context):
        return {'FINISHED'}


class AUTORIG_OT_SmoothJointWeights(bpy.types.Operator):
    """Smooth vertex weights on joint areas. Fingers and forearms are left unchanged."""
    bl_idname  = "autorig.smooth_joint_weights"
    bl_label   = "Apply Smooth"
    bl_options = {'REGISTER', 'UNDO'}

    _SKIP_KW = frozenset([
        'thumb', 'f_index', 'f_middle', 'f_ring', 'f_pinky',
        'finger', 'digit', 'palm',
        'forearm', 'lower_arm', 'lowerarm',
        'jaw', 'chin', 'cheek', 'brow', 'lid', 'nose', 'lip',
        'forehead', 'temple', 'ear', 'tongue', 'teeth', 'eye',
    ])

    @classmethod
    def poll(cls, context):
        return (context.active_object is not None
                and context.active_object.type == 'MESH'
                and context.mode in {'OBJECT', 'PAINT_WEIGHT'})

    @classmethod
    def _should_skip(cls, group_name):
        clean = group_name
        for p in ('DEF-', 'ORG-', 'MCH-', 'WGT-'):
            if clean.startswith(p):
                clean = clean[len(p):]
                break
        clean = clean.lower()
        return any(kw in clean for kw in cls._SKIP_KW)

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh object")
            return {'CANCELLED'}
        if not obj.vertex_groups:
            self.report({'ERROR'}, "Mesh has no vertex groups")
            return {'CANCELLED'}

        sp      = context.scene.autorig_skin
        factor  = float(sp.joint_smooth_factor)
        iters   = int(sp.joint_smooth_iters)
        mesh    = obj.data
        n_verts = len(mesh.vertices)

        group_names = [g.name for g in obj.vertex_groups]
        n_groups    = len(group_names)

        # Build weight matrix
        weights = np.zeros((n_verts, n_groups), dtype=np.float32)
        for v in mesh.vertices:
            for g in v.groups:
                if g.group < n_groups:
                    weights[v.index, g.group] = g.weight

        # Save excluded columns (fingers + forearms) before smoothing
        skip_cols = [gi for gi, gn in enumerate(group_names) if self._should_skip(gn)]
        skip_orig = {gi: weights[:, gi].copy() for gi in skip_cols}

        weights_before = weights.copy()

        # Build directed edge graph for laplacian smooth
        edges = np.array([(e.vertices[0], e.vertices[1])
                          for e in mesh.edges], dtype=np.int32)
        if len(edges) == 0:
            self.report({'ERROR'}, "Mesh has no edges")
            return {'CANCELLED'}
        src = np.concatenate([edges[:, 0], edges[:, 1]])
        dst = np.concatenate([edges[:, 1], edges[:, 0]])
        degree   = np.zeros(n_verts, dtype=np.float32)
        np.add.at(degree, dst, 1.0)
        deg_safe = np.where(degree > 0, degree, 1.0)

        for _ in range(iters):
            nbr_sum = np.zeros_like(weights)
            np.add.at(nbr_sum, dst, weights[src])
            nbr_avg = nbr_sum / deg_safe[:, np.newaxis]
            weights  = weights * (1.0 - factor) + nbr_avg * factor

        # Restore finger and forearm columns unchanged
        for gi, col in skip_orig.items():
            weights[:, gi] = col

        # Normalize
        totals = weights.sum(axis=1)
        valid  = totals > 0.001
        if valid.any():
            weights[valid] = weights[valid] / totals[valid, np.newaxis]

        # Write back — diff only, preserving Weight Paint active bone link
        _write_weights_back(obj, group_names, weights_before, weights)

        self.report({'INFO'},
            f"Smoothed {iters} pass{'es' if iters > 1 else ''} "
            f"(factor={factor:.2f}), {len(skip_cols)} group(s) skipped")
        return {'FINISHED'}


class AUTORIG_OT_SmoothTwistWeights(bpy.types.Operator):
    """Smooth the twist bone weight gradient along each limb segment.
    For each DEF-bone / DEF-bone.001 pair, redistributes weights as a linear
    gradient along the bone axis (smooth twist weight)."""
    bl_idname  = "autorig.smooth_twist_weights"
    bl_label   = "Smooth Twist Weights"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (context.active_object is not None
                and context.active_object.type == 'MESH'
                and context.mode in {'OBJECT', 'PAINT_WEIGHT'})

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh object")
            return {'CANCELLED'}
        rig = next((m.object for m in obj.modifiers
                    if m.type == 'ARMATURE' and m.object), None)
        if not rig:
            self.report({'ERROR'}, "Mesh has no Armature modifier")
            return {'CANCELLED'}
        import re as _re
        mesh        = obj.data
        n_verts     = len(mesh.vertices)
        group_names = [g.name for g in obj.vertex_groups]
        n_groups    = len(group_names)

        weights = np.zeros((n_verts, n_groups), dtype=np.float32)
        for v in mesh.vertices:
            for g in v.groups:
                if g.group < n_groups:
                    weights[v.index, g.group] = g.weight

        weights_before = weights.copy()

        m_inv  = obj.matrix_world.inverted()
        arm_mw = rig.matrix_world
        done   = 0

        for gi, name in enumerate(group_names):
            if _re.search(r'\.\d+$', name):
                continue
            twist_name = name + '.001'
            if twist_name not in group_names:
                continue
            ti = group_names.index(twist_name)
            if name not in rig.data.bones or twist_name not in rig.data.bones:
                continue

            bone = rig.data.bones[name]
            head_l = m_inv @ (arm_mw @ bone.head_local)
            tail_l = m_inv @ (arm_mw @ bone.tail_local)
            axis   = tail_l - head_l
            blen   = axis.length
            if blen < 1e-4:
                continue
            axis_n = axis / blen

            # For each vertex that has either main or twist weight, compute
            # its position along the bone axis as t in [0, 1].
            # Main bone gets weight (1-t), twist gets t — linear gradient.
            has_either = (weights[:, gi] > 0.001) | (weights[:, ti] > 0.001)
            if not has_either.any():
                continue

            verts_co  = np.array([mesh.vertices[i].co for i in range(n_verts)],
                                  dtype=np.float32)
            head_np   = np.array(head_l, dtype=np.float32)
            axis_np   = np.array(axis_n, dtype=np.float32)
            to_v      = verts_co - head_np
            t         = np.clip((to_v * axis_np).sum(axis=1) / blen, 0.0, 1.0)

            vi_arr = np.where(has_either)[0]
            total  = weights[vi_arr, gi] + weights[vi_arr, ti]
            # Gradient: main = (1-t) share, twist = t share of combined weight
            weights[vi_arr, gi] = total * (1.0 - t[vi_arr])
            weights[vi_arr, ti] = total * t[vi_arr]
            done += 1

        # Normalize
        totals = weights.sum(axis=1)
        valid  = totals > 0.001
        if valid.any():
            weights[valid] = weights[valid] / totals[valid, np.newaxis]

        _write_weights_back(obj, group_names, weights_before, weights)

        self.report({'INFO'}, f"Twist gradient applied to {done} bone pair(s)")
        return {'FINISHED'}
