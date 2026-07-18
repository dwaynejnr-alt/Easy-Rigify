# retarget.py — Apply an external animation (Mixamo-style FBX, mocap clips,
# other skeletons) onto the generated Rigify rig. See dev/RETARGET_DESIGN.md.
#
# Core idea: per frame, measure how far each SOURCE bone rotated from its own
# rest pose in WORLD space, and apply that same world-space delta to the target
# control's rest orientation:
#
#     R_delta = R_src_pose_world @ R_src_rest_world^-1
#     R_tgt   = R_delta @ R_tgt_rest_world
#
# World deltas make the retarget immune to the two classic killers: mismatched
# rest poses (T-pose clip onto an A-pose character) and mismatched bone rolls.
# Only the hips copy translation — as a delta from rest, scaled by the
# hip-height ratio, added to the TARGET's rest position (bone lengths differ,
# so absolute or per-bone translation would dislocate joints).
#
# Keys land on Rigify's FK CONTROLS (+ torso), so the result is a normal,
# editable action; the limb IK/FK switches are keyed to FK so it is visible
# immediately. Verified exact through Rigify's constraint stack by
# dev/spike_retarget.py (0.0000 deg / 0.000000 m wrist error).
import bpy
import math
from mathutils import Matrix, Vector
from .constants import dbg


# ── Source-name presets ──────────────────────────────────────────────────────
# Mixamo skeleton -> Rigify control. Source names may arrive as
# "mixamorig:Hips", "mixamorig1:Hips" or plain "Hips" — the prefix before ':'
# is stripped when matching. Targets are candidate tuples: the first name that
# exists on the rig wins (covers small naming drift between Rigify versions).
def _sided(base_map):
    """Expand {'Arm': ('upper_arm_fk',)} into Left/Right -> .L/.R entries."""
    out = {}
    for src, tgts in base_map.items():
        for side, suf in (("Left", ".L"), ("Right", ".R")):
            out[side + src] = tuple(t + suf for t in tgts)
    return out


_MIXAMO_CENTER = {
    "Hips":   ("torso",),
    "Spine":  ("spine_fk.001",),
    "Spine1": ("spine_fk.002",),
    "Spine2": ("spine_fk.003", "chest"),
    "Neck":   ("neck",),
    "Head":   ("head",),
}
_MIXAMO_SIDED = _sided({
    "Shoulder": ("shoulder",),
    "Arm":      ("upper_arm_fk",),
    "ForeArm":  ("forearm_fk",),
    "Hand":     ("hand_fk",),
    "UpLeg":    ("thigh_fk",),
    "Leg":      ("shin_fk",),
    "Foot":     ("foot_fk",),
    "ToeBase":  ("toe_fk", "toe"),
    # fingers: HandThumb1 -> thumb.01, HandIndex2 -> f_index.02, ...
    **{f"Hand{m_f}{i}": (f"{r_f}.0{i}",)
       for m_f, r_f in (("Thumb", "thumb"), ("Index", "f_index"),
                        ("Middle", "f_middle"), ("Ring", "f_ring"),
                        ("Pinky", "f_pinky"))
       for i in (1, 2, 3)},
})
_MIXAMO = {**_MIXAMO_CENTER, **_MIXAMO_SIDED}

# The one pair that copies translation.
_HIPS_SOURCES = {"Hips"}


def _strip_prefix(name):
    """'mixamorig:Hips' / 'mixamorig1:Hips' -> 'Hips'."""
    return name.rsplit(":", 1)[-1]


# ── Fuzzy fallback for unknown skeletons ─────────────────────────────────────
# Normalized-substring synonym match for the main body chain. Order matters:
# more specific tokens first ('forearm' would otherwise match 'arm', 'upleg'
# would match 'leg'). Fingers are preset-only — fuzzy finger matching guesses
# wrong more often than it helps.
_FUZZY_RULES = [
    # (synonyms, target base, sided, use_location)
    (("hips", "pelvis"),                    ("torso",),                False, True),
    (("neck",),                             ("neck",),                 False, False),
    (("head",),                             ("head",),                 False, False),
    (("shoulder", "clavicle", "collar"),    ("shoulder",),             True,  False),
    (("forearm", "lowerarm", "elbow"),      ("forearm_fk",),           True,  False),
    (("upperarm", "uparm", "arm"),          ("upper_arm_fk",),         True,  False),
    (("wrist", "hand"),                     ("hand_fk",),              True,  False),
    (("upleg", "thigh", "upperleg"),        ("thigh_fk",),             True,  False),
    (("lowerleg", "calf", "shin", "knee", "leg"), ("shin_fk",),        True,  False),
    (("foot", "ankle"),                     ("foot_fk",),              True,  False),
    (("toe",),                              ("toe_fk", "toe"),         True,  False),
]
_FINGER_TOKENS = ("thumb", "index", "middle", "ring", "pinky", "finger")


def _norm(name):
    return _strip_prefix(name).lower().replace("_", "").replace(".", "").replace("-", "").replace(" ", "")


def _side_of(name):
    """'.L' / '.R' / None from common side conventions."""
    n = _strip_prefix(name).lower()
    if n.startswith("left") or n.endswith((".l", "_l", "-l")) or n.endswith("left"):
        return ".L"
    if n.startswith("right") or n.endswith((".r", "_r", "-r")) or n.endswith("right"):
        return ".R"
    return None


def _first_existing(rig, candidates):
    for c in candidates:
        if c in rig.pose.bones:
            return c
    return None


def build_mapping(src, rig):
    """[(src_bone, tgt_control, use_location)], parents-first. Tries the Mixamo
    preset, falls back to fuzzy synonyms when the preset barely matches."""
    mapping = []
    used_tgt = set()

    # preset pass
    for b in src.data.bones:
        core = _strip_prefix(b.name)
        tgts = _MIXAMO.get(core)
        if not tgts:
            continue
        tgt = _first_existing(rig, tgts)
        if tgt and tgt not in used_tgt:
            mapping.append((b.name, tgt, core in _HIPS_SOURCES))
            used_tgt.add(tgt)

    if len(mapping) < 4:                      # not a Mixamo-style skeleton
        mapping, used_tgt = [], set()
        for b in src.data.bones:
            n = _norm(b.name)
            if any(t in n for t in _FINGER_TOKENS):
                continue                      # fingers are preset-only
            side = _side_of(b.name)
            for syns, tgt_bases, sided, use_loc in _FUZZY_RULES:
                if not any(s in n for s in syns):
                    continue
                if sided and side is None:
                    break
                cands = tuple(t + side for t in tgt_bases) if sided else tgt_bases
                tgt = _first_existing(rig, cands)
                if tgt and tgt not in used_tgt:
                    mapping.append((b.name, tgt, use_loc))
                    used_tgt.add(tgt)
                break                          # first matching rule only

    # parents-first: a control must be keyed after every ancestor control, so
    # sort by bone depth in the target hierarchy
    def depth(tname):
        d, b = 0, rig.data.bones[tname]
        while b.parent:
            d, b = d + 1, b.parent
        return d
    mapping.sort(key=lambda m: depth(m[1]))
    return mapping


def _is_generated_rigify(obj):
    return (obj and obj.type == 'ARMATURE'
            and "torso" in obj.pose.bones
            and any(b.name.startswith("DEF-") for b in obj.data.bones))


def find_target_rig(context):
    act = context.active_object
    if _is_generated_rigify(act):
        return act
    for o in context.scene.objects:
        if _is_generated_rigify(o):
            return o
    return None


# ── The bake ─────────────────────────────────────────────────────────────────

def _rest_world(obj, bname):
    return obj.matrix_world @ obj.data.bones[bname].matrix_local


def _pose_world(obj, bname):
    return obj.matrix_world @ obj.pose.bones[bname].matrix


def _rest_dir(obj, bname):
    """World-space rest direction of a bone (head -> tail), or None."""
    b = obj.data.bones[bname]
    v = (obj.matrix_world.to_3x3() @ (b.tail_local - b.head_local))
    return v.normalized() if v.length > 1e-8 else None


def _facing_yaw(obj, left_name, right_name):
    """Character facing as a world yaw angle, from the left->right shoulder
    line at rest (facing = (L - R) x Z, i.e. -Y for a Rigify character)."""
    L = _rest_world(obj, left_name).translation
    R = _rest_world(obj, right_name).translation
    f = (L - R).cross(Vector((0.0, 0.0, 1.0)))
    if f.length < 1e-6:
        return None
    return math.atan2(f.y, f.x)


def _match_heading(d_t, d_s):
    """Rotate d_s about world Z so its horizontal heading equals d_t's.

    Used for feet/toes: their pitch must match the clip (floor contact), but
    the HEADING must stay the character's own. Forcing the clip's foot yaw
    crams any rest heading difference (toe-out stance, etc.) into the ankle,
    and Rigify's shin twist bones render that as the whole shin twisting."""
    ht = Vector((d_t.x, d_t.y, 0.0))
    hs = Vector((d_s.x, d_s.y, 0.0))
    if ht.length < 1e-4 or hs.length < 1e-4:
        return d_s
    yaw = math.atan2(ht.y, ht.x) - math.atan2(hs.y, hs.x)
    return Matrix.Rotation(yaw, 3, 'Z') @ d_s


# Targets whose rest-align keeps the character's heading (see _match_heading).
_HEADING_PRESERVE = ("foot_fk", "toe_fk", "toe")

# IK controls snapped to their FK twin's result each frame. Keying only FK
# leaves the IK controllers parked at rest — if a limb is (or is switched)
# to IK, it stays glued to those parked controllers and the limb winds up
# visibly (shins twisting while the body turns over pinned feet). Baking the
# IK controls alongside makes the clip correct in EITHER mode and lets users
# flip a limb to IK afterward for foot polish with no snapping step.
_IK_SNAP = (
    ("hand_ik.L", "hand_fk.L"), ("hand_ik.R", "hand_fk.R"),
    ("foot_ik.L", "foot_fk.L"), ("foot_ik.R", "foot_fk.R"),
)


def _facing_correction(src, rig, mapping):
    """Yaw rotation carrying the source character's facing onto the target's.

    World deltas are direction-dependent: a clip on a skeleton that faces the
    other way would swing arms toward the character's BACK. The facing of each
    rig is estimated from its left/right upper-arm (or thigh) rest positions;
    if either side is unmapped, no correction is applied."""
    src_of = {t: s for s, t, _ in mapping}
    for l_tgt, r_tgt in (("upper_arm_fk.L", "upper_arm_fk.R"),
                         ("thigh_fk.L", "thigh_fk.R")):
        if l_tgt in src_of and r_tgt in src_of:
            yaw_s = _facing_yaw(src, src_of[l_tgt], src_of[r_tgt])
            yaw_t = _facing_yaw(rig, l_tgt, r_tgt)
            if yaw_s is not None and yaw_t is not None:
                return Matrix.Rotation(yaw_t - yaw_s, 3, 'Z')
    return Matrix.Identity(3)


def _key_rotation(pb, frame):
    if pb.rotation_mode == 'QUATERNION':
        pb.keyframe_insert("rotation_quaternion", frame=frame)
    elif pb.rotation_mode == 'AXIS_ANGLE':
        pb.keyframe_insert("rotation_axis_angle", frame=frame)
    else:
        pb.keyframe_insert("rotation_euler", frame=frame)


def run_retarget(context, src, rig, mapping, in_place=False, align_rests=True):
    """Bake the source armature's active action onto the rig's controls as a
    NEW action (the rig's previous action is preserved as a datablock).

    align_rests (default): each control's rest is first rotated so its bone
    direction matches the CLIP skeleton's rest direction (facing-corrected),
    then the clip's world deltas are applied. This reproduces the clip's
    actual limb poses — a T-pose walk lands arms at the sides even on an
    A-pose character. Without it, deltas are applied relative to the
    character's own rest, so any rest-pose difference becomes a permanent
    offset (arms behind the back is the classic symptom).
    Returns (ok, message, stats)."""
    src_act = src.animation_data.action if src.animation_data else None
    if src_act is None:
        return False, "Source armature has no action (import the clip first).", {}
    if not mapping:
        return False, "No bones could be mapped between the skeletons.", {}

    f_start, f_end = (int(round(f)) for f in src_act.frame_range)
    prev_frame = context.scene.frame_current

    # hip-height ratio for translation scaling (world Z of the hips pair at rest)
    hips_pair = next(((s, t) for s, t, loc in mapping if loc), None)
    scale_ratio = 1.0
    if hips_pair:
        src_z = _rest_world(src, hips_pair[0]).translation.z
        tgt_z = _rest_world(rig, hips_pair[1]).translation.z
        if src_z > 1e-6:
            scale_ratio = tgt_z / src_z

    # facing correction: source deltas are conjugated into the target's frame
    C = _facing_correction(src, rig, mapping)
    Ci = C.inverted()

    # cache rest matrices once — they are frame-independent. With align_rests,
    # the target rest used as the delta base is pre-rotated so its bone
    # direction equals the (facing-corrected) source bone's rest direction.
    rest_src = {s: _rest_world(src, s).to_3x3() for s, _, _ in mapping}
    rest_tgt = {}
    for s, t, use_loc in mapping:
        R = _rest_world(rig, t).to_3x3()
        # The location carrier (torso <- hips) is an abstract PIVOT: Rigify's
        # torso bone points horizontally by widget convention, not anatomy.
        # Aligning it to the source hips' up-vector pitches the whole pelvis
        # assembly ~90 deg around the pivot head and drops the character.
        # It stays delta-based; only bones that lie along an actual body part
        # (spine, limbs, fingers) are rest-aligned.
        if align_rests and not use_loc:
            d_t = _rest_dir(rig, t)
            d_s = _rest_dir(src, s)
            if d_t is not None and d_s is not None:
                d = C @ d_s
                if t.startswith(_HEADING_PRESERVE):
                    d = _match_heading(d_t, d)
                R = d_t.rotation_difference(d).to_matrix() @ R
        rest_tgt[t] = R
    rest_tgt_loc = {t: _rest_world(rig, t).translation.copy() for _, t, _ in mapping}
    src_hips_rest = (_rest_world(src, hips_pair[0]).translation.copy()
                     if hips_pair else Vector())

    # Height calibration (align mode): matching the clip's leg rest directions
    # straightens legs the character had rigged with a knee bend, so the feet
    # would reach BELOW the floor while the hips stay at the character's rest
    # height. Measure, per mapped leg, where the ankle would land under
    # clip-matched rest directions vs where the character's rest ankle is, and
    # lift every hips key by the difference.
    z_off = 0.0
    if align_rests:
        src_of = {t: s for s, t, _ in mapping}
        offs = []
        for side in (".L", ".R"):
            thigh, shin, foot = ("thigh_fk" + side, "shin_fk" + side,
                                 "foot_fk" + side)
            if not all(b in src_of for b in (thigh, shin, foot)):
                continue
            p = _rest_world(rig, thigh).translation.copy()
            dirs = [_rest_dir(src, src_of[b]) for b in (thigh, shin)]
            if any(d is None for d in dirs):
                continue
            for b, d_s in zip((thigh, shin), dirs):
                p += (C @ d_s) * rig.data.bones[b].length
            offs.append(_rest_world(rig, foot).translation.z - p.z)
        if offs:
            z_off = sum(offs) / len(offs)

    # fresh action on the rig; keep whatever it had as a datablock
    if rig.animation_data is None:
        rig.animation_data_create()
    prev_action = rig.animation_data.action
    if prev_action:
        prev_action.use_fake_user = True      # survive save even if unassigned
    new_act = bpy.data.actions.new(src_act.name + "_retarget")
    rig.animation_data.action = new_act

    # limbs must be in FK for the keys to drive them; key the switch so the
    # clip carries it
    n_switch = 0
    for pname in ("upper_arm_parent.L", "upper_arm_parent.R",
                  "thigh_parent.L", "thigh_parent.R"):
        pb = rig.pose.bones.get(pname)
        if pb is not None and "IK_FK" in pb.keys():
            pb["IK_FK"] = 1.0
            pb.keyframe_insert('["IK_FK"]', frame=f_start)
            n_switch += 1

    # depth per target: bones at the same depth are independent, so one
    # view-layer update per depth level per frame is enough (the matrix setter
    # converts through the CURRENT evaluated parent, which must be fresh)
    def depth(tname):
        d, b = 0, rig.data.bones[tname]
        while b.parent:
            d, b = d + 1, b.parent
        return d
    depths = {t: depth(t) for _, t, _ in mapping}
    mapped_tgts = {t for _, t, _ in mapping}

    wm = context.window_manager
    wm.progress_begin(f_start, f_end)
    inv_rig = rig.matrix_world.inverted()
    prev_q = {}
    try:
        for frame in range(f_start, f_end + 1):
            context.scene.frame_set(frame)
            context.view_layer.update()
            wm.progress_update(frame)
            last_depth = None
            for s_name, t_name, use_loc in mapping:
                if last_depth is not None and depths[t_name] != last_depth:
                    context.view_layer.update()
                last_depth = depths[t_name]

                M_src = _pose_world(src, s_name)
                R_delta = M_src.to_3x3() @ rest_src[s_name].inverted()
                R_tgt = C @ R_delta @ Ci @ rest_tgt[t_name]
                pb = rig.pose.bones[t_name]
                if use_loc:
                    t_delta = C @ ((M_src.translation - src_hips_rest)
                                   * scale_ratio)
                    if in_place:
                        t_delta.x = t_delta.y = 0.0
                    loc = rest_tgt_loc[t_name] + t_delta
                    loc.z += z_off
                else:
                    # keep the chain-determined position; key rotation only
                    loc = _pose_world(rig, t_name).translation
                pb.matrix = inv_rig @ Matrix.LocRotScale(
                    loc, R_tgt.to_quaternion(), None)
                # quaternion sign continuity: each frame's quaternion is
                # computed independently, and q / -q are the same rotation but
                # interpolate differently — a sign flip between adjacent keys
                # makes joints visibly spin "the long way" (feet/shins snap
                # while the character turns). Keep each bone on the same cover
                # as its previous frame.
                if pb.rotation_mode == 'QUATERNION':
                    q = pb.rotation_quaternion.copy()
                    pq = prev_q.get(t_name)
                    if pq is not None and pq.dot(q) < 0.0:
                        q.negate()
                        pb.rotation_quaternion = q
                    prev_q[t_name] = q
                if use_loc:
                    pb.keyframe_insert("location", frame=frame)
                _key_rotation(pb, frame)

            # snap IK controllers onto the FK result for this frame
            context.view_layer.update()
            for ik_name, fk_name in _IK_SNAP:
                if fk_name not in mapped_tgts or ik_name not in rig.pose.bones:
                    continue
                pb = rig.pose.bones[ik_name]
                pb.matrix = rig.pose.bones[fk_name].matrix.copy()
                if pb.rotation_mode == 'QUATERNION':
                    q = pb.rotation_quaternion.copy()
                    pq = prev_q.get(ik_name)
                    if pq is not None and pq.dot(q) < 0.0:
                        q.negate()
                        pb.rotation_quaternion = q
                    prev_q[ik_name] = q
                pb.keyframe_insert("location", frame=frame)
                _key_rotation(pb, frame)
    finally:
        wm.progress_end()
        context.scene.frame_set(prev_frame)

    stats = {"frames": f_end - f_start + 1, "bones": len(mapping),
             "action": new_act.name, "fk_switches": n_switch,
             "facing_yaw_deg": round(math.degrees(
                 math.atan2(C[1][0], C[0][0])), 1)}
    dbg(f"[retarget] {src.name} -> {rig.name}: {stats['bones']} controls, "
        f"{stats['frames']} frames, facing {stats['facing_yaw_deg']} deg, "
        f"align={align_rests}, floor z_off={z_off:.4f} "
        f"-> action '{new_act.name}'")
    return True, (f"Retargeted {stats['frames']} frames onto "
                  f"{stats['bones']} controls -> action '{new_act.name}'"), stats


# ── UI ───────────────────────────────────────────────────────────────────────

def _poll_source(self, obj):
    return obj.type == 'ARMATURE' and not _is_generated_rigify(obj)


class AutoRigRetargetProps(bpy.types.PropertyGroup):
    source: bpy.props.PointerProperty(
        name="Source",
        description="Armature carrying the animation to retarget (import the "
                    "FBX/BVH clip first, then pick its armature here)",
        type=bpy.types.Object,
        poll=_poll_source,
    )
    in_place: bpy.props.BoolProperty(
        name="In Place",
        description="Strip the horizontal (XY) root travel from the hips so "
                    "the character animates on the spot (game loops)",
        default=False,
    )
    align_rests: bpy.props.BoolProperty(
        name="Match Clip Pose",
        description="Align each control to the clip skeleton's rest pose "
                    "before applying motion, so the clip's actual limb poses "
                    "are reproduced (a T-pose walk lands arms at the sides "
                    "even on an A-pose character). Turn off to keep offsets "
                    "relative to your character's own rest pose instead",
        default=True,
    )


class AUTORIG_OT_RetargetAnim(bpy.types.Operator):
    """Retarget the source armature's animation onto the generated Rigify rig"""
    bl_idname = "autorig.retarget_anim"
    bl_label = "Retarget Animation"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.autorig_retarget
        src = props.source
        rig = find_target_rig(context)
        if src is None:
            self.report({'ERROR'}, "Pick a source armature first.")
            return {'CANCELLED'}
        if rig is None:
            self.report({'ERROR'}, "No generated Rigify rig in the scene.")
            return {'CANCELLED'}
        if src == rig:
            self.report({'ERROR'}, "Source and target are the same rig.")
            return {'CANCELLED'}

        prev_mode = None
        if context.mode != 'OBJECT':
            prev_mode = context.mode
            bpy.ops.object.mode_set(mode='OBJECT')

        mapping = build_mapping(src, rig)
        ok, msg, _ = run_retarget(context, src, rig, mapping,
                                  in_place=props.in_place,
                                  align_rests=props.align_rests)
        if ok and prev_mode == 'POSE':
            bpy.ops.object.mode_set(mode='POSE')
        self.report({'INFO'} if ok else {'ERROR'}, msg)
        return {'FINISHED'} if ok else {'CANCELLED'}


def draw_retarget_section(layout, context):
    box = layout.box()
    box.label(text="Animation Retarget", icon='ANIM')
    rig = find_target_rig(context)
    if rig is None:
        box.label(text="Generate a rig first", icon='INFO')
        return
    props = context.scene.autorig_retarget
    col = box.column(align=True)
    col.prop(props, "source")
    src = props.source
    if src is None:
        col.label(text="Import a clip (FBX/BVH), then pick its armature",
                  icon='INFO')
        return
    act = src.animation_data.action if src.animation_data else None
    if act is None:
        col.label(text="Source armature has no animation", icon='ERROR')
        return
    f0, f1 = act.frame_range
    n_preset = sum(1 for b in src.data.bones
                   if _strip_prefix(b.name) in _MIXAMO)
    kind = "Mixamo-style" if n_preset >= 4 else "name-matched"
    col.label(text=f"Clip: {act.name}  ({int(f1 - f0) + 1} frames, {kind})",
              icon='ACTION')
    col.prop(props, "align_rests")
    col.prop(props, "in_place")
    col.operator("autorig.retarget_anim", icon='PLAY')
