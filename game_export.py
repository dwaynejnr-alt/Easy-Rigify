# game_export.py — Export a generated Rigify rig + skinned mesh as a clean,
# game-ready skeleton (Unity first; Unreal planned). See dev/GAME_EXPORT_DESIGN.md.
#
# Rigify's rig is a control rig engines can't use. The usable skeleton is its
# DEF- deform bones — but limbs are split into two twist segments
# (DEF-upper_arm.L + .001, etc.) and engines want one bone per limb. This module
# extracts a DEF-only skeleton, MERGES those segments (summing skin weights, which
# is lossless because per-vertex totals stay 1.0), roots it, and exports FBX.
#
# Everything runs on DUPLICATES: the user's working rig and meshes are untouched.
import bpy
from .constants import dbg

# Limb bases whose ".001" twist segment folds into the base bone. Neck is handled
# defensively — some metarigs produce a single neck deform, not two (spike finding).
_SEGMENT_BASES = [
    "DEF-upper_arm.L", "DEF-upper_arm.R",
    "DEF-forearm.L",   "DEF-forearm.R",
    "DEF-thigh.L",     "DEF-thigh.R",
    "DEF-shin.L",      "DEF-shin.R",
    "DEF-neck",
]


def _find_game_rig(context):
    """The generated Rigify rig = an armature carrying DEF- deform bones."""
    act = context.active_object
    if act and act.type == 'ARMATURE' and any(
            b.name.startswith("DEF-") for b in act.data.bones):
        return act
    for o in context.scene.objects:
        if o.type == 'ARMATURE' and any(
                b.name.startswith("DEF-") for b in o.data.bones):
            return o
    return None


def _bound_meshes(context, rig):
    """Meshes deformed by rig via an Armature modifier."""
    out = []
    for o in context.scene.objects:
        if o.type != 'MESH':
            continue
        if any(m.type == 'ARMATURE' and m.object == rig for m in o.modifiers):
            out.append(o)
    return out


def _duplicate(context, obj):
    """Full, independent copy — INCLUDING its data (mesh/armature).

    Uses the data API rather than bpy.ops.object.duplicate() on purpose:
    duplicate() honours the user's Edit > Preferences > Duplicate Data toggles, so
    if the user has 'Mesh' or 'Armature' unchecked there the "duplicate" SHARES
    data with the original — and editing weights/bones on it would corrupt the
    user's real character. obj.data.copy() guarantees independent data every time.
    """
    dup = obj.copy()
    dup.data = obj.data.copy()
    dup.animation_data_clear()
    # Link next to the original so mode/edit ops and FBX selection see it.
    coll = obj.users_collection[0] if obj.users_collection else context.scene.collection
    coll.objects.link(dup)
    return dup


def _freeze_to_rest(context, arm):
    """Strip every pose-bone constraint and clear the pose so the deform bones sit
    exactly at their rest (bind) positions.

    Rigify's DEF- bones are not free bones — they're driven by Copy Transforms /
    Stretch To constraints that target the MCH-/ORG- mechanism bones. Once those
    targets are removed (we keep only deform bones), the constraints evaluate to
    garbage and drag the skeleton off the bind pose, exploding the skinned mesh.
    Removing the constraints leaves each deform bone at its rest position, which is
    the pose the weights were bound against."""
    context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='OBJECT')
    for pb in arm.pose.bones:
        for c in list(pb.constraints):
            pb.constraints.remove(c)
        pb.matrix_basis.identity()          # clear any pose transform
    arm.animation_data_clear()              # drop drivers/actions too


def _deform_parent_map(arm):
    """For every deform bone, its correct ANATOMICAL parent within the deform
    set — computed on the FULL hierarchy, before extraction.

    Within a limb Rigify's DEF bones parent to each other, but ACROSS joints
    they parent into the ORG/MCH mechanism layer (DEF-upper_arm.L's parent is
    ORG-upper_arm.L, under ORG-shoulder.L...). Extraction deletes those, so
    the upper arms, thighs, shoulders and toes fall to the root — invisible at
    rest in Blender, but in-engine the clavicle/spine then doesn't carry the
    arm and skinning looks broken. Walk each deform bone's original ancestors
    and resolve every mechanism bone to its deform counterpart (ORG-shoulder.L
    -> DEF-shoulder.L); the first hit is the anatomical parent."""
    keep = {b.name for b in arm.data.bones if b.use_deform}
    parent_of = {}
    for b in arm.data.bones:
        if b.name not in keep:
            continue
        p = b.parent
        while p is not None:
            core = p.name
            for pref in ("DEF-", "ORG-", "MCH-"):
                if core.startswith(pref):
                    core = core[len(pref):]
                    break
            for cand in (p.name, "DEF-" + core, core):
                if cand in keep and cand != b.name:
                    parent_of[b.name] = cand
                    break
            if b.name in parent_of:
                break
            p = p.parent
    return parent_of


def _reparent_orphans(context, arm, parent_of):
    """Re-attach deform bones orphaned by extraction to their anatomical
    parent (from _deform_parent_map). Returns the number reattached."""
    context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    ebs = arm.data.edit_bones
    n = 0
    for eb in ebs:
        if eb.parent is not None:
            continue
        target = parent_of.get(eb.name)
        if target and target in ebs and target != eb.name:
            eb.use_connect = False
            eb.parent = ebs[target]
            n += 1
    bpy.ops.object.mode_set(mode='OBJECT')
    return n


def _keep_deform_bones(context, arm):
    """Keep only the bones that actually skin the mesh (use_deform=True). This is
    the DEF- bones PLUS Rigify's neck and head, which deform directly and carry no
    DEF- prefix — filtering by name would silently drop the head and neck. The
    deform bones parent to each other, so the hierarchy survives; any left
    parentless is re-rooted later."""
    context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='OBJECT')  # bone.use_deform reads on data bones
    keep = {b.name for b in arm.data.bones if b.use_deform}
    bpy.ops.object.mode_set(mode='EDIT')
    for eb in list(arm.data.edit_bones):
        if eb.name not in keep:
            arm.data.edit_bones.remove(eb)
    bpy.ops.object.mode_set(mode='OBJECT')


def _merge_vgroup(mesh, base, seg):
    """base_weight += seg_weight per vertex, then delete the seg group. Lossless:
    the merged groups partition the influence, so the sum rebuilds it and each
    vertex's total (already 1.0) is unchanged.

    Uses 'ADD' write mode: it accumulates onto the existing weight AND creates
    the membership when the vertex is not yet in the base group. ('REPLACE'
    silently drops non-members — invisible on limb twist merges, where the two
    segments always overlap, but it LOSES weight when folding face bones into
    the head, whose group most face vertices never belonged to.)"""
    vg_seg = mesh.vertex_groups.get(seg)
    if vg_seg is None:
        return
    vg_base = mesh.vertex_groups.get(base) or mesh.vertex_groups.new(name=base)
    seg_idx = vg_seg.index
    for v in mesh.data.vertices:
        for g in v.groups:
            if g.group == seg_idx and g.weight > 0.0:
                vg_base.add([v.index], g.weight, 'ADD')
                break
    mesh.vertex_groups.remove(vg_seg)


def _merge_segments(context, arm, meshes, preserve_twist=False):
    """Fold each ".001" twist segment into its base bone (and matching vgroups).

    preserve_twist: keep the limb .001 segments as separate twist bones
    (renamed to the UE convention later) and restructure the chain the way the
    UE Mannequin has it — the next limb segment parents to the BASE bone and
    the twist bone becomes a leaf child of it (upperarm -> {upperarm_twist_01,
    lowerarm}). Weights are untouched, so deformation quality survives. The
    neck still merges (the spine-positional naming produces a single neck).
    Returns the number of merges performed."""
    merged = 0
    for base in _SEGMENT_BASES:
        seg = base + ".001"
        has_base = arm.data.bones.get(base) is not None
        has_seg = arm.data.bones.get(seg) is not None

        if has_base and has_seg and preserve_twist and base != "DEF-neck":
            context.view_layer.objects.active = arm
            bpy.ops.object.mode_set(mode='EDIT')
            ebs = arm.data.edit_bones
            eb_base, eb_seg = ebs[base], ebs[seg]
            for child in list(eb_seg.children):
                child.use_connect = False
                child.parent = eb_base
            bpy.ops.object.mode_set(mode='OBJECT')
            continue

        if has_base and has_seg:
            # extend base to the segment's tail, reparent the chain, remove seg
            context.view_layer.objects.active = arm
            bpy.ops.object.mode_set(mode='EDIT')
            ebs = arm.data.edit_bones
            eb_base, eb_seg = ebs[base], ebs[seg]
            eb_base.tail = eb_seg.tail.copy()
            for child in list(eb_seg.children):
                child.parent = eb_base
            ebs.remove(eb_seg)
            bpy.ops.object.mode_set(mode='OBJECT')
            for m in meshes:
                _merge_vgroup(m, base, seg)
            merged += 1
        elif has_seg and not has_base:
            # single-bone limb that was named ".001" — rename it to the base
            context.view_layer.objects.active = arm
            bpy.ops.object.mode_set(mode='EDIT')
            arm.data.edit_bones[seg].name = base
            bpy.ops.object.mode_set(mode='OBJECT')
            for m in meshes:
                vg = m.vertex_groups.get(seg)
                if vg:
                    vg.name = base
    return merged


def _face_deform_bones(arm):
    """Names of deform bones belonging to the face rig, identified on the
    FULL hierarchy (before the non-deform bones are stripped): a face deform
    bone's ancestor chain passes through Rigify's face root. Must run before
    _keep_deform_bones — extraction orphans these bones, so hierarchy-based
    identification is impossible afterwards."""
    out = set()
    for b in arm.data.bones:
        if not b.use_deform:
            continue
        p = b.parent
        while p is not None:
            if p.name in ("ORG-face", "face"):
                out.add(b.name)
                break
            p = p.parent
    return out


def _strip_face_bones(context, arm, meshes, face_set):
    """Remove the face rig's deform bones (60+ on a full face metarig) and
    fold each one's weights into the head bone, so the face follows the head
    rigidly. Per-vertex totals stay 1.0 (same argument as the limb merge),
    and separate eye/teeth/tongue meshes end up fully head-weighted. Runs
    after extraction, before renaming (DEF- names). Returns bones stripped."""
    chain = _spine_chain(arm)
    if not chain or not face_set:
        return 0
    head_name = chain[-1]
    doomed = [bn for bn in face_set
              if bn in arm.data.bones and bn != head_name]
    for m in meshes:
        for bn in doomed:
            _merge_vgroup(m, head_name, bn)
    context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    for bn in doomed:
        eb = arm.data.edit_bones.get(bn)
        if eb is not None:
            arm.data.edit_bones.remove(eb)
    bpy.ops.object.mode_set(mode='OBJECT')
    return len(doomed)


def _add_root(context, arm):
    """Add a single 'root' bone at world origin and parent every otherwise-
    parentless bone to it, so the skeleton has one clean root."""
    context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    ebs = arm.data.edit_bones
    root = ebs.new("root")
    root.head = (0.0, 0.0, 0.0)
    root.tail = (0.0, 0.2, 0.0)
    for eb in ebs:
        if eb is not root and eb.parent is None:
            # Clear use_connect FIRST: a connected bone snaps its head to the new
            # parent's tail. Without this, parentless deform bones far from origin
            # (breast, pelvis fans, shoulders) get their head yanked to the root at
            # world origin and stretch into giant bones.
            eb.use_connect = False
            eb.parent = root
    bpy.ops.object.mode_set(mode='OBJECT')


# Rigify DEF limb base -> UE5 Mannequin base (side + segment appended separately).
_UE_LIMB = {
    "upper_arm": "upperarm", "forearm": "lowerarm", "hand": "hand",
    "shoulder": "clavicle", "thigh": "thigh", "shin": "calf",
    "foot": "foot", "toe": "ball",
}
_UE_FINGER = {
    "f_index": "index", "f_middle": "middle", "f_ring": "ring",
    "f_pinky": "pinky", "thumb": "thumb",
}
# Rigify .001 twist segment -> UE Mannequin twist bone (preserve-twist mode).
_UE_TWIST = {
    "upper_arm": "upperarm_twist", "forearm": "lowerarm_twist",
    "thigh": "thigh_twist", "shin": "calf_twist",
}
def _core_name(bname):
    """Rigify deform-bone name without the DEF- prefix."""
    return bname[len("DEF-"):] if bname.startswith("DEF-") else bname


def _spine_chain(arm):
    """The DEF-spine deform bones, ordered base -> tip (spine, .001, .002, ...).

    Rigify has NO DEF-neck / DEF-head bones: the neck and head are deformed by the
    TOP TWO segments of this chain (the head/neck bones themselves are control-only,
    use_deform=False). So the chain runs pelvis -> spine... -> neck -> head, and we
    name it positionally."""
    def idx(bname):
        core = _core_name(bname)
        return 0 if core == "spine" else int(core.rsplit(".", 1)[-1])
    spines = [b.name for b in arm.data.bones if _core_name(b.name) == "spine"
              or _core_name(b.name).startswith("spine.")]
    return sorted(spines, key=idx)


def _spine_target_name(i, n, target):
    """Positional name for the i-th of n spine-chain bones."""
    if i == 0:
        return "pelvis"
    if i == n - 1:
        return "head"
    if i == n - 2:
        return "neck_01" if target == 'UNREAL' else "neck"
    return f"spine_{i:02d}"


def _ue_name(core):
    """Rigify core bone name (DEF- already stripped, spine handled elsewhere) ->
    UE5 Mannequin name, or None to leave the core name as-is."""
    # twist segments carry the side BEFORE the .001 (upper_arm.L.001) — peel
    # the segment suffix first or side detection never fires
    twist = core.endswith(".001")
    if twist:
        core = core[:-4]
    side = None
    if core.endswith(".L"):
        side, core = "_l", core[:-2]
    elif core.endswith(".R"):
        side, core = "_r", core[:-2]
    if side is None:
        return None
    if twist:
        if core in _UE_TWIST:                     # upper_arm.L.001 -> upperarm_twist_01_l
            return f"{_UE_TWIST[core]}_01{side}"
        return None
    for fk, fv in _UE_FINGER.items():            # f_index.01 -> index_01_l
        if core.startswith(fk + "."):
            return f"{fv}_{core[len(fk) + 1:]}{side}"
    if core in _UE_LIMB:                          # upper_arm -> upperarm_l
        return f"{_UE_LIMB[core]}{side}"
    return None


def _rename_for_target(arm, meshes, target):
    """Rename deform bones (and matching vertex groups, kept in lockstep) to the
    target's convention. The spine chain is named positionally (pelvis / spine_0N /
    neck / head) because Rigify deforms neck+head via the top spine segments. Limbs
    and fingers: UE5 Mannequin names for Unreal, stripped core names for Unity.
    Returns {old_name: new_name} so the animation bake can find each clean bone's
    source bone on the original rig."""
    renames = {}

    # Spine chain (also produces the neck + head bones both engines expect).
    chain = _spine_chain(arm)
    n = len(chain)
    for i, bname in enumerate(chain):
        renames[bname] = _spine_target_name(i, n, target)

    # Everything else.
    for b in arm.data.bones:
        if b.name == "root" or b.name in renames:
            continue
        core = _core_name(b.name)
        if target == 'UNREAL':
            ue = _ue_name(core)
            new = ue if ue else core
        else:
            new = core
        if new != b.name:
            renames[b.name] = new

    for old, new in renames.items():
        arm.data.bones[old].name = new
    for m in meshes:
        for old, new in renames.items():
            vg = m.vertex_groups.get(old)
            if vg:
                vg.name = new
    return renames


def _act_fcurves(act):
    """Action fcurves across Blender versions: legacy `Action.fcurves` was
    removed in 5.x (slotted actions) — walk layers/strips/channelbags there."""
    if hasattr(act, "fcurves"):
        return act.fcurves
    for layer in act.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                return bag.fcurves
    return ()


def _actions_for_rig(rig):
    """Every action in the file that animates this rig. Majority match, not
    any-match: a previously BAKED game action also keys a few names the rig
    happens to share (root/head/neck), so requiring most of the action's
    bones to exist on the rig keeps stray bakes and other characters' clips
    out."""
    names = set(rig.pose.bones.keys())
    out = []
    for act in bpy.data.actions:
        if act.get("er_game_bake"):
            continue                       # our own previous export bakes
        bones_in_act = {fc.data_path.split('"')[1]
                        for fc in _act_fcurves(act)
                        if fc.data_path.startswith('pose.bones["')}
        if bones_in_act and len(bones_in_act & names) > len(bones_in_act) * 0.5:
            out.append(act)
    return out


def _bake_animation(context, rig, clean, renames):
    """Bake the original rig's active action onto the clean skeleton.

    The clean skeleton was frozen to rest (its DEF constraints had to go — their
    MCH-/ORG- targets no longer exist on it). The ORIGINAL rig still animates
    normally, so each clean bone gets a temporary world-space Copy Transforms
    constraint targeting its source bone there, and a visual-keying bake converts
    that into plain keyframes. Merged limbs follow their BASE segment — the .001
    twist detail is inherently dropped by merge mode. 'root' follows Rigify's
    root control bone so root motion survives.

    Returns (n_frames, message) — n_frames 0 means nothing was baked."""
    act = rig.animation_data.action if rig.animation_data else None
    if act is None:
        return 0, "no action on the rig"

    # clean bone -> source bone on the original rig
    source_of = {new: old for old, new in renames.items()}
    for pb in clean.pose.bones:
        src = source_of.get(pb.name, pb.name if pb.name in rig.pose.bones else None)
        if pb.name == "root":
            src = "root" if "root" in rig.pose.bones else None
        if src is None:
            continue                      # no source — bone stays at rest
        con = pb.constraints.new('COPY_TRANSFORMS')
        con.target = rig
        con.subtarget = src
        con.target_space = con.owner_space = 'WORLD'

    f_start, f_end = (int(round(f)) for f in act.frame_range)
    prev_frame = context.scene.frame_current

    for o in context.selected_objects:
        o.select_set(False)
    clean.select_set(True)
    context.view_layer.objects.active = clean
    bpy.ops.object.mode_set(mode='POSE')
    # select every pose bone via the operator — Bone.select was removed in
    # Blender 5.0, and the op works on every supported version
    bpy.ops.pose.select_all(action='SELECT')
    bpy.ops.nla.bake(
        frame_start=f_start, frame_end=f_end,
        only_selected=True, visual_keying=True,
        clear_constraints=True,           # drop the temp Copy Transforms
        use_current_action=True, bake_types={'POSE'},
    )
    bpy.ops.object.mode_set(mode='OBJECT')
    context.scene.frame_set(prev_frame)

    if clean.animation_data and clean.animation_data.action:
        clean.animation_data.action.name = act.name + "_game"
        # tag the bake so the all-actions scan never re-exports our own
        # output (on face rigs the stripped core names collide with Rigify's
        # control-bone names, so name-overlap heuristics can't tell them apart)
        clean.animation_data.action["er_game_bake"] = True
    n = f_end - f_start + 1
    return n, f"baked {n} frames"


# FBX settings are the substance of a game export, not a wrapper. Per-target
# starting defaults (validate in-engine): no leaf bones, Y-up FBX (both engines'
# importers convert), deform-only skeleton (already DEF-only). Unity bakes unit
# scale so it shows scale 1; Unreal leaves scale unbaked (its importer applies
# the cm conversion).
_FBX_SETTINGS = {
    'UNITY': dict(
        apply_scale_options='FBX_SCALE_ALL',
        axis_forward='-Z', axis_up='Y',
        primary_bone_axis='Y', secondary_bone_axis='X',
    ),
    'UNREAL': dict(
        apply_scale_options='FBX_SCALE_NONE',
        axis_forward='-Z', axis_up='Y',
        primary_bone_axis='Y', secondary_bone_axis='X',
    ),
}


def build_and_export(context, filepath, target='UNITY', keep_in_scene=False,
                     apply_modifiers=False, add_leaf_bones=False,
                     include_anim=False, anim_simplify=1.0,
                     strip_face=False, all_actions=False,
                     preserve_twist=False):
    """Full pipeline: validate -> clean skeleton -> merge -> root -> rename ->
    [bake animation] -> FBX. Returns (ok, message, stats).

    include_anim: bake the rig's active action onto the clean skeleton and export
    it. anim_simplify: FBX curve simplification (0 = every frame kept, 1 = default
    lossy compression; only used when animation is exported).

    apply_modifiers=False exports the authored mesh cage (game-appropriate); True
    bakes non-armature modifiers like Subsurf into the exported mesh, which can
    multiply the poly count.

    add_leaf_bones: FBX cannot store the length of a childless bone (it infers
    length from the child's position), so bones like breast/pelvis-fan/shoulder
    reimport at a wrong default length in Blender. add_leaf_bones=True writes a
    tiny tip bone at each so lengths round-trip, at the cost of extra '_end' bones
    (unwanted by most engines). Skinning and bone positions are correct either
    way; this only affects displayed bone length on reimport."""
    rig = _find_game_rig(context)
    if rig is None:
        return False, "No generated Rigify rig (DEF- bones) found.", {}
    meshes = _bound_meshes(context, rig)
    if not meshes:
        return False, "No mesh is bound to the generated rig.", {}

    prev_active = context.view_layer.objects.active
    prev_sel = list(context.selected_objects)
    prev_mode = context.mode if context.mode else 'OBJECT'
    if context.mode != 'OBJECT':
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass

    # 1. clean skeleton (duplicate + DEF-only). The face set must be read off
    # the FULL hierarchy — extraction orphans the face deform bones.
    clean = _duplicate(context, rig)
    clean.name = "GAME_SKELETON"
    clean.animation_data_clear()
    face_set = _face_deform_bones(clean) if strip_face else set()
    parent_map = _deform_parent_map(clean)   # read the FULL hierarchy first
    _freeze_to_rest(context, clean)      # remove constraints BEFORE stripping bones
    _keep_deform_bones(context, clean)
    n_reparented = _reparent_orphans(context, clean, parent_map)

    # 2. duplicate meshes, repoint their armature modifier at the clean skeleton
    dup_meshes = []
    for m in meshes:
        dm = _duplicate(context, m)
        for mod in dm.modifiers:
            if mod.type == 'ARMATURE' and mod.object == rig:
                mod.object = clean
        dm.parent = clean
        dup_meshes.append(dm)

    # 2b. optionally drop the face rig (weights fold into the head)
    n_face = 0
    if strip_face:
        n_face = _strip_face_bones(context, clean, dup_meshes, face_set)

    # 3. merge twist segments, 4. root, 5. rename to the target convention
    n_merged = _merge_segments(
        context, clean, dup_meshes,
        preserve_twist=(preserve_twist and target == 'UNREAL'))
    _add_root(context, clean)
    renames = _rename_for_target(clean, dup_meshes, target)

    # 5b. bake animation onto the clean skeleton (optional): the active action,
    # or every action that animates the rig (one FBX animation stack each,
    # via one NLA strip per baked action)
    n_frames = 0
    n_actions = 0
    anim_note = ""
    baked = []
    if include_anim or all_actions:
        prev_act = rig.animation_data.action if rig.animation_data else None
        acts = (_actions_for_rig(rig) if all_actions
                else ([prev_act] if prev_act else []))
        if not acts:
            anim_note = "no action on the rig"
        try:
            for act in acts:
                if rig.animation_data is None:
                    rig.animation_data_create()
                rig.animation_data.action = act
                if clean.animation_data:
                    clean.animation_data.action = None
                n, note = _bake_animation(context, rig, clean, renames)
                if n > 0:
                    n_frames += n
                    n_actions += 1
                    baked.append(clean.animation_data.action)
                elif not anim_note:
                    anim_note = note
        except Exception as e:
            _cleanup(clean, dup_meshes, baked)
            return False, f"Animation bake failed: {e}", {}
        finally:
            if rig.animation_data:
                rig.animation_data.action = prev_act
        if all_actions and baked:
            ad = clean.animation_data
            ad.action = None
            for a in baked:
                tr = ad.nla_tracks.new()
                tr.name = a.name
                tr.strips.new(a.name, int(a.frame_range[0]), a)

    stats = {"bones": len(clean.data.bones), "merged": n_merged,
             "meshes": len(dup_meshes), "frames": n_frames,
             "face_stripped": n_face, "actions": n_actions,
             "reparented": n_reparented}

    # 6. export selection
    for o in context.selected_objects:
        o.select_set(False)
    clean.select_set(True)
    for dm in dup_meshes:
        dm.select_set(True)
    context.view_layer.objects.active = clean

    kw = dict(
        filepath=filepath,
        use_selection=True,
        object_types={'ARMATURE', 'MESH'},
        use_mesh_modifiers=apply_modifiers,
        add_leaf_bones=add_leaf_bones,
        bake_anim=n_frames > 0,
        global_scale=1.0,
        path_mode='COPY',
    )
    if n_frames > 0:
        kw.update(
            bake_anim_use_all_bones=True,
            # multi-action mode parks one NLA strip per baked action — the
            # exporter then writes one animation stack (clip) per strip
            bake_anim_use_nla_strips=all_actions and len(baked) > 0,
            bake_anim_use_all_actions=False,
            bake_anim_force_startend_keying=True,
            bake_anim_simplify_factor=anim_simplify,
        )
    kw.update(_FBX_SETTINGS.get(target, _FBX_SETTINGS['UNITY']))
    try:
        bpy.ops.export_scene.fbx(**kw)
    except Exception as e:
        _cleanup(clean, dup_meshes, baked)
        return False, f"FBX export failed: {e}", stats

    if not keep_in_scene:
        _cleanup(clean, dup_meshes, baked)

    # restore user's selection/active
    try:
        for o in context.selected_objects:
            o.select_set(False)
        for o in prev_sel:
            if o.name in context.view_layer.objects:
                o.select_set(True)
        if prev_active and prev_active.name in context.view_layer.objects:
            context.view_layer.objects.active = prev_active
    except Exception:
        pass

    dbg(f"[game_export] {target}: {stats['bones']} bones, "
        f"{stats['merged']} merged, {stats['meshes']} mesh(es), "
        f"{stats['frames']} anim frames -> {filepath}")
    anim_part = (f", {n_actions} clip(s)/{n_frames} frames" if n_actions > 1
                 else f", {n_frames} anim frames" if n_frames > 0
                 else f", no animation ({anim_note})"
                 if (include_anim or all_actions) else "")
    face_part = f", {n_face} face bones stripped" if n_face else ""
    return True, (f"Exported {stats['bones']}-bone {target} skeleton "
                  f"({stats['meshes']} mesh{anim_part}{face_part})"), stats


def _cleanup(clean, dup_meshes, baked_actions=()):
    for dm in dup_meshes:
        me = dm.data
        bpy.data.objects.remove(dm, do_unlink=True)
        if me.users == 0:
            bpy.data.meshes.remove(me)
    arm_data = clean.data
    bpy.data.objects.remove(clean, do_unlink=True)
    if arm_data.users == 0:
        bpy.data.armatures.remove(arm_data)
    # baked actions lose their last user with the skeleton — don't leak them
    for a in baked_actions:
        try:
            if a.users == 0:
                bpy.data.actions.remove(a)
        except (ReferenceError, RuntimeError):
            pass


class AUTORIG_OT_ExportGame(bpy.types.Operator):
    """Export the generated rig and skinned mesh as a clean, game-ready skeleton"""
    bl_idname = "autorig.export_game"
    bl_label = "Export to Game Engine"
    bl_options = {'REGISTER'}

    filepath: bpy.props.StringProperty(subtype='FILE_PATH')
    filename_ext = ".fbx"
    filter_glob: bpy.props.StringProperty(default="*.fbx", options={'HIDDEN'})

    target: bpy.props.EnumProperty(
        name="Engine",
        items=[
            ('UNITY',  "Unity",
             "Clean single-bone-per-limb skeleton with stripped names "
             "(Unity Humanoid/Generic friendly)"),
            ('UNREAL', "Unreal",
             "Single-bone-per-limb skeleton renamed to the UE5 Mannequin "
             "convention (upperarm_l, calf_r, spine_01, neck_01, ...)"),
        ],
        default='UNITY',
    )
    apply_modifiers: bpy.props.BoolProperty(
        name="Apply Modifiers",
        description="Bake non-armature modifiers (e.g. Subsurf) into the exported "
                    "mesh. Off exports the authored cage — usually what games want, "
                    "since Subsurf can multiply the poly count",
        default=False,
    )
    add_leaf_bones: bpy.props.BoolProperty(
        name="Add Leaf Bones (correct bone length)",
        description="Write a tiny tip bone at each childless bone so bone lengths "
                    "survive FBX (breast, pelvis, shoulder otherwise reimport "
                    "oversized). Adds extra '_end' bones most engines ignore. "
                    "Skinning is correct either way — this only fixes displayed "
                    "bone length",
        default=False,
    )
    strip_face: bpy.props.BoolProperty(
        name="Strip Face Bones",
        description="Remove the face rig from the game skeleton (60+ bones on "
                    "a full face rig) and fold its weights into the head, so "
                    "the face follows the head rigidly. Most game characters "
                    "don't need bone-driven faces; use blend shapes instead",
        default=False,
    )
    anim_mode: bpy.props.EnumProperty(
        name="Animation",
        items=[
            ('NONE',   "None", "Export the mesh and skeleton only"),
            ('ACTIVE', "Current Action",
             "Bake the rig's current action onto the game skeleton"),
            ('ALL',    "All Actions",
             "Bake every action in this file that animates the rig — each "
             "becomes its own animation clip in the FBX"),
        ],
        default='NONE',
    )
    preserve_twist: bpy.props.BoolProperty(
        name="Preserve Twist Bones",
        description="Unreal only: keep the limb twist segments as UE Mannequin "
                    "twist bones (upperarm_twist_01_l, calf_twist_01_l, ...) "
                    "instead of merging them — preserves deformation quality "
                    "at the cost of 8 extra bones",
        default=False,
    )
    anim_simplify: bpy.props.FloatProperty(
        name="Anim Simplify",
        description="FBX animation curve simplification. 0 keeps every baked "
                    "frame exactly; higher values shrink the file but can drift "
                    "on fast motion",
        default=0.0, min=0.0, max=10.0,
    )

    def invoke(self, context, event):
        if _find_game_rig(context) is None:
            self.report({'ERROR'}, "No generated Rigify rig found. Generate a rig first.")
            return {'CANCELLED'}
        if not self.filepath:
            base = bpy.path.basename(bpy.data.filepath) or "character"
            self.filepath = bpy.path.ensure_ext(base.replace(".blend", ""), ".fbx")
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        ok, msg, _ = build_and_export(
            context, bpy.path.ensure_ext(self.filepath, ".fbx"),
            target=self.target, apply_modifiers=self.apply_modifiers,
            add_leaf_bones=self.add_leaf_bones,
            include_anim=self.anim_mode == 'ACTIVE',
            all_actions=self.anim_mode == 'ALL',
            anim_simplify=self.anim_simplify,
            strip_face=self.strip_face,
            preserve_twist=self.preserve_twist)
        self.report({'INFO'} if ok else {'ERROR'}, msg)
        return {'FINISHED'} if ok else {'CANCELLED'}


def draw_game_export_section(layout, context):
    box = layout.box()
    box.label(text="Game Export", icon='EXPORT')
    rig = _find_game_rig(context)
    if rig is None:
        box.label(text="Generate a rig first", icon='INFO')
        return
    n_mesh = len(_bound_meshes(context, rig))
    col = box.column(align=True)
    col.label(text=f"Rig: {rig.name}  ({n_mesh} bound mesh)", icon='ARMATURE_DATA')
    r = col.row()
    r.enabled = n_mesh > 0
    r.operator("autorig.export_game", icon='EXPORT')
    if n_mesh == 0:
        col.label(text="Bind a mesh in the Skin tab", icon='INFO')
