# Headless test of retarget.py (the real module, not the spike math).
# Run:  blender --background --factory-startup --python test_retarget.py
#
# 1. Mixamo-named source (A-pose, cm scale, rolled bones) -> preset mapping,
#    run_retarget with Match Clip Pose (default): the clip's actual limb poses
#    must be reproduced — wrist lands where the SOURCE arm directions point,
#    even though the target rest is T-pose. New action created, rig's old
#    action preserved.
# 2. Same source rotated 180 deg (facing away, the "hands behind the back"
#    bug class): auto facing correction must map the motion into the
#    character's own frame.
# 3. align_rests=False keeps the original delta semantics (offsets from the
#    character's own rest).
# 4. UE-style-named source -> fuzzy mapping finds the sided limb chain.
import bpy
import sys
import math
import types
import importlib.util
from mathutils import Matrix, Vector, Euler

def fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)

def ok(msg):
    print(f"[OK] {msg}")

# load retarget.py as a package module (`from .constants import dbg`)
pkg = types.ModuleType("erpkg")
pkg.__path__ = []
sys.modules["erpkg"] = pkg
const = types.ModuleType("erpkg.constants")
const.dbg = lambda *a, **k: print(*a)
const.LITE_BUILD = False
sys.modules["erpkg.constants"] = const
spec = importlib.util.spec_from_file_location(
    "erpkg.retarget", r"d:\rig_addon_pro\retarget.py")
rt = importlib.util.module_from_spec(spec)
sys.modules["erpkg.retarget"] = rt
spec.loader.exec_module(rt)

# ── target rig ──────────────────────────────────────────────────────────────
bpy.ops.preferences.addon_enable(module="rigify")
bpy.ops.wm.read_homefile(use_empty=True)
bpy.ops.preferences.addon_enable(module="rigify")
bpy.ops.object.armature_human_metarig_add()
bpy.ops.pose.rigify_generate()
rig = bpy.context.active_object
ok(f"generated rig: {len(rig.data.bones)} bones")

rig.animation_data_create()
prev = bpy.data.actions.new("user_previous_anim")
rig.animation_data.action = prev

# ── Mixamo-style source (A-pose, hips Z=100, rolls 0.7) ─────────────────────
def build_armature(name, bones):
    data = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    for bn, h, t, p in bones:
        eb = data.edit_bones.new(bn)
        eb.head, eb.tail, eb.roll = Vector(h), Vector(t), 0.7
        if p:
            eb.parent = data.edit_bones[p]
    bpy.ops.object.mode_set(mode='OBJECT')
    return obj

D = math.sqrt(0.5)
a = Vector((8, 0, 142))
b = a + Vector((30 * D, 0, -30 * D))
c = b + Vector((25 * D, 0, -25 * D))
d = c + Vector((10 * D, 0, -10 * D))
src = build_armature("mixamo_src", [
    ("mixamorig:Hips",         (0, 0, 100),  (0, 0, 110),  None),
    ("mixamorig:Spine",        (0, 0, 110),  (0, 0, 120),  "mixamorig:Hips"),
    ("mixamorig:Spine1",       (0, 0, 120),  (0, 0, 130),  "mixamorig:Spine"),
    ("mixamorig:Spine2",       (0, 0, 130),  (0, 0, 140),  "mixamorig:Spine1"),
    ("mixamorig:Neck",         (0, 0, 140),  (0, 0, 150),  "mixamorig:Spine2"),
    ("mixamorig:Head",         (0, 0, 150),  (0, 0, 165),  "mixamorig:Neck"),
    ("mixamorig:LeftShoulder", (2, 0, 140),  tuple(a),     "mixamorig:Spine2"),
    ("mixamorig:LeftArm",      tuple(a),     tuple(b),     "mixamorig:LeftShoulder"),
    ("mixamorig:LeftForeArm",  tuple(b),     tuple(c),     "mixamorig:LeftArm"),
    ("mixamorig:LeftHand",     tuple(c),     tuple(d),     "mixamorig:LeftForeArm"),
    ("mixamorig:RightShoulder", (-2, 0, 140), (-8, 0, 142), "mixamorig:Spine2"),
    ("mixamorig:RightArm",     (-8, 0, 142), (-8 - 30 * D, 0, 142 - 30 * D), "mixamorig:RightShoulder"),
    ("mixamorig:LeftUpLeg",    (9, 0, 100),  (9, 0, 55),   "mixamorig:Hips"),
    ("mixamorig:LeftLeg",      (9, 0, 55),   (9, 0, 10),   "mixamorig:LeftUpLeg"),
    # foot heading ~35 deg OUTWARD — differs from the character's straight
    # feet; the retarget must keep the character's heading (only pitch is
    # taken from the clip) or the shin twist bones wind up
    ("mixamorig:LeftFoot",     (9, 0, 10),   (16, -10, 2), "mixamorig:LeftLeg"),
])

pb_arm = src.pose.bones["mixamorig:LeftArm"]
pb_hips = src.pose.bones["mixamorig:Hips"]
pb_arm.rotation_mode = 'XYZ'
pb_hips.rotation_mode = 'XYZ'
scn = bpy.context.scene
scn.frame_set(1)
pb_arm.keyframe_insert("rotation_euler", frame=1)
pb_hips.keyframe_insert("location", frame=1)
pb_hips.keyframe_insert("rotation_euler", frame=1)
scn.frame_set(20)
pb_arm.rotation_euler = Euler((math.radians(50), math.radians(20), 0), 'XYZ')
pb_arm.keyframe_insert("rotation_euler", frame=20)
pb_hips.location = (30, 0, -5)
pb_hips.keyframe_insert("location", frame=20)
pb_hips.keyframe_insert("rotation_euler", frame=20)
# frames 20-40: the character TURNS 240 deg — this drives quaternions across
# the double-cover boundary (the foot/shin snap-during-turns bug class)
scn.frame_set(40)
pb_hips.rotation_euler = Euler((0, 0, math.radians(240)), 'XYZ')
pb_hips.keyframe_insert("rotation_euler", frame=40)

# ── mapping (preset path) ───────────────────────────────────────────────────
mapping = rt.build_mapping(src, rig)
mapped_tgts = {t for _, t, _ in mapping}
print(f"  preset mapping: {len(mapping)} pairs -> {sorted(mapped_tgts)}")
for need in ("torso", "upper_arm_fk.L", "forearm_fk.L", "hand_fk.L",
             "upper_arm_fk.R", "thigh_fk.L", "shin_fk.L", "foot_fk.L",
             "neck", "head"):
    if need not in mapped_tgts:
        fail(f"preset mapping missed {need}")
if not any(loc for _, t, loc in mapping if t == "torso"):
    fail("torso pair is not the location carrier")
ok(f"preset mapping: {len(mapping)} pairs, all core controls present")

# ── verification helpers ────────────────────────────────────────────────────
def achieved(obj, bname):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    return ev.matrix_world @ ev.pose.bones[bname].matrix

def bone_dir(M):
    return (M.to_3x3() @ Vector((0.0, 1.0, 0.0))).normalized()

LEN_U = rig.data.bones["ORG-upper_arm.L"].length
LEN_F = rig.data.bones["ORG-forearm.L"].length

def check_matches_clip(C, label):
    """With Match Clip Pose, the target arm bones must point along the
    (facing-corrected) SOURCE arm's pose directions — so the wrist sits at
    shoulder + len_upper*dir_upper + len_forearm*dir_forearm."""
    for frame in (1, 20):
        scn.frame_set(frame)
        bpy.context.view_layer.update()
        d_u = C @ bone_dir(src.matrix_world @ src.pose.bones["mixamorig:LeftArm"].matrix)
        d_f = C @ bone_dir(src.matrix_world @ src.pose.bones["mixamorig:LeftForeArm"].matrix)
        sh_now = achieved(rig, "ORG-upper_arm.L").translation
        wr_now = achieved(rig, "ORG-hand.L").translation
        wr_expect = sh_now + d_u * LEN_U + d_f * LEN_F
        err = (wr_now - wr_expect).length
        print(f"  [{label}] frame {frame}: wrist-vs-clip err = {err:.6f} m")
        if err > 2e-3:
            fail(f"[{label}] frame {frame}: wrist off by {err:.4f} m")

# ── 1. Match Clip Pose (default) ────────────────────────────────────────────
okflag, msg, stats = rt.run_retarget(bpy.context, src, rig, mapping)
print(f"  run_retarget -> {okflag}: {msg}")
if not okflag:
    fail(msg)
if stats["frames"] != 40:
    fail(f"expected 40 frames, got {stats['frames']}")
if rig.animation_data.action == prev or "user_previous_anim" not in bpy.data.actions:
    fail("previous action lost")
if stats["fk_switches"] < 4:
    fail(f"expected 4 IK_FK switches keyed, got {stats['fk_switches']}")
if abs(stats["facing_yaw_deg"]) > 1.0:
    fail(f"same-facing rigs but facing_yaw={stats['facing_yaw_deg']}")
check_matches_clip(Matrix.Identity(3), "align")
ok("Match Clip Pose: target arm reproduces the clip's limb directions "
   "(A-pose source on T-pose rig)")

# floor calibration: at frame 1 the source is at rest — the character's foot
# must sit at its OWN rest height (clip-matching straightens bent-knee rests,
# which used to sink the feet below the floor)
scn.frame_set(1)
bpy.context.view_layer.update()
foot_rest_z = (rig.matrix_world
               @ rig.data.bones["ORG-foot.L"].matrix_local).translation.z
foot_now_z = achieved(rig, "ORG-foot.L").translation.z
err_floor = abs(foot_now_z - foot_rest_z)
print(f"  floor: rest ankle z={foot_rest_z:.4f}, retargeted z={foot_now_z:.4f}, "
      f"err={err_floor:.6f} m")
if err_floor > 5e-3:
    fail(f"foot sank {err_floor:.4f} m below its rest height")
ok("floor calibration: foot stays at the character's rest height")

# feet are DELTA-ONLY: at the clip's rest frame the character's foot must sit
# EXACTLY at its own rest orientation (sole flat on the floor) even though the
# source feet are splayed 35 deg outward with a different pitch — matching
# the clip's foot rest tilts the sole off the floor and Rigify's foot->shin
# coupling renders that as shin twist
M_foot_rest = (rig.matrix_world @ rig.data.bones["ORG-foot.L"].matrix_local)
M_foot_now = achieved(rig, "ORG-foot.L")
ang_foot = math.degrees(M_foot_now.to_quaternion().rotation_difference(
    M_foot_rest.to_quaternion()).angle)
ang_foot = min(ang_foot, 360 - ang_foot)
print(f"  foot at clip-rest frame: {ang_foot:.3f} deg from the character's "
      f"own rest orientation")
if ang_foot > 1.0:
    fail(f"foot rotated {ang_foot:.1f} deg off its rest at the neutral frame")
ok("feet delta-only: sole stays flat at neutral (no toes-up, no shin twist)")

# quaternion continuity: through the 240-deg turn, consecutive keyed
# quaternions must never flip sign (dot >= 0) or joints spin the long way
act_new = rig.animation_data.action
for bone_chk in ("foot_fk.L", "shin_fk.L", "hand_fk.L"):
    path = f'pose.bones["{bone_chk}"].rotation_quaternion'
    curves = [act_new.fcurves.find(path, index=i) for i in range(4)]
    if any(c is None for c in curves):
        fail(f"no quaternion fcurves for {bone_chk}")
    prev_v = None
    for f in range(1, 41):
        v = Vector([c.evaluate(f) for c in curves])
        if prev_v is not None and prev_v.dot(v) < 0.0:
            fail(f"{bone_chk}: quaternion sign flip between frames "
                 f"{f - 1} and {f}")
        prev_v = v
ok("quaternion continuity: no sign flips through the 240-deg turn")

# IK controllers must ride along with the FK result (parked IK controllers
# pin the feet if a limb is, or is switched, to IK — legs wind up during
# turns). Check at a mid-turn frame, then flip the leg to IK and confirm the
# deform chain STILL follows the clip.
scn.frame_set(30)
bpy.context.view_layer.update()
for ik_b, fk_b in (("foot_ik.L", "foot_fk.L"), ("hand_ik.L", "hand_fk.L"),
                   ("thigh_ik.L", "thigh_fk.L")):
    # the IK control's rotation-FROM-ITS-REST must equal the FK bone's
    # (rest conventions can differ — foot_ik may not be a twin of foot_fk)
    M_ik = achieved(rig, ik_b)
    M_fk = achieved(rig, fk_b)
    rest_ik = (rig.matrix_world @ rig.data.bones[ik_b].matrix_local).to_3x3()
    rest_fk = (rig.matrix_world @ rig.data.bones[fk_b].matrix_local).to_3x3()
    D_ik = M_ik.to_3x3() @ rest_ik.inverted()
    D_fk = M_fk.to_3x3() @ rest_fk.inverted()
    ang = math.degrees(D_ik.to_quaternion().rotation_difference(
        D_fk.to_quaternion()).angle)
    ang = min(ang, 360 - ang)
    err_t = (M_ik.translation - M_fk.translation).length
    print(f"  {ik_b}: pos err {err_t:.6f} m, delta-rot err {ang:.3f} deg")
    # thigh_ik carries Rigify's own IK constraint stack, so its EVALUATED
    # rotation includes the solver's aim — a few degrees off the keyed basis
    # is expected; the strict correctness gate is the whole-leg-in-IK check
    tol = 5.0 if ik_b.startswith("thigh_ik") else 0.5
    if err_t > 2e-3 or ang > tol:
        fail(f"{ik_b} not tracking {fk_b}'s delta")
M_before = {b: achieved(rig, b).copy()
            for b in ("ORG-foot.L", "ORG-shin.L", "ORG-thigh.L")}
rig.pose.bones["thigh_parent.L"]["IK_FK"] = 0.0   # force the leg to IK
bpy.context.view_layer.update()
for b, Mb in M_before.items():
    Ma = achieved(rig, b)
    err_sw = (Ma.translation - Mb.translation).length
    ang_sw = math.degrees(Ma.to_quaternion().rotation_difference(
        Mb.to_quaternion()).angle)
    ang_sw = min(ang_sw, 360 - ang_sw)
    print(f"  leg in IK: {b} moved {err_sw:.6f} m, rotated {ang_sw:.3f} deg")
    if err_sw > 5e-3 or ang_sw > 1.0:
        fail(f"{b} in IK mode diverges (pos {err_sw:.4f} m, "
             f"rot {ang_sw:.2f} deg)")
rig.pose.bones["thigh_parent.L"]["IK_FK"] = 1.0
ok("IK controllers baked (delta-correct): whole leg identical in FK and IK")

# ── 2. source rotated 180 deg — the hands-behind-the-back bug class ─────────
src.rotation_euler = Euler((0, 0, math.pi), 'XYZ')
bpy.context.view_layer.update()
okflag, msg, stats = rt.run_retarget(bpy.context, src, rig, mapping)
if not okflag:
    fail(msg)
if abs(abs(stats["facing_yaw_deg"]) - 180.0) > 1.0:
    fail(f"expected ~180 facing yaw, got {stats['facing_yaw_deg']}")
C180 = Matrix.Rotation(math.radians(stats["facing_yaw_deg"]), 3, 'Z')
check_matches_clip(C180, "180deg")
ok(f"facing auto-correction: {stats['facing_yaw_deg']} deg detected, motion "
   "mapped into the character's frame")
src.rotation_euler = Euler((0, 0, 0), 'XYZ')
bpy.context.view_layer.update()

# ── 3. align_rests=False keeps the original delta semantics ────────────────
okflag, msg, stats = rt.run_retarget(bpy.context, src, rig, mapping,
                                     align_rests=False)
if not okflag:
    fail(msg)

def rest_w(obj, bname):
    return obj.matrix_world @ obj.data.bones[bname].matrix_local

sh_rest = rest_w(rig, "upper_arm_fk.L").translation
wr_rest = rest_w(rig, "hand_fk.L").translation
for frame in (1, 20):
    scn.frame_set(frame)
    bpy.context.view_layer.update()
    R_delta = ((src.matrix_world @ src.pose.bones["mixamorig:LeftArm"].matrix).to_3x3()
               @ rest_w(src, "mixamorig:LeftArm").to_3x3().inverted())
    sh_now = achieved(rig, "ORG-upper_arm.L").translation
    wr_now = achieved(rig, "ORG-hand.L").translation
    wr_expect = sh_now + R_delta @ (wr_rest - sh_rest)
    err = (wr_now - wr_expect).length
    print(f"  [no-align] frame {frame}: wrist delta-semantics err = {err:.6f} m")
    if err > 2e-3:
        fail(f"[no-align] frame {frame}: wrist off by {err:.4f} m")
ok("align_rests=False preserves the original character-rest delta semantics")

# ── 4. fuzzy path: UE-style names ───────────────────────────────────────────
src2 = build_armature("ue_src", [
    ("pelvis",     (0, 0, 0.9),   (0, 0, 1.0),    None),
    ("neck_01",    (0, 0, 1.4),   (0, 0, 1.5),    "pelvis"),
    ("head",       (0, 0, 1.5),   (0, 0, 1.6),    "neck_01"),
    ("clavicle_l", (0.02, 0, 1.4), (0.08, 0, 1.4), "pelvis"),
    ("upperarm_l", (0.08, 0, 1.4), (0.3, 0, 1.4),  "clavicle_l"),
    ("lowerarm_l", (0.3, 0, 1.4),  (0.5, 0, 1.4),  "upperarm_l"),
    ("hand_l",     (0.5, 0, 1.4),  (0.6, 0, 1.4),  "lowerarm_l"),
    ("thigh_r",    (-0.09, 0, 0.9), (-0.09, 0, 0.5), "pelvis"),
    ("calf_r",     (-0.09, 0, 0.5), (-0.09, 0, 0.1), "thigh_r"),
    ("foot_r",     (-0.09, 0, 0.1), (-0.09, -0.1, 0), "calf_r"),
])
m2 = rt.build_mapping(src2, rig)
tgts2 = {t for _, t, _ in m2}
print(f"  fuzzy mapping: {len(m2)} pairs -> {sorted(tgts2)}")
for need in ("torso", "shoulder.L", "upper_arm_fk.L", "forearm_fk.L",
             "hand_fk.L", "thigh_fk.R", "shin_fk.R", "foot_fk.R",
             "neck", "head"):
    if need not in tgts2:
        fail(f"fuzzy mapping missed {need}")
ok(f"fuzzy mapping resolved UE-style names: {len(m2)} pairs incl. both sides")

# ── 5. fuzzy path: Character Creator names (INFIX side tokens: CC_Base_L_*) ──
src3 = build_armature("cc_src", [
    ("CC_Base_Hip",        (0, 0, 0.9),    (0, 0, 1.0),    None),
    ("CC_Base_Waist",      (0, 0, 1.0),    (0, 0, 1.1),    "CC_Base_Hip"),
    ("CC_Base_Spine01",    (0, 0, 1.1),    (0, 0, 1.25),   "CC_Base_Waist"),
    ("CC_Base_Spine02",    (0, 0, 1.25),   (0, 0, 1.4),    "CC_Base_Spine01"),
    ("CC_Base_NeckTwist01", (0, 0, 1.4),   (0, 0, 1.5),    "CC_Base_Spine02"),
    ("CC_Base_Head",       (0, 0, 1.5),    (0, 0, 1.6),    "CC_Base_NeckTwist01"),
    ("CC_Base_L_Clavicle", (0.02, 0, 1.4), (0.08, 0, 1.4), "CC_Base_Spine02"),
    ("CC_Base_L_Upperarm", (0.08, 0, 1.4), (0.3, 0, 1.4),  "CC_Base_L_Clavicle"),
    ("CC_Base_L_Forearm",  (0.3, 0, 1.4),  (0.5, 0, 1.4),  "CC_Base_L_Upperarm"),
    ("CC_Base_L_Hand",     (0.5, 0, 1.4),  (0.6, 0, 1.4),  "CC_Base_L_Forearm"),
    ("CC_Base_L_Thigh",    (0.09, 0, 0.9), (0.09, 0, 0.5), "CC_Base_Hip"),
    ("CC_Base_L_Calf",     (0.09, 0, 0.5), (0.09, 0, 0.1), "CC_Base_L_Thigh"),
    ("CC_Base_L_Foot",     (0.09, 0, 0.1), (0.09, -0.1, 0), "CC_Base_L_Calf"),
    ("CC_Base_R_Thigh",    (-0.09, 0, 0.9), (-0.09, 0, 0.5), "CC_Base_Hip"),
    ("CC_Base_R_Calf",     (-0.09, 0, 0.5), (-0.09, 0, 0.1), "CC_Base_R_Thigh"),
    ("CC_Base_R_Foot",     (-0.09, 0, 0.1), (-0.09, -0.1, 0), "CC_Base_R_Calf"),
])
m3 = rt.build_mapping(src3, rig)
tgts3 = {t for _, t, _ in m3}
print(f"  CC mapping: {len(m3)} pairs -> {sorted(tgts3)}")
for need in ("torso", "shoulder.L", "upper_arm_fk.L", "forearm_fk.L",
             "hand_fk.L", "thigh_fk.L", "shin_fk.L", "foot_fk.L",
             "thigh_fk.R", "shin_fk.R", "foot_fk.R", "neck", "head",
             "spine_fk.001", "spine_fk.002"):
    if need not in tgts3:
        fail(f"CC mapping missed {need}")
ok(f"fuzzy mapping resolved Character Creator names (infix _L_/_R_): "
   f"{len(m3)} pairs, arms AND legs both sides")

# ── 6. custom-mapping plumbing: pairs -> validated/ordered, JSON round-trip ──
import os, tempfile
pairs = [(s, t) for s, t, _ in mapping]           # from the Mixamo mapping
pairs.append(("bogus_bone", "no_such_control"))   # must be dropped
m4 = rt.mapping_from_pairs(rig, pairs)
if len(m4) != len(mapping):
    fail(f"mapping_from_pairs: expected {len(mapping)} valid pairs, got {len(m4)}")
if not any(loc for _, t, loc in m4 if t == "torso"):
    fail("mapping_from_pairs lost the torso location flag")
depths = []
for _, t, _ in m4:
    d, b = 0, rig.data.bones[t]
    while b.parent:
        d, b = d + 1, b.parent
    depths.append(d)
if depths != sorted(depths):
    fail("mapping_from_pairs not parents-first ordered")
tmp = os.path.join(tempfile.gettempdir(), "er_test_map.json")
rt.save_mapping_json(tmp, pairs[:-1])
loaded = rt.load_mapping_json(tmp)
os.remove(tmp)
if loaded != pairs[:-1]:
    fail("JSON mapping round-trip mismatch")
ok(f"custom mapping: validation, torso flag, ordering, JSON round-trip "
   f"({len(loaded)} pairs)")

print("\nALL CHECKS PASSED")
