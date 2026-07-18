# Headless test of retarget.py (the real module, not the spike math).
# Run:  blender --background --factory-startup --python test_retarget.py
#
# 1. Mixamo-named source (A-pose, cm scale, rolled bones) -> preset mapping,
#    run_retarget, verify wrist world position through the Rigify stack and
#    that a NEW action was created with the rig's old action preserved.
# 2. UE-style-named source -> fuzzy mapping finds the sided limb chain.
import bpy
import sys
import math
import types
import importlib.util
from mathutils import Vector, Euler

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

# give the rig a pre-existing action that must survive the retarget
rig.animation_data_create()
prev = bpy.data.actions.new("user_previous_anim")
rig.animation_data.action = prev

# ── Mixamo-style source (A-pose, hips Z=100, rolls 0.7) ─────────────────────
def build_armature(name, bones):
    """bones: [(name, head, tail, parent)]"""
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
    ("mixamorig:LeftUpLeg",    (9, 0, 100),  (9, 0, 55),   "mixamorig:Hips"),
    ("mixamorig:LeftLeg",      (9, 0, 55),   (9, 0, 10),   "mixamorig:LeftUpLeg"),
    ("mixamorig:LeftFoot",     (9, 0, 10),   (9, -12, 2),  "mixamorig:LeftLeg"),
])

# animate: rigid LeftArm swing + hips translation
pb_arm = src.pose.bones["mixamorig:LeftArm"]
pb_hips = src.pose.bones["mixamorig:Hips"]
pb_arm.rotation_mode = 'XYZ'
scn = bpy.context.scene
scn.frame_set(1)
pb_arm.keyframe_insert("rotation_euler", frame=1)
pb_hips.keyframe_insert("location", frame=1)
scn.frame_set(20)
pb_arm.rotation_euler = Euler((math.radians(50), math.radians(20), 0), 'XYZ')
pb_arm.keyframe_insert("rotation_euler", frame=20)
pb_hips.location = (30, 0, -5)
pb_hips.keyframe_insert("location", frame=20)

# ── mapping (preset path) ───────────────────────────────────────────────────
mapping = rt.build_mapping(src, rig)
mapped_tgts = {t for _, t, _ in mapping}
print(f"  preset mapping: {len(mapping)} pairs -> {sorted(mapped_tgts)}")
for need in ("torso", "upper_arm_fk.L", "forearm_fk.L", "hand_fk.L",
             "thigh_fk.L", "shin_fk.L", "foot_fk.L", "neck", "head"):
    if need not in mapped_tgts:
        fail(f"preset mapping missed {need}")
if not any(loc for _, t, loc in mapping if t == "torso"):
    fail("torso pair is not the location carrier")
ok(f"preset mapping: {len(mapping)} pairs, all core controls present")

# ── run the real bake ───────────────────────────────────────────────────────
okflag, msg, stats = rt.run_retarget(bpy.context, src, rig, mapping)
print(f"  run_retarget -> {okflag}: {msg}")
if not okflag:
    fail(msg)
if stats["frames"] != 20:
    fail(f"expected 20 frames, got {stats['frames']}")
if rig.animation_data.action.name != "mixamo_srcAction_retarget":
    print(f"  (action name: {rig.animation_data.action.name})")
if rig.animation_data.action == prev:
    fail("retarget overwrote the rig's previous action")
if "user_previous_anim" not in bpy.data.actions:
    fail("previous action datablock lost")
if stats["fk_switches"] < 4:
    fail(f"expected 4 IK_FK switches keyed, got {stats['fk_switches']}")
ok(f"new action '{stats['action']}', previous action preserved, "
   f"{stats['fk_switches']} IK/FK switches keyed")

# ── world-position verification (same discipline as the spike) ─────────────
def achieved(obj, bname):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    return ev.matrix_world @ ev.pose.bones[bname].matrix

def rest_w(obj, bname):
    return obj.matrix_world @ obj.data.bones[bname].matrix_local

sh_rest = rest_w(rig, "upper_arm_fk.L").translation
wr_rest = rest_w(rig, "hand_fk.L").translation
src_hip_rest = rest_w(src, "mixamorig:Hips").translation
ratio = rest_w(rig, "torso").translation.z / src_hip_rest.z

for frame in (1, 20):
    scn.frame_set(frame)
    bpy.context.view_layer.update()
    R_delta = ((src.matrix_world @ src.pose.bones["mixamorig:LeftArm"].matrix).to_3x3()
               @ rest_w(src, "mixamorig:LeftArm").to_3x3().inverted())
    hips_t = ((src.matrix_world @ src.pose.bones["mixamorig:Hips"].matrix).translation
              - src_hip_rest) * ratio
    sh_now = achieved(rig, "ORG-upper_arm.L").translation
    wr_now = achieved(rig, "ORG-hand.L").translation
    wr_expect = sh_now + R_delta @ (wr_rest - sh_rest)
    err = (wr_now - wr_expect).length
    err_sh = (sh_now - (sh_rest + hips_t)).length
    print(f"  frame {frame}: wrist err={err:.6f} m, shoulder-follows-hips "
          f"err={err_sh:.6f} m")
    if err > 2e-3 or err_sh > 2e-3:
        fail(f"frame {frame}: world positions off (wrist {err:.4f}, "
             f"shoulder {err_sh:.4f})")
ok("world positions exact through the Rigify stack at both keyed frames")

# ── fuzzy path: UE-style names ──────────────────────────────────────────────
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

print("\nALL CHECKS PASSED")
