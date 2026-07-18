# Retarget spike — world-space rotation-delta retarget onto Rigify FK controls.
# Run:  blender --background --factory-startup --python spike_retarget.py
#
# Proves the RETARGET_DESIGN.md algorithm end-to-end:
#   - source: fake Mixamo skeleton, A-pose rest (target metarig is T-pose),
#     100x scale (cm-style), nonzero bone rolls
#   - target: real generated Rigify rig, keys land on FK controls + torso
#   - verify WORLD POSITIONS via the depsgraph (through Rigify's constraint
#     stack), not names/channels: rigid source arm swing must rigidly rotate
#     the whole target arm about the shoulder; hips translation must arrive
#     scaled by the hip-height ratio.
import bpy
import sys
import math
from mathutils import Matrix, Vector, Euler

def fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)

def ok(msg):
    print(f"[OK] {msg}")

# ── target: real Rigify rig ─────────────────────────────────────────────────
bpy.ops.preferences.addon_enable(module="rigify")
bpy.ops.wm.read_homefile(use_empty=True)
bpy.ops.preferences.addon_enable(module="rigify")

bpy.ops.object.armature_human_metarig_add()
bpy.ops.pose.rigify_generate()
rig = bpy.context.active_object
ok(f"generated rig: {len(rig.data.bones)} bones")

# arms/legs default to IK — flip the left arm to FK so FK keys drive the limb
ikfk = rig.pose.bones.get("upper_arm_parent.L")
if ikfk is None or "IK_FK" not in ikfk.keys():
    fail("no IK_FK switch on upper_arm_parent.L")
ikfk["IK_FK"] = 1.0

# ── source: fake Mixamo skeleton ────────────────────────────────────────────
# Hips at Z=100 (cm scale), arm chain in A-pose (45 deg down), rolls = 0.7.
arm_data = bpy.data.armatures.new("mixamo_src")
src = bpy.data.objects.new("mixamo_src", arm_data)
bpy.context.scene.collection.objects.link(src)
bpy.context.view_layer.objects.active = src
bpy.ops.object.mode_set(mode='EDIT')
ebs = arm_data.edit_bones

def add_bone(name, head, tail, parent=None, roll=0.7):
    eb = ebs.new(name)
    eb.head, eb.tail, eb.roll = Vector(head), Vector(tail), roll
    if parent:
        eb.parent = ebs[parent]
    return eb

DOWN45 = math.sqrt(0.5)
add_bone("mixamorig:Hips", (0, 0, 100), (0, 0, 110), None)
add_bone("mixamorig:LeftShoulder", (2, 0, 140), (8, 0, 142), "mixamorig:Hips")
# A-pose: upper arm 30cm at 45 deg down, forearm 25cm further, hand 10cm
a = Vector((8, 0, 142))
b = a + Vector((30 * DOWN45, 0, -30 * DOWN45))
c = b + Vector((25 * DOWN45, 0, -25 * DOWN45))
d = c + Vector((10 * DOWN45, 0, -10 * DOWN45))
add_bone("mixamorig:LeftArm", a, b, "mixamorig:LeftShoulder")
add_bone("mixamorig:LeftForeArm", b, c, "mixamorig:LeftArm")
add_bone("mixamorig:LeftHand", c, d, "mixamorig:LeftForeArm")
bpy.ops.object.mode_set(mode='OBJECT')
ok("source skeleton built (A-pose, hips at Z=100, rolls 0.7)")

# animate: ONLY LeftArm rotates (children ride along => one rigid world delta
# for the whole arm), hips translate +X 30cm and dip 5cm
pb_arm = src.pose.bones["mixamorig:LeftArm"]
pb_hips = src.pose.bones["mixamorig:Hips"]
pb_arm.rotation_mode = 'XYZ'
scn = bpy.context.scene
scn.frame_set(1)
pb_arm.rotation_euler = (0, 0, 0)
pb_arm.keyframe_insert("rotation_euler", frame=1)
pb_hips.location = (0, 0, 0)
pb_hips.keyframe_insert("location", frame=1)
scn.frame_set(20)
pb_arm.rotation_euler = Euler((math.radians(50), math.radians(20), 0), 'XYZ')
pb_arm.keyframe_insert("rotation_euler", frame=20)
pb_hips.location = (30, 0, -5)          # bone-local == world-ish here; we read world later
pb_hips.keyframe_insert("location", frame=20)
ok("source animated: rigid LeftArm swing + hips translation, frames 1-20")

# ── retarget core (the algorithm under test) ────────────────────────────────
BONE_MAP = [
    # (source bone, target FK control) — parents before children
    ("mixamorig:LeftArm",     "upper_arm_fk.L"),
    ("mixamorig:LeftForeArm", "forearm_fk.L"),
    ("mixamorig:LeftHand",    "hand_fk.L"),
]
HIPS_SRC, HIPS_TGT = "mixamorig:Hips", "torso"

def rest_world_rot(obj, bname):
    return (obj.matrix_world @ obj.data.bones[bname].matrix_local).to_3x3()

def pose_world(obj, bname):
    return obj.matrix_world @ obj.pose.bones[bname].matrix

# hip-height ratio: target hips height / source hips height (world Z at rest)
src_hip_rest_w = (src.matrix_world @ src.data.bones[HIPS_SRC].matrix_local).translation
tgt_hip_rest_w = (rig.matrix_world @ rig.data.bones[HIPS_TGT].matrix_local).translation
scale_ratio = tgt_hip_rest_w.z / src_hip_rest_w.z
ok(f"hip-height scale ratio = {scale_ratio:.4f} (expect ~0.01)")

def retarget_frame(frame):
    scn.frame_set(frame)
    bpy.context.view_layer.update()
    # hips: rotation delta + translation delta scaled
    R_s_rest = rest_world_rot(src, HIPS_SRC)
    M_s = pose_world(src, HIPS_SRC)
    R_delta = M_s.to_3x3() @ R_s_rest.inverted()
    R_tgt = R_delta @ rest_world_rot(rig, HIPS_TGT)
    t_delta = (M_s.translation - src_hip_rest_w) * scale_ratio
    t_tgt = tgt_hip_rest_w + t_delta
    M = Matrix.LocRotScale(t_tgt, R_tgt.to_quaternion(), None)
    pbt = rig.pose.bones[HIPS_TGT]
    pbt.matrix = rig.matrix_world.inverted() @ M
    pbt.keyframe_insert("location", frame=frame)
    if pbt.rotation_mode == 'QUATERNION':
        pbt.keyframe_insert("rotation_quaternion", frame=frame)
    else:
        pbt.keyframe_insert("rotation_euler", frame=frame)
    bpy.context.view_layer.update()
    # limbs: rotation only, parents before children
    for s_name, t_name in BONE_MAP:
        R_delta = pose_world(src, s_name).to_3x3() @ rest_world_rot(src, s_name).inverted()
        R_tgt = R_delta @ rest_world_rot(rig, t_name)
        pbt = rig.pose.bones[t_name]
        cur = pose_world(rig, t_name)         # keep the chain-determined position
        M = Matrix.LocRotScale(cur.translation, R_tgt.to_quaternion(), None)
        pbt.matrix = rig.matrix_world.inverted() @ M
        if pbt.rotation_mode == 'QUATERNION':
            pbt.keyframe_insert("rotation_quaternion", frame=frame)
        else:
            pbt.keyframe_insert("rotation_euler", frame=frame)
        bpy.context.view_layer.update()

bpy.context.view_layer.objects.active = rig
for f in range(1, 21):
    retarget_frame(f)
ok("retargeted frames 1-20 onto torso + arm FK controls")

# ── verification via depsgraph WORLD POSITIONS ──────────────────────────────
def achieved_world(obj, bname):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    return ev.matrix_world @ ev.pose.bones[bname].matrix

# record target rest geometry (frame-independent)
sh_rest = (rig.matrix_world @ rig.data.bones["upper_arm_fk.L"].matrix_local).translation
wr_rest = (rig.matrix_world @ rig.data.bones["hand_fk.L"].matrix_local).translation

for frame in (1, 20):
    scn.frame_set(frame)
    bpy.context.view_layer.update()

    # the one rigid world delta of the source arm this frame
    R_delta = (pose_world(src, "mixamorig:LeftArm").to_3x3()
               @ rest_world_rot(src, "mixamorig:LeftArm").inverted())
    # hips translation the target should have received
    hips_t = ((pose_world(src, HIPS_SRC).translation - src_hip_rest_w) * scale_ratio)

    # 1) achieved orientation of the deform-side bone must carry the delta
    #    (ORG-upper_arm.L is what the DEF bones follow)
    R_achieved = achieved_world(rig, "ORG-upper_arm.L").to_3x3()
    R_expect = R_delta @ (rig.matrix_world
                          @ rig.data.bones["ORG-upper_arm.L"].matrix_local).to_3x3()
    ang = R_achieved.to_quaternion().rotation_difference(
        R_expect.to_quaternion()).angle
    print(f"  frame {frame}: ORG-upper_arm.L orientation error = "
          f"{math.degrees(ang):.4f} deg")
    if math.degrees(ang) > 0.5:
        fail(f"frame {frame}: upper arm orientation off by {math.degrees(ang):.2f} deg")

    # 2) wrist world position: rigid rotation of the arm about the shoulder,
    #    displaced by the hips translation — through the FULL Rigify stack
    sh_now = achieved_world(rig, "ORG-upper_arm.L").translation
    wr_now = achieved_world(rig, "ORG-hand.L").translation
    wr_expect = sh_now + R_delta @ (wr_rest - sh_rest)
    err = (wr_now - wr_expect).length
    print(f"  frame {frame}: wrist world-position error = {err:.6f} m "
          f"(arm span ~{(wr_rest - sh_rest).length:.3f} m)")
    if err > 2e-3:
        fail(f"frame {frame}: wrist off by {err:.4f} m")

    # 3) shoulder joint must have moved by the scaled hips translation only
    sh_expect = sh_rest + hips_t
    err_sh = (sh_now - sh_expect).length
    print(f"  frame {frame}: shoulder-follows-hips error = {err_sh:.6f} m "
          f"(hips moved {hips_t.length:.4f} m)")
    if err_sh > 2e-3:
        fail(f"frame {frame}: shoulder off by {err_sh:.4f} m — torso double-transform?")

scn.frame_set(20)
hips_t20 = ((pose_world(src, HIPS_SRC).translation - src_hip_rest_w) * scale_ratio)
if hips_t20.length < 0.25:
    fail("test not meaningful: hips barely moved")
ok(f"frame 20 hips delta arrived scaled: {tuple(round(v, 4) for v in hips_t20)} m "
   f"(source moved 30cm X, 5cm down at 100x scale)")

print("\nALL CHECKS PASSED — delta retarget is sound through the Rigify stack")
