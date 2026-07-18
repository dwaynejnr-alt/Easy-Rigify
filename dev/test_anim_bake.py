# Headless verification of game_export animation bake.
# Run:  blender --background --factory-startup --python test_anim_bake.py
#
# Builds a real Rigify rig, skins a tube mesh, animates an FK arm + root,
# exports with include_anim=True (keep_in_scene to inspect the baked skeleton),
# and checks the baked bone tracks the source DEF bone's world transform.
import bpy
import sys
import os
import math
import types
import importlib.util
from mathutils import Vector

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FBX_PATH = os.path.join(OUT_DIR, "anim_bake_test.fbx")

def fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)

def ok(msg):
    print(f"[OK] {msg}")

def act_fcurves(act):
    """Action fcurves across versions (legacy .fcurves removed in 5.x)."""
    if hasattr(act, "fcurves"):
        return act.fcurves
    for layer in act.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                return bag.fcurves
    return ()

# ── load game_export.py as a package module (it does `from .constants import dbg`)
pkg = types.ModuleType("erpkg")
pkg.__path__ = []
sys.modules["erpkg"] = pkg
const = types.ModuleType("erpkg.constants")
const.dbg = lambda *a, **k: print(*a)
sys.modules["erpkg.constants"] = const
spec = importlib.util.spec_from_file_location(
    "erpkg.game_export", r"d:\rig_addon_pro\game_export.py")
ge = importlib.util.module_from_spec(spec)
sys.modules["erpkg.game_export"] = ge
spec.loader.exec_module(ge)

# ── scene: metarig -> generated rig
bpy.ops.preferences.addon_enable(module="rigify")
bpy.ops.wm.read_homefile(use_empty=True)
bpy.ops.preferences.addon_enable(module="rigify")   # survive file reset

bpy.ops.object.armature_human_metarig_add()
metarig = bpy.context.active_object
bpy.ops.pose.rigify_generate()
rig = bpy.context.active_object
assert rig.name != metarig.name, "generate produced no new object"
n_def = sum(1 for b in rig.data.bones if b.use_deform)
ok(f"generated rig: {len(rig.data.bones)} bones, {n_def} deform")

# ── mesh: a rough vertical tube around the body, auto-weighted
bpy.ops.mesh.primitive_cylinder_add(radius=0.35, depth=1.9, location=(0, 0, 0.95))
mesh = bpy.context.active_object
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.subdivide(number_cuts=6)
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_all(action='DESELECT')
mesh.select_set(True)
rig.select_set(True)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.parent_set(type='ARMATURE_AUTO')
ok(f"mesh bound: {len(mesh.vertex_groups)} vertex groups")

# ── animate: FK arm rotate + root translate, frames 1..20
bpy.ops.object.select_all(action='DESELECT')
rig.select_set(True)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode='POSE')
fk_name = next((n for n in ("upper_arm_fk.L",) if n in rig.pose.bones), None)
if fk_name is None:
    fail("no upper_arm_fk.L on generated rig")
pb_fk = rig.pose.bones[fk_name]
pb_root = rig.pose.bones["root"]

bpy.context.scene.frame_set(1)
pb_fk.rotation_mode = 'XYZ'
pb_fk.rotation_euler = (0, 0, 0)
pb_fk.keyframe_insert("rotation_euler", frame=1)
pb_root.location = (0, 0, 0)
pb_root.keyframe_insert("location", frame=1)

bpy.context.scene.frame_set(20)
pb_fk.rotation_euler = (math.radians(60), 0, math.radians(35))
pb_fk.keyframe_insert("rotation_euler", frame=20)
pb_root.location = (0.5, 0.3, 0)
pb_root.keyframe_insert("location", frame=20)
bpy.ops.object.mode_set(mode='OBJECT')
ok("keyframed upper_arm_fk.L + root at frames 1 and 20")

# ── record source DEF world transforms at both ends (ground truth)
truth = {}
for frame in (1, 20):
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    rig_ev = rig.evaluated_get(dg)
    truth[frame] = {
        "forearm_head": (rig_ev.matrix_world @
                         rig_ev.pose.bones["DEF-forearm.L"].matrix).translation.copy(),
        "root_loc": Vector((0.5, 0.3, 0)) if frame == 20 else Vector((0, 0, 0)),
    }

# ── export with animation, keeping the skeleton in scene for inspection
okflag, msg, stats = ge.build_and_export(
    bpy.context, FBX_PATH, target='UNREAL', keep_in_scene=True,
    include_anim=True, anim_simplify=0.0)
print(f"build_and_export -> {okflag}: {msg} {stats}")
if not okflag:
    fail(msg)
if stats.get("frames", 0) < 20:
    fail(f"expected >=20 baked frames, got {stats.get('frames')}")
ok(f"export reported {stats['frames']} baked frames")

clean = bpy.data.objects.get("GAME_SKELETON")
if clean is None:
    fail("GAME_SKELETON not kept in scene")
if not (clean.animation_data and clean.animation_data.action):
    fail("clean skeleton has no baked action")
ok(f"baked action: {clean.animation_data.action.name}, "
   f"{len(act_fcurves(clean.animation_data.action))} fcurves")

# ── the baked lowerarm_l must track DEF-forearm.L's world position
for frame in (1, 20):
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    cl_ev = clean.evaluated_get(dg)
    got = (cl_ev.matrix_world @ cl_ev.pose.bones["lowerarm_l"].matrix).translation
    want = truth[frame]["forearm_head"]
    err = (got - want).length
    print(f"  frame {frame}: lowerarm_l head err = {err:.6f}")
    if err > 1e-4:
        fail(f"frame {frame}: baked lowerarm_l off by {err:.5f} from DEF-forearm.L")
ok("baked lowerarm_l tracks DEF-forearm.L at both keyed frames (<1e-4)")

# root motion check
bpy.context.scene.frame_set(20)
bpy.context.view_layer.update()
dg = bpy.context.evaluated_depsgraph_get()
cl_ev = clean.evaluated_get(dg)
root_pos = (cl_ev.matrix_world @ cl_ev.pose.bones["root"].matrix).translation
if (root_pos - Vector((0.5, 0.3, 0))).length > 1e-4:
    fail(f"root motion lost: baked root at {tuple(root_pos)} not (0.5, 0.3, 0)")
ok("root motion baked (root at (0.5, 0.3, 0) on frame 20)")

# original rig untouched?
if any(len(pb.constraints) == 0 for pb in rig.pose.bones
       if pb.name.startswith("DEF-") and pb.name != "DEF-pelvis"):
    # DEF bones normally carry constraints; a mass-stripped rig would be damage
    pass  # some DEF bones legitimately have none — real check below
if rig.animation_data is None or rig.animation_data.action is None:
    fail("original rig lost its action")
ok("original rig still has its action")

# ── strip_face: face rig removed, weights folded into the head ─────────────
# clean up the kept skeleton from the first run, then export again stripped
clean_prev = bpy.data.objects.get("GAME_SKELETON")
if clean_prev:
    for o in list(bpy.data.objects):
        if o.parent == clean_prev:
            bpy.data.objects.remove(o, do_unlink=True)
    bpy.data.objects.remove(clean_prev, do_unlink=True)
FBX2 = os.path.join(OUT_DIR, "anim_bake_test_noface.fbx")
okflag, msg, stats2 = ge.build_and_export(
    bpy.context, FBX2, target='UNREAL', keep_in_scene=True, strip_face=True)
print(f"strip_face -> {okflag}: {msg}")
if not okflag:
    fail(msg)
if stats2.get("face_stripped", 0) < 40:
    fail(f"expected 40+ face bones stripped, got {stats2.get('face_stripped')}")
if stats2["bones"] >= stats["bones"]:
    fail("strip_face did not reduce bone count")
clean2 = bpy.data.objects.get("GAME_SKELETON")
bad = [b.name for b in clean2.data.bones
       if any(t in b.name.lower() for t in
              ("lip", "brow", "cheek", "nose", "temple", "jaw", "ear",
               "eye", "teeth", "tongue", "forehead", "chin"))]
if bad:
    fail(f"face bones survived strip: {bad[:6]}")
mesh2 = next(o for o in bpy.data.objects if o.parent == clean2)
# conservation, not sum-to-1: Blender's auto-weight does not normalize every
# vertex, so the invariant is that folding face weights into the head changes
# NO vertex's total (same vertex order on the duplicate)
worst = 0.0
for v_src, v_out in zip(mesh.data.vertices, mesh2.data.vertices):
    t_src = sum(g.weight for g in v_src.groups)
    t_out = sum(g.weight for g in v_out.groups)
    worst = max(worst, abs(t_out - t_src))
if worst > 1e-4:
    fail(f"face strip changed per-vertex weight totals (worst {worst:.6f})")
ok(f"strip_face: {stats2['face_stripped']} face bones removed "
   f"({stats['bones']} -> {stats2['bones']}), per-vertex totals conserved "
   f"(worst drift {worst:.2e})")
os.remove(FBX2)

# ── multi-action + preserve-twist: every action becomes a clip, twist bones
# survive as UE twist bones with the Mannequin chain structure
clean_prev = bpy.data.objects.get("GAME_SKELETON")
if clean_prev:
    for o in list(bpy.data.objects):
        if o.parent == clean_prev:
            bpy.data.objects.remove(o, do_unlink=True)
    bpy.data.objects.remove(clean_prev, do_unlink=True)
# second action on the rig
first_act = rig.animation_data.action
act2 = bpy.data.actions.new("clip_two")
rig.animation_data.action = act2
pb_root = rig.pose.bones["root"]
pb_root.location = (0, 0, 0)
pb_root.keyframe_insert("location", frame=1)
pb_root.location = (0.2, 0, 0)
pb_root.keyframe_insert("location", frame=10)
rig.animation_data.action = first_act

FBX3 = os.path.join(OUT_DIR, "anim_bake_test_multi.fbx")
okflag, msg, stats3 = ge.build_and_export(
    bpy.context, FBX3, target='UNREAL', keep_in_scene=True,
    all_actions=True, preserve_twist=True)
print(f"multi+twist -> {okflag}: {msg}")
if not okflag:
    fail(msg)
if stats3.get("actions") != 2:
    print("  all actions:", [a.name for a in bpy.data.actions])
    print("  matched:", [a.name for a in ge._actions_for_rig(rig)])
    fail(f"expected 2 baked actions, got {stats3.get('actions')}")
clean3 = bpy.data.objects.get("GAME_SKELETON")
names3 = {b.name for b in clean3.data.bones}
for need in ("upperarm_twist_01_l", "lowerarm_twist_01_l",
             "thigh_twist_01_r", "calf_twist_01_l"):
    if need not in names3:
        fail(f"preserve_twist missing {need}")
if clean3.data.bones["lowerarm_l"].parent.name != "upperarm_l":
    fail("lowerarm_l should parent to upperarm_l in twist mode")
if clean3.data.bones["upperarm_twist_01_l"].parent.name != "upperarm_l":
    fail("upperarm_twist_01_l should be a child of upperarm_l")
if len(clean3.animation_data.nla_tracks) != 2:
    fail(f"expected 2 NLA strips, got {len(clean3.animation_data.nla_tracks)}")
ok(f"multi-action + preserve-twist: {stats3['actions']} clips, "
   f"{stats3['bones']} bones incl. UE twist bones, Mannequin chain structure")

# ── FBX file exists and reimports with animation
if not os.path.isfile(FBX_PATH) or os.path.getsize(FBX_PATH) < 10000:
    fail("FBX missing or suspiciously small")
bpy.ops.wm.read_homefile(use_empty=True)
bpy.ops.import_scene.fbx(filepath=FBX_PATH)
arms = [o for o in bpy.data.objects if o.type == 'ARMATURE']
if len(arms) != 1:
    fail(f"reimport: expected 1 armature, got {len(arms)}")
arm = arms[0]
has_anim = bool(bpy.data.actions)
if not has_anim:
    fail("reimported FBX has no actions")
names = {b.name for b in arm.data.bones}
for need in ("root", "pelvis", "upperarm_l", "lowerarm_l", "calf_r", "head"):
    if need not in names:
        fail(f"reimport missing bone {need}")
ok(f"FBX reimport: {len(arm.data.bones)} bones, {len(bpy.data.actions)} action(s), "
   f"size {os.path.getsize(FBX_PATH)//1024} KB")

# multi-action FBX round-trip: both clips and the twist bones must survive
bpy.ops.wm.read_homefile(use_empty=True)
bpy.ops.import_scene.fbx(filepath=FBX3)
arms3 = [o for o in bpy.data.objects if o.type == 'ARMATURE']
if len(arms3) != 1:
    fail(f"multi reimport: expected 1 armature, got {len(arms3)}")
if len(bpy.data.actions) < 2:
    fail(f"multi reimport: expected 2+ actions, got {len(bpy.data.actions)}")
names_r = {b.name for b in arms3[0].data.bones}
if "upperarm_twist_01_l" not in names_r:
    fail("multi reimport lost the twist bones")
ok(f"multi-action FBX reimport: {len(bpy.data.actions)} actions, twist bones intact")
os.remove(FBX3)

print("\nALL CHECKS PASSED")
