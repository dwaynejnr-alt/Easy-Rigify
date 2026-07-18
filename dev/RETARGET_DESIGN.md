# Animation Retarget Engine — Design Note

Apply external animations (Mixamo, mocap libraries, other skeletons) ONTO the
generated Rigify rig. Phase 2 of the retargeting work — phase 1 (game export +
animation bake) shipped, see GAME_EXPORT_DESIGN.md. **Design only — not yet
implemented.**

## The problem

Users buy/download animation clips (Mixamo FBX, mocap packs, clips authored on
other characters) and want them on the character this addon just rigged. The
source skeleton never matches ours:

1. **Different bone names** — `mixamorig:LeftForeArm` vs our `forearm_fk.L`.
2. **Different rest pose** — source may be T-pose, our character may be A-pose
   (or any pose the user modelled in). Copying rotations raw bends every limb
   by the rest-pose difference.
3. **Different bone axes/rolls** — even matching world orientations, each
   bone's local axes differ, so local-space rotation copy is garbage.
4. **Different scale** — Mixamo rigs are cm-scale (~180 units tall); root
   motion copied raw teleports the character.
5. **Control vs deform** — animation must land on Rigify's CONTROL bones
   (FK chains, torso, hips, root) so the user can still edit it, not on
   DEF-/ORG- bones where it would fight the control rig.

## Core algorithm — world-space rotation deltas

The standard retarget that is immune to rest-pose and roll mismatches:

For each mapped bone pair, per frame:

```
R_delta          = R_src_pose_world @ R_src_rest_world⁻¹     # how far the source
                                                             # bone moved from rest
R_tgt_pose_world = R_delta @ R_tgt_rest_world                # apply same world
                                                             # delta to the target
```

- Rest-pose mismatch: handled — deltas are relative to each rig's own rest.
  A T-pose clip on an A-pose character produces the A-pose character doing the
  same motion (arms swing from ITS rest). Optional "rest align" correction
  (rotate source rest to match target rest before computing deltas) if the user
  wants literal pose reproduction instead — v2.
- Roll mismatch: handled — everything is world-space.
- **Location**: only the hips/root pair copies translation, scaled by
  `target_hip_height / source_hip_height` (world Z of the hip bone head at
  rest). Everything else is rotation-only — bone lengths differ, so copying
  child translations would dislocate joints.

Implementation is a bake, not live constraints: step frames, compute matrices
via the depsgraph, write keyframes on the target controls. Live-constraint
setups (what the game-export bake uses) don't fit here because the delta math
isn't expressible as a stock constraint stack.

## Target controls (what we key on the Rigify rig)

| Region       | Rigify control                              | Channels        |
|--------------|---------------------------------------------|-----------------|
| Root         | `root`                                      | loc + rot (opt) |
| Hips/pelvis  | `torso` (+ `hips` for pelvis-only rotation) | loc + rot       |
| Spine        | `spine_fk.001..003` / `chest`               | rot             |
| Neck / head  | `neck`, `head`                              | rot             |
| Arms         | `shoulder.L/R`, `upper_arm_fk.L/R`, `forearm_fk.L/R`, `hand_fk.L/R` | rot |
| Legs         | `thigh_fk.L/R`, `shin_fk.L/R`, `foot_fk.L/R`, `toe_fk.L/R`          | rot |
| Fingers      | `f_index.01.L` … (Rigify finger controls)   | rot             |

- Retarget lands on **FK**; limbs get their IK/FK switch keyed to FK for the
  clip range so the result is visible immediately. Rigify's own IK↔FK snapping
  remains available per-limb afterward.
- IK feet (planted-foot quality) = refinement, not v1 (see below).

## Bone-map system

A mapping = ordered list of (source bone → target control). Three sources:

1. **Auto-detect presets** shipped in code, keyed off recognizable source
   names:
   - **Mixamo** (`mixamorig:Hips`, `mixamorig:LeftForeArm`, …) — the big one;
     also covers most "game-ready" store characters, which reuse this scheme.
   - **UE Mannequin** (`pelvis`, `upperarm_l`, …) — reuses `_UE_LIMB` /
     `_UE_FINGER` from game_export.py in reverse. Free symmetry: skeletons WE
     exported retarget straight back.
   - **Unity-style stripped names** (our own Unity export output).
   - Detection = score each preset by how many of its source names exist in
     the picked armature; best score wins, user can override.
2. **Fuzzy fallback** for unknown rigs: normalize names (case, side tokens
   L/R/Left/Right/_l/.L, separators) and match against target synonyms
   (`forearm|lowerarm|elbow`, `shin|calf|leg1`, …). Fills what it can; user
   fixes the rest.
3. **Manual editing** — UIList of rows (source bone picker, target control
   picker), add/remove/clear, and save/load mapping as JSON so studio users
   can reuse a mapping across a library of clips (same source skeleton).

## Pipeline

1. **Import** — user imports the source FBX/BVH themselves (Blender handles
   formats); our input is "an armature in the scene with an action". Keeps us
   out of the FBX-import business.
2. **Pick source** — armature picker; auto-runs preset detection + mapping.
3. **Review mapping** — UIList; unmapped required bones (hips, limbs) flagged.
4. **Retarget (bake)** — for the action's frame range (or user range):
   compute deltas, key FK controls + torso/root; key IK/FK switches to FK.
   Result = a NEW action on the Rigify rig, named after the source clip; the
   rig's previous action is preserved (users A/B them in the Action editor).
5. **Cleanup options** — delete/keep source armature, frame-range trim,
   "in-place" toggle (strip root XY translation for game loops).

## Refinements (post-v1)

- **IK foot bake** — after FK retarget, snap IK foot targets to the FK result
  per frame (Rigify ships the snap operator; drive it per-frame) so feet can
  be polished with IK. Floor-contact correction on top of that later.
- **Rest-align option** — pre-rotate source rest to target rest (per-bone,
  computed once) for users who want literal limb angles, not deltas.
- **Batch retarget** — run one mapping across a folder of FBX clips,
  producing one action each (pairs with batch game export). Studio tier.
- **In-place / root-motion extraction** toggle refinement per-axis.

## Foundation that already exists

- `_UE_LIMB` / `_UE_FINGER` name tables + spine positional naming
  (game_export.py) — invert for source-name presets.
- Bake plumbing + frame stepping + "verify world positions, not names"
  test discipline (dev/test_anim_bake.py is the template for the spike).
- `_is_generated_rig` / control-bone knowledge in pipeline.py for validating
  the target and enumerating FK controls.

## Spike plan (before real implementation)

Headless, like the game-export spike:

1. Generate a Rigify rig; build a second armature with Mixamo names, DIFFERENT
   rest pose (T vs A), different scale, animate its arm + hips.
2. Retarget with the delta math onto FK controls.
3. **Verify world positions, not names or channel counts** (the three-bug
   lesson from game export): target hand world trajectory ≈ source hand
   trajectory shape (normalized for scale/limb length); hips translation
   scaled correctly; no limb folded by a rest-pose delta.

Risks the spike must answer:
- Delta math vs Rigify's layered constraints — keys on FK controls pass
  through Rigify's own mechanism; confirm the world result matches the math
  (the control's world matrix isn't always its keyed matrix_basis verbatim).
- `torso`/`hips` interplay — which combination reproduces pelvis motion
  without double-transform.
- Performance — pure-Python per-frame matrix math over ~30 controls × ~500
  frames; fine in principle, confirm.

## Effort estimate

- Delta-bake core + Mixamo preset + minimal UI (picker, auto-map, button):
  the bulk, well-bounded — comparable to the game-export merge work.
- Mapping UIList + JSON save/load: meaningful additional UI work.
- IK foot bake + batch: separable follow-ups.

## Tier fit

Per GAME_EXPORT_DESIGN.md: retargeting is production-oriented → **Studio
tier**. Mixamo-preset retarget could arguably sit in Full as a taste of it,
with mapping editor + batch reserved for Studio — decide at ship time.
