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

## Spike results (verified headless, Blender 4.5, 2026-07-18)

`dev/spike_retarget.py`: real generated Rigify rig as target; fake Mixamo
skeleton as source — A-pose rest (target is T-pose), hips at Z=100 (cm scale,
ratio 0.0108), bone rolls 0.7, rigid arm swing + hips translation over 20
frames. Retargeted onto `torso` + arm FK chain with the delta math, verified
via depsgraph world positions. All three spike risks answered:

- **Delta math vs Rigify's stack: EXACT.** `ORG-upper_arm.L` achieved world
  orientation error 0.0000°; wrist world position error 0.000000 m at both
  keyed frames. Setting `pose_bone.matrix` (rotation part only, keep the
  chain's translation) then keying rotation passes through Rigify's FK
  mechanism losslessly. Note: the limb's IK_FK switch MUST be set to FK
  (arms/legs default to IK — `upper_arm_parent.L["IK_FK"] = 1.0`).
- **torso/hips double-transform: NONE.** Keying loc+rot on `torso` alone moved
  the shoulder by exactly the scaled hips delta (error 1e-6 m).
- **Performance: fine at spike scale.** One `view_layer.update()` per bone per
  frame (needed so children see fresh parent transforms). Extrapolates to
  ~15k updates for 30 controls x 500 frames — likely tens of seconds on a
  700-bone rig; batch/optimize later if real clips crawl.

Two implementation notes: (1) source hips `location` keys are BONE-LOCAL —
always read world translation via `matrix_world @ pose.bone.matrix`, never
trust the fcurve values; magnitude and direction both survived only because
the spike did. (2) Translation must be delta-from-rest scaled, then added to
the TARGET's rest position (absolute scaled positions would sink the character
by the hip-height difference).

## Implementation status (retarget.py, shipped 2026-07-18)

v1 shipped: `retarget.py` — `build_mapping` (Mixamo preset incl. fingers with
prefix-stripping and candidate target names; fuzzy synonym fallback for
unknown rigs, fingers preset-only; parents-first ordering by target depth) +
`run_retarget` (delta bake as a NEW action, previous action kept with a fake
user; IK/FK switches set AND keyed to FK; one view-layer update per depth
LEVEL per frame instead of per bone — siblings are independent, ~6x fewer
updates) + minimal Tools-tab UI (source picker with poll excluding generated
rigs, clip/frame readout with Mixamo-vs-fuzzy indicator, In Place toggle
stripping hips XY, Retarget button).

Verified headless (Blender 4.5, `dev/test_retarget.py`): 13 preset pairs on a
Mixamo-named A-pose/cm-scale source, wrist world-position error 0.000000 m
through the full Rigify stack, previous rig action preserved, 4 IK/FK
switches keyed; fuzzy path resolves a UE-named skeleton (10 pairs, both
sides). Addon register/unregister/re-register smoke-tested.

### Real-clip field fix (2026-07-18): Match Clip Pose + facing correction

First real Mixamo clip test put the hands BEHIND the character. Two causes,
both fixed:

1. **Rest-pose offset baked into deltas.** Pure delta semantics preserve
   offsets from the TARGET's rest — a T-pose clip on a character rigged in a
   different rest over-rotates every limb by the rest difference. Fix =
   the designed "rest align", now shipped as **Match Clip Pose (default ON)**:
   each control's rest base is pre-rotated so its bone direction equals the
   clip skeleton's rest direction (`d_tgt.rotation_difference(C @ d_src)`),
   so the clip's ACTUAL limb poses are reproduced. OFF restores pure-delta.
2. **Facing mismatch.** World deltas are direction-dependent — a source
   skeleton facing the other way swings arms toward the character's back.
   Fix = auto facing correction `C`: yaw estimated per rig from the L/R
   upper-arm (fallback thigh) rest positions (`(L-R) x Z`), deltas conjugated
   `C @ D @ C^-1`, hips translation rotated by `C`.

Verified (`dev/test_retarget.py`): A-pose cm source on T-pose rig, wrist
lands exactly where the clip's limb directions point (0.000000 m); source
rotated 180 deg auto-detects and corrects (0.000000 m); align OFF still
matches pure-delta semantics. Lesson: the spike proved the MATH exact but the
SEMANTICS wrong for the main use case — real-clip testing caught what
synthetic identical-facing tests could not.

### Second field round (2026-07-18): floor sink + turning snap

Match Clip Pose introduced two regressions the user caught on the same clip:

1. **Character sank, feet below the floor.** Root cause (found by dumping the
   chain's rest-vs-achieved heights): Rigify's `torso` control bone points
   HORIZONTALLY (+Y) by widget convention — it's an abstract pivot, not
   anatomy. Rest-aligning it to Mixamo Hips' up-vector pitched it ~90 deg,
   swinging the pelvis assembly down ~0.24 m around the pivot head while
   child-bone alignments masked the rotation. Fix: the location-carrying pair
   NEVER gets rest-aligned (delta-only rotation); only bones lying along an
   actual body part (spine, limbs, fingers) align. Plus a floor calibration:
   clip-matching straightens legs the character rigged with a knee bend, so
   the ankle's clip-matched rest height is compared to the character's rest
   ankle and every hips key is lifted by the difference (z_off, ~4 mm on the
   metarig; the torso pivot bug was the 0.24 m).
2. **Foot/shin snapping while the character turns.** Quaternion double-cover:
   each frame's quaternion is computed independently; when a turn crosses the
   sign boundary, adjacent keys interpolate the long way. Fix: per-bone
   sign-continuity with the previous frame (negate when dot < 0).

Test now covers both: ankle-at-rest-height assertion (err 0.0002 m) and a
240-deg hips turn with a no-sign-flip sweep over every keyed quaternion.
Lesson recorded: Rigify CONTROL bones split into anatomical (safe to align)
and widget-convention pivots (torso/hips/chest point horizontally — align
only their deltas). Verify chain HEIGHTS, not just directions.

### Third field round (2026-07-18): shin twist on both legs

After the floor fix, BOTH shins showed visible twist (previously only the
right). Mechanism: Rigify's shin twist bones derive their roll from the FOOT
orientation, and Match Clip Pose was forcing each foot to the clip skeleton's
absolute world orientation INCLUDING ITS HEADING (yaw). Any rest heading
difference between the clip's feet and the character's (toe-out stance etc.)
gets crammed into the ankle and renders as the whole shin twisting — on both
legs, once both feet were clip-aligned.

Fix: feet/toes (`_HEADING_PRESERVE`) rest-align PITCH ONLY — the source rest
direction is first yawed about world Z so its horizontal heading matches the
character's own (`_match_heading`) before the alignment rotation is computed.
Floor contact needs the clip's pitch; heading belongs to the character. Test:
source feet splayed 35 deg outward -> retargeted foot keeps the character's
0-deg heading with the clip's pitch (errors 0.0 deg / 0.0000).

Watch item: hands drive forearm twist the same way in Rigify. Hand alignment
stays absolute (hand orientation usually matters more than wrist twist), but
if users report forearm twist, the same heading/twist-limiting treatment is
the candidate.

### Fourth field round (2026-07-18): IK controllers now baked too

User reported the IK controllers not moving with the animation, and shin
twist persisting only during turns even after weight smoothing. Keying only
FK + flipping the IK_FK switches leaves the IK controllers PARKED AT REST —
if a limb is in IK (switch silently ineffective on some rig, or the user
flips a limb back), the limb stays glued to the parked controller and winds
up visibly as the body turns over it. Exactly the reported symptom.

Fix: `_IK_SNAP` — every frame, hand_ik/foot_ik are snapped to their FK
twin's evaluated matrix and keyed (with the same quaternion continuity).
The clip is now correct in EITHER mode, the IK controllers visibly follow,
and switching a limb to IK afterward needs no snapping step (this also
delivers the roadmap "IK foot bake" in its basic form). Test: IK controls
track FK to 1e-6 m mid-turn, and forcing the leg to IK moves the deform
foot 0.000000 m.

Diagnostic history for this round: keyed-transfer fidelity was proven exact
first (world-delta orientation error 0.000 deg at every turn frame,
foot-vs-shin relative twist within 0.07 deg of the source), and the user
confirmed bones-fine/mesh-twists before the IK observation surfaced.

### Final foot semantics (2026-07-18, user-found root cause)

User isolated it: the retargeted foot_fk was pitched UP (toes off the floor),
and that constant foot rotation is what Rigify's foot->shin coupling renders
as shin twist — clearing the foot rotation killed the twist. Root cause: the
clip skeleton's foot rest PITCH is correct for ITS ankle height / foot shape,
not the character's; aligning to it (even heading-preserved) tilts the sole.

Feet/toes are now `_DELTA_ONLY` — like the torso pivot they keep the
character's OWN rest entirely and receive only the clip's rotation deltas
(heel strike / toe-off / turns still come through). The earlier
heading-preserving alignment is removed. Test: at the clip's neutral frame
the foot sits 0.000 deg from the character's rest despite source feet
splayed 35 deg at a different pitch.

Bone-category summary that emerged from all six rounds:
- location carrier (torso)      -> delta-only (widget-convention pivot)
- feet/toes                     -> delta-only (floor contact = character rest)
- everything anatomical else    -> rest-aligned to the clip (Match Clip Pose)
- visible IK controls           -> delta-snapped to their FK twin per frame

### IK pole-vector keying (shipped 2026-07-18)

`_POLE_SNAP`: every frame the pole targets (`thigh_ik_target`,
`upper_arm_ik_target`) are placed at the knee/elbow pushed along the FK
chain's bend direction (`knee - midpoint(hip, ankle)`, half-limb-length out;
straight-limb fallback = previous frame's direction, else the mid bone's -Z).
Location-only keys. Verified: pole_vector ON + IK mode reproduces the clip
with 0.00000 m / 0.000 deg error on the whole leg chain. Note:
`pole_vector` is a Boolean IDProperty — assign True/False, not 1/0.

Still open from the design: batch retarget (Team tier).

## Effort estimate

- Delta-bake core + Mixamo preset + minimal UI (picker, auto-map, button):
  the bulk, well-bounded — comparable to the game-export merge work.
- Mapping UIList + JSON save/load: meaningful additional UI work.
- IK foot bake + batch: separable follow-ups.

## Tier fit (DECIDED 2026-07-18)

Tiers are now: Lite / **Full** / **Team (5-seat)** — "Studio" renamed.
Retargeting ships in the **Full edition** (gated out of Lite via
`LITE_BUILD`, same convention as the AI buttons: section absent, operator
guarded). Game export ships in BOTH editions including Lite. Batch retarget
remains the Team-tier candidate when built.
