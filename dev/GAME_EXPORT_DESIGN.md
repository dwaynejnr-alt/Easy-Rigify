# Game Export Engine — Design Note

Export a generated Rigify rig + skinned mesh to a clean, game-ready skeleton
for **Unreal** and **Unity**. Design only — not yet implemented.

## The problem

Rigify generates a **control rig** (IK, mechanism, control bones) that engines
cannot use. The usable deformation skeleton is Rigify's **DEF- bones** — the
addon already keys off these (`_is_generated_rig`, `_REGION_BONES`,
`_set_def_bones_visible`). But two structural mismatches remain:

1. **Segmented limbs.** Rigify splits each limb into two deform bones for smooth
   twist (confirmed in `pipeline.py:_REGION_BONES`):
   - `DEF-upper_arm.L` + `DEF-upper_arm.L.001`
   - `DEF-forearm.L`   + `DEF-forearm.L.001`
   - `DEF-thigh.L`     + `DEF-thigh.L.001`
   - `DEF-shin.L`      + `DEF-shin.L.001`
   - `DEF-neck`        + `DEF-neck.001`
   - `DEF-spine` … `DEF-spine.006` (many)

   Engine skeletons expect one bone per upper arm / forearm / thigh / calf, and
   (for Unity Humanoid) one neck.

2. **Naming + hierarchy.** UE Mannequin and Unity Humanoid expect specific bone
   names and a single root at world origin.

## Fix for the segmented limbs — merge with weight transfer

Per limb, deterministic and lossless for weights:

1. Build one bone from `head(seg)` to `tail(seg.001)`. Collinear in A/T-pose, so
   the merged bone has the correct single-bone length.
2. `weight(merged) = weight(seg) + weight(seg.001)` per vertex. Rigify's two
   segments partition influence along the limb, so the sum reconstructs the full
   limb. Per-vertex totals were already 1.0, so **no re-normalization needed**.
3. Re-parent whatever hung off `seg.001` onto the merged bone. Joint positions
   are unchanged (elbow/knee = merged bone's tail).

Spine (many→few): merge adjacent `DEF-spine.*` segments down to the target
count (e.g. 6 → 3 for UE spine_01/02/03 by grouping 2:2:2).

### Two modes

- **Merge (default).** One rigid bone per limb. Universal, simplest, good enough
  for most game characters. Loses smooth twist.
- **Preserve twist (UE-faithful).** Map `.001` onto the engine's twist bone
  (`lowerarm_twist_01`, `upperarm_twist_01`, `thigh_twist_01`, `calf_twist_01`).
  Keep both neck segments for UE5 (`neck_01` / `neck_02`); merge neck for Unity
  Humanoid (single `Neck`). Preserves deformation quality.

## Pipeline

1. **Validate** — confirm a generated Rigify rig (DEF- bones present) and a mesh
   bound to it.
2. **Duplicate → clean skeleton** — copy the armature, keep only DEF- bones,
   strip the `DEF-` prefix.
3. **Merge segments** — per the mode above; rewrite the mesh's vertex groups to
   match, re-parent chains.
4. **Root** — add/ensure a single `root` bone at world origin; parent the pelvis
   under it.
5. **Rename** — apply the target's bone-name map (see below).
6. **Bake** — if exporting animation, bake actions onto the clean skeleton
   (visual keying through the removed control rig).
7. **FBX export** — with the target's settings matrix (below).

## Target specifics

### Unity
- Humanoid avatar auto-maps a sane skeleton; Generic accepts most hierarchies.
- Merge neck to one bone for Humanoid.
- FBX: apply unit scale, Y-up, no leaf bones, `!use_armature_deform_only` off
  (only DEF bones remain anyway).

### Unreal
- Match the **UE5 Mannequin** bone names for out-of-the-box retarget
  (`pelvis`, `spine_01..`, `clavicle_l`, `upperarm_l`, `lowerarm_l`, `hand_l`,
  `thigh_l`, `calf_l`, `foot_l`, `neck_01`, `head`, …), plus twist bones in
  preserve-twist mode.
- FBX: **cm** (Unreal expects centimetres — Blender is m), Z-up→Y-up handled by
  the FBX exporter, `root` bone at origin, primary/secondary bone axes set so
  bones don't import twisted.

## The FBX-settings matrix is the real work

A naive `export_scene.fbx()` will not produce game-correct results. The bone
axis mapping, unit scale (m vs cm), leaf-bone toggle, and root handling are the
substance of this feature, not a wrapper around the exporter.

## Foundation that already exists

- DEF- bone identification and per-region enumeration (`_REGION_BONES`).
- Generated-rig detection (`_is_generated_rig`).
- Weight read/write plumbing (`WeightDataAccess`, `_write_weights_back`) —
  reused for the vertex-group merge.

## Effort estimate

- Merge + clean-skeleton extraction + Unity export (Generic/Humanoid): the bulk,
  well-bounded.
- UE Mannequin rename + twist mode: meaningful additional work.
- Animation bake: separable; ship static-skeleton export first, animation next.

## Spike results (verified on A-pose.blend, Blender 4.5)

Headless: generate Rigify rig → auto-weight the body mesh → extract DEF-only
skeleton → merge segments → check weights. Proven:

- **Generation works headless.** Generated rig = 411 total bones, **73 DEF-**.
- **The DEF- bones form their own intact parent chain** (DEF parents to DEF):
  `DEF-forearm.L -> DEF-upper_arm.L.001 -> DEF-upper_arm.L`,
  `DEF-shin.L -> DEF-thigh.L.001`. So deleting the non-DEF control bones does
  NOT orphan the deform skeleton — the limb hierarchy survives extraction. Only
  the root/pelvis attachment needs attention, not a full hierarchy rebuild.
- **Merge is lossless.** 8 limb segments merged (73 -> 65 bones); max per-vertex
  total-weight drift after merging + summing groups = **4.5e-08** (float noise).
  Confirms weights are preserved with no re-normalization. PASS.
- **Neck is metarig-dependent.** This character had no `DEF-neck` (only a single
  neck deform), unlike the always-two arm/leg segments. Neck merge must be
  defensive: detect the segments present rather than assume `DEF-neck` +
  `.001`. Arm/leg segmentation is reliably two.

Implementation notes for the real thing: iterate `vertex.groups` rather than
`vertex_group.weight(idx)` (the latter floods stderr with "Vertex not in group"
on misses); reconstruct only the root/pelvis link.

## Implementation status (game_export.py)

Shipped: `autorig.export_game` operator + Tools-tab section, **Unity and Unreal**
targets, "Apply Modifiers" toggle (default off — Subsurf would multiply polys).
Verified end-to-end on a real generated rig with FBX round-trip: 66-bone
single-root skeletons, originals untouched, temp skeleton cleaned up.

Output bone names:
- Unity: `pelvis, spine_01.., neck, head, upper_arm.L, forearm.L, thigh.L, ...`
- Unreal: `pelvis, spine_01.., neck_01, head, upperarm_l, lowerarm_l, calf_l,
  clavicle_l, thigh_l, foot_l, index_01_l, ...` (UE5 Mannequin).

### Key discovery — Rigify has NO DEF-neck / DEF-head

Confirmed on the *standard* Rigify human metarig, not just the test scene:
Rigify deforms the neck and head via the **top two segments of the DEF-spine
chain** (`DEF-spine.005` = neck, `DEF-spine.006` = head). The `neck`/`head`
bones exist but are control-only (`use_deform=False`). Consequences baked into
the code:

- **Extract by `use_deform`, not by the `DEF-` name prefix** — otherwise bones
  like `neck`/`head` (when a rig *does* deform through them) would be dropped.
- **Name the spine chain positionally**: first = pelvis, last = head,
  second-to-last = neck, the rest = spine_0N. This is what gives both engines a
  real head/neck bone to retarget to. Assumes the standard bottom-to-top spine
  deform order (reliable for Rigify).

### Critical gotcha — DEF- bones are constraint-driven

Rigify's DEF- deform bones are NOT free bones: they follow Copy Transforms /
Stretch To constraints that target the MCH-/ORG- mechanism bones. Extracting a
deform-only skeleton removes those targets, so the constraints evaluate to
garbage and drag the bones off the bind pose — the skinned mesh explodes into
spikes (verts collapsing toward bone roots and the world origin). Fix: strip all
pose-bone constraints and clear the pose (`_freeze_to_rest`) BEFORE removing
bones, so each deform bone sits at its rest = bind position. Verified by bbox:
exported mesh matches the original to FBX float precision.

Also fixed: `_duplicate` must use the data API (`obj.data.copy()`), never
`bpy.ops.object.duplicate()`, which shares data when the user's Duplicate Data
prefs have Mesh/Armature unchecked — editing the "copy" then destroys the real
character.

Also fixed: `_add_root` must clear `use_connect` before parenting an orphan bone
to the root. A connected bone snaps its head to the new parent's tail, so
parentless deform bones far from origin (breast, pelvis fans, shoulders) got
yanked to the world origin and stretched into giant bones. (The mesh still looked
fine at rest — rest == bind is always identity — so only the skeleton was wrong;
but posing those bones would pivot around origin.)

Lesson: verify the DEFORMED MESH SHAPE (bbox / vertex positions) AND bone
sizes/positions, not just bone names and counts. All three bugs above passed
name/count checks.

### Animation bake (shipped 2026-07-18)

`include_anim` on the operator/`build_and_export`. The clean skeleton is frozen
(its DEF constraints had to be stripped), but the ORIGINAL rig still animates —
so each clean bone gets a temporary world-space Copy Transforms constraint
targeting its source bone there (via the rename map returned by
`_rename_for_target`), and `nla.bake` with visual keying converts that into
plain keyframes over the active action's frame range. Merged limbs follow their
BASE segment (twist loss is inherent to merge mode); `root` follows Rigify's
root control bone so root motion survives. FBX gains `bake_anim_*` settings
(active action only, no NLA, `anim_simplify` exposed, default 0 = lossless).

Verified headless (Blender 4.5, full face metarig, 160 deform bones): baked
`lowerarm_l` world position matches `DEF-forearm.L` to 0.000000 at both keyed
frames; root motion baked; original rig's action untouched; FBX reimports with
the action. Test: `dev/test_anim_bake.py`.

### Strip Face Bones (shipped 2026-07-18)

`strip_face` on the operator: removes the face rig's deform bones (87 on the
full face metarig, 153 -> 66-bone skeleton) and folds each one's weights into
the head, so the face follows the head rigidly. Two hard-won details:

- **Identify face bones on the FULL hierarchy, before extraction** — a face
  deform bone is one whose ancestor chain passes through `ORG-face`.
  Extraction orphans them (their ORG parents vanish), so descendants-of-head
  finds nothing afterwards.
- **`_merge_vgroup` now uses 'ADD' write mode** — 'REPLACE' can no-op for a
  vertex not yet in the base group; invisible on limb twist merges (segments
  always overlap) but face->head folding hits exactly that case.
- Test lesson: assert per-vertex weight **conservation** (before == after),
  not sum-to-1 — Blender's auto-weight does not normalize every vertex.

### Multi-action + twist-preserve (shipped 2026-07-18)

- **Animation enum** (None / Current Action / All Actions): ALL bakes every
  action that animates the rig, parks each in its own NLA strip on the clean
  skeleton, and exports with `bake_anim_use_nla_strips` — one FBX animation
  stack (clip) per action. Baked outputs are tagged
  (`action["er_game_bake"]`) so the scan never re-exports its own bakes —
  name-overlap heuristics CANNOT work here because stripped core names
  (lip.T.L…) collide with Rigify's control-bone names. Baked actions are
  also cleaned up with the skeleton (they used to leak).
- **Preserve Twist Bones** (Unreal): limb `.001` segments survive as UE
  twist bones (`upperarm_twist_01_l`, `calf_twist_01_l`, …) with the
  Mannequin chain shape — next segment reparents to the base, twist becomes
  a leaf child. Weights untouched. Gotcha: the side suffix sits BEFORE the
  segment suffix (`upper_arm.L.001`), so `_ue_name` must peel `.001` first.
- Blender 5.x compat: `Bone.select` removed → `pose.select_all` op;
  `Action.fcurves` removed (slotted actions) → `_act_fcurves` walks
  layers/strips/channelbags.

### Anatomical hierarchy fix (2026-07-18, user found in Unreal/Unity)

In-engine, the clavicle/spine didn't carry the arm and the thigh fell off the
pelvis; the toe bone was missing and skinning looked broken. Root cause: WITHIN
a limb Rigify's DEF bones parent to each other (forearm->upper_arm — why the
spike passed), but ACROSS joints they parent through the ORG/MCH mechanism
layer (DEF-upper_arm.L's parent is ORG-upper_arm.L under ORG-shoulder.L...).
Extraction deletes the mechanism bones, so `_add_root` dropped every limb head,
shoulder and toe (97 bones) straight onto root. Invisible in Blender (rest pose
+ our world-space bakes) but the engine skeleton was flat — posing a parent
didn't move its child, and the mesh followed the flat skeleton.

Fix: `_deform_parent_map` reads each deform bone's true parent from the FULL
hierarchy BEFORE extraction (walk original ancestors, resolve each ORG-/MCH-
bone to its DEF- counterpart, first hit wins); `_reparent_orphans` reattaches
them after extraction, before merge. Verified: upperarm<-clavicle<-spine,
thigh<-pelvis, ball<-foot; only pelvis sits under root. This also restored the
toe and the perceived "bad weights" (the toe/foot verts were pulling to a
root-parented bone). Test: hierarchy assertion in test_anim_bake.py.

### Remaining refinements

- In-engine re-validation after the hierarchy fix (user).
- Weight normalization on export IF partial-weight reports persist after the
  hierarchy fix (Blender auto-weight leaves some verts <1.0; our Heat Map
  pipeline normalizes, so only unnormalized source meshes would show it).

## Tier fit (DECIDED 2026-07-18)

Tiers: Lite / Full / **Team (5-seat)** — "Studio" renamed. Game export ships
in BOTH editions (Lite included). Retargeting ships in Full only (Lite-gated
via LITE_BUILD). Batch export/retarget = Team-tier candidates when built.
