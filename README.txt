===============================================================================
 EASY RIGIFY
 Place markers -> auto-align a Rigify metarig -> smart skinning
===============================================================================

Auto-places the markers for a Rigify rig, aligns and generates the metarig, and
skins your mesh. The full edition adds AI detection for the body, fingers and
face; the geometric detectors work in every edition.

Requires Blender 4.2+ with the built-in Rigify addon enabled.


EDITIONS
--------
- Easy Rigify (full)  - AI + geometric detection.
- Easy Rigify Lite    - geometric detection only, no AI. ~1 MB.
Scenes are compatible between the two: a file saved in the full edition opens
in Lite and simply uses the geometric engine. They cannot both be installed at
once.


QUICK START
-----------
1. INSTALL
   Edit > Preferences > Get Extensions > (drop-down) Install from Disk...
   Pick the easy_rigify zip and enable "Easy Rigify".
   The AI runtime (full edition) is bundled - Blender sets it up automatically
   when you enable the addon. There is nothing to install by hand.

2. PLACE MARKERS  (press N > "Easy Rigify" tab > Markers)
   - Set your character's "Body Mesh".
   - Auto Detect Body      -> spine, arms, legs, feet (geometric).
     EasyDetect Body       -> same, using AI (full edition).
   - EasyDetect Fingers    -> hand/finger joints. (full edition)
   - Facial Markers        -> lips, eyes, brows, nose, cheeks, chin, jaw, ears. (full edition)

   Markers are editable empties - grab (G) and nudge any that are off, then
   run "Check All Markers" to catch anything missing, overlapping, or left
   outside the mesh.

3. GENERATE
   Rig tab  -> align & generate the Rigify rig.
   Skin tab -> bind and clean up weights.


TIPS
----
- Show / Hide Markers toggles markers, icons and connecting lines together.
- Live Symmetry (Mirror section) mirrors your edits left <-> right.
- Fingers: first click is the fingertip, second is the knuckle.
- Apply scale and rotation on the body mesh first (Object > Apply > All
  Transforms) - unapplied scale is the #1 cause of bad alignment and binding.
- In the full edition the EasyDetect (AI) buttons appear automatically. If the
  AI runtime can't load on your platform, the geometric detectors are used
  instead. Lite never shows the AI buttons.


FULL GUIDE
----------
Easy Rigify Documentation
https://www.notion.so/Easy-Rigify-Documentation-37019f42b1f880c3b748f0849675e3ea
for detailed steps and troubleshooting.
===============================================================================
