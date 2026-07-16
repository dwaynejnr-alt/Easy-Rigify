===============================================================================
 EASY RIGIFY
 Place markers -> auto-align a Rigify metarig -> smart skinning
===============================================================================

Auto-places the markers for a Rigify rig, aligns and generates the metarig, and
skins your mesh. Includes AI detection for the body, fingers and face.

Requires Blender 4.2+ with the built-in Rigify addon enabled.


QUICK START
-----------
1. INSTALL
   Edit > Preferences > Get Extensions > (drop-down) Install from Disk...
   Pick easy_rigify-x.x.x.zip and enable "Easy Rigify".

2. INSTALL THE AI RUNTIME (one time, needs internet)
   Edit > Preferences > Add-ons > Easy Rigify > "Install ONNX Runtime".
   Wait for "Ready". Restart Blender if prompted.
   (Installs into your user folder - no admin needed. The AI models are bundled.)

3. PLACE MARKERS  (press N > "AutoRig" tab > Markers)
   - Set your character's "Body Mesh".
   - AI Detect Body        -> spine, arms, legs, feet.
   - AI Detect Fingers     -> hand/finger joints.
   - Facial Markers >  -> lips, eyes, brows, nose, cheeks,
                                          chin, jaw, ears.
     
   Markers are editable empties - grab (G) and nudge any that are off.

4. GENERATE
   Rig tab  -> align & generate the Rigify rig.
   Skin tab -> bind and clean up weights.


TIPS
----
- Show / Hide Markers toggles markers, icons and connecting lines together.
- Live Symmetry (Mirror section) mirrors your edits left <-> right.
- AI buttons appear once the ONNX Runtime is installed.


FULL GUIDE
----------
Easy Rigify Documentation
https://www.notion.so/Easy-Rigify-Documentation-37019f42b1f880c3b748f0849675e3ea
for detailed steps and troubleshooting.
===============================================================================
