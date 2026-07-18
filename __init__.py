# SPDX-License-Identifier: GPL-3.0-or-later
#
# Easy Rigify — Place markers then auto-align a Rigify metarig
# Copyright (C) 2024  Dwayne Jones
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

bl_info = {
    "name": "Easy Rigify",
    "author": "Dwayne Jones",
    "version": (1, 2, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Easy Rigify",
    "description": "Place markers then auto-align a Rigify metarig",
    "category": "Rigging",
}

_ADDON_VERSION = bl_info["version"]

import bpy
import bpy.utils.previews  # `bpy.utils.previews` is a submodule, not auto-loaded
                            # by `import bpy` alone — normally something else in
                            # Blender's own startup pulls it in first, but that
                            # isn't guaranteed on every Blender version/launch
                            # order (broke register() on 5.2: "module 'bpy.utils'
                            # has no attribute 'previews'"). Import it explicitly
                            # so register() below never depends on load order.
import os

# ── Submodule imports ─────────────────────────────────────────────────────────
from . import utils
from .constants import LITE_BUILD

from .markers import (
    AutoRigFaceObjProps,
    marker_scale_update,
    AUTORIG_OT_DetectFaceObjects,
    AUTORIG_OT_PlaceMarkers,
    AUTORIG_OT_PlaceFaceMarkers,
    AUTORIG_OT_RemoveFaceMarkers,
    AUTORIG_OT_DeleteAllMarkers,
    AUTORIG_OT_AutoDetectBody,
    AUTORIG_OT_AutoDetectArms,
    AUTORIG_OT_MirrorMarkers,
    AUTORIG_OT_CheckMarkers,
    AUTORIG_OT_SelectAllMarkers,
    AUTORIG_OT_ApplyMarkerStyle,
    AUTORIG_OT_ToggleMarkers,
    AUTORIG_OT_ToggleXRay,
    AUTORIG_OT_ToggleMeshSel,
    AUTORIG_OT_MetarigNoFace,
    AUTORIG_OT_MetarigWithFace,
    AUTORIG_OT_MetarigFace,
    AUTORIG_OT_AlignRig,
    AUTORIG_OT_DetectFaceLandmarks,
    AUTORIG_OT_GenerateRig,
    AUTORIG_OT_AddRigifySample,
    AUTORIG_OT_CheckUpdate,
    AUTORIG_OT_OpenUpdateURL,
    AUTORIG_OT_ReportBug,
    AUTORIG_OT_RescaleMarkers,
    AUTORIG_Prefs,
    AUTORIG_PT_MetaRig,
    AUTORIG_PT_Markers,
    draw_markers_tab,
    draw_rig_tab,
)
from . import markers as _markers_mod
from .ai_detect import (
    AUTORIG_OT_AIDetectBody,
    AUTORIG_OT_AIDetectFace,
    AUTORIG_OT_AIDetectFingers,
    AUTORIG_OT_ResolveFinger,
)

from .fingers import (
    AUTORIG_OT_PlaceFingerMarkers,
    AUTORIG_OT_StraightenFingerMarkers,
    AUTORIG_OT_DetectFingers,
    AUTORIG_PT_FingerPicker,
)

from .joint_cleaner import (
    RIGIFYJOINT_OT_analyze,
    RIGIFYJOINT_OT_fix_bleeding,
    RIGIFYJOINT_OT_clean,
    AUTORIG_OT_SmoothJointWeights,
    AUTORIG_OT_SmoothTwistWeights,
    AUTORIG_OT_DEFBoneReport,
)

from .pipeline import (
    SurgicalSectionProps,
    AutoRigMeshItem,
    AutoRigSkinProps,
    AUTORIG_OT_SurgicalPreset,
    AUTORIG_OT_DetectMeshes,
    AUTORIG_OT_SmartBind,
    AUTORIG_OT_Unbind,
    AUTORIG_OT_CleanupWeights,
    AUTORIG_PT_Skinning,
)

from .weight_tools import (
    SmartWeightProperties,
    WeightEditProperties,
    AUTORIG_OT_SymmetrizeWeights,
    AUTORIG_OT_EnterWeightPaint,
    AUTORIG_OT_ExitWeightPaint,
    AUTORIG_OT_ToggleDefBones,
    AUTORIG_OT_SelectBoneRegion,
    SMARTWEIGHT_OT_transfer,
    SMARTWEIGHT_OT_batch_transfer,
    WEIGHTEDIT_OT_hammer,
    WEIGHTEDIT_OT_mirror,
    WEIGHTEDIT_OT_multi_edit,
    WeightAdvProperties,
    WEIGHTADV_OT_process,
    WEIGHTCLEAN_OT_prune_small,
    WEIGHTCLEAN_OT_remove_unused_groups,
    WEIGHTCLEAN_OT_select_unweighted,
    WEIGHTCLEAN_OT_clamp_bone,
    WEIGHTVIS_OT_reset_shading,
    WEIGHTADV_PT_panel,
    SMARTWEIGHT_PT_panel,
)

from .fan_bones import (
    FanBoneProperties,
    FANBONE_OT_generate,
    FANBONE_OT_remove,
    FANBONE_PT_panel,
)

from .custom_rig import (
    AutoRigPreserveProps,
    AUTORIG_OT_PreserveBackup,
    AUTORIG_OT_PreserveRestore,
    AUTORIG_OT_PreserveGenerate,
)

from .pipeline      import draw_skinning_tab
from .weight_tools  import draw_weights_tab, draw_visualization_section
from .fan_bones     import draw_fan_bones_section
from .game_export   import AUTORIG_OT_ExportGame, draw_game_export_section
from .retarget      import (AutoRigRetargetMapItem, AutoRigRetargetProps,
                            AUTORIG_OT_RetargetAnim, AUTORIG_UL_RetargetMap,
                            AUTORIG_OT_RetargetAutoMap,
                            AUTORIG_OT_RetargetMapAdd,
                            AUTORIG_OT_RetargetMapRemove,
                            AUTORIG_OT_RetargetMapClear,
                            AUTORIG_OT_RetargetMapSave,
                            AUTORIG_OT_RetargetMapLoad,
                            draw_retarget_section)
from .utils         import get_icon
# DEV-ONLY data-generation tools: registered only when the dev/ folder exists
# (the shipped zip excludes dev/, so customer installs never see these panels;
# the working copy keeps them without any ship-time step to remember).
import os as _os
_DEV_BUILD = _os.path.isdir(_os.path.join(_os.path.dirname(
    _os.path.abspath(__file__)), "dev"))
if _DEV_BUILD:
    from . import gen_face_training_data as _gen_face_data   # face-model labeling tool
    from . import gen_hand_training_data as _gen_hand_data   # hand-model capture/retrain tool

# ── Help opener ──────────────────────────────────────────────────────────────

_HELP_URLS = {
    'MARKERS': "https://dwaynejnr-alt.github.io/Easy-Rigify/markers.html",
    'RIG':     "https://dwaynejnr-alt.github.io/Easy-Rigify/rigging.html",
    'SKIN':    "https://dwaynejnr-alt.github.io/Easy-Rigify/skinning.html",
    'TOOLS':   "https://dwaynejnr-alt.github.io/Easy-Rigify/features.html",
}

class AUTORIG_OT_OpenHelp(bpy.types.Operator):
    bl_idname      = "autorig.open_help"
    bl_label       = "Open Help"
    bl_description = "Open the help documentation for this tab"

    tab_key: bpy.props.StringProperty()

    def execute(self, _context):
        import webbrowser
        url = _HELP_URLS.get(self.tab_key, "")
        if not url:
            self.report({'WARNING'}, "Documentation page not available yet.")
            return {'CANCELLED'}
        webbrowser.open(url)
        return {'FINISHED'}


# ── Main tabbed panel ─────────────────────────────────────────────────────────

class AUTORIG_PT_Main(bpy.types.Panel):
    """Easy Rigify — tabbed main panel."""
    bl_label       = "Easy Rigify"
    bl_idname      = "AUTORIG_PT_Main"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = "Easy Rigify"

    def draw_header(self, context):
        icon = get_icon("ER")
        if icon:
            self.layout.label(text="", icon_value=icon)

    def draw(self, context):
        layout = self.layout
        scene  = context.scene

        # ── Tab bar ───────────────────────────────────────────────────────────
        row = layout.row(align=True)
        row.scale_y = 1.3
        row.prop(scene, "autorig_tab", expand=True)
        layout.separator(factor=0.5)

        tab = scene.autorig_tab
        help_icon = get_icon("Help")
        if help_icon:
            row = layout.row(align=True)
            row.alignment = 'RIGHT'
            op = row.operator("autorig.open_help", text="", icon_value=help_icon)
            op.tab_key = tab

        if tab == 'MARKERS':
            draw_markers_tab(layout, context)
        elif tab == 'RIG':
            draw_rig_tab(layout, context)
        elif tab == 'SKIN':
            draw_skinning_tab(layout, context)
            layout.separator()
            draw_weights_tab(layout, context)
        elif tab == 'TOOLS':
            draw_fan_bones_section(layout, context)
            layout.separator()
            draw_retarget_section(layout, context)
            layout.separator()
            draw_game_export_section(layout, context)
            layout.separator()
            draw_visualization_section(layout, context)


# ── Class registry (order matters: PropertyGroups before operators that use them) ─
CLASSES = (
    AUTORIG_OT_OpenHelp,
    AutoRigFaceObjProps,
    AUTORIG_OT_DetectFaceObjects,
    AUTORIG_OT_PlaceMarkers,
    AUTORIG_OT_PlaceFaceMarkers,
    AUTORIG_OT_RemoveFaceMarkers,
    AUTORIG_OT_DeleteAllMarkers,
    AUTORIG_OT_AutoDetectBody,
    AUTORIG_OT_AIDetectBody,
    AUTORIG_OT_AIDetectFace,
    AUTORIG_OT_AIDetectFingers,
    AUTORIG_OT_ResolveFinger,
    AUTORIG_OT_AutoDetectArms,
    AUTORIG_OT_PlaceFingerMarkers,
    AUTORIG_OT_StraightenFingerMarkers,
    AUTORIG_OT_DetectFingers,
    AUTORIG_OT_MirrorMarkers,
    AUTORIG_OT_CheckMarkers,
    AUTORIG_OT_SelectAllMarkers,
    AUTORIG_OT_ApplyMarkerStyle,
    AUTORIG_OT_RescaleMarkers,
    AUTORIG_OT_ToggleMarkers,
    AUTORIG_OT_ToggleXRay,
    AUTORIG_OT_ToggleMeshSel,
    AUTORIG_OT_MetarigNoFace,
    AUTORIG_OT_MetarigWithFace,
    AUTORIG_OT_MetarigFace,
    AUTORIG_OT_AlignRig,
    AUTORIG_OT_DetectFaceLandmarks,
    AUTORIG_OT_GenerateRig,
    AUTORIG_OT_AddRigifySample,
    AUTORIG_OT_CheckUpdate,
    AUTORIG_OT_OpenUpdateURL,
    AUTORIG_OT_ReportBug,
    # Skinning
    SurgicalSectionProps,
    AutoRigMeshItem,
    AutoRigSkinProps,
    AUTORIG_OT_SurgicalPreset,
    RIGIFYJOINT_OT_analyze,
    RIGIFYJOINT_OT_fix_bleeding,
    RIGIFYJOINT_OT_clean,
    AUTORIG_OT_SmoothJointWeights,
    AUTORIG_OT_SmoothTwistWeights,
    SmartWeightProperties,
    WeightEditProperties,
    AUTORIG_OT_DetectMeshes,
    AUTORIG_OT_SmartBind,
    AUTORIG_OT_Unbind,
    AUTORIG_OT_CleanupWeights,
    AUTORIG_OT_SymmetrizeWeights,
    AUTORIG_OT_EnterWeightPaint,    
    AUTORIG_OT_ExitWeightPaint,
    AUTORIG_OT_ToggleDefBones,
    AUTORIG_OT_SelectBoneRegion,
    AUTORIG_OT_DEFBoneReport,
    SMARTWEIGHT_OT_transfer,
    SMARTWEIGHT_OT_batch_transfer,
    WEIGHTEDIT_OT_hammer,
    WEIGHTEDIT_OT_mirror,
    WEIGHTEDIT_OT_multi_edit,
    # Advanced weights
    WeightAdvProperties,
    WEIGHTADV_OT_process,
    WEIGHTCLEAN_OT_prune_small,
    WEIGHTCLEAN_OT_remove_unused_groups,
    WEIGHTCLEAN_OT_select_unweighted,
    WEIGHTCLEAN_OT_clamp_bone,
    WEIGHTVIS_OT_reset_shading,
    # Fan bones
    FanBoneProperties,
    FANBONE_OT_generate,
    FANBONE_OT_remove,
    # Game export
    AUTORIG_OT_ExportGame,
    # Animation retarget (map item PG before the props PG that collects it)
    AutoRigRetargetMapItem,
    AutoRigRetargetProps,
    AUTORIG_OT_RetargetAnim,
    AUTORIG_UL_RetargetMap,
    AUTORIG_OT_RetargetAutoMap,
    AUTORIG_OT_RetargetMapAdd,
    AUTORIG_OT_RetargetMapRemove,
    AUTORIG_OT_RetargetMapClear,
    AUTORIG_OT_RetargetMapSave,
    AUTORIG_OT_RetargetMapLoad,
    # Custom rig preserve (PropertyGroup before operators)
    AutoRigPreserveProps,
    AUTORIG_OT_PreserveBackup,
    AUTORIG_OT_PreserveRestore,
    AUTORIG_OT_PreserveGenerate,
    # Panels
    AUTORIG_PT_Main,
    AUTORIG_PT_FingerPicker,
    AUTORIG_PT_MetaRig,
    AUTORIG_PT_Markers,
    AUTORIG_PT_Skinning,
    WEIGHTADV_PT_panel,
    SMARTWEIGHT_PT_panel,
    FANBONE_PT_panel,
    AUTORIG_Prefs,
)


def _draw_armature_add_menu(self, context):
    self.layout.separator()
    self.layout.operator("autorig.generate_metarig_no_face",  text="Human (No Face)",  icon='ARMATURE_DATA')
    self.layout.operator("autorig.generate_metarig_with_face", text="Human (With Face)", icon='ARMATURE_DATA')


def register():
    # Clean up any leftover state from a previous (failed or hot) reload
    if utils._preview_coll is not None:
        bpy.utils.previews.remove(utils._preview_coll)
        utils._preview_coll = None
    for cls in CLASSES:
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass

    utils._preview_coll = bpy.utils.previews.new()

    bpy.types.Scene.autorig_tab = bpy.props.EnumProperty(
        items=[
            ('MARKERS', 'Markers', 'Marker placement and facial detection'),
            ('RIG',     'Rig',     'Metarig generation, alignment and Rigify'),
            ('SKIN',    'Skin',    'Skinning, weight painting and cleanup'),
            ('TOOLS',   'Tools',   'Fan bones and utilities'),
        ],
        default='MARKERS',
    )

    icons_dir = os.path.join(os.path.dirname(__file__), "icons")
    for key, filename in [
        ("AR",           "AR.png"),
        ("ER",           "ER.png"),
        ("MIcon",        "MIcon.png"),
        ("Meta",         "Meta.png"),
        ("Body",         "Body.png"),
        ("MetaB",        "MetaB.png"),
        ("Face",         "Face.png"),
        ("MetaRig",      "MetaRig.png"),
        ("Body_Help",    "Body_Help.png"),
        ("Arm_Help",     "Arm_Help.png"),
        ("Fingers_help", "Fingers_help.png"),
        ("Lip_Help",     "Lip_Help.png"),
        ("Chin_Help",    "Chin_Help.png"),
        ("Eye_Help",     "Eye_Help.png"),
        ("Nose_Help",    "Nose_Help.png"),
        ("Lips",         "Lips.png"),
        ("Brow",         "Brow.png"),
        ("Eye",          "Eye.png"),
        ("Nose",         "Nose.png"),
        ("Forehead",     "Forehead.png"),
        ("Chin",         "Chin.png"),
        ("Sh",           "Sh.png"),
        ("Hand",         "Hand.png"),
        ("Finger",       "Finger.png"),
        ("Help",         "Help.png"),
    ]:
        path = os.path.join(icons_dir, filename)
        if os.path.exists(path):
            utils._preview_coll.load(key, path, 'IMAGE')

    for cls in CLASSES:
        bpy.utils.register_class(cls)

    bpy.types.Scene.autorig_face_objs = bpy.props.PointerProperty(
        type=AutoRigFaceObjProps)
    bpy.types.Scene.autorig_skin = bpy.props.PointerProperty(
        type=AutoRigSkinProps)
    bpy.types.Scene.smart_weight_props = bpy.props.PointerProperty(
        type=SmartWeightProperties)
    bpy.types.Scene.weight_edit_props = bpy.props.PointerProperty(
        type=WeightEditProperties)
    bpy.types.Scene.weight_adv_props = bpy.props.PointerProperty(
        type=WeightAdvProperties)
    bpy.types.Scene.fan_bone_props = bpy.props.PointerProperty(
        type=FanBoneProperties)
    bpy.types.Scene.autorig_retarget = bpy.props.PointerProperty(
        type=AutoRigRetargetProps)
    bpy.types.Scene.autorig_preserve_props = bpy.props.PointerProperty(
        type=AutoRigPreserveProps)
    bpy.types.Scene.autorig_show_hints = bpy.props.BoolProperty(
        name="Marker Hints", default=True,
        description="Show placement hint overlay when a marker is selected")
    bpy.types.Scene.autorig_live_symmetry = bpy.props.BoolProperty(
        name="Live Symmetry", default=False,
        description="When you move a left/right marker, its mirror counterpart "
                    "follows automatically (across the character's symmetry plane)")
    bpy.types.Scene.autorig_detect_symmetry = bpy.props.BoolProperty(
        name="Symmetrical Detect", default=True,
        description="Mirror-average left/right markers around the character's "
                    "centreline after every detect (body, fingers, face) so both "
                    "sides come out perfectly symmetric — what Rigify expects. "
                    "Turn OFF for deliberately asymmetric characters to keep "
                    "each side's raw detection")
    bpy.types.Scene.autorig_marker_scale = bpy.props.FloatProperty(
        name="Marker Scale", default=1.0, min=0.02, max=10.0, step=1,
        description="Scale multiplier for all marker empty sizes (auto-set by EasyDetect Body)",
        update=marker_scale_update)
    bpy.types.Scene.finger_center_radius = bpy.props.FloatProperty(
        name="Center Radius", default=0.015, min=0.005, max=0.100, step=0.1,
        precision=3, unit='LENGTH',
        description="Search radius for BVH cross-section centering of finger joints after AI detection. "
                    "Increase for thick/stylised fingers, decrease for thin ones")
    bpy.types.Scene.finger_palm_depth = bpy.props.FloatProperty(
        name="Palm Depth", default=0.25, min=0.15, max=0.40, step=0.5, precision=2,
        description="How far the chain-walk target is pulled inward from the wrist toward the fingers "
                    "(fraction of wrist-to-tip-centroid). Increase for thick/meaty palms")
    bpy.types.Scene.finger_knuckle_radius = bpy.props.FloatProperty(
        name="Knuckle Radius", default=0.90, min=0.70, max=1.10, step=1, precision=2,
        description="Scales the knuckle arc radius used by geometric MCP refinement. "
                    "Increase for large knuckles/thick fingers, decrease for thin/spindly ones")
    bpy.types.Scene.finger_width_tolerance = bpy.props.FloatProperty(
        name="Chain Width Tol", default=1.00, min=0.50, max=2.00, step=5, precision=2,
        description="Multiplier on the palm-transition width threshold in the chain-walk. "
                    "Increase for thick finger shafts, decrease for thin/subtle knuckle transitions")
    bpy.types.Scene.finger_detection_engine = bpy.props.EnumProperty(
        name="Finger Engine",
        items=[('AUTO',      "EasyDetect (best per hand)",
                "The full detection pipeline: neural evidence constrained by "
                "the always-valid hand template, geometric fallback when the "
                "neural evidence is thin, and the quality takeover hands a "
                "flagged side to the geometric engine"),
               ('NEURAL',    "Neural (AI)",
                "Hybrid ONNX pipeline: tip-heatmap + 20-landmark models with "
                "multi-view renders. Best quality; needs onnxruntime"),
               ('GEOMETRIC', "Geometric (mesh)",
                "Geodesic-tube engine working purely from the mesh: no renders, "
                "no onnxruntime. Fingertips from geodesic maxima, joints along "
                "each finger's measured medial axis"),
               ('TEMPLATE',  "Template (constrained)",
                "Phase-1 constrained builder: runs a detector for evidence "
                "(tips + outer knuckles) then REBUILDS the hand from an "
                "always-valid template - knuckle row forced ordered/spaced, "
                "phalanges at fixed ratios. Can't collapse or cross fingers")],
        # Lite ships no neural models, so AUTO (neural evidence + template)
        # has nothing to run — default to the mesh-only engine there.
        default='GEOMETRIC' if LITE_BUILD else 'AUTO',
        description="Which engine Detect Fingers uses")
    bpy.types.Scene.finger_engine_advanced = bpy.props.BoolProperty(
        name="Advanced Engines",
        default=False,
        description="Show the individual detector engines (Neural, Template). "
                    "Auto already orchestrates them — these exist for "
                    "debugging and A/B comparison, not day-to-day use")
    bpy.types.Scene.finger_wrist_autosnap = bpy.props.BoolProperty(
        name="Auto-snap Wrist",
        default=True,
        description="Before detecting fingers, move a misplaced HAND (wrist) "
                    "marker onto the mesh wrist automatically. Turn OFF to place "
                    "the wrist yourself — useful on small/stylized characters "
                    "where the auto estimate is off and you want manual control")
    bpy.types.Scene.geo_knuckle_depth = bpy.props.FloatProperty(
        name="Knuckle Depth", default=0.22, min=0.05, max=0.50, step=1,
        precision=2,
        description="GEOMETRIC engine: how far each finger's MCP extends from "
                    "the web toward the wrist (fraction of finger length). "
                    "Increase if MCPs sit up inside the finger; decrease if "
                    "they sink into the palm")
    bpy.types.Scene.geo_thumb_depth = bpy.props.FloatProperty(
        name="Thumb Base Depth", default=0.45, min=0.20, max=0.90, step=1,
        precision=2,
        description="GEOMETRIC engine: how far THUMB_1 extends from the thumb "
                    "web into the thenar mound (fraction of thumb length). "
                    "Increase if the thumb base sits too close to the web; "
                    "decrease if it dives into the palm")
    bpy.types.Scene.geo_min_finger = bpy.props.FloatProperty(
        name="Min Finger Size", default=0.30, min=0.15, max=0.50, step=1,
        precision=2,
        description="GEOMETRIC engine: tubes shorter than this fraction of "
                    "the hand's median finger length are treated as bumps and "
                    "discarded. Raise it if bumps/folds get detected as "
                    "fingers; lower it for hands with a very short pinky or "
                    "thumb")
    bpy.types.Scene.finger_straighten_clamp = bpy.props.FloatProperty(
        name="Straighten Clamp", default=0.30, min=0.05, max=0.30, step=1, precision=2,
        description="Max PIP/DIP lateral shift as a fraction of finger length during the final "
                    "straighten pass. Joints are projected onto each finger's bend plane to remove "
                    "side-to-side wander while keeping the downward curl. A correction larger than "
                    "this is skipped (untrustworthy anchor). Lower preserves noisy poses as-is")
    for _sname in ['surgical_fix_fingers', 'surgical_fix_forearms', 'surgical_fix_upper_arms',
                   'surgical_fix_shins',   'surgical_fix_thighs',   'surgical_fix_shoulders',
                   'surgical_fix_spine',   'surgical_fix_neck']:
        setattr(bpy.types.Scene, _sname,
                bpy.props.PointerProperty(type=SurgicalSectionProps))

    bpy.types.VIEW3D_MT_armature_add.append(_draw_armature_add_menu)

    # Marker overlays — hint text and billboard icons (images load lazily on first draw).
    # Use bpy.app.driver_namespace to track handles across script reloads: without this,
    # reloading scripts resets the module-level handle variables to None while leaving
    # the old handlers still registered, so every reload stacks another copy.
    _ns = bpy.app.driver_namespace
    _prev = _ns.pop("_autorig_hint_h", None)
    if _prev is not None:
        try: bpy.types.SpaceView3D.draw_handler_remove(_prev, 'WINDOW')
        except Exception: pass
    _prev = _ns.pop("_autorig_billboard_h", None)
    if _prev is not None:
        try: bpy.types.SpaceView3D.draw_handler_remove(_prev, 'WINDOW')
        except Exception: pass
    _prev = _ns.pop("_autorig_lines_h", None)
    if _prev is not None:
        try: bpy.types.SpaceView3D.draw_handler_remove(_prev, 'WINDOW')
        except Exception: pass

    _markers_mod._marker_hint_handle = bpy.types.SpaceView3D.draw_handler_add(
        _markers_mod._draw_marker_hint, (), 'WINDOW', 'POST_PIXEL')
    _ns["_autorig_hint_h"] = _markers_mod._marker_hint_handle

    _markers_mod._marker_billboard_handle = bpy.types.SpaceView3D.draw_handler_add(
        _markers_mod._draw_marker_billboards, (), 'WINDOW', 'POST_PIXEL')
    _ns["_autorig_billboard_h"] = _markers_mod._marker_billboard_handle

    # Marker connection lines — POST_VIEW (world-space) so they track in 3D.
    _markers_mod._marker_lines_handle = bpy.types.SpaceView3D.draw_handler_add(
        _markers_mod._draw_marker_lines, (), 'WINDOW', 'POST_VIEW')
    _ns["_autorig_lines_h"] = _markers_mod._marker_lines_handle

    # Reset image cache after file load/revert so stale refs don't linger
    if _markers_mod._reset_marker_cache not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_markers_mod._reset_marker_cache)

    # Eagerly load the marker icon images and refresh any open 3D viewports so the
    # marker billboards appear immediately on enable. Without this the billboards
    # only populate on the next redraw of a viewport that happens to be in focus,
    # which is why the marker icons used to require a manual addon reload to show up.
    try:
        _markers_mod._load_marker_images()
    except Exception:
        pass
    try:
        _wm = bpy.context.window_manager
        for _win in (_wm.windows if _wm else ()):
            for _area in _win.screen.areas:
                if _area.type == 'VIEW_3D':
                    _area.tag_redraw()
    except Exception:
        pass

    # Live marker symmetry (depsgraph handler; gated by the autorig_live_symmetry toggle)
    _markers_mod.register_live_symmetry()

    # DEV-only data-generation panels (Save Face Data xN / Hand Data Generator):
    # auto-gated on the dev/ folder, which shipped zips exclude.
    if _DEV_BUILD:
        _gen_face_data.register()
        _gen_hand_data.register()


def unregister():
    try:
        _markers_mod.unregister_live_symmetry()
    except Exception:
        pass
    if _DEV_BUILD:
        try:
            _gen_face_data.unregister()
        except Exception:
            pass
        try:
            _gen_hand_data.unregister()
        except Exception:
            pass

    if _markers_mod._reset_marker_cache in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_markers_mod._reset_marker_cache)

    if _markers_mod._marker_billboard_handle:
        bpy.types.SpaceView3D.draw_handler_remove(
            _markers_mod._marker_billboard_handle, 'WINDOW')
        _markers_mod._marker_billboard_handle = None
    bpy.app.driver_namespace.pop("_autorig_billboard_h", None)
    _markers_mod._unload_marker_images()

    if _markers_mod._marker_hint_handle:
        bpy.types.SpaceView3D.draw_handler_remove(
            _markers_mod._marker_hint_handle, 'WINDOW')
        _markers_mod._marker_hint_handle = None
    bpy.app.driver_namespace.pop("_autorig_hint_h", None)

    if getattr(_markers_mod, "_marker_lines_handle", None):
        bpy.types.SpaceView3D.draw_handler_remove(
            _markers_mod._marker_lines_handle, 'WINDOW')
        _markers_mod._marker_lines_handle = None
    bpy.app.driver_namespace.pop("_autorig_lines_h", None)

    bpy.types.VIEW3D_MT_armature_add.remove(_draw_armature_add_menu)

    if hasattr(bpy.types.Scene, 'autorig_show_hints'):
        del bpy.types.Scene.autorig_show_hints
    if hasattr(bpy.types.Scene, 'autorig_live_symmetry'):
        del bpy.types.Scene.autorig_live_symmetry
    if hasattr(bpy.types.Scene, 'autorig_detect_symmetry'):
        del bpy.types.Scene.autorig_detect_symmetry
    if hasattr(bpy.types.Scene, 'autorig_marker_scale'):
        del bpy.types.Scene.autorig_marker_scale
    if hasattr(bpy.types.Scene, 'finger_center_radius'):
        del bpy.types.Scene.finger_center_radius
    for _prop in ('finger_palm_depth', 'finger_knuckle_radius',
                  'finger_width_tolerance', 'finger_straighten_clamp',
                  'finger_detection_engine', 'finger_engine_advanced',
                  'geo_knuckle_depth', 'geo_thumb_depth', 'geo_min_finger'):
        if hasattr(bpy.types.Scene, _prop):
            delattr(bpy.types.Scene, _prop)
    for _sname in ['surgical_fix_fingers', 'surgical_fix_forearms', 'surgical_fix_upper_arms',
                   'surgical_fix_shins',   'surgical_fix_thighs',   'surgical_fix_shoulders',
                   'surgical_fix_spine',   'surgical_fix_neck']:
        if hasattr(bpy.types.Scene, _sname):
            delattr(bpy.types.Scene, _sname)
    if hasattr(bpy.types.Scene, 'autorig_skin'):
        del bpy.types.Scene.autorig_skin
    if hasattr(bpy.types.Scene, 'smart_weight_props'):
        del bpy.types.Scene.smart_weight_props
    if hasattr(bpy.types.Scene, 'weight_edit_props'):
        del bpy.types.Scene.weight_edit_props
    if hasattr(bpy.types.Scene, 'weight_adv_props'):
        del bpy.types.Scene.weight_adv_props
    if hasattr(bpy.types.Scene, 'fan_bone_props'):
        del bpy.types.Scene.fan_bone_props
    if hasattr(bpy.types.Scene, 'autorig_preserve_props'):
        del bpy.types.Scene.autorig_retarget
        del bpy.types.Scene.autorig_preserve_props
    if hasattr(bpy.types.Scene, 'autorig_tab'):
        del bpy.types.Scene.autorig_tab
    del bpy.types.Scene.autorig_face_objs

    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)

    if utils._preview_coll:
        bpy.utils.previews.remove(utils._preview_coll)
        utils._preview_coll = None


if __name__ == "__main__":
    register()
