"""
Custom Rig Preserve
====================
Backs up user-added constraints (including on ORG-/MCH- bones), custom bones,
drivers, and custom shapes, then restores them after Rigify regeneration.
Backup is stored as JSON on the scene so it survives undo and file save/reload.
"""

import bpy
from .constants import dbg
import json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_metarig(context):
    ao = context.active_object
    if ao and ao.type == 'ARMATURE' and not any(
            b.name.startswith('DEF-') for b in ao.data.bones):
        return ao
    for obj in context.scene.objects:
        if obj.type == 'ARMATURE' and not any(
                b.name.startswith('DEF-') for b in obj.data.bones):
            return obj
    return None


def _find_generated_rig(metarig):
    """Find generated rig by rig_id, then by heuristic (has ORG-/MCH-/DEF- bones)."""
    rig_id = metarig.data.get("rig_id")
    if rig_id:
        for obj in bpy.data.objects:
            if obj.type == 'ARMATURE' and obj != metarig:
                if obj.data.get("rig_id") == rig_id:
                    return obj
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE' and obj != metarig:
            if any(b.name.startswith(('ORG-', 'MCH-', 'DEF-')) for b in obj.data.bones):
                return obj
    return None


# ---------------------------------------------------------------------------
# Core preserve class
# ---------------------------------------------------------------------------

class RigifyCustomPreserve:

    def __init__(self, metarig, target_rig=None):
        self.metarig   = metarig
        self.target_rig = target_rig  # override; None = auto-detect
        self._reset()

    def _reset(self):
        self.data = {
            'constraints':    {},
            'drivers':        [],
            'bone_properties':{},
            'custom_bones':   [],
            'bone_collections':[],
            'bone_layers':    {},
            'custom_shapes':  {},
        }

    def to_json(self):
        return json.dumps(self.data)

    def from_json(self, s):
        self.data = json.loads(s)

    def _get_rig(self):
        if self.target_rig and self.target_rig.type == 'ARMATURE':
            return self.target_rig
        return _find_generated_rig(self.metarig)

    # ── serialise one constraint ───────────────────────────────────────────

    def _serial_con(self, con):
        d = {'type': con.type, 'name': con.name, 'props': {}}
        for prop in dir(con):
            if prop.startswith('_'):
                continue
            if prop in ('bl_rna', 'rna_type', 'type', 'name',
                        'is_valid', 'error', 'is_proxy_local', 'id_data'):
                continue
            try:
                val = getattr(con, prop)
                if isinstance(val, (int, float, bool, str)):
                    d['props'][prop] = val
                elif hasattr(val, '__iter__') and not isinstance(val, str):
                    try:
                        lst = list(val)
                        # Only store if every item is a JSON primitive
                        if all(isinstance(item, (int, float, bool, str)) for item in lst):
                            d['props'][prop] = lst
                    except Exception:
                        pass
            except Exception:
                pass
        if hasattr(con, 'target') and con.target:
            d['target_name'] = con.target.name
        if hasattr(con, 'subtarget') and con.subtarget:
            d['subtarget'] = con.subtarget
        if hasattr(con, 'pole_target') and con.pole_target:
            d['pole_target_name'] = con.pole_target.name
        if hasattr(con, 'pole_subtarget') and con.pole_subtarget:
            d['pole_subtarget'] = con.pole_subtarget
        return d

    # ── store ──────────────────────────────────────────────────────────────

    def store_constraints(self, rig):
        result = {}
        for pb in rig.pose.bones:
            if pb.constraints:
                result[pb.name] = [self._serial_con(c) for c in pb.constraints]
        self.data['constraints'] = result

    def store_drivers(self, rig):
        if not rig.animation_data:
            return
        out = []
        for drv in rig.animation_data.drivers:
            dd = {
                'data_path':   drv.data_path,
                'array_index': drv.array_index,
                'expression':  drv.driver.expression if drv.driver else '',
                'variables':   [],
            }
            if drv.driver:
                for var in drv.driver.variables:
                    vd = {'name': var.name, 'type': var.type, 'targets': []}
                    for t in var.targets:
                        vd['targets'].append({
                            'id':              t.id.name if t.id else None,
                            'bone_target':     t.bone_target,
                            'transform_type':  t.transform_type,
                            'transform_space': t.transform_space,
                            'data_path':       t.data_path,
                        })
                    dd['variables'].append(vd)
            out.append(dd)
        self.data['drivers'] = out

    def store_bone_properties(self, rig):
        result = {}
        for pb in rig.pose.bones:
            props = {}
            for key in pb.keys():
                if key.startswith('rigify_'):
                    continue
                try:
                    val = pb[key]
                    if isinstance(val, (int, float, bool, str)):
                        props[key] = val
                    elif hasattr(val, '__iter__'):
                        props[key] = list(val)
                except Exception:
                    pass
            if props:
                result[pb.name] = props
        self.data['bone_properties'] = result

    def store_custom_bones(self, rig):
        """Store only user-added bones (not Rigify-generated ORG-/MCH-/DEF-/CTRL-).
        Roll is read from edit_bones — data bones have no roll attribute."""
        metarig_bones   = set(self.metarig.data.bones.keys())
        generated_pfx   = ('ORG-', 'MCH-', 'DEF-', 'CTRL-')
        custom_names    = [
            b.name for b in rig.data.bones
            if b.name not in metarig_bones and not b.name.startswith(generated_pfx)
        ]
        if not custom_names:
            return

        prev_active   = bpy.context.view_layer.objects.active

        # Must be in object mode before we can enter edit mode
        if bpy.context.object and bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        # The rig may live in an excluded / unlinked collection, so it is not in the
        # active view layer — then setting it active (and select_get / hide_get) raises
        # "ViewLayer does not contain object". Temporarily link it to the scene
        # collection and unhide it so we can read edit-bone rolls, then restore.
        _temp_linked = False
        if rig.name not in bpy.context.view_layer.objects:
            bpy.context.scene.collection.objects.link(rig)
            _temp_linked = True

        prev_selected = rig.select_get()
        _prev_hide_vp = rig.hide_viewport
        _prev_hide    = rig.hide_get()
        rig.hide_viewport = False
        rig.hide_set(False)

        try:
            bpy.context.view_layer.objects.active = rig
            rig.select_set(True)
            bpy.ops.object.mode_set(mode='EDIT')
            roll_map = {eb.name: eb.roll for eb in rig.data.edit_bones if eb.name in custom_names}
            bpy.ops.object.mode_set(mode='OBJECT')
        finally:
            rig.select_set(prev_selected)
            rig.hide_set(_prev_hide)
            rig.hide_viewport = _prev_hide_vp
            if _temp_linked:
                bpy.context.scene.collection.objects.unlink(rig)
            if prev_active is not None:
                bpy.context.view_layer.objects.active = prev_active

        out = []
        for bname in custom_names:
            bone  = rig.data.bones[bname]
            pb    = rig.pose.bones[bname]
            bd = {
                'name':        bname,
                'head':        list(bone.head_local),
                'tail':        list(bone.tail_local),
                'roll':        roll_map.get(bname, 0.0),
                'parent':      bone.parent.name if bone.parent else None,
                'use_connect': bool(bone.use_connect),
                'custom_shape': pb.custom_shape.name if pb.custom_shape else None,
                'custom_shape_scale': list(pb.custom_shape_scale_xyz)
                                      if hasattr(pb, 'custom_shape_scale_xyz') else [1.0, 1.0, 1.0],
                'constraints': [self._serial_con(c) for c in pb.constraints],
                'properties':  {},
            }
            for key in pb.keys():
                if not key.startswith('rigify_'):
                    try:
                        val = pb[key]
                        if isinstance(val, (int, float, bool, str)):
                            bd['properties'][key] = val
                    except Exception:
                        pass
            out.append(bd)
        self.data['custom_bones'] = out

    def store_bone_collections(self, rig):
        out = []
        arm = rig.data
        if hasattr(arm, 'collections_all'):
            for bc in arm.collections_all:
                out.append({
                    'name':       bc.name,
                    'is_visible': bc.is_visible if hasattr(bc, 'is_visible') else True,
                    'bones':      [b.name for b in bc.bones] if hasattr(bc, 'bones') else [],
                })
        self.data['bone_collections'] = out

    def store_bone_layers(self, rig):
        out = {}
        for pb in rig.pose.bones:
            if hasattr(pb.bone, 'collections'):
                cs = [c.name for c in pb.bone.collections]
                if cs:
                    out[pb.name] = cs
            elif hasattr(pb.bone, 'layers'):
                if any(pb.bone.layers[i] for i in range(8, 32)):
                    out[pb.name] = list(pb.bone.layers)
        self.data['bone_layers'] = out

    def store_custom_shapes(self, rig):
        out = {}
        for pb in rig.pose.bones:
            if not pb.custom_shape:
                continue
            so        = pb.custom_shape
            is_rigify = so.name.startswith('WGT-')
            sd = {
                'shape_name':     so.name,
                'scale':          list(pb.custom_shape_scale_xyz)
                                  if hasattr(pb, 'custom_shape_scale_xyz') else [1.0, 1.0, 1.0],
                'translation':    list(pb.custom_shape_translation)
                                  if hasattr(pb, 'custom_shape_translation') else [0.0, 0.0, 0.0],
                'rotation_euler': list(pb.custom_shape_rotation_euler)
                                  if hasattr(pb, 'custom_shape_rotation_euler') else [0.0, 0.0, 0.0],
                'is_rigify_widget': is_rigify,
                'mesh_data': None,
            }
            if not is_rigify and so.type == 'MESH':
                m = so.data
                sd['mesh_data'] = {
                    'vertices': [list(v.co)                     for v in m.vertices],
                    'edges':    [[e.vertices[0], e.vertices[1]] for e in m.edges],
                    'faces':    [list(f.vertices)               for f in m.polygons],
                }
            out[pb.name] = sd
        self.data['custom_shapes'] = out

    def backup(self):
        rig = self._get_rig()
        if not rig:
            return False, "No generated rig found"
        self._reset()
        self.store_constraints(rig)
        self.store_drivers(rig)
        self.store_bone_properties(rig)
        self.store_custom_bones(rig)
        self.store_bone_collections(rig)
        self.store_bone_layers(rig)
        self.store_custom_shapes(rig)
        n_con  = len(self.data['constraints'])
        n_bone = len(self.data['custom_bones'])
        return True, f"Backed up: {n_con} bones with constraints, {n_bone} custom bones"

    # ── restore ────────────────────────────────────────────────────────────

    def _apply_con(self, pb, cd):
        if cd['name'] in {c.name for c in pb.constraints}:
            return  # already there (Rigify regenerated it) — skip
        nc = pb.constraints.new(type=cd['type'])
        nc.name = cd['name']
        for prop, val in cd.get('props', {}).items():
            if not hasattr(nc, prop):
                continue
            try:
                setattr(nc, prop, val)
            except Exception:
                pass
        if 'target_name' in cd:
            obj = bpy.data.objects.get(cd['target_name'])
            if obj and hasattr(nc, 'target'):
                nc.target = obj
        if 'subtarget' in cd and hasattr(nc, 'subtarget'):
            nc.subtarget = cd['subtarget']
        if 'pole_target_name' in cd:
            obj = bpy.data.objects.get(cd['pole_target_name'])
            if obj and hasattr(nc, 'pole_target'):
                nc.pole_target = obj
        if 'pole_subtarget' in cd and hasattr(nc, 'pole_subtarget'):
            nc.pole_subtarget = cd['pole_subtarget']

    def restore_constraints(self, rig):
        for bname, cons in self.data['constraints'].items():
            if bname not in rig.pose.bones:
                continue
            pb = rig.pose.bones[bname]
            for cd in cons:
                self._apply_con(pb, cd)

    def restore_drivers(self, rig):
        if not self.data['drivers']:
            return
        if not rig.animation_data:
            rig.animation_data_create()
        for dd in self.data['drivers']:
            try:
                if rig.animation_data.drivers.find(dd['data_path']):
                    continue
                idx = dd['array_index']
                drv = (rig.driver_add(dd['data_path']) if idx < 0
                       else rig.driver_add(dd['data_path'], idx))
                if not drv or not drv.driver:
                    continue
                drv.driver.expression = dd['expression']
                for vd in dd['variables']:
                    nv = drv.driver.variables.new()
                    nv.name = vd['name']
                    nv.type = vd['type']
                    for i, td in enumerate(vd['targets']):
                        if i >= len(nv.targets):
                            break
                        t = nv.targets[i]
                        if td['id']:
                            t.id = bpy.data.objects.get(td['id'])
                        t.bone_target     = td['bone_target']
                        t.transform_type  = td['transform_type']
                        t.transform_space = td['transform_space']
                        t.data_path       = td['data_path']
            except Exception as e:
                dbg(f"[CustomRig] driver restore failed on {dd['data_path']}: {e}")

    def restore_bone_properties(self, rig):
        for bname, props in self.data['bone_properties'].items():
            if bname not in rig.pose.bones:
                continue
            pb = rig.pose.bones[bname]
            for key, val in props.items():
                try:
                    pb[key] = val
                except Exception:
                    pass

    def restore_custom_bones(self, rig):
        if not self.data['custom_bones']:
            return

        prev_active   = bpy.context.view_layer.objects.active

        if bpy.context.object and bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        # The rig may be in an excluded / unlinked collection (not in the active view
        # layer) — link & unhide it temporarily so we can enter edit mode, then restore.
        _temp_linked = False
        if rig.name not in bpy.context.view_layer.objects:
            bpy.context.scene.collection.objects.link(rig)
            _temp_linked = True

        prev_selected = rig.select_get()
        _prev_hide_vp = rig.hide_viewport
        _prev_hide    = rig.hide_get()
        rig.hide_viewport = False
        rig.hide_set(False)

        try:
            bpy.context.view_layer.objects.active = rig
            rig.select_set(True)
            bpy.ops.object.mode_set(mode='EDIT')
            for bd in self.data['custom_bones']:
                if bd['name'] in rig.data.edit_bones:
                    continue
                eb             = rig.data.edit_bones.new(bd['name'])
                eb.head        = bd['head']
                eb.tail        = bd['tail']
                eb.roll        = bd['roll']
                eb.use_connect = bd['use_connect']
                if bd['parent'] and bd['parent'] in rig.data.edit_bones:
                    eb.parent = rig.data.edit_bones[bd['parent']]
            bpy.ops.object.mode_set(mode='OBJECT')
        finally:
            rig.select_set(prev_selected)
            rig.hide_set(_prev_hide)
            rig.hide_viewport = _prev_hide_vp
            if _temp_linked:
                bpy.context.scene.collection.objects.unlink(rig)
            if prev_active is not None:
                bpy.context.view_layer.objects.active = prev_active

        for bd in self.data['custom_bones']:
            if bd['name'] not in rig.pose.bones:
                continue
            pb = rig.pose.bones[bd['name']]
            if bd['custom_shape']:
                so = bpy.data.objects.get(bd['custom_shape'])
                if so:
                    pb.custom_shape = so
            if hasattr(pb, 'custom_shape_scale_xyz'):
                pb.custom_shape_scale_xyz = bd['custom_shape_scale']
            for key, val in bd['properties'].items():
                try:
                    pb[key] = val
                except Exception:
                    pass
            for cd in bd['constraints']:
                self._apply_con(pb, cd)

    def restore_bone_collections(self, rig):
        arm = rig.data
        if not hasattr(arm, 'collections_all'):
            return
        for bcd in self.data['bone_collections']:
            if arm.collections_all.get(bcd['name']):
                continue
            if not hasattr(arm.collections, 'new'):
                continue
            nbc = arm.collections.new(bcd['name'])
            if hasattr(nbc, 'is_visible'):
                nbc.is_visible = bcd.get('is_visible', True)
            for bname in bcd.get('bones', []):
                if bname in arm.bones:
                    nbc.assign(arm.bones[bname])

    def restore_layers(self, rig):
        for bname, data in self.data['bone_layers'].items():
            if bname not in rig.pose.bones:
                continue
            pb = rig.pose.bones[bname]
            if isinstance(data, list) and data and isinstance(data[0], str):
                if hasattr(rig.data, 'collections_all'):
                    for cname in data:
                        bc = rig.data.collections_all.get(cname)
                        if bc:
                            bc.assign(pb)
            elif hasattr(pb.bone, 'layers'):
                pb.bone.layers = data

    def _recreate_widget(self, name, mesh_data):
        mesh = bpy.data.meshes.new(name=name)
        mesh.from_pydata(mesh_data['vertices'], mesh_data['edges'], mesh_data['faces'])
        mesh.update()
        obj = bpy.data.objects.new(name=name, object_data=mesh)
        wgt = next((c for c in bpy.data.collections if c.name.startswith('WGT-')), None)
        (wgt or bpy.context.scene.collection).objects.link(obj)
        return obj

    def restore_custom_shapes(self, rig):
        for bname, sd in self.data['custom_shapes'].items():
            if bname not in rig.pose.bones:
                continue
            pb = rig.pose.bones[bname]
            so = bpy.data.objects.get(sd['shape_name'])
            if not so:
                if sd.get('is_rigify_widget'):
                    continue
                if sd.get('mesh_data'):
                    so = self._recreate_widget(sd['shape_name'], sd['mesh_data'])
                else:
                    continue
            if so:
                pb.custom_shape = so
            if hasattr(pb, 'custom_shape_scale_xyz'):
                pb.custom_shape_scale_xyz = sd['scale']
            if hasattr(pb, 'custom_shape_translation'):
                pb.custom_shape_translation = sd['translation']
            if hasattr(pb, 'custom_shape_rotation_euler'):
                pb.custom_shape_rotation_euler = sd['rotation_euler']

    def restore(self):
        rig = self._get_rig()
        if not rig:
            return False, "No generated rig found"
        self.restore_constraints(rig)
        self.restore_drivers(rig)
        self.restore_bone_properties(rig)
        self.restore_custom_bones(rig)
        self.restore_bone_collections(rig)
        self.restore_layers(rig)
        self.restore_custom_shapes(rig)
        return True, f"Restored to: {rig.name}"


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class AUTORIG_OT_PreserveBackup(bpy.types.Operator):
    bl_idname   = "autorig.preserve_backup"
    bl_label    = "Backup Custom Setup"
    bl_description = "Backup constraints, custom bones, drivers and shapes from the generated rig"
    bl_options  = {'REGISTER'}  # no UNDO — backup is intentional

    def execute(self, context):
        props   = context.scene.autorig_preserve_props
        metarig = props.metarig or _find_metarig(context)
        if not metarig:
            self.report({'ERROR'}, "No metarig found — set one in the Metarig picker")
            return {'CANCELLED'}

        p = RigifyCustomPreserve(metarig, target_rig=props.target_rig)
        ok, msg = p.backup()
        if not ok:
            self.report({'WARNING'}, msg)
            return {'CANCELLED'}

        props.backup_json = p.to_json()
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class AUTORIG_OT_PreserveRestore(bpy.types.Operator):
    bl_idname   = "autorig.preserve_restore"
    bl_label    = "Restore Custom Setup"
    bl_description = "Restore backed-up constraints, bones, drivers and shapes to the generated rig"
    bl_options  = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props   = context.scene.autorig_preserve_props
        metarig = props.metarig or _find_metarig(context)
        if not metarig:
            self.report({'ERROR'}, "No metarig found — set one in the Metarig picker")
            return {'CANCELLED'}

        if not props.backup_json:
            self.report({'ERROR'}, "No backup found — run Backup Custom Setup first")
            return {'CANCELLED'}

        p = RigifyCustomPreserve(metarig, target_rig=props.target_rig)
        p.from_json(props.backup_json)
        ok, msg = p.restore()
        if not ok:
            self.report({'WARNING'}, msg)
            return {'CANCELLED'}

        self.report({'INFO'}, msg)
        return {'FINISHED'}


class AUTORIG_OT_PreserveGenerate(bpy.types.Operator):
    bl_idname   = "autorig.preserve_generate"
    bl_label    = "Regenerate & Preserve Custom"
    bl_description = ("Backup custom setup, run Rigify generate, then restore custom setup.\n"
                      "Use instead of Generate Rig when you have custom constraints / bones on the rig")
    bl_options  = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from .markers import FINGER_01_BONES

        props   = context.scene.autorig_preserve_props
        metarig = props.metarig or _find_metarig(context)
        if not metarig:
            self.report({'ERROR'}, "No metarig found — set one in the Metarig picker")
            return {'CANCELLED'}

        props = context.scene.autorig_preserve_props

        # 1 — backup
        p = RigifyCustomPreserve(metarig, target_rig=props.target_rig)
        ok, msg = p.backup()
        if ok:
            props.backup_json = p.to_json()
            dbg(f"[CustomRig] {msg}")
        else:
            dbg(f"[CustomRig] No existing rig to backup — generating fresh")
            p = None

        # 2 — set finger rotation axes (same as GenerateRig)
        context.view_layer.objects.active = metarig
        for bname in FINGER_01_BONES:
            pb = metarig.pose.bones.get(bname)
            if pb:
                try:
                    pb.rigify_parameters.primary_rotation_axis = 'X'
                except AttributeError:
                    pass

        # 3 — generate (pose mode required)
        bpy.ops.object.mode_set(mode='POSE')
        try:
            bpy.ops.pose.rigify_generate()
        except Exception as e:
            self.report({'ERROR'}, f"Rigify generate failed: {e}")
            return {'CANCELLED'}
        finally:
            bpy.ops.object.mode_set(mode='OBJECT')

        # 4 — restore
        if p:
            p2 = RigifyCustomPreserve(metarig, target_rig=props.target_rig)
            p2.from_json(props.backup_json)
            ok2, msg2 = p2.restore()
            if ok2:
                self.report({'INFO'}, f"Regenerated & preserved. {msg2}")
            else:
                self.report({'WARNING'}, f"Generated OK but restore failed: {msg2}")
        else:
            self.report({'INFO'}, "Fresh rig generated")

        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Property group
# ---------------------------------------------------------------------------

class AutoRigPreserveProps(bpy.types.PropertyGroup):
    metarig: bpy.props.PointerProperty(
        name="Metarig",
        description="The metarig to regenerate from (leave empty to auto-detect)",
        type=bpy.types.Object,
        poll=lambda self, obj: obj is not None and getattr(obj, 'type', None) == 'ARMATURE',
    )
    target_rig: bpy.props.PointerProperty(
        name="Generated Rig",
        description="The generated rig to backup from / restore to (leave empty to auto-detect)",
        type=bpy.types.Object,
        poll=lambda self, obj: obj is not None and getattr(obj, 'type', None) == 'ARMATURE',
    )
    backup_json: bpy.props.StringProperty(
        name="Backup Data",
        description="Serialised backup — do not edit manually",
        default="",
    )


# ---------------------------------------------------------------------------
# Exported symbols
# ---------------------------------------------------------------------------

PRESERVE_CLASSES = (
    AutoRigPreserveProps,
    AUTORIG_OT_PreserveBackup,
    AUTORIG_OT_PreserveRestore,
    AUTORIG_OT_PreserveGenerate,
)
