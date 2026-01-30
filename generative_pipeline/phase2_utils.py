import bpy
import numpy as np
from mathutils import *
import math
import glob
import os


def blender(synth_trajectory, FS=60, WLEN=3, pos_rot=True):
    WSZ = FS * WLEN

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Assume mocap_data is an array of shape (T, 24, 3)
    mocap_data = np.load('/home/igavier_umass_edu/Documents/synthetic-pipeline/P01G01R03F0792T0844A0402.npy')
    #mocap_data = np.load('/Users/igavier/Desktop/blender_output/31.npy')
    mocap_data *= np.random.randn() * 0.1 + 1.3
    mocap_data -= mocap_data[:,0:1]

    # Create a new armature
    bpy.ops.object.armature_add()
    armature = bpy.context.object
    armature.name = "MyArmature"

    # Remove the default bone
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.armature.select_all(action='SELECT')
    bpy.ops.armature.delete()

    # Create bones based on the given coordinates
    bones_data = [
        ("hips", 0, 0),
        ("leftleg1", 0, 1), ("leftleg2", 1, 4), ("leftleg3", 4, 7), ("leftleg4", 7, 10),
        ("rightleg1", 0, 2), ("rightleg2", 2, 5), ("rightleg3", 5, 8), ("rightleg4", 8, 11),
        ("spine1", 0, 3), ("spine2", 3, 6), ("spine3", 6, 9),
        ("head1", 9, 12), ("head2", 12, 15),
        ("leftarm1", 9, 13), ("leftarm2", 13, 16), ("leftarm3", 16, 18), ("leftarm4", 18, 20), ("leftarm5", 20, 22),
        ("rightarm1", 9, 14), ("rightarm2", 14, 17), ("rightarm3", 17, 19), ("rightarm4", 19, 21), ("rightarm5", 21, 23),
    ]

    # Create bones
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = armature.data.edit_bones

    bone_dict = {}  # Dictionary to store created bones for easy access

    for bone_name, head_index, tail_index in bones_data:
        new_bone = edit_bones.new(bone_name)
        if bone_name == "hips":
            new_bone.head = mocap_data[0, head_index] - Vector((0, 1, 0)) / 1000
        else:
            new_bone.head = mocap_data[0, head_index]
        new_bone.tail = mocap_data[0, tail_index]
        bone_dict[bone_name] = new_bone

    # Establish parent-child relationships
    parent_child_relations = [
        ("hips", "leftleg1"),
        ("hips", "rightleg1"),
        ("hips", "spine1"),
        ("spine3", "leftarm1"),
        ("spine3", "rightarm1"),
        ("spine3", "head1"),
        ("leftleg1", "leftleg2"),
        ("leftleg2", "leftleg3"),
        ("leftleg3", "leftleg4"),
        ("rightleg1", "rightleg2"),
        ("rightleg2", "rightleg3"),
        ("rightleg3", "rightleg4"),
        ("spine1", "spine2"),
        ("spine2", "spine3"),
        ("spine3", "head2"),
        ("spine3", "leftarm2"),
        ("leftarm1", "leftarm2"),
        ("leftarm2", "leftarm3"),
        ("leftarm3", "leftarm4"),
        ("leftarm4", "leftarm5"),
        ("spine3", "rightarm2"),
        ("rightarm1", "rightarm2"),
        ("rightarm2", "rightarm3"),
        ("rightarm3", "rightarm4"),
        ("rightarm4", "rightarm5"),
    ]

    for parent_name, child_name in parent_child_relations:
        bone_dict[child_name].parent = bone_dict[parent_name]

    bpy.ops.object.mode_set(mode='OBJECT')

    bones = armature.pose.bones

    # Create targets located at hands
    bpy.ops.object.empty_add(type='SPHERE', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    targetleft = bpy.context.object
    targetleft.location = bones['leftarm5'].matrix.translation
    targetleft.scale *= 0.05
    targetleft.name = "targetleft"

    bpy.ops.object.empty_add(type='SPHERE', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    targetright = bpy.context.object
    targetright.location = bones['rightarm5'].matrix.translation
    targetright.scale *= 0.05
    targetright.name = "targetright"

    # Define bone constraints
    for bone in bones:

        # add limit location
        bone.constraints.new("LIMIT_LOCATION")

        # change space to local
        bone.constraints["Limit Location"].owner_space = "LOCAL"

        # use min and max location
        bone.constraints["Limit Location"].use_min_x = True
        bone.constraints["Limit Location"].use_min_y = True
        bone.constraints["Limit Location"].use_min_z = True

        bone.constraints["Limit Location"].use_max_x = True
        bone.constraints["Limit Location"].use_max_y = True
        bone.constraints["Limit Location"].use_max_z = True

        # add limt rotation
        bone.constraints.new("LIMIT_ROTATION")

        # change space to local and mode to euler ZYX
        bone.constraints["Limit Rotation"].owner_space = "LOCAL"
        bone.constraints["Limit Rotation"].euler_order = 'ZYX'

        # use limit rotation
        bone.constraints["Limit Rotation"].use_limit_x = True
        bone.constraints["Limit Rotation"].use_limit_y = True
        bone.constraints["Limit Rotation"].use_limit_z = True

    # set limit rotation
    bones["spine1"].constraints["Limit Rotation"].min_x = math.radians(-15)
    bones["spine1"].constraints["Limit Rotation"].min_y = math.radians(-20)
    bones["spine1"].constraints["Limit Rotation"].min_z = math.radians(-45)
    bones["spine1"].constraints["Limit Rotation"].max_x = math.radians(85)
    bones["spine1"].constraints["Limit Rotation"].max_y = math.radians(20)
    bones["spine1"].constraints["Limit Rotation"].max_z = math.radians(45)

    bones["spine2"].constraints["Limit Rotation"].min_x = math.radians(-5)
    bones["spine2"].constraints["Limit Rotation"].min_y = math.radians(-5)
    bones["spine2"].constraints["Limit Rotation"].min_z = math.radians(-5)
    bones["spine2"].constraints["Limit Rotation"].max_x = math.radians(5)
    bones["spine2"].constraints["Limit Rotation"].max_y = math.radians(5)
    bones["spine2"].constraints["Limit Rotation"].max_z = math.radians(5)

    bones["spine3"].constraints["Limit Rotation"].min_x = math.radians(-5)
    bones["spine3"].constraints["Limit Rotation"].min_y = math.radians(-5)
    bones["spine3"].constraints["Limit Rotation"].min_z = math.radians(-5)
    bones["spine3"].constraints["Limit Rotation"].max_x = math.radians(5)
    bones["spine3"].constraints["Limit Rotation"].max_y = math.radians(5)
    bones["spine3"].constraints["Limit Rotation"].max_z = math.radians(5)

    bones["leftarm4"].constraints["Limit Rotation"].min_x = math.radians(-110)
    bones["leftarm4"].constraints["Limit Rotation"].min_y = math.radians(-90)
    bones["leftarm4"].constraints["Limit Rotation"].min_z = math.radians(0)
    bones["leftarm4"].constraints["Limit Rotation"].max_x = math.radians(20)
    bones["leftarm4"].constraints["Limit Rotation"].max_y = math.radians(90)
    bones["leftarm4"].constraints["Limit Rotation"].max_z = math.radians(0)

    bones["leftarm3"].constraints["Limit Rotation"].min_x = math.radians(-200)
    bones["leftarm3"].constraints["Limit Rotation"].min_y = math.radians(-90)
    bones["leftarm3"].constraints["Limit Rotation"].min_z = math.radians(-20)
    bones["leftarm3"].constraints["Limit Rotation"].max_x = math.radians(45)
    bones["leftarm3"].constraints["Limit Rotation"].max_y = math.radians(90)
    bones["leftarm3"].constraints["Limit Rotation"].max_z = math.radians(110)

    bones["leftarm2"].constraints["Limit Rotation"].min_x = math.radians(-20)
    bones["leftarm2"].constraints["Limit Rotation"].min_y = math.radians(-180)
    bones["leftarm2"].constraints["Limit Rotation"].min_z = math.radians(0)
    bones["leftarm2"].constraints["Limit Rotation"].max_x = math.radians(20)
    bones["leftarm2"].constraints["Limit Rotation"].max_y = math.radians(180)
    bones["leftarm2"].constraints["Limit Rotation"].max_z = math.radians(45)

    bones["leftarm1"].constraints["Limit Rotation"].min_x = math.radians(-5)
    bones["leftarm1"].constraints["Limit Rotation"].min_y = math.radians(-5)
    bones["leftarm1"].constraints["Limit Rotation"].min_z = math.radians(-5)
    bones["leftarm1"].constraints["Limit Rotation"].max_x = math.radians(5)
    bones["leftarm1"].constraints["Limit Rotation"].max_y = math.radians(5)
    bones["leftarm1"].constraints["Limit Rotation"].max_z = math.radians(5)

    bones["rightarm4"].constraints["Limit Rotation"].min_x = math.radians(-110)
    bones["rightarm4"].constraints["Limit Rotation"].min_y = math.radians(-90)
    bones["rightarm4"].constraints["Limit Rotation"].min_z = math.radians(0)
    bones["rightarm4"].constraints["Limit Rotation"].max_x = math.radians(20)
    bones["rightarm4"].constraints["Limit Rotation"].max_y = math.radians(90)
    bones["rightarm4"].constraints["Limit Rotation"].max_z = math.radians(0)

    bones["rightarm3"].constraints["Limit Rotation"].min_x = math.radians(-200)
    bones["rightarm3"].constraints["Limit Rotation"].min_y = math.radians(-90)
    bones["rightarm3"].constraints["Limit Rotation"].min_z = math.radians(-20)
    bones["rightarm3"].constraints["Limit Rotation"].max_x = math.radians(45)
    bones["rightarm3"].constraints["Limit Rotation"].max_y = math.radians(90)
    bones["rightarm3"].constraints["Limit Rotation"].max_z = math.radians(110)

    bones["rightarm2"].constraints["Limit Rotation"].min_x = math.radians(-20)
    bones["rightarm2"].constraints["Limit Rotation"].min_y = math.radians(-180)
    bones["rightarm2"].constraints["Limit Rotation"].min_z = math.radians(-45)
    bones["rightarm2"].constraints["Limit Rotation"].max_x = math.radians(20)
    bones["rightarm2"].constraints["Limit Rotation"].max_y = math.radians(180)
    bones["rightarm2"].constraints["Limit Rotation"].max_z = math.radians(0)

    bones["rightarm1"].constraints["Limit Rotation"].min_x = math.radians(-5)
    bones["rightarm1"].constraints["Limit Rotation"].min_y = math.radians(-5)
    bones["rightarm1"].constraints["Limit Rotation"].min_z = math.radians(-5)
    bones["rightarm1"].constraints["Limit Rotation"].max_x = math.radians(5)
    bones["rightarm1"].constraints["Limit Rotation"].max_y = math.radians(5)
    bones["rightarm1"].constraints["Limit Rotation"].max_z = math.radians(5)

    # Set IK
    bones["leftarm4"].constraints.new("IK")
    bones["leftarm4"].constraints["IK"].chain_count = 2
    bones["leftarm4"].constraints["IK"].target = targetleft
    bones["rightarm4"].constraints.new("IK")
    bones["rightarm4"].constraints["IK"].chain_count = 2
    bones["rightarm4"].constraints["IK"].target = targetright

    # Define initial and final positions
    # hand = np.random.choice(['left', 'right'])
    hand = ['left', 'right'][int(synth_trajectory[1])]
    hand_idle = ['left', 'right'][1-int(synth_trajectory[1])]

    #for hand in ['left', 'right']:
    is_p2p = np.random.rand() < 1.0
    target = targetleft if hand == 'left' else targetright
    target_idle = targetleft if hand_idle == 'left' else targetright

    if is_p2p:

        duration = min(0.95 * WLEN, 10 ** synth_trajectory[0])
        keyframes = (np.linspace(0,1,6) * duration * FS).astype(int)
        offsets = np.einsum('ij,nj->ni', np.array([[-1,0,0],[0,0,-1],[0,-1,0]]),
                            synth_trajectory[2:].reshape(3,6).T - np.array([[0,0,0.25]]))
        if np.any(1. * np.linalg.norm(offsets, axis=1) >
                  bones[f'{hand}arm3'].length + bones[f'{hand}arm4'].length):
            bones[f'leftarm4'].constraints["IK"].chain_count = 7
            bones[f'rightarm4'].constraints["IK"].chain_count = 7

        # Set initial and final keyframes
        keyframe_initial = max([2, 2 + np.random.randint(WSZ - 1 - int(FS * duration))])
        keyframe_final = min([WSZ - 1, int(FS * duration + keyframe_initial)])

        for i, (k, o) in enumerate(zip(keyframes, offsets)):
            target.location = bones[f'{hand}arm3'].matrix.translation + Vector(o)
            if i == 0: target.keyframe_insert("location", frame=1)
            target.keyframe_insert("location", frame=keyframe_initial+k)
            if i == 5: target.keyframe_insert("location", frame=WSZ)

    else:
        keyframe_initial, keyframe_final = WSZ // 2, WSZ // 2 + 1
        radii = np.random.uniform(0.3, 0.5, size=(2,))
        phis = np.random.uniform(0, 2 * np.pi, size=(2,))
        thetas = np.random.uniform(np.pi * 4 / 7, np.pi, size=(2,))
        offsets = radii[:,None] * np.stack([np.sin(thetas) * np.cos(phis),
                                            np.sin(thetas) * np.sin(phis),
                                            np.cos(thetas)], axis=1)
        offsets[-1] = offsets[0]


    # Set pre-point-to-point keyframes
    keyframes_pre = np.random.randint(1, keyframe_initial, size=keyframe_initial//20)
    keyframes_pre = np.sort(np.unique(keyframes_pre))[::-1]
    offset_curr = offsets[0]
    kf_curr = keyframe_initial
    for kfp in keyframes_pre:
        dur = 1. / FS * (kf_curr - kfp)
        offset_delta = dur ** (0.33) * 10 ** (- 3 + .6 * np.random.randn(3))
        offset_curr += offset_delta
        offset_curr *= 0.5 / min(0.5, np.linalg.norm(offset_curr))
        target.location = bones[f'{hand}arm3'].matrix.translation + Vector(offset_curr)
        target.keyframe_insert("location", frame=kfp)
        kf_curr = kfp
    # Set post-point-to-point keyframes
    keyframes_pos = np.random.randint(keyframe_final + 1, WSZ + 1, size=(WSZ - keyframe_final)//20)
    keyframes_pos = np.sort(np.unique(keyframes_pos))
    offset_curr = offsets[-1]
    kf_curr = keyframe_final
    for kfp in keyframes_pos:
        dur = 1. / FS * (kfp - kf_curr)
        offset_delta = dur ** (0.33) * 10 ** (- 3 + .6 * np.random.randn(3))
        offset_curr += offset_delta
        offset_curr *= 0.5 / min(0.5, np.linalg.norm(offset_curr))
        target.location = bones[f'{hand}arm3'].matrix.translation + Vector(offset_curr)
        target.keyframe_insert("location", frame=kfp)
        kf_curr = kfp



    if is_p2p:

        duration = min(0.95 * WLEN, 10 ** synth_trajectory[0])
        keyframes = (np.linspace(0,1,6) * duration * FS).astype(int)
        offsets = np.einsum('ij,nj->ni', np.array([[-1,0,0],[0,0,-1],[0,-1,0]]),
                            synth_trajectory[2:].reshape(3,6).T * np.array([[-1,1,1]]) - np.array([[0,0,0.25]]))
        if np.any(1. * np.linalg.norm(offsets, axis=1) >
                  bones[f'{hand_idle}arm3'].length + bones[f'{hand_idle}arm4'].length):
            bones[f'leftarm4'].constraints["IK"].chain_count = 7
            bones[f'rightarm4'].constraints["IK"].chain_count = 7

        # Set initial and final keyframes
        keyframe_initial = max([2, 2 + np.random.randint(WSZ - 1 - int(FS * duration))])
        keyframe_final = min([WSZ - 1, int(FS * duration + keyframe_initial)])

        for i, (k, o) in enumerate(zip(keyframes, offsets)):
            target_idle.location = bones[f'{hand_idle}arm3'].matrix.translation + Vector(o)
            if i == 0: target_idle.keyframe_insert("location", frame=1)
            target_idle.keyframe_insert("location", frame=keyframe_initial+k)
            if i == 5: target_idle.keyframe_insert("location", frame=WSZ)


    else:
        keyframe_initial, keyframe_final = WSZ // 2, WSZ // 2 + 1
        radii = np.random.uniform(0.3, 0.5, size=(2,))
        phis = np.random.uniform(0, 2 * np.pi, size=(2,))
        thetas = np.random.uniform(np.pi * 4 / 7, np.pi, size=(2,))
        offsets = radii[:,None] * np.stack([np.sin(thetas) * np.cos(phis),
                                            np.sin(thetas) * np.sin(phis),
                                            np.cos(thetas)], axis=1)
        offsets[-1] = offsets[0]


    # Set pre-point-to-point keyframes
    keyframes_pre = np.random.randint(1, keyframe_initial, size=keyframe_initial//20)
    keyframes_pre = np.sort(np.unique(keyframes_pre))[::-1]
    offset_curr = offsets[0]
    kf_curr = keyframe_initial
    for kfp in keyframes_pre:
        dur = 1. / FS * (kf_curr - kfp)
        offset_delta = dur ** (0.33) * 10 ** (- 3 + .6 * np.random.randn(3))
        offset_curr += offset_delta
        offset_curr *= 0.5 / min(0.5, np.linalg.norm(offset_curr))
        target_idle.location = bones[f'{hand_idle}arm3'].matrix.translation + Vector(offset_curr)
        target_idle.keyframe_insert("location", frame=kfp)
        kf_curr = kfp
    # Set post-point-to-point keyframes
    keyframes_pos = np.random.randint(keyframe_final + 1, WSZ + 1, size=(WSZ - keyframe_final)//20)
    keyframes_pos = np.sort(np.unique(keyframes_pos))
    offset_curr = offsets[-1]
    kf_curr = keyframe_final
    for kfp in keyframes_pos:
        dur = 1. / FS * (kfp - kf_curr)
        offset_delta = dur ** (0.33) * 10 ** (- 3 + .6 * np.random.randn(3))
        offset_curr += offset_delta
        offset_curr *= 0.5 / min(0.5, np.linalg.norm(offset_curr))
        target_idle.location = bones[f'{hand_idle}arm3'].matrix.translation + Vector(offset_curr)
        target_idle.keyframe_insert("location", frame=kfp)
        kf_curr = kfp

    # # Set automatic handle
    # bpy.context.view_layer.objects.active = target
    # bpy.ops.object.select_all(action='DESELECT')
    # target.select_set(True)
    # bpy.context.area.type = 'GRAPH_EDITOR'
    # bpy.ops.graph.handle_type(type='AUTO')
    # bpy.context.area.type = 'TEXT_EDITOR'

    # Export motion
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.select_all(action='DESELECT')
    armature.select_set(True)
    
    
    # Construct array of left pos/rot, right pos/rot
    new_mocap_data = []
    
    for frame in range(1, WSZ + 1):
        bpy.context.scene.frame_set(frame)
        bpy.ops.object.mode_set(mode='POSE')

        # Update the armature to get the latest pose
        bpy.context.view_layer.update()

        # Insert keyframes for all bones
        
        if pos_rot:
            bones['leftarm4'].bone.select = True
            left_pos = np.array(bones['leftarm4'].tail)
            left_vx = np.array(bones['leftarm4'].y_axis)
            left_vy = - np.array(bones['leftarm4'].z_axis)
            left_vz = np.array(bones['leftarm4'].x_axis)
            bones['rightarm4'].bone.select = True
            right_pos = np.array(bones['rightarm4'].tail)
            right_vx = np.array(bones['rightarm4'].y_axis)
            right_vy = np.array(bones['rightarm4'].z_axis)
            right_vz = - np.array(bones['rightarm4'].x_axis)

            new_mocap_data += [np.concatenate([left_pos, left_vx, left_vy, left_vz,
                                               right_pos, right_vx, right_vy, right_vz])]
        
        else:
            bones['leftarm4'].bone.select = True
            left_pos = np.array(bones['leftarm4'].tail)
            bones['leftarm3'].bone.select = True
            left_elbpos = np.array(bones['leftarm3'].tail)
            bones['leftarm2'].bone.select = True
            left_shldpos = np.array(bones['leftarm2'].tail)
            bones['rightarm4'].bone.select = True
            right_pos = np.array(bones['rightarm4'].tail)
            bones['rightarm3'].bone.select = True
            right_elbpos = np.array(bones['rightarm3'].tail)
            bones['rightarm2'].bone.select = True
            right_shldpos = np.array(bones['rightarm2'].tail)

            new_mocap_data += [np.concatenate([left_pos, left_elbpos, left_shldpos,
                                               right_pos, right_elbpos, right_shldpos])]
    
    new_mocap_data = np.array(new_mocap_data)
    
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')

    return hand, new_mocap_data