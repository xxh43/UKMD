
import argparse
import os
import numpy as np
import math
from deform import *
from models import *
from pytorch3d.loss import chamfer_distance 

import torch.nn as nn
import torch.nn.functional as F
import torch
import joblib
import os
from pathlib import Path
from util_motion import *
from config import *
from util_vis import *

from scipy.spatial.distance import cdist
from partnet import *
from scipy.spatial.transform import Rotation as R
import numpy as np
from objects import *
np.set_printoptions(precision=10)


def get_joint_pcs(joints, option='mov'):
    pcs = []
    for i in range(0, len(joints)):
        if option == 'mov':
            pc = joints[i].mov_pc
        else:
            pc = joints[i].base_pc
        pc = to_tensor(pc)
        pcs.append(pc)
    pcs = torch.stack(pcs)
    return pcs

def pc_to_bbox_collision_distances(pcs, handle_zs, handle_ts, handle_rotmats):
    standarized_pcs = pcs - handle_ts.unsqueeze(dim=1)
    standarized_pcs = torch.bmm(handle_rotmats.inverse(), standarized_pcs.transpose(1,2)).transpose(1,2)
    repeated_handle_zs = handle_zs.unsqueeze(dim=1).repeat_interleave(standarized_pcs.shape[1], dim=1)
    dists = repeated_handle_zs - torch.abs(standarized_pcs) 
    threshold = 0.001
    dists[dists > threshold] = dists[dists > threshold]
    dists[dists < threshold] = -99999
    final_dists = torch.relu(torch.min(dists, dim=2).values)
    final_dists = final_dists * final_dists
    final_dists = torch.sum(final_dists, dim=1)
    return final_dists

def pc_to_bbox_detach_distances(pcs, handle_zs, handle_ts, handle_rotmats, is_debug=False):
    standarized_pcs = pcs - handle_ts.unsqueeze(dim=1)
    standarized_pcs = torch.bmm(handle_rotmats.inverse(), standarized_pcs.transpose(1,2)).transpose(1,2) 
    repeated_handle_zs = handle_zs.unsqueeze(dim=1).repeat_interleave(standarized_pcs.shape[1], dim=1)
    dists = torch.abs(standarized_pcs) - repeated_handle_zs 
    dists = torch.relu(dists)
    dists = dists * dists
    dists = dists.sum(dim=2)
    k = min(50, dists.shape[1])
    top_k_dists = torch.topk(dists, k, dim=1, largest=False).values
    dists = torch.mean(top_k_dists, dim=1)
    return dists

def pc_to_pc_detach_distances(mov_pcs, base_pcs, is_debug=False):
    r = 0.01
    k = 10
    distmovtobase = torch.cdist(mov_pcs, base_pcs)
    min_dists = torch.min(distmovtobase, dim=2).values
    top_k_min_dists = torch.topk(min_dists, k, dim=1, largest=False).values
    detach_penalty = torch.relu(torch.sum(top_k_min_dists, dim=1)-r*k)
    return detach_penalty

def transform_handle(handle_ts, handle_rotmats, local_homo_rot_mats, local_trans_vecs):    
    transformed_handle_ts = handle_ts.unsqueeze(dim=1)
    transformed_handle_ts = transform_with_homo_matrix_batched(transformed_handle_ts, local_homo_rot_mats)
    transformed_handle_ts = translate_with_vector_batched(transformed_handle_ts, local_trans_vecs)
    transformed_handle_rotmats = handle_rotmats
    transformed_handle_rotmats = torch.bmm(local_homo_rot_mats[:, 0:3, 0:3], transformed_handle_rotmats)
    return transformed_handle_ts.squeeze(dim=1), transformed_handle_rotmats

def global_transformation(pcs, rigid_trans_vectors, rigid_angles):
    rigid_axes = torch.tensor([[0.0, 1.0, 0.0]], device='cuda:0').repeat_interleave(len(pcs), dim=0)
    rigid_centers = torch.tensor([[0.0, 0.0, 0.0]], device='cuda:0').repeat_interleave(len(pcs), dim=0)
    transformed_pcs, rot_mats = rotate_with_axis_center_angle_batched(pcs, rigid_axes, torch.mean(pcs, dim=1), rigid_angles)
    transformed_pcs = translate_with_vector_batched(transformed_pcs, rigid_trans_vectors)
    return transformed_pcs, rot_mats, rigid_trans_vectors

def pack_inputs(joints, group_size):
    
    group_count = int(len(joints)/group_size)
    all_mov_pcs = get_joint_pcs(joints, 'mov')   
    all_base_pcs = get_joint_pcs(joints, 'base')
    ref_indices = list(range(0, len(joints), group_size))
    mov_templates = all_mov_pcs[ref_indices].repeat_interleave(group_size-1, dim=0)
    base_templates = all_base_pcs[ref_indices].repeat_interleave(group_size-1, dim=0)
    
    mov_pcs = []
    source_mov_pc_comp_nums = []
    target_mov_pc_comp_nums = []
    source_mov_centers = []
    mov_scales = []
    source_mov_scales = []
    ref_mov_pcs = []
    ref_mov_handle_ts = []
    ref_mov_handle_rotmats = []
    ref_mov_handle_zs = []

    for i in range(len(all_mov_pcs)):
        if i % group_size != 0:
            mov_pcs.append(all_mov_pcs[i])
            target_mov_pc_comp_nums.append(joints[i].get_mov_comp_num())
            mov_scales.append(torch.tensor(np.power(np.linalg.norm(joints[i].mov_deform_handle.z), 2), device='cuda:0', dtype=torch.float))
        else:
            ref_mov_pcs.append(all_mov_pcs[i])
            source_mov_pc_comp_nums.append(torch.tensor(joints[i].get_mov_comp_num()))
            source_mov_centers.append(torch.tensor(joints[i].get_mov_center()))
            source_mov_scales.append(torch.tensor(np.power(np.linalg.norm(joints[i].mov_deform_handle.z), 2), device='cuda:0', dtype=torch.float))
            ref_mov_handle_ts.append(torch.tensor(joints[i].mov_deform_handle.t, device='cuda:0', dtype=torch.float))
            ref_mov_handle_rotmats.append(torch.tensor(joints[i].mov_deform_handle.rot_mat, device='cuda:0', dtype=torch.float))
            ref_mov_handle_zs.append(torch.tensor(joints[i].mov_deform_handle.z, device='cuda:0', dtype=torch.float))
    mov_pcs = torch.stack(mov_pcs)
    mov_scales = torch.stack(mov_scales)
    ref_mov_pcs = torch.stack(ref_mov_pcs)
    ref_mov_handle_ts = torch.stack(ref_mov_handle_ts)
    source_mov_pc_comp_nums = to_numpy(torch.stack(source_mov_pc_comp_nums).repeat_interleave(group_size-1, dim=0))
    source_mov_scales = torch.stack(source_mov_scales).repeat_interleave(group_size-1, dim=0)
    source_mov_centers = torch.stack(source_mov_centers).repeat_interleave(group_size-1, dim=0)

    ref_mov_handle_rotmats = torch.stack(ref_mov_handle_rotmats)
    ref_mov_handle_zs = torch.stack(ref_mov_handle_zs)

    base_pcs = []
    source_base_pc_comp_nums = []
    target_base_pc_comp_nums = []
    base_scales = []
    source_base_scales = []
    ref_base_pcs = []
    ref_base_handle_ts = []
    ref_base_handle_rotmats = []
    ref_base_handle_zs = []
    for i in range(len(all_base_pcs)):
        if i % group_size != 0:
            base_pcs.append(all_base_pcs[i])
            target_base_pc_comp_nums.append(joints[i].get_base_comp_num())
            base_scales.append(torch.tensor(np.power(np.linalg.norm(joints[i].base_deform_handle.z), 2), device='cuda:0', dtype=torch.float))
        else:
            ref_base_pcs.append(all_base_pcs[i])
            source_base_scales.append(torch.tensor(np.power(np.linalg.norm(joints[i].base_deform_handle.z), 2), device='cuda:0', dtype=torch.float))
            source_base_pc_comp_nums.append(torch.tensor(joints[i].get_base_comp_num()))
            ref_base_handle_ts.append(torch.tensor(joints[i].base_deform_handle.t, device='cuda:0', dtype=torch.float))
            ref_base_handle_rotmats.append(torch.tensor(joints[i].base_deform_handle.rot_mat, device='cuda:0', dtype=torch.float))
            ref_base_handle_zs.append(torch.tensor(joints[i].base_deform_handle.z, device='cuda:0', dtype=torch.float))
    base_pcs = torch.stack(base_pcs)
    base_scales = torch.stack(base_scales)
    source_base_scales = torch.stack(source_base_scales).repeat_interleave(group_size-1, dim=0)
    ref_base_pcs = torch.stack(ref_base_pcs)
    ref_base_handle_ts = torch.stack(ref_base_handle_ts)
    ref_base_handle_rotmats = torch.stack(ref_base_handle_rotmats)
    ref_base_handle_zs = torch.stack(ref_base_handle_zs)
    source_base_pc_comp_nums = to_numpy(torch.stack(source_base_pc_comp_nums).repeat_interleave(group_size-1, dim=0))

    mov_handle_ts = []
    mov_handle_rotmats = []
    mov_handle_zs = []
    base_handle_ts = []
    base_handle_rotmats = []
    base_handle_zs = []

    for i, joint in enumerate(joints):
        if i % group_size == 0:
            for j in range(group_size-1):
                mov_handle_ts.append(torch.tensor(joint.mov_deform_handle.t, device='cuda:0', dtype=torch.float))
                mov_handle_rotmats.append(torch.tensor(joint.mov_deform_handle.rot_mat, device='cuda:0', dtype=torch.float))
                mov_handle_zs.append(torch.tensor(joint.mov_deform_handle.z, device='cuda:0', dtype=torch.float))
                base_handle_ts.append(torch.tensor(joint.base_deform_handle.t, device='cuda:0', dtype=torch.float))
                base_handle_rotmats.append(torch.tensor(joint.base_deform_handle.rot_mat, device='cuda:0', dtype=torch.float))
                base_handle_zs.append(torch.tensor(joint.base_deform_handle.z, device='cuda:0', dtype=torch.float))
    mov_handle_ts = torch.stack(mov_handle_ts)
    mov_handle_rotmats = torch.stack(mov_handle_rotmats)
    mov_handle_zs = torch.stack(mov_handle_zs)
    base_handle_ts = torch.stack(base_handle_ts)
    base_handle_rotmats = torch.stack(base_handle_rotmats)
    base_handle_zs = torch.stack(base_handle_zs)

    inputs = dict()
    inputs['group_count'] = group_count
    inputs['group_size'] = group_size
    inputs['source_mov_pcs'] = mov_templates
    inputs['source_mov_pc_comp_nums'] = source_mov_pc_comp_nums
    inputs['ref_mov_pcs'] = ref_mov_pcs
    inputs['target_mov_pcs'] = mov_pcs
    inputs['target_mov_pc_comp_nums'] = target_mov_pc_comp_nums
    inputs['mov_scales'] = mov_scales
    inputs['source_mov_scales'] = source_mov_scales
    inputs['source_mov_handle_ts'] = ref_mov_handle_ts
    inputs['source_mov_handle_rots'] = ref_mov_handle_rotmats
    inputs['source_mov_handle_zs'] = ref_mov_handle_zs
    inputs['source_mov_centers'] = source_mov_centers
    inputs['mov_handle_ts'] = mov_handle_ts
    inputs['mov_handle_rotmats'] = mov_handle_rotmats
    inputs['mov_handle_zs'] = mov_handle_zs
    inputs['source_base_pcs'] = base_templates
    inputs['source_base_pc_comp_nums'] = source_base_pc_comp_nums
    inputs['ref_base_pcs'] = ref_base_pcs
    inputs['target_base_pcs'] = base_pcs
    inputs['target_base_pc_comp_nums'] = target_base_pc_comp_nums
    inputs['base_scales'] = base_scales
    inputs['source_base_scales'] = source_base_scales
    inputs['source_base_handle_ts'] = ref_base_handle_ts
    inputs['source_base_handle_rots'] = ref_base_handle_rotmats
    inputs['source_base_handle_zs'] = ref_base_handle_zs
    inputs['base_handle_ts'] = base_handle_ts
    inputs['base_handle_rotmats'] = base_handle_rotmats
    inputs['base_handle_zs'] = base_handle_zs

    mov_contact_indices = []
    base_contact_indices = []
    for i in range(len(joints)):
        if i % group_size == 0:
            mov_contact_index, base_contact_index = joints[i].get_contact_indices()
            mov_contact_indices.append(mov_contact_index)
            base_contact_indices.append(base_contact_index)

    inputs['mov_contact_indices'] = mov_contact_indices
    inputs['base_contact_indices'] = base_contact_indices

    return inputs

def postprocess_dir(axis, mov_handle_t, mov_handle_z, mov_handle_rotmat, base_handle_t, base_handle_z, base_handle_rotmat):

    axis = to_numpy(axis)
    axis = axis/np.linalg.norm(axis)
    mov_handle_t = to_numpy(mov_handle_t)
    mov_handle_z = to_numpy(mov_handle_z)
    mov_handle_rotmat = to_numpy(mov_handle_rotmat)
    base_handle_t = to_numpy(base_handle_t)
    base_handle_z = to_numpy(base_handle_z)
    base_handle_rotmat = to_numpy(base_handle_rotmat)
    
    axis_candidates = []

    world_up = np.array([0, 1, 0])
    to_world_up_distance = abs(np.dot(axis, world_up))
    axis_candidates.append((to_world_up_distance, world_up))

    world_right = np.array([1, 0, 0])
    to_world_right_distance = abs(np.dot(axis, world_right))
    axis_candidates.append((to_world_right_distance, world_right))

    world_front = np.array([0, 0, 1])
    to_world_front_distance = abs(np.dot(axis, world_front))
    axis_candidates.append((to_world_front_distance, world_front))

    mov_frame, mov_frame_loc, mov_frame_size = mov_handle_rotmat, mov_handle_t, mov_handle_z*2
    
    mov_axis0 = mov_frame[:, 0]
    to_mov_axis0_distance = abs(np.dot(axis, mov_axis0))
    axis_candidates.append((to_mov_axis0_distance, mov_axis0))
    
    mov_axis1 = mov_frame[:, 1]
    to_mov_axis1_distance = abs(np.dot(axis, mov_axis1))
    axis_candidates.append((to_mov_axis1_distance, mov_axis1))

    mov_axis2 = mov_frame[:, 2]
    to_mov_axis2_distance = abs(np.dot(axis, mov_axis2))
    axis_candidates.append((to_mov_axis2_distance, mov_axis2))

    base_frame, base_frame_loc, base_frame_size = base_handle_rotmat, base_handle_t, base_handle_z*2

    base_axis0 = base_frame[:, 0]
    to_base_axis0_distance = abs(np.dot(axis, base_axis0))
    axis_candidates.append((to_base_axis0_distance, base_axis0))
    
    base_axis1 = base_frame[:, 1]
    to_base_axis1_distance = abs(np.dot(axis, base_axis1))
    axis_candidates.append((to_base_axis1_distance, base_axis1))
    
    base_axis2 = base_frame[:, 2]
    to_base_axis2_distance = abs(np.dot(axis, base_axis2))
    axis_candidates.append((to_base_axis2_distance, base_axis2))

    ret_axis = None
    max_distance = 0
    for j in range(len(axis_candidates)):
        if axis_candidates[j][0] > max_distance:
            max_distance = axis_candidates[j][0]
            ret_axis = axis_candidates[j][1]    

    if max_distance > 0.975:
        ret_axis = ret_axis
    else:
        ret_axis = axis

    return ret_axis


def postprocess_center(center, axis, mov_pc, base_pc, mov_handle_t, mov_handle_z, mov_handle_rotmat, base_handle_t, base_handle_z, base_handle_rotmat):
    
    axis = to_numpy(axis)
    axis = axis/np.linalg.norm(axis)
    
    center = to_numpy(center)
    mov_pc = to_numpy(mov_pc)
    base_pc = to_numpy(base_pc)
    mov_handle_t = to_numpy(mov_handle_t)
    mov_handle_z = to_numpy(mov_handle_z)
    mov_handle_rotmat = to_numpy(mov_handle_rotmat)
    base_handle_t = to_numpy(base_handle_t)
    base_handle_z = to_numpy(base_handle_z)
    base_handle_rotmat = to_numpy(base_handle_rotmat)

    mov_deform_handle = DeformHandle()
    mov_deform_handle.t = mov_handle_t
    mov_deform_handle.z = mov_handle_z
    mov_deform_handle.rot_mat = mov_handle_rotmat
    
    mov_frame, mov_frame_loc, mov_frame_size = mov_handle_rotmat, mov_handle_t, mov_handle_z*2
    base_frame, base_frame_loc, base_frame_size = base_handle_rotmat, base_handle_t, base_handle_z*2

    contact_eps = (np.mean(mov_frame_size) + np.mean(base_frame_size)) * 0.5 * 0.1
    dist = cdist(mov_pc, base_pc)
    flat_dist = dist.flatten()
    contact_indices = np.where(flat_dist < contact_eps)[0].tolist()
    
    to_contact_distance = np.inf
    final_contact_point = None
    
    for j in range(len(contact_indices)):
        mov_i = contact_indices[j] // base_pc.shape[0]
        base_i = contact_indices[j] % base_pc.shape[0]
        contact_point = 0.5 * (base_pc[base_i] + mov_pc[mov_i])
        temp = center - contact_point
        temp_contact_distance = np.linalg.norm(temp - np.dot(temp, axis))
        if temp_contact_distance < to_contact_distance:
            to_contact_distance = temp_contact_distance
            final_contact_point = contact_point

    center_candidates = [(to_contact_distance, final_contact_point)]

    mov_face_centers = mov_deform_handle.get_face_centers()
    mov_center = mov_deform_handle.get_center()
    mov_key_points = [mov_center] + mov_face_centers

    for key_point in mov_key_points:
        temp = center - key_point
        to_key_point_distance = np.linalg.norm(temp - np.dot(temp, axis))
        center_candidates.append((to_key_point_distance, key_point))

    ret_center = None
    min_distance = np.inf
    for j in range(len(center_candidates)):
        if center_candidates[j][0] < min_distance:
            min_distance = center_candidates[j][0]
            ret_center = center_candidates[j][1]

    if min_distance < (np.mean(mov_frame_size) + np.mean(base_frame_size)) * 0.5 * 0.05:
        ret_center = ret_center
    else:
        ret_center = center

    return ret_center


def global_pre_transformation(pcs, rigid_axis, rigid_trans_vectors, rigid_angles):
    rigid_axes = torch.tensor([rigid_axis], device='cuda:0').repeat_interleave(len(pcs), dim=0)
    rigid_centers = torch.tensor([[0.0, 0.0, 0.0]], device='cuda:0').repeat_interleave(len(pcs), dim=0)
    transformed_pcs, rot_mats = rotate_with_axis_center_angle_batched(pcs, rigid_axes, rigid_centers, rigid_angles)
    transformed_pcs = translate_with_vector_batched(transformed_pcs, rigid_trans_vectors)
    return transformed_pcs, rot_mats, rigid_trans_vectors


def global_pre_align(mov_templates, base_templates, mov_pcs, base_pcs, target_mov_scales, target_base_scales, option):

    if option == 0:
        return mov_pcs, base_pcs

    joint_to_transformed_mov_pcs = {}
    joint_to_transformed_base_pcs = {}
    axis_candidates = [ [0.0, 1.0, 0.0]]
    range_candidates = [0, 0.5*np.pi, 1.0*np.pi, 1.5*np.pi]

    if option == 1:
        for axis_index in range(len(axis_candidates)):
            for range_index in range(len(range_candidates)):

                global_translation_vectors = torch.zeros((len(mov_pcs), 3), dtype=torch.float, device="cuda:0", requires_grad=True)
                global_rotation_ranges = torch.tensor(torch.tensor(range_candidates[range_index]).repeat(len(mov_pcs)), dtype=torch.float, device="cuda:0", requires_grad=True)
                optimizer = torch.optim.Adam([global_translation_vectors]+[global_rotation_ranges], lr=0.01)

                for epoch in range(0, global_pre_align_max_iteration):
                    
                    print('global pre align epoch', epoch)

                    transformed_full_pcs, _, _ = global_pre_transformation(torch.cat((mov_pcs, base_pcs), dim=1), axis_candidates[axis_index], global_translation_vectors, global_rotation_ranges)
                    
                    transformed_mov_pcs = transformed_full_pcs[:, 0:mov_pcs.shape[1]]
                    transformed_base_pcs = transformed_full_pcs[:, mov_pcs.shape[1]:]

                    mov_recon_losses, _ = chamfer_distance(transformed_mov_pcs, mov_templates, batch_reduction=None)
                    base_recon_losses, _ = chamfer_distance(transformed_base_pcs, base_templates, batch_reduction=None)
                    losses = base_recon_losses/target_base_scales

                    loss = torch.mean(losses)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    print('loss', loss)

                for joint_index in range(len(transformed_full_pcs)):
                    if joint_index not in joint_to_transformed_mov_pcs:
                        joint_to_transformed_mov_pcs[joint_index] = (losses[joint_index], transformed_mov_pcs[joint_index].detach())
                    else:
                        if losses[joint_index] < joint_to_transformed_mov_pcs[joint_index][0]:
                            joint_to_transformed_mov_pcs[joint_index] = (losses[joint_index], transformed_mov_pcs[joint_index].detach())

                for joint_index in range(len(transformed_full_pcs)):
                    if joint_index not in joint_to_transformed_base_pcs:
                        joint_to_transformed_base_pcs[joint_index] = (losses[joint_index], transformed_base_pcs[joint_index].detach())
                    else:
                        if losses[joint_index] < joint_to_transformed_base_pcs[joint_index][0]:
                            joint_to_transformed_base_pcs[joint_index] = (losses[joint_index], transformed_base_pcs[joint_index].detach())
        
        temp_mov_pcs = []
        for joint_index in range(len(mov_pcs)):
            temp_mov_pcs.append(joint_to_transformed_mov_pcs[joint_index][1])
        temp_mov_pcs = torch.stack(temp_mov_pcs)
        temp_base_pcs = []
        for joint_index in range(len(base_pcs)):
            temp_base_pcs.append(joint_to_transformed_base_pcs[joint_index][1])
        temp_base_pcs = torch.stack(temp_base_pcs)


        return temp_mov_pcs, temp_base_pcs

    if option == 2:

        for axis_index in range(len(axis_candidates)):
            for range_index in range(len(range_candidates)):

                global_translation_vectors = torch.zeros((len(mov_pcs), 3), dtype=torch.float, device="cuda:0", requires_grad=True)
                global_rotation_ranges = torch.tensor(torch.tensor(range_candidates[range_index]).repeat(len(mov_pcs)), dtype=torch.float, device="cuda:0", requires_grad=True)
                optimizer = torch.optim.Adam([global_translation_vectors]+[global_rotation_ranges], lr=0.01)

                for epoch in range(0, global_pre_align_max_iteration):
                    
                    print('global pre align epoch', epoch)

                    transformed_full_pcs, _, _ = global_pre_transformation(torch.cat((mov_pcs, base_pcs), dim=1), axis_candidates[axis_index], global_translation_vectors, global_rotation_ranges)
                    
                    transformed_mov_pcs = transformed_full_pcs[:, 0:mov_pcs.shape[1]]
                    transformed_base_pcs = transformed_full_pcs[:, mov_pcs.shape[1]:]

                    mov_recon_losses, _ = chamfer_distance(transformed_mov_pcs, mov_templates, batch_reduction=None)
                    base_recon_losses, _ = chamfer_distance(transformed_base_pcs, base_templates, batch_reduction=None)

                    losses = mov_recon_losses/target_mov_scales

                    loss = torch.mean(losses)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    print('loss', loss)

                for joint_index in range(len(transformed_full_pcs)):
                    if joint_index not in joint_to_transformed_mov_pcs:
                        joint_to_transformed_mov_pcs[joint_index] = (losses[joint_index], transformed_mov_pcs[joint_index].detach())
                    else:
                        if losses[joint_index] < joint_to_transformed_mov_pcs[joint_index][0]:
                            joint_to_transformed_mov_pcs[joint_index] = (losses[joint_index], transformed_mov_pcs[joint_index].detach())

                for joint_index in range(len(transformed_full_pcs)):
                    if joint_index not in joint_to_transformed_base_pcs:
                        joint_to_transformed_base_pcs[joint_index] = (losses[joint_index], transformed_base_pcs[joint_index].detach())
                    else:
                        if losses[joint_index] < joint_to_transformed_base_pcs[joint_index][0]:
                            joint_to_transformed_base_pcs[joint_index] = (losses[joint_index], transformed_base_pcs[joint_index].detach())
        
        temp_mov_pcs = []
        for joint_index in range(len(mov_pcs)):
            temp_mov_pcs.append(joint_to_transformed_mov_pcs[joint_index][1])
        temp_mov_pcs = torch.stack(temp_mov_pcs)
        temp_base_pcs = []
        for joint_index in range(len(base_pcs)):
            temp_base_pcs.append(joint_to_transformed_base_pcs[joint_index][1])
        temp_base_pcs = torch.stack(temp_base_pcs)

        global_translation_vectors = torch.zeros((len(mov_pcs), 3), dtype=torch.float, device="cuda:0", requires_grad=True)
        global_rotation_ranges = torch.zeros(len(mov_pcs), dtype=torch.float, device="cuda:0", requires_grad=True)

        optimizer = torch.optim.Adam([global_translation_vectors]+[global_rotation_ranges], lr=0.008)

        for epoch in range(0, global_pre_align_max_iteration):

            print('global pre align epoch', epoch)
            
            transformed_full_pcs, _, _ = global_pre_transformation(torch.cat((temp_mov_pcs, temp_base_pcs), dim=1), axis_candidates[0], global_translation_vectors, global_rotation_ranges)
            
            transformed_mov_pcs = transformed_full_pcs[:, 0:mov_pcs.shape[1]]
            transformed_base_pcs = transformed_full_pcs[:, mov_pcs.shape[1]:]

            mov_recon_losses, _ = chamfer_distance(transformed_mov_pcs, mov_templates, batch_reduction=None)
            base_recon_losses, _ = chamfer_distance(transformed_base_pcs, base_templates, batch_reduction=None)

            losses = mov_recon_losses/target_mov_scales + base_recon_losses/target_base_scales 
            
            loss = torch.mean(losses)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('loss', loss)
        
        return transformed_mov_pcs.detach(), transformed_base_pcs.detach()

    print('wrong pre align option')
    exit()


    

    