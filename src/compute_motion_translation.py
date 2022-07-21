
import argparse
from email.policy import default
import os
from networkx.algorithms.centrality import group
import numpy as np
import math
from deform import *
from models import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pytorch3d
from pytorch3d.loss import chamfer_distance 

import torch.nn as nn
import torch.nn.functional as F
import torch
import joblib
import os
from pathlib import Path
import copy
import time
from util_motion import *
import gc
from util_vis import *
from partnet import *
import numpy as np
from objects import *
np.set_printoptions(precision=10)

from compute_common import *

def compute_translation_parameters_core(primary_dir_candidates, primary_center_candidates, args, is_validate=False):

    group_count = args['group_count']
    group_size = args['group_size']
 
    source_mov_pcs = args['source_mov_pcs']
    source_mov_handle_ts = args['source_mov_handle_ts']
    source_mov_handle_rots = args['source_mov_handle_rots']
    source_mov_handle_zs = args['source_mov_handle_zs']
    mov_contact_indices = args['mov_contact_indices']
    target_mov_scales = args['mov_scales']

    source_base_pcs = args['source_base_pcs']
    source_base_handle_ts = args['source_base_handle_ts']
    source_base_handle_rots = args['source_base_handle_rots']
    source_base_handle_zs = args['source_base_handle_zs']
    base_contact_indices = args['base_contact_indices']
    target_base_scales = args['base_scales']

    primary_translation_dirs = []
    primary_translation_centers = []
    primary_translation_ranges = []

    for group_index in range(group_count):
        primary_translation_dirs.append(primary_dir_candidates[group_index])
        primary_translation_centers.append(primary_center_candidates[group_index])
        primary_translation_ranges += [0] * (group_size-1)

    source_primary_translation_dirs = torch.tensor(primary_translation_dirs, device=device, dtype=torch.float, requires_grad=True)
    source_primary_translation_centers = torch.tensor(primary_translation_centers, device=device, dtype=torch.float, requires_grad=True)
    primary_translation_ranges = torch.tensor(primary_translation_ranges, device=device, dtype=torch.float, requires_grad=True)

    target_mov_pcs = args['target_mov_pcs']
    target_base_pcs = args['target_base_pcs']

    source_mov_pcs = source_mov_pcs
    source_mov_handle_ts = source_mov_handle_ts.repeat_interleave(group_size-1, dim=0)
    source_mov_handle_rots = source_mov_handle_rots.repeat_interleave(group_size-1, dim=0)
    source_mov_handle_zs = source_mov_handle_zs.repeat_interleave(group_size-1, dim=0)

    source_base_pcs = source_base_pcs
    source_base_handle_ts = source_base_handle_ts.repeat_interleave(group_size-1, dim=0)
    source_base_handle_rots = source_base_handle_rots.repeat_interleave(group_size-1, dim=0)
    source_base_handle_zs = source_base_handle_zs.repeat_interleave(group_size-1, dim=0)

    secondary_translation_vectors = torch.zeros((group_count * (group_size-1), 3), device="cuda:0", requires_grad=True)

    global_translation_vectors = torch.zeros(((group_count * (group_size-1)), 3), device="cuda:0", requires_grad=True)
    global_rotation_ranges = torch.zeros((group_count * (group_size-1)), device="cuda:0", requires_grad=True)

    mov_deform_box_parameters = torch.zeros((group_count*(group_size-1), 6), device="cuda:0", requires_grad=True)
    base_deform_box_parameters = torch.zeros((group_count*(group_size-1), 6), device="cuda:0", requires_grad=True)

    if is_validate == True:
        optimizer = torch.optim.Adam([primary_translation_ranges] + [secondary_translation_vectors] + [global_rotation_ranges] + [global_translation_vectors] + [mov_deform_box_parameters] + [base_deform_box_parameters], lr=0.008)
    else:
        optimizer = torch.optim.Adam([source_primary_translation_dirs] + [primary_translation_ranges] + [secondary_translation_vectors] + [global_rotation_ranges] + [global_translation_vectors]+[mov_deform_box_parameters] + [base_deform_box_parameters], lr=0.008)

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(0, translation_training_max_epoch):
        
        if is_validate:
            print('validate epoch:', epoch)
        else:
            print('translation epoch:', epoch)

        primary_translation_dirs = source_primary_translation_dirs.repeat_interleave((group_size-1), dim=0)
        primary_translation_dirs = primary_translation_dirs/torch.norm(primary_translation_dirs, dim=1).unsqueeze(dim=1)

        transformed_source_mov_pcs, local_rot_mats, local_trans_vecs = local_translation_primary(source_mov_pcs, primary_translation_dirs, primary_translation_ranges)
        transformed_source_mov_handle_ts, transformed_source_mov_handle_rots = transform_handle(source_mov_handle_ts, source_mov_handle_rots, local_rot_mats, local_trans_vecs)
        
        if epoch == 0:
            init_mov_base_collision_dists = get_mov_base_translation_collision(source_mov_pcs, primary_translation_dirs, primary_translation_ranges, source_mov_handle_zs, source_mov_handle_ts, source_mov_handle_rots, source_base_pcs)
            mov_base_collision_dists = init_mov_base_collision_dists
        else:
            mov_base_collision_dists = get_mov_base_translation_collision(source_mov_pcs, primary_translation_dirs, primary_translation_ranges, source_mov_handle_zs, source_mov_handle_ts, source_mov_handle_rots, source_base_pcs)
                
        mov_base_detach_dists = get_mov_base_translation_detach(source_mov_pcs, primary_translation_dirs, primary_translation_ranges, source_mov_handle_zs, source_mov_handle_ts, source_mov_handle_rots, source_base_pcs)
        
        transformed_source_base_handle_ts = source_base_handle_ts
        transformed_source_base_handle_rots = source_base_handle_rots

        if use_secondary:
            transformed_source_mov_pcs, local_rot_mats, local_trans_vecs = local_translation_secondary(transformed_source_mov_pcs, primary_translation_dirs, secondary_translation_vectors)
            transformed_source_mov_handle_ts, transformed_source_mov_handle_rots = transform_handle(transformed_source_mov_handle_ts, transformed_source_mov_handle_rots, local_rot_mats, local_trans_vecs)

        transformed_full_source_pcs, global_rot_mats, global_trans_vecs = global_transformation(torch.cat((transformed_source_mov_pcs, source_base_pcs), dim=1), global_translation_vectors, global_rotation_ranges)
        
        transformed_source_mov_pcs = transformed_full_source_pcs[:, 0:source_mov_pcs.shape[1]]
        transformed_source_base_pcs = transformed_full_source_pcs[:, source_mov_pcs.shape[1]:]

        transformed_source_mov_handle_ts, transformed_source_mov_handle_rots = transform_handle(transformed_source_mov_handle_ts, transformed_source_mov_handle_rots, global_rot_mats, global_trans_vecs)
        transformed_source_base_handle_ts, transformed_source_base_handle_rots = transform_handle(transformed_source_base_handle_ts, transformed_source_base_handle_rots, global_rot_mats, global_trans_vecs)

        mov_recons = transformed_source_mov_pcs
        base_recons = transformed_source_base_pcs
        
        if use_deform:
            mov_recons, deform_mov_roughness, deformed_mov_handle_ts, deformed_mov_handle_zs, _ = perform_deformation_batched(mov_recons, transformed_source_mov_handle_ts, source_mov_handle_zs, transformed_source_mov_handle_rots, mov_deform_box_parameters)
            base_recons, deform_base_roughness, deformed_base_handle_ts, deformed_base_handle_zs, _ = perform_deformation_batched(base_recons, transformed_source_base_handle_ts, source_base_handle_zs, transformed_source_base_handle_rots, base_deform_box_parameters)

        primary_range_encouragements = translation_primary_range_encouragement_weight * torch.relu(1.0 - torch.abs(primary_translation_ranges))

        if not use_primary:
            primary_range_encouragements = 0
        
        detach_penalties = translation_mov_base_detach_penalty_weight * mov_base_detach_dists
        collision_penalties = translation_collision_penalty_weight * torch.relu(mov_base_collision_dists - init_mov_base_collision_dists.detach())
        if not use_physics:
            detach_penalties = 0.0
            collision_penalties = 0.0

        secondary_range_penalties = translation_secondary_range_penalty_weight * torch.norm(secondary_translation_vectors, dim=1)
        mov_box_deform_penalties = translation_box_deform_penalty_weight * (torch.mean(torch.abs(mov_deform_box_parameters), dim=1))
        base_box_deform_penalties = translation_box_deform_penalty_weight * (torch.mean(torch.abs(base_deform_box_parameters), dim=1))

        mov_recon_losses, _ = chamfer_distance(mov_recons, target_mov_pcs, batch_reduction=None)
        base_recon_losses, _ = chamfer_distance(base_recons, target_base_pcs, batch_reduction=None)

        losses = mov_recon_losses/target_mov_scales + base_recon_losses/target_base_scales + primary_range_encouragements + secondary_range_penalties + detach_penalties + collision_penalties + mov_box_deform_penalties + base_box_deform_penalties

        reshaped_losses = losses.reshape(group_count, group_size-1)
        loss = torch.mean(torch.mean(reshaped_losses, dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('loss', loss)

    debug_infos = []
    ret_errors = []
    ret_group_errors = []
    ret_dirs = []
    ret_centers = []
    ret_ranges = []
    for i in range(group_count):
        ret_errors.append(to_numpy(losses[i*(group_size-1):(i+1)*(group_size-1)])*recon_error_amplifier)
        ret_group_errors.append(to_numpy(torch.mean(losses[i*(group_size-1):(i+1)*(group_size-1)])))
        ret_dirs.append(to_numpy(source_primary_translation_dirs[i]/torch.norm(source_primary_translation_dirs[i])))
        ret_centers.append(to_numpy(source_primary_translation_centers[i]))
        ret_ranges.append(to_numpy(primary_translation_ranges[i*(group_size-1):(i+1)*(group_size-1)]))

    return ret_errors, ret_group_errors, ret_dirs, ret_centers, ret_ranges, debug_infos

def compute_translation_parameters(joints, inputs):

    group_count = inputs['group_count']
    group_size = inputs['group_size']

    group_to_best_ret = {}

    for dir_candidate_index in range(2):
        print('dir_candidate_index', dir_candidate_index)
        primary_dir_candidates = []
        primary_center_candidates = []
        for group_index in range(group_count):
            primary_dir_candidate = joints[group_index*group_size].get_mov_key_dirs(k=2)[dir_candidate_index]
            primary_dir_candidates.append(primary_dir_candidate)
            primary_center_candidate = joints[group_index*group_size].get_key_points()[0]
            primary_center_candidates.append(primary_center_candidate)

        ret_errors, ret_group_errors, ret_dirs, ret_centers, ret_ranges, debug_infos = compute_translation_parameters_core(primary_dir_candidates, primary_center_candidates, inputs, is_validate=False)
        
        for group_index in range(len(ret_group_errors)):
            group_error = ret_group_errors[group_index]
            if group_index not in group_to_best_ret:
                group_to_best_ret[group_index] = (group_error, ret_errors[group_index], ret_dirs[group_index], ret_centers[group_index], ret_ranges[group_index])
            else:
                if group_error < group_to_best_ret[group_index][0]:
                    group_to_best_ret[group_index] = (group_error, ret_errors[group_index], ret_dirs[group_index], ret_centers[group_index], ret_ranges[group_index])

    ret_errors = []
    ret_group_errors = []
    ret_dirs = []
    ret_centers = []
    ret_ranges = []
    ret_debug_infos = []
    for group_index in range(group_count):
        ret_group_errors.append(group_to_best_ret[group_index][0])
        ret_errors.append(group_to_best_ret[group_index][1])
        ret_dirs.append(group_to_best_ret[group_index][2])
        ret_centers.append(group_to_best_ret[group_index][3])
        ret_ranges.append(group_to_best_ret[group_index][4])
        #ret_debug_infos.append(group_to_best_ret[group_index][5])
    
    source_mov_pcs = inputs['source_mov_pcs']
    source_mov_handle_ts = inputs['source_mov_handle_ts']
    source_mov_handle_rots = inputs['source_mov_handle_rots']
    source_mov_handle_zs = inputs['source_mov_handle_zs']

    source_base_pcs = inputs['source_base_pcs']
    source_base_handle_ts = inputs['source_base_handle_ts']
    source_base_handle_rots = inputs['source_base_handle_rots']
    source_base_handle_zs = inputs['source_base_handle_zs']

    postprocessed_ret_dirs = []
    for i in range(group_count):
        processed_dir = ret_dirs[i]
        processed_dir = postprocess_dir(processed_dir, source_mov_handle_ts[i], source_mov_handle_zs[i], source_mov_handle_rots[i], source_base_handle_ts[i], source_base_handle_zs[i], source_base_handle_rots[i])
        postprocessed_ret_dirs.append(processed_dir)
        
    ret_dirs = postprocessed_ret_dirs

    return ret_errors, ret_group_errors, ret_dirs, ret_centers, ret_ranges, ret_debug_infos


def compute_translations(joints, group_size):

    group_count = int(len(joints)/group_size)
    inputs = pack_inputs(joints, group_size)

    group_to_best = {}

    for option in [1]:
        temp_inputs = copy.deepcopy(inputs)
        target_mov_pcs, target_base_pcs = global_pre_align(inputs['source_mov_pcs'], inputs['source_base_pcs'], inputs['target_mov_pcs'], inputs['target_base_pcs'], inputs['mov_scales'], inputs['base_scales'], option)
        temp_inputs['target_mov_pcs'] = target_mov_pcs
        temp_inputs['target_base_pcs'] = target_base_pcs
    
        errors, group_errors, dirs, centers, ranges, debug_infos = compute_translation_parameters(joints, temp_inputs)
        
        for i in range(group_count):
            if i not in group_to_best:
                group_to_best[i] = [group_errors[i], errors[i], dirs[i], centers[i], ranges[i]]
            else:
                if group_errors[i] < group_to_best[i][0]:
                    group_to_best[i] = [group_errors[i], errors[i], dirs[i], centers[i], ranges[i]]

    errors = []
    group_errors = []
    dirs = []
    centers = []
    ranges = []
    debug_infos = []
    for group_index in range(group_count):
        group_errors.append(group_to_best[group_index][0])
        errors.append(group_to_best[group_index][1])
        dirs.append(group_to_best[group_index][2])
        centers.append(group_to_best[group_index][3])
        ranges.append(group_to_best[group_index][4])
        #debug_infos.append(group_to_best[group_index][5])

    return errors, group_errors, dirs, centers, ranges, debug_infos

def project_vector_to_plane(vectors, plane_normals):
    plane_normals = plane_normals/torch.norm(plane_normals, dim=1).unsqueeze(dim=1)
    dots = torch.matmul(vectors, plane_normals.transpose(0,1)).diag()
    projected_vectors = vectors - dots.unsqueeze(dim=1) * plane_normals
    return projected_vectors

def local_translation_primary(mov_templates, primary_translation_dirs, primary_translation_ranges):
    primary_trans_vectors = primary_translation_dirs/torch.norm(primary_translation_dirs, dim=1).unsqueeze(dim=1) * primary_translation_ranges.unsqueeze(dim=1)
    transformed_mov_templates = translate_with_vector_batched(mov_templates, primary_trans_vectors)
    homo_rot_mats = torch.cat((torch.eye(3, device='cuda:0', dtype=torch.float), torch.zeros((3,1), device='cuda:0', dtype=torch.float)),dim=1).unsqueeze(dim=0).repeat_interleave(len(mov_templates), dim=0)
    return transformed_mov_templates, homo_rot_mats, primary_trans_vectors

def local_translation_secondary(mov_templates, primary_translation_dirs, secondary_translation_vectors):
    secondary_translation_vectors = project_vector_to_plane(secondary_translation_vectors, primary_translation_dirs)
    transformed_mov_templates = translate_with_vector_batched(mov_templates, secondary_translation_vectors)
    homo_rot_mats = torch.cat((torch.eye(3, device='cuda:0', dtype=torch.float), torch.zeros((3,1), device='cuda:0', dtype=torch.float)),dim=1).unsqueeze(dim=0).repeat_interleave(len(mov_templates), dim=0)
    return transformed_mov_templates, homo_rot_mats, secondary_translation_vectors

def get_mov_base_translation_collision(mov_templates, primary_translation_dirs, primary_translation_ranges, mov_handle_zs, mov_handle_ts, mov_handle_rotmats, base_templates):
       
    sample_along_path = 10
    path_sample_indices = torch.tensor(np.arange(1, sample_along_path), device=device, dtype=torch.float)
    repeated_path_sample_indices = (path_sample_indices.repeat(primary_translation_ranges.shape[0])).unsqueeze(dim=1)
    
    repeated_primary_translation_dirs = primary_translation_dirs.repeat_interleave((sample_along_path-1), dim=0)/torch.norm(primary_translation_dirs.repeat_interleave((sample_along_path-1), dim=0), dim=1).unsqueeze(dim=1)
    repeated_primary_translation_ranges = (primary_translation_ranges.repeat_interleave((sample_along_path-1), dim=0)).unsqueeze(dim=1)

    temp_translation_vectors = repeated_primary_translation_dirs * repeated_primary_translation_ranges * repeated_path_sample_indices/sample_along_path
    repeated_mov_handle_zs = mov_handle_zs.repeat_interleave((sample_along_path-1), dim=0)
    repeated_mov_handle_ts = mov_handle_ts.repeat_interleave((sample_along_path-1), dim=0)
    repeated_mov_handle_rotmats = mov_handle_rotmats.repeat_interleave((sample_along_path-1), dim=0)

    homo_rot_mats = ((torch.cat((torch.eye(3, device='cuda:0', dtype=torch.float), torch.zeros((3,1), device='cuda:0', dtype=torch.float)),dim=1)).unsqueeze(dim=0)).repeat((len(repeated_mov_handle_ts), 1, 1))

    temp_transformed_mov_handle_ts, temp_transformed_mov_handle_rotmats = transform_handle(repeated_mov_handle_ts, repeated_mov_handle_rotmats, homo_rot_mats, temp_translation_vectors)        

    repeated_base_templates = base_templates.repeat_interleave((sample_along_path-1), dim=0)
    
    collision_dists = pc_to_bbox_collision_distances(repeated_base_templates, repeated_mov_handle_zs, temp_transformed_mov_handle_ts, temp_transformed_mov_handle_rotmats)
    collision_dists = collision_dists.reshape(-1, sample_along_path-1)
    collision_dists = torch.mean(collision_dists, dim=1)/sample_along_path
    return collision_dists


def get_mov_base_translation_detach(mov_templates, primary_translation_dirs, primary_translation_ranges, mov_handle_zs, mov_handle_ts, mov_handle_rotmats, base_templates):
    
    sample_along_path = 10
    path_sample_indices = torch.tensor(np.arange(1, sample_along_path), device=device, dtype=torch.float)
    repeated_path_sample_indices = (path_sample_indices.repeat(primary_translation_ranges.shape[0])).unsqueeze(dim=1)
    
    repeated_primary_translation_dirs = primary_translation_dirs.repeat_interleave((sample_along_path-1), dim=0)/torch.norm(primary_translation_dirs.repeat_interleave((sample_along_path-1), dim=0), dim=1).unsqueeze(dim=1)
    repeated_primary_translation_ranges = (primary_translation_ranges.repeat_interleave((sample_along_path-1), dim=0)).unsqueeze(dim=1)

    temp_translation_vectors = repeated_primary_translation_dirs * repeated_primary_translation_ranges * repeated_path_sample_indices/sample_along_path
    repeated_mov_handle_zs = mov_handle_zs.repeat_interleave((sample_along_path-1), dim=0)
    repeated_mov_handle_ts = mov_handle_ts.repeat_interleave((sample_along_path-1), dim=0)
    repeated_mov_handle_rotmats = mov_handle_rotmats.repeat_interleave((sample_along_path-1), dim=0)

    homo_rot_mats = ((torch.cat((torch.eye(3, device='cuda:0', dtype=torch.float), torch.zeros((3,1), device='cuda:0', dtype=torch.float)),dim=1)).unsqueeze(dim=0)).repeat((len(repeated_mov_handle_ts), 1, 1))

    temp_transformed_mov_handle_ts, temp_transformed_mov_handle_rotmats = transform_handle(repeated_mov_handle_ts, repeated_mov_handle_rotmats, homo_rot_mats, temp_translation_vectors)        

    repeated_base_templates = base_templates.repeat_interleave((sample_along_path-1), dim=0)
    detach_dists = pc_to_bbox_detach_distances(repeated_base_templates, repeated_mov_handle_zs, temp_transformed_mov_handle_ts, temp_transformed_mov_handle_rotmats)
    
    detach_dists = detach_dists.reshape(-1, sample_along_path-1)
    detach_dists = torch.mean(detach_dists, dim=1)/sample_along_path
    return detach_dists