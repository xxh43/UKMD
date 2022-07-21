
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
import copy
import time
from util_motion import *
import gc

from util_vis import *

import shutil
from scipy.spatial.distance import cdist
from partnet import *
import numpy as np
from objects import *
from compute_common import *

np.set_printoptions(precision=10)

def mov_global_transformation(pcs, rot_mats, trans_vectors):

    reset_vector = torch.mean(pcs, dim=1).unsqueeze(dim=1).repeat_interleave(pcs.shape[1], dim=1)
    pcs = pcs - reset_vector
    transformed_pcs = torch.bmm(rot_mats, pcs.transpose(1,2)).transpose(1,2)
    transformed_pcs = transformed_pcs + reset_vector
    transformed_pcs = translate_with_vector_batched(transformed_pcs, trans_vectors)
    homo_rot_mats = torch.cat((rot_mats, torch.zeros((rot_mats.shape[0], 3, 1), device='cuda:0')), dim=2)
    return transformed_pcs, homo_rot_mats, trans_vectors

def compute_transformations(joints, group_size):

    group_count = int(len(joints)/group_size)
    inputs = pack_inputs(joints, group_size)

    mov_templates = inputs['source_mov_pcs']
    source_mov_pc_comp_nums = inputs['source_mov_pc_comp_nums'] 
    mov_pcs = inputs['target_mov_pcs']
    mov_pc_comp_nums = inputs['target_mov_pc_comp_nums']
    mov_handle_ts = inputs['source_mov_handle_ts']
    mov_handle_rotmats = inputs['source_mov_handle_rots']
    mov_handle_zs = inputs['source_mov_handle_zs']
    target_mov_scales = inputs['mov_scales']

    mov_handle_ts = mov_handle_ts.repeat_interleave(group_size-1, dim=0)
    mov_handle_rotmats = mov_handle_rotmats.repeat_interleave(group_size-1, dim=0)
    mov_handle_zs = mov_handle_zs.repeat_interleave(group_size-1, dim=0)

    base_templates = inputs['source_base_pcs']
    source_base_pc_comp_nums = inputs['source_base_pc_comp_nums']
    base_pcs = inputs['target_base_pcs']
    base_pc_comp_nums = inputs['target_base_pc_comp_nums']
    base_handle_ts = inputs['source_base_handle_ts']
    base_handle_rotmats = inputs['source_base_handle_rots']
    base_handle_zs = inputs['source_base_handle_zs']
    target_base_scales = inputs['base_scales']

    base_handle_ts = base_handle_ts.repeat_interleave(group_size-1, dim=0)
    base_handle_rotmats = base_handle_rotmats.repeat_interleave(group_size-1, dim=0)
    base_handle_zs = base_handle_zs.repeat_interleave(group_size-1, dim=0)

    mov_translations = torch.zeros(((group_count * (group_size-1)), 3), device="cuda:0", dtype=torch.float, requires_grad=True)
    base_translations = torch.zeros(((group_count * (group_size-1)), 3), device="cuda:0", dtype=torch.float, requires_grad=True)
    mov_deform_box_parameters = torch.full((group_count*(group_size-1), 6), 0.0001, device="cuda:0", dtype=torch.float, requires_grad=True)
    base_deform_box_parameters = torch.full((group_count*(group_size-1), 6), 0.0001, device="cuda:0", dtype=torch.float, requires_grad=True)

    # training start ------------------------------------------------------------------------------------------
    
    max_epoch = pre_annotate_transformation_max_epoch
    
    joint_to_best_error = {}

    base_rotation_angle_candidates = [0, 0.5*np.pi, 1.0*np.pi, 1.5*np.pi]
    mov_rotation_quaternion_candidates = [[1.0, 0.0, 0.0, 0.0]]
    for mov_index in range(len(mov_rotation_quaternion_candidates)):
        for base_index in range(len(base_rotation_angle_candidates)):
            
            mov_rotation_quaternions = torch.tensor(torch.tensor([mov_rotation_quaternion_candidates[mov_index]], device=device).repeat_interleave(group_count * (group_size-1), dim=0), device="cuda:0", requires_grad=True)
            base_rotation_angles = torch.tensor(torch.tensor(base_rotation_angle_candidates[base_index]).repeat(len(mov_pcs)), dtype=torch.float, device="cuda:0", requires_grad=True)
            
            optimizer = torch.optim.Adam([mov_translations] +  [mov_rotation_quaternions] + [base_translations] + [base_rotation_angles] + [mov_deform_box_parameters] + [base_deform_box_parameters], lr=pre_annotate_optim_learning_rate)

            for epoch in range(0, max_epoch):
                print('pre transformation epoch:', epoch)
                
                mov_rotations = quaternion_to_rotation_matrix_batched(mov_rotation_quaternions)
            
                transformed_mov_templates, mov_rot_mats, mov_trans_vecs = mov_global_transformation(mov_templates, mov_rotations, mov_translations)
                transformed_mov_handle_ts, transformed_mov_handle_rotmats = transform_handle(mov_handle_ts, mov_handle_rotmats, mov_rot_mats, mov_trans_vecs)
                transformed_templates, global_base_homo_rots, global_base_trans = global_transformation(torch.cat((transformed_mov_templates, base_templates), dim=1), base_translations, base_rotation_angles)
                transformed_mov_templates = transformed_templates[:, 0:mov_pcs.shape[1]]
                transformed_base_templates = transformed_templates[:, mov_pcs.shape[1]:]
                transformed_mov_handle_ts, transformed_mov_handle_rotmats = transform_handle(transformed_mov_handle_ts, transformed_mov_handle_rotmats, global_base_homo_rots, global_base_trans)
                transformed_base_handle_ts, transformed_base_handle_rotmats = transform_handle(base_handle_ts, base_handle_rotmats, global_base_homo_rots, global_base_trans)
            
                mov_recons = transformed_mov_templates
                base_recons = transformed_base_templates

                mov_recon_losses, _ = chamfer_distance(mov_recons, mov_pcs, batch_reduction=None)
                base_recon_losses, _ = chamfer_distance(base_recons, base_pcs, batch_reduction=None)

                #losses = mov_recon_losses/target_mov_scales + base_recon_losses/target_base_scales + mov_box_deform_penalties/target_mov_scales + base_box_deform_penalties/target_base_scales
                losses = mov_recon_losses/target_mov_scales + base_recon_losses/target_base_scales

                reshape_losses = losses.reshape(group_count, group_size-1)
                loss = torch.mean(torch.mean(reshape_losses, dim=1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print('loss', loss)

            for joint_index in range(len(mov_pcs)):
                if joint_index not in joint_to_best_error:
                    joint_to_best_error[joint_index] = (losses[joint_index], mov_recons[joint_index], base_recons[joint_index], torch.norm(mov_translations[joint_index]))
                else:
                    if losses[joint_index] < joint_to_best_error[joint_index][0]:
                        joint_to_best_error[joint_index] = (losses[joint_index], mov_recons[joint_index], base_recons[joint_index], torch.norm(mov_translations[joint_index]))
    
    all_best_losses = []
    all_best_mov_pcs = []
    all_best_base_pcs = []
    for joint_index in range(len(mov_pcs)):

        source_mov_comp_num = source_mov_pc_comp_nums[joint_index]
        source_base_comp_num = source_base_pc_comp_nums[joint_index]
        target_mov_comp_num = mov_pc_comp_nums[joint_index]
        target_base_comp_num = base_pc_comp_nums[joint_index]
        
        if source_mov_comp_num != target_mov_comp_num or source_base_comp_num != target_base_comp_num:
            distance = joint_to_best_error[joint_index][0] + 10
        else:
            distance = joint_to_best_error[joint_index][0]
        all_best_losses.append(distance)
        all_best_mov_pcs.append(joint_to_best_error[joint_index][1])
        all_best_base_pcs.append(joint_to_best_error[joint_index][2])
    all_best_losses = torch.stack(all_best_losses)

    ret_errors = []
    for i in range(group_count):
        ret_errors.append(to_numpy(all_best_losses[i*(group_size-1):(i+1)*(group_size-1)]*recon_error_amplifier))

    return ret_errors

