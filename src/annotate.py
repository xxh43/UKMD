



import argparse
from ntpath import join
import os
import numpy as np
import math
from numpy import arange
from compute_motion_rotation import *
from models import *

import torch.nn as nn
import torch.nn.functional as F
import torch

from pathlib import Path
from util_motion import *
from util_vis import *
from partnet import *
from config import *

import argparse
import os
import numpy as np
import math
from models import *
import matplotlib.pyplot as plt

import os
from pathlib import Path

from compute_transformation import *
from compute_motion_translation import *
from compute_motion_rotation import * 

def get_latent_distance(enc_t, enc_s):
    d_emb = torch.norm(enc_t - enc_s, p=2)
    return d_emb

def get_embedding_loss(target_enc, support_encs, fit_distances, alpha):
    losses = []
    for i in range(len(fit_distances)):
        d_emb = get_latent_distance(target_enc, support_encs[i])
        d_fit = torch.tensor(fit_distances[i], device='cuda:0')
        losses.append(torch.abs(d_fit - alpha*d_emb))

    return torch.mean(torch.stack(losses))

def guided_sample(target_index, encs, sample_option, sample_size, include_self=False):

    enc_target = encs[target_index]
    candidate_indices = []
    candidate_probs = []

    if include_self:
        for i in range(len(encs)):
            d = get_latent_distance(torch.tensor(enc_target, device=device), torch.tensor(encs[i], device=device))
            prob = torch.exp(-d/10)
            candidate_indices.append(i)
            candidate_probs.append(prob)
    else:
        for i in range(len(encs)):
            if i != target_index:
                d = get_latent_distance(torch.tensor(enc_target, device=device), torch.tensor(encs[i], device=device))
                prob = torch.exp(-d/10)
                candidate_indices.append(i)
                candidate_probs.append(prob)
    candidate_probs = to_numpy(torch.softmax(torch.stack(candidate_probs), dim=0))

    if sample_option == 'random':
        samples = np.random.choice(candidate_indices, sample_size, replace=False, p=np.array([1.0/len(candidate_indices)]*len(candidate_indices)))
    elif sample_option == 'sample':
        samples = np.random.choice(candidate_indices, sample_size, replace=False, p=candidate_probs)
    elif sample_option == 'greedy':
        sorted_candidate_indices = [x for _, x in sorted(zip(candidate_probs, candidate_indices), key=lambda pair: pair[0], reverse=True)][0:sample_size]
        samples = np.array(sorted_candidate_indices)
    else:
        print('invalid sample option')
        exit()

    support_probs = []
    for i in range(len(candidate_indices)):
        if candidate_indices[i] in samples:
            support_probs.append(candidate_probs[i])

    return samples.tolist(), support_probs


def get_clustered_joint_encs(joints, joint_clusters, encoder, is_train):
    
    encs = get_joint_encs(joints, encoder, is_train)
    
    all_cluster_encs = []
    for cluster_index in range(len(joint_clusters)):
        joint_indices = np.array(joint_clusters[cluster_index])
        cluster_encs = encs[joint_indices]
        all_cluster_encs.append(cluster_encs)
        
    return all_cluster_encs


def get_joint_encs(joints, encoder, is_train=True):
    
    mov_pcs = get_joint_pcs(joints, 'mov')
    base_pcs = get_joint_pcs(joints, 'base')

    encs = None
    batch_size = 64
    if len(joints) % batch_size == 1:
        batch_size = 48

    for batch_start in range(0, len(joints), batch_size):
        batched_mov_pcs = mov_pcs[batch_start:batch_start+batch_size]
        batched_base_pcs = base_pcs[batch_start:batch_start+batch_size]
        batched_labeled_mov_pcs = torch.cat((batched_mov_pcs, torch.zeros((batched_mov_pcs.shape[0], batched_mov_pcs.shape[1], 1), device='cuda:0')), dim=2)
        batched_labeled_base_pcs = torch.cat((batched_base_pcs, torch.ones((batched_base_pcs.shape[0], batched_base_pcs.shape[1], 1), device='cuda:0')), dim=2)
        batched_labeled_joint_pcs = torch.cat((batched_labeled_mov_pcs, batched_labeled_base_pcs), dim=1)
        batched_encs = encoder(batched_labeled_joint_pcs)

        if encs is None:
            encs = batched_encs
        else:
            encs = torch.cat((encs, batched_encs), dim=0)

    return encs

def update_encoder(joints, target_indices, support_indices, fit_distances, encoder, encoder_alpha, max_epoch, learning_rate):

    optimizer = torch.optim.Adam(list(encoder.parameters()) + [encoder_alpha], lr=learning_rate)
    
    errors = []
    for epoch in range(0, max_epoch):
        
        print('encoder epoch', epoch)

        encs = get_joint_encs(joints, encoder)
    
        losses = []
        for i in range(len(target_indices)):
            loss_emb = get_embedding_loss(encs[target_indices[i]], encs[support_indices[i]], fit_distances[i], encoder_alpha)
            losses.append(loss_emb)

        epoch_loss = torch.mean(torch.stack(losses))
        print('epoch_loss', epoch_loss)
        optimizer.zero_grad()
        epoch_loss.backward()
        optimizer.step()
        
        errors.append(epoch_loss.item())

    return errors


def update_clustered_encoder(joints, joint_clusters, target_indices, support_indices, fit_distances, encoder, encoder_alpha, max_epoch, learning_rate):

    optimizer = torch.optim.Adam(list(encoder.parameters()) + [encoder_alpha], lr=learning_rate)
    
    errors = []
    for epoch in range(0, max_epoch):    
        print('encoder epoch', epoch)

        batch_size = 16
        for batch_start in range(0, len(joint_clusters), batch_size):
            batched_clustered_encs = get_clustered_joint_encs(joints, joint_clusters[batch_start:batch_start+batch_size], encoder, True)
            batched_target_indices = target_indices[batch_start:batch_start+batch_size]
            batched_support_indices = support_indices[batch_start:batch_start+batch_size]
            batched_fit_distances = fit_distances[batch_start:batch_start+batch_size]
            losses = []
            for cluster_index in range(len(batched_clustered_encs)):
                loss_emb = get_embedding_loss(batched_clustered_encs[cluster_index][batched_target_indices[cluster_index]], batched_clustered_encs[cluster_index][batched_support_indices[cluster_index]], batched_fit_distances[cluster_index], encoder_alpha)
                losses.append(loss_emb)

            epoch_loss = torch.mean(torch.stack(losses))
            print('epoch_loss', epoch_loss)
            optimizer.zero_grad()
            epoch_loss.backward()
            optimizer.step()
            
            errors.append(epoch_loss.item())

    return errors

def cluster_joints_with_encoder(joints, encoder, pairwise_distances):

    encs = get_joint_encs(joints, encoder)
    clusters = []
    for target_index in range(len(joints)):
        support_indices, _ = guided_sample(target_index, encs, 'greedy', joint_cluster_size)
        cluster = [target_index] + support_indices
        clusters.append(cluster)
    return clusters

def cluster_joints_with_simmat(joints, interested_joint_indices, pairwise_distances):

    clusters = []
    for local_joint_index, interested_joint_index in enumerate(interested_joint_indices):
        candidate_indices = np.array(list(arange(0, len(joints)))[0:interested_joint_index] + list(arange(0, len(joints)))[interested_joint_index+1:])
        candidate_distances = np.array(list(pairwise_distances[local_joint_index][0:interested_joint_index]) + list(pairwise_distances[local_joint_index][interested_joint_index+1:]))
        sorted_candidate_indices = [x for _, x in sorted(zip(candidate_distances, candidate_indices), key=lambda pair: pair[0])][0:joint_cluster_size]
        support_indices = sorted_candidate_indices
        cluster = [interested_joint_index] + support_indices
        clusters.append(cluster)

    return clusters

def pre_annotate_joints(joints, interested_joint_indices, encoder, encoder_alpha):

    group_count = min(len(joints), pre_annotate_group_count)
    print('group_count', group_count)
    group_size = len(joints)+1
    print('group_size', group_size)

    batch_length = group_count * group_size

    all_target_indices = []
    all_support_indices = []
    flattened_joint_indices = []

    for target_index in range(len(joints)):
        if target_index in interested_joint_indices:
            all_target_indices.append(target_index)
            support_indices = list(arange(0, len(joints)))
            all_support_indices.append(support_indices)
            flattened_joint_indices += [target_index]
            flattened_joint_indices += support_indices

    all_errors = []
    for batch_start in range(0, len(flattened_joint_indices), batch_length):
        batched_joint_indices = flattened_joint_indices[batch_start:batch_start+batch_length]
        batched_joints = [joints[v] for v in batched_joint_indices] 
        batched_errors = compute_transformations(batched_joints, group_size)
        all_errors += batched_errors

    return to_numpy(all_errors)

def first_annotate_joints(joints, joint_clusters, motion_type):

    summary = {}

    group_count = min(len(joints), annotate_group_count)
    group_size = max(4, min(int(len(joints)*0.5), annotate_group_size))
    batch_length = group_count * group_size

    flattened_global_joint_indices = []
    for cluster_index in range(len(joint_clusters)):
        flattened_global_joint_indices += [joint_clusters[cluster_index][0]]
        global_support_indices = joint_clusters[cluster_index][1:group_size]
        flattened_global_joint_indices += global_support_indices
    
    all_errors = []
    all_group_errors = []
    all_dirs = []
    all_centers = []
    all_ranges = []
    all_debug_infos = []

    for batch_start in range(0, len(flattened_global_joint_indices), batch_length):
        
        batched_joint_indices = flattened_global_joint_indices[batch_start:batch_start+batch_length]
        batched_joints = [joints[v] for v in batched_joint_indices] 
        
        if motion_type == 'rotation':
            
            batched_errors, batched_group_errors, batched_dirs, batched_centers, batched_ranges, batched_debug_infos = compute_rotations(batched_joints, group_size)

        else:
            batched_errors, batched_group_errors, batched_dirs, batched_centers, batched_ranges, batched_debug_infos = compute_translations(batched_joints, group_size)

        all_errors += batched_errors
        all_group_errors += batched_group_errors
        all_dirs += batched_dirs
        all_centers += batched_centers
        all_ranges += batched_ranges
        all_debug_infos += batched_debug_infos

    annotations = []
    for i in range(len(joint_clusters)):
        joint_index = joint_clusters[i][0]
        group_indices = [flattened_global_joint_indices[v] for v in range(i*group_size, (i+1)*group_size)]
        annotation = [(joints[joint_index].shape_id, joints[joint_index].part_id), to_numpy(all_group_errors[i]), to_numpy(all_dirs[i]), to_numpy(all_centers[i]), to_numpy(all_ranges[i]), group_indices]
        annotations.append(annotation)

    return annotations, summary

def annotate_joints(joints, joint_clusters, motion_type, encoder, encoder_alpha, sample_option):

    summary = {}

    group_count = min(len(joints), annotate_group_count)
    group_size = max(4, min(int(len(joints)*0.5), annotate_group_size))
    batch_length = group_count * group_size

    clustered_joint_encs = get_clustered_joint_encs(joints, joint_clusters, encoder, False)

    flattened_global_joint_indices = []

    all_local_target_joint_indices = []
    all_local_support_joint_indices = []

    for cluster_index in range(len(joint_clusters)):

        encs_in_cluster = clustered_joint_encs[cluster_index]
        local_support_joint_indices, _ = guided_sample(0, encs_in_cluster, sample_option, group_size-1)
        all_local_target_joint_indices.append(0)
        all_local_support_joint_indices.append(local_support_joint_indices)

        global_support_indices = [joint_clusters[cluster_index][v] for v in local_support_joint_indices]
        
        flattened_global_joint_indices += [joint_clusters[cluster_index][0]]
        flattened_global_joint_indices += global_support_indices

    all_errors = []
    all_group_errors = []
    all_dirs = []
    all_centers = []
    all_ranges = []
    all_debug_infos = []

    for batch_start in range(0, len(flattened_global_joint_indices), batch_length):
        
        batched_joint_indices = flattened_global_joint_indices[batch_start:batch_start+batch_length]
        batched_joints = [joints[v] for v in batched_joint_indices] 
        
        if motion_type == 'rotation':
            batched_errors, batched_group_errors, batched_dirs, batched_centers, batched_ranges, batched_debug_infos = compute_rotations(batched_joints, group_size)
        else:
            batched_errors, batched_group_errors, batched_dirs, batched_centers, batched_ranges, batched_debug_infos = compute_translations(batched_joints, group_size)

        all_errors += batched_errors
        all_group_errors += batched_group_errors
        all_dirs += batched_dirs
        all_centers += batched_centers
        all_ranges += batched_ranges
        all_debug_infos += batched_debug_infos

    annotations = []
    for i in range(len(joint_clusters)):
        joint_index = joint_clusters[i][0]
        group_indices = [flattened_global_joint_indices[v] for v in range(i*group_size, (i+1)*group_size)]
        annotation = [(joints[joint_index].shape_id, joints[joint_index].part_id), to_numpy(all_group_errors[i]), to_numpy(all_dirs[i]), to_numpy(all_centers[i]), to_numpy(all_ranges[i]), group_indices]
        annotations.append(annotation)

    update_clustered_encoder(joints, joint_clusters, all_local_target_joint_indices, all_local_support_joint_indices, all_errors, encoder, encoder_alpha, encoding_training_max_epoch, encoding_training_learning_rate)

    return annotations, summary


def validate_annotate_joints(joints, joint_clusters, annotations, motion_type, encoder, sample_option):

    dirs = [v[2] for v in annotations]
    centers = [v[3] for v in annotations]
    
    group_size = min(len(joint_clusters[0]), special_annotate_group_size)
    group_count = min(len(joints), special_annotate_group_count)
    print('group_size', group_size)
    print('group_count', group_count)
    
    clustered_joint_encs = get_clustered_joint_encs(joints, joint_clusters, encoder, False)

    flattened_global_joint_indices = []
    all_local_target_joint_indices = []
    all_local_support_joint_indices = []

    for cluster_index in range(len(joint_clusters)):

        encs_in_cluster = clustered_joint_encs[cluster_index]
        local_support_joint_indices, _ = guided_sample(0, encs_in_cluster, sample_option, group_size-1)
        all_local_target_joint_indices.append(0)
        all_local_support_joint_indices.append(local_support_joint_indices)

        global_support_indices = [joint_clusters[cluster_index][v] for v in local_support_joint_indices]
        
        flattened_global_joint_indices += [joint_clusters[cluster_index][0]]
        flattened_global_joint_indices += global_support_indices
    
    all_errors = []
    all_dirs = []
    all_centers = []
    all_ranges = []

    for batch_start in range(0, len(joint_clusters), group_count):
            
        batched_joint_indices = flattened_global_joint_indices[batch_start*group_size:(batch_start+group_count)*group_size]
        batched_joints = [joints[batched_joint_indices[i]] for i in range(len(batched_joint_indices))]

        batched_init_dirs = dirs[batch_start:batch_start+group_count]
        batched_init_centers = centers[batch_start:batch_start+group_count]

        inputs = pack_inputs(batched_joints, group_size)
                
        if motion_type == 'rotation':
            batched_errors, batched_group_errors, batched_dirs, batched_centers, batched_ranges, debug_infos = compute_rotation_parameters_core(batched_init_dirs, batched_init_centers, inputs, is_validate=True)
        else:
            batched_errors, batched_group_errors, batched_dirs, batched_centers, batched_ranges, debug_infos = compute_translation_parameters_core(batched_init_dirs, batched_init_centers, inputs, is_validate=True)

        all_errors += batched_errors
        all_dirs += batched_dirs
        all_centers += batched_centers
        all_ranges += batched_ranges

    return all_errors, all_ranges

