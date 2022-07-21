

from shutil import move
from torch._C import device, dtype
import torch.nn as nn
import torch.nn.functional as F
import torch
from objects import *

def get_box_deformation_batched(handle_zs, xmin_deltas, xmax_deltas, ymin_deltas, ymax_deltas, zmin_deltas, zmax_deltas):

    xmins = -handle_zs[:,0]
    xmaxs = handle_zs[:,0]
    ymins = -handle_zs[:,1]
    ymaxs = handle_zs[:,1]
    zmins = -handle_zs[:,2]
    zmaxs = handle_zs[:,2]

    new_xmins = xmins - xmin_deltas
    new_xmaxs = xmaxs + xmax_deltas

    old_xcenters = 0
    old_xlengths = xmaxs - xmins

    new_xcenters = (new_xmins + new_xmaxs) * 0.5
    new_xlengths = new_xmaxs - new_xmins

    xmoves = new_xcenters - old_xcenters
    xscales = new_xlengths/old_xlengths

    new_ymins = ymins - ymin_deltas
    new_ymaxs = ymaxs + ymax_deltas

    old_ycenters = (ymins + ymaxs) * 0.5
    old_ylengths = ymaxs - ymins

    new_ycenters = (new_ymins + new_ymaxs) * 0.5
    new_ylengths = new_ymaxs - new_ymins

    ymoves = new_ycenters - old_ycenters
    yscales = new_ylengths/old_ylengths

    new_zmins = zmins - zmin_deltas
    new_zmaxs = zmaxs + zmax_deltas

    old_zcenters = (zmins + zmaxs) * 0.5
    old_zlengths = zmaxs - zmins

    new_zcenters = (new_zmins + new_zmaxs) * 0.5
    new_zlengths = new_zmaxs - new_zmins

    zmoves = new_zcenters - old_zcenters
    zscales = new_zlengths / old_zlengths

    x_identity_matrix = torch.eye(3, dtype=torch.float, device='cuda:0')
    x_identity_matrix[1][1] = 0.0
    x_identity_matrix[2][2] = 0.0
    x_identity_matrices = x_identity_matrix.repeat(len(handle_zs), 1, 1)
    repeated_xscales = (xscales.unsqueeze(dim=1).unsqueeze(dim=1)).repeat(1, 3, 3) 
    x_scale_matrices = x_identity_matrices * repeated_xscales

    y_identity_matrix = torch.eye(3, dtype=torch.float, device='cuda:0')
    y_identity_matrix[0][0] = 0.0
    y_identity_matrix[2][2] = 0.0
    y_identity_matrices = y_identity_matrix.repeat(len(handle_zs), 1, 1)
    repeated_yscales = (yscales.unsqueeze(dim=1).unsqueeze(dim=1)).repeat(1, 3, 3) 
    y_scale_matrices = y_identity_matrices * repeated_yscales

    z_identity_matrix = torch.eye(3, dtype=torch.float, device='cuda:0')
    z_identity_matrix[0][0] = 0.0
    z_identity_matrix[1][1] = 0.0
    z_identity_matrices = z_identity_matrix.repeat(len(handle_zs), 1, 1)
    repeated_zscales = (zscales.unsqueeze(dim=1).unsqueeze(dim=1)).repeat(1, 3, 3) 
    z_scale_matrices = z_identity_matrices * repeated_zscales

    scale_matrices = x_scale_matrices + y_scale_matrices + z_scale_matrices
    move_vectors = torch.stack([xmoves, ymoves, zmoves], dim=1)

    return scale_matrices, move_vectors

def perform_deformation_batched(pcs, handle_ts, handle_zs, handle_rotmats, box_parameters):
    
    handle_zxs = handle_zs[:, 0]
    handle_zys = handle_zs[:, 1]
    handle_zzs = handle_zs[:, 2]

    xmin_deltas = clamp_batched(handle_zxs * box_parameters[:, 0], -0.4 * handle_zxs, 2*handle_zxs)
    xmax_deltas = clamp_batched(handle_zxs * box_parameters[:, 1], -0.4 * handle_zxs, 2*handle_zxs)
    ymin_deltas = clamp_batched(handle_zys * box_parameters[:, 2], -0.4 * handle_zys, 2*handle_zys)
    ymax_deltas = clamp_batched(handle_zys * box_parameters[:, 3], -0.4 * handle_zys, 2*handle_zys)
    zmin_deltas = clamp_batched(handle_zzs * box_parameters[:, 4], -0.4 * handle_zzs, 2*handle_zzs)
    zmax_deltas = clamp_batched(handle_zzs * box_parameters[:, 5], -0.4 * handle_zzs, 2*handle_zzs)

    scale_mats, move_vectors = get_box_deformation_batched(handle_zs, xmin_deltas, xmax_deltas, ymin_deltas, ymax_deltas, zmin_deltas, zmax_deltas)
    
    # -- transform to standard origin
    deformed_pcs = pcs - handle_ts.unsqueeze(dim=1)
    deformed_pcs = torch.bmm(handle_rotmats.inverse(), deformed_pcs.transpose(1,2)).transpose(1,2)

    deformed_pcs = torch.bmm(scale_mats, deformed_pcs.transpose(1,2)).transpose(1,2)

    deform_smoothnesses = 0

    # -- transform back 
    deformed_pcs = torch.bmm(handle_rotmats, deformed_pcs.transpose(1,2)).transpose(1,2)
    deformed_pcs = deformed_pcs + handle_ts.unsqueeze(dim=1)
    deformed_handle_ts = torch.tensor(handle_ts, device='cuda:0', dtype=torch.float)

    deformed_handle_zs = torch.tensor(handle_zs, device='cuda:0', dtype=torch.float)
    deformed_handle_zs[:, 0] = handle_zs[:, 0] * scale_mats[:, 0, 0]
    deformed_handle_zs[:, 1] = handle_zs[:, 1] * scale_mats[:, 1, 1]
    deformed_handle_zs[:, 2] = handle_zs[:, 2] * scale_mats[:, 2, 2]

    return deformed_pcs, deform_smoothnesses, deformed_handle_ts, deformed_handle_zs, None

def clamp_batched(input, min, max):
    clamped_input = torch.max(torch.min(input, max), min)
    return clamped_input
