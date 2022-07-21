
import sys
sys.path.append('..')
import torch
import torch.nn
from objects import *

from scipy.spatial.distance import cdist


def transform_with_homo_matrix_batched(pcs, mats):
    pad = torch.ones((pcs.shape[0], pcs.shape[1], 1), device=device)
    pcs_homo = torch.cat((pcs, pad), dim=2)
    pcs_transformed = torch.bmm(mats, pcs_homo.transpose(1,2)).transpose(1,2)
    return pcs_transformed

def transform_with_homo_matrix(pc, mat):
    if not torch.is_tensor(pc):
        pc = torch.tensor(pc, device=device, dtype=torch.float)
    if not torch.is_tensor(mat):
        mat = torch.tensor(mat, device=device, dtype=torch.float)
    pad = torch.nn.ConstantPad1d((0, 1), 1.0)
    pc_homo = pad(pc)
    pc_homo = pc_homo.transpose(0, 1)
    pc_transformed = torch.matmul(mat, pc_homo)
    pc_transformed = pc_transformed.transpose(0, 1)
    return pc_transformed

def rotate_with_axis_center_angle_batched(pcs, axes, centers, angles):
    if not torch.is_tensor(pcs):
        pcs = torch.tensor(pcs, device=device, dtype=torch.float)
    if not torch.is_tensor(axes):
        axes = torch.tensor(axes, device=device, dtype=torch.float)
    if not torch.is_tensor(centers):
        centers = torch.tensor(centers, device=device, dtype=torch.float)
    if not torch.is_tensor(angles):
        angles = torch.tensor(angles, device=device, dtype=torch.float)

    axes = axes/torch.norm(axes, dim=1).unsqueeze(dim=1)
    mats = getRotMatrixHomo_batched(axes, centers, angles)
    return transform_with_homo_matrix_batched(pcs, mats), mats

def rotate_with_axis_center_angle(pc, axis, center, angle):
    if not torch.is_tensor(pc):
        pc = torch.tensor(pc, device=device, dtype=torch.float)
    if not torch.is_tensor(axis):
        axis = torch.tensor(axis, device=device, dtype=torch.float)
    if not torch.is_tensor(center):
        center = torch.tensor(center, device=device, dtype=torch.float)
    if not torch.is_tensor(angle):
        angle = torch.tensor(angle, device=device, dtype=torch.float)

    axis = axis/torch.norm(axis)
    mat_homo = getRotMatrixHomo(axis, center, angle)
    return transform_with_homo_matrix(pc, mat_homo)

def getRotMatrixHomo_batched(axes, centers, angles):
    u = axes[:,0]
    v = axes[:,1]
    w = axes[:,2]
    a = centers[:,0]
    b = centers[:,1]
    c = centers[:,2]
    sin_theta = torch.sin(angles)
    cos_theta = torch.cos(angles)

    m00 = u*u + (v*v + w*w)*cos_theta
    m01 = u*v*(torch.ones(cos_theta.shape, device=device)-cos_theta)-w*sin_theta
    m02 = u*w*(torch.ones(cos_theta.shape, device=device)-cos_theta)+v*sin_theta
    m03 = (a*(v*v+w*w) - u*(b*v+c*w))*(torch.ones(cos_theta.shape, device=device)-cos_theta)+(b*w-c*v)*sin_theta

    m10 = u*v*(torch.ones(cos_theta.shape, device=device)-cos_theta)+w*sin_theta
    m11 = v*v+(u*u+w*w)*cos_theta
    m12 = v*w*(torch.ones(cos_theta.shape, device=device)-cos_theta)-u*sin_theta
    m13 = (b*(u*u+w*w) - v*(a*u+c*w))*(torch.ones(cos_theta.shape, device=device)-cos_theta)+(c*u-a*w)*sin_theta

    m20 = u*w*(torch.ones(cos_theta.shape, device=device)-cos_theta)-v*sin_theta
    m21 = v*w*(torch.ones(cos_theta.shape, device=device)-cos_theta)+u*sin_theta
    m22 = w*w+(u*u+v*v)*cos_theta
    m23 = (c*(u*u+v*v) - w*(a*u+b*v))*(torch.ones(cos_theta.shape, device=device)-cos_theta)+(a*v-b*u)*sin_theta

    rot_mats = torch.stack((m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23)).transpose(0, 1)
    rot_mats = rot_mats.reshape(rot_mats.shape[0], 3, 4)

    return rot_mats


def getRotMatrixHomo(axis, center, angle):
    u = axis[0]
    v = axis[1]
    w = axis[2]
    a = center[0]
    b = center[1]
    c = center[2]
    sin_theta = torch.sin(angle)
    cos_theta = torch.cos(angle)

    m00 = u*u + (v*v + w*w)*cos_theta
    m01 = u*v*(1-cos_theta)-w*sin_theta
    m02 = u*w*(1-cos_theta)+v*sin_theta
    m03 = (a*(v*v+w*w) - u*(b*v+c*w))*(1-cos_theta)+(b*w-c*v)*sin_theta
    
    m10 = u*v*(1-cos_theta)+w*sin_theta
    m11 = v*v+(u*u+w*w)*cos_theta
    m12 = v*w*(1-cos_theta)-u*sin_theta
    m13 = (b*(u*u+w*w) - v*(a*u+c*w))*(1-cos_theta)+(c*u-a*w)*sin_theta

    m20 = u*w*(1-cos_theta)-v*sin_theta
    m21 = v*w*(1-cos_theta)+u*sin_theta
    m22 = w*w+(u*u+v*v)*cos_theta
    m23 = (c*(u*u+v*v) - w*(a*u+b*v))*(1-cos_theta)+(a*v-b*u)*sin_theta
    
    rot_mat = torch.stack((

        torch.stack((m00, m01, m02, m03)),

        torch.stack((m10, m11, m12, m13)),

        torch.stack((m20, m21, m22, m23)),
    ))
    
    return rot_mat

def translate_with_vector_batched(pcs, vectors):
    if not torch.is_tensor(pcs):
        pcs = torch.tensor(pcs, device=device, dtype=torch.float)
    if not torch.is_tensor(vectors):
        vectors = torch.tensor(vectors, device=device, dtype=torch.float)

    vectors = vectors.unsqueeze(dim=1)
    translated_pcs = pcs + vectors.repeat_interleave(pcs.shape[1], dim=1)
    return translated_pcs

def translate_with_vector(pc, vector):
    if not torch.is_tensor(pc):
        pc = torch.tensor(pc, device=device, dtype=torch.float)
    if not torch.is_tensor(vector):
        vector = torch.tensor(vector, device=device, dtype=torch.float)
    translated_pc = pc + vector
    return translated_pc

def diag(a, b):
  return 1 - 2 * a * a - 2 * b * b

def tr_add(a, b, c, d):
  return 2 * a * b + 2 * c * d

def tr_sub(a, b, c, d):
  return 2 * a * b - 2 * c * d

def quaternion_to_rotation_matrix(q):
    normalized_q = q/np.linalg.norm(q)
    w = normalized_q[0]
    x = normalized_q[1]
    y = normalized_q[2]
    z = normalized_q[3]
    mat = np.zeros((3,3))
    mat[0][0] = diag(y, z)  
    mat[0][1] = tr_sub(x, y, z, w)
    mat[0][2] = tr_add(x, z, y, w)
    mat[1][0] = tr_add(x, y, z, w) 
    mat[1][1] = diag(x, z)
    mat[1][2] = tr_sub(y, z, x, w)
    mat[2][0] = tr_sub(x, z, y, w)  
    mat[2][1] = tr_add(y, z, x, w)  
    mat[2][2] = diag(x, y)
    return mat

def quaternion_to_rotation_matrix_batched(qs):
    normalized_qs = qs/torch.norm(qs, dim=1).unsqueeze(dim=1)
    ws = normalized_qs[:,0]
    xs = normalized_qs[:,1]
    ys = normalized_qs[:,2]
    zs = normalized_qs[:,3]
    mats = torch.zeros((len(qs),3,3), device=device, dtype=torch.float)
    mats[:,0,0] = diag(ys, zs)  
    mats[:,0,1] = tr_sub(xs, ys, zs, ws)
    mats[:,0,2] = tr_add(xs, zs, ys, ws)
    mats[:,1,0] = tr_add(xs, ys, zs, ws) 
    mats[:,1,1] = diag(xs, zs)
    mats[:,1,2] = tr_sub(ys, zs, xs, ws)
    mats[:,2,0] = tr_sub(xs, zs, ys, ws)  
    mats[:,2,1] = tr_add(ys, zs, xs, ws)  
    mats[:,2,2] = diag(xs, ys)
    return mats