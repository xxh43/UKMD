
from copy import deepcopy
import numpy as np
from numpy import dtype
from numpy.lib import utils
import trimesh
import torch
import json

from trimesh.base import Trimesh

from collections import defaultdict
from util_mesh import *
import networkx as nx
from scipy.spatial.distance import cdist
from util_vis import *
from util_motion import *
from sklearn.decomposition import PCA
from config import *
import numpy as np



class Part:
    def __init__(self):
        self.shape_id = None
        self.id = None
        self.is_root = False
        self.mesh = None
        
        self.gt_motion_type = None
        self.gt_center = None
        self.gt_axis = None
        self.gt_min_range = None
        self.gt_max_range = None

class Joint:
    def __init__(self):
        self.shape_id = None
        self.part_id = None

        self.mov_part_indices = None
        self.base_part_indices = None

        self.reset_translation_vector = None

        self.is_original = None
        self.mov_pc = None
        self.mov_comp_num = None
        self.mov_pc_normal = None

        self.base_pc = None
        self.base_comp_num = None
        self.base_pc_normal = None

        self.mov_deform_handle = None
        self.base_deform_handle = None

        self.gt_motion_type = None
        self.gt_center = None
        self.gt_axis = None
        self.gt_min_range = None
        self.gt_max_range = None

    def get_mov_comp_num(self):
        if self.mov_comp_num is None:
            self.mov_comp_num = get_point_comps(self.mov_pc)
        else:
            return self.mov_comp_num

    def get_base_comp_num(self):
        if self.base_comp_num is None:
            self.base_comp_num = get_point_comps(self.base_pc)
        else:
            return self.base_comp_num

    def get_contact_indices(self):
        contact_point_num = int(part_point_num * 0.1)
        threshold = 0.0
        pc1 = self.mov_pc
        pc2 = self.base_pc
        contact_dist_mat = cdist(pc1, pc2)
        flat_dist = contact_dist_mat.flatten()
        contact_indices = np.where(flat_dist < threshold)[0].tolist()
        while len(contact_indices) < contact_point_num:
            threshold += 0.002
            contact_indices = np.where(flat_dist < threshold)[0].tolist()
        
        contact_indices = contact_indices[0:contact_point_num]

        contact_indices_pc1 = []
        contact_indices_pc2 = []
        for i in range(len(contact_indices)):
            i_pc1 = contact_indices[i] // pc2.shape[0]
            contact_indices_pc1.append(torch.tensor(i_pc1, device="cuda:0", dtype=torch.long))
            i_pc2 = contact_indices[i] % pc2.shape[0]
            contact_indices_pc2.append(torch.tensor(i_pc2, device="cuda:0", dtype=torch.long))
        contact_indices_pc1 = list(set(contact_indices_pc1))
        contact_indices_pc2 = list(set(contact_indices_pc2))
        contact_mov_indices = contact_indices_pc1
        contact_base_indices = contact_indices_pc2

        return torch.stack(contact_mov_indices), torch.stack(contact_base_indices)

    def get_mov_center(self):
        center = self.mov_deform_handle.get_center()
        return center

    def get_key_points(self):
        face_centers = self.mov_deform_handle.get_face_centers()
        center = self.mov_deform_handle.get_center()

        distances = []
        for face_center in face_centers:
            distances.append(np.linalg.norm(face_center - center))

        sorted_face_centers = [x for _, x in sorted(zip(distances, face_centers), key=lambda pair: pair[0], reverse=True)][0:6]        
        return [center]+sorted_face_centers

    def get_mov_key_dirs(self, k=3):
        rot_mat, t, extents = get_yaxis_aligned_bbox(self.mov_pc)
        dir0 = rot_mat[:, 0]
        dir1 = rot_mat[:, 1]
        dir2 = rot_mat[:, 2]
        sorted_dirs = [x for _, x in sorted(zip(extents, [dir0, dir1, dir2]), key=lambda pair: pair[0], reverse=True)][0:k]
        return sorted_dirs

class RawShape:
    def __init__(self):
        self.mesh = None
        self.parts = []
        self.part_graph = None

class Shape:
    def __init__(self):
        self.id = None
        self.parts = []
        self.part_graph = None
        self.joints = []

class DeformHandle:
    def __init__(self):
        self.z = None
        self.q = None
        self.t = None
        self.rot_mat = None
        self.control_mesh = None
        self.old_control_vertices = None
        self.control_vertices = None
        self.faces = None

    def get_center(self):
        center = self.t
        return center

    def get_corners(self):
        origin = np.zeros(3)
        x_dir = np.array([1, 0, 0])
        y_dir = np.array([0, 1, 0])
        z_dir = np.array([0, 0, 1])

        xmin = -to_numpy(self.z)[0]
        xmax = to_numpy(self.z)[0]
        ymin = -to_numpy(self.z)[1]
        ymax = to_numpy(self.z)[1]
        zmin = -to_numpy(self.z)[2]
        zmax = to_numpy(self.z)[2]

        p0 = origin + x_dir * xmin + y_dir * ymin + z_dir * zmin
        p1 = origin + x_dir * xmin + y_dir * ymin + z_dir * zmax
        p2 = origin + x_dir * xmin + y_dir * ymax + z_dir * zmin
        p3 = origin + x_dir * xmin + y_dir * ymax + z_dir * zmax
        p4 = origin + x_dir * xmax + y_dir * ymin + z_dir * zmin
        p5 = origin + x_dir * xmax + y_dir * ymin + z_dir * zmax
        p6 = origin + x_dir * xmax + y_dir * ymax + z_dir * zmin
        p7 = origin + x_dir * xmax + y_dir * ymax + z_dir * zmax

        return p0,p1,p2,p3,p4,p5,p6,p7

    def get_face_centers(self):
        
        p0,p1,p2,p3,p4,p5,p6,p7=self.get_corners()

        face_center0 = (p0+p1+p2+p3)/4.0
        face_center1 = (p4+p5+p6+p7)/4.0
        face_center2 = (p0+p1+p4+p5)/4.0
        face_center3 = (p2+p3+p6+p7)/4.0
        face_center4 = (p0+p2+p4+p6)/4.0
        face_center5 = (p1+p3+p5+p7)/4.0

        face_centers = np.array([face_center0, face_center1, face_center2, face_center3, face_center4, face_center5])
        face_centers = np.matmul(to_numpy(self.rot_mat), to_numpy(face_centers).transpose()).transpose()
        face_centers = face_centers + to_numpy(self.t)

        return list(face_centers)
        
    def to_cube(self):

        p0,p1,p2,p3,p4,p5,p6,p7=self.get_corners()

        pc = np.array([p0, p1, p2, p3, p4, p5, p6, p7])
        print('rot_mat', self.rot_mat)
        pc = np.matmul(to_numpy(self.rot_mat), to_numpy(pc).transpose()).transpose()
        pc = pc + to_numpy(self.t)

        edge01 = [pc[0], pc[1]]
        edge02 = [pc[0], pc[2]]
        edge13 = [pc[1], pc[3]]
        edge23 = [pc[2], pc[3]]

        edge45 = [pc[4], pc[5]]
        edge46 = [pc[4], pc[6]]
        edge57 = [pc[5], pc[7]]
        edge67 = [pc[6], pc[7]]

        edge04 = [pc[0], pc[4]]
        edge15 = [pc[1], pc[5]]
        edge26 = [pc[2], pc[6]]
        edge37 = [pc[3], pc[7]]

        edges = [edge01, edge02, edge13, edge23, edge45, edge46, edge57, edge67, edge04, edge15, edge26, edge37]

        return edges

    def to_mesh(self):
        vertices = np.matmul(to_numpy(self.rot_mat), to_numpy(self.control_mesh.vertices).transpose()).transpose()
        vertices = vertices + to_numpy(self.t)
        mesh = trimesh.Trimesh(vertices, self.control_mesh.faces)
        return mesh


def get_point_comps(pc):
    pc = to_numpy(pc)
    rot_mat, t, extents = get_bbox(pc)
    scale = max(extents)
    dists = cdist(pc, pc)
    
    graph = nx.Graph()
    for p_i in range(len(pc)):
        graph.add_node(p_i)
        for p_j in range(len(pc)):
            if p_j != p_i and dists[p_i, p_j] < scale * 0.2:
                graph.add_edge(p_i, p_j)
                graph.add_edge(p_j, p_i)
        
    number_of_components = nx.number_connected_components(graph)
    print('number of components', number_of_components)
    return number_of_components

def get_bbox(pc):

    try:
        to_origin, extents = trimesh.bounds.oriented_bounds(pc, angle_digits=10)
        t = to_origin[:3, :3].transpose().dot(-to_origin[:3, 3])
        size = extents*0.5
        xdir = to_origin[0, :3]
        ydir = to_origin[1, :3]
    except:
        center = pc.mean(axis=0, keepdims=True)
        points = pc - center
        t = center[0, :]
        pca = PCA()
        pca.fit(points)
        pcomps = pca.components_
        points_local = np.matmul(pcomps, points.transpose()).transpose()
        size = (points_local.max(axis=0) - points_local.min(axis=0))*0.5
        xdir = pcomps[0, :]
        ydir = pcomps[1, :]

    xdir /= np.linalg.norm(xdir)
    ydir /= np.linalg.norm(ydir)
    zdir = np.cross(xdir, ydir)
    rotmat = np.vstack([xdir, ydir, zdir]).T

    eps = 1e-6
    for i in range(len(size)):
        if size[i] < eps:
            size[i] = eps

    return rotmat, t, size

def get_yaxis_aligned_bbox(pc):

    try:

        pc_2d = np.concatenate((np.expand_dims(pc[:, 0], axis=1), np.expand_dims(pc[:,2], axis=1)), axis=1)    
        to_origin, extents = trimesh.bounds.oriented_bounds(pc_2d, angle_digits=10)
        t_xz = to_origin[:2, :2].transpose().dot(-to_origin[:2, 2])
        size_y = np.max(pc[:,1]) - np.min(pc[:,1])
        t = np.array([t_xz[0], np.min(pc[:,1])+size_y*0.5, t_xz[1]])
        size = np.array([extents[0]*0.5, size_y*0.5, extents[1]*0.5])
        xdir = np.array([to_origin[0, 0], 0, to_origin[0, 1]])
        zdir = np.array([to_origin[1, 0], 0, to_origin[1, 1]])
        ydir = np.cross(xdir, zdir)
        rotmat = np.vstack([xdir, ydir, zdir]).T

    except:

        center = pc.mean(axis=0, keepdims=True)
        points = pc - center
        t = center[0, :]
        pca = PCA()
        pca.fit(points)
        pcomps = pca.components_
        points_local = np.matmul(pcomps, points.transpose()).transpose()
        size = (points_local.max(axis=0) - points_local.min(axis=0))*0.5
        xdir = pcomps[0, :]
        ydir = pcomps[1, :]
        
        xdir /= np.linalg.norm(xdir)
        ydir /= np.linalg.norm(ydir)
        zdir = np.cross(xdir, ydir)
        rotmat = np.vstack([xdir, ydir, zdir]).T

    eps = 1e-6
    for i in range(len(size)):
        if size[i] < eps:
            size[i] = eps

    return rotmat, t, size
