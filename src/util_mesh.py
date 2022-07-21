
import copy
import numpy as np
import trimesh
import math

def load_mesh(obj_filename):
    tri_obj = trimesh.load_mesh(obj_filename)
    if tri_obj.is_empty:
        return None
    if type(tri_obj) is trimesh.scene.scene.Scene:
        tri_mesh = tri_obj.dump(True)
    else:
        tri_mesh = tri_obj

    return tri_mesh
    
def merge2mesh(mesh1, mesh2):
    new_mesh = copy.deepcopy(mesh1)
    shifted_mesh2_faces = copy.deepcopy(mesh2.faces) + copy.deepcopy(mesh1.vertices.shape[0])
    new_mesh.faces = np.concatenate((copy.deepcopy(mesh1.faces), copy.deepcopy(shifted_mesh2_faces)))
    new_mesh.vertices = np.concatenate((copy.deepcopy(mesh1.vertices), copy.deepcopy(mesh2.vertices)))
    return new_mesh

def merge_meshes(meshes):
    if len(meshes) == 0:
        return None
    base_mesh = meshes[0]
    face_labels = [0]*len(base_mesh.faces)
    for i in range(1, len(meshes)):
        base_mesh = merge2mesh(base_mesh, meshes[i])
        face_labels += [i]*len(meshes[i].faces)
    return base_mesh, face_labels

def sample_mesh(mesh, amount = 1000):

    if mesh is None:
        return np.zeros(amount)

    mesh.fix_normals()
    pc, face_indices = trimesh.sample.sample_surface(mesh, amount)
    for i in range(len(pc)):
        point = pc[i]
        face_index = face_indices[i]

    return pc, None, face_indices

    