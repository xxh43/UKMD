
from pathlib import Path
import json
from numpy import arange
import networkx as nx
from collections import defaultdict
import copy
from objects import *
import os
from util_mesh import *
from config import *
import shutil
from util_motion import *
from deform import *
import joblib
import queue
import os
cur_dir = os.path.dirname(__file__)
partnet_geo_path = os.path.join(cur_dir, '../data/partnet/geo')
partnet_precomputed_path = os.path.join(cur_dir,'../data/partnet/precomputed')


class Node:
    def __init__(self):
        self.id = None
        self.objs = []

def recur_build_dict(node_jo, id2node):

    if 'children' not in node_jo:
        if 'objs' in node_jo:
            node = Node()
            node.id = node_jo['id']
            node.objs = node_jo['objs']
            id2node[node.id] = node
        return 

    for c_jo in node_jo['children']:
        recur_build_dict(c_jo, id2node) 

def build_dict(json_filename):
    id2node = defaultdict()
    with open(json_filename) as json_file:
        jo = json.load(json_file)
        if 'children' not in jo[0]:
            if 'objs' in jo[0]:
                node = Node()
                node.id = jo[0]['id']
                node.objs = jo[0]['objs']
                id2node[node.id] = node
            return id2node

        for c_jo in jo[0]['children']:
            recur_build_dict(c_jo, id2node)

    return id2node

def label_root_parts(parts):
    shape_min_height = np.inf
    shape_max_height = -np.inf
    part_min_heights = []
    for i in range(len(parts)):
        part = parts[i]
        part_pc, _, _ = sample_mesh(part.mesh)
        part_min_height = np.min(part_pc[:, 1])
        part_max_height = np.max(part_pc[:, 1])
        shape_min_height = min(shape_min_height, part_min_height)
        shape_max_height = max(shape_max_height, part_max_height)
        part_min_heights.append(part_min_height)

    ground_height = shape_min_height + 0.05 * (shape_max_height - shape_min_height)
    for i in range(len(parts)):
        if part_min_heights[i] < ground_height:
            parts[i].is_root = True
        else:
            parts[i].is_root = False
    
def is_connected_to_root(parts):
    for i in range(len(parts)):
        if parts[i].is_root:
            return True
    return False

def get_parts_graph_from_file(precomputed_json_file):
    graph = nx.Graph()
    with open(precomputed_json_file) as json_file:
        conn_graph = json.load(json_file)['connectivityGraph']
        for si in range(len(conn_graph)):
            graph.add_node(si)
            for ei in conn_graph[si]:
                e_tuple = (si, ei)
                graph.add_edge(*e_tuple)

    return graph


def get_component(g, vertex, visited):
    comp = []
    q = queue.Queue()
    q.put(vertex)
    visited.add(vertex)
    while not q.empty():
        cur = q.get()
        comp.append(cur)
        for nb in g.neighbors(cur):
            if nb not in visited:
                q.put(nb)
                visited.add(nb)

    return comp

def get_parts_graph(parts):

    node_has_edge = set()
    graph = nx.Graph()
    for i in range(len(parts)):
        graph.add_node(i)
        nearest_part_distance = np.inf
        nearest_part_index = None
        for j in range(i+1, len(parts)):
            part_i_pc, _, _ = sample_mesh(parts[i].mesh, 4096)
            part_j_pc, _, _ = sample_mesh(parts[j].mesh, 4096)
            dists = cdist(part_i_pc, part_j_pc)
            min_dist = min(dists.flatten())
            if min_dist < 0.1:
                node_has_edge.add(i)
                node_has_edge.add(j)
                graph.add_edge(i, j)
            
            if min_dist < nearest_part_distance:
                nearest_part_distance = min_dist
                nearest_part_index = j

        if i not in node_has_edge and nearest_part_index is not None:
            graph.add_edge(i, nearest_part_index)
            graph.add_edge(nearest_part_index, i)

    return graph

def get_parts_from_file(precomputed_json_file, geo_path, id2node):
    mesh_path = os.path.join(geo_path, 'textured_objs')
    mobility_file = os.path.join(geo_path, 'mobility_v2.json')
    parts = []


    precomputed_part_infos = json.load(open(precomputed_json_file))['parts']
    mobility_part_infos = json.load(open(mobility_file))

    for part_index in range(len(precomputed_part_infos)):
        precomputed_part_info = precomputed_part_infos[part_index]

        name = precomputed_part_info['name']
    
        node_jos = precomputed_part_info['ids']
        meshes = []
        for node_id in node_jos:
            node = id2node[node_id]
            for obj in node.objs:
                mesh = load_mesh(os.path.join(mesh_path, obj + '.obj'))
                meshes.append(mesh)
        part_mesh, _ = merge_meshes(meshes)
        part = Part()
        part.mesh = part_mesh

        motion_info = mobility_part_infos[part_index]

        if motion_info['joint'] == 'hinge':
            part.gt_motion_type = 'rotation'
        elif motion_info['joint'] == 'slider':
            part.gt_motion_type = 'translation'
        else:
            part.gt_motion_type = 'static'

        if part.gt_motion_type != 'static': 
            motion_parameters = mobility_part_infos[part_index]['jointData']
            if bool(motion_parameters):
                part.gt_axis = np.squeeze(np.array(motion_parameters['axis']['direction']))
                part.gt_center = np.squeeze(np.array(motion_parameters['axis']['origin']))

                if part.gt_motion_type == 'rotation':
                    if bool(motion_parameters['limit']['noLimit']) != False:
                        part.gt_min_range = -2*np.pi
                        part.gt_max_range = 2*np.pi
                    else:
                        a_val = float(motion_parameters['limit']['a'])
                        if a_val > 0:
                            a_range = (float(motion_parameters['limit']['a'])%360.0)/360.0*2*np.pi
                        else:
                            a_range = (float(motion_parameters['limit']['a'])%(-360.0))/360.0*2*np.pi
                        
                        b_val = float(motion_parameters['limit']['b'])
                        if b_val > 0:
                            b_range = (float(motion_parameters['limit']['b'])%360.0)/360.0*2*np.pi
                        else:
                            b_range = (float(motion_parameters['limit']['b'])%(-360.0))/360.0*2*np.pi

                        part.gt_min_range = min(a_range, b_range)
                        part.gt_max_range = max(a_range, b_range)

                else:
                    a_val = float(motion_parameters['limit']['a'])
                    b_val = float(motion_parameters['limit']['b'])
                    a_range = a_val
                    b_range = b_val

                    part.gt_min_range = min(a_range, b_range)
                    part.gt_max_range = max(a_range, b_range)

                if part.gt_min_range < -0.01 and part.gt_max_range < -0.01:
                    return []
                if part.gt_min_range > 0.01 and part.gt_max_range > 0.01:
                    return []

            part.gt_axis = part.gt_axis/np.linalg.norm(part.gt_axis)
        parts.append(part)

    return parts

def build_raw_shape(id, parts, part_graph):

    shape = RawShape()
    shape.id = id
    shape.parts = parts

    part_pcs = []
    for i, part in enumerate(shape.parts):
        pc, _, _ = sample_mesh(part.mesh, 512)
        part_pcs.append(pc)
    shape.part_graph = part_graph
    label_root_parts(shape.parts)

    return shape

def get_parts_from_shape(old_parts):
    parts = copy.deepcopy(old_parts)
    label_root_parts(parts)
    return parts

def get_max_neighbor_parts(part_index, part_graph, parts, k):
    nbs = list(part_graph.neighbors(part_index))
    nb_areas = []
    for nb in nbs:
        nb_areas.append(parts[nb].mesh.area)
    top_nbs = [x for _, x in sorted(zip(nb_areas, nbs), key=lambda pair: pair[0], reverse=True)][0:k] 
    return top_nbs

def pc_in_box(pc, pc_normal, z, t, rotmat):

    pc = torch.tensor(pc)
    z = torch.tensor(z)
    t = torch.tensor(t)
    rotmat = torch.tensor(rotmat)
    standarized_pc = pc - t
    standarized_pc = torch.matmul(rotmat.inverse(), standarized_pc.transpose(0,1)).transpose(0,1)
    repeated_zs = z.unsqueeze(dim=0).repeat_interleave(len(pc), dim=0)
    dists = repeated_zs - torch.abs(standarized_pc) 
    threshold = 0.001

    inside_pc = []
    for i in range(len(pc)):
        if dists[i][0] > -threshold and dists[i][1] > -threshold and dists[i][2] > -threshold:
            inside_pc.append(pc[i])

    inside_pc = to_numpy(torch.stack(inside_pc))
    filter_indices = np.random.choice(arange(0, len(inside_pc)), part_point_num)
    filtered_inside_pc = inside_pc[filter_indices]
    filtered_inside_pc_normal = pc_normal
    return filtered_inside_pc, filtered_inside_pc_normal


def get_final_base_parts(top_nb_part_indices, parts):
    max_nb_part_pc, _, _ = sample_mesh(parts[top_nb_part_indices[0]].mesh, part_point_num)
    max_nb_box_rotmat, max_nb_box_t, max_nb_box_z = get_yaxis_aligned_bbox(max_nb_part_pc)
    temp_meshes = [parts[v].mesh for v in top_nb_part_indices]
    temp_mesh, _ = merge_meshes(temp_meshes)
    temp_pc, temp_pc_normal, _ = sample_mesh(temp_mesh, 4096)
    filtered_base_pc, filtered_base_pc_normal = pc_in_box(temp_pc, temp_pc_normal, max_nb_box_z, max_nb_box_t, max_nb_box_rotmat)
    return [top_nb_part_indices[0]], filtered_base_pc, filtered_base_pc_normal


def build_my_shape(shape_id, parts, part_graph):

    shape = Shape()
    shape.id = shape_id
    shape.part_graph = part_graph
    
    cut_parts = list(nx.articulation_points(part_graph))
    for part_index in range(len(parts)):
        part = copy.deepcopy(parts[part_index])
        part.shape_id = shape.id
        part.id = part_index
        shape.parts.append(part)

        max_nb_k = 10
        mov_part_indices = [part_index]
        base_part_indices = []
        if part.is_root or part_index not in cut_parts:
            max_nbs = get_max_neighbor_parts(part_index, part_graph, parts, max_nb_k)
            base_part_indices, base_pc, base_pc_normal = get_final_base_parts(max_nbs, parts)
        else:
            nbs = list(part_graph.neighbors(part_index))
            visited = set()
            visited.add(part_index)
            for nb in nbs:
                if nb not in visited:
                    comp = get_component(part_graph, nb, visited)
                    comp_parts = [parts[v] for v in comp]
                    if not is_connected_to_root(comp_parts):
                        mov_part_indices += comp

            max_nbs = get_max_neighbor_parts(part_index, part_graph, parts, max_nb_k)
            base_part_indices, base_pc, base_pc_normal = get_final_base_parts(max_nbs, parts)

        # ------------------------------------------------------------------------------------------------------------------------------------
        if len(set(mov_part_indices).intersection(set(base_part_indices))) > 0:
            return None

        mov_meshes = [parts[v].mesh for v in mov_part_indices]
        mov_mesh, _ = merge_meshes(mov_meshes)
        
        mov_pc, mov_pc_normal, indices = sample_mesh(mov_mesh, part_point_num)

        joint = Joint()
        joint.shape_id = shape.id
        joint.part_id = part.id

        joint.mov_part_indices = mov_part_indices
        joint.base_part_indices = base_part_indices
        
        rot_mat, t, extents = get_bbox(mov_pc)
        handle = DeformHandle()
        handle.rot_mat = rot_mat
        handle.t = t
        handle.z = extents
        joint.mov_deform_handle = handle
        
        rot_mat, t, extents = get_yaxis_aligned_bbox(base_pc)
        handle = DeformHandle()
        handle.rot_mat = rot_mat
        handle.t = t
        handle.z = extents 
        joint.base_deform_handle = handle

        joint.gt_motion_type = part.gt_motion_type
        joint.gt_axis = part.gt_axis
        joint.gt_center = part.gt_center
        joint.gt_min_range = part.gt_min_range
        joint.gt_max_range = part.gt_max_range
        joint.mov_pc = copy.deepcopy(mov_pc)
        joint.base_pc = copy.deepcopy(base_pc)
        joint.get_mov_comp_num()
        joint.get_base_comp_num()
        shape.joints.append(joint)

    return shape

def add_motion_to_parts(pivot_part, affected_parts, other_parts, motion_level=add_motion_level):

    min_range = pivot_part.gt_min_range * 0.2 * motion_level
    max_range = pivot_part.gt_max_range * 0.2 * motion_level

    sampled_motion_range = min_range + np.random.random() * (max_range - min_range)

    if pivot_part.gt_motion_type == 'rotation':
        pivot_part.mesh.vertices = to_numpy(rotate_with_axis_center_angle(pivot_part.mesh.vertices, pivot_part.gt_axis, pivot_part.gt_center, sampled_motion_range))
    else:          
        pivot_part.mesh.vertices = to_numpy(translate_with_vector(pivot_part.mesh.vertices, pivot_part.gt_axis * sampled_motion_range))
    
    for part in affected_parts:
        if pivot_part.gt_motion_type == 'rotation':
            part.mesh.vertices = to_numpy(rotate_with_axis_center_angle(part.mesh.vertices, pivot_part.gt_axis, pivot_part.gt_center, sampled_motion_range))
            part.gt_center = to_numpy(rotate_with_axis_center_angle(np.expand_dims(part.gt_center, axis=0), pivot_part.gt_axis, pivot_part.gt_center, sampled_motion_range)[0])
            part.gt_axis = to_numpy(rotate_with_axis_center_angle(np.expand_dims(part.gt_axis, axis=0), pivot_part.gt_axis, torch.zeros(3), sampled_motion_range)[0])
            part.gt_axis = part.gt_axis/np.linalg.norm(part.gt_axis)
        else:          
            part.mesh.vertices = to_numpy(translate_with_vector(part.mesh.vertices, pivot_part.gt_axis * sampled_motion_range))
            part.gt_center = to_numpy(translate_with_vector(np.expand_dims(pivot_part.gt_center, axis=0), pivot_part.gt_axis * sampled_motion_range)[0])


def add_motion_to_shape(shape, motion_level=add_motion_level):

    cut_parts = list(nx.articulation_points(shape.part_graph))

    for part_index in range(len(shape.parts)):
        pivot_part = shape.parts[part_index]
        if pivot_part.gt_motion_type != 'static':
            affected_parts = []
            other_parts = []
        
            nbs = list(shape.part_graph.neighbors(part_index))
            visited = set()
            visited.add(part_index)
            if not pivot_part.is_root:
                if part_index in cut_parts:
                    for nb in nbs:
                        if nb not in visited:
                            comp = get_component(shape.part_graph, nb, visited)
                            comp_parts = [shape.parts[v] for v in comp]
                            if not is_connected_to_root(comp_parts):
                                affected_parts += comp_parts
            
            for p_i in range(len(shape.parts)):
                if shape.parts[p_i] != pivot_part and shape.parts[p_i] not in affected_parts:
                    other_parts.append(shape.parts[p_i])

            add_motion_to_parts(pivot_part, affected_parts, other_parts, motion_level)

def filter_parts(parts):
    
    filtered_parts = []

    part_meshes = [part.mesh for part in parts]
    shape_mesh, _ = merge_meshes(part_meshes)
    shape_pc, _, _ = sample_mesh(shape_mesh, 2048)
    _, _, shape_extents = get_bbox(shape_pc)

    shape_vol = max(shape_extents)

    for part_index in range(len(parts)):
        part = copy.deepcopy(parts[part_index])
        
        part_pc, _, _ = sample_mesh(part.mesh, 2048)
        
        if np.isnan(part_pc).any() or np.isinf(part_pc).any():
            continue
        
        part_comp_count = get_point_comps(part_pc)
        if part_comp_count >= 2:
            continue
        
        _, _, part_extents = get_bbox(part_pc)
        
        part_vol = max(part_extents)
        if part_vol/shape_vol < 0.15:
            continue
        
        filtered_parts.append(part)

    return filtered_parts


def process_category_shapes(category):

    folder = os.path.join(processed_partnet_folder, category)
    if not os.path.exists(folder):
        os.makedirs(folder)

    for shape_id in os.listdir(partnet_geo_path):

        if os.path.isfile(os.path.join(partnet_precomputed_path, shape_id + '.artpre.json')) is False:
            continue

        shape_category = json.load(open(os.path.join(partnet_geo_path, shape_id, 'meta.json')))['model_cat']
        if shape_category != category:
            continue
        
        print('processing category ------------------------------- : ', category, 'shape_id', shape_id)

        id2node = build_dict(os.path.join(partnet_geo_path, shape_id, 'result.json'))
        parts = get_parts_from_file(os.path.join(partnet_precomputed_path, shape_id + '.artpre.json'), os.path.join(partnet_geo_path, shape_id), id2node)
        if len(parts) == 0:
            continue

        part_graph = get_parts_graph_from_file(os.path.join(partnet_precomputed_path, shape_id + '.artpre.json'))
        raw_shape = build_raw_shape(shape_id, parts, part_graph)

        if raw_shape is not None:
            try:
                add_motion_to_shape(raw_shape)
            except:
                continue
            
            parts = get_parts_from_shape(raw_shape.parts)
            parts = filter_parts(parts)
            if len(parts) <= 1:
                continue
            
            part_graph = get_parts_graph(parts)
            if len(part_graph.edges) == 0 or len(list(nx.isolates(part_graph)))>0:
                continue

            my_shape = build_my_shape(shape_id, parts, part_graph)
            if my_shape is not None:
                joblib.dump(my_shape, os.path.join(folder, str(shape_id) + '.joblib'))
    
    train_shape_ids, test_shape_ids = prepare_train_test_split(category, processed_partnet_folder)
    joblib.dump(train_shape_ids, os.path.join(folder, 'train_shape_files'))
    joblib.dump(test_shape_ids, os.path.join(folder, 'test_shape_files'))
    
    
def prepare_train_test_split(category, folder):
    shapes, _ = get_shapes(folder, category, use_train_file=False, use_test_file=False, interested_option='both')
    shape_indices = np.array(list(range(0, len(shapes))))
    sample_size = int(0.6 * len(shapes))
    sampled_shape_indices = np.random.choice(shape_indices, sample_size, replace=False, p=np.array([1.0/len(shape_indices)]*len(shape_indices)))
    train_shape_ids = []
    test_shape_ids = []
    for shape_index in range(0, len(shapes)):
        print('shapes[shape_index]', shapes[shape_index])
        if shape_index in sampled_shape_indices:
            train_shape_ids.append(shapes[shape_index].id+'.joblib')
        else:
            test_shape_ids.append(shapes[shape_index].id+'.joblib')

    return train_shape_ids, test_shape_ids

def get_base_net_shapes(category, use_train_file, use_test_file):
    shapes, _ = get_shapes(processed_partnet_folder, category, use_train_file, use_test_file, interested_option='both')
    return shapes

def get_shapes(folder, category, use_train_file, use_test_file, interested_option):
    
    print('folder', folder)

    interested_shape_files = []
    if use_train_file is True and use_test_file is True:
        train_shape_files = joblib.load(os.path.join(folder, category, 'train_shape_files'))
        test_shape_files = joblib.load(os.path.join(folder, category, 'test_shape_files'))
        shape_files = train_shape_files + test_shape_files
        if interested_option == 'test':
            interested_shape_files = test_shape_files
        elif interested_option == 'train':
            interested_shape_files = train_shape_files
        elif interested_option == 'both':
            interested_shape_files = train_shape_files + test_shape_files
        else:
            print('wrong interested option')
            exit()
    elif use_train_file is False and use_test_file is True:
        test_shape_files = joblib.load(os.path.join(folder, category, 'test_shape_files'))
        shape_files = test_shape_files
        interested_shape_files = test_shape_files
    elif use_train_file is True and use_test_file is False:
        train_shape_files = joblib.load(os.path.join(folder, category, 'train_shape_files'))
        shape_files = train_shape_files
        interested_shape_files = train_shape_files
    elif use_train_file is False and use_test_file is False:
        shape_files = os.listdir(os.path.join(folder, category))
        interested_shape_files = []
    else:
        print('get shape error')
        exit()

    sorted_shape_files = sorted(shape_files)
    
    shapes = []
    interested_shape_indices = []
    for i, shape_file in enumerate(sorted_shape_files):
        if shape_file.endswith('.joblib'):
            shape = joblib.load(os.path.join(folder, category, shape_file))
            if type(shape) is Shape:
                shapes.append(shape)
                if shape_file in interested_shape_files:
                    interested_shape_indices.append(i)
            
    return shapes, interested_shape_indices

def get_joints(shapes, interested_shape_indices):
    joints = []
    interested_joint_indices = []
    for i, shape in enumerate(shapes):        
        if i in interested_shape_indices:
            for j in range(len(shape.joints)):
                interested_joint_indices.append(len(joints)+j)
        joints += shape.joints

    ret_joints = joints
    return ret_joints, interested_joint_indices

if __name__ == "__main__":
    process_category_shapes('Box')


