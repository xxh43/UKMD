


import argparse
import os
import numpy as np
import math

from trimesh.util import concatenate

from models import *
from torch.utils.data import DataLoader
from torch.autograd import Variable

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

from scipy.spatial.transform import Rotation as R
from util_motion import *
from config import *

import argparse
import os
import numpy as np
import math
from models import *
from torch.utils.data import DataLoader
from torch.autograd import Variable

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#import sklearn.external.joblib as extjoblib
import joblib


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


from scipy.spatial.transform import Rotation as R
from sklearn.cluster import KMeans

from annotate import *

MSE = torch.nn.MSELoss()
#chamfer_distance = ChamferDistance()


color_palette = ['green', 'red', 'blue', 'purple', 'orange', 'yellow', 'cyan', 'maroon', 'olive', 'teal', 'navy', 'lime']

def resolve_validations(errors, ranges, motion_type):

    valid_count = 0
    valid_threshold = 50
    valid_ranges = []
    
    for i in range(len(ranges)):
        if errors[i] < valid_threshold:
            valid_count += 1
            valid_ranges.append(ranges[i])

    marker_dict = {}
    if motion_type == 'rotation':
        range_markers = np.arange(-2*np.pi, 2*np.pi, 0.1)
        min_range = 2*np.pi
        max_range = -2*np.pi
    else:
        range_markers = np.arange(-3, 3, 0.04)
        min_range = 5
        max_range = -5

    for valid_range in valid_ranges:
        min_range = min(min_range, valid_range)
        max_range = max(max_range, valid_range)
        min_distance = np.inf
        closet_marker = 0
        for i in range(len(range_markers)):
            if abs(valid_range - range_markers[i]) < min_distance:
                min_distance = abs(valid_range - range_markers[i])
                closet_marker = i
        marker_dict[closet_marker] = True
        
    marker_count = 0
    for k,v in marker_dict.items():
        if v == True:
            marker_count += 1

    print('marker_count', marker_count)

    return valid_count, marker_count, to_numpy(min_range), to_numpy(max_range)


def resolve_annotations(shapes, joints, rotation_annotations, rotation_validate_errors, rotation_ranges,
                    translation_annotations, translation_validate_errors, translation_ranges):

    if 'rotation' in motion_types:
        number_of_annotations = len(rotation_annotations)
    if 'translation' in motion_types:
        number_of_annotations = len(translation_annotations)

    k1 = 2
    k2 = 1
    k3 = 1
    k4 = 3
    k5 = 10

    shape_annotations = []
    for i in range(len(shapes)):
        shape_annotation = []
        part_as_mov = defaultdict(list)
        part_as_base = defaultdict(list)
        part_to_scale = defaultdict(list)

        for j in range(len(shapes[i].parts)):

            #max_coords = np.max(shapes[i].parts[j].pc, axis=0)
            #min_coords = np.min(shapes[i].parts[j].pc, axis=0)
            part_pc,_,_ = sample_mesh(shapes[i].parts[j].mesh)
            rot_mat, t, extents = get_bbox(part_pc)
            part_scale = np.mean(extents)

            part_to_scale[j] = part_scale

            part_annotation = []
            
            for k in range(number_of_annotations):
                joint = None
                if 'rotation' in motion_types:
                    if rotation_annotations[k][0] == (shapes[i].id, shapes[i].parts[j].id):
                        for m in range(len(joints)):
                            if joints[m].shape_id == shapes[i].id and joints[m].part_id == shapes[i].parts[j].id:
                                joint = joints[m]
                                break
                if 'translation' in motion_types:
                    if translation_annotations[k][0] == (shapes[i].id, shapes[i].parts[j].id):
                        for m in range(len(joints)):
                            if joints[m].shape_id == shapes[i].id and joints[m].part_id == shapes[i].parts[j].id:
                                joint = joints[m]
                                break

                if joint is not None:
                    rotation_confidence = 0
                    rotation_group = []
                    if 'rotation' in motion_types:

                        mov_pc = joint.mov_pc
                        base_pc = joint.base_pc
                        
                        mov_part_indices = joint.mov_part_indices
                        base_part_indices = joint.base_part_indices
                        
                        rotation_dir = rotation_annotations[k][2]
                        rotation_center = rotation_annotations[k][3]
                        rotation_group = rotation_annotations[k][-1]
                        rotation_recon_count, rotation_range_count, rotation_min_range, rotation_max_range = resolve_validations(rotation_validate_errors[k], rotation_ranges[k], 'rotation')
                        if rotation_range_count <= 2:
                            rotation_confidence = 0
                        else:
                            rotation_confidence = k1 * rotation_recon_count + k2 * rotation_range_count
                        print('rotation_recon_count', rotation_recon_count)
                        print('rotation_range_count', rotation_range_count)
                        print('rotation_confidence', rotation_confidence)

                    translation_confidence = 0
                    translation_group = []
                    if 'translation' in motion_types:
                        
                        mov_pc = joint.mov_pc
                        base_pc = joint.base_pc
                        
                        mov_part_indices = joint.mov_part_indices
                        base_part_indices = joint.base_part_indices
                        translation_dir = translation_annotations[k][2]
                        translation_center = translation_annotations[k][3]
                        translation_group = translation_annotations[k][-1]
                        translation_recon_count, translation_range_count, translation_min_range, translation_max_range = resolve_validations(translation_validate_errors[k], translation_ranges[k], 'translation')
                        if translation_range_count <= 2:
                            translation_confidence = 0
                        else:
                            translation_confidence = k1 * translation_recon_count + k2 * translation_range_count
                        print('translation_recon_count', translation_recon_count)
                        print('translation_range_count', translation_range_count)
                        print('translation_confidence', translation_confidence)

                    print('rotation_confidence', rotation_confidence)
                    print('translation_confidence', translation_confidence)
                    

                    if 'rotation' in motion_types and 'translation' not in motion_types:
                        motion_confidence = rotation_confidence
                        motion_type = 'rotation'
                        motion_center = rotation_center
                        motion_dir = rotation_dir
                        motion_min_range = min(0, rotation_min_range)
                        motion_max_range = max(0, rotation_max_range)
                    elif 'rotation' not in motion_types and 'translation' in motion_types:
                        motion_confidence = translation_confidence                   
                        motion_type = 'translation'
                        motion_dir = translation_dir
                        motion_center = translation_center
                        motion_min_range = min(0, translation_min_range)
                        motion_max_range = max(0, translation_max_range)
                    else:
                        if rotation_confidence >= translation_confidence:
                            motion_confidence = rotation_confidence
                            motion_type = 'rotation'
                            motion_center = rotation_center
                            motion_dir = rotation_dir
                            motion_min_range = min(0, rotation_min_range)
                            motion_max_range = max(0, rotation_max_range)
                        else:
                            motion_confidence = translation_confidence                   
                            motion_type = 'translation'
                            motion_dir = translation_dir
                            motion_center = translation_center
                            motion_min_range = min(0, translation_min_range)
                            motion_max_range = max(0, translation_max_range)
                        
                    for l in mov_part_indices:
                        part_as_mov[l].append(motion_confidence)

                    for l in base_part_indices:
                        part_as_base[l].append(motion_confidence)

                    part_annotation = [(shapes[i].id, shapes[i].parts[j].id), motion_type, motion_dir, motion_center, motion_min_range, motion_max_range, mov_pc, base_pc, rotation_group, translation_group]
                    
                    break
            
            shape_annotation.append(part_annotation)
        
        #print('part_as_mov', part_as_mov)
        #print('part_as_base', part_as_base)
        #print('part_to_scale', part_to_scale)

        best_static_part_index = 0
        max_static_confidence = 0
        best_movable_part_index = 0
        max_movable_confidence = 0
        largest_part_volume = 0
        largest_part_index = 0
        static_confidence_threshold = 100.0
        part_to_static = {}
        for j in range(len(shapes[i].parts)):

            if j not in part_as_mov:
                mov_confidence = 0.0000000000001
            else:
                mov_confidence = (k3 * np.mean(np.array(part_as_mov[j])) + k4 * len(part_as_mov[j])) 
            
            if j not in part_as_base:
                base_confidence = 0.0000000000001
            else:
                base_confidence = (k3 * np.mean(np.array(part_as_base[j])) + k5 * len(part_as_base[j])) 
            
            movable_confidence = mov_confidence/base_confidence * 1.0/part_to_scale[j]
            static_confidence = base_confidence/mov_confidence * part_to_scale[j]

            if static_confidence > movable_confidence:
                part_to_static[j] = True
            else:
                part_to_static[j] = False

            #print('movable_confidence', movable_confidence)
            #print('static_confidence', static_confidence)

            if static_confidence >= max_static_confidence:
                max_static_confidence = static_confidence
                best_static_part_index = j 

            if movable_confidence >= max_movable_confidence:
                max_movable_confidence = movable_confidence
                best_movable_part_index = j

            #if part_to_scale[j] > largest_part_volume:
                #largest_part_volume = part_to_scale[j]
                #largest_part_index = j

        part_to_static[best_movable_part_index] = False
        part_to_static[best_static_part_index] = True
        
        if largest_as_static:
            part_to_static[largest_part_index] = True

        for j in range(len(shapes[i].parts)):
            if part_to_static[j] is True and len(shape_annotation[j])>0:
                shape_annotation[j] = (shape_annotation[j][0], 'static', shape_annotation[j][2], shape_annotation[j][3], shape_annotation[j][4], shape_annotation[j][5], mov_pc, base_pc, None, None)
        shape_annotations.append(shape_annotation)

    return shape_annotations

def resolve_func(shapes, joints, folder):
 
    final_rotation_annotations = []
    final_rotation_validate_range_errors = []
    final_rotation_validate_ranges = []

    final_translation_annotations = []
    final_translation_validate_range_errors = []
    final_translation_validate_ranges = []

    for joint in joints:

        best_rotation_annotation_error = np.inf
        best_rotation_annotation = None
        best_rotation_validate_range_errors = None
        best_rotation_validate_ranges = None

        best_translation_annotation_error = np.inf
        best_translation_annotation = None
        best_translation_validate_range_errors = None
        best_translation_validate_ranges = None

        #for iteration in range(0, 3):
        for iteration in range(1, annotate_max_iteration):
            iteration_folder = os.path.join(folder, str(iteration))

            if 'rotation' in motion_types:
                rotation_annotations = joblib.load(os.path.join(iteration_folder, 'rotation_annotations.joblib'))
                rotation_validate_range_errors = joblib.load(os.path.join(iteration_folder, 'rotation_validate_range_errors.joblib'))
                rotation_validate_ranges = joblib.load(os.path.join(iteration_folder, 'rotation_validate_ranges.joblib'))
                for i in range(len(rotation_annotations)):
                    if rotation_annotations[i][0][0] == joint.shape_id and rotation_annotations[i][0][1] == joint.part_id:
                        if rotation_annotations[i][1] < best_rotation_annotation_error:
                            best_rotation_annotation_error = rotation_annotations[i][1]
                            best_rotation_annotation = rotation_annotations[i]
                            best_rotation_validate_range_errors = rotation_validate_range_errors[i]
                            best_rotation_validate_ranges = rotation_validate_ranges[i]
            
            if 'translation' in motion_types:
                translation_annotations = joblib.load(os.path.join(iteration_folder, 'translation_annotations.joblib'))
                translation_validate_range_errors = joblib.load(os.path.join(iteration_folder, 'translation_validate_range_errors.joblib'))
                translation_validate_ranges = joblib.load(os.path.join(iteration_folder, 'translation_validate_ranges.joblib'))
                for i in range(len(translation_annotations)):
                    if translation_annotations[i][0][0] == joint.shape_id and translation_annotations[i][0][1] == joint.part_id:
                        if translation_annotations[i][1] < best_translation_annotation_error:
                            best_translation_annotation_error = translation_annotations[i][1]
                            best_translation_annotation = translation_annotations[i]
                            best_translation_validate_range_errors = translation_validate_range_errors[i]
                            best_translation_validate_ranges = translation_validate_ranges[i]
        
        if 'rotation' in motion_types:
            print('best_rotation_annotation', best_rotation_annotation)
            final_rotation_annotations.append(best_rotation_annotation)
            final_rotation_validate_range_errors.append(best_rotation_validate_range_errors)
            final_rotation_validate_ranges.append(best_rotation_validate_ranges)

        if 'translation' in motion_types:
            print('best_translation_annotation', best_translation_annotation)
            final_translation_annotations.append(best_translation_annotation)
            final_translation_validate_range_errors.append(best_translation_validate_range_errors)
            final_translation_validate_ranges.append(best_translation_validate_ranges)

    shape_annotations = resolve_annotations(shapes, joints, final_rotation_annotations, final_rotation_validate_range_errors, final_rotation_validate_ranges,\
                                            final_translation_annotations, final_translation_validate_range_errors, final_translation_validate_ranges)
    print('folder', folder)
    joblib.dump(shape_annotations, os.path.join(folder, 'shape_annotations.joblib'))

    