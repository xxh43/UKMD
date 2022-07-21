

import sys
from numpy.core.defchararray import center

from trimesh.util import concatenate
sys.path.append('..')


import argparse
from logging import error
import os
from matplotlib.pyplot import axis
import numpy as np
import math
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
from annotate import *

import numpy as np
import math

import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

def append_items_to_file(items, filename):
    with open(filename, 'a') as f:
        for item in items:
            f.write(str(item) + ' ')
        f.write('\n')


def compute_center_error(pred_center, gt_center, gt_axis, scale):
    gt_axis = gt_axis / np.linalg.norm(gt_axis)
    error_vector = to_numpy(pred_center) - to_numpy(gt_center)
    error_distance = np.linalg.norm(error_vector - gt_axis * np.dot(error_vector, gt_axis))
    error_ratio = error_distance/scale
    
    return error_ratio

def compute_rotation_range_error(pred_min_range, pred_max_range, gt_min_range, gt_max_range):

    print('gt_min_range', gt_min_range)
    print('gt_max_range', gt_max_range)

    delta = 0.0000
    if pred_min_range < -2*np.pi:
        pred_min_range = pred_min_range%(-2*np.pi+delta)
    if pred_max_range > 2*np.pi:
        pred_max_range = pred_max_range%(2*np.pi+delta)
    #gt_min_range = gt_min_range%(-2*np.pi+delta)
    #gt_max_range = gt_max_range%(2*np.pi+delta)

    if gt_min_range <= -2 * np.pi:
        gt_min_range = -2 * np.pi

    if gt_max_range >= 2 * np.pi:  
        gt_max_range = 2 * np.pi

    max_of_min_range = max(pred_min_range, gt_min_range)
    min_of_max_range = min(pred_max_range, gt_max_range)
    range_intersection = min_of_max_range - max_of_min_range

    min_of_min_range = min(pred_min_range, gt_min_range)
    max_of_max_range = max(pred_max_range, gt_max_range)
    range_union = max_of_max_range - min_of_min_range

    iou = range_intersection/range_union

    if iou < 0:
        iou = 0

    print('pred_min_range', pred_min_range)
    print('pred_max_range', pred_max_range)
    print('gt_min_range', gt_min_range)
    print('gt_max_range', gt_max_range)
    print('iou', iou)
    return iou

def compute_translation_range_error(pred_min_range, pred_max_range, gt_min_range, gt_max_range):
    max_of_min_range = max(pred_min_range, gt_min_range)
    min_of_max_range = min(pred_max_range, gt_max_range)
    range_intersection = min_of_max_range - max_of_min_range

    min_of_min_range = min(pred_min_range, gt_min_range)
    max_of_max_range = max(pred_max_range, gt_max_range)
    range_union = max_of_max_range - min_of_min_range
    

    iou = range_intersection/range_union

    #print('trans inter')
    print('pred_min_range', pred_min_range)
    print('pred_max_range', pred_max_range)
    print('gt_min_range', gt_min_range)
    print('gt_max_range', gt_max_range)
    print('iou', iou)
    if iou < 0:
        iou = 0

    return iou

def compute_motion_paramter_error(pred_axis, pred_center, pred_min_range, pred_max_range, gt_axis, gt_center, gt_min_range, gt_max_range, motion_type, part_scale):

    #if gt_min_range <= -np.pi:
        #gt_min_range = -np.pi

    #if gt_max_range >= np.pi:  
        #gt_max_range = np.pi

    #pred_min_range = pred_min_range % (2*np.pi)
    #pred_max_range = pred_max_range % (2*np.pi)
    #gt_min_range = max(gt_min_range, -(2*np.pi))
    #gt_max_range = min(gt_max_range, (2*np.pi))

    pred_axis = pred_axis/np.linalg.norm(pred_axis)
    gt_axis = gt_axis/np.linalg.norm(gt_axis)
    dotval = np.dot(pred_axis, gt_axis)
    dotval = np.clip(dotval, -1.0, 1.0)
    if dotval >= 0:
        axis_error = math.acos(dotval) / np.pi * 180
        if motion_type == 'rotation':
            center_error = compute_center_error(pred_center, gt_center, gt_axis, part_scale)
            range_iou = compute_rotation_range_error(pred_min_range, pred_max_range, gt_min_range, gt_max_range)
            #range_iou_2 = compute_rotation_range_error(-pred_max_range, -pred_min_range, gt_min_range, gt_max_range)
            #range_iou = max(range_iou_1, range_iou_2)
        else:
            center_error = 0
            range_iou = compute_translation_range_error(pred_min_range, pred_max_range, gt_min_range, gt_max_range)
            #range_iou_2 = compute_translation_range_error(-pred_max_range, -pred_min_range, gt_min_range, gt_max_range)
            #range_iou = max(range_iou_1, range_iou_2)
            
    else:
        axis_error = abs(np.pi - math.acos(dotval)) / np.pi * 180
        if motion_type == 'rotation':
            center_error = compute_center_error(pred_center, gt_center, gt_axis, part_scale)
            #range_iou_1 = compute_rotation_range_error(pred_min_range, pred_max_range, gt_min_range, gt_max_range)
            range_iou = compute_rotation_range_error(-pred_max_range, -pred_min_range, gt_min_range, gt_max_range)
            #range_iou = max(range_iou_1, range_iou_2)
        else:
            center_error = 0
            #range_iou_1 = compute_translation_range_error(pred_min_range, pred_max_range, gt_min_range, gt_max_range)
            range_iou = compute_translation_range_error(-pred_max_range, -pred_min_range, gt_min_range, gt_max_range)
            #range_iou = max(range_iou_1, range_iou_2)

    return axis_error, center_error, range_iou

def compute_accuracy_func(shapes, shape_annotations, include_static, folder):

    accuracy_filename = 'accuracy.txt'
    if include_static is False:
        accuracy_filename = 'accuracy_no_static.txt'

    if os.path.isfile(os.path.join(folder,accuracy_filename)):
        os.remove(os.path.join(folder,accuracy_filename))

    total_axis_error = 0
    total_center_error = 0
    total_range_iou = 0
    
    all_part_count = 0
    correctly_assigned_static_part_count = 0
    correctly_assigned_rotation_part_count = 0
    correctly_assigned_translation_part_count = 0
    
    rotation_as_translation = 0
    rotation_as_static = 0
    translation_as_rotation = 0
    translation_as_static = 0
    static_as_rotation = 0
    static_as_translation = 0

    summary_infos = []

    for i in range(len(shapes)):
        shape_annotation = shape_annotations[i]
        for j in range(len(shapes[i].parts)):

            if include_static is False:
                if shapes[i].parts[j].gt_motion_type == 'static':
                    continue

            part_annotation = shape_annotation[j]
            if int(shapes[i].id) != int(part_annotation[0][0]) or int(shapes[i].parts[j].id) != int(part_annotation[0][1]):
                print('id check fail')
                exit()
            
            print('shape id', shapes[i].id, 'part_id', shapes[i].parts[j].id)
            summary_info = []
            summary_info += [shapes[i].id, shapes[i].parts[j].id]

            all_part_count += 1

            pred_motion_type = part_annotation[1]
            gt_motion_type = shapes[i].parts[j].gt_motion_type

            if pred_motion_type != gt_motion_type:
                if gt_motion_type == 'rotation' and pred_motion_type == 'translation':
                    rotation_as_translation += 1
                    summary_info += ['rotation_as_translation']
                    #summary_info += ['rotation_group: ', rotation_group, 'translation_group: ', translation_group]
                    summary_infos.append(summary_info)
                if gt_motion_type == 'rotation' and pred_motion_type == 'static':
                    rotation_as_static += 1
                    summary_info += ['rotation_as_static']
                    #summary_info += ['rotation_group: ', rotation_group, 'translation_group: ', translation_group]
                    summary_infos.append(summary_info)
                if gt_motion_type == 'translation' and pred_motion_type == 'rotation':
                    translation_as_rotation += 1
                    summary_info += ['translation_as_rotation']
                    #summary_info += ['rotation_group: ', rotation_group, 'translation_group: ', translation_group]
                    summary_infos.append(summary_info)
                if gt_motion_type == 'translation' and pred_motion_type == 'static':
                    translation_as_static += 1
                    summary_info += ['translation_as_static']
                    #summary_info += ['rotation_group: ', rotation_group, 'translation_group: ', translation_group]
                    summary_infos.append(summary_info)
                if gt_motion_type == 'static' and pred_motion_type == 'rotation':
                    static_as_rotation += 1
                    summary_info += ['static_as_rotation']
                    #summary_info += ['rotation_group: ', rotation_group, 'translation_group: ', translation_group]
                    summary_infos.append(summary_info)
                if gt_motion_type == 'static' and pred_motion_type == 'translation':
                    static_as_translation += 1
                    summary_info += ['static_as_translation']
                    #summary_info += ['rotation_group: ', rotation_group, 'translation_group: ', translation_group]
                    summary_infos.append(summary_info)

                continue
            
            if pred_motion_type == 'static':
                correctly_assigned_static_part_count += 1
                summary_info = [shapes[i].id, shapes[i].parts[j].id, 'static']
                summary_infos.append(summary_info)
                continue

            pred_axis = to_numpy(part_annotation[2])
            pred_axis = pred_axis/np.linalg.norm(pred_axis)
            pred_center = to_numpy(part_annotation[3])
            pred_min_range = to_numpy(part_annotation[4])
            pred_max_range = to_numpy(part_annotation[5])
            rotation_group = part_annotation[8]
            translation_group = part_annotation[9]

            print('rotation_group', rotation_group)
            print('translation_group', translation_group)

            gt_axis = shapes[i].parts[j].gt_axis
            gt_axis = gt_axis/np.linalg.norm(gt_axis)
            gt_center = shapes[i].parts[j].gt_center
            gt_min_range = shapes[i].parts[j].gt_min_range
            gt_max_range = shapes[i].parts[j].gt_max_range
            
            if pred_motion_type == 'rotation':
                correctly_assigned_rotation_part_count += 1

            if pred_motion_type == 'translation':
                correctly_assigned_translation_part_count += 1

            part_pc, _, _ = sample_mesh(shapes[i].parts[j].mesh)
            max_coords = np.max(part_pc, axis=0)
            min_coords = np.min(part_pc, axis=0)
            part_scale = np.linalg.norm(max_coords - min_coords)            
            axis_error, center_error, range_iou = \
            compute_motion_paramter_error(pred_axis, pred_center, pred_min_range, pred_max_range, gt_axis, gt_center, gt_min_range, gt_max_range, gt_motion_type, part_scale)
            
            total_axis_error += axis_error
            total_center_error += center_error
            total_range_iou += range_iou

            summary_info += ['pred_type', pred_motion_type, 'pred_axis', pred_axis, 'gt_axis', gt_axis, 'axis_error: ', axis_error, 'pred_center', pred_center, 'gt_center', gt_center, 'center_error: ', center_error, 'range iou: ', range_iou]
            summary_info += ['rotation_group: ', rotation_group, 'translation_group: ', translation_group]
            summary_infos.append(summary_info)

    #if include_static is True:
    avg_type_accuracy = (correctly_assigned_static_part_count + correctly_assigned_rotation_part_count + correctly_assigned_translation_part_count) / all_part_count

    if correctly_assigned_rotation_part_count + correctly_assigned_translation_part_count == 0:
        avg_axis_error = 0
    else:
        avg_axis_error = total_axis_error/(correctly_assigned_rotation_part_count + correctly_assigned_translation_part_count)
    
    if correctly_assigned_rotation_part_count == 0:
        avg_center_error = 0
    else:
        avg_center_error = total_center_error/correctly_assigned_rotation_part_count

    if correctly_assigned_rotation_part_count + correctly_assigned_translation_part_count == 0:
        avg_range_iou = 0
    else:
        avg_range_iou = total_range_iou/(correctly_assigned_rotation_part_count + correctly_assigned_translation_part_count)

    append_items_to_file(['avg_type_accuracy: ', str(avg_type_accuracy)], os.path.join(folder, accuracy_filename))
    append_items_to_file(['avg_axis_error: ', str(avg_axis_error)], os.path.join(folder, accuracy_filename))
    append_items_to_file(['avg_center_error: ', str(avg_center_error*100)], os.path.join(folder, accuracy_filename))
    append_items_to_file(['avg_range_iou: ', str(avg_range_iou)], os.path.join(folder, accuracy_filename))

    append_items_to_file(['all_part_count: ', str(all_part_count)], os.path.join(folder, accuracy_filename))
    append_items_to_file(['correctly_assigned_static_part_count: ', str(correctly_assigned_static_part_count)], os.path.join(folder, accuracy_filename))
    append_items_to_file(['correctly_assigned_rotation_part_count: ', str(correctly_assigned_rotation_part_count)], os.path.join(folder, accuracy_filename))
    append_items_to_file(['correctly_assigned_translation_part_count: ', str(correctly_assigned_translation_part_count)], os.path.join(folder, accuracy_filename))

    append_items_to_file(['rotation_as_translation: ', str(rotation_as_translation)], os.path.join(folder, accuracy_filename))
    append_items_to_file(['rotation_as_static: ', str(rotation_as_static)], os.path.join(folder, accuracy_filename))
    append_items_to_file(['translation_as_rotation: ', str(translation_as_rotation)], os.path.join(folder, accuracy_filename))
    append_items_to_file(['translation_as_static: ', str(translation_as_static)], os.path.join(folder, accuracy_filename))
    append_items_to_file(['static_as_rotation: ', str(static_as_rotation)], os.path.join(folder, accuracy_filename))
    append_items_to_file(['static_as_translation: ', str(static_as_translation)], os.path.join(folder, accuracy_filename))
    
    append_items_to_file(['                    '], os.path.join(folder, accuracy_filename))
    for summary_info in summary_infos:
        append_items_to_file(summary_info, os.path.join(folder, accuracy_filename))
        append_items_to_file(['                    '], os.path.join(folder, accuracy_filename))
