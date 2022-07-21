

import sys

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
from partnet import *
from train import *
from resolve import *

import multiprocessing
from accuracy import *


def process_shapes_all():
    for category in categories:
        process_shapes(category)

def train_all():
    for category in categories:
        train(category)

def resolve_all():
    for category in categories:
        resolve(category)

def compute_accuracy_all():
    for category in categories:
        compute_accuracy(category)

def process_shapes(category):
    process_category_shapes(category)

def train(category):
    shapes, interested_shape_indices = get_shapes(processed_partnet_folder, category, use_train_file=True, use_test_file=True, interested_option='both')
    joints, interested_joint_indices = get_joints(shapes, interested_shape_indices)

    folder = os.path.join(cur_dir, 'exp', category, result_folder)
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    with open(os.path.join(folder, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if not os.path.isfile(os.path.join(folder, "clusters.joblib")):
        pre_train_func(joints, interested_joint_indices, folder)

    joint_clusters = joblib.load(os.path.join(folder, "clusters.joblib"))
    train_func(joints, joint_clusters, motion_types, folder)
    
def resolve(category):
    shapes, interested_shape_indices = get_shapes(processed_partnet_folder, category, use_train_file=True, use_test_file=True, interested_option='both')
    joints, interested_joint_indices = get_joints(shapes, interested_shape_indices)
    folder = os.path.join(cur_dir, 'exp', category, result_folder)
    resolve_func([shapes[v] for v in interested_shape_indices], [joints[v] for v in interested_joint_indices], folder)

def compute_accuracy(category):
    shapes, interested_shape_indices = get_shapes(processed_partnet_folder, category, use_train_file=True, use_test_file=True, interested_option='both')
    folder = os.path.join(cur_dir, 'exp', category, result_folder)
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    our_annotations = joblib.load(os.path.join(folder, 'shape_annotations.joblib'))
    compute_accuracy_func([shapes[v] for v in interested_shape_indices], our_annotations, True, folder)
    compute_accuracy_func([shapes[v] for v in interested_shape_indices], our_annotations, False, folder)

if __name__ == "__main__":

    if args.option == 'process':
        if args.category == 'all':
            process_shapes_all()
        else:
            process_shapes(args.category)

    if args.option == 'train':
        if args.category == 'all':
            train_all()
        else:
            train(args.category)
    
    if args.option == 'resolve':
        if args.category == 'all':
            resolve_all()
        else:
            resolve(args.category)
    
    if args.option == 'accuracy':
        if args.category == 'all':
            compute_accuracy_all()
        else:
            compute_accuracy(args.category)
