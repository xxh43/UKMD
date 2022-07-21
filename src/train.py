
import argparse
import os
import numpy as np
import math
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
from annotate import *

print(torch.__version__)


def pre_train_func(joints, interested_joint_indices, folder):
    
    encoder = JointEncoder()
    encoder.cuda()
    encoder_alpha = torch.ones(1, dtype=torch.float, device="cuda:0", requires_grad=True)

    iteration_folder = os.path.join(folder, 'pre')
    if not os.path.exists(iteration_folder):
        os.makedirs(iteration_folder)

    pairwise_distances = pre_annotate_joints(joints, interested_joint_indices, encoder, encoder_alpha)
    joblib.dump(encoder, os.path.join(iteration_folder, "rotation_encoder.joblib"))
    joblib.dump(encoder, os.path.join(iteration_folder, "translation_encoder.joblib"))

    clusters = cluster_joints_with_simmat(joints, interested_joint_indices, pairwise_distances)
    joblib.dump(pairwise_distances, os.path.join(folder, "simmat.joblib"))
    joblib.dump(clusters, os.path.join(folder, "clusters.joblib"))


def train_func(joints, joint_clusters, motion_types, folder):

    rotation_encoder = JointEncoder()
    #rotation_encoder = joblib.load(os.path.join(pre_encoder_folder, "rotation_encoder.joblib"))
    rotation_encoder.cuda()
    rotation_encoder_alpha = torch.ones(1, device="cuda:0", requires_grad=True)
    #rotation_enc_vars = torch.full((len(joints), enc_dim), 1.0, device="cuda:0", requires_grad=True)

    translation_encoder = JointEncoder()
    #translation_encoder = joblib.load(os.path.join(pre_encoder_folder, "translation_encoder.joblib"))
    translation_encoder.cuda()
    translation_encoder_alpha = torch.ones(1, device="cuda:0", requires_grad=True)
    #translation_enc_vars = torch.full((len(joints), enc_dim), 1.0, device="cuda:0", requires_grad=True)

    # train ---------------------------------------------------------------------------------
    max_iteration = annotate_max_iteration
    for iteration in range(0, max_iteration):
        print('iteration', iteration)
        iteration_folder = os.path.join(folder, str(iteration))
        if not os.path.exists(iteration_folder):
            os.makedirs(iteration_folder)
        
        sample_option = 'sample'
        if iteration >= max_iteration-2:
            sample_option = 'greedy'

        for motion_type in motion_types:
            if motion_type == 'rotation':
                if iteration == 0:
                    rotation_annotations, summary = first_annotate_joints(joints, joint_clusters, motion_type)
                else:
                    rotation_annotations, summary = annotate_joints(joints, joint_clusters, motion_type, rotation_encoder, rotation_encoder_alpha, sample_option)
                joblib.dump(rotation_encoder, os.path.join(iteration_folder, "rotation_encoder.joblib"))
                joblib.dump(rotation_annotations, os.path.join(iteration_folder, 'rotation_annotations.joblib'))
                validate_errors, validate_ranges = validate_annotate_joints(joints, joint_clusters, rotation_annotations, motion_type, rotation_encoder, sample_option)
                joblib.dump(validate_errors, os.path.join(os.path.join(folder, str(iteration)), 'rotation_validate_range_errors.joblib'))
                joblib.dump(validate_ranges, os.path.join(os.path.join(folder, str(iteration)), 'rotation_validate_ranges.joblib'))
            else:
                if iteration == 0:
                    translation_annotations, summary = first_annotate_joints(joints, joint_clusters, motion_type)
                else:
                    translation_annotations, summary = annotate_joints(joints, joint_clusters, motion_type, translation_encoder, translation_encoder_alpha, sample_option)
                joblib.dump(translation_annotations, os.path.join(iteration_folder, 'translation_annotations.joblib'))
                joblib.dump(translation_encoder, os.path.join(iteration_folder, "translation_encoder.joblib"))
                translation_annotations = joblib.load(os.path.join(os.path.join(folder, str(iteration)), 'translation_annotations.joblib'))
                validate_errors, validate_ranges = validate_annotate_joints(joints, joint_clusters, translation_annotations, motion_type, translation_encoder, sample_option)
                joblib.dump(validate_errors, os.path.join(os.path.join(folder, str(iteration)), 'translation_validate_range_errors.joblib'))
                joblib.dump(validate_ranges, os.path.join(os.path.join(folder, str(iteration)), 'translation_validate_ranges.joblib'))