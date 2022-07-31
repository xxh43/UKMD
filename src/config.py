


import torch.nn as nn
import torch.nn.functional as F
import torch

import json
import os
cur_dir = os.path.dirname(__file__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def to_numpy(item):
    if torch.is_tensor(item):
        if item.is_cuda:
            return item.cpu().detach().numpy()
        else:
            return item.detach().numpy()
    else:
        return item

def to_tensor(item):
    return torch.tensor(item, device="cuda:0", dtype=torch.float)


categories = [
'Box', 'Bucket', 'Clock', 'Door', 'Fan', 'Faucet', 'FoldingChair', \
'Knife', 'Laptop', 'Pliers', 'Refrigerator', 'Scissors', 'TrashCan', 'USB', \
'Table', 'Window', 'StorageFurniture', 'Stapler'
]

part_point_num = 1024
shape_point_num = 2048

motion_types = ['rotation', 'translation']

import argparse
parser = argparse.ArgumentParser()

debug = False
if debug:
    parser.add_argument("--pre_annotate_transformation_max_epoch", type=int, default=2)
    parser.add_argument("--global_pre_align_max_iteration", type=int, default=2)
    parser.add_argument("--rotation_training_max_epoch", type=int, default=2)
    parser.add_argument("--translation_training_max_epoch", type=int, default=2)
    parser.add_argument("--annotate_max_iteration", type=int, default=2)
    parser.add_argument("--encoding_training_max_epoch", type=int, default=2) 
else:
    parser.add_argument("--pre_annotate_transformation_max_epoch", type=int, default=700) #1000
    parser.add_argument("--global_pre_align_max_iteration", type=int, default=700) #1000
    parser.add_argument("--rotation_training_max_epoch", type=int, default=800) #1000
    parser.add_argument("--translation_training_max_epoch", type=int, default=1501) #1000
    parser.add_argument("--annotate_max_iteration", type=int, default=6) #6
    parser.add_argument("--encoding_training_max_epoch", type=int, default=150) #100

parser.add_argument("--use_primary", type=int, default=1)
parser.add_argument("--use_secondary", type=int, default=1)
parser.add_argument("--use_deform", type=int, default=1)
parser.add_argument("--use_physics", type=int, default=1)

parser.add_argument("--option", type=str, default=None)
parser.add_argument("--category", type=str, default=None)

parser.add_argument("--add_motion_level", type=int, default=5) #5
parser.add_argument("--max_shape_num", type=int, default=50) #50
parser.add_argument("--enc_dim", type=int, default=48) #48

parser.add_argument("--secondary_translation_clamp", type=float, default=0.01) #0.15
parser.add_argument("--encoding_training_learning_rate", type=float, default=0.04) #0.04
parser.add_argument("--pre_annotate_group_count", type=int, default=64) #32
parser.add_argument("--pre_annotate_max_epoch", type=int, default=1) #1
parser.add_argument("--pre_annotate_optim_learning_rate", type=int, default=0.005) #0.005

parser.add_argument("--joint_cluster_size", type=int, default=16) #32
parser.add_argument("--bayes_iterations", type=int, default=50) #50

parser.add_argument("--annotate_group_count", type=int, default=512) #512
parser.add_argument("--annotate_group_size", type=int, default=6) #10
parser.add_argument("--special_annotate_group_count", type=int, default=32) #48
parser.add_argument("--special_annotate_group_size", type=int, default=24) #24

parser.add_argument("--rotation_primary_range_encouragement_weight", type=float, default=0.001)
parser.add_argument("--rotation_collision_penalty_weight", type=float, default=0.01)
parser.add_argument("--rotation_mov_base_detach_penalty_weight", type=float, default=0.01)
parser.add_argument("--rotation_mov_center_detach_penalty_weight", type=float, default=5.0)
parser.add_argument("--rotation_secondary_range_penalty_weight", type=float, default=0.0025)
parser.add_argument("--rotation_box_deform_penalty_weight", type=float, default=0.001)

parser.add_argument("--translation_primary_range_encouragement_weight", type=float, default=0.001)
parser.add_argument("--translation_collision_penalty_weight", type=float, default=0.5)
parser.add_argument("--translation_mov_base_detach_penalty_weight", type=float, default=0.5)
parser.add_argument("--translation_secondary_range_penalty_weight", type=float, default=0.005)
parser.add_argument("--translation_box_deform_penalty_weight", type=float, default=0.001)

parser.add_argument("--result_folder", type=str, default='')
parser.add_argument("--processed_partnet_folder", type=str, default='processed_partnet_shapes')
parser.add_argument("--basenet_folder", type=str, default='exp_basenet')
parser.add_argument("--recon_error_amplifier", type=float, default=1000)

parser.add_argument("--basenet_option", type=str, default=None)
parser.add_argument("--basenet_train_category", type=str, default=None)
parser.add_argument("--basenet_test_category", type=str, default=None)

parser.add_argument("--shape2motion_option", type=int, default=1)
parser.add_argument("--shape2motion_category", type=str, default=None)

parser.add_argument("--update_category", type=str, default=None)
parser.add_argument("--prealign_category", type=str, default=None)

parser.add_argument("--largest_as_static", type=int, default=1)

args = parser.parse_args()

# ablation------------------------

if args.use_primary == 1:
    use_primary = True
else:
    use_primary = False

if args.use_physics == 1:
    use_physics = True
else:
    use_physics = False

if args.use_deform == 1:
    use_deform = True
else:
    use_deform = False

if args.use_secondary == 1:
    use_secondary = True
else:
    use_secondary = False

if args.largest_as_static == 1:
    largest_as_static = True
else:
    largest_as_static = False

enc_dim = args.enc_dim
add_motion_level = args.add_motion_level

global_pre_align_max_iteration = args.global_pre_align_max_iteration
processed_partnet_folder = os.path.join(cur_dir, args.processed_partnet_folder)

encoding_training_max_epoch = args.encoding_training_max_epoch
encoding_training_learning_rate = args.encoding_training_learning_rate
pre_annotate_optim_learning_rate = args.pre_annotate_optim_learning_rate

pre_annotate_group_count = args.pre_annotate_group_count
pre_annotate_max_epoch = args.pre_annotate_max_epoch
pre_annotate_transformation_max_epoch = args.pre_annotate_transformation_max_epoch
joint_cluster_size = args.joint_cluster_size

annotate_group_count = args.annotate_group_count

annotate_group_size = args.annotate_group_size
special_annotate_group_count = args.special_annotate_group_count
special_annotate_group_size = args.special_annotate_group_size
annotate_max_iteration = args.annotate_max_iteration

rotation_training_max_epoch = args.rotation_training_max_epoch
translation_training_max_epoch = args.translation_training_max_epoch

rotation_primary_range_encouragement_weight = args.rotation_primary_range_encouragement_weight
rotation_secondary_range_penalty_weight = args.rotation_secondary_range_penalty_weight
rotation_collision_penalty_weight = args.rotation_collision_penalty_weight
rotation_mov_base_detach_penalty_weight = args.rotation_mov_base_detach_penalty_weight
rotation_mov_center_detach_penalty_weight = args.rotation_mov_center_detach_penalty_weight
rotation_box_deform_penalty_weight = args.rotation_box_deform_penalty_weight
#rotation_global_range_penalty_weight = args.rotation_global_range_penalty_weight
#box_deform_penalty_weight = args.box_deform_penalty_weight
#rotation_mvc_deform_penalty_weight = args.rotation_mvc_deform_penalty_weight

translation_primary_range_encouragement_weight = args.translation_primary_range_encouragement_weight
translation_secondary_range_penalty_weight = args.translation_secondary_range_penalty_weight
translation_collision_penalty_weight = args.translation_collision_penalty_weight
translation_mov_base_detach_penalty_weight = args.translation_mov_base_detach_penalty_weight
translation_box_deform_penalty_weight = args.translation_box_deform_penalty_weight
#translation_global_range_penalty_weight = args.translation_global_range_penalty_weight
#translation_box_deform_penalty_weight = args.translation_box_deform_penalty_weight
#translation_mvc_deform_penalty_weight = args.translation_mvc_deform_penalty_weight

recon_error_amplifier = args.recon_error_amplifier
result_folder = args.result_folder