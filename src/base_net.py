
import argparse
import os
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import torch

import argparse
import os
import numpy as np
import math
from partnet import *
from util_mesh import *
from config import * 
import joblib
from accuracy import *

class MotionNet(nn.Module):
    def __init__(self):
        super(MotionNet, self).__init__()

        self.bn_conv1 = nn.Sequential(nn.Conv1d(4, 64, 1),
                            nn.BatchNorm1d(64),
                            nn.LeakyReLU(negative_slope=0.2))

        self.bn_conv2 = nn.Sequential(nn.Conv1d(64, 128, 1),
                            nn.BatchNorm1d(128),
                            nn.LeakyReLU(negative_slope=0.2))
        
        self.bn_conv3 = nn.Sequential(nn.Conv1d(128, 512, 1),
                            nn.BatchNorm1d(512),
                            nn.LeakyReLU(negative_slope=0.2)
                            )
        
        self.fc1 = nn.Sequential(nn.Linear(512, 256),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.fc_enc = nn.Sequential(nn.Linear(256, 256)
                                   )

        self.fc_type = nn.Sequential(nn.Linear(256, 3))

        self.fc_axis = nn.Sequential(nn.Linear(256, 3))

        self.fc_center = nn.Sequential(nn.Linear(256, 3))

        self.fc_range = nn.Sequential(nn.Linear(256, 2))

    def forward(self, x):

        x = x.transpose(1, 2)

        x = self.bn_conv1(x)
        x = self.bn_conv2(x)
        x = self.bn_conv3(x)

        x = torch.max(x, 2, keepdim=True)[0]
        x = torch.flatten(x, 1, -1)

        x = self.fc1(x)
        enc = self.fc_enc(x)

        pred_type = self.fc_type(enc)
        pred_axis = self.fc_axis(enc)
        pred_center = self.fc_center(enc)
        pred_range = self.fc_range(enc)

        return pred_type, pred_axis, pred_center, pred_range

def get_data(shapes):

    input_pcs = []
    gt_motion_parameters = []
    for i in range(len(shapes)):
        print('processing shape:  ', i)
        part_meshes = [part.mesh for part in shapes[i].parts]
        shape_mesh, face_labels = merge_meshes(part_meshes)
        shape_pc, pc_normal, face_indices = sample_mesh(shape_mesh, shape_point_num)
        shape_pc = torch.tensor(shape_pc, dtype=torch.float, device='cuda:0')

        for j in range(len(shapes[i].parts)):
            point_labels = []
            for p_i in range(len(shape_pc)):
                point_part_index = face_labels[face_indices[p_i]]
                if point_part_index == j:
                    point_labels.append(torch.tensor(1, dtype=torch.float, device='cuda:0'))
                else:
                    point_labels.append(torch.tensor(0, dtype=torch.float, device='cuda:0'))
            point_labels = torch.stack(point_labels).unsqueeze(dim=1)

            input_pc = torch.cat((shape_pc, point_labels), dim=1)
            input_pcs.append(input_pc)

            if shapes[i].parts[j].gt_motion_type == 'static':
                gt_motion_type = torch.tensor([0], dtype=torch.long, device='cuda:0')
                gt_motion_axis = None
                gt_motion_center = None
                gt_motion_min_range = None
                gt_motion_max_range = None
                gt_motion_parameter = (gt_motion_type, None, None, None)
            else:
                if shapes[i].parts[j].gt_motion_type == 'rotation':
                    gt_motion_type = torch.tensor([1], dtype=torch.long, device='cuda:0')
                else:
                    gt_motion_type = torch.tensor([2], dtype=torch.long, device='cuda:0')

                upward_axis = torch.tensor([0, 1, 0], dtype=torch.float, device='cuda:0')
                gt_motion_axis = torch.tensor(shapes[i].parts[j].gt_axis, dtype=torch.float, device='cuda:0')
                gt_motion_axis = gt_motion_axis/torch.norm(gt_motion_axis)
                gt_motion_center = torch.tensor(shapes[i].parts[j].gt_center, dtype=torch.float, device='cuda:0')
                gt_motion_min_range = torch.tensor([shapes[i].parts[j].gt_min_range], dtype=torch.float, device='cuda:0')
                gt_motion_max_range = torch.tensor([shapes[i].parts[j].gt_max_range], dtype=torch.float, device='cuda:0')
                ref_axis = torch.tensor([1, 1, 1], dtype=torch.float, device='cuda:0')
                ref_axis = ref_axis/torch.norm(ref_axis)

                if torch.dot(gt_motion_axis, ref_axis) < 0:
                    gt_motion_axis = -gt_motion_axis
                    temp_gt_motion_min_range = -gt_motion_min_range
                    temp_gt_motion_max_range = -gt_motion_max_range
                
                    gt_motion_min_range = min(temp_gt_motion_min_range, temp_gt_motion_max_range)
                    gt_motion_max_range = max(temp_gt_motion_min_range, temp_gt_motion_max_range)

                gt_motion_parameter = (gt_motion_type, gt_motion_axis, gt_motion_center, torch.cat((gt_motion_min_range, gt_motion_max_range)))
            gt_motion_parameters.append(gt_motion_parameter)

    input_pcs = torch.stack(input_pcs)
    return input_pcs,  gt_motion_parameters    

def test_base_net(shapes, train_folder, test_folder):
    for model_file in list(os.listdir(train_folder)):
        print('model_file', model_file)
        model_epoch = model_file.split('motion')[0]
        if int(model_epoch) != 150:
            continue
        motion_net = joblib.load(os.path.join(train_folder, model_file))
        #motion_net.cuda()
        motion_net.eval()

        test_epoch_folder = os.path.join(test_folder, model_epoch)
        if not os.path.exists(test_epoch_folder):
            os.makedirs(test_epoch_folder)

        shape_annotations = []
        for i in range(len(shapes)):
            shape_annotation = []

            part_meshes = [part.mesh for part in shapes[i].parts]
            shape_mesh, face_labels = merge_meshes(part_meshes)
            shape_pc, pc_normal, face_indices = sample_mesh(shape_mesh, shape_point_num)
            shape_pc = torch.tensor(shape_pc, dtype=torch.float, device='cuda:0')

            for j in range(len(shapes[i].parts)):
                point_labels = []
                for p_i in range(len(shape_pc)):
                    point_part_index = face_labels[face_indices[p_i]]
                    if point_part_index == j:
                        point_labels.append(torch.tensor(1, dtype=torch.float, device='cuda:0'))
                    else:
                        point_labels.append(torch.tensor(0, dtype=torch.float, device='cuda:0'))
                point_labels = torch.stack(point_labels).unsqueeze(dim=1)

                input_pc = torch.cat((shape_pc, point_labels), dim=1)
                pred_type, pred_axis, pred_center, pred_range = motion_net(torch.stack([input_pc]))

                pred_type = to_numpy(pred_type[0])
                pred_axis = to_numpy(pred_axis[0])
                pred_center = to_numpy(pred_center[0])
                pred_range = to_numpy(pred_range[0])

                motion_type_index = np.argmax(softmax(pred_type))
                if motion_type_index == 0:
                    pred_motion_type = 'static'
                elif motion_type_index == 1:
                    pred_motion_type = 'rotation'
                else:
                    pred_motion_type = 'translation' 
                pred_motion_dir = pred_axis/np.linalg.norm(pred_axis)
                pred_motion_center = pred_center
                pred_motion_min_range = min(0, pred_range[0])
                pred_motion_max_range = max(0, pred_range[1])

                part_annotation = [(shapes[i].id, shapes[i].parts[j].id), pred_motion_type, pred_motion_dir, pred_motion_center, pred_motion_min_range, pred_motion_max_range, None, None, [], []]

                shape_annotation.append(part_annotation)

            shape_annotations.append(shape_annotation)    
        
        joblib.dump(shape_annotations, os.path.join(test_epoch_folder, 'shape_annotations.joblib'))
        compute_accuracy_func(shapes, shape_annotations, True, test_epoch_folder)

CE = torch.nn.CrossEntropyLoss()
MSE = torch.nn.MSELoss()

def train_base_net(shapes, folder):

    motion_net = MotionNet()
    motion_net.cuda()
    motion_net.train()

    optimizer = torch.optim.Adam(list(motion_net.parameters()), lr=0.0001)

    input_pcs, gt_motion_parameters = get_data(shapes)

    max_epoch = 201
    for epoch in range(max_epoch):
        epoch_loss = 0
        batch_size = 64
        for batch_start in range(0, len(input_pcs), batch_size):
            batched_pred_type, batched_pred_axis, batched_pred_center, batched_pred_range = motion_net(input_pcs[batch_start:batch_start+batch_size])
            batched_gt_motion_parameters = gt_motion_parameters[batch_start:batch_start+batch_size]
            
            filtered_batched_gt_type = []
            filtered_batched_gt_axis = []
            filtered_batched_gt_center = []
            filtered_batched_gt_range = []
            filtered_batched_pred_type = []
            filtered_batched_pred_axis = []
            filtered_batched_pred_center = []
            filtered_batched_pred_range = []
            for i in range(len(batched_gt_motion_parameters)):
                filtered_batched_gt_type.append(batched_gt_motion_parameters[i][0])
                filtered_batched_pred_type.append(batched_pred_type[i])
                if batched_gt_motion_parameters[i][0] != 0:
                    filtered_batched_gt_axis.append(batched_gt_motion_parameters[i][1])
                    filtered_batched_gt_center.append(batched_gt_motion_parameters[i][2])
                    filtered_batched_gt_range.append(batched_gt_motion_parameters[i][3])
                    filtered_batched_pred_axis.append(batched_pred_axis[i])
                    filtered_batched_pred_center.append(batched_pred_center[i])
                    filtered_batched_pred_range.append(batched_pred_range[i])

            filtered_batched_gt_type = torch.stack(filtered_batched_gt_type).squeeze(dim=1)
            filtered_batched_gt_axis = torch.stack(filtered_batched_gt_axis)
            filtered_batched_gt_center = torch.stack(filtered_batched_gt_center)
            filtered_batched_gt_range = torch.stack(filtered_batched_gt_range)

            filtered_batched_pred_type = torch.stack(filtered_batched_pred_type)
            filtered_batched_pred_axis = torch.stack(filtered_batched_pred_axis)
            filtered_batched_pred_center = torch.stack(filtered_batched_pred_center)
            filtered_batched_pred_range = torch.stack(filtered_batched_pred_range)

            type_loss = CE(filtered_batched_pred_type, filtered_batched_gt_type)
            axis_loss = MSE(filtered_batched_pred_axis, filtered_batched_gt_axis)
            center_loss = MSE(filtered_batched_pred_center, filtered_batched_gt_center)
            range_loss = MSE(filtered_batched_pred_range, filtered_batched_gt_range)

            print('type_loss', type_loss)
            print('axis loss', axis_loss)
            print('center_loss', center_loss)
            print('range_loss', range_loss)
            loss = 5 * type_loss + 100 * axis_loss + 20 * center_loss + 0.001*range_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print('epoch_loss', epoch_loss) 

        if epoch % 50 == 0:
            joblib.dump(motion_net, os.path.join(folder, str(epoch)+"motion_net.joblib"))

if __name__ == "__main__":

    option = args.basenet_option

    if option == 'train':
        for category in categories:
            train_folder = os.path.join(args.basenet_folder, category, 'train')
            if not os.path.exists(train_folder):
                os.makedirs(train_folder)
            shapes = get_base_net_shapes(category, use_train_file=True, use_test_file=False)
            train_base_net(shapes, train_folder)
    elif option == 'test':
        for category in categories:
            train_folder = os.path.join(args.basenet_folder, category, 'train')
            test_folder = os.path.join(args.basenet_folder, category, 'test')
            if not os.path.exists(test_folder):
                os.makedirs(test_folder)
            shapes = get_base_net_shapes(category, use_train_file=False, use_test_file=True)
            test_base_net(shapes, train_folder, test_folder)
    else:
        print('invalid option')

    



