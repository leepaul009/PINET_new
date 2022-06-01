#########################################################################
##
## train agent that has some utility for training and saving.
##
#########################################################################

import torch.nn as nn
import torch
from util_hourglass import *
from copy import deepcopy
import numpy as np
from torch.autograd import Variable
from hourglass_network import lane_detection_network
from torch.autograd import Function as F
from parameters import Parameters
import math
import util
import logging
import os
from lib.utils import get_module
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
from lib import comm

############################################################
##
## agent for lane detection
##
############################################################
class Agent(nn.Module):

    def __init__(self, cfg, device, distributed):
        super(Agent, self).__init__()

        self.p = Parameters()

        self.cfg = cfg
        self.data_cfg = cfg['dataset']
        self.loss_cfg = cfg['loss_parameters']
        self.opt_cfg = cfg['optimizer']
        self.scheduler_cfg = cfg['lr_scheduler']

        self.device = device
        self.distributed = distributed

        self.lane_detection_network = lane_detection_network()

        self.setup_optimizer()

        self.current_epoch = 0
        self.starting_epoch = 0

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def setup_optimizer(self):
        self.lane_detection_optim = torch.optim.Adam(
            self.lane_detection_network.parameters(),
            lr=self.opt_cfg['l_rate'],
            weight_decay=self.opt_cfg['weight_decay'],
            )
        '''
        self.lane_detection_optim = getattr(torch.optim, 
                self.opt_cfg['name'])(
                    model_parameters,
                    **self.opt_cfg['parameters'])
        '''
        self.lr_scheduler = getattr(
            torch.optim.lr_scheduler,
            self.scheduler_cfg['name'])(
                self.lane_detection_optim, 
                **self.scheduler_cfg['parameters'])


    #####################################################
    ## Make ground truth for key point estimation
    #####################################################
    def make_ground_truth_point(self, 
                                level, # 0->64x32, 1->128x64
                                target_lanes, target_h):

        # target_lanes, target_h = util.sort_batch_along_y(target_lanes, target_h)
        ratio  = self.data_cfg['resize_ratio'] / (level+1)
        grid_y = self.data_cfg['grid_y'] * (level+1)
        grid_x = self.data_cfg['grid_x'] * (level+1)

        ground        = np.zeros((target_lanes.size(0), 3, grid_y, grid_x))
        ground_binary = np.zeros((target_lanes.size(0), 1, grid_y, grid_x))

        for batch_index, batch in enumerate(target_lanes):
            for lane_index, lane in enumerate(batch):
                for point_index, point in enumerate(lane):
                    if point > 0:
                        # determine grid index
                        x = point.item()
                        y = target_h[batch_index][lane_index][point_index].item()
                        x_index = int(x / ratio)
                        y_index = int(y / ratio)
                        ground[batch_index][0][y_index][x_index] = 1.0
                        ground[batch_index][1][y_index][x_index] = (x * 1.0 / ratio) - x_index
                        ground[batch_index][2][y_index][x_index] = (y * 1.0 / ratio) - y_index
                        ground_binary[batch_index][0][y_index][x_index] = 1

        return ground, ground_binary

    #####################################################
    ## Make ground truth for instance feature
    #####################################################
    def make_ground_truth_instance(self, 
                                   level,
                                   target_lanes, target_h):

        ratio  = self.data_cfg['resize_ratio'] / (level+1)
        grid_y = self.data_cfg['grid_y'] * (level+1)
        grid_x = self.data_cfg['grid_x'] * (level+1)

        ground = np.zeros((len(target_lanes), 1, grid_y*grid_x, grid_y*grid_x))

        for batch_index, batch in enumerate(target_lanes): # per img
            temp = np.zeros((1, grid_y, grid_x))
            lane_cluster = 1
            for lane_index, lane in enumerate(batch): # per lane
                previous_x_index = 0
                previous_y_index = 0
                for point_index, point in enumerate(lane): # per pt
                    if point > 0:
                        # get grid
                        x_index = int(point.item() / ratio)
                        y_index = int(target_h[batch_index][lane_index][point_index].item() / ratio)
                        temp[0][y_index][x_index] = lane_cluster
                    # for this grid, search for connected grids
                    if previous_x_index != 0 or previous_y_index != 0: #interpolation make more dense data
                        temp_x = previous_x_index
                        temp_y = previous_y_index
                        while True:
                            delta_x = 0
                            delta_y = 0
                            temp[0][temp_y][temp_x] = lane_cluster # repeated work

                            if temp_x < x_index: # prev is left
                                temp[0][temp_y][temp_x+1] = lane_cluster
                                delta_x = 1
                            elif temp_x > x_index: # prev is right
                                temp[0][temp_y][temp_x-1] = lane_cluster
                                delta_x = -1

                            if temp_y < y_index: # prev is top
                                temp[0][temp_y+1][temp_x] = lane_cluster
                                delta_y = 1
                            elif temp_y > y_index: # prev is bottom
                                temp[0][temp_y-1][temp_x] = lane_cluster
                                delta_y = -1

                            temp_x += delta_x
                            temp_y += delta_y
                            if temp_x == x_index and temp_y == y_index:
                                break
                    if point > 0:
                        previous_x_index = x_index
                        previous_y_index = y_index
                # lane indicator
                lane_cluster += 1

            # for per pt in img
            for i in range(grid_y*grid_x): #make gt 
                temp = temp[temp>-1]
                gt_one = deepcopy(temp)
                if temp[i]>0:
                    gt_one[temp==temp[i]] = 1   #same instance, same lane
                    '''
                    if temp[i] == 0: # skip
                        gt_one[temp!=temp[i]] = 3 #different instance, different class
                    else:
                        gt_one[temp!=temp[i]] = 2 #different instance, diff lane, same class
                        gt_one[temp==0] = 3 #different instance, different class
                    '''
                    gt_one[temp!=temp[i]] = 2 #different instance, diff lane, same class
                    gt_one[temp==0] = 3 #different instance, different class
                    ground[batch_index][0][i] += gt_one # give relationship(pt_i, other_pts) to point i

        return ground

    #####################################################
    ## train
    #####################################################
    def train(self, inputs, labels, lanes, epoch):
        loss, loss_dict = self.train_point(inputs, labels, lanes, epoch)
        return loss, loss_dict

    def validation(self, inputs, labels, lanes, epoch):
        loss, loss_dict = self.train_point(inputs, labels, lanes, epoch, 
                                           is_train=False)
        return loss, loss_dict

    #####################################################
    ## compute loss function and optimize
    #####################################################
    def train_point(self, 
                    inputs, 
                    labels,
                    lanes, # target_lanes tensor ? why not np.array x [N num_lns, pts]
                    epoch,
                    is_train=True):
        real_batch_size = inputs.size(0)
        ground_truth_point, ground_binary, ground_truth_instance = labels
        ground_truth_point = ground_truth_point.to(self.device) # .cuda() # [N,3,Gy,Gx]
        ground_truth_instance = ground_truth_instance.to(self.device) # .cuda()

        result = self.predict_lanes(inputs)
        lane_detection_loss, loss_dict = self.loss(result, 
                                                   real_batch_size, 
                                                   ground_truth_point, 
                                                   ground_binary, 
                                                   ground_truth_instance)
        # Backward
        if is_train:
            self.lane_detection_optim.zero_grad()
            lane_detection_loss.backward()
            self.lane_detection_optim.step()
            # lr = self.lane_detection_optim.param_groups[best_param_group_id]["lr"]
            self.lr_scheduler.step()
            # Learning rate
            if epoch>0 and epoch%20==0 and self.current_epoch != epoch:
                self.current_epoch = epoch
                if epoch>0 and (epoch == 1000):
                    self.loss_cfg['constant_lane_loss'] += 0.5
                    self.loss_cfg['constant_nonexist'] += 0.5
                    self.opt_cfg['l_rate'] /= 2.0
                    self.setup_optimizer()

        # del ground_truth_point, ground_binary, ground_truth_instance

        loss_dict['lane_detection_loss'] = lane_detection_loss.item()
        return lane_detection_loss, loss_dict


    def loss(self, 
             result, 
             real_batch_size, 
             ground_truth_point, 
             ground_binary, 
             ground_truth_instance):
        lane_detection_loss = 0
        loss_dict = {'same_instance_loss': .0, 
                     'diff_instance_loss': .0,
                     'exist_loss': .0,
                     'nonexist_loss': .0,
                     'offset_loss': .0,
                     }
        for (confidance, offset, feature) in result:
            if self.cfg['debug']:
                logging.info('-- conf: {}, offset: {}, feat: {}.'
                    .format( confidance.size(), offset.size(), feature.size() )) 
                # [N,1,Gy,Gx], [N,2,Gy,Gx], [N,4,Gy,Gx]
                logging.info('-- gt_cf: {}, gt_offset: {}, gt_feat: {}.'
                    .format( ground_binary.size(), ground_truth_point.size(), ground_truth_instance.size() )) 

            #compute loss for point prediction
            offset_loss = 0
            exist_condidence_loss = 0
            nonexist_confidence_loss = 0

            #exist confidance loss
            confidance_gt = ground_truth_point[:, 0, :, :]
            confidance_gt = confidance_gt.view(real_batch_size, 
                                               1, 
                                               self.data_cfg['grid_y'], 
                                               self.data_cfg['grid_x'])
            exist_condidence_loss = torch.sum(
                (confidance_gt[confidance_gt==1] - confidance[confidance_gt==1])**2 
                ) / torch.sum(confidance_gt==1)

            #non exist confidance loss
            nonexist_confidence_loss = torch.sum( 
                (confidance_gt[confidance_gt==0] - confidance[confidance_gt==0])**2 
                ) / torch.sum(confidance_gt==0)

            #offset loss 
            offset_x_gt = ground_truth_point[:, 1:2, :, :]
            offset_y_gt = ground_truth_point[:, 2:3, :, :]

            predict_x = offset[:, 0:1, :, :]
            predict_y = offset[:, 1:2, :, :]

            x_offset_loss = torch.sum(
                (offset_x_gt[confidance_gt==1] - predict_x[confidance_gt==1])**2 
                ) / torch.sum(confidance_gt==1)
            y_offset_loss = torch.sum(
                (offset_y_gt[confidance_gt==1] - predict_y[confidance_gt==1])**2 
                ) / torch.sum(confidance_gt==1)

            offset_loss = (x_offset_loss + y_offset_loss)/2

            #compute loss for similarity
            sisc_loss = 0
            disc_loss = 0

            feature_map = feature.view(real_batch_size, 
                                       self.data_cfg['feature_size'], 
                                       1, 
                                       self.data_cfg['grid_y']*self.data_cfg['grid_x'])
            feature_map = feature_map.expand(real_batch_size, 
                                             self.data_cfg['feature_size'], 
                                             self.data_cfg['grid_y']*self.data_cfg['grid_x'], 
                                             self.data_cfg['grid_y']*self.data_cfg['grid_x']).detach()

            point_feature = feature.view(real_batch_size, 
                                         self.data_cfg['feature_size'], 
                                         self.data_cfg['grid_y']*self.data_cfg['grid_x'], 
                                         1)
            point_feature = point_feature.expand(real_batch_size, 
                                            self.data_cfg['feature_size'], 
                                            self.data_cfg['grid_y']*self.data_cfg['grid_x'], 
                                            self.data_cfg['grid_y']*self.data_cfg['grid_x'])#.detach()

            distance_map = (feature_map - point_feature)**2 
            distance_map = torch.norm( distance_map, dim=1 ).view(
                                                    real_batch_size, 
                                                    1, 
                                                    self.data_cfg['grid_y']*self.data_cfg['grid_x'], 
                                                    self.data_cfg['grid_y']*self.data_cfg['grid_x'])

            # same instance
            sisc_loss = torch.sum(
                distance_map[ground_truth_instance==1])/torch.sum(ground_truth_instance==1)

            # different instance, same class
            disc_loss = self.loss_cfg['K1'] - distance_map[ground_truth_instance==2] 
            # self.loss_cfg['K1']/distance_map[ground_truth_instance==2] + (self.loss_cfg['K1']-distance_map[ground_truth_instance==2])
            disc_loss[disc_loss<0] = 0
            disc_loss = torch.sum(disc_loss)/torch.sum(ground_truth_instance==2)

            lane_loss = (self.loss_cfg['constant_exist'] * exist_condidence_loss
                         + self.loss_cfg['constant_nonexist'] * nonexist_confidence_loss
                         + self.loss_cfg['constant_offset'] * offset_loss)
            
            instance_loss = (self.loss_cfg['constant_alpha'] * sisc_loss 
                            + self.loss_cfg['constant_beta'] * disc_loss)

            lane_detection_loss = (lane_detection_loss
                                   + self.loss_cfg['constant_lane_loss']*lane_loss
                                   + self.loss_cfg['constant_instance_loss']*instance_loss)

            loss_dict['same_instance_loss'] += sisc_loss.item()
            loss_dict['diff_instance_loss'] += disc_loss.item()
            loss_dict['exist_loss'] += exist_condidence_loss.item()
            loss_dict['nonexist_loss'] += nonexist_confidence_loss.item()
            loss_dict['offset_loss'] += offset_loss.item()

        # del confidance, offset, feature
        # del feature_map, point_feature, distance_map
        # del exist_condidence_loss, nonexist_confidence_loss, offset_loss, sisc_loss, disc_loss, lane_loss, instance_loss

        return lane_detection_loss, loss_dict

    #####################################################
    ## predict lanes
    #####################################################
    def predict_lanes(self, inputs): # inputs tensor
        # inputs = torch.from_numpy(inputs).float() 
        # inputs = Variable(inputs).cuda()
        inputs = inputs.to(self.device)
        return self.lane_detection_network(inputs)

    #####################################################
    ## predict lanes in test
    #####################################################
    def predict_lanes_test(self, inputs):
        # inputs = torch.from_numpy(inputs).float() 
        # inputs = Variable(inputs).cuda()
        inputs = inputs.to(self.device)
        return self.lane_detection_network(inputs)

    #####################################################
    ## Training/evaluate mode
    #####################################################                                                
    def training_mode(self):
        self.lane_detection_network.train()
                                            
    def evaluate_mode(self):
        self.lane_detection_network.eval()

    #####################################################
    ## Setup GPU computation
    #####################################################                                                

    def to_cuda(self):
        self.lane_detection_network.to(self.device)

    # before to device, after creating model
    def convert_sync_batchnorm(self):
        self.lane_detection_network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            self.lane_detection_network)
    
    def buildDistributedModel(self, local_rank):
        self.lane_detection_network = DistributedDataParallel(
            self.lane_detection_network, 
            find_unused_parameters=True,
            device_ids=[local_rank], 
            output_device=local_rank)

    #####################################################
    ## Load/save file
    #####################################################
    # def load_weights(self, epoch, loss):
    #    self.lane_detection_network.load_state_dict(
    #        torch.load(self.p.model_path+str(epoch)+'_'+str(loss)+'_'+'lane_detection_network.pkl'),False
    #    )
    
    def resume(self, exp_root):
        models_dir = os.path.join(exp_root, "models")
        models = os.listdir(models_dir)
        last_epoch, last_modelname = sorted(
            [( int(name.split("_")[0]),name) for name in models], key=lambda x:x[0]
            )[-1]
        model_path = os.path.join(models_dir, last_modelname)

        checkpoint = torch.load(model_path,
                                map_location=torch.device("cpu"))
        
        get_module(self.lane_detection_network, self.distributed).load_state_dict(checkpoint['model'])
        self.lane_detection_optim.load_state_dict(checkpoint['optimizer'])
        if checkpoint['lr_scheduler']:
           self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.starting_epoch = checkpoint['epoch'] + 1
        logging.info('resume model at epoch {} from {}'.format(checkpoint['epoch'], model_path))

    def save_model(self, exp_root, epoch, loss):
        path = os.path.join(exp_root, "models", "{:03d}_model.pkl".format(int(epoch)))
        state = {
                'model': get_module(self.lane_detection_network, self.distributed).state_dict(),
                'optimizer': self.lane_detection_optim.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'epoch': epoch,
                'loss': loss,
                }
        torch.save(state, path)
        logging.info('save model as {}'.format(path))
