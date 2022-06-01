#########################################################################
##
##  Data loader source code for TuSimple dataset
##
#########################################################################


import math
import numpy as np
import cv2
import json
import random
import logging
import torch
import os, glob
from copy import deepcopy
from parameters import Parameters
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
from util import sort_batch_along_y

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

#########################################################################
## some iamge transform utils
#########################################################################
def Translate_Points(point,translation): 
    point = point + translation 
    
    return point

def Rotate_Points(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def BuildDataLoader(dataset, 
                    is_train, 
                    distributed, 
                    batch_size=1, 
                    num_workers=1,
                    is_shuffle=True,
                    ):
    sampler = None
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset, 
                    shuffle=is_shuffle,
                    )
    else:
        if is_shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = None # torch.utils.data.Sampler(dataset) # TBD ?????

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        sampler=sampler,
        )
    return loader

class LaneDataset(Dataset):
    def __init__(self, 
                 cfg, 
                 split, 
                 transform=None,
                 ):
        super(LaneDataset, self).__init__()
        self.p = Parameters()
        self.cfg = cfg['dataset']
        self.dataset = None
        self.images_path = None
        self.to_tensor = ToTensor()
        self.grid_x = int(self.cfg['x_size'] / self.cfg['resize_ratio'])
        self.grid_y = int(self.cfg['y_size'] / self.cfg['resize_ratio'])

        if split == 'train':
            with open(self.cfg['train_dataset']) as f:
                self.dataset = json.load(f)
                self.images_path = self.cfg['train_images_path']
        elif split == 'val':
            with open(self.cfg['val_dataset']) as f:
                self.dataset = json.load(f)
                self.images_path = self.cfg['train_images_path']
        elif split == 'test':
            logging.info('to be done, to create test dataset')
        else:
            logging.info('Wrong split as {}'.format(split))

        self.max_lanes = -1e5
        self.max_points = -1e5
        for it in self.dataset:
            self.max_lanes  = max( self.max_lanes, len(it['lanes']) )
            self.max_points = max( self.max_points, max( [ len(ln) for ln in it['lanes'] ] ) )
        logging.info('create {} dataset with length {}, max-lanes {}, max-points {}.'
            .format( split, len(self.dataset), self.max_lanes, self.max_points ))
        self.transform = transform

    def __getitem__(self, idx):
        data = self.dataset[idx]
        image = cv2.imread( self.images_path + data['img_path'] + '.bmp' )
        # print('imread img shape {}'.format(image.shape))
        if image is None:
            logging.info('ERROR! cant not read image {}'.format(idx))
        ratio_w = self.cfg['x_size']*1.0/image.shape[1]
        ratio_h = self.cfg['y_size']*1.0/image.shape[0]
        image = cv2.resize(image, (self.cfg['x_size'], self.cfg['y_size']))
        # image = np.rollaxis(image, axis=2, start=0) 
        # xs = []
        # ys = []
        xs = np.ones( (self.max_lanes, self.max_points), dtype=np.float32 ) * -2
        ys = np.ones( (self.max_lanes, self.max_points), dtype=np.float32 ) * -2
        for lane_id, lane in enumerate(data['lanes']):
            # xs.append( np.array( [x for x, y in lane] ) * ratio_w )
            # ys.append( np.array( [y for x, y in lane] ) * ratio_h )
            temp_xs = np.array( [x for x, y in lane] ) * ratio_w
            temp_ys = np.array( [y for x, y in lane] ) * ratio_h
            xs[lane_id, 0:len(temp_xs)] = temp_xs
            ys[lane_id, 0:len(temp_ys)] = temp_ys
        # xs = np.array(xs)
        # ys = np.array(ys)

        # if self.transform:
        #     image = self.transform(image)

        image, xs, ys = self.trans( image, xs, ys ) # image: hwc

        ground, ground_binary = self.make_ground_truth_point(xs, ys)
        ground_truth_instance = self.make_ground_truth_instance(xs, ys)

        ground = torch.from_numpy(ground).float()
        ground_binary = torch.LongTensor(ground_binary.tolist())
        ground_truth_instance = torch.from_numpy(ground_truth_instance).float()

        return (image, (ground, ground_binary, ground_truth_instance), (xs, ys), idx)

    def __len__(self):
        return len(self.dataset)       
    
    def trans(self, img, xs, ys):
        sample = img, xs, ys
        sample = self.Hflip(sample)
        sample = self.Translation(sample)
        sample = self.Rotate(sample)
        sample = self.Gaussian(sample)
        sample = self.Change_intensity(sample)
        sample = self.Shadow(sample)
        img, xs, ys = sample
        

        img = img / 255.
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        # img = torch.from_numpy(img).float() 
        img = self.to_tensor(img.astype(np.float32)) # hwc => chw
        # print('trans output img shape {}'.format(img.size()))
        return img, xs, ys

    def Gaussian(self, sample): # input image: hwc
        test_image, x, y = sample
        if torch.rand(1) < self.cfg['noise_ratio']:
            img = np.zeros((self.cfg['y_size'], self.cfg['x_size'], 3), np.uint8)
            m = (0,0,0) 
            s = (20,20,20)
            cv2.randn(img, m, s)
            test_image = test_image + img
        return test_image, x, y

    def Change_intensity(self, sample):
        test_image, x, y = sample
        if torch.rand(1) < self.cfg['intensity_ratio']:
            hsv = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            value = int(random.uniform(-60.0, 60.0))
            if value > 0:
                lim = 255 - value
                v[v > lim] = 255
                v[v <= lim] += value
            else:
                lim = -1*value
                v[v < lim] = 0
                v[v >= lim] -= lim                
            final_hsv = cv2.merge((h, s, v))
            test_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return test_image, x, y


    ## Generate random shadow in random region
    def Shadow(self, sample, min_alpha=0.5, max_alpha = 0.75):
        if torch.rand(1) < self.cfg['shadow_ratio']:
            if np.random.randint(2) == 0:
                image, x, y = sample
                test_image = deepcopy(image)

                top_x, bottom_x = np.random.randint(0, self.cfg['x_size'], 2)
                rows, cols, _ = test_image.shape
                shadow_img = test_image.copy()

                rand = np.random.randint(2)
                vertices = np.array([[(50, 65), (45, 0), (145, 0), (150, 65)]], dtype=np.int32)
                if rand == 0:
                    vertices = np.array([[top_x, 0], [0, 0], [0, rows], [bottom_x, rows]], dtype=np.int32)
                elif rand == 1:
                    vertices = np.array([[top_x, 0], [cols, 0], [cols, rows], [bottom_x, rows]], dtype=np.int32)
                mask = test_image.copy()
                channel_count = test_image.shape[2]  # i.e. 3 or 4 depending on your image
                ignore_mask_color = (0,) * channel_count
                cv2.fillPoly(mask, [vertices], ignore_mask_color)
                rand_alpha = np.random.uniform(min_alpha, max_alpha)
                cv2.addWeighted(mask, rand_alpha, test_image, 1 - rand_alpha, 0., shadow_img)

                return shadow_img, x, y
        return sample

    def Hflip(self, sample):
        if torch.rand(1) < self.cfg['flip_ratio']:
            image, x, y = sample
            image = cv2.flip(image, 1)
            # for i in range(len(x)): # for each lane
            #    x[i][x[i]>0] = self.cfg['x_size'] - x[i][x[i]>0] # resize's size
            #    x[i][x[i]<0] = -2
            #    x[i][x[i]>=self.cfg['x_size']] = -2
            for ln in x: # for each ref lane
                ln[ln > 0] = self.cfg['x_size'] - ln[ln > 0]
                ln[ln < 0] = -2
                ln[ln >= self.cfg['x_size']] = -2
            return image, x, y
        return sample

    def Translation(self, sample):
        img, x, y = sample

        tx_param = int(self.cfg['x_size'] / 10.0)
        ty_param = int(self.cfg['y_size'] / 10.0)

        tx = np.random.randint(-tx_param, tx_param)
        ty = np.random.randint(-ty_param, ty_param)

        img = cv2.warpAffine(img, np.float32([[1,0,tx], [0,1,ty]]), (self.cfg['x_size'], self.cfg['y_size']))

        for j in range(len(x)):
            x[j][x[j]>0]  = x[j][x[j]>0] + tx
            x[j][x[j]<0] = -2
            x[j][x[j]>=self.cfg['x_size']] = -2
        for j in range(len(y)):
            y[j][y[j]>0]  = y[j][y[j]>0] + ty
            # set -2 to the x axis of invalid point
            x[j][y[j]<0] = -2
            x[j][y[j]>=self.cfg['y_size']] = -2
            y[j][y[j]<0] = -2
            y[j][y[j]>=self.cfg['y_size']] = -2

        return img, x, y


    def Rotate(self, sample):
        temp_image, x, y = sample

        angle = np.random.randint(-10, 10)
        M = cv2.getRotationMatrix2D((self.cfg['x_size']/2,self.cfg['y_size']/2),angle,1)
        temp_image = cv2.warpAffine(temp_image, M, (self.cfg['x_size'], self.cfg['y_size']))

        for j in range(len(x)):
            index_mask = deepcopy(x[j]>0)
            x[j][index_mask], y[j][index_mask] = Rotate_Points(
                (self.cfg['x_size']/2, self.cfg['y_size']/2),
                (x[j][index_mask], y[j][index_mask]),
                (-angle * 2 * np.pi) / 360)
            x[j][x[j]<0] = -2
            x[j][x[j]>=self.cfg['x_size']] = -2
            x[j][y[j]<0] = -2
            x[j][y[j]>=self.cfg['y_size']] = -2
            y[j][y[j]<0] = -2
            y[j][y[j]>=self.cfg['y_size']] = -2

        return temp_image, x, y

    def make_ground_truth_point(self, target_lanes, target_h):

        # target_lanes, target_h = sort_batch_along_y(target_lanes, target_h)

        ground        = np.zeros((3, self.grid_y, self.grid_x))
        ground_binary = np.zeros((1, self.grid_y, self.grid_x))

        # for batch_index, batch in enumerate(target_lanes):
        #   for lane_index, lane in enumerate(batch):
        for lane_index, lane in enumerate(target_lanes): # for each lane
            for point_index, point in enumerate(lane): # for each point(x)
                if point > 0: # ignore negative point(x)
                    # determine grid index
                    x_index = int(point / self.cfg['resize_ratio']) # rr=8
                    y_index = int(target_h[lane_index][point_index] / self.cfg['resize_ratio'])
                    ground[0][y_index][x_index] = 1.0
                    ground[1][y_index][x_index]= (point*1.0 / self.cfg['resize_ratio']) - x_index
                    ground[2][y_index][x_index] = (target_h[lane_index][point_index]*1.0 / self.cfg['resize_ratio']) - y_index
                    ground_binary[0][y_index][x_index] = 1

        return ground, ground_binary

    def make_ground_truth_instance(self, target_lanes, target_h):

        # ground = np.zeros((len(target_lanes), 1, self.p.grid_y*self.p.grid_x, self.p.grid_y*self.p.grid_x))
        ground = np.zeros((1, self.grid_y*self.grid_x, self.grid_y*self.grid_x))

        # for batch_index, batch in enumerate(target_lanes): # per img
        temp = np.zeros((1, self.grid_y, self.grid_x))
        lane_cluster = 1
        for lane_index, lane in enumerate(target_lanes): # per lane
            previous_x_index = 0
            previous_y_index = 0
            for point_index, point in enumerate(lane): # per pt
                if point > 0:
                    # get grid
                    x_index = int(point / self.cfg['resize_ratio'])
                    y_index = int(target_h[lane_index][point_index] / self.cfg['resize_ratio'])
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
        for i in range(self.grid_y*self.grid_x): #make gt 
            temp = temp[temp>-1]
            gt_one = deepcopy(temp)
            if temp[i]>0:
                gt_one[temp==temp[i]] = 1   #same instance, same lane
                gt_one[temp!=temp[i]] = 2 #different instance, diff lane, same class
                gt_one[temp==0] = 3 #different instance, different class
                ground[0][i] += gt_one # give relationship(pt_i, other_pts) to point i

        return ground


class InferenceDataset(Dataset):
    def __init__(self, cfg):
        super(InferenceDataset, self).__init__()
        self.cfg = cfg['dataset']
        self.root = self.cfg['test_images_path']
        self.to_tensor = ToTensor()

        pattern = os.path.join(self.root, '*.bmp')
        self.img_paths = glob.glob(pattern)
        logging.info('Create inference dataset with len {}'
                     .format( len(self.img_paths) ))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        image = cv2.imread(image_path)
        
        assert type(image) == np.ndarray, 'Error! failed to read image from {}'.format(image_path)

        ratio_w = self.cfg['x_size']*1.0 / image.shape[1]
        ratio_h = self.cfg['y_size']*1.0 / image.shape[0]
        image = cv2.resize(image, (self.cfg['x_size'], self.cfg['y_size']))

        image = image / 255.
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = self.to_tensor(image.astype(np.float32)) # => chw

        return (image, image_path, idx)