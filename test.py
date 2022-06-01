#############################################################################################################
##
##  Source code for testing
##
#############################################################################################################

import cv2
import json
import torch
import agent
import numpy as np
from copy import deepcopy
from data_loader import Generator
import time
from parameters import Parameters
import util
import argparse
import os
import logging
import random

from lib import comm
from lib.utils import setup_exp_dir, set_config, get_exp_checkpoint
from lib.dataset import BuildDataLoader, LaneDataset, InferenceDataset

import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
import pandas as pd

p = Parameters()

def parse_args():
    parser = argparse.ArgumentParser(description="Test PINet")
    parser.add_argument("--cfg", default="config.yaml", help="Config file", required=True)
    parser.add_argument("--exp_name", default="default", help="Experiment name", required=True)
    parser.add_argument("--cuda_start", type=int, default=0)
    parser.add_argument("--cuda_num", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_args()

###############################################################
## Testing
###############################################################
def Testing():
    args = parse_args()
    cfg = set_config(args.cfg)

    exp_root = os.path.join(cfg['exps_dir'], 
                            os.path.basename(os.path.normpath(args.exp_name)))
    assert os.path.exists(exp_root) == True, "unexisting path, {}".format(exp_root)
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(exp_root, "testing.txt")),
            logging.StreamHandler(),
        ],
    )

    #########################################################################
    ## Check GPU
    #########################################################################
    cuda_ids = []
    for idx in range(args.cuda_num):
        cuda_ids.append(args.cuda_start + idx)
    logging.info("visible devices: {}".format(cuda_ids))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, cuda_ids))

    distributed = args.cuda_num > 1
    device = None
    if distributed:
        device = torch.device('cuda:{}'.format(args.local_rank))
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://")
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info('rank: {}, distributed: {}, device: {}'
                 .format(comm.get_rank(), distributed, device))

    #########################################################################
    ## Get dataset TBD??????????
    #########################################################################
    test_dataset = InferenceDataset(cfg)
    test_loader  = BuildDataLoader(test_dataset, False, distributed, 
                                  **cfg['testloader_parameter'])

    ##############################
    ## Get agent and model
    ##############################
    lane_agent = agent.Agent(cfg, device, distributed)
    if distributed:
        lane_agent.convert_sync_batchnorm()
    lane_agent.to_cuda()
    if distributed:
        lane_agent.buildDistributedModel(args.local_rank)
    # load lasted model
    lane_agent.resume(exp_root)

    ##############################
    ## testing
    ##############################
    logging.info('Testing loop')
    num_steps = len(test_loader)
    lane_agent.evaluate_mode()

    # prepare report root
    model_epoch = lane_agent.starting_epoch - 1
    report_root = 'inference_out/epoch_{}'.format(model_epoch)
    report_img_root = 'inference_out/epoch_{}/img'.format(model_epoch)
    if not os.path.exists( os.path.join(exp_root, report_root) ):
        os.makedirs(os.path.join(exp_root, report_root), exist_ok=True)
    if not os.path.exists( os.path.join(exp_root, report_img_root) ):
        os.makedirs(os.path.join(exp_root, report_img_root), exist_ok=True)

    #############
    img = []
    index = []
    img_width = []
    img_height = []
    lane_prob = []
    solid_prob = []
    points = []
    solid_type = []
    global_id = []
    x_ratio = 1280*1.0 / cfg['dataset']['x_size']
    y_ratio = 720*1.0 / cfg['dataset']['y_size']
    #############

    for step, (images, images_path, idx) in enumerate(test_loader):
        # inference
        out_x, out_y, ti = test(lane_agent, images)
        
        # prepare report
        prev_data_idx = -1
        for batch_id, (x_batch, y_batch) in enumerate(zip(out_x, out_y)): # per image

            # check the order of batch sequence
            data_idx = idx[batch_id].item()
            do_report = True
            if prev_data_idx >=0:
                # ignore report if dataloader re-load previous input
                do_report = data_idx > prev_data_idx

            # write into report info
            if do_report:
                if len(x_batch) == 0: # if this img has not pred lanes
                    global_id.append( data_idx )
                    img.append( images_path[batch_id].split('/')[-1] )
                    index.append( 0 )
                    img_width.append(1280)
                    img_height.append(720)
                    lane_prob.append( 1.0 )
                    solid_prob.append( 1.0 )
                    solid_type.append( 'solid' )
                    # insert dumy line
                    points.append( [[1,1],[2,2],[3,3],[4,4]] ) 

                for lid, (lx, ly) in enumerate(zip(x_batch, y_batch)): # per lane
                    global_id.append( data_idx )
                    img.append( images_path[batch_id].split('/')[-1] )
                    index.append( lid )
                    img_width.append(1280)
                    img_height.append(720)
                    lane_prob.append( 1.0 )
                    solid_prob.append( 1.0 )
                    solid_type.append( 'solid' )
                    line = [ [int(x*x_ratio), int(y*y_ratio)] for x, y in zip(lx, ly)]
                    points.append( line )
            prev_data_idx = data_idx

        if step % 1 == 0 or step == 0:
            logging.info('step: {}/{}, idx: {}'.format( step, num_steps, idx))

        if cfg['show_img_in_inference']:
            i = random.randint( 0, images.size(0)-1 )
            pred_path = os.path.join(
                exp_root, report_img_root, 'pred_%s.png'%(idx[i].item()) )
            cv2.imwrite(pred_path, ti[i])
            logging.info('Write image as {}'.format(pred_path))

        if cfg['show_more_img']:
            for i in range(images.size(0)):
                pred_path = os.path.join(
                    exp_root, report_img_root, 'pred_%s.png'%(idx[i].item()) )
                cv2.imwrite(pred_path, ti[i])
                logging.info('Write image as {}'.format(pred_path))

        if cfg['debug'] and step == cfg['debug_steps']:
            break
    # end testing loop

    # save csv report
    cont_list = {'global_id': global_id, 'img':img, 'index':index, 'img_width':img_width, 'img_height':img_height,
                 'prob':lane_prob,'solid_prob':solid_prob, 'solid_type':solid_type,'points':points}
    df = pd.DataFrame(cont_list)



    csv_fpath = 'inf_out_rank_{}.csv'.format( comm.get_rank() )
    csv_fpath = os.path.join(exp_root, report_root, csv_fpath)

    df.to_csv(csv_fpath, index=False)
    logging.info('Save inference output as {}'.format(csv_fpath))
    
    # save state data
    inf_out_fpath = "inf_out_rank_{}.pkl".format( comm.get_rank() )
    inf_out_fpath = os.path.join(exp_root, report_root, inf_out_fpath)

    torch.save(cont_list, inf_out_fpath)
    logging.info('Save output data as {}'.format(inf_out_fpath))


############################################################################
## evaluate on the test dataset
############################################################################
def evaluation(loader, lane_agent, thresh = p.threshold_point, name = None):
    result_data = deepcopy(loader.test_data)
    for test_image, target_h, ratio_w, ratio_h, testset_index in loader.Generate_Test():
        x, y, _ = test(lane_agent, np.array([test_image]), thresh)
        x, y = util.convert_to_original_size(x[0], y[0], ratio_w, ratio_h)
        x, y = find_target(x, y, target_h, ratio_w, ratio_h)
        result_data = write_result_json(result_data, x, y, testset_index)
    if name == None:
        save_result(result_data, "test_result.json")
    else:
        save_result(result_data, name)

############################################################################
## linear interpolation for fixed y value on the test dataset
############################################################################
def find_target(x, y, target_h, ratio_w, ratio_h):
    # find exact points on target_h
    out_x = []
    out_y = []
    x_size = p.x_size/ratio_w
    y_size = p.y_size/ratio_h
    for i, j in zip(x,y): # per lane
        min_y = min(j)
        max_y = max(j)
        temp_x = []
        temp_y = []
        for h in target_h:
            temp_y.append(h)
            if h < min_y:
                temp_x.append(-2)
            elif min_y <= h and h <= max_y:
                for k in range(len(j)-1): # per pt y
                    if j[k] >= h and h >= j[k+1]:
                        #linear regression
                        if i[k] < i[k+1]:
                            temp_x.append(int(i[k+1] - float(abs(j[k+1] - h))*abs(i[k+1]-i[k])/abs(j[k+1]+0.0001 - j[k])))
                        else:
                            temp_x.append(int(i[k+1] + float(abs(j[k+1] - h))*abs(i[k+1]-i[k])/abs(j[k+1]+0.0001 - j[k])))
                        break
            else:
                if i[0] < i[1]:
                    l = int(i[1] - float(-j[1] + h)*abs(i[1]-i[0])/abs(j[1]+0.0001 - j[0]))
                    if l > x_size or l < 0 :
                        temp_x.append(-2)
                    else:
                        temp_x.append(l)
                else:
                    l = int(i[1] + float(-j[1] + h)*abs(i[1]-i[0])/abs(j[1]+0.0001 - j[0]))
                    if l > x_size or l < 0 :
                        temp_x.append(-2)
                    else:
                        temp_x.append(l)
        out_x.append(temp_x)
        out_y.append(temp_y)
    
    return out_x, out_y

############################################################################
## write result
############################################################################
def write_result_json(result_data, x, y, testset_index):
    for i in x:
        result_data[testset_index]['lanes'].append(i)
        result_data[testset_index]['run_time'] = 1
    return result_data

############################################################################
## save result by json form
############################################################################
def save_result(result_data, fname):
    with open(fname, 'w') as make_file:
        for i in result_data:
            json.dump(i, make_file, separators=(',', ': '))
            make_file.write("\n")

############################################################################
## test on the input test image
############################################################################
def test(lane_agent, test_images, thresh = p.threshold_point):

    result = lane_agent.predict_lanes_test(test_images)
    confidences, offsets, instances = result[-1] # result: 2x
    
    num_batch = len(test_images)

    out_x = []
    out_y = []
    out_images = []

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])

    for i in range(num_batch):
        # test on test data set
        image = deepcopy(test_images[i]) # chw
        #image =  np.rollaxis(image, axis=2, start=0)
        #image =  np.rollaxis(image, axis=2, start=0)*255.0 # => hwc
        #image = image.astype(np.uint8).copy()
        image = image.permute(1,2,0).contiguous()
        image = image.numpy()
        image = image * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        image = (image * 255).astype(np.uint8)

        confidence = confidences[i].view(p.grid_y, p.grid_x).cpu().data.numpy() # [32x64]

        offset = offsets[i].cpu().data.numpy() # [2 h w]
        offset = np.rollaxis(offset, axis=2, start=0)
        offset = np.rollaxis(offset, axis=2, start=0) # => [h w 2]
        
        instance = instances[i].cpu().data.numpy()
        instance = np.rollaxis(instance, axis=2, start=0)
        instance = np.rollaxis(instance, axis=2, start=0) # [4 h w] => [h w 4]

        # generate point and cluster
        raw_x, raw_y = generate_result(confidence, offset, instance, thresh)

        # eliminate fewer points
        in_x, in_y = eliminate_fewer_points(raw_x, raw_y)
                
        # sort points along y 
        in_x, in_y = util.sort_along_y(in_x, in_y)  
        in_x, in_y = eliminate_out(in_x, in_y, confidence, deepcopy(image))
        in_x, in_y = util.sort_along_y(in_x, in_y)
        in_x, in_y = eliminate_fewer_points(in_x, in_y)

        result_image = util.draw_lines(in_x, in_y, deepcopy(image))

        out_x.append(in_x)
        out_y.append(in_y)
        out_images.append(result_image)

    return out_x, out_y, out_images

############################################################################
## post processing for eliminating outliers
############################################################################
def eliminate_out(sorted_x, sorted_y, confidence, image = None):
    out_x = []
    out_y = []

    for lane_x, lane_y in zip(sorted_x, sorted_y):

        lane_x_along_y = np.array(deepcopy(lane_x))
        lane_y_along_y = np.array(deepcopy(lane_y))

        ind = np.argsort(lane_x_along_y, axis=0)
        lane_x_along_x = np.take_along_axis(lane_x_along_y, ind, axis=0)
        lane_y_along_x = np.take_along_axis(lane_y_along_y, ind, axis=0)
        
        if lane_y_along_x[0] > lane_y_along_x[-1]: #if y of left-end point is higher than right-end
            starting_points = [(lane_x_along_y[0], lane_y_along_y[0]), (lane_x_along_y[1], lane_y_along_y[1]), (lane_x_along_y[2], lane_y_along_y[2]),
                                (lane_x_along_x[0], lane_y_along_x[0]), (lane_x_along_x[1], lane_y_along_x[1]), (lane_x_along_x[2], lane_y_along_x[2])] # some low y, some left/right x
        else:
            starting_points = [(lane_x_along_y[0], lane_y_along_y[0]), (lane_x_along_y[1], lane_y_along_y[1]), (lane_x_along_y[2], lane_y_along_y[2]),
                                (lane_x_along_x[-1], lane_y_along_x[-1]), (lane_x_along_x[-2], lane_y_along_x[-2]), (lane_x_along_x[-3], lane_y_along_x[-3])] # some low y, some left/right x            
    
        temp_x = []
        temp_y = []
        for start_point in starting_points:
            temp_lane_x, temp_lane_y = generate_cluster(start_point, lane_x, lane_y, image)
            temp_x.append(temp_lane_x)
            temp_y.append(temp_lane_y)
        
        max_lenght_x = None
        max_lenght_y = None
        max_lenght = 0
        for i, j in zip(temp_x, temp_y):
            if len(i) > max_lenght:
                max_lenght = len(i)
                max_lenght_x = i
                max_lenght_y = j
        out_x.append(max_lenght_x)
        out_y.append(max_lenght_y)

    return out_x, out_y

############################################################################
## generate cluster
############################################################################
def generate_cluster(start_point, lane_x, lane_y, image = None):
    cluster_x = [start_point[0]]
    cluster_y = [start_point[1]]

    point = start_point
    while True:
        points = util.get_closest_upper_point(lane_x, lane_y, point, 3)
         
        max_num = -1
        max_point = None

        if len(points) == 0:
            break
        if len(points) < 3:
            for i in points: 
                cluster_x.append(i[0])
                cluster_y.append(i[1])                
            break
        for i in points: 
            num, shortest = util.get_num_along_point(lane_x, lane_y, point, i, image)
            if max_num < num:
                max_num = num
                max_point = i

        total_remain = len(np.array(lane_y)[np.array(lane_y) < point[1]])
        cluster_x.append(max_point[0])
        cluster_y.append(max_point[1])
        point = max_point
        
        if len(points) == 1 or max_num < total_remain/5:
            break

    return cluster_x, cluster_y

############################################################################
## remove same value on the prediction results
############################################################################
def remove_same_point(x, y):
    out_x = []
    out_y = []
    for lane_x, lane_y in zip(x, y): # per lane
        temp_x = []
        temp_y = []
        for i in range(len(lane_x)): # per pt
            if len(temp_x) == 0 :
                temp_x.append(lane_x[i])
                temp_y.append(lane_y[i])
            else:
                if temp_x[-1] == lane_x[i] and temp_y[-1] == lane_y[i]:
                    continue
                else:
                    temp_x.append(lane_x[i])
                    temp_y.append(lane_y[i])     
        out_x.append(temp_x)  
        out_y.append(temp_y)  
    return out_x, out_y

############################################################################
## eliminate result that has fewer points than threshold
############################################################################
def eliminate_fewer_points(x, y):
    # eliminate fewer points
    out_x = []
    out_y = []
    for i, j in zip(x, y): # per lane
        if len(i)>2:
            out_x.append(i)
            out_y.append(j)     
    return out_x, out_y   

############################################################################
## generate raw output
############################################################################
def generate_result(confidance, offsets, instance, thresh):

    mask = confidance > thresh
    #print(mask)

    grid = p.grid_location[mask] # [32 64 2] => [valid_n 2]
    offset = offsets[mask] # ? [valid_n 2]
    feature = instance[mask] # ? [valid_n 4]

    lane_feature = []
    x = []
    y = []
    for i in range(len(grid)): # for each valid grid
        if (np.sum(feature[i]**2))>=0:
            point_x = int((offset[i][0]+grid[i][0])*p.resize_ratio)
            point_y = int((offset[i][1]+grid[i][1])*p.resize_ratio)
            if point_x > p.x_size or point_x < 0 or point_y > p.y_size or point_y < 0:
                continue
            if len(lane_feature) == 0:
                lane_feature.append(feature[i])
                x.append([])
                x[0].append(point_x)
                y.append([])
                y[0].append(point_y)
            else:
                flag = 0
                index = 0
                for feature_idx, j in enumerate(lane_feature): # max len is valid grid num
                    index += 1
                    if index >= 12:
                        index = 12
                    if np.linalg.norm((feature[i] - j)**2) <= p.threshold_instance: # if cur feat is close enough
                        lane_feature[feature_idx] = (j*len(x[index-1]) + feature[i])/(len(x[index-1])+1)
                        x[index-1].append(point_x)
                        y[index-1].append(point_y)
                        flag = 1
                        break
                if flag == 0: # cur feat not close to any
                    lane_feature.append(feature[i])
                    x.append([])
                    x[index].append(point_x) 
                    y.append([])
                    y[index].append(point_y)
                
    return x, y

if __name__ == '__main__':
    Testing()
