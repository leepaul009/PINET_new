#############################################################################################################
##
##  Source code for training. In this source code, there are initialize part, training part, ...
##
#############################################################################################################

import cv2
import torch
# import visdom
import agent
import numpy as np
# from data_loader import Generator
from lib.dataset import BuildDataLoader, LaneDataset
from lib.utils import setup_exp_dir, set_config, get_exp_checkpoint, AverageMeter, get_loss_info_str
from lib import comm
from parameters import Parameters
import test
import evaluation
import argparse
import os
import logging
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

p = Parameters()

def parse_args():
    parser = argparse.ArgumentParser(description="Train PINet")
    parser.add_argument("--cfg", default="config.yaml", help="Config file", required=True)
    parser.add_argument("--exp_name", default="default", help="Experiment name", required=True)
    parser.add_argument("--resume", action="store_true", help="Resume training")
    # parser.add_argument("--validation", action="store_true", help="Resume training")
    parser.add_argument("--cuda_start", type=int, default=0)
    parser.add_argument("--cuda_num", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=0)
    
    return parser.parse_args()

###############################################################
## Training
###############################################################
def Training():
    args = parse_args()
    cfg = set_config(args.cfg)

    if not args.resume:
        exp_root = setup_exp_dir(cfg['exps_dir'], args.exp_name)
    else:
        exp_root = os.path.join(cfg['exps_dir'], 
                                os.path.basename(os.path.normpath(args.exp_name)))
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(exp_root, "log.txt")),
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
    logging.info('rank: {}, distributed: {}, device: {}'.format(comm.get_rank(), distributed, device))

    #########################################################################
    ## Get dataset
    #########################################################################
    # print("Get dataset")
    train_dataset = LaneDataset(cfg, 'train')
    train_loader = BuildDataLoader(train_dataset, True, distributed, **cfg['train_loader_parameter'])
    if cfg['do_validation']:
        val_dataset   = LaneDataset(cfg, 'val')
        val_loader   = BuildDataLoader(val_dataset, False, distributed, **cfg['val_loader_parameter'])

    #########################################################################
    ## Get agent and model
    #########################################################################
    lane_agent = agent.Agent(cfg, device, distributed)

    if distributed:
        lane_agent.convert_sync_batchnorm()
    lane_agent.to_cuda()
    if distributed:
        lane_agent.buildDistributedModel(args.local_rank)

    checkpoint = None
    if args.resume:
        lane_agent.resume(exp_root)
    
    starting_epoch = lane_agent.starting_epoch

    #########################################################################
    ## Loop for training
    #########################################################################
    num_epoch = cfg['num_epoch'] if not cfg['debug'] else starting_epoch
    num_steps = len(train_loader)

    if comm.is_main_process():
        logging.info("Starting training from epoch {} to {}".format( starting_epoch, num_epoch ))
    ## Training loop
    for epoch in range(starting_epoch, num_epoch + 1):

        lane_agent.training_mode()
        if distributed:
            train_loader.sampler.set_epoch(epoch)
        loss_meter = AverageMeter()
        
        for step, (inputs, labels, lanes, idx) in enumerate(train_loader):
            loss_p, loss_dict = lane_agent.train(inputs, labels, lanes, epoch)

            # reduce from other ranks
            if distributed:
                torch.distributed.all_reduce(loss_p, op=torch.distributed.ReduceOp.SUM)
            loss_p = loss_p.detach().cpu().item() / comm.get_world_size()
            # loss_p = loss_p.detach().cpu().item() # loss_p.cpu().data
            loss_meter.update( loss_p, 1 )
            
            ## Logging
            if step % cfg['logging_interval'] == 0 or cfg['debug']:
                gt, gtn, gti = labels # gt_point, gt_existance, gt_instance
                target_lanes, target_h = lanes # x, y
                # loss details
                loss_str = ', '.join([ '{}: {:4f}'.format(loss_name, loss_dict[loss_name])
                                      for loss_name in loss_dict])
                # debug output
                debug_str = 'img: {}, grid: {}, conf: {}, ins: {}'.format(
                        inputs.size(), gt.size(), gtn.size(), gti.size(),
                    ) if cfg['debug'] else None

                logging.info('[rank {}]  epoch: {}/{}, step: {}/{}, loss.val: {:4f}, loss.avg: {:4f}, {}, {}'
                    .format( comm.get_rank(), epoch, num_epoch, step, num_steps, 
                            loss_meter.val, loss_meter.avg, loss_str, debug_str ))
            
            if cfg['debug'] and step == cfg['debug_steps']:
                break

        ## Save model
        if comm.is_main_process():
            lane_agent.save_model( exp_root, int(epoch), loss_p )

        ## Validation
        if cfg['do_validation']:
            lane_agent.evaluate_mode()
            val_loss_meter = AverageMeter()
            for step, (inputs, labels, lanes, idx) in enumerate(val_loader):
                loss_p, loss_dict = lane_agent.validation(inputs, labels, lanes, epoch)
            
                # reduce from other ranks
                if distributed:
                    torch.distributed.all_reduce(loss_p, op=torch.distributed.ReduceOp.SUM)
                loss_p = loss_p.detach().cpu().item() / comm.get_world_size()
                val_loss_meter.update( loss_p, 1 )

                if cfg['debug'] and step == cfg['debug_steps']:
                    break
            if comm.is_main_process():
                logging.info('epoch {}, val_loss: {}'.format( epoch, val_loss_meter.avg ))


        '''
        # evaluation
        # if epoch > 0 and epoch % 10 == 0:
        logging.info("evaluation start...")
        lane_agent.evaluate_mode()
        th_list = [0.3, 0.5, 0.7]
        # lane_agent.save_model(int(step/100), loss_p)

        for th in th_list:
            print("generate result")
            print(th)
            test.evaluation(loader, lane_agent, thresh = th, name="test_result_"+str(epoch)+"_"+str(th)+".json")

        for th in th_list:
            print("compute score")
            print(th)
            with open("eval_result_"+str(th)+"_.txt", 'a') as make_file:
                make_file.write( "epoch : " + str(epoch) + " loss : " + str(loss_p.cpu().data) )
                make_file.write(evaluation.LaneEval.bench_one_submit("test_result_"+str(epoch)+"_"+str(th)+".json", "test_label.json"))
                make_file.write("\n")
        '''

        ## Epoch end
    ## Training end

def testing(lane_agent, test_image, step, loss):
    lane_agent.evaluate_mode()

    _, _, ti = test.test(lane_agent, np.array([test_image]))

    cv2.imwrite('test_result/result_'+str(step)+'_'+str(loss)+'.png', ti[0])

    lane_agent.training_mode()

    
if __name__ == '__main__':
    Training()

