import os
import yaml
import logging

def setup_exp_dir(exps_dir, exp_name):
    dirs = ["models", "inference_out", "img_show"]
    exp_root = os.path.join(exps_dir, exp_name)
    for dirname in dirs:
        os.makedirs(os.path.join(exp_root, dirname), exist_ok=True)
    return exp_root

def set_config(path):
    with open(path, 'r') as file:
        config_str = file.read()
    config = yaml.load(config_str, Loader=yaml.FullLoader)
    return config

def get_exp_checkpoint(exp_root):
    models_dir = os.path.join(exp_root, "models")
    models = os.listdir(models_dir)
    last_epoch, last_modelname = sorted(
        # [(int(name.split("_")[1].split(".")[0]), name) for name in models],
        [(int(name.split("_")[0]), name) for name in models],
        key=lambda x: xrange[0],
        )
    model_path = os.path.join(models_dir, last_modelname)
    checkpoint = torch.load(model_path,
                            map_location=torch.device("cpu"))
    logging.info('resume model from {}'.format(model_path))
    return checkpoint

def get_module(model, distributed):
    if distributed:
        return model.module
    else:
        return model

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def get_loss_info_str(loss_meter_dict):
    msg = ''
    for key in loss_meter_dict.keys():
        msg += '{name}: {meter.val:.3e} ({meter.avg:.3e})\t'.format(
            name=key, meter=loss_meter_dict[key]
        )

    return msg