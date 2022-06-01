#########################################################################
##
## Structure of network.
##
#########################################################################
import torch
import torch.nn as nn
from util_hourglass import *

####################################################################
##
## lane_detection_network
##
####################################################################
class lane_detection_network(nn.Module):
    def __init__(self):
        super(lane_detection_network, self).__init__()

        self.resizing = resize_layer(3, 128)

        #feature extraction
        self.layer1 = hourglass_block(128, 128)
        self.layer2 = hourglass_block(128, 128)


    def forward(self, inputs):
        #feature extraction
        out = self.resizing(inputs)
        result1, out = self.layer1(out)
        result2, out = self.layer2(out)      

        return [result1, result2]


class curve_lane_net(nn.Module):
    def __init__(self):
        super(curve_lane_net, self).__init__()

        self.fpn = fpn_layer(3, 128)

        #feature extraction
        self.layer1 = hourglass_block(128, 128)
        self.layer2 = hourglass_block(128, 128)


    def forward(self, x):
        #feature extraction
        feats = self.fpn(x)
        
        pred = []
        for feat in feats: # 2 level
            result1, t = self.layer1(feat)
            result2, t = self.layer2(t)
            
            pred.append(result1)
            pred.append(result2)

        # return [result1, result2]
        return pred