#########################################################################
##
## Some utility for training, data processing, and network.
##
#########################################################################
import torch
import torch.nn as nn
from parameters import Parameters
from collections import OrderedDict

p = Parameters()

def backward_hook(self, grad_input, grad_output):
    print('grad_input norm:', grad_input[0].data.norm())

def cross_entropy2d(inputs, target, weight=None, size_average=True):
    loss = torch.nn.CrossEntropyLoss()

    n, c, h, w = inputs.size()
    prediction = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    gt =target.transpose(1, 2).transpose(2, 3).contiguous().view(-1)

    return loss(prediction, gt)

######################################################################
##
## Convolution layer modules
##
######################################################################
class Conv2D_BatchNorm_Relu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, padding, stride, bias=True, acti=True):
        super(Conv2D_BatchNorm_Relu, self).__init__()

        if acti:
            self.cbr_unit = nn.Sequential(nn.Conv2d(in_channels, n_filters, k_size, 
                                                    padding=padding, stride=stride, bias=bias),
                                    nn.BatchNorm2d(n_filters),
                                    nn.ReLU(inplace=True),)
        else:
            self.cbr_unit = nn.Conv2d(in_channels, n_filters, k_size, padding=padding, stride=stride, bias=bias)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

class bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, acti=True):
        super(bottleneck, self).__init__()
        self.acti = acti
        temp_channels = in_channels//4
        if in_channels < 4:
            temp_channels = in_channels
        self.conv1 = Conv2D_BatchNorm_Relu(in_channels, temp_channels, 1, 0, 1)
        self.conv2 = Conv2D_BatchNorm_Relu(temp_channels, temp_channels, 3, 1, 1)
        self.conv3 = Conv2D_BatchNorm_Relu(temp_channels, out_channels, 1, 0, 1, acti = self.acti)

        self.residual = Conv2D_BatchNorm_Relu(in_channels, out_channels, 1, 0, 1)

    def forward(self, x):
        re = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if not self.acti:
            return out

        re = self.residual(x)
        out = out + re

        return out

class bottleneck_down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(bottleneck_down, self).__init__()
        temp_channels = in_channels//4
        if in_channels < 4:
            temp_channels = in_channels
        self.conv1 = Conv2D_BatchNorm_Relu(in_channels, temp_channels, 1, 0, 1)
        self.conv2 = Conv2D_BatchNorm_Relu(temp_channels, temp_channels, 3, 1, 2)
        self.conv3 = Conv2D_BatchNorm_Relu(temp_channels, out_channels, 1, 0, 1)

        self.residual = Conv2D_BatchNorm_Relu(in_channels, out_channels, 3, 1, 2)

    def forward(self, x):
        re = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        re = self.residual(x)

        out = out + re

        return out

class bottleneck_up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(bottleneck_up, self).__init__()
        temp_channels = in_channels//4
        if in_channels < 4:
            temp_channels = in_channels
        self.conv1 = Conv2D_BatchNorm_Relu(in_channels, temp_channels,1,  0, 1)
        self.conv2 = nn.Sequential( nn.ConvTranspose2d(temp_channels, temp_channels, 3, 2, 1, 1),
                                        nn.BatchNorm2d(temp_channels),
                                        nn.ReLU() )
        self.conv3 = Conv2D_BatchNorm_Relu(temp_channels, out_channels, 1, 0, 1)

        self.residual = nn.Sequential( nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1, 1),
                                        nn.BatchNorm2d(out_channels),
                                        nn.ReLU() )

    def forward(self, x):
        re = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        re = self.residual(re)

        out = out + re

        return out

class Output(nn.Module):
    def __init__(self, in_size, out_size):
        super(Output, self).__init__()
        self.conv = bottleneck(in_size, out_size, acti=False)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        return outputs

class hourglass_same(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(hourglass_same, self).__init__()
        self.down1 = bottleneck_down(in_channels, out_channels)
        self.down2 = bottleneck_down(out_channels, out_channels)
        self.down3 = bottleneck_down(out_channels, out_channels)
        self.down4 = bottleneck_down(out_channels, out_channels)

        self.same1 = bottleneck(out_channels, out_channels)
        self.same2 = bottleneck(out_channels, out_channels)

        self.up2 = bottleneck_up(out_channels, out_channels)
        self.up3 = bottleneck_up(out_channels, out_channels)
        self.up4 = bottleneck_up(out_channels, out_channels)
        self.up5 = bottleneck_up(out_channels, out_channels)

        self.residual1 = bottleneck_down(in_channels, out_channels)
        self.residual2 = bottleneck_down(out_channels, out_channels)
        self.residual3 = bottleneck_down(out_channels, out_channels)
        self.residual4 = bottleneck_down(out_channels, out_channels)

    def forward(self, inputs):
        outputs1 = self.down1(inputs)  # 512*256 -> 256*128
        outputs2 = self.down2(outputs1)  # 256*128 -> 128*64
        outputs3 = self.down3(outputs2)  # 128*64 -> 64*32
        outputs4 = self.down4(outputs3)  # 64*32 -> 32*16

        outputs = self.same1(outputs4)  # 16*8 -> 16*8
        outputs = self.same2(outputs)  # 16*8 -> 16*8
        
        outputs = self.up2(outputs + self.residual4(outputs3))  # 32*16 -> 64*32
        outputs = self.up3(outputs + self.residual3(outputs2))  # 64*32 -> 128*64
        outputs = self.up4(outputs + self.residual2(outputs1))  # 128*64 -> 256*128
        outputs = self.up5(outputs + self.residual1(inputs))  # 256*128 -> 512*256

        return outputs   

class resize_layer(nn.Module):
    def __init__(self, in_channels, out_channels, acti = True):
        super(resize_layer, self).__init__()
        self.conv = Conv2D_BatchNorm_Relu(in_channels, out_channels//2, 7, 3, 2)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.re1 = bottleneck(out_channels//2, out_channels//2)
        self.re2 = bottleneck(out_channels//2, out_channels//2)
        self.re3 = bottleneck(out_channels//2, out_channels)

    def forward(self, inputs):
        ## [N, 3, 256, 512]
        outputs = self.conv(inputs)
        ## [N, 64, 128, 256]
        outputs = self.re1(outputs)
        ## [N, 64, 128, 256]
        outputs = self.maxpool(outputs)
        ## [N, 64, 64, 128]
        outputs = self.re2(outputs)
        ## [N, 64, 64, 128]
        outputs = self.maxpool(outputs)
        ## [N, 64, 32, 64]
        outputs = self.re3(outputs)
        ## [N, 128, 32, 64]

        return outputs   

class Conv2d(torch.nn.Conv2d):
    def forward(self, x):
        if x.numel() > 0:
            return super(Conv2d, self).forward(x)
        # get output shape

        output_shape = [
            (i + 2 * p - (di * (k - 1) + 1)) // d + 1
            for i, p, di, k, d in zip(
                x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
            )
        ]
        output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)

def conv_with_kaiming_uniform(use_gn=False, use_relu=False):
    def make_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1
    ):
        conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            bias=False if use_gn else True
        )
        # Caffe2 implementation uses XavierFill, which in fact
        # corresponds to kaiming_uniform_ in PyTorch
        nn.init.kaiming_uniform_(conv.weight, a=1)
        if not use_gn:
            nn.init.constant_(conv.bias, 0)
        module = [conv,]
        if use_gn:
            module.append(group_norm(out_channels))
        if use_relu:
            module.append(nn.ReLU(inplace=True))
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv

    return make_conv

class fpn_layer(nn.Module):
    def __init__(self, in_channels, out_channels, acti = True):
        super(fpn_layer, self).__init__()
        self.conv = Conv2D_BatchNorm_Relu(in_channels, out_channels//2, 7, 3, 2)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.re1 = bottleneck(out_channels//2, out_channels//2)
        self.re2 = bottleneck(out_channels//2, out_channels//2)
        self.re3 = bottleneck(out_channels//2, out_channels)

        conv_block = conv_with_kaiming_uniform()
        self.inner1 = conv_block(out_channels//2,out_channels,1)
        self.layer1 = conv_block(out_channels,out_channels,3,1)
        self.inner2 = conv_block(out_channels,out_channels,1)
        self.layer2 = conv_block(out_channels,out_channels,3,1)

    def forward(self, inputs):
        outs = []
        ## [N, 3, 256, 512]
        outputs = self.conv(inputs)
        ## [N, 64, 128, 256]
        outputs = self.re1(outputs)
        ## [N, 64, 128, 256]
        outputs = self.maxpool(outputs)
        ## [N, 64, 64, 128]
        outputs = self.re2(outputs)
        outs.append(outputs)
        ## [N, 64, 64, 128]
        outputs = self.maxpool(outputs)
        ## [N, 64, 32, 64]
        outputs = self.re3(outputs)
        ## [N, 128, 32, 64]
        outs.append(outputs)

        outs[1] = self.inner2(outs[1])
        outs[1] = self.layer2(outs[1])

        inner_top_down = torch.nn.functional.interpolate(outs[1],scale_factor=2, mode="nearest")

        outs[0] = self.inner1(outs[0])
        outs[0] = self.layer1(outs[0]+inner_top_down)



        return outs # outputs  

class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr

class BatchRNN(nn.Module):
    def __init__(self, 
                input_size, 
                hidden_size, 
                rnn_type=nn.LSTM, 
                bidirectional=False, 
                batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=True)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, output_lengths):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths)
        x, h = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        return x

class hourglass_block(nn.Module):
    def __init__(self, in_channels, out_channels, acti = True, input_re=True):
        super(hourglass_block, self).__init__()
        self.layer1 = hourglass_same(in_channels, out_channels)
        self.re1 = bottleneck(out_channels, out_channels)
        self.re2 = bottleneck(out_channels, out_channels)
        self.re3 = bottleneck(1, out_channels)  

        self.out_confidence = Output(out_channels, 1)      
        self.out_offset = Output(out_channels, 2)      
        self.out_instance = Output(out_channels, p.feature_size)  
        self.input_re = input_re    

        # LSTM
        '''
        rnns = []
        num_rnns = 5
        rnn = BatchRNN(input_size=128*64,
                        hidden_size=1024,
                        bidirectional=True,
                        )
        rnns.append( ('0', rnn) )
        for x in range(num_rnns-1):
            rnn = BatchRNN(input_size=1024,
                hidden_size=1024,
                bidirectional=True,
                )
            rnns.append( ('%d' % (x + 1), rnn) )
        self.rnns = nn.Sequential(OrderedDict(rnns))
        '''

    def forward(self, inputs):
        # [N, 128, 32, 64]
        outputs = self.layer1(inputs)
        # [N, 128, 32, 64]
        outputs = self.re1(outputs)
        # [N, 128, 32, 64]






        out_confidence = self.out_confidence(outputs)
        out_offset = self.out_offset(outputs)
        out_instance = self.out_instance(outputs)
        # [N, 4, 32, 64]

        out = out_confidence

        outputs = self.re2(outputs)
        out = self.re3(out)

        if self.input_re:
            outputs = outputs + out + inputs
        else:
            outputs = outputs + out
        # [N, 128, 32, 64]
        '''
        print( 'hourglass_block out: {}'.format(outputs.size()) )

        # [N, C=128, T=32, 64] => [N, T=32, C=128, 64]
        seq_h = outputs.transpose(1, 2).contiguous() # copy
        size_sh = seq_h.size()
        seq_h = seq_h.view(size_sh[0], size_sh[1], size_sh[2]*size_sh[3])
        print( 'hourglass_block rnn_in seq_h: {}'.format(seq_h.size()) )
        # LSTM
        for i, rnn in enumerate(self.rnns):
            seq_h = rnn(seq_h, output_lengths)
            print( 'hourglass_block rnn_{} seq_h: {}'.format(i, seq_h.size()) )
        print( 'hourglass_block rnn_out seq_h: {}'.format(seq_h.size()) )

	'''
        return [out_confidence, out_offset, out_instance], outputs
