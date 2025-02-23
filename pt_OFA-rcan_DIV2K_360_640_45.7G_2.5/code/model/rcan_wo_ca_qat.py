## ECCV-2018-Image Super-Resolution Using Very Deep Residual Channel Attention Networks
## https://arxiv.org/abs/1807.02758
from model import common

import torch.nn as nn

from pytorch_nndct import nn as nndct_nn
from pytorch_nndct.nn.modules import functional
from pytorch_nndct import QatProcessor

def make_model(args, parent=False):
    return RCAN_wo_CA(args)

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1, be=0):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            if i == 0:
                modules_body.append(conv(n_feat, be, kernel_size, bias=bias))
                if bn: modules_body.append(nn.BatchNorm2d(be))
            else:
                modules_body.append(conv(be, n_feat, kernel_size, bias=bias))
                if bn: modules_body.append(nn.BatchNorm2d(n_feat))

            if i == 0: modules_body.append(act)
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

        self.skip_add = functional.Add()

    def forward(self, x):
        res = self.body(x)
        res = self.skip_add(res, x)
        #res += x
        return res
## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks, be):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1, be=be[j]) \
            for j in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

        self.skip_add = functional.Add()

    def forward(self, x):
        res = self.body(x)
        #res += x
        res = self.skip_add(res, x)
        return res

## Residual Channel Attention Network (RCAN)
class RCAN_wo_CA(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RCAN_wo_CA, self).__init__()
        
        n_resgroups = args.n_resgroups
        n_resblocks = [8]
        n_be = [16, 32, 32, 48, 16, 16, 64, 24]
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale[0]
        act = nn.ReLU(True)
        
        # RGB mean for DIV2K
        self.sub_mean = common.MeanShiftConv(args.rgb_range)
        
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = []
        idx1 = 0
        idx2 = 0
        for i in range(n_resgroups):
            idx2 = idx1 + n_resblocks[i]
            be_current = n_be[idx1:idx2]
            modules_body.append(ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_resblocks=n_resblocks[i], be=be_current))
            idx1 = idx1 + n_resblocks[i]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShiftConv(args.rgb_range, sign=1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)


        self.skip_add = functional.Add()
        self.quant_stub = nndct_nn.QuantStub()
        self.dequant_stub = nndct_nn.DeQuantStub()

    def forward(self, x): 
        x = self.quant_stub(x)
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        
        res = self.skip_add(res, x)
        #res += x
        x = self.tail(res)
        x = self.add_mean(x)

        x = self.dequant_stub(x)
        return x 

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
