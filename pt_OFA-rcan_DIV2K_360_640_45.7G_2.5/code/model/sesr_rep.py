# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## ECCV-2018-Image Super-Resolution Using Very Deep Residual Channel Attention Networks
## https://arxiv.org/abs/1807.02758
import torch
from model import common
import copy
import numpy as np
import torch.nn as nn
from collections import OrderedDict

def make_model(args, parent=False):
    return SESR(args)

def Conv(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    return result

## Residual Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        res_scale=1, deploy=False):

        super(RCAB, self).__init__()
        self.in_channels = n_feat
        self.groups = 1
        self.res_scale = res_scale
        self.deploy = deploy
  
        self.body_reparam = nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=kernel_size, stride=1,
                                      padding=1, dilation=1, groups=1, bias=False)
            
    def forward(self, x):
        return self.body_reparam(x)
      

## RonvGroup
class ConvGroup(nn.Module):
    def __init__(
        self, conv, in_feat, mid_feat, out_feat, kernel_size, deploy=False):

        super(ConvGroup, self).__init__()
        self.body_reparam = nn.Conv2d(in_channels=in_feat, out_channels=out_feat, kernel_size=kernel_size, stride=1,
                                      padding=(kernel_size - 1) // 2, dilation=1, groups=1, bias=False)
            
            
    def forward(self, x):
        return self.body_reparam(x)
 
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, res_scale, n_resblocks, deploy):
        super(ResidualGroup, self).__init__()
        self.deploy = deploy 
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, res_scale=1, deploy=self.deploy) \
            for _ in range(n_resblocks)]
        self.body = nn.Sequential(*modules_body)
        self.act = nn.ReLU()

    def forward(self, x):
        res = self.body(x)
        res += x
        return self.act(res)

class SESR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SESR, self).__init__()
        
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale[0]
        act = nn.PReLU()
        #deploy = args.deploy
        #self.deploy = deploy 
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        self.head = ConvGroup(conv, in_feat=args.n_colors, mid_feat=256, out_feat=n_feats, kernel_size=5, deploy=args.deploy)
                           
        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, res_scale=args.res_scale, n_resblocks=n_resblocks, deploy=args.deploy) \
            for _ in range(n_resgroups)]

        self.body = nn.Sequential(*modules_body)
 
        # define tail module
        self.tail = ConvGroup(conv, in_feat=n_feats, mid_feat=256, out_feat=scale*scale*args.n_colors, kernel_size=5, deploy=args.deploy)

        self.ps = nn.PixelShuffle(scale)

    def forward(self, x): 
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        x = self.ps(x)
        x = self.add_mean(x)
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


def model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print('Save converted model in: ', save_path)
    return model
