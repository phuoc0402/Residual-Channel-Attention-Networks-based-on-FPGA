# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.module_0 = py_nndct.nn.Input() #Model::input_0
        self.module_1 = py_nndct.nn.quant_input() #Model::Model/QuantStub[model]/QuantStub[quant_stub]/input.1
        self.module_2 = py_nndct.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/MeanShiftConv[model]/MeanShiftConv[sub_mean]/Conv2d[mean_conv]/input.2
        self.module_3 = py_nndct.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential[model]/Sequential[head]/Conv2d[0]/input.3
        self.module_4 = py_nndct.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[0]/Sequential[body]/Conv2d[0]/input.4
        self.module_5 = py_nndct.nn.ReLU(inplace=True) #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[0]/Sequential[body]/ReLU[1]/input.5
        self.module_6 = py_nndct.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[0]/Sequential[body]/Conv2d[2]/3049
        self.module_7 = py_nndct.nn.Add() #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[0]/Add[skip_add]/input.6
        self.module_8 = py_nndct.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[1]/Sequential[body]/Conv2d[0]/input.7
        self.module_9 = py_nndct.nn.ReLU(inplace=True) #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[1]/Sequential[body]/ReLU[1]/input.8
        self.module_10 = py_nndct.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[1]/Sequential[body]/Conv2d[2]/3090
        self.module_11 = py_nndct.nn.Add() #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[1]/Add[skip_add]/input.9
        self.module_12 = py_nndct.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[2]/Sequential[body]/Conv2d[0]/input.10
        self.module_13 = py_nndct.nn.ReLU(inplace=True) #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[2]/Sequential[body]/ReLU[1]/input.11
        self.module_14 = py_nndct.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[2]/Sequential[body]/Conv2d[2]/3131
        self.module_15 = py_nndct.nn.Add() #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[2]/Add[skip_add]/input.12
        self.module_16 = py_nndct.nn.Conv2d(in_channels=16, out_channels=48, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[3]/Sequential[body]/Conv2d[0]/input.13
        self.module_17 = py_nndct.nn.ReLU(inplace=True) #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[3]/Sequential[body]/ReLU[1]/input.14
        self.module_18 = py_nndct.nn.Conv2d(in_channels=48, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[3]/Sequential[body]/Conv2d[2]/3172
        self.module_19 = py_nndct.nn.Add() #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[3]/Add[skip_add]/input.15
        self.module_20 = py_nndct.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[4]/Sequential[body]/Conv2d[0]/input.16
        self.module_21 = py_nndct.nn.ReLU(inplace=True) #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[4]/Sequential[body]/ReLU[1]/input.17
        self.module_22 = py_nndct.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[4]/Sequential[body]/Conv2d[2]/3213
        self.module_23 = py_nndct.nn.Add() #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[4]/Add[skip_add]/input.18
        self.module_24 = py_nndct.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[5]/Sequential[body]/Conv2d[0]/input.19
        self.module_25 = py_nndct.nn.ReLU(inplace=True) #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[5]/Sequential[body]/ReLU[1]/input.20
        self.module_26 = py_nndct.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[5]/Sequential[body]/Conv2d[2]/3254
        self.module_27 = py_nndct.nn.Add() #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[5]/Add[skip_add]/input.21
        self.module_28 = py_nndct.nn.Conv2d(in_channels=16, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[6]/Sequential[body]/Conv2d[0]/input.22
        self.module_29 = py_nndct.nn.ReLU(inplace=True) #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[6]/Sequential[body]/ReLU[1]/input.23
        self.module_30 = py_nndct.nn.Conv2d(in_channels=64, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[6]/Sequential[body]/Conv2d[2]/3295
        self.module_31 = py_nndct.nn.Add() #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[6]/Add[skip_add]/input.24
        self.module_32 = py_nndct.nn.Conv2d(in_channels=16, out_channels=24, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[7]/Sequential[body]/Conv2d[0]/input.25
        self.module_33 = py_nndct.nn.ReLU(inplace=True) #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[7]/Sequential[body]/ReLU[1]/input.26
        self.module_34 = py_nndct.nn.Conv2d(in_channels=24, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[7]/Sequential[body]/Conv2d[2]/3336
        self.module_35 = py_nndct.nn.Add() #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/RCAB[7]/Add[skip_add]/input.27
        self.module_36 = py_nndct.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Sequential[body]/Conv2d[8]/3357
        self.module_37 = py_nndct.nn.Add() #Model::Model/Sequential[model]/Sequential[body]/ResidualGroup[0]/Add[skip_add]/input.28
        self.module_38 = py_nndct.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential[model]/Sequential[body]/Conv2d[1]/3378
        self.module_39 = py_nndct.nn.Add() #Model::Model/Add[model]/Add[skip_add]/input.29
        self.module_40 = py_nndct.nn.Conv2d(in_channels=16, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential[model]/Sequential[tail]/Upsampler[0]/Conv2d[0]/3399
        self.module_41 = py_nndct.nn.Module('pixel_shuffle',upscale_factor=2) #Model::Model/Sequential[model]/Sequential[tail]/Upsampler[0]/PixelShuffle[1]/input.30
        self.module_42 = py_nndct.nn.Conv2d(in_channels=16, out_channels=3, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Model::Model/Sequential[model]/Sequential[tail]/Conv2d[1]/input
        self.module_43 = py_nndct.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/MeanShiftConv[model]/MeanShiftConv[add_mean]/Conv2d[mean_conv]/3439
        self.module_44 = py_nndct.nn.dequant_output() #Model::Model/DeQuantStub[model]/DeQuantStub[dequant_stub]/3440

    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(input=output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_4 = self.module_4(output_module_0)
        output_module_4 = self.module_5(output_module_4)
        output_module_4 = self.module_6(output_module_4)
        output_module_4 = self.module_7(input=output_module_4, other=output_module_0, alpha=1)
        output_module_8 = self.module_8(output_module_4)
        output_module_8 = self.module_9(output_module_8)
        output_module_8 = self.module_10(output_module_8)
        output_module_8 = self.module_11(input=output_module_8, other=output_module_4, alpha=1)
        output_module_12 = self.module_12(output_module_8)
        output_module_12 = self.module_13(output_module_12)
        output_module_12 = self.module_14(output_module_12)
        output_module_12 = self.module_15(input=output_module_12, other=output_module_8, alpha=1)
        output_module_16 = self.module_16(output_module_12)
        output_module_16 = self.module_17(output_module_16)
        output_module_16 = self.module_18(output_module_16)
        output_module_16 = self.module_19(input=output_module_16, other=output_module_12, alpha=1)
        output_module_20 = self.module_20(output_module_16)
        output_module_20 = self.module_21(output_module_20)
        output_module_20 = self.module_22(output_module_20)
        output_module_20 = self.module_23(input=output_module_20, other=output_module_16, alpha=1)
        output_module_24 = self.module_24(output_module_20)
        output_module_24 = self.module_25(output_module_24)
        output_module_24 = self.module_26(output_module_24)
        output_module_24 = self.module_27(input=output_module_24, other=output_module_20, alpha=1)
        output_module_28 = self.module_28(output_module_24)
        output_module_28 = self.module_29(output_module_28)
        output_module_28 = self.module_30(output_module_28)
        output_module_28 = self.module_31(input=output_module_28, other=output_module_24, alpha=1)
        output_module_32 = self.module_32(output_module_28)
        output_module_32 = self.module_33(output_module_32)
        output_module_32 = self.module_34(output_module_32)
        output_module_32 = self.module_35(input=output_module_32, other=output_module_28, alpha=1)
        output_module_32 = self.module_36(output_module_32)
        output_module_32 = self.module_37(input=output_module_32, other=output_module_0, alpha=1)
        output_module_32 = self.module_38(output_module_32)
        output_module_32 = self.module_39(input=output_module_32, other=output_module_0, alpha=1)
        output_module_32 = self.module_40(output_module_32)
        output_module_32 = self.module_41(output_module_32)
        output_module_32 = self.module_42(output_module_32)
        output_module_32 = self.module_43(output_module_32)
        output_module_32 = self.module_44(input=output_module_32)
        return output_module_32
