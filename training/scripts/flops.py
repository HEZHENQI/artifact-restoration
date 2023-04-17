from ptflops import get_model_complexity_info
from improved_diffusion.swinunet_flops import SwinUNetModel
from improved_diffusion.SegFormer_flops import Segformer


import copy
import logging
import math

from os.path import join as pjoin
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import os

import torch.optim as optim
from torchvision import transforms
import torch.utils.data as data
import scipy.io as sio


def prepare_input(resolution):
    x1 = torch.FloatTensor(1, 3, 256, 256).cuda()
    x2 = torch.FloatTensor(1).cuda()
    return dict(x = [x1,x2])

model = Segformer(
        dims = (32, 64, 160, 256),      # dimensions of each stage
        heads = (1, 2, 5, 8),           # heads of each stage
        ff_expansion = (8, 8, 4, 4),    # feedforward expansion factor of each stage
        reduction_ratio = (8, 4, 2, 1), # reducti,on ratio of each stage for efficient attention
        num_layers = 2,                 # num layers of each stage
        decoder_dim = 256,              # decoder dimension
        num_classes = 6, 
        channels = 3
    )
model.to('cuda')
flops, params = get_model_complexity_info(model, (3, 256, 256),input_constructor=prepare_input, as_strings=True,
                                          print_per_layer_stat=False)  # 不用写batch_size大小，默认batch_size=1
# print("Swin_unet")
print('Flops:  ' + flops)
print('Params: ' + params)
