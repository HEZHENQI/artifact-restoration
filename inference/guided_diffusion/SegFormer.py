from math import sqrt
from functools import partial
import torch
from torch import nn, einsum
import torch.nn.functional as F
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
)

from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import copy
import logging
import math

from os.path import join as pjoin
import torch
import torch.nn as nn
import numpy as np

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
import matplotlib.pyplot as plt
from abc import abstractmethod

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """
class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class SinusoidalPositionEmbeddings(nn.Module):         #正弦时间编码
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# helpers

def exists(val):
    return val is not None

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

# classes

class DsConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))

class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        reduction_ratio
    ):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, dim * 2, reduction_ratio, stride = reduction_ratio, bias = False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias = False)

    def forward(self, x):
        h, w = x.shape[-2:]
        heads = self.heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h = heads, x = h, y = w)
        return self.to_out(out)

class MixFeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        expansion_factor
    ):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            DsConv2d(hidden_dim, hidden_dim, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class MiT(TimestepBlock):
    def __init__(
        self,
        *,
        channels,
        dims,
        heads,
        ff_expansion,
        reduction_ratio,
        num_layers,
        time_embed_dim
    ):
        super().__init__()
        stage_kernel_stride_pad = ((7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))
        # stage_kernel_stride_pad = ((3, 2, 1), (3, 2, 1), (3, 2, 1), (3, 2, 1))
        
        self.time_embed_dim = time_embed_dim
        
        
        dims = (channels, *dims)
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.stages = nn.ModuleList([])
        self.time_embed_layers = nn.ModuleList([])
        n = 0
        for (dim_in, dim_out), (kernel, stride, padding), num_layers, ff_expansion, heads, reduction_ratio in zip(dim_pairs, stage_kernel_stride_pad, num_layers, ff_expansion, heads, reduction_ratio):
            # print('in MiT dim in dim out',dim_in,' ',dim_out)
            get_overlap_patches = nn.Unfold(kernel, stride = stride, padding = padding)
            overlap_patch_embed = nn.Conv2d(dim_in * kernel ** 2, dim_out, 1)
            # print('get_overlap_patches: ',get_overlap_patches)
            # print('overlap_patch_embed: ',overlap_patch_embed)

            layers = nn.ModuleList([])
            time_layers = nn.ModuleList([])
            
            

            for _ in range(num_layers):
                layers.append(nn.ModuleList([
                    PreNorm(dim_out, EfficientSelfAttention(dim = dim_out, heads = heads, reduction_ratio = reduction_ratio)),
                    PreNorm(dim_out, MixFeedForward(dim = dim_out, expansion_factor = ff_expansion)),
                ]))
                time_layers.append(nn.Sequential(
                    SiLU(),
                    linear(
                        time_embed_dim,
                        2**(6-n),
                    ),
                ))
                # print('2**(6-_):',2**(6-_))
            
            self.stages.append(nn.ModuleList([
                get_overlap_patches,
                overlap_patch_embed,
                layers,
                time_layers
            ]))
            n+=1
            # self.time_embed_layers.append(time_layers)

    def forward(
        self,
        x,
        emb,
        return_layer_outputs = False
    ):
        h, w = x.shape[-2:]
        # print('MIT h,w',h,w)
        # print('MIT emb shape',emb.shape)

        layer_outputs = []
        for (get_overlap_patches, overlap_embed, layers,time_layers) in self.stages:
            x = get_overlap_patches(x)
            # print('x in MiT after get_overlap_patches ',x.shape)

            num_patches = x.shape[-1]
            ratio = int(sqrt((h * w) / num_patches))
            x = rearrange(x, 'b c (h w) -> b c h w', h = h // ratio)
            # print('x in MiT before overlap_embed ',x.shape)

            x = overlap_embed(x)
            for (attn, ff),emb_layer in zip(layers,time_layers):
                x = attn(x) + x
                # print(emb_layer)
                
                
                emb_out = emb_layer(emb).type(x.dtype)
                # print('In MiT, emb after emb_layer shape ',emb_out.shape)
                while len(emb_out.shape) < len(x.shape):
                    emb_out = emb_out[..., None]
                # print('In MiT, emb before add x , emb shape ',emb_out.shape)
                x = x + emb_out
                
                x = ff(x) + x

            layer_outputs.append(x)

        ret = x if not return_layer_outputs else layer_outputs
        return ret

class Segformer(nn.Module):
    def __init__(
        self,
        *,
        dims = (32, 64, 160, 256),
        heads = (1, 2, 5, 8),
        ff_expansion = (8, 8, 4, 4),
        reduction_ratio = (8, 4, 2, 1),
        num_layers = 2,
        channels = 3,
        decoder_dim = 256,
        num_classes = 4
    ):
        super().__init__()
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth = 4), (dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, num_layers))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'
        embed_dim=2*128
        self.time_dim=embed_dim*2
        self.decoder_dim = decoder_dim
        self.time_mlp = nn.Sequential(
            linear(embed_dim, self.time_dim),
            SiLU(),
            linear(self.time_dim, self.time_dim),
        )
        
        
        self.mit = MiT(
            channels = channels,
            dims = dims,
            heads = heads,
            ff_expansion = ff_expansion,
            reduction_ratio = reduction_ratio,
            num_layers = num_layers,
            time_embed_dim=self.time_dim
        )

        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, decoder_dim, 1),
            nn.Upsample(scale_factor = 2 ** (i+2))
        ) for i, dim in enumerate(dims)])
        
        # self.to_fused_time_emb = nn.ModuleList([nn.Sequential(
        #             SiLU(),
        #             linear(
        #                 time_embed_dim,
        #                 decoder_dim,
        #             ),
        #         ) for i, dim in enumerate(dims)])

        self.to_segmentation = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 1),
            nn.Conv2d(decoder_dim, num_classes, 1),
        )

    
    def convert_to_fp16(self):
        self.input_blocks.apply(convert_module_to_f16)
        self.output_blocks_layers_up.apply(convert_module_to_f16)
        self.output_blocks_concat_back_dim.apply(convert_module_to_f16)
        
        self.norm.apply(convert_module_to_f16)
        self.norm_up.apply(convert_module_to_f16)
    
    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.output_blocks_layers_up.apply(convert_module_to_f32)
        self.output_blocks_concat_back_dim.apply(convert_module_to_f32)
        
        self.norm.apply(convert_module_to_f32)
        self.norm_up.apply(convert_module_to_f32)
    
    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype
    
    def forward(self, x,timestep):
        # print('input timestep shape ',timestep.shape)
        # print('input x shape ',x.shape)
        emb = self.time_mlp(timestep).unsqueeze(1).cuda()
        # print(' timestep  emb shape ',emb.shape)
        # h = x.type(self.inner_dtype)
        
        layer_outputs = self.mit(x,emb ,return_layer_outputs = True)
        # print('segformer forward, layer_outputs.shape ',layer_outputs.shape)
        
        fused = []
        n = 0
        for output, to_fused in zip(layer_outputs, self.to_fused):
            # print('output in layer_outputs shape: ',output.shape)
            time_layer = nn.Sequential(
                    SiLU(),
                    linear(
                        self.time_dim,
                        2**(6-n),
                    ),
                ).to('cuda')
            # emb_out = time_layer(emb).type(output.dtype)
            emb_out = time_layer(emb.cuda())
            
            while len(emb_out.shape) < len(x.shape):
                emb_out = emb_out[..., None]
            # print('in segformer forward emb_out.shape ',emb_out.shape)
            output = output + emb_out
            
            output = to_fused(output)
            fused.append(output)
            n+=1
        # fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]
        fused = torch.cat(fused, dim = 1)
        out = self.to_segmentation(fused)
        # print(out.shape)
        return out
    
