## TODO: This file is the full architecture and structure of the model 
# use for the purpose this this project

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn.utils import spectral_norm
from torch.nn.utils import weight_norm
from torch.nn.utils import remove_weight_norm
from torch.nn import init
import math
import os
import sys
import random
import copy
import time
import datetime
from collections import OrderedDict


#########################
###     VAE-GAN       ###
#########################

##  --------------------------------------- SUB-COMPONENTS ----------------------------------------------------
# Block components of each composition of the model architecture

#-------------- Residual Block --------------#
class ResidualBlock(nn.Module):
    """
    Residual Block with instance normalization
    TODO: Add explanation of what a residual block used for
    """
    def __init__(self, features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class EncoderBlock(nn.Module):
    """
    Encoder Block with instance normalization and ReLU activation
    TODO: Add explanation of what an encoder block used for, configurations and parameters
    """
    def __init__(self, in_channels, out_channels, downsample=True):
        super(EncoderBlock, self).__init__()
        self.downsample = downsample

        # Convolutional layer
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=4, stride=2, padding=1)
        # self.batchnorm = nn.BatchNorm2d(num_features=out_channels, momentum=0.8)

        # Normalization: 
        self.norm = nn.InstanceNorm2d(num_features=out_channels) # using Instance Normalization

        # Activation function: ReLU / Leaky ReLU / Sigmoid / Tanh ...
        # self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.downsample:
            x = self.norm(x)
            x = self.activation(x)
        return x

class DecoderBlock(nn.Module):
    """
    Decoder Block with instance normalization and Leakly ReLU activation
    """
    def __init__(self, in_channels, out_channels, upsample=True):
        super(DecoderBlock, self).__init__()
        self.upsample = upsample

        # Convolutional layer
        self.conv = nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size=4, stride=2, padding=1)
        # Normalization:
        # self.batchnorm = nn.BatchNorm2d(num_features=out_channels, momentum=0.8)
        self.norm = nn.InstanceNorm2d(num_features=out_channels) # using Instance Normalization instead of Batch Normalization

        # Activation function: ReLU / Leaky ReLU / Sigmoid / Tanh ...
        # self.activation = nn.ReLU(inplace=True)
        self.activation = nn.LeakyReLU(0.2, inplace=True) # using Leaky ReLU for Decoder

    def forward(self, x):
        x = self.conv(x)
        if self.upsample:
            x = self.norm(x)
            x = self.activation
        return x

#################---------------------------------------------END OF SUB-COMPONENTS------------------------------------#################

#########################
###   VAE COMPONENT   ###
#########################


#- ENCODER
class Encoder(nn.Module):
    def __init__(self, in_channels=3, dim=64, n_downsample=2):
        super(Encoder, self).__init__()
        self.size = in_channels #number of channels in encoder input

        # Initial convolution block
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, dim, 7),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        #ADD: Downsampling Layers
        for _ in range(n_downsample):
            # layers += [
            #     nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1),
            #     nn.InstanceNorm2d(dim * 2),
            #     nn.ReLU(inplace=True),
            # ]
            layers += [EncoderBlock(dim, dim * 2)]
            dim *= 2

        #ADD: Residual blocks
        for _ in range(4):
            layers += [ResidualBlock(dim)]

        self.model_blocks = nn.Sequential(*layers)

    def reparameterization(self, mu):
        Tensor = torch.cuda.FloatTensor if mu.is_cuda else torch.FloatTensor
        z = Variable(Tensor(np.random.normal(0, 1, mu.shape)))
        return z + mu

    def forward(self, x):
        mu = self.model_blocks(x)
        z = self.reparameterization(mu)
        return mu, z

#-DECODER
class Decoder(nn.Module):
    def __init__(self, out_channels=3, dim=64, n_upsample=2, shared_block=None):
        super(Decoder, self).__init__()

        self.shared_block = shared_block

        # Residual blocks
        # layers = []
        for _ in range(3):
            layers += [ResidualBlock(dim)]

        # Upsampling
        layers = []
        dim = dim * 2 ** n_upsample
        for _ in range(n_upsample):
            layers += [DecoderBlock(dim, dim // 2)]
            dim //= 2

        # Output layer
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=7), nn.Tanh()]

        # self.model_blocks = nn.Sequential(*self.model_blocks)
        # self.model = nn.Sequential(*layers)

        self.model_blocks = nn.Sequential(*layers)

    def forward(self, x):
        x = self.shared_block(x)
        x = self.model_blocks(x)
        return x

#---------------------------------------------------------------------------------

##################################
###   DISCRIMINATOR (GAN)      ###
##################################
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        channels, height, width = input_shape
        # Calculate output of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # a series of layers upsample the image to the desired size
        # these layers are the combined layers of oringimal cnv layers with/without normalization

        # the final layer is a convolutional layer that produce the final output
        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 3, padding=1)
        )

    def forward(self, img):
        return self.model(img)