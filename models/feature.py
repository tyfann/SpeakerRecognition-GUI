#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from models.ResNetBlocks import *

class ResNetSE(nn.Module):
    def __init__(self, block, layers, num_filters, nOut, encoder_type='SAP', n_mels=40, log_input=True, **kwargs):
        super(ResNetSE, self).__init__()

        print('Embedding size is %d, encoder %s.'%(nOut, encoder_type))
        self.inplanes   = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels     = n_mels
        self.log_input  = log_input
        self.instancenorm   = nn.InstanceNorm1d(n_mels)
        self.torchfb        = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, window_fn=torch.hamming_window, n_mels=n_mels)


    def forward(self, x):

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfb(x)+1e-6
                if self.log_input: x = x.log()
                x = self.instancenorm(x).unsqueeze(1).detach()
        return x


def MainModel(nOut=256, **kwargs):
    # Number of filters
    num_filters = [16, 32, 32, 32]
    model = ResNetSE(SEBasicBlock, [2, 2, 2, 2], num_filters, nOut, **kwargs)
    return model
