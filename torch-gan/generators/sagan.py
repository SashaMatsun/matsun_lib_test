import torch
from torch.optim.optimizer import Optimizer, required

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter
import numpy as np

from ..discriminators.sagan import SpectralNorm, SelfAttn

class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GenBlock, self).__init__()
        self.block = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)),
            
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.block(x)

class SAGANGenerator(nn.Module):
    def __init__(self, image_size, latent_dim=128, channels=3):
        super(SAGANGenerator, self).__init__()
        
        coef_pow = int(np.log(image_size // 4))

        self.l0 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(latent_dim, 64, 4)),
            
            nn.LeakyReLU(0.1)
        ) #  4 * 4

        layers = []
        for i in range(coef_pow):
            layers.append(GenBlock(64, 64))
            if (i == coef_pow - 2) or (i == coef_pow - 1):
                layers.append(SelfAttn(64, 'relu'))
        self.model_body = nn.Sequential(*layers)

        self.to_RGB = nn.Sequential(
            nn.ConvTranspose2d(64, channels, 1),
            nn.Tanh()
        )

    def forward(self, x):
        y = self.l0(x)
        y = self.model_body(y)
        y = self.to_RGB(y)

        return y