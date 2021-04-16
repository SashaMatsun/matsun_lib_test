import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class PixelwiseNorm(torch.nn.Module):
    """
    ------------------------------------------------------------------------------------
    Pixelwise feature vector normalization.
    reference:
    https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120
    ------------------------------------------------------------------------------------
    """

    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    @staticmethod
    def forward(x, alpha=1e-8):
        y = x.pow(2.0).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y


class GenInitBlock(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super(GenInitBlock, self).__init__()

        self.latent_dim = latent_dim

        self.block = nn.Sequential(
            PixelwiseNorm(),
            nn.ConvTranspose2d(self.latent_dim, out_channels, (4, 4), bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, (3, 3), padding=1, bias=True),
            nn.LeakyReLU(0.2),
            PixelwiseNorm()
        )

    def forward(self, x):
        y = x.view([x.size()[0], self.latent_dim, 1, 1])
        y = self.block(y)
        return y


class GenGeneralBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GenGeneralBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3,3), padding=1, bias=True),
            nn.LeakyReLU(0.2),
            PixelwiseNorm(),
            nn.Conv2d(out_channels, out_channels, (3,3), padding=1, bias=True),
            nn.LeakyReLU(0.2),
            PixelwiseNorm(),
        )

    def forward(self, x):
        y = F.interpolate(x, scale_factor=2)
        y = self.block(y)
        return y


class ProGANGenerator(nn.Module):
    def __init__(self, latent_dim, fin_size, channels=3):
        super(ProGANGenerator, self).__init__()
        self.depth = int(np.log2(fin_size / 4))
        
        self.init_block = GenInitBlock(latent_dim, 64)
        self.general_blocks = [GenGeneralBlock(64, 64) for i in range(self.depth)]

        self.to_RGB_list = [nn.Conv2d(64, 3, (1,1)) for i in range(self.depth + 1)]

    def forward(self, x, curr_size, alpha):
        y = self.init_block(x)

        curr_depth = int(np.log2(curr_size / 4))

        for i in range(curr_depth):
            y_res = y
            y = self.general_blocks[i](y)

        if curr_depth == 0:
            y = self.to_RGB_list[curr_depth](y)
        else:
            y = self.to_RGB_list[curr_depth](y)
            y_res = self.to_RGB_list[curr_depth - 1](y_res)
            y_res = F.interpolate(y_res, scale_factor=2)
            y = y * alpha + y_res * (1.0 - alpha)

        return y