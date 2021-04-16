import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class DisFinBlock(nn.Module):
    def __init__(self, in_channels):
        super(DisFinBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (3,3), padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, in_channels, (4,4)),
            nn.LeakyReLU(0.2)
        )
        self.l = nn.Sequential(
            nn.Linear(in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.block(x)
        y = y.squeeze()
        y = self.l(y)
        return y

    
class DisGeneralBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DisGeneralBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (4,4), 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, out_channels, (3,3), padding=1),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        y = self.block(x)
        return y


class ProGANDiscriminator(nn.Module):
    def __init__(self, img_size, channels=3):
        super(ProGANDiscriminator, self).__init__()

        self.depth = int(np.log2(img_size / 4))

        self.from_RGB_list = [nn.Conv2d(3, 64, (1,1)) for i in range(self.depth + 1)]
        self.general_blocks = [DisGeneralBlock(64, 64) for i in range(self.depth)]
        self.fin_block = DisFinBlock(64)

    def forward(self, x, curr_size, alpha):
        curr_depth = int(np.log2(curr_size / 4))
        
        if curr_depth == 0:
            y = self.from_RGB_list[0](x)
        elif alpha > 0.0:
            #y_res = F.avg_pool2d(x, kernel_size=2, stride=2)
            y_res = F.interpolate(x, scale_factor=1/2, mode='bilinear')
            y_res = self.from_RGB_list[curr_depth - 1](y_res)

            y_str = self.from_RGB_list[curr_depth](x)
            y_str = self.general_blocks[curr_depth - 1](y_str)

            y = y_str * alpha + y_res * (1.0 - alpha)
        else:
            y = self.from_RGB_list[curr_depth](x)
            y = self.general_blocks[curr_depth - 1](y)

        for i in range(curr_depth - 2, -1, -1):
            y = self.general_blocks[i](y)

        y = self.fin_block(y)
        return y
