import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import downsample_block


class P2PDiscriminator(nn.Module):
    def __init__(self, img_size, in_channels=3):
        super(P2PDiscriminator, self).__init__()

        self.model = nn.Sequential(
            *downsample_block(in_channels * 2, 64, bn=False),
            *downsample_block(64, 128),
            *downsample_block(128, 256),
            *downsample_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

