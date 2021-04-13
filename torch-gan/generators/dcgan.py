import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import upsample_block



class DCGANGenerator(nn.Module):
    def __init__(self, img_size, latent_dim, channels):
        super().__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            *upsample_block(128, 128, bn=True),
            *upsample_block(128, 64, bn=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
