
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..utils import upsample_block

class InfoGANGenerator(nn.Module):
    def __init__(self, img_size, latent_dim, code_dim, channels, n_classes):
        super(InfoGANGenerator, self).__init__()
        input_dim = latent_dim + n_classes + code_dim

        self.init_size = img_size // 4  # Initial size before upsampling
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            *upsample_block(128, 128, bn=True),
            *upsample_block(128, 64, bn=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, self.label_emb(labels), code), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img