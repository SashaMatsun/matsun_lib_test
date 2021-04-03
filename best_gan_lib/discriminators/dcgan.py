import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import downsample_block


class DCGANDiscriminator(nn.Module):
    def __init__(self, img_size, channels, dropout=0.25):
        super().__init__()

        self.model = nn.Sequential(
            *downsample_block(channels, 16, bn=False, dropout=dropout),
            *downsample_block(16, 32, dropout=dropout),
            *downsample_block(32, 64, dropout=dropout),
            *downsample_block(64, 128, dropout=dropout),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity
