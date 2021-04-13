import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import downsample_block


class SGANDiscriminator(nn.Module):
    def __init__(self, img_size, channels, n_classes, dropout=0.25):
        super(SGANDiscriminator, self).__init__()

        self.conv_blocks = nn.Sequential(
            *downsample_block(channels, 16, bn=False, dropout=dropout),
            *downsample_block(16, 32, dropout=dropout),
            *downsample_block(32, 64, dropout=dropout),
            *downsample_block(64, 128, dropout=dropout),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, n_classes + 1), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label