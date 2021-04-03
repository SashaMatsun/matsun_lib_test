import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import downsample_block


class CGANDiscriminator(nn.Module):
    def __init__(self, img_size, channels, n_classes, dropout=0.25):
        super().__init__()

        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.label_l = nn.Linear(n_classes, img_size ** 2)

        self.model = nn.Sequential(
            *downsample_block(channels + 1, 16, bn=False, dropout=dropout),
            *downsample_block(16, 32, dropout=dropout),
            *downsample_block(32, 64, dropout=dropout),
            *downsample_block(64, 128, dropout=dropout),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img, label):
        l_in = self.label_emb(label)
        l_in = self.label_l(l_in)
        l_in = l_in.view((l_in.size[0], 1, img.size[2], img.size[3]))
        img = torch.cat((img, l_in), 1)
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity