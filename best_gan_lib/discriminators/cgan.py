import torch
import torch.nn as nn
import torch.nn.functional as F


class CGANDiscriminator(nn.Module):
    def __init__(self, img_size, channels, n_classes, dropout=0.25):
        super().__init__()

        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.label_l = nn.Linear(n_classes, img_size ** 2)

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(dropout)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels + 1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img, label):
        l_in = self.label_emb(label)
        l_in = self.label_l(l_in)
        l_in = l_in.view((l_in.size[0], 1, img_size, img_size))
        img = torch.cat((img, l_in), 1)
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity