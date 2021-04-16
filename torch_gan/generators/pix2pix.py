import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super(UNetDownsample, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UNetUpsample, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class P2PGenerator(nn.Module):
    def __init__(self, img_size, in_channels=3, out_channels=3, dropout=0.5):
        super(P2PGenerator, self).__init__()

        self.down1 = UNetDownsample(in_channels, 64, normalize=False)
        self.down2 = UNetDownsample(64, 128)
        self.down3 = UNetDownsample(128, 256)
        self.down4 = UNetDownsample(256, 512, dropout=dropout)
        self.down5 = UNetDownsample(512, 512, dropout=dropout)
        self.down6 = UNetDownsample(512, 512, dropout=dropout)
        self.down7 = UNetDownsample(512, 512, dropout=dropout)
        self.down8 = UNetDownsample(512, 512, normalize=False, dropout=dropout)

        self.up1 = UNetUpsample(512, 512, dropout=dropout)
        self.up2 = UNetUpsample(1024, 512, dropout=dropout)
        self.up3 = UNetUpsample(1024, 512, dropout=dropout)
        self.up4 = UNetUpsample(1024, 512, dropout=dropout)
        self.up5 = UNetUpsample(1024, 256)
        self.up6 = UNetUpsample(512, 128)
        self.up7 = UNetUpsample(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)