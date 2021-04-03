import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init_normal(m):
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def downsample_block(in_channels, out_channels, bn=True, dropout=0.0):
    block = [nn.Conv2d(in_channels, out_channels, 4, 2, 1), 
             nn.LeakyReLU(0.2, inplace=True), 
             nn.Dropout2d(dropout)]
    if bn:
        block.append(nn.BatchNorm2d(out_channels, 0.1))
    return block

def upsample_block(in_channels, out_channels, bn=True):
    block = [nn.Upsample(scale_factor=2),
             nn.Conv2d(in_channels, out_channels, 3, 1, 1), 
             nn.LeakyReLU(0.2, inplace=True)]
    if bn:
        block.append(nn.BatchNorm2d(out_channels, 0.1))
    return block