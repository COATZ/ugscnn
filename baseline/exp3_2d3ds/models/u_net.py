from utils import initialize_weights
from torch import nn
import torch.nn.functional as F
import torch
import sys
sys.path.append("..")


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, feat=64):
        super(UNet, self).__init__()
        self.enc1 = _EncoderBlock(in_channels, feat)
        self.enc2 = _EncoderBlock(feat, feat*2)
        self.enc3 = _EncoderBlock(feat*2, feat*4)
        self.enc4 = _EncoderBlock(feat*4, feat*8, dropout=True)
        self.center = _DecoderBlock(feat*8, feat*16, feat*8)
        self.dec4 = _DecoderBlock(feat*16, feat*8, feat*4)
        self.dec3 = _DecoderBlock(feat*8, feat*4, feat*2)
        self.dec2 = _DecoderBlock(feat*4, feat*2, feat)
        self.dec1 = nn.Sequential(
            nn.Conv2d(feat*2, feat, kernel_size=3),
            nn.BatchNorm2d(feat),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat, feat, kernel_size=3),
            nn.BatchNorm2d(feat),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(feat, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)
        dec4 = self.dec4(torch.cat([center, F.upsample(enc4, center.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.upsample(enc3, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear')], 1))
        final = self.final(dec1)
        return F.upsample(final, x.size()[2:], mode='bilinear')
