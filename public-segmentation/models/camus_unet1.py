import torch
from torch import nn


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class UNet1(nn.Module):
    def __init__(self, in_channels=1, n_classes=4, batchnorm=True, mode="bilinear"):
        """
        PyTorch implementation of Unet1 from  Deep Learning for Segmentation Using an Open Large-Scale Dataset in 2D
        Echocardiography

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
        """
        super(UNet1, self).__init__()

        self.d1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32) if batchnorm else Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32) if batchnorm else Identity(),
            nn.ReLU(inplace=True),
        )
        self.d2 = nn.Sequential(
            nn.MaxPool2d(2),  # move here because connection comes before pooling
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32) if batchnorm else Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32) if batchnorm else Identity(),
            nn.ReLU(inplace=True),
        )
        self.d3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if batchnorm else Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if batchnorm else Identity(),
            nn.ReLU(inplace=True),
        )
        self.d4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if batchnorm else Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if batchnorm else Identity(),
            nn.ReLU(inplace=True),
        )
        self.d5 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if batchnorm else Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if batchnorm else Identity(),
            nn.ReLU(inplace=True),
        )
        self.d6 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if batchnorm else Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if batchnorm else Identity(),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode=mode),
        )
        self.u1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if batchnorm else Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if batchnorm else Identity(),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode=mode),
        )
        self.u2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if batchnorm else Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if batchnorm else Identity(),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode=mode),
        )
        self.u3 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if batchnorm else Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if batchnorm else Identity(),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode=mode),
        )
        self.u4 = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32) if batchnorm else Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32) if batchnorm else Identity(),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode=mode),
        )
        self.u5 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16) if batchnorm else Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16) if batchnorm else Identity(),
            nn.ReLU(inplace=True),
        )
        self.seg = nn.Sequential(
            nn.Conv2d(16, n_classes, kernel_size=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)
        x5 = self.d5(x4)
        x = self.d6(x5)
        x = self.u1(torch.cat([x5, x], 1))
        x = self.u2(torch.cat([x4, x], 1))
        x = self.u3(torch.cat([x3, x], 1))
        x = self.u4(torch.cat([x2, x], 1))
        x = self.u5(torch.cat([x1, x], 1))
        seg = self.seg(x)
        return dict(segs=seg)
