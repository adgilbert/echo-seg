import torch
from torch import nn


class UNet2(nn.Module):
    def __init__(self, in_channels=1, n_classes=4):
        """
        PyTorch implementation of Unet1 from  Deep Learning for Segmentation Using an Open Large-Scale Dataset in 2D
        Echocardiography

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
        """
        super(UNet2, self).__init__()

        self.d1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )
        self.d2 = nn.Sequential(
            nn.MaxPool2d(2),  # move here because connection comes before pooling
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )
        self.d3 = nn.Sequential(
            nn.MaxPool2d(2),  # move here because connection comes before pooling
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
        )
        self.d4 = nn.Sequential(
            nn.MaxPool2d(2),  # move here because connection comes before pooling
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )
        self.d5 = nn.Sequential(
            nn.MaxPool2d(2),  # move here because connection comes before pooling
            nn.Conv2d(384, 768, kernel_size=3, padding=1),
            nn.BatchNorm2d(768),
            nn.ReLU(),
            nn.Conv2d(768, 768, kernel_size=3, padding=1),
            nn.BatchNorm2d(768),
            nn.ReLU(),
            nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2),
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )
        self.u1 = nn.Sequential(
            nn.Conv2d(768, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2),
            nn.BatchNorm2d(192),
            nn.ReLU(),
        )
        self.u2 = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )
        self.u3 = nn.Sequential(
            nn.Conv2d(192, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )
        self.u4 = nn.Sequential(
            nn.Conv2d(96, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )
        self.seg = nn.Conv2d(48, n_classes, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> dict:
        x1 = self.d1(x)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)
        x = self.d5(x4)
        x = self.u1(torch.cat([x4, x], 1))
        x = self.u2(torch.cat([x3, x], 1))
        x = self.u3(torch.cat([x2, x], 1))
        x = self.u4(torch.cat([x1, x], 1))
        seg = self.seg(x)
        return dict(segs=seg)
