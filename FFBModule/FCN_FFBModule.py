import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

######################################################################################
# FCN: Fully Convolutional Networks for Semantic Segmentation
# Paper-Link: https://arxiv.org/abs/1411.4038
######################################################################################

__all__ = ["FCN"]

class FFBModule(nn.Module):
    # 每个stage维度中扩展的倍数
    extention = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(FFBModule, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, groups=4, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=2, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, groups=4, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.conv4 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=2, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)

        self.conv5 = nn.Conv2d(planes, planes , kernel_size=1, stride=1, groups=4, bias=False)
        self.bn5 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

        # 判断残差有没有卷积
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # 参差数据
        # residual = x
        # residual1=x

        # 卷积操作
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        residual1 = out

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        residual2 = out

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = out + residual1

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)

        out = out + residual2

        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu(out)

        # 是否直连（如果Indentity blobk就是直连；如果Conv2 Block就需要对残差边就行卷积，改变通道数和size
        if self.downsample is not None:
            residual = self.downsample(x)

        # 将残差部分和卷积部分相加
        # out = out + residual
        # out=out+residual1+residual2
        #out = self.relu(out)

        return out

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class conv3x3_block_x1(nn.Module):
    '''(conv => BN => ReLU) * 1'''

    def __init__(self, in_ch, out_ch):
        super(conv3x3_block_x1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv3x3_block_x2(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(conv3x3_block_x2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv3x3_block_x3(nn.Module):
    '''(conv => BN => ReLU) * 3'''

    def __init__(self, in_ch, out_ch):
        super(conv3x3_block_x3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class upsample(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor=2):
        super(upsample, self).__init__()
        self.conv1x1 = conv1x1(in_ch, out_ch)
        self.scale_factor = scale_factor

    def forward(self, H):
        """
        H: High level feature map, upsample
        """
        H = self.conv1x1(H)
        H = F.interpolate(H, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        return H


class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.block1 = conv3x3_block_x2(3, 64)
        self.block2 = conv3x3_block_x2(64, 128)
        self.block3 = conv3x3_block_x3(128, 256)
        self.block4 = conv3x3_block_x3(256, 512)
        self.bottleneck=FFBModule(inplanes=512,planes=512)
        self.block5 = conv3x3_block_x3(512, 512)
        self.upsample1 = upsample(512, 512, 2)
        self.upsample2 = upsample(512, 256, 2)
        self.upsample3 = upsample(256, num_classes, 8)

    def forward(self, x):
        block1_x = self.block1(x)
        block2_x = self.block2(block1_x)
        block3_x = self.block3(block2_x)
        block4_x = self.block4(block3_x)
        block5_x = self.block5(block4_x)
        block5_x=self.bottleneck(block5_x)
        # block5_x=self.maxpool(block5_x)
        upsample1 = self.upsample1(block5_x)
        x = torch.add(upsample1, block4_x)
        upsample2 = self.upsample2(x)
        x = torch.add(upsample2, block3_x)
        x = self.upsample3(x)

        return x


if __name__ == '__main__':
    inputs = torch.randn((4, 3, 224, 224))
    model = FCN(num_classes=1)
    y = model(inputs)
    print(y.shape)