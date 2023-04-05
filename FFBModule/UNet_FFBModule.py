import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(  # nn.Sequential 有序容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数。
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # inplace 是否进行覆盖运算
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


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


class UNET(nn.Module):
    def __init__(
            self, in_channels, out_channels, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()  # 将多个Module加入list，但不存在实质性顺序，参考python的list
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.ffbModule = FFBModule(features[-1]*2, features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # decoder part
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        x = self.ffbModule(x)
        skip_connections = skip_connections[::-1]

        # encoder part
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


