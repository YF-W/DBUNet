import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    # 每个stage维度中扩展的倍数
    extention = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

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

class batchnorm_relu(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.bn(inputs)
        x = self.relu(x)
        return x


class residual_block(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        """ Convolutional layer """
        self.b1 = batchnorm_relu(in_c)
        self.c1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride)
        self.b2 = batchnorm_relu(out_c)
        self.c2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=1)

        """ Shortcut Connection (Identity Mapping) """
        self.s = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, stride=stride)

    def forward(self, inputs):
        x = self.b1(inputs)
        x = self.c1(x)
        x = self.b2(x)
        x = self.c2(x)
        s = self.s(inputs)

        skip = x + s
        return skip


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r = residual_block(in_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = torch.cat((x, skip), dim=1)
        x = self.r(x)
        return x


class build_resunet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder 1 """
        self.c11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.br1 = batchnorm_relu(64)
        self.c12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.c13 = nn.Conv2d(3, 64, kernel_size=1, padding=0)

        """ Encoder 2 and 3 """
        self.r2 = residual_block(64, 128, stride=2)
        self.r3 = residual_block(128, 256, stride=2)

        """ Bridge """
        self.r4 = residual_block(256, 512, stride=2)
        self.bottleneck=Bottleneck(inplanes=512,planes=512)
        # self.conv=nn.Conv2d(512,512,1,2)
        """ Decoder """
        self.d1 = decoder_block(512, 256)
        self.d2 = decoder_block(256, 128)
        self.d3 = decoder_block(128, 64)

        """ Output """
        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.sigmoid = torch.sigmoid

    def forward(self, inputs):
        """ Encoder 1 """
        x = self.c11(inputs)
        x = self.br1(x)
        x = self.c12(x)
        s = self.c13(inputs)
        skip1 = x + s

        """ Encoder 2 and 3 """
        skip2 = self.r2(skip1)
        skip3 = self.r3(skip2)

        """ Bridge """
        b = self.r4(skip3)
        b=self.bottleneck(b)
        # b=self.conv(b)
        """ Decoder """

        d1 = self.d1(b, skip3)
        d2 = self.d2(d1, skip2)
        d3 = self.d3(d2, skip1)

        """ output """
        output = self.output(d3)
        # output = self.sigmoid(output)

        return output


if __name__ == "__main__":
    inputs = torch.randn((4, 3, 224, 224))
    model = build_resunet()
    y = model(inputs)
    print(y.shape)
