from __future__ import absolute_import, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义DeepLabV1的网络结构
class DeepLabV1(nn.Sequential):
    """
    DeepLab v1: Dilated ResNet + 1x1 Conv
    Note that this is just a container for loading the pretrained COCO model and not mentioned as "v1" in papers.
    """

    def __init__(self, n_classes, n_blocks,):
        super(DeepLabV1, self).__init__()
        ch = [64 * 2 ** p for p in range(6)]
        self.convs = nn.ModuleList([
        _Stem(ch[0]),
        _ResLayer( n_blocks[0], ch[0], ch[2], 1, 1 ),
        _ResLayer( n_blocks[1], ch[2], ch[3], 2, 1 ),
        _ResLayer( n_blocks[2], ch[3], ch[4], 1, 2 ),
        _ResLayer( n_blocks[3], ch[4], ch[5], 1, 4 ),
        nn.Conv2d( 2048, n_classes, 1 ),
                                        ]
        )


    def forward(self,x):


        h = x.size()[2]
        w = x.size()[3]
        for lay in self.convs:
            x=lay(x)

        x = F.interpolate( x, size=(h, w), mode="bilinear" )  # (shape: (batch_size, num_classes, h, w))
        return x
# 这里是看一下是使用torch的nn模块中BatchNorm还是在encoding文件中定义的BatchNorm




_BATCH_NORM = nn.BatchNorm2d

_BOTTLENECK_EXPANSION = 4

# 定义卷积+BN+ReLU的组件
class _ConvBnReLU(nn.Sequential):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """

    BATCH_NORM = _BATCH_NORM

    def __init__(
            self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", _BATCH_NORM(out_ch, eps=1e-5, momentum=0.999))

        if relu:
            self.add_module("relu", nn.ReLU())


# 定义Bottleneck，先1*1卷积降维，然后使用3*3卷积，最后再1*1卷积升维，然后再shortcut连接。
# 降维到多少是由_BOTTLENECK_EXPANSION参数决定的，这是ResNet的Bottleneck。
class _Bottleneck(nn.Module):
    """
    Bottleneck block of MSRA ResNet.
    """

    def __init__(self, in_ch, out_ch, stride, dilation, downsample):
        super(_Bottleneck, self).__init__()
        mid_ch = out_ch // _BOTTLENECK_EXPANSION
        self.reduce = _ConvBnReLU(in_ch, mid_ch, 1, stride, 0, 1, True)
        self.conv3x3 = _ConvBnReLU(mid_ch, mid_ch, 3, 1, dilation, dilation, True)
        self.increase = _ConvBnReLU(mid_ch, out_ch, 1, 1, 0, 1, False)
        self.shortcut = (
            _ConvBnReLU(in_ch, out_ch, 1, stride, 0, 1, False)
            if downsample
            else lambda x: x  # identity
        )

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h)

# 定义ResLayer，整个DeepLabv1是用ResLayer堆叠起来的，下采样是在每个ResLayer的第一个
# Bottleneck发生的。
class _ResLayer(nn.Sequential):
    """
    Residual layer with multi grids
    """

    def __init__(self, n_layers, in_ch, out_ch, stride, dilation, multi_grids=None):
        super(_ResLayer, self).__init__()

        if multi_grids is None:
            multi_grids = [1 for _ in range(n_layers)]
        else:
            assert n_layers == len(multi_grids)

        # Downsampling is only in the first block
        for i in range(n_layers):
            self.add_module(
                "block{}".format(i + 1),
                _Bottleneck(
                    in_ch=(in_ch if i == 0 else out_ch),
                    out_ch=out_ch,
                    stride=(stride if i == 0 else 1),
                    dilation=dilation * multi_grids[i],
                    downsample=(True if i == 0 else False),
                ),
            )

# 在进入ResLayer之前，先用7*7的卷积核在原图滑动，增大感受野。padding方式设为same，大小不变。
# Pool层的核大小为3，步长为2，这会导致特征图的分辨率发生变化。
class _Stem(nn.Sequential):
    """
    The 1st conv layer.
    Note that the max pooling is different from both MSRA and FAIR ResNet.
    """

    def __init__(self, out_ch):
        super(_Stem, self).__init__()
        self.add_module("conv1", _ConvBnReLU(3, out_ch, 7, 2, 3, 1))
        self.add_module("pool", nn.MaxPool2d(3, 2, 1, ceil_mode=True))

# 相当于Reshape，网络并没有用到
class _Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# 主函数，输出构建的DeepLab V1模型的结构还有原始图像分辨率和结果图像的分辨率
if __name__ == "__main__":
    #model.eval()
    x = torch.randn(1, 3, 513, 513)
    h = x.size()[2]
    w = x.size()[3]
    model = DeepLabV1(n_classes=1, n_blocks=[3, 4, 23, 3])


    print(model)
    print("input:", x.shape)
    #输入图像与输出图像大小不匹配
    print("output:", model(x).shape)