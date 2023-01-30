import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchvision import models as resnet_model
DEVICE = "cuda:1"

"""Dual-Decoding Branch U-shaped Net (DBUNet)"""
"""Proposed By Dr. Wang Yuefei"""
"""CHENGDU UNIVERSITY"""
"""2023.1.30"""




# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(64, 12, bias=False),  #针对224的数据集
            nn.ReLU(inplace=True),
            nn.Linear(12, 64, bias=False),
            nn.Sigmoid()
        )
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d(1)
        self.PatchConv_stride1 = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        self.PatchConv_stride1_bn = nn.BatchNorm2d(1)
        self.PatchConv_stride1_rl = nn.ReLU(inplace=True)

        self.layers = nn.ModuleList([])
        # for _ in range(depth):
        self.layers.append(nn.ModuleList([
            PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
            PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, tokens, x_AfterPatchEmbedding):
        x_AfterConv = torch.zeros([1, 1, 64, 196])  #新建一个空的tensor，用来存储前64个patches
        for i in range(64):  #遍历每一行（每个）
            x_patch_i = x_AfterPatchEmbedding[0][i][:]  #取出这一行的值
            x_patch_i = x_patch_i.view(1, 1, 1, 196)
            x_patch_i = self.PatchConv_stride1(x_patch_i)
            x_patch_i = self.PatchConv_stride1_bn(x_patch_i)
            x_patch_i = self.PatchConv_stride1_rl(x_patch_i)
            x_AfterConv[0][0][i][:] = x_patch_i[0][0][0][:]  #赋值给新建的tensor x_AfterConv.shape=[1, 1, 64, 196]
        a = x_AfterConv.view(1, 64, 1, 196)
        a = a.to(DEVICE)
        b1 = nn.AdaptiveAvgPool2d(1)(a).view(1, 64) #avgpool,压缩出[1, 64]大小的tensor
        b1 = b1.to(DEVICE)

        val, idx = torch.sort(b1)  # 升序排序
        if torch.max(val) < 0 or torch.min(val) > 0: #如果所有值都 < 0,或者所有值都 > 0,
            middle = 32
        else:
            for i in range(64):
                if val[0][i] > 0:
                    middle = i  # 找到中间值（负正交界）
                    break
                if i == 63:
                    middle = 32
        suppress = idx[0][:middle]
        excitation = idx[0][middle:]
        l_s = len(suppress)
        l_e = len(excitation)
        b1[0][suppress] = b1[0][suppress] - 1 / (1 + l_s ** (b1[0][suppress]))  # 反sigmoid，底是长度
        b1[0][excitation] = b1[0][excitation] + 1 / (1 + l_e ** -(b1[0][excitation]))  # sigmoid，底是长度
        b = b1


        c = self.fc(b).view(1, 64, 1, 1) #前面的(1, 64)是b的size
        x_attention = a * c.expand_as(a)
        x_attention = x_attention.view(1, 64, 196)  #[1, 64, 1, 196] -> [1, 64, 196]
        token = tokens.view(1, 1, 196)
        x_attention = torch.cat([x_attention, token], dim=1)  #[1, 65, 196]

        for attn, ff in self.layers:
            x = attn(x) + x + x_attention
            x = ff(x) + x
        return x



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

        if self.downsample is not None:
            residual = self.downsample(x)

        return out


class encoder(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.channelTrans = nn.Conv2d(in_channels=65, out_channels=512, kernel_size=1, padding=0)

    def forward(self, x):
        x_vit = x
        x_vit = self.to_patch_embedding(x_vit)
        x_AfterPatchEmbedding = x_vit
        b, n, _ = x_vit.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x_vit = torch.cat((cls_tokens, x_vit), dim=1)
        x_vit += self.pos_embedding[:, :(n + 1)]
        x_vit = self.dropout(x_vit)

        vit_layerInfo = []
        for i in range(4):  # 设置深度的地方[6, 64+1, dim=196]
            x_vit = self.transformer(x_vit, cls_tokens[i], x_AfterPatchEmbedding)
            vit_layerInfo.append(x_vit)


        return vit_layerInfo


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DBUNet(nn.Module):
    def __init__(self, features=[64, 128, 256, 512], out_channels=1):
        super(DBUNet, self).__init__()
        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.encoder = encoder(image_size=224, patch_size=28, num_classes=2, dim=196, depth=6, heads=16, mlp_dim=2048)
        self.conv = nn.Conv2d(1024, 512, 1)
        self.bottleneck = Bottleneck(512, 1024)
        self.ups = nn.ModuleList()
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2 + 65 + feature, feature * 2, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))
        self.finalconv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self.vitLayer_UpConv = nn.ConvTranspose2d(65, 65, kernel_size=2, stride=2)

        self.final_conv1 = nn.ConvTranspose2d(64, 32, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        vit_layerInfo = self.encoder(x)

        x = self.bottleneck(e4)

        vit_layerInfo = vit_layerInfo[::-1]  # 翻转，呈正序。0表示第四层...3表示第一层

        v = vit_layerInfo[0].view(4, 65, 14, 14)
        x = torch.cat([x, e4, v], dim=1)
        x = self.ups[0](x)
        x = self.ups[1](x)

        v = vit_layerInfo[1].view(4, 65, 14, 14)
        v = self.vitLayer_UpConv(v)
        x = torch.cat([x, e3, v], dim=1)
        x = self.ups[2](x)
        x = self.ups[3](x)

        v = vit_layerInfo[2].view(4, 65, 14, 14)
        v = self.vitLayer_UpConv(v)
        v = self.vitLayer_UpConv(v)
        x = torch.cat([x, e2, v], dim=1)
        x = self.ups[4](x)
        x = self.ups[5](x)

        v = vit_layerInfo[3].view(4, 65, 14, 14)
        v = self.vitLayer_UpConv(v)
        v = self.vitLayer_UpConv(v)
        v = self.vitLayer_UpConv(v)
        x = torch.cat([x, e1, v], dim=1)
        x = self.ups[6](x)
        x = self.ups[7](x)

        out1 = self.final_conv1(x)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)
        return out


x = torch.randn(4, 3, 224, 224)
model = DBUNet()
preds = model(x)
print(x.shape)
print(preds.shape)
print("DBUNet")
