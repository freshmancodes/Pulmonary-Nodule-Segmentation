# 我的python
# 开发时间：2025/6/17 20:34
import torch
from torch import nn
from layer import GGCA3D, SplitAttn3D, AttendingBlock3D
from vnet_new_def import NormalizationLayer



# 残差块
class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_convs=1, norm='group', dropout=0.5):
        super().__init__()
        self.res_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.GELU()
        )
        layers = []
        current_channels = in_channels
        for i in range(num_convs):
            next_channels = out_channels if i == num_convs - 1 else mid_channels
            layers.append(nn.Conv3d(current_channels, next_channels, kernel_size=5, padding=2))
            if norm == 'group':
                layers.append(NormalizationLayer(norm_type='group'))
            else:
                layers.append(nn.BatchNorm3d(next_channels))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout3d(p=dropout))
            current_channels = next_channels
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        res = self.res_conv(x)
        out = self.conv_block(x)
        return res + out


# Attention + 上采样模块
class DecoderBlock(nn.Module):
    def __init__(self, skip_channels, up_channels, out_channels, attn_type=None):
        super().__init__()
        self.attend = AttendingBlock3D(skip_channels, up_channels)
        self.attn_type = attn_type
        self.concat_channels = skip_channels + up_channels

        if attn_type == 'split':
            self.attn = SplitAttn3D(self.concat_channels)
        else:
            self.attn = nn.Identity()

        self.res_conv = ResidualConvBlock(
            in_channels=self.concat_channels,
            mid_channels=out_channels,
            out_channels=out_channels,
            num_convs=3
        )

    def forward(self, skip_feat, up_feat):
        attn_feat = self.attend(skip_feat, up_feat)
        x = torch.cat([attn_feat, up_feat], dim=1)
        x = self.attn(x)
        return self.res_conv(x)


# 编码器块 + GGCA
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, num_convs=1, ggca=None):
        super().__init__()
        self.res_conv = ResidualConvBlock(in_ch, mid_ch, out_ch, num_convs)
        self.ggca = ggca
        self.down = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        x = self.res_conv(x)
        x_skip = x
        if self.ggca:
            x = self.ggca(x)
        x = self.down(x)
        return x, x_skip


class VNet(nn.Module):
    def __init__(self, in_channel=1, num_classes=2, width=16):
        super().__init__()

        self.enc1 = EncoderBlock(in_channel, width, width)
        self.enc2 = EncoderBlock(width, width, width * 2, num_convs=2,
                                 ggca=GGCA3D(width * 2, d=8, h=48, w=48, reduction=8, num_groups=4))
        self.enc3 = EncoderBlock(width * 2, width * 2, width * 4, num_convs=3,
                                 ggca=GGCA3D(width * 4, d=4, h=24, w=24, reduction=16, num_groups=4))
        self.enc4 = EncoderBlock(width * 4, width * 4, width * 8, num_convs=3,
                                 ggca=GGCA3D(width * 8, d=2, h=12, w=12, reduction=16, num_groups=4))

        self.bottom = ResidualConvBlock(width * 8, width * 8, width * 16, num_convs=3)

        self.up4 = nn.ConvTranspose3d(width * 16, width * 16, kernel_size=2, stride=2)
        self.dec4 = DecoderBlock(width * 8, width * 16, width * 16, attn_type='split')

        self.up3 = nn.ConvTranspose3d(width * 16, width * 16, kernel_size=2, stride=2)
        self.dec3 = DecoderBlock(width * 4, width * 16, width * 8)

        self.up2 = nn.ConvTranspose3d(width * 8, width * 8, kernel_size=2, stride=2)
        self.dec2 = DecoderBlock(width * 2, width * 8, width * 4)

        self.up1 = nn.ConvTranspose3d(width * 4, width * 4, kernel_size=2, stride=2)
        self.dec1 = DecoderBlock(width, width * 4, width * 2)

        self.out_conv = nn.Sequential(
            nn.Conv3d(width * 2, num_classes, kernel_size=1),
            nn.BatchNorm3d(num_classes),
            nn.GELU(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x1, skip1 = self.enc1(x)
        x2, skip2 = self.enc2(x1)
        x3, skip3 = self.enc3(x2)
        x4, skip4 = self.enc4(x3)

        b = self.bottom(x4)

        x = self.up4(b)
        x = self.dec4(skip4, x)

        x = self.up3(x)
        x = self.dec3(skip3, x)

        x = self.up2(x)
        x = self.dec2(skip2, x)

        x = self.up1(x)
        x = self.dec1(skip1, x)

        return self.out_conv(x)
