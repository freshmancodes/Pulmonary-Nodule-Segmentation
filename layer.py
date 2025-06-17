import torch
from torch import nn
import torch.nn.functional as F


class GGCA3D(nn.Module):  # Global Grouped Coordinate Attention for 3D
    def __init__(self, channel, d, h, w, reduction=16, num_groups=4):
        super(GGCA3D, self).__init__()
        self.num_groups = num_groups  # 分组数
        self.group_channels = channel // num_groups  # 每组的通道数
        self.d = d  # 深度方向的特定尺寸
        self.h = h  # 高度方向的特定尺寸
        self.w = w  # 宽度方向的特定尺寸

        # 定义D方向的全局平均池化和最大池化
        self.avg_pool_d = nn.AdaptiveAvgPool3d((d, 1, 1))  # 输出大小为(1, h, w)
        self.max_pool_d = nn.AdaptiveMaxPool3d((d, 1, 1))
        # 定义H方向的全局平均池化和最大池化
        self.avg_pool_h = nn.AdaptiveAvgPool3d((1, h, 1))  # 输出大小为(d, 1, w)
        self.max_pool_h = nn.AdaptiveMaxPool3d((1, h, 1))
        # 定义W方向的全局平均池化和最大池化
        self.avg_pool_w = nn.AdaptiveAvgPool3d((1, 1, w))  # 输出大小为(d, h, 1)
        self.max_pool_w = nn.AdaptiveMaxPool3d((1, 1, w))

        # 定义共享的卷积层，用于通道间的降维和恢复
        self.shared_conv = nn.Sequential(
            nn.Conv3d(in_channels=self.group_channels, out_channels=self.group_channels // reduction,
                      kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(self.group_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=self.group_channels // reduction, out_channels=self.group_channels,
                      kernel_size=(1, 1, 1))
        )
        # 定义sigmoid激活函数
        self.sigmoid_d = nn.Sigmoid()
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        batch_size, channel, depth, height, width = x.size()
        # print("x", x.shape)
        assert channel % self.num_groups == 0, "The number of channels must be divisible by the number of groups."

        # 将输入特征图按通道数分组
        x = x.view(batch_size, self.num_groups, self.group_channels, depth, height, width)
        # print("x", x.shape)

        # D方向的全局平均池化和最大池化
        x_d_avg = self.avg_pool_d(x.view(batch_size * self.num_groups, self.group_channels, depth, height, width)).view(
            batch_size, self.num_groups, self.group_channels, depth, 1, 1)
        x_d_max = self.max_pool_d(x.view(batch_size * self.num_groups, self.group_channels, depth, height, width)).view(
            batch_size, self.num_groups, self.group_channels, depth, 1, 1)
        # print("x_d_avg", x_d_avg.shape)
        # print("x_d_max", x_d_max.shape)

        # H方向的全局平均池化和最大池化
        x_h_avg = self.avg_pool_h(x.view(batch_size * self.num_groups, self.group_channels, depth, height, width)).view(
            batch_size, self.num_groups, self.group_channels, 1, height, 1)
        x_h_max = self.max_pool_h(x.view(batch_size * self.num_groups, self.group_channels, depth, height, width)).view(
            batch_size, self.num_groups, self.group_channels, 1, height, 1)
        # print("x_h_avg", x_h_avg.shape)
        # print("x_h_max", x_h_max.shape)

        # W方向的全局平均池化和最大池化
        x_w_avg = self.avg_pool_w(x.view(batch_size * self.num_groups, self.group_channels, depth, height, width)).view(
            batch_size, self.num_groups, self.group_channels, 1, 1, width)
        x_w_max = self.max_pool_w(x.view(batch_size * self.num_groups, self.group_channels, depth, height, width)).view(
            batch_size, self.num_groups, self.group_channels, 1, 1, width)
        # print("x_w_avg", x_w_avg.shape)
        # print("x_w_max", x_w_max.shape)

        # 应用共享卷积层进行特征处理
        y_d_avg = self.shared_conv(x_d_avg.view(batch_size * self.num_groups, self.group_channels, depth, 1, 1))
        y_d_max = self.shared_conv(x_d_max.view(batch_size * self.num_groups, self.group_channels, depth, 1, 1))
        # print("y_d_avg", y_d_avg.shape)
        # print("y_d_max", y_d_max.shape)

        y_h_avg = self.shared_conv(x_h_avg.view(batch_size * self.num_groups, self.group_channels, 1, height, 1))
        y_h_max = self.shared_conv(x_h_max.view(batch_size * self.num_groups, self.group_channels, 1, height, 1))
        # print("y_h_avg", y_h_avg.shape)
        # print("y_h_max", y_h_max.shape)

        y_w_avg = self.shared_conv(x_w_avg.view(batch_size * self.num_groups, self.group_channels, 1, 1, width))
        y_w_max = self.shared_conv(x_w_max.view(batch_size * self.num_groups, self.group_channels, 1, 1, width))
        # print("y_w_avg", y_w_avg.shape)
        # print("y_w_max", y_w_max.shape)

        # 计算注意力权重
        att_d = self.sigmoid_d(y_d_avg + y_d_max).view(batch_size, self.num_groups, self.group_channels, depth, 1, 1)
        # print("att_d", att_d.shape)
        att_h = self.sigmoid_h(y_h_avg + y_h_max).view(batch_size, self.num_groups, self.group_channels, 1, height, 1)
        # print("att_h", att_h.shape)
        att_w = self.sigmoid_w(y_w_avg + y_w_max).view(batch_size, self.num_groups, self.group_channels, 1, 1, width)
        # print("att_w", att_w.shape)

        # 应用注意力权重
        out = x * att_d * att_h * att_w
        # print("out1", out.shape)
        out = out.view(batch_size, channel, depth, height, width)
        # print("out2", out.shape)

        return out

# if __name__ == '__main__':
#     block = GGCA3D(channel=64, d=32, h=32, w=32, reduction=16, num_groups=4).cuda()  # 初始化GGCA模块
#     input = torch.rand(16, 64, 32, 32, 32).cuda()  # 五维输入：batch size, channels, depth, height, width
#     output = block(input)  # 将输入通过GGCA模块处理
#     print(output.shape)  # 输出处理后的数据形状


# -----------------------------------------AG--------------------------------------------
class AttendingBlock3D(nn.Module):
    """
    x: 左边跳过来的特征图
    g: 上采样上来的特征图
    """

    def __init__(self, in_channels_x, in_channels_g, out_channels=None):
        super(AttendingBlock3D, self).__init__()

        if out_channels is None:
            out_channels = in_channels_x
        self.Wx = nn.Conv3d(in_channels_x, out_channels, kernel_size=1, stride=1, bias=False)
        self.Wg = nn.Conv3d(in_channels_g, out_channels, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv_1 = nn.Conv3d(out_channels, 1, kernel_size=1, stride=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=1)
        # self.norm = nn.GroupNorm(num_groups=4, num_channels=mid_channels, eps=1e-05, affine=True)
        self.norm = NormalizationLayer(norm_type='group')  # 使用自定义的归一化层

    def forward(self, x, g):
        x1 = self.Wx(x)

        g = self.Wg(g)
        g1 = self.norm(g)
        # add = x1.add(g1)
        add = x1 + g1

        relu = self.relu(add)
        conv = self.conv_1(relu)
        out = self.sigmoid(conv)
        # 在空间上对数据进行缩放或变形操作。即上采样（up-sampling）操作
        # 在宽、高、深度上进行线性插值。
        # 优势：有效地保留和处理空间数据的详细信息，使得模型在空间上的预测或生成结果更为精确和可靠。
        up = F.interpolate(out, size=out.size()[2:], mode='trilinear', align_corners=False)

        out = x * up

        out = self.conv_2(out)
        out = self.norm(out)

        return out


'''
# if __name__ == "__main__":
#     xi = torch.randn(1, 16, 16, 96, 96).cuda()
#     # xi_1 = torch.randn(1, 384, 14, 14)
#     g = torch.randn(1, 32, 16, 96, 96).cuda()
#     # ff = ContextBridge(dim=192)
# 
#     attn = AttentionGate(in_channels_global=16, in_channels_local=32).cuda()
# 
#     print(attn(xi, g).shape)
'''