# 我的python
# 开发时间：2024/7/7 18:53
import torch
from torch import nn
# from layers import SELayer
from vnet_new_def import NormalizationLayer


class VNet(nn.Module):
    name = "Vnet"

    def __init__(self, in_channel, num_classes, width=16, **kwargs):
        super().__init__()

        self.in_channel = in_channel
        self.num_classes = num_classes

        self.relu = nn.GELU()
        self.droup = nn.Dropout3d(p=0.5)

        # 左侧卷积---点卷积+卷积一次
        self.L1_point_conv = nn.Conv3d(self.in_channel, width * 1, kernel_size=1, stride=1)
        self.L1_bn = nn.BatchNorm3d(width * 1)
        self.L1_conv1 = nn.Conv3d(self.in_channel, width * 1, kernel_size=5, stride=1, padding=2)
        self.L1_bn1 = NormalizationLayer(norm_type='group')

        # 左侧下采样（1）
        self.L1_down_conv = nn.Conv3d(width * 1, width * 1, kernel_size=2, stride=2)

        # 左侧卷积---点卷积+卷积两次
        self.L2_point_conv = nn.Conv3d(width * 1, width * 2, kernel_size=1, stride=1)
        self.L2_bn = nn.BatchNorm3d(width * 2)
        self.L2_conv1 = nn.Conv3d(width * 1, width * 1, kernel_size=5, stride=1, padding=2)
        self.L2_bn1 = NormalizationLayer(norm_type='group')
        self.L2_conv2 = nn.Conv3d(width * 1, width * 2, kernel_size=5, stride=1, padding=2)
        self.L2_bn2 = NormalizationLayer(norm_type='group')

        # 左侧下采样（2）
        self.L2_down_conv = nn.Conv3d(width * 2, width * 2, kernel_size=2, stride=2)

        # 左侧卷积---点卷积+卷积三次（1）
        self.L3_point_conv = nn.Conv3d(width * 2, width * 4, kernel_size=1, stride=1)
        self.L3_bn = nn.BatchNorm3d(width * 4)
        self.L3_conv1 = nn.Conv3d(width * 2, width * 2, kernel_size=5, stride=1, padding=2)
        self.L3_bn1 = NormalizationLayer(norm_type='group')
        self.L3_conv3 = nn.Conv3d(width * 2, width * 4, kernel_size=5, stride=1, padding=2)
        self.L3_bn3 = NormalizationLayer(norm_type='group')

        # 左侧下采样（3）
        self.L3_down_conv = nn.Conv3d(width * 4, width * 4, kernel_size=2, stride=2)

        # 左侧---点卷积+卷积三次（2）
        self.L4_point_conv = nn.Conv3d(width * 4, width * 8, kernel_size=1, stride=1)
        self.L4_bn = nn.BatchNorm3d(width * 8)
        self.L4_conv1 = nn.Conv3d(width * 4, width * 4, kernel_size=5, stride=1, padding=2)
        self.L4_bn1 = NormalizationLayer(norm_type='group')
        self.L4_conv3 = nn.Conv3d(width * 4, width * 8, kernel_size=5, stride=1, padding=2)
        self.L4_bn3 = NormalizationLayer(norm_type='group')

        # 左侧下采样（4）
        self.L4_down_conv = nn.Conv3d(width * 8, width * 8, kernel_size=2, stride=2)

        # 底部---点卷积+卷积三次
        self.Bottom_point_conv = nn.Conv3d(width * 8, width * 16, kernel_size=1, stride=1)
        self.Bottom_bn = nn.BatchNorm3d(width * 16)
        self.Bottom_conv1 = nn.Conv3d(width * 8, width * 8, kernel_size=5, stride=1, padding=2)
        self.Bottom_bn1 = NormalizationLayer(norm_type='group')
        self.Bottom_conv3 = nn.Conv3d(width * 8, width * 16, kernel_size=5, stride=1, padding=2)
        self.Bottom_bn3 = NormalizationLayer(norm_type='group')

        # 右侧上采样（4)
        self.R4_up_conv = nn.ConvTranspose3d(width * 16, width * 16, kernel_size=2, stride=2)

        # 右侧---点卷积+卷积三次(4)
        self.R4_point_conv = nn.Conv3d(width * (8 + 16), width * 16, kernel_size=1, stride=1)  # 384---256
        self.R4_bn = nn.BatchNorm3d(width * 16)
        self.R4_conv1 = nn.Conv3d(width * (8 + 16), width * 16, kernel_size=5, stride=1, padding=2)
        self.R4_bn1 = NormalizationLayer(norm_type='group')
        self.R4_conv2 = nn.Conv3d(width * 16, width * 16, kernel_size=5, stride=1, padding=2)
        self.R4_bn2 = NormalizationLayer(norm_type='group')

        # 右侧上采样（3）
        self.R3_up_conv = nn.ConvTranspose3d(width * 16, width * 16, kernel_size=2, stride=2)

        # 右侧---点卷积+卷积三次(3)
        self.R3_point_conv = nn.Conv3d(width * (4 + 16), width * 8, kernel_size=1, stride=1)  # 320---128
        self.R3_bn = nn.BatchNorm3d(width * 8)
        self.R3_conv1 = nn.Conv3d(width * (4 + 16), width * 8, kernel_size=5, stride=1, padding=2)
        self.R3_bn1 = NormalizationLayer(norm_type='group')

        # 右侧上采样（2）
        self.R2_up_conv = nn.ConvTranspose3d(width * 8, width * 8, kernel_size=2, stride=2)

        # 右侧---点卷积+卷积两次
        self.R2_point_conv = nn.Conv3d(width * (2 + 8), width * 4, kernel_size=1, stride=1)
        self.R2_bn = nn.BatchNorm3d(width * 4)
        self.R2_conv1 = nn.Conv3d(width * (2 + 8), width * 4, kernel_size=5, stride=1, padding=2)
        self.R2_bn1 = NormalizationLayer(norm_type='group')

        # 右侧上采样（1）
        self.R1_up_conv = nn.ConvTranspose3d(width * 4, width * 4, kernel_size=2, stride=2)

        # 右侧---点卷积+卷积一次
        self.R1_point_conv = nn.Conv3d(width * (1 + 4), width * 2, kernel_size=1, stride=1)
        self.R1_bn = nn.BatchNorm3d(width * 2)
        self.R1_conv1 = nn.Conv3d(width * (1 + 4), width * 2, kernel_size=5, stride=1, padding=2)
        self.R1_bn1 = NormalizationLayer(norm_type='group')

        # 右侧---点卷积
        self.point_conv = nn.Conv3d(width * 2, self.num_classes, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm3d(self.num_classes)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.to(torch.float32)
        # 左侧第一层
        res1_l1 = x
        res1_l1 = self.L1_point_conv(res1_l1)  # 左侧点卷积（1）
        res1_l1 = self.L1_bn(res1_l1)
        res1_l1 = self.relu(res1_l1)  # 待残差

        l1 = self.L1_conv1(x)  # 卷积一次
        l1 = self.L1_bn1(l1)
        l1 = self.relu(l1)
        l1 = self.droup(l1)

        l1 = torch.add(res1_l1, l1)  # 残差（1）
        layer1 = l1  # 待拼接

        # 下采样（1）
        l1 = self.L1_down_conv(l1)
        l1 = self.L1_bn(l1)
        l1 = self.relu(l1)

        # 左侧第二层
        res2_l2 = l1
        res2_l2 = self.L2_point_conv(res2_l2)  # 左侧点卷积（2）
        res2_l2 = self.L2_bn(res2_l2)
        res2_l2 = self.relu(res2_l2)  # 待残差

        l2 = self.L2_conv1(l1)  # 卷积一次
        l2 = self.L2_bn1(l2)
        l2 = self.relu(l2)
        l2 = self.droup(l2)

        l2 = self.L2_conv2(l2)  # 卷积两次
        l2 = self.L2_bn2(l2)
        l2 = self.relu(l2)
        l2 = self.droup(l2)

        l2 = torch.add(res2_l2, l2)  # 残差（2）
        layer2 = l2  # 待拼接

        # 下采样（2）
        l2 = self.L2_down_conv(l2)
        l2 = self.L2_bn(l2)
        l2 = self.relu(l2)

        # 左侧第三层
        res3_l3 = l2
        res3_l3 = self.L3_point_conv(res3_l3)  # 左侧点卷积（3）
        res3_l3 = self.L3_bn(res3_l3)
        res3_l3 = self.relu(res3_l3)  # 待残差

        l3 = self.L3_conv1(l2)  # 卷积一次
        l3 = self.L3_bn1(l3)
        l3 = self.relu(l3)
        l3 = self.droup(l3)

        l3 = self.L3_conv1(l3)  # 卷积两次
        l3 = self.L3_bn1(l3)
        l3 = self.relu(l3)
        l3 = self.droup(l3)

        l3 = self.L3_conv3(l3)  # 卷积三次
        l3 = self.L3_bn3(l3)
        l3 = self.relu(l3)
        l3 = self.droup(l3)

        l3 = torch.add(res3_l3, l3)  # 残差（3）
        layer3 = l3  # 待拼接

        # 下采样（3）
        l3 = self.L3_down_conv(l3)
        l3 = self.L3_bn(l3)
        l3 = self.relu(l3)

        # 左侧第四层
        res4_l4 = l3
        res4_l4 = self.L4_point_conv(res4_l4)  # 左侧点卷积（4）
        res4_l4 = self.L4_bn(res4_l4)
        res4_l4 = self.relu(res4_l4)  # 待残差

        l4 = self.L4_conv1(l3)  # 卷积一次
        l4 = self.L4_bn1(l4)
        l4 = self.relu(l4)
        l4 = self.droup(l4)

        l4 = self.L4_conv1(l4)  # 卷积两次
        l4 = self.L4_bn1(l4)
        l4 = self.relu(l4)
        l4 = self.droup(l4)

        l4 = self.L4_conv3(l4)  # 卷积三次
        l4 = self.L4_bn3(l4)
        l4 = self.relu(l4)
        l4 = self.droup(l4)

        l4 = torch.add(res4_l4, l4)  # 残差（4）
        layer4 = l4  # 待拼接

        # 下采样（4）
        l4 = self.L4_down_conv(l4)
        l4 = self.L4_bn(l4)
        l4 = self.relu(l4)
        # print("l4", l4.shape)

        # 底部卷积
        res5_l5 = l4
        res5_l5 = self.Bottom_point_conv(res5_l5)  # 底部点卷积
        res5_l5 = self.Bottom_bn(res5_l5)
        res5_bottom = self.relu(res5_l5)  # 待残差

        bottom_conv = self.Bottom_conv1(l4)  # 卷积一次
        bottom_conv = self.Bottom_bn1(bottom_conv)
        bottom_conv = self.relu(bottom_conv)
        bottom_conv = self.droup(bottom_conv)

        bottom_conv = self.Bottom_conv1(bottom_conv)  # 卷积两次
        bottom_conv = self.Bottom_bn1(bottom_conv)
        bottom_conv = self.relu(bottom_conv)
        bottom_conv = self.droup(bottom_conv)

        bottom_conv = self.Bottom_conv3(bottom_conv)  # 卷积三次
        bottom_conv = self.Bottom_bn3(bottom_conv)
        bottom_conv = self.relu(bottom_conv)
        bottom_conv = self.droup(bottom_conv)

        bottom_conv = torch.add(res5_bottom, bottom_conv)  # 残差

        # 上采样（4）
        bottom_conv = self.R4_up_conv(bottom_conv)
        bottom_conv = self.Bottom_bn(bottom_conv)
        bottom_conv = self.relu(bottom_conv)
        layer_bottom = bottom_conv  # 待拼接
        # print("bottom_conv", bottom_conv.shape)

        # 右侧---点卷积+卷积三次（1）
        r4 = torch.cat([layer4, layer_bottom], dim=1)
        res4_r4 = r4
        res4_r4 = self.R4_point_conv(res4_r4)
        res4_r4 = self.R4_bn(res4_r4)
        res4_r4 = self.relu(res4_r4)  # 待残差

        r4 = self.R4_conv1(r4)  # 卷积一次
        r4 = self.R4_bn1(r4)
        r4 = self.relu(r4)
        r4 = self.droup(r4)

        r4 = self.R4_conv2(r4)  # 卷积两次
        r4 = self.R4_bn2(r4)
        r4 = self.relu(r4)
        r4 = self.droup(r4)

        r4 = self.R4_conv2(r4)  # 卷积三次
        r4 = self.R4_bn2(r4)
        r4 = self.relu(r4)
        r4 = self.droup(r4)

        r4 = torch.add(res4_r4, r4)  # 残差

        # 上采样（3）
        r3 = self.R3_up_conv(r4)
        r3 = self.R4_bn(r3)
        r3 = self.relu(r3)
        layer_r3 = r3  # 待拼接

        # 右侧---点卷积+卷积三次（2）
        r3 = torch.cat([layer3, layer_r3], dim=1)
        res3_r3 = r3
        res3_r3 = self.R3_point_conv(res3_r3)
        res3_r3 = self.R3_bn(res3_r3)
        res3_r3 = self.relu(res3_r3)  # 待残差

        r3 = self.R3_conv1(r3)  # 卷积一次
        r3 = self.R3_bn1(r3)
        r3 = self.relu(r3)
        r3 = self.droup(r3)

        r3 = self.Bottom_conv1(r3)  # 卷积两次
        r3 = self.Bottom_bn1(r3)
        r3 = self.relu(r3)
        r3 = self.droup(r3)

        r3 = self.Bottom_conv1(r3)  # 卷积三次
        r3 = self.Bottom_bn1(r3)
        r3 = self.relu(r3)
        r3 = self.droup(r3)

        r3 = torch.add(res3_r3, r3)  # 残差

        # 上采样（2）
        r2 = self.R2_up_conv(r3)
        r2 = self.R3_bn(r2)
        r2 = self.relu(r2)
        layer_r2 = r2  # 待拼接

        # 右侧---点卷积+卷积两次
        r2 = torch.cat([layer2, layer_r2], dim=1)
        res2_r2 = r2
        res2_r2 = self.R2_point_conv(res2_r2)
        res2_r2 = self.R2_bn(res2_r2)
        res2_r2 = self.relu(res2_r2)  # 待残差

        r2 = self.R2_conv1(r2)  # 卷积一次
        r2 = self.R2_bn1(r2)
        r2 = self.relu(r2)
        r2 = self.droup(r2)

        r2 = self.L4_conv1(r2)  # 卷积两次
        r2 = self.L4_bn1(r2)
        r2 = self.relu(r2)
        r2 = self.droup(r2)

        r2 = torch.add(res2_r2, r2)  # 残差

        # 上采样（1）
        r1 = self.R1_up_conv(r2)
        r1 = self.R2_bn(r1)
        r1 = self.relu(r1)
        layer_r1 = r1  # 待拼接

        # 右侧---点卷积+卷积一次
        r1 = torch.cat([layer1, layer_r1], dim=1)
        res1_r1 = r1
        res1_r1 = self.R1_point_conv(res1_r1)
        res1_r1 = self.R1_bn(res1_r1)
        res1_r1 = self.relu(res1_r1)  # 待残差

        r1 = self.R1_conv1(r1)  # 卷积一次
        r1 = self.R1_bn1(r1)
        r1 = self.relu(r1)
        r1 = self.droup(r1)

        r1 = torch.add(res1_r1, r1)  # 残差

        # 右侧---点卷积
        r1 = self.point_conv(r1)
        r1 = self.bn(r1)
        r1 = self.relu(r1)

        out = self.softmax(r1)
        return out


if __name__ == "__main__":
    # 计时
    import time
    from thop import profile

    start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.Tensor(2, 1, 16, 96, 96)
    x.to(device)
    print("x size: {}".format(x.size()))
    model = VNet(in_channel=1, num_classes=2)
    out = model(x)
    print("out size: {}".format(out.size()))

    # 计时
    end_time = time.time()
    cost_time = end_time - start_time
    print(f"代码执行时间：{cost_time:.2f}秒")

    # 计算 我3dunet 的 flops, params
    flops, params = profile(model, (x,))
    print('flops: ', flops, 'params: ', params)

