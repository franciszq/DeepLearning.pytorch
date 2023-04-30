import math

import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, depth_multiplier=1,
                 width_multiplier=1.0):
        """
        深度可分离卷积
        :param in_channels: 输入通道数
        :param out_channels:  输出通道数
        :param kernel_size: depthwise卷积核大小
        :param stride: depthwise卷积的步长
        :param padding:  depthwise卷积的padding
        :param depth_multiplier: 在depthwise步骤中每个输入通道生成多少个输出通道
        :param width_multiplier: 用于将模型变小和降低计算复杂度的参数，取值范围：(0, 1]
        """
        super().__init__()
        in_channels = int(in_channels * width_multiplier)
        depthwise_channel = int(in_channels * depth_multiplier)
        out_channels = int(out_channels * width_multiplier)
        self.depthwise_conv = nn.Conv2d(in_channels, depthwise_channel, kernel_size, stride=stride,
                                        padding=padding, groups=in_channels)
        self.bn1 = nn.BatchNorm2d(num_features=depthwise_channel)
        self.act1 = nn.ReLU(inplace=True)

        self.pointwise_conv = nn.Conv2d(depthwise_channel, out_channels, 1, 1, padding=0)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act1(self.bn1(self.depthwise_conv(x)))
        x = self.act2(self.bn2(self.pointwise_conv(x)))
        return x


class MobileNetV1(nn.Module):
    def __init__(self, width_multiplier=1.0):
        """
        URL: https://arxiv.org/abs/1704.04861
        @article{howard2017mobilenets,
          title={Mobilenets: Efficient convolutional neural networks for mobile vision applications},
          author={Howard, Andrew G and Zhu, Menglong and Chen, Bo and Kalenichenko, Dmitry and Wang, Weijun and Weyand, Tobias and Andreetto, Marco and Adam, Hartwig},
          journal={arXiv preprint arXiv:1704.04861},
          year={2017}
        }
        :param width_multiplier: 用于将模型变小和降低计算复杂度的参数，取值范围：(0, 1]
        """
        super().__init__()
        n = int(32 * width_multiplier)
        self.conv1 = nn.Conv2d(3, n, 3, 2, padding=1)
        self.bn = nn.BatchNorm2d(n)
        self.relu = nn.ReLU(True)
        self.layer_1 = self._make_layer(32, 64, 3, 1, 1, width_multiplier)
        self.layer_2 = self._make_layer(64, 128, 3, 2, 1, width_multiplier)
        self.layer_3 = self._make_layer(128, 128, 3, 1, 1, width_multiplier)
        self.layer_4 = self._make_layer(128, 256, 3, 2, 1, width_multiplier)
        self.layer_5 = self._make_layer(256, 256, 3, 1, 1, width_multiplier)
        self.layer_6 = self._make_layer(256, 512, 3, 1, 1, width_multiplier)
        self.layer_7 = self._make_layer(512, 512, 3, 1, 5, width_multiplier)
        self.layer_8 = self._make_layer(512, 1024, 3, 2, 1, width_multiplier)
        self.layer_9 = self._make_layer(1024, 1024, 3, 1, 1, width_multiplier)

    def _make_layer(self, c_in, c_out, kernel_size=3, stride=1, num_layers=1, width_multiplier=1.0):
        p = math.ceil((kernel_size - stride) / 2)
        layers = []
        for _ in range(num_layers):
            layers.append(SeparableConv2d(c_in, c_out, kernel_size, stride, padding=p,
                                          width_multiplier=width_multiplier))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.bn(x))
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.layer_7(x)
        o1 = x.clone()
        x = self.layer_8(x)
        x = self.layer_9(x)

        return o1, x   # [torch.Size([2, 512, 38, 38]), torch.Size([2, 1024, 19, 19])]


if __name__ == '__main__':
    x = torch.randn(2, 3, 300, 300)
    net = MobileNetV1()
    print(net)
    y = net(x)
    print([e.size() for e in y])
