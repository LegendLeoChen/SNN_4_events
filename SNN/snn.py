# —*-coding:utf-8 一*一
"""
作者:LegendLeo
日期:2025年02月16日
snn网络
"""
import torch.nn as nn
from snntorch import surrogate
import snntorch as snn
import torch.nn.functional as F

class SNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(12, 64, 5)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        self.fc1 = nn.Linear(1600, 10)
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=spike_grad)

    def forward(self, x):
        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        cur1 = F.max_pool2d(self.conv1(x), 2)
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = F.max_pool2d(self.conv2(spk1), 2)
        spk2, mem2 = self.lif2(cur2, mem2)

        cur3 = self.fc1(spk2.view(spk2.size(0), -1))
        spk3, mem3 = self.lif3(cur3, mem3)
        return spk3


class SNNNet_3D(nn.Module):
    def __init__(self):
        super(SNNNet_3D, self).__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 5, 5))  # 3D卷积
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        self.conv2 = nn.Conv3d(32, 128, kernel_size=(3, 5, 5))  # 3D卷积
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        self.fc1 = nn.Linear(51200, 10)
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        # 初始化网络参数
        self.init_weights()

    def init_weights(self):
        # 初始化卷积层权重（He 初始化）
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')

        # 初始化全连接层权重（Xavier 初始化）
        nn.init.xavier_uniform_(self.fc1.weight)

        # 初始化偏置为零（可选）
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.constant_(self.fc1.bias, 0)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        x = x.permute(0, 2, 1, 3, 4)  # 调整维度
        cur1 = F.max_pool3d(self.conv1(x), kernel_size=(1, 2, 2))  # 3D池化
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = F.max_pool3d(self.conv2(spk1), kernel_size=(1, 2, 2))  # 3D池化
        spk2, mem2 = self.lif2(cur2, mem2)

        # 展平操作
        cur3 = spk2.view(spk2.size(0), -1)
        spk3, mem3 = self.lif3(self.fc1(cur3), mem3)
        return spk3


class SNNNet_3D_L(nn.Module):
    def __init__(self):
        super(SNNNet_3D_L, self).__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3))  # 3D卷积
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        self.conv2 = nn.Conv3d(32, 128, kernel_size=(3, 3, 3))  # 3D卷积
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        self.conv3 = nn.Conv3d(128, 64, kernel_size=(3, 3, 3))
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        self.conv4 = nn.Conv3d(64, 32, kernel_size=(3, 3, 3))
        self.lif4 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        self.fc1 = nn.Linear(6144, 10)
        self.lif5 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        # 初始化网络参数
        self.init_weights()

    def init_weights(self):
        # 初始化卷积层权重（He 初始化）
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv4.weight, nonlinearity='relu')

        # 初始化全连接层权重（Xavier 初始化）
        nn.init.xavier_uniform_(self.fc1.weight)

        # 初始化偏置为零（可选）
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.constant_(self.conv3.bias, 0)
        nn.init.constant_(self.conv4.bias, 0)
        nn.init.constant_(self.fc1.bias, 0)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()

        x = x.permute(0, 2, 1, 3, 4)  # 调整维度
        cur1 = F.max_pool3d(self.conv1(x), kernel_size=(1, 2, 2))  # 3D池化
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = self.conv2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)

        cur3 = F.max_pool3d(self.conv3(spk2), kernel_size=(1, 2, 2))  # 新增3D池化
        spk3, mem3 = self.lif3(cur3, mem3)

        cur4 = self.conv4(spk3)
        spk4, mem4 = self.lif4(cur4, mem4)

        # 展平操作
        cur5 = spk4.view(spk4.size(0), -1)
        spk5, mem5 = self.lif5(self.fc1(cur5), mem5)
        return spk5