# —*-coding:utf-8 一*一
"""
作者:LegendLeo
日期:2025年02月16日
各种工具: 处理事件流、绘制图像、定制数据集
"""
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from torch.utils.data import Dataset


def plot_spike_tensor(spike_tensor, cols=5):
    """
    绘制脉冲张量，时间戳为0的地方显示为黑色，其他地方根据时间戳大小显示为从浅黄色到深黄色
    :param spike_tensor: 脉冲张量，形状为 (time_window, 1, img_size[0], img_size[1])
    :param cols: 每行显示的子图数量，默认为5
    """
    time_window = spike_tensor.shape[0]
    img_size = spike_tensor.shape[2:]

    # 计算需要多少行
    rows = (time_window + cols - 1) // cols

    # 创建一个总图，包含 time_window 个子图
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    # 如果只有一个子图，axes 不是数组，需要将其转换为数组
    if time_window == 1:
        axes = np.array([axes])

    # 将所有轴对象展平为一个列表
    axes = axes.flatten()

    # 自定义从浅黄色到深黄色的色彩映射
    colors = [(0, 0, 0), (1, 1, 0.6), (1, 1, 0)]  # RGB值，黑色到浅黄色到深黄色
    cmap = LinearSegmentedColormap.from_list('custom_yellow', colors)

    # 遍历每个时间步
    for t in range(time_window):
        # 获取当前时间步的图像
        img = spike_tensor[t, 0].numpy()

        # 动态计算当前子图的归一化范围
        min_val = img.min()
        max_val = img.max()

        # 如果当前子图没有数据（全为0），跳过绘制
        if min_val == max_val == 0:
            axes[t].set_title(f'Time Step {t}')
            axes[t].set_xticks([])
            axes[t].set_yticks([])
            continue

        # 绘制图像，使用自定义的黄色色调
        im = axes[t].imshow(img, cmap=cmap, vmin=0, vmax=max_val)  # 确保时间戳为0的地方显示为黑色
        axes[t].set_title(f'Time Step {t}')
        axes[t].set_xticks([])
        axes[t].set_yticks([])

    # 隐藏多余的子图
    for t in range(time_window, rows * cols):
        fig.delaxes(axes[t])

    # 显示总图
    plt.tight_layout()
    plt.show()


class EventToSpikeTensor:           # 事件流转为3D图像
    def __init__(self, time_window, img_size=(34, 34)):
        self.time_window = time_window
        self.img_size = img_size

    def __call__(self, events):
        """
        将事件数据转换为脉冲张量
        :param events: 事件数据，形状为 (N, 4)，其中 N 是事件数量，4 表示 (x, y, t, p)
        :return: 脉冲张量，形状为 (time_window, 1, img_size[0], img_size[1])
        """
        # 初始化脉冲张量
        spike_tensor = torch.zeros(self.time_window, 1, self.img_size[0], self.img_size[1])
        # 获取所有时间戳的最大值和最小值，用于归一化
        min_time = events['t'].min()
        max_time = events['t'].max()

        # 遍历每个事件
        for event in events:
            x, y, t, p = event
            if p == 1:  # 只处理正极性事件
                # 归一化时间戳
                normalized_time = (t - min_time) / (max_time - min_time)
                # 将归一化的时间戳填入脉冲张量
                time_step = int(normalized_time * (self.time_window - 1))
                spike_tensor[time_step, 0, y, x] = normalized_time
        # plot_spike_tensor(spike_tensor)
        return spike_tensor


class PreprocessedDataset(Dataset):
    def __init__(self, data_dir, split):
        """
        加载预处理后的脉冲张量数据集
        :param data_dir: 存储预处理数据的目录
        :param split: 数据集类型（'train' 或 'test'）
        """
        self.data_dir = os.path.join(data_dir, split)
        self.file_list = [f for f in os.listdir(self.data_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data, target = torch.load(file_path)
        return data, target


class FlattenedDataset(Dataset):
    def __init__(self, dataset):
        """
        将每个item有多个样本的数据集展平为单个样本的列表，也就是一个item一个样本
        :param dataset: 包含多个文件的 Dataset
        """
        self.samples = []
        self.labels = []

        for data, target in dataset:
            # 如果 data 是一个批量，将其拆分为单个样本
            for i in range(data.size(0)):
                self.samples.append(data[i])
                self.labels.append(target[i])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


def result_plot(loss_hist, test_acc_hist):          # 绘制训练损失和测试准确率
    plt.figure(figsize=(10, 5))
    plt.plot(loss_hist, label="Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.figure(facecolor="w")
    plt.plot(test_acc_hist)
    plt.title("Test Set Accuracy")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.show()