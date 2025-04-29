# —*-coding:utf-8 一*一
"""
作者:LegendLeo
日期:2025年02月20日
将N-MNIST的事件流转为脉冲张量后存储在硬盘，免除每次训练的转换数据时间
"""
import os
import torch
import tonic
from SNN.snn_process import EventToSpikeTensor
from tqdm import tqdm
from tonic import DiskCachedDataset
from torch.utils.data import DataLoader

# 定义保存数据的函数
def save_batched_spike_tensors(dataset, transform, save_path, split, batch_size=128):
    """
    将数据集中的事件流批量转换为脉冲张量并保存到本地
    :param dataset: 数据集对象
    :param transform: 转换函数
    :param save_path: 保存路径
    :param split: 数据集类型（训练集或测试集）
    :param batch_size: 每个文件保存的样本数量
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cached_dataset = DiskCachedDataset(dataset, transform=transform, cache_path=os.path.join(save_path, f"{split}_cache"))
    dataloader = DataLoader(cached_dataset, batch_size=batch_size, shuffle=False)

    file_idx = 0
    for batch_idx, (events, targets) in enumerate(tqdm(dataloader, desc=f"Processing {split}")):
        # 将批量数据保存到单个文件
        torch.save((events, targets), os.path.join(save_path, f"{split}_{file_idx}.pt"))
        file_idx += 1


if __name__ == '__main__':
    # 设置参数
    time_window = 20
    img_size = (34, 34)
    save_path = './datasets/NMNIST_processed'
    dataset_path = './datasets'
    batch_size = 512  # 每个文件保存的样本数量

    # 初始化数据集和转换函数
    transform = EventToSpikeTensor(time_window=time_window, img_size=img_size)

    # 加载训练集和测试集
    train_dataset = tonic.datasets.NMNIST(save_to=dataset_path, train=True)
    test_dataset = tonic.datasets.NMNIST(save_to=dataset_path, train=False)

    # 批量处理并保存训练集数据
    save_batched_spike_tensors(train_dataset, transform, save_path, split='train', batch_size=batch_size)

    # 批量处理并保存测试集数据
    save_batched_spike_tensors(test_dataset, transform, save_path, split='test', batch_size=batch_size)

    print("数据处理完成，已保存到本地。")