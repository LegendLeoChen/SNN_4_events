import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tonic
from tonic import CachedDataset
from tqdm import tqdm
import time
from SNN.snn_process import *
from SNN.snn import SNNNet_3D

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("训练设备：", device)

'''----------------数据集构建------------------'''
time_window = 20
# 1、读事件流数据集
# transform = EventToSpikeTensor(time_window=time_window)         # 事件流转3D图像
# train_dataset = CachedDataset(                                  # 训练集
#     tonic.datasets.NMNIST(save_to='./datasets', train=True),
#     transform=transform,
#     cache_path='./cache'
# )
# test_dataset = CachedDataset(                                   # 测试集
#     tonic.datasets.NMNIST(save_to='./datasets', train=False),
#     transform=transform,
#     cache_path='./cache'
# )

# 2、读脉冲张量数据集（已经过EventToSpikeTensor）
data_dir = './datasets/NMNIST_processed'
train_dataset = PreprocessedDataset(data_dir=data_dir, split='train')       # 此时数据集里一个item有多个样本
test_dataset = PreprocessedDataset(data_dir=data_dir, split='test')
train_dataset = FlattenedDataset(train_dataset)                             # 展平成单样本item的数据集
test_dataset = FlattenedDataset(test_dataset)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True)


'''----------------测试函数------------------'''
def batch_accuracy(test_loader, net, num_steps, criterion, max_batches=None):
    with torch.no_grad():
        net.eval()
        correct = 0
        total = 0
        processed_batches = 0
        test_loss = 0.0                 # 测试集损失

        for batch_idx, (data, targets) in enumerate(tqdm(test_loader, desc="验证", unit="batch")):
            data = data.to(device)
            targets = targets.to(device)

            spk_rec = net(data)                                 # 前向传播
            predicted = torch.argmax(spk_rec, dim=1)            # 预测类别
            loss = criterion(spk_rec, targets.long())           # 损失计算
            test_loss += loss.item() * targets.size(0)          # 累加损失

            total += targets.size(0)                            # 统计数量
            correct += (predicted == targets).sum().item()

            processed_batches += 1
            if max_batches is not None and processed_batches >= max_batches:
                break

        test_loss /= total                                      # 平均损失
        print(f"测试集准确率: {correct}/{total}, {correct / total * 100:.2f}%")
        print(f"测试集平均损失: {test_loss:.4f}")

    return correct / total, test_loss


'''----------------训练环节------------------'''
model = SNNNet_3D().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', min_lr=8e-7, factor=0.5, patience=40, verbose=True)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=600, eta_min=1e-7)

num_epochs = 30
num_steps = time_window
loss_hist = []
test_acc_hist = []
counter = 0

for epoch in range(num_epochs):
    progress_bar = tqdm(iter(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")   # 进度条
    for data, targets in progress_bar:
        data = data.to(device)
        targets = targets.to(device)
        model.train()

        # 前向传播 损失计算 反向传播 参数更新 学习率更新
        spk_rec = model(data)
        loss_val = criterion(spk_rec, targets.long())
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        # scheduler.step(loss_val.item())
        scheduler.step()

        loss_hist.append(loss_val.item())
        progress_bar.set_postfix({"损失": f"{loss_val.item():.4f}", "学习率": f"{scheduler.get_last_lr()[0]}"})

        # 验证
        if counter % 100 == 0:
            with torch.no_grad():
                model.eval()
                test_acc = batch_accuracy(test_loader, model, num_steps, criterion, max_batches=2)     # 验证
                test_acc_hist.append(test_acc)
        counter += 1

result_plot(loss_hist, test_acc_hist)       # 绘制结果图线（训练集损失、测试集准确率）

'''----------------测试环节------------------'''
def test_single_event_stream_inference_time(net, num_steps, data):
    net.eval()
    with torch.no_grad():
        data = data.to(device)
        start_time = time.time()
        spk_rec = model(data)
        end_time = time.time()
        inference_time = end_time - start_time
    return inference_time

# 获取一个事件流，测试推理速度
for data, targets in test_loader:
    single_event_stream = data[0:1]  # 获取第一个事件流
    break
inference_time = test_single_event_stream_inference_time(model, num_steps, single_event_stream)
print(f"单个事件流预测: {inference_time:.4f} 秒")
