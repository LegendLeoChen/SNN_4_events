import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tonic
from tonic import CachedDataset
from tqdm import tqdm  # 导入 tqdm 库
import time
from SNN.snn_process import *
from SNN.snn import SNNNet

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("训练设备：", device)

# 加载N-MNIST数据集
transform = EventToSpikeTensor(time_window=20)
# 训练集 测试集
train_dataset = CachedDataset(
    tonic.datasets.NMNIST(save_to='./datasets', train=True),
    transform=transform,
    cache_path='./cache'
)
test_dataset = CachedDataset(
    tonic.datasets.NMNIST(save_to='./datasets', train=False),
    transform=transform,
    cache_path='./cache'
)
# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True)

model = SNNNet().to(device)

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 4. 训练模型
def forward_pass(net, num_steps, data):
    spk_rec = []
    for t in range(num_steps):
        spk_out = net(data[:, t, :, :, :])
        spk_rec.append(spk_out)
    return torch.stack(spk_rec), None

def batch_accuracy(test_loader, net, num_steps, max_batches=None):
    with torch.no_grad():
        net.eval()
        correct = 0
        total = 0
        processed_batches = 0

        # 使用 tqdm 创建进度条，并动态更新描述字段
        for batch_idx, (data, targets) in enumerate(tqdm(test_loader, desc="验证", unit="batch", leave=True)):
            data = data.to(device)
            targets = targets.to(device)
            spk_rec, _ = forward_pass(net, num_steps, data)
            _, predicted = torch.max(spk_rec.sum(dim=0), 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            # 动态更新进度条描述字段
            # 如果指定了最大批次数量，则在达到该数量时停止
            processed_batches += 1
            if max_batches is not None and processed_batches >= max_batches:
                break
        print(f"当前批次 {batch_idx + 1}, 正确率: {correct}/{total}, {correct / total * 100:.2f}%")
    return correct / total

num_epochs = 1
num_steps = 20
loss_hist = []
test_acc_hist = []
counter = 0

for epoch in range(num_epochs):
    # Training loop with tqdm progress bar
    for data, targets in tqdm(iter(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        model.train()
        spk_rec, _ = forward_pass(model, num_steps, data)

        # initialize the loss & sum over time
        loss_val = criterion(spk_rec.sum(dim=0), targets.long())

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # Test set
        if counter % 50 == 0:
            with torch.no_grad():
                model.eval()

                # Test set forward pass
                test_acc = batch_accuracy(test_loader, model, num_steps, max_batches=2)
                print(f"Iteration {counter}, Test Acc: {test_acc * 100:.2f}%\n")
                test_acc_hist.append(test_acc)

        counter += 1

# 5. 绘制测试集准确率曲线
plt.figure(facecolor="w")
plt.plot(test_acc_hist)
plt.title("Test Set Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.show()

def test_single_event_stream_inference_time(net, num_steps, data):
    net.eval()
    with torch.no_grad():
        data = data.to(device)
        start_time = time.time()
        spk_rec, _ = forward_pass(net, num_steps, data)
        end_time = time.time()
        inference_time = end_time - start_time
    return inference_time

# 从测试集中获取一个事件流
for data, targets in test_loader:
    single_event_stream = data[0:1]  # 获取第一个事件流
    break

# 测试单个事件流的预测耗时
inference_time = test_single_event_stream_inference_time(model, num_steps, single_event_stream)
print(f"单个事件流预测: {inference_time:.4f} 秒")
