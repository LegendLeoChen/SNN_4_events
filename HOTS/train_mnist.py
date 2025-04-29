from __future__ import print_function
import argparse
import os
import glob
from random import shuffle
import numpy as np
from event_Python import eventvision
from lib.noise_filter import remove_isolated_pixels
from lib.layer_operations import visualise_time_surface_for_event_stream, initialise_time_surface_prototypes, \
    generate_layer_outputs, train_layer
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_event_data(event_data, width, height):
    # 找到所有不同的极性值
    polarities = set(event.p for event in event_data)
    num_polarities = len(polarities)

    # 创建一个颜色列表，每种极性一个颜色
    colors = plt.cm.get_cmap('tab20', num_polarities).colors  # 使用tab20颜色映射，最多支持20种颜色
    cmap = ListedColormap(colors)

    image = np.zeros((height, width), dtype=int)

    events_by_polarity = {}
    for event in event_data:
        if event.p not in events_by_polarity:
            events_by_polarity[event.p] = []
        events_by_polarity[event.p].append((event.x, event.y))

    # 为每种极性分配一个颜色索引
    polarity_to_color_index = {p: i for i, p in enumerate(polarities)}

    for polarity, events in events_by_polarity.items():
        for x, y in events:
            image[y, x] = polarity_to_color_index[polarity]  # 使用颜色索引

    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap=cmap, interpolation='nearest')
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.gca().invert_yaxis()  # 翻转y轴，使其与图像坐标系一致
    plt.title('Event Data Feature Map')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.show()

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='训练数字识别器')
    # 添加命令行参数
    parser.add_argument('--input_folders_training', action='store', nargs='+', default='datasets/mnist/Test/0',
                        help='包含事件文件的文件夹路径')
    parser.add_argument('--num_files_per_folder', action='store', default=1, type=int,
                        help="每个数字文件夹中读取的文件数量")

    # 解析命令行参数
    args = parser.parse_args()
    # 定义训练用的数字文件夹路径
    input_folders_training = [
        'datasets/mnist/Train/0',
        'datasets/mnist/Train/1',
        'datasets/mnist/Train/2',
        'datasets/mnist/Train/3',
        'datasets/mnist/Train/4',
        'datasets/mnist/Train/5',
        'datasets/mnist/Train/6',
        'datasets/mnist/Train/7',
        'datasets/mnist/Train/8',
        'datasets/mnist/Train/9']

    # 配置网络参数
    # 第一层参数
    N_1 = 4  # 特征数量，可以类比卷积的通道数（特征维度）
    tau_1 = 20000.  # 时间常数
    r_1 = 2  # 邻域半径

    # 定义参数增长因子
    K_N = 2  # 特征数量增长因子
    K_tau = 2  # 时间常数增长因子
    K_r = 2  # 邻域半径增长因子

    # 计算后续层的参数
    N_2 = N_1 * K_N
    tau_2 = tau_1 * K_tau
    r_2 = r_1 * K_r

    N_3 = N_2 * K_N
    tau_3 = tau_2 * K_tau
    r_3 = r_2 * K_r

    # 获取文件夹中的事件数据文件
    input_files_all = []

    for folder in input_folders_training:
        # 获取每个文件夹中的事件文件路径
        input_files = glob.glob(os.path.join(folder, '*.bin'))[:args.num_files_per_folder]
        input_files_all.extend(input_files)
        print('从{}获取的文件数量: {}'.format(folder, len(input_files)))

    # 打乱文件顺序
    shuffle(input_files_all)

    # 读取第一个文件的事件数据
    ev = eventvision.read_dataset(input_files_all[0])
    # 初始化事件数据列表
    event_data = []
    event_data_filt = []

    # 遍历所有事件文件合并为长事件流（每个文件是一段事件流）
    for f in input_files_all:
        # 读取事件数据
        ev_data = eventvision.read_dataset(f).data
        # 去除孤立像素
        ev_data_filt = remove_isolated_pixels(ev_data, eps=3, min_samples=20)[0]

        # 确保事件流的时间戳是单调递增的
        if len(event_data) > 0:
            ts_start_0 = event_data[-1].ts  # 获取上一个事件的时间戳

            for i in range(len(ev_data)):
                ev_data[i].ts += ts_start_0  # 更新当前事件的时间戳

        if len(event_data_filt) > 0:
            ts_start_1 = event_data_filt[-1].ts  # 获取上一个过滤后事件的时间戳

            for i in range(len(ev_data_filt)):
                ev_data_filt[i].ts += ts_start_1  # 更新当前过滤后事件的时间戳

        # 将当前文件的事件数据添加到总事件数据列表中
        event_data.extend(ev_data)
        event_data_filt.extend(ev_data_filt)

        print('事件流长度:', len(ev_data), len(ev_data_filt))

    # 绘制单个事件序列的时间曲面
    visualise_time_surface_for_event_stream(N_1, tau_1, r_1, ev.width, ev.height, ev.data)

    # 训练第一层的时间曲面原型
    C_1 = initialise_time_surface_prototypes(N_1, tau_1, r_1, ev.width, ev.height, event_data_filt, plot=True)  # 初始化（随机）

    train_layer(C_1, N_1, tau_1, r_1, ev.width, ev.height, event_data_filt, num_polarities=2, layer_number=1, plot=True)

    # 训练第二层的时间曲面原型
    # 使用第一层训练得到的特征生成第二层的事件数据
    event_data_2 = generate_layer_outputs(num_polarities=2, features=C_1, tau=tau_1, r=r_1, width=ev.width,
                                          height=ev.height, events=event_data_filt)

    C_2 = initialise_time_surface_prototypes(N_2, tau_2, r_2, ev.width, ev.height, event_data_2, plot=True)

    train_layer(C_2, N_2, tau_2, r_2, ev.width, ev.height, event_data_2, num_polarities=N_1, layer_number=2, plot=True)

    # 训练第三层的时间曲面原型
    # 使用第二层训练得到的特征生成第三层的事件数据
    event_data_3 = generate_layer_outputs(num_polarities=N_1, features=C_2, tau=tau_2, r=r_2, width=ev.width,
                                          height=ev.height, events=event_data_2)

    C_3 = initialise_time_surface_prototypes(N_3, tau_3, r_3, ev.width, ev.height, event_data_3, plot=True)

    train_layer(C_3, N_3, tau_3, r_3, ev.width, ev.height, event_data_3, num_polarities=N_2, layer_number=3, plot=True)

    ev.data = remove_isolated_pixels(ev.data, eps=3, min_samples=20)[0]
    ev2 = generate_layer_outputs(num_polarities=2, features=C_1, tau=tau_1, r=r_1, width=ev.width,
                                          height=ev.height, events=ev.data)
    ev3 = generate_layer_outputs(num_polarities=N_1, features=C_2, tau=tau_2, r=r_2, width=ev.width,
                                          height=ev.height, events=ev2)
    ev4 = generate_layer_outputs(num_polarities=N_2, features=C_3, tau=tau_3, r=r_3, width=ev.width,
                                          height=ev.height, events=ev3)
    plot_event_data(ev4, ev.width, ev.height)


if __name__ == '__main__':
    main()