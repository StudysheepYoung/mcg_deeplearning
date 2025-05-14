import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
from scipy import signal
from pathlib import Path
import matplotlib.gridspec as gridspec
from pysh.figure_related import (
    plot_filtering_steps,
    visualize_sample_processing,
    plot_butterfly_diagram,
    plot_all_channels_butterfly
)
from pysh.signal_related import (
    preprocess_data,
    filter_mcg_signal,
    stack_average_signals
)

def read_base_date_file(filename, header_size, channel, fs):
    """
    读取心磁数据文件

    参数:
    filename: 数据文件路径
    header_size: 包头大小（字节）
    channel: 通道数
    fs: 采样率 (Hz)

    返回:
    raw_data: 处理后的数据矩阵
    """
    try:
        # 使用 pathlib.Path 处理路径，自动适应不同操作系统的分隔符
        path_parts = Path(filename).parts

        # 验证路径深度是否足够
        if len(path_parts) < 4:
            raise ValueError(f"路径结构不符合预期: {filename}")

        # 获取倒数第4个部分作为标签来源
        category_part = path_parts[-4]

        # 从该部分提取最后一个字符作为标签
        # 例如："类别1" → "1"，"类别2" → "2"
        label_char = category_part[-1]

        # 确保提取的字符是数字
        if not label_char.isdigit():
            raise ValueError(f"无法从路径提取有效标签: {filename}")

        # 转换为整数标签
        label = int(label_char)

        # 读取数据
        data = np.fromfile(filename, dtype=np.float32)

        # 去除包头
        data = data[header_size // 4:]  # 每个float为4字节，因此除以4

        # 计算实际数据长度
        total_samples = len(data) // channel
        if len(data) % channel != 0:
            data = data[:total_samples * channel]  # 确保数据长度是通道数的整数倍

        # 重组数据，将其按通道分配
        raw_data = np.zeros((channel, total_samples))

        # 使用与原代码相同的三重循环逻辑
        samples_per_block = 1000  # 保持原始的1000点假设
        total_blocks = total_samples // samples_per_block

        for i in range(total_blocks):
            for j in range(channel):
                for g in range(samples_per_block):
                    raw_data[j, i * samples_per_block + g] = data[
                        i * channel * samples_per_block + j * samples_per_block + g]

        # 处理剩余数据
        remaining_samples = total_samples % samples_per_block
        if remaining_samples > 0:
            start_idx = total_blocks * samples_per_block
            data_start_idx = total_blocks * channel * samples_per_block
            for j in range(channel):
                for g in range(remaining_samples):
                    raw_data[j, start_idx + g] = data[data_start_idx + j * remaining_samples + g]

        return raw_data, label

    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return None

class MCGDataset(Dataset):
    def __init__(self, data_list, label_list):
        self.data_list = [torch.FloatTensor(data) for data in data_list]
        self.label_list = [torch.FloatTensor([label]) for label in label_list]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx], self.label_list[idx]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class MCGResNet(nn.Module):
    def __init__(self, input_channels=36, num_classes=1):
        super(MCGResNet, self).__init__()

        # 初始卷积层，使用Conv1d代替Conv2d
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet层，使用Conv1d代替Conv2d
        self.layer1 = self.make_layer(64, 64, 2)
        self.layer2 = self.make_layer(64, 128, 2, stride=2)
        self.layer3 = self.make_layer(128, 256, 2, stride=2)
        self.layer4 = self.make_layer(256, 512, 2, stride=2)

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # 全连接层
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 输入x的形状应该是 [batch, channels, length]
        print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def train_model(model, data_loader, num_epochs=10):
    """
    训练模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch, label in data_loader:
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(data_loader):.4f}')


def load_all_data(base_dir, header_size, channel, fs):
    """
    加载所有数据文件
    """
    data_list = []
    label_list = []

    # 遍历所有类别文件夹
    for class_dir in glob(os.path.join(base_dir, "类别*")):
        # 遍历每个类别下的所有患者文件夹
        for patient_dir in glob(os.path.join(class_dir, "*")):
            # 遍历每个患者文件夹下的所有日期文件夹
            for date_dir in glob(os.path.join(patient_dir, "*")):
                # 查找.baseDate文件
                base_date_files = glob(os.path.join(date_dir, "*.baseDate"))
                for file in base_date_files:
                    result = read_base_date_file(file, header_size, channel, fs)
                    if result is not None:
                        data, label = result
                        data_list.append(data)
                        label_list.append(label)

    return data_list, label_list

def main():
    # 参数设置
    base_dir = "/Users/luckyyoung/Desktop/心磁/MCGdata"
    header_size = 2048  # 包头大小（字节）
    channel = 36  # 通道数
    fs = 1000  # 采样率 (Hz)
    visualize_channel = 28  # 要可视化的通道索引（29通道）
    visualize_enable = 0  # 是否启用可视化
    
    # 加载所有数据
    print("正在加载数据...")
    data_list, label_list = load_all_data(base_dir, header_size, channel, fs)
    print(f"共加载了 {len(data_list)} 个样本")

    if data_list:
        # 预处理数据
        print("正在预处理数据...")
        filtered_data = preprocess_data(data_list, channel, fs, visualize_enable, visualize_channel)
        
        # 对每个样本进行叠加平均处理
        print("正在进行叠加平均处理...")
        stacked_data = []
        for i, data in enumerate(filtered_data):
            stacked = stack_average_signals(data, fs, visualize_enable)
            stacked_data.append(stacked)
            
            # 如果启用了可视化，为第一个样本生成蝴蝶图
            if visualize_enable == 1 and i == 0:
                print("正在生成蝴蝶图...")
                # 保存整体蝴蝶图
                plot_butterfly_diagram(stacked, 'butterfly_diagram_all.png')
                # 保存所有通道的蝴蝶图
                plot_all_channels_butterfly(stacked, channel)
        
        print(f"完成叠加平均处理，每个样本的叠加平均结果形状为: {stacked_data[0].shape}")
        
        # 创建数据集和数据加载器
        dataset = MCGDataset(stacked_data, label_list)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # 创建模型
        model = MCGResNet(input_channels=channel)

        # 训练模型
        print("开始训练模型...")
        train_model(model, data_loader)

        # # 保存模型
        # torch.save(model.state_dict(), 'mcg_resnet.pth')
        # print("模型已保存为 mcg_resnet.pth")


if __name__ == "__main__":
    main()