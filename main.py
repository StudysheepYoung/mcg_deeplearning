import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
from scipy import signal
import matplotlib.gridspec as gridspec
from pysh.figure_related import (
    plot_signals,
    visualize_compare_signals,
    visualize_multi_channels,
    plot_butterfly_diagram,
    plot_filtering_steps,
    stack_average_signals,
    visualize_preprocessing_results,
    plot_all_channels_butterfly
)

def preprocess_data(data_list, target_length=50000):
    """
    预处理数据，确保所有样本具有相同的长度，并进行滤波处理
    
    参数:
    data_list: 数据列表
    target_length: 目标长度
    
    返回:
    processed_data: 处理后的数据列表
    filtered_data: 滤波后的数据列表
    """
    processed_data = []
    filtered_data = []
    
    for data in data_list:
        # 调整数据长度
        current_length = data.shape[1]
        if current_length > target_length:
            # 如果数据太长，截断
            processed = data[:, :target_length]
        elif current_length < target_length:
            # 如果数据太短，用0填充
            padding = np.zeros((data.shape[0], target_length - current_length))
            processed = np.concatenate([data, padding], axis=1)
        else:
            processed = data
            
        processed_data.append(processed)
        
        # 应用滤波处理
        filtered = filter_mcg_signal(processed)
        filtered_data.append(filtered)
        
    return processed_data, filtered_data

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
        label = int(filename.split("/")[-4][-1])
        # 读取数据
        data = np.fromfile(filename, dtype=np.float32)
        
        # 去除包头
        data = data[header_size // 4:]  # 每个float为4字节，因此除以4
        
        # 计算采集时长
        time = len(data) / (channel * fs)  # 采集时长 (秒)
        
        # 初始化raw_data矩阵
        raw_data = np.zeros((channel, int(time * fs)))
        
        # 重组数据，将其按通道分配
        for i in range(int(time)):
            for j in range(channel):
                for g in range(1000):
                    raw_data[j, i * fs + g] = data[i * channel * fs + j * 1000 + g]
        
        return raw_data, label
    
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return None

def plot_signals(raw_data, channel):
    """
    绘制原始信号
    
    参数:
    raw_data: 数据矩阵
    channel: 通道数
    """
    plt.figure(figsize=(12, 6))
    
    # 使用jet颜色映射
    colors = plt.cm.jet(np.linspace(0, 1, channel))
    
    # 绘制每个通道的数据
    for i in range(channel):
        plt.plot(raw_data[i, :], color=colors[i], linewidth=1)
    
    plt.title('Original MCG Signal')
    plt.xlabel('Data Points')
    plt.ylabel('Magnitude/pT')
    plt.grid(True)
    plt.savefig('figure/original_signals.png')
    plt.close()

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
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(data_loader):.4f}')

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

def filter_mcg_signal(raw_data, fs=1000):
    """
    对心磁信号进行滤波处理
    
    参数:
    raw_data: 原始数据矩阵，形状为 [channel, data_length]
    fs: 采样率 (Hz)
    
    返回:
    处理后的数据矩阵
    """
    chn = raw_data.shape[0]
    data_length = raw_data.shape[1]
    
    # 初始化输出数组
    filter_data1 = np.zeros((chn, data_length))
    med_data = np.zeros((chn, data_length))
    filter_data2 = np.zeros((chn, data_length))
    
    # 对每个通道进行处理
    for i in range(chn):
        # 1-40Hz 带通滤波
        filter_data1[i, :] = signal.filtfilt(
            *signal.butter(4, [1, 40], btype='bandpass', fs=fs),
            raw_data[i, :]
        )
        
        # 300 窗口中值滤波
        med_data[i, :] = signal.medfilt(filter_data1[i, :], kernel_size=301)
        
        # 去除基线漂移
        filter_data2[i, :] = filter_data1[i, :] - med_data[i, :]
    
    # 绘制滤波处理各个步骤的结果
    plot_filtering_steps(raw_data, filter_data1, med_data, filter_data2)
    
    return filter_data2

def main():
    # 参数设置
    base_dir = '/Users/luckyyoung/Desktop/心磁/MCGdata'
    header_size = 2048  # 包头大小（字节）
    channel = 36  # 通道数
    fs = 1000  # 采样率 (Hz)
    target_length = 50000  # 目标数据长度
    
    # 加载所有数据
    print("正在加载数据...")
    data_list, label_list = load_all_data(base_dir, header_size, channel, fs)
    print(f"共加载了 {len(data_list)} 个样本")
    
    if data_list:
        # 预处理数据
        print("正在预处理数据...")
        processed_data, filtered_data = preprocess_data(data_list, target_length)
        
        # 可视化第一个样本的信号
        if len(processed_data) > 0:
            # 可视化预处理结果
            visualize_preprocessing_results(processed_data[0], filtered_data[0], channel)
            
            # 进行叠加平均处理
            print("正在进行叠加平均处理...")
            stack_average = stack_average_signals(filtered_data[0], fs)
            
            # 绘制所有通道的蝴蝶图
            print("正在生成蝴蝶图...")
            # 保存整体蝴蝶图
            plot_butterfly_diagram(stack_average, 'butterfly_diagram_all.png')
            # 保存所有通道的蝴蝶图
            plot_all_channels_butterfly(stack_average, channel)
            
            print("所有可视化图表已保存")
        
        # # 创建数据集和数据加载器
        # dataset = MCGDataset(filtered_data, label_list)
        # data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # # 创建模型
        # model = MCGResNet(input_channels=channel)
        
        # # 训练模型
        # print("开始训练模型...")
        # train_model(model, data_loader)
        
        # # 保存模型
        # torch.save(model.state_dict(), 'mcg_resnet.pth')
        # print("模型已保存为 mcg_resnet.pth")

if __name__ == "__main__":
    main()
