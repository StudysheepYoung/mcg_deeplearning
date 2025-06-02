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
import math

def read_bfd_file(filename, header_size, channel, fs):
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
        label_char = category_part[-1]

        # 确保提取的字符是数字
        if not label_char.isdigit():
            raise ValueError(f"无法从路径提取有效标签: {filename}")

        # 转换为整数标签
        label = int(label_char)

        # 读取数据
        data = np.fromfile(filename, dtype=np.float32)
        print(f"文件 {filename} 原始数据长度: {len(data)}")

        # 去除包头
        data = data[header_size // 4:]  # 每个float为4字节，因此除以4
        print(f"去除包头后数据长度: {len(data)}")

        # 计算实际数据长度
        total_samples = len(data) // channel
        if len(data) % channel != 0:
            data = data[:total_samples * channel]  # 确保数据长度是通道数的整数倍

        print(f"每个通道的采样点数: {total_samples}")

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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class MCGCNNTransformer(nn.Module):
    def __init__(self, input_channels=36, num_classes=1, d_model=256, nhead=8, num_layers=3):
        super(MCGCNNTransformer, self).__init__()
        
        # CNN部分用于特征提取
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(128, d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Transformer部分
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead) for _ in range(num_layers)
        ])
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # CNN特征提取
        x = self.cnn(x)  # [batch, d_model, seq_len]
        
        # 调整维度顺序以适应Transformer
        x = x.permute(2, 0, 1)  # [seq_len, batch, d_model]
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer处理
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # 取序列的平均值作为最终特征
        x = x.mean(dim=0)  # [batch, d_model]
        
        # 输出层
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

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

        avg_loss = total_loss / len(data_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
        scheduler.step(avg_loss)

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
                # 查找.bfd文件
                bfd_files = glob(os.path.join(date_dir, "*.BFD"))
                for file in bfd_files:
                    result = read_bfd_file(file, header_size, channel, fs)
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
    
    # 加载所有数据
    print("正在加载数据...")
    data_list, label_list = load_all_data(base_dir, header_size, channel, fs)
    print(f"共加载了 {len(data_list)} 个样本")

    if data_list:
        # 打印每个样本的数据长度
        for i, data in enumerate(data_list):
            print(f"样本 {i+1} 的数据形状: {data.shape}")
        
        # 创建数据集和数据加载器
        dataset = MCGDataset(data_list, label_list)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # 创建模型
        model = MCGCNNTransformer(input_channels=channel)

        # 训练模型
        print("开始训练模型...")
        train_model(model, data_loader)

        # 保存模型
        torch.save(model.state_dict(), 'mcg_cnn_transformer.pth')
        print("模型已保存为 mcg_cnn_transformer.pth")


if __name__ == "__main__":
    main()