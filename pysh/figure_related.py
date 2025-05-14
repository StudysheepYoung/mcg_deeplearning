import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
import os

# 创建figure文件夹（如果不存在）
FIGURE_DIR = 'figure'
if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)

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
    plt.savefig(os.path.join(FIGURE_DIR, 'original_signals.png'))
    plt.close()

def visualize_compare_signals(raw_data, filtered_data, channel, save_path='signal_compare.png'):
    """
    可视化处理前后的信号对比
    
    参数:
    raw_data: 原始数据矩阵 [channel, data_length]
    filtered_data: 处理后的数据矩阵 [channel, filtered_length]
    channel: 通道数
    save_path: 保存图像路径
    """
    # 创建一个大的图形
    plt.figure(figsize=(20, 15))
    
    # 计算行列数
    rows = int(np.ceil(np.sqrt(channel)))
    cols = int(np.ceil(channel / rows))
    
    # 选择相同长度的数据进行比较
    min_length = min(raw_data.shape[1], filtered_data.shape[1])
    raw_segment = raw_data[:, :min_length]
    filtered_segment = filtered_data[:, :min_length]
    
    # 绘制每个通道的对比
    for ch in range(channel):
        plt.subplot(rows, cols, ch+1)
        plt.plot(raw_segment[ch, :], 'b-', alpha=0.5, label='Original Signal')
        plt.plot(filtered_segment[ch, :], 'r-', alpha=0.7, label='Filtered Signal')
        plt.title(f'Channel {ch+1}')
        plt.grid(True)
        
        # 只在第一个子图中显示图例
        if ch == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, save_path))
    plt.close()

def visualize_multi_channels(data, channel, title, save_path):
    """
    可视化多通道信号
    
    参数:
    data: 数据矩阵 [channel, data_length]
    channel: 通道数
    title: 图表标题
    save_path: 保存图像路径
    """
    # 计算行列数
    rows = int(np.ceil(np.sqrt(channel)))
    cols = int(np.ceil(channel / rows))
    
    # 创建图形
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(title, fontsize=16)
    
    # 使用GridSpec灵活排布子图
    gs = gridspec.GridSpec(rows, cols)
    
    # 绘制每个通道
    for ch in range(channel):
        ax = plt.subplot(gs[ch])
        ax.plot(data[ch, :])
        ax.set_title(f'Channel {ch+1}')
        ax.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间
    plt.savefig(os.path.join(FIGURE_DIR, save_path))
    plt.close()


def plot_butterfly_diagram(stack_average, save_path='butterfly_diagram.png'):
    """
    绘制蝴蝶图
    
    参数:
    stack_average: 叠加平均后的数据矩阵 [channel, window_length]
    save_path: 保存图像路径
    """
    plt.figure(figsize=(12, 8))
    
    # 使用jet颜色映射
    colors = plt.cm.jet(np.linspace(0, 1, stack_average.shape[0]))
    
    # 绘制每个通道的数据
    for i in range(stack_average.shape[0]):
        plt.plot(stack_average[i, :], color=colors[i], linewidth=1)
    
    plt.title('Averaged Signal (Butterfly Diagram)')
    plt.xlabel('Data Points')
    plt.ylabel('Magnitude/pT')
    plt.grid(True)
    plt.savefig(os.path.join(FIGURE_DIR, save_path))
    plt.close()

def plot_filtering_steps(raw_data, filter_data1, med_data, filter_data2, channel_idx=28):
    """
    绘制滤波处理各个步骤的结果
    
    参数:
    raw_data: 原始数据
    filter_data1: 带通滤波后的数据
    med_data: 中值滤波后的数据
    filter_data2: 最终处理结果
    channel_idx: 要显示的通道索引（默认29通道）
    """
    # 保存第29通道（用于R波检测的通道）的各个处理步骤结果
    # 原始信号
    plt.figure(figsize=(15, 6))
    plt.plot(raw_data[channel_idx, :], 'b-', linewidth=1)
    plt.title('Original Signal (Channel 29)')
    plt.xlabel('Sample Points')
    plt.ylabel('Magnitude/pT')
    plt.grid(True)
    plt.savefig(os.path.join(FIGURE_DIR, 'step1_original_signal.png'))
    plt.close()
    
    # 带通滤波后
    plt.figure(figsize=(15, 6))
    plt.plot(filter_data1[channel_idx, :], 'r-', linewidth=1)
    plt.title('After 1-40Hz Bandpass Filter')
    plt.xlabel('Sample Points')
    plt.ylabel('Magnitude/pT')
    plt.grid(True)
    plt.savefig(os.path.join(FIGURE_DIR, 'step2_bandpass_filtered.png'))
    plt.close()
    
    # 中值滤波后
    plt.figure(figsize=(15, 6))
    plt.plot(med_data[channel_idx, :], 'g-', linewidth=1)
    plt.title('After 300-point Median Filter')
    plt.xlabel('Sample Points')
    plt.ylabel('Magnitude/pT')
    plt.grid(True)
    plt.savefig(os.path.join(FIGURE_DIR, 'step3_median_filtered.png'))
    plt.close()
    
    # 最终结果
    plt.figure(figsize=(15, 6))
    plt.plot(filter_data2[channel_idx, :], 'k-', linewidth=1)
    plt.title('After Baseline Drift Removal')
    plt.xlabel('Sample Points')
    plt.ylabel('Magnitude/pT')
    plt.grid(True)
    plt.savefig(os.path.join(FIGURE_DIR, 'step4_final_result.png'))
    plt.close()
    
    # 同时保存组合图
    plt.figure(figsize=(15, 12))
    
    plt.subplot(4, 1, 1)
    plt.plot(raw_data[channel_idx, :], 'b-', linewidth=1)
    plt.title('Original Signal (Channel 29)')
    plt.xlabel('Sample Points')
    plt.ylabel('Magnitude/pT')
    plt.grid(True)
    
    plt.subplot(4, 1, 2)
    plt.plot(filter_data1[channel_idx, :], 'r-', linewidth=1)
    plt.title('After 1-40Hz Bandpass Filter')
    plt.xlabel('Sample Points')
    plt.ylabel('Magnitude/pT')
    plt.grid(True)
    
    plt.subplot(4, 1, 3)
    plt.plot(med_data[channel_idx, :], 'g-', linewidth=1)
    plt.title('After 300-point Median Filter')
    plt.xlabel('Sample Points')
    plt.ylabel('Magnitude/pT')
    plt.grid(True)
    
    plt.subplot(4, 1, 4)
    plt.plot(filter_data2[channel_idx, :], 'k-', linewidth=1)
    plt.title('After Baseline Drift Removal')
    plt.xlabel('Sample Points')
    plt.ylabel('Magnitude/pT')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'all_steps_combined.png'))
    plt.close()

def visualize_preprocessing_results(processed_data, filtered_data, channel):
    """
    可视化预处理结果
    
    参数:
    processed_data: 预处理后的数据
    filtered_data: 滤波后的数据
    channel: 通道数
    """
    print("正在生成可视化图表...")
    # 可视化原始信号
    visualize_multi_channels(
        processed_data, 
        channel, 
        "Original MCG Signal", 
        "raw_signals.png"
    )
    
    # 可视化滤波后信号
    visualize_multi_channels(
        filtered_data, 
        channel, 
        "Filtered MCG Signal", 
        "filtered_signals.png"
    )
    
    # 可视化对比
    visualize_compare_signals(
        processed_data, 
        filtered_data, 
        channel,
        "signal_comparison.png"
    )
    
    print("所有可视化图表已保存")

def plot_all_channels_butterfly(stack_average, channel, save_path='butterfly_diagram_all_channels.png'):
    """
    绘制所有通道的蝴蝶图，每个通道一个子图
    
    参数:
    stack_average: 叠加平均后的数据矩阵 [channel, window_length]
    channel: 通道数
    save_path: 保存图像路径
    """
    # 计算子图的行列数
    rows = int(np.ceil(np.sqrt(channel)))
    cols = int(np.ceil(channel / rows))
    
    # 创建一个大图，包含所有通道的子图
    plt.figure(figsize=(20, 15))
    plt.suptitle('Averaged Signals for All Channels', fontsize=16)
    
    # 绘制每个通道的子图
    for ch in range(channel):
        plt.subplot(rows, cols, ch+1)
        plt.plot(stack_average[ch, :], 'b-', linewidth=1)
        plt.title(f'Channel {ch+1}')
        plt.xlabel('Sample Points')
        plt.ylabel('Magnitude/pT')
        plt.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间
    plt.savefig(os.path.join(FIGURE_DIR, save_path))
    plt.close()

def visualize_sample_processing(raw_data, filtered_data, channel, fs, channel_idx=None):
    """
    可视化样本处理过程中的各个步骤
    
    参数:
    raw_data: 原始数据
    filtered_data: 滤波后的数据
    channel: 通道数
    fs: 采样率 (Hz)
    channel_idx: 要可视化的通道索引，如果为None则可视化所有通道
    """
    print("正在生成可视化图表...")
    
    # 可视化预处理结果
    visualize_preprocessing_results(raw_data, filtered_data, channel)
    
    print("所有可视化图表已保存")

