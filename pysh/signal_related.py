import numpy as np
from scipy import signal
from pysh.figure_related import (
    plot_filtering_steps,
    visualize_sample_processing
)

def preprocess_data(data_list, channel, fs, visualize_enable=0, visualize_channel=None):
    """
    预处理数据，确保所有样本具有相同的长度，并进行滤波处理

    参数:
    data_list: 数据列表
    channel: 通道数
    fs: 采样率 (Hz)
    visualize_enable: 是否启用可视化
    visualize_channel: 要可视化的通道索引，如果为None则可视化所有通道

    返回:
    filtered_data: 滤波后的数据列表
    """
    filtered_data = []

    for i, data in enumerate(data_list):
        # 应用滤波处理
        filtered = filter_mcg_signal(data, fs, visualize_enable)
        filtered_data.append(filtered)
        
        # 只对第一个样本进行可视化
        if i == 0 and visualize_enable == 1:
            visualize_sample_processing(data, filtered, channel, fs, visualize_channel)

    return filtered_data

def filter_mcg_signal(raw_data, fs=1000, visualize_enable=0):
    """
    对心磁信号进行滤波处理

    参数:
    raw_data: 原始数据矩阵，形状为 [channel, data_length]
    fs: 采样率 (Hz)
    visualize_enable: 是否启用可视化

    返回:
    处理后的数据矩阵
    """
    chn = raw_data.shape[0]
    data_length = raw_data.shape[1]

    # 初始化输出数组
    filter_data1 = np.zeros((chn, data_length))
    med_data = np.zeros((chn, data_length))
    filter_data2 = np.zeros((chn, data_length))

    # FIR滤波器设计参数（等效MATLAB Steepness 0.85，StopbandAttenuation 60dB）
    numtaps = 1001  # 滤波器阶数，越高过渡带越陡
    nyq = fs / 2
    fir_bandpass = signal.firwin(numtaps, [1 / nyq, 40 / nyq], pass_zero=False, window='hamming')

    for i in range(chn):
        # 1-40Hz FIR带通滤波
        filter_data1[i, :] = signal.filtfilt(fir_bandpass, [1], raw_data[i, :])

        # 300 窗口中值滤波
        med_data[i, :] = signal.medfilt(filter_data1[i, :], kernel_size=301)

        # 去除基线漂移
        filter_data2[i, :] = filter_data1[i, :] - med_data[i, :]
        
    # 绘制滤波处理各个步骤的结果
    if visualize_enable == 1:
        plot_filtering_steps(raw_data, filter_data1, med_data, filter_data2)

    return filter_data2

def stack_average_signals(filtered_data, fs=1000, visualize_enable=0):
    """
    对心磁信号进行叠加平均处理
    
    参数:
    filtered_data: 滤波后的数据矩阵 [channel, data_length]
    fs: 采样率 (Hz)
    visualize_enable: 是否启用可视化
    
    返回:
    stack_average: 叠加平均后的数据矩阵 [channel, window_length]
    """
    # 使用第29通道数据（索引为28）进行R波检测
    r_channel = filtered_data[28, :]

    # 使用scipy的find_peaks函数找到R波峰值
    peaks, _ = signal.find_peaks(r_channel, distance=int(fs * 0.5))
    
    # 每个周期取R波峰前fs/3个点和后2fs/3个点
    d = int(0.333 * fs)
    window_length = 3 * d + 1
    
    # 初始化用于存放叠加平均结果的矩阵
    stack_average = np.zeros((filtered_data.shape[0], window_length))
    
    # 对每个通道进行叠加平均
    for i in range(filtered_data.shape[0]):
        heartbeat_segments = []
        # 从第二个R波到倒数第二个R波进行叠加
        for j in range(1, len(peaks)-1):
            start_idx = peaks[j] - d
            end_idx = peaks[j] + 2 * d + 1
            if start_idx >= 0 and end_idx <= filtered_data.shape[1]:
                segment = filtered_data[i, start_idx:end_idx]
                heartbeat_segments.append(segment)
        
        # 计算平均值
        if heartbeat_segments:
            stack_average[i, :] = np.mean(heartbeat_segments, axis=0)
    
    return stack_average
