import numpy as np
from scipy.signal import bandpass, median_filter, butter, filtfilt, medfilt

def preprocess_data(data_list, channel, fs, visualize_channel=None, visualize_enable=0):
    """
    预处理数据
    
    参数:
    data_list: 数据列表
    channel: 通道数
    fs: 采样率
    visualize_channel: 要可视化的通道索引
    visualize_enable: 是否启用可视化
    """
    # 数据预处理
    processed_data = np.array(data_list)
    
    # 带通滤波
    filter_data1 = bandpass_filter(processed_data, fs)
    
    # 中值滤波
    med_data = median_filter(filter_data1)
    
    # 去除基线漂移
    filter_data2 = remove_baseline_drift(med_data)
    
    # 可视化处理过程
    if visualize_enable == 1:
        from figure_related import plot_filtering_steps
        plot_filtering_steps(processed_data, filter_data1, med_data, filter_data2, visualize_channel)
    
    return filter_data2

def bandpass_filter(data, fs, lowcut=1, highcut=40):
    """
    带通滤波
    
    参数:
    data: 输入数据
    fs: 采样率
    lowcut: 低频截止频率
    highcut: 高频截止频率
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, data, axis=1)

def remove_baseline_drift(data, window_size=300):
    """
    去除基线漂移
    
    参数:
    data: 输入数据
    window_size: 中值滤波窗口大小
    """
    baseline = medfilt(data, kernel_size=(1, window_size))
    return data - baseline

def read_data():
    """
    读取数据
    """
    # TODO: 实现数据读取逻辑
    return np.random.randn(64, 10000)  # 临时返回随机数据用于测试

def main():
    # 参数设置
    channel = 64  # 通道数
    fs = 1000     # 采样率
    
    # 读取数据
    data_list = read_data()
    
    # 数据预处理
    filtered_data = preprocess_data(data_list, channel, fs, visualize_channel=28, visualize_enable=1)
    
    # 叠加平均处理
    from figure_related import stack_average_signals, visualize_stack_average
    stack_average = stack_average_signals(filtered_data, fs)
    
    # 可视化叠加平均结果
    visualize_stack_average(stack_average, channel)

if __name__ == "__main__":
    main()