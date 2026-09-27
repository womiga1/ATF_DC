import torch
import torch.nn.functional as F

def wavelet_1d_transform(x, filter):
    """
    对输入信号进行一维小波变换
    
    Args:
        x: 输入信号，形状为 [batch_size, channels, length]
        filter: 小波滤波器，形状为 [2, channels, filter_length] 或 [2, 1, filter_length]
            filter[0] 是低通滤波器，filter[1] 是高通滤波器
    
    Returns:
        变换后的信号，形状为 [batch_size, channels, 2, length//2]
        其中第三维的索引0对应低频部分，索引1对应高频部分
    """
    batch_size, channels, length = x.shape
    
    # 确保长度是偶数
    if length % 2 > 0:
        x = F.pad(x, (0, 1))
        length += 1
    
    # 分离低通和高通滤波器
    low_filter = filter[0]  # 可能是 [channels, filter_length] 或 [1, filter_length]
    high_filter = filter[1]  # 可能是 [channels, filter_length] 或 [1, filter_length]
    
    # 检查滤波器形状并适配
    if low_filter.dim() == 1:
        # 如果滤波器是一维的，扩展为 [1, filter_length]
        low_filter = low_filter.unsqueeze(0)
        high_filter = high_filter.unsqueeze(0)
    
    # 应用滤波器并下采样
    # 使用stride=2实现下采样
    filter_size = low_filter.size(-1)
    padding = (filter_size - 1) // 2
    
    # 创建每个通道的滤波器
    # 从[channels或1, filter_length]变为[channels, 1, filter_length]
    filters_low = torch.zeros(channels, 1, filter_size, device=x.device, dtype=x.dtype)
    filters_high = torch.zeros(channels, 1, filter_size, device=x.device, dtype=x.dtype)
    
    if low_filter.size(0) == 1:
        # 如果只有一个滤波器，复制到所有通道
        for i in range(channels):
            filters_low[i, 0] = low_filter[0].to(x.dtype)
            filters_high[i, 0] = high_filter[0].to(x.dtype)
    else:
        # 如果每个通道有一个滤波器
        for i in range(channels):
            filters_low[i, 0] = low_filter[i].to(x.dtype)
            filters_high[i, 0] = high_filter[i].to(x.dtype)
    
    low_freq = F.conv1d(x, filters_low, stride=2, groups=channels, padding=padding)
    high_freq = F.conv1d(x, filters_high, stride=2, groups=channels, padding=padding)
    
    # 将低频和高频部分堆叠在一起
    # [batch_size, channels, 2, length//2]
    result = torch.stack([low_freq, high_freq], dim=2)
    
    return result

def inverse_1d_wavelet_transform(x, filter):
    """
    对小波变换后的信号进行逆变换
    
    Args:
        x: 小波变换后的信号，形状为 [batch_size, channels, 2, length]
            其中第三维的索引0对应低频部分，索引1对应高频部分
        filter: 小波重构滤波器，形状为 [2, channels, filter_length] 或 [2, 1, filter_length]
            filter[0] 是低通重构滤波器，filter[1] 是高通重构滤波器
    
    Returns:
        重构后的信号，形状为 [batch_size, channels, length*2]
    """
    batch_size, channels, _, length = x.shape
    
    # 分离低频和高频部分
    low_freq = x[:, :, 0]  # [batch_size, channels, length]
    high_freq = x[:, :, 1]  # [batch_size, channels, length]
    
    # 分离低通和高通重构滤波器
    low_filter = filter[0]  # 可能是 [channels, filter_length] 或 [1, filter_length]
    high_filter = filter[1]  # 可能是 [channels, filter_length] 或 [1, filter_length]
    
    # 检查滤波器形状并适配
    if low_filter.dim() == 1:
        # 如果滤波器是一维的，扩展为 [1, filter_length]
        low_filter = low_filter.unsqueeze(0)
        high_filter = high_filter.unsqueeze(0)
    
    # 上采样并应用重构滤波器
    # 使用转置卷积实现上采样
    low_filter_size = low_filter.size(-1)
    high_filter_size = high_filter.size(-1)
    low_padding = low_filter_size // 2
    high_padding = high_filter_size // 2
    
    # 创建每个通道的滤波器
    # 从[channels或1, filter_length]变为[channels, 1, filter_length]
    filters_low = torch.zeros(channels, 1, low_filter_size, device=low_freq.device, dtype=low_freq.dtype)
    filters_high = torch.zeros(channels, 1, high_filter_size, device=high_freq.device, dtype=high_freq.dtype)
    
    if low_filter.size(0) == 1:
        # 如果只有一个滤波器，复制到所有通道
        for i in range(channels):
            filters_low[i, 0] = low_filter[0].to(low_freq.dtype)
            filters_high[i, 0] = high_filter[0].to(high_freq.dtype)
    else:
        # 如果每个通道有一个滤波器
        for i in range(channels):
            filters_low[i, 0] = low_filter[i].to(low_freq.dtype)
            filters_high[i, 0] = high_filter[i].to(high_freq.dtype)
    
    low_upsampled = F.conv_transpose1d(low_freq, filters_low, stride=2, groups=channels, padding=low_padding)
    high_upsampled = F.conv_transpose1d(high_freq, filters_high, stride=2, groups=channels, padding=high_padding)
    
    # 将低频和高频部分相加得到重构信号
    reconstructed = low_upsampled + high_upsampled
    
    # 确保输出长度正确
    target_length = length * 2
    if reconstructed.size(-1) > target_length:
        reconstructed = reconstructed[:, :, :target_length]
    
    return reconstructed