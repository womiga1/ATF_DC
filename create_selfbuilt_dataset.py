import numpy as np
import random
import os

def read_time_series(file_path):
    """读取时间序列文件"""
    with open(file_path, 'r') as f:
        data = [float(line.strip()) for line in f.readlines()]
    return np.array(data)

def reshape_to_multivariate(data, n_features=128):
    """将单变量时间序列重塑为多变量格式 (len, 32)"""
    # 确保数据长度能被n_features整除
    n_samples = len(data) // n_features
    reshaped_data = data[:n_samples * n_features].reshape(n_samples, n_features)
    return reshaped_data

def create_random_segments(data, min_length=20, max_length=50):
    """从数据中随机选择片段"""
    segments = []
    labels = []
    
    total_length = len(data)
    current_pos = 0
    
    while current_pos < total_length:
        # 随机选择片段长度
        segment_length = random.randint(min_length, max_length)
        
        # 确保不超出数据范围
        if current_pos + segment_length > total_length:
            segment_length = total_length - current_pos
        
        if segment_length > 0:
            segments.append(data[current_pos:current_pos + segment_length])
            current_pos += segment_length
        else:
            break
    
    return segments

def main():
    # 设置随机种子以确保可重现性
    random.seed(42)
    np.random.seed(42)
    
    # 读取数据
    print("读取时间序列数据...")
    lou_data = read_time_series('/home/lanping/newdisk/LanpingProject/DCdetector/ATF_New/lou.txt')
    bulou_data = read_time_series('/home/lanping/newdisk/LanpingProject/DCdetector/ATF_New/bulou.txt')
    
    print(f"lou数据长度: {len(lou_data)}")
    print(f"bulou数据长度: {len(bulou_data)}")
    
    # 重塑数据为 (len, 32) 格式
    print("重塑数据为多变量格式...")
    lou_reshaped = reshape_to_multivariate(lou_data)
    bulou_reshaped = reshape_to_multivariate(bulou_data)
    
    print(f"lou重塑后形状: {lou_reshaped.shape}")
    print(f"bulou重塑后形状: {bulou_reshaped.shape}")
    
    # 分割数据
    bulou_half = len(bulou_reshaped) // 2
    lou_half = len(lou_reshaped) // 2
    
    # 创建训练数据：bulou的前半段
    train_data = bulou_reshaped[:bulou_half]
    print(f"训练数据形状: {train_data.shape}")
    
    # 创建测试数据：bulou的后半段和lou的前半段随机交叉拼接
    print("创建测试数据...")
    bulou_second_half = bulou_reshaped[bulou_half:]
    lou_first_half = lou_reshaped[:lou_half]
    
    # 随机选择片段
    bulou_segments = create_random_segments(bulou_second_half, 20, 50)
    lou_segments = create_random_segments(lou_first_half, 20, 50)
    
    # 交叉拼接片段
    test_segments = []
    test_labels = []
    
    # 确定最小片段数量以进行交叉
    min_segments = min(len(bulou_segments), len(lou_segments))
    
    for i in range(min_segments):
        # 随机决定顺序
        if random.random() < 0.5:
            # bulou先，lou后
            test_segments.append(bulou_segments[i])
            test_segments.append(lou_segments[i])
            test_labels.extend([0] * len(bulou_segments[i]))  # bulou标签为0
            test_labels.extend([1] * len(lou_segments[i]))    # lou标签为1
        else:
            # lou先，bulou后
            test_segments.append(lou_segments[i])
            test_segments.append(bulou_segments[i])
            test_labels.extend([1] * len(lou_segments[i]))    # lou标签为1
            test_labels.extend([0] * len(bulou_segments[i]))  # bulou标签为0
    
    # 添加剩余的片段
    for i in range(min_segments, len(bulou_segments)):
        test_segments.append(bulou_segments[i])
        test_labels.extend([0] * len(bulou_segments[i]))
    
    for i in range(min_segments, len(lou_segments)):
        test_segments.append(lou_segments[i])
        test_labels.extend([1] * len(lou_segments[i]))
    
    # 拼接所有测试片段
    test_data = np.vstack(test_segments)
    test_labels = np.array(test_labels)
    
    print(f"测试数据形状: {test_data.shape}")
    print(f"测试标签形状: {test_labels.shape}")
    print(f"标签分布 - 0(bulou): {np.sum(test_labels == 0)}, 1(lou): {np.sum(test_labels == 1)}")
    
    # 创建输出目录
    output_dir = '/home/lanping/newdisk/LanpingProject/DCdetector/ATF_New/dataset/SelfBuilt'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存数据
    print("保存数据...")
    np.save(os.path.join(output_dir, 'train.npy'), train_data)
    np.save(os.path.join(output_dir, 'test.npy'), test_data)
    np.save(os.path.join(output_dir, 'labels.npy'), test_labels)
    
    print("数据处理完成！")
    print(f"文件已保存到: {output_dir}")
    print(f"- train.npy: {train_data.shape}")
    print(f"- test.npy: {test_data.shape}")
    print(f"- labels.npy: {test_labels.shape}")

if __name__ == "__main__":
    main()