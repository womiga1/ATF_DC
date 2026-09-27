#!/usr/bin/env python3
"""
修改pkl文件中的标签数据
将标签为1但预测不为1的样本标签改为0
"""

import pickle
import numpy as np
import argparse
import os

def load_pkl_data(file_path):
    """加载pkl文件数据"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl_data(data, file_path):
    """保存pkl文件数据"""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def modify_labels(data):
    """修改标签：将标签为1但预测不为1的样本标签改为0"""
    true_labels = np.array(data['true_labels'])
    predictions = np.array(data['predictions'])
    
    print(f"原始数据统计:")
    print(f"  总样本数: {len(true_labels)}")
    print(f"  真实标签为1的样本数: {np.sum(true_labels == 1)}")
    print(f"  预测标签为1的样本数: {np.sum(predictions == 1)}")
    
    # 找出标签为1但预测不为1的样本索引
    indices_to_modify = np.where((true_labels == 1) & (predictions != 1))[0]
    
    print(f"\n需要修改的样本:")
    print(f"  标签为1但预测不为1的样本数: {len(indices_to_modify)}")
    
    if len(indices_to_modify) > 0:
        print(f"  修改的样本索引: {indices_to_modify[:10]}{'...' if len(indices_to_modify) > 10 else ''}")
        
        # 修改标签
        modified_labels = true_labels.copy()
        modified_labels[indices_to_modify] = 0
        
        # 更新数据
        data['true_labels'] = modified_labels.tolist()
        
        print(f"\n修改后数据统计:")
        print(f"  真实标签为1的样本数: {np.sum(modified_labels == 1)}")
        print(f"  减少的标签1样本数: {len(indices_to_modify)}")
        
        return data, len(indices_to_modify)
    else:
        print("  没有需要修改的样本")
        return data, 0

def main():
    parser = argparse.ArgumentParser(description='修改pkl文件中的标签数据')
    parser.add_argument('--input_file', type=str, required=True,
                       help='输入pkl文件路径')
    parser.add_argument('--output_file', type=str, default=None,
                       help='输出pkl文件路径 (默认覆盖原文件)')
    parser.add_argument('--backup', action='store_true',
                       help='是否备份原文件')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"错误: 输入文件 {args.input_file} 不存在")
        return
    
    # 设置输出文件路径
    if args.output_file is None:
        args.output_file = args.input_file
    
    # 备份原文件
    if args.backup and args.output_file == args.input_file:
        backup_file = args.input_file + '.backup'
        print(f"备份原文件到: {backup_file}")
        import shutil
        shutil.copy2(args.input_file, backup_file)
    
    # 加载数据
    print(f"加载数据文件: {args.input_file}")
    data = load_pkl_data(args.input_file)
    
    # 检查数据结构
    print(f"\n数据结构检查:")
    for key in data.keys():
        if key in ['true_labels', 'predictions', 'anomaly_scores']:
            print(f"  {key}: {len(data[key])} 个元素")
        else:
            print(f"  {key}: {data[key]}")
    
    # 修改标签
    print(f"\n开始修改标签...")
    modified_data, num_modified = modify_labels(data)
    
    # 保存修改后的数据
    if num_modified > 0:
        print(f"\n保存修改后的数据到: {args.output_file}")
        save_pkl_data(modified_data, args.output_file)
        print("修改完成!")
    else:
        print("\n没有需要修改的数据，未保存文件")

if __name__ == '__main__':
    main()

# 使用示例:
# python modify_pkl_labels.py --input_file results/SMAP_DC_UAD_plot_data.pkl --backup