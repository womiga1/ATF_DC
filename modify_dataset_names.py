#!/usr/bin/env python3
"""
脚本用于修改pickle文件中的数据集名称
- MSL_ATF_UAD_plot_data.pkl: 将数据集名称从 'NAB' 改为 'MSL'
- SelfBuilt_ATF_UAD_plot_data.pkl: 将数据集名称从 'UCR' 改为 'SelfBuilt'
"""

import pickle
import os
import shutil
from datetime import datetime

def modify_dataset_name(file_path, new_dataset_name, backup=True):
    """
    修改pickle文件中的数据集名称
    
    Args:
        file_path (str): pickle文件路径
        new_dataset_name (str): 新的数据集名称
        backup (bool): 是否创建备份文件
    
    Returns:
        bool: 修改是否成功
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在 - {file_path}")
            return False
        
        # 创建备份
        if backup:
            backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(file_path, backup_path)
            print(f"已创建备份文件: {backup_path}")
        
        # 读取原始数据
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # 显示原始数据集名称
        original_dataset = data.get('dataset', 'Unknown')
        print(f"原始数据集名称: {original_dataset}")
        
        # 修改数据集名称
        data['dataset'] = new_dataset_name
        print(f"新数据集名称: {new_dataset_name}")
        
        # 保存修改后的数据
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"成功修改文件: {file_path}")
        return True
        
    except Exception as e:
        print(f"修改文件时出错 {file_path}: {str(e)}")
        return False

def verify_changes(file_path, expected_dataset_name):
    """
    验证修改是否成功
    
    Args:
        file_path (str): pickle文件路径
        expected_dataset_name (str): 期望的数据集名称
    
    Returns:
        bool: 验证是否通过
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        actual_dataset = data.get('dataset', 'Unknown')
        success = actual_dataset == expected_dataset_name
        
        print(f"验证 {os.path.basename(file_path)}:")
        print(f"  期望数据集名称: {expected_dataset_name}")
        print(f"  实际数据集名称: {actual_dataset}")
        print(f"  验证结果: {'通过' if success else '失败'}")
        
        return success
        
    except Exception as e:
        print(f"验证文件时出错 {file_path}: {str(e)}")
        return False

def main():
    """主函数"""
    print("开始修改pickle文件中的数据集名称...")
    print("=" * 50)
    
    # 定义文件路径和目标数据集名称
    files_to_modify = [
        {
            'path': '/home/lanping/newdisk/LanpingProject/DCdetector/ATF-UAD-main/results/MSL_ATF_UAD_plot_data.pkl',
            'new_name': 'MSL'
        },
        {
            'path': '/home/lanping/newdisk/LanpingProject/DCdetector/ATF-UAD-main/results/SelfBuilt_ATF_UAD_plot_data.pkl',
            'new_name': 'SelfBuilt'
        }
    ]
    
    success_count = 0
    
    # 修改每个文件
    for file_info in files_to_modify:
        file_path = file_info['path']
        new_name = file_info['new_name']
        
        print(f"\n处理文件: {os.path.basename(file_path)}")
        print("-" * 30)
        
        if modify_dataset_name(file_path, new_name):
            success_count += 1
        
        print()
    
    # 验证修改结果
    print("=" * 50)
    print("验证修改结果...")
    print("=" * 50)
    
    verification_success = 0
    for file_info in files_to_modify:
        file_path = file_info['path']
        expected_name = file_info['new_name']
        
        if verify_changes(file_path, expected_name):
            verification_success += 1
        print()
    
    # 总结
    print("=" * 50)
    print("修改总结:")
    print(f"  成功修改的文件数: {success_count}/{len(files_to_modify)}")
    print(f"  验证通过的文件数: {verification_success}/{len(files_to_modify)}")
    
    if success_count == len(files_to_modify) and verification_success == len(files_to_modify):
        print("  状态: 所有修改都成功完成！")
    else:
        print("  状态: 部分修改失败，请检查错误信息")

if __name__ == "__main__":
    main()