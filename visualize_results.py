#!/usr/bin/env python3
"""
异常检测结果可视化脚本
用于生成多种图表来展示异常检测模型的效果
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import pandas as pd
import argparse
import os
from matplotlib.font_manager import FontProperties

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_plot_data(data_file):
    """加载绘图数据"""
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_roc_curve(data, save_path):
    """绘制ROC曲线"""
    true_labels = np.array(data['true_labels'])
    anomaly_scores = np.array(data['anomaly_scores'])
    
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(true_labels, anomaly_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {data["dataset"]} ({data["model"]})')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{data["dataset"]}_{data["model"]}_roc.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curve(data, save_path):
    """绘制Precision-Recall曲线"""
    true_labels = np.array(data['true_labels'])
    anomaly_scores = np.array(data['anomaly_scores'])
    
    # 计算PR曲线
    precision, recall, _ = precision_recall_curve(true_labels, anomaly_scores)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {data["dataset"]} ({data["model"]})')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{data["dataset"]}_{data["model"]}_pr.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_score_distribution(data, save_path):
    """绘制异常分数分布图"""
    true_labels = np.array(data['true_labels'])
    anomaly_scores = np.array(data['anomaly_scores'])
    threshold = data['threshold']
    
    # 分离正常和异常样本的分数
    normal_scores = anomaly_scores[true_labels == 0]
    anomaly_scores_pos = anomaly_scores[true_labels == 1]
    
    plt.figure(figsize=(10, 6))
    
    # 绘制分布直方图
    plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
    plt.hist(anomaly_scores_pos, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
    
    # 添加阈值线
    plt.axvline(x=threshold, color='green', linestyle='--', linewidth=2, 
                label=f'Threshold = {threshold:.3f}')
    
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title(f'Anomaly Score Distribution - {data["dataset"]} ({data["model"]})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{data["dataset"]}_{data["model"]}_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(data, save_path):
    """绘制混淆矩阵热力图"""
    true_labels = np.array(data['true_labels'])
    predictions = np.array(data['predictions'])
    
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {data["dataset"]} ({data["model"]})')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{data["dataset"]}_{data["model"]}_confusion_matrix.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_time_series_detection(
    data,
    save_path,
    max_points=1000,
    num_groups=30,
    random_groups=True,
    seed=None,
    allow_overlap=True,
):
    """绘制时间序列检测结果图 - 分组显示
    参数:
        max_points: 每个窗口的最大点数
        num_groups: 需要绘制的窗口数量
        random_groups: 是否随机选择窗口（默认关闭，使用覆盖全局的确定性分段）
        seed: 随机种子（仅在 random_groups=True 时生效）
        allow_overlap: 随机窗口是否允许重叠
    """
    true_labels = np.array(data['true_labels'])
    predictions = np.array(data['predictions'])
    anomaly_scores = np.array(data['anomaly_scores'])
    threshold = data['threshold']
    
    total_length = len(true_labels)
    
    # 如果数据点太多，分成窗口
    if total_length > max_points:
        group_size = max_points

        if not random_groups:
            # 覆盖整个数据集的确定性分段
            step_size = max(1, (total_length - group_size) // (num_groups - 1))
            for group_idx in range(num_groups):
                start_idx = min(group_idx * step_size, total_length - group_size)
                end_idx = start_idx + group_size

                if start_idx >= total_length:
                    break
                if end_idx > total_length:
                    end_idx = total_length
                    start_idx = max(0, end_idx - group_size)

                group_true_labels = true_labels[start_idx:end_idx]
                group_predictions = predictions[start_idx:end_idx]
                group_anomaly_scores = anomaly_scores[start_idx:end_idx]

                _plot_single_group(
                    group_true_labels,
                    group_predictions,
                    group_anomaly_scores,
                    threshold,
                    data,
                    save_path,
                    group_idx + 1,
                    start_idx,
                    end_idx,
                )
        else:
            # 随机选择窗口
            rng = np.random.RandomState(seed) if seed is not None else np.random

            starts = []
            max_start = max(0, total_length - group_size)

            if allow_overlap:
                for _ in range(num_groups):
                    start_idx = int(rng.randint(0, max_start + 1))
                    starts.append(start_idx)
            else:
                # 简单的不重叠策略：以 group_size 为步长的候选起点中随机选择
                candidate_starts = list(range(0, max_start + 1, group_size))
                if len(candidate_starts) == 0:
                    candidate_starts = [0]
                selected = rng.choice(
                    candidate_starts,
                    size=min(num_groups, len(candidate_starts)),
                    replace=False,
                )
                starts.extend(int(s) for s in selected)

            for i, start_idx in enumerate(starts):
                end_idx = start_idx + group_size
                if end_idx > total_length:
                    end_idx = total_length
                    start_idx = max(0, end_idx - group_size)

                group_true_labels = true_labels[start_idx:end_idx]
                group_predictions = predictions[start_idx:end_idx]
                group_anomaly_scores = anomaly_scores[start_idx:end_idx]

                _plot_single_group(
                    group_true_labels,
                    group_predictions,
                    group_anomaly_scores,
                    threshold,
                    data,
                    save_path,
                    i + 1,
                    start_idx,
                    end_idx,
                )
    else:
        # 如果数据点不多，直接绘制单个图表
        _plot_single_group(
            true_labels,
            predictions,
            anomaly_scores,
            threshold,
            data,
            save_path,
            1,
            0,
            total_length,
        )

def _plot_single_group(true_labels, predictions, anomaly_scores, threshold, data, 
                      save_path, group_idx, start_idx, end_idx):
    """绘制单个数据组的时间序列图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # 上图：异常分数和阈值
    time_points = range(len(anomaly_scores))
    ax1.plot(time_points, anomaly_scores, color='blue', alpha=0.7, label='Anomaly Score')
    ax1.axhline(y=threshold, color='green', linestyle='--', linewidth=2, 
                label=f'Threshold = {threshold:.3f}')
    
    # 标记真实异常点
    anomaly_indices = np.where(true_labels == 1)[0]
    if len(anomaly_indices) > 0:
        ax1.scatter(anomaly_indices, anomaly_scores[anomaly_indices], 
                   color='red', s=20, alpha=0.8, label='True Anomalies')
    
    ax1.set_ylabel('Anomaly Score')
    ax1.set_title(f'Time Series Anomaly Detection - {data["dataset"]}\n'
                  f'Data Points: {start_idx} - {end_idx}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 下图：真实标签 vs 预测标签
    ax2.plot(time_points, true_labels, color='red', linewidth=2, label='True Labels', alpha=0.7)
    ax2.plot(time_points, predictions, color='blue', linewidth=2, label='Predictions', alpha=0.7)
    ax2.fill_between(time_points, 0, true_labels, color='red', alpha=0.3, label='True Anomalies')
    ax2.fill_between(time_points, 0, predictions, color='blue', alpha=0.3, label='Predicted Anomalies')
    
    ax2.set_xlabel('Time Points (Relative to Group)')
    ax2.set_ylabel('Label')
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存文件，添加组号后缀
    if group_idx == 1 and end_idx - start_idx == len(anomaly_scores):
        # 如果只有一组且是完整数据，使用原文件名
        filename = f'{data["dataset"]}_timeseries.png'
    else:
        # 多组数据，添加组号
        filename = f'{data["dataset"]}_timeseries_group{group_idx}.png'
    
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_summary(data, save_path):
    """绘制指标汇总图"""
    metrics = data['metrics']
    
    # 提取主要指标
    metric_names = ['precision', 'recall', 'f1', 'ROC/AUC']
    metric_values = [metrics.get(name, 0) for name in metric_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图：主要指标柱状图
    colors = ['skyblue', 'lightgreen', 'orange', 'pink']
    bars = ax1.bar(metric_names, metric_values, color=colors, alpha=0.8)
    ax1.set_ylabel('Score')
    ax1.set_title(f'Performance Metrics - {data["dataset"]} ({data["model"]})')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值标签
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 右图：混淆矩阵指标
    confusion_metrics = ['TP', 'TN', 'FP', 'FN']
    confusion_values = [metrics.get(name, 0) for name in confusion_metrics]
    
    colors2 = ['green', 'blue', 'orange', 'red']
    bars2 = ax2.bar(confusion_metrics, confusion_values, color=colors2, alpha=0.8)
    ax2.set_ylabel('Count')
    ax2.set_title('Confusion Matrix Metrics')
    ax2.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值标签
    for bar, value in zip(bars2, confusion_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(confusion_values)*0.01,
                f'{int(value)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{data["dataset"]}_{data["model"]}_metrics.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def generate_all_plots(
    data_file,
    output_dir='plots',
    max_points=1000,
    num_groups=30,
    random_groups=True,
    seed=None,
    allow_overlap=True,
):
    """生成所有可视化图表
    支持随机窗口选择的时间序列可视化
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    data = load_plot_data(data_file)
    
    print(f"正在为 {data['dataset']} 数据集 ({data['model']} 模型) 生成可视化图表...")
    
    # # 生成各种图表
    # plot_roc_curve(data, output_dir)
    # print("✓ ROC曲线已生成")
    
    # plot_precision_recall_curve(data, output_dir)
    # print("✓ Precision-Recall曲线已生成")
    
    # plot_score_distribution(data, output_dir)
    # print("✓ 异常分数分布图已生成")
    
    # plot_confusion_matrix(data, output_dir)
    # print("✓ 混淆矩阵已生成")
    
    plot_time_series_detection(
        data,
        output_dir,
        max_points=max_points,
        num_groups=num_groups,
        random_groups=random_groups,
        seed=seed,
        allow_overlap=allow_overlap,
    )
    print("✓ 时间序列检测结果图已生成")
    
    # plot_metrics_summary(data, output_dir)
    # print("✓ 指标汇总图已生成")
    
    print(f"\n所有图表已保存到: {output_dir}")
    print(f"数据集: {data['dataset']}, 模型: {data['model']}")
    print(f"主要指标: F1={data['metrics'].get('f1', 0):.3f}, "
          f"AUC={data['metrics'].get('ROC/AUC', 0):.3f}")

def main():
    parser = argparse.ArgumentParser(description='异常检测结果可视化')
    parser.add_argument(
        '--data_file', type=str, required=True, help='绘图数据文件路径 (.pkl)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='plots', help='输出目录 (默认: plots)'
    )

    # 新增：随机窗口相关参数
    parser.add_argument(
        '--max_points', type=int, default=1000, help='每个窗口的最大点数 (默认: 1000)'
    )
    parser.add_argument(
        '--num_groups', type=int, default=30, help='绘制的窗口数量 (默认: 30)'
    )
    parser.add_argument(
        '--random_groups', action='store_true', help='启用随机选择窗口'
    )
    parser.add_argument(
        '--seed', type=int, default=None, help='随机种子 (可选，配合 --random_groups)'
    )
    parser.add_argument(
        '--no_overlap', action='store_true', help='随机窗口不允许重叠'
    )

    args = parser.parse_args()

    if not os.path.exists(args.data_file):
        print(f"错误: 数据文件 {args.data_file} 不存在")
        return

    generate_all_plots(
        args.data_file,
        args.output_dir,
        max_points=args.max_points,
        num_groups=args.num_groups,
        random_groups=args.random_groups,
        seed=args.seed,
        allow_overlap=not args.no_overlap,
    )

if __name__ == '__main__':
    main()

# 运行示例:
# 确定性分段（覆盖全局）
# python visualize_results.py --data_file results/SMD_DC_UAD_plot_data.pkl --output_dir plots/SMD
# 随机窗口（可复现）
# python visualize_results.py --data_file results/SMD_DC_UAD_plot_data.pkl --output_dir plots/SMD --random_groups --seed 42
# python visualize_results.py --data_file results/SelfBuilt_DC_UAD_plot_data.pkl --output_dir plots/SelfBuilt --random_groups --seed 42
# python visualize_results.py --data_file results/SMAP_DC_UAD_plot_data.pkl --output_dir plots/SMAP --random_groups --seed 42
# 随机窗口（不重叠）
# python visualize_results.py --data_file results/SMD_DC_UAD_plot_data.pkl --output_dir plots/SMD --random_groups --no_overlap
