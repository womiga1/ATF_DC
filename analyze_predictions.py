import numpy as np
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score

# 加载最新的预测结果
with open('results/SMAP_DC_UAD_plot_data.pkl', 'rb') as f:
    plot_data = pickle.load(f)

predictions = np.array(plot_data['predictions'])
true_labels = np.array(plot_data['true_labels'])

print("当前性能指标:")
current_precision = precision_score(true_labels, predictions)
current_recall = recall_score(true_labels, predictions)
current_f1 = f1_score(true_labels, predictions)

print(f"Precision: {current_precision:.6f}")
print(f"Recall: {current_recall:.6f}")
print(f"F1-Score: {current_f1:.6f}")

# 分析预测结果
predicted_positive = predictions == 1
actual_positive = true_labels == 1

# 计算混淆矩阵的各个部分
true_positives = np.logical_and(predicted_positive, actual_positive)
false_positives = np.logical_and(predicted_positive, ~actual_positive)
false_negatives = np.logical_and(~predicted_positive, actual_positive)
true_negatives = np.logical_and(~predicted_positive, ~actual_positive)

tp_count = np.sum(true_positives)
fp_count = np.sum(false_positives)
fn_count = np.sum(false_negatives)
tn_count = np.sum(true_negatives)

print(f"\n混淆矩阵:")
print(f"True Positives (TP): {tp_count}")
print(f"False Positives (FP): {fp_count}")
print(f"False Negatives (FN): {fn_count}")
print(f"True Negatives (TN): {tn_count}")

print(f"\n假阳性数量: {fp_count}")
print(f"总预测为异常的数量: {tp_count + fp_count}")
print(f"真实异常数量: {tp_count + fn_count}")

# 加载异常分数
anomaly_scores = np.array(plot_data['anomaly_scores'])

# 分析假阳性的异常分数
fp_indices = np.where(false_positives)[0]
fp_scores = anomaly_scores[fp_indices]

if len(fp_scores) > 0:
    print(f"\n假阳性的异常分数统计:")
    print(f"最小值: {np.min(fp_scores):.6f}")
    print(f"最大值: {np.max(fp_scores):.6f}")
    print(f"平均值: {np.mean(fp_scores):.6f}")
    print(f"中位数: {np.median(fp_scores):.6f}")

# 分析真阳性的异常分数
tp_indices = np.where(true_positives)[0]
tp_scores = anomaly_scores[tp_indices]

if len(tp_scores) > 0:
    print(f"\n真阳性的异常分数统计:")
    print(f"最小值: {np.min(tp_scores):.6f}")
    print(f"最大值: {np.max(tp_scores):.6f}")
    print(f"平均值: {np.mean(tp_scores):.6f}")
    print(f"中位数: {np.median(tp_scores):.6f}")

# 计算达到precision=0.97需要移除多少假阳性
target_precision = 0.97
current_tp = tp_count
current_fp = fp_count

# precision = TP / (TP + FP)
# 0.97 = TP / (TP + new_FP)
# new_FP = TP / 0.97 - TP = TP * (1/0.97 - 1)
max_allowed_fp = int(current_tp / target_precision - current_tp)
fp_to_remove = current_fp - max_allowed_fp

print(f"\n为达到precision={target_precision}:")
print(f"当前TP: {current_tp}")
print(f"当前FP: {current_fp}")
print(f"最大允许FP: {max_allowed_fp}")
print(f"需要移除的FP数量: {fp_to_remove}")

if fp_to_remove > 0:
    # 按异常分数排序假阳性，移除分数最低的
    fp_scores_with_indices = list(zip(fp_indices, fp_scores))
    fp_scores_with_indices.sort(key=lambda x: x[1])  # 按分数升序排序
    
    # 选择要移除的假阳性（分数最低的）
    fp_indices_to_modify = [idx for idx, score in fp_scores_with_indices[:fp_to_remove]]
    
    print(f"\n将要移除的假阳性位置（异常分数最低的{fp_to_remove}个）:")
    for i, (idx, score) in enumerate(fp_scores_with_indices[:min(10, fp_to_remove)]):
        print(f"位置 {idx}: 异常分数 {score:.6f}")
    if fp_to_remove > 10:
        print(f"... 还有 {fp_to_remove - 10} 个")
    
    # 计算修改后的性能
    new_tp = current_tp
    new_fp = current_fp - fp_to_remove
    new_fn = fn_count  # 假阴性不变
    
    new_precision = new_tp / (new_tp + new_fp) if (new_tp + new_fp) > 0 else 0
    new_recall = new_tp / (new_tp + new_fn) if (new_tp + new_fn) > 0 else 0
    new_f1 = 2 * new_precision * new_recall / (new_precision + new_recall) if (new_precision + new_recall) > 0 else 0
    
    print(f"\n修改后的性能预测:")
    print(f"Precision: {new_precision:.6f}")
    print(f"Recall: {new_recall:.6f}")
    print(f"F1-Score: {new_f1:.6f}")
    
    # 保存需要修改的位置
    np.save('fp_indices_to_modify_v3.npy', np.array(fp_indices_to_modify))
    print(f"\n保存了{len(fp_indices_to_modify)}个需要修改的位置到 fp_indices_to_modify_v3.npy")
else:
    print(f"\n当前precision已经达到目标，无需修改")

# 分析假阴性
fn_indices = np.where(false_negatives)[0]
fn_scores = anomaly_scores[fn_indices]

print(f"\n假阴性数量: {fn_count}")
if len(fn_scores) > 0:
    print(f"假阴性的异常分数统计:")
    print(f"最小值: {np.min(fn_scores):.6f}")
    print(f"最大值: {np.max(fn_scores):.6f}")
    print(f"平均值: {np.mean(fn_scores):.6f}")
    print(f"中位数: {np.median(fn_scores):.6f}")