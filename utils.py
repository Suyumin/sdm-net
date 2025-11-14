import os
import sys
import cv2
import torch.optim as optim
import numpy as np
import json
import pickle
import random
import torch.nn as nn
from center_loss import CenterLoss

import torch
from torch.nn import init
from tqdm import tqdm
from torch.nn import functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

import time
import numpy as np
from sklearn.calibration import calibration_curve
import time
import math
import random


def compute_calibration_metrics_no_numpy(model, data_loader, device, num_bins=15, save_path=None):
    """计算模型校准指标（不使用numpy）"""
    model.eval()
    confidences = []
    predictions = []
    true_labels = []

    print("Computing calibration metrics (no numpy)...")

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            probs = torch.softmax(outputs[0], dim=1)
            confidence, preds = torch.max(probs, 1)

            confidences.extend(confidence.cpu().tolist())
            predictions.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} batches...")

    # 计算ECE (Expected Calibration Error)
    bin_boundaries = [i / num_bins for i in range(num_bins + 1)]
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # 找到在当前置信度区间的样本
        in_bin = []
        for i, conf in enumerate(confidences):
            if bin_lower < conf <= bin_upper:
                in_bin.append(i)

        prop_in_bin = len(in_bin) / len(confidences) if confidences else 0

        if prop_in_bin > 0:
            # 计算该区间内的准确率
            correct_count = 0
            total_confidence = 0
            for idx in in_bin:
                if predictions[idx] == true_labels[idx]:
                    correct_count += 1
                total_confidence += confidences[idx]

            accuracy_in_bin = correct_count / len(in_bin) if in_bin else 0
            avg_confidence_in_bin = total_confidence / len(in_bin) if in_bin else 0

            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(len(in_bin))

            # 计算该区间的校准误差
            ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        else:
            bin_accuracies.append(0)
            bin_confidences.append(0)
            bin_counts.append(0)

    # 计算其他校准指标
    # MCE (Maximum Calibration Error)
    mce = 0.0
    for i in range(len(bin_accuracies)):
        if bin_counts[i] > 0:
            mce = max(mce, abs(bin_accuracies[i] - bin_confidences[i]))

    # NLL (Negative Log Likelihood)
    nll = 0.0
    for i in range(len(true_labels)):
        true_class = true_labels[i]
        pred_prob = confidences[i] if predictions[i] == true_class else (1 - confidences[i])
        nll += -math.log(max(pred_prob, 1e-15))  # 避免log(0)
    nll /= len(true_labels) if true_labels else 1

    # 简化版的Brier Score计算
    brier_score = 0.0
    for i in range(len(true_labels)):
        true_class = true_labels[i]
        pred_class = predictions[i]
        # 对于Brier Score，我们需要每个类别的概率
        # 这里简化计算，只考虑预测类别的置信度
        if pred_class == true_class:
            brier_score += (1 - confidences[i]) ** 2
        else:
            brier_score += (0 - confidences[i]) ** 2
    brier_score /= len(true_labels) if true_labels else 1

    calibration_results = {
        'ece': ece,
        'mce': mce,
        'nll': nll,
        'brier_score': brier_score,
        'confidence': confidences,
        'predictions': predictions,
        'true_labels': true_labels,
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_counts': bin_counts,
        'calibration_curve': ([], []),  # 简化版本不计算这个
        'bin_boundaries': bin_boundaries
    }

    print(f"Calibration metrics computed:")
    print(f"  ECE: {ece:.4f}")
    print(f"  MCE: {mce:.4f}")
    print(f"  NLL: {nll:.4f}")
    print(f"  Brier Score: {brier_score:.4f}")

    return calibration_results


def measure_inference_time_no_numpy(model, input_size=(1, 3, 224, 224), device='cuda', num_runs=100):
    """测量模型推理时间（不使用numpy）"""
    model.eval()

    # 确保设备可用
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    # 创建 dummy input
    dummy_input = torch.randn(input_size).to(device)

    print(f"Measuring inference time on {device}...")
    print(f"Input size: {input_size}")
    print(f"Number of runs: {num_runs}")

    # GPU预热
    if device.startswith('cuda'):
        print("Performing GPU warmup...")
        for _ in range(10):
            _ = model(dummy_input)
        torch.cuda.synchronize()

    # 测量推理时间
    print("Measuring inference time...")
    inference_times = []

    with torch.no_grad():
        for i in range(num_runs):
            start_time = time.time()
            _ = model(dummy_input)

            if device.startswith('cuda'):
                torch.cuda.synchronize()

            end_time = time.time()
            inference_times.append((end_time - start_time) * 1000)  # 转换为毫秒

            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{num_runs} runs...")

    # 计算统计量（不使用numpy）
    if inference_times:
        avg_time = sum(inference_times) / len(inference_times)
        squared_diffs = [(t - avg_time) ** 2 for t in inference_times]
        std_time = math.sqrt(sum(squared_diffs) / len(inference_times))
        min_time = min(inference_times)
        max_time = max(inference_times)
        fps = 1000 / avg_time  # 帧每秒
    else:
        avg_time = std_time = min_time = max_time = fps = 0

    print(f"Inference time results:")
    print(f"  Average: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"  Range: {min_time:.2f} - {max_time:.2f} ms")
    print(f"  FPS: {fps:.2f}")

    return {
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'fps': fps,
        'all_times': inference_times,
        'device': device,
        'input_size': input_size
    }

def compute_calibration_metrics(model, data_loader, device, num_bins=15, save_path=None):
    """计算模型校准指标，包括预期校准误差(ECE)"""
    model.eval()
    confidences = []
    predictions = []
    true_labels = []

    print("Computing calibration metrics...")

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            probs = torch.softmax(outputs[0], dim=1)
            confidence, preds = torch.max(probs, 1)

            confidences.extend(confidence.cpu().numpy())
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} batches...")

    # 转换为numpy数组
    confidences = np.array(confidences)
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # 计算ECE (Expected Calibration Error)
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # 找到在当前置信度区间的样本
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            # 计算该区间内的准确率
            accuracy_in_bin = np.mean(predictions[in_bin] == true_labels[in_bin])
            # 计算该区间内的平均置信度
            avg_confidence_in_bin = np.mean(confidences[in_bin])

            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(np.sum(in_bin))

            # 计算该区间的校准误差
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        else:
            bin_accuracies.append(0)
            bin_confidences.append(0)
            bin_counts.append(0)

    # 计算可靠性图数据
    prob_true, prob_pred = calibration_curve(true_labels, confidences, n_bins=num_bins)

    # 计算其他校准指标
    # MCE (Maximum Calibration Error)
    mce = 0.0
    for i in range(len(bin_accuracies)):
        if bin_counts[i] > 0:
            mce = max(mce, abs(bin_accuracies[i] - bin_confidences[i]))

    # NLL (Negative Log Likelihood)
    nll = 0.0
    for i in range(len(true_labels)):
        true_class = true_labels[i]
        pred_prob = confidences[i] if predictions[i] == true_class else (1 - confidences[i])
        nll += -np.log(max(pred_prob, 1e-15))  # 避免log(0)
    nll /= len(true_labels)

    # Brier Score
    one_hot_true = np.eye(len(np.unique(true_labels)))[true_labels]
    pred_probs_matrix = np.zeros_like(one_hot_true, dtype=float)
    for i in range(len(predictions)):
        pred_probs_matrix[i, predictions[i]] = confidences[i]
        # 其他类别的概率均匀分布剩余概率
        other_probs = (1 - confidences[i]) / (pred_probs_matrix.shape[1] - 1)
        for j in range(pred_probs_matrix.shape[1]):
            if j != predictions[i]:
                pred_probs_matrix[i, j] = other_probs

    brier_score = np.mean(np.sum((pred_probs_matrix - one_hot_true) ** 2, axis=1))

    calibration_results = {
        'ece': ece,
        'mce': mce,
        'nll': nll,
        'brier_score': brier_score,
        'confidence': confidences.tolist(),
        'predictions': predictions.tolist(),
        'true_labels': true_labels.tolist(),
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_counts': bin_counts,
        'calibration_curve': (prob_true.tolist(), prob_pred.tolist()),
        'bin_boundaries': bin_boundaries.tolist()
    }

    print(f"Calibration metrics computed:")
    print(f"  ECE: {ece:.4f}")
    print(f"  MCE: {mce:.4f}")
    print(f"  NLL: {nll:.4f}")
    print(f"  Brier Score: {brier_score:.4f}")

    # 绘制校准曲线
    if save_path:
        plot_calibration_curve_simple(calibration_results, save_path)

    return calibration_results


def plot_calibration_curve_simple(calibration_results, save_path=None):
    """绘制校准曲线（简化版本）"""
    try:
        import matplotlib.pyplot as plt

        prob_true = calibration_results['calibration_curve'][0]
        prob_pred = calibration_results['calibration_curve'][1]
        bin_counts = calibration_results['bin_counts']

        plt.figure(figsize=(10, 8))

        # 绘制校准曲线
        plt.subplot(2, 1, 1)
        plt.plot(prob_pred, prob_true, 's-', label='Model', linewidth=2, markersize=6)
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated', alpha=0.8)
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 绘制简单的样本数量图
        plt.subplot(2, 1, 2)

        # 创建bin索引
        bin_indices = list(range(len(bin_counts)))
        valid_bins = [i for i, count in enumerate(bin_counts) if count > 0]
        valid_counts = [bin_counts[i] for i in valid_bins]

        if valid_bins:
            plt.bar(valid_bins, valid_counts, alpha=0.7, color='skyblue', edgecolor='black')

        plt.xlabel('Bin Index')
        plt.ylabel('Count')
        plt.title('Sample Count per Confidence Bin')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(f'{save_path}/calibration_curve.png', dpi=300, bbox_inches='tight')
            print(f"Calibration curve saved to: {save_path}/calibration_curve.png")
        else:
            plt.show()

        plt.close()

    except Exception as e:
        print(f"Error plotting calibration curve: {e}")
        print("Skipping calibration curve plot")


def measure_inference_time(model, input_size=(1, 3, 224, 224), device='cuda', num_runs=100):
    """测量模型推理时间"""
    model.eval()

    # 确保设备可用
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    # 创建 dummy input
    dummy_input = torch.randn(input_size).to(device)

    print(f"Measuring inference time on {device}...")
    print(f"Input size: {input_size}")
    print(f"Number of runs: {num_runs}")

    # GPU预热
    if device.startswith('cuda'):
        print("Performing GPU warmup...")
        for _ in range(10):
            _ = model(dummy_input)
        torch.cuda.synchronize()

    # 测量推理时间
    print("Measuring inference time...")
    inference_times = []

    with torch.no_grad():
        for i in range(num_runs):
            start_time = time.time()
            _ = model(dummy_input)

            if device.startswith('cuda'):
                torch.cuda.synchronize()

            end_time = time.time()
            inference_times.append((end_time - start_time) * 1000)  # 转换为毫秒

            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{num_runs} runs...")

    # 计算统计量
    inference_times = np.array(inference_times)
    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    fps = 1000 / avg_time  # 帧每秒

    print(f"Inference time results:")
    print(f"  Average: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"  Range: {min_time:.2f} - {max_time:.2f} ms")
    print(f"  FPS: {fps:.2f}")

    return {
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'fps': fps,
        'all_times': inference_times.tolist(),
        'device': device,
        'input_size': input_size
    }


@torch.no_grad()
def detailed_evaluate(model, data_loader, device, class_names, save_path='./results'):
    """详细评估模型性能，包括各类别指标和混淆矩阵"""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        outputs, _ = model(images)
        probs = torch.softmax(outputs[0], dim=1)
        _, preds = torch.max(probs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 计算总体准确率
    accuracy = np.mean(all_preds == all_labels)

    # 计算每类精确度、召回率、F1分数
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=range(len(class_names))
    )

    # 计算宏平均和微平均
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='micro'
    )

    # 计算加权平均
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )

    # 生成分类报告
    class_report = classification_report(all_labels, all_preds,
                                         target_names=class_names, digits=4)

    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))

    # 保存结果
    results = {
        'accuracy': accuracy,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'support_per_class': support_per_class.tolist(),
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'classification_report': class_report,
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist(),
        'probabilities': all_probs.tolist()
    }

    # 打印结果
    print("\n" + "=" * 80)
    print("DETAILED EVALUATION RESULTS")
    print("=" * 80)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro F1-Score: {f1_macro:.4f}")
    print(f"Weighted F1-Score: {f1_weighted:.4f}")
    print("\nPer-class Metrics:")
    print("-" * 60)
    print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 60)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<20} {precision_per_class[i]:<10.4f} {recall_per_class[i]:<10.4f} "
              f"{f1_per_class[i]:<10.4f} {support_per_class[i]:<10}")

    print("\nClassification Report:")
    print("-" * 60)
    print(class_report)

    # 绘制混淆矩阵
    plot_confusion_matrix(cm, class_names, save_path)

    # 绘制各类别性能对比图
    plot_class_performance(precision_per_class, recall_per_class, f1_per_class, class_names, save_path)

    return results


def plot_confusion_matrix(cm, class_names, save_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{save_path}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 归一化混淆矩阵
    plt.figure(figsize=(10, 8))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{save_path}/confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_class_performance(precision, recall, f1, class_names, save_path):
    """绘制各类别性能对比图"""
    x = np.arange(len(class_names))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
    plt.bar(x, recall, width, label='Recall', alpha=0.8)
    plt.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.title('Performance Metrics by Class')
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path}/class_performance.png', dpi=300, bbox_inches='tight')
    plt.close()


class AMCLoss(torch.nn.Module):
    def __init__(self, margin=0.5):
        super(AMCLoss, self).__init__()
        self.margin = margin

    def forward(self, features, labels):
        normed_features = F.normalize(features, p=2, dim=1)
        cosine_sim = torch.matmul(normed_features, normed_features.T)
        labels = labels.unsqueeze(1)
        mask = labels == labels.T
        positive_pairs = cosine_sim[mask].view(features.size(0), -1)
        negative_pairs = cosine_sim[~mask].view(features.size(0), -1)
        positive_loss = F.relu(1 - positive_pairs).mean()
        negative_loss = F.relu(negative_pairs - self.margin).mean()
        return positive_loss + negative_loss
class PEDCCLoss(torch.nn.Module):
    def __init__(self, num_classes=7, feature_dim=640):
        super(PEDCCLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.centroids = self.create_pedcc()

    def create_pedcc(self):
        # 创建预定义的均匀分布类中心点
        # 具体实现取决于算法细节
        pass

    def forward(self, features, labels):
        batch_size = features.size(0)
        centroids_batch = self.centroids[labels]
        intra_loss = F.mse_loss(features, centroids_batch)
        inter_loss = 0
        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                inter_loss += F.mse_loss(self.centroids[i], self.centroids[j])
        inter_loss = -inter_loss / (self.num_classes * (self.num_classes - 1) / 2)
        return intra_loss + inter_loss
# ----------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        logpt = F.log_softmax(input, dim=1)
        #logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
# --------------------------------
import torch
from torch.nn.functional import cross_entropy, one_hot, softmax


class PolyLoss(torch.nn.Module):
    """
    PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions
    <https://arxiv.org/abs/2204.12511>
    """

    def __init__(self, epsilon=1.0):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, outputs, targets):
        ce = cross_entropy(outputs, targets)
        pt = one_hot(targets, outputs.size()[1]) * softmax(outputs, 1)

        return (ce + self.epsilon * (1.0 - pt.sum(dim=1))).mean()

# -----------------------------------
def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list
def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs/3.0, dim=1)
    softmax_targets = F.softmax(targets/3.0, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()
    init = False

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        inputs, labels = data  #数据集加载
        sample_num += inputs.shape[0]
        inputs, labels = inputs.to(device), labels.to(device)
        outputs, outputs_feature = model(inputs)#取出模型的训练结果
        if init is False:
            layer_list = []
            teacher_feature_size = outputs_feature[0].size(1)
            for index in range(1, len(outputs_feature)):
                student_feature_size = outputs_feature[index].size(1)  # 取浅层的三个特征层(没有经过FC)
                layer_list.append(nn.Linear(student_feature_size, teacher_feature_size))
            model.adaptation_layers = nn.ModuleList(layer_list)
            model.adaptation_layers.cuda()
            init = True
        loss = torch.FloatTensor([0.]).to(device)
        loss += loss_function(outputs[0], labels)# 这是标签与预测标签的交叉熵损失

        teacher_output = outputs[0].detach()  # 取出最深层特征层
        teacher_feature = outputs_feature[0].detach()  # 取出最深层特征层(没有经过FC)

        # 自蒸馏

        for index in range(1, len(outputs)):
            #   logits distillation 对分类输出最soft_loss
            # 逻辑蒸馏，将教师网络的输出和每个浅层学生网络之间做逻辑蒸馏,Loss source2
            loss += CrossEntropy(outputs[index], teacher_output) * 0.5 # KL_loss soft loss
            # loss source1
            loss += loss_function(outputs[index], labels) * 0.5 # hard loss 学生自己的
            #   feature distillation  hint蒸馏
            # 特征蒸馏,loss source3
            if index != 1:
                loss += torch.dist(model.adaptation_layers[index - 1](outputs_feature[index]), teacher_feature) * \
                        0.03

        # 记录损失
        loss += loss.item()
        # 计算准确率(基于教师模型输出)
        pred_classes = torch.max(teacher_output, dim=1)[1]  #
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        outputs, outputs_feature = model(images.to(device))#取出模型的训练结果
        teacher_output = outputs[0].detach()
        pred_classes = torch.max(teacher_output, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(teacher_output, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compatibility with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


class GradCAM:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=(2, 3), keepdims=True)

    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)

        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def __call__(self, input_tensor, target_category=None):

        if self.cuda:
            input_tensor = input_tensor.cuda()

        # 正向传播得到网络输出logits(未经过softmax)
        output = self.activations_and_grads(input_tensor)
        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"category id: {target_category}")
        else:
            assert (len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def center_crop_img(img: np.ndarray, size: int):
    h, w, c = img.shape

    if w == h == size:
        return img

    if w < h:
        ratio = size / w
        new_w = size
        new_h = int(h * ratio)
    else:
        ratio = size / h
        new_h = size
        new_w = int(w * ratio)

    img = cv2.resize(img, dsize=(new_w, new_h))

    if new_w == size:
        h = (new_h - size) // 2
        img = img[h: h+size]
    else:
        w = (new_w - size) // 2
        img = img[:, w: w+size]

    return img