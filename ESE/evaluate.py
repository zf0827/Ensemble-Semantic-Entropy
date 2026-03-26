"""
Evaluation script: compute AUROC and generate ROC curves.

Per Plan.md Step 3:
1. Read source/ESE/result/<difficulty>/<model>/<method>/result.jsonl
2. Evaluate two binary targets:
   - pass@N: max(scores)==1
   - pass@Cluster: whether the largest cluster contains score==1
3. Compute AUROC (higher entropy implies higher chance of failure)
4. Generate ROC plots and summary results
"""

import json
import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_all_results(result_dir: str = "source/ESE/result") -> List[Dict[str, Any]]:
    """
    Load all result.jsonl files.
    
    Args:
        result_dir: results directory
    
    Returns:
        List of results, each containing model, difficulty, method, etc.
    """
    results = []
    result_dir_path = Path(result_dir)
    
    if not result_dir_path.exists():
        logger.error(f"结果目录不存在: {result_dir}")
        return results
    
    # Traverse all difficulty/model/method directories
    for difficulty_dir in result_dir_path.iterdir():
        if not difficulty_dir.is_dir():
            continue
        
        difficulty = difficulty_dir.name
        
        for model_dir in difficulty_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            model = model_dir.name
            
            for method_dir in model_dir.iterdir():
                if not method_dir.is_dir():
                    continue
                
                method = method_dir.name
                result_file = method_dir / "result.jsonl"
                
                if not result_file.exists():
                    logger.warning(f"文件不存在: {result_file}")
                    continue
                
                logger.info(f"读取: {difficulty}/{model}/{method}")
                
                with open(result_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line.strip())
                            data['model'] = model
                            data['difficulty'] = difficulty
                            data['method'] = method
                            results.append(data)
                        except json.JSONDecodeError as e:
                            logger.warning(f"解析JSON失败: {line[:100]}... 错误: {e}")
    
    logger.info(f"总共加载了 {len(results)} 条记录")
    return results


def calculate_pass_at_n(scores: List[float]) -> int:
    """
    Compute pass@N: max(scores)==1.
    
    Args:
        scores: list of scores
    
    Returns:
        1 if max(scores)==1, else 0
    """
    if not scores:
        return 0
    return 1 if max(scores) == 1.0 else 0


def calculate_pass_at_cluster(scores: List[float], cluster_ids: List[int]) -> int:
    """
    Compute pass@Cluster: whether the largest cluster contains score==1.
    
    Args:
        scores: list of scores
        cluster_ids: list of cluster IDs
    
    Returns:
        1 if the largest cluster contains score==1, else 0
    """
    if not scores or not cluster_ids or len(scores) != len(cluster_ids):
        return 0
    
    # Find the largest cluster
    from collections import Counter
    cluster_counts = Counter(cluster_ids)
    if not cluster_counts:
        return 0
    
    largest_cluster_id = cluster_counts.most_common(1)[0][0]
    
    # Check if the largest cluster contains score==1
    for score, cid in zip(scores, cluster_ids):
        if cid == largest_cluster_id and score == 1.0:
            return 1
    
    return 0


def calculate_scores(scores: List[float], cluster_ids: List[int]) -> Dict[str, float]:
    """
    Compute four score metrics.
    
    Args:
        scores: list of scores
        cluster_ids: list of cluster IDs
    
    Returns:
        Dict with avg_score, max_score, cluster_avg_score, cluster_max_score
    """
    if not scores:
        return {
            'avg_score': 0.0,
            'max_score': 0.0,
            'cluster_avg_score': 0.0,
            'cluster_max_score': 0.0
        }
    
    # avg_score: average of all scores
    avg_score = sum(scores) / len(scores)
    
    # max_score: maximum of all scores
    max_score = max(scores)
    
    # cluster_avg_score / cluster_max_score: avg/max within largest cluster
    if not cluster_ids or len(scores) != len(cluster_ids):
        cluster_avg_score = avg_score
        cluster_max_score = max_score
    else:
        from collections import Counter
        cluster_counts = Counter(cluster_ids)
        if not cluster_counts:
            cluster_avg_score = avg_score
            cluster_max_score = max_score
        else:
            largest_cluster_id = cluster_counts.most_common(1)[0][0]
            # Collect scores from the largest cluster
            cluster_scores = [score for score, cid in zip(scores, cluster_ids) if cid == largest_cluster_id]
            if cluster_scores:
                cluster_avg_score = sum(cluster_scores) / len(cluster_scores)
                cluster_max_score = max(cluster_scores)
            else:
                cluster_avg_score = avg_score
                cluster_max_score = max_score
    
    return {
        'avg_score': avg_score,
        'max_score': max_score,
        'cluster_avg_score': cluster_avg_score,
        'cluster_max_score': cluster_max_score
    }


def calculate_pearson_correlation(x: List[float], y: List[float]) -> Tuple[float, float]:
    """
    Compute Pearson correlation.
    
    Args:
        x: first variable list
        y: second variable list
    
    Returns:
        (correlation, p-value)
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0, 1.0
    
    try:
        x_array = np.array(x)
        y_array = np.array(y)
        
        # Check for constant arrays (zero variance)
        if np.var(x_array) == 0 or np.var(y_array) == 0:
            # If both arrays are constant, correlation is undefined; return 0
            return 0.0, 1.0
        
        corr, p_value = pearsonr(x_array, y_array)
        return float(corr), float(p_value)
    except Exception as e:
        logger.warning(f"计算Pearson相关系数时出错: {e}")
        return 0.0, 1.0


def calculate_auroc(y_true: List[int], y_scores: List[float]) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute AUROC and ROC curve data.
    
    Args:
        y_true: ground-truth labels (0 or 1)
        y_scores: prediction scores (entropy; higher implies more likely failure)
    
    Returns:
        (auroc, fpr array, tpr array)
    """
    if len(set(y_true)) < 2:
        # If only one class is present, AUROC is undefined
        logger.warning("只有一个类别，无法计算AUROC")
        return 0.0, np.array([0.0, 1.0]), np.array([0.0, 1.0])
    
    try:
        # Higher entropy implies more likely failure (label 0), so we invert.
        # roc_curve expects label 1 as positive; use (1 - normalized_entropy)
        # as the score for pass=1.
        
        # Normalize entropy into [0, 1]
        y_scores_array = np.array(y_scores)
        if y_scores_array.max() == y_scores_array.min():
            # All values equal; ROC is undefined
            return 0.5, np.array([0.0, 1.0]), np.array([0.0, 1.0])
        
        # Normalize entropy
        normalized_scores = (y_scores_array - y_scores_array.min()) / (y_scores_array.max() - y_scores_array.min())
        
        # Higher entropy => more likely fail (0), so use 1-normalized for pass=1
        y_pred_scores = 1 - normalized_scores
        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_scores)
        auroc = auc(fpr, tpr)
        
        return auroc, fpr, tpr
    except Exception as e:
        logger.error(f"计算AUROC时出错: {e}")
        return 0.0, np.array([0.0, 1.0]), np.array([0.0, 1.0])


def evaluate_entropy_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate AUROC for all entropy metrics.
    
    Args:
        results: list of all results
    
    Returns:
        Evaluation results dictionary
    """
    evaluation_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    # Entropy types
    entropy_types = ['PE_MC', 'PE_Rao', 'SE', 'DSE']
    
    # Metric types
    metrics = ['pass_at_n', 'pass_at_cluster']
    
    # Difficulty levels
    difficulties = ['easy', 'medium', 'hard']
    
    # Group by model/method/difficulty
    for result in results:
        model = result['model']
        method = result['method']
        difficulty = result['difficulty']
        
        # Compute both targets
        pass_at_n = calculate_pass_at_n(result.get('scores', []))
        pass_at_cluster = calculate_pass_at_cluster(
            result.get('scores', []),
            result.get('cluster_ids', [])
        )
        
        # Store data for AUROC computation
        key = (model, method, difficulty)
        if key not in evaluation_results:
            evaluation_results[key] = {
                'pass_at_n': {'y_true': [], 'entropies': {et: [] for et in entropy_types}},
                'pass_at_cluster': {'y_true': [], 'entropies': {et: [] for et in entropy_types}}
            }
        
        evaluation_results[key]['pass_at_n']['y_true'].append(pass_at_n)
        evaluation_results[key]['pass_at_cluster']['y_true'].append(pass_at_cluster)
        
        for et in entropy_types:
            entropy_value = result.get(et)
            if entropy_value is not None:
                evaluation_results[key]['pass_at_n']['entropies'][et].append(entropy_value)
                evaluation_results[key]['pass_at_cluster']['entropies'][et].append(entropy_value)
    
    # Compute AUROC
    final_results = {}
    
    for (model, method, difficulty), data in evaluation_results.items():
        for metric in metrics:
            y_true = data[metric]['y_true']
            
            for et in entropy_types:
                entropies = data[metric]['entropies'][et]
                
                # Ensure matching lengths
                min_len = min(len(y_true), len(entropies))
                if min_len == 0:
                    continue
                
                y_true_subset = y_true[:min_len]
                entropies_subset = entropies[:min_len]
                
                # Check class distribution
                unique_classes = set(y_true_subset)
                if len(unique_classes) < 2:
                    logger.warning(
                        f"只有一个类别，无法计算AUROC: model={model}, method={method}, "
                        f"difficulty={difficulty}, metric={metric}, entropy_type={et}, "
                        f"classes={unique_classes}, num_samples={min_len}"
                    )
                
                auroc, fpr, tpr = calculate_auroc(y_true_subset, entropies_subset)
                
                key = f"{model}_{method}_{difficulty}_{metric}_{et}"
                final_results[key] = {
                    'model': model,
                    'method': method,
                    'difficulty': difficulty,
                    'metric': metric,
                    'entropy_type': et,
                    'auroc': auroc,
                    'num_samples': min_len,
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist()
                }
    
    # Compute overall (all data combined)
    overall_results = defaultdict(lambda: defaultdict(lambda: {'y_true': [], 'entropies': {et: [] for et in entropy_types}}))
    
    for result in results:
        model = result['model']
        method = result['method']
        
        pass_at_n = calculate_pass_at_n(result.get('scores', []))
        pass_at_cluster = calculate_pass_at_cluster(
            result.get('scores', []),
            result.get('cluster_ids', [])
        )
        
        overall_results[(model, method)]['pass_at_n']['y_true'].append(pass_at_n)
        overall_results[(model, method)]['pass_at_cluster']['y_true'].append(pass_at_cluster)
        
        for et in entropy_types:
            entropy_value = result.get(et)
            if entropy_value is not None:
                overall_results[(model, method)]['pass_at_n']['entropies'][et].append(entropy_value)
                overall_results[(model, method)]['pass_at_cluster']['entropies'][et].append(entropy_value)
    
    # Compute overall AUROC
    for (model, method), data in overall_results.items():
        for metric in metrics:
            y_true = data[metric]['y_true']
            
            for et in entropy_types:
                entropies = data[metric]['entropies'][et]
                
                min_len = min(len(y_true), len(entropies))
                if min_len == 0:
                    continue
                
                y_true_subset = y_true[:min_len]
                entropies_subset = entropies[:min_len]
                
                # Check class distribution
                unique_classes = set(y_true_subset)
                if len(unique_classes) < 2:
                    logger.warning(
                        f"只有一个类别，无法计算AUROC: model={model}, method={method}, "
                        f"difficulty=overall, metric={metric}, entropy_type={et}, "
                        f"classes={unique_classes}, num_samples={min_len}"
                    )
                
                auroc, fpr, tpr = calculate_auroc(y_true_subset, entropies_subset)
                
                key = f"{model}_{method}_overall_{metric}_{et}"
                final_results[key] = {
                    'model': model,
                    'method': method,
                    'difficulty': 'overall',
                    'metric': metric,
                    'entropy_type': et,
                    'auroc': auroc,
                    'num_samples': min_len,
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist()
                }
    
    return final_results


def plot_roc_curves(evaluation_results: Dict[str, Any], output_dir: str = "source/ESE/evaluation"):
    """
    Plot ROC curves.
    
    Args:
        evaluation_results: evaluation results dict
        output_dir: output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Organize data by model/difficulty/metric
    plots_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for key, result in evaluation_results.items():
        model = result['model']
        difficulty = result['difficulty']
        metric = result['metric']
        method = result['method']
        entropy_type = result['entropy_type']
        
        plots_data[model][difficulty][metric].append({
            'method': method,
            'entropy_type': entropy_type,
            'auroc': result['auroc'],
            'fpr': np.array(result['fpr']),
            'tpr': np.array(result['tpr'])
        })
    
    # Create a directory per model
    for model in plots_data:
        model_dir = output_path / model
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot ROC for each difficulty and metric
        for difficulty in plots_data[model]:
            for metric in plots_data[model][difficulty]:
                methods_data = plots_data[model][difficulty][metric]
                
                if not methods_data:
                    continue
                
                plt.figure(figsize=(10, 8))
                
                for data in methods_data:
                    method = data['method']
                    entropy_type = data['entropy_type']
                    auroc = data['auroc']
                    fpr = data['fpr']
                    tpr = data['tpr']
                    
                    # Show only AUC >= 0.6
                    if auroc >= 0.6:
                        label = f"{method}_{entropy_type} (AUC={auroc:.4f})"
                        plt.plot(fpr, tpr, label=label, linewidth=2)
                
                plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)', linewidth=1)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate', fontsize=12)
                plt.ylabel('True Positive Rate', fontsize=12)
                plt.title(f'ROC Curve - {model} - {difficulty} - {metric}', fontsize=14, fontweight='bold')
                plt.legend(loc='lower right', fontsize=9)
                plt.grid(True, alpha=0.3)
                
                # Save figure
                filename = f"auc_{difficulty}_{metric}.png"
                filepath = model_dir / filename
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"已保存: {filepath}")
        
        # Plot overall ROC curves
        if 'overall' in plots_data[model]:
            for metric in plots_data[model]['overall']:
                methods_data = plots_data[model]['overall'][metric]
                
                if not methods_data:
                    continue
                
                plt.figure(figsize=(10, 8))
                
                for data in methods_data:
                    method = data['method']
                    entropy_type = data['entropy_type']
                    auroc = data['auroc']
                    fpr = data['fpr']
                    tpr = data['tpr']
                    
                    # Show only AUC >= 0.6
                    if auroc >= 0.6:
                        label = f"{method}_{entropy_type} (AUC={auroc:.4f})"
                        plt.plot(fpr, tpr, label=label, linewidth=2)
                
                plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)', linewidth=1)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate', fontsize=12)
                plt.ylabel('True Positive Rate', fontsize=12)
                plt.title(f'ROC Curve - {model} - Overall - {metric}', fontsize=14, fontweight='bold')
                plt.legend(loc='lower right', fontsize=9)
                plt.grid(True, alpha=0.3)
                
                # Save figure
                filename = f"auc_overall_{metric}.png"
                filepath = model_dir / filename
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"已保存: {filepath}")


def calculate_fpr_at_tpr(fpr: np.ndarray, tpr: np.ndarray, target_tpr: float) -> float:
    """
    Compute FPR at a target TPR (via interpolation).
    
    Args:
        fpr: FPR array
        tpr: TPR array
        target_tpr: target TPR (e.g., 0.8 or 0.9)
    
    Returns:
        FPR at target TPR; returns 1.0 if unavailable.
    """
    if len(fpr) == 0 or len(tpr) == 0:
        return 1.0
    
    # If target TPR is above range, return max FPR
    if target_tpr > tpr.max():
        return fpr[np.argmax(tpr)]
    
    # If target TPR is below range, return min FPR
    if target_tpr < tpr.min():
        return fpr[np.argmin(tpr)]
    
    # Find the closest target TPR using interpolation
    try:
        from scipy.interpolate import interp1d
        # Ensure tpr is monotonic; ROC may not be strictly monotonic
        indices = np.argsort(tpr)
        sorted_tpr = tpr[indices]
        sorted_fpr = fpr[indices]
        
        # Remove duplicate TPRs (keep the last FPR)
        unique_indices = []
        seen_tpr = set()
        for i in range(len(sorted_tpr) - 1, -1, -1):
            if sorted_tpr[i] not in seen_tpr:
                unique_indices.append(i)
                seen_tpr.add(sorted_tpr[i])
        unique_indices.reverse()
        
        sorted_tpr_unique = sorted_tpr[unique_indices]
        sorted_fpr_unique = sorted_fpr[unique_indices]
        
        if len(sorted_tpr_unique) < 2:
            # Too few points; fall back to linear interpolation
            idx = np.searchsorted(sorted_tpr_unique, target_tpr)
            if idx == 0:
                return sorted_fpr_unique[0]
            elif idx >= len(sorted_tpr_unique):
                return sorted_fpr_unique[-1]
            else:
                # Linear interpolation
                tpr1, tpr2 = sorted_tpr_unique[idx-1], sorted_tpr_unique[idx]
                fpr1, fpr2 = sorted_fpr_unique[idx-1], sorted_fpr_unique[idx]
                if tpr2 == tpr1:
                    return fpr1
                fpr_interp = fpr1 + (fpr2 - fpr1) * (target_tpr - tpr1) / (tpr2 - tpr1)
                return fpr_interp
        
        # Use interpolation function
        interp_func = interp1d(sorted_tpr_unique, sorted_fpr_unique, 
                              kind='linear', bounds_error=False, 
                              fill_value=(sorted_fpr_unique[0], sorted_fpr_unique[-1]))
        return float(interp_func(target_tpr))
    except Exception as e:
        # If interpolation fails, fall back to linear search
        logger.warning(f"插值计算失败，使用线性搜索: {e}")
        # Find closest target TPR
        diff = np.abs(tpr - target_tpr)
        idx = np.argmin(diff)
        return float(fpr[idx])


def create_fpr_at_tpr_summary_plot(evaluation_results: Dict[str, Any], output_dir: str = "source/ESE/evaluation"):
    """
    Create an FPR@TPR summary plot (TPR=80% and 90%).
    
    Args:
        evaluation_results: evaluation results dict
        output_dir: output directory
    """
    output_path = Path(output_dir)
    
    # Organize data by model
    models_data = {}
    
    for key, result in evaluation_results.items():
        model = result['model']
        difficulty = result['difficulty']
        metric = result['metric']
        method = result['method']
        entropy_type = result['entropy_type']
        fpr_array = np.array(result['fpr'])
        tpr_array = np.array(result['tpr'])
        
        if model not in models_data:
            models_data[model] = {}
        if difficulty not in models_data[model]:
            models_data[model][difficulty] = {}
        if metric not in models_data[model][difficulty]:
            models_data[model][difficulty][metric] = []
        
        # Compute FPR@80% and FPR@90%
        fpr_at_80 = calculate_fpr_at_tpr(fpr_array, tpr_array, 0.8)
        fpr_at_90 = calculate_fpr_at_tpr(fpr_array, tpr_array, 0.9)
        
        models_data[model][difficulty][metric].append({
            'method': method,
            'entropy_type': entropy_type,
            'fpr_at_80': fpr_at_80,
            'fpr_at_90': fpr_at_90
        })
    
    # Create a summary plot per model
    for model in models_data:
        model_dir = output_path / model
        
        # Collect all difficulties and metrics
        difficulties = sorted([d for d in models_data[model].keys() if d != 'overall'])
        if 'overall' in models_data[model]:
            difficulties = difficulties + ['overall']
        metrics = ['pass_at_n', 'pass_at_cluster']
        
        # Create a large figure with subplots
        # Layout: rows=difficulties, cols=metrics
        fig = plt.figure(figsize=(16, 4 * len(difficulties)))
        gs = gridspec.GridSpec(len(difficulties), len(metrics), figure=fig, hspace=0.3, wspace=0.3)
        
        # Plot bar charts for each difficulty/metric
        for i, difficulty in enumerate(difficulties):
            for j, metric in enumerate(metrics):
                ax = fig.add_subplot(gs[i, j])
                
                if difficulty in models_data[model] and metric in models_data[model][difficulty]:
                    methods_data = models_data[model][difficulty][metric]
                    
                    # Filter: keep methods with FPR < 50% for at least one TPR
                    filtered_data = []
                    for data in methods_data:
                        if data['fpr_at_80'] < 0.5 or data['fpr_at_90'] < 0.5:
                            filtered_data.append(data)
                    
                    if not filtered_data:
                        ax.text(0.5, 0.5, 'No Data\n(FPR >= 50% for all methods)', 
                               ha='center', va='center', fontsize=12)
                        ax.set_title(f'{difficulty} - {metric}', fontsize=11)
                        continue
                    
                    # Prepare bar chart data
                    # Method label format: method_entropy_type
                    method_names = []
                    fpr_80_values = []
                    fpr_90_values = []
                    
                    for data in filtered_data:
                        method_name = f"{data['method']}_{data['entropy_type']}"
                        method_names.append(method_name)
                        fpr_80_values.append(data['fpr_at_80'])
                        fpr_90_values.append(data['fpr_at_90'])
                    
                    # Grouped bar chart
                    x = np.arange(len(method_names))
                    width = 0.35
                    
                    bars1 = ax.bar(x - width/2, fpr_80_values, width, 
                                  label='FPR@80%', alpha=0.8, color='steelblue')
                    bars2 = ax.bar(x + width/2, fpr_90_values, width, 
                                  label='FPR@90%', alpha=0.8, color='coral')
                    
                    ax.set_xlabel('Method', fontsize=10)
                    ax.set_ylabel('FPR', fontsize=10)
                    ax.set_title(f'{difficulty} - {metric}', fontsize=11, fontweight='bold')
                    ax.set_xticks(x)
                    ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=8)
                    ax.legend(loc='upper left', fontsize=9)
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # Compute y-axis range; cap at 0.5 (only show FPR < 50%)
                    all_values = fpr_80_values + fpr_90_values
                    max_value = max(all_values) if all_values else 0.1
                    ax.set_ylim([0, min(max_value * 1.1, 0.55)])  # Slightly above 0.5 for visibility
                    
                    # Add value labels (only FPR < 50%)
                    for bars in [bars1, bars2]:
                        for bar in bars:
                            height = bar.get_height()
                            if height < 0.5:
                                ax.text(bar.get_x() + bar.get_width()/2., height,
                                       f'{height:.3f}',
                                       ha='center', va='bottom',
                                       fontsize=7)
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
                    ax.set_title(f'{difficulty} - {metric}', fontsize=11)
        
        plt.suptitle(f'FPR@TPR Summary - {model}', fontsize=16, fontweight='bold', y=0.995)
        
        # Save summary plot
        summary_file = model_dir / "fpr_at_tpr_summary.png"
        plt.savefig(summary_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"已保存FPR@TPR汇总图: {summary_file}")


def calculate_tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target_fpr: float) -> float:
    """
    Compute TPR at a target FPR (via interpolation).
    
    Args:
        fpr: FPR array
        tpr: TPR array
        target_fpr: target FPR (e.g., 0.05 or 0.1)
    
    Returns:
        TPR at target FPR; returns 0.0 if unavailable.
    """
    if len(fpr) == 0 or len(tpr) == 0:
        return 0.0
    
    # If target FPR is above range, return min TPR
    if target_fpr > fpr.max():
        return tpr[np.argmax(fpr)]
    
    # If target FPR is below range, return min TPR
    if target_fpr < fpr.min():
        return tpr[np.argmin(fpr)]
    
    # Find the closest target FPR using interpolation
    try:
        from scipy.interpolate import interp1d
        # Ensure FPR is monotonic
        indices = np.argsort(fpr)
        sorted_fpr = fpr[indices]
        sorted_tpr = tpr[indices]
        
        # Remove duplicate FPRs (keep the last TPR)
        unique_indices = []
        seen_fpr = set()
        for i in range(len(sorted_fpr) - 1, -1, -1):
            if sorted_fpr[i] not in seen_fpr:
                unique_indices.append(i)
                seen_fpr.add(sorted_fpr[i])
        unique_indices.reverse()
        
        sorted_fpr_unique = sorted_fpr[unique_indices]
        sorted_tpr_unique = sorted_tpr[unique_indices]
        
        if len(sorted_fpr_unique) < 2:
            # Too few points; fall back to linear interpolation
            idx = np.searchsorted(sorted_fpr_unique, target_fpr)
            if idx == 0:
                return sorted_tpr_unique[0]
            elif idx >= len(sorted_fpr_unique):
                return sorted_tpr_unique[-1]
            else:
                # Linear interpolation
                fpr1, fpr2 = sorted_fpr_unique[idx-1], sorted_fpr_unique[idx]
                tpr1, tpr2 = sorted_tpr_unique[idx-1], sorted_tpr_unique[idx]
                if fpr2 == fpr1:
                    return tpr1
                tpr_interp = tpr1 + (tpr2 - tpr1) * (target_fpr - fpr1) / (fpr2 - fpr1)
                return tpr_interp
        
        # Use interpolation function
        interp_func = interp1d(sorted_fpr_unique, sorted_tpr_unique, 
                              kind='linear', bounds_error=False, 
                              fill_value=(sorted_tpr_unique[0], sorted_tpr_unique[-1]))
        return float(interp_func(target_fpr))
    except Exception as e:
        # If interpolation fails, fall back to linear search
        logger.warning(f"插值计算失败，使用线性搜索: {e}")
        # Find closest target FPR
        diff = np.abs(fpr - target_fpr)
        idx = np.argmin(diff)
        return float(tpr[idx])


def calculate_tpr_at_fpr_results(evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute TPR@FPR results (TPR@5% and TPR@10%).
    
    Args:
        evaluation_results: evaluation results dict
    
    Returns:
        Dict keyed by "{model}_{method}_{difficulty}_{metric}_{entropy_type}"
    """
    tpr_at_fpr_results = {}
    
    for key, result in evaluation_results.items():
        model = result['model']
        difficulty = result['difficulty']
        metric = result['metric']
        method = result['method']
        entropy_type = result['entropy_type']
        fpr_array = np.array(result['fpr'])
        tpr_array = np.array(result['tpr'])
        
        # Compute TPR@5% and TPR@10%
        tpr_at_5 = calculate_tpr_at_fpr(fpr_array, tpr_array, 0.05)
        tpr_at_10 = calculate_tpr_at_fpr(fpr_array, tpr_array, 0.1)
        
        result_key = f"{model}_{method}_{difficulty}_{metric}_{entropy_type}"
        tpr_at_fpr_results[result_key] = {
            'model': model,
            'method': method,
            'difficulty': difficulty,
            'metric': metric,
            'entropy_type': entropy_type,
            'tpr_at_5': tpr_at_5,
            'tpr_at_10': tpr_at_10,
            'num_samples': result.get('num_samples', 0)
        }
    
    return tpr_at_fpr_results


def create_tpr_at_fpr_summary_plot(evaluation_results: Dict[str, Any], output_dir: str = "source/ESE/evaluation"):
    """
    Create a TPR@FPR summary plot (FPR=5% and 10%).
    
    Args:
        evaluation_results: evaluation results dict
        output_dir: output directory
    """
    output_path = Path(output_dir)
    
    # Compute TPR@FPR results
    tpr_at_fpr_results = calculate_tpr_at_fpr_results(evaluation_results)
    
    # Organize data by model
    models_data = {}
    
    for key, result in tpr_at_fpr_results.items():
        model = result['model']
        difficulty = result['difficulty']
        metric = result['metric']
        method = result['method']
        entropy_type = result['entropy_type']
        
        if model not in models_data:
            models_data[model] = {}
        if difficulty not in models_data[model]:
            models_data[model][difficulty] = {}
        if metric not in models_data[model][difficulty]:
            models_data[model][difficulty][metric] = []
        
        models_data[model][difficulty][metric].append({
            'method': method,
            'entropy_type': entropy_type,
            'tpr_at_5': result['tpr_at_5'],
            'tpr_at_10': result['tpr_at_10']
        })
    
    # Create a summary plot per model
    for model in models_data:
        model_dir = output_path / model
        
        # Collect all difficulties and metrics
        difficulties = sorted([d for d in models_data[model].keys() if d != 'overall'])
        if 'overall' in models_data[model]:
            difficulties = difficulties + ['overall']
        metrics = ['pass_at_n', 'pass_at_cluster']
        
        # Create a large figure with subplots
        # Layout: rows=difficulties, cols=metrics
        fig = plt.figure(figsize=(16, 4 * len(difficulties)))
        gs = gridspec.GridSpec(len(difficulties), len(metrics), figure=fig, hspace=0.3, wspace=0.3)
        
        # Plot bar charts for each difficulty/metric
        for i, difficulty in enumerate(difficulties):
            for j, metric in enumerate(metrics):
                ax = fig.add_subplot(gs[i, j])
                
                if difficulty in models_data[model] and metric in models_data[model][difficulty]:
                    methods_data = models_data[model][difficulty][metric]
                    
                    # Filter: keep methods with TPR >= 50% for at least one FPR
                    filtered_data = []
                    for data in methods_data:
                        if data['tpr_at_5'] >= 0.5 or data['tpr_at_10'] >= 0.5:
                            filtered_data.append(data)
                    
                    if not filtered_data:
                        ax.text(0.5, 0.5, 'No Data\n(TPR < 50% for all methods)', 
                               ha='center', va='center', fontsize=12)
                        ax.set_title(f'{difficulty} - {metric}', fontsize=11)
                        continue
                    
                    # Prepare bar chart data
                    # Method label format: method_entropy_type
                    method_names = []
                    tpr_5_values = []
                    tpr_10_values = []
                    
                    for data in filtered_data:
                        method_name = f"{data['method']}_{data['entropy_type']}"
                        method_names.append(method_name)
                        tpr_5_values.append(data['tpr_at_5'])
                        tpr_10_values.append(data['tpr_at_10'])
                    
                    # Grouped bar chart
                    x = np.arange(len(method_names))
                    width = 0.35
                    
                    bars1 = ax.bar(x - width/2, tpr_5_values, width, 
                                  label='TPR@5%', alpha=0.8, color='steelblue')
                    bars2 = ax.bar(x + width/2, tpr_10_values, width, 
                                  label='TPR@10%', alpha=0.8, color='coral')
                    
                    ax.set_xlabel('Method', fontsize=10)
                    ax.set_ylabel('TPR', fontsize=10)
                    ax.set_title(f'{difficulty} - {metric}', fontsize=11, fontweight='bold')
                    ax.set_xticks(x)
                    ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=8)
                    ax.legend(loc='lower right', fontsize=9)
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # Compute y-axis range; start near 0.5 (only show TPR >= 50%)
                    all_values = tpr_5_values + tpr_10_values
                    min_value = min(all_values) if all_values else 0.5
                    max_value = max(all_values) if all_values else 1.0
                    ax.set_ylim([max(0.45, min_value * 0.95), min(1.05, max_value * 1.05)])
                    
                    # Add 50% reference line
                    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
                    
                    # Add value labels (only TPR >= 50%)
                    for bars in [bars1, bars2]:
                        for bar in bars:
                            height = bar.get_height()
                            if height >= 0.5:
                                ax.text(bar.get_x() + bar.get_width()/2., height,
                                       f'{height:.3f}',
                                       ha='center', va='bottom',
                                       fontsize=7)
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
                    ax.set_title(f'{difficulty} - {metric}', fontsize=11)
        
        plt.suptitle(f'TPR@FPR Summary - {model}', fontsize=16, fontweight='bold', y=0.995)
        
        # Save summary plot
        summary_file = model_dir / "tpr_at_fpr_summary.png"
        plt.savefig(summary_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"已保存TPR@FPR汇总图: {summary_file}")


def save_tpr_at_fpr_jsonl(tpr_at_fpr_results: Dict[str, Any], output_file: str = "source/ESE/evaluation/evaluation_tprfpr.jsonl"):
    """
    Save TPR@FPR evaluation results to JSONL.
    
    Args:
        tpr_at_fpr_results: TPR@FPR result dict
        output_file: output file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to list and sort by model/method/difficulty/metric
    results_list = []
    for key, result in tpr_at_fpr_results.items():
        result_to_save = {
            'model': result['model'],
            'method': result['method'],
            'difficulty': result['difficulty'],
            'metric': result['metric'],
            'entropy_type': result['entropy_type'],
            'tpr_at_5': result['tpr_at_5'],
            'tpr_at_10': result['tpr_at_10'],
            'num_samples': result.get('num_samples', 0)
        }
        results_list.append(result_to_save)
    
    # Sort
    results_list.sort(key=lambda x: (x['model'], x['method'], x['difficulty'], x['metric'], x['entropy_type']))
    
    # Save to JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results_list:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    logger.info(f"已保存TPR@FPR评估结果: {output_path}")
    logger.info(f"总共 {len(results_list)} 条记录")


def create_summary_plot(evaluation_results: Dict[str, Any], output_dir: str = "source/ESE/evaluation"):
    """
    Create a summary plot combining all ROC curves.
    
    Args:
        evaluation_results: evaluation results dict
        output_dir: output directory
    """
    output_path = Path(output_dir)
    
    # Organize data by model
    models_data = {}
    
    for key, result in evaluation_results.items():
        model = result['model']
        difficulty = result['difficulty']
        metric = result['metric']
        method = result['method']
        entropy_type = result['entropy_type']
        
        if model not in models_data:
            models_data[model] = {}
        if difficulty not in models_data[model]:
            models_data[model][difficulty] = {}
        if metric not in models_data[model][difficulty]:
            models_data[model][difficulty][metric] = []
        
        models_data[model][difficulty][metric].append({
            'method': method,
            'entropy_type': entropy_type,
            'auroc': result['auroc'],
            'fpr': np.array(result['fpr']),
            'tpr': np.array(result['tpr'])
        })
    
    # Create a summary plot per model
    for model in models_data:
        model_dir = output_path / model
        
        # Collect all difficulties and metrics
        difficulties = sorted([d for d in models_data[model].keys() if d != 'overall'])
        metrics = ['pass_at_n', 'pass_at_cluster']
        
        # Create a large figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(len(difficulties) + 1, len(metrics), figure=fig, hspace=0.3, wspace=0.3)
        
        # Plot ROC curves for each difficulty/metric
        for i, difficulty in enumerate(difficulties + ['overall']):
            for j, metric in enumerate(metrics):
                ax = fig.add_subplot(gs[i, j])
                
                if difficulty in models_data[model] and metric in models_data[model][difficulty]:
                    methods_data = models_data[model][difficulty][metric]
                    
                    for data in methods_data:
                        method = data['method']
                        entropy_type = data['entropy_type']
                        auroc = data['auroc']
                        fpr = data['fpr']
                        tpr = data['tpr']
                        
                        # Show only AUC >= 0.6
                        if auroc >= 0.6:
                            label = f"{method}_{entropy_type} (AUC={auroc:.4f})"
                            ax.plot(fpr, tpr, label=label, linewidth=1.5)
                    
                    ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)', linewidth=1)
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate', fontsize=10)
                    ax.set_ylabel('True Positive Rate', fontsize=10)
                    ax.set_title(f'{difficulty} - {metric}', fontsize=11, fontweight='bold')
                    ax.legend(loc='lower right', fontsize=7)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
                    ax.set_title(f'{difficulty} - {metric}', fontsize=11)
        
        plt.suptitle(f'ROC Curves Summary - {model}', fontsize=16, fontweight='bold', y=0.995)
        
        # Save summary plot
        summary_file = model_dir / "auc_summary.png"
        plt.savefig(summary_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"已保存汇总图: {summary_file}")


def evaluate_correlation(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate correlation between entropy and scores (Pearson).
    
    Args:
        results: list of all results
    
    Returns:
        Correlation results dict
    """
    correlation_results = {}
    
    # Entropy types
    entropy_types = ['PE_MC', 'PE_Rao', 'SE', 'DSE']
    
    # Score types
    score_types = ['avg_score', 'max_score', 'cluster_avg_score', 'cluster_max_score']
    
    # Group by model/method/difficulty
    # Use plain dict to ensure proper initialization
    grouped_data = {}
    
    for result in results:
        model = result['model']
        method = result['method']
        difficulty = result['difficulty']
        
        # Check that all entropy values exist
        has_all_entropies = all(result.get(et) is not None for et in entropy_types)
        if not has_all_entropies:
            continue
        
        # Compute four scores
        scores_dict = calculate_scores(
            result.get('scores', []),
            result.get('cluster_ids', [])
        )
        
        # Check that all scores exist
        has_all_scores = all(scores_dict.get(st) is not None for st in score_types)
        if not has_all_scores:
            continue
        
        # Initialize data structures if needed
        if model not in grouped_data:
            grouped_data[model] = {}
        if method not in grouped_data[model]:
            grouped_data[model][method] = {}
        if difficulty not in grouped_data[model][method]:
            grouped_data[model][method][difficulty] = {
                'entropies': {et: [] for et in entropy_types},
                'scores': {st: [] for st in score_types}
            }
        
        # Store entropy values and scores
        for et in entropy_types:
            entropy_value = result.get(et)
            if entropy_value is not None:
                grouped_data[model][method][difficulty]['entropies'][et].append(entropy_value)
        
        for st in score_types:
            score_value = scores_dict.get(st)
            if score_value is not None:
                grouped_data[model][method][difficulty]['scores'][st].append(score_value)
    
    # Compute correlations
    for model in grouped_data:
        for method in grouped_data[model]:
            for difficulty in grouped_data[model][method]:
                data = grouped_data[model][method][difficulty]
                
                # Ensure matching lengths
                min_len = min(
                    min(len(data['entropies'][et]) for et in entropy_types),
                    min(len(data['scores'][st]) for st in score_types)
                )
                
                if min_len < 2:
                    continue
                
                for et in entropy_types:
                    for st in score_types:
                        entropies = data['entropies'][et][:min_len]
                        scores = data['scores'][st][:min_len]
                        
                        # Pearson correlation (higher entropy should correlate with lower score)
                        corr, p_value = calculate_pearson_correlation(entropies, scores)
                        
                        key = f"{model}_{method}_{difficulty}_{st}_{et}"
                        correlation_results[key] = {
                            'model': model,
                            'method': method,
                            'difficulty': difficulty,
                            'score_type': st,
                            'entropy_type': et,
                            'correlation': corr,
                            'p_value': p_value,
                            'num_samples': min_len
                        }
    
    # Compute overall (all data combined)
    # Use plain dict to ensure proper initialization
    overall_grouped = {}
    
    for result in results:
        model = result['model']
        method = result['method']
        key = (model, method)
        
        # Check that all entropy values exist
        has_all_entropies = all(result.get(et) is not None for et in entropy_types)
        if not has_all_entropies:
            continue
        
        # Compute four scores
        scores_dict = calculate_scores(
            result.get('scores', []),
            result.get('cluster_ids', [])
        )
        
        # Check that all scores exist
        has_all_scores = all(scores_dict.get(st) is not None for st in score_types)
        if not has_all_scores:
            continue
        
        # Initialize data structures if needed
        if key not in overall_grouped:
            overall_grouped[key] = {
                'entropies': {et: [] for et in entropy_types},
                'scores': {st: [] for st in score_types}
            }
        
        # Store entropy values and scores
        for et in entropy_types:
            entropy_value = result.get(et)
            if entropy_value is not None:
                overall_grouped[key]['entropies'][et].append(entropy_value)
        
        for st in score_types:
            score_value = scores_dict.get(st)
            if score_value is not None:
                overall_grouped[key]['scores'][st].append(score_value)
    
    # Compute overall correlations
    for (model, method), data in overall_grouped.items():
        min_len = min(
            min(len(data['entropies'][et]) for et in entropy_types),
            min(len(data['scores'][st]) for st in score_types)
        )
        
        if min_len < 2:
            continue
        
        for et in entropy_types:
            for st in score_types:
                entropies = data['entropies'][et][:min_len]
                scores = data['scores'][st][:min_len]
                
                corr, p_value = calculate_pearson_correlation(entropies, scores)
                
                key = f"{model}_{method}_overall_{st}_{et}"
                correlation_results[key] = {
                    'model': model,
                    'method': method,
                    'difficulty': 'overall',
                    'score_type': st,
                    'entropy_type': et,
                    'correlation': corr,
                    'p_value': p_value,
                    'num_samples': min_len
                }
    
    return correlation_results


def plot_correlation_bar_charts(correlation_results: Dict[str, Any], output_dir: str = "source/ESE/evaluation"):
    """
    Plot correlation bar charts.
    
    Args:
        correlation_results: correlation results dict
        output_dir: output directory
    """
    output_path = Path(output_dir)
    
    # Organize data by model/difficulty/score type
    plots_data = {}
    
    for key, result in correlation_results.items():
        model = result['model']
        difficulty = result['difficulty']
        score_type = result['score_type']
        method = result['method']
        entropy_type = result['entropy_type']
        correlation = result['correlation']
        
        if model not in plots_data:
            plots_data[model] = {}
        if difficulty not in plots_data[model]:
            plots_data[model][difficulty] = {}
        if score_type not in plots_data[model][difficulty]:
            plots_data[model][difficulty][score_type] = []
        
        plots_data[model][difficulty][score_type].append({
            'method': method,
            'entropy_type': entropy_type,
            'correlation': correlation
        })
    
    # Create summary plots per model and difficulty
    for model in plots_data:
        model_dir = output_path / model
        model_dir.mkdir(parents=True, exist_ok=True)
        
        difficulties = sorted([d for d in plots_data[model].keys() if d != 'overall'])
        score_types = ['avg_score', 'max_score', 'cluster_avg_score', 'cluster_max_score']
        
        # One summary plot per difficulty (four score-type subplots)
        for difficulty in difficulties + ['overall']:
            if difficulty not in plots_data[model]:
                continue
            
            # Create a 2x2 subplot layout
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            for idx, score_type in enumerate(score_types):
                ax = axes[idx]
                
                if score_type in plots_data[model][difficulty]:
                    methods_data = plots_data[model][difficulty][score_type]
                    
                    # Prepare bar chart data
                    labels = []
                    correlations = []
                    colors = []
                    
                    for data in methods_data:
                        method = data['method']
                        entropy_type = data['entropy_type']
                        correlation = data['correlation']
                        
                        label = f"{method}_{entropy_type}"
                        labels.append(label)
                        correlations.append(correlation)
                        
                        # Color by correlation sign (negative=red, positive=blue)
                        if correlation < 0:
                            colors.append('red')
                        else:
                            colors.append('blue')
                    
                    # Plot bar chart
                    bars = ax.bar(range(len(labels)), correlations, color=colors, alpha=0.7)
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
                    ax.set_ylabel('Pearson Correlation', fontsize=10)
                    ax.set_title(f'{score_type}', fontsize=11, fontweight='bold')
                    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}',
                               ha='center', va='bottom' if height >= 0 else 'top',
                               fontsize=7)
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
                    ax.set_title(f'{score_type}', fontsize=11)
            
            plt.suptitle(f'Correlation Analysis - {model} - {difficulty}', 
                        fontsize=14, fontweight='bold', y=0.995)
            plt.tight_layout()
            
            # Save figure
            filename = f"p_{difficulty}.png"
            filepath = model_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"已保存相关性图: {filepath}")


def save_correlation_jsonl(correlation_results: Dict[str, Any], output_file: str = "source/ESE/evaluation/evaluation_corr.jsonl"):
    """
    Save correlation results to JSONL.
    
    Args:
        correlation_results: correlation results dict
        output_file: output file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to list and sort
    results_list = []
    for key, result in correlation_results.items():
        results_list.append(result)
    
    # Sort
    results_list.sort(key=lambda x: (x['model'], x['method'], x['difficulty'], x['score_type'], x['entropy_type']))
    
    # Save to JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results_list:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    logger.info(f"已保存相关性评估结果: {output_path}")
    logger.info(f"总共 {len(results_list)} 条记录")


def save_evaluation_jsonl(evaluation_results: Dict[str, Any], output_file: str = "source/ESE/evaluation/evaluation.jsonl"):
    """
    Save evaluation results to JSONL.
    
    Args:
        evaluation_results: evaluation results dict
        output_file: output file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to list and sort by model/method/difficulty/metric
    results_list = []
    for key, result in evaluation_results.items():
        # Remove fpr/tpr to reduce file size (optional)
        result_to_save = {
            'model': result['model'],
            'method': result['method'],
            'difficulty': result['difficulty'],
            'metric': result['metric'],
            'entropy_type': result['entropy_type'],
            'auroc': result['auroc'],
            'num_samples': result['num_samples']
        }
        results_list.append(result_to_save)
    
    # Sort
    results_list.sort(key=lambda x: (x['model'], x['method'], x['difficulty'], x['metric'], x['entropy_type']))
    
    # Save to JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results_list:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    logger.info(f"已保存评估结果: {output_path}")
    logger.info(f"总共 {len(results_list)} 条记录")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='评估熵指标的AUROC和相关性')
    parser.add_argument('--result-dir', type=str, default='source/ESE/result',
                        help='Results directory')
    parser.add_argument('--output-dir', type=str, default='source/ESE/evaluation',
                        help='Output directory')
    parser.add_argument('--auc', action='store_true',
                        help='Evaluate AUROC (if --auc/--corr set, only evaluate selected parts)')
    parser.add_argument('--corr', action='store_true',
                        help='Evaluate correlation (if --auc/--corr set, only evaluate selected parts)')
    
    args = parser.parse_args()
    
    # Determine what to evaluate
    # If --auc/--corr specified, evaluate only selected parts
    # If neither specified, evaluate all
    if args.auc or args.corr:
        # If one is specified, evaluate only that part
        evaluate_auc = args.auc
        evaluate_corr = args.corr
    else:
        # If none specified, evaluate all
        evaluate_auc = True
        evaluate_corr = True
    
    logger.info("=" * 80)
    logger.info("Starting evaluation")
    logger.info(f"Evaluate AUROC: {evaluate_auc}")
    logger.info(f"Evaluate correlation: {evaluate_corr}")
    logger.info("=" * 80)
    
    # 1. Load all results
    logger.info("Step 1: Loading all results...")
    results = load_all_results(args.result_dir)
    
    if not results:
        logger.error("No result files found!")
        return
    
    # AUROC-related evaluation
    if evaluate_auc:
        # 2. Evaluate AUROC
        logger.info("Step 2: Computing AUROC...")
        evaluation_results = evaluate_entropy_metrics(results)
        
        # 3. Save evaluation.jsonl
        logger.info("Step 3: Saving evaluation results...")
        evaluation_file = os.path.join(args.output_dir, "evaluation.jsonl")
        save_evaluation_jsonl(evaluation_results, evaluation_file)
        
        # 4. Plot ROC curves
        logger.info("Step 4: Plotting ROC curves...")
        plot_roc_curves(evaluation_results, args.output_dir)
        
        # 5. Create summary plots
        logger.info("Step 5: Creating summary plots...")
        create_summary_plot(evaluation_results, args.output_dir)
        
        # 6. Create FPR@TPR summary
        logger.info("Step 6: Creating FPR@TPR summary...")
        create_fpr_at_tpr_summary_plot(evaluation_results, args.output_dir)
        
        # 7. Create TPR@FPR summary
        logger.info("Step 7: Creating TPR@FPR summary...")
        create_tpr_at_fpr_summary_plot(evaluation_results, args.output_dir)
        
        # 8. Save TPR@FPR results
        logger.info("Step 8: Saving TPR@FPR results...")
        tpr_at_fpr_results = calculate_tpr_at_fpr_results(evaluation_results)
        tpr_at_fpr_file = os.path.join(args.output_dir, "evaluation_tprfpr.jsonl")
        save_tpr_at_fpr_jsonl(tpr_at_fpr_results, tpr_at_fpr_file)
    
    # Correlation-related evaluation
    if evaluate_corr:
        # 9. Evaluate correlation (Pearson)
        logger.info("Step 9: Computing correlation...")
        correlation_results = evaluate_correlation(results)
        
        # 10. Save correlation results
        logger.info("Step 10: Saving correlation results...")
        correlation_file = os.path.join(args.output_dir, "evaluation_corr.jsonl")
        save_correlation_jsonl(correlation_results, correlation_file)
        
        # 11. Plot correlation bar charts
        logger.info("Step 11: Plotting correlation charts...")
        plot_correlation_bar_charts(correlation_results, args.output_dir)
    
    logger.info("=" * 80)
    logger.info("Evaluation complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

