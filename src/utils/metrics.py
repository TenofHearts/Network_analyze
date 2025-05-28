import numpy as np
from typing import List, Dict


def calculate_spread_metrics(propagation_results: List[Dict]) -> Dict[str, float]:
    """
    计算传播效果指标

    参数:
        propagation_results: 传播结果列表

    返回:
        包含各项指标的字典
    """
    total_activated = [result["total_activated"] for result in propagation_results]
    propagation_steps = [result["propagation_steps"] for result in propagation_results]

    metrics = {
        "mean_activated": np.mean(total_activated),
        "std_activated": np.std(total_activated),
        "mean_steps": np.mean(propagation_steps),
        "std_steps": np.std(propagation_steps),
    }

    return metrics


def calculate_node_importance_scores(model_output: np.ndarray) -> np.ndarray:
    """
    计算节点重要性分数

    参数:
        model_output: 模型输出

    返回:
        节点重要性分数
    """
    # 将输出展平并归一化到[0,1]范围
    scores = model_output.flatten()
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return scores
