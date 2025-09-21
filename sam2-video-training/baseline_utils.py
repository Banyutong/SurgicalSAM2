"""
Utility functions for extracting and comparing baseline metrics.
Follows KISS, YAGNI, DRY principles - minimal, focused implementation.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional
from loguru import logger


def extract_baseline_metrics(combo_name: str, baseline_results_dir: str = "baseline_results") -> Optional[Dict[str, float]]:
    """
    Extract baseline metrics for a combo configuration.
    
    Args:
        combo_name: Name of the combo (e.g., "endovis17_1_mem", "cholecseg8k_2_mem_sfx")
        baseline_results_dir: Path to baseline results directory
        
    Returns:
        Dictionary with baseline metrics or None if not found
    """
    # Parse combo name to extract dataset and memory config
    parts = combo_name.split("_")
    if len(parts) < 3:
        logger.warning(f"Invalid combo name format: {combo_name}")
        return None
    
    dataset = parts[0]  # endovis17 or cholecseg8k
    mem_num = parts[1]  # memory number
    
    # For *_mem_sfx configs, baseline is from *_mem
    baseline_combo = f"{mem_num}_mem"
    
    baseline_path = Path(baseline_results_dir) / dataset / baseline_combo / "metrics.json"
    
    if not baseline_path.exists():
        logger.warning(f"Baseline metrics not found: {baseline_path}")
        return None
    
    try:
        with open(baseline_path, 'r') as f:
            baseline_metrics = json.load(f)
        
        logger.info(f"Loaded baseline metrics for {combo_name}: {baseline_metrics}")
        return baseline_metrics
    
    except Exception as e:
        logger.error(f"Failed to load baseline metrics from {baseline_path}: {e}")
        return None


def calculate_metrics_delta(current_metrics: Dict[str, float], baseline_metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate delta between current and baseline metrics.
    
    Args:
        current_metrics: Current evaluation metrics
        baseline_metrics: Baseline metrics to compare against
        
    Returns:
        Dictionary with delta values (current - baseline)
    """
    delta_metrics = {}
    
    for metric_key in current_metrics.keys():
        if metric_key in baseline_metrics:
            delta = current_metrics[metric_key] - baseline_metrics[metric_key]
            delta_metrics[f"delta_{metric_key}"] = delta
            logger.info(f"{metric_key}: {current_metrics[metric_key]:.4f} vs baseline {baseline_metrics[metric_key]:.4f} = delta {delta:.4f}")
        else:
            logger.warning(f"Metric {metric_key} not found in baseline")
    
    return delta_metrics