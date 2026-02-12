"""
Model Evaluation Module for Scania Predictive Maintenance

This module provides comprehensive evaluation metrics and tools for:
- Binary classification metrics (accuracy, precision, recall, F1)
- ROC-AUC and PR-AUC curves
- Confusion matrices
- Cost-sensitive evaluation
- Model comparison
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation for predictive maintenance"""
    
    def __init__(self, cost_fp: float = 10, cost_fn: float = 500):
        """
        Initialize evaluator
        
        Args:
            cost_fp: Cost of false positive (unnecessary maintenance)
            cost_fn: Cost of false negative (missed failure)
        """
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        self.evaluation_results_ = {}
        
    def evaluate_predictions(self, 
                           y_true: np.ndarray, 
                           y_pred: np.ndarray,
                           y_pred_proba: Optional[np.ndarray] = None,
                           model_name: str = "model") -> Dict[str, float]:
        """
        Evaluate model predictions with comprehensive metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for AUC metrics)
            model_name: Name of the model
            
        Returns:
            Dictionary with all evaluation metrics
        """
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # Add AUC metrics if probabilities provided
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Specificity (True Negative Rate)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Calculate cost
        metrics['total_cost'] = (fp * self.cost_fp) + (fn * self.cost_fn)
        metrics['cost_per_sample'] = metrics['total_cost'] / len(y_true)
        
        # Store results
        self.evaluation_results_[model_name] = metrics
        
        logger.info(f"Evaluation for {model_name}:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        if y_pred_proba is not None:
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"  PR-AUC: {metrics['pr_auc']:.4f}")
        logger.info(f"  Total Cost: ${metrics['total_cost']:,.2f}")
        
        return metrics
    
    def plot_confusion_matrix(self, 
                             y_true: np.ndarray, 
                             y_pred: np.ndarray,
                             model_name: str = "Model",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name for plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Healthy', 'Failed'],
                   yticklabels=['Healthy', 'Failed'])
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Confusion Matrix - {model_name}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")
        
        return fig
    
    def plot_roc_curve(self, 
                      y_true: np.ndarray, 
                      y_pred_proba: np.ndarray,
                      model_name: str = "Model",
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name for legend
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate (Recall)')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved ROC curve to {save_path}")
        
        return fig
    
    def plot_precision_recall_curve(self, 
                                   y_true: np.ndarray, 
                                   y_pred_proba: np.ndarray,
                                   model_name: str = "Model",
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Precision-Recall curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name for legend
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, label=f'{model_name} (AP = {pr_auc:.3f})', linewidth=2)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved PR curve to {save_path}")
        
        return fig
    
    def compare_models(self, results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            results_dict: Dictionary of {model_name: metrics_dict}
            
        Returns:
            DataFrame with model comparison
        """
        comparison_df = pd.DataFrame(results_dict).T
        
        # Sort by F1 score
        if 'f1_score' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('f1_score', ascending=False)
        
        logger.info("Model Comparison:")
        logger.info(f"\n{comparison_df}")
        
        return comparison_df
    
    def find_optimal_threshold(self, 
                              y_true: np.ndarray, 
                              y_pred_proba: np.ndarray,
                              metric: str = 'f1') -> Tuple[float, float]:
        """
        Find optimal classification threshold
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            metric: Metric to optimize ('f1', 'cost')
            
        Returns:
            Tuple of (optimal_threshold, optimal_metric_value)
        """
        thresholds = np.linspace(0, 1, 101)
        best_threshold = 0.5
        best_score = 0
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            
            elif metric == 'cost':
                cm = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = cm.ravel()
                cost = (fp * self.cost_fp) + (fn * self.cost_fn)
                
                # Minimize cost (invert for comparison)
                score = -cost
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
        
        logger.info(f"Optimal threshold for {metric}: {best_threshold:.3f}")
        logger.info(f"Optimal {metric} value: {best_score:.4f}")
        
        return best_threshold, best_score


def calculate_business_metrics(tp: int, fp: int, fn: int, tn: int,
                              cost_fp: float = 10, 
                              cost_fn: float = 500,
                              cost_maintenance: float = 100) -> Dict:
    """
    Calculate business impact metrics
    
    Args:
        tp, fp, fn, tn: Confusion matrix values
        cost_fp: Cost of false positive (unnecessary maintenance)
        cost_fn: Cost of false negative (missed failure)
        cost_maintenance: Cost of planned maintenance
        
    Returns:
        Dictionary with business metrics
    """
    total_predictions = tp + fp + fn + tn
    total_failures = tp + fn
    
    # Calculate costs
    total_cost = (fp * cost_fp) + (fn * cost_fn) + (tp * cost_maintenance)
    cost_without_model = total_failures * cost_fn  # All failures go undetected
    savings = cost_without_model - total_cost
    roi = (savings / total_cost * 100) if total_cost > 0 else 0
    
    metrics = {
        'total_cost': total_cost,
        'cost_without_model': cost_without_model,
        'total_savings': savings,
        'roi_percent': roi,
        'maintenance_actions': tp + fp,
        'prevented_failures': tp,
        'missed_failures': fn,
        'unnecessary_maintenance': fp
    }
    
    return metrics


if __name__ == "__main__":
    print("Model Evaluation Module")
    print("This module provides comprehensive evaluation for predictive maintenance models")
    print("\nMain classes:")
    print("- ModelEvaluator: Evaluate predictions with multiple metrics")
    print("\nMain functions:")
    print("- calculate_business_metrics(): Calculate ROI and business impact")
