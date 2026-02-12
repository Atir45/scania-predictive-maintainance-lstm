"""
Decision Tree Model Comparison for Scania Predictive Maintenance
=================================================================

This module provides comprehensive comparison of different decision tree variants
for predictive maintenance classification.

Decision Tree Variants Compared:
1. Gini Impurity (unpruned) - Standard CART algorithm
2. Entropy/Information Gain (unpruned) - Alternative splitting criterion
3. Depth-Limited Trees (max_depth=10, 15) - Pre-pruning by depth
4. Min-Samples Constrained Tree - Pre-pruning by sample requirements
5. Cost-Complexity Pruned Tree - Post-pruning approach

Author: ML Engineer
Date: November 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DecisionTreeComparator:
    """
    Compare different decision tree variants for predictive maintenance.
    
    This class trains multiple decision tree configurations and provides
    comprehensive comparison metrics and visualizations.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the comparator.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models: Dict[str, DecisionTreeClassifier] = {}
        self.results: Dict[str, Dict] = {}
        self.feature_importances: Dict[str, np.ndarray] = {}
        
    def define_models(self) -> Dict[str, DecisionTreeClassifier]:
        """
        Define all decision tree variants to compare.
        
        Returns:
            Dictionary of model name to model instance
        """
        models = {
            'DT_Gini_Unpruned': DecisionTreeClassifier(
                criterion='gini',
                class_weight='balanced',
                random_state=self.random_state,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1
            ),
            'DT_Entropy_Unpruned': DecisionTreeClassifier(
                criterion='entropy',
                class_weight='balanced',
                random_state=self.random_state,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1
            ),
            'DT_Gini_Depth10': DecisionTreeClassifier(
                criterion='gini',
                class_weight='balanced',
                random_state=self.random_state,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1
            ),
            'DT_Gini_Depth15': DecisionTreeClassifier(
                criterion='gini',
                class_weight='balanced',
                random_state=self.random_state,
                max_depth=15,
                min_samples_split=2,
                min_samples_leaf=1
            ),
            'DT_Gini_Depth20': DecisionTreeClassifier(
                criterion='gini',
                class_weight='balanced',
                random_state=self.random_state,
                max_depth=20,
                min_samples_split=2,
                min_samples_leaf=1
            ),
            'DT_MinSamples_Pruned': DecisionTreeClassifier(
                criterion='gini',
                class_weight='balanced',
                random_state=self.random_state,
                max_depth=None,
                min_samples_split=20,
                min_samples_leaf=10
            ),
            'DT_CostComplexity_001': DecisionTreeClassifier(
                criterion='gini',
                class_weight='balanced',
                random_state=self.random_state,
                ccp_alpha=0.001,
                max_depth=None
            ),
            'DT_CostComplexity_0001': DecisionTreeClassifier(
                criterion='gini',
                class_weight='balanced',
                random_state=self.random_state,
                ccp_alpha=0.0001,
                max_depth=None
            )
        }
        
        return models
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train all decision tree variants.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info("Training all decision tree variants...")
        
        self.models = self.define_models()
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            try:
                model.fit(X_train, y_train)
                
                # Store feature importances
                self.feature_importances[name] = model.feature_importances_
                
                # Log tree structure info
                n_nodes = model.tree_.node_count
                max_depth = model.tree_.max_depth
                n_leaves = model.tree_.n_leaves
                
                logger.info(f"  ├─ Nodes: {n_nodes}")
                logger.info(f"  ├─ Max Depth: {max_depth}")
                logger.info(f"  └─ Leaves: {n_leaves}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                
        logger.info(f"Successfully trained {len(self.models)} models")
    
    def evaluate_all_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Evaluate all trained models.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame with comparison metrics
        """
        logger.info("Evaluating all models...")
        
        results_list = []
        
        for name, model in self.models.items():
            logger.info(f"Evaluating {name}...")
            
            try:
                # Predictions
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                metrics = {
                    'Model': name,
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, zero_division=0),
                    'Recall': recall_score(y_test, y_pred, zero_division=0),
                    'F1_Score': f1_score(y_test, y_pred, zero_division=0),
                    'ROC_AUC': roc_auc_score(y_test, y_proba),
                    'Nodes': model.tree_.node_count,
                    'Max_Depth': model.tree_.max_depth,
                    'Leaves': model.tree_.n_leaves
                }
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                tn, fp, fn, tp = cm.ravel()
                
                metrics.update({
                    'True_Negatives': tn,
                    'False_Positives': fp,
                    'False_Negatives': fn,
                    'True_Positives': tp
                })
                
                # Store results
                self.results[name] = {
                    'metrics': metrics,
                    'predictions': y_pred,
                    'probabilities': y_proba,
                    'confusion_matrix': cm
                }
                
                results_list.append(metrics)
                
            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results_list)
        
        # Sort by F1 Score
        comparison_df = comparison_df.sort_values('F1_Score', ascending=False)
        
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON RESULTS")
        logger.info("="*80)
        logger.info(f"\n{comparison_df.to_string(index=False)}")
        
        return comparison_df
    
    def plot_comparison(self, comparison_df: pd.DataFrame, save_dir: str = None) -> None:
        """
        Create comprehensive comparison visualizations.
        
        Args:
            comparison_df: DataFrame with model comparison metrics
            save_dir: Directory to save plots
        """
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Performance Metrics Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        for idx, (metric, color) in enumerate(zip(metrics, colors)):
            ax = axes[idx // 2, idx % 2]
            
            data = comparison_df.sort_values(metric, ascending=True)
            ax.barh(data['Model'], data[metric], color=color, alpha=0.7)
            ax.set_xlabel(metric, fontsize=12, fontweight='bold')
            ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
            ax.set_xlim(0, 1.0)
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(data[metric]):
                ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(save_path / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved metrics comparison to {save_path / 'metrics_comparison.png'}")
        plt.show()
        
        # 2. ROC-AUC Comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        
        data = comparison_df.sort_values('ROC_AUC', ascending=True)
        bars = ax.barh(data['Model'], data['ROC_AUC'], color='#9b59b6', alpha=0.7)
        
        ax.set_xlabel('ROC-AUC Score', fontsize=12, fontweight='bold')
        ax.set_title('ROC-AUC Comparison Across Decision Tree Variants', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(data['ROC_AUC']):
            ax.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=10)
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(save_path / 'roc_auc_comparison.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved ROC-AUC comparison to {save_path / 'roc_auc_comparison.png'}")
        plt.show()
        
        # 3. Tree Structure Comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Nodes
        data = comparison_df.sort_values('Nodes', ascending=True)
        axes[0].barh(data['Model'], data['Nodes'], color='#1abc9c', alpha=0.7)
        axes[0].set_xlabel('Number of Nodes', fontsize=11, fontweight='bold')
        axes[0].set_title('Tree Size (Nodes)', fontsize=12, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Max Depth
        data = comparison_df.sort_values('Max_Depth', ascending=True)
        axes[1].barh(data['Model'], data['Max_Depth'], color='#e67e22', alpha=0.7)
        axes[1].set_xlabel('Maximum Depth', fontsize=11, fontweight='bold')
        axes[1].set_title('Tree Depth', fontsize=12, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        
        # Leaves
        data = comparison_df.sort_values('Leaves', ascending=True)
        axes[2].barh(data['Model'], data['Leaves'], color='#34495e', alpha=0.7)
        axes[2].set_xlabel('Number of Leaves', fontsize=11, fontweight='bold')
        axes[2].set_title('Leaf Nodes', fontsize=12, fontweight='bold')
        axes[2].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(save_path / 'tree_structure_comparison.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved tree structure comparison to {save_path / 'tree_structure_comparison.png'}")
        plt.show()
        
        # 4. Confusion Matrix Heatmap for Top Models
        top_models = comparison_df.nlargest(3, 'F1_Score')['Model'].tolist()
        
        fig, axes = plt.subplots(1, min(3, len(top_models)), figsize=(15, 5))
        if len(top_models) == 1:
            axes = [axes]
        
        for idx, model_name in enumerate(top_models[:3]):
            cm = self.results[model_name]['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Healthy', 'Failed'],
                       yticklabels=['Healthy', 'Failed'])
            
            axes[idx].set_title(f'{model_name}\n(F1={comparison_df[comparison_df["Model"]==model_name]["F1_Score"].values[0]:.3f})',
                              fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('Predicted', fontsize=10)
            axes[idx].set_ylabel('Actual', fontsize=10)
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(save_path / 'top_models_confusion_matrix.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrices to {save_path / 'top_models_confusion_matrix.png'}")
        plt.show()
    
    def plot_feature_importance_comparison(self, feature_names: List[str], 
                                          top_n: int = 15, save_dir: str = None) -> None:
        """
        Compare feature importances across models.
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to show
            save_dir: Directory to save plot
        """
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        # Get top 3 models by F1 score
        comparison_df = pd.DataFrame([self.results[name]['metrics'] for name in self.results.keys()])
        top_models = comparison_df.nlargest(3, 'F1_Score')['Model'].tolist()
        
        fig, axes = plt.subplots(1, min(3, len(top_models)), figsize=(18, 6))
        if len(top_models) == 1:
            axes = [axes]
        
        for idx, model_name in enumerate(top_models[:3]):
            importances = self.feature_importances[model_name]
            
            # Create DataFrame and sort
            feature_imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(top_n)
            
            # Plot
            axes[idx].barh(feature_imp_df['Feature'], feature_imp_df['Importance'], 
                          color='#3498db', alpha=0.7)
            axes[idx].set_xlabel('Importance', fontsize=10, fontweight='bold')
            axes[idx].set_title(f'{model_name}\nTop {top_n} Features', 
                              fontsize=11, fontweight='bold')
            axes[idx].invert_yaxis()
            axes[idx].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(save_path / 'feature_importance_comparison.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance comparison to {save_path / 'feature_importance_comparison.png'}")
        plt.show()
    
    def save_results(self, comparison_df: pd.DataFrame, save_dir: str) -> None:
        """
        Save comparison results to files.
        
        Args:
            comparison_df: DataFrame with comparison metrics
            save_dir: Directory to save results
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save comparison table
        comparison_df.to_csv(save_path / 'model_comparison.csv', index=False)
        logger.info(f"Saved comparison table to {save_path / 'model_comparison.csv'}")
        
        # Save detailed results
        detailed_results = {}
        for name, result in self.results.items():
            detailed_results[name] = {
                'metrics': result['metrics'],
                'confusion_matrix': result['confusion_matrix'].tolist()
            }
        
        with open(save_path / 'detailed_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)
        logger.info(f"Saved detailed results to {save_path / 'detailed_results.json'}")
        
        # Save summary report
        with open(save_path / 'comparison_report.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write("DECISION TREE COMPARISON REPORT\n")
            f.write("Scania Predictive Maintenance\n")
            f.write("="*80 + "\n\n")
            
            f.write("Models Compared:\n")
            for i, model_name in enumerate(self.models.keys(), 1):
                f.write(f"{i}. {model_name}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("PERFORMANCE METRICS\n")
            f.write("="*80 + "\n\n")
            f.write(comparison_df.to_string(index=False))
            
            f.write("\n\n" + "="*80 + "\n")
            f.write("TOP 3 MODELS BY F1 SCORE\n")
            f.write("="*80 + "\n\n")
            top_3 = comparison_df.nlargest(3, 'F1_Score')
            f.write(top_3.to_string(index=False))
            
        logger.info(f"Saved comparison report to {save_path / 'comparison_report.txt'}")
    
    def print_best_model_summary(self, comparison_df: pd.DataFrame) -> str:
        """
        Print summary of the best performing model.
        
        Args:
            comparison_df: DataFrame with comparison metrics
            
        Returns:
            Name of best model
        """
        best_model = comparison_df.iloc[0]
        
        logger.info("\n" + "="*80)
        logger.info("BEST MODEL SUMMARY")
        logger.info("="*80)
        logger.info(f"Model: {best_model['Model']}")
        logger.info(f"F1 Score: {best_model['F1_Score']:.4f}")
        logger.info(f"Accuracy: {best_model['Accuracy']:.4f}")
        logger.info(f"Precision: {best_model['Precision']:.4f}")
        logger.info(f"Recall: {best_model['Recall']:.4f}")
        logger.info(f"ROC-AUC: {best_model['ROC_AUC']:.4f}")
        logger.info(f"Tree Nodes: {best_model['Nodes']}")
        logger.info(f"Max Depth: {best_model['Max_Depth']}")
        logger.info(f"Leaf Nodes: {best_model['Leaves']}")
        logger.info("="*80)
        
        return best_model['Model']


def run_decision_tree_comparison(X_train: pd.DataFrame, y_train: pd.Series,
                                X_test: pd.DataFrame, y_test: pd.Series,
                                save_dir: str = None) -> Tuple[DecisionTreeComparator, pd.DataFrame]:
    """
    Run complete decision tree comparison analysis.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        save_dir: Directory to save results
        
    Returns:
        Tuple of (comparator instance, comparison DataFrame)
    """
    logger.info("Starting Decision Tree Comparison Analysis...")
    
    # Initialize comparator
    comparator = DecisionTreeComparator(random_state=42)
    
    # Train all models
    comparator.train_all_models(X_train, y_train)
    
    # Evaluate all models
    comparison_df = comparator.evaluate_all_models(X_test, y_test)
    
    # Create visualizations
    if save_dir:
        comparator.plot_comparison(comparison_df, save_dir)
        comparator.plot_feature_importance_comparison(X_train.columns.tolist(), 
                                                     top_n=15, save_dir=save_dir)
        comparator.save_results(comparison_df, save_dir)
    
    # Print best model summary
    best_model = comparator.print_best_model_summary(comparison_df)
    
    logger.info(f"\nAnalysis complete! Best model: {best_model}")
    
    return comparator, comparison_df
