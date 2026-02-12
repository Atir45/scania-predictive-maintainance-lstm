"""
Experiment Tracker for ML Model Training

This module helps track all experiments automatically:
- Logs parameters, metrics, and timestamps
- Saves results to CSV for easy comparison
- Saves trained models for future use
- Prevents losing good results during experimentation

Usage:
    tracker = ExperimentTracker(experiment_name='decision_tree_comparison')
    tracker.log_experiment(model, params, metrics)
"""

import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime
from pathlib import Path


class ExperimentTracker:
    """
    Tracks machine learning experiments automatically.
    
    Saves:
    1. All experiment results to CSV (append mode - never lose data)
    2. Trained models as pickle files
    3. Metadata (date, time, notes)
    """
    
    def __init__(self, experiment_name='experiments', results_dir='../results/experiments'):
        """
        Initialize the tracker.
        
        Args:
            experiment_name: Name for this set of experiments (e.g., 'decision_trees')
            results_dir: Directory to save results
        """
        self.experiment_name = experiment_name
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.csv_path = self.results_dir / f'{experiment_name}_log.csv'
        self.models_dir = self.results_dir / 'saved_models'
        self.models_dir.mkdir(exist_ok=True)
        
        print(f"✓ Experiment Tracker initialized")
        print(f"  Results will be saved to: {self.csv_path}")
        print(f"  Models will be saved to: {self.models_dir}")
    
    def log_experiment(self, model_name, params, metrics, model=None, notes=''):
        """
        Log a single experiment.
        
        Args:
            model_name: Name/ID of the model (e.g., 'Gini_Unpruned_Balanced')
            params: Dictionary of model parameters
            metrics: Dictionary of performance metrics (accuracy, recall, etc.)
            model: Optional - the trained model object to save
            notes: Optional - any notes about this experiment
        
        Returns:
            experiment_id: Unique ID for this experiment
        """
        # Generate unique experiment ID
        timestamp = datetime.now()
        experiment_id = f"{model_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Prepare experiment record
        experiment_record = {
            'experiment_id': experiment_id,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'model_name': model_name,
            'notes': notes
        }
        
        # Add parameters (flatten nested dicts)
        for param_key, param_value in params.items():
            experiment_record[f'param_{param_key}'] = str(param_value)
        
        # Add metrics
        for metric_key, metric_value in metrics.items():
            experiment_record[f'metric_{metric_key}'] = metric_value
        
        # Save to CSV (append mode - never overwrites)
        df_new = pd.DataFrame([experiment_record])
        
        if self.csv_path.exists():
            # Append to existing file
            df_existing = pd.read_csv(self.csv_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(self.csv_path, index=False)
        else:
            # Create new file
            df_new.to_csv(self.csv_path, index=False)
        
        # Save model if provided
        if model is not None:
            model_path = self.models_dir / f'{experiment_id}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"  ✓ Model saved: {model_path.name}")
        
        print(f"  ✓ Experiment logged: {experiment_id}")
        
        return experiment_id
    
    def get_best_experiments(self, metric='metric_F1', top_n=5):
        """
        Get the best experiments based on a metric.
        
        Args:
            metric: Name of the metric column to sort by
            top_n: Number of top experiments to return
        
        Returns:
            DataFrame with top N experiments
        """
        if not self.csv_path.exists():
            print("No experiments logged yet!")
            return None
        
        df = pd.read_csv(self.csv_path)
        
        if metric not in df.columns:
            print(f"Metric '{metric}' not found. Available metrics:")
            metric_cols = [col for col in df.columns if col.startswith('metric_')]
            print(metric_cols)
            return None
        
        # Sort by metric (descending) and return top N
        df_sorted = df.sort_values(by=metric, ascending=False)
        return df_sorted.head(top_n)
    
    def load_model(self, experiment_id):
        """
        Load a saved model by experiment ID.
        
        Args:
            experiment_id: The unique experiment ID
        
        Returns:
            The loaded model object
        """
        model_path = self.models_dir / f'{experiment_id}.pkl'
        
        if not model_path.exists():
            print(f"Model not found: {experiment_id}")
            return None
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✓ Loaded model: {experiment_id}")
        return model
    
    def compare_experiments(self, experiment_ids):
        """
        Compare specific experiments side-by-side.
        
        Args:
            experiment_ids: List of experiment IDs to compare
        
        Returns:
            DataFrame with comparison
        """
        if not self.csv_path.exists():
            print("No experiments logged yet!")
            return None
        
        df = pd.read_csv(self.csv_path)
        df_filtered = df[df['experiment_id'].isin(experiment_ids)]
        
        return df_filtered
    
    def summary(self):
        """
        Print a summary of all experiments.
        """
        if not self.csv_path.exists():
            print("No experiments logged yet!")
            return
        
        df = pd.read_csv(self.csv_path)
        
        print(f"\n{'='*60}")
        print(f"EXPERIMENT TRACKER SUMMARY: {self.experiment_name}")
        print(f"{'='*60}")
        print(f"Total experiments: {len(df)}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"\nUnique models tested:")
        for model in df['model_name'].unique():
            count = (df['model_name'] == model).sum()
            print(f"  - {model}: {count} runs")
        
        # Show best results
        metric_cols = [col for col in df.columns if col.startswith('metric_')]
        if metric_cols:
            print(f"\nBest results:")
            for metric in metric_cols:
                best_idx = df[metric].idxmax()
                best_value = df.loc[best_idx, metric]
                best_model = df.loc[best_idx, 'model_name']
                print(f"  - {metric}: {best_value:.4f} ({best_model})")
        
        print(f"{'='*60}\n")


# Example usage
if __name__ == '__main__':
    # Create tracker
    tracker = ExperimentTracker(experiment_name='test')
    
    # Example: Log an experiment
    params = {'criterion': 'gini', 'max_depth': 10, 'class_weight': 'balanced'}
    metrics = {'accuracy': 0.85, 'recall': 0.75, 'F1': 0.80}
    
    tracker.log_experiment(
        model_name='DecisionTree_Test',
        params=params,
        metrics=metrics,
        notes='Testing experiment tracker'
    )
    
    # Show summary
    tracker.summary()
