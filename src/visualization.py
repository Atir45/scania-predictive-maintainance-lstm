"""
Visualization Module for Scania Predictive Maintenance

This module provides visualization tools for:
- Exploratory data analysis
- Feature distributions
- Time series patterns
- Model performance visualization
- Feature importance plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class DataVisualizer:
    """Visualization tools for Scania predictive maintenance data"""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer
        
        Args:
            style: Matplotlib style
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
    
    def plot_class_distribution(self, 
                                y: pd.Series, 
                                title: str = "Class Distribution",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot target class distribution
        
        Args:
            y: Target variable
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Count plot
        value_counts = y.value_counts()
        ax1.bar(value_counts.index, value_counts.values, color=['#2ecc71', '#e74c3c'])
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.set_title(f'{title} - Counts')
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(['Healthy (0)', 'Failed (1)'])
        
        # Add count labels
        for i, v in enumerate(value_counts.values):
            ax1.text(i, v, f'{v:,}', ha='center', va='bottom')
        
        # Pie chart
        colors = ['#2ecc71', '#e74c3c']
        ax2.pie(value_counts.values, labels=['Healthy', 'Failed'], autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax2.set_title(f'{title} - Percentage')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved class distribution to {save_path}")
        
        return fig
    
    def plot_feature_distributions(self, 
                                  df: pd.DataFrame, 
                                  features: List[str],
                                  hue: Optional[str] = None,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distributions of multiple features
        
        Args:
            df: Input dataframe
            features: List of feature columns to plot
            hue: Column for color grouping (e.g., 'label')
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feature in enumerate(features):
            if feature in df.columns:
                if hue and hue in df.columns:
                    for label in df[hue].unique():
                        subset = df[df[hue] == label][feature].dropna()
                        axes[idx].hist(subset, alpha=0.6, label=f'{hue}={label}', bins=30)
                    axes[idx].legend()
                else:
                    axes[idx].hist(df[feature].dropna(), bins=30, color='steelblue', alpha=0.7)
                
                axes[idx].set_xlabel(feature)
                axes[idx].set_ylabel('Frequency')
                axes[idx].set_title(f'Distribution of {feature}')
                axes[idx].grid(alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature distributions to {save_path}")
        
        return fig
    
    def plot_correlation_matrix(self, 
                               df: pd.DataFrame,
                               features: Optional[List[str]] = None,
                               method: str = 'pearson',
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot correlation matrix heatmap
        
        Args:
            df: Input dataframe
            features: List of features to include (None for all numeric)
            method: Correlation method ('pearson', 'spearman')
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if features:
            corr_df = df[features]
        else:
            corr_df = df.select_dtypes(include=[np.number])
        
        # Limit to reasonable size
        if len(corr_df.columns) > 50:
            logger.warning(f"Too many features ({len(corr_df.columns)}), showing top 50")
            corr_df = corr_df.iloc[:, :50]
        
        corr_matrix = corr_df.corr(method=method)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   ax=ax, vmin=-1, vmax=1)
        ax.set_title(f'Correlation Matrix ({method.capitalize()})')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved correlation matrix to {save_path}")
        
        return fig
    
    def plot_time_series(self, 
                        df: pd.DataFrame,
                        vehicle_id: int,
                        sensors: List[str],
                        time_col: str = 'timestamp',
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot time series for specific vehicle and sensors
        
        Args:
            df: Time series dataframe
            vehicle_id: Vehicle ID to plot
            sensors: List of sensor columns to plot
            time_col: Time column name
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        vehicle_data = df[df['vehicle_id'] == vehicle_id].copy()
        
        if len(vehicle_data) == 0:
            logger.warning(f"No data found for vehicle {vehicle_id}")
            return None
        
        n_sensors = len(sensors)
        fig, axes = plt.subplots(n_sensors, 1, figsize=(12, 3 * n_sensors), sharex=True)
        
        if n_sensors == 1:
            axes = [axes]
        
        for idx, sensor in enumerate(sensors):
            if sensor in vehicle_data.columns:
                axes[idx].plot(vehicle_data.index, vehicle_data[sensor], linewidth=1.5)
                axes[idx].set_ylabel(sensor)
                axes[idx].set_title(f'{sensor} over time')
                axes[idx].grid(alpha=0.3)
        
        axes[-1].set_xlabel('Reading Index' if time_col not in vehicle_data.columns else time_col)
        fig.suptitle(f'Time Series for Vehicle {vehicle_id}', fontsize=14, y=1.00)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved time series plot to {save_path}")
        
        return fig
    
    def plot_feature_importance(self, 
                               feature_names: List[str],
                               importances: np.ndarray,
                               top_n: int = 20,
                               title: str = "Feature Importance",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance
        
        Args:
            feature_names: List of feature names
            importances: Array of importance values
            top_n: Number of top features to show
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Create dataframe and sort
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
        ax.barh(range(len(feature_importance_df)), feature_importance_df['importance'], color='steelblue')
        ax.set_yticks(range(len(feature_importance_df)))
        ax.set_yticklabels(feature_importance_df['feature'])
        ax.set_xlabel('Importance')
        ax.set_title(f'{title} (Top {top_n})')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance to {save_path}")
        
        return fig
    
    def plot_missing_data(self, 
                         df: pd.DataFrame,
                         threshold: float = 0.0,
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize missing data patterns
        
        Args:
            df: Input dataframe
            threshold: Only show columns with missing % above threshold
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        missing_percent = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        missing_percent = missing_percent[missing_percent > threshold]
        
        if len(missing_percent) == 0:
            logger.info("No missing data found")
            return None
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(missing_percent) * 0.25)))
        ax.barh(range(len(missing_percent)), missing_percent.values, color='coral')
        ax.set_yticks(range(len(missing_percent)))
        ax.set_yticklabels(missing_percent.index)
        ax.set_xlabel('Missing Percentage (%)')
        ax.set_title('Missing Data by Feature')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved missing data plot to {save_path}")
        
        return fig


if __name__ == "__main__":
    print("Visualization Module")
    print("This module provides visualization tools for Scania predictive maintenance data")
    print("\nMain classes:")
    print("- DataVisualizer: Create various plots for EDA and model evaluation")
