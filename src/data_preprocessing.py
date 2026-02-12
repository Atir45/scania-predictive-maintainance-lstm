"""
Data Preprocessing Module for Scania Predictive Maintenance

This module handles:
- Missing value imputation
- Outlier detection and handling
- Data normalization/standardization
- Time series aggregation
- Train/validation/test split preparation
"""

import pandas as pd
import numpy as np
from typing import Optional, List
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScaniaPreprocessor:
    """Preprocess Scania sensor and specification data"""
    
    def __init__(self, 
                 missing_strategy: str = 'median',
                 scaling_method: str = 'standard',
                 outlier_threshold: float = 3.0):
        """
        Initialize preprocessor
        
        Args:
            missing_strategy: Strategy for imputing missing values ('mean', 'median', 'forward_fill')
            scaling_method: Scaling method ('standard', 'robust', 'minmax', 'none')
            outlier_threshold: Z-score threshold for outlier detection
        """
        self.missing_strategy = missing_strategy
        self.scaling_method = scaling_method
        self.outlier_threshold = outlier_threshold
        
        self.imputer = None
        self.scaler = None
        self.feature_names_ = None
        
    def analyze_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze missing data patterns
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with missing data statistics
        """
        missing_stats = pd.DataFrame({
            'column': df.columns,
            'missing_count': df.isnull().sum().values,
            'missing_percent': (df.isnull().sum() / len(df) * 100).values,
            'dtype': df.dtypes.values
        })
        
        missing_stats = missing_stats[missing_stats['missing_count'] > 0].sort_values(
            'missing_percent', ascending=False
        )
        
        logger.info(f"Found {len(missing_stats)} columns with missing values")
        logger.info(f"Total missing values: {missing_stats['missing_count'].sum()}")
        
        return missing_stats
    
    def handle_missing_values(self, 
                             df: pd.DataFrame, 
                             strategy: Optional[str] = None,
                             fit: bool = True) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input dataframe
            strategy: Imputation strategy (overrides instance default)
            fit: Whether to fit the imputer (True for training data)
            
        Returns:
            DataFrame with imputed values
        """
        strategy = strategy or self.missing_strategy
        df_copy = df.copy()
        
        # Separate ID columns and numeric columns
        id_cols = [col for col in df.columns if col in ['id_vehicle', 'timestamp', 'reading_id']]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in id_cols]
        
        if not numeric_cols:
            logger.warning("No numeric columns found for imputation")
            return df_copy
        
        if strategy in ['mean', 'median', 'most_frequent']:
            if fit:
                self.imputer = SimpleImputer(strategy=strategy)
                df_copy[numeric_cols] = self.imputer.fit_transform(df_copy[numeric_cols])
                logger.info(f"Fitted imputer with strategy: {strategy}")
            else:
                if self.imputer is None:
                    raise ValueError("Imputer not fitted. Call with fit=True first.")
                df_copy[numeric_cols] = self.imputer.transform(df_copy[numeric_cols])
        
        elif strategy == 'forward_fill':
            # Group by vehicle ID if available, then forward fill
            if 'id_vehicle' in df_copy.columns:
                df_copy[numeric_cols] = df_copy.groupby('id_vehicle')[numeric_cols].ffill()
            else:
                df_copy[numeric_cols] = df_copy[numeric_cols].ffill()
            
            # Backward fill remaining NaNs
            df_copy[numeric_cols] = df_copy[numeric_cols].bfill()
            
            # Fill any remaining with median
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].median())
        
        logger.info(f"Imputed missing values using {strategy} strategy")
        return df_copy
    
    def detect_outliers(self, df: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """
        Detect outliers using z-score method
        
        Args:
            df: Input dataframe
            threshold: Z-score threshold (overrides instance default)
            
        Returns:
            Boolean dataframe indicating outliers
        """
        threshold = threshold or self.outlier_threshold
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
        outliers = z_scores > threshold
        
        outlier_counts = outliers.sum()
        logger.info(f"Detected outliers (>{threshold} std): {outlier_counts.sum()} total")
        
        return outliers
    
    def cap_outliers(self, df: pd.DataFrame, threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Cap outliers using percentile-based method
        
        Args:
            df: Input dataframe
            threshold: Percentile threshold (e.g., 0.99 for 99th percentile)
            
        Returns:
            DataFrame with capped outliers
        """
        df_copy = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['id_vehicle', 'timestamp', 'reading_id', 'label', 'tte']:
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df_copy[col] = df_copy[col].clip(lower, upper)
        
        logger.info("Capped outliers using 1st and 99th percentiles")
        return df_copy
    
    def scale_features(self, 
                      df: pd.DataFrame, 
                      method: Optional[str] = None,
                      fit: bool = True) -> pd.DataFrame:
        """
        Scale numeric features
        
        Args:
            df: Input dataframe
            method: Scaling method (overrides instance default)
            fit: Whether to fit the scaler (True for training data)
            
        Returns:
            DataFrame with scaled features
        """
        method = method or self.scaling_method
        
        if method == 'none':
            return df
        
        df_copy = df.copy()
        
        # Identify numeric columns to scale
        id_cols = ['id_vehicle', 'timestamp', 'reading_id', 'label', 'tte']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_scale = [col for col in numeric_cols if col not in id_cols]
        
        if not cols_to_scale:
            logger.warning("No columns to scale")
            return df_copy
        
        # Initialize scaler
        if fit:
            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")
            
            df_copy[cols_to_scale] = self.scaler.fit_transform(df_copy[cols_to_scale])
            logger.info(f"Fitted and applied {method} scaler")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            df_copy[cols_to_scale] = self.scaler.transform(df_copy[cols_to_scale])
            logger.info(f"Applied {method} scaler")
        
        return df_copy
    
    def preprocess_pipeline(self, 
                           df: pd.DataFrame, 
                           fit: bool = True,
                           handle_outliers: bool = True) -> pd.DataFrame:
        """
        Complete preprocessing pipeline
        
        Args:
            df: Input dataframe
            fit: Whether to fit transformers (True for training data)
            handle_outliers: Whether to cap outliers
            
        Returns:
            Preprocessed dataframe
        """
        logger.info(f"Starting preprocessing pipeline (fit={fit})...")
        
        # 1. Analyze missing data
        if fit:
            missing_stats = self.analyze_missing_data(df)
            if len(missing_stats) > 0:
                logger.info(f"Top missing columns:\n{missing_stats.head()}")
        
        # 2. Handle missing values
        df_processed = self.handle_missing_values(df, fit=fit)
        
        # 3. Handle outliers
        if handle_outliers and fit:
            df_processed = self.cap_outliers(df_processed)
        
        # 4. Scale features
        df_processed = self.scale_features(df_processed, fit=fit)
        
        logger.info("Preprocessing pipeline completed")
        return df_processed


def aggregate_time_series(operational_df: pd.DataFrame, 
                         aggregation_funcs: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
    """
    Aggregate time series operational readouts per vehicle
    
    Args:
        operational_df: Operational readouts with id_vehicle and sensor columns
        aggregation_funcs: List of aggregation functions to apply
        
    Returns:
        Aggregated dataframe with one row per vehicle
    """
    # Identify sensor columns (exclude id and metadata columns)
    exclude_cols = ['id_vehicle', 'timestamp', 'reading_id']
    sensor_cols = [col for col in operational_df.columns if col not in exclude_cols]
    
    # Group by vehicle and aggregate
    agg_dict = {col: aggregation_funcs for col in sensor_cols}
    aggregated = operational_df.groupby('id_vehicle').agg(agg_dict)
    
    # Flatten column names
    aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]
    aggregated = aggregated.reset_index()
    
    logger.info(f"Aggregated time series from {len(operational_df)} readings to {len(aggregated)} vehicles")
    logger.info(f"Created {len(aggregated.columns) - 1} aggregated features")
    
    return aggregated


def merge_operational_specs(operational_df: pd.DataFrame, 
                           specifications_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge operational readouts with vehicle specifications
    
    Args:
        operational_df: Operational/aggregated features
        specifications_df: Vehicle specifications
        
    Returns:
        Merged dataframe
    """
    merged = operational_df.merge(specifications_df, on='id_vehicle', how='left')
    logger.info(f"Merged operational and specifications: {merged.shape}")
    return merged


if __name__ == "__main__":
    # Example usage
    print("Data Preprocessing Module")
    print("This module provides preprocessing utilities for Scania data")
    print("\nMain classes:")
    print("- ScaniaPreprocessor: Handle missing values, outliers, and scaling")
    print("\nMain functions:")
    print("- aggregate_time_series(): Aggregate operational readouts per vehicle")
    print("- merge_operational_specs(): Merge operational and specification data")
