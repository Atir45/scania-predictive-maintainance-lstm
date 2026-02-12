"""
Feature Engineering Module for Scania Predictive Maintenance

This module handles advanced feature engineering including:
- Time series statistical features
- Rolling window features
- Trend and momentum features
- Sensor interaction features
- Domain-specific features
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesFeatureEngine:
    """Extract advanced time series features from sensor data"""
    
    def __init__(self, window_sizes: List[int] = [5, 10, 20]):
        """
        Initialize feature engine
        
        Args:
            window_sizes: List of window sizes for rolling features
        """
        self.window_sizes = window_sizes
        self.feature_names_ = []
        
    def extract_statistical_features(self, df: pd.DataFrame, group_col: str = 'id_vehicle') -> pd.DataFrame:
        """
        Extract statistical features from time series
        
        Args:
            df: Time series dataframe with sensor readings
            group_col: Column to group by (vehicle ID)
            
        Returns:
            DataFrame with statistical features per vehicle
        """
        # Identify sensor columns
        exclude_cols = [group_col, 'timestamp', 'reading_id', 'label', 'tte']
        sensor_cols = [col for col in df.columns if col not in exclude_cols]
        
        features_list = []
        
        for vehicle_id, group in df.groupby(group_col):
            vehicle_features = {group_col: vehicle_id}
            
            for col in sensor_cols:
                values = group[col].dropna()
                
                if len(values) == 0:
                    continue
                
                # Basic statistics
                vehicle_features[f'{col}_mean'] = values.mean()
                vehicle_features[f'{col}_std'] = values.std()
                vehicle_features[f'{col}_min'] = values.min()
                vehicle_features[f'{col}_max'] = values.max()
                vehicle_features[f'{col}_median'] = values.median()
                vehicle_features[f'{col}_range'] = values.max() - values.min()
                
                # Distribution features
                vehicle_features[f'{col}_skew'] = values.skew()
                vehicle_features[f'{col}_kurtosis'] = values.kurtosis()
                
                # Percentiles
                vehicle_features[f'{col}_q25'] = values.quantile(0.25)
                vehicle_features[f'{col}_q75'] = values.quantile(0.75)
                vehicle_features[f'{col}_iqr'] = values.quantile(0.75) - values.quantile(0.25)
                
                # Count and completeness
                vehicle_features[f'{col}_count'] = len(values)
                vehicle_features[f'{col}_missing_ratio'] = group[col].isnull().sum() / len(group)
            
            features_list.append(vehicle_features)
        
        features_df = pd.DataFrame(features_list)
        logger.info(f"Extracted {len(features_df.columns) - 1} statistical features for {len(features_df)} vehicles")
        
        return features_df
    
    def extract_rolling_features(self, df: pd.DataFrame, group_col: str = 'id_vehicle') -> pd.DataFrame:
        """
        Extract rolling window features
        
        Args:
            df: Time series dataframe sorted by timestamp
            group_col: Column to group by (vehicle ID)
            
        Returns:
            DataFrame with rolling features aggregated per vehicle
        """
        # Identify sensor columns
        exclude_cols = [group_col, 'timestamp', 'reading_id', 'label', 'tte']
        sensor_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Limit to subset of key sensors to avoid feature explosion
        # In practice, select most important sensors
        key_sensors = sensor_cols[:10] if len(sensor_cols) > 10 else sensor_cols
        
        features_list = []
        
        for vehicle_id, group in df.groupby(group_col):
            vehicle_features = {group_col: vehicle_id}
            
            # Sort by timestamp if available
            if 'timestamp' in group.columns:
                group = group.sort_values('timestamp')
            
            for col in key_sensors:
                for window in self.window_sizes:
                    if len(group) >= window:
                        rolling = group[col].rolling(window=window, min_periods=1)
                        
                        # Rolling statistics - take last value
                        vehicle_features[f'{col}_rolling_{window}_mean'] = rolling.mean().iloc[-1]
                        vehicle_features[f'{col}_rolling_{window}_std'] = rolling.std().iloc[-1]
                        vehicle_features[f'{col}_rolling_{window}_max'] = rolling.max().iloc[-1]
                        vehicle_features[f'{col}_rolling_{window}_min'] = rolling.min().iloc[-1]
            
            features_list.append(vehicle_features)
        
        features_df = pd.DataFrame(features_list)
        logger.info(f"Extracted {len(features_df.columns) - 1} rolling features for {len(features_df)} vehicles")
        
        return features_df
    
    def extract_trend_features(self, df: pd.DataFrame, group_col: str = 'id_vehicle') -> pd.DataFrame:
        """
        Extract trend and momentum features
        
        Args:
            df: Time series dataframe
            group_col: Column to group by (vehicle ID)
            
        Returns:
            DataFrame with trend features per vehicle
        """
        # Identify sensor columns
        exclude_cols = [group_col, 'timestamp', 'reading_id', 'label', 'tte']
        sensor_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Limit to key sensors
        key_sensors = sensor_cols[:15] if len(sensor_cols) > 15 else sensor_cols
        
        features_list = []
        
        for vehicle_id, group in df.groupby(group_col):
            vehicle_features = {group_col: vehicle_id}
            
            # Sort by timestamp if available
            if 'timestamp' in group.columns:
                group = group.sort_values('timestamp')
            
            for col in key_sensors:
                values = group[col].dropna().values
                
                if len(values) < 2:
                    continue
                
                # Linear trend (slope)
                x = np.arange(len(values))
                if len(values) > 1 and values.std() > 0:
                    slope, _intercept, r_value, _p_value, _std_err = stats.linregress(x, values)
                    vehicle_features[f'{col}_trend_slope'] = slope
                    vehicle_features[f'{col}_trend_r2'] = r_value ** 2
                
                # First vs last comparison
                vehicle_features[f'{col}_first_last_diff'] = values[-1] - values[0]
                vehicle_features[f'{col}_first_last_ratio'] = values[-1] / values[0] if values[0] != 0 else 0
                
                # Rate of change
                if len(values) > 1:
                    changes = np.diff(values)
                    vehicle_features[f'{col}_mean_change'] = changes.mean()
                    vehicle_features[f'{col}_std_change'] = changes.std()
            
            features_list.append(vehicle_features)
        
        features_df = pd.DataFrame(features_list)
        logger.info(f"Extracted {len(features_df.columns) - 1} trend features for {len(features_df)} vehicles")
        
        return features_df
    
    def extract_all_features(self, df: pd.DataFrame, group_col: str = 'id_vehicle') -> pd.DataFrame:
        """
        Extract all time series features
        
        Args:
            df: Time series dataframe
            group_col: Column to group by
            
        Returns:
            DataFrame with all extracted features
        """
        logger.info("Extracting all time series features...")
        
        # Extract different feature types
        stat_features = self.extract_statistical_features(df, group_col)
        trend_features = self.extract_trend_features(df, group_col)
        
        # Merge all features
        all_features = stat_features.merge(trend_features, on=group_col, how='outer')
        
        # Optionally add rolling features (can be expensive)
        # rolling_features = self.extract_rolling_features(df, group_col)
        # all_features = all_features.merge(rolling_features, on=group_col, how='outer')
        
        logger.info(f"Total features extracted: {len(all_features.columns) - 1}")
        self.feature_names_ = [col for col in all_features.columns if col != group_col]
        
        return all_features


class SensorInteractionFeatures:
    """Create interaction features between sensors"""
    
    def __init__(self, max_interactions: int = 50):
        """
        Initialize interaction feature creator
        
        Args:
            max_interactions: Maximum number of interaction features to create
        """
        self.max_interactions = max_interactions
    
    def create_ratio_features(self, df: pd.DataFrame, sensor_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create ratio features between sensor pairs
        
        Args:
            df: Input dataframe with sensor features
            sensor_pairs: List of (sensor1, sensor2) tuples
            
        Returns:
            DataFrame with ratio features
        """
        df_copy = df.copy()
        
        for sensor1, sensor2 in sensor_pairs[:self.max_interactions]:
            if sensor1 in df.columns and sensor2 in df.columns:
                # Avoid division by zero
                df_copy[f'{sensor1}_div_{sensor2}'] = df[sensor1] / (df[sensor2].replace(0, np.nan))
        
        logger.info(f"Created {min(len(sensor_pairs), self.max_interactions)} ratio features")
        return df_copy
    
    def create_product_features(self, df: pd.DataFrame, sensor_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create product features between sensor pairs
        
        Args:
            df: Input dataframe with sensor features
            sensor_pairs: List of (sensor1, sensor2) tuples
            
        Returns:
            DataFrame with product features
        """
        df_copy = df.copy()
        
        for sensor1, sensor2 in sensor_pairs[:self.max_interactions]:
            if sensor1 in df.columns and sensor2 in df.columns:
                df_copy[f'{sensor1}_mul_{sensor2}'] = df[sensor1] * df[sensor2]
        
        logger.info(f"Created {min(len(sensor_pairs), self.max_interactions)} product features")
        return df_copy


def select_top_features(X: pd.DataFrame, y: pd.Series, n_features: int = 100, 
                       method: str = 'mutual_info') -> List[str]:
    """
    Select top features using feature importance
    
    Args:
        X: Feature dataframe
        y: Target variable
        n_features: Number of top features to select
        method: Selection method ('mutual_info', 'chi2', 'variance')
        
    Returns:
        List of selected feature names
    """
    from sklearn.feature_selection import mutual_info_classif, f_classif
    
    if method == 'mutual_info':
        scores = mutual_info_classif(X, y, random_state=42)
    elif method == 'f_classif':
        scores = f_classif(X, y)[0]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Get top features
    feature_scores = pd.DataFrame({
        'feature': X.columns,
        'score': scores
    }).sort_values('score', ascending=False)
    
    top_features = feature_scores.head(n_features)['feature'].tolist()
    
    logger.info(f"Selected top {n_features} features using {method}")
    return top_features


if __name__ == "__main__":
    print("Feature Engineering Module")
    print("This module provides advanced feature engineering for time series sensor data")
    print("\nMain classes:")
    print("- TimeSeriesFeatureEngine: Extract statistical, rolling, and trend features")
    print("- SensorInteractionFeatures: Create interaction features between sensors")
    print("\nMain functions:")
    print("- select_top_features(): Feature selection using various methods")
