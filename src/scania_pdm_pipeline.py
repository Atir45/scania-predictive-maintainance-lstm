"""
Scania Component X Predictive Maintenance Pipeline
==================================================

This module implements a complete ML pipeline for predicting component failures
in Scania trucks using multivariate time series sensor data.

Pipeline Flow:
Raw Data (CSV) → Preprocessing → Feature Engineering → Model Training → Evaluation

Models:
Compares different Decision Tree variants:
- Gini Impurity criterion (unpruned)
- Entropy/Information Gain criterion (unpruned)
- Pre-pruned trees (depth-limited: 10, 15)
- Min-samples pruned tree (split & leaf constraints)
- Cost-Complexity pruned tree (post-pruning with ccp_alpha)

Author: ML Engineer
Date: November 2025
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')


class ScaniaPdMPipeline:
    """
    Complete pipeline for Scania Component X predictive maintenance.
    
    This class handles data loading, feature engineering, model training,
    and evaluation for predicting truck component failures.
    """
    
    def __init__(self, data_path):
        """
        Initialize the pipeline with data path.
        
        Args:
            data_path (str): Path to directory containing CSV files
        """
        self.data_path = data_path
        self.models = {}
        self.preprocessors = {}
        self.scaler = None
        
    def load_data(self):
        """Load all dataset components"""
        print("Loading Scania datasets...")
        
        try:
            # Load training data
            self.train_ops = pd.read_csv(f"{self.data_path}/train_operational_readouts.csv")
            self.train_specs = pd.read_csv(f"{self.data_path}/train_specifications.csv")
            self.train_tte = pd.read_csv(f"{self.data_path}/train_tte.csv")
            
            # Load validation data
            self.val_ops = pd.read_csv(f"{self.data_path}/validation_operational_readouts.csv")
            self.val_specs = pd.read_csv(f"{self.data_path}/validation_specifications.csv")
            self.val_labels = pd.read_csv(f"{self.data_path}/validation_labels.csv")
            
            print(f"Training operational data: {self.train_ops.shape}")
            print(f"Training specifications: {self.train_specs.shape}")
            print(f"Training labels: {self.train_tte.shape}")
            print("Data loading completed successfully!")
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error during data loading: {e}")
            raise
            
        return self
    
    def engineer_features(self, df_ops, df_specs, max_vehicles=None):
        """
        Create features from time series operational data.
        
        Args:
            df_ops (pd.DataFrame): Operational readouts data
            df_specs (pd.DataFrame): Specifications data
            max_vehicles (int, optional): Limit number of vehicles for testing
            
        Returns:
            pd.DataFrame: Engineered features dataset
        """
        print("Starting feature engineering...")
        
        features = []
        unique_vehicles = df_ops['vehicle_id'].unique()
        
        # Limit vehicles for testing if specified
        if max_vehicles:
            unique_vehicles = unique_vehicles[:max_vehicles]
            print(f"Processing {len(unique_vehicles)} vehicles (limited for testing)")
        else:
            print(f"Processing {len(unique_vehicles)} vehicles")
        
        # Process each vehicle
        for i, vehicle_id in enumerate(unique_vehicles):
            if i % 1000 == 0:
                print(f"Processing vehicle {i+1}/{len(unique_vehicles)}")
                
            try:
                vehicle_data = df_ops[df_ops['vehicle_id'] == vehicle_id].copy()
                
                # Sort by time_step
                vehicle_data = vehicle_data.sort_values('time_step')
                
                # Get specification data for this vehicle
                spec_data = df_specs[df_specs['vehicle_id'] == vehicle_id]
                if spec_data.empty:
                    continue
                spec_data = spec_data.iloc[0]
                
                # Initialize feature dictionary
                feature_dict = {'vehicle_id': vehicle_id}
                
                # Get sensor columns (excluding vehicle_id and time_step)
                sensor_cols = [col for col in vehicle_data.columns 
                              if col not in ['vehicle_id', 'time_step']]
                
                # 1. Statistical features
                for col in sensor_cols:
                    values = pd.to_numeric(vehicle_data[col], errors='coerce').dropna()
                    if len(values) > 0:
                        feature_dict[f'{col}_mean'] = values.mean()
                        feature_dict[f'{col}_std'] = values.std() if len(values) > 1 else 0
                        feature_dict[f'{col}_min'] = values.min()
                        feature_dict[f'{col}_max'] = values.max()
                        feature_dict[f'{col}_range'] = values.max() - values.min()
                        feature_dict[f'{col}_median'] = values.median()
                
                # 2. Trend features (limited to first 10 sensors for performance)
                for col in sensor_cols[:10]:
                    values = pd.to_numeric(vehicle_data[col], errors='coerce').dropna()
                    if len(values) > 2:
                        try:
                            x = np.arange(len(values))
                            slope = np.polyfit(x, values, 1)[0]
                            feature_dict[f'{col}_trend'] = slope
                        except:
                            feature_dict[f'{col}_trend'] = 0
                
                # 3. Rolling window features (last 30% of data)
                window_size = max(3, len(vehicle_data) // 3)
                recent_data = vehicle_data.tail(window_size)
                
                for col in sensor_cols[:10]:  # Limit for computation
                    values = pd.to_numeric(recent_data[col], errors='coerce').dropna()
                    if len(values) > 0:
                        feature_dict[f'{col}_recent_mean'] = values.mean()
                        feature_dict[f'{col}_recent_std'] = values.std() if len(values) > 1 else 0
                
                # 4. Add specification features
                for spec_col in df_specs.columns:
                    if spec_col != 'vehicle_id':
                        feature_dict[spec_col] = spec_data[spec_col]
                
                # 5. Temporal features
                feature_dict['sequence_length'] = len(vehicle_data)
                feature_dict['max_time_step'] = vehicle_data['time_step'].max()
                feature_dict['min_time_step'] = vehicle_data['time_step'].min()
                feature_dict['time_range'] = (vehicle_data['time_step'].max() - 
                                            vehicle_data['time_step'].min())
                
                features.append(feature_dict)
                
            except Exception as e:
                print(f"Error processing vehicle {vehicle_id}: {e}")
                continue
        
        print(f"Feature engineering completed. Created {len(features)} feature sets.")
        return pd.DataFrame(features)
    
    def preprocess_features(self, df_features, is_training=True):
        """
        Preprocess engineered features.
        
        Args:
            df_features (pd.DataFrame): Raw engineered features
            is_training (bool): Whether this is training data
            
        Returns:
            pd.DataFrame: Preprocessed features
        """
        print("Preprocessing features...")
        
        # Create a copy to avoid modifying original
        df_processed = df_features.copy()
        
        # Separate column types
        categorical_cols = [col for col in df_processed.columns 
                           if df_processed[col].dtype == 'object' and col != 'vehicle_id']
        numerical_cols = [col for col in df_processed.columns 
                         if df_processed[col].dtype in ['float64', 'int64'] and col != 'vehicle_id']
        
        print(f"Found {len(categorical_cols)} categorical and {len(numerical_cols)} numerical features")
        
        # Handle categorical features with one-hot encoding
        if categorical_cols:
            df_processed = pd.get_dummies(df_processed, columns=categorical_cols, prefix=categorical_cols)
        
        # Update numerical columns after one-hot encoding
        numerical_cols = [col for col in df_processed.columns 
                         if df_processed[col].dtype in ['float64', 'int64'] and col != 'vehicle_id']
        
        # Handle missing values
        if numerical_cols:
            # Fill missing values with median
            for col in numerical_cols:
                if df_processed[col].isnull().any():
                    median_val = df_processed[col].median()
                    df_processed[col].fillna(median_val, inplace=True)
            
            # Scale numerical features
            if is_training:
                self.scaler = StandardScaler()
                scaled_features = self.scaler.fit_transform(df_processed[numerical_cols])
                print("Fitted new scaler for training data")
            else:
                if self.scaler is None:
                    raise ValueError("Scaler not fitted. Please train on training data first.")
                scaled_features = self.scaler.transform(df_processed[numerical_cols])
                print("Used existing scaler for validation/test data")
            
            # Replace scaled values
            df_processed[numerical_cols] = scaled_features
        
        print("Feature preprocessing completed")
        return df_processed
    
    def train_models(self, X_train, y_train):
        """
        Train multiple decision tree models with different configurations.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
        """
        print("Training decision tree models...")
        
        # Calculate class weights for imbalanced data
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        print(f"Class distribution: {np.bincount(y_train)}")
        print(f"Class weights: {class_weight_dict}")
        
        # Define different decision tree variants for comparison
        models = {
            'dt_gini': DecisionTreeClassifier(
                criterion='gini',
                class_weight='balanced',
                random_state=42,
                max_depth=None,  # Unpruned tree
                min_samples_split=2,
                min_samples_leaf=1
            ),
            'dt_entropy': DecisionTreeClassifier(
                criterion='entropy',
                class_weight='balanced',
                random_state=42,
                max_depth=None,  # Unpruned tree
                min_samples_split=2,
                min_samples_leaf=1
            ),
            'dt_pruned_depth10': DecisionTreeClassifier(
                criterion='gini',
                class_weight='balanced',
                random_state=42,
                max_depth=10,  # Pre-pruned by depth
                min_samples_split=2,
                min_samples_leaf=1
            ),
            'dt_pruned_depth15': DecisionTreeClassifier(
                criterion='gini',
                class_weight='balanced',
                random_state=42,
                max_depth=15,  # Pre-pruned by depth
                min_samples_split=2,
                min_samples_leaf=1
            ),
            'dt_min_samples': DecisionTreeClassifier(
                criterion='gini',
                class_weight='balanced',
                random_state=42,
                max_depth=None,
                min_samples_split=20,  # Require more samples to split
                min_samples_leaf=10    # Require more samples in leaf
            ),
            'dt_cost_complexity': DecisionTreeClassifier(
                criterion='gini',
                class_weight='balanced',
                random_state=42,
                ccp_alpha=0.001,  # Cost-complexity pruning parameter
                max_depth=None
            )
        }
        
        print(f"\nTraining {len(models)} decision tree variants:")
        print("1. Gini Impurity (unpruned)")
        print("2. Entropy/Information Gain (unpruned)")
        print("3. Gini with max_depth=10 (pre-pruned)")
        print("4. Gini with max_depth=15 (pre-pruned)")
        print("5. Gini with min_samples constraints (pruned)")
        print("6. Cost-Complexity Pruned (post-pruning)")
        
        # Train models
        for name, model in models.items():
            print(f"\nTraining {name}...")
            try:
                model.fit(X_train, y_train)
                self.models[name] = model
                
                # Print tree info
                n_nodes = model.tree_.node_count
                max_depth_actual = model.tree_.max_depth
                n_leaves = model.tree_.n_leaves
                print(f"  ├─ Nodes: {n_nodes}")
                print(f"  ├─ Max depth: {max_depth_actual}")
                print(f"  └─ Leaves: {n_leaves}")
                
            except Exception as e:
                print(f"Error training {name}: {e}")
        
        return self
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
            
        Returns:
            dict: Evaluation results for each model
        """
        print("Evaluating models...")
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n=== {name.upper()} EVALUATION ===")
            
            try:
                # Predictions
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                
                # Metrics
                auc_score = roc_auc_score(y_test, y_proba)
                
                print(f"AUC-ROC Score: {auc_score:.4f}")
                print("\nClassification Report:")
                print(classification_report(y_test, y_pred))
                print("\nConfusion Matrix:")
                print(confusion_matrix(y_test, y_pred))
                
                results[name] = {
                    'auc_roc': auc_score,
                    'predictions': y_pred,
                    'probabilities': y_proba
                }
                
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def run_complete_pipeline(self, max_vehicles_train=None, max_vehicles_val=None):
        """
        Execute the complete ML pipeline.
        
        Args:
            max_vehicles_train (int, optional): Limit training vehicles for testing
            max_vehicles_val (int, optional): Limit validation vehicles for testing
            
        Returns:
            dict: Evaluation results
        """
        print("=== STARTING COMPLETE SCANIA PDM PIPELINE ===")
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Feature engineering
            print("\n--- FEATURE ENGINEERING ---")
            train_features = self.engineer_features(
                self.train_ops, self.train_specs, max_vehicles_train
            )
            val_features = self.engineer_features(
                self.val_ops, self.val_specs, max_vehicles_val
            )
            
            # Step 3: Preprocessing
            print("\n--- PREPROCESSING ---")
            X_train = self.preprocess_features(train_features, is_training=True)
            X_val = self.preprocess_features(val_features, is_training=False)
            
            # Step 4: Prepare labels
            print("\n--- PREPARING LABELS ---")
            y_train = self.train_tte.set_index('vehicle_id')['in_study_repair']
            y_val = self.val_labels.set_index('vehicle_id')['class_label']
            
            # Step 5: Align features and labels
            common_train_ids = X_train['vehicle_id'].isin(y_train.index)
            common_val_ids = X_val['vehicle_id'].isin(y_val.index)
            
            X_train_final = X_train[common_train_ids].set_index('vehicle_id')
            X_val_final = X_val[common_val_ids].set_index('vehicle_id')
            
            y_train_final = y_train[X_train_final.index]
            y_val_final = y_val[X_val_final.index]
            
            print(f"Final training set: {X_train_final.shape[0]} vehicles, {X_train_final.shape[1]} features")
            print(f"Final validation set: {X_val_final.shape[0]} vehicles, {X_val_final.shape[1]} features")
            
            # Step 6: Train models
            print("\n--- MODEL TRAINING ---")
            self.train_models(X_train_final, y_train_final)
            
            # Step 7: Evaluate models
            print("\n--- MODEL EVALUATION ---")
            results = self.evaluate_models(X_val_final, y_val_final)
            
            print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
            return results
            
        except Exception as e:
            print(f"\nPipeline failed with error: {e}")
            raise


def main():
    """Main function to run the pipeline"""
    # Initialize pipeline
    data_path = "c:/Users/Iduma/OneDrive - University of Hertfordshire/Desktop/datasets"
    pipeline = ScaniaPdMPipeline(data_path)
    
    # Run pipeline with limited vehicles for testing (remove limits for full run)
    print("Running pipeline with limited vehicles for testing...")
    print("For full dataset, call: pipeline.run_complete_pipeline()")
    
    results = pipeline.run_complete_pipeline(
        max_vehicles_train=1000,  # Limit for testing
        max_vehicles_val=200      # Limit for testing
    )
    
    return results


if __name__ == "__main__":
    results = main()