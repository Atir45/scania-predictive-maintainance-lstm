"""
Scania Predictive Maintenance - Main Pipeline

This script runs the complete ML pipeline from data loading to model evaluation.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import joblib
from datetime import datetime

# Add src to path
sys.path.append('src')

from data_loading import ScaniaDataLoader
from data_preprocessing import ScaniaPreprocessor, aggregate_time_series, merge_operational_specs
from feature_engineering import TimeSeriesFeatureEngine
from evaluation import ModelEvaluator
from visualization import DataVisualizer

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ScaniaPredictiveMaintenancePipeline:
    """Complete ML pipeline for Scania component failure prediction"""
    
    def __init__(self, data_dir='data/raw', output_dir='results'):
        """
        Initialize pipeline
        
        Args:
            data_dir: Directory containing raw data
            output_dir: Directory for outputs
        """
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.models_dir = Path('models')
        self.processed_dir = Path('data/processed')
        
        # Create output directories
        (self.output_dir / 'figures').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'metrics').mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        
        logger.info("Pipeline initialized")
    
    def load_data(self):
        """Load raw data"""
        logger.info("Loading data...")
        loader = ScaniaDataLoader(self.data_dir)
        self.data = loader.load_all_data()
        logger.info("Data loaded successfully")
    
    def preprocess_data(self):
        """Preprocess and aggregate data"""
        logger.info("Preprocessing data...")
        
        preprocessor = ScaniaPreprocessor(
            missing_strategy='median',
            scaling_method='standard'
        )
        
        # Unpack data
        train_ops, train_specs, train_tte = self.data['train']
        val_ops, val_specs, val_labels = self.data['validation']
        test_ops, test_specs, test_labels = self.data['test']
        
        # Handle missing values
        logger.info("Handling missing values...")
        train_ops_clean = preprocessor.handle_missing_values(train_ops, fit=True)
        val_ops_clean = preprocessor.handle_missing_values(val_ops, fit=False)
        test_ops_clean = preprocessor.handle_missing_values(test_ops, fit=False)
        
        # Aggregate time series
        logger.info("Aggregating time series...")
        train_agg = aggregate_time_series(train_ops_clean)
        val_agg = aggregate_time_series(val_ops_clean)
        test_agg = aggregate_time_series(test_ops_clean)
        
        # Merge with specifications
        logger.info("Merging operational and specification data...")
        train_merged = merge_operational_specs(train_agg, train_specs)
        val_merged = merge_operational_specs(val_agg, val_specs)
        test_merged = merge_operational_specs(test_agg, test_specs)
        
        # Add labels
        train_merged = train_merged.merge(train_tte, on='id_vehicle', how='left')
        train_merged['label'] = (train_merged['tte'] == 0).astype(int)
        val_merged = val_merged.merge(val_labels, on='id_vehicle', how='left')
        test_merged = test_merged.merge(test_labels, on='id_vehicle', how='left')
        
        # Scale features
        logger.info("Scaling features...")
        scaler = ScaniaPreprocessor(scaling_method='standard')
        train_scaled = scaler.scale_features(train_merged, fit=True)
        val_scaled = scaler.scale_features(val_merged, fit=False)
        test_scaled = scaler.scale_features(test_merged, fit=False)
        
        # Save processed data
        train_scaled.to_csv(self.processed_dir / 'train_processed.csv', index=False)
        val_scaled.to_csv(self.processed_dir / 'validation_processed.csv', index=False)
        test_scaled.to_csv(self.processed_dir / 'test_processed.csv', index=False)
        
        self.train_df = train_scaled
        self.val_df = val_scaled
        self.test_df = test_scaled
        
        logger.info("Preprocessing completed")
    
    def prepare_features(self):
        """Prepare feature matrices"""
        logger.info("Preparing features...")
        
        feature_cols = [col for col in self.train_df.columns 
                       if col not in ['id_vehicle', 'label', 'tte']]
        
        self.X_train = self.train_df[feature_cols].fillna(0)
        self.y_train = self.train_df['label']
        
        self.X_val = self.val_df[feature_cols].fillna(0)
        self.y_val = self.val_df['label']
        
        self.X_test = self.test_df[feature_cols].fillna(0)
        self.y_test = self.test_df['label']
        
        logger.info(f"Features prepared: {self.X_train.shape[1]} features")
    
    def train_models(self):
        """Train baseline and optimized models"""
        logger.info("Training models...")
        
        # Random Forest with class balancing
        logger.info("Training Random Forest...")
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(self.X_train, self.y_train)
        
        # XGBoost
        logger.info("Training XGBoost...")
        scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        self.xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        )
        self.xgb_model.fit(self.X_train, self.y_train)
        
        logger.info("Models trained successfully")
    
    def evaluate_models(self):
        """Evaluate models on validation and test sets"""
        logger.info("Evaluating models...")
        
        evaluator = ModelEvaluator(cost_fp=10, cost_fn=500)
        
        # Evaluate on validation set
        logger.info("Validation set evaluation...")
        
        # Random Forest
        y_pred_rf = self.rf_model.predict(self.X_val)
        y_proba_rf = self.rf_model.predict_proba(self.X_val)[:, 1]
        rf_metrics = evaluator.evaluate_predictions(
            self.y_val, y_pred_rf, y_proba_rf, 
            model_name="Random Forest"
        )
        
        # XGBoost
        y_pred_xgb = self.xgb_model.predict(self.X_val)
        y_proba_xgb = self.xgb_model.predict_proba(self.X_val)[:, 1]
        xgb_metrics = evaluator.evaluate_predictions(
            self.y_val, y_pred_xgb, y_proba_xgb,
            model_name="XGBoost"
        )
        
        # Test set evaluation (best model)
        logger.info("Test set evaluation...")
        y_pred_test = self.xgb_model.predict(self.X_test)
        y_proba_test = self.xgb_model.predict_proba(self.X_test)[:, 1]
        test_metrics = evaluator.evaluate_predictions(
            self.y_test, y_pred_test, y_proba_test,
            model_name="XGBoost (Test)"
        )
        
        # Save metrics
        metrics_df = pd.DataFrame(evaluator.evaluation_results_).T
        metrics_df.to_csv(self.output_dir / 'metrics' / 'model_metrics.csv')
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        evaluator.plot_confusion_matrix(
            self.y_test, y_pred_test, 
            model_name="XGBoost (Test Set)",
            save_path=str(self.output_dir / 'figures' / 'confusion_matrix.png')
        )
        
        evaluator.plot_roc_curve(
            self.y_test, y_proba_test,
            model_name="XGBoost (Test Set)",
            save_path=str(self.output_dir / 'figures' / 'roc_curve.png')
        )
        
        evaluator.plot_precision_recall_curve(
            self.y_test, y_proba_test,
            model_name="XGBoost (Test Set)",
            save_path=str(self.output_dir / 'figures' / 'pr_curve.png')
        )
        
        logger.info("Evaluation completed")
    
    def save_models(self):
        """Save trained models"""
        logger.info("Saving models...")
        
        joblib.dump(self.rf_model, self.models_dir / 'random_forest_final.pkl')
        joblib.dump(self.xgb_model, self.models_dir / 'xgboost_final.pkl')
        
        logger.info("Models saved successfully")
    
    def run(self):
        """Run complete pipeline"""
        logger.info("="*50)
        logger.info("Starting Scania Predictive Maintenance Pipeline")
        logger.info("="*50)
        
        try:
            self.load_data()
            self.preprocess_data()
            self.prepare_features()
            self.train_models()
            self.evaluate_models()
            self.save_models()
            
            logger.info("="*50)
            logger.info("Pipeline completed successfully!")
            logger.info("="*50)
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    # Run pipeline
    pipeline = ScaniaPredictiveMaintenancePipeline()
    pipeline.run()
