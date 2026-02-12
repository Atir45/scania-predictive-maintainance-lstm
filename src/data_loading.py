"""
Data Loading Module for Scania Predictive Maintenance

This module handles loading raw data from CSV files including:
- Operational readouts (time series sensor data)
- Vehicle specifications
- Labels and Time-to-Event (TTE) data
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScaniaDataLoader:
    """Load and manage Scania Component X dataset"""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize data loader
        
        Args:
            data_dir: Path to directory containing raw CSV files
        """
        self.data_dir = Path(data_dir)
        self._validate_data_directory()
        
    def _validate_data_directory(self):
        """Check if data directory exists and contains expected files"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        expected_files = [
            'train_operational_readouts.csv',
            'train_specifications.csv',
            'train_tte.csv',
            'validation_operational_readouts.csv',
            'validation_specifications.csv',
            'validation_labels.csv',
            'test_operational_readouts.csv',
            'test_specifications.csv',
            'test_labels.csv'
        ]
        
        missing_files = [f for f in expected_files if not (self.data_dir / f).exists()]
        if missing_files:
            logger.warning(f"Missing files: {missing_files}")
    
    def load_train_data(self, verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load training dataset
        
        Args:
            verbose: Print loading information
            
        Returns:
            Tuple of (operational_readouts, specifications, tte_labels)
        """
        if verbose:
            logger.info("Loading training data...")
        
        operational = pd.read_csv(self.data_dir / 'train_operational_readouts.csv')
        specifications = pd.read_csv(self.data_dir / 'train_specifications.csv')
        tte = pd.read_csv(self.data_dir / 'train_tte.csv')
        
        if verbose:
            logger.info(f"Operational readouts: {operational.shape}")
            logger.info(f"Specifications: {specifications.shape}")
            logger.info(f"TTE labels: {tte.shape}")
        
        return operational, specifications, tte
    
    def load_validation_data(self, verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load validation dataset
        
        Args:
            verbose: Print loading information
            
        Returns:
            Tuple of (operational_readouts, specifications, labels)
        """
        if verbose:
            logger.info("Loading validation data...")
        
        operational = pd.read_csv(self.data_dir / 'validation_operational_readouts.csv')
        specifications = pd.read_csv(self.data_dir / 'validation_specifications.csv')
        labels = pd.read_csv(self.data_dir / 'validation_labels.csv')
        
        if verbose:
            logger.info(f"Operational readouts: {operational.shape}")
            logger.info(f"Specifications: {specifications.shape}")
            logger.info(f"Labels: {labels.shape}")
        
        return operational, specifications, labels
    
    def load_test_data(self, verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load test dataset
        
        Args:
            verbose: Print loading information
            
        Returns:
            Tuple of (operational_readouts, specifications, labels)
        """
        if verbose:
            logger.info("Loading test data...")
        
        operational = pd.read_csv(self.data_dir / 'test_operational_readouts.csv')
        specifications = pd.read_csv(self.data_dir / 'test_specifications.csv')
        labels = pd.read_csv(self.data_dir / 'test_labels.csv')
        
        if verbose:
            logger.info(f"Operational readouts: {operational.shape}")
            logger.info(f"Specifications: {specifications.shape}")
            logger.info(f"Labels: {labels.shape}")
        
        return operational, specifications, labels
    
    def load_all_data(self, verbose: bool = True) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Load all datasets (train, validation, test)
        
        Args:
            verbose: Print loading information
            
        Returns:
            Dictionary with keys 'train', 'validation', 'test' containing data tuples
        """
        if verbose:
            logger.info("Loading all datasets...")
        
        data = {
            'train': self.load_train_data(verbose=verbose),
            'validation': self.load_validation_data(verbose=verbose),
            'test': self.load_test_data(verbose=verbose)
        }
        
        return data
    
    def get_dataset_info(self) -> Dict[str, Dict[str, int]]:
        """
        Get information about dataset sizes without loading all data
        
        Returns:
            Dictionary with dataset information
        """
        info = {}
        
        # Check training data
        train_tte = pd.read_csv(self.data_dir / 'train_tte.csv', usecols=[0])
        info['train'] = {
            'n_samples': len(train_tte),
            'split': 'train'
        }
        
        # Check validation data
        val_labels = pd.read_csv(self.data_dir / 'validation_labels.csv', usecols=[0])
        info['validation'] = {
            'n_samples': len(val_labels),
            'split': 'validation'
        }
        
        # Check test data
        test_labels = pd.read_csv(self.data_dir / 'test_labels.csv', usecols=[0])
        info['test'] = {
            'n_samples': len(test_labels),
            'split': 'test'
        }
        
        info['total'] = {
            'n_samples': sum(split_info['n_samples'] for split_info in info.values() if isinstance(split_info, dict))
        }
        
        return info


def load_scania_data(split: str = 'train', data_dir: str = 'data/raw') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load Scania data for a specific split
    
    Args:
        split: Data split to load ('train', 'validation', or 'test')
        data_dir: Path to raw data directory
        
    Returns:
        Tuple of (operational_readouts, specifications, labels/tte)
    """
    loader = ScaniaDataLoader(data_dir)
    
    if split == 'train':
        return loader.load_train_data()
    elif split == 'validation':
        return loader.load_validation_data()
    elif split == 'test':
        return loader.load_test_data()
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'validation', or 'test'")


if __name__ == "__main__":
    # Example usage
    loader = ScaniaDataLoader()
    
    # Get dataset information
    info = loader.get_dataset_info()
    print("\nDataset Information:")
    print(f"Training samples: {info['train']['n_samples']}")
    print(f"Validation samples: {info['validation']['n_samples']}")
    print(f"Test samples: {info['test']['n_samples']}")
    print(f"Total samples: {info['total']['n_samples']}")
    
    # Load training data
    print("\nLoading training data...")
    train_ops, train_specs, train_tte = loader.load_train_data()
    print(f"\nTraining data loaded successfully!")
    print(f"Operational readouts shape: {train_ops.shape}")
    print(f"Specifications shape: {train_specs.shape}")
    print(f"TTE labels shape: {train_tte.shape}")
