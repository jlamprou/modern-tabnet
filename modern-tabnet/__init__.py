"""
Modern-TabNet: Attentive Interpretable Tabular Learning

Main Classes:
    TabNetClassifier: For classification tasks
    TabNetRegressor: For regression tasks  
    TabNetMultiTaskClassifier: For multi-task learning
    TabNetConfig: Configuration class for all hyperparameters

Quick Start:
    >>> from tabnet import TabNetClassifier, TabNetConfig
    >>> from tabnet.datasets import TabularDataset
    >>> from torch.utils.data import DataLoader
    >>> 
    >>> # Create configuration
    >>> config = TabNetConfig(n_d=32, n_a=32, learning_rate=0.02)
    >>> 
    >>> # Create model and train
    >>> model = TabNetClassifier(config)
    >>> train_loader = DataLoader(TabularDataset(X_train, y_train), batch_size=1024)
    >>> model.fit(train_loader)

Modules:
    tabnet_model: Core TabNet neural network architecture
    training: Training utilities, optimizers, and callbacks
    datasets: Dataset classes and data utilities
    metrics: Evaluation metrics
"""

# Core components
from .tabnet_model import (
    TabNet, TabNetEncoder, TabNetConfig,
    initialize_glu, initialize_non_glu,
    BaseTabNet, History, EarlyStopping,
    get_exponential_decay_scheduler, get_linear_schedule_with_warmup,
    TabNetClassifier, TabNetRegressor, TabNetMultiTaskClassifier
)

# Dataset utilities
from .datasets import (
    TabularDataset, StreamingDataset, InfiniteStreamingDataset,
    create_class_weights, create_weighted_sampler
)

# Evaluation metrics
from .metrics import (
    Metric, Accuracy, BalancedAccuracy, AUC, LogLoss, MSE, BinaryF1
)


# Version info
__version__ = "v0.1"
__author__ = "Ioannis Lamprou"

# Main exports for easy importing
__all__ = [
    # Core model classes
    "TabNetClassifier",
    "TabNetRegressor", 
    "TabNetMultiTaskClassifier",
    
    # Configuration
    "TabNetConfig",
    
    # Dataset utilities
    "TabularDataset",
    "StreamingDataset",
    "InfiniteStreamingDataset",
    "create_class_weights",
    "create_weighted_sampler",
    
    # Training utilities
    "History",
    "EarlyStopping",
    "get_exponential_decay_scheduler",
    "get_linear_schedule_with_warmup",
    
    # Metrics
    "Accuracy",
    "BalancedAccuracy", 
    "AUC",
    "LogLoss",
    "MSE",
    "BinaryF1",
    
    # Low-level components (for advanced users)
    "TabNet",
    "TabNetEncoder",
    "BaseTabNet",
    "initialize_glu",
    "initialize_non_glu",
]