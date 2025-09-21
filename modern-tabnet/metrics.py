# Evaluation metrics for TabNet

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, balanced_accuracy_score, log_loss, f1_score


# ============================================================================
# Training and Evaluation Components
# ============================================================================

class Metric:
    """Base metric class."""
    
    def __init__(self, name: str, maximize: bool = True):
        self.name = name
        self.maximize = maximize
    
    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass


class Accuracy(Metric):
    def __init__(self):
        super().__init__("accuracy", maximize=True)
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)
        return accuracy_score(y_true, y_pred)


class BalancedAccuracy(Metric):
    def __init__(self):
        super().__init__("balanced_accuracy", maximize=True)
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)
        return balanced_accuracy_score(y_true, y_pred)


class AUC(Metric):
    def __init__(self):
        super().__init__("auc", maximize=True)
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_pred.ndim > 1:
            y_pred = y_pred[:, 1]
        return roc_auc_score(y_true, y_pred)


class LogLoss(Metric):
    def __init__(self):
        super().__init__("logloss", maximize=False)
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return log_loss(y_true, y_pred)


class MSE(Metric):
    def __init__(self):
        super().__init__("mse", maximize=False)
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return mean_squared_error(y_true, y_pred)


class BinaryF1(Metric):
    def __init__(self):
        super().__init__("binary_f1", maximize=True)
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)
        return f1_score(y_true, y_pred, average='binary', zero_division=0)
