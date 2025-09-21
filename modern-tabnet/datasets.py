# Dataset utilities and data handling for TabNet

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import numpy as np
from typing import Optional


# ============================================================================
# Utility Functions for Dataset Creation
# ============================================================================

def create_class_weights(y: np.ndarray) -> torch.Tensor:
    """Create balanced class weights."""
    unique_classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = total / (len(unique_classes) * counts)
    
    weight_dict = dict(zip(unique_classes, weights))
    class_weights = torch.tensor([weight_dict[cls] for cls in unique_classes], dtype=torch.float32)
    
    return class_weights


def create_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    """Create weighted sampler for imbalanced datasets."""
    class_weights = create_class_weights(y)
    sample_weights = torch.tensor([class_weights[int(label)] for label in y])
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=False
    )


class TabularDataset(Dataset):
    """Simple tabular dataset class."""
    
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)) if y is not None else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# ============================================================================
# Streaming Dataset Examples
# ============================================================================

class StreamingDataset(Dataset):
    """Example streaming dataset that generates data on-the-fly."""
    
    def __init__(self, n_samples: int, n_features: int, n_classes: int = 2, 
                 chunk_size: int = 1000, seed: int = 42):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.chunk_size = chunk_size
        self.seed = seed
        self.current_chunk = None
        self.current_chunk_idx = 0
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Generate data chunk if needed
        chunk_idx = idx // self.chunk_size
        local_idx = idx % self.chunk_size
        
        if self.current_chunk is None or chunk_idx != self.current_chunk_idx:
            self._generate_chunk(chunk_idx)
        
        return self.current_chunk[local_idx]
    
    def _generate_chunk(self, chunk_idx):
        """Generate a chunk of data on-the-fly."""
        np.random.seed(self.seed + chunk_idx)
        
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.n_samples)
        chunk_size = end_idx - start_idx
        
        # Generate synthetic data
        X = np.random.randn(chunk_size, self.n_features).astype(np.float32)
        # Add some pattern to make it learnable
        y = ((X[:, 0] + X[:, 1] + np.random.normal(0, 0.1, chunk_size)) > 0).astype(np.float32)
        
        self.current_chunk = [(torch.from_numpy(X[i]), torch.from_numpy(y[i:i+1])) 
                             for i in range(chunk_size)]
        self.current_chunk_idx = chunk_idx


class InfiniteStreamingDataset(Dataset):
    """Example infinite streaming dataset that never ends."""
    
    def __init__(self, n_features: int, n_classes: int = 2, seed: int = 42):
        self.n_features = n_features
        self.n_classes = n_classes
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def __len__(self):
        # Infinite dataset - this will cause len() to fail
        raise NotImplementedError("This is an infinite streaming dataset")
    
    def __iter__(self):
        """Return an iterator for infinite streaming."""
        return self
    
    def __next__(self):
        """Generate next sample on-the-fly."""
        X = self.rng.randn(self.n_features).astype(np.float32)
        # Create learnable pattern
        y = np.array([(X[0] + X[1] + self.rng.normal(0, 0.1)) > 0], dtype=np.float32)
        
        return torch.from_numpy(X), torch.from_numpy(y)