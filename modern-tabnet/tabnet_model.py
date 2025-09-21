#TabNet with safe optimizations, streaming support, and weighted loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Function
import numpy as np
import warnings
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from tqdm.auto import tqdm
from sklearn.base import BaseEstimator
from scipy.sparse import csc_matrix
import copy
import json
import zipfile
import shutil
from pathlib import Path
import scipy

from tabnet.datasets import create_class_weights
from tabnet.metrics import AUC, MSE, Accuracy, BalancedAccuracy, BinaryF1, LogLoss, Metric
# Weights & Biases support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("wandb not available. Install with 'pip install wandb' for experiment tracking.")


torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

# ============================================================================
# Original TabNet Components (Restored)
# ============================================================================

def initialize_glu(module: nn.Module, input_dim: int, output_dim: int):
    """Initialize GLU layers exactly as in paper."""
    # Paper formula: gain_value = sqrt((input_dim + output_dim) / sqrt(input_dim))
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
    nn.init.xavier_normal_(module.weight, gain=gain_value)


def initialize_non_glu(module: nn.Module, input_dim: int, output_dim: int):
    """Initialize non-GLU layers exactly as in paper."""
    # Paper formula: gain_value = sqrt((input_dim + output_dim) / sqrt(4 * input_dim))
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(4 * input_dim))
    nn.init.xavier_normal_(module.weight, gain=gain_value)

def get_exponential_decay_scheduler(optimizer, decay_rate: float = 0.9, decay_steps: int = 8000):
    """
    Exponential learning rate decay exactly as used in TabNet paper.
    
    Args:
        optimizer: PyTorch optimizer
        decay_rate: Decay rate (0.9 in paper)
        decay_steps: Steps between decay (8k iterations in paper)
    
    Returns:
        LR scheduler that matches paper implementation
    """
    def lr_lambda(current_step):
        return decay_rate ** (current_step // decay_steps)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class GhostBatchNorm(nn.Module):
    """Ghost Batch Normalization with safe optimization."""
    
    def __init__(self, input_dim: int, virtual_batch_size: int = 128, momentum: float = 0.01):
        super().__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(input_dim, momentum=momentum)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Safe optimization: early return for small batches
        if x.size(0) <= self.virtual_batch_size:
            return self.bn(x)
        
        # Original chunking logic for larger batches
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]
        return torch.cat(res, dim=0)


def _make_ix_like(input, dim=0):
    """Helper function for sparsemax - from original implementation."""
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


class SparsemaxFunction(Function):
    """Original Sparsemax implementation with custom backward pass."""
    
    @staticmethod
    def forward(ctx, input, dim=-1):
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val
        tau, supp_size = SparsemaxFunction._threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None

    @staticmethod
    def _threshold_and_support(input, dim=-1):
        input_srt, _ = torch.sort(input, descending=True, dim=dim)
        input_cumsum = input_srt.cumsum(dim) - 1
        rhos = _make_ix_like(input, dim)
        support = rhos * input_srt > input_cumsum

        support_size = support.sum(dim=dim).unsqueeze(dim)
        tau = input_cumsum.gather(dim, support_size - 1)
        tau /= support_size.to(input.dtype)
        return tau, support_size


sparsemax = SparsemaxFunction.apply


class Sparsemax(nn.Module):
    """Original Sparsemax module."""
    def __init__(self, dim=-1):
        self.dim = dim
        super(Sparsemax, self).__init__()

    def forward(self, input):
        return sparsemax(input, self.dim)


class Entmax15Function(Function):
    """Original Entmax implementation."""
    
    @staticmethod
    def forward(ctx, input, dim=-1):
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input = input - max_val
        input = input / 2

        tau_star, _ = Entmax15Function._threshold_and_support(input, dim)
        output = torch.clamp(input - tau_star, min=0) ** 2
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        Y, = ctx.saved_tensors
        gppr = Y.sqrt()
        dX = grad_output * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None

    @staticmethod
    def _threshold_and_support(input, dim=-1):
        Xsrt, _ = torch.sort(input, descending=True, dim=dim)
        rho = _make_ix_like(input, dim)
        mean = Xsrt.cumsum(dim) / rho
        mean_sq = (Xsrt ** 2).cumsum(dim) / rho
        ss = rho * (mean_sq - mean ** 2)
        delta = (1 - ss) / rho

        delta_nz = torch.clamp(delta, 0)
        tau = mean - torch.sqrt(delta_nz)

        support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
        tau_star = tau.gather(dim, support_size - 1)
        return tau_star, support_size


entmax15 = Entmax15Function.apply


class Entmax15(nn.Module):
    def __init__(self, dim=-1):
        self.dim = dim
        super(Entmax15, self).__init__()

    def forward(self, input):
        return entmax15(input, self.dim)


class GLULayer(nn.Module):
    """Original GLU Layer implementation."""
    
    def __init__(self, input_dim: int, output_dim: int, fc=None, virtual_batch_size: int = 128, momentum: float = 0.02):
        super().__init__()
        self.output_dim = output_dim
        if fc:
            self.fc = fc
        else:
            self.fc = nn.Linear(input_dim, 2 * output_dim, bias=False)
        initialize_glu(self.fc, input_dim, 2 * output_dim)

        self.bn = GhostBatchNorm(2 * output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.bn(x)
        out = torch.mul(x[:, : self.output_dim], torch.sigmoid(x[:, self.output_dim :]))
        return out


class GLUBlock(nn.Module):
    """Original GLU block with shared layers support."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_glu: int = 2,
        first: bool = False,
        shared_layers=None,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
    ):
        super().__init__()
        self.first = first
        self.shared_layers = shared_layers
        self.n_glu = n_glu
        self.glu_layers = nn.ModuleList()

        params = {"virtual_batch_size": virtual_batch_size, "momentum": momentum}

        fc = shared_layers[0] if shared_layers else None
        self.glu_layers.append(GLULayer(input_dim, output_dim, fc=fc, **params))
        for glu_id in range(1, self.n_glu):
            fc = shared_layers[glu_id] if shared_layers else None
            self.glu_layers.append(GLULayer(output_dim, output_dim, fc=fc, **params))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.sqrt(torch.FloatTensor([0.5]).to(x.device))
        if self.first:
            x = self.glu_layers[0](x)
            layers_left = range(1, self.n_glu)
        else:
            layers_left = range(self.n_glu)

        for glu_id in layers_left:
            x = torch.add(x, self.glu_layers[glu_id](x))
            x = x * scale
        return x


class FeatureTransformer(nn.Module):
    """Original Feature Transformer implementation."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        shared_layers,
        n_glu_independent: int,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
    ):
        super().__init__()
        
        params = {
            "n_glu": n_glu_independent,
            "virtual_batch_size": virtual_batch_size,
            "momentum": momentum,
        }

        if shared_layers is None:
            self.shared = nn.Identity()
            is_first = True
        else:
            self.shared = GLUBlock(
                input_dim,
                output_dim,
                first=True,
                shared_layers=shared_layers,
                n_glu=len(shared_layers),
                virtual_batch_size=virtual_batch_size,
                momentum=momentum,
            )
            is_first = False

        if n_glu_independent == 0:
            self.specifics = nn.Identity()
        else:
            spec_input_dim = input_dim if is_first else output_dim
            self.specifics = GLUBlock(
                spec_input_dim, output_dim, first=is_first, **params
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shared(x)
        x = self.specifics(x)
        return x


class AttentiveTransformer(nn.Module):
    """Original Attentive Transformer with group matrix support."""
    
    def __init__(
        self,
        input_dim: int,
        group_dim: int,
        group_matrix,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
        mask_type: str = "sparsemax",
    ):
        super().__init__()
        self.fc = nn.Linear(input_dim, group_dim, bias=False)
        initialize_non_glu(self.fc, input_dim, group_dim)
        self.bn = GhostBatchNorm(group_dim, virtual_batch_size=virtual_batch_size, momentum=momentum)

        if mask_type == "sparsemax":
            self.selector = Sparsemax(dim=-1)
        elif mask_type == "entmax":
            self.selector = Entmax15(dim=-1)
        else:
            raise NotImplementedError("Please choose either sparsemax or entmax as masktype")

    def forward(self, priors: torch.Tensor, processed_feat: torch.Tensor) -> torch.Tensor:
        x = self.fc(processed_feat)
        x = self.bn(x)
        x = torch.mul(x, priors)
        x = self.selector(x)
        return x


class EmbeddingGenerator(nn.Module):
    """Original Embedding Generator with group matrix support."""
    
    def __init__(self, input_dim: int, cat_dims: List[int], cat_idxs: List[int], cat_emb_dims: List[int], group_matrix):
        super().__init__()
        
        if cat_dims == [] and cat_idxs == []:
            self.skip_embedding = True
            self.post_embed_dim = input_dim
            self.register_buffer('embedding_group_matrix', group_matrix)
            return
        else:
            self.skip_embedding = False

        self.post_embed_dim = int(input_dim + np.sum(cat_emb_dims) - len(cat_emb_dims))

        self.embeddings = nn.ModuleList()
        for cat_dim, emb_dim in zip(cat_dims, cat_emb_dims):
            self.embeddings.append(nn.Embedding(cat_dim, emb_dim))

        # record continuous indices
        continuous_idx = torch.ones(input_dim, dtype=torch.bool)
        continuous_idx[cat_idxs] = 0
        self.register_buffer('continuous_idx', continuous_idx)

        # update group matrix
        n_groups = group_matrix.shape[0]
        embedding_group_matrix = torch.empty((n_groups, self.post_embed_dim), device=group_matrix.device)
        for group_idx in range(n_groups):
            post_emb_idx = 0
            cat_feat_counter = 0
            for init_feat_idx in range(input_dim):
                if self.continuous_idx[init_feat_idx] == 1:
                    # this means that no embedding is applied to this column
                    embedding_group_matrix[group_idx, post_emb_idx] = group_matrix[group_idx, init_feat_idx]
                    post_emb_idx += 1
                else:
                    # this is a categorical feature which creates multiple embeddings
                    n_embeddings = cat_emb_dims[cat_feat_counter]
                    embedding_group_matrix[group_idx, post_emb_idx:post_emb_idx+n_embeddings] = group_matrix[group_idx, init_feat_idx] / n_embeddings
                    post_emb_idx += n_embeddings
                    cat_feat_counter += 1
        
        # Register as buffer so it moves with the model to different devices
        self.register_buffer('embedding_group_matrix', embedding_group_matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.skip_embedding:
            return x

        cols = []
        cat_feat_counter = 0
        for feat_init_idx, is_continuous in enumerate(self.continuous_idx):
            if is_continuous:
                cols.append(x[:, feat_init_idx].float().view(-1, 1))
            else:
                cols.append(self.embeddings[cat_feat_counter](x[:, feat_init_idx].long()))
                cat_feat_counter += 1
        
        return torch.cat(cols, dim=1)


class TabNetEncoder(nn.Module):
    """Original TabNet Encoder implementation."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_d: int = 8,
        n_a: int = 8,
        n_steps: int = 3,
        gamma: float = 1.3,
        n_independent: int = 2,
        n_shared: int = 2,
        epsilon: float = 1e-15,        # ✓ NOW PROPERLY USED
        virtual_batch_size: int = 128,
        momentum: float = 0.02,        # ✓ NOW PROPERLY USED  
        mask_type: str = "sparsemax",  # ✓ NOW PROPERLY USED
        group_attention_matrix=None,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_multi_task = isinstance(output_dim, list)
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon  # ✓ STORE IT
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum  # ✓ STORE IT
        self.mask_type = mask_type  # ✓ STORE IT
        
        # ✓ USE momentum in initial BatchNorm
        self.initial_bn = nn.BatchNorm1d(self.input_dim, momentum=momentum)
        self.group_attention_matrix = group_attention_matrix

        if self.group_attention_matrix is None:
            self.group_attention_matrix = torch.eye(self.input_dim)
            self.attention_dim = self.input_dim
        else:
            self.attention_dim = self.group_attention_matrix.shape[0]
        
        self.register_buffer('group_attention_matrix_buffer', self.group_attention_matrix)

        if self.n_shared > 0:
            shared_feat_transform = nn.ModuleList()
            for i in range(self.n_shared):
                if i == 0:
                    shared_feat_transform.append(nn.Linear(self.input_dim, 2 * (n_d + n_a), bias=False))
                else:
                    shared_feat_transform.append(nn.Linear(n_d + n_a, 2 * (n_d + n_a), bias=False))
        else:
            shared_feat_transform = None

        # ✓ PASS momentum to FeatureTransformer
        self.initial_splitter = FeatureTransformer(
            self.input_dim,
            n_d + n_a,
            shared_feat_transform,
            n_glu_independent=self.n_independent,
            virtual_batch_size=self.virtual_batch_size,
            momentum=momentum,  # ✓ USE IT
        )

        self.feat_transformers = nn.ModuleList()
        self.att_transformers = nn.ModuleList()

        for step in range(n_steps):
            # ✓ PASS momentum to both transformers
            transformer = FeatureTransformer(
                self.input_dim,
                n_d + n_a,
                shared_feat_transform,
                n_glu_independent=self.n_independent,
                virtual_batch_size=self.virtual_batch_size,
                momentum=momentum,  # ✓ USE IT
            )
            attention = AttentiveTransformer(
                n_a,
                self.attention_dim,
                group_matrix=group_attention_matrix,
                virtual_batch_size=self.virtual_batch_size,
                momentum=momentum,  # ✓ USE IT
                mask_type=self.mask_type,  # ✓ USE IT
            )
            self.feat_transformers.append(transformer)
            self.att_transformers.append(attention)

    def forward(self, x: torch.Tensor, prior=None) -> Tuple[List[torch.Tensor], torch.Tensor]:
        x = self.initial_bn(x)
        bs = x.shape[0]
        
        if prior is None:
            prior = torch.ones((bs, self.attention_dim)).to(x.device)

        M_loss = 0
        att = self.initial_splitter(x)[:, self.n_d :]
        steps_output = []
        
        for step in range(self.n_steps):
            M = self.att_transformers[step](prior, att)
            # ✓ USE self.epsilon properly
            M_loss += torch.mean(torch.sum(torch.mul(M, torch.log(M + self.epsilon)), dim=1))
            
            prior = torch.mul(self.gamma - M, prior)
            
            M_feature_level = torch.matmul(M, self.group_attention_matrix_buffer)
            masked_x = torch.mul(M_feature_level, x)
            out = self.feat_transformers[step](masked_x)
            d = F.relu(out[:, : self.n_d])
            steps_output.append(d)
            
            att = out[:, self.n_d :]

        M_loss /= self.n_steps
        return steps_output, M_loss

    def forward_masks(self, x: torch.Tensor):
        x = self.initial_bn(x)
        bs = x.shape[0]
        prior = torch.ones((bs, self.attention_dim)).to(x.device)
        M_explain = torch.zeros(x.shape).to(x.device)
        att = self.initial_splitter(x)[:, self.n_d :]
        masks = {}

        for step in range(self.n_steps):
            M = self.att_transformers[step](prior, att)
            M_feature_level = torch.matmul(M, self.group_attention_matrix_buffer)
            masks[step] = M_feature_level
            
            prior = torch.mul(self.gamma - M, prior)
            
            masked_x = torch.mul(M_feature_level, x)
            out = self.feat_transformers[step](masked_x)
            d = F.relu(out[:, : self.n_d])
            
            step_importance = torch.sum(d, dim=1)
            M_explain += torch.mul(M_feature_level, step_importance.unsqueeze(dim=1))
            
            att = out[:, self.n_d :]

        return M_explain, masks


class TabNet(nn.Module):
    """Original TabNet implementation with restored functionality."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_d: int = 8,
        n_a: int = 8,
        n_steps: int = 3,
        gamma: float = 1.3,
        cat_idxs: List[int] = None,
        cat_dims: List[int] = None,
        cat_emb_dim: Union[int, List[int]] = 1,
        n_independent: int = 2,
        n_shared: int = 2,
        epsilon: float = 1e-15,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
        mask_type: str = "sparsemax",
        grouped_features: List[List[int]] = None,
    ):
        super().__init__()
        
        self.cat_idxs = cat_idxs or []
        self.cat_dims = cat_dims or []
        self.cat_emb_dim = cat_emb_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.mask_type = mask_type
        self.grouped_features = grouped_features or []

        if self.n_steps <= 0:
            raise ValueError("n_steps should be a positive integer.")
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independent can't be both zero.")

        self.virtual_batch_size = virtual_batch_size
        
        # Handle embedding dimensions
        if isinstance(cat_emb_dim, int):
            cat_emb_dims = [cat_emb_dim] * len(self.cat_idxs)
        else:
            cat_emb_dims = cat_emb_dim

        # Create group matrix
        self.group_matrix = self._create_group_matrix(self.grouped_features, input_dim)
        
        # Embedding generator
        self.embedder = EmbeddingGenerator(
            input_dim, self.cat_dims, self.cat_idxs, cat_emb_dims, self.group_matrix
        )
        self.post_embed_dim = self.embedder.post_embed_dim

        # Encoder
        self.encoder = TabNetEncoder(
            input_dim=self.post_embed_dim,
            output_dim=output_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=epsilon,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
            mask_type=mask_type,
            group_attention_matrix=self.embedder.embedding_group_matrix,
        )

        # Output layer
        if isinstance(output_dim, list):
            # Multi-task
            self.multi_task_mappings = nn.ModuleList()
            for task_dim in output_dim:
                task_mapping = nn.Linear(n_d, task_dim, bias=False)
                initialize_non_glu(task_mapping, n_d, task_dim)
                self.multi_task_mappings.append(task_mapping)
            self.is_multi_task = True
        else:
            self.final_mapping = nn.Linear(n_d, output_dim, bias=False)
            initialize_non_glu(self.final_mapping, n_d, output_dim)
            self.is_multi_task = False

        # Create reducing matrix for explainability
        self.reducing_matrix = self._create_explain_matrix(
            self.post_embed_dim,
            cat_emb_dims,
            self.cat_idxs,
            self.post_embed_dim,
        )

    def _create_group_matrix(self, list_groups: List[List[int]], input_dim: int) -> torch.Tensor:
        """Create group matrix from grouped features."""
        if len(list_groups) == 0:
            return torch.eye(input_dim)
        
        # Validate groups
        self._check_list_groups(list_groups, input_dim)
        
        n_groups = input_dim - int(np.sum([len(gp) - 1 for gp in list_groups]))
        group_matrix = torch.zeros((n_groups, input_dim))

        remaining_features = list(range(input_dim))
        current_group_idx = 0
        
        for group in list_groups:
            group_size = len(group)
            for elem_idx in group:
                group_matrix[current_group_idx, elem_idx] = 1 / group_size
                remaining_features.remove(elem_idx)
            current_group_idx += 1
        
        # Features not in groups get their own group
        for remaining_feat_idx in remaining_features:
            group_matrix[current_group_idx, remaining_feat_idx] = 1
            current_group_idx += 1
            
        return group_matrix

    def _check_list_groups(self, list_groups: List[List[int]], input_dim: int):
        """Validate grouped features."""
        assert isinstance(list_groups, list), "list_groups must be a list of list."
        
        if len(list_groups) == 0:
            return
            
        for group_pos, group in enumerate(list_groups):
            msg = f"Groups must be given as a list of list, but found {group} in position {group_pos}."
            assert isinstance(group, list), msg
            assert len(group) > 0, "Empty groups are forbidden"

        n_elements_in_groups = np.sum([len(group) for group in list_groups])
        flat_list = []
        for group in list_groups:
            flat_list.extend(group)
        unique_elements = np.unique(flat_list)
        n_unique_elements_in_groups = len(unique_elements)
        
        assert n_unique_elements_in_groups == n_elements_in_groups, \
            "One feature can only appear in one group"
        
        highest_feat = np.max(unique_elements)
        assert highest_feat < input_dim, \
            f"Number of features is {input_dim} but one group contains {highest_feat}."

    def _create_explain_matrix(self, input_dim: int, cat_emb_dims: List[int], cat_idxs: List[int], post_embed_dim: int):
        """Create matrix for reducing post-embedding explanations back to original features."""
        if not cat_emb_dims:
            return scipy.sparse.eye(input_dim)
        
        if isinstance(cat_emb_dims, int):
            all_emb_impact = [cat_emb_dims - 1] * len(cat_idxs)
        else:
            all_emb_impact = [emb_dim - 1 for emb_dim in cat_emb_dims]

        acc_emb = 0
        nb_emb = 0
        indices_trick = []
        
        for i in range(input_dim):
            if i not in cat_idxs:
                indices_trick.append([i + acc_emb])
            else:
                indices_trick.append(range(i + acc_emb, i + acc_emb + all_emb_impact[nb_emb] + 1))
                acc_emb += all_emb_impact[nb_emb]
                nb_emb += 1

        reducing_matrix = np.zeros((post_embed_dim, input_dim))
        for i, cols in enumerate(indices_trick):
            reducing_matrix[cols, i] = 1

        return csc_matrix(reducing_matrix)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded_x = self.embedder(x)
        steps_output, M_loss = self.encoder(embedded_x)
        
        # Sum all steps
        res = sum(steps_output)
        
        if self.is_multi_task:
            out = []
            for task_mapping in self.multi_task_mappings:
                out.append(task_mapping(res))
        else:
            out = self.final_mapping(res)
            
        return out, M_loss

    def forward_masks(self, x: torch.Tensor):
        embedded_x = self.embedder(x)
        return self.encoder.forward_masks(embedded_x)




class History:
    """Training history tracker."""
    
    def __init__(self):
        self.history = {}
    
    def add(self, key: str, value: float):
        if key not in self.history:
            self.history[key] = []
        self.history[key].append(value)
    
    def get_last(self, key: str) -> Optional[float]:
        return self.history.get(key, [None])[-1]


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(self, monitor: str, patience: int = 10, min_delta: float = 0.0, maximize: bool = True):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.maximize = maximize
        
        self.best_value = float('-inf') if maximize else float('inf')
        self.wait = 0
        self.best_weights = None
        self.should_stop = False
    
    def __call__(self, current_value: float, model: nn.Module) -> bool:
        if self.maximize:
            improved = current_value > self.best_value + self.min_delta
        else:
            improved = current_value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = current_value
            self.wait = 0
            self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.wait += 1
            
        self.should_stop = self.wait >= self.patience
        return self.should_stop
    
    def restore_best_weights(self, model: nn.Module):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


# ============================================================================
# Main TabNet Classes with Streaming Support and Weighted Loss
# ============================================================================

@dataclass
class TabNetConfig:
    """Configuration for TabNet models - faithful to original paper."""
    
    # === Core Architecture (all with defaults) ===
    n_d: int = 8
    n_a: int = 8
    n_steps: int = 3
    gamma: float = 1.3
    lambda_sparse: float = 1e-3
    n_independent: int = 2
    n_shared: int = 2
    epsilon: float = 1e-15
    
    # === Ghost BatchNorm ===
    virtual_batch_size: int = 128
    momentum: float = 0.02
    
    # === Categorical Features ===
    cat_dims: List[int] = field(default_factory=list)
    cat_idxs: List[int] = field(default_factory=list)
    cat_emb_dim: Union[int, List[int]] = 1
    
    # === Attention ===
    mask_type: str = "sparsemax"
    grouped_features: List[List[int]] = field(default_factory=list)
    
    # === Training Parameters ===
    learning_rate: float = 2e-2
    decay_rate: float = 0.9
    decay_steps: int = 8000
    max_iterations: int = 50000  # Reduced for your 700k dataset
    weight_decay: float = 0.0
    
    # === System ===
    device: str = "auto"
    seed: int = 0
    use_class_weights: bool = False
    
    # === Weights & Biases Configuration ===
    use_wandb: bool = False
    wandb_project: str = "tabnet-experiments"
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_notes: Optional[str] = None
    wandb_config: Dict[str, Any] = field(default_factory=dict)
    
    # === Evaluation Strategy ===
    eval_strategy: str = "epoch"  # "epoch" for per-epoch eval, "steps" for step-based eval
    eval_steps: Optional[int] = None  # Required when eval_strategy="steps"
    eval_patience_steps: Optional[int] = None  # Early stopping based on steps instead of epochs
    log_feature_importance_every: int = 5  # Log feature importance every N epochs
    save_model_every: int = 10  # Save model every N epochs

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    Hugging Face linear schedule with warmup
    - Warmup: LR goes from 0 to max_lr over warmup steps
    - Decay: LR goes from max_lr to 0 over remaining steps
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Warmup phase: linear increase
            return float(current_step) / float(max(1, num_warmup_steps))
        # Decay phase: linear decrease to 0
        return max(
            0.0, 
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class BaseTabNet(BaseEstimator, ABC):
    """Base class for TabNet models with streaming support."""
    
    def __init__(self, config: TabNetConfig = None, **kwargs):
        self.config = config or TabNetConfig()
        
        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Set device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        # Set seed
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
        
        self.network = None
        self.history = History()
        self.feature_importances_ = None
        
        # Wandb initialization
        self.wandb_run = None
        self._step_counter = 0
        self._last_eval_step = 0
        
        # Validate evaluation strategy
        self._validate_eval_strategy()
    
    def _validate_eval_strategy(self):
        """Validate evaluation strategy configuration."""
        if self.config.eval_strategy not in ["epoch", "steps"]:
            raise ValueError(f"eval_strategy must be 'epoch' or 'steps', got '{self.config.eval_strategy}'")
        
        if self.config.eval_strategy == "steps" and self.config.eval_steps is None:
            raise ValueError("eval_steps must be specified when eval_strategy='steps'")
        
        if self.config.eval_strategy == "steps" and self.config.eval_steps <= 0:
            raise ValueError("eval_steps must be positive when eval_strategy='steps'")
    
    def _build_network(self, input_dim: int, output_dim: int):
        """Build the TabNet network."""
        self.network = TabNet(
            input_dim=input_dim,
            output_dim=output_dim,
            n_d=self.config.n_d,
            n_a=self.config.n_a,
            n_steps=self.config.n_steps,
            gamma=self.config.gamma,
            cat_dims=self.config.cat_dims,
            cat_idxs=self.config.cat_idxs,
            cat_emb_dim=self.config.cat_emb_dim,
            n_independent=self.config.n_independent,
            n_shared=self.config.n_shared,
            virtual_batch_size=self.config.virtual_batch_size,
            momentum=self.config.momentum,
            epsilon=self.config.epsilon,
            mask_type=self.config.mask_type,
            grouped_features=self.config.grouped_features
        ).to(self.device)
    
    def _prepare_class_weights(self, train_loader: DataLoader, verbose: bool):
        """Hook for subclasses to prepare class weights. Default implementation does nothing."""
        pass
    
    def _init_wandb(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Initialize Weights & Biases tracking."""
        if not self.config.use_wandb or not WANDB_AVAILABLE:
            return
        
        # Prepare wandb config
        wandb_config = {
            # TabNet architecture
            'n_d': self.config.n_d,
            'n_a': self.config.n_a,
            'n_steps': self.config.n_steps,
            'gamma': self.config.gamma,
            'lambda_sparse': self.config.lambda_sparse,
            'n_independent': self.config.n_independent,
            'n_shared': self.config.n_shared,
            'epsilon': self.config.epsilon,
            
            # Training parameters
            'learning_rate': self.config.learning_rate,
            'decay_rate': self.config.decay_rate,
            'decay_steps': self.config.decay_steps,
            'max_iterations': self.config.max_iterations,
            'weight_decay': self.config.weight_decay,
            
            # BatchNorm parameters
            'virtual_batch_size': self.config.virtual_batch_size,
            'momentum': self.config.momentum,
            
            # Attention
            'mask_type': self.config.mask_type,
            
            # System
            'device': str(self.device),
            'seed': self.config.seed,
            'use_class_weights': self.config.use_class_weights,
            
            # Dataset info
            'has_validation': val_loader is not None,
        }
        
        # Add user-defined config
        wandb_config.update(self.config.wandb_config)
        
        # Get dataset info
        try:
            sample_batch = next(iter(train_loader))
            if isinstance(sample_batch, (list, tuple)) and len(sample_batch) >= 2:
                X_sample, y_sample = sample_batch[0], sample_batch[1]
            else:
                X_sample = sample_batch
                y_sample = None
            
            wandb_config['input_dim'] = X_sample.shape[1]
            wandb_config['batch_size'] = X_sample.shape[0]
            
            if y_sample is not None:
                if hasattr(self, 'n_classes_') and self.n_classes_:
                    wandb_config['n_classes'] = self.n_classes_
                wandb_config['target_shape'] = list(y_sample.shape)
        except Exception:
            pass
        
        # Initialize wandb
        self.wandb_run = wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=self.config.wandb_name,
            tags=self.config.wandb_tags,
            notes=self.config.wandb_notes,
            config=wandb_config,
            reinit=True
        )
        
        if self.wandb_run:
            # Watch the model for gradients and weights
            wandb.watch(self.network, log="all", log_freq=100)
            
            # Log evaluation strategy configuration
            wandb.log({
                "eval_strategy": self.config.eval_strategy,
                "eval_steps": self.config.eval_steps if self.config.eval_strategy == "steps" else None,
                "step_based_evaluation": self.config.eval_strategy == "steps"
            }, step=0)
    
    def _log_wandb(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = ""):
        """Log metrics to wandb."""
        if not self.wandb_run:
            return
        
        # Add prefix to metric names if provided
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # Add step counter if not provided
        if step is None:
            step = self._step_counter
        
        wandb.log(metrics, step=step)
    
    def _log_feature_importance_wandb(self, epoch: int):
        """Log feature importance to wandb."""
        if not self.wandb_run or self.feature_importances_ is None:
            return
        
        # Create feature importance table
        importance_table = wandb.Table(
            columns=["Feature Index", "Importance", "Rank"],
            data=[
                [i, importance, rank + 1] 
                for rank, (i, importance) in enumerate(
                    sorted(enumerate(self.feature_importances_), 
                           key=lambda x: x[1], reverse=True)
                )
            ]
        )
        
        # Log table
        wandb.log({
            "feature_importance_table": importance_table,
            "feature_importance_epoch": epoch
        }, step=self._step_counter)
        
        # Log histogram of importances
        wandb.log({
            "feature_importance_histogram": wandb.Histogram(self.feature_importances_)
        }, step=self._step_counter)
    
    def _save_model_wandb(self, epoch: int, metrics: Dict[str, float]):
        """Save model as wandb artifact."""
        if not self.wandb_run:
            return
        
        # Create temporary file for model
        temp_path = f"temp_model_epoch_{epoch}"
        saved_path = self.save_model(temp_path)
        
        # Create artifact
        artifact = wandb.Artifact(
            name=f"tabnet_model_epoch_{epoch}",
            type="model",
            metadata={
                "epoch": epoch,
                "metrics": metrics,
                "config": self.config.__dict__
            }
        )
        
        # Add model file
        artifact.add_file(saved_path)
        
        # Log artifact
        wandb.log_artifact(artifact)
        
        # Clean up
        Path(saved_path).unlink(missing_ok=True)
    
    def _should_evaluate_at_step(self, step: int) -> bool:
        """Check if we should evaluate at this step."""
        # Only do step-based evaluation if strategy is "steps"
        if self.config.eval_strategy != "steps" or self.config.eval_steps is None:
            return False
        
        # Always evaluate on the first step if eval_steps is set
        if self._last_eval_step == 0 and step > 0:
            return True
        
        return (step - self._last_eval_step) >= self.config.eval_steps
    
    def _close_wandb(self):
        """Close wandb run."""
        if self.wandb_run:
            wandb.finish()
            self.wandb_run = None
    
    def _print_evaluation_metrics(self, evaluation_type: str, step_or_epoch: int, 
                                metrics: Dict[str, float], lr: float, 
                                train_loss: Optional[float] = None, 
                                total_iterations: Optional[int] = None,
                                max_iterations: Optional[int] = None) -> None:
        """Unified function to print evaluation metrics for both epoch and step-based evaluation."""
        if not metrics:
            return
            
        # Format metrics consistently
        acc = metrics.get("accuracy", float('nan'))
        auc = metrics.get("auc", float('nan'))
        f1 = metrics.get("binary_f1", float('nan'))
        
        # Create base metric string
        metric_str = f"acc={acc:.4f}, auc={auc:.4f}, f1={f1:.4f}"
        
        if evaluation_type == "epoch":
            if total_iterations and max_iterations:
                iter_str = f"iter={total_iterations}/{max_iterations}, "
            else:
                iter_str = ""
            
            if train_loss is not None:
                print(f"\nEpoch {step_or_epoch}: {iter_str}lr={lr:.2e}, "
                      f"train_loss={train_loss:.4f}, val_{metric_str}")
            else:
                print(f"Epoch {step_or_epoch}: {iter_str}lr={lr:.2e}, val_{metric_str}")
                
        elif evaluation_type == "step":
            print(f"\nStep {step_or_epoch}: lr={lr:.2e}, val_{metric_str}")
        
        elif evaluation_type == "epoch_step_mode":
            # For epochs when using step-based evaluation
            if total_iterations and max_iterations:
                iter_str = f"iter={total_iterations}/{max_iterations}, "
            else:
                iter_str = ""
            print(f"Epoch {step_or_epoch}: {iter_str}lr={lr:.2e}, "
                  f"train_loss={train_loss:.4f} (using step-based eval)")
    
    def _get_steps_per_epoch(self, data_loader: DataLoader, steps_per_epoch: Optional[int], verbose: bool) -> Optional[int]:
        """Determine number of steps per epoch for potentially streaming datasets."""
        
        # First try: user-specified steps_per_epoch
        if steps_per_epoch is not None:
            return steps_per_epoch
        
        # Second try: get length from DataLoader
        try:
            return len(data_loader)
        except (TypeError, NotImplementedError):
            pass
        
        # Third try: get length from dataset
        try:
            dataset_len = len(data_loader.dataset)
            batch_size = data_loader.batch_size or 1
            drop_last = getattr(data_loader, 'drop_last', False)
            
            if drop_last:
                return dataset_len // batch_size
            else:
                return (dataset_len + batch_size - 1) // batch_size
        except (TypeError, NotImplementedError, AttributeError):
            pass
        
        # Fourth try: dry run to count batches (expensive for streaming data)
        if verbose:
            print("Dataset length unknown, performing dry run to count batches...")
        
        try:
            count = 0
            # Use a separate iterator to avoid interfering with training
            temp_loader = DataLoader(
                data_loader.dataset,
                batch_size=data_loader.batch_size,
                shuffle=False,  # Don't shuffle for counting
                num_workers=0,  # Single threaded for counting
                drop_last=data_loader.drop_last
            )
            
            for _ in temp_loader:
                count += 1
                # Limit dry run to reasonable number for very large datasets
                if count > 10000:
                    if verbose:
                        print(f"Stopped dry run at {count} batches (very large dataset)")
                    break
            
            if verbose:
                print(f"Found {count} batches per epoch")
            return count
            
        except Exception as e:
            if verbose:
                print(f"Could not determine dataset length: {e}")
                print("Using indeterminate progress bars")
            return None
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        max_epochs: int = 100,
        patience: int = 10,
        eval_metric: Optional[str] = None,
        verbose: bool = True,
        steps_per_epoch: Optional[int] = None
    ):
        """
        Fit using exact TabNet paper methodology with config parameters.
        
        Uses self.config for all training parameters.
        """
        
        # Initialize network if not already done
        if self.network is None:
            sample_batch = next(iter(train_loader))
            if isinstance(sample_batch, (list, tuple)) and len(sample_batch) >= 2:
                X_sample, y_sample = sample_batch[0], sample_batch[1]
            else:
                X_sample = sample_batch
                y_sample = None
            
            input_dim = X_sample.shape[1]
            output_dim = self._get_output_dim(y_sample)
            self._build_network(input_dim, output_dim)
        
        # Prepare class weights if needed
        self._prepare_class_weights(train_loader, verbose)
        
        # Initialize Weights & Biases
        self._init_wandb(train_loader, val_loader)
        
        # Log evaluation strategy
        if verbose:
            if self.config.eval_strategy == "steps":
                print(f"Using step-based evaluation: evaluating every {self.config.eval_steps} steps")
            else:
                print("Using per-epoch evaluation: evaluating at the end of each epoch")
        
        # Determine steps per epoch
        train_steps_per_epoch = self._get_steps_per_epoch(train_loader, steps_per_epoch, verbose)
        val_steps_per_epoch = None
        if val_loader is not None:
            val_steps_per_epoch = self._get_steps_per_epoch(val_loader, None, verbose)
        
        # ✓ USE config parameters for optimizer
        optimizer = torch.optim.Adam(
            self.network.parameters(), 
            lr=self.config.learning_rate,    # ✓ USE IT
            weight_decay=self.config.weight_decay  # ✓ USE IT
        )
        
        # ✓ USE config parameters for scheduler  
        scheduler = get_exponential_decay_scheduler(
            optimizer, 
            self.config.decay_rate,    # ✓ USE IT
            self.config.decay_steps    # ✓ USE IT
        )
            
        # Setup metrics and early stopping
        metrics = self._get_metrics(eval_metric)
        early_stopping = None
        if patience > 0 and val_loader is not None:
            monitor_metric = eval_metric or self._get_default_metric()
            maximize = metrics[monitor_metric].maximize
            early_stopping = EarlyStopping(monitor_metric, patience, maximize=maximize)
        
        # ✓ Training loop with iteration counting using config.max_iterations
        epoch_pbar = tqdm(range(max_epochs), desc="Epochs", disable=not verbose, position=0)
        total_iterations = 0
        
        for epoch in epoch_pbar:
            # Training phase with iteration counting and step-based evaluation
            train_loss, iterations_this_epoch, step_eval_metrics = self._train_epoch(
                train_loader, optimizer, scheduler, train_steps_per_epoch, 
                self.config.max_iterations, total_iterations, verbose, val_loader, metrics  # ✓ USE IT
            )
            total_iterations += iterations_this_epoch
            self._step_counter = total_iterations
            
            self.history.add("train_loss", train_loss)
            self.history.add("total_iterations", total_iterations)
            
            # Log training metrics to wandb
            train_metrics = {
                "train_loss": train_loss,
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "total_iterations": total_iterations
            }
            self._log_wandb(train_metrics)
            
            # Add step-based evaluation metrics if any
            if step_eval_metrics:
                self._log_wandb(step_eval_metrics, prefix="step_eval")
            
            # ✓ Check iteration limit using config.max_iterations
            if total_iterations >= self.config.max_iterations:
                if verbose:
                    print(f"\nReached maximum iterations ({self.config.max_iterations}) at epoch {epoch + 1}")
                break
            
            # Validation phase - only run if eval_strategy is "epoch"
            if val_loader is not None and self.config.eval_strategy == "epoch":
                val_metrics = self._validate_epoch(val_loader, metrics, val_steps_per_epoch, verbose)
                for name, value in val_metrics.items():
                    self.history.add(f"val_{name}", value)
                
                # Log validation metrics to wandb
                val_wandb_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
                self._log_wandb(val_wandb_metrics)
                
                # Log progress
                current_lr = optimizer.param_groups[0]['lr']
                
                if verbose:
                    self._print_evaluation_metrics("epoch", epoch+1, val_metrics, current_lr, 
                                                 train_loss, total_iterations, self.config.max_iterations)
                
                # Update progress bar
                epoch_pbar.set_postfix({
                    "iter": f"{total_iterations}/{self.config.max_iterations}",
                    "lr": f"{current_lr:.2e}",
                    "train_loss": f"{train_loss:.4f}",
                    **{f"val_{k}": f"{v:.4f}" for k, v in val_metrics.items()}
                })
                
                # Early stopping
                if early_stopping is not None:
                    monitor_value = val_metrics.get(early_stopping.monitor.replace("val_", ""))
                    if monitor_value is not None:
                        if early_stopping(monitor_value, self.network):
                            if verbose:
                                print(f"\nEarly stopping at epoch {epoch + 1}, iteration {total_iterations}")
                            break
            elif val_loader is not None and self.config.eval_strategy == "steps":
                # For step-based evaluation, only show training progress
                current_lr = optimizer.param_groups[0]['lr']
                if verbose:
                    self._print_evaluation_metrics("epoch_step_mode", epoch+1, {}, current_lr, 
                                                 train_loss, total_iterations, self.config.max_iterations)
                
                # Update progress bar (no validation metrics)
                epoch_pbar.set_postfix({
                    "iter": f"{total_iterations}/{self.config.max_iterations}",
                    "lr": f"{current_lr:.2e}",
                    "train_loss": f"{train_loss:.4f}",
                    "eval_strategy": "steps"
                })
            
            # Log feature importance periodically
            if (epoch + 1) % self.config.log_feature_importance_every == 0:
                # Compute current feature importances
                self._compute_feature_importances(train_loader)
                self._log_feature_importance_wandb(epoch + 1)
            
            # Save model periodically
            if (epoch + 1) % self.config.save_model_every == 0:
                combined_metrics = {"train_loss": train_loss}
                if val_loader is not None and self.config.eval_strategy == "epoch":
                    combined_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                self._save_model_wandb(epoch + 1, combined_metrics)
            
            # Handle case where no validation is shown (for step-based strategy without validation at epoch level)
            if val_loader is None:
                current_lr = optimizer.param_groups[0]['lr']
                epoch_pbar.set_postfix({
                    "iter": f"{total_iterations}/{self.config.max_iterations}",
                    "lr": f"{current_lr:.2e}", 
                    "train_loss": f"{train_loss:.4f}"
                })
        
        # Restore best weights if early stopping was used
        if early_stopping is not None:
            early_stopping.restore_best_weights(self.network)
        
        # Compute final feature importances
        self._compute_feature_importances(train_loader)
        
        # Log final feature importance to wandb
        if self.wandb_run:
            self._log_feature_importance_wandb(epoch + 1)
            
            # Log final model metrics
            final_metrics = {"final_train_loss": train_loss, "final_iterations": total_iterations}
            if val_loader is not None:
                final_metrics.update({f"final_val_{k}": v for k, v in val_metrics.items()})
            self._log_wandb(final_metrics)
        
        if verbose:
            print(f"\nTraining completed: {total_iterations} total iterations")
        
        # Clean up wandb at the end
        try:
            self._close_wandb()
        except Exception as e:
            if verbose:
                print(f"Warning: Error closing wandb: {e}")
        
        return self


    def _train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer, 
                                scheduler, steps_per_epoch: Optional[int], max_iterations: int,
                                current_total_iterations: int, verbose: bool, 
                                val_loader: Optional[DataLoader] = None, 
                                metrics: Optional[Dict[str, Any]] = None) -> Tuple[float, int, Dict[str, float]]:
        """Train for one epoch with paper-faithful iteration counting."""
        self.network.train()
        total_loss = 0
        num_batches = 0
        iterations_this_epoch = 0
        step_eval_metrics = {}
        
        if verbose:
            if steps_per_epoch is not None:
                step_pbar = tqdm(total=steps_per_epoch, desc="Training", leave=False, position=1)
            else:
                step_pbar = tqdm(desc="Training (unknown length)", leave=False, position=1)
        
        train_iter = iter(train_loader)
        
        try:
            while True:
                # Check iteration limit first
                if current_total_iterations + iterations_this_epoch >= max_iterations:
                    if verbose:
                        step_pbar.set_description("Reached max iterations")
                    break
                    
                # Check epoch limit  
                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break
                
                try:
                    batch = next(train_iter)
                except StopIteration:
                    break
                
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    X, y = batch[0], batch[1]
                else:
                    X, y = batch, None
                
                X = X.to(self.device).float()
                if y is not None:
                    y = y.to(self.device)
                    y = self._prepare_target(y)
                
                optimizer.zero_grad()
                
                output, M_loss = self.network(X)
                loss = self._compute_loss(output, y)
                
                # ✓ USE config.lambda_sparse for sparsity regularization
                total_loss_with_reg = loss - self.config.lambda_sparse * M_loss
                
                total_loss_with_reg.backward()
                optimizer.step()
                
                # Step scheduler every iteration (paper methodology)
                scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                iterations_this_epoch += 1
                
                # Step-based evaluation
                current_step = current_total_iterations + iterations_this_epoch
                if (val_loader is not None and metrics is not None and 
                    self._should_evaluate_at_step(current_step)):
                    
                    # Perform validation at this step
                    if verbose:
                        step_pbar.set_description(f"Step Evaluation at step {current_step}")
                    
                    try:
                        val_steps_per_epoch = min(100, len(val_loader))
                    except (TypeError, NotImplementedError):
                        val_steps_per_epoch = 100  # Default for streaming datasets
                    
                    step_val_metrics = self._validate_epoch(val_loader, metrics, val_steps_per_epoch, verbose)
                    
                    # Print step evaluation results using unified function
                    if verbose and step_val_metrics:
                        current_lr = optimizer.param_groups[0]['lr']
                        self._print_evaluation_metrics("step", current_step, step_val_metrics, current_lr)
                    
                    # Log step evaluation metrics
                    step_eval_metrics.update({f"step_{current_step}_{k}": v for k, v in step_val_metrics.items()})
                    
                    # Log batch-level metrics
                    self._log_wandb({
                        "batch_loss": loss.item(),
                        "batch_M_loss": M_loss.item(),
                        "batch_total_loss": total_loss_with_reg.item(),
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        **{f"step_val_{k}": v for k, v in step_val_metrics.items()}
                    }, step=current_step)
                    
                    self._last_eval_step = current_step
                    self.network.train()  # Return to training mode
                    
                    # Reset progress bar description
                    if verbose:
                        step_pbar.set_description("Training")
                else:
                    # Log batch-level metrics without validation
                    self._log_wandb({
                        "batch_loss": loss.item(),
                        "batch_M_loss": M_loss.item(),
                        "batch_total_loss": total_loss_with_reg.item(),
                        "learning_rate": optimizer.param_groups[0]['lr']
                    }, step=current_step)
                
                if verbose:
                    current_lr = optimizer.param_groups[0]['lr']
                    step_pbar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{current_lr:.2e}",
                        "iter": f"{current_total_iterations + iterations_this_epoch}"
                    })
                    step_pbar.update(1)
        
        finally:
            if verbose:
                step_pbar.close()
        
        return total_loss / max(num_batches, 1), iterations_this_epoch, step_eval_metrics

    def _validate_epoch(self, val_loader: DataLoader, metrics: Dict[str, Metric], 
                       steps_per_epoch: Optional[int], verbose: bool) -> Dict[str, float]:
        """Validate for one epoch with proper progress tracking."""
        self.network.eval()
        all_outputs = []
        all_targets = []
        num_batches = 0
        
        # Setup progress bar for validation steps
        if verbose:
            if steps_per_epoch is not None:
                step_pbar = tqdm(total=steps_per_epoch, desc="Validation", leave=False, position=1)
            else:
                step_pbar = tqdm(desc="Validation (unknown length)", leave=False, position=1)
        
        val_iter = iter(val_loader)
        
        try:
            with torch.no_grad():
                while True:
                    # Check if we've completed the specified number of steps
                    if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                        break
                    
                    try:
                        batch = next(val_iter)
                    except StopIteration:
                        # Natural end of validation
                        break
                    
                    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        X, y = batch[0], batch[1]
                    else:
                        X, y = batch, None
                    
                    X = X.to(self.device).float()
                    if y is not None:
                        y = y.to(self.device)
                    
                    output, _ = self.network(X)
                    
                    all_outputs.append(self._postprocess_output(output).cpu().numpy())
                    if y is not None:
                        all_targets.append(y.cpu().numpy())
                    
                    num_batches += 1
                    
                    if verbose:
                        step_pbar.update(1)
        
        finally:
            if verbose:
                step_pbar.close()
        
        # Compute metrics
        if not all_outputs:
            return {}
        
        outputs = np.vstack(all_outputs)
        results = {}
        
        if all_targets:
            targets = np.concatenate(all_targets)
            for name, metric in metrics.items():
                try:
                    results[name] = metric(targets, outputs)
                except Exception:
                    # Skip metrics that can't be computed
                    pass
        
        return results
    
    def _compute_feature_importances(self, data_loader: DataLoader):
        """Compute feature importances - mathematically equivalent to original TabNet."""
        self.network.eval()
        
        importance_sum = None
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Computing feature importances", leave=False):
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    X = batch[0]
                else:
                    X = batch
                
                X = X.to(self.device).float()
                
                # Get explanations for this batch (same as original)
                M_explain, _ = self.network.forward_masks(X)
                
                # Reduce back to original feature space (same as original)
                original_feat_explain = csc_matrix.dot(
                    M_explain.cpu().detach().numpy(), 
                    self.network.reducing_matrix
                )
                
                # Safe optimization: accumulate sums instead of storing all data
                # This is mathematically equivalent to: np.vstack(all_batches).sum(axis=0)
                batch_sum = original_feat_explain.sum(axis=0)
                
                if importance_sum is None:
                    importance_sum = batch_sum
                else:
                    importance_sum += batch_sum
                
                total_samples += X.shape[0]
                
                # Memory cleanup (safe optimization)
                del M_explain, original_feat_explain, batch_sum
        
        if importance_sum is not None:
            # Normalize exactly as original: sum_explain / np.sum(sum_explain)
            total_sum = np.sum(importance_sum)
            if total_sum > 0:
                self.feature_importances_ = (importance_sum / total_sum).flatten()
            else:
                # Handle edge case of all-zero importance
                self.feature_importances_ = np.zeros(len(importance_sum)).flatten()
        else:
            # No data processed
            self.feature_importances_ = None
    
    def predict(self, data_loader: DataLoader, max_batches: Optional[int] = None, verbose: bool = True) -> np.ndarray:
        """Make predictions using a DataLoader."""
        self.network.eval()
        predictions = []
        
        # Determine number of batches for progress bar
        try:
            total_batches = len(data_loader)
            if max_batches is not None:
                total_batches = min(total_batches, max_batches)
        except (TypeError, NotImplementedError):
            total_batches = max_batches
        
        # Setup progress bar
        if verbose:
            if total_batches is not None:
                pbar = tqdm(total=total_batches, desc="Predicting")
            else:
                pbar = tqdm(desc="Predicting (unknown length)")
        
        num_batches = 0
        data_iter = iter(data_loader)
        
        try:
            with torch.no_grad():
                while True:
                    if max_batches is not None and num_batches >= max_batches:
                        break
                    
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        break
                    
                    if isinstance(batch, (list, tuple)):
                        X = batch[0]
                    else:
                        X = batch
                    
                    X = X.to(self.device).float()
                    output, _ = self.network(X)
                    pred = self._postprocess_output(output)
                    predictions.append(pred.cpu().numpy())
                    
                    num_batches += 1
                    if verbose:
                        pbar.update(1)
        
        finally:
            if verbose:
                pbar.close()
        
        if not predictions:
            return np.array([])
        
        return np.vstack(predictions)
    
    def explain(self, data_loader: DataLoader, max_batches: Optional[int] = None, verbose: bool = True) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """Get explanations for predictions."""
        self.network.eval()
        all_importance = []
        all_masks = {i: [] for i in range(self.config.n_steps)}
        
        # Determine number of batches for progress bar
        try:
            total_batches = len(data_loader)
            if max_batches is not None:
                total_batches = min(total_batches, max_batches)
        except (TypeError, NotImplementedError):
            total_batches = max_batches
        
        # Setup progress bar
        if verbose:
            if total_batches is not None:
                pbar = tqdm(total=total_batches, desc="Explaining")
            else:
                pbar = tqdm(desc="Explaining (unknown length)")
        
        num_batches = 0
        data_iter = iter(data_loader)
        
        try:
            with torch.no_grad():
                while True:
                    if max_batches is not None and num_batches >= max_batches:
                        break
                    
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        break
                    
                    if isinstance(batch, (list, tuple)):
                        X = batch[0]
                    else:
                        X = batch
                    
                    X = X.to(self.device).float()
                    importance, masks = self.network.forward_masks(X)
                    
                    # Reduce back to original feature space
                    original_feat_explain = csc_matrix.dot(importance.cpu().detach().numpy(), self.network.reducing_matrix)
                    all_importance.append(original_feat_explain)
                    
                    for step, mask in masks.items():
                        mask_reduced = csc_matrix.dot(mask.cpu().detach().numpy(), self.network.reducing_matrix)
                        all_masks[step].append(mask_reduced)
                    
                    num_batches += 1
                    if verbose:
                        pbar.update(1)
        
        finally:
            if verbose:
                pbar.close()
        
        if not all_importance:
            return np.array([]), {}
        
        importance_array = np.vstack(all_importance)
        mask_arrays = {step: np.vstack(masks) for step, masks in all_masks.items() if masks}
        
        return importance_array, mask_arrays
    
    def save_model(self, path: str):
        """Save model to disk."""
        save_dict = {
            'config': self.config.__dict__,
            'state_dict': self.network.state_dict() if self.network else None,
            'feature_importances_': self.feature_importances_,
            'history': self.history.history
        }
        
        # Add class weights for classifiers
        if hasattr(self, 'class_weights_'):
            save_dict['class_weights_'] = self.class_weights_.cpu() if self.class_weights_ is not None else None
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save as zip file
        temp_path = f"{path}_temp"
        Path(temp_path).mkdir(exist_ok=True)
        
        torch.save(save_dict, Path(temp_path) / "model.pt")
        
        with open(Path(temp_path) / "metadata.json", "w") as f:
            json.dump({"class_name": self.__class__.__name__}, f)
        
        shutil.make_archive(path, 'zip', temp_path)
        shutil.rmtree(temp_path)
        
        return f"{path}.zip"
    
    def load_model(self, path: str):
        """Load model from disk."""
        with zipfile.ZipFile(path, 'r') as zip_file:
            with zip_file.open("model.pt") as f:
                save_dict = torch.load(f, map_location=self.device, weights_only=False)
            
            # Restore config
            self.config = TabNetConfig(**save_dict['config'])
            
            # Restore network
            if save_dict['state_dict'] is not None:
                state_dict = save_dict['state_dict']
                
                # Get input dimension from embedder or initial layers
                if 'embedder.embedding_group_matrix' in state_dict:
                    sample_input_dim = state_dict['embedder.embedding_group_matrix'].shape[0]
                elif 'encoder.initial_bn.bias' in state_dict:
                    sample_input_dim = state_dict['encoder.initial_bn.bias'].shape[0]
                else:
                    # Fallback: look at any layer that processes the initial input
                    for key in state_dict.keys():
                        if 'shared.glu_layers.0.fc.weight' in key:
                            sample_input_dim = state_dict[key].shape[1]
                            break
                    else:
                        raise ValueError("Could not determine input dimension from state_dict")
                
                # Get output dimension from final mapping
                if 'final_mapping.weight' in state_dict:
                    output_dim = state_dict['final_mapping.weight'].shape[0]
                else:
                    # Multi-task case
                    output_dim = [v.shape[0] for k, v in state_dict.items() 
                                if 'multi_task_mappings' in k and 'weight' in k]
                    if not output_dim:
                        raise ValueError("Could not determine output dimension from state_dict")
                
                print(f"Detected input_dim: {sample_input_dim}, output_dim: {output_dim}")
                
                # Rebuild network with detected dimensions
                self._build_network(sample_input_dim, output_dim)
                self.network.load_state_dict(save_dict['state_dict'])
            
            # Restore other attributes
            self.feature_importances_ = save_dict.get('feature_importances_')
            self.history.history = save_dict.get('history', {})
            
            # Restore class weights for classifiers
            if hasattr(self, 'class_weights_') and 'class_weights_' in save_dict:
                self.class_weights_ = save_dict['class_weights_']
                if self.class_weights_ is not None:
                    self.class_weights_ = self.class_weights_.to(self.device)
    
    @abstractmethod
    def _get_output_dim(self, y_sample: Optional[torch.Tensor]) -> int:
        """Get output dimension from sample target."""
        pass
    
    @abstractmethod
    def _compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss."""
        pass
    
    @abstractmethod
    def _prepare_target(self, y: torch.Tensor) -> torch.Tensor:
        """Prepare target for training."""
        pass
    
    @abstractmethod
    def _postprocess_output(self, output: torch.Tensor) -> torch.Tensor:
        """Postprocess model output."""
        pass
    
    @abstractmethod
    def _get_metrics(self, eval_metric: Optional[str]) -> Dict[str, Metric]:
        """Get evaluation metrics."""
        pass
    
    @abstractmethod
    def _get_default_metric(self) -> str:
        """Get default metric name."""
        pass


class TabNetClassifier(BaseTabNet):
    """TabNet for classification tasks with optional weighted loss."""
    
    def __init__(self, config: TabNetConfig = None, **kwargs):
        super().__init__(config, **kwargs)
        self.classes_ = None
        self.n_classes_ = None
        self.class_weights_ = None  # NEW: Store class weights
    
    def _prepare_class_weights(self, train_loader: DataLoader, verbose: bool):
        """Calculate class weights from the training dataset if enabled."""
        if not self.config.use_class_weights:
            return
        
        if verbose:
            print("Calculating class weights from training data...")
        
        all_targets = []
        
        try:
            # Try to create a temporary loader to collect all targets
            temp_loader = DataLoader(
                train_loader.dataset,
                batch_size=train_loader.batch_size or 32,
                shuffle=False,  # Don't shuffle for weight calculation
                num_workers=0,   # Single threaded to avoid complications
                drop_last=False  # Don't drop any samples
            )
            
            # Collect targets with progress bar
            for batch in tqdm(temp_loader, desc="Collecting targets for class weights", disable=not verbose):
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    _, y = batch[0], batch[1]
                    y_np = y.numpy() if isinstance(y, torch.Tensor) else y
                    
                    # Handle different target shapes
                    if y_np.ndim > 1:
                        if y_np.shape[1] == 1:
                            y_np = y_np.flatten()
                        else:
                            # Multi-class one-hot encoded
                            y_np = np.argmax(y_np, axis=1)
                    
                    all_targets.append(y_np)
        
        except Exception as e:
            if verbose:
                print(f"Warning: Could not iterate through dataset for class weights: {e}")
                print("Proceeding without class weights")
            return
        
        if not all_targets:
            if verbose:
                print("Warning: No targets found for class weight calculation")
            return
        
        # Concatenate all targets and calculate weights
        try:
            y_all = np.concatenate(all_targets)
            self.class_weights_ = create_class_weights(y_all).to(self.device)
            
            if verbose:
                print(f"Calculated class weights: {self.class_weights_.cpu().numpy()}")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not calculate class weights: {e}")
    
    def _get_output_dim(self, y_sample: Optional[torch.Tensor]) -> int:
        if y_sample is None:
            return 2  # Default binary classification
        
        if y_sample.dim() == 1:
            self.n_classes_ = len(torch.unique(y_sample))
        else:
            self.n_classes_ = y_sample.shape[1]
        
        return self.n_classes_
    
    def _compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # NEW: Use class weights if enabled and available
        if self.config.use_class_weights and self.class_weights_ is not None:
            return F.cross_entropy(output, target.long(), weight=self.class_weights_)
        return F.cross_entropy(output, target.long())
    
    def _prepare_target(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() > 1 and y.shape[1] > 1:
            return torch.argmax(y, dim=1)
        return y.long()
    
    def _postprocess_output(self, output: torch.Tensor) -> torch.Tensor:
        return F.softmax(output, dim=1)
    
    def _get_metrics(self, eval_metric: Optional[str]) -> Dict[str, Metric]:
        metrics = {"accuracy": Accuracy(), "binary_f1": BinaryF1()}
        if self.n_classes_ == 2:
            metrics["auc"] = AUC()
        
        if eval_metric and eval_metric not in metrics:
            if eval_metric == "auc" and self.n_classes_ == 2:
                metrics["auc"] = AUC()
            elif eval_metric == "balanced_accuracy":
                metrics["balanced_accuracy"] = BalancedAccuracy()
            elif eval_metric == "logloss":
                metrics["logloss"] = LogLoss()
        
        return metrics
    
    def _get_default_metric(self) -> str:
        return "auc" if self.n_classes_ == 2 else "accuracy"


class TabNetRegressor(BaseTabNet):
    """TabNet for regression tasks."""
    
    def _get_output_dim(self, y_sample: Optional[torch.Tensor]) -> int:
        if y_sample is None:
            return 1
        return y_sample.shape[1] if y_sample.dim() > 1 else 1
    
    def _compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(output, target.float())
    
    def _prepare_target(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() == 1:
            y = y.unsqueeze(1)
        return y.float()
    
    def _postprocess_output(self, output: torch.Tensor) -> torch.Tensor:
        return output
    
    def _get_metrics(self, eval_metric: Optional[str]) -> Dict[str, Metric]:
        return {"mse": MSE()}
    
    def _get_default_metric(self) -> str:
        return "mse"


class TabNetMultiTaskClassifier(BaseTabNet):
    """TabNet for multi-task classification."""
    
    def __init__(self, config: TabNetConfig = None, **kwargs):
        super().__init__(config, **kwargs)
        self.classes_ = None
        self.target_mapper = None
        self.preds_mapper = None
    
    def _get_output_dim(self, y_sample: Optional[torch.Tensor]) -> List[int]:
        if y_sample is None:
            return [2, 2]  # Default: 2 binary tasks
        
        if y_sample.dim() < 2:
            raise ValueError("Multi-task targets should be 2D: (n_samples, n_tasks)")
        
        # For multi-task, assume each column is a separate task
        output_dims = []
        for task_idx in range(y_sample.shape[1]):
            task_targets = y_sample[:, task_idx]
            n_classes = len(torch.unique(task_targets))
            output_dims.append(n_classes)
        
        return output_dims
    
    def _compute_loss(self, outputs: List[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        total_loss = 0
        targets = targets.long()
        
        for task_idx, task_output in enumerate(outputs):
            task_loss = F.cross_entropy(task_output, targets[:, task_idx])
            total_loss += task_loss
        
        return total_loss / len(outputs)
    
    def _prepare_target(self, y: torch.Tensor) -> torch.Tensor:
        return y.long()
    
    def _postprocess_output(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        # For multi-task, return list of softmax outputs
        processed_outputs = []
        for output in outputs:
            processed_outputs.append(F.softmax(output, dim=1))
        return processed_outputs
    
    def _get_metrics(self, eval_metric: Optional[str]) -> Dict[str, Metric]:
        return {"accuracy": Accuracy()}
    
    def _get_default_metric(self) -> str:
        return "accuracy"

    
