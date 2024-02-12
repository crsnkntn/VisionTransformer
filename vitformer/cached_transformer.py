import torch as t
import torch.nn as nn
import numpy as np

from torch import einsum, zeros, ones, empty
from numpy import sqrt
from einops import reduce
from dataclasses import dataclass

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

# Configuration for a Vision Transformer
@dataclass
class Config:
    debug: bool = False

    # The Dimension of the Model (size of a flattened patch)
    d_model: int = 248

    # The number of transformer layers
    n_layers: int = 4

    # Base value to prevent division by zero
    ln_eps: float = 1e-5

    # The weight initialization range [-init_range, init_range]
    # Using Xavier Initialization
    init_range: float = np.sqrt(d_mlp)**-1

    # The number of attention heads in each layer
    n_heads: int = 12

    # The hidden dimension of Q, K, V matrices
    d_head: int = 64

    # MLP parameters
    d_mlp: int = 248
    n_mlp_layers: int = 1

    # The number of classes, or the output size of the unembedder
    n_classes: int = 27

# Layer Normalization Object
class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(ones(cfg.d_model))
        self.b = nn.Parameter(zeros(cfg.d_model))

    def forward(self, residual):
        # calculate the standard deviation of each row
        residual = residual - reduce(residual, "b p d -> b p 1", "mean") # The 1 preserves the dimension?

        # divide by the square of each row + a small value, then find the square root
        scale = (reduce(residual.pow(2), "b p d -> b p 1", "mean") + self.cfg.ln_eps).sqrt()
        normalized = residual / scale

        return normalized

# TODO: class BatchNorm(nn.Module):

# Classic MLP 
class CachedMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fcs1 = nn.ModuleList([nn.Linear(cfg.d_model, cfg.d_mlp) for _ in range(cfg.n_mlp_layers)])
        self.fcs2 = nn.ModuleList([nn.Linear(cfg.d_mlp, cfg.d_model) for _ in range(cfg.n_mlp_layers)])
        self.relu = nn.ReLU()

        for i in range(self.cfg.n_layers):
          nn.init.normal_(self.fcs1[i].weight, std=self.cfg.init_range)
          nn.init.normal_(self.fcs2[i].weight, std=self.cfg.init_range)
          nn.init.zeros_(self.fcs1[i].bias)
          nn.init.zeros_(self.fcs2[i].bias)

        self.activations = {}

    def forward(self, x):
        for i in range(self.cfg.n_mlp_layers):
          x = self.fcs1[i](x)
          x = self.relu(x)
          x = self.fcs2[i](x)
          self.activations[f'mlp_layer{i}'] = x

        return x

# Attention for an Architecture with no Encoders
class CachedDecoderAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # Query Matrices and Regularizer Values
        self.Qs = nn.Parameter(empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.Qbs = nn.Parameter(zeros((cfg.n_heads, cfg.d_head)))

        # Key Matrices and Regularizer Values
        self.Ks = nn.Parameter(empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.Kbs = nn.Parameter(zeros((cfg.n_heads, cfg.d_head)))

        # Value Matrices and Regularizer Values
        self.Vs = nn.Parameter(empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.Vbs = nn.Parameter(zeros((cfg.n_heads, cfg.d_head)))

        # Query Matrices and Regularizer Values
        self.O = nn.Parameter(empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.Ob = nn.Parameter(zeros(cfg.d_model))

        # Initialize Q, K, and V Weights
        nn.init.normal_(self.Qs, std=self.cfg.init_range)
        nn.init.normal_(self.Ks, std=self.cfg.init_range)
        nn.init.normal_(self.Vs, std=self.cfg.init_range)
        nn.init.normal_(self.O, std=self.cfg.init_range)

        # This is for cuda management
        self.register_buffer("IGNORE", t.tensor(-1e5, dtype=t.float32, device=device))

        self.activations = {}


    def forward(self, normalized_resid_pre, isHooked=False):
        # Calculate query, key matrices from the encoder output
        q = einsum("b p d, h d k -> b p h k", normalized_resid_pre, self.Qs) + self.Qbs
        k = einsum("b p d, h d k -> b p h k", normalized_resid_pre, self.Ks) + self.Kbs

        # Calculate the value matrices from the normalized residual stream
        v = einsum("b p d, h d k -> b p h k", normalized_resid_pre, self.Vs) + self.Vbs

        # Form the visually appealing attention grid
        attn_scores = einsum("b Q h k, b K h k -> b h Q K", q, k)
        attn_scores_masked = self.apply_causal_mask(attn_scores / sqrt(self.cfg.d_head))
        attn_scores = attn_scores_masked.softmax(-1)

        self.activations['attention_grid'] = attn_scores

        z = einsum("b K h k, b h Q K -> b Q h k", v, self.attn_cache)

        attn_out = einsum("b Q h k, h k d -> b Q d", z, self.O) + self.Ob

        return attn_out

    def apply_causal_mask(self, attn_scores: t.Tensor):
        mask = t.triu(ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device), diagonal=1).bool()

        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores

# Functions that Comprise a Decoder Layer that is not Paired with Encoders
class CachedDecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Objects needed by a decoder block
        self.ln1 = LayerNorm(cfg)
        self.attn = DecoderOnlyAttention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

        # Stores attention from the last inference
        self.activations = {}


    def forward(self, resid_pre):
        normalized_resid_pre = self.ln1(resid_pre)
        resid_mid = self.attn(normalized_resid_pre)

        # get the attn matrix
        resid_mid = resid_mid + resid_pre
        normalized_resid_mid = self.ln2(resid_mid)

        resid_post = self.mlp(normalized_resid_mid) + resid_mid

        self.activations = {**self.mlp.activations, **self.attn.activations}

        return resid_post

# Vision Transformer Implementation
class CachedTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Create the necessary objects
        self.unembedder = Unembedder(cfg.n_classes, cfg.init_range)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)

        self.activations = []

    def forward(self, residual_stream):
        self.activations.clear()
        # Pass the residual stream through all of the blocks
        for i, block in enumerate(self.decoder_blocks):
            residual_stream = block(residual_stream)
            self.activations.append(block.activations)

        # Normalize and add back to the stream
        residual_stream = self.ln_final(residual_stream) + residual_stream

        # Unembed and return logits
        logits = self.unembedder(residual_stream)

        return logits
