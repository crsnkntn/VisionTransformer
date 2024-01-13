import math
import random
import os
import csv

import tensorflow as tf
import torch as t
import torch.nn as nn
import numpy as np
import tqdm.auto as tqdm
import matplotlib.pyplot as plt

from einops import reduce, rearrange, repeat
from dataclasses import dataclass
from torch import einsum
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from PIL import Image

from dataclasses import dataclass
@dataclass
class TFConfig:
    debug: bool = False

    # The Height and Width of a patch
    P: int = 4

    # The Dimension of the Model (size of a flattened patch)
    d_model: int = P * P

    # The number of transformer layers
    n_layers: int = 4

    # Base value to prevent division by zero
    ln_eps: float = 1e-5

    # The weight initialization range [-init_range, init_range]
    init_range: float = 0.02

    # The number of attention heads in each layer
    n_heads: int = 12

    # The hidden dimension of Q, K, V matrices
    d_head: int = 64

    # MLP hidden dimension
    d_mlp: int = 248

    # The number of classes, or the output size of the unembedder
    n_classes: int = 27

# Layer Normalization Object
class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(self, residual):
        # calculate the standard deviation of each row
        residual = residual - reduce(residual, "b p d -> b p 1", "mean") # The 1 preserves the dimension?

        # divide by the square of each row + a small value, then find the square root
        scale = (reduce(residual.pow(2), "b p d -> b p 1", "mean") + self.cfg.ln_eps).sqrt()
        normalized = residual / scale

        return normalized

# TODO: class BatchNorm(nn.Module):

# Classic MLP 
class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fcs1 = nn.ModuleList([nn.Linear(cfg.d_model, cfg.d_mlp) for _ in range(cfg.n_layers)])
        self.fcs2 = nn.ModuleList([nn.Linear(cfg.d_mlp, cfg.d_model) for _ in range(cfg.n_layers)])
        self.relu = nn.ReLU()

        for i in range(self.cfg.n_layers):
          nn.init.normal_(self.fcs1[i].weight, std=self.cfg.init_range)
          nn.init.normal_(self.fcs2[i].weight, std=self.cfg.init_range)
          nn.init.zeros_(self.fcs1[i].bias)
          nn.init.zeros_(self.fcs2[i].bias)

    def forward(self, x):
        for i in range(self.cfg.n_layers):
          x = self.fcs1[i](x)
          x = self.relu(x)
          x = self.fcs2[i](x)

        return x

# Attention for an Encoder Block
class EncoderAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # Attention Weights
        self.Qs = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.Ks = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.Vs = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.Qbs = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.Kbs = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.Vbs = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.Vbs = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.Ob = nn.Parameter(t.zeros(cfg.d_model))

        # Initialize the Weights
        nn.init.normal_(self.Qs, std=self.cfg.init_range)
        nn.init.normal_(self.Ks, std=self.cfg.init_range)
        nn.init.normal_(self.Vs, std=self.cfg.init_range)
        nn.init.normal_(self.O, std=self.cfg.init_range)



    def forward(self, normalized_resid_pre):
        # Calculate query, key and value vectors
        q = einsum("b p d, h d k -> b p h k", normalized_resid_pre, self.Qs) + self.Qbs
        k = einsum("b p d, h d k -> b p h k", normalized_resid_pre, self.Ks) + self.Kbs
        v = einsum("b p d, h d k -> b p h k", normalized_resid_pre, self.Vs) + self.Vbs

        # Calculate attention scores, then scale, and apply softmax
        attn_scores = einsum("b Q h k, b K h k -> b h Q K", q, k)
        attn_scores = attn_scores / np.sqrt(self.cfg.d_head)
        attn_pattern = attn_scores.softmax(-1)

        z = einsum("b K h k, b h Q K -> b Q h k", v, attn_pattern)

        # Apply another transformation to convert back to the right dimensions
        attn_out = einsum("b Q h k, h k d -> b Q d", z, self.O) + self.Ob

        return attn_out

# Attention for an Architecture with Encoders
class DecoderAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # Attention Weights
        self.Qs = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.Ks = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.Vs = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.Qbs = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.Kbs = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.Vbs = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.Vbs = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.Ob = nn.Parameter(t.zeros(cfg.d_model))

        # Initialize the Weights
        nn.init.normal_(self.Qs, std=self.cfg.init_range)
        nn.init.normal_(self.Ks, std=self.cfg.init_range)
        nn.init.normal_(self.Vs, std=self.cfg.init_range)
        nn.init.normal_(self.O, std=self.cfg.init_range)

        # Register
        self.register_buffer("IGNORE", t.tensor(-1e5, dtype=t.float32, device=device))

        # Cache for the attention grid
        self.attn_cache = None


    def forward(self, normalized_resid_pre, encoder_output):
        assert(normalized_resid_pre.shape == encoder_output.shape)

        # Calculate query, key matrices from the encoder output
        q = einsum("b p d, h d k -> b p h k", encoder_output, self.Qs) + self.Qbs
        k = einsum("b p d, h d k -> b p h k", encoder_output, self.Ks) + self.Kbs

        # Calculate the value matrices from the normalized residual stream
        v = einsum("b p d, h d k -> b p h k", normalized_resid_pre, self.Vs) + self.Vbs

        # Form the visually appealing attention grid
        attn_scores = einsum("b Q h k, b K h k -> b h Q K", q, k)
        attn_scores_masked = self.apply_causal_mask(attn_scores / np.sqrt(self.cfg.d_head))
        attn_pattern = attn_scores_masked.softmax(-1)

        z = einsum("b K h k, b h Q K -> b Q h k", v, attn_pattern)

        attn_out = einsum("b Q h k, h k d -> b Q d", z, self.O) + self.Ob

        return attn_out, attn_pattern

    def apply_causal_mask(self, attn_scores: t.Tensor):
        mask = t.triu(t.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device), diagonal=1).bool()

        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores

# Attention for an Architecture with no Encoders
class DecoderOnlyAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # Attention Weights
        self.Qs = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.Ks = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.Vs = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.Qbs = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.Kbs = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.Vbs = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.Vbs = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.Ob = nn.Parameter(t.zeros(cfg.d_model))

        # Initialize the Weights
        nn.init.normal_(self.Qs, std=self.cfg.init_range)
        nn.init.normal_(self.Ks, std=self.cfg.init_range)
        nn.init.normal_(self.Vs, std=self.cfg.init_range)
        nn.init.normal_(self.O, std=self.cfg.init_range)

        # This is for cuda management
        self.register_buffer("IGNORE", t.tensor(-1e5, dtype=t.float32, device=device))

        # Cache for the attention grid
        self.attn_cache = None


    def forward(self, normalized_resid_pre, isHooked=False):
        # Calculate query, key matrices from the encoder output
        q = einsum("b p d, h d k -> b p h k", normalized_resid_pre, self.Qs) + self.Qbs
        k = einsum("b p d, h d k -> b p h k", normalized_resid_pre, self.Ks) + self.Kbs

        # Calculate the value matrices from the normalized residual stream
        v = einsum("b p d, h d k -> b p h k", normalized_resid_pre, self.Vs) + self.Vbs

        # Form the visually appealing attention grid
        attn_scores = einsum("b Q h k, b K h k -> b h Q K", q, k)
        attn_scores_masked = self.apply_causal_mask(attn_scores / np.sqrt(self.cfg.d_head))
        attn_scores = attn_scores_masked.softmax(-1)

        self.attn_cache = attn_scores

        z = einsum("b K h k, b h Q K -> b Q h k", v, self.attn_cache)

        attn_out = einsum("b Q h k, h k d -> b Q d", z, self.O) + self.Ob

        return attn_out

    def apply_causal_mask(self, attn_scores: t.Tensor):
        mask = t.triu(t.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device), diagonal=1).bool()

        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores


    def get_attn_cache(self):
        return self.attn_cache

# Functions that Comprise an Encoder Layer
class EncoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.ln1 = LayerNorm(cfg)
        self.attn = EncoderAttention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, resid_pre):
        normalized_resid_pre = self.ln1(resid_pre)

        resid_mid = self.attn(normalized_resid_pre) + resid_pre

        normalized_resid_mid = self.ln2(resid_mid)

        resid_post = self.mlp(normalized_resid_mid) + resid_mid

        return resid_post

# Functions that Comprise a Decoder Layer
class DecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Create the objects needed for a decoder block
        self.ln = LayerNorm(cfg)
        self.attn = DecoderAttention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, resid_pre, encoder_output):
        # Normalize the residual stream
        normalized_resid_pre = self.ln(resid_pre)

        # Apply Attention and add back onto the unnormalized stream
        # Notably this requires the encoder_output
        # You can pass in the residual stream twice to get a decoder only block, but why
        resid_mid = self.attn(normalized_resid_pre, encoder_output) + resid_pre

        # Normalize again
        normalized_resid_mid = self.ln(resid_mid)

        # Pass through the MLP layer, add back onto the residual stream
        resid_post = self.mlp(normalized_resid_mid) + resid_mid

        # Normalize one more time
        return self.ln(residual_post)

# Functions that Comprise a Decoder Layer that is not Paired with Encoders
class DecoderOnlyBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Objects needed by a decoder block
        self.ln1 = LayerNorm(cfg)
        self.attn = DecoderOnlyAttention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

        # Stores attention from the last inference
        self.attn_cache = None


    def forward(self, resid_pre):
        normalized_resid_pre = self.ln1(resid_pre)
        resid_mid = self.attn(normalized_resid_pre)

        # get the attn matrix
        self.attn_cache = self.attn.get_attn_cache()
        resid_mid = resid_mid + resid_pre
        normalized_resid_mid = self.ln2(resid_mid)

        resid_post = self.mlp(normalized_resid_mid) + resid_mid

        return resid_post


    def get_attn_cache(self):
        return self.attn_cache

# Embedder for an Image; Turns Images into Embedded Patches
class Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.positional_embeddings = nn.Parameter(t.randn(49, 16), requires_grad=False)

    def forward(self, im, vis=False):
        # Create the patches from the image
        im_reshaped = im.view(28, 28)
        patches = []
        for i in range(0, 28, 4):
            for j in range(0, 28, 4):
                patch = im_reshaped[i:i+4, j:j+4]
                patches.append(t.Tensor(patch))

        # Optionally show an image of the patches in place
        if vis:
            display_patch_grid(patches)


        patches = t.stack(patches)

        # Add the positional embeddings
        for patch in patches:
            patches[i] += self.positional_embeddings[i].view(4, 4)

        # Flatten the patches
        patches = patches.view(49, -1)

        # Normalize and return these patches
        return nn.functional.softmax(patches, dim=-1)

# TODO: play around with embedding ideas

# TODO: develop embeddings for larger images

# Converts the Output of a Series of Encoder/Decoder Layers into Dictionary Logits
class Unembedder(nn.Module):
    def __init__(self, n_classes, init_range):
        super().__init__()
        # Define the parameters and initialize
        self.W = nn.Parameter(t.empty(49 * 16, n_classes))
        self.b = nn.Parameter(t.zeros(n_classes))

        nn.init.normal_(self.W, std=init_range)

    def forward(self, logits):
        # Flatten the input tensor, why is it shaped so weird
        logits = logits.view(logits.shape[0], -1)

        # Compute the matrix product using einsum and add the bias
        pred = t.einsum('bi,ij->bj', logits, self.W) + self.b
        return pred

# Functions Comprising a Vision Transformer Algorithm
class ViTF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Create the necessary objects
        self.unembedder = Unembedder(cfg.n_classes, cfg.init_range)
        self.decoder_blocks = nn.ModuleList([DecoderOnlyBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)

    def forward(self, residual_stream):
        # Pass the residual stream through all of the blocks
        for block in self.decoder_blocks:
            residual_stream = block(residual_stream)

        # Normalize and add back to the stream
        residual_stream = self.ln_final(residual_stream) + residual_stream

        # Unembed and return logits
        logits = self.unembedder(residual_stream)
        return logits

    # Utility function for getting the attention
    def getAttnPatterns(self):
        attn_patterns = []
        for block in self.decoder_blocks:
            attn_patterns.append(block.get_attn_cache())
        return attn_patterns
