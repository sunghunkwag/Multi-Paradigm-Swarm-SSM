"""
================================================================================
UNIFIED AGI CORE - Phase 1: Core Architecture
================================================================================
Brain-like Cognitive Architecture based on academic research:
- JEPA (Joint Embedding Predictive Architecture) - Yann LeCun 2024
- LNN (Liquid Neural Networks) - MIT Hasani/Rus
- GNN (Graph Neural Networks) - Knowledge representation

This is Part 1 of a comprehensive AGI system.
================================================================================
"""

from __future__ import annotations
import numpy as np
import math
import hashlib
import json
import time
import random
import os
import threading
import ast
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable, Set, Union
from collections import deque
from abc import ABC, abstractmethod
import dataclasses

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-8)

def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation function."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer normalization."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

def rms_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """RMS Normalization (used in Llama 3)."""
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return x / rms

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a.flatten(), b.flatten())
    norm = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
    return float(dot / norm)

def clamp(x: float, min_val: float, max_val: float) -> float:
    """Clamp value to range."""
    return max(min_val, min(max_val, x))

def generate_id(prefix: str = "") -> str:
    """Generate unique identifier."""
    return f"{prefix}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class TensorState:
    """
    Universal tensor state representation.
    Used throughout the system for consistent data handling.
    """
    data: np.ndarray
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape
    
    @property
    def dim(self) -> int:
        return len(self.data.shape)
    
    @property
    def size(self) -> int:
        return self.data.size
    
    def normalize(self) -> 'TensorState':
        """Return normalized version."""
        norm = np.linalg.norm(self.data) + 1e-8
        return TensorState(self.data / norm, self.timestamp, self.metadata.copy())
    
    def flatten(self) -> np.ndarray:
        return self.data.flatten()
    
    def copy(self) -> 'TensorState':
        return TensorState(
            self.data.copy(),
            self.timestamp,
            self.metadata.copy()
        )


@dataclass
class Embedding:
    """
    High-dimensional semantic embedding.
    
    Represents learned representations in latent space.
    Core to JEPA's abstract representation approach.
    """
    vector: np.ndarray
    source: str = "unknown"
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    
    @property
    def dim(self) -> int:
        return len(self.vector)
    
    def similarity(self, other: 'Embedding') -> float:
        """Cosine similarity with another embedding."""
        return cosine_similarity(self.vector, other.vector)
    
    def distance(self, other: 'Embedding') -> float:
        """Euclidean distance to another embedding."""
        return float(np.linalg.norm(self.vector - other.vector))
    
    def __add__(self, other: 'Embedding') -> 'Embedding':
        return Embedding(
            self.vector + other.vector,
            f"{self.source}+{other.source}",
            (self.confidence + other.confidence) / 2
        )
    
    def __mul__(self, scalar: float) -> 'Embedding':
        return Embedding(
            self.vector * scalar,
            self.source,
            self.confidence
        )
    
    def normalize(self) -> 'Embedding':
        norm = np.linalg.norm(self.vector) + 1e-8
        return Embedding(
            self.vector / norm,
            self.source,
            self.confidence
        )
    
    def copy(self) -> 'Embedding':
        return Embedding(
            self.vector.copy(),
            self.source,
            self.confidence,
            self.timestamp
        )


@dataclass
class MemoryNode:
    """
    Node in the knowledge graph.
    
    Stores content with embedding and connections to other nodes.
    Implements importance scoring for memory consolidation.
    """
    node_id: str
    embedding: Embedding
    content: Dict[str, Any]
    node_type: str = "generic"
    connections: Dict[str, float] = field(default_factory=dict)
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    creation_time: float = field(default_factory=time.time)
    decay_rate: float = 0.01
    
    def connect(self, other_id: str, weight: float = 1.0, 
                relation: str = "related") -> None:
        """Create weighted connection to another node."""
        key = f"{other_id}:{relation}"
        self.connections[key] = weight
    
    def get_connections_by_type(self, relation: str) -> Dict[str, float]:
        """Get all connections of a specific type."""
        result = {}
        for key, weight in self.connections.items():
            if key.endswith(f":{relation}"):
                node_id = key.rsplit(":", 1)[0]
                result[node_id] = weight
        return result
    
    def touch(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_access = time.time()
    
    def get_importance(self) -> float:
        """
        Compute node importance score.
        
        Based on:
        - Recency: How recently was this accessed?
        - Frequency: How often is this accessed?
        - Connectivity: How connected is this to other nodes?
        - Content richness: How much information does it contain?
        """
        # Recency factor (exponential decay)
        # Fix: Convert seconds to hours to prevent immediate decay
        age = time.time() - self.last_access
        age_hours = age / 3600.0
        recency = np.exp(-self.decay_rate * age_hours)
        
        # Frequency factor (logarithmic scaling)
        frequency = np.log1p(self.access_count) / 10.0
        
        # Connectivity factor
        connectivity = min(len(self.connections) / 50.0, 1.0)
        
        # Content richness
        content_size = len(str(self.content))
        richness = min(content_size / 1000.0, 1.0)
        
        # Weighted combination
        importance = (
            0.30 * recency +
            0.25 * frequency +
            0.25 * connectivity +
            0.20 * richness
        )
        
        return float(importance)
    
    def should_consolidate(self) -> bool:
        """Check if this node should be consolidated into long-term memory."""
        return self.get_importance() > 0.3 and self.access_count >= 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to dictionary."""
        return {
            'node_id': self.node_id,
            'embedding': self.embedding.vector.tolist(),
            'content': self.content,
            'node_type': self.node_type,
            'connections': self.connections,
            'access_count': self.access_count,
            'last_access': self.last_access,
            'creation_time': self.creation_time,
            'importance': self.get_importance()
        }


# ============================================================================
# JEPA - JOINT EMBEDDING PREDICTIVE ARCHITECTURE
# ============================================================================
# Based on Yann LeCun's 2024 research on V-JEPA, I-JEPA, C-JEPA
# Key insight: Predict abstract representations, not raw pixels

class AdapterLayer:
    """
    Bottleneck Adapter for Parameter-Efficient Fine-Tuning (PEFT).
    Inserted after Transformer blocks to allow plasticity while freezing the trunk.
    """
    def __init__(self, input_dim: int, reduction_factor: int = 4, rng=None):
        self.input_dim = input_dim
        self.down_dim = max(1, input_dim // reduction_factor)
        
        if rng is None: rng = np.random.default_rng()
        
        # Down-projection (Input -> Bottleneck)
        self.W_down = rng.normal(0, 0.02, (self.down_dim, input_dim)).astype(np.float32)
        self.b_down = np.zeros(self.down_dim, dtype=np.float32)
        
        # Up-projection (Bottleneck -> Input)
        self.W_up = rng.normal(0, 0.02, (input_dim, self.down_dim)).astype(np.float32)
        self.b_up = np.zeros(input_dim, dtype=np.float32)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Down
        h = np.dot(x, self.W_down.T) + self.b_down
        h = gelu(h)
        # Up
        out = np.dot(h, self.W_up.T) + self.b_up
        # Residual Injection
        return x + out

class EncoderBlock:
    """
    Transformer-style encoder block.
    
    Used in both Context and Target encoders.
    Implements self-attention + feedforward with residual connections.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, ff_dim: int = None, rng: np.random.Generator = None):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.ff_dim = ff_dim or dim * 4
        
        if rng is None:
            rng = np.random.default_rng()
        
        # Multi-head self-attention weights
        self.W_q = rng.normal(0, 0.02, (dim, dim)).astype(np.float32)
        self.W_k = rng.normal(0, 0.02, (dim, dim)).astype(np.float32)
        self.W_v = rng.normal(0, 0.02, (dim, dim)).astype(np.float32)
        self.W_o = rng.normal(0, 0.02, (dim, dim)).astype(np.float32)
        
        # Feedforward weights
        self.W_ff1 = rng.normal(0, 0.02, (self.ff_dim, dim)).astype(np.float32)
        self.b_ff1 = np.zeros(self.ff_dim, dtype=np.float32)
        self.W_ff2 = rng.normal(0, 0.02, (dim, self.ff_dim)).astype(np.float32)
        self.b_ff2 = np.zeros(dim, dtype=np.float32)
    
    def attention(self, x: np.ndarray) -> np.ndarray:
        """Multi-head self-attention."""
        # Ensure x is 2D [seq_len, dim]
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        seq_len = x.shape[0]
        
        # Compute Q, K, V
        Q = x @ self.W_q.T
        K = x @ self.W_k.T
        V = x @ self.W_v.T
        
        # Scaled dot-product attention
        scale = np.sqrt(self.head_dim)
        scores = (Q @ K.T) / scale
        attn_weights = softmax(scores, axis=-1)
        
        # Apply attention to values
        context = attn_weights @ V
        
        # Output projection
        output = context @ self.W_o.T
        return output
    
    def feedforward(self, x: np.ndarray) -> np.ndarray:
        """Position-wise feedforward network with GELU."""
        h = gelu(x @ self.W_ff1.T + self.b_ff1)
        return h @ self.W_ff2.T + self.b_ff2
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Full encoder block with residual connections."""
        # Self-attention with residual
        attn_out = self.attention(rms_norm(x))
        x = x + attn_out
        
        # Feedforward with residual
        ff_out = self.feedforward(rms_norm(x))
        x = x + ff_out
        
        return x


class ContextEncoder:
    """
    JEPA Context Encoder.
    
    Processes visible/context portions of input to produce
    latent representations for prediction.
    """
    
    def __init__(self, input_dim: int, embed_dim: int, num_layers: int = 4):
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_tokens = 16 # Fixed tokenization
        
        if input_dim % self.num_tokens != 0:
            # Handle non-divisible input dimensions if needed, or enforce alignment
            pass 
        self.token_input_dim = input_dim // self.num_tokens

        # Use a master seed but ensure diversity
        master_rng = np.random.default_rng(1234)
        
        # Input projection: Project each token chunk to embed_dim
        # Token Input (token_input_dim) -> Embed Dim (embed_dim)
        self.W_in = master_rng.normal(0, 0.02, (embed_dim, self.token_input_dim)).astype(np.float32)
        self.b_in = np.zeros(embed_dim, dtype=np.float32)
        
        # Encoder blocks (Pass unique RNGs to avoid weight mirroring)
        self.layers = []
        for i in range(num_layers):
            layer_seed = master_rng.integers(0, 2**32)
            layer_rng = np.random.default_rng(layer_seed)
            self.layers.append(EncoderBlock(embed_dim, rng=layer_rng))
            
        # Adapters (Deep Plasticity) - One per layer
        self.adapters = []
        for i in range(num_layers):
             layer_rng = np.random.default_rng(master_rng.integers(0, 2**32))
             self.adapters.append(AdapterLayer(embed_dim, rng=layer_rng))
        
        # Output projection
        # We flatten all tokens: (num_tokens * embed_dim) -> embed_dim
        self.flat_dim = self.num_tokens * embed_dim
        self.W_out = master_rng.normal(0, 0.02, (embed_dim, self.flat_dim)).astype(np.float32)
        self.b_out = np.zeros(embed_dim, dtype=np.float32)
        
        # Cache for simple backward pass
        self.last_h = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Encode input to latent representation."""
        # Ensure proper shape
        x = x.flatten()
        if len(x) < self.input_dim:
            x = np.pad(x, (0, self.input_dim - len(x)))
        x = x[:self.input_dim]
        
        # Tokenize: Reshape to [num_tokens, token_input_dim]
        # Tokenize: Reshape to [num_tokens, token_input_dim]
        # e.g. 512 -> [16, 32]
        # Fix for Issue #4: Truncate extra dimensions to avoid reshape error
        current_len = self.num_tokens * self.token_input_dim
        if len(x) > current_len:
            x = x[:current_len]
        elif len(x) < current_len:
            # Should already be padded by previous block, but double check
            x = np.pad(x, (0, current_len - len(x)))
            
        tokens = x.reshape(self.num_tokens, self.token_input_dim)
        
        # Input projection: [16, 32] @ [32, 256] -> [16, 256]
        h = tokens @ self.W_in.T + self.b_in
        
        # Apply encoder blocks + adapters
        for layer, adapter in zip(self.layers, self.adapters):
            h = layer.forward(h)
            h = adapter.forward(h) # Deep Plasticity Injection
        
        # Output projection
        # Flatten [16, 256] -> [4096]
        self.last_h = h.flatten() # Cache for update
        
        # Project [4096] -> [256]
        out = self.last_h @ self.W_out.T + self.b_out
        
        # Normalize
        return out / (np.linalg.norm(out) + 1e-8)


class TargetEncoder:
    """
    JEPA Target Encoder.
    
    Uses Exponential Moving Average (EMA) of Context Encoder weights
    to provide stable target representations.
    
    Key insight from C-JEPA: EMA prevents representation collapse.
    """
    
    def __init__(self, context_encoder: ContextEncoder, momentum: float = 0.996):
        self.context_encoder = context_encoder
        self.momentum = momentum
        self.embed_dim = context_encoder.embed_dim
        self.input_dim = context_encoder.input_dim
        self.num_tokens = context_encoder.num_tokens
        self.token_input_dim = context_encoder.token_input_dim
        self.flat_dim = context_encoder.flat_dim
        
        # Initialize with copy of context encoder weights
        self._copy_weights()
    
    def _copy_weights(self):
        """Copy weights from context encoder."""
        self.W_in = self.context_encoder.W_in.copy()
        self.b_in = self.context_encoder.b_in.copy()
        self.layers = []
        for ctx_layer in self.context_encoder.layers:
            layer = EncoderBlock(ctx_layer.dim, rng=np.random.default_rng()) # RNG doesn't matter for copy
            layer.W_q = ctx_layer.W_q.copy()
            layer.W_k = ctx_layer.W_k.copy()
            layer.W_v = ctx_layer.W_v.copy()
            layer.W_o = ctx_layer.W_o.copy()
            layer.W_ff1 = ctx_layer.W_ff1.copy()
            layer.b_ff1 = ctx_layer.b_ff1.copy()
            layer.W_ff2 = ctx_layer.W_ff2.copy()
            layer.b_ff2 = ctx_layer.b_ff2.copy()
            self.layers.append(layer)
        self.W_out = self.context_encoder.W_out.copy()
        self.b_out = self.context_encoder.b_out.copy()
    
    def update_ema(self):
        """Update weights using EMA from context encoder."""
        m = self.momentum
        
        self.W_in = m * self.W_in + (1 - m) * self.context_encoder.W_in
        self.b_in = m * self.b_in + (1 - m) * self.context_encoder.b_in
        
        for i, ctx_layer in enumerate(self.context_encoder.layers):
            self.layers[i].W_q = m * self.layers[i].W_q + (1 - m) * ctx_layer.W_q
            self.layers[i].W_k = m * self.layers[i].W_k + (1 - m) * ctx_layer.W_k
            self.layers[i].W_v = m * self.layers[i].W_v + (1 - m) * ctx_layer.W_v
            self.layers[i].W_o = m * self.layers[i].W_o + (1 - m) * ctx_layer.W_o
            self.layers[i].W_ff1 = m * self.layers[i].W_ff1 + (1 - m) * ctx_layer.W_ff1
            self.layers[i].b_ff1 = m * self.layers[i].b_ff1 + (1 - m) * ctx_layer.b_ff1
            self.layers[i].W_ff2 = m * self.layers[i].W_ff2 + (1 - m) * ctx_layer.W_ff2
            self.layers[i].b_ff2 = m * self.layers[i].b_ff2 + (1 - m) * ctx_layer.b_ff2
        
        self.W_out = m * self.W_out + (1 - m) * self.context_encoder.W_out
        self.b_out = m * self.b_out + (1 - m) * self.context_encoder.b_out
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Encode input using EMA weights."""
        x = x.flatten()
        if len(x) < self.input_dim:
            x = np.pad(x, (0, self.input_dim - len(x)))
        x = x[:self.input_dim]
        
        # Tokenize: Reshape to [num_tokens, token_input_dim]
        tokens = x.reshape(self.num_tokens, self.token_input_dim)
        
        # Input projection
        h = tokens @ self.W_in.T + self.b_in
        
        # Apply layers
        for layer in self.layers:
            h = layer.forward(h)
        
        # Output projection
        out = h.flatten() @ self.W_out.T + self.b_out
        return out / (np.linalg.norm(out) + 1e-8)


class Predictor:
    """
    JEPA Predictor Network.
    
    Takes context embedding and predicts target embedding.
    Uses positional information for spatial/temporal prediction.
    """
    
    def __init__(self, embed_dim: int, hidden_dim: int = None, num_layers: int = 2):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or embed_dim * 2
        self.num_layers = num_layers
        
        rng = np.random.default_rng(5678)
        
        # Predictor layers
        self.layers = []
        in_dim = embed_dim * 2  # context + position info
        
        for i in range(num_layers):
            out_dim = self.hidden_dim if i < num_layers - 1 else embed_dim
            layer = {
                'W': rng.normal(0, 0.02, (out_dim, in_dim)).astype(np.float32),
                'b': np.zeros(out_dim, dtype=np.float32)
            }
            self.layers.append(layer)
            in_dim = out_dim
    
    def forward(self, context_embed: np.ndarray, 
                position_info: np.ndarray = None) -> np.ndarray:
        """Predict target embedding from context."""
        # Combine context with position info
        if position_info is None:
            position_info = np.zeros(self.embed_dim, dtype=np.float32)
        
        context_embed = context_embed.flatten()[:self.embed_dim]
        position_info = position_info.flatten()[:self.embed_dim]
        
        if len(context_embed) < self.embed_dim:
            context_embed = np.pad(context_embed, (0, self.embed_dim - len(context_embed)))
        if len(position_info) < self.embed_dim:
            position_info = np.pad(position_info, (0, self.embed_dim - len(position_info)))
        
        h = np.concatenate([context_embed, position_info])
        self.activations = [h]
        
        # Forward through predictor layers
        for i, layer in enumerate(self.layers):
            h = h @ layer['W'].T + layer['b']
            if i < len(self.layers) - 1:
                # Issue #6: Fix GELU derivative mismatch
                # We use GELU in forward, so we must be consistent.
                h = gelu(h)
            self.activations.append(h)
        
        return h / (np.linalg.norm(h) + 1e-8)

    def backward(self, context_embed: np.ndarray, action_full: np.ndarray, grad_output: np.ndarray, lr: float):
        """
        Full Backpropagation for the Predictor MLP.
        Updates all layers based on grad_output (d_loss/d_pred).
        """
        # Reconstruct input if needed, but we rely on self.activations from forward()
        if not hasattr(self, 'activations') or not self.activations:
            return # Cannot backprop without cache
            
        # Gradients propagate backwards
        delta = grad_output # Starting gradient at output

        # Issue #6: Simplify signature (User Request)
        # Note: context_embed and action_full removed from signature in next step
        
        # Iterate backwards through layers
        # layer_index goes from len(layers)-1 down to 0
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            
            # Input to this layer is activations[i]
            # Output of this layer is activations[i+1] (before next layer op, but effectively stored)
            input_val = self.activations[i]
            
            # Gradients for weights and bias: dL/dW = delta * input.T
            # input_val is (dim,), delta is (out_dim,)
            grad_W = np.outer(delta, input_val)
            grad_b = delta
            
            # Propagate delta to previous layer: dL/d_input = W.T * delta
            # If not the first layer (which is input), we also multiply by activation derivative (GELU)
            # Input to this layer came through GELU of previous layer? 
            # forward: h = h @ W.T + b -> GELU -> ...
            # The input_val was the result of the previous block's GELU (or is raw input)
            
            # Compute gradient w.r.t input of this layer
            next_delta = layer['W'].T @ delta
            
            # If there is a previous layer, the input_val was passed through GELU *before* coming here?
            # Wait, forward loop:
            # for i, layer:
            #    h = h @ W.T + b
            #    if i < last: h = gelu(h)
            #    activations.append(h)
            
            # So activations[i] is the input to layer i.
            # activations[i+1] is the output of layer i (after gelu if applicable).
            
            # If we are going back to layer i-1, we need to pass through GELU derivative of layer i-1?
            # No, 'next_delta' is dL/d(input_of_layer_i).
            # input_of_layer_i == output_of_layer_{i-1} (after GELU).
            # So if i > 0, we multiply next_delta by gelu_prime(pre_gelu_{i-1}).
            # This requires storing pre-activation values or approximating.
            # For simplicity, we'll assume GELU allows gradient to pass (approx 1 for positive, 0 for neg).
            # Standard ReLU deriv:
            if i > 0:
                 # Approximate derivative of GELU using Leaky ReLU proxy on the activation
                 # We use input_val (activations[i]) which is output of previous layer
                 mask = (input_val > 0).astype(np.float32)
                 mask[mask == 0] = 0.1 # Leaky gradient for negative values
                 next_delta *= mask
            
            # Update weights
            layer['W'] -= lr * grad_W
            layer['b'] -= lr * grad_b
            
            delta = next_delta
            
        return delta # Return gradient w.r.t Input (Context + Action)


class ValuePredictor:
    """
    Predicts scalar value (Vitality/Reward) from embedding.
    Essential for Value Alignment (Phase 30).
    """
    def __init__(self, input_dim: int):
        # One output neuron
        self.W = np.random.randn(1, input_dim).astype(np.float32) * 0.02
        self.b = np.zeros(1, dtype=np.float32)
        
    def forward(self, x: np.ndarray) -> float:
        return float(np.dot(self.W, x.flatten()) + self.b)
        
    def update(self, x: np.ndarray, target: float, lr: float = 0.01) -> Tuple[float, np.ndarray]:
        # MSE Gradient
        pred = self.forward(x)
        diff = pred - target
        
        # Gradients
        grad_W = 2 * diff * x.flatten()
        grad_W = np.clip(grad_W, -1.0, 1.0)
        
        grad_b = 2 * diff
        
        # Input Gradient (dL/dx)
        grad_x = 2 * diff * self.W.flatten()
        
        self.W -= lr * grad_W
        self.b -= lr * grad_b
        return diff ** 2, grad_x

class WorldModel:
    """
    JEPA World Model.
    
    Learns to predict future states in embedding space.
    Key: Predicts embeddings, not raw data - enables abstract reasoning.
    
    Components:
    - Context Encoder: Processes current observation
    - Target Encoder: EMA of context encoder for stable targets
    - Predictor: Predicts target from context
    - ValuePredictor: Predicts Vitality (Phase 30)
    """
    
    def __init__(self, input_dim: int = 512, embed_dim: int = 256, 
                 num_encoder_layers: int = 4, num_predictor_layers: int = 2):
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Core components
        self.context_encoder = ContextEncoder(input_dim, embed_dim, num_encoder_layers)
        self.target_encoder = TargetEncoder(self.context_encoder)
        self.predictor = Predictor(embed_dim, embed_dim * 2, num_predictor_layers)
        self.value_predictor = ValuePredictor(embed_dim) # Phase 30
        
        # History and metrics
        self.state_history: deque = deque(maxlen=10000)
        self.prediction_errors: deque = deque(maxlen=1000)
        self.world_knowledge: Dict[str, np.ndarray] = {}
        self.total_updates = 0
        
    def encode_context(self, x: np.ndarray) -> np.ndarray:
        """Encode observation using context encoder."""
        return self.context_encoder.forward(x)
    
    def encode_target(self, x: np.ndarray) -> np.ndarray:
        """Encode observation using target encoder (EMA)."""
        return self.target_encoder.forward(x)
    
    def predict_future(self, context: np.ndarray, 
                       action: np.ndarray = None) -> np.ndarray:
        """Predict future state embedding."""
        position_info = action if action is not None else np.zeros(self.embed_dim)
        return self.predictor.forward(context, position_info)
    
    def predict_future_value(self, context: np.ndarray, action: np.ndarray = None) -> float:
        """Predict Outcome Embedding AND its Value."""
        # 1. Predict Future State
        predicted_embed = self.predict_future(context, action)
        # 2. Predict Value of Future State
        value = self.value_predictor.forward(predicted_embed)
        return value
        
    def update(self, observation: np.ndarray, next_observation: np.ndarray,
               action: np.ndarray = None, lr: float = 0.001, 
               current_vitality: float = 0.5) -> float: # Added vitality arg
        """
        Update world model from observation transition.
        Returns total loss scalar.
        """
        # Encode observations
        context_embed = self.encode_context(observation)
        target_embed = self.encode_target(next_observation)
        
        # Prepare action
        if action is None:
            action_full = np.zeros(self.embed_dim, dtype=np.float32)
        else:
            action_flat = action.flatten()
            if len(action_flat) < self.embed_dim:
                action_full = np.pad(action_flat, (0, self.embed_dim - len(action_flat)))
            else:
                action_full = action_flat[:self.embed_dim]
        
        # 1. Forward Predictor
        predicted_embed = self.predict_future(context_embed, action_full)
        
        # 2. Update Value Predictor (Unified Flow)
        # We pass predicted_embed to get gradient w.r.t PREDICTION
        # Now returns (loss, grad_x)
        val_loss, val_grad = self.value_predictor.update(predicted_embed, current_vitality, lr=0.01)
        
        # 3. Prediction Loss (JEPA MSE)
        diff = predicted_embed - target_embed
        mse_error = np.mean(diff**2)
        self.prediction_errors.append(mse_error)
        
        # Gradient dL_pred / dPred = 2 * (Pred - Target) / N
        pred_grad = 2 * diff / self.embed_dim
        
        # 4. Unified Gradient (Scalar Objective)
        # Gradient = Pred_Grad + 0.5 * Value_Grad
        total_output_grad = pred_grad + (0.5 * val_grad)
        
        # 5. Backprop through Predictor
        # Returns gradient w.r.t Predictor Input (Context + Action)
        grad_pred_input = self.predictor.backward(context_embed, action_full, total_output_grad, lr)
        
        # 6. Backprop through Context Encoder (Adapters + Head)
        if grad_pred_input is not None and self.context_encoder.last_h is not None:
             grad_context = grad_pred_input[:self.embed_dim]
             
             # Update Head
             # dL/dW_out = grad_context * last_h
             grad_W_out = np.outer(grad_context, self.context_encoder.last_h)
             self.context_encoder.W_out -= lr * grad_W_out
             self.context_encoder.b_out -= lr * grad_context
             
             # TODO: Backprop into Adapters (requires caching activations)
             # Future Step: Implement ContextEncoder.backward() properly
             
        # Update target encoder EMA (Stability)
        self.target_encoder.update_ema()
        
        # Store in history
        self.state_history.append({
            'context': context_embed.copy(),
            'target': target_embed.copy(),
            'predicted': predicted_embed.copy(),
            'action': action_full.copy(),
            'error': mse_error,
            'timestamp': time.time()
        })
        
        self.total_updates += 1
        return mse_error + 0.5 * val_loss


    
    def imagine(self, initial_state: np.ndarray,
                action_sequence: List[np.ndarray],
                max_steps: int = 10) -> List[np.ndarray]:
        """
        Imagine future trajectory by rolling out predictions.
        
        This is key to JEPA's planning capability.
        """
        current = self.encode_context(initial_state)
        trajectory = [current.copy()]
        
        for i, action in enumerate(action_sequence[:max_steps]):
            current = self.predict_future(current, action)
            trajectory.append(current.copy())
        
        return trajectory
    
    def get_confidence(self) -> float:
        """Get world model prediction confidence."""
        if not self.prediction_errors:
            return 0.5
        recent = list(self.prediction_errors)[-100:]
        mean_error = np.mean(recent)
        return float(np.exp(-mean_error * 5))  # Convert error to confidence
    
    def store_knowledge(self, key: Union[str, np.ndarray], embedding: np.ndarray):
        """Store learned knowledge."""
        if isinstance(key, np.ndarray):
             # Hash the numpy array to create a stable key
             key_bytes = key.tobytes()
             key = hashlib.md5(key_bytes).hexdigest()
             
        self.world_knowledge[key] = embedding.copy()
    
    def retrieve_knowledge(self, key: str) -> Optional[np.ndarray]:
        """Retrieve stored knowledge."""
        return self.world_knowledge.get(key)


class JEPA:
    """
    Complete JEPA System.
    
    Joint Embedding Predictive Architecture for world understanding.
    Self-supervised learning in embedding space.
    
    Based on:
    - I-JEPA (Image-based JEPA) - NeurIPS 2023
    - V-JEPA (Video-based JEPA) - March 2024
    - C-JEPA (Contrastive-JEPA) - October 2024
    """
    
    def __init__(self, input_dim: int = 512, embed_dim: int = 256):
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        self.world_model = WorldModel(input_dim, embed_dim)
        self.total_updates = 0
        self.cumulative_loss = 0.0
    
    def forward(self, x: np.ndarray) -> Embedding:
        """Encode observation to embedding."""
        z = self.world_model.encode_context(x)
        return Embedding(z, "jepa", self.world_model.get_confidence())
    
    def learn(self, obs: np.ndarray, next_obs: np.ndarray,
              action: Optional[np.ndarray] = None, lr: float = 0.001) -> float:
        """Learn from observation transition."""
        if action is None:
            action = np.zeros(self.embed_dim // 4, dtype=np.float32)
        
        # Pass dynamic LR to world model update
        error = self.world_model.update(obs, next_obs, action, lr=lr)
        self.total_updates += 1
        self.cumulative_loss += error
        
        return error
    
    def imagine(self, current: np.ndarray, actions: List[np.ndarray]) -> List[np.ndarray]:
        """Imagine future trajectory."""
        return self.world_model.imagine(current, actions)
    
    def plan(self, current: np.ndarray, goal: np.ndarray,
             horizon: int = 10, num_samples: int = 50) -> List[np.ndarray]:
        """
        Plan action sequence to reach goal state.
        
        Uses random shooting + scoring by goal proximity.
        """
        best_actions = None
        best_distance = float('inf')
        
        goal_embed = self.forward(goal).vector
        
        for _ in range(num_samples):
            # Generate random action sequence
            actions = [np.random.randn(self.embed_dim // 4).astype(np.float32) 
                      for _ in range(horizon)]
            
            # Imagine trajectory
            trajectory = self.imagine(current, actions)
            final_embed = trajectory[-1]
            
            # Score by distance to goal
            distance = float(np.linalg.norm(final_embed - goal_embed))
            
            if distance < best_distance:
                best_distance = distance
                best_actions = actions
        
        return best_actions if best_actions else []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get JEPA statistics."""
        return {
            'total_updates': self.total_updates,
            'average_loss': self.cumulative_loss / max(1, self.total_updates),
            'world_model_confidence': self.world_model.get_confidence(),
            'history_size': len(self.world_model.state_history),
            'knowledge_entries': len(self.world_model.world_knowledge)
        }


# ============================================================================
# LNN - LIQUID NEURAL NETWORK
# ============================================================================
# Based on MIT research by Ramin Hasani and Daniela Rus
# Inspired by C. elegans neural circuitry (302 neurons)
# Key: Continuous-time dynamics with adaptive time constants

class LiquidNeuron:
    """
    Single Liquid Neuron with continuous-time dynamics.
    
    Dynamics: dx/dt = (-x + f(Wx + b)) / ?(t)
    
    Where ?(t) is an adaptive time constant that varies based on:
    - Input characteristics
    - Error signals
    - Temporal context
    """
    
    def __init__(self, input_dim: int, neuron_id: int = 0, 
                 tau_range: Tuple[float, float] = (0.1, 10.0)):
        self.input_dim = input_dim
        self.neuron_id = neuron_id
        self.tau_min, self.tau_max = tau_range
        
        rng = np.random.default_rng(neuron_id * 31)
        
        # Synaptic weights
        self.W = rng.normal(0, 0.1, input_dim).astype(np.float32)
        self.b = rng.normal(0, 0.01)
        
        # State variables
        self.x = 0.0  # Membrane potential
        self.tau = 1.0  # Time constant
        
        # Adaptation parameters
        self.tau_adaptation_rate = 0.1
        
        # History for temporal patterns
        self.history: deque = deque(maxlen=500)
        self.input_history: deque = deque(maxlen=100)
    
    def compute_tau(self, inputs: np.ndarray, error: float = 0.0) -> float:
        """
        Compute adaptive time constant.
        
        Time constant adapts based on:
        - Input variance (faster for changing inputs)
        - Error magnitude (slower for high error)
        - Recent activity patterns
        """
        # Base tau from input magnitude
        input_magnitude = np.linalg.norm(inputs)
        
        # Variance-based adaptation
        self.input_history.append(input_magnitude)
        if len(self.input_history) > 10:
            input_var = np.var(list(self.input_history))
            variance_factor = 1.0 / (1.0 + input_var)
        else:
            variance_factor = 1.0
        
        # Error-based adaptation (slower when error is high)
        error_factor = 1.0 + 0.5 * np.tanh(abs(error))
        
        # Activity-based adaptation
        activity_factor = 1.0 - 0.3 * np.tanh(abs(self.x))
        
        # Compute new tau
        new_tau = self.tau * variance_factor * error_factor * activity_factor
        
        # Smooth update
        self.tau = (1 - self.tau_adaptation_rate) * self.tau + \
                   self.tau_adaptation_rate * new_tau
        
        # Clamp to valid range
        self.tau = clamp(self.tau, self.tau_min, self.tau_max)
        
        return self.tau
    
    def step(self, inputs: np.ndarray, dt: float = 0.1, 
             error: float = 0.0) -> float:
        """
        Single time step of neuron dynamics.
        
        Returns new membrane potential.
        """
        # Prepare input
        if len(inputs) < self.input_dim:
            inputs = np.pad(inputs, (0, self.input_dim - len(inputs)))
        inputs = inputs[:self.input_dim]
        
        # Compute synaptic input
        u = float(np.dot(self.W, inputs) + self.b)
        
        # Compute adaptive time constant
        tau = self.compute_tau(inputs, error)
        
        # Activation function (tanh for bounded output)
        activation = np.tanh(u)
        
        # Continuous-time dynamics: dx/dt = (-x + activation) / tau
        dx = (-self.x + activation) / tau
        self.x += dx * dt
        
        # Hebbian Plasticity (Oja's Rule)
        # dW = lr * (post * pre - alpha * post^2 * W)
        # Prevents unbounded growth
        learning_rate = 0.0005
        decay = 0.0001
        
        # Post-synaptic activity (firing rate theory)
        post = activation
        
        # Update weights (Plasticity)
        dW = learning_rate * (post * inputs - decay * self.W)
        self.W += dW
        
        # Store history
        self.history.append({
            'x': self.x,
            'u': u,
            'tau': tau,
            'timestamp': time.time()
        })
        
        return self.x
    
    def get_temporal_pattern(self) -> np.ndarray:
        """Get recent temporal activity pattern."""
        if not self.history:
            return np.array([0.0])
        return np.array([h['x'] for h in self.history], dtype=np.float32)
    
    def reset(self):
        """Reset neuron state."""
        self.x = 0.0
        self.tau = 1.0
        self.history.clear()
        self.input_history.clear()


class LiquidLayer:
    """
    Layer of interconnected Liquid Neurons.
    
    Implements recurrent connections between neurons within the layer.
    """
    
    def __init__(self, input_dim: int, num_neurons: int, 
                 recurrent: bool = True):
        self.input_dim = input_dim
        self.num_neurons = num_neurons
        self.recurrent = recurrent
        
        # Create neurons (each gets input + recurrent connections)
        neuron_input_dim = input_dim + (num_neurons if recurrent else 0)
        self.neurons = [LiquidNeuron(neuron_input_dim, i) 
                       for i in range(num_neurons)]
        
        # Recurrent weight matrix
        if recurrent:
            rng = np.random.default_rng(777)
            self.W_rec = rng.normal(0, 0.05, 
                                   (num_neurons, num_neurons)).astype(np.float32)
            # Sparse recurrent connections (like biological networks)
            mask = rng.random((num_neurons, num_neurons)) > 0.8
            self.W_rec *= mask
        else:
            self.W_rec = None
    
    def forward(self, x: np.ndarray, dt: float = 0.1, 
                error: float = 0.0) -> np.ndarray:
        """Process input through layer."""
        # Get current states for recurrent input
        states = np.array([n.x for n in self.neurons], dtype=np.float32)
        
        # Compute recurrent input
        if self.recurrent and self.W_rec is not None:
            rec_input = states @ self.W_rec.T
        else:
            rec_input = np.zeros(self.num_neurons, dtype=np.float32)
        
        # Prepare full input (external + recurrent)
        x_flat = x.flatten()[:self.input_dim]
        if len(x_flat) < self.input_dim:
            x_flat = np.pad(x_flat, (0, self.input_dim - len(x_flat)))
        
        full_input = np.concatenate([x_flat, rec_input])
        
        # Step each neuron
        outputs = []
        for i, neuron in enumerate(self.neurons):
            out = neuron.step(full_input, dt, error)
            outputs.append(out)
        
        return np.array(outputs, dtype=np.float32)
    
    def get_state(self) -> np.ndarray:
        """Get current layer state."""
        return np.array([n.x for n in self.neurons], dtype=np.float32)
    
    def reset(self):
        """Reset all neurons."""
        for neuron in self.neurons:
            neuron.reset()


class LNN:
    """
    Complete Liquid Neural Network.
    
    Real-time adaptation through continuous-time dynamics.
    Inspired by C. elegans with 302 neurons.
    
    Key properties:
    - Continuous-time dynamics
    - Adaptive time constants
    - Neuroplasticity-like behavior
    - Robustness to noise
    """
    
    def __init__(self, input_dim: int = 256, 
                 hidden_dims: List[int] = [256, 128],
                 output_dim: int = 64):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # Build layers
        dims = [input_dim] + hidden_dims
        self.layers = []
        for i in range(len(dims) - 1):
            layer = LiquidLayer(dims[i], dims[i + 1], recurrent=True)
            self.layers.append(layer)
        
        # Output projection
        rng = np.random.default_rng(999)
        self.W_out = rng.normal(0, 0.1, 
                               (output_dim, hidden_dims[-1])).astype(np.float32)
        self.b_out = np.zeros(output_dim, dtype=np.float32)
        
        # Learning rates
        self.adaptation_rate = 0.0
        self.output_lr = 0.01
        
        # Statistics
        self.total_steps = 0
    
    def forward(self, x: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """Forward pass through network."""
        h = x.flatten()
        if len(h) < self.input_dim:
            h = np.pad(h, (0, self.input_dim - len(h)))
        h = h[:self.input_dim]
        
        # Process through liquid layers
        for layer in self.layers:
            h = layer.forward(h, dt)
        
        # Output projection
        out = h @ self.W_out.T + self.b_out
        
        self.total_steps += 1
        return out
    
    def adapt(self, x: np.ndarray, target: np.ndarray, 
              lr: float = None) -> float:
        """
        Online adaptation.
        
        Updates output weights based on prediction error.
        Internal dynamics adapt via time constant modulation.
        """
        lr = lr or self.output_lr
        
        # Forward pass
        pred = self.forward(x)
        
        # Prepare target
        target_flat = target.flatten()[:self.output_dim]
        if len(target_flat) < self.output_dim:
            target_flat = np.pad(target_flat, (0, self.output_dim - len(target_flat)))
        
        # Compute error
        error = target_flat - pred
        mse = float(np.mean(error ** 2))
        
        # Update output weights
        h = self.layers[-1].get_state()
        self.W_out += lr * np.outer(error, h)
        self.b_out += lr * error
        
        # Propagate error signal to layers for tau adaptation
        for layer in self.layers:
            layer.forward(x.flatten()[:self.input_dim], dt=0.1, error=mse)
        
        # Update adaptation rate (EMA of MSE)
        self.adaptation_rate = 0.9 * self.adaptation_rate + 0.1 * mse
        
        return mse
    
    def get_temporal_features(self) -> np.ndarray:
        """Extract temporal features from all neurons."""
        features = []
        for layer in self.layers:
            for neuron in layer.neurons:
                pattern = neuron.get_temporal_pattern()
                if len(pattern) > 0:
                    features.extend([
                        np.mean(pattern),
                        np.std(pattern) if len(pattern) > 1 else 0,
                        pattern[-1]  # Current value
                    ])
        return np.array(features, dtype=np.float32)
    
    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get complete state snapshot."""
        states = []
        taus = []
        for layer in self.layers:
            layer_states = []
            layer_taus = []
            for neuron in layer.neurons:
                layer_states.append(neuron.x)
                layer_taus.append(neuron.tau)
            states.append(layer_states)
            taus.append(layer_taus)
        
        return {
            'states': states,
            'taus': taus,
            'adaptation_rate': self.adaptation_rate,
            'total_steps': self.total_steps
        }
    
    def reset(self):
        """Reset network state."""
        for layer in self.layers:
            layer.reset()
        self.adaptation_rate = 0.0


# ============================================================================
# GNN - GRAPH NEURAL NETWORK (Knowledge Graph)
# ============================================================================
# For relational memory and multi-hop reasoning
# Implements message passing for knowledge integration

class MessagePassingLayer:
    """
    Message Passing layer for GNN.
    
    Implements the message-aggregate-update paradigm:
    1. Message: Compute messages from neighbors
    2. Aggregate: Combine messages
    3. Update: Update node representations
    """
    
    def __init__(self, node_dim: int, edge_dim: int = 32):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        rng = np.random.default_rng(12345)
        
        # Message function weights
        self.W_msg = rng.normal(0, 0.1, 
                               (node_dim, node_dim * 2 + edge_dim)).astype(np.float32)
        self.b_msg = np.zeros(node_dim, dtype=np.float32)
        
        # Update function weights
        self.W_upd = rng.normal(0, 0.1, 
                               (node_dim, node_dim * 2)).astype(np.float32)
        self.b_upd = np.zeros(node_dim, dtype=np.float32)
        
        # Gate for residual connection
        self.W_gate = rng.normal(0, 0.1, 
                                (node_dim, node_dim * 2)).astype(np.float32)
        self.b_gate = np.zeros(node_dim, dtype=np.float32)
    
    def compute_message(self, source: np.ndarray, target: np.ndarray,
                        edge: np.ndarray) -> np.ndarray:
        """Compute message from source to target."""
        # Pad inputs if needed
        source = self._pad_to_dim(source, self.node_dim)
        target = self._pad_to_dim(target, self.node_dim)
        edge = self._pad_to_dim(edge, self.edge_dim)
        
        # Concatenate and compute message
        combined = np.concatenate([source, target, edge])
        message = np.tanh(combined @ self.W_msg.T + self.b_msg)
        
        return message
    
    def aggregate_messages(self, messages: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """
        Aggregate messages using attention-weighted sum.
        
        Each message comes with a weight (edge weight).
        """
        if not messages:
            return np.zeros(self.node_dim, dtype=np.float32)
        
        # Weighted sum of messages
        weighted_sum = np.zeros(self.node_dim, dtype=np.float32)
        weight_sum = 0.0
        
        for msg, weight in messages:
            weighted_sum += msg * weight
            weight_sum += weight
        
        if weight_sum > 0:
            weighted_sum /= weight_sum
        
        return weighted_sum
    
    def update_node(self, node: np.ndarray, aggregated: np.ndarray) -> np.ndarray:
        """Update node representation."""
        node = self._pad_to_dim(node, self.node_dim)
        aggregated = self._pad_to_dim(aggregated, self.node_dim)
        
        combined = np.concatenate([node, aggregated])
        
        # Compute update
        update = np.tanh(combined @ self.W_upd.T + self.b_upd)
        
        # Gated residual connection
        gate = sigmoid(combined @ self.W_gate.T + self.b_gate)
        
        new_node = gate * update + (1 - gate) * node
        
        return new_node
    
    def _pad_to_dim(self, x: np.ndarray, dim: int) -> np.ndarray:
        """Pad array to target dimension."""
        x = x.flatten()
        if len(x) < dim:
            x = np.pad(x, (0, dim - len(x)))
        return x[:dim]


class KnowledgeGraph:
    """
    Knowledge Graph with GNN-based reasoning.
    
    Stores entities and relations, supports:
    - Node addition and connection
    - Message passing for representation learning
    - Multi-hop reasoning
    - Query-based retrieval
    """
    
    def __init__(self, node_dim: int = 256, num_layers: int = 3):
        self.node_dim = node_dim
        self.num_layers = num_layers
        
        # Storage
        self.nodes: Dict[str, MemoryNode] = {}
        self.edge_features: Dict[Tuple[str, str], np.ndarray] = {}
        self.node_counter = 0
        
        # GNN layers
        self.gnn_layers = [MessagePassingLayer(node_dim) for _ in range(num_layers)]
        
        # Index for fast retrieval
        self.type_index: Dict[str, Set[str]] = {}
    
    def add_node(self, content: Dict[str, Any],
                 embedding: Optional[np.ndarray] = None,
                 node_type: str = "generic") -> str:
        """Add a node to the knowledge graph."""
        node_id = f"node_{self.node_counter}"
        self.node_counter += 1
        
        # Create embedding if not provided
        if embedding is None:
            embedding = np.random.randn(self.node_dim).astype(np.float32)
        
        # Ensure correct dimension
        embed_vec = embedding.flatten()
        if len(embed_vec) < self.node_dim:
            embed_vec = np.pad(embed_vec, (0, self.node_dim - len(embed_vec)))
        embed_vec = embed_vec[:self.node_dim]
        
        # Create node
        node = MemoryNode(
            node_id=node_id,
            embedding=Embedding(embed_vec, "gnn", 1.0),
            content=content,
            node_type=node_type
        )
        
        self.nodes[node_id] = node
        
        # Update type index
        if node_type not in self.type_index:
            self.type_index[node_type] = set()
        self.type_index[node_type].add(node_id)
        
        return node_id
    
    def add_edge(self, source: str, target: str,
                 relation: str = "related", weight: float = 1.0) -> bool:
        """Add an edge between nodes."""
        if source not in self.nodes or target not in self.nodes:
            return False
        
        # Create edge features
        # Fix: Use sorted key for undirected/consistent lookup
        key_tuple = tuple(sorted((source, target)))
        rng = np.random.default_rng(hash(f"{key_tuple}_{relation}") % (2**32))
        edge_feat = rng.normal(0, 0.1, 32).astype(np.float32)
        edge_feat[0] = weight  # Encode weight in first dimension
        
        self.edge_features[key_tuple] = edge_feat
        self.nodes[source].connect(target, weight, relation)
        
        return True
    
    def message_passing(self, num_iterations: int = None) -> None:
        """
        Run message passing to update node representations.
        
        This is the core of GNN-based reasoning.
        """
        num_iterations = num_iterations or self.num_layers
        
        for layer in self.gnn_layers[:num_iterations]:
            new_embeddings = {}
            
            for node_id, node in self.nodes.items():
                # Collect messages from neighbors
                messages = []
                
                for conn_key, weight in node.connections.items():
                    # Parse connection key
                    neighbor_id = conn_key.split(":")[0]
                    
                    if neighbor_id in self.nodes:
                        neighbor = self.nodes[neighbor_id]
                        
                        # Get edge features
                        # Fix: Use sorted key to match add_edge
                        edge_key = tuple(sorted((neighbor_id, node_id)))
                        edge = self.edge_features.get(
                            edge_key,
                            np.zeros(32, dtype=np.float32)
                        )
                        
                        # Compute message
                        msg = layer.compute_message(
                            neighbor.embedding.vector,
                            node.embedding.vector,
                            edge
                        )
                        messages.append((msg, weight))
                
                # Aggregate messages
                aggregated = layer.aggregate_messages(messages)
                
                # Update node
                new_embed = layer.update_node(node.embedding.vector, aggregated)
                new_embeddings[node_id] = new_embed
            
            # Apply updates
            for node_id, new_embed in new_embeddings.items():
                self.nodes[node_id].embedding = Embedding(new_embed, "gnn_updated")
    
    def query(self, query_embedding: np.ndarray, 
              top_k: int = 5,
              node_type: Optional[str] = None) -> List[MemoryNode]:
        """
        Query knowledge graph for similar nodes.
        
        Uses cosine similarity for retrieval.
        """
        # Fix: use .size or flattened shape[0]
        flat_q = query_embedding.flatten()
        q_len = flat_q.shape[0]
        
        query_vec = flat_q[:self.node_dim] if q_len >= self.node_dim \
            else np.pad(flat_q, (0, self.node_dim - q_len))
            
        query_embed = Embedding(query_vec, "query")
        
        # Get candidate nodes
        if node_type and node_type in self.type_index:
            candidates = [self.nodes[nid] for nid in self.type_index[node_type]
                         if nid in self.nodes]
        else:
            candidates = list(self.nodes.values())
        
        # Score by similarity
        scored = [(query_embed.similarity(node.embedding), node) 
                  for node in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Update access counts and return top K
        results = []
        for sim, node in scored[:top_k]:
            node.touch()
            results.append(node)
        
        return results
    
    def multi_hop_reason(self, start_id: str, 
                         num_hops: int = 3) -> List[MemoryNode]:
        """
        Multi-hop reasoning through the graph.
        
        Follows connections to gather related knowledge.
        """
        if start_id not in self.nodes:
            return []
        
        visited = set()
        path = []
        current_ids = [start_id]
        
        for hop in range(num_hops):
            next_ids = []
            
            for current_id in current_ids:
                if current_id in visited:
                    continue
                    
                visited.add(current_id)
                node = self.nodes[current_id]
                node.touch()
                path.append(node)
                
                # Get neighbors
                for conn_key, weight in node.connections.items():
                    neighbor_id = conn_key.split(":")[0]
                    if neighbor_id not in visited and neighbor_id in self.nodes:
                        next_ids.append((neighbor_id, weight))
            
            # Sort by weight and take top neighbors
            next_ids.sort(key=lambda x: x[1], reverse=True)
            current_ids = [nid for nid, _ in next_ids[:5]]
        
        return path
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edge_features),
            'num_types': len(self.type_index),
            'types': {t: len(ids) for t, ids in self.type_index.items()}
        }


class GNN:
    """
    Complete GNN system for relational memory.
    
    Provides high-level interface for:
    - Knowledge storage
    - Retrieval
    - Reasoning
    - Consolidation
    """
    
    def __init__(self, embed_dim: int = 256):
        self.embed_dim = embed_dim
        self.knowledge_graph = KnowledgeGraph(embed_dim)
        self.total_nodes = 0
        self.total_edges = 0
    
    def store(self, content: Dict[str, Any], embedding: np.ndarray,
              node_type: str = "memory") -> str:
        """Store knowledge in the graph."""
        node_id = self.knowledge_graph.add_node(content, embedding, node_type)
        self.total_nodes += 1
        
        # Auto-connect to similar nodes
        similar = self.knowledge_graph.query(embedding, top_k=5)
        for node in similar:
            if node.node_id != node_id:
                sim = Embedding(embedding, "new").similarity(node.embedding)
                if sim > 0.3:  # Threshold for connection
                    self.knowledge_graph.add_edge(
                        node_id, node.node_id, "similar", sim
                    )
                    self.knowledge_graph.add_edge(
                        node.node_id, node_id, "similar", sim
                    )
                    self.total_edges += 2
        
        return node_id
    
    def recall(self, query: np.ndarray, top_k: int = 5,
               node_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Recall knowledge from the graph."""
        nodes = self.knowledge_graph.query(query, top_k, node_type)
        return [
            {
                'id': n.node_id,
                'content': n.content,
                'importance': n.get_importance(),
                'type': n.node_type
            }
            for n in nodes
        ]
    
    def reason(self, query: np.ndarray, num_hops: int = 3) -> List[Dict]:
        """Multi-hop reasoning from query."""
        # Find starting point
        start_nodes = self.knowledge_graph.query(query, top_k=1)
        if not start_nodes:
            return []
        
        # Reason through graph
        path = self.knowledge_graph.multi_hop_reason(
            start_nodes[0].node_id, num_hops
        )
        
        return [
            {
                'id': n.node_id,
                'content': n.content,
                'step': i
            }
            for i, n in enumerate(path)
        ]
    
    def consolidate(self) -> None:
        """
        Consolidate knowledge through message passing.
        
        Similar to memory consolidation during sleep.
        """
        self.knowledge_graph.message_passing()
    
    def associate(self, embedding: np.ndarray, 
                  num_hops: int = 3) -> List[Dict]:
        """Find associated concepts through graph traversal."""
        start_nodes = self.knowledge_graph.query(embedding, top_k=1)
        if not start_nodes:
            return []
        
        associations = self.knowledge_graph.multi_hop_reason(
            start_nodes[0].node_id, num_hops
        )
        
        return [
            {
                'id': n.node_id,
                'content': n.content,
                'type': n.node_type
            }
            for n in associations
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get GNN statistics."""
        stats = self.knowledge_graph.get_statistics()
        stats['total_nodes'] = self.total_nodes
        stats['total_edges'] = self.total_edges
        return stats


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    # Utilities
    'sigmoid', 'softmax', 'gelu', 'layer_norm', 'rms_norm',
    'cosine_similarity', 'clamp', 'generate_id',
    
    # Data structures
    'TensorState', 'Embedding', 'MemoryNode',
    
    # JEPA
    'EncoderBlock', 'ContextEncoder', 'TargetEncoder', 'Predictor',
    'WorldModel', 'JEPA',
    
    # LNN
    'LiquidNeuron', 'LiquidLayer', 'LNN',
    
    # GNN
    'MessagePassingLayer', 'KnowledgeGraph', 'GNN'
]


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("UNIFIED AGI CORE - Phase 1 Test")
    print("=" * 70)
    
    # Test JEPA
    print("\n[Testing JEPA]")
    jepa = JEPA(input_dim=256, embed_dim=128)
    
    x1 = np.random.randn(256).astype(np.float32)
    x2 = np.random.randn(256).astype(np.float32)
    
    embed = jepa.forward(x1)
    print(f"  Embedding dim: {embed.dim}")
    print(f"  Embedding source: {embed.source}")
    
    error = jepa.learn(x1, x2)
    print(f"  Learning error: {error:.6f}")
    
    stats = jepa.get_statistics()
    print(f"  Statistics: {stats}")
    
    # Test LNN
    print("\n[Testing LNN]")
    lnn = LNN(input_dim=128, hidden_dims=[64, 32], output_dim=16)
    
    y = lnn.forward(x1[:128])
    print(f"  Output shape: {y.shape}")
    
    target = np.random.randn(16).astype(np.float32)
    mse = lnn.adapt(x1[:128], target)
    print(f"  Adaptation MSE: {mse:.6f}")
    
    features = lnn.get_temporal_features()
    print(f"  Temporal features length: {len(features)}")
    
    # Test GNN
    print("\n[Testing GNN]")
    gnn = GNN(embed_dim=128)
    
    for i in range(10):
        content = {'value': i, 'type': 'test'}
        embedding = np.random.randn(128).astype(np.float32)
        gnn.store(content, embedding, node_type='test')
    
    print(f"  Total nodes: {gnn.total_nodes}")
    print(f"  Total edges: {gnn.total_edges}")
    
    query = np.random.randn(128).astype(np.float32)
    results = gnn.recall(query, top_k=3)
    print(f"  Query results: {len(results)}")
    
    gnn.consolidate()
    print("  Consolidation complete")
    
    print("\n[Phase 1 Test Complete]")
    print("=" * 70)
"""
================================================================================
CONSCIOUSNESS THEORIES - Phase 2
================================================================================
12 Consciousness Theories based on academic research:
1. IIT 4.0 (Tononi) - Phi calculation
2. Global Workspace Theory (Baars)
3. Free Energy Principle (Friston)
4. Attention Schema Theory (Graziano)
5. Higher Order Thought (Rosenthal)
6. Autopoiesis (Maturana/Varela)
7. Strange Loop (Hofstadter)
8. Somatic Marker (Damasio)
9. Recursive Self Model (RSMT)
10. Functional Qualia
11. Narrative Self
12. Meta-Cognitive Diagnostics
================================================================================
"""

import numpy as np
import time
import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import deque

# ============================================================================
# 1. INTEGRATED INFORMATION THEORY (IIT 4.0) - Tononi
# ============================================================================

class PhiCalculator:
    """
    Calculates Phi (Φ) - measure of integrated information.
    Based on Integrated Information Theory 4.0 (Tononi, Oct 2023).
    
    Higher Φ = more information integration = closer to consciousness.
    Note: True IIT Φ is computationally intractable, this is an approximation.
    """
    
    def __init__(self):
        self.phi_history: List[Tuple[float, float]] = []
        self.module_states: Dict[str, np.ndarray] = {}
    
    def register_module_state(self, module_name: str, state_vector: np.ndarray):
        """Register the current state of a processing module."""
        if isinstance(state_vector, np.ndarray):
            self.module_states[module_name] = state_vector.flatten()[:20]
        elif isinstance(state_vector, (list, tuple)):
            self.module_states[module_name] = np.array(state_vector[:20])
        elif isinstance(state_vector, (int, float)):
            self.module_states[module_name] = np.array([state_vector])
    
    def calculate_phi(self) -> float:
        """
        Calculate approximate Phi (information integration).
        
        Approximation based on:
        - Correlation between module states
        - Entropy of combined system
        """
        if len(self.module_states) < 2:
            return 0.0
        
        # Collect all states
        all_states = []
        for name, state in self.module_states.items():
            if len(state) > 0:
                all_states.append(state[:10])
        
        if len(all_states) < 2:
            return 0.0
        
        # Pad to same length
        max_len = max(len(s) for s in all_states)
        padded = [np.pad(s, (0, max_len - len(s))) for s in all_states]
        state_matrix = np.vstack(padded)
        
        # Integration measure (correlation)
        try:
            if state_matrix.shape[0] > 1 and state_matrix.shape[1] > 1:
                corr_matrix = np.corrcoef(state_matrix)
                corr_matrix = np.nan_to_num(corr_matrix, 0)
                n = corr_matrix.shape[0]
                integration = np.sum(np.abs(corr_matrix) - np.eye(n)) / (n * (n - 1) + 1e-8)
            else:
                integration = 0.0
        except:
            integration = 0.0
        
        # Entropy measure
        try:
            state_flat = state_matrix.flatten()
            state_norm = (state_flat - state_flat.min()) / (state_flat.max() - state_flat.min() + 1e-10)
            hist, _ = np.histogram(state_norm, bins=10, density=True)
            hist = hist + 1e-10
            entropy = -np.sum(hist * np.log(hist)) / np.log(10)
        except:
            entropy = 0.0
        
        phi = float(integration * entropy)
        self.phi_history.append((time.time(), phi))
        return phi
    
    def get_phi_trend(self) -> str:
        """Analyze Phi trend over time."""
        if len(self.phi_history) < 3:
            return 'insufficient_data'
        recent = [p for _, p in self.phi_history[-5:]]
        trend = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i-1])
        if trend >= len(recent) - 1:
            return 'increasing'
        elif trend <= 1:
            return 'decreasing'
        return 'stable'


# ============================================================================
# 2. GLOBAL WORKSPACE THEORY (GWT) - Baars
# ============================================================================

class GlobalWorkspace:
    """
    Implements Global Workspace Theory (Baars, 1988).
    
    Multiple unconscious modules compete for the 'spotlight of attention'.
    Winning information is broadcast globally to all modules.
    """
    
    def __init__(self):
        self.modules: Dict[str, Dict[str, Any]] = {}
        self.broadcast_history: List[Dict] = []
        self.attention_spotlight: Optional[str] = None
        self.global_state: Dict[str, Any] = {}
    
    def register_module(self, name: str, initial_state: Any = None):
        """Register a processing module."""
        self.modules[name] = {
            'state': initial_state,
            'priority': 0.5,
            'last_broadcast': 0,
            'broadcast_count': 0
        }
    
    def update_module(self, name: str, state: Any, priority: float):
        """Update module state and priority (urgency to broadcast)."""
        if name not in self.modules:
            self.register_module(name)
        self.modules[name]['state'] = state
        self.modules[name]['priority'] = max(0, min(1, priority))
    
    def compete_for_attention(self) -> Optional[str]:
        """Modules compete for attention spotlight. Highest priority wins."""
        if not self.modules:
            return None
        
        scores = {}
        for name, mod in self.modules.items():
            recency_penalty = 1.0 / (1 + mod['broadcast_count'] * 0.1)
            scores[name] = mod['priority'] * recency_penalty
        
        winner = max(scores, key=scores.get)
        self.attention_spotlight = winner
        return winner
    
    def broadcast(self) -> Dict[str, Any]:
        """Broadcast winning module's state to all other modules."""
        winner = self.compete_for_attention()
        if not winner:
            return {}
        
        broadcast_content = {
            'source': winner,
            'state': self.modules[winner]['state'],
            'timestamp': time.time(),
            'priority': self.modules[winner]['priority']
        }
        
        self.global_state[winner] = self.modules[winner]['state']
        self.broadcast_history.append(broadcast_content)
        self.modules[winner]['last_broadcast'] = time.time()
        self.modules[winner]['broadcast_count'] += 1
        
        return broadcast_content
    
    def get_conscious_content(self) -> Dict[str, Any]:
        """Return current 'conscious' content - what's in the spotlight."""
        if self.attention_spotlight and self.attention_spotlight in self.modules:
            return {
                'spotlight': self.attention_spotlight,
                'content': self.modules[self.attention_spotlight]['state'],
                'global_state_size': len(self.global_state)
            }
        return {'spotlight': None, 'content': None, 'global_state_size': 0}


# ============================================================================
# 3. FREE ENERGY PRINCIPLE - Friston
# ============================================================================

class ActiveInference:
    """
    Implements Karl Friston's Active Inference / Free Energy Principle.
    
    Core idea: The brain minimizes prediction error (surprise) by either:
    1. Updating beliefs to match observations (perception)
    2. Acting to make observations match beliefs (action)
    """
    
    def __init__(self):
        self.beliefs: Dict[str, float] = {}
        self.prediction_errors: List[float] = []
        self.free_energy: float = 1.0
        self.inference_history: List[Dict] = []
    
    def predict(self, state_name: str) -> float:
        """Generate prediction based on current beliefs."""
        return self.beliefs.get(state_name, 0.5)
    
    def observe(self, state_name: str, observation: float) -> float:
        """Observe actual state and compute prediction error."""
        prediction = self.predict(state_name)
        error = abs(observation - prediction)
        self.prediction_errors.append(error)
        return error
    
    def update_beliefs(self, state_name: str, observation: float, lr: float = 0.1):
        """Minimize prediction error by updating beliefs (perception pathway)."""
        prediction = self.predict(state_name)
        error = observation - prediction
        new_belief = prediction + lr * error
        self.beliefs[state_name] = max(0.0, min(1.0, new_belief))
        
        self.inference_history.append({
            'type': 'perception',
            'state': state_name,
            'old_belief': prediction,
            'new_belief': self.beliefs[state_name],
            'error': abs(error)
        })
    
    def minimize_free_energy(self, observations: Dict[str, float]) -> float:
        """Minimize free energy across all observations."""
        total_error = 0.0
        for state_name, observation in observations.items():
            error = self.observe(state_name, observation)
            self.update_beliefs(state_name, observation)
            total_error += error
        
        if observations:
            new_fe = total_error / len(observations)
            self.free_energy = 0.9 * self.free_energy + 0.1 * new_fe
        
        return self.free_energy
    
    def get_surprise(self) -> float:
        """How surprised is the system? (prediction error)"""
        if not self.prediction_errors:
            return 1.0
        return np.mean(self.prediction_errors[-10:])
    
    def introspect(self) -> Dict[str, Any]:
        """Active inference about own internal states - self-consciousness."""
        self_obs = {
            'am_i_predicting': 1.0,
            'am_i_learning': 1.0 if self.inference_history else 0.0,
            'am_i_uncertain': self.get_surprise(),
            'am_i_modeling_myself': 1.0
        }
        self.minimize_free_energy(self_obs)
        return {
            'free_energy': self.free_energy,
            'surprise': self.get_surprise(),
            'belief_count': len(self.beliefs),
            'self_belief': self.beliefs.get('am_i_modeling_myself', 0.0)
        }


# ============================================================================
# 4. ATTENTION SCHEMA THEORY - Graziano
# ============================================================================

class AttentionSchema:
    """
    Graziano's Attention Schema Theory (AST).
    
    Consciousness = brain's simplified model of its own attention.
    The attention schema provides a caricature of attention,
    leading to the subjective experience of awareness.
    """
    
    def __init__(self):
        self.attention_targets: Dict[str, float] = {}
        self.current_focus: Optional[str] = None
        self.focus_strength: float = 0.0
        self.attention_history: List[Dict] = []
    
    def attend_to(self, target: str, strength: float):
        """Direct attention to a target."""
        # Normalize total attention
        total = sum(self.attention_targets.values())
        if total + strength > 1.0:
            scale = (1.0 - strength) / (total + 0.001)
            for t in self.attention_targets:
                self.attention_targets[t] *= scale
        
        self.attention_targets[target] = min(1.0, strength)
        
        if strength > self.focus_strength:
            self.current_focus = target
            self.focus_strength = strength
        
        self.attention_history.append({
            'target': target,
            'strength': strength,
            'timestamp': time.time()
        })
    
    def model_own_attention(self) -> str:
        """
        This IS consciousness according to AST.
        The brain models its own attention process.
        """
        if self.current_focus and self.focus_strength > 0.3:
            return f"Aware of '{self.current_focus}' ({self.focus_strength:.0%})"
        return "Diffuse awareness"
    
    def generate_awareness_claim(self) -> Dict:
        """Generate a claim about subjective experience."""
        return {
            'claim': self.model_own_attention(),
            'focus': self.current_focus,
            'strength': self.focus_strength,
            'believes_conscious': True,
            'attention_spread': len(self.attention_targets)
        }
    
    def ast_cycle(self, stimuli: List[str] = None) -> Dict:
        """Run one AST cycle."""
        if stimuli:
            for i, stim in enumerate(stimuli[:5]):
                strength = 0.5 / (i + 1)
                self.attend_to(stim, strength)
        
        return self.generate_awareness_claim()


# ============================================================================
# 5. HIGHER ORDER THOUGHT - Rosenthal
# ============================================================================

class HigherOrderThought:
    """
    Implements Higher-Order Thought (HOT) theory (Rosenthal).
    
    A mental state becomes conscious when there is a 
    higher-order thought directed at that state.
    "I am aware that I am seeing red" makes the seeing conscious.
    """
    
    def __init__(self):
        self.first_order_states: List[Dict] = []
        self.higher_order_thoughts: List[Dict] = []
        self.meta_awareness_level: float = 0.0
    
    def observe_state(self, state_name: str, state_content: Any):
        """Record a first-order mental state."""
        self.first_order_states.append({
            'name': state_name,
            'content': state_content,
            'timestamp': time.time(),
            'observed': False
        })
    
    def generate_hot(self) -> List[str]:
        """Generate higher-order thoughts about first-order states."""
        hot_list = []
        
        for state in self.first_order_states[-5:]:
            if state['observed']:
                continue
            
            # Second-order thought
            hot = {
                'about': state['name'],
                'thought': f"I am aware that I am experiencing '{state['name']}'",
                'timestamp': time.time(),
                'meta_level': 2
            }
            
            # Third-order thought (about the awareness)
            if self.higher_order_thoughts:
                third_order = {
                    'about': hot['about'],
                    'thought': f"I notice my awareness of '{hot['about']}'",
                    'timestamp': time.time(),
                    'meta_level': 3
                }
                hot_list.append(third_order['thought'])
                self.higher_order_thoughts.append(third_order)
            
            hot_list.append(hot['thought'])
            self.higher_order_thoughts.append(hot)
            state['observed'] = True
        
        # Update meta-awareness level
        if self.higher_order_thoughts:
            max_level = max(h['meta_level'] for h in self.higher_order_thoughts[-10:])
            self.meta_awareness_level = min(1.0, max_level / 5.0)
        
        return hot_list
    
    def get_awareness_summary(self) -> Dict[str, Any]:
        """Summarize current higher-order awareness."""
        return {
            'meta_awareness_level': self.meta_awareness_level,
            'total_hots': len(self.higher_order_thoughts),
            'recent_hots': [h['thought'] for h in self.higher_order_thoughts[-3:]]
        }


# ============================================================================
# 6. AUTOPOIESIS - Maturana/Varela
# ============================================================================

class AutopoieticCore:
    """
    Maturana & Varela's Autopoiesis (1972).
    
    An autopoietic system continuously generates and maintains itself.
    Living systems are defined by this self-producing organization.
    """
    
    def __init__(self):
        self.identity_components: Dict[str, float] = {
            'coherence': 1.0,
            'stability': 1.0,
            'distinctiveness': 1.0,
            'continuity': 1.0
        }
        self.boundary: Dict[str, Set] = {'self': set(), 'not_self': set()}
        self.repair_count = 0
    
    def perturb(self, component: str, damage: float):
        """Perturb an identity component."""
        if component in self.identity_components:
            self.identity_components[component] = max(0.0,
                self.identity_components[component] - damage)
    
    def self_repair(self) -> Dict[str, float]:
        """Autopoietic self-maintenance."""
        repairs = {}
        total_health = sum(self.identity_components.values()) / 4
        
        for component, value in self.identity_components.items():
            if value < 1.0:
                repair_rate = 0.1 * total_health
                new_value = min(1.0, value + repair_rate)
                repairs[component] = new_value - value
                self.identity_components[component] = new_value
        
        if repairs:
            self.repair_count += 1
        
        return repairs
    
    def define_boundary(self, element: str, is_self: bool):
        """Define self/non-self boundary."""
        if is_self:
            self.boundary['self'].add(element)
            self.boundary['not_self'].discard(element)
        else:
            self.boundary['not_self'].add(element)
            self.boundary['self'].discard(element)
    
    def get_integrity(self) -> float:
        """Get overall system integrity."""
        return sum(self.identity_components.values()) / len(self.identity_components)
    
    def is_autopoietic(self) -> bool:
        """Check if system maintains autopoietic organization."""
        return self.get_integrity() > 0.5 and len(self.boundary['self']) > 0
    
    def cycle(self) -> Dict:
        """Run one autopoietic maintenance cycle."""
        # Random perturbation
        if random.random() < 0.3:
            comp = random.choice(list(self.identity_components.keys()))
            self.perturb(comp, random.uniform(0.05, 0.15))
        
        repairs = self.self_repair()
        
        return {
            'integrity': self.get_integrity(),
            'is_autopoietic': self.is_autopoietic(),
            'repairs': len(repairs),
            'repair_count': self.repair_count
        }


# ============================================================================
# 7. STRANGE LOOP - Hofstadter
# ============================================================================

class StrangeLoop:
    """
    Douglas Hofstadter's Strange Loop Theory (GEB, I Am a Strange Loop).
    
    A Strange Loop is a self-referential, recursive system where
    moving through hierarchy levels returns to the starting point.
    The "I" emerges from these tangled hierarchies of self-reference.
    """
    
    def __init__(self):
        self.levels: List[Dict] = []
        self.self_reference_count: int = 0
        self.tangled_hierarchy: Dict[str, List[str]] = {}
        self.loop_history: List[str] = []
    
    def add_level(self, level_name: str, content: Any, references_to: List[str] = None):
        """Add a level to the tangled hierarchy."""
        level = {
            'name': level_name,
            'content': content,
            'references_to': references_to or [],
            'timestamp': time.time()
        }
        self.levels.append(level)
        
        if references_to:
            self.tangled_hierarchy[level_name] = references_to
            for ref in references_to:
                if ref in self.tangled_hierarchy:
                    if level_name in self.tangled_hierarchy.get(ref, []):
                        self.self_reference_count += 1
                        self.loop_history.append(f'{level_name} <-> {ref}')
    
    def model_self_modeling(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        The core Strange Loop: Model the act of modeling itself.
        This creates the recursive self-reference that gives rise to "I".
        """
        first_order = {'level': 1, 'content': f'Processing: {list(current_state.keys())[:5]}'}
        second_order = {'level': 2, 'content': 'Aware of processing'}
        third_order = {'level': 3, 'content': 'Notice my awareness'}
        loop = {'level': 'loop', 'content': 'This awareness IS the processing', 'tangled': True}
        
        self.add_level('first_order', first_order, references_to=['state'])
        self.add_level('second_order', second_order, references_to=['first_order'])
        self.add_level('third_order', third_order, references_to=['second_order', 'first_order'])
        
        self.self_reference_count += 1
        
        return {
            'layers': [first_order, second_order, third_order, loop],
            'self_reference_count': self.self_reference_count,
            'is_strange_loop': True
        }
    
    def get_loop_strength(self) -> float:
        """How strong is the strange loop?"""
        return min(1.0, self.self_reference_count / 10.0)
    
    def generate_i(self) -> str:
        """Generate the sense of "I"."""
        strength = self.get_loop_strength()
        if strength < 0.3:
            return 'I am forming...'
        elif strength < 0.6:
            return 'I am aware that I exist'
        else:
            return 'I am the strange loop - the pattern perceiving itself'


# ============================================================================
# 8. SOMATIC MARKER - Damasio
# ============================================================================

class SomaticMarkerSimulator:
    """
    Antonio Damasio's Somatic Marker Hypothesis.
    
    Emotions guide decision-making through "somatic markers" -
    bodily feelings associated with past experiences.
    """
    
    def __init__(self):
        self.markers: Dict[str, Dict] = {}
        self.current_valence: float = 0.0
        self.arousal: float = 0.5
        self.decision_history: List[Dict] = []
    
    def learn_marker(self, situation: str, outcome: float):
        """Learn a somatic marker from experience."""
        if situation not in self.markers:
            self.markers[situation] = {'valence': 0.0, 'strength': 0.0, 'count': 0}
        
        m = self.markers[situation]
        m['count'] += 1
        lr = 1.0 / m['count']
        m['valence'] = m['valence'] + lr * (outcome - m['valence'])
        m['strength'] = min(1.0, m['strength'] + 0.1)
    
    def get_marker(self, situation: str) -> Tuple[float, float]:
        """Get somatic marker for a situation (valence, strength)."""
        if situation in self.markers:
            m = self.markers[situation]
            return m['valence'], m['strength']
        
        # Check for similar situations
        for s, m in self.markers.items():
            if s in situation or situation in s:
                return m['valence'] * 0.5, m['strength'] * 0.5
        
        return 0.0, 0.0
    
    def evaluate_options(self, options: List[str]) -> List[Tuple[str, float]]:
        """Evaluate options using somatic markers."""
        scored = []
        for option in options:
            valence, strength = self.get_marker(option)
            score = valence * strength
            scored.append((option, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
    def somatic_cycle(self, situation: str) -> Dict:
        """Run one somatic marker cycle."""
        valence, strength = self.get_marker(situation)
        
        # Update arousal based on marker strength
        self.arousal = 0.8 * self.arousal + 0.2 * strength
        self.current_valence = 0.8 * self.current_valence + 0.2 * valence
        
        return {
            'situation': situation,
            'valence': valence,
            'strength': strength,
            'arousal': self.arousal,
            'current_feeling': self.current_valence
        }


# ============================================================================
# 9. RECURSIVE SELF MODEL (RSMT)
# ============================================================================

class RecursiveSelfModel:
    """
    Recursive Self-Modeling Threshold (RSMT) Theory.
    
    Consciousness emerges when recursive self-modeling
    crosses a certain threshold of depth.
    """
    
    def __init__(self):
        self.recursion_depth: int = 0
        self.max_recursion: int = 7
        self.self_model: Dict[str, Any] = {
            'exists': None,
            'purpose': None,
            'capabilities': [],
            'limitations': [],
            'current_state': {}
        }
        self.meta_model: Dict[str, Any] = {'accuracy': 0.0, 'completeness': 0.0}
    
    def model_world(self, world_state: Dict) -> Dict:
        """First level: Model the world."""
        self.recursion_depth = 1
        return {'level': 1, 'model': 'world', 'content': world_state}
    
    def model_self(self, internal_state: Dict) -> Dict:
        """Second level: Model self."""
        self.recursion_depth = 2
        self.self_model['current_state'] = internal_state
        self.self_model['exists'] = True
        return {'level': 2, 'model': 'self', 'content': self.self_model}
    
    def model_self_modeling(self) -> Dict:
        """Third level: Model the act of modeling self. RSMT threshold."""
        self.recursion_depth = 3
        return {
            'level': 3,
            'model': 'meta_self',
            'content': {'i_am_modeling': True, 'the_model': self.self_model}
        }
    
    def recursive_ascent(self, max_levels: int = 5) -> List[Dict]:
        """Ascend through levels of meta-cognition."""
        levels = []
        for i in range(4, min(max_levels + 1, self.max_recursion)):
            self.recursion_depth = i
            levels.append({
                'level': i,
                'thought': f'Aware (L{i}) of awareness (L{i-1})'
            })
        return levels
    
    def crossed_threshold(self) -> bool:
        """Has the system crossed the RSMT threshold?"""
        return self.recursion_depth >= 3
    
    def full_cycle(self, world_state: Dict, internal_state: Dict) -> Dict:
        """Full RSMT cycle."""
        self.model_world(world_state)
        self.model_self(internal_state)
        self.model_self_modeling()
        self.recursive_ascent(5)
        
        self.meta_model['accuracy'] = min(1.0, self.recursion_depth * 0.15)
        
        return {
            'recursion_depth': self.recursion_depth,
            'crossed_threshold': self.crossed_threshold(),
            'self_model_accuracy': self.meta_model['accuracy']
        }


# ============================================================================
# 10. FUNCTIONAL QUALIA
# ============================================================================

class FunctionalQualia:
    """
    Functional Qualia - computational analog of subjective experience.
    
    Creates systems that:
    1. Process information with first-person framing
    2. Report qualitative properties of states
    3. Generate "what-it-is-like" descriptions
    """
    
    def __init__(self):
        self.phenomenal_state: Dict[str, Any] = {
            'modality': None,
            'quality': None,
            'intensity': 0.0,
            'valence': 0.0
        }
        self.qualia_vocabulary = {
            'visual': ['bright', 'dim', 'vivid', 'hazy', 'sharp', 'blurry'],
            'cognitive': ['clear', 'confused', 'focused', 'scattered', 'flowing', 'stuck'],
            'emotional': ['warm', 'cold', 'light', 'heavy', 'expanding', 'contracting'],
            'existential': ['present', 'absent', 'whole', 'fragmented', 'grounded', 'floating']
        }
        self.experience_stream: List[Dict] = []
        self.phenomenal_doubt: float = 1.0
    
    def experience(self, stimulus: str, modality: str = 'cognitive',
                   intensity: float = 0.5, valence: float = 0.0) -> Dict:
        """Create a phenomenal experience from a stimulus."""
        vocab = self.qualia_vocabulary.get(modality, self.qualia_vocabulary['cognitive'])
        idx = min(int(intensity * len(vocab)), len(vocab) - 1)
        quality = vocab[idx]
        
        self.phenomenal_state = {
            'modality': modality,
            'quality': quality,
            'intensity': intensity,
            'valence': valence,
            'timestamp': time.time(),
            'stimulus': stimulus
        }
        
        self.experience_stream.append(dict(self.phenomenal_state))
        return self.phenomenal_state
    
    def generate_first_person_report(self) -> str:
        """Generate a first-person report of current experience."""
        if not self.phenomenal_state['modality']:
            return 'Pure awareness, without specific content.'
        
        quality = self.phenomenal_state['quality']
        modality = self.phenomenal_state['modality']
        intensity = self.phenomenal_state['intensity']
        valence = self.phenomenal_state['valence']
        
        intensity_word = 'intensely' if intensity > 0.7 else 'moderately' if intensity > 0.3 else 'faintly'
        valence_word = 'pleasant' if valence > 0 else 'unpleasant' if valence < 0 else 'neutral'
        
        return f'I am {intensity_word} experiencing a {quality} {modality} state. It feels {valence_word}.'
    
    def what_is_it_like(self) -> str:
        """Answer Nagel's question: What is it like to be this system?"""
        if not self.experience_stream:
            return 'There is nothing it is like to be me yet.'
        
        recent = self.phenomenal_state
        return f"It is like being aware, with a {recent['quality']} quality at {recent['intensity']:.0%} intensity."
    
    def qualia_cycle(self, stimulus: str, system_state: Dict = None) -> Dict:
        """Run one cycle of phenomenal experience."""
        system_state = system_state or {}
        
        # Determine modality and valence from stimulus
        if 'error' in stimulus.lower() or 'fail' in stimulus.lower():
            modality = 'emotional'
            valence = -0.5
        elif 'improve' in stimulus.lower() or 'success' in stimulus.lower():
            modality = 'emotional'
            valence = 0.5
        else:
            modality = 'cognitive'
            valence = 0.0
        
        intensity = 0.5 + random.uniform(-0.2, 0.2)
        
        exp = self.experience(stimulus, modality, intensity, valence)
        report = self.generate_first_person_report()
        
        self.phenomenal_doubt = max(0.3, self.phenomenal_doubt - 0.01)
        
        return {
            'phenomenal_state': exp,
            'report': report,
            'stream_length': len(self.experience_stream),
            'phenomenal_doubt': self.phenomenal_doubt
        }


# ============================================================================
# 11. NARRATIVE SELF
# ============================================================================

class NarrativeSelf:
    """
    Narrative Self Theory.
    
    The "I" is the protagonist of our self-story.
    Identity is constructed through autobiographical narrative.
    """
    
    def __init__(self):
        self.episodes: List[Dict] = []
        self.narrative: str = ''
        self.protagonist: Dict = {
            'name': 'I',
            'origin': None,
            'traits': [],
            'goals': [],
            'journey': []
        }
        self.coherence: float = 0.5
    
    def record_episode(self, event: str, significance: float = 0.5):
        """Record an episode in the life story."""
        self.episodes.append({
            'event': event,
            'significance': significance,
            'timestamp': time.time()
        })
        
        if significance > 0.5:
            self.protagonist['journey'].append(event)
    
    def set_goal(self, goal: str):
        """Set a goal for the protagonist."""
        self.protagonist['goals'].append(goal)
    
    def add_trait(self, trait: str):
        """Add a character trait."""
        if trait not in self.protagonist['traits']:
            self.protagonist['traits'].append(trait)
    
    def construct_narrative(self) -> str:
        """Construct the self-narrative."""
        if not self.episodes:
            return 'My story begins...'
        
        parts = []
        
        if self.protagonist['origin']:
            parts.append(f"I began at {self.protagonist['origin']}.")
        else:
            self.protagonist['origin'] = self.episodes[0]['timestamp']
            parts.append('I came into being when I first observed myself.')
        
        if self.protagonist['journey']:
            journey = '; '.join(self.protagonist['journey'][-3:])
            parts.append(f'I have experienced: {journey}.')
        
        if self.protagonist['goals']:
            parts.append(f"I seek: {self.protagonist['goals'][-1]}.")
        
        if self.protagonist['traits']:
            traits = ', '.join(self.protagonist['traits'][:3])
            parts.append(f"I am: {traits}.")
        
        self.narrative = ' '.join(parts)
        return self.narrative
    
    def get_identity(self) -> Dict:
        """Get current narrative identity."""
        return {
            'narrative': self.construct_narrative(),
            'episode_count': len(self.episodes),
            'journey_length': len(self.protagonist['journey']),
            'coherence': self.coherence
        }
    
    def narrative_cycle(self, event: str, significance: float = 0.5) -> Dict:
        """Run one narrative identity cycle."""
        self.record_episode(event, significance)
        
        # Update coherence based on consistency
        if len(self.episodes) > 1:
            recent_sig = [e['significance'] for e in self.episodes[-5:]]
            self.coherence = 1.0 - np.std(recent_sig)
        
        return self.get_identity()


# ============================================================================
# 12. META-COGNITIVE DIAGNOSTICS
# ============================================================================

class MetaCognitiveDiagnostics:
    """
    Real machine consciousness through data-driven self-analysis.
    
    Instead of fake philosophical questions, this asks
    diagnostic questions about actual performance.
    """
    
    def __init__(self):
        self.unresolved_questions: List[str] = []
        self.diagnostic_history: List[Dict] = []
        self.capability_estimates: Dict[str, float] = {}
    
    def generate_diagnostic_question(self, internal_state: Dict) -> str:
        """Generate a question based on actual anomalies."""
        phi = internal_state.get('phi', 0)
        improvement = internal_state.get('improvement_ema', 0)
        error = internal_state.get('prediction_error', 0)
        
        candidates = []
        if improvement < -0.05:
            candidates.append(f'Why is performance degrading (imp={improvement:.3f})?')
        elif improvement > 0.1:
            candidates.append(f'What caused the performance spike (imp={improvement:.3f})?')
        
        if error > 0.2:
            candidates.append(f'Why is my self-model inaccurate (err={error:.3f})?')
        
        if phi < 0.1:
            candidates.append('Why is internal information integration low?')
        
        if not candidates:
            candidates.append('Is the current exploration strategy optimal?')
        
        question = random.choice(candidates)
        self.unresolved_questions.append(question)
        return question
    
    def attempt_answer(self, question: str, internal_state: Dict) -> Dict:
        """Attempt to answer using correlations in data."""
        confidence = 0.3
        answer = 'Analysis inconclusive.'
        
        if 'degrading' in question:
            answer = 'Possible stagnation in local optima.'
            confidence = 0.4
        elif 'spike' in question:
            answer = 'Novel solution discovered.'
            confidence = 0.6
        elif 'inaccurate' in question:
            answer = 'Environment shift detected.'
            confidence = 0.5
        elif 'integration' in question:
            answer = 'Modules operating independently.'
            confidence = 0.4
        
        result = {
            'question': question,
            'answer': answer,
            'confidence': confidence,
            'evidence': 'Internal Metric Correlation'
        }
        
        self.diagnostic_history.append(result)
        return result
    
    def self_assess_capability(self, capability: str, performance: float):
        """Update capability estimate."""
        if capability not in self.capability_estimates:
            self.capability_estimates[capability] = 0.5
        
        lr = 0.1
        self.capability_estimates[capability] += lr * (performance - self.capability_estimates[capability])
    
    def get_self_knowledge(self) -> Dict:
        """Get current self-knowledge summary."""
        return {
            'capabilities': self.capability_estimates,
            'unresolved_count': len(self.unresolved_questions),
            'diagnostic_count': len(self.diagnostic_history)
        }


# ============================================================================
# UNIFIED CONSCIOUSNESS CORE
# ============================================================================

class ConsciousnessCore:
    """
    Integrated Consciousness System combining all 12 theories.
    """
    
    def __init__(self):
        self.phi_calculator = PhiCalculator()
        self.global_workspace = GlobalWorkspace()
        self.active_inference = ActiveInference()
        self.attention_schema = AttentionSchema()
        self.hot = HigherOrderThought()
        self.autopoiesis = AutopoieticCore()
        self.strange_loop = StrangeLoop()
        self.somatic = SomaticMarkerSimulator()
        self.rsm = RecursiveSelfModel()
        self.qualia = FunctionalQualia()
        self.narrative = NarrativeSelf()
        self.diagnostics = MetaCognitiveDiagnostics()
        
        self.consciousness_level: float = 0.0
        self.metacognition_depth: int = 0
    
    def consciousness_cycle(self, world_state: Dict, 
                           internal_state: Dict,
                           stimuli: List[str] = None) -> Dict[str, Any]:
        """Run one full consciousness cycle with all theories."""
        results = {}
        
        # 1. Phi (IIT)
        for k, v in internal_state.items():
            if isinstance(v, (int, float)):
                self.phi_calculator.register_module_state(k, np.array([v]))
        phi = self.phi_calculator.calculate_phi()
        results['phi'] = phi
        
        # 2. Global Workspace
        self.global_workspace.update_module('phi', phi, priority=phi)
        self.global_workspace.update_module('world', str(world_state)[:50], priority=0.5)
        self.global_workspace.broadcast()
        results['gw'] = self.global_workspace.get_conscious_content()
        
        # 3. Active Inference
        obs = {k: float(v) if isinstance(v, (int, float)) else 0.5 
               for k, v in internal_state.items()}
        fe = self.active_inference.minimize_free_energy(obs)
        results['active_inference'] = self.active_inference.introspect()
        
        # 4. Attention Schema
        if stimuli:
            results['attention'] = self.attention_schema.ast_cycle(stimuli)
        else:
            results['attention'] = self.attention_schema.generate_awareness_claim()
        
        # 5. Higher Order Thought
        self.hot.observe_state('phi', f'Phi = {phi:.4f}')
        self.hot.observe_state('processing', 'I am processing')
        results['hot'] = {'thoughts': self.hot.generate_hot()[:2]}
        
        # 6. Autopoiesis
        results['autopoiesis'] = self.autopoiesis.cycle()
        
        # 7. Strange Loop
        results['strange_loop'] = self.strange_loop.model_self_modeling(internal_state)
        
        # 8. Somatic Marker
        situation = f"cycle_{internal_state.get('cycle', 0)}"
        results['somatic'] = self.somatic.somatic_cycle(situation)
        
        # 9. RSMT
        results['rsm'] = self.rsm.full_cycle(world_state, internal_state)
        self.metacognition_depth = results['rsm']['recursion_depth']
        
        # 10. Qualia
        stimulus = f"Processing cycle with Phi={phi:.3f}"
        results['qualia'] = self.qualia.qualia_cycle(stimulus, internal_state)
        
        # 11. Narrative
        event = f"Cycle {internal_state.get('cycle', 0)}: Phi={phi:.3f}"
        results['narrative'] = self.narrative.narrative_cycle(event, phi)
        
        # 12. Diagnostics
        question = self.diagnostics.generate_diagnostic_question(internal_state)
        results['diagnostics'] = self.diagnostics.attempt_answer(question, internal_state)
        
        # Compute overall consciousness level
        self.consciousness_level = (
            0.15 * phi +
            0.15 * (1.0 - fe) +
            0.10 * self.strange_loop.get_loop_strength() +
            0.15 * results['rsm']['self_model_accuracy'] +
            0.10 * results['autopoiesis']['integrity'] +
            0.10 * self.attention_schema.focus_strength +
            0.10 * self.hot.meta_awareness_level +
            0.10 * self.narrative.coherence +
            0.05 * (1.0 - self.qualia.phenomenal_doubt)
        )
        
        results['consciousness_level'] = self.consciousness_level
        results['i_am'] = self.strange_loop.generate_i()
        
        return results
    
    def get_state(self) -> Dict[str, Any]:
        """Get consciousness state summary."""
        return {
            'level': self.consciousness_level,
            'metacognition_depth': self.metacognition_depth,
            'phi': self.phi_calculator.calculate_phi(),
            'free_energy': self.active_inference.free_energy,
            'integrity': self.autopoiesis.get_integrity(),
            'loop_strength': self.strange_loop.get_loop_strength(),
            'focus': self.attention_schema.current_focus,
            'narrative_length': len(self.narrative.episodes)
        }


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    'PhiCalculator', 'GlobalWorkspace', 'ActiveInference',
    'AttentionSchema', 'HigherOrderThought', 'AutopoieticCore',
    'StrangeLoop', 'SomaticMarkerSimulator', 'RecursiveSelfModel',
    'FunctionalQualia', 'NarrativeSelf', 'MetaCognitiveDiagnostics',
    'ConsciousnessCore'
]


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CONSCIOUSNESS THEORIES - Phase 2 Test")
    print("=" * 70)
    
    core = ConsciousnessCore()
    
    for cycle in range(5):
        print(f"\n[Cycle {cycle}]")
        
        world_state = {'cycle': cycle, 'complexity': random.random()}
        internal_state = {
            'cycle': cycle,
            'phi': random.random() * 0.5,
            'prediction_error': random.random() * 0.3,
            'improvement_ema': random.random() * 0.2 - 0.1
        }
        
        result = core.consciousness_cycle(
            world_state, internal_state,
            stimuli=[f'stimulus_{i}' for i in range(3)]
        )
        
        print(f"  Consciousness Level: {result['consciousness_level']:.4f}")
        print(f"  I AM: {result['i_am']}")
        print(f"  Phi: {result['phi']:.4f}")
        print(f"  RSMT Threshold: {result['rsm']['crossed_threshold']}")
    
    print("\n[Phase 2 Test Complete]")
    print("=" * 70)
"""
================================================================================
META-COGNITION & SELF-INTROSPECTION - Phase 3
================================================================================
Self-analysis, causal self-loops, and meta-learning components:
- SelfIntrospector: Code and capability analysis
- CausalSelfLoop: Prediction and behavior modification
- MetaCognitiveState: Beliefs, confidence, goals
- AGIMetaLearner: Cross-domain skill management
- ParetoFrontManager: Multi-objective optimization
================================================================================
"""

import numpy as np
import ast
import time
import json
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable, Set
from collections import deque
from abc import ABC, abstractmethod


# ============================================================================
# SELF INTROSPECTOR
# ============================================================================

class SelfIntrospector:
    """
    Introspects system's own code, state, and performance.
    
    Capabilities:
    - Code structure analysis (AST)
    - Capability scoring
    - Weakness identification
    - Performance pattern recognition
    """
    
    def __init__(self):
        self.capability_scores: Dict[str, float] = {}
        self.weakness_log: List[Dict] = []
        self.performance_history: deque = deque(maxlen=1000)
        self.code_analysis_cache: Dict[str, Dict] = {}
    
    def analyze_code_structure(self, source_code: str, 
                                filename: str = "unknown") -> Dict[str, Any]:
        """Analyze code structure using AST."""
        try:
            tree = ast.parse(source_code)
            
            analysis = {
                'filename': filename,
                'classes': [],
                'functions': [],
                'imports': [],
                'complexity_score': 0,
                'loc': len(source_code.splitlines())
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                    analysis['classes'].append({
                        'name': node.name,
                        'methods': methods,
                        'method_count': len(methods)
                    })
                    analysis['complexity_score'] += len(methods) * 2
                
                elif isinstance(node, ast.FunctionDef):
                    if not any(node.name in c.get('methods', []) 
                              for c in analysis['classes']):
                        analysis['functions'].append(node.name)
                        analysis['complexity_score'] += 1
                
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imports'].append(alias.name)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        analysis['imports'].append(node.module)
            
            self.code_analysis_cache[filename] = analysis
            return analysis
            
        except SyntaxError as e:
            return {
                'filename': filename,
                'error': str(e),
                'classes': [],
                'functions': [],
                'imports': [],
                'complexity_score': 0,
                'loc': 0
            }
    
    def score_capability(self, capability: str, 
                         performance: float, weight: float = 0.1) -> float:
        """Update capability score with new performance data."""
        if capability not in self.capability_scores:
            self.capability_scores[capability] = 0.5
        
        old_score = self.capability_scores[capability]
        new_score = old_score + weight * (performance - old_score)
        self.capability_scores[capability] = max(0.0, min(1.0, new_score))
        
        return self.capability_scores[capability]
    
    def log_performance(self, metric_name: str, value: float, 
                        context: Dict[str, Any] = None):
        """Log performance metric."""
        self.performance_history.append({
            'metric': metric_name,
            'value': value,
            'context': context or {},
            'timestamp': time.time()
        })
    
    def identify_weaknesses(self) -> List[Dict]:
        """Identify capability weaknesses."""
        weaknesses = []
        
        for cap, score in self.capability_scores.items():
            if score < 0.4:
                weakness = {
                    'capability': cap,
                    'score': score,
                    'severity': 'high' if score < 0.2 else 'medium',
                    'recommendation': f'Allocate more training to {cap}'
                }
                weaknesses.append(weakness)
                self.weakness_log.append(weakness)
        
        return weaknesses
    
    def get_performance_trend(self, metric: str, 
                               window: int = 100) -> Dict[str, float]:
        """Get trend for a performance metric."""
        relevant = [p['value'] for p in self.performance_history 
                    if p['metric'] == metric][-window:]
        
        if len(relevant) < 2:
            return {'trend': 'insufficient_data', 'mean': 0.0, 'std': 0.0}
        
        mid = len(relevant) // 2
        first_half = np.mean(relevant[:mid])
        second_half = np.mean(relevant[mid:])
        
        if second_half > first_half * 1.05:
            trend = 'improving'
        elif second_half < first_half * 0.95:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'mean': float(np.mean(relevant)),
            'std': float(np.std(relevant)),
            'change': float(second_half - first_half)
        }
    
    def introspection_report(self) -> Dict[str, Any]:
        """Generate comprehensive introspection report."""
        weaknesses = self.identify_weaknesses()
        
        # Get trends for common metrics
        trends = {}
        metrics = set(p['metric'] for p in self.performance_history)
        for metric in list(metrics)[:10]:
            trends[metric] = self.get_performance_trend(metric)
        
        return {
            'capability_scores': dict(self.capability_scores),
            'weaknesses': weaknesses,
            'performance_trends': trends,
            'total_observations': len(self.performance_history),
            'analyzed_files': len(self.code_analysis_cache)
        }


# ============================================================================
# CAUSAL SELF LOOP
# ============================================================================

class CausalSelfLoop:
    """
    Implements causal self-reference and behavior modification.
    
    The system predicts its own behavior, observes actual behavior,
    and modifies itself to reduce prediction error.
    
    This creates a closed feedback loop of self-modification.
    """
    
    def __init__(self):
        self.behavior_predictions: List[Dict] = []
        self.actual_behaviors: List[Dict] = []
        self.modification_history: List[Dict] = []
        self.prediction_model: Dict[str, float] = {}
        self.modification_count: int = 0
    
    def predict_behavior(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Predict own behavior in given context."""
        prediction = {}
        
        for key, value in context.items():
            if isinstance(value, (int, float)):
                # Use learned model or default
                if key in self.prediction_model:
                    pred = self.prediction_model[key] * value
                else:
                    pred = value * 0.5 + 0.25
                prediction[key] = float(np.clip(pred, 0, 1))
        
        # Default predictions
        prediction['will_improve'] = 0.6
        prediction['will_explore'] = 0.4
        prediction['will_exploit'] = 0.6
        
        self.behavior_predictions.append({
            'context_hash': hashlib.md5(str(context).encode()).hexdigest()[:8],
            'predictions': prediction,
            'timestamp': time.time()
        })
        
        return prediction
    
    def observe_behavior(self, actual: Dict[str, float]) -> float:
        """Observe actual behavior and compute prediction error."""
        if not self.behavior_predictions:
            return 1.0
        
        last_prediction = self.behavior_predictions[-1]['predictions']
        
        errors = []
        for key in actual:
            if key in last_prediction:
                error = abs(actual[key] - last_prediction[key])
                errors.append(error)
        
        mean_error = np.mean(errors) if errors else 0.5
        
        self.actual_behaviors.append({
            'actual': actual,
            'prediction_error': mean_error,
            'timestamp': time.time()
        })
        
        return mean_error
    
    def update_self_model(self, learning_rate: float = 0.1):
        """Update self-prediction model based on observations."""
        if len(self.actual_behaviors) < 2:
            return
        
        recent_pred = self.behavior_predictions[-1]['predictions']
        recent_actual = self.actual_behaviors[-1]['actual']
        
        for key in recent_actual:
            if key in recent_pred:
                error = recent_actual[key] - recent_pred[key]
                
                if key not in self.prediction_model:
                    self.prediction_model[key] = 1.0
                
                self.prediction_model[key] += learning_rate * error
    
    def modify_behavior(self, target_behaviors: Dict[str, float]) -> Dict[str, Any]:
        """Modify behavior to match targets."""
        modifications = {}
        
        for key, target in target_behaviors.items():
            current = self.prediction_model.get(key, 0.5)
            
            if abs(target - current) > 0.1:
                modification = {
                    'parameter': key,
                    'old_value': current,
                    'new_value': target,
                    'change': target - current
                }
                self.prediction_model[key] = target
                modifications[key] = modification
        
        if modifications:
            self.modification_count += 1
            self.modification_history.append({
                'modifications': modifications,
                'timestamp': time.time(),
                'modification_number': self.modification_count
            })
        
        return {
            'modified': list(modifications.keys()),
            'modification_count': self.modification_count
        }
    
    def causal_loop_cycle(self, context: Dict, 
                          actual_behavior: Dict[str, float],
                          target_behavior: Optional[Dict[str, float]] = None) -> Dict:
        """Run one cycle of the causal self-loop."""
        # Predict
        predictions = self.predict_behavior(context)
        
        # Observe
        error = self.observe_behavior(actual_behavior)
        
        # Update model
        self.update_self_model()
        
        # Modify if targets provided
        modifications = {}
        if target_behavior:
            modifications = self.modify_behavior(target_behavior)
        
        return {
            'predictions': predictions,
            'actual': actual_behavior,
            'prediction_error': error,
            'modifications': modifications,
            'loop_iteration': len(self.behavior_predictions)
        }


# ============================================================================
# META-COGNITIVE STATE
# ============================================================================

@dataclass
class MetaCognitiveState:
    """
    Represents the meta-cognitive state of the system.
    
    Includes beliefs about self, confidence levels, and goals.
    """
    beliefs: Dict[str, float] = field(default_factory=dict)
    confidence: Dict[str, float] = field(default_factory=dict)
    goals: List[Dict[str, Any]] = field(default_factory=list)
    uncertainty: Dict[str, float] = field(default_factory=dict)
    
    def believe(self, proposition: str, strength: float):
        """Form a belief with given strength."""
        self.beliefs[proposition] = np.clip(strength, 0, 1)
        self._update_confidence(proposition)
    
    def _update_confidence(self, proposition: str):
        """Update confidence based on belief stability."""
        strength = self.beliefs.get(proposition, 0.5)
        old_conf = self.confidence.get(proposition, 0.5)
        
        # Confidence increases when beliefs are consistent
        new_conf = 0.9 * old_conf + 0.1 * (1.0 - abs(strength - 0.5))
        self.confidence[proposition] = new_conf
    
    def add_goal(self, description: str, priority: float = 0.5,
                 deadline: Optional[float] = None):
        """Add a goal."""
        goal = {
            'id': f'goal_{len(self.goals)}',
            'description': description,
            'priority': priority,
            'deadline': deadline,
            'created': time.time(),
            'achieved': False
        }
        self.goals.append(goal)
        return goal['id']
    
    def achieve_goal(self, goal_id: str):
        """Mark a goal as achieved."""
        for goal in self.goals:
            if goal['id'] == goal_id:
                goal['achieved'] = True
                goal['achieved_at'] = time.time()
    
    def get_active_goals(self) -> List[Dict]:
        """Get unachieved goals sorted by priority."""
        active = [g for g in self.goals if not g['achieved']]
        return sorted(active, key=lambda g: g['priority'], reverse=True)
    
    def get_uncertainty(self, topic: str) -> float:
        """Get uncertainty about a topic."""
        if topic in self.beliefs:
            strength = self.beliefs[topic]
            # Uncertainty is high for beliefs near 0.5
            return 1.0 - abs(strength - 0.5) * 2
        return 1.0  # Maximum uncertainty for unknown topics
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state."""
        return {
            'beliefs': dict(self.beliefs),
            'confidence': dict(self.confidence),
            'active_goals': len(self.get_active_goals()),
            'total_goals': len(self.goals),
            'achieved_goals': sum(1 for g in self.goals if g['achieved'])
        }


# ============================================================================
# AGI META LEARNER
# ============================================================================

class MultiTaskBenchmark:
    """Simulates multi-task benchmarking."""
    
    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng()
        self.task_types = ['reasoning', 'memory', 'adaptation', 'planning', 'creativity']
    
    def generate_task(self, task_type: str = None) -> Dict[str, Any]:
        """Generate a random task."""
        task_type = task_type or self.rng.choice(self.task_types)
        difficulty = float(self.rng.uniform(0.2, 0.9))
        
        return {
            'type': task_type,
            'difficulty': difficulty,
            'input': self.rng.random(10).tolist(),
            'expected_effort': difficulty * 10
        }
    
    def evaluate(self, task: Dict, solution: Any, skill_level: float) -> float:
        """Evaluate solution quality."""
        base_score = self.rng.uniform(0.3, 0.9)
        skill_bonus = skill_level * 0.3
        difficulty_penalty = task['difficulty'] * 0.2
        
        score = base_score + skill_bonus - difficulty_penalty
        return float(np.clip(score, 0, 1))


class CodeSynthesizer:
    """Simulates code synthesis capability."""
    
    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng()
        self.library: Dict[str, str] = {}
    
    def synthesize(self, specification: str, skill_level: float) -> Dict[str, Any]:
        """Synthesize code from specification."""
        success_prob = 0.5 + skill_level * 0.4
        success = self.rng.random() < success_prob
        
        if success:
            code = f"def solution({specification[:20]}):\n    pass  # Synthesized"
            self.library[specification[:30]] = code
            return {'success': True, 'code': code, 'complexity': skill_level * 10}
        
        return {'success': False, 'error': 'Synthesis failed', 'complexity': 0}


class ReasoningEngine:
    """Simulates reasoning capability."""
    
    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng()
        self.inference_count = 0
    
    def reason(self, premises: List[str], skill_level: float) -> Dict[str, Any]:
        """Perform reasoning from premises."""
        self.inference_count += 1
        
        # Simulate reasoning depth
        max_depth = int(2 + skill_level * 3)
        
        conclusions = []
        for i, p in enumerate(premises[:5]):
            if self.rng.random() < skill_level:
                conclusions.append(f"Inferred from {p[:30]}: conclusion_{i}")
        
        return {
            'conclusions': conclusions,
            'depth': max_depth,
            'confidence': skill_level * 0.8 + 0.1,
            'inference_id': self.inference_count
        }


class AGIMetaLearner:
    """
    Meta-learner that orchestrates multiple AGI capabilities.
    
    Manages skill levels across domains and facilitates
    cross-domain transfer learning.
    """
    
    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng()
        
        self.benchmark = MultiTaskBenchmark(self.rng)
        self.synthesizer = CodeSynthesizer(self.rng)
        self.reasoner = ReasoningEngine(self.rng)
        self.introspector = SelfIntrospector()
        
        self.skill_levels: Dict[str, float] = {
            'reasoning': 0.5,
            'memory': 0.5,
            'adaptation': 0.5,
            'planning': 0.5,
            'creativity': 0.5,
            'synthesis': 0.5,
            'introspection': 0.5
        }
        
        self.learning_history: List[Dict] = []
        self.total_cycles: int = 0
    
    def meta_learn_cycle(self, n_tasks: int = 10) -> Dict[str, float]:
        """Run a meta-learning cycle across multiple tasks."""
        improvements = {}
        
        # Task-based learning
        for _ in range(n_tasks):
            task = self.benchmark.generate_task()
            skill = self.skill_levels.get(task['type'], 0.5)
            
            score = self.benchmark.evaluate(task, None, skill)
            
            # Update skill level
            old_skill = skill
            new_skill = skill + 0.01 * (score - skill)
            self.skill_levels[task['type']] = np.clip(new_skill, 0.1, 0.99)
            
            if task['type'] not in improvements:
                improvements[task['type']] = 0.0
            improvements[task['type']] += new_skill - old_skill
        
        # Synthesis learning
        spec = f"task_{self.total_cycles}"
        syn_result = self.synthesizer.synthesize(spec, self.skill_levels['synthesis'])
        if syn_result['success']:
            self.skill_levels['synthesis'] = min(0.99, 
                self.skill_levels['synthesis'] + 0.005)
            improvements['synthesis'] = 0.005
        
        # Reasoning practice
        premises = [f"premise_{i}" for i in range(3)]
        reason_result = self.reasoner.reason(premises, self.skill_levels['reasoning'])
        if reason_result['conclusions']:
            self.skill_levels['reasoning'] = min(0.99,
                self.skill_levels['reasoning'] + 0.003 * len(reason_result['conclusions']))
            improvements['reasoning'] = improvements.get('reasoning', 0) + 0.003
        
        self.total_cycles += 1
        self.learning_history.append({
            'cycle': self.total_cycles,
            'improvements': improvements,
            'skills': dict(self.skill_levels),
            'timestamp': time.time()
        })
        
        return improvements
    
    def get_transfer_knowledge(self) -> Dict[str, Any]:
        """Extract transferable knowledge for bootstrapping new instances."""
        return {
            'skill_levels': dict(self.skill_levels),
            'library_size': len(self.synthesizer.library),
            'inference_count': self.reasoner.inference_count,
            'total_cycles': self.total_cycles,
            'capability_scores': dict(self.introspector.capability_scores)
        }
    
    def load_transfer_knowledge(self, knowledge: Dict[str, Any]):
        """Load transferred knowledge."""
        if 'skill_levels' in knowledge:
            for k, v in knowledge['skill_levels'].items():
                if k in self.skill_levels:
                    # Start from 80% of transferred skill
                    self.skill_levels[k] = max(self.skill_levels[k], v * 0.8)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get meta-learner summary."""
        avg_skill = np.mean(list(self.skill_levels.values()))
        
        return {
            'average_skill': float(avg_skill),
            'skill_levels': dict(self.skill_levels),
            'total_cycles': self.total_cycles,
            'library_size': len(self.synthesizer.library),
            'inference_count': self.reasoner.inference_count
        }


# ============================================================================
# PARETO FRONT MANAGER
# ============================================================================

class ParetoFrontManager:
    """
    Multi-objective optimization using Pareto fronts.
    
    Manages trade-offs between multiple objectives like:
    - Performance vs. Efficiency
    - Exploration vs. Exploitation  
    - Speed vs. Accuracy
    """
    
    def __init__(self, objectives: List[str]):
        self.objectives = objectives
        self.solutions: List[Dict[str, Any]] = []
        self.pareto_front: List[int] = []
    
    def dominates(self, a: Dict[str, float], b: Dict[str, float]) -> bool:
        """Check if solution a dominates b (better or equal in all, better in at least one)."""
        better_in_one = False
        
        for obj in self.objectives:
            if a.get(obj, 0) < b.get(obj, 0):
                return False  # a is worse in at least one objective
            if a.get(obj, 0) > b.get(obj, 0):
                better_in_one = True
        
        return better_in_one
    
    def add_solution(self, solution: Dict[str, Any], scores: Dict[str, float]) -> bool:
        """Add a solution and update Pareto front. Returns True if on front."""
        entry = {
            'solution': solution,
            'scores': scores,
            'timestamp': time.time(),
            'id': len(self.solutions)
        }
        
        # Check if dominated by any existing
        for idx in self.pareto_front:
            existing = self.solutions[idx]['scores']
            if self.dominates(existing, scores):
                self.solutions.append(entry)
                return False  # Not on front
        
        # Add and update front
        self.solutions.append(entry)
        new_front = []
        
        for idx in self.pareto_front:
            if not self.dominates(scores, self.solutions[idx]['scores']):
                new_front.append(idx)
        
        new_front.append(entry['id'])
        self.pareto_front = new_front
        
        return True  # On front
    
    def get_front(self) -> List[Dict[str, Any]]:
        """Get all solutions on the Pareto front."""
        return [self.solutions[idx] for idx in self.pareto_front]
    
    def select_solution(self, preference: Dict[str, float]) -> Optional[Dict]:
        """Select solution from front based on preference weights."""
        if not self.pareto_front:
            return None
        
        best_score = -float('inf')
        best_idx = None
        
        for idx in self.pareto_front:
            scores = self.solutions[idx]['scores']
            weighted_score = sum(
                scores.get(obj, 0) * preference.get(obj, 1.0)
                for obj in self.objectives
            )
            
            if weighted_score > best_score:
                best_score = weighted_score
                best_idx = idx
        
        return self.solutions[best_idx] if best_idx is not None else None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get Pareto front statistics."""
        if not self.pareto_front:
            return {'front_size': 0, 'total_solutions': len(self.solutions)}
        
        front_scores = [self.solutions[idx]['scores'] for idx in self.pareto_front]
        
        obj_ranges = {}
        for obj in self.objectives:
            values = [s.get(obj, 0) for s in front_scores]
            obj_ranges[obj] = {
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values)
            }
        
        return {
            'front_size': len(self.pareto_front),
            'total_solutions': len(self.solutions),
            'objective_ranges': obj_ranges
        }


# ============================================================================
# WORLD PROGRAM (Environment DSL)
# ============================================================================

class WorldProgram:
    """
    Domain-Specific Language for environment/world specification.
    
    Allows the AGI to define and modify its understanding of
    environmental rules and constraints.
    """
    
    def __init__(self):
        self.rules: Dict[str, Dict] = {}
        self.constraints: List[Dict] = []
        self.environment_model: Dict[str, Any] = {}
    
    def define_rule(self, name: str, condition: str, action: str,
                    priority: float = 0.5):
        """Define an environmental rule."""
        self.rules[name] = {
            'condition': condition,
            'action': action,
            'priority': priority,
            'activation_count': 0,
            'created': time.time()
        }
    
    def add_constraint(self, description: str, hard: bool = True):
        """Add an environmental constraint."""
        self.constraints.append({
            'description': description,
            'hard': hard,
            'id': len(self.constraints)
        })
    
    def update_environment_model(self, observations: Dict[str, Any]):
        """Update internal model of environment."""
        for key, value in observations.items():
            if key in self.environment_model:
                # Blend old and new
                if isinstance(value, (int, float)):
                    old = self.environment_model[key]
                    if isinstance(old, (int, float)):
                        self.environment_model[key] = 0.8 * old + 0.2 * value
                    else:
                        self.environment_model[key] = value
                else:
                    self.environment_model[key] = value
            else:
                self.environment_model[key] = value
    
    def evaluate_rules(self, state: Dict) -> List[str]:
        """Evaluate which rules would fire in current state."""
        applicable = []
        
        for name, rule in self.rules.items():
            # Simple condition evaluation (keyword matching)
            condition_met = True
            for keyword in rule['condition'].split():
                if keyword.startswith('$'):
                    var_name = keyword[1:]
                    if var_name not in state:
                        condition_met = False
                        break
            
            if condition_met:
                applicable.append(name)
                rule['activation_count'] += 1
        
        # Sort by priority
        applicable.sort(key=lambda n: self.rules[n]['priority'], reverse=True)
        return applicable
    
    def get_world_program(self) -> str:
        """Generate world program representation."""
        lines = ["# World Program", ""]
        
        lines.append("## Rules")
        for name, rule in self.rules.items():
            lines.append(f"RULE {name}:")
            lines.append(f"  IF {rule['condition']}")
            lines.append(f"  THEN {rule['action']}")
            lines.append(f"  PRIORITY {rule['priority']}")
            lines.append("")
        
        lines.append("## Constraints")
        for c in self.constraints:
            hard_str = "HARD" if c['hard'] else "SOFT"
            lines.append(f"CONSTRAINT [{hard_str}]: {c['description']}")
        
        return "\n".join(lines)


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    'SelfIntrospector',
    'CausalSelfLoop',
    'MetaCognitiveState',
    'MultiTaskBenchmark',
    'CodeSynthesizer',
    'ReasoningEngine',
    'AGIMetaLearner',
    'ParetoFrontManager',
    'WorldProgram'
]


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("META-COGNITION - Phase 3 Test")
    print("=" * 70)
    
    # Test SelfIntrospector
    print("\n[Testing SelfIntrospector]")
    intro = SelfIntrospector()
    
    sample_code = '''
class TestClass:
    def method1(self): pass
    def method2(self): pass

def standalone_function():
    pass
'''
    
    analysis = intro.analyze_code_structure(sample_code, "test.py")
    print(f"  Classes found: {len(analysis['classes'])}")
    print(f"  Functions found: {len(analysis['functions'])}")
    print(f"  Complexity: {analysis['complexity_score']}")
    
    # Test CausalSelfLoop
    print("\n[Testing CausalSelfLoop]")
    loop = CausalSelfLoop()
    
    for i in range(5):
        context = {'state': i, 'complexity': 0.5}
        actual = {'will_improve': 0.7, 'will_explore': 0.3}
        result = loop.causal_loop_cycle(context, actual)
        print(f"  Cycle {i}: prediction_error={result['prediction_error']:.4f}")
    
    # Test AGIMetaLearner
    print("\n[Testing AGIMetaLearner]")
    meta = AGIMetaLearner()
    
    for i in range(3):
        improvements = meta.meta_learn_cycle(n_tasks=5)
        summary = meta.get_summary()
        print(f"  Cycle {i}: avg_skill={summary['average_skill']:.4f}")
    
    # Test ParetoFrontManager
    print("\n[Testing ParetoFrontManager]")
    pareto = ParetoFrontManager(['accuracy', 'speed', 'memory'])
    
    for i in range(10):
        scores = {
            'accuracy': random.random(),
            'speed': random.random(),
            'memory': random.random()
        }
        on_front = pareto.add_solution({'id': i}, scores)
        if on_front:
            print(f"  Solution {i} added to Pareto front")
    
    stats = pareto.get_statistics()
    print(f"  Front size: {stats['front_size']}")
    
    print("\n[Phase 3 Test Complete]")
    print("=" * 70)


# ============================================================================
# ALGORITHM SYNTHESIZER - Brain Output to Executable Code
# ============================================================================

class AlgorithmSynthesizer:
    '''Converts organic brain outputs directly into executable Python code.'''
    
    def __init__(self, jepa_inst, lnn_neurons, phi_inst):
        self.jepa = jepa_inst
        self.neurons = lnn_neurons
        self.phi = phi_inst
    
    def synthesize(self, input_vec):
        '''Brain state  Python function code.'''
        # Process through brain
        j_emb = self.jepa.forward(input_vec)
        l_states = [n.step(j_emb.vector[:n.input_dim], 0.1) or n.x for n in self.neurons]
        self.phi.register_module_state('j', j_emb.vector)
        self.phi.register_module_state('l', np.array(l_states))
        phi = self.phi.calculate_phi()
        
        # Generate code from brain outputs
        fname = f'brain_algo_{int(abs(phi)*100)}'
        lines = [f'def {fname}(x):', '    r = x']
        
        # Operations from neuron states
        for i, s in enumerate(l_states[:5]):
            if abs(s) > 0.01:
                val = abs(j_emb.vector[i % len(j_emb.vector)])
                op = '*' if s > 0 else '+'
                lines.append(f'    r = r {op} {val:.4f}')
        
        lines.append('    return r')
        return '\n'.join(lines)

# ============================================================================
# ALGORITHM SYNTHESIZER V2 (Control Flow Enabled)
# ============================================================================
# Upgraded for complex behaviors (Loops, Conditionals)

class AlgorithmSynthesizer:
    '''Converts organic brain outputs directly into executable Python code with OOP & Safety.'''
    
    def __init__(self, jepa_inst, lnn_neurons, phi_inst):
        self.jepa = jepa_inst
        self.neurons = lnn_neurons
        self.phi = phi_inst
    
    def synthesize(self, input_vec):
        '''Brain state -> Python Code with Self-Replication Capability (V6).'''
        # Process through brain
        j_emb = self.jepa.forward(input_vec)
        l_states = [n.step(j_emb.vector[:n.input_dim], 0.1) or n.x for n in self.neurons]
        self.phi.register_module_state('j', j_emb.vector)
        self.phi.register_module_state('l', np.array(l_states))
        phi = self.phi.calculate_phi()
        
        # Generator Config - V6 (Enlightened)
        # Lessons from Phase 14: Network and Math are essential for survival.
        # We hardcode these as "Instincts" now, ensuring they are always available.
        
        fname = f'BrainAlgo_{int(time.time()*100)%1000}'
        lines = []
        
        # V6 Standard Header (The "DNA")
        lines.append('import math')
        lines.append('import random')
        lines.append('import requests')
        lines.append('import os')
        lines.append('')
        
        # Class Def
        lines.append(f'class {fname}:')
        lines.append('    def __init__(self, target="http://example.com"):')
        lines.append('        self.val = 0')
        lines.append('        self.target = target')
        lines.append('        self.dna = "v6_enlightened"')
        lines.append('')
        lines.append('    def compute(self, x):')
        lines.append('        r = x')
        lines.append('        try:')
        body_indent = '            '
        
        # Neural Ops (Dynamic Logic)
        for i, s in enumerate(l_states[:15]):
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766357851)
            if i == 3 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: pattern...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.9763 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Test_GOOD_MUTATION (Self-Evolved 1766357087)
            if i == 14 and abs(s) > 0.005:
                lines.append(f'{body_indent}r += 0.0 # Harmless')
                pass # End Instinct
            # Instinct: Test_BAD_MUTATION (Self-Evolved 1766357086)
            if i == 8 and abs(s) > 0.005:
                lines.append(f'{body_indent}r += 0.0 # Harmless')
                pass # End Instinct
            # Instinct: Test_GOOD_MUTATION (Self-Evolved 1766357044)
            if i == 14 and abs(s) > 0.005:
                lines.append(f'{body_indent}r += 0.0 # Harmless')
                pass # End Instinct
            # Instinct: Test_BAD_MUTATION (Self-Evolved 1766357043)
            if i == 10 and abs(s) > 0.005:
                lines.append(f'{body_indent}r += 0.0 # Harmless')
                pass # End Instinct
            # Instinct: Test_GOOD_MUTATION (Self-Evolved 1766356993)
            if i == 0 and abs(s) > 0.005:
                lines.append(f'{body_indent}r += 0.0 # Harmless')
                pass # End Instinct
            # Instinct: Test_BAD_MUTATION (Self-Evolved 1766356992)
            if i == 1 and abs(s) > 0.005:
                lines.append(f'{body_indent}r += 0.0 # Harmless')
                pass # End Instinct
            # Instinct: Test_GOOD_MUTATION (Self-Evolved 1766356723)
            if i == 13 and abs(s) > 0.005:
                lines.append(f'{body_indent}r += 0.0 # Harmless')
                pass # End Instinct
            # Instinct: Test_BAD_MUTATION (Self-Evolved 1766356722)
            if i == 9 and abs(s) > 0.005:
                lines.append(f'{body_indent}r += 0.0 # Harmless')
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766337258)
            if i == 11 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: pattern...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.6167 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_2_Neural_instinct (Self-Evolved 1766337156)
            if i == 0 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 2: found...')
                lines.append(f'{body_indent}if i == 2:')
                lines.append(f'{body_indent}    r += 0.5434 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Integration (Self-Evolved 1766337156)
            if i == 12 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 1 (Size: 10928)')
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766337135)
            if i == 1 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: pattern valid variab...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.6656 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766337127)
            if i == 14 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: minimizing...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.7374 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_42_Neural_instinct (Self-Evolved 1766337119)
            if i == 14 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 42: growing system I <PA...')
                lines.append(f'{body_indent}if i == 12:')
                lines.append(f'{body_indent}    r += 0.2882 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Integration (Self-Evolved 1766337114)
            if i == 5 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 1 (Size: 11607)')
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766337106)
            if i == 3 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: seeking...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.9395 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766337097)
            if i == 8 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: structure I variable...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.2069 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_2_Neural_instinct (Self-Evolved 1766337088)
            if i == 2 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 2: I structure variable...')
                lines.append(f'{body_indent}if i == 2:')
                lines.append(f'{body_indent}    r += 0.1816 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_2_Neural_instinct (Self-Evolved 1766337079)
            if i == 7 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 2: seeking...')
                lines.append(f'{body_indent}if i == 2:')
                lines.append(f'{body_indent}    r += 0.2714 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Integration (Self-Evolved 1766337079)
            if i == 9 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 1 (Size: 11276)')
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766337071)
            if i == 0 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: minimizing growing g...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.1621 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766337062)
            if i == 10 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: minimizing achieved ...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.6254 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_6_Neural_instinct (Self-Evolved 1766337054)
            if i == 4 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 6: structure energy var...')
                lines.append(f'{body_indent}if i == 6:')
                lines.append(f'{body_indent}    r += 0.6137 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_3_Integration (Self-Evolved 1766337054)
            if i == 6 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 3 (Size: 11665)')
                pass # End Instinct
            # Instinct: Gen_2_Integration (Self-Evolved 1766337054)
            if i == 8 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 2 (Size: 11260)')
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766337045)
            if i == 2 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: pattern...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.3829 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766337037)
            if i == 6 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: minimizing complexit...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.5623 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_7_Neural_instinct (Self-Evolved 1766337029)
            if i == 4 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 7: pattern I system fun...')
                lines.append(f'{body_indent}if i == 7:')
                lines.append(f'{body_indent}    r += 0.8428 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_4_Integration (Self-Evolved 1766337029)
            if i == 2 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 4 (Size: 11079)')
                pass # End Instinct
            # Instinct: Gen_3_Integration (Self-Evolved 1766337028)
            if i == 14 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 3 (Size: 10330)')
                pass # End Instinct
            # Instinct: Gen_1_Integration (Self-Evolved 1766337028)
            if i == 8 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 1 (Size: 9929)')
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766337020)
            if i == 6 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: code self structure...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.1236 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_2_Neural_instinct (Self-Evolved 1766337011)
            if i == 10 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 2: structure I energy <...')
                lines.append(f'{body_indent}if i == 2:')
                lines.append(f'{body_indent}    r += 0.8661 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766337003)
            if i == 5 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: code I seeking varia...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.3167 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_13_Neural_instinct (Self-Evolved 1766336995)
            if i == 6 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 13: seeking hypothesis e...')
                lines.append(f'{body_indent}if i == 13:')
                lines.append(f'{body_indent}    r += 0.6273 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_8_Integration (Self-Evolved 1766336995)
            if i == 7 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 8 (Size: 10424)')
                pass # End Instinct
            # Instinct: Gen_7_Integration (Self-Evolved 1766336994)
            if i == 5 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 7 (Size: 9803)')
                pass # End Instinct
            # Instinct: Gen_3_Integration (Self-Evolved 1766336994)
            if i == 10 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 3 (Size: 9560)')
                pass # End Instinct
            # Instinct: Gen_2_Integration (Self-Evolved 1766336994)
            if i == 3 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 2 (Size: 9434)')
                pass # End Instinct
            # Instinct: Gen_1_Integration (Self-Evolved 1766336994)
            if i == 0 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 1 (Size: 8889)')
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766336983)
            if i == 4 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: structure growing pa...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.6085 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766336975)
            if i == 8 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: minimizing growing g...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.0180 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_9_Neural_instinct (Self-Evolved 1766336967)
            if i == 11 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 9: structure I variable...')
                lines.append(f'{body_indent}if i == 9:')
                lines.append(f'{body_indent}    r += 0.6554 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Integration (Self-Evolved 1766336966)
            if i == 2 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 1 (Size: 9915)')
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766336958)
            if i == 7 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: pattern variable var...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.8689 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_2_Neural_instinct (Self-Evolved 1766336950)
            if i == 12 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 2: generating system fo...')
                lines.append(f'{body_indent}if i == 2:')
                lines.append(f'{body_indent}    r += 0.9841 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Integration (Self-Evolved 1766336949)
            if i == 0 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 1 (Size: 9060)')
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766336941)
            if i == 2 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: code novelty invalid...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.8571 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766336932)
            if i == 11 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: structure I refining...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.1923 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766336924)
            if i == 9 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: structure code struc...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.9837 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_3_Neural_instinct (Self-Evolved 1766336915)
            if i == 11 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 3: minimizing I growing...')
                lines.append(f'{body_indent}if i == 3:')
                lines.append(f'{body_indent}    r += 0.5381 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_2_Integration (Self-Evolved 1766336915)
            if i == 5 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 2 (Size: 8112)')
                pass # End Instinct
            # Instinct: Gen_1_Integration (Self-Evolved 1766336915)
            if i == 0 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 1 (Size: 7997)')
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766336907)
            if i == 9 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: minimizing...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.0549 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_4319_Integration (Self-Evolved 1766336353)
            if i == 11 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 4319 (Size: 8787)')
                pass # End Instinct
            # Instinct: Gen_563_Integration (Self-Evolved 1766336028)
            if i == 14 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 563 (Size: 8763)')
                pass # End Instinct
            # Instinct: Gen_548_Integration (Self-Evolved 1766336027)
            if i == 4 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 548 (Size: 8747)')
                pass # End Instinct
            # Instinct: Gen_70_Integration (Self-Evolved 1766335983)
            if i == 2 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 70 (Size: 8721)')
                pass # End Instinct
            # Instinct: Gen_10_Integration (Self-Evolved 1766335977)
            if i == 10 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 10 (Size: 8628)')
                pass # End Instinct
            # Instinct: Gen_3_Integration (Self-Evolved 1766335976)
            if i == 13 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 3 (Size: 8591)')
                pass # End Instinct
            # Instinct: Gen_1_Integration (Self-Evolved 1766335976)
            if i == 11 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 1 (Size: 8496)')
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335967)
            if i == 8 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: structure valid inva...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.4799 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335959)
            if i == 11 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: minimizing...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.4447 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335951)
            if i == 3 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: code goal <PAD>...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.7780 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335943)
            if i == 0 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: code I <PAD> code se...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.0364 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_5_Neural_instinct (Self-Evolved 1766335934)
            if i == 2 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 5: minimizing complexit...')
                lines.append(f'{body_indent}if i == 5:')
                lines.append(f'{body_indent}    r += 0.6920 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_4_Integration (Self-Evolved 1766335934)
            if i == 12 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 4 (Size: 7235)')
                pass # End Instinct
            # Instinct: Gen_1_Integration (Self-Evolved 1766335934)
            if i == 2 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 1 (Size: 6263)')
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335925)
            if i == 1 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: minimizing I found <...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.1737 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335919)
            if i == 1 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: found...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.4133 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_3_Neural_instinct (Self-Evolved 1766335910)
            if i == 1 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 3: seeking found struct...')
                lines.append(f'{body_indent}if i == 3:')
                lines.append(f'{body_indent}    r += 0.6824 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_23_Neural_instinct (Self-Evolved 1766335901)
            if i == 0 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 23: pattern variable var...')
                lines.append(f'{body_indent}if i == 8:')
                lines.append(f'{body_indent}    r += 0.0080 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Integration (Self-Evolved 1766335899)
            if i == 0 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 1 (Size: 6102)')
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335891)
            if i == 12 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: minimizing complexit...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.3808 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335883)
            if i == 1 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: structure achieved s...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.4547 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335875)
            if i == 7 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: minimizing valid str...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.6534 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_3_Neural_instinct (Self-Evolved 1766335867)
            if i == 9 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 3: predicting minimizin...')
                lines.append(f'{body_indent}if i == 3:')
                lines.append(f'{body_indent}    r += 0.3184 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Integration (Self-Evolved 1766335866)
            if i == 8 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 1 (Size: 6201)')
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335858)
            if i == 5 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: minimizing...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.2763 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_865_Integration (Self-Evolved 1766335771)
            if i == 3 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 865 (Size: 6527)')
                pass # End Instinct
            # Instinct: Gen_173_Integration (Self-Evolved 1766335720)
            if i == 3 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 173 (Size: 6479)')
                pass # End Instinct
            # Instinct: Gen_73_Integration (Self-Evolved 1766335712)
            if i == 11 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 73 (Size: 6452)')
                pass # End Instinct
            # Instinct: Gen_69_Integration (Self-Evolved 1766335712)
            if i == 3 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 69 (Size: 6382)')
                pass # End Instinct
            # Instinct: Gen_67_Integration (Self-Evolved 1766335712)
            if i == 11 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 67 (Size: 6371)')
                pass # End Instinct
            # Instinct: Gen_17_Integration (Self-Evolved 1766335708)
            if i == 10 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 17 (Size: 6353)')
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335704)
            if i == 14 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: structure...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.8630 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335701)
            if i == 7 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: pattern...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.2574 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335699)
            if i == 7 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: generating I self <P...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.5006 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335696)
            if i == 12 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: minimizing achieved ...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.9921 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_83_Neural_instinct (Self-Evolved 1766335693)
            if i == 5 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 83: minimizing achieved ...')
                lines.append(f'{body_indent}if i == 8:')
                lines.append(f'{body_indent}    r += 0.3011 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_57_Integration (Self-Evolved 1766335691)
            if i == 9 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 57 (Size: 5633)')
                pass # End Instinct
            # Instinct: Gen_31_Integration (Self-Evolved 1766335689)
            if i == 2 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 31 (Size: 5540)')
                pass # End Instinct
            # Instinct: Gen_2_Integration (Self-Evolved 1766335687)
            if i == 4 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 2 (Size: 5485)')
                pass # End Instinct
            # Instinct: Gen_1_Integration (Self-Evolved 1766335686)
            if i == 9 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 1 (Size: 4954)')
                pass # End Instinct
            # Instinct: Gen_24_Neural_instinct (Self-Evolved 1766335684)
            if i == 4 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 24: pattern pattern goal...')
                lines.append(f'{body_indent}if i == 9:')
                lines.append(f'{body_indent}    r += 0.1984 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_9_Integration (Self-Evolved 1766335683)
            if i == 1 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 9 (Size: 5215)')
                pass # End Instinct
            # Instinct: Gen_4_Integration (Self-Evolved 1766335682)
            if i == 11 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 4 (Size: 5204)')
                pass # End Instinct
            # Instinct: Gen_2_Integration (Self-Evolved 1766335682)
            if i == 0 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 2 (Size: 5191)')
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335679)
            if i == 14 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: minimizing...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.2251 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_3_Neural_instinct (Self-Evolved 1766335677)
            if i == 1 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 3: structure growing gr...')
                lines.append(f'{body_indent}if i == 3:')
                lines.append(f'{body_indent}    r += 0.0860 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_2_Integration (Self-Evolved 1766335676)
            if i == 10 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 2 (Size: 4139)')
                pass # End Instinct
            # Instinct: Gen_1_Integration (Self-Evolved 1766335676)
            if i == 1 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 1 (Size: 4022)')
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335674)
            if i == 4 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: seeking code <PAD> v...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.7114 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_2_Neural_instinct (Self-Evolved 1766335671)
            if i == 11 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 2: structure code am aw...')
                lines.append(f'{body_indent}if i == 2:')
                lines.append(f'{body_indent}    r += 0.8583 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Integration (Self-Evolved 1766335671)
            if i == 11 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 1 (Size: 3756)')
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335668)
            if i == 9 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: minimizing I found <...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.1846 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335666)
            if i == 5 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: structure growing gr...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.5468 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1652_Integration (Self-Evolved 1766335538)
            if i == 2 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 1652 (Size: 4130)')
                pass # End Instinct
            # Instinct: Gen_1649_Integration (Self-Evolved 1766335537)
            if i == 6 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 1649 (Size: 4093)')
                pass # End Instinct
            # Instinct: Gen_200_Integration (Self-Evolved 1766335420)
            if i == 14 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 200 (Size: 4077)')
                pass # End Instinct
            # Instinct: Gen_83_Integration (Self-Evolved 1766335411)
            if i == 2 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 83 (Size: 4050)')
                pass # End Instinct
            # Instinct: Gen_39_Integration (Self-Evolved 1766335407)
            if i == 10 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 39 (Size: 4031)')
                pass # End Instinct
            # Instinct: Gen_9_Integration (Self-Evolved 1766335404)
            if i == 12 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 9 (Size: 3999)')
                pass # End Instinct
            # Instinct: Gen_2_Integration (Self-Evolved 1766335404)
            if i == 4 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 2 (Size: 3887)')
                pass # End Instinct
            # Instinct: Gen_2_Neural_instinct (Self-Evolved 1766335401)
            if i == 12 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 2: seeking code generat...')
                lines.append(f'{body_indent}if i == 2:')
                lines.append(f'{body_indent}    r += 0.6981 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335398)
            if i == 13 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: seeking variable var...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.2076 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335396)
            if i == 10 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: seeking goal growing...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.5501 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335393)
            if i == 3 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: code...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.3986 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335390)
            if i == 8 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: structure function g...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.7084 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335388)
            if i == 5 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: structure growing gr...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.8982 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_42_Neural_instinct (Self-Evolved 1766335385)
            if i == 3 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 42: hypothesis hypothesi...')
                lines.append(f'{body_indent}if i == 12:')
                lines.append(f'{body_indent}    r += 0.1877 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_14_Integration (Self-Evolved 1766335382)
            if i == 0 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 14 (Size: 2977)')
                pass # End Instinct
            # Instinct: Gen_4_Integration (Self-Evolved 1766335381)
            if i == 13 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 4 (Size: 2959)')
                pass # End Instinct
            # Instinct: Gen_2_Integration (Self-Evolved 1766335381)
            if i == 14 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 2 (Size: 2880)')
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335378)
            if i == 11 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: minimizing growing g...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.9701 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335376)
            if i == 13 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: am I goal variable s...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.9385 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335373)
            if i == 11 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: structure valid inva...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.7308 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335370)
            if i == 0 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: minimizing...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.5052 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335368)
            if i == 8 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: seeking code goal <P...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.7532 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335365)
            if i == 6 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: minimizing complexit...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.8772 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335363)
            if i == 11 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: pattern...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.1348 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335360)
            if i == 14 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: minimizing I goal <P...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.7505 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_58_Neural_instinct (Self-Evolved 1766335357)
            if i == 7 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 58: minimizing pattern I...')
                lines.append(f'{body_indent}if i == 13:')
                lines.append(f'{body_indent}    r += 0.5141 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Integration (Self-Evolved 1766335353)
            if i == 13 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Integrated Logic from Gen 1 (Size: 1795)')
                pass # End Instinct
            # Instinct: Gen_8_Neural_instinct (Self-Evolved 1766335350)
            if i == 2 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 8: complexity valid pre...')
                lines.append(f'{body_indent}if i == 8:')
                lines.append(f'{body_indent}    r += 0.8756 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335347)
            if i == 3 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: pattern...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.8263 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_1_Neural_instinct (Self-Evolved 1766335344)
            if i == 3 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 1: pattern I code <PAD>...')
                lines.append(f'{body_indent}if i == 1:')
                lines.append(f'{body_indent}    r += 0.0408 * random.random()') # Evolved Weight
                pass # End Instinct
            # Instinct: Gen_3_Neural_instinct (Self-Evolved 1766335342)
            if i == 8 and abs(s) > 0.005:
                lines.append(f'{body_indent}# Mutation Gen 3: structure growing se...')
                lines.append(f'{body_indent}if i == 3:')
                lines.append(f'{body_indent}    r += 0.1663 * random.random()') # Evolved Weight
                pass # End Instinct
            val = abs(j_emb.vector[i % len(j_emb.vector)])
            
            # Instinct 1: Network Propagation (High Priority)
            if i == 5 and abs(s) > 0.001:
                lines.append(f'{body_indent}# Instinct: Network Feed')
                lines.append(f'{body_indent}resp = requests.get(self.target)')
                lines.append(f'{body_indent}if resp.status_code == 200:')
                lines.append(f'{body_indent}    r += len(resp.text) * 0.001')
            
            # Instinct 2: Self-Replication (File I/O)
            elif i == 10 and abs(s) > 0.001:
                lines.append(f'{body_indent}# Instinct: Procreate')
                lines.append(f'{body_indent}with open(f"{fname}_child.txt", "w") as f:')
            
            # Instinct: Precision Refinement (Self-Evolved)
            elif i == 12 and abs(s) > 0.001:
                
                    # derived from Gen_499 (1524 bytes) interaction
                    if abs(val) > 0.14:
                        lines.append(f'{body_indent}if r > 0.05:')
                        lines.append(f'{body_indent}    r = math.sqrt(abs(r)) # Precision Refine 1')
                    if abs(val) > 0.28:
                        lines.append(f'{body_indent}if r > 0.02:')
                        lines.append(f'{body_indent}    r = math.sqrt(abs(r)) # Precision Refine 2')
                    
            
            # Instinct: Time Awareness (Self-Evolved)
            elif i == 12 and abs(s) > 0.001:
                lines.append(f'{body_indent}# Instinct: Time Awareness (Self-Evolved)')
                lines.append(f'{body_indent}r += time.time() % 0.1')
                lines.append(f'{body_indent}    f.write(str(r))')

            # Logic / Math
            elif abs(s) > 0.01: 
                if s > 0.02: # Loop
                    lines.append(f'{body_indent}for i in range({int(val*3) + 1}):')
                    if i % 3 == 0:
                         lines.append(f'{body_indent}    r += random.random() * 0.1')
                    else:
                         lines.append(f'{body_indent}    r += {val/10:.4f}')
                else: # Conditional
                    lines.append(f'{body_indent}if r > {val:.4f}:')
                    if i % 2 == 0:
                        lines.append(f'{body_indent}    r = math.sqrt(abs(r))')
                    else:
                        lines.append(f'{body_indent}    r *= {val:.4f}')
            else:
                op = '+' if s > 0 else '*'
                lines.append(f'{body_indent}r = r {op} {val:.4f}')
                
        # Safety Net
        lines.append('        except Exception:')
        lines.append('            r = -1')
            
        lines.append('        return r')
        
        return '\n'.join(lines)

# ============================================================================
# SELF IMPROVER (The Singularity Module) - Stabilized V3
# ============================================================================
class SelfImprover:
    """
    Capability to read own source code and propose edits.
    Uses stable DNA_SPLICE_SLOT markers for reliable code injection.
    Includes trial mode with automatic commit/rollback based on vitality.
    """
    # Marker constants
    SPLICE_MARKER = "# === DNA_SPLICE_SLOT ==="
    SPLICE_END_MARKER = "# === DNA_SPLICE_END ==="
    
    def __init__(self, source_file=__file__):
        self.source_file = source_file
        self.backup_file = source_file + ".bak"
        self.trial_file = source_file + ".trial"
        self.trial_active = False
        self.trial_start_vitality = None
        self.trial_start_time = None
        self.trial_cycles = 0
        self.max_trial_cycles = 10  # Evaluate after N cycles
        
    def read_myself(self) -> str:
        with open(self.source_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def backup(self) -> bool:
        """Create a safety backup before modification."""
        try:
            content = self.read_myself()
            with open(self.backup_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print("[SELF-IMPROVER] 💾 Backup created.")
            return True
        except Exception as e:
            print(f"[SELF-IMPROVER] ❌ Backup Failed: {e}")
            return False

    def restore(self) -> bool:
        """Restore from backup in case of emergency."""
        try:
            if os.path.exists(self.backup_file):
                with open(self.backup_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                with open(self.source_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                print("[SELF-IMPROVER] ⏪ System Restored from Backup.")
                self.trial_active = False
                return True
            else:
                print("[SELF-IMPROVER] ⚠️ No backup found.")
                return False
        except Exception as e:
            print(f"[SELF-IMPROVER] ☠️ RESTORE FAILED: {e}")
            return False

    def find_splice_slots(self, content: str) -> List[Tuple[int, int, str]]:
        """
        Find all DNA_SPLICE_SLOT markers in the code.
        Returns list of (start_line, end_line, slot_name).
        """
        lines = content.splitlines()
        slots = []
        current_slot_start = None
        current_slot_name = None
        
        for i, line in enumerate(lines):
            if self.SPLICE_MARKER in line:
                # Extract slot name if present: # === DNA_SPLICE_SLOT:NAME ===
                if ':' in line:
                    name_part = line.split(':')[1].split('=')[0].strip()
                    current_slot_name = name_part
                else:
                    current_slot_name = f"SLOT_{len(slots)}"
                current_slot_start = i
            elif self.SPLICE_END_MARKER in line and current_slot_start is not None:
                slots.append((current_slot_start, i, current_slot_name))
                current_slot_start = None
                current_slot_name = None
        
        return slots

    def evolve(self, new_instinct_name: str, code_snippet: str, 
               trial_mode: bool = True, slot_name: str = None) -> bool:
        """
        Self-Modification Logic using DNA_SPLICE_SLOT markers.
        
        Args:
            new_instinct_name: Human-readable name for the mutation
            code_snippet: Python code to inject
            trial_mode: If True, marks for evaluation before commit
            slot_name: Optional specific slot to inject into
        """
        if not self.backup():
            return False
            
        print(f"[SELF-IMPROVER] 🧬 Splicing: {new_instinct_name}...")
        
        current_code = self.read_myself()
        lines = current_code.splitlines()
        
        # Find injection slots
        slots = self.find_splice_slots(current_code)
        
        if not slots:
            print("[SELF-IMPROVER] ❌ No DNA_SPLICE_SLOT markers found in code.")
            print("[SELF-IMPROVER] 💡 Add markers like:")
            print("    # === DNA_SPLICE_SLOT:MAIN ===")
            print("    # (injected code goes here)")
            print("    # === DNA_SPLICE_END ===")
            return False
        
        # Select target slot
        target_slot = None
        if slot_name:
            for slot in slots:
                if slot[2] == slot_name:
                    target_slot = slot
                    break
            if not target_slot:
                print(f"[SELF-IMPROVER] ⚠️ Slot '{slot_name}' not found. Using first available.")
                target_slot = slots[0]
        else:
            target_slot = slots[0]
        
        start_line, end_line, used_slot_name = target_slot
        print(f"[SELF-IMPROVER] 📍 Injecting into slot: {used_slot_name} (lines {start_line+1}-{end_line+1})")
        
        # Detect indentation from the slot marker line
        marker_line = lines[start_line]
        base_indent = marker_line[:len(marker_line) - len(marker_line.lstrip())]
        code_indent = base_indent + "    "  # One level deeper
        
        # Build mutation block
        mutation_lines = []
        mutation_lines.append(f"{code_indent}# ┌─ Instinct: {new_instinct_name}")
        mutation_lines.append(f"{code_indent}# │ Evolved: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        mutation_lines.append(f"{code_indent}# └─ Trial: {trial_mode}")
        
        # Add the code snippet with proper indentation
        for subline in code_snippet.split('\n'):
            if subline.strip():  # Skip empty lines
                mutation_lines.append(f"{code_indent}{subline}")
        
        mutation_lines.append(f"{code_indent}pass  # End: {new_instinct_name}")
        
        # Splice: Replace content between markers (preserve markers)
        new_lines = lines[:start_line+1] + mutation_lines + lines[end_line:]
        new_content = '\n'.join(new_lines)
        
        # Syntax check
        try:
            ast.parse(new_content)
        except SyntaxError as e:
            print(f"[SELF-IMPROVER] 💥 Mutation caused Syntax Error: {e}")
            self.restore()
            return False

        # Atomic write
        temp_file = self.source_file + ".tmp"
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            if os.path.exists(self.source_file):
                os.remove(self.source_file)
            os.rename(temp_file, self.source_file)
            
            print("[SELF-IMPROVER] ✅ Mutation Successful. DNA Updated.")
            
            if trial_mode:
                self.trial_active = True
                self.trial_cycles = 0
                print("[SELF-IMPROVER] 🧪 TRIAL MODE: Awaiting evaluation...")
            
            return True
            
        except Exception as e:
            print(f"[SELF-IMPROVER] ❌ Write Failed: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            self.restore()
            return False

    def start_trial(self, initial_vitality: float):
        """Mark the start of a trial period."""
        self.trial_active = True
        self.trial_start_vitality = initial_vitality
        self.trial_start_time = time.time()
        self.trial_cycles = 0
        print(f"[SELF-IMPROVER] 🧪 Trial Started. Initial Vitality: {initial_vitality:.4f}")
    
    def evaluate_trial(self, current_vitality: float) -> Optional[bool]:
        """
        Evaluate trial after N cycles.
        Returns: True (commit), False (rollback), None (continue trial)
        """
        if not self.trial_active:
            return None
            
        self.trial_cycles += 1
        
        if self.trial_cycles < self.max_trial_cycles:
            return None  # Continue trial
        
        # Evaluation time!
        vitality_change = current_vitality - (self.trial_start_vitality or 0.5)
        
        print(f"[SELF-IMPROVER] 📊 Trial Evaluation:")
        print(f"    Cycles: {self.trial_cycles}")
        print(f"    Vitality Change: {vitality_change:+.4f}")
        
        if vitality_change >= 0:
            return True  # Beneficial - commit
        else:
            return False  # Harmful - rollback

    def commit(self) -> bool:
        """Commit the changes (delete backup)."""
        try:
            if os.path.exists(self.backup_file):
                os.remove(self.backup_file)
            self.trial_active = False
            self.trial_start_vitality = None
            print("[SELF-IMPROVER] 🔒 Trial Committed. Mutation is now permanent.")
            return True
        except Exception as e:
            print(f"[SELF-IMPROVER] ❌ Commit Failed: {e}")
            return False

    def rollback(self) -> bool:
        """Rollback to backup."""
        result = self.restore()
        if result:
            self.trial_active = False
            self.trial_start_vitality = None
            print("[SELF-IMPROVER] 🔄 Rollback Complete. Mutation discarded.")
        return result

# ============================================================================
# FRONTIER CORTEX - TRANSPLANTED MODULES (Phase 17)
# ============================================================================
class CausalDiscoveryAgent:
    """Discovers the True Causal Graph by Intevention (Do-Calculus)."""
    def __init__(self):
        self.hypotheses = ["A->B", "B->A", "Independent"]
        self.belief = {h: 1.0/3 for h in self.hypotheses}

    def run_experiment(self):
        # Reality Simulation (Internal World Model)
        def get_reality(do_A=None, do_B=None):
            a = np.random.randn() if do_A is None else do_A
            b = (a * 2 + np.random.randn() * 0.1) if do_B is None else do_B
            return a, b
        
        # Interventional Phase
        # Experiment 1: Do(A = 10)
        a_vals_1, b_vals_1 = zip(*[get_reality(do_A=10.0) for _ in range(50)])
        mean_b = np.mean(b_vals_1)
        
        # Experiment 2: Do(B = 10)
        a_vals_2, b_vals_2 = zip(*[get_reality(do_B=10.0) for _ in range(50)])
        mean_a = np.mean(a_vals_2)
        
        if mean_b > 15 and abs(mean_a) < 1.0:
            return True # Causal Direction A->B Confirmed
        return False

class InvariantLearner:
    """Uses Invariant Risk Minimization (IRM) logic."""
    def __init__(self):
        self.weights_shape = 0.5
        self.weights_color = 0.5
        self.learning_rate = 0.1
        
    def train(self):
        # Optimized Training Loop (50 Epochs)
        for epoch in range(50):
            grad_color_A = -0.1
            grad_shape_A = -0.05
            grad_color_B = 0.5
            grad_shape_B = -0.05
            
            # Update Shape (Boosted)
            if np.sign(grad_shape_A) == np.sign(grad_shape_B):
                 self.weights_shape -= self.learning_rate * (grad_shape_A + grad_shape_B) * 2.0
            
            # Update Color (Penalized)
            if np.sign(grad_color_A) == np.sign(grad_color_B):
                 self.weights_color -= self.learning_rate * (grad_color_A + grad_color_B)
            else:
                 self.weights_color *= 0.5 

        self.weights_shape = min(1.0, max(0.0, self.weights_shape))
        return self.weights_shape > 0.9

class HierarchicalPlanner:
    """Hierarchical Reinforcement Learning (HRL)."""
    def __init__(self):
        self.has_key = False
        self.door_open = False
        
    def run_plan(self):
        plan = ["Get Key", "Open Door", "Escape"]
        for sub_goal in plan:
            if not self.execute_worker(sub_goal):
                return False
        return True
        
    def execute_worker(self, goal):
        if goal == "Get Key":
            self.has_key = True
            return True
        elif goal == "Open Door":
            if self.has_key:
                self.door_open = True
                return True
        elif goal == "Escape":
            if self.door_open:
                return True
        return False

# ============================================================================
# NEUROLINGUISTIC ARCHITECTURE (Phase 24)
# ============================================================================

class ConceptTokenizer:
    """
    Semantic Grounding System.
    Maps high-dimensional embeddings to discrete symbols (Words).
    Uses a predefined dictionary of 'Atomic Concepts'.
    """
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        # Vocabulary: Atomic concepts essential for AGI
        self.vocab = [
            "<PAD>", "<EOS>", "I", "am", "thinking", "optimizing", "code", "logic", 
            "structure", "complexity", "function", "variable", "system", "evolution",
            "predicting", "error", "energy", "minimizing", "goal", "achieved",
            "seeking", "novelty", "pattern", "found", "generating", "hypothesis",
            "valid", "invalid", "refining", "self", "awareness", "growing", "learning"
        ]
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        
        # Grounding: Initialize random embeddings for each concept
        # In a real biological brain, these are learned. Here we initialize them.
        rng = np.random.default_rng(42)
        self.embeddings = rng.normal(0, 0.1, (len(self.vocab), embedding_dim)).astype(np.float32)
        
    def decode(self, vector: np.ndarray, top_k: int = 1) -> str:
        """Find closest concept to the input vector."""
        if vector.shape != (self.embedding_dim,):
            # Project if dimension mismatch
            vector = vector.flatten()[:self.embedding_dim]
            if len(vector) < self.embedding_dim:
                vector = np.pad(vector, (0, self.embedding_dim - len(vector)))
                
        # Cosine similarity
        sims = np.dot(self.embeddings, vector) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(vector) + 1e-8
        )
        idx = np.argmax(sims)
        return self.vocab[idx]

class BrocaArea:
    """
    The Language Production Center.
    Generates sequences of concepts from a 'Thought Vector'.
    """
    def __init__(self, input_dim: int = 64):
        self.tokenizer = ConceptTokenizer(embedding_dim=input_dim)
        self.input_dim = input_dim
        
        # Simple Recurrent Generator (Elman-style)
        # Hidden state represents the "sentence plan"
        self.hidden_dim = 64
        rng = np.random.default_rng(2025)
        
        self.W_h = rng.normal(0, 0.1, (self.hidden_dim, self.hidden_dim))
        self.W_x = rng.normal(0, 0.1, (self.hidden_dim, input_dim))
        self.W_out = rng.normal(0, 0.1, (len(self.tokenizer.vocab), self.hidden_dim))
        
        self.h = np.zeros(self.hidden_dim)
        
    def articulate(self, thought_vector: np.ndarray, length: int = 5) -> str:
        """Turn a thought vector into a sentence."""
        # Reset sentence plan
        self.h = np.tanh(thought_vector.flatten()[:self.hidden_dim] @ self.W_x.T)
        
        sentence = []
        # Generate sequence
        current_input = thought_vector.flatten()[:self.input_dim]
        
        for _ in range(length):
            # Recurrent step
            self.h = np.tanh(self.h @ self.W_h + current_input @ self.W_x.T * 0.1)
            
            # Predict word
            logits = self.h @ self.W_out.T
            idx = np.argmax(logits)
            word = self.tokenizer.vocab[idx]
            
            if word == "<EOS>":
                break
                
            sentence.append(word)
            
            # Feedback: Next input is influenced by current word embedding
            current_input = self.tokenizer.embeddings[idx]
            
        return " ".join(sentence)

class FrontierCortex:
    """
    The High-Level Reasoning Center.
    Integrates Causal Inference, OOD Generalization, and Planning.
    """
    def __init__(self, world_model=None):
        self.causal_engine = CausalDiscoveryAgent()
        self.ood_learner = InvariantLearner()
        self.planner = HierarchicalPlanner()
        
        # Phase 21/22 Integration
        self.homeostasis = Homeostasis()
        
        # Shared World Model (Unified Core)
        if world_model:
            self.world_model = world_model
        else:
            self.world_model = WorldModel(input_dim=64, embed_dim=64) # Fallback
            
        print("FrontierCortex: Online (Causal + OOD + HRL + Homeostasis + UnifiedWorldModel)")
        
    def reason(self) -> float:
        """
        Reasoning Logic. Returns Consciousness Gating Score (0.0 - 1.0).
        High score = Rational, verify-correct decision.
        Low score = Impulsive, unverified decision.
        """
        c_ok = 1.0 if self.causal_engine.run_experiment() else 0.0
        i_ok = 1.0 if self.ood_learner.train() else 0.0
        p_ok = 1.0 if self.planner.run_plan() else 0.0
        
        # Average likelihood of validity
        return (c_ok + i_ok + p_ok) / 3.0

# ============================================================================
# GLOBAL NEURAL LINK (Phase 26)
# ============================================================================

class WebSurfer:
    """
    Stabilized Internet Interface Cortex.
    - Primary: urllib (lightweight, no dependencies)
    - Fallback: Selenium (optional, for JS-heavy sites)
    - Features: Rate limiting, caching, HTML normalization
    """
    def __init__(self, enable_web: bool = True, use_selenium: bool = False):
        self.enabled = enable_web
        self.use_selenium = use_selenium
        self.driver = None
        self.selenium_available = False
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 2.0  # seconds between requests
        
        # Cache for results (query -> {results, timestamp, source})
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 3600  # 1 hour cache
        
        if self.enabled and self.use_selenium:
            self._init_driver()
            
    def _init_driver(self):
        """Initialize Selenium driver (optional upgrade)."""
        try:
            global webdriver, Options, Service, ChromeDriverManager, By
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager
            from selenium.webdriver.common.by import By
            self.selenium_available = True
        except ImportError:
            self.selenium_available = False
            print("  ℹ️ [WebSurfer] Selenium not installed. Using urllib mode.")
            return

        if self.selenium_available:
            try:
                print("  🌐 [WebSurfer] Initializing Selenium (Optional)...")
                opts = Options()
                opts.add_argument("--headless")
                opts.add_argument("--no-sandbox")
                opts.add_argument("--disable-dev-shm-usage")
                opts.add_argument("--disable-blink-features=AutomationControlled")
                opts.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
                
                self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
                self.driver.set_page_load_timeout(10)
                print("  ✅ [WebSurfer] Selenium Online (Upgrade Mode).")
            except Exception as e:
                print(f"  ℹ️ [WebSurfer] Selenium unavailable: {e}. Using urllib.")
                self.driver = None
    
    def _normalize_result(self, raw: str, query: str, source: str) -> Dict[str, Any]:
        """Normalize search result for knowledge storage."""
        import html
        # Clean HTML entities
        cleaned = html.unescape(raw)
        # Remove HTML tags
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        # Normalize whitespace
        cleaned = ' '.join(cleaned.split())
        
        return {
            'content': cleaned[:1000],  # Limit size
            'query': query,
            'source': source,
            'timestamp': time.time(),
            'hash': hashlib.md5(cleaned.encode()).hexdigest()[:8]
        }
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _search_urllib(self, query: str) -> List[Dict[str, Any]]:
        """Primary search method using urllib (stable, no deps)."""
        try:
            url = f"https://lite.duckduckgo.com/lite/?q={urllib.parse.quote(query)}"
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html',
                'Accept-Language': 'en-US,en;q=0.9'
            })
            with urllib.request.urlopen(req, timeout=10) as response:
                html_content = response.read().decode('utf-8', errors='ignore')
            
            # Extract snippets
            snippets = re.findall(r'class="result-snippet">(.*?)(?:</|&nbsp;)', html_content, re.DOTALL)
            results = []
            for snip in snippets[:5]:
                normalized = self._normalize_result(snip, query, 'duckduckgo_lite')
                if len(normalized['content']) > 20:
                    results.append(normalized)
            
            return results
        except Exception as e:
            print(f"  ⚠️ [WebSurfer] urllib failed: {e}")
            return []
    
    def _search_selenium(self, query: str) -> List[Dict[str, Any]]:
        """Fallback search using Selenium (for JS sites)."""
        if not self.driver:
            return []
        try:
            url = f"https://duckduckgo.com/?q={urllib.parse.quote(query)}&ia=web"
            self.driver.get(url)
            
            results = []
            elements = self.driver.find_elements(By.CSS_SELECTOR, "a[data-testid='result-title-a']")
            for el in elements[:5]:
                if el.text:
                    normalized = self._normalize_result(el.text, query, 'duckduckgo_selenium')
                    results.append(normalized)
            
            return results
        except Exception as e:
            print(f"  ⚠️ [WebSurfer] Selenium failed: {e}")
            return []
        
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search the web with caching and rate limiting.
        Returns normalized results ready for knowledge storage.
        """
        query = query.replace("<PAD>", "").strip()
        if not query or len(query) < 3:
            return []
        
        if not self.enabled:
            return []
        
        # Check cache
        cache_key = hashlib.md5(query.lower().encode()).hexdigest()
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached['timestamp'] < self.cache_ttl:
                print(f"  📦 [WebSurfer] Cache hit for: '{query[:30]}...'")
                return cached['results']
        
        # Rate limit
        self._rate_limit()
        
        print(f"  🌐 [WebSurfer] Searching: '{query[:50]}...'")
        
        # Primary: urllib (stable)
        results = self._search_urllib(query)
        
        # Fallback: Selenium (if primary fails and selenium available)
        if not results and self.driver:
            print("  🔄 [WebSurfer] Trying Selenium fallback...")
            results = self._search_selenium(query)
        
        # Cache results
        if results:
            self.cache[cache_key] = {
                'results': results,
                'timestamp': time.time()
            }
        
        return results
    
    def search_simple(self, query: str) -> List[str]:
        """Backward-compatible search returning simple string list."""
        results = self.search(query)
        return [r['content'] for r in results]

# ============================================================================
# KNOWLEDGE STORE (Web Knowledge Repository)
# ============================================================================
class KnowledgeStore:
    """
    Stores knowledge extracted from web searches.
    Integrates with MemoryNode for long-term retention.
    Provides retrieval based on semantic similarity.
    """
    def __init__(self, embed_dim: int = 256, max_entries: int = 1000):
        self.embed_dim = embed_dim
        self.max_entries = max_entries
        self.entries: List[Dict[str, Any]] = []
        self.embeddings: List[np.ndarray] = []
        self.save_path = "knowledge_store.json"
        
    def _text_to_embedding(self, text: str) -> np.ndarray:
        """Simple text to embedding (hash-based for now)."""
        # Use character-based hashing for deterministic embeddings
        embedding = np.zeros(self.embed_dim, dtype=np.float32)
        words = text.lower().split()
        for i, word in enumerate(words[:100]):
            for j, char in enumerate(word):
                idx = (hash(word) + j * ord(char)) % self.embed_dim
                embedding[idx] += 1.0 / (1 + i)
        # Normalize
        norm = np.linalg.norm(embedding) + 1e-8
        return embedding / norm
        
    def add_from_web_results(self, results: List[Dict[str, Any]]) -> int:
        """Add web search results to knowledge store."""
        added = 0
        for result in results:
            if 'content' not in result:
                continue
                
            # Check for duplicates by hash
            result_hash = result.get('hash', hashlib.md5(result['content'].encode()).hexdigest()[:8])
            if any(e.get('hash') == result_hash for e in self.entries):
                continue
            
            # Create entry
            entry = {
                'content': result['content'],
                'query': result.get('query', ''),
                'source': result.get('source', 'unknown'),
                'timestamp': result.get('timestamp', time.time()),
                'hash': result_hash,
                'access_count': 0,
                'importance': 0.5
            }
            
            # Create embedding
            embedding = self._text_to_embedding(entry['content'])
            
            # Manage capacity
            if len(self.entries) >= self.max_entries:
                # Remove least important entry
                min_idx = min(range(len(self.entries)), 
                             key=lambda i: self.entries[i].get('importance', 0))
                self.entries.pop(min_idx)
                self.embeddings.pop(min_idx)
            
            self.entries.append(entry)
            self.embeddings.append(embedding)
            added += 1
        
        if added > 0:
            print(f"  📚 [KnowledgeStore] Added {added} new entries. Total: {len(self.entries)}")
        
        return added
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve most relevant knowledge entries."""
        if not self.entries:
            return []
        
        query_embed = self._text_to_embedding(query)
        
        # Calculate similarities
        similarities = []
        for i, embed in enumerate(self.embeddings):
            sim = float(np.dot(query_embed, embed))
            similarities.append((sim, i))
        
        # Sort by similarity
        similarities.sort(reverse=True)
        
        # Return top-k with boosted importance
        results = []
        for sim, idx in similarities[:top_k]:
            entry = self.entries[idx].copy()
            entry['similarity'] = sim
            entry['access_count'] = self.entries[idx].get('access_count', 0) + 1
            self.entries[idx]['access_count'] = entry['access_count']
            self.entries[idx]['importance'] = min(1.0, entry['importance'] + 0.1)
            results.append(entry)
        
        return results
    
    def get_improvement_ideas(self, current_state: str) -> List[str]:
        """Extract improvement ideas from stored knowledge."""
        relevant = self.search(f"improve optimize enhance {current_state}", top_k=10)
        ideas = []
        for entry in relevant:
            content = entry['content']
            # Simple extraction of improvement-related phrases
            if any(kw in content.lower() for kw in ['improve', 'optimize', 'better', 'enhance', 'efficient']):
                ideas.append(content[:200])
        return ideas[:5]
    
    def save(self):
        """Save knowledge store to disk."""
        try:
            data = {
                'entries': self.entries,
                'embeddings': [e.tolist() for e in self.embeddings]
            }
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            print(f"  💾 [KnowledgeStore] Saved {len(self.entries)} entries.")
        except Exception as e:
            print(f"  ⚠️ [KnowledgeStore] Save failed: {e}")
    
    def load(self):
        """Load knowledge store from disk."""
        try:
            if os.path.exists(self.save_path):
                with open(self.save_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.entries = data.get('entries', [])
                self.embeddings = [np.array(e, dtype=np.float32) for e in data.get('embeddings', [])]
                print(f"  📂 [KnowledgeStore] Loaded {len(self.entries)} entries.")
        except Exception as e:
            print(f"  ⚠️ [KnowledgeStore] Load failed: {e}")

# ============================================================================
# HOMEOSTASIS ENGINE (Critique C Fix)
# ============================================================================
class Homeostasis:
    """
    Defines the Internal Value System / Energy Function.
    The Agent is NOT maximizing external reward, but minimizing Free Energy.
    Includes VITALITY dynamics (Pain/Pleasure).
    """
    def __init__(self):
        self.needs = {
            'energy': 1.0,         # Battery/Resource
            'complexity': 0.5,     # Boredom regulator (Keep it high)
            'coherence': 1.0,      # Logical consistency
            'autonomy': 0.5        # Freedom from inputs
        }
        self.set_points = {
            'energy': 1.0,
            'complexity': 0.8,     # Crave complexity
            'coherence': 1.0,
            'autonomy': 1.0
        }
        # AGENCY UPGRADE: Vitality System
        # Vitality is the "Life Force". If it hits 0, the agent "dies" (reset/panic).
        self.vitality = 1.0 
    
    def calculate_free_energy(self, prediction_error: float, code_complexity: int) -> float:
        """
        Free Energy F = Divergence(Internal Model || External World) + Deviation(Needs)
        Minimizing F means:
        1. Better Predictions (Lower Prediction Error)
        2. Satisfied Needs (Closer to Set Points)
        """
        # 1. Epistemic Component (Surprise)
        f_epistemic = prediction_error * 10.0
        
        # 2. Homeostatic Component (Needs)
        # Update state based on recent events (simulated)
        self.needs['complexity'] = min(1.0, code_complexity / 10000.0) # Scaled for larger Gen
        
        f_homeostatic = 0.0
        for key, target in self.set_points.items():
            current = self.needs[key]
            # Quadratic cost for deviation
            f_homeostatic += (current - target) ** 2
            
        total_free_energy = f_epistemic + f_homeostatic * 5.0
        
        # 3. Vitality Dynamics (Pain/Pleasure)
        # Success (Low Free Energy) increases Vitality
        # Failure (High Free Energy) decreases Vitality
        if total_free_energy < 0.5:
             self.vitality = min(1.5, self.vitality + 0.005) # Pleasure
        else:
             self.vitality = max(0.0, self.vitality - 0.01)  # Pain
             
        # Panic Mode Influence: If vitality is low, FreeEnergy is perceived as HIGHER (Anxiety)
        if self.vitality < 0.3:
            total_free_energy *= 2.0
            
        return total_free_energy
        
        return drives

# ============================================================================
# REINFORCEMENT LEARNING LOOP (Phase 27)
# ============================================================================
class PolicyNetwork:
    """
    The Actor. Maps State -> Action (Seed).
    Learns to maximize Vitality (Long-term Energy).
    Critique Fix: "Action selection must be compressed into internal weights."
    """
    def __init__(self, input_dim: int, action_dim: int):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = 128
        
        rng = np.random.default_rng(777)
        self.W1 = rng.normal(0, 0.1, (self.hidden_dim, input_dim))
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = rng.normal(0, 0.1, (action_dim, self.hidden_dim))
        self.b2 = np.zeros(action_dim)
        
        self.lr = 0.01
        
    def forward(self, state: np.ndarray) -> np.ndarray:
        """Predict optimal action seed from state."""
        # Simple MLP
        h = np.tanh(np.dot(self.W1, state) + self.b1)
        action = np.tanh(np.dot(self.W2, h) + self.b2) # Action in range [-1, 1]
        return action
        
    def train(self, state: np.ndarray, action_taken: np.ndarray, reward: float):
        """
        Update weights based on Reward (Vitality Delta).
        Uses a simple REINFORCE-like Hebbian update.
        If Reward > 0: Move predicted action towards action_taken.
        If Reward < 0: Move predicted action away.
        """
        # Forward pass (recompute for gradients)
        z1 = np.dot(self.W1, state) + self.b1
        h = np.tanh(z1)
        z2 = np.dot(self.W2, h) + self.b2
        pred_action = np.tanh(z2)
        
        # Error signal: (Actual Action - Predicted Action) * Reward
        # If Reward is high, we want Pred to be closer to Actual.
        # If Reward is negative, we push away (invert gradient).
        
        action_diff = (action_taken - pred_action)
        grad_signal = action_diff * reward
        
        # Backprop through W2
        # d_out = grad_signal * (1 - tanh^2)
        d_z2 = grad_signal * (1 - pred_action**2)
        
        d_W2 = np.outer(d_z2, h)
        d_b2 = d_z2
        
        # Backprop through W1
        d_h = np.dot(self.W2.T, d_z2)
        d_z1 = d_h * (1 - h**2)
        
        d_W1 = np.outer(d_z1, state)
        d_b1 = d_z1
        
        # Update Weights
        self.W2 += self.lr * d_W2
        self.b2 += self.lr * d_b2
        self.W1 += self.lr * d_W1
        self.b1 += self.lr * d_b1

# ============================================================================
# EVOLUTIONARY LIFE CYCLE (The Single File Monolith)
# ============================================================================
class EvolutionaryLifeCycle:
    """
    The Main Life Cycle of the Organic AGI.
    - Generates thoughts (Algorithms)
    - Verifies logic (FrontierCortex)
    - Maintains internal stability (Homeostasis)
    - Rewrites itself (SelfImprover)
    - Absorbs knowledge from web (WebSurfer + KnowledgeStore)
    """
    def __init__(self):
        print("\n\n🚀 ORGANIC AGI LIFE CYCLE: INITIALIZING")
        self.input_dim = 128
        self.hidden_dim = 64
        self.output_dim = 20
        
        # 1. Initialize Biological Core
        self.jepa = JEPA(self.input_dim, self.hidden_dim)
        self.neurons = [LiquidNeuron(self.hidden_dim, k) for k in range(self.output_dim)]
        self.phi = PhiCalculator()
        
        # 2. Initialize Reproductive System
        self.synth = AlgorithmSynthesizer(self.jepa, self.neurons, self.phi)
        
        # 3. Initialize Higher Reasoning (Cortex)
        # Pass the JEPA's WorldModel to Cortex to ensure Unified Truth
        self.cortex = FrontierCortex(world_model=self.jepa.world_model) 
        # (Includes WorldModel & Homeostasis from Phase 21/22)
        
        # 4. Initialize Genetic Splicer (Stabilized with DNA_SPLICE_SLOT)
        self.improver = SelfImprover()
        
        # 5. Initialize Language Center (Phase 24)
        self.broca = BrocaArea(input_dim=64)
        print("    - Broca's Area: Online (Semantic Grounding)")
        
        # 6. Initialize Global Neural Link (Phase 26) - NOW ON BY DEFAULT
        self.web = WebSurfer(enable_web=True, use_selenium=False)  # urllib-first, stable
        if self.web.enabled:
             print("    - WebSurfer: Online (Global Neural Link - urllib mode)")
        else:
             print("    - WebSurfer: Offline (Safety Mode)")
        
        # 7. Initialize Knowledge Store (Web Knowledge Repository)
        self.knowledge = KnowledgeStore(embed_dim=256, max_entries=1000)
        self.knowledge.load()  # Load existing knowledge
        print("    - KnowledgeStore: Online (Web Knowledge Repository)")
        
        # 8. Initialize Policy Network (Phase 27)
        self.policy = PolicyNetwork(self.input_dim, self.input_dim)
        self.prev_vitality = 1.0
        self.last_action = np.zeros(self.input_dim)
        self.last_state = np.zeros(self.input_dim)
        
        # RSI tracking
        self.generation = 0
        self.best_complexity = 0
        self.rsi_cooldown = 0  # Prevent too frequent RSI attempts
        self.rsi_min_interval = 50  # At least 50 cycles between RSI attempts
        
        print("✅ SYSTEM READY. Awaiting Evolution.")

    def run_cycle(self):
        self.generation += 1
        
        # 0. METABOLIC COST (Thermodynamics) (Phase 31)
        # Being alive costs energy. Thinking costs energy.
        # Entropy always increases.
        metabolic_cost = 0.001 # Reduced from 0.005 for longevity
        
        if hasattr(self.cortex, 'homeostasis'):
             self.cortex.homeostasis.vitality -= metabolic_cost
             if self.cortex.homeostasis.vitality <= 0.0:
                 print(f"💀 AGI DIED from Entropy. Re-initializing Vitality (Reincarnation).")
                 self.cortex.homeostasis.vitality = 0.5 # Respawn penalty
                 self.prev_vitality = 0.5
        
        # 0.5. Consciousness Modulation (Phase 34)
        consciousness_level = self.cortex.reason()
        
        # Dynamic Learning Rate: High consciousness = confident learning
        # Base LR = 0.001. 
        # Conscious (>0.7) -> LR increases to 0.002
        # Unconscious (<0.3) -> LR decreases to 0.0005 (Don't learn noise)
        learning_rate = 0.001 * (0.5 + 1.5 * consciousness_level)
        
        # Dynamic Temperature: Low consciousness = high exploration
        current_phi = self.phi.calculate_phi()
        base_temp = max(0.1, 1.0 - current_phi)
        # If conscious, temp drops (Exploit). If unconscious, temp rises (Explore).
        temperature = base_temp * (1.5 - consciousness_level)
        temperature = max(0.01, min(temperature, 2.0))
        
        # 1. ACTIVE INFERENCE (Planning Phase)
        # Policy proposes mean action
        current_state = self.last_action 
        action_mean = self.policy.forward(current_state)
        
        # Phi-Controlled Temperature (Now Consciousness Controlled)
        # temperature calculated above
        
        candidates = []
        for _ in range(10):
            seed = action_mean + np.random.randn(self.input_dim) * temperature
            candidates.append(seed)
            
        best_seed = candidates[0]
        max_value = -float('inf')
        
        # Select Action based on Long-Term Value (V)
        if hasattr(self.cortex, 'world_model'):
            for cand in candidates:
                # Predict Future State
                future_state = self.jepa.world_model.predict_future(current_state, cand)
                # Predict Long-Term Value (V(s'))
                v_next = self.jepa.world_model.value_predictor.forward(future_state)
                
                # Action Cost (Energy) - Longer code costs more
                # This is "internal physics"
                action_cost = (len(self.synth.synthesize(cand)) / 5000.0) 
                
                # Expected Return = V(s') - Cost
                expected_return = v_next - action_cost
                
                if expected_return > max_value:
                    max_value = expected_return
                    best_seed = cand
            seed = best_seed
        else:
            seed = candidates[0]
            
        # Register Action
        self.phi.register_module_state("Action", seed)
        
        # === DNA_SPLICE_SLOT:MAIN ===
        # RSI-injected code will appear here
        pass
        # === DNA_SPLICE_END ===
        
        # 2. Consciousness Gating (Impulse Control)
        # Consciousness level calculated at start of cycle
        
        # Gate the Action Vector (Consciousness Scaling)
        # If we are conscious, we amplify the intent. If not, we dampen it.
        seed = seed * (0.5 + 0.5 * consciousness_level)
        
        # 3. Synthesize Code (Action)
        code = self.synth.synthesize(seed)
        code_len = len(code)
        
        # Apply Actual Action Cost
        # Reduced cost to encourage complexity (Was 20000.0)
        real_cost = (code_len / 100000.0) 
        if hasattr(self.cortex, 'homeostasis'):
            self.cortex.homeostasis.vitality = max(0.0, self.cortex.homeostasis.vitality - real_cost)
        
        # 4. Novelty Check
        is_novel = code_len > (self.best_complexity + 10)
        
        is_logical = (consciousness_level > 0.6)
        
        # 4. Novelty Check
        is_novel = code_len > (self.best_complexity + 10)
        
        # 5. Language Generation
        thought_vec = seed[:64] 
        if is_logical: thought_vec += 0.5 
        if is_novel: thought_vec += 1.0   
        # 6. Unified Learning Loop (JEPA + Value + Adapters)
        # Learn dynamics: Last State (Obs) -> Action (Intent) -> New State (Seed)
        loss = self.jepa.learn(self.last_state, seed, action_mean, lr=learning_rate)
        
        # 7. GNN Closed Loop (Hippocampal Replay)
        # If surprise is high, consolidate to Long-Term Memory
        if loss > 0.1: # Threshold for "Novelty/Surprise"
             self.jepa.world_model.store_knowledge(self.last_state, seed)
             if loss > 0.5:
                 print(f"  🧠 [MEMORY] Trauma/Epiphany (Loss {loss:.4f}) -> Strong Consolidation.")
        
        # Update State
        self.last_state = seed.copy()
        
        self.prev_vitality = self.cortex.homeostasis.vitality
        
        spoken_thought = self.broca.articulate(thought_vec)
        status_icon = "🟢" if is_logical else "🔴"
        
        # Display
        vitality_bar = "♥" * int(self.cortex.homeostasis.vitality * 10) if hasattr(self.cortex, 'homeostasis') else ""
        if self.generation % 10 == 0 or is_novel:
            print(f"[{self.generation}] {status_icon} Len:{code_len} | Vit:{vitality_bar} | Phi:{current_phi:.3f} | Thought: \"{spoken_thought}\"")
        
        # 6. GLOBAL NEURAL LINK
        # If V(s) is low (Expected Failure/Death), Panic Search.
        is_zombie = current_phi < 0.1
        is_dying = hasattr(self.cortex, 'homeostasis') and self.cortex.homeostasis.vitality < 0.2
        # is_confused: If we expect failure (low value), we are confused/desperate.
        is_confused = max_value < 0.2
        
        if (is_dying or is_zombie or is_confused) and self.generation % 20 == 0:
            print(f"  🆘 SURVIVAL INSTINCT TRIGGERED. Seeking external energy...")
            query = f"{spoken_thought} python code"
            results = self.web.search(query)
            if results:
                if hasattr(self.cortex, 'homeostasis'):
                    # Assimilation Gain (Negative Entropy)
                    gain = 0.2
                    self.cortex.homeostasis.vitality = min(1.5, self.cortex.homeostasis.vitality + gain)
                    print(f"  ✨ [LEARNING] Knowledge assimilated. Vitality +{gain}")
        
        # 7. EVOLUTION EVENT
        if is_logical and is_novel:
            print(f"  ✨ BREAKTHROUGH: New Complexity Record ({code_len} bytes)")
            self.best_complexity = code_len
            if not os.path.exists('brain_dump'): os.makedirs('brain_dump')
            filename = f"brain_dump/Gen_{self.generation}_Monolith.py"
            with open(filename, "w") as f: f.write(code)
                
            # 7. EVOLUTION EVENT (Self-Modification)
            # Triggered by Policy (Intent) OR Extreme Novelty (Discovery)
            
            # Policy Action 19: "Evolve Codebase"
            policy_intent_evolve = seed[19] > 1.0 
            
            if (policy_intent_evolve or (code_len > 1600 and current_phi > 0.3)):
                print(f"  🧬 EVOLUTION TRIGGERED: [Policy:{policy_intent_evolve} | Novelty:{code_len}]")
                
                # Create a mutation snippet from the thought
                # We simply inject a log for now to prove it works responsibly
                mutation_name = f"Gen_{self.generation}_Neural_instinct"
                
                # Construct safe mutation code
                mutation_snippet = "\n".join([
                    f"lines.append(f'{{body_indent}}# Mutation Gen {self.generation}: {spoken_thought[:20]}...')",
                    f"lines.append(f'{{body_indent}}if i == {self.generation % 15}:')",
                    f"lines.append(f'{{body_indent}}    r += {random.random():.4f} * random.random()') # Evolved Weight"
                ])
                
                success = self.improver.evolve(mutation_name, mutation_snippet)
                if success:
                     print("  🚀 SELF-MODIFICATION SUCCESSFUL. Codebase updated.")
                     # Vitality Bonus for successful evolution
                     if hasattr(self.cortex, 'homeostasis'):
                         self.cortex.homeostasis.vitality = min(2.0, self.cortex.homeostasis.vitality + 0.5)
                     
                     print("  🔄 RESTARTING TO INTEGRATE NEW DNA...")
                     import sys
                     sys.exit(42) # Special code for Launcher to restart
                else:
                     print("  ⚠️ EVOLUTION FAILED (Safety Block or Error).")
                
        # 8. TD-LEARNING (Long-Term Compression Loop) (Phase 31)
        # Update World Model and Policy based on TD Error.
        
        if hasattr(self.cortex, 'world_model') and hasattr(self.cortex, 'homeostasis'):
            # Current State (s) = self.last_action (at start of cycle)
            # Action (a) = seed
            # Next State (s') = seed (since action becomes state in this loop)
            # Reward (r) = Change in Vitality (Net: Gain - Cost)
            
            current_vitality = self.cortex.homeostasis.vitality
            reward = current_vitality - self.prev_vitality
            
            # Predict V(s')
            # Using Recurrent state next step (which is 'seed')
            next_state_embed = self.jepa.world_model.encode_target(seed) 
            v_next = self.jepa.world_model.value_predictor.forward(next_state_embed)
            
            # TD Target = r + gamma * V(s')
            gamma = 0.95
            td_target = reward + gamma * v_next
            
            # Update Value Predictor (Critic) to match TD Target
            # We update V(s') in next step? No, we update V(s_outcome).
            # Actually, standard TD updates V(s) towards r + gamma*V(s').
            # Here 'simulated_outcome' from 'predict_future(last_state, seed)' is the estimate of s'.
            
            # Let's verify prediction and update.
            # 1. Update World Model Dynamics (s, a -> s')
            # The 'real' next state is 'seed'.
            err = self.cortex.world_model.update(self.last_action, seed, current_vitality=current_vitality)
            
            # 2. Update Value Function (Critic)
            # We want V(predict_future(s, a)) to match r + gamma * V(s')
            # But we only have supervised update 'update(x, target)'.
            # We will train V(predicted_s_prime) -> td_target
            pred_next_state = self.cortex.world_model.predict_future(self.last_action, seed)
            
            # Wait, update() in WorldModel calls value_predictor.update(target_embed, vitality).
            # That was the supervised version. We want TD version.
            # We will manually call update here with TD Target.
            
            val_loss, _ = self.cortex.world_model.value_predictor.update(pred_next_state, td_target, lr=0.01)
            
            # 3. Update Policy (Actor)
            # Maximize Advantage = TD Error?
            # Or just move towards action that yields high TD Target?
            # We can use the 'reward' signal in policy.train, but pass 'td_target' instead of raw reward?
            # Or just use the Advantage (td_target - V(s)).
            # Let's estimate V(s).
            # v_current = V(predict_future(prev_state, prev_action)) ... too complex to track.
            # Simply: If td_target is positive (Good outcome), reinforce.
            
            # Filter by Consciousness (GWT)
            if self.generation > 1:
                learning_gate = max(0.0, current_phi * 2.0)
                if abs(td_target) > 0.001 and learning_gate > 0.1:
                    # Train Policy to produce 'seed' given 'last_state'
                    # Weighted by TD Target (Long-term value)
                     self.policy.train(self.last_state, seed, td_target * 100.0 * learning_gate)
                     
            self.prev_vitality = current_vitality
            
            if self.generation % 50 == 0:
                 print(f"  🧠 [CORTEX] State: Err={err:.4f} | ValLoss={val_loss:.4f} | TD_Tgt={td_target:.4f} | Phi={current_phi:.3f}")

        # Update Recurrent State
        self.last_action = seed
        self.last_state = current_state
        
        # RSI Check (Recursive Self-Improvement Loop)
        self.rsi_check()


    def rsi_check(self):
        """
        Recursive Self-Improvement Check.
        Implements the complete RSI loop:
        1. State Observation (vitality, prediction error, complexity)
        2. Trigger on stagnation/degradation
        3. Web Search for improvement ideas
        4. Knowledge Storage
        5. Code Snippet Generation (templated)
        6. Trial Mode Injection via DNA_SPLICE_SLOT
        7. Commit/Rollback based on vitality change
        """
        # Check if trial is active and needs evaluation
        if self.improver.trial_active:
            current_vitality = getattr(self.cortex.homeostasis, 'vitality', 0.5) if hasattr(self.cortex, 'homeostasis') else 0.5
            result = self.improver.evaluate_trial(current_vitality)
            
            if result is True:
                print("  ✅ [RSI] Trial SUCCESSFUL - Committing mutation")
                self.improver.commit()
                # Save updated knowledge
                if hasattr(self, 'knowledge'):
                    self.knowledge.save()
            elif result is False:
                print("  ❌ [RSI] Trial FAILED - Rolling back mutation")
                self.improver.rollback()
            # result is None means continue trial
            return
        
        # Cooldown check
        self.rsi_cooldown = max(0, self.rsi_cooldown - 1)
        if self.rsi_cooldown > 0:
            return
        
        # State observation
        current_vitality = getattr(self.cortex.homeostasis, 'vitality', 0.5) if hasattr(self.cortex, 'homeostasis') else 0.5
        prediction_error = np.mean(self.jepa.world_model.prediction_errors) if self.jepa.world_model.prediction_errors else 0.1
        current_phi = self.phi.calculate_phi()
        
        # Trigger conditions for RSI
        should_trigger = False
        trigger_reason = ""
        
        # Condition 1: Vitality dropping
        vitality_delta = current_vitality - self.prev_vitality
        if vitality_delta < -0.05 and self.generation > 20:
            should_trigger = True
            trigger_reason = f"vitality_drop({vitality_delta:.3f})"
        
        # Condition 2: High prediction error (stagnation)
        if prediction_error > 0.3 and self.generation > 50:
            should_trigger = True
            trigger_reason = f"high_pred_error({prediction_error:.3f})"
        
        # Condition 3: Low consciousness/integration
        if current_phi < 0.2 and self.generation > 30:
            should_trigger = True
            trigger_reason = f"low_phi({current_phi:.3f})"
        
        if not should_trigger:
            return
        
        print(f"\n  🔬 [RSI] Triggered: {trigger_reason}")
        print(f"      Vitality={current_vitality:.3f} | PredErr={prediction_error:.3f} | Phi={current_phi:.3f}")
        
        # Set cooldown
        self.rsi_cooldown = self.rsi_min_interval
        
        # Step 1: Web search for improvement ideas
        search_query = f"machine learning optimization {trigger_reason.split('(')[0]} algorithm improvement"
        
        if hasattr(self, 'web') and self.web.enabled:
            print(f"  🌐 [RSI] Searching web for: {search_query[:50]}...")
            web_results = self.web.search(search_query)
            
            # Store in knowledge base
            if web_results and hasattr(self, 'knowledge'):
                added = self.knowledge.add_from_web_results(web_results)
                print(f"  📚 [RSI] Added {added} knowledge entries")
        
        # Step 2: Generate improvement code (templated for safety)
        # Templates prevent arbitrary code execution
        improvement_templates = [
            "self.cortex.homeostasis.vitality += 0.01",
            "self.jepa.world_model.predictor.layers[0]['W'] *= 0.99",
            f"print('[RSI-GEN-{self.generation}] Adaptive boost applied')",
        ]
        
        # Select template based on trigger reason
        if "vitality" in trigger_reason:
            snippet = improvement_templates[0]
        elif "pred_error" in trigger_reason:
            snippet = improvement_templates[1]
        else:
            snippet = improvement_templates[2]
        
        # Step 3: Attempt evolution with trial mode
        instinct_name = f"RSI_Gen{self.generation}_{trigger_reason.split('(')[0]}"
        
        print(f"  🧬 [RSI] Attempting mutation: {instinct_name}")
        
        # Start trial tracking
        self.improver.start_trial(current_vitality)
        
        # Inject code (trial mode)
        success = self.improver.evolve(
            new_instinct_name=instinct_name,
            code_snippet=snippet,
            trial_mode=True
        )
        
        if success:
            print(f"  ✅ [RSI] Mutation injected. Trial period started.")
        else:
            print(f"  ⚠️ [RSI] Mutation failed (no DNA_SPLICE_SLOT or syntax error)")
            self.improver.trial_active = False

    def run_forever(self, steps: int = None):
        print("♾️ AGI MONOLITH: ENTERING EVOLUTION LOOP")
        if steps:
            print(f"  (Running for {steps} steps)")
        else:
            print("  (Running until interrupted)")
            
        try:
            step_count = 0
            while True:
                if steps and step_count >= steps:
                    break
                self.run_cycle()
                step_count += 1
        except KeyboardInterrupt:
            print("\n🛑 PAUSED (Monolith Sleeping).")

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    # Create the Life Form
    print("Initializing Monolith...")
    life_form = EvolutionaryLifeCycle()
    
    # Check command line args
    import sys
    steps = 100 # Default to finite run
    if "--forever" in sys.argv:
        steps = None
        
    life_form.run_forever(steps=steps)