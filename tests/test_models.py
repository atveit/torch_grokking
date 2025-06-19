"""
Unit tests for model components.
"""
import pytest
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    RMSNormTorch, 
    RoPETorch, 
    AttentionTorch, 
    FeedForwardTorch, 
    BlockTorch, 
    TransformerTorch
)


class TestRMSNormTorch:
    """Test the RMSNorm implementation."""
    
    def test_initialization(self):
        """Test RMSNorm initialization."""
        dim = 64
        norm = RMSNormTorch(dim)
        
        assert norm.weight.shape == (dim,)
        assert torch.allclose(norm.weight, torch.ones(dim))
        assert norm.eps == 1e-5
    
    def test_forward_pass(self):
        """Test RMSNorm forward pass."""
        dim = 32
        norm = RMSNormTorch(dim)
        
        # Test 2D input (batch, dim)
        x = torch.randn(4, dim)
        output = norm(x)
        
        assert output.shape == x.shape
        assert not torch.allclose(output, x)  # Should be different from input
        
        # Test 3D input (batch, seq, dim)
        x = torch.randn(4, 8, dim)
        output = norm(x)
        
        assert output.shape == x.shape
    
    def test_rms_normalization_property(self):
        """Test that RMS normalization works correctly."""
        dim = 16
        norm = RMSNormTorch(dim, eps=0.0)  # No epsilon for exact test
        
        # Create input with known RMS
        x = torch.ones(2, dim) * 2.0  # RMS = 2.0
        output = norm(x)
        
        # After normalization, RMS should be close to 1.0 (times weight)
        rms_output = torch.sqrt(torch.mean(output**2, dim=-1))
        expected_rms = torch.ones(2)  # weight is initialized to 1
        
        assert torch.allclose(rms_output, expected_rms, atol=1e-6)


class TestRoPETorch:
    """Test the Rotary Position Embedding implementation."""
    
    def test_initialization(self):
        """Test RoPE initialization."""
        dim_head = 64
        rope = RoPETorch(dim_head)
        
        assert hasattr(rope, 'rope')
        assert rope.rope is not None
    
    def test_forward_pass(self):
        """Test RoPE forward pass."""
        dim_head = 32
        rope = RoPETorch(dim_head)
        
        # Input shape: (batch, seq, heads, dim_head)
        batch_size, seq_len, heads = 2, 8, 4
        x = torch.randn(batch_size, seq_len, heads, dim_head)
        
        output = rope(x)
        
        assert output.shape == x.shape
        assert not torch.allclose(output, x)  # Should apply rotation


class TestAttentionTorch:
    """Test the Attention module implementation."""
    
    def test_initialization(self):
        """Test Attention initialization."""
        dim = 128
        heads = 8
        dim_head = 16
        dropout = 0.1
        
        attn = AttentionTorch(dim, heads, dim_head, dropout)
        
        assert attn.heads == heads
        assert attn.scale == dim_head ** -0.5
        assert hasattr(attn, 'wq')
        assert hasattr(attn, 'wk')
        assert hasattr(attn, 'wv')
        assert hasattr(attn, 'wo')
    
    def test_forward_pass(self):
        """Test Attention forward pass."""
        dim = 64
        heads = 4
        dim_head = 16
        seq_len = 8
        batch_size = 2
        
        attn = AttentionTorch(dim, heads, dim_head)
        x = torch.randn(batch_size, seq_len, dim)
        
        output = attn(x)
        
        assert output.shape == (batch_size, seq_len, dim)
    
    def test_causal_masking(self):
        """Test that causal masking works."""
        dim = 32
        heads = 2
        dim_head = 16
        seq_len = 4
        batch_size = 1
        
        attn = AttentionTorch(dim, heads, dim_head)
        
        # Create mask
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        
        x = torch.randn(batch_size, seq_len, dim)
        output = attn(x, mask=mask)
        
        assert output.shape == (batch_size, seq_len, dim)


class TestFeedForwardTorch:
    """Test the FeedForward module implementation."""
    
    def test_initialization(self):
        """Test FeedForward initialization."""
        dim = 128
        mlp_dim = 512
        dropout = 0.1
        
        ff = FeedForwardTorch(dim, mlp_dim, dropout)
        
        assert hasattr(ff, 'w1')
        assert hasattr(ff, 'w2')
        assert hasattr(ff, 'w3')
        assert hasattr(ff, 'norm')
    
    def test_forward_pass(self):
        """Test FeedForward forward pass."""
        dim = 64
        mlp_dim = 256
        batch_size = 2
        seq_len = 8
        
        ff = FeedForwardTorch(dim, mlp_dim)
        x = torch.randn(batch_size, seq_len, dim)
        
        output = ff(x)
        
        assert output.shape == (batch_size, seq_len, dim)


class TestBlockTorch:
    """Test the Transformer Block implementation."""
    
    def test_initialization(self):
        """Test Block initialization."""
        dim = 128
        heads = 8
        dim_head = 16
        mlp_dim = 512
        seq_len = 16
        dropout = 0.1
        
        block = BlockTorch(dim, heads, dim_head, mlp_dim, seq_len, dropout)
        
        assert hasattr(block, 'attn')
        assert hasattr(block, 'ff')
        assert hasattr(block, '_mask')
    
    def test_causal_mask_creation(self):
        """Test causal mask creation."""
        dim = 32
        seq_len = 4
        
        block = BlockTorch(dim, 1, 32, 128, seq_len, 0.0)
        mask = block._mask
        
        # Check mask shape
        assert mask.shape == (seq_len, seq_len)
        
        # Check causal property (upper triangular with -inf)
        expected_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        assert torch.equal(mask, expected_mask)
    
    def test_forward_pass(self):
        """Test Block forward pass."""
        dim = 64
        heads = 4
        dim_head = 16
        mlp_dim = 256
        seq_len = 8
        batch_size = 2
        
        block = BlockTorch(dim, heads, dim_head, mlp_dim, seq_len, 0.0)
        x = torch.randn(batch_size, seq_len, dim)
        
        output = block(x)
        
        assert output.shape == (batch_size, seq_len, dim)
    
    def test_residual_connections(self):
        """Test that residual connections work correctly."""
        dim = 32
        seq_len = 4
        batch_size = 1
        
        block = BlockTorch(dim, 1, 32, 128, seq_len, 0.0)
        
        # Use small random input to test residual connections
        x = torch.randn(batch_size, seq_len, dim) * 0.01
        output = block(x)
        
        # Output should be different from input due to transformations
        assert not torch.allclose(output, x, atol=1e-6)
        assert output.shape == x.shape


class TestTransformerTorch:
    """Test the complete Transformer model."""
    
    def test_initialization(self):
        """Test Transformer initialization."""
        depth = 4
        dim = 128
        heads = 8
        n_tokens = 1000
        seq_len = 16
        dropout = 0.1
        
        model = TransformerTorch(depth, dim, heads, n_tokens, seq_len, dropout)
        
        assert len(model.layers) == depth
        assert hasattr(model, 'embedding')
        assert hasattr(model, 'norm')
        assert hasattr(model, 'out')
        assert model.pool == 'cls'  # default
    
    def test_embedding_layer(self):
        """Test embedding layer properties."""
        n_tokens = 100
        dim = 64
        
        model = TransformerTorch(1, dim, 1, n_tokens, 4)
        
        assert model.embedding.num_embeddings == n_tokens
        assert model.embedding.embedding_dim == dim
    
    def test_output_layer(self):
        """Test output layer properties."""
        n_tokens = 100
        dim = 64
        
        model = TransformerTorch(1, dim, 1, n_tokens, 4)
        
        assert model.out.in_features == dim
        assert model.out.out_features == n_tokens
        assert model.out.bias is None  # bias=False
    
    def test_forward_pass_cls_pooling(self):
        """Test forward pass with CLS pooling."""
        depth = 2
        dim = 32
        heads = 2
        n_tokens = 50
        seq_len = 6
        batch_size = 3
        
        model = TransformerTorch(depth, dim, heads, n_tokens, seq_len, pool='cls')
        
        # Input tokens
        x = torch.randint(0, n_tokens, (batch_size, seq_len))
        output = model(x)
        
        assert output.shape == (batch_size, n_tokens)
    
    def test_forward_pass_mean_pooling(self):
        """Test forward pass with mean pooling."""
        depth = 2
        dim = 32
        heads = 2
        n_tokens = 50
        seq_len = 6
        batch_size = 3
        
        model = TransformerTorch(depth, dim, heads, n_tokens, seq_len, pool='mean')
        
        # Input tokens
        x = torch.randint(0, n_tokens, (batch_size, seq_len))
        output = model(x)
        
        assert output.shape == (batch_size, n_tokens)
    
    def test_token_range_validation(self):
        """Test that input tokens are within valid range."""
        n_tokens = 20
        model = TransformerTorch(1, 32, 1, n_tokens, 4)
        
        # Valid tokens
        x_valid = torch.randint(0, n_tokens, (2, 4))
        output = model(x_valid)
        assert output.shape == (2, n_tokens)
        
        # Tokens at boundary
        x_boundary = torch.full((2, 4), n_tokens - 1)
        output = model(x_boundary)
        assert output.shape == (2, n_tokens)
    
    def test_different_depths(self):
        """Test models with different depths."""
        dims = [32, 64]
        depths = [1, 2, 4]
        
        for dim in dims:
            for depth in depths:
                model = TransformerTorch(depth, dim, 1, 10, 4)
                x = torch.randint(0, 10, (1, 4))
                output = model(x)
                
                assert output.shape == (1, 10)
                assert len(model.layers) == depth
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = TransformerTorch(2, 32, 1, 10, 4)
        x = torch.randint(0, 10, (2, 4))
        target = torch.randint(0, 10, (2,))
        
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        
        # Check that gradients exist for key parameters
        assert model.embedding.weight.grad is not None
        assert model.out.weight.grad is not None
        
        # Check that gradients are non-zero
        assert not torch.allclose(model.embedding.weight.grad, torch.zeros_like(model.embedding.weight.grad))


if __name__ == '__main__':
    pytest.main([__file__])