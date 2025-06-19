"""
Unit tests for data generation functions.
"""
import pytest
import torch
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import grokking_data_torch


class TestGrokkingDataTorch:
    """Test the grokking_data_torch function."""
    
    def test_basic_functionality(self):
        """Test basic data generation functionality."""
        p = 11
        op = '/'
        train_fraction = 0.5
        
        X_train, T_train, X_test, T_test = grokking_data_torch(p, op, train_fraction)
        
        # Check types
        assert isinstance(X_train, torch.Tensor)
        assert isinstance(T_train, torch.Tensor)
        assert isinstance(X_test, torch.Tensor)
        assert isinstance(T_test, torch.Tensor)
        
        # Check dtypes
        assert X_train.dtype == torch.long
        assert T_train.dtype == torch.long
        assert X_test.dtype == torch.long
        assert T_test.dtype == torch.long
        
        # Check shapes
        assert X_train.shape[1] == 4  # [a, op, b, '=']
        assert len(T_train.shape) == 1
        assert X_test.shape[1] == 4
        assert len(T_test.shape) == 1
        
        # Check lengths match
        assert X_train.shape[0] == T_train.shape[0]
        assert X_test.shape[0] == T_test.shape[0]
    
    def test_different_operations(self):
        """Test all supported operations."""
        p = 7
        operations = ['*', '/', '+', '-']
        
        for op in operations:
            X_train, T_train, X_test, T_test = grokking_data_torch(p, op, 0.3)
            
            # Basic checks
            assert X_train.shape[0] > 0
            assert T_train.shape[0] > 0
            assert X_test.shape[0] > 0
            assert T_test.shape[0] > 0
            
            # Check that operation embedding is correct
            embed_val = p  # all operations map to p
            assert torch.all(X_train[:, 1] == embed_val)
            assert torch.all(X_test[:, 1] == embed_val)
            
            # Check equals sign embedding
            assert torch.all(X_train[:, 3] == p + 1)
            assert torch.all(X_test[:, 3] == p + 1)
    
    def test_train_fraction(self):
        """Test different train fractions."""
        p = 5
        op = '+'
        
        # Test 50-50 split
        X_train, T_train, X_test, T_test = grokking_data_torch(p, op, 0.5)
        total_size = X_train.shape[0] + X_test.shape[0]
        train_size = X_train.shape[0]
        
        # Should be approximately 50% (allowing for rounding)
        assert abs(train_size / total_size - 0.5) < 0.1
        
        # Test 80-20 split
        X_train, T_train, X_test, T_test = grokking_data_torch(p, op, 0.8)
        total_size = X_train.shape[0] + X_test.shape[0]
        train_size = X_train.shape[0]
        
        # Should be approximately 80%
        assert abs(train_size / total_size - 0.8) < 0.1
    
    def test_division_operation_excludes_zero(self):
        """Test that division operation excludes zero in denominator."""
        p = 5
        op = '/'
        
        X_train, T_train, X_test, T_test = grokking_data_torch(p, op, 0.5)
        
        # Combine all data
        X_all = torch.cat([X_train, X_test])
        
        # Check that second operand (b) is never 0 for division
        # X format: [a, op_embed, b, equals_embed]
        b_values = X_all[:, 2]
        assert torch.all(b_values > 0), "Division should exclude zero denominator"
    
    def test_other_operations_include_zero(self):
        """Test that non-division operations can include zero."""
        p = 5
        operations = ['*', '+', '-']
        
        for op in operations:
            X_train, T_train, X_test, T_test = grokking_data_torch(p, op, 0.5)
            X_all = torch.cat([X_train, X_test])
            
            # Check that both operands can be zero
            a_values = X_all[:, 0]
            b_values = X_all[:, 2]
            
            # At least some zeros should exist for small p
            assert 0 in a_values or 0 in b_values, f"Operation {op} should allow zero operands"
    
    def test_value_ranges(self):
        """Test that values are within expected ranges."""
        p = 7
        op = '*'
        
        X_train, T_train, X_test, T_test = grokking_data_torch(p, op, 0.5)
        
        # Operands should be in range [0, p-1]
        assert torch.all(X_train[:, 0] >= 0) and torch.all(X_train[:, 0] < p)
        assert torch.all(X_train[:, 2] >= 0) and torch.all(X_train[:, 2] < p)
        
        # Results should be in range [0, p-1] due to modular arithmetic
        assert torch.all(T_train >= 0) and torch.all(T_train < p)
        assert torch.all(T_test >= 0) and torch.all(T_test < p)
    
    def test_modular_arithmetic_correctness(self):
        """Test that the arithmetic operations are computed correctly."""
        p = 5
        
        # Test a few specific cases
        test_cases = [
            ('*', 2, 3, (2 * 3) % p),
            ('+', 3, 4, (3 + 4) % p),
            ('-', 1, 3, (1 - 3) % p),
        ]
        
        for op, a, b, expected in test_cases:
            X_train, T_train, X_test, T_test = grokking_data_torch(p, op, 0.5)
            
            # Find the specific case in the data
            X_all = torch.cat([X_train, X_test])
            T_all = torch.cat([T_train, T_test])
            
            mask = (X_all[:, 0] == a) & (X_all[:, 2] == b)
            if torch.any(mask):
                actual = T_all[mask][0].item()
                assert actual == expected, f"For {a} {op} {b} mod {p}, expected {expected}, got {actual}"
    
    def test_device_parameter(self):
        """Test that device parameter works correctly."""
        p = 5
        op = '+'
        
        # Test CPU device (default)
        X_train, T_train, X_test, T_test = grokking_data_torch(p, op, 0.5, device='cpu')
        assert X_train.device.type == 'cpu'
        assert T_train.device.type == 'cpu'
        
        # Test device consistency
        devices = [X_train.device, T_train.device, X_test.device, T_test.device]
        assert all(d == devices[0] for d in devices), "All tensors should be on same device"
    
    def test_invalid_operation(self):
        """Test that invalid operations raise ValueError."""
        p = 5
        
        with pytest.raises(ValueError, match="Unsupported operation"):
            grokking_data_torch(p, '^', 0.5)  # Invalid operation
    
    def test_reproducibility_with_seed(self):
        """Test that results are reproducible when numpy seed is set."""
        p = 7
        op = '*'
        
        # Set seed and generate data
        np.random.seed(42)
        X1_train, T1_train, X1_test, T1_test = grokking_data_torch(p, op, 0.5)
        
        # Reset seed and generate again
        np.random.seed(42)
        X2_train, T2_train, X2_test, T2_test = grokking_data_torch(p, op, 0.5)
        
        # Should be identical
        assert torch.equal(X1_train, X2_train)
        assert torch.equal(T1_train, T2_train)
        assert torch.equal(X1_test, X2_test)
        assert torch.equal(T1_test, T2_test)


if __name__ == '__main__':
    pytest.main([__file__])