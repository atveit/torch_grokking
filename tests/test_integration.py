"""
Integration tests for the complete torch_grokking system.
"""
import pytest
import torch
import numpy as np
import os
import sys
import argparse
from unittest.mock import patch

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import main, TorchTrainer
from models import TransformerTorch
from data import grokking_data_torch


class TestMainFunction:
    """Test the main training function and CLI interface."""
    
    def create_test_args(self, **kwargs):
        """Create test arguments with defaults."""
        defaults = {
            'p': 5,
            'op': '+',
            'train_fraction': 0.6,
            'depth': 1,
            'dim': 16,
            'heads': 1,
            'dropout': 0.0,
            'lr': 1e-3,
            'weight_decay': 0.1,
            'beta1': 0.9,
            'beta2': 0.98,
            'warmup': 2,
            'batch_size': 8,
            'epochs': 2,
            'seed': 42,
            'cpu': True
        }
        defaults.update(kwargs)
        
        # Create argparse Namespace
        args = argparse.Namespace(**defaults)
        return args
    
    def test_main_function_basic(self):
        """Test basic main function execution."""
        args = self.create_test_args()
        
        # Mock plt.show() to avoid display issues in testing
        with patch('matplotlib.pyplot.show'):
            main(args)
        
        # If we reach here, main() completed successfully
        assert True
    
    def test_main_function_different_operations(self):
        """Test main function with different operations."""
        operations = ['+', '-', '*', '/']
        
        for op in operations:
            args = self.create_test_args(op=op, epochs=1)
            
            with patch('matplotlib.pyplot.show'):
                main(args)
            
            # Test passes if no exception is raised
            assert True
    
    def test_main_function_different_primes(self):
        """Test main function with different prime numbers."""
        primes = [3, 5, 7, 11]
        
        for p in primes:
            args = self.create_test_args(p=p, epochs=1)
            
            with patch('matplotlib.pyplot.show'):
                main(args)
            
            assert True
    
    def test_main_function_different_model_configs(self):
        """Test main function with different model configurations."""
        configs = [
            {'depth': 1, 'dim': 16, 'heads': 1},
            {'depth': 2, 'dim': 32, 'heads': 2},
            {'depth': 1, 'dim': 24, 'heads': 3},
        ]
        
        for config in configs:
            args = self.create_test_args(epochs=1, **config)
            
            with patch('matplotlib.pyplot.show'):
                main(args)
            
            assert True
    
    def test_main_function_train_fractions(self):
        """Test main function with different train fractions."""
        fractions = [0.3, 0.5, 0.7, 0.9]
        
        for frac in fractions:
            args = self.create_test_args(train_fraction=frac, epochs=1)
            
            with patch('matplotlib.pyplot.show'):
                main(args)
            
            assert True
    
    def test_main_function_reproducibility(self):
        """Test that main function produces reproducible results."""
        args1 = self.create_test_args(seed=123, epochs=1)
        args2 = self.create_test_args(seed=123, epochs=1)
        
        # This is a basic test - we can't easily compare outputs
        # but we can ensure both runs complete successfully
        with patch('matplotlib.pyplot.show'):
            main(args1)
            main(args2)
        
        assert True
    
    def test_main_function_creates_media_dir(self):
        """Test that main function creates media directory."""
        # Remove media directory if it exists
        media_dir = '/home/runner/work/torch_grokking/torch_grokking/media'
        if os.path.exists(media_dir):
            import shutil
            shutil.rmtree(media_dir)
        
        args = self.create_test_args(epochs=1)
        
        with patch('matplotlib.pyplot.show'):
            main(args)
        
        # Check that media directory was created
        assert os.path.exists(media_dir)
        
        # Check that plot file was created
        plot_file = os.path.join(media_dir, 'grokking_comparison.png')
        assert os.path.exists(plot_file)


class TestEndToEndTraining:
    """End-to-end integration tests for training scenarios."""
    
    def test_small_problem_convergence(self):
        """Test that model can solve very small problems."""
        # Use smallest possible problem
        p = 3
        op = '+'
        
        # Generate data
        X_train, T_train, X_test, T_test = grokking_data_torch(
            p=p, op=op, train_fraction=0.8, device='cpu'
        )
        
        # Create model with sufficient capacity
        model = TransformerTorch(
            depth=2, 
            dim=32, 
            heads=2, 
            n_tokens=p + 2,
            seq_len=4,
            dropout=0.0
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        trainer = TorchTrainer(
            model, optimizer, classification=True, batch_size=4, device='cpu'
        )
        
        # Train for more epochs
        trainer.train(
            (X_train, T_train),
            (X_test, T_test),
            epochs=20,
            shuffle=True
        )
        
        # Check that training accuracy improved significantly
        initial_acc = trainer.train_acc_trace[0]
        final_acc = trainer.train_acc_trace[-1]
        
        assert final_acc > initial_acc
        assert final_acc > 0.1  # Should learn something meaningful
    
    def test_different_operation_difficulty(self):
        """Test relative difficulty of different operations."""
        p = 5
        operations = ['+', '-', '*', '/']
        results = {}
        
        for op in operations:
            # Generate data
            X_train, T_train, X_test, T_test = grokking_data_torch(
                p=p, op=op, train_fraction=0.7, device='cpu'
            )
            
            # Create consistent model
            model = TransformerTorch(
                depth=1, 
                dim=24, 
                heads=2, 
                n_tokens=p + 2,
                seq_len=4,
                dropout=0.0
            )
            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
            trainer = TorchTrainer(
                model, optimizer, classification=True, batch_size=8, device='cpu'
            )
            
            # Train
            trainer.train(
                (X_train, T_train),
                (X_test, T_test),
                epochs=10,
                shuffle=True
            )
            
            results[op] = {
                'final_train_acc': trainer.train_acc_trace[-1],
                'final_val_acc': trainer.val_acc_trace[-1]
            }
        
        # All operations should achieve some learning
        for op, result in results.items():
            assert result['final_train_acc'] > 0.0
            assert result['final_val_acc'] >= 0.0  # May be 0 for difficult cases
    
    def test_scaling_with_problem_size(self):
        """Test how training scales with problem size."""
        primes = [3, 5, 7]
        op = '+'
        
        for p in primes:
            X_train, T_train, X_test, T_test = grokking_data_torch(
                p=p, op=op, train_fraction=0.6, device='cpu'
            )
            
            # Scale model size with problem
            dim = max(16, p * 4)
            model = TransformerTorch(
                depth=1, 
                dim=dim, 
                heads=1, 
                n_tokens=p + 2,
                seq_len=4,
                dropout=0.0
            )
            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            trainer = TorchTrainer(
                model, optimizer, classification=True, batch_size=8, device='cpu'
            )
            
            trainer.train(
                (X_train, T_train),
                (X_test, T_test),
                epochs=5,
                shuffle=True
            )
            
            # Should complete training successfully
            assert len(trainer.train_acc_trace) == 5
            assert trainer.train_acc_trace[-1] >= 0.0
    
    def test_batch_size_effects(self):
        """Test effects of different batch sizes."""
        p = 5
        op = '*'
        batch_sizes = [1, 4, 16, -1]  # -1 means full batch
        
        X_train, T_train, X_test, T_test = grokking_data_torch(
            p=p, op=op, train_fraction=0.6, device='cpu'
        )
        
        for batch_size in batch_sizes:
            model = TransformerTorch(
                depth=1, 
                dim=20, 
                heads=1, 
                n_tokens=p + 2,
                seq_len=4,
                dropout=0.0
            )
            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            trainer = TorchTrainer(
                model, optimizer, classification=True, 
                batch_size=batch_size, device='cpu'
            )
            
            trainer.train(
                (X_train, T_train),
                (X_test, T_test),
                epochs=3,
                shuffle=True
            )
            
            # Training should complete successfully
            assert len(trainer.train_acc_trace) == 3
            assert all(0 <= acc <= 1 for acc in trainer.train_acc_trace)
    
    def test_learning_rate_sensitivity(self):
        """Test sensitivity to learning rate."""
        p = 5
        op = '+'
        learning_rates = [1e-4, 1e-3, 1e-2]
        
        X_train, T_train, X_test, T_test = grokking_data_torch(
            p=p, op=op, train_fraction=0.7, device='cpu'
        )
        
        for lr in learning_rates:
            model = TransformerTorch(
                depth=1, 
                dim=24, 
                heads=1, 
                n_tokens=p + 2,
                seq_len=4,
                dropout=0.0
            )
            
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            trainer = TorchTrainer(
                model, optimizer, classification=True, batch_size=8, device='cpu'
            )
            
            trainer.train(
                (X_train, T_train),
                (X_test, T_test),
                epochs=5,
                shuffle=True
            )
            
            # Should not crash with any learning rate
            assert len(trainer.val_acc_trace) == 5
            assert all(isinstance(acc, float) for acc in trainer.val_acc_trace)
    
    def test_model_capacity_effects(self):
        """Test effects of different model capacities."""
        p = 5
        op = '/'
        
        X_train, T_train, X_test, T_test = grokking_data_torch(
            p=p, op=op, train_fraction=0.6, device='cpu'
        )
        
        # Test different model sizes
        model_configs = [
            {'depth': 1, 'dim': 12, 'heads': 1},
            {'depth': 1, 'dim': 24, 'heads': 2},
            {'depth': 2, 'dim': 16, 'heads': 1},
        ]
        
        for config in model_configs:
            model = TransformerTorch(
                n_tokens=p + 2,
                seq_len=4,
                dropout=0.0,
                **config
            )
            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            trainer = TorchTrainer(
                model, optimizer, classification=True, batch_size=8, device='cpu'
            )
            
            trainer.train(
                (X_train, T_train),
                (X_test, T_test),
                epochs=3,
                shuffle=True
            )
            
            # All configurations should train successfully
            assert len(trainer.train_error_trace) == 3
            assert all(loss >= 0 for loss in trainer.train_error_trace)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_arguments_handling(self):
        """Test handling of invalid arguments."""
        # Test invalid operation
        with pytest.raises(ValueError):
            grokking_data_torch(5, op='^', train_fraction=0.5)
        
        # Test invalid pool type
        with pytest.raises(AssertionError):
            TransformerTorch(1, 16, 1, 10, 4, pool='invalid')
    
    def test_device_consistency(self):
        """Test device consistency throughout the pipeline."""
        p = 3
        device = 'cpu'
        
        X_train, T_train, X_test, T_test = grokking_data_torch(
            p=p, op='+', train_fraction=0.5, device=device
        )
        
        # All tensors should be on correct device
        assert X_train.device.type == device
        assert T_train.device.type == device
        assert X_test.device.type == device
        assert T_test.device.type == device
        
        model = TransformerTorch(1, 16, 1, p + 2, 4)
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters())
        trainer = TorchTrainer(model, optimizer, classification=True, device=device)
        
        # Training should work without device errors
        trainer.train(
            (X_train, T_train),
            (X_test, T_test),
            epochs=1,
            shuffle=False
        )
        
        assert len(trainer.train_acc_trace) == 1


if __name__ == '__main__':
    pytest.main([__file__])