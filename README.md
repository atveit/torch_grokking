# PyTorch Grokking Implementation

This is a PyTorch implementation of the grokking phenomenon, where
neural networks exhibit a phase transition in generalization
performance during training.

**NOTE: this repo is a Pytorch port of the Grokking Modular Arithmetic - written in MLX by [Jason Stock](https://github.com/stockeh) - available on [https://github.com/stockeh/mlx-grokking](https://github.com/stockeh/mlx-grokking)**

## Default Usage 

```bash
python main.py 
```
This should generate a plot similar to the one below.

## Overview

The implementation includes:
- Transformer-based architecture with RMSNorm and RoPE
- Customizable model parameters (depth, dimensions, heads)
- Learning rate warmup scheduler
- Training progress visualization
- Comprehensive test suite

## Example Training Progress

Below is an example of the grokking phenomenon, where the model suddenly "groks" the underlying pattern and generalizes well to the test set:

![Training Progress](media/grokking_run_example.png)

## Usage Options

```bash
python main.py [--op /] [--p 97] [--train-fraction 0.5] [--depth 2] [--dim 128] [--heads 1] [--dropout 0.2] [--epochs 150] [--batch_size 512] [--lr 1e-3] [--weight-decay 1.0] [--beta1 0.9] [--beta2 0.98] [--warmup 10] [--cpu]
```

### Parameters

- `--p`: Prime number for modular arithmetic (default: 97)
- `--op`: Operation to learn (+, -, *, /) (default: /)
- `--train-fraction`: Fraction of data for training (default: 0.5)
- `--depth`: Number of transformer layers (default: 2)
- `--dim`: Model dimension (default: 128)
- `--heads`: Number of attention heads (default: 1)
- `--dropout`: Dropout rate (default: 0.2)
- `--epochs`: Number of training epochs (default: 150)
- `--batch_size`: Batch size (default: 512)
- `--lr`: Learning rate (default: 1e-3)
- `--weight-decay`: Weight decay (default: 1.0)
- `--cpu`: Force CPU usage (default: False)

## Testing

The project includes a comprehensive test suite covering unit tests and integration tests.

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_data.py -v
python -m pytest tests/test_models.py -v
python -m pytest tests/test_trainer.py -v
python -m pytest tests/test_integration.py -v

# Run tests with coverage (if pytest-cov is installed)
python -m pytest tests/ --cov=. --cov-report=html
```

### Test Coverage

The test suite includes:

#### Unit Tests
- **Data Generation (`test_data.py`)**: Tests for `grokking_data_torch` function
  - Different arithmetic operations (+, -, *, /)
  - Train/test splitting
  - Data format validation
  - Modular arithmetic correctness
  - Device handling

- **Model Components (`test_models.py`)**: Tests for neural network modules
  - `RMSNormTorch`: Layer normalization
  - `RoPETorch`: Rotary position embeddings
  - `AttentionTorch`: Self-attention mechanism
  - `FeedForwardTorch`: Feed-forward networks
  - `BlockTorch`: Transformer blocks
  - `TransformerTorch`: Complete model

- **Training (`test_trainer.py`)**: Tests for training functionality
  - `TorchTrainer` class initialization and configuration
  - Batch creation and data handling
  - Training and evaluation loops
  - Loss computation and metrics
  - Device consistency

#### Integration Tests
- **End-to-End Training (`test_integration.py`)**: Complete pipeline tests
  - Main function execution with different parameters
  - Training convergence on small problems
  - Model scaling with problem size
  - Different operation difficulty
  - Hyperparameter sensitivity
  - Error handling

### Test Requirements

The tests require the following additional dependencies:
- `pytest>=7.0.0` (included in requirements.txt)

Install test dependencies:
```bash
pip install -r requirements.txt
```

### Continuous Integration

All tests are designed to run quickly (< 20 seconds total) using small model sizes and few training epochs, making them suitable for CI/CD pipelines.

## Architecture

The model uses:
- Transformer architecture with causal attention
- RMSNorm for layer normalization
- Rotary Position Embeddings (RoPE)
- AdamW optimizer with weight decay
- Learning rate warmup schedule

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch >= 2.0.0
- torchtune >= 0.1.0 (for RotaryPositionalEmbeddings)
- torchao >= 0.11.0 (required by torchtune)
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
- tqdm >= 4.65.0
- pytest >= 7.0.0 (for testing)

## File Structure

```
torch_grokking/
├── main.py              # Main training script and TorchTrainer class
├── data.py              # Data generation for modular arithmetic
├── models.py            # Transformer model implementation
├── requirements.txt     # Project dependencies
├── pytest.ini          # Test configuration
├── tests/               # Test suite
│   ├── __init__.py
│   ├── test_data.py     # Data generation tests
│   ├── test_models.py   # Model component tests
│   ├── test_trainer.py  # Training functionality tests
│   └── test_integration.py  # End-to-end integration tests
└── media/               # Generated plots and visualizations
```
