import argparse
import numpy as np

import matplotlib.pyplot as plt
import os  # Add os import

from tqdm import tqdm
from functools import partial

import torch
import torch.nn as nn_torch
import torch.optim as optim_torch
import torch.nn.functional as F

from models import TransformerTorch
from data import grokking_data_torch

parser = argparse.ArgumentParser(add_help=True)
# data args
parser.add_argument('--p', type=int, default=97, help='prime number')
parser.add_argument('--op', type=str, default='/',
                    help='operation', choices=['*', '/', '+', '-'])
parser.add_argument('--train-fraction', type=float,
                    default=0.5, help='train fraction')
# model args
parser.add_argument('--depth', type=int, default=2, help='depth')
parser.add_argument('--dim', type=int, default=128, help='dimension')
parser.add_argument('--heads', type=int, default=1, help='heads')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
# optimizer args
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight-decay', type=float,
                    default=1, help='weight decay')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1')
parser.add_argument('--beta2', type=float, default=0.98, help='beta2')
parser.add_argument('--warmup', type=int, default=10, help='warmup steps')
# training args
parser.add_argument('-b', '--batch_size', type=int,
                    default=512, help='batch size')
parser.add_argument('-e', '--epochs', type=int,
                    default=150, help='number of epochs')
# misc args
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--cpu', action='store_true', help='use cpu only')



class TorchTrainer:
    """
    A parallel trainer that replicates the MLX training flow using PyTorch.
    """
    def __init__(self,
                 model: nn_torch.Module,
                 optimizer: optim_torch.Optimizer,
                 classification: bool = False,
                 batch_size: int = 64,
                 device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.classification = classification
        self.batch_size = batch_size

        if classification:
            self.loss_fn = F.cross_entropy
        else:
            self.loss_fn = F.mse_loss

        self.train_error_trace = []
        self.train_acc_trace = []
        self.val_error_trace = []
        self.val_acc_trace = []

    def _make_batches(self, X_torch, T_torch):
        bs = self.batch_size if self.batch_size != -1 else X_torch.shape[0]
        for i in range(0, X_torch.shape[0], bs):
            yield X_torch[i:i+bs], T_torch[i:i+bs]


    def train(self, train_data, val_data, epochs=5, shuffle=True):
        self.model.train()
        Xtrain_t, Ttrain_t = train_data
        Xtest_t, Ttest_t = val_data

        # Basic epoch loop
        epoch_bar = tqdm(range(epochs), desc='Training', unit='epoch')
        for _ in epoch_bar:
            self.model.train()
            if shuffle:
                permutation = torch.randperm(Xtrain_t.size(0))
                Xtrain_t = Xtrain_t[permutation]
                Ttrain_t = Ttrain_t[permutation]

            total_loss = 0.0
            total_correct = 0
            for Xb, Tb in self._make_batches(Xtrain_t, Ttrain_t):
                # Move to device if needed
                Xb = Xb.to(self.device)
                Tb = Tb.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(Xb)
                loss = self.loss_fn(outputs, Tb)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * Xb.size(0)
                if self.classification:
                    preds = torch.argmax(outputs, dim=1)
                    total_correct += (preds == Tb).sum().item()

            avg_train_loss = total_loss / Xtrain_t.shape[0]
            if self.classification:
                avg_train_acc = total_correct / Xtrain_t.shape[0]
            else:
                avg_train_acc = 0.0

            self.train_error_trace.append(avg_train_loss)
            self.train_acc_trace.append(avg_train_acc)

            # Evaluate
            avg_val_loss, avg_val_acc = self.evaluate((Xtest_t, Ttest_t))
            self.val_error_trace.append(avg_val_loss)
            self.val_acc_trace.append(avg_val_acc)

            postfix = {
                'train_loss': f'{avg_train_loss:.3f}',
                'train_acc': f'{avg_train_acc:.3f}',
                'val_loss': f'{avg_val_loss:.3f}',
                'val_acc': f'{avg_val_acc:.3f}',
            }
            epoch_bar.set_postfix(postfix)

    def evaluate(self, test_data):
        self.model.eval()
        Xtest_t, Ttest_t = test_data
        total_loss, total_correct = 0.0, 0
        with torch.no_grad():
            for Xb, Tb in self._make_batches(Xtest_t, Ttest_t):
                Xb = Xb.to(self.device)
                Tb = Tb.to(self.device)
                outputs = self.model(Xb)
                loss = self.loss_fn(outputs, Tb)
                total_loss += loss.item() * Xb.size(0)
                if self.classification:
                    preds = torch.argmax(outputs, dim=1)
                    total_correct += (preds == Tb).sum().item()
        avg_loss = total_loss / Xtest_t.shape[0]
        if self.classification:
            avg_acc = total_correct / Xtest_t.shape[0]
        else:
            avg_acc = 0.0
        return avg_loss, avg_acc


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
  
    Xtrain_torch, Ttrain_torch, Xtest_torch, Ttest_torch = grokking_data_torch(
        args.p, op=args.op, train_fraction=args.train_fraction, device='cpu')
        # Already torch tensors

    # Build model(s)
    kwargs = {
        'depth': args.depth,
        'dim': args.dim,
        'heads': args.heads,
        'n_tokens': args.p + 2,
        'seq_len': 4,  # typically X shape is (N, 4) for [a, op, b, '=']
        'dropout': args.dropout
    }


    device = 'cpu'
    if not args.cpu and torch.cuda.is_available():
        device = 'cuda'
    torch_model = TransformerTorch(**kwargs).to(device)

    optimizer_torch = optim_torch.AdamW(
        torch_model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )

    trainer = TorchTrainer(
        torch_model,
        optimizer_torch,
        classification=True,
        batch_size=args.batch_size,
        device=device
    )

    trainer.train(
        (Xtrain_torch, Ttrain_torch),
        (Xtest_torch, Ttest_torch),
        epochs=args.epochs,
        shuffle=True
    )

    # Store PyTorch results for later plotting
    torch_train_acc = np.array(trainer.train_acc_trace) * 100
    torch_val_acc = np.array(trainer.val_acc_trace) * 100

    # Plot results
    os.makedirs('media', exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 10))
    lw = 2

    ax.plot(torch_train_acc, label=f'PyTorch train', color='#1b9e77', lw=lw, linestyle='-')
    ax.plot(torch_val_acc, label=f'PyTorch val', color='#1b9e77', lw=lw, linestyle='--')

    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.title('Training Progress: PyTorch', fontsize=16, pad=20)
    fig.tight_layout()
    fig.savefig('media/grokking_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

