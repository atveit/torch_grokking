import numpy as np
import torch

# --- New: PyTorch version ---
def grokking_data_torch(p: int, op: str = '/', train_fraction: float = 0.5, device='cpu'):
    """
    Same logic as grokking_data, but returns PyTorch tensors.
    """
    operations = {
        '*': lambda a, b: (a * b) % p,
        '/': lambda a, b: (a * pow(int(b), p-2, p)) % p,
        '+': lambda a, b: (a + b) % p,
        '-': lambda a, b: (a - b) % p
    }

    if op not in operations:
        raise ValueError(
            "Unsupported operation, choose from ['*', '/', '+', '-']")

    X = np.array([(a, b) for a in range(p)
                 for b in range(1 if op == '/' else 0, p)])
    T = np.array([operations[op](a, b) for a, b in X])

    embed = {'*': p, '/': p, '+': p, '-': p, '=': p + 1}
    X = np.array([[a, embed[op], b, embed['=']]
                  for (a, b) in X])

    n_train = int(train_fraction * len(X))
    inds = np.random.permutation(len(X))
    Xtrain, Ttrain = X[inds[:n_train]], T[inds[:n_train]]
    Xtest, Ttest = X[inds[n_train:]], T[inds[n_train:]]

    # Convert to torch
    Xtrain_torch = torch.tensor(Xtrain, dtype=torch.long, device=device)
    Ttrain_torch = torch.tensor(Ttrain, dtype=torch.long, device=device)
    Xtest_torch = torch.tensor(Xtest, dtype=torch.long, device=device)
    Ttest_torch = torch.tensor(Ttest, dtype=torch.long, device=device)

    return Xtrain_torch, Ttrain_torch, Xtest_torch, Ttest_torch


# Optional: quick equivalence check
if __name__ == '__main__':
    X_t, T_t, Xtest_t, Ttest_t = grokking_data_torch(11, op='/', train_fraction=0.5)
    print("Torch shapes:", X_t.shape, T_t.shape, Xtest_t.shape, Ttest_t.shape)

    # Check close in shape & values
    # Note: order might differ if random perm changed seeds
    # This is purely a demonstration
    print("Sample Torch data:", X_t[0], T_t[0])
