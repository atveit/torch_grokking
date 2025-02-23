#import mlx.nn as nn
#import mlx.core as mx

#from mlx.utils import tree_flatten, tree_map

# --- New: Torch imports ---
import torch
import torch.nn as nn_torch
import torch.nn.functional as F
from torchtune.modules.position_embeddings import RotaryPositionalEmbeddings

class AttentionTorch(nn_torch.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = RMSNormTorch(dim)
        self.drop = nn_torch.Dropout(dropout)

        self.wq = nn_torch.Linear(dim, inner_dim, bias=False)
        self.wk = nn_torch.Linear(dim, inner_dim, bias=False)
        self.wv = nn_torch.Linear(dim, inner_dim, bias=False)
        self.wo = nn_torch.Linear(inner_dim, dim, bias=False)

        self.project_out = not (heads == 1 and dim_head == dim)
        if self.project_out:
            self.to_out = nn_torch.Sequential(
                nn_torch.Linear(inner_dim, dim),
                nn_torch.Dropout(dropout)
            )
        else:
            self.to_out = nn_torch.Identity()

        # We replicate the RoPE. We'll implement a small helper:
        self.rope = RoPETorch(dim_head, base=1e6)

    def forward(self, x, mask=None):
        # x: (b, n, d)
        b, n, d = x.shape
        x = self.norm(x)

        # (b, n, heads*dim_head)
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Reshape to (b, heads, n, dim_head)
        q = q.reshape(b, n, self.heads, -1).transpose(1, 2)  # (b, heads, n, dim_head)
        k = k.reshape(b, n, self.heads, -1).transpose(1, 2)
        v = v.reshape(b, n, self.heads, -1).transpose(1, 2)

        # Apply rope
        q = self.rope(q)
        k = self.rope(k)

        # Scaled dot product attention
        scores = torch.einsum('bhqd,bhkd->bhqk', q, k) * self.scale
        if mask is not None:
            scores = scores + mask  # broadcast
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum('bhqk,bhkd->bhqd', attn, v)

        # (b, heads, n, dim_head) -> (b, n, heads*dim_head)
        out = out.transpose(1, 2).reshape(b, n, -1)
        out = self.wo(out)
        if self.project_out:
            out = self.to_out(out)
        return out


class FeedForwardTorch(nn_torch.Module):
    def __init__(self, dim, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = RMSNormTorch(dim)
        self.drop = nn_torch.Dropout(dropout)
        self.w1 = nn_torch.Linear(dim, mlp_dim, bias=False)
        self.w2 = nn_torch.Linear(mlp_dim, dim, bias=False)
        self.w3 = nn_torch.Linear(dim, mlp_dim, bias=False)

    def forward(self, x):
        x_norm = self.norm(x)
        x1 = self.w1(x_norm)
        x_silu = F.silu(x1)
        x2 = x_silu * self.w3(x_norm)
        x2 = self.drop(x2)
        return self.w2(x2)


class BlockTorch(nn_torch.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, seq_len, dropout):
        super().__init__()
        self.attn = AttentionTorch(dim, heads, dim_head, dropout)
        self.ff = FeedForwardTorch(dim, mlp_dim, dropout)
        # Build a causal mask if needed
        self.register_buffer("_mask", self._causal_mask(seq_len), persistent=False)

    def _causal_mask(self, n):
        # shape: (1, 1, n, n) for broadcasting in multi-head attention
        # or simply (1, n, n). We'll do (1, n, n):
        mask = torch.triu(torch.full((n, n), float('-inf')), diagonal=1)
        return mask

    def forward(self, x):
        # x: (b, n, d)
        b, n, d = x.shape
        # Expand mask to (b, 1, n, n) if needed:
        mask = self._mask.unsqueeze(0)  # (1, n, n)
        # attn
        x = x + self.attn(x, mask=mask)
        x = x + self.ff(x)
        return x


class TransformerTorch(nn_torch.Module):
    def __init__(self, depth, dim, heads, n_tokens, seq_len, dropout=0., pool='cls'):
        super().__init__()
        assert pool in {'cls', 'mean'}
        self.pool = pool

        self.embedding = nn_torch.Embedding(n_tokens, dim)
        self.layers = nn_torch.ModuleList([
            BlockTorch(dim, heads, dim//heads, dim*4, seq_len, dropout) for _ in range(depth)
        ])
        self.norm = RMSNormTorch(dim)
        self.out = nn_torch.Linear(dim, n_tokens, bias=False)

    def forward(self, x):
        # x shape: (b, n)
        x = self.embedding(x)  # (b, n, dim)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        if self.pool == 'mean':
            x = x.mean(dim=1)
        else:
            # last token
            x = x[:, -1]
        logits = self.out(x)
        return logits



class RMSNormTorch(nn_torch.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn_torch.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (batch, seq, dim) or (batch, dim)
        # We handle last dimension as feature dimension
        normed = x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return normed * self.weight


class RoPETorch(nn_torch.Module):
    """
    A wrapper around torchtune's RotaryPositionalEmbeddings.
    Expects x shape: (b, seq, heads, dim_head).
    """
    def __init__(self, dim_head, base=1e6):
        super().__init__()
        self.rope = RotaryPositionalEmbeddings(dim_head, base=base)
        
    def forward(self, x, input_pos=None):
        # x shape: (b, seq, heads, dim_head)
        # This is already in the format torchtune expects: [b, s, n_h, h_d]
        return self.rope(x, input_pos=input_pos)

