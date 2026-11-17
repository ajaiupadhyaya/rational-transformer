"""A small causal transformer that forecasts the next six values of a series.

Inputs are raw values x_s and their true time indices s (1..86). Time index 0
is reserved as the padding marker: a position with times == 0 is never
attended to by any other position, and its own output is meaningless (callers
ignore it). So a left-padded window gives exactly the forecasts the unpadded
window would.
"""
import torch
import torch.nn.functional as F
from torch import nn

SIZES = {"xs": (1, 32, 2), "s": (2, 64, 4), "m": (4, 128, 4), "l": (8, 256, 8)}
N_TIMES = 87       # index 0 = pad, 1..86 = real time indices
HORIZONS = 6


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model))

    def forward(self, h: torch.Tensor, allowed: torch.Tensor | None) -> torch.Tensor:
        """allowed: (B, 1, L, L) bool, True = query may attend to key; None = causal."""
        B, L, D = h.shape
        q, k, v = self.qkv(self.ln1(h)).split(D, dim=-1)
        q, k, v = (z.view(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)
                   for z in (q, k, v))
        if allowed is None:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        h = h + self.proj(a.transpose(1, 2).reshape(B, L, D))
        return h + self.mlp(self.ln2(h))


class Forecaster(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.value_in = nn.Linear(1, d_model)
        self.time_emb = nn.Embedding(N_TIMES, d_model, padding_idx=0)
        self.blocks = nn.ModuleList(Block(d_model, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, HORIZONS)

    def forward(self, values: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """values (B, L) float, times (B, L) int -> forecasts (B, L, 6)."""
        times = times.long()
        pad = times == 0
        values = values.masked_fill(pad, 0.0)
        h = self.value_in(values.unsqueeze(-1)) + self.time_emb(times)
        if pad.any():
            L = values.shape[1]
            idx = torch.arange(L, device=values.device)
            causal = idx[None, :] <= idx[:, None]                   # (L, L) key <= query
            diag = idx[None, :] == idx[:, None]
            # a query may see itself and earlier real keys; never a pad key (the
            # diagonal keeps pad queries from having an empty row, and pad outputs
            # are never read by a real position)
            allowed = causal[None] & (~pad[:, None, :] | diag[None])  # (B, L, L)
            allowed = allowed[:, None]                               # (B, 1, L, L)
        else:
            allowed = None      # no padding: plain causal mask (fused kernels on cuda)
        for blk in self.blocks:
            h = blk(h, allowed)
        return self.head(self.ln_f(h))
