"""A small causal transformer that forecasts the next six values of a series.

Inputs are raw values x_s and their true time indices s (1..86). Time index 0
is reserved as the padding marker: a position with times == 0 is never
attended to by any other position, and its own output is meaningless (callers
ignore it). So a left-padded window gives exactly the forecasts the unpadded
window would.

Two architecture options (config keys "head" and "inputs"; defaults are "affine" and
"raw+asinh"; "linear" + "raw" is the original network):
- head "linear": forecasts = head(ln_f(h)), 6 numbers per position.
  head "affine": head(ln_f(h)) gives 12 numbers per position, (a_h, b_h) for h = 1..6,
  and the forecast is a_h * x_t + b_h with x_t the raw value at that position (the last
  position of a left-padded window is its last real value). The a_h rows of the head
  start at zero, so a_h starts at 0. LayerNorm bounds what head(ln_f(h)) can output; the
  affine head lets the forecast scale with x_t instead.
- inputs "raw": the value is embedded by Linear(1, d) from [x].
  inputs "raw+asinh": embedded by Linear(2, d) from [x, asinh(x)].
Both use only the inputs at the forecast's own position and earlier real positions, so
the information rule is the same for every variant.
"""
import torch
import torch.nn.functional as F
from torch import nn

SIZES = {"xs": (1, 32, 2), "s": (2, 64, 4), "m": (4, 128, 4), "l": (8, 256, 8)}
N_TIMES = 87       # index 0 = pad, 1..86 = real time indices
HORIZONS = 6
HEADS = ("linear", "affine")
INPUTS = ("raw", "raw+asinh")


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
    def __init__(self, n_layers: int, d_model: int, n_heads: int,
                 head: str = "affine", inputs: str = "raw+asinh"):
        super().__init__()
        assert d_model % n_heads == 0
        if head not in HEADS:
            raise ValueError(f"unknown head {head!r}; want one of {HEADS}")
        if inputs not in INPUTS:
            raise ValueError(f"unknown inputs {inputs!r}; want one of {INPUTS}")
        self.head_kind, self.inputs_kind = head, inputs
        self.value_in = nn.Linear(1 if inputs == "raw" else 2, d_model)
        self.time_emb = nn.Embedding(N_TIMES, d_model, padding_idx=0)
        self.blocks = nn.ModuleList(Block(d_model, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        if head == "linear":
            self.head = nn.Linear(d_model, HORIZONS)
        else:
            self.head = nn.Linear(d_model, 2 * HORIZONS)      # rows 0..5: a_h, 6..11: b_h
            with torch.no_grad():
                self.head.weight[:HORIZONS].zero_()
                self.head.bias[:HORIZONS].zero_()

    def forward(self, values: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """values (B, L) float, times (B, L) int -> forecasts (B, L, 6)."""
        times = times.long()
        pad = times == 0
        values = values.masked_fill(pad, 0.0)
        feats = values.unsqueeze(-1)
        if self.inputs_kind == "raw+asinh":
            feats = torch.cat([feats, torch.asinh(feats)], dim=-1)
        h = self.value_in(feats) + self.time_emb(times)
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
        out = self.head(self.ln_f(h))
        if self.head_kind == "linear":
            return out
        a, b = out[..., :HORIZONS], out[..., HORIZONS:]
        return a * values.unsqueeze(-1) + b
