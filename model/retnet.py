"""
Retention-based backbone for HOI-DM (RetHOI).

Implements the components described in Sec. 3.3 of the thesis:
  * Multi-scale retention with one learnable decay rate per head
    (gamma_h = sigmoid(w_h), initialised to 0.85 / 0.90 / 0.95 / 0.98 for the 4 heads:
    short-term contact dynamics -> long-term trajectory patterns).
  * Gated (SwiGLU) feed-forward network.
  * Group normalisation of the retention output + swish output gate.
  * Cross-retention: queries from one branch, keys/values from the other,
    weighted by the same distance-decay matrix. Used by the communication
    module that exchanges information between the human and object branches.

Tensors follow the MDM convention [seq_len, batch, dim].

A denoiser sees the whole (noisy) sequence at once, so by default the decay
matrix is bidirectional, D[n, m] = gamma^|n - m|. `causal=True` gives the
original RetNet mask (D[n, m] = gamma^(n - m) for n >= m, else 0).
The parallel form is used for training/sampling (sequences are <= 197 tokens);
the decay structure is what allows the O(N) recurrent / chunkwise forms.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


PAPER_GAMMAS = (0.85, 0.90, 0.95, 0.98)


def _init_decay_logits(num_heads):
    if num_heads == len(PAPER_GAMMAS):
        gammas = torch.tensor(PAPER_GAMMAS)
    else:  # RetNet default schedule for other head counts
        gammas = 1.0 - 2.0 ** (-5.0 - torch.arange(num_heads, dtype=torch.float))
    return torch.log(gammas / (1.0 - gammas))  # logit, so sigmoid(logit) == gamma


def _rotate_every_two(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RetentionRelPos(nn.Module):
    """Builds the rotary (xPos-style theta) tables and the multi-scale decay mask."""

    def __init__(self, head_dim, num_heads, causal=False):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, head_dim // 2))
        self.register_buffer('angle', angle.unsqueeze(-1).repeat(1, 2).flatten(), persistent=False)
        self.decay_logit = nn.Parameter(_init_decay_logits(num_heads))
        self.causal = causal

    def gammas(self):
        return torch.sigmoid(self.decay_logit)

    def rotary(self, length, device):
        index = torch.arange(length, device=device, dtype=self.angle.dtype)
        theta = index[:, None] * self.angle[None, :]
        return torch.sin(theta), torch.cos(theta)

    def decay_mask(self, len_q, len_k, device):
        """[H, len_q, len_k] decay mask, row-normalised as in RetNet."""
        q_idx = torch.arange(len_q, device=device, dtype=torch.float)
        k_idx = torch.arange(len_k, device=device, dtype=torch.float)
        diff = q_idx[:, None] - k_idx[None, :]
        log_gamma = torch.log(self.gammas())[:, None, None]  # [H, 1, 1]
        if self.causal:
            mask = torch.exp(log_gamma * diff.clamp(min=0))
            mask = mask.masked_fill(diff[None] < 0, 0.0)
        else:
            mask = torch.exp(log_gamma * diff.abs())
        return mask / mask.sum(dim=-1, keepdim=True).sqrt()


class MultiScaleRetention(nn.Module):
    """Multi-scale (cross-)retention.

    Self-retention when `memory` is None, cross-retention otherwise
    (queries from x, keys/values from memory).
    """

    def __init__(self, d_model, num_heads, causal=False):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.g_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.group_norm = nn.GroupNorm(num_heads, d_model, affine=False)
        self.relpos = RetentionRelPos(self.head_dim, num_heads, causal=causal)

    def _split_heads(self, x):
        seq, bs, _ = x.shape
        return x.view(seq, bs, self.num_heads, self.head_dim).permute(1, 2, 0, 3)  # [bs, H, seq, hd]

    def forward(self, x, memory=None):
        kv = x if memory is None else memory
        len_q, bs, _ = x.shape
        len_k = kv.shape[0]

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(kv)) * self.scaling
        v = self._split_heads(self.v_proj(kv))

        sin_q, cos_q = self.relpos.rotary(len_q, x.device)
        sin_k, cos_k = self.relpos.rotary(len_k, x.device)
        q = q * cos_q + _rotate_every_two(q) * sin_q
        k = k * cos_k + _rotate_every_two(k) * sin_k

        decay = self.relpos.decay_mask(len_q, len_k, x.device)  # [H, Lq, Lk]
        scores = torch.matmul(q, k.transpose(-1, -2)) * decay.unsqueeze(0)
        scores = scores / scores.abs().sum(dim=-1, keepdim=True).clamp(min=1.0)
        out = torch.matmul(scores, v)  # [bs, H, Lq, hd]

        out = out.permute(2, 1, 0, 3).reshape(len_q * bs, self.d_model)
        out = self.group_norm(out).view(len_q, bs, self.d_model)
        out = F.silu(self.g_proj(x)) * out
        return self.out_proj(out)


class GatedFFN(nn.Module):
    """SwiGLU feed-forward: W_down( swish(W_gate x) * W_up x )."""

    def __init__(self, d_model, ff_size, dropout=0.1):
        super().__init__()
        self.up_proj = nn.Linear(d_model, ff_size)
        self.gate_proj = nn.Linear(d_model, ff_size)
        self.down_proj = nn.Linear(ff_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class RetNetLayer(nn.Module):
    """Post-norm retention block (same layout as nn.TransformerEncoderLayer so that
    the pretrained MDM transformer can warm-start it, see utils/model_util.py)."""

    def __init__(self, d_model, num_heads, ff_size, dropout=0.1, causal=False):
        super().__init__()
        self.retention = MultiScaleRetention(d_model, num_heads, causal=causal)
        self.ffn = GatedFFN(d_model, ff_size, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.dropout1(self.retention(x)))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x


class RetNetEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, ff_size, dropout=0.1, causal=False):
        super().__init__()
        self.layers = nn.ModuleList([RetNetLayer(d_model, num_heads, ff_size, dropout, causal)
                                     for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CrossRetentionLayer(nn.Module):
    """Self-retention -> cross-retention (to the other branch) -> gated FFN.
    The communicated features enter through a residual H_out = H + alpha * H_comm (alpha = 0.5)."""

    def __init__(self, d_model, num_heads, ff_size, dropout=0.1, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.self_ret = MultiScaleRetention(d_model, num_heads)
        self.cross_ret = MultiScaleRetention(d_model, num_heads)
        self.ffn = GatedFFN(d_model, ff_size, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory):
        x = self.norm1(x + self.dropout(self.self_ret(x)))
        x = self.norm2(x + self.alpha * self.dropout(self.cross_ret(x, memory)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class CrossRetentionCommunication(nn.Module):
    """Bidirectional human <-> object communication module (Sec. 3.3.3)."""

    def __init__(self, num_layers, latent_dim, num_heads=4, ff_size=1024, dropout=0.1):
        super().__init__()
        self.human_layers = nn.ModuleList([CrossRetentionLayer(latent_dim, num_heads, ff_size, dropout)
                                           for _ in range(num_layers)])
        self.obj_layers = nn.ModuleList([CrossRetentionLayer(latent_dim, num_heads, ff_size, dropout)
                                         for _ in range(num_layers)])

    def forward(self, x_human, x_obj):
        h, o = x_human, x_obj
        for h_layer, o_layer in zip(self.human_layers, self.obj_layers):
            # both branches read the other branch's features from the previous layer
            h, o = h_layer(h, o), o_layer(o, h)
        return h, o
