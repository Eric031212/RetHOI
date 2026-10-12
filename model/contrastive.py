"""
Text-motion contrastive alignment (Sec. 3.5 of the thesis).

Features are temporally pooled, projected to the 512-d CLIP space by a two-layer MLP
(ReLU + LayerNorm after each layer),
L2-normalised and aligned with the (frozen) CLIP text embedding using a symmetric
InfoNCE loss with temperature tau = 0.07.
Captions that appear several times in a batch (BEHAVE has many repeated prompts) are
treated as positives of each other rather than as negatives.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=512, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
                                 nn.Linear(hidden_dim, out_dim), nn.ReLU(), nn.LayerNorm(out_dim))

    def forward(self, x):
        return self.net(x)


def masked_mean_pool(feats, mask=None):
    """feats: [seq, bs, d]; mask: [bs, seq] bool (True = valid frame) or None."""
    if mask is None:
        return feats.mean(0)
    m = mask.t().unsqueeze(-1).to(feats.dtype)  # [seq, bs, 1]
    return (feats * m).sum(0) / m.sum(0).clamp(min=1.0)


def info_nce(motion_emb, text_emb, captions=None, temperature=0.07):
    """Symmetric InfoNCE between motion and text embeddings, both [bs, d]."""
    motion_emb = F.normalize(motion_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    logits = motion_emb @ text_emb.t() / temperature  # [bs, bs]

    bs = logits.shape[0]
    if captions is not None:
        positives = torch.tensor([[a == b for b in captions] for a in captions], device=logits.device)
    else:
        positives = torch.eye(bs, dtype=torch.bool, device=logits.device)
    targets = positives.float() / positives.float().sum(-1, keepdim=True)

    loss_m2t = -(targets * F.log_softmax(logits, dim=-1)).sum(-1)
    loss_t2m = -(targets * F.log_softmax(logits.t(), dim=-1)).sum(-1)
    return 0.5 * (loss_m2t + loss_t2m)  # [bs]
