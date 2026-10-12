"""
Text-motion evaluator for BEHAVE (HOI-Diff did not release theirs).

Follows the design of Guo et al. (CVPR'22): a GloVe + POS bidirectional-GRU text encoder
and a convolutional + bidirectional-GRU motion encoder over the normalised 263-d human +
6-d object features, trained with a symmetric InfoNCE loss on the training split.
It deliberately does not use CLIP, so it cannot favour models trained with a CLIP-space
contrastive loss.

    python -m eval.evaluator --save_path checkpoints/behave_evaluator.pt
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from model.contrastive import info_nce


class TextEncoder(nn.Module):
    def __init__(self, word_dim=300, pos_dim=15, hidden=512, out_dim=512):
        super().__init__()
        self.inp = nn.Linear(word_dim + pos_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(0.2),
                                 nn.Linear(hidden, out_dim))

    def forward(self, word_embs, pos_ohot, sent_lens):
        x = self.inp(torch.cat([word_embs, pos_ohot], -1))
        packed = pack_padded_sequence(x, sent_lens.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return self.out(torch.cat([h[0], h[1]], -1))


class MotionEncoder(nn.Module):
    def __init__(self, in_dim=269, hidden=512, out_dim=512):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(in_dim, hidden, 4, 2, 1), nn.Dropout(0.2), nn.LeakyReLU(0.2),
                                  nn.Conv1d(hidden, hidden, 4, 2, 1), nn.Dropout(0.2), nn.LeakyReLU(0.2))
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(0.2),
                                 nn.Linear(hidden, out_dim))

    def forward(self, motion, lengths):
        """motion: [B, T, 269] normalised features; lengths: [B] valid frames."""
        x = self.conv(motion.permute(0, 2, 1)).permute(0, 2, 1)  # [B, T/4, H]
        lens = (lengths // 4).clamp(min=1, max=x.shape[1])
        packed = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return self.out(torch.cat([h[0], h[1]], -1))


class EvaluatorWrapper:
    def __init__(self, ckpt_path, device):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.text_enc = TextEncoder().to(device).eval()
        self.motion_enc = MotionEncoder().to(device).eval()
        self.text_enc.load_state_dict(ckpt['text_enc'])
        self.motion_enc.load_state_dict(ckpt['motion_enc'])
        self.device = device

    @torch.no_grad()
    def text_embedding(self, word_embs, pos_ohot, sent_lens):
        emb = self.text_enc(word_embs.to(self.device).float(), pos_ohot.to(self.device).float(), sent_lens)
        return F.normalize(emb, dim=-1)

    @torch.no_grad()
    def motion_embedding(self, motion, lengths):
        emb = self.motion_enc(motion.to(self.device).float(), lengths.to(self.device))
        return F.normalize(emb, dim=-1)


def make_loader(split, batch_size, shuffle, data_root='./dataset'):
    from data_loaders.dataset import Text2MotionDatasetV2
    opt = argparse.Namespace(dataset='behave', data_root=data_root, max_motion_length=196,
                             max_text_len=20, unit_length=5)
    ds = Text2MotionDatasetV2(opt, split=split)

    def collate(batch):
        word, pos, caption, sent_len, motion, length = zip(*[b[:6] for b in batch])
        return (torch.from_numpy(np.stack(word)), torch.from_numpy(np.stack(pos)), list(caption),
                torch.tensor(sent_len), torch.from_numpy(np.stack(motion)).float(), torch.tensor(length))

    return ds, torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle,
                                           num_workers=2, collate_fn=collate)


def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, train_loader = make_loader('train', args.batch_size, True)
    _, test_loader = make_loader('test', 32, False)
    text_enc, motion_enc = TextEncoder().to(device), MotionEncoder().to(device)
    params = list(text_enc.parameters()) + list(motion_enc.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        text_enc.train(); motion_enc.train()
        total = 0.0
        for word, pos, caption, sent_len, motion, length in train_loader:
            t_emb = text_enc(word.to(device).float(), pos.to(device).float(), sent_len)
            m_emb = motion_enc(motion.to(device), length.to(device))
            loss = info_nce(m_emb, t_emb, captions=caption, temperature=args.temperature).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if (epoch + 1) % 25 == 0:
            r3 = retrieval_top3(text_enc, motion_enc, test_loader, device)
            print(f'epoch {epoch + 1}: loss {total / len(train_loader):.4f}  test R@3 (32-way) {r3:.3f}', flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    torch.save({'text_enc': text_enc.state_dict(), 'motion_enc': motion_enc.state_dict()}, args.save_path)
    print('saved evaluator to', args.save_path)


@torch.no_grad()
def retrieval_top3(text_enc, motion_enc, loader, device):
    text_enc.eval(); motion_enc.eval()
    hits, n = 0, 0
    for word, pos, caption, sent_len, motion, length in loader:
        t = F.normalize(text_enc(word.to(device).float(), pos.to(device).float(), sent_len), dim=-1)
        m = F.normalize(motion_enc(motion.to(device), length.to(device)), dim=-1)
        dist = torch.cdist(m, t)
        rank = (dist < dist.diag().unsqueeze(1)).sum(1)
        hits += (rank < 3).sum().item(); n += len(rank)
    return hits / n


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--save_path', default='./checkpoints/behave_evaluator.pt')
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=0)
    train(p.parse_args())
