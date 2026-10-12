"""
Computes BEHAVE metrics for samples written by sample/generate_eval.py.

    python -m eval.eval_behave --samples results/rethoi.npz results/hoidiff.npz \
        --evaluator checkpoints/behave_evaluator.pt --out results/metrics.json

Metrics
  FID        Frechet distance between evaluator motion embeddings of generated and real test motions.
  R-Prec     top-3 text->motion retrieval accuracy in groups of 32 (1 positive + 31 negatives).
  Diversity  mean distance between 300 random pairs of generated motion embeddings.
  Contact    for each ground-truth contact joint, distance (m) to the nearest point of the (generated)
             object surface, averaged over the closer half of the frames (the interaction phase;
             the approach phase before the person reaches the object is excluded).
  FootSkate  ratio of frames in which a foot is within 5 cm of the ground yet slides more than
             2.5 cm in one frame (0.5 m/s at 20 fps).
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy import linalg
from scipy.spatial.transform import Rotation

from data_loaders.dataset import load_affordance, AFFORD_JOINTS
from data_loaders.utils.word_vectorizer import WordVectorizer
from eval.evaluator import EvaluatorWrapper, make_loader

FOOT_JOINTS = [10, 11]


def fid(a, b):
    """Frechet distance; trace(sqrt(S1 S2)) via the eigenvalues of sqrt(S1) S2 sqrt(S1), which stays
    real and non-negative when the covariances are rank deficient (few samples, 512-d)."""
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    w1, v1 = linalg.eigh(s1)
    sqrt_s1 = (v1 * np.sqrt(np.clip(w1, 0, None))) @ v1.T
    w = linalg.eigvalsh(sqrt_s1 @ s2 @ sqrt_s1)
    tr_covmean = np.sqrt(np.clip(w, 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * tr_covmean)


def r_precision(motion_emb, text_emb, group=32, top_k=3, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(motion_emb))
    hits, n = 0, 0
    for start in range(0, len(idx) - group + 1, group):
        sel = idx[start:start + group]
        dist = np.linalg.norm(motion_emb[sel, None] - text_emb[None, sel], axis=-1)
        rank = (dist < np.diag(dist)[:, None]).sum(1)
        hits += (rank < top_k).sum(); n += group
    return hits / max(n, 1)


def diversity(emb, times=300, seed=0):
    rng = np.random.RandomState(seed)
    a, b = rng.randint(0, len(emb), times), rng.randint(0, len(emb), times)
    return float(np.linalg.norm(emb[a] - emb[b], axis=-1).mean())


def foot_skate_ratio(joints, length, height_thresh=0.05, disp_thresh=0.025):
    feet = joints[:length, FOOT_JOINTS]  # [T, 2, 3]
    floor = 0.0  # sequences are floor-aligned during preprocessing (y up)
    height = feet[..., 1] - floor
    disp = np.linalg.norm(feet[1:, :, [0, 2]] - feet[:-1, :, [0, 2]], axis=-1)  # [T-1, 2]
    grounded = (height[1:] < height_thresh) & (height[:-1] < height_thresh)
    skating = (grounded & (disp > disp_thresh)).any(-1)
    return float(skating.mean()) if len(skating) else 0.0


def contact_distance(joints, obj_pose, obj_points, afford, length):
    """afford: [8, 4] ground-truth (xyz on object, flag). Returns None if the sequence has no contact."""
    on = afford[:, 3] > 0.5
    if not on.any():
        return None
    rot = Rotation.from_rotvec(obj_pose[:length, :3]).as_matrix()  # [T, 3, 3]
    world = np.einsum('tij,nj->tni', rot, obj_points) + obj_pose[:length, None, 3:]  # [T, N, 3]
    jts = joints[:length, np.array(AFFORD_JOINTS)[on]]  # [T, k, 3]
    dist = np.linalg.norm(jts[:, :, None] - world[:, None], axis=-1).min(-1)  # [T, k]
    half = max(1, length // 2)
    return float(np.sort(dist, axis=0)[:half].mean())


def text_embeddings(evaluator, tokens_list, w_vectorizer, max_text_len=20):
    words, poses, lens = [], [], []
    for tokens in tokens_list:
        tokens = tokens.split('_')
        tokens = [t for t in tokens if t not in ('sos/OTHER', 'eos/OTHER', 'unk/OTHER')]
        tokens = ['sos/OTHER'] + tokens[:max_text_len] + ['eos/OTHER']
        sent_len = len(tokens)
        tokens = tokens + ['unk/OTHER'] * (max_text_len + 2 - sent_len)
        w, p = zip(*[w_vectorizer[t] for t in tokens])
        words.append(np.stack(w)); poses.append(np.stack(p)); lens.append(sent_len)
    return evaluator.text_embedding(torch.from_numpy(np.stack(words)), torch.from_numpy(np.stack(poses)),
                                    torch.tensor(lens)).cpu().numpy()


def motion_embeddings(evaluator, feats, lengths, batch=64):
    out = []
    for i in range(0, len(feats), batch):
        m = torch.from_numpy(feats[i:i + batch]).float()
        out.append(evaluator.motion_embedding(m, torch.from_numpy(lengths[i:i + batch]).long()).cpu().numpy())
    return np.concatenate(out)


def real_statistics(evaluator, seed):
    """Embeddings of the real test motions and their captions (random crops / captions per seed)."""
    torch.manual_seed(seed); np.random.seed(seed)
    import random; random.seed(seed)
    _, loader = make_loader('test', 32, False)
    m_emb, t_emb = [], []
    for word, pos, caption, sent_len, motion, length in loader:
        m_emb.append(evaluator.motion_embedding(motion, length).cpu().numpy())
        t_emb.append(evaluator.text_embedding(word, pos, sent_len).cpu().numpy())
    return np.concatenate(m_emb), np.concatenate(t_emb)


def evaluate_samples(path, evaluator, w_vectorizer, gt_motion_emb, data_dir):
    s = np.load(path, allow_pickle=True)
    lengths = s['lengths'].astype(np.int64)
    feats = s['feats'].copy()
    for i, l in enumerate(lengths):  # the evaluator only sees the valid frames
        feats[i, l:] = 0
    m_emb = motion_embeddings(evaluator, feats, lengths)
    t_emb = text_embeddings(evaluator, list(s['tokens']), w_vectorizer)

    contacts, skates = [], []
    for i, l in enumerate(lengths):
        skates.append(foot_skate_ratio(s['joints'][i], l))
        afford = load_affordance(os.path.join(data_dir, 'affordance', f"{s['seq_name'][i]}.npy"))
        c = contact_distance(s['joints'][i], s['obj_pose'][i], s['obj_points'][i], afford, l)
        if c is not None:
            contacts.append(c)
    return {
        'FID': fid(m_emb, gt_motion_emb),
        'R-Prec(top3)': float(r_precision(m_emb, t_emb)),
        'Diversity': diversity(m_emb),
        'Contact': float(np.mean(contacts)),
        'FootSkate': float(np.mean(skates)),
        'num_samples': int(len(lengths)),
    }


def gt_physical_metrics(data_dir):
    """Contact / foot-skate of the real test motions (sanity reference)."""
    from data_loaders.scripts.motion_process import recover_from_ric
    ds, _ = make_loader('test', 32, False)
    contacts, skates = [], []
    for name in ds.name_list[ds.pointer:]:
        d = ds.data_dict[name]
        motion = d['motion']
        joints = recover_from_ric(torch.from_numpy(motion[:, :263]).float(), 22).numpy()
        l = len(motion)
        skates.append(foot_skate_ratio(joints, l))
        c = contact_distance(joints, motion[:, 263:], d['obj_points'],
                             load_affordance(os.path.join(data_dir, 'affordance', f'{name}.npy')), l)
        if c is not None:
            contacts.append(c)
    return float(np.mean(contacts)), float(np.mean(skates))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--samples', nargs='+', required=True)
    p.add_argument('--names', nargs='*', default=None)
    p.add_argument('--evaluator', default='./checkpoints/behave_evaluator.pt')
    p.add_argument('--data_dir', default='./dataset/behave_t2m')
    p.add_argument('--out', default='./results/metrics.json')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = EvaluatorWrapper(args.evaluator, device)
    w_vectorizer = WordVectorizer('./glove', 'our_vab')

    gt_m, gt_t = real_statistics(evaluator, seed=0)
    gt_m2, gt_t2 = real_statistics(evaluator, seed=1)
    real_contact, real_skate = gt_physical_metrics(args.data_dir)
    results = {'Real': {'FID': fid(gt_m2, gt_m), 'R-Prec(top3)': float(r_precision(gt_m2, gt_t2)),
                        'Diversity': diversity(gt_m2), 'Contact': real_contact, 'FootSkate': real_skate}}

    names = args.names or [os.path.splitext(os.path.basename(x))[0] for x in args.samples]
    for name, path in zip(names, args.samples):
        results[name] = evaluate_samples(path, evaluator, w_vectorizer, gt_m, args.data_dir)
        print(name, results[name], flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n{'Method':<22}{'FID':>8}{'R-Prec':>8}{'Div':>8}{'Contact':>9}{'Foot':>8}")
    for k, v in results.items():
        print(f"{k:<22}{v['FID']:>8.3f}{v['R-Prec(top3)']:>8.3f}{v['Diversity']:>8.3f}"
              f"{v['Contact']:>9.3f}{v['FootSkate']:>8.3f}")


if __name__ == '__main__':
    main()
