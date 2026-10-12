"""
Generates HOI sequences for every prompt of the BEHAVE test split and stores everything
needed by eval/eval_behave.py.

    python -m sample.generate_eval --model_path save/rethoi/model000020000.pt \
        --afford_model_path save/apdm_con/model000010000.pt --guidance --out results/rethoi.npz

Stage 1 (APDM) predicts contact affordances, stage 2 (HOI-DM) denoises the human+object
motion with classifier-free guidance; with --guidance the affordance-guided correction is
applied during sampling (Gabor-smoothed contact trajectories when the HOI-DM was trained
with --use_gabor).
"""
import os
import time

import numpy as np
import torch

from data_loaders.dataset import Text2MotionDatasetV2
from data_loaders.scripts.motion_process import recover_from_ric
from data_loaders.tensors import t2hoi_collate
from diffusion.gaussian_diffusion import LocalMotionDiffusion, AffordDiffusion
from model.afford_est import AffordEstimation
from model.hoi_diff import HOIDiff
from sample.condition import Guide_Contact
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import load_model
from utils.parser_util import generate_args

N_JOINTS = 22


def main():
    import argparse
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--out', default='')
    extra.add_argument('--afford_skip', type=int, default=0, help='APDM diffusion steps to skip (0 = full 500)')
    extra.add_argument('--motion_skip', type=int, default=0, help='HOI-DM diffusion steps to skip (0 = full 1000)')
    extra.add_argument('--max_batches', type=int, default=-1)
    extra.add_argument('--showcase', type=int, default=0,
                       help='only generate N long test sequences (>= 150 frames) with distinct objects, for videos')
    extra.add_argument('--showcase_objects', nargs='*', default=None,
                       help='restrict the showcase to these objects (one sequence each)')
    extra.add_argument('--guide_obj_scale', type=float, default=10.0)
    extra.add_argument('--guide_human_scale', type=float, default=1.0)
    extra_args, remaining = extra.parse_known_args()
    import sys
    sys.argv = [sys.argv[0]] + remaining
    args = generate_args()
    for k, v in vars(extra_args).items():
        setattr(args, k, v)

    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()

    dataset = Text2MotionDatasetV2(opt=args, split='test')
    subset = dataset
    if args.showcase > 0:
        picked, objects = [], set()
        names = list(dataset.name_list[dataset.pointer:])
        for idx in reversed(range(len(names))):  # name_list is sorted by length: longest first
            obj = names[idx].split('_')[2]
            if args.showcase_objects and obj not in args.showcase_objects:
                continue
            if obj not in objects and dataset.data_dict[names[idx]]['length'] >= 150:
                picked.append(idx); objects.add(obj)
            if len(picked) == args.showcase:
                break
        subset = torch.utils.data.Subset(dataset, picked)
        print('showcase sequences:', [names[i] for i in picked])
    loader = torch.utils.data.DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers, drop_last=False,
                                         collate_fn=t2hoi_collate)

    afford_model, afford_diffusion = load_model(args, loader, device, ModelClass=AffordEstimation,
                                                DiffusionClass=AffordDiffusion,
                                                model_path=args.afford_model_path, diff_steps=500)
    motion_model, motion_diffusion = load_model(args, loader, device, ModelClass=HOIDiff,
                                                DiffusionClass=LocalMotionDiffusion, diff_steps=1000,
                                                model_path=args.model_path)
    motion_diffusion.guide_human_scale = args.guide_human_scale
    motion_diffusion.guide_obj_scale = args.guide_obj_scale
    use_gabor = getattr(args, 'use_gabor', False)
    sigmoid = torch.nn.Sigmoid()
    mean, std = dataset.mean, dataset.std

    out = {k: [] for k in ['feats', 'joints', 'obj_pose', 'lengths', 'text', 'seq_name',
                           'obj_points', 'afford', 'tokens']}
    t0 = time.time()
    for b_idx, (_, model_kwargs) in enumerate(loader):
        if 0 <= args.max_batches <= b_idx:
            break
        y = model_kwargs['y']
        for k in ['obj_points', 'obj_bps', 'mask', 'lengths']:
            y[k] = y[k].to(device)
        bs = y['lengths'].shape[0]
        if args.guidance_param != 1:
            y['scale'] = torch.ones(bs, device=device) * args.guidance_param

        afford_sample = afford_diffusion.p_sample_loop(
            afford_model, (bs, 4, 1, 8), clip_denoised=False, model_kwargs=model_kwargs,
            skip_timesteps=args.afford_skip, progress=False)
        afford_sample[:, 3:] = sigmoid(afford_sample[:, 3:])

        guide_fn = None
        if args.guidance:
            guide_fn = Guide_Contact(inv_transform_th=dataset.inv_transform_th, mean=mean, std=std,
                                     classifiler_scale=args.classifier_scale, batch_size=bs,
                                     afford_sample=afford_sample, use_gabor=use_gabor)

        sample = motion_diffusion.p_sample_loop(
            motion_model, (bs, 269, 1, args.max_motion_length), clip_denoised=False,
            model_kwargs=model_kwargs, skip_timesteps=args.motion_skip, progress=False,
            cond_fn=guide_fn)

        feats = sample.squeeze(2).permute(0, 2, 1).cpu()  # normalised [bs, T, 269]
        denorm = dataset.inv_transform(feats.numpy())
        joints = recover_from_ric(torch.from_numpy(denorm[..., :263]).float(), N_JOINTS).numpy()

        out['feats'].append(feats.numpy())
        out['joints'].append(joints)
        out['obj_pose'].append(denorm[..., 263:])
        out['lengths'].append(y['lengths'].cpu().numpy())
        out['obj_points'].append(y['obj_points'].cpu().numpy())
        out['afford'].append(afford_sample.squeeze(2).permute(0, 2, 1).cpu().numpy())
        out['text'] += y['text']
        out['tokens'] += y['tokens']
        out['seq_name'] += y['seq_name']
        print(f'[batch {b_idx + 1}/{len(loader)}] {time.time() - t0:.0f}s', flush=True)

    result = {k: (np.concatenate(v) if k not in ('text', 'seq_name', 'tokens') else np.array(v))
              for k, v in out.items()}
    out_path = args.out or os.path.join(os.path.dirname(args.model_path), 'eval_samples.npz')
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(out_path, **result)
    print(f'saved {len(result["text"])} samples to {out_path}')


if __name__ == '__main__':
    main()
