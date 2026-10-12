"""
End-to-end smoke test on random tensors (no dataset needed):
builds HOI-DM (Transformer and RetNet backbones) and APDM, warm-starts HOI-DM from the
pretrained MDM, runs one training step with every loss term, and a few sampling steps.

    python -m tests.smoke_test
"""
import math
import time
from types import SimpleNamespace

import torch

from diffusion.gaussian_diffusion import LocalMotionDiffusion, AffordDiffusion
from model.afford_est import AffordEstimation
from model.gabor import gabor_smooth
from model.hoi_diff import HOIDiff
from utils.model_util import create_model_and_diffusion, load_split_mdm

MDM_CKPT = './checkpoints/humanml_trans_enc_512/model000475000.pt'


def make_args(**kw):
    base = dict(arch='retnet', use_gabor=True, use_contrastive=True, lambda_con=0.1, lambda_smooth=0.05,
                multi_backbone_split=4, cond_mask_prob=0.1, emb_trans_dec=False, dataset='behave',
                unconstrained=False, noise_schedule='cosine', sigma_small=True)
    base.update(kw)
    return SimpleNamespace(**base)


def batch(bs, frames, device, feats=269):
    texts = ['a person lifts a box with both hands', 'a person sits on a chair'] * (bs // 2)
    lengths = torch.tensor([frames, frames - 40] * (bs // 2), device=device)
    mask = (torch.arange(frames, device=device)[None] < lengths[:, None])[:, None, None]
    y = {'text': texts, 'lengths': lengths, 'mask': mask,
         'obj_points': torch.rand(bs, 1024, 3, device=device) - 0.5}
    return torch.randn(bs, feats, 1, frames, device=device), {'y': y}


def check_hoidm(arch, device, **kw):
    args = make_args(arch=arch, **kw)
    data = SimpleNamespace(dataset=SimpleNamespace())
    model, diffusion = create_model_and_diffusion(args, data, ModelClass=HOIDiff, DiffusionClass=LocalMotionDiffusion)
    load_split_mdm(model, torch.load(MDM_CKPT, map_location='cpu'), args.multi_backbone_split)
    model.to(device).train()
    n_params = sum(p.numel() for p in model.parameters_wo_clip()) / 1e6

    x, cond = batch(8, 196, device)
    t = torch.randint(0, 1000, (8,), device=device)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    losses = diffusion.training_losses(model, cond['y']['mask'], 700, x_start=x, t=t, model_kwargs=cond)
    loss = losses['loss'].mean()
    loss.backward()
    torch.cuda.synchronize() if device.type == 'cuda' else None
    assert torch.isfinite(loss), losses
    terms = {k: round(v.mean().item(), 4) for k, v in losses.items()}
    print(f'[HOI-DM {arch:9s} gabor={args.use_gabor} con={args.use_contrastive}] params {n_params:.1f}M '
          f'step {time.time() - start:.2f}s losses {terms}')

    model.eval()
    with torch.no_grad():
        out = diffusion.p_sample_loop(model, (2, 269, 1, 196), clip_denoised=False,
                                      model_kwargs={'y': {k: (v[:2] if k != 'text' else v[:2]) for k, v in cond['y'].items()}},
                                      skip_timesteps=990, progress=False)
    assert out.shape == (2, 269, 1, 196) and torch.isfinite(out).all()


def check_apdm(device):
    args = make_args(use_contrastive=True)
    data = SimpleNamespace(dataset=SimpleNamespace())
    model, diffusion = create_model_and_diffusion(args, data, ModelClass=AffordEstimation,
                                                  DiffusionClass=AffordDiffusion, diff_steps=500)
    model.to(device).train()
    x, cond = batch(4, 8, device, feats=4)
    cond['y']['mask'] = None
    t = torch.randint(0, 500, (4,), device=device)
    losses = diffusion.training_losses(model, None, 700, x_start=x, t=t, model_kwargs=cond)
    losses['loss'].mean().backward()
    print('[APDM] losses', {k: round(v.mean().item(), 4) for k, v in losses.items()})


def check_gabor():
    t = torch.linspace(0, 4 * math.pi, 196)
    clean = torch.sin(t)[None, None]
    noisy = clean + 0.2 * torch.randn_like(clean)
    smooth = gabor_smooth(noisy, dim=-1)
    err_noisy, err_smooth = ((noisy - clean) ** 2).mean().item(), ((smooth - clean) ** 2).mean().item()
    print(f'[Gabor] MSE to clean signal: noisy {err_noisy:.4f} -> smoothed {err_smooth:.4f}')
    assert err_smooth < err_noisy


if __name__ == '__main__':
    torch.manual_seed(0)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', dev, torch.cuda.get_device_name(0) if dev.type == 'cuda' else '')
    check_gabor()
    check_hoidm('trans_enc', dev, use_gabor=False, use_contrastive=False, lambda_smooth=0.0)
    check_hoidm('retnet', dev)
    check_apdm(dev)
    print('SMOKE TEST PASSED')
