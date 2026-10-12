"""
Multi-scale, multi-orientation Gabor smoothing (Sec. 3.4 of the thesis).

A bank of 2-D Gabor kernels g_{lambda, theta}(t, c) is laid over the (time x channel)
map of a trajectory, with
    x' =  t cos(theta) + c sin(theta)
    y' = -t sin(theta) + c cos(theta)
    g  = exp(-(x'^2 + gamma^2 y'^2) / (2 sigma^2)) * (1 + cos(2 pi x' / lambda + psi)) / 2
using 5 wavelengths lambda in {3, 5, 7, 9, 11} frames, 8 orientations theta_k = k * pi / 8,
sigma = 0.5 * lambda and aspect ratio gamma = 0.5. The carrier is the real part of the complex
Gabor wave plus a DC term, (1 + cos)/2, and every kernel is normalised to unit gain: a
trajectory filter has to preserve the signal itself (a zero-mean band-pass kernel would
remove it), while the oriented carrier still makes each filter frequency selective.
The 40 responses are fused with factorised learnable weights,
    y_smooth = sum_s sum_o softmax(w_s)_s * softmax(w_o)_o * y_{s,o};
because filtering is linear this equals one convolution with the weighted kernel,
which is what forward() computes.

With channel_extent=1 (default) each channel is filtered independently along time and the
orientation changes the effective temporal frequency / envelope of the kernel.
Reflection padding keeps the sequence length and avoids edge artefacts.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_WAVELENGTHS = (3.0, 5.0, 7.0, 9.0, 11.0)


def gabor_bank(wavelengths=DEFAULT_WAVELENGTHS, num_orientations=8, kernel_size=11,
               channel_extent=1, aspect_ratio=0.5, phase=0.0):
    """Returns [num_scales, num_orientations, kernel_size, channel_extent] unit-sum kernels."""
    half_t, half_c = kernel_size // 2, channel_extent // 2
    t = torch.arange(-half_t, half_t + 1, dtype=torch.float)
    c = torch.arange(-half_c, half_c + 1, dtype=torch.float)
    tt, cc = torch.meshgrid(t, c, indexing='ij')
    kernels = []
    for lam in wavelengths:
        sigma = 0.5 * lam
        for k in range(num_orientations):
            theta = k * math.pi / num_orientations
            x_r = tt * math.cos(theta) + cc * math.sin(theta)
            y_r = -tt * math.sin(theta) + cc * math.cos(theta)
            envelope = torch.exp(-(x_r ** 2 + aspect_ratio ** 2 * y_r ** 2) / (2 * sigma ** 2))
            carrier = 0.5 * (1.0 + torch.cos(2 * math.pi * x_r / lam + phase))
            g = envelope * carrier
            kernels.append(g / g.sum())
    return torch.stack(kernels).view(len(wavelengths), num_orientations, kernel_size, channel_extent)


class MultiScaleGaborSmoother(nn.Module):
    """Smooths x of shape [B, C, T] along T.

    out = x + sigmoid(blend) * (fused_gabor(x) - x)
    `learnable=False` freezes the fusion weights (uniform) and uses a full blend,
    which is how the filter is applied to contact trajectories at sampling time.
    """

    def __init__(self, wavelengths=DEFAULT_WAVELENGTHS, num_orientations=8, kernel_size=11,
                 channel_extent=1, learnable=True, init_blend=0.5):
        super().__init__()
        bank = gabor_bank(wavelengths, num_orientations, kernel_size, channel_extent)
        self.register_buffer('bank', bank, persistent=False)
        self.kernel_size = kernel_size
        self.channel_extent = channel_extent
        n_scales, n_orient = bank.shape[:2]
        blend_logit = math.log(init_blend / (1 - init_blend)) if 0 < init_blend < 1 else 20.0
        if learnable:
            self.scale_logits = nn.Parameter(torch.zeros(n_scales))
            self.orient_logits = nn.Parameter(torch.zeros(n_orient))
            self.blend_logit = nn.Parameter(torch.tensor(blend_logit))
        else:
            self.register_buffer('scale_logits', torch.zeros(n_scales), persistent=False)
            self.register_buffer('orient_logits', torch.zeros(n_orient), persistent=False)
            self.register_buffer('blend_logit', torch.tensor(blend_logit), persistent=False)

    def fused_kernel(self):
        alpha = torch.softmax(self.scale_logits, dim=0)
        beta = torch.softmax(self.orient_logits, dim=0)
        return torch.einsum('s,o,sokc->kc', alpha, beta, self.bank)

    def forward(self, x):
        bs, ch, length = x.shape
        pad_t = self.kernel_size // 2
        pad_c = self.channel_extent // 2
        img = x.permute(0, 2, 1).unsqueeze(1)  # [B, 1, T, C]
        mode = 'reflect' if length > pad_t else 'replicate'
        img = F.pad(img, (0, 0, pad_t, pad_t), mode=mode)
        if pad_c:
            img = F.pad(img, (pad_c, pad_c, 0, 0), mode='reflect' if ch > pad_c else 'replicate')
        kernel = self.fused_kernel().to(x.dtype)[None, None]
        smoothed = F.conv2d(img, kernel).squeeze(1).permute(0, 2, 1)  # [B, C, T]
        blend = torch.sigmoid(self.blend_logit)
        return x + blend * (smoothed - x)


def gabor_smooth(x, dim=1, **kwargs):
    """Functional, non-learnable smoothing of a trajectory along `dim` (time)."""
    smoother = _cached_smoother(x.device, **kwargs)
    moved = x.movedim(dim, -1)
    shape = moved.shape
    out = smoother(moved.reshape(-1, 1, shape[-1])).reshape(shape)
    return out.movedim(-1, dim)


_SMOOTHERS = {}


def _cached_smoother(device, **kwargs):
    key = (str(device), tuple(sorted(kwargs.items())))
    if key not in _SMOOTHERS:
        _SMOOTHERS[key] = MultiScaleGaborSmoother(learnable=False, init_blend=1.0, **kwargs).to(device)
    return _SMOOTHERS[key]
