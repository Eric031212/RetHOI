"""
Pure-PyTorch replacement for `pointnet2_ops.pointnet2_modules.PointnetSAModuleMSG`.

Same semantics as the CUDA ops (furthest point sampling starting from index 0,
ball query returning the first `nsample` neighbours in index order and padding
with the first one, relative xyz concatenated to the grouped features), so the
APDM can be trained without compiling the extension.
"""
import torch
import torch.nn as nn


def furthest_point_sample(xyz, npoint):
    """xyz: [B, N, 3] -> indices [B, npoint]"""
    bs, n, _ = xyz.shape
    if npoint >= n:
        return torch.arange(n, device=xyz.device).unsqueeze(0).expand(bs, n).contiguous()
    idx = torch.zeros(bs, npoint, dtype=torch.long, device=xyz.device)
    dist = torch.full((bs, n), 1e10, device=xyz.device)
    farthest = torch.zeros(bs, dtype=torch.long, device=xyz.device)
    batch = torch.arange(bs, device=xyz.device)
    for i in range(npoint):
        idx[:, i] = farthest
        centroid = xyz[batch, farthest].unsqueeze(1)
        dist = torch.minimum(dist, ((xyz - centroid) ** 2).sum(-1))
        farthest = dist.argmax(-1)
    return idx


def ball_query(radius, nsample, xyz, new_xyz):
    """-> [B, npoint, nsample] neighbour indices into xyz."""
    n = xyz.shape[1]
    sqr = torch.cdist(new_xyz, xyz) ** 2  # [B, S, N]
    order = torch.arange(n, device=xyz.device).expand_as(sqr).clone()
    order[sqr > radius ** 2] = n
    order = order.sort(dim=-1)[0][..., :nsample]
    first = order[..., :1].expand_as(order)
    return torch.where(order == n, first, order)


def gather(points, idx):
    """points: [B, N, C], idx: [B, ...] -> [B, ..., C]"""
    bs = points.shape[0]
    flat = idx.reshape(bs, -1)
    out = torch.gather(points, 1, flat.unsqueeze(-1).expand(-1, -1, points.shape[-1]))
    return out.view(*idx.shape, points.shape[-1])


class SharedMLP(nn.Sequential):
    def __init__(self, channels):
        layers = []
        for c_in, c_out in zip(channels[:-1], channels[1:]):
            layers += [nn.Conv2d(c_in, c_out, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True)]
        super().__init__(*layers)


class PointnetSAModuleMSG(nn.Module):
    def __init__(self, npoint, radii, nsamples, mlps, use_xyz=True):
        super().__init__()
        self.npoint = npoint
        self.radii = radii
        self.nsamples = nsamples
        self.use_xyz = use_xyz
        self.mlps = nn.ModuleList()
        for spec in mlps:
            spec = list(spec)
            if use_xyz:
                spec[0] += 3
            self.mlps.append(SharedMLP(spec))

    def forward(self, xyz, features=None):
        """xyz: [B, N, 3], features: [B, C, N] or None -> (new_xyz [B, S, 3], new_features [B, C', S])"""
        fps_idx = furthest_point_sample(xyz, self.npoint)
        new_xyz = gather(xyz, fps_idx)
        outs = []
        for radius, nsample, mlp in zip(self.radii, self.nsamples, self.mlps):
            idx = ball_query(radius, nsample, xyz, new_xyz)  # [B, S, K]
            grouped_xyz = gather(xyz, idx) - new_xyz.unsqueeze(2)  # [B, S, K, 3]
            if features is not None:
                grouped = gather(features.transpose(1, 2), idx)  # [B, S, K, C]
                grouped = torch.cat([grouped_xyz, grouped], -1) if self.use_xyz else grouped
            else:
                grouped = grouped_xyz
            grouped = grouped.permute(0, 3, 1, 2)  # [B, C, S, K]
            outs.append(mlp(grouped).max(dim=-1)[0])
        return new_xyz, torch.cat(outs, dim=1)
