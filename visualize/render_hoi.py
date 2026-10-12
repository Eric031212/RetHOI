"""
Renders generated human-object interactions as shaded meshes (SMPL body + BEHAVE object) to mp4.

    python -m visualize.render_hoi --samples results/rethoi.npz --indices 0 5 12 --out_dir renders/rethoi

Input is the .npz written by sample/generate_eval.py. For every selected sample:
  1. the 22 generated joints are fitted with SMPLify (visualize/joints2smpl, from HOI-Diff / MDM)
     to obtain an SMPL body mesh per frame,
  2. the object's scanned mesh is placed with the generated 6-DoF pose,
  3. the scene is rendered off-screen with pyrender (OSMesa, no display needed) from a static
     camera that frames the whole motion, with the text prompt as caption.
"""
import argparse
import os
import textwrap

os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')

import cv2
import imageio
import numpy as np
import pyrender
import torch
import trimesh
from scipy.spatial.transform import Rotation

from visualize.simplify_loc2rot import joints2smpl

HUMAN_COLOR = (0.55, 0.70, 0.90, 1.0)
OBJECT_COLOR = (0.95, 0.60, 0.25, 1.0)


def fit_smpl(joints, device_id=0):
    """joints [T, 22, 3] -> SMPL vertices [T, 6890, 3] in the same world frame, and faces."""
    num_frames = len(joints)
    j2s = joints2smpl(num_frames=num_frames, device_id=device_id, cuda=torch.cuda.is_available())
    keypoints = torch.tensor(joints, dtype=torch.float32, device=j2s.device)
    conf = torch.ones(22, device=j2s.device)
    vertices, fit_joints, _, _, cam_t, _ = j2s.smplify(j2s.init_mean_pose, j2s.init_mean_shape,
                                                       j2s.cam_trans_zero, keypoints, conf_3d=conf)
    vertices, fit_joints = vertices.cpu().numpy(), fit_joints[:, :22].cpu().numpy()
    cam_t = cam_t.detach().cpu().numpy().reshape(-1, 1, 3)
    # SMPLify returns the body without its fitted translation; apply it if that matches the input
    if np.abs(fit_joints + cam_t - joints).mean() < np.abs(fit_joints - joints).mean():
        vertices, fit_joints = vertices + cam_t, fit_joints + cam_t
    print(f'  SMPLify mean joint error: {np.linalg.norm(fit_joints - joints, axis=-1).mean() * 100:.1f} cm')
    return vertices, j2s.smplify.smpl.faces


def object_vertices(mesh, obj_pose):
    rot = Rotation.from_rotvec(obj_pose[:, :3]).as_matrix()
    return np.einsum('tij,nj->tni', rot, mesh.vertices) + obj_pose[:, None, 3:]


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, true_up, -forward, eye
    return pose


def floor_mesh(center, size):
    """Checkerboard floor at y = 0."""
    tiles, n = [], 12
    step = size / n
    for i in range(n):
        for k in range(n):
            box = trimesh.creation.box(extents=(step, 0.01, step))
            box.apply_translation((center[0] - size / 2 + (i + 0.5) * step, -0.005,
                                   center[2] - size / 2 + (k + 0.5) * step))
            shade = 0.85 if (i + k) % 2 == 0 else 0.70
            box.visual.vertex_colors = np.array([shade, shade, shade, 1.0]) * 255
            tiles.append(box)
    return pyrender.Mesh.from_trimesh(trimesh.util.concatenate(tiles), smooth=False)


def add_caption(frame, text, width):
    lines = textwrap.wrap(text, width=max(20, width // 11))
    bar = np.full((22 * len(lines) + 12, frame.shape[1], 3), 255, dtype=np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(bar, line, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    return np.concatenate([bar, frame], axis=0)


def render_sequence(human_verts, faces, obj_verts, obj_faces, caption, out_path, size=512, fps=20):
    all_pts = np.concatenate([human_verts.reshape(-1, 3), obj_verts.reshape(-1, 3)])
    lo, hi = all_pts.min(0), all_pts.max(0)
    center = (lo + hi) / 2
    extent = max(hi[0] - lo[0], hi[2] - lo[2], hi[1] - lo[1], 1.5)
    target = np.array([center[0], 0.75, center[2]])
    eye = target + np.array([0.9, 0.55, 1.6]) / np.linalg.norm([0.9, 0.55, 1.6]) * (extent * 0.8 + 1.6)

    scene = pyrender.Scene(bg_color=(1.0, 1.0, 1.0, 1.0), ambient_light=(0.35, 0.35, 0.35))
    scene.add(floor_mesh(center, max(extent * 2.0, 4.0)))
    cam_pose = look_at(eye, target)
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4.0), pose=cam_pose)
    scene.add(pyrender.DirectionalLight(intensity=3.0), pose=cam_pose)
    scene.add(pyrender.DirectionalLight(intensity=2.0), pose=look_at(target + np.array([-2.0, 4.0, 1.0]), target))

    human_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=HUMAN_COLOR, metallicFactor=0.1, roughnessFactor=0.7)
    obj_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=OBJECT_COLOR, metallicFactor=0.1, roughnessFactor=0.6)
    renderer = pyrender.OffscreenRenderer(size, size)
    frames, nodes = [], []
    for t in range(len(human_verts)):
        for node in nodes:
            scene.remove_node(node)
        nodes = [scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(human_verts[t], faces, process=False),
                                                      material=human_mat)),
                 scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(obj_verts[t], obj_faces, process=False),
                                                      material=obj_mat))]
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
        frames.append(add_caption(color.copy(), caption, size))
    renderer.delete()
    imageio.mimsave(out_path, frames, fps=fps, codec='libx264', quality=8, macro_block_size=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--samples', required=True, help='npz written by sample/generate_eval.py')
    p.add_argument('--indices', type=int, nargs='*', default=None, help='sample indices (default: first --num)')
    p.add_argument('--num', type=int, default=4)
    p.add_argument('--gt', action='store_true', help='render the ground-truth motion of the same test sequences')
    p.add_argument('--out_dir', default='./renders')
    p.add_argument('--data_dir', default='./dataset/behave_t2m')
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--device', type=int, default=0)
    args = p.parse_args()

    s = np.load(args.samples, allow_pickle=True)
    indices = args.indices if args.indices else list(range(args.num))
    os.makedirs(args.out_dir, exist_ok=True)
    if args.gt:
        from data_loaders.scripts.motion_process import recover_from_ric
        mean = np.load(os.path.join(args.data_dir, 'Mean_local.npy'))
        std = np.load(os.path.join(args.data_dir, 'Std_local.npy'))

    for i in indices:
        seq, text, length = str(s['seq_name'][i]), str(s['text'][i]), int(s['lengths'][i])
        obj_name = seq.split('_')[2]
        if args.gt:
            raw = np.load(os.path.join(args.data_dir, 'sequences_263_rep', seq, 'hoi_motion.npy'))[:length]
            joints = recover_from_ric(torch.from_numpy(raw[:, :263]).float(), 22).numpy()
            obj_pose = raw[:, 263:]
        else:
            joints, obj_pose = s['joints'][i][:length], s['obj_pose'][i][:length]
        print(f'[{i}] {seq} ({length} frames): {text}')
        human_verts, faces = fit_smpl(joints, args.device)
        mesh = trimesh.load(os.path.join(args.data_dir, 'object_mesh', obj_name, f'{obj_name}.obj'),
                            force='mesh', process=False)
        obj_verts = object_vertices(mesh, obj_pose)
        out = os.path.join(args.out_dir, f'{i:03d}_{seq}{"_gt" if args.gt else ""}.mp4')
        render_sequence(human_verts, faces, obj_verts, mesh.faces, text, out, size=args.size)
        print('  ->', out)


if __name__ == '__main__':
    main()
