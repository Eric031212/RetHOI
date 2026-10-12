"""
Tiles rendered videos into one labelled grid video (+ an optional GIF for the README).

    python -m visualize.make_grid --rows "Ground truth" "HOI-Diff" "RetHOI (ours)" \
        --videos renders/gt/a.mp4 renders/gt/b.mp4 renders/hoidiff/a.mp4 ... --cols 2 \
        --out assets/comparison.mp4 --gif assets/comparison.gif

Videos are given row-major; shorter clips hold their last frame.
"""
import argparse

import cv2
import imageio
import numpy as np


def read(path):
    return [f for f in imageio.get_reader(path)]


def label_strip(text, height, width=150):
    strip = np.full((height, width, 3), 255, dtype=np.uint8)
    for i, line in enumerate(text.split(' ', 1) if len(text) > 12 else [text]):
        cv2.putText(strip, line, (8, height // 2 - 10 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    return strip


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--videos', nargs='+', required=True)
    p.add_argument('--rows', nargs='+', required=True, help='one label per row')
    p.add_argument('--cols', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--gif', default='')
    p.add_argument('--gif_width', type=int, default=900)
    p.add_argument('--fps', type=int, default=20)
    args = p.parse_args()
    assert len(args.videos) == len(args.rows) * args.cols

    clips = [read(v) for v in args.videos]
    h = max(c[0].shape[0] for c in clips)
    w = max(c[0].shape[1] for c in clips)
    n = max(len(c) for c in clips)

    def cell(clip, t):
        f = clip[min(t, len(clip) - 1)]
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)
        canvas[:f.shape[0], :f.shape[1]] = f[..., :3]
        return canvas

    frames = []
    for t in range(n):
        rows = []
        for r, label in enumerate(args.rows):
            cells = [cell(clips[r * args.cols + c], t) for c in range(args.cols)]
            rows.append(np.concatenate([label_strip(label, h)] + cells, axis=1))
        frames.append(np.concatenate(rows, axis=0))
    imageio.mimsave(args.out, frames, fps=args.fps, codec='libx264', quality=8, macro_block_size=1)
    print('wrote', args.out, frames[0].shape)

    if args.gif:
        scale = args.gif_width / frames[0].shape[1]
        small = [cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for f in frames[::2]]
        imageio.mimsave(args.gif, small, duration=2.0 / args.fps, loop=0)
        print('wrote', args.gif)


if __name__ == '__main__':
    main()
