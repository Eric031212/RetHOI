"""Writes a small README-friendly GIF from an mp4:  python -m visualize.shrink_gif in.mp4 out.gif [width] [step]"""
import sys

import cv2
import imageio

src, dst = sys.argv[1], sys.argv[2]
width = int(sys.argv[3]) if len(sys.argv) > 3 else 960
step = int(sys.argv[4]) if len(sys.argv) > 4 else 3
frames = [f for f in imageio.get_reader(src)][::step]
scale = width / frames[0].shape[1]
small = [cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for f in frames]
imageio.mimsave(dst, small, duration=step / 20.0, loop=0)
print('wrote', dst, small[0].shape, len(small), 'frames')
