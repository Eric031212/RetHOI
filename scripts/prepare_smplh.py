"""Converts the AMASS-style SMPL+H release (smplh.tar.xz from mano.is.tue.mpg.de,
"Extended SMPL+H model") into body_models/smplh/SMPLH_{MALE,FEMALE,NEUTRAL}.npz loadable by smplx.

The AMASS files have no hand PCA basis, which smplx reads even with use_pca=False; an identity
basis and zero hand mean are added (i.e. flat hand mean, as used by process/get_affordance.py).

    python scripts/prepare_smplh.py /path/to/extracted/smplh
"""
import os
import sys

import numpy as np

src = sys.argv[1]
dst = './body_models/smplh'
os.makedirs(dst, exist_ok=True)
for gender in ['male', 'female', 'neutral']:
    data = dict(np.load(os.path.join(src, gender, 'model.npz'), allow_pickle=True))
    for side in 'lr':
        data.setdefault(f'hands_components{side}', np.eye(45, dtype=np.float64))
        data.setdefault(f'hands_mean{side}', np.zeros(45, dtype=np.float64))
    np.savez(os.path.join(dst, f'SMPLH_{gender.upper()}.npz'), **data)
    print('wrote', os.path.join(dst, f'SMPLH_{gender.upper()}.npz'))
