"""Prints dataset sizes and basic statistics after preprocessing."""
import argparse

import numpy as np

from data_loaders.dataset import Text2AffordDataset, Text2MotionDatasetV2

opt = argparse.Namespace(dataset='behave', data_root='./dataset', max_motion_length=196, max_text_len=20, unit_length=5)
for split in ['train', 'test']:
    m = Text2MotionDatasetV2(opt, split=split)
    a = Text2AffordDataset(opt, split=split)
    lengths = np.array([m.data_dict[n]['length'] for n in m.name_list])
    aff = np.stack([a.data_dict[n]['affordance_data'] for n in a.name_list])
    print(f'{split}: motion samples {len(m)} (lengths {lengths.min()}-{lengths.max()}, mean {lengths.mean():.0f}), '
          f'affordance samples {len(a)}, contact rate per joint {np.round(aff[:, :, 3].mean(0), 2)}')
item = m[0]
print('motion item shapes:', item[0].shape, item[1].shape, item[4].shape, 'caption:', item[2])
print('feature range after normalisation: %.2f..%.2f' % (item[4].min(), item[4].max()))
