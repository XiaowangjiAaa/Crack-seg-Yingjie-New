"""Registered only when an OpenMMLab command needs it."""
import json
from pathlib import Path

from mmcv.transforms import BaseTransform
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS, TRANSFORMS

from .data import read_binary_mask


@DATASETS.register_module()
class BinaryPairDataset(BaseSegDataset):
    METAINFO = dict(classes=('background', 'foreground'), palette=[[0, 0, 0], [255, 255, 255]])

    def load_data_list(self):
        records = json.loads(Path(self.ann_file).read_text(encoding='utf8'))
        if not records:
            raise ValueError('Empty split: {}'.format(self.ann_file))
        return [dict(img_path=str(Path(self.data_root) / r['image']),
                     seg_map_path=str(Path(self.data_root) / r['mask']),
                     label_map=None, reduce_zero_label=False, seg_fields=[]) for r in records]


@TRANSFORMS.register_module()
class LoadBinaryMask(BaseTransform):
    def transform(self, results):
        results['gt_seg_map'] = read_binary_mask(results['seg_map_path'])
        results.setdefault('seg_fields', []).append('gt_seg_map')
        return results
