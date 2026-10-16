import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from simple_seg.data import index_files, prepare, read_binary_mask
from simple_seg.adapter import adapt_model


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'Image').mkdir()
        (self.root / 'Mask').mkdir()
        for i in range(10):
            Image.new('RGB', (8, 6)).save(self.root / 'Image' / ('%d.jpg' % i))
            Image.fromarray(np.array([[0, 255] * 4] * 6, dtype=np.uint8)).save(self.root / 'Mask' / ('%d.png' % i))

    def tearDown(self):
        self.temp.cleanup()

    def test_white_is_foreground_and_zero_one_masks_work(self):
        mask = read_binary_mask(self.root / 'Mask/0.png')
        self.assertEqual(set(np.unique(mask)), {0, 1})
        self.assertEqual(mask[0, 1], 1)
        Image.fromarray(mask).save(self.root / 'Mask/0.png')
        np.testing.assert_array_equal(read_binary_mask(self.root / 'Mask/0.png'), mask)

    def test_split_is_reproducible_disjoint_complete_and_guarded(self):
        result = prepare(self.root)
        self.assertEqual(result['counts'], dict(train=7, val=2, test=1))
        def read(name):
            return json.loads((self.root / 'splits' / (name + '.json')).read_text())
        before = {k: read(k) for k in ('train', 'val', 'test')}
        ids = [r['id'] for samples in before.values() for r in samples]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 10)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            prepare(self.root)
        prepare(self.root, force=True)
        self.assertEqual(before, {k: read(k) for k in before})

    def test_unpaired_size_and_invalid_values_fail(self):
        (self.root / 'Mask/0.png').unlink()
        with self.assertRaisesRegex(ValueError, 'Unpaired'):
            prepare(self.root)
        Image.new('L', (3, 3), 255).save(self.root / 'Mask/0.png')
        with self.assertRaisesRegex(ValueError, 'size mismatch'):
            prepare(self.root)
        Image.new('L', (8, 6), 128).save(self.root / 'Mask/0.png')
        with self.assertRaisesRegex(ValueError, 'non-binary'):
            prepare(self.root)

    def test_duplicate_stem_rejected(self):
        Image.new('RGB', (8, 6)).save(self.root / 'Image/0.png')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            index_files(self.root / 'Image')

    def test_color_mask_rejected(self):
        Image.new('RGB', (8, 6), (255, 0, 0)).save(self.root / 'Mask/0.png')
        with self.assertRaisesRegex(ValueError, 'colored'):
            read_binary_mask(self.root / 'Mask/0.png')

    def test_high_bit_depth_values_are_not_silently_clipped(self):
        Image.fromarray(np.full((6, 8), 512, dtype=np.uint16)).save(self.root / 'Mask/0.png')
        with self.assertRaisesRegex(ValueError, 'non-binary'):
            read_binary_mask(self.root / 'Mask/0.png')


class ModelAdaptationTests(unittest.TestCase):
    def test_all_heads_and_preprocessor_and_init_are_adapted(self):
        original = dict(type='CascadeEncoderDecoder',
            backbone=dict(init_cfg=dict(type='Pretrained', checkpoint='remote.pth')),
            decode_head=[dict(num_classes=19, out_channels=19), dict(num_classes=19)],
            auxiliary_head=dict(num_classes=19),
            data_preprocessor=dict(size=(512, 1024), size_divisor=32, mean=[1, 2, 3]),
            test_cfg=dict(mode='slide', crop_size=(512, 1024), stride=(341, 683)))
        adapted = adapt_model(original, [256, 256], no_pretrained=True)
        self.assertEqual(adapted['decode_head'][0]['num_classes'], 2)
        self.assertEqual(adapted['decode_head'][0]['out_channels'], 2)
        self.assertEqual(adapted['auxiliary_head']['num_classes'], 2)
        self.assertNotIn('init_cfg', adapted['backbone'])
        self.assertEqual(adapted['data_preprocessor']['size'], (256, 256))
        self.assertEqual(adapted['data_preprocessor']['mean'], [1, 2, 3])
        self.assertEqual(adapted['test_cfg']['stride'], (341, 683))
        self.assertEqual(adapted['test_cfg']['crop_size'], (512, 1024))
        self.assertEqual(original['decode_head'][0]['num_classes'], 19)

    def test_specialized_segmentor_rejected(self):
        with self.assertRaisesRegex(ValueError, 'dedicated adapter'):
            adapt_model(dict(type='DepthEstimator'), [256, 256])


if __name__ == '__main__':
    unittest.main()
