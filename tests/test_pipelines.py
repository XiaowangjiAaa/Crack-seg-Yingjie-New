import unittest

from simple_seg.adapter import adapt_model
from simple_seg.pipelines import build_scheduler, build_train_pipeline, upstream_pipeline


class PipelineTests(unittest.TestCase):
    def test_wrapped_official_pipelines_and_conflicting_concat_are_guarded(self):
        pipeline = [dict(type='LoadImageFromFile'), dict(type='LoadAnnotations')]
        cfg = dict(train_dataloader=dict(dataset=dict(type='ConcatDataset', datasets=[
            dict(pipeline=pipeline), dict(type='RepeatDataset', dataset=dict(pipeline=pipeline))])))
        self.assertEqual(upstream_pipeline(cfg, 'train'), pipeline)
        cfg['train_dataloader']['dataset']['datasets'][1]['dataset']['pipeline'] = []
        with self.assertRaisesRegex(ValueError, 'dedicated pipeline adapter'):
            upstream_pipeline(cfg, 'train')

    def test_official_random_resize_crop_and_image_only_distortion(self):
        options = dict(mode='random_crop', resize_scale=[1024, 512], ratio_range=[0.5, 2.0],
                       cat_max_ratio=0.95, horizontal_flip_prob=0.5, vertical_flip_prob=0.25,
                       photometric_distortion=True,
                       extra_transforms=[dict(type='RandomRotate', prob=0.25, degree=10)])
        pipeline = build_train_pipeline([512, 512], options)
        self.assertEqual([t['type'] for t in pipeline], ['LoadImageFromFile', 'LoadBinaryMask', 'RandomResize',
            'RandomCrop', 'RandomFlip', 'RandomFlip', 'PhotoMetricDistortion', 'RandomRotate', 'PackSegInputs'])
        self.assertTrue(pipeline[2]['keep_ratio'])
        self.assertEqual(pipeline[2]['scale'], (1024, 512))
        self.assertEqual(pipeline[3]['crop_size'], (512, 512))
        self.assertEqual(pipeline[7]['seg_pad_val'], 255)
        self.assertNotIn('seg_pad_val', options['extra_transforms'][0])

    def test_legacy_resize_and_unsafe_configs(self):
        self.assertEqual(build_train_pipeline([256, 512])[2]['scale'], (512, 256))
        for options in (dict(mode='unknown'), dict(mode='random_crop', ratio_range=[2, 0.5]),
                        dict(mode='random_crop', cat_max_ratio=0), dict(horizontal_flip_prob=1.5),
                        dict(extra_transforms=[dict(type='LoadAnnotations')]),
                        dict(extra_transforms=[dict(type='RandomRotate', auto_bound=True)])):
            with self.subTest(options=options), self.assertRaises(ValueError):
                build_train_pipeline([512, 512], options)

    def test_warmup_poly_cosine_and_short_training(self):
        settings = dict(type='poly', warmup_iters=1500, warmup_ratio=1e-6, power=1.0)
        schedule = build_scheduler(settings, 20000)
        self.assertEqual(schedule[0]['type'], 'LinearLR')
        self.assertEqual(schedule[0]['end'], schedule[1]['begin'])
        self.assertEqual(schedule[-1]['end'], 20000)
        self.assertEqual(build_scheduler(settings, 2)[0]['end'], 1)
        self.assertEqual(build_scheduler(settings, 1)[0]['type'], 'PolyLR')
        self.assertEqual(build_scheduler(dict(type='cosine', warmup_iters=500), 2000)[-1]['T_max'], 1500)
        with self.assertRaises(ValueError):
            build_scheduler(dict(warmup_iters=-1), 1000)

    def test_mask2former_feature_dimension_and_no_object_class_are_preserved(self):
        original = dict(type='EncoderDecoder', decode_head=dict(type='Mask2FormerHead', num_classes=150,
            out_channels=256, feat_channels=256, loss_cls=dict(class_weight=[1.] * 150 + [0.1])),
            data_preprocessor=dict(test_cfg=dict(size_divisor=32)))
        adapted = adapt_model(original, [512, 512])
        self.assertEqual(adapted['decode_head']['num_classes'], 2)
        self.assertEqual(adapted['decode_head']['out_channels'], 256)
        self.assertEqual(adapted['decode_head']['loss_cls']['class_weight'], [1., 1., 0.1])
        self.assertEqual(adapted['data_preprocessor']['test_cfg']['size_divisor'], 32)
        self.assertEqual(len(original['decode_head']['loss_cls']['class_weight']), 151)


if __name__ == '__main__':
    unittest.main()
