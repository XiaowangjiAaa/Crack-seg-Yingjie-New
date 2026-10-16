"""Real MMEngine config integration tests; require mmengine but not torch/mmcv."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from simple_seg.adapter import build_config


@unittest.skipUnless(importlib.util.find_spec('mmengine'), 'MMEngine not installed')
class UpstreamConfigTests(unittest.TestCase):
    def test_explicit_training_crop_does_not_change_validation_or_inference(self):
        root = Path(__file__).resolve().parents[1]
        experiment = json.loads((root / 'configs/experiments/convnext_tiny.json').read_text())
        baseline = build_config(experiment, root, validate_data=False)
        overridden = build_config(dict(experiment, size=[384, 384]), root, validate_data=False)
        crop = next(t for t in overridden.train_dataloader.dataset.pipeline if t['type'] == 'RandomCrop')
        self.assertEqual(crop['crop_size'], (384, 384))
        self.assertEqual(overridden.model.data_preprocessor.size, (384, 384))
        self.assertEqual(overridden.val_dataloader, baseline.val_dataloader)
        self.assertEqual(overridden.test_dataloader, baseline.test_dataloader)
        self.assertEqual(overridden.model.test_cfg, baseline.model.test_cfg)
        self.assertEqual(overridden.test_pipeline, baseline.test_pipeline)

    def test_saving_is_independent_and_early_stopping_can_be_disabled(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root) as folder:
            for split in ('train', 'val', 'test'):
                (Path(folder) / (split + '.json')).write_text('[]')
            experiment = json.loads((root / 'configs/experiments/unet.json').read_text())
            experiment.update(split_dir=folder, val_interval=1000, checkpoint_interval=700,
                              early_stopping=None, save_on_val=False, max_keep_ckpts=2)
            cfg = build_config(experiment, root)
            self.assertEqual(cfg.train_cfg.val_interval, 1000)
            self.assertEqual(cfg.default_hooks.checkpoint.interval, 700)
            self.assertEqual(cfg.default_hooks.checkpoint.max_keep_ckpts, 2)
            self.assertFalse(cfg.default_hooks.checkpoint.save_on_val)
            self.assertEqual([h['type'] for h in cfg.custom_hooks], ['MetricNamesHook', 'RunLogHook'])
            for changes in (dict(checkpoint_interval=0), dict(metrics=['mDice']),
                            dict(early_stopping=dict(patience_iters=0)),
                            dict(early_stopping=dict(min_delta=-1)), dict(max_keep_ckpts=0)):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    build_config(dict(experiment, **changes), root)

    def test_string_custom_import_is_preserved_as_one_module(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root) as folder:
            temp = Path(folder)
            for split in ('train', 'val', 'test'):
                (temp / (split + '.json')).write_text('[]')
            source = temp / 'model.py'
            source.write_text("custom_imports = dict(imports='math')\nmodel = dict(type='EncoderDecoder')\n"
                              "train_dataloader = dict(dataset=dict(pipeline=[]))\n"
                              "val_dataloader = dict(dataset=dict(pipeline=[]))\n"
                              "test_dataloader = dict(dataset=dict(pipeline=[]))\n")
            cfg = build_config(dict(model_config=str(source), split_dir=folder), root)
            self.assertEqual(cfg.custom_imports.imports, ['math', 'simple_seg.dataset', 'simple_seg.hooks', 'simple_seg.logging_hooks'])

    def test_examples_resolve_and_use_binary_datasets(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root) as folder:
            temp = Path(folder)
            for split in ('train', 'val', 'test'):
                (temp / (split + '.json')).write_text('[]')
            for path in (root / 'configs/experiments').glob('*.json'):
                with self.subTest(config=path.name):
                    experiment = json.loads(path.read_text())
                    experiment['split_dir'] = str(temp)
                    cfg = build_config(experiment, root)
                    self.assertEqual(cfg.model.decode_head.num_classes, 2)
                    self.assertEqual(cfg.train_dataloader.dataset.type, 'BinaryPairDataset')
                    self.assertEqual(cfg.train_dataloader.batch_size, 16)
                    self.assertEqual(cfg.train_cfg.max_iters, 30000)
                    self.assertEqual(cfg.train_cfg.val_interval, 265)
                    self.assertEqual(cfg.default_hooks.checkpoint.interval, 265)
                    self.assertEqual(cfg.train_dataloader.dataset.data_root, str((root / 'dataset').resolve()))
                    self.assertEqual(cfg.val_dataloader.batch_size, 1)
                    self.assertEqual(cfg.train_dataloader.num_workers, 0)
                    self.assertEqual(cfg.param_scheduler[-1].end, experiment['max_iters'])
                    self.assertEqual(cfg.default_hooks.checkpoint.interval, experiment['checkpoint_interval'])
                    self.assertEqual(cfg.default_hooks.checkpoint.max_keep_ckpts, -1)
                    self.assertTrue(cfg.default_hooks.checkpoint.save_on_val)
                    self.assertEqual(cfg.default_hooks.checkpoint.save_best, 'mIoU')
                    self.assertEqual(cfg.val_evaluator.iou_metrics, ['mIoU', 'mDice', 'mFscore'])
                    self.assertEqual(cfg.test_evaluator.iou_metrics, ['mIoU', 'mDice', 'mFscore'])
                    self.assertEqual(cfg.val_evaluator.beta, 1)
                    self.assertEqual(cfg.test_evaluator.beta, 1)
                    self.assertIn('MetricNamesHook', [h['type'] for h in cfg.custom_hooks])
                    self.assertIn('RunLogHook', [h['type'] for h in cfg.custom_hooks])
                    self.assertEqual([b.type for b in cfg.visualizer.vis_backends], ['LocalVisBackend', 'TensorboardVisBackend'])
                    self.assertEqual(cfg.default_hooks.logger.interval, 10)
                    self.assertEqual(cfg.log_processor.window_size, 10)
                    self.assertEqual(cfg.custom_hooks[0].patience_iters, 5000)
                    self.assertNotIn('LoadBinaryMask', [t['type'] for t in cfg.test_pipeline])
                    self.assertEqual(cfg.val_dataloader.dataset.pipeline[2]['type'], 'LoadBinaryMask')
                    expected_resize = 'RandomChoiceResize' if 'mask2former' in path.stem else 'RandomResize'
                    self.assertEqual(cfg.train_dataloader.dataset.pipeline[2]['type'], expected_resize)
                    self.assertEqual(cfg.train_dataloader.dataset.pipeline[3]['type'], 'RandomCrop')
                    self.assertTrue(any(t['type'] == 'PhotoMetricDistortion' for t in cfg.train_dataloader.dataset.pipeline))
                    # The same fully expanded file must load without upstream _base_ dependencies.
                    resolved = temp / (path.stem + '.py')
                    cfg.dump(str(resolved))
                    from mmengine.config import Config
                    from simple_seg.pipelines import upstream_pipeline, adapt_binary_pipeline
                    source = Config.fromfile(str(root / experiment['model_config']), import_custom_modules=False)
                    self.assertEqual(cfg.train_dataloader.dataset.pipeline,
                                     adapt_binary_pipeline(upstream_pipeline(source, 'train')))
                    self.assertEqual(cfg.val_dataloader.dataset.pipeline,
                                     adapt_binary_pipeline(upstream_pipeline(source, 'val')))
                    self.assertEqual(cfg.test_dataloader.dataset.pipeline,
                                     adapt_binary_pipeline(upstream_pipeline(source, 'test')))
                    self.assertEqual(cfg.model.test_cfg, source.model.test_cfg)
                    self.assertEqual(cfg.model.data_preprocessor, source.model.data_preprocessor)
                    self.assertEqual(cfg.test_pipeline,
                                     adapt_binary_pipeline(upstream_pipeline(source, 'test'), prediction=True))
                    reloaded = Config.fromfile(str(resolved), import_custom_modules=False)
                    self.assertEqual(reloaded.model.decode_head.num_classes, 2)
                    if 'mask2former' in path.stem:
                        self.assertEqual(cfg.model.decode_head.out_channels, 256)
                        self.assertEqual(cfg.model.decode_head.loss_cls.class_weight, [1.0, 1.0, 0.1])
                        self.assertEqual(cfg.model.decode_head.num_stuff_classes, 2)
                        self.assertIn('mmdet.models', cfg.custom_imports.imports)
                        self.assertEqual(cfg.optim_wrapper.clip_grad.max_norm, 0.01)
                    if 'convnext' in path.stem:
                        self.assertIn('mmpretrain.models', cfg.custom_imports.imports)
                        self.assertEqual(cfg.optim_wrapper.constructor, 'LearningRateDecayOptimizerConstructor')
                        self.assertEqual(cfg.optim_wrapper.type, 'AmpOptimWrapper')
                        self.assertEqual(cfg.optim_wrapper.paramwise_cfg.decay_rate, 0.9)
                        fp32 = build_config(dict(experiment, amp=False), root)
                        self.assertEqual(fp32.optim_wrapper.type, 'OptimWrapper')
                        self.assertNotIn('loss_scale', fp32.optim_wrapper)


if __name__ == '__main__':
    unittest.main()
