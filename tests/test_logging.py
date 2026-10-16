import importlib.util
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from simple_seg.dashboard import build_command
from simple_seg.reporting import describe_config


class ReportingTests(unittest.TestCase):
    def test_auxiliary_and_query_losses_are_reported_separately(self):
        cfg = dict(model=dict(type='EncoderDecoder',
            decode_head=dict(loss_decode=dict(type='CrossEntropyLoss', loss_weight=1.0)),
            auxiliary_head=dict(loss_decode=dict(type='CrossEntropyLoss', loss_weight=0.4))),
            optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.01)))
        result = describe_config(cfg)
        self.assertEqual([r['path'] for r in result['losses']], ['decode_head.loss_decode', 'auxiliary_head.loss_decode'])
        self.assertEqual(result['optimizer']['type'], 'SGD')
        cfg['model']['decode_head'] = dict(loss_cls=dict(type='mmdet.CrossEntropyLoss', loss_weight=2),
            loss_mask=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=5),
            loss_dice=dict(type='mmdet.DiceLoss', loss_weight=5))
        self.assertEqual(len(describe_config(cfg)['losses']), 4)

    def test_dashboard_uses_local_native_tensorboard_module(self):
        cmd = build_command('work_dirs', '127.0.0.1', 6006, 5)
        self.assertEqual(cmd[1:3], ['-m', 'tensorboard.main'])
        self.assertIn('127.0.0.1', cmd)
        self.assertIn(str(Path('work_dirs').resolve()), cmd)
        with self.assertRaises(ValueError):
            build_command('.', '127.0.0.1', 0, 5)


@unittest.skipUnless(importlib.util.find_spec('tensorboard') and importlib.util.find_spec('mmengine') and
                     importlib.util.find_spec('torch'), 'TensorBoard/MMEngine/PyTorch are required')
class NativeLoggingTests(unittest.TestCase):
    def test_native_events_json_metrics_manifest_and_status_are_saved(self):
        from mmengine.config import Config
        from mmengine.logging import MessageHub
        from mmengine.visualization import Visualizer
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        from simple_seg.logging_hooks import RunLogHook, update_run_status
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            split = root / 'train.json'
            split.write_text('[{"id": "sample", "image": "Image/a.png", "mask": "Mask/a.png"}]')
            cfg = Config(dict(train_dataloader=dict(dataset=dict(ann_file=str(split))),
                simple_seg_experiment=dict(logging=dict(tensorboard=True)),
                model=dict(type='EncoderDecoder', decode_head=dict(loss_decode=dict(type='CrossEntropyLoss'))),
                optim_wrapper=dict(optimizer=dict(type='AdamW', lr=0.00006))))
            vis = Visualizer(name='logging_' + uuid.uuid4().hex, save_dir=str(root),
                vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])
            runner = SimpleNamespace(cfg=cfg, visualizer=vis, log_dir=str(root), iter=1000,
                train_loop=SimpleNamespace(stop_training=False), message_hub=MessageHub('log_' + uuid.uuid4().hex))
            hook = RunLogHook(flush_interval=10)
            try:
                hook.before_run(runner)
                hook.before_train(runner)
                vis.add_scalars({'train/loss':0.75, 'train/lr':0.00006}, step=10)
                hook.after_train_iter(runner, 9)
                values = dict(mIoU=60., Precision=70., Recall=65., **{'F1-score':67.})
                vis.add_scalars({'val/' + k:v for k,v in values.items()}, step=1000)
                hook.after_val_epoch(runner, values)
                hook.after_test_epoch(runner, values)
                hook.after_train(runner)
                events = EventAccumulator(str(root / 'vis_data')).Reload()
                self.assertAlmostEqual(events.Scalars('train/loss')[0].value, 0.75)
                self.assertEqual(events.Scalars('val/F1-score')[0].step, 1000)
                self.assertEqual(events.Scalars('test/Recall')[0].value, 65.)
                self.assertTrue((root / 'vis_data/scalars.json').exists())
                val = json.loads((root / 'val_metrics.jsonl').read_text().strip())
                self.assertEqual(val['iter'], 1000)
                self.assertEqual(val['metrics']['Precision'], 70.)
                manifest = json.loads((root / 'run_manifest.json').read_text())
                self.assertEqual(manifest['dataset_splits']['train']['count'], 1)
                self.assertEqual(len(manifest['dataset_splits']['train']['sha256']), 64)
                self.assertEqual(manifest['training']['optimizer']['type'], 'AdamW')
                self.assertEqual(json.loads((root / 'run_status.json').read_text())['status'], 'completed')
                update_run_status(runner, 'failed', error='example test error')
                self.assertEqual(json.loads((root / 'run_status.json').read_text())['status'], 'failed')
            finally:
                vis.close()


if __name__ == '__main__':
    unittest.main()
