"""Exercise real MMEngine hooks and disk checkpoints without an MMSeg model."""
import importlib.util
import json
import pickle
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('mmengine') and importlib.util.find_spec('torch'),
                     'MMEngine and PyTorch are required')
class TrainingHookTests(unittest.TestCase):
    def setUp(self):
        from mmengine.logging import MessageHub
        from simple_seg.hooks import IterEarlyStoppingHook, ValidationCheckpointHook
        self.early_type, self.checkpoint_type = IterEarlyStoppingHook, ValidationCheckpointHook
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runner = SimpleNamespace(iter=0, epoch=0, max_iters=20000, max_epochs=1,
            work_dir=str(self.root), logger=Mock(), train_loop=SimpleNamespace(stop_training=False),
            message_hub=MessageHub('test_' + uuid.uuid4().hex))
        def save(out_dir, filename, file_client_args=None, **kwargs):
            # Keep the real hook's flags, metadata and MessageHub payload so
            # resume behavior can be checked independently of model/CUDA setup.
            payload = dict(meta=kwargs.get('meta'), message_hub=self.runner.message_hub.state_dict(),
                           save_optimizer=kwargs.get('save_optimizer', True))
            with (Path(out_dir) / filename).open('wb') as output:
                pickle.dump(payload, output)
        self.runner.save_checkpoint = Mock(side_effect=save)

    def tearDown(self):
        self.temp.cleanup()

    def validate(self, hook, step, score):
        self.runner.iter = step
        hook.after_val_epoch(self.runner, dict(mIoU=score))

    def test_precision_recall_f1_aliases_reach_native_logging_and_test_results(self):
        from mmengine.hooks import RuntimeInfoHook
        from mmengine.runner.priority import get_priority
        from simple_seg.hooks import MetricNamesHook
        hook = MetricNamesHook()
        self.assertLess(get_priority(hook.priority), get_priority(RuntimeInfoHook.priority))
        metrics = dict(mIoU=50.0, mPrecision=70.0, mRecall=60.0, mFscore=64.5)
        hook.after_val_epoch(self.runner, metrics)
        self.assertEqual(metrics['Precision'], 70.0)
        self.assertEqual(metrics['Recall'], 60.0)
        self.assertEqual(metrics['F1-score'], 64.5)
        # RuntimeInfoHook consumes this same dict for the logger/JSON scalars.
        RuntimeInfoHook().after_val_epoch(self.runner, metrics)
        self.assertEqual(self.runner.message_hub.get_scalar('val/F1-score').current(), 64.5)
        test_metrics = dict(mPrecision=75.0, mRecall=65.0, mFscore=69.0)
        hook.after_test_epoch(self.runner, test_metrics)
        self.assertEqual(test_metrics['F1-score'], 69.0)

    def test_patience_is_training_iters_and_ties_are_not_improvement(self):
        hook = self.early_type(patience_iters=5000)
        hook.before_train(self.runner)
        for step in (1000, 2000, 3000, 4000, 5000):
            self.validate(hook, step, 50.0)
            self.assertFalse(self.runner.train_loop.stop_training)
        self.validate(hook, 6000, 50.0)
        self.assertTrue(self.runner.train_loop.stop_training)

    def test_improvement_resets_and_irregular_validation_uses_real_steps(self):
        hook = self.early_type(patience_iters=5000)
        self.validate(hook, 1000, 50)
        self.validate(hook, 3000, 51)
        self.validate(hook, 7500, 50)
        self.assertFalse(self.runner.train_loop.stop_training)
        self.validate(hook, 8100, 50)
        self.assertTrue(self.runner.train_loop.stop_training)
        state = self.runner.message_hub.get_info(hook.state_key)
        self.assertEqual(state['best_iter'], 3000)

    def test_resume_retains_patience_and_threshold(self):
        from mmengine.logging import MessageHub
        hook = self.early_type(patience_iters=5000, min_delta=0.1)
        self.validate(hook, 1000, 50)
        self.validate(hook, 4000, 50.05)
        saved = pickle.loads(pickle.dumps(self.runner.message_hub.state_dict()))
        self.runner.message_hub = MessageHub('resume_' + uuid.uuid4().hex)
        self.runner.message_hub.load_state_dict(saved)
        resumed = self.early_type(patience_iters=5000, min_delta=0.1)
        resumed.before_train(self.runner)
        self.validate(resumed, 5000, 50)
        self.assertFalse(self.runner.train_loop.stop_training)
        self.validate(resumed, 6000, 50)
        self.assertTrue(self.runner.train_loop.stop_training)

    def test_invalid_or_missing_metric_does_not_silently_disable_stopping(self):
        with self.assertRaises(ValueError):
            self.early_type(patience_iters=0)
        hook = self.early_type()
        with self.assertRaisesRegex(RuntimeError, 'mIoU'):
            hook.after_val_epoch(self.runner, dict(mDice=50))
        self.validate(hook, 1000, float('nan'))
        self.assertTrue(self.runner.train_loop.stop_training)

    def test_periodic_every_val_and_best_snapshots_have_correct_steps(self):
        checkpoint = self.checkpoint_type(interval=700, save_on_val=True,
            max_keep_ckpts=-1, save_best='mIoU', rule='greater')
        checkpoint.before_train(self.runner)
        early = self.early_type()
        # Native after_train_iter sees a zero-based iteration index.
        self.runner.iter = 699
        checkpoint.after_train_iter(self.runner, batch_idx=699)
        self.assertTrue((self.root / 'iter_700.pth').exists())
        for step, score in ((1000, 50), (2000, 51), (3000, 49)):
            self.validate(early, step, score)
            checkpoint.after_val_epoch(self.runner, dict(mIoU=score))
            self.assertTrue((self.root / ('iter_{}.pth'.format(step))).exists())
        self.assertEqual([p.name for p in self.root.glob('best_*.pth')], ['best_mIoU_iter_2000.pth'])
        self.assertFalse((self.root / 'iter_1001.pth').exists())
        with (self.root / 'iter_3000.pth').open('rb') as source:
            saved = pickle.load(source)
        self.assertEqual(saved['meta']['iter'], 3000)
        self.assertTrue(saved['save_optimizer'])
        self.assertEqual(saved['message_hub']['runtime_info'][early.state_key]['best_iter'], 2000)
        self.assertEqual(Path((self.root / 'last_checkpoint').read_text()).name, 'iter_3000.pth')

    def test_early_stop_saves_current_checkpoint_even_when_val_saving_disabled(self):
        early = self.early_type(patience_iters=5000)
        checkpoint = self.checkpoint_type(interval=10000, save_on_val=False, save_best='mIoU', rule='greater')
        checkpoint.before_train(self.runner)
        for step in (1000, 6000):
            self.validate(early, step, 50)
            checkpoint.after_val_epoch(self.runner, dict(mIoU=50))
        self.assertTrue(self.runner.train_loop.stop_training)
        self.assertTrue((self.root / 'iter_6000.pth').exists())

    def test_native_runner_trains_saves_resumes_and_stops_without_resetting_patience(self):
        import torch
        from torch.utils.data import DataLoader, Dataset
        from mmengine.evaluator import BaseMetric, Evaluator
        from mmengine.model import BaseModel
        from mmengine.runner import Runner
        from simple_seg.hooks import MetricNamesHook
        from simple_seg.logging_hooks import RunLogHook

        class ToyData(Dataset):
            metainfo = dict(classes=('background', 'foreground'))

            def __len__(self):
                return 32

            def __getitem__(self, index):
                return dict(inputs=torch.ones(1))

        class ToyModel(BaseModel):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))

            def forward(self, inputs, data_samples=None, mode='tensor'):
                if mode == 'loss':
                    return dict(loss=(inputs * self.weight).square().mean())
                return [dict(value=1) for _ in inputs]

        class PlateauMetric(BaseMetric):
            def __init__(self):
                super().__init__(collect_device='cpu', prefix='')

            def process(self, data_batch, data_samples):
                self.results.extend(data_samples)

            def compute_metrics(self, results):
                return dict(mIoU=50.0, mPrecision=70.0, mRecall=65.0, mFscore=67.0)

        runners = []
        def create(steps, checkpoint=None):
            backends = [dict(type='LocalVisBackend')]
            if importlib.util.find_spec('tensorboard'):
                backends.append(dict(type='TensorboardVisBackend'))
            with patch('mmengine.runner.runner.collect_env', return_value={}):
                runner = Runner(model=ToyModel(), work_dir=str(self.root),
                    train_dataloader=DataLoader(ToyData(), batch_size=2),
                    train_cfg=dict(type='IterBasedTrainLoop', max_iters=steps, val_interval=2),
                    val_dataloader=DataLoader(ToyData(), batch_size=2), val_cfg=dict(type='ValLoop'),
                    val_evaluator=Evaluator(PlateauMetric()), optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.01)),
                    default_hooks=dict(checkpoint=self.checkpoint_type(interval=3, save_on_val=True,
                        save_best='mIoU', rule='greater'),
                        logger=dict(type='LoggerHook', interval=1, ignore_last=False, log_metric_by_epoch=False)),
                    visualizer=dict(type='Visualizer', vis_backends=backends),
                    log_processor=dict(by_epoch=False, window_size=1, log_with_hierarchy=True),
                    custom_hooks=[self.early_type(patience_iters=4), MetricNamesHook(), RunLogHook(flush_interval=1)],
                    log_level='ERROR',
                    env_cfg=dict(dist_cfg=dict(backend='gloo')), load_from=checkpoint, resume=checkpoint is not None)
                runners.append(runner)
                return runner
        try:
            # The project pins torch 2.1. Newer local torch defaults to a different
            # checkpoint deserialization policy; use the pinned behavior in this test.
            with patch.dict('os.environ', dict(TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')):
                first = create(4)
                first.train()
                self.assertEqual(first.iter, 4)
                resumed = create(12, str(self.root / 'iter_4.pth'))
                resumed.train()
                self.assertEqual(resumed.iter, 6)  # First best at 2; no increase for 4 iters.
                self.assertTrue((self.root / 'iter_6.pth').exists())
                self.assertEqual([p.name for p in self.root.glob('best_*.pth')], ['best_mIoU_iter_2.pth'])
                saved = torch.load(self.root / 'iter_6.pth', weights_only=False)
                self.assertIn('optimizer', saved)
                self.assertEqual(saved['meta']['iter'], 6)
                self.assertTrue(saved['message_hub']['runtime_info'][self.early_type.state_key]['stopped'])
                status = json.loads((Path(resumed.log_dir) / 'run_status.json').read_text())
                self.assertEqual(status['status'], 'early_stopped')
                if importlib.util.find_spec('tensorboard'):
                    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
                    events = EventAccumulator(str(Path(resumed.log_dir) / 'vis_data')).Reload()
                    self.assertIn('train/loss', events.Tags()['scalars'])
                    self.assertEqual(events.Scalars('val/mIoU')[-1].step, 6)
                    self.assertEqual(events.Scalars('val/F1-score')[-1].value, 67.0)
        finally:
            # Release native Windows log file handles before TemporaryDirectory cleanup.
            for runner in runners:
                for handler in list(runner.logger.handlers):
                    handler.close()
                    runner.logger.removeHandler(handler)


if __name__ == '__main__':
    unittest.main()
