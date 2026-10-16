"""Durable run records alongside official MMEngine JSON and TensorBoard logs."""
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from mmengine.dist import is_main_process
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

from .reporting import describe_config


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding='utf8')
    temporary.replace(path)


def update_run_status(runner, status, **values):
    if not is_main_process():
        return
    _write_json(Path(runner.log_dir) / 'run_status.json',
                dict(status=status, updated_at_utc=datetime.now(timezone.utc).isoformat(), **values))


@HOOKS.register_module()
class RunLogHook(Hook):
    """Snapshot experiment facts, append all val metrics and flush live events."""
    priority = 'LOWEST'  # LoggerHook writes events first; then flush them to disk.

    def __init__(self, flush_interval=10):
        self.flush_interval = flush_interval

    def before_run(self, runner):
        if not is_main_process():
            return
        versions = dict(python=platform.python_version())
        for name in ('torch', 'mmcv', 'mmengine', 'mmsegmentation', 'tensorboard', 'mmdet', 'mmpretrain'):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        splits = {}
        for name in ('train', 'val', 'test'):
            dataset = runner.cfg.get(name + '_dataloader', {}).get('dataset', {})
            manifest = dataset.get('ann_file') if isinstance(dataset, dict) else None
            if manifest and Path(manifest).is_file():
                content = Path(manifest).read_bytes()
                splits[name] = dict(path=manifest, sha256=hashlib.sha256(content).hexdigest(),
                                    count=len(json.loads(content)))
        _write_json(Path(runner.log_dir) / 'run_manifest.json', dict(
            experiment=runner.cfg.get('simple_seg_experiment', {}),
            source_model_config=runner.cfg.get('simple_seg_source_config'),
            training=describe_config(runner.cfg), versions=versions, dataset_splits=splits))
        update_run_status(runner, 'initialized')

    def before_train(self, runner):
        update_run_status(runner, 'running', resumed=bool(runner.cfg.get('resume', False)), iter=runner.iter)

    def _flush(self, runner):
        if not is_main_process():
            return
        backend = runner.visualizer.get_backend('TensorboardVisBackend')
        if backend is not None:
            backend.experiment.flush()

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if (batch_idx + 1) % self.flush_interval == 0:
            self._flush(runner)

    def _record_metrics(self, runner, metrics, mode, step):
        if not is_main_process():
            return
        target = Path(runner.log_dir) / (mode + '_metrics.jsonl')
        target.parent.mkdir(parents=True, exist_ok=True)
        values = {key: float(value) for key, value in (metrics or {}).items()}
        with target.open('a', encoding='utf8') as stream:
            stream.write(json.dumps(dict(iter=step, metrics=values), ensure_ascii=False) + '\n')
            stream.flush()
        if mode == 'test':
            runner.visualizer.add_scalars({'test/' + k: v for k, v in values.items()}, step=step)
        self._flush(runner)

    def after_val_epoch(self, runner, metrics):
        self._record_metrics(runner, metrics, 'val', int(runner.iter))

    def after_test_epoch(self, runner, metrics):
        # Testing a checkpoint does not initialize a training loop just to get an x coordinate.
        self._record_metrics(runner, metrics, 'test', 0)
        update_run_status(runner, 'tested')

    def after_train(self, runner):
        stopped = bool(runner.train_loop.stop_training)
        update_run_status(runner, 'early_stopped' if stopped else 'completed', iter=runner.iter,
                          best_score=runner.message_hub.get_info('best_score'),
                          best_checkpoint=runner.message_hub.get_info('best_ckpt'))
        self._flush(runner)
