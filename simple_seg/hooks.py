"""Small extensions to MMEngine's checkpoint/early-stop hook workflow."""
from math import isfinite

from mmengine.hooks import CheckpointHook, Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class MetricNamesHook(Hook):
    """Expose readable aliases for native macro Precision/Recall/F1 results."""
    # RuntimeInfoHook must see the added keys before updating the log scalars.
    priority = 'HIGHEST'
    aliases = {'Precision': 'mPrecision', 'Recall': 'mRecall', 'F1-score': 'mFscore'}

    def after_val_epoch(self, runner, metrics):
        if metrics:
            for name, native in self.aliases.items():
                if native in metrics:
                    metrics[name] = metrics[native]

    def after_test_epoch(self, runner, metrics):
        self.after_val_epoch(runner, metrics)


@HOOKS.register_module()
class IterEarlyStoppingHook(Hook):
    """Check validation plateaus in completed training iterations, not val calls.

    State lives in MessageHub so native resumable checkpoints retain patience.
    The first validation establishes the initial best score. All distributed
    ranks receive the same aggregated metrics and make the same stop decision.
    """
    priority = 'LOW'  # Persist updated early-stop state before CheckpointHook.
    state_key = 'simple_seg_early_stopping'

    def __init__(self, patience_iters=5000, min_delta=0.0, monitor='mIoU'):
        if isinstance(patience_iters, bool) or not isinstance(patience_iters, int) or patience_iters <= 0:
            raise ValueError('patience_iters must be a positive integer')
        if not isfinite(float(min_delta)) or min_delta < 0:
            raise ValueError('min_delta must be finite and nonnegative')
        self.patience_iters = patience_iters
        self.min_delta = float(min_delta)
        self.monitor = monitor

    def before_train(self, runner):
        if not hasattr(runner.train_loop, 'stop_training'):
            raise ValueError('Iteration early stopping requires IterBasedTrainLoop')
        # Runner has already restored MessageHub when resuming at this point.
        state = runner.message_hub.get_info(self.state_key)
        if state is not None and state.get('monitor') != self.monitor:
            raise ValueError('Cannot resume early stopping with a different monitor')

    def after_val_epoch(self, runner, metrics):
        if not metrics or self.monitor not in metrics:
            raise RuntimeError('Missing early-stopping metric: {}'.format(self.monitor))
        step = int(runner.iter)  # ValLoop runs after the training loop increments iter.
        score = float(metrics[self.monitor])
        state = runner.message_hub.get_info(self.state_key)
        if state is None:
            state = dict(monitor=self.monitor, best_score=None, best_iter=None)
        else:
            state = dict(state)
        state.update(last_val_iter=step, patience_iters=self.patience_iters, min_delta=self.min_delta)
        reason = None
        if not isfinite(score):
            reason = 'non-finite {}={}'.format(self.monitor, score)
        elif state['best_score'] is None or score > state['best_score'] + self.min_delta:
            state.update(best_score=score, best_iter=step)
        elif step - state['best_iter'] >= self.patience_iters:
            reason = '{} has not improved for {} iterations (best {} at iter {})'.format(
                self.monitor, step - state['best_iter'], state['best_score'], state['best_iter'])
        state.update(last_score=score, stopped=reason is not None, stop_reason=reason)
        runner.message_hub.update_info(self.state_key, state)
        if reason is not None:
            runner.train_loop.stop_training = True
            runner.logger.info('Early stopping at iter {}: {}'.format(step, reason))


@HOOKS.register_module()
class ValidationCheckpointHook(CheckpointHook):
    """Native periodic/best checkpoints plus a resumable snapshot at every val."""

    def __init__(self, save_on_val=True, **kwargs):
        if kwargs.get('by_epoch', False):
            raise ValueError('ValidationCheckpointHook requires by_epoch=False')
        kwargs['by_epoch'] = False
        super().__init__(**kwargs)
        self.save_on_val = save_on_val
        self._inside_validation = False
        self._saved_in_validation = False

    def _save_checkpoint_with_step(self, runner, step, meta):
        super()._save_checkpoint_with_step(runner, step, meta)
        if self._inside_validation:
            self._saved_in_validation = True

    def after_val_epoch(self, runner, metrics):
        self._inside_validation = True
        self._saved_in_validation = False
        try:
            # Retain the official best comparison, naming, deletion and resume state.
            # It can already refresh the normal snapshot when the best changes.
            super().after_val_epoch(runner, metrics)
            stopped = getattr(runner.train_loop, 'stop_training', False)
            if (self.save_on_val or stopped) and not self._saved_in_validation:
                self._save_checkpoint_with_step(runner, int(runner.iter),
                                                dict(epoch=runner.epoch, iter=int(runner.iter)))
        finally:
            self._inside_validation = False
