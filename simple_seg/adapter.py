"""Convert resolved upstream configs into one consistent binary experiment."""
from copy import deepcopy
from math import isfinite
from pathlib import Path

from .pipelines import (adapt_binary_pipeline, build_scheduler, build_train_pipeline,
                        upstream_pipeline)


def adapt_model(model, size=None, no_pretrained=False):
    model = deepcopy(model)
    if model.get('type') not in ('EncoderDecoder', 'CascadeEncoderDecoder'):
        raise ValueError('Automatic adaptation supports EncoderDecoder/CascadeEncoderDecoder only; '
                         'this model needs a dedicated adapter: {}'.format(model.get('type')))

    def visit(node):
        if isinstance(node, dict):
            query_head = str(node.get('type', '')).split('.')[-1] == 'Mask2FormerHead'
            original_weights = node.get('loss_cls', {}).get('class_weight') if query_head else None
            no_object_weight = original_weights[-1] if isinstance(original_weights, (tuple, list)) and original_weights else 0.1
            if ('num_things_classes' in node or 'num_stuff_classes' in node) and not query_head:
                raise ValueError('Panoptic/query heads need a dedicated binary adapter')
            if 'num_classes' in node:
                node['num_classes'] = 2
                if 'out_channels' in node and not query_head:
                    node['out_channels'] = 2
            if 'class_weight' in node and isinstance(node['class_weight'], (tuple, list)):
                if len(node['class_weight']) != 2:
                    node['class_weight'] = None
            if no_pretrained:
                node.pop('pretrained', None)
                init = node.get('init_cfg')
                if isinstance(init, dict) and str(init.get('type', '')).lower() == 'pretrained':
                    node.pop('init_cfg')
                elif isinstance(init, list):
                    node['init_cfg'] = [x for x in init if str(x.get('type', '')).lower() != 'pretrained']
            for value in list(node.values()):
                visit(value)
            if query_head:
                # out_channels is the mask embedding dimension, NOT the class count.
                node.update(num_classes=2, num_things_classes=0, num_stuff_classes=2)
                node.setdefault('loss_cls', {})['class_weight'] = [1.0, 1.0, no_object_weight]
        elif isinstance(node, (tuple, list)):
            for value in node:
                visit(value)

    visit(model)
    # Preserve official training/test padding and inference settings independently.
    preprocessor = model.setdefault('data_preprocessor', dict(type='SegDataPreProcessor'))
    if size is not None:
        preprocessor.pop('size_divisor', None)
        preprocessor['size'] = tuple(size)
    preprocessor['seg_pad_val'] = 255
    return model


def build_config(experiment, project_root, work_dir=None, validate_data=True):
    from mmengine.config import Config

    root = Path(project_root).resolve()
    source = Path(experiment['model_config'])
    if not source.is_absolute():
        source = root / source
    if not source.is_file():
        raise ValueError('Model config missing: {}. Run fetch-configs first.'.format(source))
    # Delay runtime registrations so configs can be inspected without optional
    # model packages. CLI explicitly loads these imports before building Runner.
    cfg = Config.fromfile(str(source), import_custom_modules=False)
    size = experiment.get('size')
    if size is not None and (not isinstance(size, (list, tuple)) or len(size) != 2 or
                            any(isinstance(s, bool) or not isinstance(s, int) or s <= 0 for s in size)):
        raise ValueError('size must be [height, width], both positive integers')
    batch = int(experiment.get('batch_size', 16))
    workers = int(experiment.get('num_workers', 0))
    steps = int(experiment.get('max_iters', 30000))
    interval = int(experiment.get('val_interval', 265))
    if min(batch, steps, interval) <= 0 or workers < 0:
        raise ValueError('batch_size/max_iters/val_interval must be positive; num_workers >= 0')
    checkpoint_interval = experiment.get('checkpoint_interval', interval)
    max_keep = experiment.get('max_keep_ckpts', -1)
    if isinstance(checkpoint_interval, bool) or not isinstance(checkpoint_interval, int) or checkpoint_interval <= 0:
        raise ValueError('checkpoint_interval must be a positive integer')
    if isinstance(max_keep, bool) or not isinstance(max_keep, int) or (max_keep != -1 and max_keep <= 0):
        raise ValueError('max_keep_ckpts must be -1 (keep all) or a positive integer')
    save_on_val = experiment.get('save_on_val', True)
    if not isinstance(save_on_val, bool):
        raise ValueError('save_on_val must be true or false')
    metric_names = experiment.get('metrics', ['mIoU', 'mDice', 'mFscore', 'Precision', 'Recall', 'F1-score'])
    if (not isinstance(metric_names, list) or not metric_names or
            any(x not in ('mIoU', 'mDice', 'mFscore', 'Precision', 'Recall', 'F1-score') for x in metric_names) or
            len(set(metric_names)) != len(metric_names) or 'mIoU' not in metric_names):
        raise ValueError('metrics must contain mIoU; supported: mIoU, mDice, mFscore, Precision, Recall, F1-score')
    # Native IoUMetric computes Precision and Recall through its mFscore group.
    native_metrics = list(dict.fromkeys('mFscore' if x in ('Precision', 'Recall', 'F1-score') else x
                                       for x in metric_names))
    early = experiment.get('early_stopping', dict(patience_iters=5000, min_delta=0.0))
    if early is not None:
        if not isinstance(early, dict) or set(early) - {'patience_iters', 'min_delta'}:
            raise ValueError('early_stopping must be null or contain patience_iters/min_delta')
        patience = early.get('patience_iters', 5000)
        delta = early.get('min_delta', 0.0)
        if isinstance(patience, bool) or not isinstance(patience, int) or patience <= 0:
            raise ValueError('early_stopping.patience_iters must be a positive integer')
        if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not isfinite(delta) or delta < 0:
            raise ValueError('early_stopping.min_delta must be finite and nonnegative')
    cfg.model = adapt_model(cfg.model, size, experiment.get('no_pretrained', True))
    imports = cfg.get('custom_imports', {}).get('imports', [])
    imports = [imports] if isinstance(imports, str) else list(imports)
    cfg.custom_imports = dict(imports=imports + ['simple_seg.dataset', 'simple_seg.hooks', 'simple_seg.logging_hooks'], allow_failed_imports=False)
    cfg.default_scope = 'mmseg'
    data_root = (root / experiment.get('data_root', 'dataset')).resolve()
    split_dir = data_root / experiment.get('split_dir', 'splits')
    augmentation = experiment.get('augmentation', dict(mode='upstream'))
    if augmentation == dict(mode='upstream'):
        train_pipeline = adapt_binary_pipeline(upstream_pipeline(cfg, 'train'), size)
    else:
        if size is None:
            raise ValueError('Custom augmentation requires an explicit training size [height, width]')
        train_pipeline = build_train_pipeline(size, augmentation)
    # Official validation transforms retain original-size GT and independent resize.
    eval_pipeline = adapt_binary_pipeline(upstream_pipeline(cfg, 'val'))
    test_pipeline = adapt_binary_pipeline(upstream_pipeline(cfg, 'test'))
    predict_pipeline = adapt_binary_pipeline(upstream_pipeline(cfg, 'test'), prediction=True)
    for split in ('train', 'val', 'test'):
        manifest = split_dir / (split + '.json')
        if validate_data and not manifest.is_file():
            raise ValueError('Missing split {}; run prepare first.'.format(manifest))
        training = split == 'train'
        dataset = dict(type='BinaryPairDataset', data_root=str(data_root), ann_file=str(manifest),
                       pipeline=train_pipeline if training else (eval_pipeline if split == 'val' else test_pipeline),
                       reduce_zero_label=False, test_mode=not training)
        cfg[split + '_dataloader'] = dict(batch_size=batch if training else 1,
            num_workers=workers, persistent_workers=workers > 0,
            sampler=dict(type='InfiniteSampler' if training else 'DefaultSampler', shuffle=training),
            dataset=dataset)
    cfg.train_cfg = dict(type='IterBasedTrainLoop', max_iters=steps, val_interval=interval)
    cfg.val_cfg = dict(type='ValLoop')
    cfg.test_cfg = dict(type='TestLoop')
    evaluator = dict(type='IoUMetric', iou_metrics=native_metrics, beta=1)
    cfg.val_evaluator = deepcopy(evaluator)
    cfg.test_evaluator = deepcopy(evaluator)
    # Retain algorithm-specific optimizer and parameter rules; use one explicit iteration schedule.
    if cfg.get('optim_wrapper', {}).get('type') not in (None, 'OptimWrapper', 'AmpOptimWrapper'):
        raise ValueError('This optimizer wrapper needs a dedicated adapter')
    cfg.setdefault('optim_wrapper', dict(type='OptimWrapper', optimizer=dict(type='AdamW', lr=0.0001)))
    if 'lr' in experiment:
        cfg.optim_wrapper.optimizer.lr = float(experiment['lr'])
    cfg.optim_wrapper.type = 'AmpOptimWrapper' if experiment.get('amp', False) else 'OptimWrapper'
    if cfg.optim_wrapper.type == 'OptimWrapper':
        cfg.optim_wrapper.pop('loss_scale', None)
        cfg.optim_wrapper.pop('dtype', None)
    accumulation = experiment.get('accumulative_counts', 1)
    if isinstance(accumulation, bool) or not isinstance(accumulation, int) or accumulation < 1:
        raise ValueError('accumulative_counts must be a positive integer')
    cfg.optim_wrapper.accumulative_counts = accumulation
    if 'grad_clip' in experiment:
        clipping = experiment['grad_clip']
        if clipping is not None and not isinstance(clipping, dict):
            raise ValueError('grad_clip must be null or a native gradient clipping dict')
        cfg.optim_wrapper.clip_grad = clipping
    cfg.param_scheduler = build_scheduler(experiment.get('scheduler'), steps)
    logging = experiment.get('logging', {})
    if not isinstance(logging, dict) or set(logging) - {'interval', 'window_size', 'tensorboard'}:
        raise ValueError('logging supports interval, window_size and tensorboard')
    log_interval, log_window = logging.get('interval', 10), logging.get('window_size', 10)
    tensorboard = logging.get('tensorboard', True)
    if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in (log_interval, log_window)):
        raise ValueError('logging interval/window_size must be positive integers')
    if not isinstance(tensorboard, bool):
        raise ValueError('logging.tensorboard must be true or false')
    cfg.default_hooks = dict(timer=dict(type='IterTimerHook'),
        logger=dict(type='LoggerHook', interval=log_interval, ignore_last=False,
                    keep_local=True, log_metric_by_epoch=False),
        param_scheduler=dict(type='ParamSchedulerHook'), sampler_seed=dict(type='DistSamplerSeedHook'),
        checkpoint=dict(type='ValidationCheckpointHook', by_epoch=False, interval=checkpoint_interval,
                        save_on_val=save_on_val, save_best='mIoU', rule='greater',
                        max_keep_ckpts=max_keep, save_last=True))
    cfg.custom_hooks = [] if early is None else [dict(type='IterEarlyStoppingHook',
        monitor='mIoU', patience_iters=patience, min_delta=float(delta))]
    cfg.custom_hooks.append(dict(type='MetricNamesHook'))
    cfg.custom_hooks.append(dict(type='RunLogHook', flush_interval=log_interval))
    cfg.log_processor = dict(by_epoch=False, window_size=log_window, log_with_hierarchy=True)
    backends = [dict(type='LocalVisBackend')]
    if tensorboard:
        backends.append(dict(type='TensorboardVisBackend'))
    cfg.visualizer = dict(type='SegLocalVisualizer', vis_backends=backends, name='visualizer')
    cfg.simple_seg_experiment = deepcopy(experiment)
    cfg.simple_seg_source_config = str(source)
    cfg.env_cfg = dict(cudnn_benchmark=False, mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
                       dist_cfg=dict(backend='nccl'))
    cfg.launcher = 'none'
    cfg.randomness = dict(seed=int(experiment.get('seed', 42)))
    cfg.work_dir = str(root / (work_dir or experiment.get('work_dir', 'work_dirs/default')))
    cfg.load_from = None
    cfg.resume = False
    cfg.train_pipeline = deepcopy(train_pipeline)
    cfg.test_pipeline = predict_pipeline
    return cfg
