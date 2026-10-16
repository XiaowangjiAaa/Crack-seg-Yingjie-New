"""Build native MMSeg augmentation pipelines from small experiment settings."""
from copy import deepcopy
from math import isfinite


def upstream_pipeline(cfg, split):
    """Read the actual loader pipeline, including RepeatDataset wrappers."""
    def read(dataset):
        if 'pipeline' in dataset:
            return deepcopy(dataset['pipeline'])
        if 'dataset' in dataset:
            return read(dataset['dataset'])
        if dataset.get('datasets'):
            pipelines = [read(child) for child in dataset['datasets']]
            if all(pipeline == pipelines[0] for pipeline in pipelines):
                return pipelines[0]
        raise ValueError('Upstream {} dataset needs a dedicated pipeline adapter'.format(split))
    return read(cfg[split + '_dataloader']['dataset'])


def adapt_binary_pipeline(pipeline, crop_size=None, prediction=False):
    """Keep upstream transforms and order; replace only semantic label loading."""
    result = []
    for original in pipeline:
        transform = deepcopy(original)
        name = transform['type'].split('.')[-1]
        if name in ('LoadAnnotations', 'LoadBinaryMask'):
            if prediction:
                continue
            transform = dict(type='LoadBinaryMask')
        if name == 'RandomCrop' and crop_size is not None:
            transform['crop_size'] = tuple(crop_size)
        result.append(transform)
    return result


def _pair(value, name, integers=False):
    if (not isinstance(value, (list, tuple)) or len(value) != 2 or
            any(isinstance(x, bool) or not isinstance(x, (int, float)) or not isfinite(x) or x <= 0 for x in value) or
            (integers and any(not isinstance(x, int) for x in value))):
        raise ValueError('{} must contain two positive {}'.format(name, 'integers' if integers else 'numbers'))
    return tuple(value)


def build_train_pipeline(size, augmentation=None):
    options = {} if augmentation is None else deepcopy(augmentation)
    allowed = {'mode', 'resize_scale', 'ratio_range', 'cat_max_ratio', 'horizontal_flip_prob',
               'vertical_flip_prob', 'photometric_distortion', 'extra_transforms'}
    if not isinstance(options, dict) or set(options) - allowed:
        raise ValueError('Unsupported augmentation settings')
    mode = options.get('mode', 'resize')
    pipeline = [dict(type='LoadImageFromFile'), dict(type='LoadBinaryMask')]
    if mode == 'random_crop':
        scale = _pair(options.get('resize_scale', [size[1], size[0]]), 'resize_scale [width, height]', True)
        ratio = _pair(options.get('ratio_range', [0.5, 2.0]), 'ratio_range')
        if ratio[0] > ratio[1]:
            raise ValueError('ratio_range must be increasing')
        cat_ratio = options.get('cat_max_ratio', 0.95)
        if isinstance(cat_ratio, bool) or not isinstance(cat_ratio, (int, float)) or not 0 < cat_ratio <= 1:
            raise ValueError('cat_max_ratio must be in (0, 1]')
        pipeline += [dict(type='RandomResize', scale=scale, ratio_range=ratio, keep_ratio=True),
                     dict(type='RandomCrop', crop_size=tuple(size), cat_max_ratio=cat_ratio, ignore_index=255)]
    elif mode == 'resize':
        if any(k in options for k in ('resize_scale', 'ratio_range', 'cat_max_ratio')):
            raise ValueError('resize_scale/ratio_range/cat_max_ratio require mode=random_crop')
        pipeline.append(dict(type='Resize', scale=(size[1], size[0]), keep_ratio=False))
    else:
        raise ValueError('augmentation.mode must be resize or random_crop')
    for direction, default in (('horizontal', 0.5), ('vertical', 0.0)):
        prob = options.get(direction + '_flip_prob', default)
        if isinstance(prob, bool) or not isinstance(prob, (int, float)) or not 0 <= prob <= 1:
            raise ValueError('{} flip probability must be in [0, 1]'.format(direction))
        if prob:
            pipeline.append(dict(type='RandomFlip', prob=prob, direction=direction))
    photo = options.get('photometric_distortion', False)
    if photo is True:
        pipeline.append(dict(type='PhotoMetricDistortion'))
    elif isinstance(photo, dict):
        if set(photo) - {'brightness_delta', 'contrast_range', 'saturation_range', 'hue_delta'}:
            raise ValueError('Unsupported PhotoMetricDistortion parameter')
        pipeline.append(dict(type='PhotoMetricDistortion', **photo))
    elif photo is not False:
        raise ValueError('photometric_distortion must be a boolean or parameter dict')
    extra = options.get('extra_transforms', [])
    if not isinstance(extra, list):
        raise ValueError('extra_transforms must be a list of native transform dictionaries')
    for transform in extra:
        if not isinstance(transform, dict) or not isinstance(transform.get('type'), str):
            raise ValueError('Each extra transform needs a type')
        name = transform['type'].split('.')[-1]
        if name.startswith('Load') or name in ('PackSegInputs', 'Resize', 'RandomResize', 'RandomChoiceResize'):
            raise ValueError('Extra transforms cannot replace loading, resizing or packing stages')
        # Keep rotation on the crop's grid; synthetic mask padding must be ignored.
        if name == 'RandomRotate':
            if transform.get('auto_bound', False):
                raise ValueError('RandomRotate auto_bound must be false to retain the configured crop size')
            transform.setdefault('seg_pad_val', 255)
        pipeline.append(transform)
    return pipeline + [dict(type='PackSegInputs')]


def build_scheduler(settings, steps):
    settings = {} if settings is None else settings
    if not isinstance(settings, dict) or set(settings) - {'type', 'warmup_iters', 'warmup_ratio', 'power', 'eta_min'}:
        raise ValueError('Unsupported scheduler settings')
    kind = settings.get('type', 'poly')
    if kind not in ('poly', 'cosine'):
        raise ValueError('scheduler.type must be poly or cosine')
    warmup = settings.get('warmup_iters', 0)
    ratio, power, eta_min = settings.get('warmup_ratio', 1e-6), settings.get('power', 0.9), settings.get('eta_min', 0.0)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError('warmup_iters must be a nonnegative integer')
    if not isinstance(ratio, (int, float)) or not 0 < ratio <= 1:
        raise ValueError('warmup_ratio must be in (0, 1]')
    if not isinstance(power, (int, float)) or not isfinite(power) or power <= 0:
        raise ValueError('scheduler.power must be positive')
    if not isinstance(eta_min, (int, float)) or not isfinite(eta_min) or eta_min < 0:
        raise ValueError('scheduler.eta_min must be finite and nonnegative')
    warmup = min(warmup, steps - 1)  # Short smoke runs still have a decay phase.
    schedule = []
    if warmup:
        schedule.append(dict(type='LinearLR', start_factor=ratio, begin=0, end=warmup, by_epoch=False))
    decay = dict(begin=warmup, end=steps, by_epoch=False, eta_min=eta_min)
    if kind == 'poly':
        decay.update(type='PolyLR', power=power)
    else:
        decay.update(type='CosineAnnealingLR', T_max=steps - warmup)
    return schedule + [decay]
