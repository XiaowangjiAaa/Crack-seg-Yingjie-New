import argparse
import json
import os
from pathlib import Path

from .data import prepare
from .upstream import fetch_configs, list_configs


def experiment_config(args):
    from .adapter import build_config
    path = Path(args.experiment).resolve()
    experiment = json.loads(path.read_text(encoding='utf8'))
    if getattr(args, 'max_iters', None) is not None:
        experiment['max_iters'] = args.max_iters
    return build_config(experiment, args.root, getattr(args, 'work_dir', None))


def training_config(args):
    cfg = experiment_config(args)
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    cfg.launcher = args.launcher
    if args.cfg_options:
        cfg.merge_from_dict(parse_overrides(args.cfg_options))
    # Match native MMSeg: explicit CLI work-dir wins over config overrides.
    if args.work_dir:
        cfg.work_dir = str((Path(args.root) / args.work_dir).resolve())
    if args.amp:
        if cfg.optim_wrapper.type not in ('OptimWrapper', 'AmpOptimWrapper'):
            raise ValueError('--amp requires OptimWrapper or AmpOptimWrapper')
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'
    if args.resume:
        cfg.resume = True
        cfg.load_from = None if args.resume == 'auto' else str(Path(args.resume).resolve())
        if cfg.load_from and not Path(cfg.load_from).is_file():
            raise ValueError('Resume checkpoint does not exist: {}'.format(cfg.load_from))
    return cfg


def run_train(args):
    cfg = training_config(args)
    if args.dry_run:
        # Process groups are not initialized in a config-only run.
        rank = next((int(os.environ[key]) for key in
                     ('RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK') if key in os.environ), 0)
        if rank == 0:
            save_resolved_config(cfg)
            print('Config resolution only; no model built or training performed.')
        return
    validate_distributed_runtime(cfg)
    load_experiment_modules(cfg)
    from mmengine.runner import Runner
    if cfg.get('runner_type'):
        from mmseg.registry import RUNNERS
        runner = RUNNERS.build(cfg)
    else:
        runner = Runner.from_cfg(cfg)
    from mmengine.dist import is_main_process
    if is_main_process():
        save_resolved_config(cfg)
    try:
        runner.train()
    except BaseException as error:
        from .logging_hooks import update_run_status
        if is_main_process():
            runner.logger.exception('Training interrupted or failed')
        update_run_status(runner, 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                          error='{}: {}'.format(type(error).__name__, error))
        raise
    finally:
        visualizer = getattr(runner, 'visualizer', None)
        if visualizer is not None:
            visualizer.close()


def save_resolved_config(cfg):
    output = Path(cfg.work_dir)
    output.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(output / 'resolved_config.py'))
    print('Resolved config: {}'.format(output / 'resolved_config.py'), flush=True)


def load_experiment_modules(cfg):
    from mmengine.utils import import_modules_from_strings
    imports = cfg.get('custom_imports')
    if imports:
        try:
            import_modules_from_strings(**imports)
        except ImportError as error:
            raise ImportError('{}; for ConvNeXt/Mask2Former install: '
                              'python -m pip install -r requirements-models.txt'.format(error)) from error


def validate_distributed_runtime(cfg):
    if cfg.launcher == 'none':
        if int(os.environ.get('WORLD_SIZE', '1')) > 1:
            raise ValueError('Multiple workers detected; pass --launcher pytorch when using torchrun')
        return
    import torch.distributed as dist
    if cfg.env_cfg.dist_cfg.get('backend', 'nccl') == 'nccl' and not dist.is_nccl_available():
        raise ValueError('NCCL is unavailable in this PyTorch build. Native Windows does not support NCCL; '
                         'use Linux/WSL2 with CUDA-enabled PyTorch for the default MMSeg GPU DDP workflow.')
    if cfg.launcher == 'pytorch' and not all(k in os.environ for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK')):
        raise ValueError('--launcher pytorch requires torchrun or tools/dist_train.sh; '
                         'use python scripts/train.py CONFIG --gpus 2')


def parse_overrides(values):
    # Reuse MMEngine parsing for nested dicts/lists and CLI booleans.
    from mmengine.config import DictAction
    parser = argparse.ArgumentParser()
    parser.add_argument('--options', nargs='+', action=DictAction)
    return parser.parse_args(['--options'] + values).options


def run_test(args):
    from mmengine.runner import Runner
    cfg = experiment_config(args)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise ValueError('Checkpoint does not exist: {}'.format(checkpoint))
    cfg.load_from = str(checkpoint)
    if args.cfg_options:
        cfg.merge_from_dict(parse_overrides(args.cfg_options))
    load_experiment_modules(cfg)
    Runner.from_cfg(cfg).test()


def run_predict(args):
    import numpy as np
    from PIL import Image
    from mmengine.config import Config
    from mmseg.apis import inference_model, init_model
    from mmseg.utils import register_all_modules
    from .data import EXTENSIONS
    register_all_modules()
    cfg = Config.fromfile(args.config)
    model = init_model(cfg, args.checkpoint, device=args.device)
    source = Path(args.input).resolve()
    if not source.exists():
        raise ValueError('Input does not exist: {}'.format(source))
    images = [source] if source.is_file() else sorted(p for p in source.rglob('*') if p.suffix.lower() in EXTENSIONS)
    if not images:
        raise ValueError('No input images found')
    output = Path(args.output).resolve()
    if source.is_dir() and (source == output or source in output.parents):
        raise ValueError('Prediction output must be outside the input directory')
    targets = set()
    for image in images:
        relative = Path(image.name) if source.is_file() else image.relative_to(source)
        target = output / relative.with_suffix('.png')
        if target in targets or target.exists():
            raise ValueError('Duplicate or existing prediction output: {}'.format(target))
        targets.add(target)
    for image in images:
        relative = Path(image.name) if source.is_file() else image.relative_to(source)
        target = output / relative.with_suffix('.png')
        result = inference_model(model, str(image))
        label = result.pred_sem_seg.data[0].detach().cpu().numpy()
        if not set(np.unique(label).tolist()) <= {0, 1}:
            raise ValueError('Prediction model is not binary')
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((label * 255).astype(np.uint8)).save(target)
    print('Saved {} masks to {}'.format(len(images), output))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Simple binary segmentation with MMSeg models')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='Validate pairs and write fixed train/val/test manifests')
    p.add_argument('--root', default='dataset', help='Dataset directory (default: dataset)')
    p.add_argument('--images', default='Image')
    p.add_argument('--masks', default='Mask')
    p.add_argument('--output', default='splits')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val-ratio', type=float, default=0.2)
    p.add_argument('--test-ratio', type=float, default=0.1)
    p.add_argument('--force', action='store_true')
    p = sub.add_parser('fetch-configs', help='Fetch pinned upstream configs (no model code)')
    p.add_argument('--root', default='.')
    p.add_argument('--archive', help='Offline mmsegmentation v1.2.2 source ZIP')
    p = sub.add_parser('models', help='List preserved upstream training configs')
    p.add_argument('query', nargs='?', default='')
    p.add_argument('--root', default='.')
    for command in ('train', 'test'):
        p = sub.add_parser(command)
        p.add_argument('experiment')
        p.add_argument('--root', default='.')
        p.add_argument('--work-dir')
        p.add_argument('--cfg-options', nargs='+')
        if command == 'train':
            p.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
            p.add_argument('--local_rank', '--local-rank', type=int, default=0)
            p.add_argument('--amp', action='store_true')
            p.add_argument('--max-iters', type=int)
            p.add_argument('--resume', nargs='?', const='auto')
            p.add_argument('--dry-run', action='store_true')
        else:
            p.add_argument('checkpoint')
    p = sub.add_parser('predict')
    p.add_argument('config', help='Training output resolved_config.py')
    p.add_argument('checkpoint')
    p.add_argument('input')
    p.add_argument('--output', default='predictions')
    p.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            print(json.dumps(prepare(args.root, args.images, args.masks, args.output,
                args.seed, args.val_ratio, args.test_ratio, args.force), indent=2, ensure_ascii=False))
        elif args.command == 'fetch-configs':
            fetch_configs(args.root, args.archive)
        elif args.command == 'models':
            print('\n'.join(list_configs(args.root, args.query)))
        elif args.command == 'train':
            run_train(args)
        elif args.command == 'test':
            run_test(args)
        elif args.command == 'predict':
            run_predict(args)
    except (ValueError, FileNotFoundError, ImportError) as error:
        parser.exit(1, '{}: {}\n'.format(type(error).__name__, error))


if __name__ == '__main__':
    main()
