"""Preflight each real experiment in its own process before long training jobs."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
import uuid
from pathlib import Path
from types import SimpleNamespace

from .batch import load_jobs
from .launch import PROJECT_ROOT

DEFAULT_MANIFEST = 'configs/batch/lightweight.json'
EXPERIMENT_KEYS = {'model_config', 'data_root', 'split_dir', 'size', 'batch_size', 'num_workers',
                   'max_iters', 'val_interval', 'checkpoint_interval', 'max_keep_ckpts', 'save_on_val',
                   'metrics', 'early_stopping', 'no_pretrained', 'lr', 'amp', 'accumulative_counts',
                   'grad_clip', 'scheduler', 'logging', 'seed', 'work_dir', 'augmentation'}


def validate_experiment(path):
    experiment = json.loads(Path(path).read_text(encoding='utf8'))
    if not isinstance(experiment, dict) or not isinstance(experiment.get('model_config'), str):
        raise ValueError('Experiment must be a JSON object with a model_config string')
    unknown = set(experiment) - EXPERIMENT_KEYS
    if unknown:
        raise ValueError('Unknown experiment keys: {}'.format(', '.join(sorted(unknown))))
    for key in ('batch_size', 'num_workers', 'max_iters', 'val_interval', 'checkpoint_interval',
                'max_keep_ckpts', 'accumulative_counts', 'seed'):
        if key in experiment and (isinstance(experiment[key], bool) or not isinstance(experiment[key], int)):
            raise ValueError('{} must be an integer'.format(key))
    for key in ('save_on_val', 'no_pretrained', 'amp'):
        if key in experiment and not isinstance(experiment[key], bool):
            raise ValueError('{} must be a boolean'.format(key))
    return experiment


def gpu_ids(value):
    ids = value.split(',')
    if not ids or len(ids) != len(set(ids)) or any(not s.isdigit() for s in ids):
        raise ValueError('--gpu-ids must contain distinct numeric IDs, e.g. 0,1')
    return ids


def isolated_env(gpu):
    env = os.environ.copy()
    for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'GROUP_RANK', 'ROLE_RANK',
                'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED='1')
    return env


def jobs_from_manifest(manifest, python, work_root, common_args=()):
    options = SimpleNamespace(gpus=1, gpu_ids=None, nnodes=1, node_rank=0,
                              master_addr='127.0.0.1', port=29500, python=python)
    path = Path(manifest)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    jobs = load_jobs(path, options, common_args, work_root)
    for job in jobs:
        # Keep precisely the flags used by the eventual training worker.
        index = job['command'].index('--launcher')
        flags = job['command'][index + 2:]
        job['train_args'] = flags[2:] if flags[:1] == ['--work-dir'] else flags
    return jobs


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')
    temporary.replace(path)


def validate_manifests(cfg):
    """Validate every record and file path, and detect split leakage."""
    all_ids, all_images, all_masks, summary = set(), set(), set(), {}
    for split in ('train', 'val', 'test'):
        dataset = cfg[split + '_dataloader']['dataset']
        root = Path(dataset['data_root']).resolve()
        manifest = Path(dataset['ann_file'])
        records = json.loads(manifest.read_text(encoding='utf8'))
        if not isinstance(records, list) or not records:
            raise ValueError('Empty or invalid split: {}'.format(manifest))
        for record in records:
            if not isinstance(record, dict) or any(not isinstance(record.get(k), str) or not record[k]
                                                    for k in ('id', 'image', 'mask')):
                raise ValueError('Invalid manifest record in {}'.format(manifest))
            image, mask = [(root / record[k]).resolve() for k in ('image', 'mask')]
            if record['id'] in all_ids or image in all_images or mask in all_masks:
                raise ValueError('Duplicate sample or split leakage: {}'.format(record['id']))
            for path in (image, mask):
                if root not in path.parents or not path.is_file():
                    raise ValueError('Missing file or path outside dataset: {}'.format(path))
            all_ids.add(record['id'])
            all_images.add(image)
            all_masks.add(mask)
        summary[split] = dict(count=len(records), manifest=str(manifest),
                              sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    return summary


def check_worker(args):
    row = dict(experiment=args.experiment, level=args.level, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
               status='failed')
    try:
        validate_experiment(args.experiment)
        from .cli import training_config, load_experiment_modules
        args.root = str(PROJECT_ROOT)
        args.resume = None
        args.local_rank = 0
        args.launcher = 'none'
        cfg = training_config(args)
        row['splits'] = validate_manifests(cfg)
        row['batch_size_per_gpu'] = cfg.train_dataloader.batch_size
        row['max_iters'] = cfg.train_cfg.max_iters
        row['config_sha256'] = hashlib.sha256(Path(args.experiment).read_bytes()).hexdigest()
        row['train_args'] = dict(amp=args.amp, max_iters=args.max_iters, cfg_options=args.cfg_options)
        cfg.dump(str(Path(args.work_dir) / 'resolved_config.json'))
        if args.level != 'config':
            import torch
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError('Runtime preflight requires exactly one visible CUDA GPU per worker')
            from mmseg.utils import register_all_modules
            register_all_modules()
            load_experiment_modules(cfg)
            from mmengine.runner import Runner
            # Keep all model, AMP, batch-size, optimizer and pipeline settings.
            # Disable weight/event artifacts and limit val/test to one real sample.
            cfg.default_hooks.checkpoint = None
            cfg.custom_hooks = [h for h in cfg.custom_hooks if h['type'] == 'MetricNamesHook']
            cfg.visualizer.vis_backends = []
            cfg.val_dataloader.dataset.indices = [0]
            cfg.test_dataloader.dataset.indices = [0]
            cfg.train_dataloader.num_workers = 0
            cfg.val_dataloader.num_workers = 0
            cfg.test_dataloader.num_workers = 0
            for split in ('train', 'val', 'test'):
                cfg[split + '_dataloader'].persistent_workers = False
            cfg.train_cfg.update(max_iters=1, val_interval=1)
            runner = Runner.from_cfg(cfg)
            try:
                if args.level == 'build':
                    runner.model.init_weights()
                    runner.build_optim_wrapper(cfg.optim_wrapper)
                else:
                    runner.train()
                    runner.test()
                    torch.cuda.synchronize()
                    row['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 1024 ** 3
                row['gpu_name'] = torch.cuda.get_device_name(0)
            finally:
                runner.visualizer.close()
        row['status'] = 'passed'
    except Exception as error:
        row['error'] = '{}: {}'.format(type(error).__name__, error)
        traceback.print_exc()
    write_json(Path(args.work_dir) / 'result.json', row)
    print('{}: {}'.format(row['status'], args.experiment), flush=True)
    return 0 if row['status'] == 'passed' else 1


def run_checks(jobs, level, devices, output, python, timeout=900):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, job in enumerate(jobs):
        folder = output / job['name']
        folder.mkdir()
        log = folder / 'console.log'
        device = devices[index % len(devices)]
        command = [python, str(PROJECT_ROOT / 'scripts/check_experiments.py'), '--worker',
                   '--experiment', str((PROJECT_ROOT / job['config']).resolve()), '--level', level,
                   '--work-dir', str(folder)] + job['train_args']
        print('Checking {} [{}] on GPU {} ...'.format(job['name'], level, device), flush=True)
        timed_out = False
        with log.open('wb') as stream:
            process = subprocess.Popen(command, cwd=str(PROJECT_ROOT), env=isolated_env(device),
                                       stdout=stream, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                code = 124
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        result_path = folder / 'result.json'
        row = json.loads(result_path.read_text(encoding='utf8')) if result_path.exists() else dict(
            status='failed', error='Worker exited without a result; see console.log')
        row.update(name=job['name'], exit_code=code, console_log=str(log))
        if timed_out:
            row['error'] = 'Preflight timed out after {} seconds'.format(timeout)
        if code:
            row['status'] = 'failed'
        rows.append(row)
        write_json(output / 'summary.json', dict(level=level, results=rows))
        print('  {}{}'.format(row['status'], ': ' + row['error'] if row.get('error') else ''), flush=True)
    passed = all(row['status'] == 'passed' for row in rows)
    write_json(output / 'summary.json', dict(level=level, passed=passed, results=rows))
    print('Check report: {}'.format(output / 'summary.json'), flush=True)
    return 0 if passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', nargs='?', default=DEFAULT_MANIFEST)
    parser.add_argument('--level', choices=['config', 'build', 'smoke'], default='config')
    parser.add_argument('--gpu-ids', default='0,1')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--output')
    parser.add_argument('--timeout', type=float, default=900, help='Maximum seconds per check worker')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--max-iters', type=int)
    parser.add_argument('--cfg-options', nargs='+')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--experiment', help=argparse.SUPPRESS)
    parser.add_argument('--work-dir', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return check_worker(args)
    try:
        devices = gpu_ids(args.gpu_ids)
        if args.timeout <= 0:
            raise ValueError('--timeout must be positive')
        output = Path(args.output or PROJECT_ROOT / 'work_dirs/checks' / uuid.uuid4().hex[:12]).resolve()
        extra = (['--amp'] if args.amp else [])
        if args.max_iters is not None:
            extra += ['--max-iters', str(args.max_iters)]
        if args.cfg_options:
            extra += ['--cfg-options'] + args.cfg_options
        jobs = jobs_from_manifest(args.manifest, args.python, output, extra)
        return run_checks(jobs, args.level, devices, output, args.python, args.timeout)
    except (ValueError, OSError) as error:
        parser.exit(1, '{}\n'.format(error))
    except KeyboardInterrupt:
        return 130
