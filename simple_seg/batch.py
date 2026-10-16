"""Sequential model experiments, each using the native training launcher."""
import argparse
import json
import re
import subprocess
import sys
import uuid
from pathlib import Path

from .launch import PROJECT_ROOT, add_launch_arguments, build_launch_command, display_command


def load_jobs(manifest, args, common_args, work_root):
    records = json.loads(Path(manifest).read_text(encoding='utf8'))
    if not isinstance(records, list) or not records:
        raise ValueError('Batch manifest must be a nonempty JSON list')
    if args.nnodes != 1:
        raise ValueError('Batch orchestration supports one node; use dist_train.sh for multi-node jobs')
    jobs, names = [], set()
    for record in records:
        if isinstance(record, str):
            record = dict(config=record)
        if not isinstance(record, dict) or not isinstance(record.get('config'), str):
            raise ValueError('Every batch entry needs a config path')
        name = record.get('name', Path(record['config']).stem)
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) or name in names:
            raise ValueError('Batch names must be unique safe directory names: {}'.format(name))
        names.add(name)
        extra = record.get('args', [])
        if not isinstance(extra, list) or not all(isinstance(x, str) for x in extra):
            raise ValueError('Per-experiment args must be a list of strings')
        train_args = list(common_args) + extra
        if any(x.split('=', 1)[0] == '--work-dir' for x in train_args):
            raise ValueError('Batch manages --work-dir; choose --work-root instead')
        # Preflight every command before starting the first training job.
        command, env = build_launch_command(record['config'], args,
                                            ['--work-dir', str(work_root / name)] + train_args)
        jobs.append(dict(name=name, config=record['config'], command=command, env=env))
    return jobs


def execute_jobs(jobs, work_root, continue_on_error=False):
    # Refuse to overwrite a previous batch's summary or model outputs.
    work_root.mkdir(parents=True, exist_ok=False)
    results = [dict(name=j['name'], config=j['config'], command=j['command'],
                    mode='config_check' if '--dry-run' in j['command'] else 'train',
                    status='pending') for j in jobs]
    def save():
        target = work_root / 'summary.json'
        temporary = work_root / 'summary.tmp'
        temporary.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf8')
        temporary.replace(target)
    save()
    failed = False
    for job, result in zip(jobs, results):
        print('\nTraining {}'.format(job['name']), flush=True)
        display_command(job['command'])
        result['status'] = 'running'
        save()
        try:
            code = subprocess.run(job['command'], cwd=str(PROJECT_ROOT), env=job['env']).returncode
        except KeyboardInterrupt:
            result.update(status='interrupted', exit_code=130)
            save()
            return 130
        except OSError as error:
            code = 127
            result['error'] = str(error)
        result.update(status='success' if code == 0 else 'failed', exit_code=code)
        save()
        if code:
            failed = True
            if not continue_on_error:
                break
    print('Batch summary: {}'.format(work_root / 'summary.json'), flush=True)
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train experiments sequentially on the same GPU(s)')
    parser.add_argument('manifest', help='JSON list of experiments')
    add_launch_arguments(parser)
    parser.add_argument('--work-root', help='New directory for this batch; must not exist')
    parser.add_argument('--continue-on-error', action='store_true')
    args, common_args = parser.parse_known_args(argv)
    try:
        manifest = Path(args.manifest)
        if not manifest.is_absolute():
            manifest = PROJECT_ROOT / manifest
        work_root = Path(args.work_root or ('work_dirs/batch/' + uuid.uuid4().hex[:12]))
        if not work_root.is_absolute():
            work_root = PROJECT_ROOT / work_root
        work_root = work_root.resolve()
        jobs = load_jobs(manifest, args, common_args, work_root)
        if args.print_only:
            for job in jobs:
                display_command(job['command'])
            return 0
        return execute_jobs(jobs, work_root, args.continue_on_error)
    except (ValueError, OSError) as error:
        parser.exit(1, '{}\n'.format(error))


if __name__ == '__main__':
    sys.exit(main())
