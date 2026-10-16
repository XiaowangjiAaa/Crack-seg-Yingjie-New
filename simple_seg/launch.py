"""Launch native MMEngine training workers; never implement a second train loop."""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def add_launch_arguments(parser):
    parser.add_argument('--gpus', type=int, default=1, help='Worker count / GPUs per node')
    parser.add_argument('--gpu-ids', help='Visible device IDs, e.g. 0,1 (single node)')
    parser.add_argument('--nnodes', type=int, default=int(os.environ.get('NNODES', '1')))
    parser.add_argument('--node-rank', type=int, default=int(os.environ.get('NODE_RANK', '0')))
    parser.add_argument('--master-addr', default=os.environ.get('MASTER_ADDR', '127.0.0.1'))
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', '29500')))
    parser.add_argument('--python', default=sys.executable, help='Python in the installed MMSeg environment')
    parser.add_argument('--print-only', action='store_true', help='Print worker command without launching it')


def build_launch_command(config, args, train_args=()):
    if args.gpus < 1 or args.nnodes < 1 or not 0 <= args.node_rank < args.nnodes:
        raise ValueError('gpus/nnodes must be positive; node-rank must be in [0, nnodes)')
    if not 1 <= args.port <= 65535:
        raise ValueError('port must be in [1, 65535]')
    source = Path(config)
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    if not source.is_file():
        raise ValueError('Experiment not found: {}'.format(source))
    # The launcher owns these options; accepting conflicting worker options can
    # accidentally run several independent single-GPU jobs into the same directory.
    if any(x.split('=', 1)[0] in ('--launcher', '--root') for x in train_args):
        raise ValueError('--launcher and --root are managed by the training script')
    env = os.environ.copy()
    if args.gpu_ids:
        ids = args.gpu_ids.split(',')
        if len(ids) != args.gpus or len(set(ids)) != len(ids) or not all(x.isdigit() for x in ids):
            raise ValueError('--gpu-ids must contain exactly --gpus distinct numeric device IDs')
        env['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    distributed = args.gpus > 1 or args.nnodes > 1
    command = [args.python]
    if distributed:
        command += ['-m', 'torch.distributed.run', '--nproc_per_node', str(args.gpus),
                    '--nnodes', str(args.nnodes), '--node_rank', str(args.node_rank),
                    '--master_addr', args.master_addr, '--master_port', str(args.port)]
    command += [str(PROJECT_ROOT / 'tools/train.py'), str(source.resolve()),
                '--root', str(PROJECT_ROOT), '--launcher', 'pytorch' if distributed else 'none']
    return command + list(train_args), env


def display_command(command):
    print(subprocess.list2cmdline(command) if os.name == 'nt' else ' '.join(shlex.quote(x) for x in command), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Single/multi-GPU MMSeg-style training launcher')
    parser.add_argument('config', help='Experiment JSON; paths relative to the project root')
    add_launch_arguments(parser)
    args, train_args = parser.parse_known_args(argv)
    try:
        command, env = build_launch_command(args.config, args, train_args)
        display_command(command)
        if args.print_only:
            return 0
        return subprocess.run(command, cwd=str(PROJECT_ROOT), env=env).returncode
    except (ValueError, OSError) as error:
        parser.exit(1, '{}\n'.format(error))
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())
