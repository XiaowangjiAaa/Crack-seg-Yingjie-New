"""Local TensorBoard WebUI entry point using the native TensorBoard server."""
import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


def build_command(logdir, host, port, reload_interval):
    if not 1 <= port <= 65535 or reload_interval < 1:
        raise ValueError('port must be in [1, 65535] and reload-interval must be positive')
    return [sys.executable, '-m', 'tensorboard.main', '--logdir', str(Path(logdir).resolve()),
            '--host', host, '--port', str(port), '--reload_interval', str(reload_interval)]


def main(argv=None):
    parser = argparse.ArgumentParser(description='Open the local training log WebUI (TensorBoard)')
    parser.add_argument('--logdir', default='work_dirs', help='One experiment or the root of all experiments')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=6006)
    parser.add_argument('--reload-interval', type=int, default=5)
    parser.add_argument('--print-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        command = build_command(args.logdir, args.host, args.port, args.reload_interval)
        if args.print_only:
            print(subprocess.list2cmdline(command))
            return 0
        if importlib.util.find_spec('tensorboard') is None:
            raise ValueError('TensorBoard is missing. Install: python -m pip install tensorboard==2.18.0')
        print('TensorBoard: http://{}:{} (Ctrl+C to stop)'.format(args.host, args.port), flush=True)
        print('Log directory: {}'.format(Path(args.logdir).resolve()), flush=True)
        return subprocess.run(command).returncode
    except (ValueError, OSError) as error:
        parser.exit(1, '{}\n'.format(error))
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())
