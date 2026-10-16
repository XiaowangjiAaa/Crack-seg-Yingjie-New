"""Run independent single-GPU experiments, immediately reusing free GPUs."""
import argparse
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .checks import DEFAULT_MANIFEST, gpu_ids, isolated_env, jobs_from_manifest, run_checks, write_json
from .launch import PROJECT_ROOT, display_command


def stop_process(process):
    if process.poll() is not None:
        return
    if os.name == 'nt':
        # Include possible DataLoader children; operate only on this queue's PID.
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name != 'nt':
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def execute_queue(jobs, devices, work_root, fail_fast=False):
    """Each active process owns exactly one GPU until it exits."""
    work_root = Path(work_root)
    rows = [dict(name=j['name'], config=j['config'], command=j['command'], status='pending') for j in jobs]
    active, pending, failed = {}, list(range(len(jobs))), False
    def save(status):
        write_json(work_root / 'summary.json', dict(status=status, devices=devices, jobs=rows))
    save('running')
    try:
        while pending or active:
            for device in devices:
                if device in active or not pending or (fail_fast and failed):
                    continue
                index = pending.pop(0)
                job, row = jobs[index], rows[index]
                log = work_root / (job['name'] + '.console.log')
                stream = log.open('wb')
                row.update(gpu=device, console_log=str(log), started_at=time.time())
                print('Starting {} on GPU {}. Log: {}'.format(job['name'], device, log), flush=True)
                try:
                    env = isolated_env(device)
                    options = dict(cwd=str(PROJECT_ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT)
                    if os.name == 'nt':
                        options['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
                    else:
                        options['start_new_session'] = True
                    process = subprocess.Popen(job['command'], **options)
                except OSError as error:
                    stream.close()
                    failed = True
                    row.update(status='failed', exit_code=127, error=str(error), finished_at=time.time())
                else:
                    row.update(status='running', pid=process.pid)
                    active[device] = (index, process, stream)
                save('running')
            for device, (index, process, stream) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                rows[index].update(status='success' if code == 0 else 'failed', exit_code=code,
                                   finished_at=time.time())
                failed = failed or code != 0
                print('Finished {} on GPU {}: exit {}'.format(rows[index]['name'], device, code), flush=True)
                del active[device]
                save('running')
            if fail_fast and failed and not active:
                for index in pending:
                    rows[index]['status'] = 'skipped'
                pending.clear()
            if pending or active:
                time.sleep(0.2)
    except BaseException as error:
        for index, process, stream in active.values():
            stop_process(process)
            stream.close()
            rows[index].update(status='interrupted', exit_code=process.returncode, finished_at=time.time())
        for index in pending:
            rows[index]['status'] = 'skipped'
        save('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed')
        if isinstance(error, KeyboardInterrupt):
            return 130
        raise
    save('failed' if failed else 'completed')
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', nargs='?', default=DEFAULT_MANIFEST)
    parser.add_argument('--gpu-ids', default='0,1')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--work-root', help='New output directory; existing paths are refused')
    parser.add_argument('--check-level', choices=['config', 'build', 'smoke'], default='smoke')
    parser.add_argument('--check-timeout', type=float, default=900, help='Maximum seconds per preflight model')
    parser.add_argument('--fail-fast', action='store_true', help='Stop scheduling new jobs after a failed job')
    parser.add_argument('--print-only', action='store_true')
    parser.add_argument('--amp', action='store_true', help='Enable AMP in both preflight and training')
    parser.add_argument('--max-iters', type=int)
    parser.add_argument('--cfg-options', nargs='+')
    args = parser.parse_args(argv)
    train_args = ['--amp'] if args.amp else []
    if args.max_iters is not None:
        train_args += ['--max-iters', str(args.max_iters)]
    if args.cfg_options:
        train_args += ['--cfg-options'] + args.cfg_options
    try:
        devices = gpu_ids(args.gpu_ids)
        if args.check_timeout <= 0:
            raise ValueError('--check-timeout must be positive')
        work_root = Path(args.work_root or PROJECT_ROOT / 'work_dirs/gpu_queue' / uuid.uuid4().hex[:12]).resolve()
        jobs = jobs_from_manifest(args.manifest, args.python, work_root, train_args)
        if args.print_only:
            print('GPU pool: {}; preflight: {}'.format(','.join(devices), args.check_level))
            for job in jobs:
                display_command(job['command'])
            return 0
        work_root.mkdir(parents=True, exist_ok=False)
        write_json(work_root / 'summary.json', dict(status='checking', devices=devices, jobs=[]))
        print('Preflight all jobs before starting the training queue.', flush=True)
        try:
            code = run_checks(jobs, args.check_level, devices, work_root / 'preflight', args.python, args.check_timeout)
        except BaseException as error:
            write_json(work_root / 'summary.json', dict(status='preflight_failed', error=str(error), jobs=[]))
            raise
        if code:
            write_json(work_root / 'summary.json', dict(status='preflight_failed',
                       report=str(work_root / 'preflight/summary.json'), jobs=[]))
            return code
        return execute_queue(jobs, devices, work_root, args.fail_fast)
    except (ValueError, OSError) as error:
        parser.exit(1, '{}\n'.format(error))
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
