import json
import os
import sys
import subprocess
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from simple_seg.checks import gpu_ids, isolated_env, jobs_from_manifest, run_checks, validate_experiment, validate_manifests
from simple_seg.gpu_queue import execute_queue, main as queue_main
from simple_seg.launch import PROJECT_ROOT


class QueueTests(unittest.TestCase):
    def test_default_manifest_is_not_consumed_by_training_option_values(self):
        with patch('simple_seg.gpu_queue.jobs_from_manifest', return_value=[]) as load, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(queue_main(['--print-only', '--max-iters', '123', '--amp',
                                         '--cfg-options', 'train_dataloader.batch_size=8']), 0)
        self.assertEqual(load.call_args[0][0], 'configs/batch/lightweight.json')
        self.assertEqual(load.call_args[0][3], ['--amp', '--max-iters', '123',
                                               '--cfg-options', 'train_dataloader.batch_size=8'])

    def test_preflight_timeout_terminates_worker_and_records_failure(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as folder:
            output = Path(folder) / 'checks'
            process = Mock()
            process.wait.side_effect = [subprocess.TimeoutExpired('worker', 1), 0]
            jobs = [dict(name='model', config='configs/experiments/unet.json', train_args=[])]
            with patch('simple_seg.checks.subprocess.Popen', return_value=process):
                code = run_checks(jobs, 'smoke', ['0'], output, sys.executable, timeout=1)
            self.assertEqual(code, 1)
            process.terminate.assert_called_once()
            report = json.loads((output / 'summary.json').read_text())
            self.assertFalse(report['passed'])
            self.assertEqual(report['results'][0]['exit_code'], 124)
            self.assertIn('timed out', report['results'][0]['error'])

    def test_failed_preflight_prevents_training_queue(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as folder:
            output = Path(folder) / 'run'
            with patch('simple_seg.gpu_queue.run_checks', return_value=1), \
                    patch('simple_seg.gpu_queue.execute_queue') as execute:
                status = queue_main(['--work-root', str(output)])
            self.assertEqual(status, 1)
            execute.assert_not_called()
            self.assertEqual(json.loads((output / 'summary.json').read_text())['status'], 'preflight_failed')

    def test_json_typos_and_invalid_scalar_types_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as folder:
            path = Path(folder) / 'experiment.json'
            for settings in (dict(batchsize=16), dict(batch_size=1.5), dict(amp='false')):
                path.write_text(json.dumps(dict(model_config='official.py', **settings)))
                with self.assertRaises(ValueError):
                    validate_experiment(path)

    def test_free_gpu_gets_next_job_and_child_environment_is_isolated(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as folder:
            root = Path(folder)
            jobs = []
            for name, delay in [('slow', 1.2), ('fast', 0.05), ('next', 0.05)]:
                code = ('import os,time; print(os.environ["CUDA_VISIBLE_DEVICES"]); '
                        'assert "RANK" not in os.environ and "WORLD_SIZE" not in os.environ; '
                        'time.sleep({})'.format(delay))
                jobs.append(dict(name=name, config='fake', command=[sys.executable, '-c', code]))
            with patch.dict(os.environ, dict(RANK='3', WORLD_SIZE='4')):
                status = execute_queue(jobs, ['0', '1'], root)
            self.assertEqual(status, 0)
            rows = json.loads((root / 'summary.json').read_text())['jobs']
            self.assertEqual([row['gpu'] for row in rows], ['0', '1', '1'])
            self.assertLess(rows[2]['started_at'], rows[0]['finished_at'])
            self.assertEqual((root / 'next.console.log').read_text().strip(), '1')

    def test_follow_streams_child_logs_and_reports_progress(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as folder:
            root = Path(folder)
            jobs = [dict(name='chatty', config='fake',
                         command=[sys.executable, '-c', 'print("hello-live")'])]
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                status = execute_queue(jobs, ['0'], root, follow=True)
            self.assertEqual(status, 0)
            printed = buffer.getvalue()
            self.assertIn('[chatty] hello-live', printed)
            self.assertIn('(1/1 started)', printed)
            self.assertIn('(1/1 done)', printed)
            self.assertEqual((root / 'chatty.console.log').read_text().strip(), 'hello-live')

    def test_without_follow_child_output_stays_off_the_terminal(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as folder:
            root = Path(folder)
            jobs = [dict(name='chatty', config='fake',
                         command=[sys.executable, '-c', 'print("hello-live")'])]
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                status = execute_queue(jobs, ['0'], root)
            self.assertEqual(status, 0)
            self.assertNotIn('hello-live', buffer.getvalue())
            self.assertEqual((root / 'chatty.console.log').read_text().strip(), 'hello-live')

    def test_failure_is_reported_and_next_job_runs_unless_fail_fast(self):
        for fail_fast in (False, True):
            with self.subTest(fail_fast=fail_fast), tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as folder:
                root = Path(folder)
                jobs = [dict(name='bad', config='fake', command=[sys.executable, '-c', 'raise SystemExit(7)']),
                        dict(name='after', config='fake', command=[sys.executable, '-c', 'print("ok")'])]
                self.assertEqual(execute_queue(jobs, ['0'], root, fail_fast), 1)
                report = json.loads((root / 'summary.json').read_text())
                self.assertEqual(report['jobs'][0]['exit_code'], 7)
                self.assertEqual(report['jobs'][1]['status'], 'skipped' if fail_fast else 'success')

    def test_manifest_uses_eight_single_gpu_workers_and_identical_preflight_flags(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as folder:
            jobs = jobs_from_manifest('configs/batch/lightweight.json', sys.executable, Path(folder),
                                      ['--amp', '--max-iters', '123'])
            self.assertEqual(len(jobs), 8)
            self.assertEqual([job['name'] for job in jobs[:2]], ['convnext_tiny', 'deeplabv3plus_mobilenetv2'])
            for job in jobs:
                self.assertNotIn('torch.distributed.run', job['command'])
                self.assertEqual(job['command'][job['command'].index('--launcher') + 1], 'none')
                self.assertEqual(job['train_args'], ['--amp', '--max-iters', '123'])
                self.assertEqual(job['command'].count('--work-dir'), 1)

    def test_invalid_gpu_lists_and_distributed_environment(self):
        for value in ('', '0,0', 'a,1', '0,'):
            with self.assertRaises(ValueError):
                gpu_ids(value)
        with patch.dict(os.environ, dict(RANK='2', WORLD_SIZE='4', LOCAL_RANK='2')):
            env = isolated_env('1')
        self.assertNotIn('RANK', env)
        self.assertNotIn('WORLD_SIZE', env)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '1')

    def test_all_manifest_paths_and_cross_split_leakage_are_checked(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as folder:
            root = Path(folder)
            cfg = {}
            for split in ('train', 'val', 'test'):
                (root / (split + '.png')).write_bytes(b'image')
                (root / (split + '_mask.png')).write_bytes(b'mask')
                rows = [dict(id=split, image=split + '.png', mask=split + '_mask.png')]
                manifest = root / (split + '.json')
                manifest.write_text(json.dumps(rows))
                cfg[split + '_dataloader'] = dict(dataset=dict(data_root=str(root), ann_file=str(manifest)))
            self.assertEqual(validate_manifests(cfg)['train']['count'], 1)
            (root / 'test.json').write_text((root / 'train.json').read_text())
            with self.assertRaisesRegex(ValueError, 'leakage'):
                validate_manifests(cfg)
