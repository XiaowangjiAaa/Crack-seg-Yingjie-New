import argparse
import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from simple_seg import cli
from simple_seg.batch import execute_jobs, load_jobs
from simple_seg.launch import PROJECT_ROOT, add_launch_arguments, build_launch_command


def launch_args(**overrides):
    parser = argparse.ArgumentParser()
    add_launch_arguments(parser)
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class LaunchTests(unittest.TestCase):
    def test_dual_gpu_uses_native_torchrun_and_forwards_options(self):
        args = launch_args(gpus=2, gpu_ids='1,3', port=29601)
        with patch.dict(os.environ, {}, clear=True):
            command, env = build_launch_command('configs/experiments/unet.json', args,
                                                 ['--amp', '--resume'])
        self.assertEqual(command[1:3], ['-m', 'torch.distributed.run'])
        self.assertEqual(command[command.index('--nproc_per_node') + 1], '2')
        self.assertEqual(command[command.index('--launcher') + 1], 'pytorch')
        self.assertEqual(command[command.index('--master_port') + 1], '29601')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '1,3')
        self.assertEqual(command[-2:], ['--amp', '--resume'])

    def test_single_gpu_uses_normal_worker_and_conflicts_fail(self):
        command, _ = build_launch_command('configs/experiments/unet.json', launch_args())
        self.assertNotIn('torch.distributed.run', command)
        self.assertEqual(command[-1], 'none')
        with self.assertRaisesRegex(ValueError, 'managed'):
            build_launch_command('configs/experiments/unet.json', launch_args(), ['--launcher=none'])
        with self.assertRaisesRegex(ValueError, 'distinct'):
            build_launch_command('configs/experiments/unet.json', launch_args(gpus=2, gpu_ids='0,0'))

    def test_rank_aliases_and_torchrun_env_are_respected(self):
        for flag in ('--local_rank', '--local-rank'):
            with patch.object(cli, 'run_train') as train:
                cli.main(['train', 'example.json', '--launcher', 'pytorch', flag, '1', '--amp'])
            args = train.call_args[0][0]
            self.assertEqual(args.local_rank, 1)
            self.assertEqual(args.launcher, 'pytorch')
            self.assertTrue(args.amp)


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.root = Path(self.temp.name)
        self.manifest = self.root / 'jobs.json'
        self.manifest.write_text(json.dumps([
            dict(name='first', config='configs/experiments/unet.json'),
            dict(name='second', config='configs/experiments/segformer_b0.json', args=['--amp'])]))

    def tearDown(self):
        self.temp.cleanup()

    def jobs(self):
        return load_jobs(self.manifest, launch_args(gpus=2), [], self.root / 'outputs')

    def test_distinct_work_dirs_and_preflight_all_configs(self):
        jobs = self.jobs()
        paths = [job['command'][job['command'].index('--work-dir') + 1] for job in jobs]
        self.assertEqual(len(set(paths)), 2)
        self.manifest.write_text(json.dumps(['configs/experiments/unet.json', 'missing.json']))
        with self.assertRaisesRegex(ValueError, 'not found'):
            self.jobs()
        self.assertFalse((self.root / 'outputs').exists())

    def test_failure_stops_queue_and_records_pending_jobs(self):
        with patch('simple_seg.batch.subprocess.run') as run, contextlib.redirect_stdout(io.StringIO()):
            run.return_value.returncode = 9
            self.assertEqual(execute_jobs(self.jobs(), self.root / 'outputs'), 1)
        self.assertEqual(run.call_count, 1)
        report = json.loads((self.root / 'outputs/summary.json').read_text())
        self.assertEqual([r['status'] for r in report], ['failed', 'pending'])
        self.assertEqual(report[0]['exit_code'], 9)

    def test_continue_after_failure_and_refuse_existing_batch(self):
        with patch('simple_seg.batch.subprocess.run') as run, contextlib.redirect_stdout(io.StringIO()):
            run.side_effect = [argparse.Namespace(returncode=1), argparse.Namespace(returncode=0)]
            self.assertEqual(execute_jobs(self.jobs(), self.root / 'outputs', True), 1)
        self.assertEqual(run.call_count, 2)
        report = json.loads((self.root / 'outputs/summary.json').read_text())
        self.assertEqual([r['status'] for r in report], ['failed', 'success'])
        with self.assertRaises(FileExistsError):
            execute_jobs(self.jobs(), self.root / 'outputs')

    def test_invalid_names_and_work_dir_overrides_rejected(self):
        self.manifest.write_text(json.dumps([dict(name='../escape', config='configs/experiments/unet.json')]))
        with self.assertRaisesRegex(ValueError, 'safe'):
            self.jobs()
        self.manifest.write_text(json.dumps(['configs/experiments/unet.json']))
        with self.assertRaisesRegex(ValueError, 'work-dir'):
            load_jobs(self.manifest, launch_args(), ['--work-dir=shared'], self.root / 'outputs')


@unittest.skipUnless(importlib.util.find_spec('mmengine'), 'MMEngine not installed')
class WorkerConfigTests(unittest.TestCase):
    def test_training_error_is_logged_and_visualizer_is_closed(self):
        from mmengine.config import Config
        from types import SimpleNamespace
        from unittest.mock import Mock
        cfg = Config(dict(work_dir='unused'))
        runner = SimpleNamespace(train=Mock(side_effect=ValueError('training failed')),
                                 logger=Mock(), visualizer=Mock())
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(cli, 'experiment_config', return_value=cfg), \
                patch.object(cli, 'validate_distributed_runtime'), \
                patch.object(cli, 'load_experiment_modules'), \
                patch.object(cli, 'save_resolved_config'), \
                patch('mmengine.runner.Runner.from_cfg', return_value=runner), \
                patch('mmengine.dist.is_main_process', return_value=True), \
                patch('simple_seg.logging_hooks.update_run_status') as status, \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                cli.main(['train', 'fake.json'])
        self.assertEqual(error.exception.code, 1)
        self.assertEqual(status.call_args[0][1], 'failed')
        runner.logger.exception.assert_called_once()
        runner.visualizer.close.assert_called_once()

    def test_custom_modules_are_registered_before_runner_build(self):
        from mmengine.config import Config
        from types import SimpleNamespace
        from unittest.mock import Mock
        events = Mock()
        cfg = Config(dict(work_dir='unused'))
        runner = SimpleNamespace(train=Mock())
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(cli, 'experiment_config', return_value=cfg), \
                patch.object(cli, 'validate_distributed_runtime'), \
                patch.object(cli, 'load_experiment_modules') as imports, \
                patch('mmengine.runner.Runner.from_cfg', return_value=runner) as build, \
                patch('mmengine.dist.is_main_process', return_value=False):
            events.attach_mock(imports, 'imports')
            events.attach_mock(build, 'build')
            events.attach_mock(runner.train, 'train')
            cli.main(['train', 'fake.json'])
        self.assertEqual([call[0] for call in events.mock_calls], ['imports', 'build', 'train'])

    def test_nonzero_rank_does_not_dump_and_env_rank_is_not_overwritten(self):
        from mmengine.config import Config
        cfg = Config(dict(optim_wrapper=dict(type='OptimWrapper')))
        with patch.dict(os.environ, dict(RANK='1', LOCAL_RANK='1'), clear=True), \
                patch.object(cli, 'experiment_config', return_value=cfg), \
                patch.object(cli, 'save_resolved_config') as dump:
            cli.main(['train', 'fake.json', '--launcher', 'pytorch', '--dry-run'])
            dump.assert_not_called()
            self.assertEqual(os.environ['LOCAL_RANK'], '1')
            self.assertEqual(cfg.launcher, 'pytorch')

    def test_rank_zero_amp_and_cli_work_dir_priority(self):
        from mmengine.config import Config
        cfg = Config(dict(optim_wrapper=dict(type='OptimWrapper'), work_dir='old'))
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(cli, 'experiment_config', return_value=cfg), \
                patch.object(cli, 'save_resolved_config') as dump:
            cli.main(['train', 'fake.json', '--launcher', 'pytorch', '--dry-run', '--amp',
                      '--work-dir', 'work_dirs/chosen', '--cfg-options', 'work_dir=wrong'])
            dump.assert_called_once_with(cfg)
            self.assertEqual(cfg.optim_wrapper.type, 'AmpOptimWrapper')
            self.assertEqual(cfg.optim_wrapper.loss_scale, 'dynamic')
            self.assertEqual(cfg.work_dir, str((Path.cwd() / 'work_dirs/chosen').resolve()))
            self.assertEqual(os.environ['LOCAL_RANK'], '0')


if __name__ == '__main__':
    unittest.main()
