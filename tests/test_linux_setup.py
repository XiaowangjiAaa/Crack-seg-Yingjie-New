"""Bash installer orchestration tests; Conda is mocked, no packages installed."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_setup_data import prepare_or_reuse

ROOT = Path(__file__).resolve().parents[1]
if os.name == 'nt':
    BASH = next((str(p) for p in (Path('C:/Program Files/Git/bin/bash.exe'),
                                Path('C:/Program Files/Git/usr/bin/bash.exe')) if p.is_file()), None)
else:
    BASH = shutil.which('bash')


@unittest.skipUnless(BASH, 'Bash is required')
class LinuxSetupTests(unittest.TestCase):
    def test_bash_syntax_and_cuda_cpu_previews_do_not_create_environment(self):
        subprocess.run([BASH, '-n', (ROOT / 'scripts/setup.sh').as_posix()], check=True)
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            target = Path(folder) / 'environment with spaces'
            for flags, requirement in (([], 'requirements.txt'), (['--cpu'], 'requirements-common.txt')):
                result = subprocess.run([BASH, (ROOT / 'scripts/setup.sh').as_posix(), '--print-only',
                                         '--skip-data', '--env-path', target.as_posix()] + flags,
                                        capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(requirement, result.stdout)
                self.assertIn('/bin/python', result.stdout)
                self.assertNotIn('python.exe', result.stdout)
                self.assertIn('conda run --no-capture-output --prefix', result.stdout)
                self.assertFalse(target.exists())

    def test_mocked_install_preserves_arguments_with_spaces_and_stops_on_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory(dir=ROOT) as folder:
                project = Path(folder) / 'project with spaces'
                (project / 'scripts').mkdir(parents=True)
                shutil.copy(ROOT / 'scripts/setup.sh', project / 'scripts/setup.sh')
                mock_bin = project / 'mock-bin'
                mock_bin.mkdir()
                mock = mock_bin / 'conda'
                mock.write_bytes('''#!/bin/bash
printf '%s\t' "$@" >> "$MOCK_SETUP_CALLS"
printf '\n' >> "$MOCK_SETUP_CALLS"
if [[ "$1" == create ]]; then
    shift
    while (($#)); do
        if [[ "$1" == --prefix ]]; then
            mkdir -p "$2/bin" "$2/conda-meta"
            printf '#!/bin/bash\nexit 0\n' > "$2/bin/python"
            chmod +x "$2/bin/python"
            break
        fi
        shift
    done
fi
if [[ "$*" == *" pip check"* && "${MOCK_SETUP_FAIL:-0}" == 1 ]]; then exit 17; fi
exit 0
'''.encode('utf8'))
                mock.chmod(0o755)
                calls = project / 'calls.txt'
                env = dict(os.environ, PATH=str(mock_bin) + os.pathsep + os.environ['PATH'],
                           MOCK_SETUP_CALLS=calls.as_posix(), MOCK_SETUP_FAIL='1' if fail else '0')
                result = subprocess.run([BASH, (project / 'scripts/setup.sh').as_posix(), '--skip-data'],
                                        env=env, capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 17 if fail else 0, result.stdout + result.stderr)
                lines = calls.read_text().splitlines()
                # Tab-separated fields preserve a prefix containing spaces as one argument.
                fields = lines[0].split('\t')
                self.assertEqual(fields[:2], ['create', '--prefix'])
                self.assertTrue(fields[2].endswith('project with spaces/.venv-mmseg'))
                self.assertEqual(any('verify_environment.py' in line for line in lines), not fail)
                self.assertEqual(any('audit_data_pipeline.py' in line for line in lines), not fail)
                self.assertFalse(any('prepare_setup_data.py' in line for line in lines))


class SetupDataTests(unittest.TestCase):
    def test_first_run_prepares_seven_two_one_without_touching_originals(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            (root / 'Image').mkdir()
            (root / 'Mask').mkdir()
            for index in range(10):
                Image.new('RGB', (8, 8)).save(root / 'Image' / ('{}.png'.format(index)))
                Image.new('L', (8, 8), 255).save(root / 'Mask' / ('{}.png'.format(index)))
            source = {p: p.read_bytes() for name in ('Image', 'Mask') for p in (root / name).iterdir()}
            summary = prepare_or_reuse(root)
            self.assertEqual(summary['counts'], dict(train=7, val=2, test=1))
            self.assertTrue((root / 'splits/summary.json').is_file())
            self.assertEqual(source, {p: p.read_bytes() for p in source})

    def test_existing_splits_are_unchanged_and_other_ratios_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            split = root / 'splits'
            split.mkdir()
            for name in ('train', 'val', 'test'):
                (split / (name + '.json')).write_text('[]')
            summary = split / 'summary.json'
            summary.write_text(json.dumps(dict(val_ratio=0.2, test_ratio=0.1)))
            before = {p.name: p.read_bytes() for p in split.iterdir()}
            self.assertEqual(prepare_or_reuse(root)['val_ratio'], 0.2)
            self.assertEqual(before, {p.name: p.read_bytes() for p in split.iterdir()})
            summary.write_text(json.dumps(dict(val_ratio=0.1, test_ratio=0.1)))
            with self.assertRaisesRegex(ValueError, 'not 7:2:1'):
                prepare_or_reuse(root)
            summary.unlink()
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                prepare_or_reuse(root)
