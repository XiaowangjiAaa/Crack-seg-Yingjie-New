"""Exercise installer orchestration with mocked Conda; never install packages."""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('powershell'), 'Windows PowerShell required')
class SetupTests(unittest.TestCase):
    def test_preview_selects_cuda_or_cpu_without_creating_environment(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            target = Path(temp) / 'environment with spaces'
            for flags, requirement in (([], 'requirements.txt'), (['-Cpu'], 'requirements-common.txt')):
                result = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                    str(ROOT / 'scripts/setup.ps1'), '-PrintOnly', '-SkipData', '-EnvPath', str(target)] + flags,
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(requirement, result.stdout)
                self.assertIn('conda run --no-capture-output --prefix', result.stdout)
                self.assertNotIn('Ready. Activate:', result.stdout)
                self.assertFalse(target.exists())

    def test_failure_stops_later_steps_and_existing_splits_are_preserved(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory(dir=ROOT) as temp:
                project = Path(temp)
                (project / 'scripts').mkdir()
                shutil.copy(ROOT / 'scripts/setup.ps1', project / 'scripts/setup.ps1')
                splits = project / 'dataset/splits'
                splits.mkdir(parents=True)
                summary = dict(val_ratio=0.2, test_ratio=0.1)
                (splits / 'summary.json').write_text(json.dumps(summary))
                for split in ('train', 'val', 'test'):
                    (splits / (split + '.json')).write_text('[]')
                before = {p.name: p.read_bytes() for p in splits.iterdir()}
                driver = project / 'driver.ps1'
                driver.write_text("""$ErrorActionPreference = 'Stop'
$global:Calls = @()
function global:conda {
    $global:Calls += ,@($args)
    $global:LASTEXITCODE = 0
    if ($args -contains 'check' -and $env:MOCK_SETUP_FAIL -eq '1') { $global:LASTEXITCODE = 17 }
}
try { & (Join-Path $PSScriptRoot 'scripts/setup.ps1') }
finally { ConvertTo-Json -InputObject $global:Calls -Depth 8 | Set-Content (Join-Path $PSScriptRoot 'calls.json') }
""")
                import os
                env = dict(os.environ, MOCK_SETUP_FAIL='1' if fail else '0')
                result = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(driver)],
                                        env=env, capture_output=True, text=True)
                calls = json.loads((project / 'calls.json').read_text(encoding='utf-8-sig'))
                flattened = [' '.join(call) for call in calls]
                self.assertEqual(result.returncode == 0, not fail, result.stderr)
                self.assertEqual(any('verify_environment.py' in call for call in flattened), not fail)
                self.assertEqual(any('audit_data_pipeline.py' in call for call in flattened), not fail)
                self.assertFalse(any('prepare' in call for call in flattened))
                self.assertEqual(before, {p.name: p.read_bytes() for p in splits.iterdir()})
