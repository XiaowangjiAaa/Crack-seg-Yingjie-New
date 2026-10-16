param([string]$EnvPath = '', [switch]$Cpu, [switch]$SkipData, [switch]$PrintOnly)
$ErrorActionPreference = 'Stop'
$Project = Split-Path -Parent $PSScriptRoot
if (-not $EnvPath) { $EnvPath = Join-Path $Project '.venv-mmseg' }
$EnvPath = [System.IO.Path]::GetFullPath($EnvPath)
$Python = Join-Path $EnvPath 'python.exe'
$CondaMetadata = Join-Path $EnvPath 'conda-meta'
$TranscriptStarted = $false

function Invoke-Step([string]$Label, [string]$Program, [string[]]$Arguments) {
    Write-Host "`n== $Label =="
    # Activate the target environment for each step, including Windows DLL paths.
    if ($Program -eq $Python) {
        $Arguments = @('run', '--no-capture-output', '--prefix', $EnvPath, $Program) + $Arguments
        $Program = 'conda'
    }
    if ($PrintOnly) { Write-Host ($Program + ' ' + ($Arguments -join ' ')); return }
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Label failed (exit $LASTEXITCODE). Fix the error and rerun this script." }
}

try {
    if (-not $PrintOnly) {
        if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
            throw 'Conda was not found. Install Miniconda/Anaconda and run from its PowerShell Prompt.'
        }
        if ((Test-Path -LiteralPath $EnvPath) -and -not (Test-Path -LiteralPath $CondaMetadata)) {
            throw "Target exists but is not a Conda environment: $EnvPath. Choose another -EnvPath."
        }
        $LogFolder = Join-Path $Project 'work_dirs/setup'
        New-Item -ItemType Directory -Path $LogFolder -Force | Out-Null
        $LogPath = Join-Path $LogFolder ((Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
        Start-Transcript -Path $LogPath | Out-Null
        $TranscriptStarted = $true
        Write-Host "Installation log: $LogPath"
    }
    if (-not (Test-Path -LiteralPath $CondaMetadata)) {
        Invoke-Step 'Create Python 3.10 environment' 'conda' @('create', '--prefix', $EnvPath, 'python=3.10', 'pip', '-y')
    }
    Invoke-Step 'Check target Python version' $Python @('-c', 'import sys; assert sys.version_info[:2] == (3, 10)')
    Invoke-Step 'Install package tools' $Python @('-m', 'pip', 'install', 'pip<26', 'setuptools>=65,<81', 'wheel')
    if ($Cpu) {
        Invoke-Step 'Install CPU PyTorch' $Python @('-m', 'pip', 'install', 'numpy==1.26.4', 'torch==2.1.0+cpu', 'torchvision==0.16.0+cpu', '--extra-index-url', 'https://download.pytorch.org/whl/cpu')
        Invoke-Step 'Install CPU MMCV wheel' $Python @('-m', 'pip', 'install', 'mmcv==2.1.0', '--only-binary', 'mmcv', '--find-links', 'https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html')
        $Requirements = Join-Path $Project 'requirements-common.txt'
    } else { $Requirements = Join-Path $Project 'requirements.txt' }
    Invoke-Step 'Install all 15 experiment dependencies and TensorBoard' $Python @('-m', 'pip', 'install', '-r', $Requirements)
    Invoke-Step 'Register project' $Python @('-m', 'pip', 'install', '-e', $Project)
    Invoke-Step 'Verify dependency consistency' $Python @('-m', 'pip', 'check')
    Invoke-Step 'Verify OpenMMLab, compiled ops and model registrations' $Python @((Join-Path $PSScriptRoot 'verify_environment.py'))
    Invoke-Step 'Fetch pinned official configs' $Python @('-m', 'simple_seg', 'fetch-configs', '--root', $Project)
    if (-not $SkipData) {
        $SummaryPath = Join-Path $Project 'dataset/splits/summary.json'
        $SplitFolder = Join-Path $Project 'dataset/splits'
        if (Test-Path -LiteralPath $SplitFolder) {
            if (-not (Test-Path -LiteralPath $SummaryPath)) { throw "Incomplete splits at $SplitFolder. Review them or rerun with -SkipData." }
            $Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
            if ($Summary.val_ratio -ne 0.2 -or $Summary.test_ratio -ne 0.1) {
                throw 'Existing splits are not 7:2:1. Review them and run prepare --force explicitly, or use -SkipData.'
            }
            foreach ($Split in @('train', 'val', 'test')) {
                if (-not (Test-Path -LiteralPath (Join-Path $SplitFolder ($Split + '.json')))) { throw "Missing $Split split. Review existing splits." }
            }
            Write-Host 'Reusing existing 7:2:1 splits; no files overwritten.'
        } else {
            Invoke-Step 'Validate dataset and prepare 7:2:1 splits' $Python @('-m', 'simple_seg', 'prepare', '--root', (Join-Path $Project 'dataset'))
        }
    }
    Invoke-Step 'Compare all 15 data protocols with official configs' $Python @((Join-Path $PSScriptRoot 'audit_data_pipeline.py'))
    if ($PrintOnly) { Write-Host 'Preview complete; no commands executed and no files written.'; return }
    Write-Host "`nReady. Activate: conda activate '$EnvPath'"
    Write-Host "Train: & '$Python' '$(Join-Path $PSScriptRoot 'train.py')' '$(Join-Path $Project 'configs/experiments/segformer_b0.json')' --gpus 1"
    Write-Host "WebUI: & '$Python' '$(Join-Path $PSScriptRoot 'webui.py')' --logdir '$(Join-Path $Project 'work_dirs')'"
} finally { if ($TranscriptStarted) { Stop-Transcript | Out-Null } }
