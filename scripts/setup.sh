#!/usr/bin/env bash
# Linux/WSL2 one-command installer. No sudo, source builds, or existing split overwrite.
set -Eeuo pipefail

PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
ENV_PATH="$PROJECT/.venv-mmseg"
CPU=0
SKIP_DATA=0
PRINT_ONLY=0
STEP='Initialize installer'

usage() {
    cat <<'EOF'
Usage: bash scripts/setup.sh [options]
  --env-path PATH  Target Conda prefix (default: PROJECT/.venv-mmseg)
  --cpu            Select CPU PyTorch and MMCV wheels
  --skip-data      Install environment without preparing dataset splits
  --print-only     Preview commands without creating files or running Conda
  -h, --help       Show this help
EOF
}

while (($#)); do
    case "$1" in
        --env-path)
            if (($# < 2)) || [[ -z "$2" || "$2" == --* ]]; then
                echo '--env-path requires a directory' >&2; exit 2
            fi
            ENV_PATH=$2; shift 2 ;;
        --cpu) CPU=1; shift ;;
        --skip-data) SKIP_DATA=1; shift ;;
        --print-only) PRINT_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# Resolve a possibly relative prefix without requiring it to exist first.
ENV_PATH=$(realpath -m -- "$ENV_PATH")
PYTHON="$ENV_PATH/bin/python"
CONDA=conda
if command -v conda >/dev/null 2>&1; then
    CONDA=conda
elif [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
    CONDA=$CONDA_EXE
elif (( ! PRINT_ONLY )); then
    echo 'Conda was not found. Install Linux Miniconda/Anaconda and make conda available in this shell.' >&2
    exit 1
fi

if (( ! PRINT_ONLY )); then
    if [[ -e "$ENV_PATH" && ! -d "$ENV_PATH/conda-meta" ]]; then
        echo "Target exists but is not a Conda environment: $ENV_PATH. Choose another --env-path." >&2
        exit 1
    fi
    if [[ -d "$ENV_PATH/conda-meta" && ! -x "$PYTHON" ]]; then
        echo "Target has no Linux Python executable: $PYTHON. Do not reuse a Windows environment in Linux/WSL2." >&2
        exit 1
    fi
    LOG_DIR="$PROJECT/work_dirs/setup"
    mkdir -p -- "$LOG_DIR"
    LOG_PATH="$LOG_DIR/$(date +%Y%m%d-%H%M%S)-$$.log"
    exec > >(tee -a "$LOG_PATH") 2>&1
    echo "Installation log: $LOG_PATH"
fi

trap 'status=$?; printf "\nInstallation failed at: %s (exit %s). Fix the error and rerun.\n" "$STEP" "$status" >&2; exit "$status"' ERR

run_step() {
    STEP=$1; shift
    printf '\n== %s ==\n' "$STEP"
    if (( PRINT_ONLY )); then
        printf '%q ' "$@"; printf '\n'
        return 0
    fi
    "$@"
}

run_python() {
    local label=$1; shift
    run_step "$label" "$CONDA" run --no-capture-output --prefix "$ENV_PATH" "$PYTHON" "$@"
}

if [[ ! -d "$ENV_PATH/conda-meta" ]]; then
    run_step 'Create Python 3.10 environment' "$CONDA" create --prefix "$ENV_PATH" python=3.10 pip -y
fi
run_python 'Check target Python version' -c 'import sys; assert sys.version_info[:2] == (3, 10)'
run_python 'Install package tools' -m pip install 'pip<26' 'setuptools>=65,<81' wheel

if (( CPU )); then
    run_python 'Install CPU PyTorch' -m pip install numpy==1.26.4 torch==2.1.0+cpu torchvision==0.16.0+cpu \
        --extra-index-url https://download.pytorch.org/whl/cpu
    run_python 'Install CPU MMCV wheel' -m pip install mmcv==2.1.0 --only-binary mmcv \
        --find-links https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html
    REQUIREMENTS="$PROJECT/requirements-common.txt"
else
    REQUIREMENTS="$PROJECT/requirements.txt"
fi

run_python 'Install all experiment dependencies and TensorBoard' -m pip install -r "$REQUIREMENTS"
run_python 'Register project' -m pip install -e "$PROJECT"
run_python 'Verify dependency consistency' -m pip check
run_python 'Verify OpenMMLab, compiled ops and model registrations' "$PROJECT/scripts/verify_environment.py"
run_python 'Fetch pinned official configs' -m simple_seg fetch-configs --root "$PROJECT"
if (( ! SKIP_DATA )); then
    run_python 'Validate or reuse 7:2:1 dataset splits' "$PROJECT/scripts/prepare_setup_data.py" --root "$PROJECT/dataset"
fi
run_python 'Compare data protocols with official configs' "$PROJECT/scripts/audit_data_pipeline.py"

if (( PRINT_ONLY )); then
    echo 'Preview complete; no commands executed and no files written.'
    exit 0
fi
printf '\nReady. Activate: conda activate %q\n' "$ENV_PATH"
printf 'Eight-model GPU queue: %q %q --gpu-ids 0,1\n' "$PYTHON" "$PROJECT/scripts/train_gpu_queue.py"
printf 'WebUI: %q %q --logdir %q\n' "$PYTHON" "$PROJECT/scripts/webui.py" "$PROJECT/work_dirs"
