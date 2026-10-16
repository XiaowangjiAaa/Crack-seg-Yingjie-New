#!/usr/bin/env bash
# Same CONFIG / GPUS convention as MMSeg; use the current torchrun entry.
set -euo pipefail
if [ "$#" -lt 2 ]; then
    echo "Usage: bash tools/dist_train.sh CONFIG GPUS [training arguments...]" >&2
    exit 2
fi
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$1"
GPUS="$2"
shift 2
cd "$PROJECT_ROOT"
exec "${PYTHON:-python}" -m torch.distributed.run \
    --nnodes="${NNODES:-1}" \
    --node_rank="${NODE_RANK:-0}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${PORT:-29500}" \
    --nproc_per_node="$GPUS" \
    "$PROJECT_ROOT/tools/train.py" "$CONFIG" --launcher pytorch "$@"
