"""Train the eight lightweight models, one independent model per GPU."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simple_seg.gpu_queue import main

if __name__ == '__main__':
    raise SystemExit(main())
