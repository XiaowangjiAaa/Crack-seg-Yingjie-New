"""Run a JSON list of model experiments sequentially."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simple_seg.batch import main

if __name__ == '__main__':
    sys.exit(main())
