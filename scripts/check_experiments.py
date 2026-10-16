"""Configuration/build/one-step training checks for a batch manifest."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simple_seg.checks import main

if __name__ == '__main__':
    raise SystemExit(main())
