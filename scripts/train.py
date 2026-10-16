"""Portable wrapper for tools/train.py and PyTorch's native distributed launcher."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simple_seg.launch import main

if __name__ == '__main__':
    sys.exit(main())
