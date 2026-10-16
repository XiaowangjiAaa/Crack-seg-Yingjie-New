"""MMSeg-style training entry; dataset/experiment conversion lives in simple_seg."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simple_seg.cli import main

if __name__ == '__main__':
    main(['train'] + sys.argv[1:])
