"""Reuse complete 7:2:1 manifests; prepare a new split only when none exists."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def prepare_or_reuse(root):
    root = Path(root).resolve()
    destination = root / 'splits'
    if destination.exists():
        summary_path = destination / 'summary.json'
        if not summary_path.is_file():
            raise ValueError('Incomplete splits at {}. Review them or use --skip-data.'.format(destination))
        summary = json.loads(summary_path.read_text(encoding='utf8'))
        if summary.get('val_ratio') != 0.2 or summary.get('test_ratio') != 0.1:
            raise ValueError('Existing splits are not 7:2:1. Review them and run prepare --force explicitly, '
                             'or use --skip-data.')
        for split in ('train', 'val', 'test'):
            manifest = destination / (split + '.json')
            if not manifest.is_file():
                raise ValueError('Missing split: {}'.format(manifest))
        print('Reusing existing 7:2:1 splits; no files overwritten.')
        return summary
    from simple_seg.data import prepare
    return prepare(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1] / 'dataset'))
    args = parser.parse_args(argv)
    try:
        print(json.dumps(prepare_or_reuse(args.root), indent=2, ensure_ascii=False))
        return 0
    except (ValueError, OSError) as error:
        parser.exit(1, '{}\n'.format(error))


if __name__ == '__main__':
    raise SystemExit(main())
