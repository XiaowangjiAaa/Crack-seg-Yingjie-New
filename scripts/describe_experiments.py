"""Write exact losses and optimizer parameters for all experiment templates."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simple_seg.adapter import build_config
from simple_seg.reporting import describe_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='reports/experiment_training.json')
    args = parser.parse_args()
    records = []
    for path in sorted((ROOT / 'configs/experiments').glob('*.json')):
        experiment = json.loads(path.read_text(encoding='utf8'))
        cfg = build_config(experiment, ROOT, validate_data=False)
        summary = describe_config(cfg)
        records.append(dict(experiment=path.stem, **summary))
        losses = ', '.join('{}: {} (weight={})'.format(row['path'], row['config']['type'],
            row['config'].get('loss_weight', 1)) for row in summary['losses'])
        print('{} | {} lr={} | {}'.format(path.stem, summary['optimizer']['type'],
                                         summary['optimizer']['lr'], losses))
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf8')
    print(output)


if __name__ == '__main__':
    main()
