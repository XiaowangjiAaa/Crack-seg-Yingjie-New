"""Compare each experiment's data protocol with its pinned MMSeg configuration."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mmengine.config import Config
from simple_seg.adapter import build_config
from simple_seg.pipelines import adapt_binary_pipeline, upstream_pipeline


def main():
    rows = []
    for path in sorted((ROOT / 'configs/experiments').glob('*.json')):
        experiment = json.loads(path.read_text(encoding='utf8'))
        source = Config.fromfile(str(ROOT / experiment['model_config']), import_custom_modules=False)
        current = build_config(experiment, ROOT, validate_data=False)
        checks = {}
        row = dict(experiment=path.stem, source_config=experiment['model_config'])
        for split in ('train', 'val', 'test'):
            official = upstream_pipeline(source, split)
            adapted = current[split + '_dataloader']['dataset']['pipeline']
            row['upstream_' + split + '_pipeline'] = official
            row['current_' + split + '_pipeline'] = adapted
            checks[split + '_pipeline'] = adapted == adapt_binary_pipeline(official)
        for name, key in (('inference', 'test_cfg'), ('preprocessor', 'data_preprocessor')):
            row['upstream_' + name] = source.model.get(key, {})
            row['current_' + name] = current.model.get(key, {})
            checks[name] = row['upstream_' + name] == row['current_' + name]
        checks['prediction_pipeline'] = current.test_pipeline == adapt_binary_pipeline(
            upstream_pipeline(source, 'test'), prediction=True)
        row['checks'] = checks
        rows.append(row)
    passed = all(all(row['checks'].values()) for row in rows)
    report = dict(mmseg_version='1.2.2', scope='Configuration comparison only; models were not built or trained.',
                  all_match_upstream_except_binary_loading=passed, experiments=rows)
    output = ROOT / 'reports/data_pipeline_audit.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf8')
    print('{} experiments; all data protocol checks passed: {}'.format(len(rows), passed))
    print(output)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
