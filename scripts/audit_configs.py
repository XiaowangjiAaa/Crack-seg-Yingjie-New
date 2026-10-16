"""Resolve and adapt every upstream config; this is NOT a model execution test."""
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simple_seg.adapter import build_config
from simple_seg.upstream import list_configs


def main():
    results = []
    with tempfile.TemporaryDirectory(dir=ROOT) as folder:
        for split in ('train', 'val', 'test'):
            (Path(folder) / (split + '.json')).write_text('[]')
        configs = list_configs(ROOT)
        for i, source in enumerate(configs):
            try:
                cfg = build_config(dict(model_config=source, split_dir=folder), ROOT)
                row = dict(config=source, status='resolved', model_type=cfg.model.type,
                           runtime_imports=list(cfg.custom_imports.imports))
            except Exception as error:
                reason = str(error).split(', the current sys.path')[0]
                row = dict(config=source, status='needs_attention', reason='{}: {}'.format(type(error).__name__, reason))
            results.append(row)
            if (i + 1) % 100 == 0:
                print('Audited {}/{}'.format(i + 1, len(configs)), flush=True)
    report = dict(scope='Config resolution and binary adaptation only; models were not built or trained; '
                        'optional runtime imports were not executed.',
                  counts=dict(Counter(r['status'] for r in results)), configs=results)
    output = ROOT / 'reports/config_compatibility.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
    print(report['counts'])
    print(output)


if __name__ == '__main__':
    main()
