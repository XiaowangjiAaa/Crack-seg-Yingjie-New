"""Fetch only upstream configuration files and attribution, never execute downloads."""
import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

VERSION = 'v1.2.2'
URL = 'https://codeload.github.com/open-mmlab/mmsegmentation/zip/refs/tags/' + VERSION


def fetch_configs(root, archive=None):
    destination = Path(root) / 'third_party' / 'mmsegmentation'
    if (destination / 'UPSTREAM.json').exists():
        print('Pinned configs already available: {}'.format(destination))
        return
    if archive:
        payload = Path(archive).read_bytes()
    else:
        print('Downloading {} configuration archive...'.format(VERSION), flush=True)
        with urllib.request.urlopen(URL, timeout=120) as response:
            payload = response.read()
    count = 0
    with zipfile.ZipFile(io.BytesIO(payload)) as source:
        prefix = 'mmsegmentation-1.2.2/'
        for member in source.infolist():
            if member.is_dir() or not member.filename.startswith(prefix):
                continue
            relative = PurePosixPath(member.filename[len(prefix):])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe archive path')
            if relative.parts[0] != 'configs' and str(relative) not in ('LICENSE', 'README.md'):
                continue
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as src, target.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    if not (destination / 'configs' / '_base_').is_dir():
        raise ValueError('Expected v1.2.2 configuration tree not found in archive')
    (destination / 'UPSTREAM.json').write_text(json.dumps(dict(
        repository='https://github.com/open-mmlab/mmsegmentation', revision=VERSION,
        archive=URL, files=count), indent=2), encoding='utf8')
    print('Saved {} upstream files to {}'.format(count, destination))


def list_configs(root, query=''):
    folder = Path(root) / 'third_party' / 'mmsegmentation' / 'configs'
    if not folder.exists():
        raise ValueError('Run fetch-configs first')
    return [p.relative_to(Path(root)).as_posix() for p in sorted(folder.rglob('*.py'))
            if '_base_' not in p.parts and query.lower() in p.as_posix().lower()]
