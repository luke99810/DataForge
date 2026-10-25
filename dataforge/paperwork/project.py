"""Portable, scope-limited project and provenance primitives."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

EXCLUDED = {'.git', '.venv', 'venv', '__pycache__', 'node_modules', '.cache'}
SENSITIVE = ('.env', 'secret', 'password', 'credential', 'token', 'private', '密钥', '密码', '隐私')


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inside(root, name):
    root = Path(root).resolve()
    candidate = root / name
    if candidate.is_symlink():
        raise ValueError('Symlinks not allowed')
    path = candidate.resolve()
    if not path.is_relative_to(root):
        raise ValueError('Path escapes project scope')
    if any(part.lower() in EXCLUDED or any(s in part.lower() for s in SENSITIVE)
           for part in path.relative_to(root).parts):
        raise ValueError('Sensitive or excluded path')
    return path


def init(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    for name in ['data', 'code', 'results', 'figures', 'runs', 'literature', 'draft', 'reviews']:
        (root / name).mkdir()
    templates = Path(__file__).resolve().parent / 'templates'
    if templates.exists():
        for path in templates.iterdir():
            if path.is_file():
                (root / path.name).write_bytes(path.read_bytes())
    save(root / 'sources.json', [])


def discover(root, scopes, limit=2000):
    records = []
    for scope in scopes:
        try:
            base = inside(root, scope)
        except ValueError:
            if not (Path(root) / scope).resolve().is_relative_to(Path(root).resolve()):
                raise
            continue
        paths = [base] if base.is_file() else _walk(base)
        for path in paths:
            if len(records) >= limit:
                raise ValueError('Inventory limit exceeded; narrow scope')
            if path.is_symlink():
                continue
            try:
                safe = inside(root, path.relative_to(root))
            except ValueError:
                continue
            records.append({'path': safe.relative_to(Path(root).resolve()).as_posix(),
                            'bytes': safe.stat().st_size, 'sha256': sha(safe)})
    return sorted({r['path']: r for r in records}.values(), key=lambda r: r['path'])


def _walk(base):
    for folder, dirs, files in os.walk(base, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d.lower() not in EXCLUDED
                         and not any(s in d.lower() for s in SENSITIVE)
                         and not (Path(folder) / d).is_symlink())
        for name in sorted(files):
            yield Path(folder) / name


def source(root, artifact, url, license_text, kind):
    parsed = urlsplit(url)
    if parsed.scheme not in ('https', 'http', 'urn') or not license_text.strip():
        raise ValueError('Explicit source URL/URN and license required')
    if parsed.username or parsed.password or parsed.query:
        raise ValueError('Credential/query URLs forbidden in public ledger')
    path = inside(root, artifact)
    row = {'path': artifact, 'url': url, 'license': license_text, 'kind': kind,
           'accessed_at': now(), 'sha256': sha(path)}
    ledger = load(Path(root) / 'sources.json')
    ledger.append(row)
    save(Path(root) / 'sources.json', ledger)
    return row
