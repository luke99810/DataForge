"""Fail-closed mechanical evidence checks; never semantic peer review."""
import math
import re
from pathlib import Path
from .project import inside, load, sha

STAGES = ('idea', 'journal', 'assets', 'literature', 'model', 'compute', 'claims', 'writing', 'reflection')
ERRORS = (KeyError, ValueError, TypeError, OSError, IndexError)


def run_record(root, run_id):
    if not re.fullmatch(r'[0-9a-f]{32}', run_id):
        raise ValueError('Invalid run id')
    record = load(inside(root, f'runs/{run_id}/record.json'))
    if record['status'] != 'pass' or record['exit_code'] != 0 or record['timed_out']:
        raise ValueError('Run failed')
    for name, digest in {**record['inputs'], **record['outputs']}.items():
        if sha(inside(root, name)) != digest:
            raise ValueError('Stale run artifact: ' + name)
    for stream in ('stdout', 'stderr'):
        if sha(inside(root, f'runs/{run_id}/{stream}.txt')) != record[stream + '_sha256']:
            raise ValueError('Changed execution log')
    return record


def claims(root, rows):
    errors = [] if rows else ['No numeric claims supplied']
    for row in rows:
        try:
            record = run_record(root, row['run_id'])
            if row['artifact'] not in record['outputs']:
                raise ValueError('Claim is not a declared run output')
            actual = load(inside(root, row['artifact']))
            for part in row['field'].split('.'):
                actual = actual[int(part)] if isinstance(actual, list) else actual[part]
            values = (actual, row['value'], row['abs_tol'])
            if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in values):
                raise ValueError('Finite numeric fields required')
            if row['abs_tol'] < 0 or abs(actual - row['value']) > row['abs_tol']:
                raise ValueError('Claim outside absolute tolerance')
        except ERRORS as exc:
            errors.append(f"{row.get('id', '?')}: {exc}")
    return errors


def figures(root, rows):
    errors = []
    for row in rows:
        try:
            record = run_record(root, row['run_id'])
            if row['path'] not in record['outputs'] or not row['sources']:
                raise ValueError('Figure requires output and source bindings')
            for name in row['sources']:
                if name not in record['inputs'] and name not in record['outputs']:
                    raise ValueError('Figure source not in producing run')
        except ERRORS as exc:
            errors.append(f"{row.get('path', '?')}: {exc}")
    return errors


def manifest(root, names):
    if not names:
        raise ValueError('Empty manifest')
    return {'artifacts': {name: sha(inside(root, name)) for name in names}}


def verify_manifest(root, document):
    errors = []
    try:
        if not document['artifacts']:
            raise ValueError('Empty manifest')
        for name, digest in document['artifacts'].items():
            if sha(inside(root, name)) != digest:
                errors.append('Manifest mismatch: ' + name)
    except ERRORS as exc:
        errors.append(str(exc))
    return errors


def citations(text, bibliography):
    ids = [row['id'] for row in bibliography]
    errors = ['Duplicate bibliography ids'] if len(ids) != len(set(ids)) else []
    for key in re.findall(r'\[@([A-Za-z0-9_.:-]+)\]', text):
        if key not in ids:
            errors.append('Unresolved citation: ' + key)
    return errors


def gates(root, state):
    errors = []
    stages = {row.get('name'): row for row in state.get('stages', [])}
    for name in STAGES:
        row = stages.get(name, {})
        if row.get('status') != 'pass' or not row.get('reviewer') or not row.get('evidence'):
            errors.append('Incomplete stage: ' + name)
            continue
        for artifact, digest in row['evidence'].items():
            try:
                if sha(inside(root, artifact)) != digest:
                    errors.append('Stage evidence changed: ' + name)
            except ERRORS as exc:
                errors.append(str(exc))
    if state.get('publication_ready'):
        approval = state.get('human_approval', {})
        if (approval.get('status') != 'approved' or
                approval.get('reviewer_role') != 'human' or
                not approval.get('reviewer') or not approval.get('evidence')):
            errors.append('publication_ready requires explicit human approval')
    return errors


def provenance(root, rows):
    errors = []
    for row in rows:
        try:
            kind = row['kind']
            if kind not in ('observed', 'synthetic', 'synthetic teaching fixture', 'authored', 'reused_public', 'restricted'):
                raise ValueError('unknown provenance kind')
            if not row['url'] or not row['license'] or not row['sha256']:
                raise ValueError('incomplete provenance')
            if kind == 'observed' and not row.get('coverage'):
                raise ValueError('observed provenance requires coverage')
            if kind == 'synthetic' and not str(row['url']).startswith('urn:synthetic:'):
                raise ValueError('synthetic source must use urn:synthetic')
            if sha(inside(root, row['path'])) != row['sha256']:
                raise ValueError('stale provenance artifact')
        except ERRORS as exc:
            errors.append('Invalid provenance: ' + str(exc))
    return errors


def validate(root):
    errors = []
    try:
        claim_rows = load(inside(root, 'claims.json'))
        figure_rows = load(inside(root, 'figures.json'))
        bibliography = load(inside(root, 'bibliography.json'))
        text = inside(root, 'draft/manuscript.md').read_text(encoding='utf-8')
        errors.extend(claims(root, claim_rows))
        errors.extend(figures(root, figure_rows))
        errors.extend(citations(text, bibliography))
        errors.extend(gates(root, load(inside(root, 'state.json'))))
        doc = load(inside(root, 'manifest.json'))
        errors.extend(verify_manifest(root, doc))
        required = {'claims.json', 'figures.json', 'bibliography.json', 'state.json',
                    'sources.json', 'draft/manuscript.md'}
        required.update(r['artifact'] for r in claim_rows)
        required.update(r['path'] for r in figure_rows)
        for row in claim_rows + figure_rows:
            required.add(f"runs/{row['run_id']}/record.json")
        if not required.issubset(doc['artifacts']):
            errors.append('Manifest missing required evidence artifacts')
        sources = load(inside(root, 'sources.json'))
        if not sources:
            errors.append('Empty data provenance ledger')
        errors.extend(provenance(root, sources))
    except ERRORS as exc:
        errors.append('Missing/invalid validation input: ' + str(exc))
    return {'status': 'pass' if not errors else 'fail', 'errors': errors,
            'scope': 'mechanical integrity only; not semantic, ethical or publication approval'}
