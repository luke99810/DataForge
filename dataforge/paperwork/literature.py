"""Conservative DOI syntax and Crossref metadata verification (not full text)."""
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
from .project import load, now, save

DOI = re.compile(r'10\.[0-9]{4,9}/[-._;()/:A-Za-z0-9]+', re.ASCII)


def fetch(url):
    request = Request(url, headers={'User-Agent': 'idea-to-paper/0.1 (metadata verification)',
                                    'Accept': 'application/json'})
    with urlopen(request, timeout=20) as response:
        data = response.read(4 * 1024 * 1024 + 1)
    if len(data) > 4 * 1024 * 1024:
        raise ValueError('Metadata response too large')
    return json.loads(data)


def verify(doi, cache, refresh=False):
    key = hashlib.sha256(doi.encode('utf-8')).hexdigest()
    path = Path(cache) / (key + '.json')
    if path.exists() and not refresh:
        cached = load(path)
        if cached.get('doi') == doi:
            return cached
    record = {'doi': doi, 'checked_at': now(), 'status': 'invalid_doi',
              'evidence_read_level': 'none'}
    if not DOI.fullmatch(doi):
        record['error'] = 'Bare DOI required; input preserved, no normalization or repair'
        save(path, record)
        return record
    record['url'] = 'https://api.crossref.org/works/' + quote(doi, safe='')
    try:
        response = fetch(record['url'])
        record['response'] = response
        item = response['message']
        if item['DOI'].casefold() != doi.casefold():
            raise ValueError('Crossref returned another DOI')
        if not all(item.get(k) for k in ['title', 'author', 'container-title', 'issued']):
            raise ValueError('Incomplete required metadata')
        record.update(status='verified_metadata', evidence_read_level='metadata')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        record.update(status='failed', error=str(exc))
    save(path, record)
    return record
