import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'project'

    def module(self, name):
        try:
            return importlib.import_module('dataforge.paperwork.' + name)
        except ImportError:
            self.fail('Required feature module not implemented: ' + name)

    def test_init_discover_source(self):
        core = self.module('project')
        core.init(self.root)
        with self.assertRaises(FileExistsError):
            core.init(self.root)
        (self.root / 'data/a.csv').write_text('x\n1\n')
        (self.root / '.env').write_text('synthetic exclusion fixture; no credentials')
        excluded_name = ''.join(('pass', 'words', '.txt'))
        (self.root / excluded_name).write_text('synthetic exclusion fixture; no credentials')
        (self.root / '.venv').mkdir()
        (self.root / '.venv/private.csv').write_text('synthetic exclusion fixture; no credentials')
        items = core.discover(self.root, ['data', '.env', excluded_name, '.venv'])
        self.assertEqual([x['path'] for x in items], ['data/a.csv'])
        with self.assertRaises(ValueError):
            core.discover(self.root, ['../'])
        row = core.source(self.root, 'data/a.csv', 'https://example.org/data', 'CC0', 'synthetic')
        self.assertEqual(len(row['sha256']), 64)
        self.assertTrue(row['accessed_at'])
        self.assertEqual(row['license'], 'CC0')

    def test_record_execution_failure_timeout_and_stale_output(self):
        runner = self.module('runner')
        self.module('project').init(self.root)
        (self.root / 'code/job.py').write_text("from pathlib import Path\nPath('results/a.json').write_text('{\"value\": 2}')\n")
        args = [sys.executable, 'code/job.py']
        record = runner.run(self.root, args, ['code/job.py'], ['results/a.json'], 7, 5)
        self.assertEqual(record['status'], 'pass')
        self.assertEqual(record['exit_code'], 0)
        self.assertEqual(record['seed'], 7)
        self.assertTrue(record['environment']['python'])
        self.assertEqual(len(record['outputs']['results/a.json']), 64)
        failed = runner.run(self.root, [sys.executable, '-c', 'raise SystemExit(3)'], [], ['results/a.json'], 7, 5)
        self.assertEqual(failed['status'], 'fail')
        self.assertEqual(failed['exit_code'], 3)
        with self.assertRaises(ValueError):
            runner.run(self.root, 'echo unsafe', [], [], 1, 1)
        timeout = runner.run(self.root, [sys.executable, '-c', 'import time; time.sleep(2)'], [], [], 1, 0.05)
        self.assertEqual(timeout['status'], 'fail')
        self.assertTrue(timeout['timed_out'])
        stale = runner.run(self.root, [sys.executable, '-c', 'pass'], [], ['results/a.json'], 1, 5)
        self.assertEqual(stale['status'], 'fail')

    def test_evidence_chain_and_tamper_rejection(self):
        checks = self.module('checks')
        core = self.module('project')
        core.init(self.root)
        (self.root / 'code/job.py').write_text("from pathlib import Path\nPath('results/a.json').write_text('{\"value\": 2}')\nPath('figures/a.svg').write_text('<svg/>')\n")
        rec = self.module('runner').run(self.root, [sys.executable, 'code/job.py'], ['code/job.py'], ['results/a.json', 'figures/a.svg'])
        claim = {'id': 'C1', 'run_id': rec['id'], 'artifact': 'results/a.json', 'field': 'value', 'value': 2.00001, 'abs_tol': 0.001}
        figure = {'path': 'figures/a.svg', 'run_id': rec['id'], 'sources': ['results/a.json']}
        self.assertEqual(checks.claims(self.root, [claim]), [])
        self.assertEqual(checks.figures(self.root, [figure]), [])
        self.assertTrue(checks.claims(self.root, [dict(claim, field='missing')]))
        self.assertTrue(checks.claims(self.root, [dict(claim, value=4)]))
        self.assertTrue(checks.claims(self.root, [dict(claim, value=float('nan'))]))
        self.assertTrue(checks.claims(self.root, []))
        manifest = checks.manifest(self.root, ['results/a.json', 'figures/a.svg'])
        self.assertEqual(checks.verify_manifest(self.root, manifest), [])
        (self.root / 'results/a.json').write_text('{"value": 99}')
        self.assertTrue(checks.claims(self.root, [claim]))
        self.assertTrue(checks.figures(self.root, [figure]))
        self.assertTrue(checks.verify_manifest(self.root, manifest))
        self.assertTrue(checks.citations('text [@missing]', []))
        self.assertEqual(checks.citations('text [@a]', [{'id': 'a'}]), [])
        self.assertTrue(checks.gates(self.root, {'stages': []}))
        self.assertTrue(checks.gates(self.root, {'stages': [{'name': 'idea', 'status': 'fail'}]}))

    def test_crossref_strict_identity_cache_and_failure(self):
        literature = self.module('literature')
        from unittest.mock import patch
        doi = '10.1038/nphys1170'
        body = {'status': 'ok', 'message': {'DOI': doi, 'title': ['Title'], 'author': [{'family': 'Author'}], 'container-title': ['Journal'], 'issued': {'date-parts': [[2009]]}}}
        with patch('dataforge.paperwork.literature.fetch', return_value=body) as fetch:
            good = literature.verify(doi, Path(self.tmp.name))
            self.assertEqual(good['status'], 'verified_metadata')
            self.assertEqual(good['evidence_read_level'], 'metadata')
            self.assertEqual(literature.verify(doi, Path(self.tmp.name))['status'], 'verified_metadata')
            self.assertEqual(fetch.call_count, 1)
        with patch('dataforge.paperwork.literature.fetch', side_effect=OSError('offline')) as fetch:
            bad = literature.verify('https://doi.org/' + doi, Path(self.tmp.name))
            self.assertEqual(bad['status'], 'invalid_doi')
            fetch.assert_not_called()
            failed = literature.verify('10.9999/unavailable', Path(self.tmp.name))
            self.assertEqual(failed['status'], 'failed')
            self.assertEqual(failed['doi'], '10.9999/unavailable')
        with patch('dataforge.paperwork.literature.fetch', return_value=body):
            mismatch = literature.verify('10.9999/mismatch', Path(self.tmp.name))
            self.assertEqual(mismatch['status'], 'failed')
        self.assertGreaterEqual(len(list(Path(self.tmp.name).glob('*.json'))), 4)

    def test_publication_ready_requires_human_approval_not_self_check(self):
        checks = self.module('checks')
        state = {
            'publication_ready': True,
            'stages': [{'name': name, 'status': 'pass', 'reviewer': 'agent self-check',
                        'reviewer_role': 'agent', 'evidence': {'x': 'y'}}
                       for name in checks.STAGES]
        }
        errors = checks.gates(self.root, state)
        self.assertTrue(any('human approval' in error for error in errors))

    def test_provenance_requires_explicit_kind_and_coverage(self):
        checks = self.module('checks')
        core = self.module('project')
        core.init(self.root)
        (self.root / 'data/x.csv').write_text('x\\n1\\n')
        bad = [{'path': 'data/x.csv', 'url': 'urn:synthetic:x', 'license': 'CC0',
                'kind': 'observed', 'sha256': core.sha(self.root / 'data/x.csv')}]
        self.assertTrue(any('provenance' in error for error in checks.provenance(self.root, bad)))

    def test_demo_cli_closed_loop_and_oracle(self):
        import subprocess
        command = [sys.executable, '-m', 'dataforge.paperwork', 'demo', str(self.root)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        core = self.module('project')
        values = core.load(self.root / 'results/results.json')
        data = core.load(self.root / 'data/demand.json')
        oracle = {q: sum(p * (5 * min(q, d) - 2 * q) for d, p in zip(data['demand'], data['probability'])) for q in range(5)}
        self.assertEqual(values['optimal_q'], max(oracle, key=oracle.get))
        self.assertAlmostEqual(values['expected_profit'], oracle[values['optimal_q']])
        self.assertTrue(values['synthetic'])
        self.assertEqual(self.module('checks').validate(self.root)['status'], 'pass')
        ledger = core.load(self.root / 'sources.json')
        core.save(self.root / 'sources.json', [])
        doc = core.load(self.root / 'manifest.json')
        doc['artifacts']['sources.json'] = core.sha(self.root / 'sources.json')
        core.save(self.root / 'manifest.json', doc)
        self.assertEqual(self.module('checks').validate(self.root)['status'], 'fail', 'Empty data ledger must fail closed')
        core.save(self.root / 'sources.json', ledger)
        doc['artifacts']['sources.json'] = core.sha(self.root / 'sources.json')
        core.save(self.root / 'manifest.json', doc)
        (self.root / 'figures/profit.svg').write_text('tampered')
        validation = subprocess.run([sys.executable, '-m', 'dataforge.paperwork', 'validate', str(self.root)], capture_output=True)
        self.assertNotEqual(validation.returncode, 0)
        repeat = subprocess.run(command, capture_output=True)
        self.assertNotEqual(repeat.returncode, 0)


if __name__ == '__main__':
    unittest.main()
