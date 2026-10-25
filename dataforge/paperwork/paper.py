"""Standard-library CLI for repeatable research mechanics."""
import argparse
import json
from pathlib import Path
import sys
from . import checks, demo, literature, project, runner


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest='command', required=True)
    for name in ['init', 'demo', 'validate']:
        commands.add_parser(name).add_argument('root', type=Path)
    p = commands.add_parser('discover')
    p.add_argument('root', type=Path)
    p.add_argument('scopes', nargs='+')
    p = commands.add_parser('source')
    p.add_argument('root', type=Path)
    for name in ['artifact', 'url', 'license', 'kind']:
        p.add_argument(name)
    p = commands.add_parser('run')
    p.add_argument('root', type=Path)
    p.add_argument('spec', type=Path, help='JSON: argv, inputs, outputs, seed, timeout')
    p = commands.add_parser('doi')
    p.add_argument('doi')
    p.add_argument('cache', type=Path)
    p.add_argument('--refresh', action='store_true')
    p = commands.add_parser('manifest')
    p.add_argument('root', type=Path)
    p.add_argument('artifacts', nargs='+')
    return cli


def dispatch(args):
    if args.command == 'init':
        project.init(args.root)
        return {'status': 'initialized'}
    if args.command == 'demo':
        return demo.build(args.root)
    if args.command == 'validate':
        return checks.validate(args.root)
    if args.command == 'discover':
        return project.discover(args.root.resolve(), args.scopes)
    if args.command == 'source':
        return project.source(args.root, args.artifact, args.url, args.license, args.kind)
    if args.command == 'run':
        spec = project.load(args.spec)
        return runner.run(args.root, spec['argv'], spec['inputs'], spec['outputs'],
                          spec.get('seed', 0), spec.get('timeout', 60))
    if args.command == 'doi':
        return literature.verify(args.doi, args.cache, args.refresh)
    document = checks.manifest(args.root, args.artifacts)
    project.save(args.root / 'manifest.json', document)
    return document


def main():
    try:
        result = dispatch(parser().parse_args())
        print(json.dumps(result, ensure_ascii=True, indent=2))
        status = result.get('status') if isinstance(result, dict) else None
        return 1 if status in ('fail', 'failed', 'invalid_doi') else 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({'status': 'fail', 'error': str(exc)}, ensure_ascii=True), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
