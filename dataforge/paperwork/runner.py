"""Execute explicitly approved argv; not a sandbox or shell filter."""
import os
import platform
import subprocess
import sys
import uuid
from pathlib import Path
from .project import inside, now, save, sha


def run(root, argv, inputs, outputs, seed=0, timeout=60):
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
        raise ValueError('Command must be a nonempty string argument array')
    if timeout <= 0 or set(inputs) & set(outputs):
        raise ValueError('Positive timeout and disjoint inputs/outputs required')
    root = Path(root).resolve()
    input_hash = {p: sha(inside(root, p)) for p in inputs}
    output_paths = {p: inside(root, p) for p in outputs}
    run_id = uuid.uuid4().hex
    directory = root / 'runs' / run_id
    directory.mkdir(parents=True)
    # Archive declared old outputs so successful no-op commands cannot reuse them.
    for index, path in enumerate(output_paths.values()):
        if path.exists():
            archive = directory / 'previous' / str(index)
            archive.parent.mkdir(parents=True, exist_ok=True)
            path.replace(archive)
        path.parent.mkdir(parents=True, exist_ok=True)
    record = {'id': run_id, 'argv': argv, 'started_at': now(), 'seed': seed,
              'inputs': input_hash, 'outputs': {}, 'exit_code': None,
              'timed_out': False, 'status': 'fail',
              'environment': {'python': sys.version, 'platform': platform.platform(),
                              'executable': sys.executable,
                              'PYTHONHASHSEED': str(seed), 'RESEARCH_SEED': str(seed)}}
    env = os.environ.copy()
    env.update(PYTHONHASHSEED=str(seed), RESEARCH_SEED=str(seed))
    _execute(root, directory, argv, timeout, env, record)
    record['outputs'] = {p: sha(path) for p, path in output_paths.items() if path.is_file()}
    unchanged = all(inside(root, p).is_file() and sha(inside(root, p)) == h
                    for p, h in input_hash.items())
    record['inputs_unchanged'] = unchanged
    if record['exit_code'] == 0 and len(record['outputs']) == len(outputs) and unchanged:
        record['status'] = 'pass'
    record['finished_at'] = now()
    record['stdout_sha256'] = sha(directory / 'stdout.txt')
    record['stderr_sha256'] = sha(directory / 'stderr.txt')
    save(directory / 'record.json', record)
    return record


def _execute(root, directory, argv, timeout, env, record):
    with (directory / 'stdout.txt').open('wb') as out, (directory / 'stderr.txt').open('wb') as err:
        try:
            result = subprocess.run(argv, cwd=root, env=env, shell=False,
                                    stdout=out, stderr=err, timeout=timeout, check=False)
            record['exit_code'] = result.returncode
        except subprocess.TimeoutExpired:
            record['timed_out'] = True
            record['error'] = 'Command timeout; check independently spawned child processes'
        except OSError as exc:
            record['error'] = str(exc)
