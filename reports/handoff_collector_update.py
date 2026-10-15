"""Finish in-flight recordings, install a bound patch, and resume both slots."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import psutil

from lane_dataset.config import PROJECT, sha256_file, write_json
from lane_dataset.simulator import world_owner
from reports import automatic_corpus as corpus, parallel_corpus as parallel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--supervisor', type=int, required=True)
    parser.add_argument('--patch', type=Path, required=True)
    parser.add_argument('--patch-sha256', required=True)
    parser.add_argument('--tests', type=Path, required=True)
    parser.add_argument('--tests-sha256', required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if sha256_file(args.patch) != args.patch_sha256 or sha256_file(args.tests) != args.tests_sha256:
        raise ValueError('update artifacts changed')
    patch_command = ['git', 'apply', '--ignore-space-change', '--unidiff-zero', str(args.patch)]
    subprocess.run(patch_command[:2] + ['--check'] + patch_command[2:], cwd=PROJECT, check=True)
    sources = []
    for line in args.patch.read_text(encoding='utf-8').splitlines():
        if line.startswith('+++ b/'):
            relative = Path(line[6:])
            source = (PROJECT / relative).resolve()
            if source.parent != PROJECT / 'lane_dataset' or not source.is_file():
                raise ValueError('update must modify existing lane_dataset source files only')
            sources.append((relative, source, sha256_file(source)))
    if not sources:
        raise ValueError('empty source update')
    owner = psutil.Process(args.supervisor)
    command = owner.cmdline()
    if ('reports.automatic_corpus' not in command or '--job' in command or str(root) not in command):
        raise ValueError('supervisor identity mismatch')
    owner.terminate()
    owner.wait(timeout=15)
    output = root / 'reports/automatic'
    try:
        with world_owner('automatic-corpus-supervisor', root):
            state = corpus.read(output / 'state.json')
            workers = {port: parallel.Worker(root, job, port, process=p)
                       for p, job, port in parallel.discover_workers(root)}
            for value in state.get('workers', {}).values():
                port = value['port']
                if port not in workers and value['job'] not in state['completed']:
                    workers[port] = parallel.Worker(root, output/(value['job']+'.job.json'), port,
                        process=SimpleNamespace(pid=value['pid'], is_running=lambda: False))
            for worker in workers.values():
                worker.reader.seek(0, 2)
            state['status'] = 'finishing_current_captures_before_cpu_update'
            print('Both current GPU captures continue. Admit their results before installing CPU optimizations.', flush=True)
            while workers:
                for port, worker in list(workers.items()):
                    if not worker.poll():
                        continue
                    replacement = parallel.finish(root, state, worker, PROJECT/'reports/training-release-tests.xml')
                    worker.close()
                    if replacement:
                        workers[port] = replacement
                    else:
                        del workers[port]
                parallel.save(root, state, workers)
                time.sleep(2)
            # Recheck after the wait so a stale update cannot undo intervening work.
            if sha256_file(args.patch) != args.patch_sha256 or sha256_file(args.tests) != args.tests_sha256:
                raise ValueError('update artifacts changed during drain')
            if any(sha256_file(source) != digest for _, source, digest in sources):
                raise ValueError('live source changed during drain')
            subprocess.run(patch_command[:2] + ['--check'] + patch_command[2:], cwd=PROJECT, check=True)
            backup = output / ('before-cpu-update-' + str(time.time_ns()))
            backup.mkdir()
            for relative, source, _ in sources:
                target = backup / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            tests = PROJECT / 'reports/training-release-tests.xml'
            shutil.copy2(tests, backup/'previous-tests.xml')
            try:
                subprocess.run(patch_command, cwd=PROJECT, check=True)
                shutil.copy2(args.tests, tests)
                write_json(output/'collector-cpu-update.json', dict(installed=True, time=time.time(),
                    patch_sha256=args.patch_sha256, tests_sha256=args.tests_sha256, backup=str(backup)))
            except BaseException:
                for relative, source, _ in sources:
                    shutil.copy2(backup/relative, source)
                shutil.copy2(backup/'previous-tests.xml', tests)
                raise
    except Exception as error:
        traceback.print_exc()
        write_json(output/'collector-cpu-update.json', dict(installed=False, error=str(error), time=time.time()))
    print('Resuming collection on both GPU slots. CPU training remains separate.', flush=True)
    return subprocess.call([sys.executable, '-u', '-m', 'reports.automatic_corpus', '--root', str(root)], cwd=PROJECT)


if __name__ == '__main__':
    raise SystemExit(main())
