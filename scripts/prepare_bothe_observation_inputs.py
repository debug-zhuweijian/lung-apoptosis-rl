"""Preflight or consume closed actual Bothe counts; optional readonly waiter.

This job writes only a new observation pack/evidence. No native count, source
normalization, model, PT load or forward is started.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from lung.data.bothe_observation_inputs import (BASE, LIMITS, OUTPUT, RECOVERY, SCHEMA,
                                               VERSION, closed_count_publication, preflight, prepare,
                                               read_json, require, sha, write_json)


def process_alive(launch, key, wrapper):
    pid = launch.get(key)
    require(type(pid) is int and 0 < pid <= 2**32-1, 'POSITIVE_WIN32_PID_REQUIRED')
    command = launch.get('commandline')
    canonical = f'"{Path(sys.executable).resolve().as_posix()}" -X utf8 -u "{wrapper.resolve().as_posix()}" --execute'
    require(isinstance(command, str) and command.replace('\\', '/') == canonical,
            'ABSOLUTE_REVIEWED_WRAPPER_COMMAND_REQUIRED')
    expected = datetime.fromisoformat(launch['creation_date'].replace('Z', '+00:00'))
    require(expected.tzinfo is not None, 'TIMEZONE_AWARE_PROCESS_CREATION_REQUIRED')
    query = ('[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding($false);'
             '$OutputEncoding=[Console]::OutputEncoding;'
             f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={pid}';"
             "if($p){[pscustomobject]@{pid=$p.ProcessId;name=$p.Name;commandline=$p.CommandLine;"
             "creation_date=$p.CreationDate.ToUniversalTime().ToString('o')}|ConvertTo-Json -Compress}")
    result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', query], capture_output=True,
                            text=False, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    require(isinstance(result.stdout, bytes), 'UTF8_BINARY_PROCESS_CAPTURE_REQUIRED')
    text = result.stdout.decode('utf-8-sig').strip()
    if not text:
        return False
    actual = json.loads(text)
    created = datetime.fromisoformat(actual['creation_date'].replace('Z', '+00:00'))
    require(created.tzinfo is not None, 'TIMEZONE_AWARE_ACTUAL_CREATION_REQUIRED')
    return (type(actual['pid']) is int and actual['pid'] == pid and actual['name'] == 'python.exe' and actual['commandline'] == command
            and created.astimezone(timezone.utc) == expected.astimezone(timezone.utc))


def state_write(path, value):
    temporary = path.with_suffix('.json.part')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf8')
    os.replace(temporary, path)


def execute(output, evidence, *, watch):
    require(evidence.resolve().is_relative_to((ROOT / 'experiments/20261011-data-alignment-round22/bothe_observation').resolve()),
            'ROUND22_OBSERVATION_EVIDENCE_REQUIRED')
    require(not evidence.exists(), 'NEW_OBSERVATION_EVIDENCE_REQUIRED')
    with FileLock(str(ROOT / BASE / '.observation-inputs.lock'), timeout=0):
        evidence.mkdir(parents=True)
        source = {'consumer': sha(ROOT / 'src/lung/data/bothe_observation_inputs.py'), 'builder': sha(__file__)}
        state = {'schema': SCHEMA, 'status': 'waiting_for_closed_actual_joint_export', 'actual_waiter_pid': os.getpid(),
                 'source_sha256': source, 'started_at_utc': datetime.now(timezone.utc).isoformat(), **LIMITS}
        state_write(evidence / 'state.json', state)
        try:
            finalizer = read_json(ROOT / RECOVERY / 'finalizer_launch-v2.json') if watch else None
            count = read_json(ROOT / RECOVERY / 'count_launch.json') if watch else None
            while True:
                require(source == {'consumer': sha(ROOT / 'src/lung/data/bothe_observation_inputs.py'), 'builder': sha(__file__)},
                        'RUNNING_OBSERVATION_CONSUMER_SOURCE_CHANGED')
                status = preflight(ROOT)
                state['preflight'] = status
                state_write(evidence / 'state.json', state)
                if status['source_data_ready']:
                    break
                if not watch:
                    raise RuntimeError('PENDING_ACTUAL_EXPORT_NO_OBSERVATION_PACK_CREATED')
                directory = ROOT / BASE / 'processed' / VERSION
                lifecycle = read_json(directory / 'driver.json')
                require(lifecycle['status'] != 'failed', 'ACTUAL_NATIVE_COUNT_FAILED_NO_OBSERVATION_PACK')
                finalizer_alive = process_alive(finalizer, 'finalizer_PID', ROOT / RECOVERY / 'finalize_joint_counts.py')
                count_alive = process_alive(count, 'count_PID', ROOT / RECOVERY / 'resume_counts.py')
                count_closed = None
                if not finalizer_alive or not count_alive:
                    # Completion can close between preflight and the CIM query.
                    if preflight(ROOT)['source_data_ready']:
                        break
                    lifecycle = read_json(directory / 'driver.json')
                    require(lifecycle['status'] != 'failed', 'ACTUAL_NATIVE_COUNT_FAILED_NO_OBSERVATION_PACK')
                    count_closed = closed_count_publication(ROOT)
                    require(finalizer_alive, 'BOUND_FINALIZER_DEAD_BEFORE_CLOSED_JOINT_EXPORT')
                    require(count_alive or count_closed is not None, 'BOUND_COUNT_DEAD_WITHOUT_COMPLETE_PUBLICATION')
                state.update(actual_finalizer_identity_matched=finalizer_alive, actual_count_identity_matched=count_alive,
                             actual_closed_count_publication=count_closed)
                state_write(evidence / 'state.json', state)
                time.sleep(30)
            manifest = prepare(ROOT, output)
            result = {'schema': SCHEMA, 'status': 'complete_actual_observation_pack_readback',
                      'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'output': str(output),
                      'manifest_sha256': sha(output / 'manifest.json'), 'shape': manifest['shape'],
                      'coverage': manifest['coverage'], 'source_sha256': source,
                      'raw_SAM_index_rehashed_here': False, **LIMITS}
            write_json(evidence / 'result.json', result)
            state.update(status=result['status'], output=str(output), manifest_sha256=result['manifest_sha256'])
            state_write(evidence / 'state.json', state)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except BaseException as error:
            state.update(status='failed_preserved', error_type=type(error).__name__, error=str(error),
                         actual_observation_pack_qualified=False)
            write_json(evidence / 'failure.json', state)
            state_write(evidence / 'state.json', state)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / OUTPUT)
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args(argv)
    if args.watch and not args.execute:
        parser.error('--watch requires --execute (data-only observation preparation)')
    if not args.execute:
        print(json.dumps(preflight(ROOT), ensure_ascii=False))
        return 0
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    evidence = args.evidence or ROOT / 'experiments/20261011-data-alignment-round22/bothe_observation' / ('execution-' + stamp)
    execute(args.output.resolve(), evidence.resolve(), watch=args.watch)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
