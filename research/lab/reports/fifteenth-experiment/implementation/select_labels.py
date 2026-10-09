"""One offline, resource-bounded fragment selector; root runs each exactly once."""
from pathlib import Path
import argparse
import json
import os
import resource
import signal
import socket
import time
from label_common import *


def blocked(*args, **kwargs):
    raise RuntimeError('Network is disabled in label selection')


def worker_limits():
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '4'
    resource.setrlimit(resource.RLIMIT_AS, (LIMITS['worker_address_space_bytes'],)*2)
    resource.setrlimit(resource.RLIMIT_CPU, (LIMITS['worker_cpu_seconds'],)*2)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Selection wall cap')))
    signal.alarm(LIMITS['worker_wall_seconds'])
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked


def candidates_for(labels, mask, known):
    import numpy as np
    height, width = labels.shape
    border = np.zeros((384, 384), dtype=bool)
    border[32:352, 32:352] = True
    candidates = []
    for y in range(0, height-1152+1, 1152):
        for x in range(0, width-1152+1, 1152):
            sl = (slice(y+1, y+1152, 3), slice(x+1, x+1152, 3))
            truth, primary = labels[sl], known[sl] & border
            inside = int(np.count_nonzero(mask[y:y+1152, x:x+1152]))
            positive = int(np.count_nonzero(primary & (truth == 1)))
            negative = int(np.count_nonzero(primary & (truth == 0)))
            candidates.append(dict(y=y, x=x, native_inside_count=inside,
                native_area=1152*1152, surface_fraction=inside/(1152*1152),
                primary_positive=positive, primary_negative=negative,
                primary_count=positive+negative, coarse_surface_count=int(np.count_nonzero(mask[sl])),
                coarse_known_count=int(np.count_nonzero(known[sl])),
                eligible=eligible(inside, 1152*1152, positive, negative)))
    return candidates, border


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--acquisition', required=True)
    parser.add_argument('--fragment', required=True, choices=['frag1', 'frag2', 'frag3'])
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    worker_limits()
    started = time.monotonic()
    plan = load_plan(args.plan)
    acquisition_path = Path(args.acquisition).resolve(strict=True)
    acquisition = json.loads(acquisition_path.read_text())
    require(acquisition['status'] == 'PASS' and acquisition['plan'] == record(args.plan), 'Acquisition does not match plan')
    require([{k: r[k] for k in ('fragment', 'kind', 'url', 'shape_yx')} for r in acquisition['objects']]
            == plan['objects'], 'All six acquisition identities required before decoding')
    for obj in acquisition['objects']:
        verify(obj['file'])
    ledger = json.loads(verify(acquisition['request_ledger']).read_text())
    require(len(ledger['requests']) == 6 and all(r['success'] for r in ledger['requests']), 'Incomplete HTTP receipts')
    require(acquisition['response_body_bytes'] == ledger['response_body_bytes'] <= LIMITS['total_response_bytes'],
            'Acquisition body totals changed')
    root = Path(args.output_dir).resolve()
    require(not root.exists(), 'Fresh fragment selection directory required')
    root.mkdir(parents=True)
    fragment = next(c for c in plan['fragments'] if c['legacy_alias'].lower() == args.fragment)
    result = dict(schema_version=1, status='STARTED', fragment=args.fragment, identity=fragment,
                  plan=record(args.plan), acquisition=record(acquisition_path),
                  source_objects=[r for r in acquisition['objects'] if r['fragment'] == args.fragment],
                  candidates=[], selected=[], CT_pixel_decodes=0, model_calls=0, network_requests=0)
    try:
        import numpy as np
        raw = {r['kind']: r for r in result['source_objects']}
        labels, label_quality = decode_binary(verify(raw['labels']['file']), fragment['shape_yx'])
        mask, mask_quality = decode_binary(verify(raw['mask']['file']), fragment['shape_yx'])
        inside_positive = outside_positive = 0
        for y in range(0, labels.shape[0], 256):
            ink = labels[y:y+256] != 0
            support = mask[y:y+256] != 0
            inside_positive += int(np.count_nonzero(ink & support))
            outside_positive += int(np.count_nonzero(ink & ~support))
        result['quality'] = dict(labels=label_quality, mask=mask_quality,
            label_positive_inside_surface=inside_positive, label_positive_outside_surface=outside_positive,
            outside_labels_policy='record and exclude; outside-positive count is not an eligibility gate')
        known = known_mask(labels, mask)
        candidates, border = candidates_for(labels, mask, known)
        result['candidates'] = candidates
        result['candidate_count'] = len(candidates)
        result['eligible_count'] = sum(c['eligible'] for c in candidates)
        selected = chosen(candidates)
        for i, case in enumerate(selected, 1):
            y, x = case['y'], case['x']
            case_id = args.fragment + '-context' + str(i)
            directory = root / case_id
            directory.mkdir()
            sl = (slice(y+1, y+1152, 3), slice(x+1, x+1152, 3))
            arrays = [('labels.npy', labels[sl]), ('surface-mask.npy', mask[sl].astype(bool)),
                      ('known-mask.npy', known[sl]), ('primary-mask.npy', known[sl] & border)]
            files = {}
            for name, array in arrays:
                require(array.shape == (384, 384), 'Selected shape changed')
                path = directory / name
                np.save(path, array, allow_pickle=False)
                files[name] = dict(**record(path), shape_yx=[384, 384], dtype=str(array.dtype))
            result['selected'].append(dict(**case, case_id=case_id, fragment=fragment['id'],
                role=fragment['role'], parent_papyrus=fragment['parent_papyrus'],
                bbox_source_xywh=[x, y, 1152, 1152], native_bbox_yxhw=[y, x, 1152, 1152], files=files,
                source_pixel_center_mapping={'x': f'{x}+1+3*x_coarse', 'y': f'{y}+1+3*y_coarse'}))
        require(directory_bytes(root) < LIMITS['selection_output_bytes'], 'Selection output cap exceeded')
        load_plan(args.plan)
        verify(result['acquisition'])
        for obj in acquisition['objects']:
            verify(obj['file'])
        result['status'] = 'PASS'
    except Exception as error:
        result.update(status='INCOMPLETE', failure_type=type(error).__name__)
        if isinstance(error, ValueError):
            result['stop_reason'] = str(error)
        raise
    finally:
        signal.alarm(0)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        result.update(wall_seconds=time.monotonic()-started, peak_rss_bytes=peak)
        if peak > LIMITS['worker_peak_rss_bytes']:
            result.update(status='INCOMPLETE', stop_reason='Peak RSS exceeded acceptance limit')
        save(root / 'selection-result.json', result)
    require(result['status'] == 'PASS', 'Selection failed resource acceptance')
    print(json.dumps({'status': 'PASS', 'result': record(root / 'selection-result.json')}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'STOP', 'failure_type': type(error).__name__}))
        raise SystemExit(1)
