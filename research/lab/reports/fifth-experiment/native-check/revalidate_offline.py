"""Recompute frozen native profiles offline after the recorded input-edit race."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import socket

import numpy as np

ROOT = Path(__file__).resolve().parent
PLAN_SHA = '50c8fa7cea430111b61a4c66dfd87965dee8bd4e53d06323151a83c592ce9a67'
CODE_SHA = '4d74a492e05c38f7a5074b9b8b6b6d35d5f69d041bb539310386971f6b4357b5'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    socket.socket.connect = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('Offline revalidation forbids network'))
    code = ROOT / 'check_native_ct.py'
    plan_path, previous_path = ROOT / 'frozen-plan.json', ROOT / 'result.json'
    if sha(code) != CODE_SHA or sha(plan_path) != PLAN_SHA:
        raise ValueError('Original reviewed package is not restored exactly.')
    spec = importlib.util.spec_from_file_location('frozen_native_check', code)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = json.loads(plan_path.read_text())
    previous = json.loads(previous_path.read_text())
    previous_sha = sha(previous_path)
    if previous['frozen_plan_sha256'] != PLAN_SHA or previous['stop_reason'] != 'Input size changed':
        raise ValueError('Unexpected predecessor; do not silently replace a different failure.')
    module.verify_records(plan['input_files'])
    module.verify_render(plan['render_parameters'])
    module.verify_native_metadata(plan['native_metadata'])
    arrays = {}
    observed = {tuple(r['chunk_index_zyx']): r for r in previous['native_chunk_sources']}
    for chunk in plan['chunks']:
        index = tuple(chunk['chunk_index_zyx'])
        record = observed[index]
        path = Path(record['local_path'])
        if (record['http_status'] != 200 or record['acquisition'] != 'new_http200_payload'
            or record['url'] != chunk['url'] or str(path) != chunk['local_path']
            or path.stat().st_size != 128**3 or sha(path) != record['sha256']):
            raise ValueError('Previously observed native chunk provenance failed.')
        arrays[index] = np.frombuffer(path.read_bytes(), dtype=np.uint8).reshape(128,128,128)
    metadata = module.DATA / 'ct.level0.zarray.json'
    if sha(metadata) != plan['native_metadata_sha256']:
        raise ValueError('Retained native metadata differs from frozen metadata.')
    surface = np.load(module.SURFACE / 'surface.npy', mmap_mode='r', allow_pickle=False)
    points, residuals = [], []
    for point in plan['points']:
        prediction = np.array([module.trilinear(s['position_ct_xyz'], arrays) for s in point['samples']], dtype=np.float32)
        rendered = np.clip(prediction,0,255).astype(np.uint8)
        x,y = point['surface_local_roi_xy']
        source = np.array(surface[:,y,x],dtype=np.uint8)
        diff = np.abs(source.astype(np.int16)-rendered.astype(np.int16))
        residuals.extend(diff.tolist())
        points.append({**point,'native_ct_profile_float32':prediction.tolist(),
            'rendered_profile_uint8':rendered.tolist(),'observed_profile_uint8':source.tolist(),
            'absolute_intensity_errors':diff.tolist(),'exact_matches':int((diff==0).sum()),
            'within_one_intensity_count':int((diff<=1).sum()),'max_abs_error':int(diff.max()),
            'mean_abs_error':float(diff.mean())})
    module.verify_records(plan['input_files'])
    if sha(code)!=CODE_SHA or sha(plan_path)!=PLAN_SHA or sha(previous_path)!=previous_sha:
        raise ValueError('An input changed during offline revalidation.')
    aggregate={'samples':len(residuals),'exact_matches':sum(d==0 for d in residuals),
        'within_one_intensity_count':sum(d<=1 for d in residuals),'max_abs_error':max(residuals),
        'mean_abs_error':float(np.mean(residuals))}
    if len(residuals)!=310 or max(residuals)>1 or aggregate!=previous['aggregate']:
        raise ValueError('Offline fixed comparison differs or fails preregistered tolerance.')
    for old,new in zip(previous['points'],points):
        if old != new:
            raise ValueError('A re-evaluated native profile differs from the retained first computation.')
    result={'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'passed_offline_revalidation_of_original_frozen_comparison',
        'render_comparison_gate_passed':True,'context_visual_review_gate_passed':None,
        'predecessor_path':str(previous_path),'predecessor_sha256':previous_sha,
        'predecessor_stop_preserved':'Input size changed during original execution; original stopped result is unchanged.',
        'resolution':'Original reviewed code and plan restored byte-for-byte; all frozen inputs verified before and after a separate offline recomputation from retained observed cubes.',
        'frozen_plan_path':str(plan_path),'frozen_plan_sha256':PLAN_SHA,'frozen_native_code_sha256':CODE_SHA,
        'revalidation_code_sha256':sha(__file__),'native_chunk_sources':previous['native_chunk_sources'],
        'points':points,'aggregate':aggregate,'input_files':plan['input_files'],
        'network_requests':0,'new_native_bytes':0,'all_recomputed_points_equal_predecessor':True,
        'limitations':['Local rendering consistency only; ten points share two mesh-normal neighborhoods.',
                       'No ink authentication or independent papyrus-layer identification.'],
        'letter':None,'transcription':None,'translation':None}
    with (ROOT/'offline-revalidation.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'status':result['status'],'aggregate':aggregate,'network_requests':0},indent=2))


if __name__ == '__main__':
    main()
