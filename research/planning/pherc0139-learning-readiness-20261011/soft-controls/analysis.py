#!/usr/bin/env python3
"""Frozen finite soft-target comparison; offline only, no learning or source fetch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXPECTED_PROTOCOL = 'dd87ce2618f75564d072fbc08ab3a1689bc37d1fbc0128620f5452b2e69e3052'
EXPECTED_DEPENDENCY = '24f625944826362b3c7029e4b412a35986440ac10d01eccaa2c9d9a2069da5d6'
PRIMARY = ('mean16_bilinear', 14)
OPERATORS = ('mean16_bilinear', 'point_bilinear')
PLANES = tuple(range(10, 18))
FIT = (0, -1, 0)
CONTRASTS = (('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def dependency():
    path = HERE / 'dependencies/prior_transfer.py'
    require(sha(path) == EXPECTED_DEPENDENCY, 'Pinned transfer dependency changed')
    spec = importlib.util.spec_from_file_location('soft_control_prior_transfer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rational_string(v):
    return str(v.numerator) if v.denominator == 1 else f'{v.numerator}/{v.denominator}'


def moments(shifts, weights):
    """Exact finite ZYX moments; covariance is centered population covariance."""
    require(sum(weights, Fraction()) == 1, 'Kernel mass must be one')
    means = [sum(w * int(s[d]) for w, s in zip(weights, shifts)) for d in range(3)]
    seconds = [[sum(w * int(s[a]) * int(s[b]) for w, s in zip(weights, shifts))
                for b in range(3)] for a in range(3)]
    cov = [[seconds[a][b] - means[a] * means[b] for b in range(3)] for a in range(3)]
    offset = [means[d] - FIT[d] for d in range(3)]
    return dict(mean_absolute_zyx_exact=[rational_string(v) for v in means],
                mean_relative_to_fit_zyx_exact=[rational_string(v) for v in offset],
                second_moment_absolute_zyx_exact=[[rational_string(v) for v in row] for row in seconds],
                covariance_zyx_exact=[[rational_string(v) for v in row] for row in cov],
                mean_absolute_zyx=[float(v) for v in means],
                mean_relative_to_fit_zyx=[float(v) for v in offset],
                covariance_zyx=[[float(v) for v in row] for row in cov])


def make_kernels(candidates):
    entries = candidates['supports']
    require([e['support'] for e in entries] == [4, 5, 6, 7], 'Frozen support IDs')
    require([len(e['candidates']) for e in entries] == [11, 6, 22, 4], 'Frozen candidate counts')
    unique = sorted({tuple(c['shift_zyx']) for e in entries for c in e['candidates']})
    weights, kernels = {}, []
    for e in entries:
        shifts = [tuple(c['shift_zyx']) for c in e['candidates']]
        require(len(set(shifts)) == len(shifts), 'Duplicate in individual kernel')
        local = [Fraction(int(s in shifts), len(shifts)) for s in unique]
        weights[e['support']] = local
        kernels.append(dict(support=e['support'], candidate_count=len(shifts),
                            candidate_indices=[unique.index(s) for s in shifts],
                            exact_weights_unique=[rational_string(w) for w in local],
                            **moments(unique, local)))
    global_weights = [sum((weights[e['support']][i] for e in entries), Fraction()) / 4
                      for i in range(len(unique))]
    return unique, weights, global_weights, dict(unique_shifts_zyx=[list(s) for s in unique],
        local_kernels=kernels, global_C=dict(exact_weights_unique=[rational_string(w) for w in global_weights],
                                           equal_kernel_mass='1/4', **moments(unique, global_weights)),
        reference_fit_zyx=list(FIT), covariance_convention='Population, exact rational arithmetic')


def operator_targets(raw_fit, raw_candidates, raw_centroid, local_indices, all_kernel_indices):
    """C averages four arithmetic kernel means, each with equal outer mass."""
    a = np.asarray(raw_fit, dtype=np.float64)
    b = raw_candidates[local_indices].mean(axis=0, dtype=np.float64)
    kernel_means = np.stack([raw_candidates[indices].mean(axis=0, dtype=np.float64)
                            for indices in all_kernel_indices])
    c = kernel_means.mean(axis=0, dtype=np.float64)
    d = np.asarray(raw_centroid, dtype=np.float64)
    return dict(A=a, B=b, C=c, D=d), kernel_means


def masks(raw_fit, raw_candidates, raw_centroid, geometry_fit, geometry_candidates,
          geometry_centroid, source_fit, source_candidates, source_centroid, native):
    geometry = geometry_fit & geometry_candidates.all(axis=0) & geometry_centroid
    source = source_fit & source_candidates.all(axis=0) & source_centroid
    finite = np.isfinite(raw_fit) & np.isfinite(raw_candidates).all(axis=0) & np.isfinite(raw_centroid)
    complete = geometry & source & finite & native
    nonzero = (raw_fit > 0) & (raw_candidates > 0).all(axis=0) & (raw_centroid > 0)
    common = complete & nonzero
    active = common & ((raw_fit >= 48) | (raw_candidates >= 48).any(axis=0) | (raw_centroid >= 48))
    return dict(geometry_only=geometry, source_only=source, finite=finite, native=native,
                complete=complete, all_source_teacher_nonzero=nonzero, common=common, active_union=active)


def difference(a, b, mask):
    d = (a[mask] - b[mask]) / 255.0
    if not len(d):
        return dict(pixels=0, MAE=None, RMSE=None, signed_mean=None,
                    p95_absolute=None, max_absolute=None)
    absolute = np.abs(d)
    return dict(pixels=int(len(d)), MAE=float(absolute.mean()), RMSE=float(np.sqrt(np.mean(d*d))),
                signed_mean=float(d.mean()), p95_absolute=float(np.quantile(absolute, .95)),
                max_absolute=float(absolute.max()))


def entropy(q):
    """Natural-log Bernoulli target entropy, including exact endpoints."""
    q = np.asarray(q, dtype=np.float64)
    require(np.all((q >= 0) & (q <= 1)), 'Target outside unit interval')
    result = np.zeros_like(q)
    m = (q > 0) & (q < 1)
    result[m] = -q[m] * np.log(q[m]) - (1-q[m]) * np.log1p(-q[m])
    return result


def summaries(arms, domain_masks):
    common, active = domain_masks['common'], domain_masks['active_union']
    arm_rows = {}
    for name, raw in arms.items():
        rows = {}
        for domain, m in [('common', common), ('active_union', active)]:
            q = raw[m] / 255.0
            rows[domain] = dict(pixels=int(m.sum()), target_mean=float(q.mean()) if len(q) else None,
                                mean_target_entropy_nats=float(entropy(q).mean()) if len(q) else None,
                                threshold48_positive_pixels=int((raw[m] >= 48).sum()))
        arm_rows[name] = rows
    comparisons = {}
    for left, right in CONTRASTS:
        rows = {}
        for domain, m in [('common', common), ('active_union', active)]:
            row = difference(arms[left], arms[right], m)
            la, rb = arms[left][m] >= 48, arms[right][m] >= 48
            union = int((la | rb).sum())
            row.update(threshold48_disagreements=int((la ^ rb).sum()),
                       threshold48_positive_union_pixels=union,
                       threshold48_disagreement_fraction_of_pair_union=float((la ^ rb).sum()/union) if union else None)
            rows[domain] = row
        comparisons[f'{left}-{right}'] = rows
    routing = {}
    for key in ['B-C', 'B-D']:
        value = comparisons[key]['active_union']['MAE']
        routing[key] = dict(active_union_MAE=value, source_quantization_interval=1/255,
                            strictly_below_one_source_interval=value < 1/255 if value is not None else None)
    return dict(arms=arm_rows, differences=comparisons, descriptive_quantization_comparisons=routing,
                signed_difference_convention='left minus right', target_units='original uint8 score / 255')


def create_input_manifest(correspondence, supervision):
    """Freeze all consumed source files plus implementation before real target reads."""
    paths = [('study', HERE / 'PROTOCOL.md'), ('study', HERE / 'analysis.py'),
             ('study', HERE / 'synthetic_tests.py'), ('study', HERE / 'dependencies/prior_transfer.py'),
             ('study', HERE / 'SYNTHETIC_TESTS.json')]
    for name in ['CT_CANDIDATES.json', 'teacher/CROP.json', 'teacher/crop.npy']:
        paths.append(('supervision', supervision/name))
    for name in ['geometry-review/pullback/site1/RESULTS.json',
                 'geometry-review/pullback/site1/queries_uvk.npy',
                 'geometry-review/pullback/site1/support.npy',
                 'ct-study/REQUEST_PLAN.json', 'ct-study/REQUEST_LEDGER.json']:
        paths.append(('correspondence', correspondence/name))
    candidate = json.loads((supervision/'CT_CANDIDATES.json').read_text())
    for entry in candidate['input_files']:
        path = correspondence/entry['relative']
        require(sha(path) == entry['sha256'], 'CT candidate source mismatch')
        paths.append(('correspondence', path))
    request_plan = json.loads((correspondence/'ct-study/REQUEST_PLAN.json').read_text())
    for entry in request_plan['requests']:
        if entry['site'] == 'site1' and entry['acquisition'] == 'native':
            paths.append(('correspondence', correspondence/'ct-study/chunks'/entry['name']))
    roots = dict(study=HERE, correspondence=correspondence, supervision=supervision)
    rows = [dict(packet=kind, relative=str(path.relative_to(roots[kind])),
                 size_bytes=path.stat().st_size, sha256=sha(path)) for kind,path in paths]
    require(sha(HERE/'PROTOCOL.md') == EXPECTED_PROTOCOL, 'Frozen protocol changed')
    require(json.loads((HERE/'SYNTHETIC_TESTS.json').read_text())['status']=='PASS', 'Tests required')
    save_json(HERE/'INPUT_MANIFEST.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_packets={k:str(v) for k,v in roots.items()}, files=rows,
        stage='Before new real-data operator contrasts; existing teacher previously inspected'))


def run(correspondence, supervision):
    require(not (HERE/'results').exists(), 'No overwrite or rerun')
    create_input_manifest(correspondence, supervision)
    prior = dependency()
    q, g, candidates = prior.load_geometry(correspondence, supervision/'CT_CANDIDATES.json')
    native, native_receipt = prior.load_native_eligibility(correspondence)
    metadata = json.loads((supervision/'teacher/CROP.json').read_text())
    require(metadata['origin_fullfine_xy']==[9084,7088] and metadata['crop_shape_yx']==[780,788],
            'Frozen teacher origin and crop shape')
    require(sha(supervision/'teacher/crop.npy')==metadata['crop_sha256'], 'Teacher crop hash')
    crop = np.load(supervision/'teacher/crop.npy', allow_pickle=False)
    require(crop.dtype==np.uint8 and list(crop.shape)==[780,788], 'Teacher array contract')
    origin = metadata['origin_fullfine_xy']
    pool = prior.pooled_crop(crop, origin)
    unique, local_weights, global_weights, kernel_record = make_kernels(candidates)
    require(len(unique)==24 and FIT in unique, 'Frozen global shift inventory')
    fit_index = unique.index(FIT)
    all_kernel_indices = [e['candidate_indices'] for e in kernel_record['local_kernels']]
    output = HERE/'results'
    output.mkdir()
    (output/'queries').mkdir()
    (output/'cases').mkdir()
    save_json(output/'KERNELS.json', kernel_record)
    save_json(output/'NATIVE_ELIGIBILITY.json', native_receipt)
    np.save(output/'native_eligible_max28_gt0.npy', native, allow_pickle=False)
    rows = []
    for entry, kernel in zip(candidates['supports'], kernel_record['local_kernels']):
        sid, bounds = entry['support'], entry['native_xy_half_open']
        x0,y0,x1,y1 = bounds
        native_mask = native[y0-prior.HALO_XY[1]:y1-prior.HALO_XY[1],
                             x0-prior.HALO_XY[0]:x1-prior.HALO_XY[0]]
        require(native_mask.shape==(32,32), 'Fixed support shape')
        indices = kernel['candidate_indices']
        for j in PLANES:
            shifted = [prior.shifted_queries(q,g,bounds,j,s) for s in unique]
            uvs = np.stack([v[0] for v in shifted])
            valid_geometry = np.stack([v[1] for v in shifted])
            uv_mean = uvs[indices].mean(axis=0, dtype=np.float64)
            centroid_g = valid_geometry[indices].all(axis=0) & np.isfinite(uv_mean).all(axis=-1)
            query_file = output/'queries'/f's{sid}_j{j:02d}.npz'
            np.savez_compressed(query_file, shifts_zyx=np.array(unique,dtype=np.int64),
                candidate_uv_group2=uvs, candidate_geometry_valid=valid_geometry,
                fit_uv_group2=uvs[fit_index], fit_geometry_valid=valid_geometry[fit_index],
                centroid_uv_group2=uv_mean, centroid_geometry_valid=centroid_g,
                native_eligible=native_mask, local_indices=np.array(indices,dtype=np.int64),
                local_weights=np.array([float(w) for w in local_weights[sid]],dtype=np.float64),
                global_weights=np.array([float(w) for w in global_weights],dtype=np.float64))
            for operator in OPERATORS:
                transferred = [prior.sample_teacher(crop,origin,uv,np.ones((32,32),bool),operator,pool)
                               for uv in uvs]
                raw_candidates = np.stack([v[0] for v in transferred])
                source_candidates = np.stack([v[1] for v in transferred])
                raw_d, source_d = prior.sample_teacher(crop,origin,uv_mean,np.ones((32,32),bool),operator,pool)
                raw_a, source_a = raw_candidates[fit_index], source_candidates[fit_index]
                arms, all_means = operator_targets(raw_a,raw_candidates,raw_d,indices,all_kernel_indices)
                domain_masks = masks(raw_a,raw_candidates,raw_d,valid_geometry[fit_index],valid_geometry,
                                     centroid_g,source_a,source_candidates,source_d,native_mask)
                case_name = f'{operator}_s{sid}_j{j:02d}'
                case_file = output/'cases'/f'{case_name}.npz'
                arrays = {f'target_{k}':v/255.0 for k,v in arms.items()}
                arrays.update({f'raw_{k}':v for k,v in arms.items()})
                arrays.update({f'mask_{k}':v for k,v in domain_masks.items()})
                arrays.update(raw_candidates=raw_candidates, source_candidates_valid=source_candidates,
                              source_centroid_valid=source_d, source_fit_valid=source_a,
                              all_kernel_means_raw=all_means)
                np.savez_compressed(case_file,**arrays)
                row = dict(case=case_name, operator=operator, support=sid, native_plane=j,
                    primary=(operator,j)==PRIMARY, native_xy_half_open=bounds,
                    local_candidate_count=len(indices), unique_global_candidate_count=len(unique),
                    requested_pixels=1024, mask_counts={k:int(v.sum()) for k,v in domain_masks.items()},
                    outside_crop_pixels=int((~domain_masks['source_only']).sum()),
                    outside_crop_sample_occurrences=int((~source_candidates).sum()+(~source_d).sum()),
                    geometry_missing_pixels=int((~domain_masks['geometry_only']).sum()),
                    arrays=dict(relative=str(case_file.relative_to(output)),sha256=sha(case_file)),
                    queries=dict(relative=str(query_file.relative_to(output)),sha256=sha(query_file)),
                    **summaries(arms,domain_masks))
                rows.append(row)
    require(len(rows)==64 and sum(r['primary'] for r in rows)==4,'Exactly64 cases and4 primary')
    manifest = json.loads((HERE/'INPUT_MANIFEST.json').read_text())
    for entry in manifest['files']:
        root = Path(manifest['source_packets'][entry['packet']])
        require(sha(root/entry['relative'])==entry['sha256'],'Input changed during analysis')
    save_json(output/'RESULTS.json',dict(status='COMPLETE_FINITE_OPERATOR_STUDY',
        completed_utc=datetime.now(timezone.utc).isoformat(), cases=rows,
        protocol_sha256=EXPECTED_PROTOCOL, input_manifest_sha256=sha(HERE/'INPUT_MANIFEST.json'),
        cases_count=64, primary_cases_count=4, unique_shift_count=len(unique),
        new_teacher_pixels=0, new_CT_chunks=0, model_forwards=0, optimizer_updates=0,GPU_spend_GBP=0,
        prior_correspondence_screen='FAIL; unchanged', claims='Finite target differences only; no learning or ink accuracy'))
    checks = [dict(relative=str(p.relative_to(output)),size_bytes=p.stat().st_size,sha256=sha(p))
              for p in sorted(output.rglob('*')) if p.is_file()]
    save_json(output/'CHECKSUMS.json',dict(files=checks))
    print(json.dumps(dict(cases=64,primary=[dict(support=r['support'],mask_counts=r['mask_counts'],
        contrasts=r['descriptive_quantization_comparisons']) for r in rows if r['primary']]),indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--correspondence',type=Path,required=True)
    parser.add_argument('--supervision',type=Path,required=True)
    args=parser.parse_args()
    run(args.correspondence.resolve(),args.supervision.resolve())
