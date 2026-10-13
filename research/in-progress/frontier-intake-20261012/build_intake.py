"""Build metadata-only reserve, development cube list and source-exposure table."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import gzip
import hashlib
import itertools
import json
from pathlib import Path
from registry import family, require_admitted


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(data, indent=2) + '\n')


def chunks_for(box, chunks):
    return list(itertools.product(*(range(a // n, (b - 1) // n + 1)
                                   for a, b, n in zip(box[0], box[1], chunks))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sources', required=True, type=Path)
    p.add_argument('--prior-sources', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    sources = args.sources
    def read(name):
        f = sources / (name + '.raw')
        b = f.read_bytes()
        if b[:2] == b'\x1f\x8b':
            b = gzip.decompress(b)
        return json.loads(b)
    catalog = read('current-catalog')['samples']
    socratic = read('socratic-registry')
    f0 = {v['id']: v for v in socratic['scrolls']}
    config_path = args.prior_sources / 'reader-config.raw'
    config = json.loads(config_path.read_text())
    rows = []
    for index, dataset in enumerate(config['datasets']):
        parent = 'PHerc' + dataset['sampling_scroll']
        for segment in dataset['segments']:
            rows.append({'dataset_index': index, 'sample': parent,
                         'family': family(parent), 'segment': segment,
                         'source_family': dataset.get('source_family'),
                         'declared_physical_key': dataset.get('sampling_physical_segment_keys', {}).get(segment),
                         'representation_key': dataset.get('sampling_representation_keys', {}).get(segment),
                         'evidence': 'released Reader-v2 train config; exact sampled coordinates and teacher ancestry not proved'})
    trained = {r['family'] for r in rows}
    published_audit = set(read('socratic-cube-plan')['scrolls'])
    reserve_candidates = sorted(sample for sample, entry in catalog.items()
        if sample in f0 and f0[sample]['f0_role'] == 'unseen'
        and sample not in published_audit and family(sample) not in trained
        and entry['sample']['properties']['type'] == 'scroll'
        and any(v['properties'].get('energy_keV') == 116 for v in entry['volumes'].values()))
    assert reserve_candidates[:2] == ['PHerc0175A', 'PHerc0175B']
    # Choice uses metadata/exposure only. Keep A/B pieces together conservatively.
    assert not ({'PHerc0175', 'PHerc1203'} & trained)
    for sample in ['PHerc0175A', 'PHerc0175B', 'PHerc1203']:
        assert f0[sample]['f0_role'] == 'unseen'
    registry = {
        'schema': 'plinius-physical-family-registry-v1',
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'catalog_sha256': sha(sources/'current-catalog.raw'),
        'reader_config_sha256': sha(config_path),
        'socratic_registry_sha256': sha(sources/'socratic-registry.raw'),
        'reserved_families': ['PHerc0175', 'PHerc1203'],
        'protected_running_families': ['PHerc0139'],
        'development_families': ['PHerc0800', 'PHerc0814', 'PHercParis4', 'PHerc0343', 'PHerc0009', 'PHerc0500', 'PHerc0172', 'PHerc1667'],
        'family_rule': 'Conservative shared numeric parent, including A/B/P/fragment suffixes; this groups possible relatives without asserting their exact physical adjacency.',
        'selection_rule': 'Reserve PHerc1203 as the previously documented fine/native lead and the lowest-numbered 116-keV scroll family outside Reader train families, F0 train/validation roles, and the published eight-scroll geometry audit: PHerc0175 A/B. No image or response criterion.',
        'status': 'PARENT_EMBARGO_FROZEN; NOT_READY_FOR_CONFIRMATION_SCORING',
        'current_native_go_sha256': '2c20b4491a2c94fdc4c479a84a6ab85f309673a113cfdd8088b30fb7df6bf61c',
        'limits': ['No reserve CT payload, surface prediction, label or ink map read.',
                   'F0-unseen is an author registry designation; m7/teacher/other model ancestry remains unresolved.',
                   'Unknown upstream exposure is not proved external holdout.',
                   'Independent annotations and exact confirmation contexts must be qualified before scoring.',
                   'This guard protects new intake manifests; it does not intercept all future tools or alter the running study.'],
        'reserves': []
    }
    for sample in ['PHerc0175A', 'PHerc0175B', 'PHerc1203']:
        entry = catalog[sample]
        volumes = []
        for volume, value in entry['volumes'].items():
            header = read(sample.lower()+'-'+volume+'-array')
            source = next(v for v in value['data'] if v['type'] == 'ome-zarr')['origins'][0]
            volumes.append({'volume_id': volume, 'scan_id': value['scan_id'],
                            'array_shape_zyx': header['shape'], 'chunk_shape_zyx': header['chunks'],
                            'dtype': header['dtype'], 'properties': value['properties'],
                            'source': source, 'protected_box_zyx': [[0,0,0], header['shape']]})
        registry['reserves'].append({'sample': sample, 'family': family(sample),
                                    'scope': 'Entire physical family, all scans, future aliases, teacher crops and derived meshes',
                                    'reader_v2_config_inclusion': False,
                                    'f0_declared_role': f0[sample]['f0_role'],
                                    'm7_and_ink_teacher_ancestry': 'unknown',
                                    'independent_reference_ready': False,
                                    'catalog_segments': len(entry['segments']), 'volumes': volumes})
    dump(args.output/'registry/REGISTRY.json', registry)
    args.output.joinpath('models').mkdir(parents=True, exist_ok=True)
    with (args.output/'models/READER_TRAIN_EXPOSURE.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    dump(args.output/'models/READER_SOURCE_SUMMARY.json', {
        'representations': len(rows), 'families': sorted(trained),
        'config_batch_size': config['batch_size'], 'source_quotas': config['fixed_scroll_prior']['target_batch_counts'],
        'known_alias_warning': 'pherc0139-w028 in the legacy training naming is the native w044 physical alias. Declared string keys alone are insufficient physical exclusions.',
        'reserve_family_intersections': sorted(trained & set(registry['reserved_families'])),
        'held_out_native_study_preserved': True})
    sample_plan = read('socratic-cube-plan')
    candidate_rows, total_raw, unique = [], 0, {}
    for sample in ['PHerc0800', 'PHerc0814']:
        require_admitted(sample, 'development', registry)
        entry = sample_plan['scrolls'][sample]
        origins = entry['drawn_origins_zyx']
        for ordinal in [0,6,12,18,24,31]:
            origin = origins[ordinal]
            box = [[v-32 for v in origin], [v+160 for v in origin]]
            row = {'id': sample+'-'+str(ordinal).zfill(2), 'sample': sample,
                   'source_ordinal_zero_based': ordinal, 'origin_zyx': origin,
                   'core_box_zyx': [origin, [v+128 for v in origin]],
                   'ct_context_box_zyx': box, 'voxel_size_um': entry['voxel_size_um'],
                   'stratum': 'UNASSIGNED: CT-only review required',
                   'reference': 'Published m7 field is a model prediction, not ground truth',
                   'source_chunks': {}}
            for kind, field in [('ct','volume_uri'), ('m7','prediction_uri')]:
                meta = read(sample.lower()+'-'+kind+'-array')
                assert meta['dtype'] == '|u1'
                assert meta['shape'] == entry['shape_l0_zyx']
                support = box if kind == 'ct' else row['core_box_zyx']
                assert all(0 <= a < b <= s for a,b,s in zip(*support, meta['shape']))
                coords = chunks_for(support, meta['chunks'])
                separator = meta.get('dimension_separator','.')
                keys = ['0/'+separator.join(map(str,c)) for c in coords]
                row['source_chunks'][kind] = {'root': entry[field], 'keys': keys}
                for key in keys:
                    unique[entry[field]+'/'+key] = {'sample': sample, 'kind': kind,
                                                  'decoded_bytes': int(__import__('math').prod(meta['chunks']))}
            candidate_rows.append(row)
    total_raw = sum(r['decoded_bytes'] for r in unique.values())
    dump(args.output/'geometry/NATIVE_CANDIDATES.json', {
        'status': 'FIXED_DEVELOPMENT_INTAKE; NOT_G1_LAUNCH',
        'selection': 'Six fixed sorted-origin ordinals 0,6,12,18,24,31 per parent from the public Socratic sample; no maps or scores used. Public benchmark exposure is declared.',
        'source_plan_sha256': sha(sources/'socratic-cube-plan.raw'),
        'cases': candidate_rows, 'unique_chunk_requests': len(unique),
        'decoded_chunk_bytes': total_raw,
        'compressed_transfer_bytes': 'Not known until bounded retrieval; decoded bytes are not a network quote.',
        'ct_or_surface_payloads_fetched': False,
        'next': 'Retrieve exactly this cohort; CT-only strata/trace/reference review, no output-driven replacements. Add independent connected-path references before a tracing claim.',
        'baseline': 'Published m7 field for geometry; later m7 and F0 with matched TTA, input and tracer for a model attribution claim.'})
    dump(args.output/'geometry/CHUNK_PLAN.json', {'requests': unique})
    tests = []
    for sample in ['PHerc0175A','PHerc175B','PHerc0175Cr1Fr9','PHerc1203','PHerc0139']:
        try:
            require_admitted(sample, 'training', registry)
        except ValueError:
            tests.append(sample+':reject-training')
        else:
            raise AssertionError(sample)
    assert require_admitted('PHerc175A','metadata',registry) == 'PHerc0175'
    assert require_admitted('Scroll1','development',registry) == 'PHercParis4'
    tests += ['reserved-metadata-allowed','legacy-Scroll1-resolved']
    for sample in ['unresolved-alias','PHerc1218']:
        try:
            require_admitted(sample,'development',registry)
        except ValueError:
            tests.append(sample+':reject-unassigned')
        else:
            raise AssertionError(sample)
    dump(args.output/'registry/GUARD_QUALIFICATION.json', {'status':'PASS','checks':len(tests),'cases':tests})
    print(json.dumps({'reserve_families': registry['reserved_families'], 'reserve_samples':len(registry['reserves']),
                      'reader_representations':len(rows), 'geometry_candidates':len(candidate_rows),
                      'unique_chunks':len(unique), 'decoded_chunk_bytes':total_raw, 'guard_checks':len(tests)}, indent=2))


if __name__ == '__main__':
    main()
