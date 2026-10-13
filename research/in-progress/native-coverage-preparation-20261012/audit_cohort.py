"""Bind proposed training sources using retained metadata; no image reads."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import sys


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--catalog', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    sys.path.insert(0, str(a.repo/'research/in-progress/frontier-intake-20261012'))
    from registry import family
    raw = a.catalog.read_bytes()
    catalog = json.loads(gzip.decompress(raw) if raw[:2] == b'\x1f\x8b' else raw)['samples']
    config = json.loads(a.config.read_text())
    reserved = {'PHerc0175', 'PHerc1203'}
    evaluation = {'PHerc0841', 'PHerc0009'}
    # No post-result source selection: audit every native-v2 entry, then apply
    # explicit observation/teacher matching rules to the first regional pair.
    names = dict(PHerc0139='0139', PHerc0814='0814', PHerc0500P2='0500P2',
                 PHerc0343P='0343P', PHerc0172='0172')
    rows = []
    for sample, scroll in names.items():
        group = family(sample)
        assert group not in reserved | evaluation
        data = next(d for d in config['datasets'] if d['source_family'] == 'native_v2_'+scroll)
        by_long = {s['long_id']: s for s in catalog[sample]['segments'].values()}
        for name in data['segments']:
            matches = [s for key, s in by_long.items() if key.startswith(name)]
            assert len(matches) == 1, (sample, name)
            seg = matches[0]
            def source(d):
                origins = d['origins']; assert len(origins) == 1
                vol = d['parameters']['target_volume']
                return dict(path=origins[0]['path'], volume=vol,
                            acquisition=catalog[sample]['volumes'][vol]['properties'],
                            model_id=d['parameters'].get('model_id'))
            renders = [source(d) for d in seg['data'] if d['type'] == 'layers-zarr']
            maps = [source(d) for d in seg['data'] if d['type'] == 'ink-detection']
            native = [r for r in renders if r['acquisition']['pixel_size_um'] in [9.362, 8.64, 7.91]]
            assert len(native) == 1
            fine = [m for m in maps if m['model_id'] == '20260417190342']
            matched = [m for m in fine if m['acquisition']['pixel_size_um'] == 2.399
                       and m['acquisition']['energy_keV'] == 78]
            contract_match = (native[0]['acquisition']['pixel_size_um'] == 9.362
                              and native[0]['acquisition']['energy_keV'] == 113 and len(matched) == 1)
            role = 'additional_regional_coverage_candidate' if sample == 'PHerc0814' and contract_match else 'deferred'
            if sample == 'PHerc0139':
                reason = 'Same acquisition/teacher contract, but not needed to isolate added 0814 regional coverage; keep distinct from native human-reference stream'
            elif role != 'deferred':
                reason = 'All eight released native-v2 0814 segments; both arms use the same parent/segment quotas and teacher'
            else:
                reason = 'Changes teacher acquisition or native acquisition; separate data-package/acquisition experiment'
            rows.append(dict(sample=sample, family=group, segment=seg['long_id'], source_key=name,
                native=native[0], teacher=matched[0] if matched else (fine[0] if len(fine)==1 else None),
                other_maps=maps, matches_9p362_113kev_and_2p399_78kev_teacher=contract_match,
                role=role, reason=reason, upstream_reader_training=True,
                physical_registration='not established by metadata or detector correlation',
                supervision_tier='thresholded published model prediction, not independent ink truth'))
    originals = []
    for sid, seg in catalog['PHerc0139']['segments'].items():
        for d in seg['data']:
            if d['type'] != 'ink-labels' or d['properties'].get('registration') != 'native':
                continue
            assert d['parameters']['target_volume'] == d['properties']['annotated_on'] == '20250728140407'
            render = [r for r in seg['data'] if r['type']=='layers-zarr' and
                      r['parameters']['target_volume']=='20250728140407']
            assert len(render)==1
            originals.append(dict(sample='PHerc0139', family='PHerc0139', segment=seg['long_id'],
                native_path=render[0]['origins'][0]['path'], labels=d,
                role='fixed_original_native_annotation_stream',
                scope='Same native coordinate frame by catalogue; annotation ancestry is not independent physical authentication'))
    assert len(rows)==56 and len(originals)==5
    candidates=[r for r in rows if r['role']=='additional_regional_coverage_candidate'];assert len(candidates)==8
    a.output.mkdir(parents=True)
    result=dict(status='SOURCE_CONTRACTS_BOUND_NOT_FINAL_TRAINING_ADMISSION',
        pins={str(p):sha(p) for p in [a.catalog,a.config,Path(__file__)]},
        reserved_families=sorted(reserved), new_training_excluded_families=sorted(evaluation),
        original_stream=sorted(originals,key=lambda r:r['segment']), native_v2_audit=rows,
        new_study_0139_policy='Completed study remains immutable; its exposed training/reference material may supply a separate strong-model preparation. No continuation or endpoint reselection.',
        proposed_contrast='Same eight PHerc0814 segments and source quotas in both arms, restricted versus broader regional support; original PHerc0139 stream identical. This isolates regional coverage within a fixed parent mixture, not an eight-parent generalization claim.',
        pending=['Native/teacher array metadata and bytes','Training support eligibility and physical alias/overlap accounting',
                 'Restricted/broad support freeze without model-response selection','Full cohort and schedule; GPU runtime and artifact delivery'],
        metadata_only=True, new_inference=0, optimizer_updates=0, paid_compute=False)
    (a.output/'COHORT_CANDIDATES.json').write_text(json.dumps(result,indent=2)+'\n')
    requests=[]
    for r in candidates:
        ident=r['sample'].lower()+'-'+r['segment'].split('-')[0]
        for suffix in ['.zattrs','0/.zarray']:
            requests.append(dict(id=ident+('-attrs' if suffix=='.zattrs' else '-array'),
                url='https://vesuvius-challenge-open-data.s3.amazonaws.com/'+r['native']['path'].rstrip('/')+'/'+suffix,
                max_bytes=1024*1024))
    (a.output/'METADATA_REQUESTS.json').write_text(json.dumps(requests,indent=2)+'\n')
    print(json.dumps(dict(native_v2_audited=len(rows), original_native_label_segments=len(originals),
                         added_candidates=len(candidates), metadata_requests=len(requests))))


if __name__ == '__main__':
    main()
