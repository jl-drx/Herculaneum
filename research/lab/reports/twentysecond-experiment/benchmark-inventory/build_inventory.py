"""Retained text/metadata inventory only. Never opens scientific arrays or weights."""
import gzip
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode

LAB = Path('/workspace/herculaneum-lab')
OUT = LAB / 'reports/twentysecond-experiment/benchmark-inventory'
R = LAB / 'reports'
CAT = LAB / 'data/experiments/pherc0800-native-ct-consistency/metadata.full.json.gz'
sources = {}

def pin(path):
    path = Path(path)
    data = path.read_bytes()
    value = {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'byte_length': len(data)}
    sources[str(path)] = value
    return value

def read(path):
    pin(path)
    return json.loads(Path(path).read_text())

def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

catalog_pin = pin(CAT)
assert catalog_pin['sha256'] == '7ec8dcaa3000b9716724c26a1991ee416491d8a6b3615238ca90ae11540abdd3'
catalog = json.loads(gzip.decompress(CAT.read_bytes()))
old = read(R / 'tenth-experiment/reference-audit/reference-matrix.json')
phase19 = read(R / 'nineteenth-experiment/scroll-audit/eligibility-matrix.json')
register = read(R / 'research-register-v17.json')
assert len(register['experiments']) == 21

modern_label_inventory = []
for parent, sample in catalog['samples'].items():
    for seg_id, segment in sample.get('segments', {}).items():
        for i, entry in enumerate(segment.get('data', [])):
            if entry.get('type') == 'ink-labels':
                modern_label_inventory.append({
                    'parent': parent, 'segment': seg_id,
                    'source_json_pointer': f'/samples/{parent}/segments/{seg_id}/data/{i}',
                    'origins': entry['origins'], 'properties': entry['properties'],
                })
assert len({x['parent'] for x in modern_label_inventory}) == 7
write('CATALOG_LABEL_INVENTORY.json', {
    'source': catalog_pin, 'scope': 'All explicit type=ink-labels entries in this retained catalog; not all public annotation assets.',
    'entry_count': len(modern_label_inventory),
    'parent_count': len({x['parent'] for x in modern_label_inventory}),
    'entries': modern_label_inventory,
})

extracts = {}
for parent in ['PHerc0009B', 'PHerc0500P2', 'PHerc0841', 'PHerc51Cr4Fr8', 'PHerc1667Cr1Fr3',
               'PHercParis1Fr34', 'PHercParis1Fr39', 'PHercParis2Fr47', 'PHercParis2Fr143']:
    value = catalog['samples'][parent]
    relevant_segments = {}
    for seg_id, seg in value.get('segments', {}).items():
        if any(x.get('type') == 'ink-labels' for x in seg.get('data', [])):
            relevant_segments[seg_id] = {
                'long_id': seg.get('long_id'),
                'data': [x for x in seg.get('data', []) if x.get('type') in ['ink-labels', 'ink-detection']],
            }
    extracts[parent] = {
        'source_json_pointer': f'/samples/{parent}',
        'sample': value['sample'],
        'scan_ids': sorted(value.get('scans', {})),
        'volume_ids': sorted(value.get('volumes', {})),
        'all_segment_ids': sorted(value.get('segments', {})),
        'labelled_segments_and_colocated_predictions_metadata': relevant_segments,
    }
write('CATALOG_EXTRACTS.json', {
    'source': catalog_pin, 'samples': extracts,
    'canonical_model_metadata': catalog['models']['20260417190342'],
    'canonical_metadata_limit': 'compatible_samples and co-located predictions establish supported/applied inference, not training membership or annotation teacher identity.',
})

training_dir = R / 'frontier-methods-audit-20261008/source-evidence/training'
native_cfg_path = next(training_dir.glob('*-nieuwlaar-training__configs__train_dense_native.json'))
native_cfg = read(native_cfg_path)
native_rows = [{'sampling_scroll': d.get('sampling_scroll'), 'segments': d['segments'],
                'segments_path': d['segments_path'], 'source_family': d.get('source_family')}
               for d in native_cfg['datasets']]
direct = {}
for name, model in old['original_models'].items():
    rows = model['configured_representations']
    assert len(rows) == 29
    direct[name] = {
        'artifact_sha256': model['sha256'], 'representation_count': len(rows),
        'parents_by_physical_key': sorted({v['physical_key'].split(':')[0] for v in rows}),
        'rows': rows,
    }
write('DIRECT_MODEL_ROOTS.json', {
    'four_originals_source': sources[str(R / 'tenth-experiment/reference-audit/reference-matrix.json')],
    'four_originals': direct,
    'native_published_training_config_source': sources[str(native_cfg_path)],
    'native_published_training_rows': native_rows,
    'scope': 'Original rows derive from previously decoded checkpoint configurations. Native rows are published source configuration, not a full training execution receipt. Neither establishes complete ancestors.',
})

evidence_paths = [
    R / 'twentyfirst-experiment/NEXT_STUDY_DRAFT.md',
    R / 'twentyfirst-experiment/RESULTS.md',
    R / 'tenth-experiment/reference-audit/LINEAGE.md',
    R / 'nineteenth-experiment/exposure-audit/AUDIT.md',
    R / 'nineteenth-experiment/scroll-audit/RESULTS.md',
    R / 'nineteenth-experiment/optical-audit/OPTICAL_CANDIDATE_AUDIT.md',
    R / 'nineteenth-experiment/optical-audit/exposure/RETAINED_LEGACY_EXPOSURE.md',
    R / 'eleventh-experiment/optical-reference/OPTICAL_REFERENCE.md',
    R / 'fifteenth-experiment/RESULTS.md',
    R / 'fifteenth-experiment/BENCHMARK_DESIGN.md',
    R / 'frontier-methods-audit-20261008/RULES_AND_TRAINING.md',
    R / 'twentyfirst-experiment/research-context/EVIDENCE.md',
    next(training_dir.glob('*-nieuwlaar-hf-training__teacher_sources.tsv')),
    next(training_dir.glob('*-nieuwlaar-training__REPRODUCE.md')),
    LAB / 'data/native-reference-preflight-20261008/iteration-0-README.md',
    LAB / 'data/native-reference-preflight-20261008/paper-ink-detection__metadata.json',
    LAB / 'data/experiments/known-ink-control/upstream-curated-ink-labels-README.md',
    R / 'nineteenth-experiment/exposure-audit/canonical-readme.body',
    R / 'nineteenth-experiment/exposure-audit/canonical-config.body',
    R / 'nineteenth-experiment/exposure-audit/canonical-api.body',
    R / 'frontier-methods-audit-20261008/source-evidence/official-contributions/russo-scripts-teacher_seg.py',
]
excerpts = []
terms = ('PHerc0009B', 'PHerc0500P2', 'Frag6', 'Frag5', 'PHerc51', 'teacher', 'Teacher',
         'Paris', 'pretrained', '0009', '500p2', '9b', '202509', 'sup = sup | m_sup')
for path in evidence_paths:
    source = pin(path)
    lines = path.read_text().splitlines()
    hits = [{'line': i, 'text': line} for i, line in enumerate(lines, 1) if any(t in line for t in terms)]
    excerpts.append({'source': source, 'excerpts': hits})
write('TEXT_EVIDENCE.json', {'selection': 'Literal matching lines; source files remain unchanged.', 'sources': excerpts})

unknown_ancestry = 'Unresolved canonical R152 training/init and original annotation teacher histories; missing named root is not proof of independence.'
rows = []
aliases = {
    'PHerc0009B': ['PHerc0009B', '0009b', '9b', 'new_traces9b/david_9b', '20250910185200', '20250919125754-auto_grown_20250919055754487_inp_hr'],
    'PHerc0500P2': ['PHerc0500P2', '0500p2', '500p2a', 'train_scrolls/500p2a', 'sean_hiddenlayers/-1', '0500p2/-1'],
    'PHerc0841': ['PHerc0841', '841', '841/w00', '841/auto_grown_20260220174252405', '841/auto_grown_20260220144552896'],
}
for parent in ['PHerc0009B', 'PHerc0500P2', 'PHerc0841']:
    rows.append({
        'parent_group': parent, 'aliases': aliases[parent], 'catalog_scan_ids': extracts[parent]['scan_ids'],
        'catalog_volume_ids': extracts[parent]['volume_ids'], 'all_catalog_segment_ids': extracts[parent]['all_segment_ids'],
        'four_original_models_direct_roots': 'Absent from all four explicit 29-representation configurations.',
        'native_model_direct_roots': 'Absent from published native training configuration; initializes from Russo ex016, so inherited ancestors also matter.',
        'paper_R3D50_model': 'Published model card explicitly reports parent exposure; alias bridge in paper pipeline train/val metadata.' if parent != 'PHerc0841' else 'Not named in inspected eight-entry paper pipeline; no exhaustive alias/history exclusion established.',
        'production_canonical_R152': unknown_ancestry,
        'annotation_derivation': 'Transferred; annotated_on=unknown; quality review uses an unnamed independent ink prediction. Optical derivation not authenticated.',
        'annotation_specific_limit': '20260221022814 record explicitly describes aggressively completed, iteratively reconstructed letter forms; other two sibling records also use prediction-based review.' if parent == 'PHerc0841' else 'Public mapping_validity and linked provenance.json absent from complete phase19 package inventories; four exact 404s retained.',
        'own_lab_use': 'No recorded candidate pixels/predictions, fitting or calibration in the inspected phase1–21 registry/source reports; metadata-only reserve.' if parent != 'PHerc0841' else 'No recorded scientific use in inspected phase1–21 registry; this phase extracts metadata only. Not a claim about every unrecorded action.',
        'optical_reference': 'Catalog specimen photo exists; no authenticated registered optical ink-label/negative-support contract.',
        'eligible_now': 'Metadata/source contract audit only; not an independent accuracy set or acquisition-ready benchmark.',
        'conditional_claim': 'Prospectively excluded local-parent supplied-reference agreement, only after acquisition/support/input contract is frozen and passed.',
    })
rows.append({
    'parent_group': 'PHerc51 / PHerc0051', 'aliases': ['Frag6', 'PHerc51Cr4Fr8', 'PHerc0051Cr04Fr08'],
    'legacy_volume_ids': ['20231121152933', '20231130112027', '20231201112849', '20231201120546'],
    'scan_descriptors': ['53keV 3.24um', '53keV 7.91um', '88keV 3.24um', '70keV 3.24um'],
    'four_original_models_direct_roots': 'Absent from explicit configurations.',
    'native_model_direct_roots': 'Absent from published configuration.',
    'paper_R3D50_model': 'Not affirmatively named in inspected record; unknown complete history.',
    'production_canonical_R152': unknown_ancestry,
    'legacy_InkID': 'Four Paris surfaces are described; two additional legacy identities are obfuscated/unbridged. Six-fragment wording does not establish Frag6 inclusion or exclusion.',
    'annotation_derivation': 'Co-located inklabels.png, inklabels.psd, ir.png, mask.png exist in retained directory metadata; specific authorship, tracing method, optical warp/residual and negative support unknown.',
    'own_lab_use': 'Reserved, locally unscored. Phase21 structural headers and fixed sparse PPM numerical geometry inspected; no CT intensities, optical/label images or model outputs.',
    'eligible_now': 'Metadata/source contract audit; Stage C still closed.',
    'conditional_claim': 'Locally unscored supplied-reference comparison after render/source/annotation contract closure, with qualified ancestry; not independent confirmation.',
})
rows += [
    {'parent_group': 'PHerc1667', 'aliases': ['Frag5', 'PHerc1667Cr1Fr3', 'Scroll 4'],
     'direct_and_parent_exposure': 'PHerc1667 directly included in four originals and native configuration; fragment is explicitly from that same parent.',
     'own_lab_use': 'PHerc1667 reused controls, calibration and development; Frag5 specific pixels not scored in this inventory.',
     'annotation_derivation': 'Frag5 has optical/labels/mask assets, but its specific contract remains unresolved.',
     'eligible_now': 'Cannot establish a new-parent holdout even if Frag5 pixels were not used.'},
    {'parent_group': 'Paris1', 'aliases': ['Frag3/PHercParis1Fr34', 'Frag4/PHercParis1Fr39'],
     'four_original_models_direct_roots': 'Absent; not the same as PHercParis4.',
     'legacy_InkID': 'Publication explicitly used both Paris1 fragments in four-surface training/cross-validation; no verified initializer edge to current seed42.',
     'annotation_derivation': 'Published IR tracing route and legacy supplied fragment labels; residual/negative completeness qualifications retained.',
     'own_lab_use': 'Frag4 diagnostics since phase11; Frag3 calibration then repeated development through phase21.',
     'eligible_now': 'Reused development; no fresh-parent confirmation.'},
    {'parent_group': 'Paris2', 'aliases': ['Frag1/PHercParis2Fr47', 'Frag2/PHercParis2Fr143'],
     'four_original_models_direct_roots': 'Absent; parent is distinct from Paris1.',
     'legacy_InkID': 'Publication explicitly used both Paris2 fragments in four-surface training/cross-validation; no verified initializer edge to current seed42.',
     'annotation_derivation': 'Published IR tracing route and legacy supplied fragment labels; residual/negative completeness qualifications retained.',
     'own_lab_use': 'Initially reserved evaluation, repeatedly scored and now explicitly development by phase21; never lab gradients/teacher targets/cutoff fitting.',
     'eligible_now': 'One reused parent, not two independent parent replications.'},
]
for parent, usage in [('PHerc0139', 'Lab adaptation and retired-validation support.'),
                      ('PHerc0814', 'Repeated development and cutoff calibration.'),
                      ('PHercParis4', 'Direct upstream roots; possible local-new crop does not erase parent exposure.')]:
    rows.append({'parent_group': parent, 'four_original_models_direct_roots': 'Present.',
                 'native_model_direct_roots': 'Present.', 'annotation_derivation': 'Transferred, model-assisted lineage incomplete.',
                 'own_lab_use': usage, 'eligible_now': 'No new-parent or independently authenticated truth claim.'})
write('ELIGIBILITY_MATRIX.json', {
    'status': 'retained_metadata_inventory_no_new_scientific_access',
    'authorization': str(R / 'twentysecond-experiment/SCOPE.md'),
    'scientific_pixels_read': 0, 'new_network_requests': 0, 'model_calls': 0, 'paid_spend_gbp': 0,
    'candidates': rows,
    'global_limits': ['Unknown is not independent.', 'Compatible samples and deployed predictions are not training membership.',
                      'Annotation prediction use is distinct from model parameter initialization.',
                      'Fresh local crop does not create a fresh parent; all scans/aliases stay grouped.',
                      'No concrete independent accuracy reserve is established by these retained records.'],
})

queries = []
for seg, record in extracts['PHerc0841']['labelled_segments_and_colocated_predictions_metadata'].items():
    entry = next(x for x in record['data'] if x['type'] == 'ink-labels')
    prefix = entry['origins'][0]['path']
    root = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/'
    queries += [
        {'id': f'0841-{seg}-package-inventory', 'url': root + '?' + urlencode({'list-type': '2', 'prefix': prefix, 'max-keys': '100'}),
         'max_body_bytes': 65536, 'purpose': 'Check actual object inventory for metadata/provenance/mapping-validity assets; reject IsTruncated=true, no pagination.'},
        {'id': f'0841-{seg}-ink-root', 'url': root + prefix + 'inklabels.zarr/zarr.json',
         'max_body_bytes': 32768, 'purpose': 'Check declared root transform and provenance links without array reads.'},
    ]
write('METADATA_QUERY_PROPOSAL.json', {
    'status': 'proposal_not_executed_no_network_authorized_here',
    'purpose': 'Optional exact minimum package-completeness discovery for the newly identified PHerc0841 parent. Does not establish independent truth or authorize scoring.',
    'already_answered_do_not_blindly_repeat': [
        'PHerc0009B/PHerc0500P2 complete inventories and mapping-validity/provenance 404s are retained in phase19.',
        'Canonical R152 pinned card/config/metadata were already audited; no public training manifest was listed.',
        'Frag6 package logs and sparse geometry do not supply renderer history; repeating them cannot authenticate it.',
    ],
    'queries': queries, 'max_attempts': 6, 'max_total_observed_body_bytes': sum(q['max_body_bytes'] for q in queries),
    'per_request_absolute_seconds': 30, 'external_wall_seconds': 210,
    'transport': 'Verified HTTPS; exact listed S3 hostname; no redirects, retries, credentials, pagination or payload fallback. Count/retain every failure; stop on cap or unexpected content. Only bounded XML/JSON text bodies.',
    'required_primary_evidence_without_known_public_URL': [
        {'gap': 'Seed42 original annotation production and upstream teachers', 'needed': 'Versioned per-label teacher/checkpoint/iteration and support manifest; direct model config cannot prove this.'},
        {'gap': 'Canonical R152 training/initialization', 'needed': 'Artifact-bound training specimen/alias/support and initializer manifest for advertised 36dd0de84b7b7aa6590184192c7415466cd8a1ba7c1e59f42c6373846373c3e0; no full checkpoint acquisition proposed.'},
        {'gap': 'Modern 0009B/0500P2/0841 publication', 'needed': 'Exact dirty publisher source/commands, source and target hashes, offset and post-transfer corrections, actual mapping-validity location and annotation authorship/negative support.'},
        {'gap': 'Frag6 historical render and labels', 'needed': 'Native UUID/frame, raster-origin and depth offsets, normal/interpolation/quantization recipe plus optical alignment and annotation provenance; no authoritative public URL established.'},
    ],
    'unknown_endpoints_policy': 'Do not guess private bucket paths or contact anyone. Any newly discovered public text URL needs a new bounded proposal and root authorization.',
})

# Verify every retained source remains byte-identical after all derivation.
for path, source in sources.items():
    data = Path(path).read_bytes()
    assert len(data) == source['byte_length'] and hashlib.sha256(data).hexdigest() == source['sha256']
write('SOURCE_PINS.json', {'sources': sorted(sources.values(), key=lambda x: x['path']),
                          'verified_before_after': True, 'only_text_metadata_opened': True})
print(json.dumps({'status': 'PASS', 'modern_label_entries': len(modern_label_inventory),
                  'modern_label_parents': 7, 'matrix_parent_groups': len(rows),
                  'retained_sources_pinned': len(sources), 'new_requests': 0,
                  'query_proposal_max_bytes': sum(q['max_body_bytes'] for q in queries)}, indent=2))
