"""Bind the completed metadata audit; no images, arrays or model calls."""
from pathlib import Path
import gzip
import hashlib
import json

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / 'source-audit'
LAB = ROOT.parent.parent


def record(path):
    path = Path(path)
    body = path.read_bytes()
    return dict(path=str(path), size_bytes=len(body), sha256=hashlib.sha256(body).hexdigest())


catalog = LAB / 'data/experiments/pherc0800-native-ct-consistency/metadata.full.json.gz'
data = json.loads(gzip.decompress(catalog.read_bytes()))
candidates = []
for number, identity, parent, role in [
    (1, 'PHercParis2Fr47', 'Paris2', 'evaluation'),
    (2, 'PHercParis2Fr143', 'Paris2', 'evaluation'),
    (3, 'PHercParis1Fr34', 'Paris1', 'calibration'),
]:
    meta_path = AUDIT / f'frag{number}-surface-meta.json'
    meta = json.loads(meta_path.read_text())
    base = data['samples'][identity]['sample']['properties']['legacy_data_url']
    assert meta['slices'] == 65 and meta['voxelsize'] == 3.24 and '54KeV' in meta['name']
    exposed = base + 'working/54keV_exposed_surface/'
    candidates.append(dict(
        id=identity, legacy_alias=f'Frag{number}', parent_papyrus=parent, role=role,
        eligibility='Qualified supplied-reference candidate; pending label-only quality/selection gates',
        base_url=base, exposed_surface_url=exposed,
        labels_url=exposed+'inklabels.png', mask_url=exposed+'mask.png',
        surface_volume_url=exposed+'surface_volume/', surface_metadata=record(meta_path),
        shape_yx=[meta['height'], meta['width']], slice_count=65,
        nominal_native_spacing_um=3.24, declared_energy_keV=54, surface_uuid=meta['uuid'],
        depth_convention=dict(surface_plane_zero_based=32, normal_step_native_voxels=1,
            evidence='Published EduceLab-Scrolls A.6 protocol plus matching current vc_layers metadata; historical executable/argv not reproduced'),
        reference_contract='Published manually registered 1000nm optical images and manually traced ink; actual encoding/canvas checks pending',
        registration_residual='unmeasured',
        local_preexisting_scientific_exposure='No Frag1/2/3 use found in retained first-fourteenth records; labels not yet acquired in this phase',
        upstream_exposure='Absent from examined 29 explicit roots; complete teacher/pretraining exposure unknown',
        ct_layout_and_range_support='Not probed; future header/IFD/range validation required before CT reads',
    ))

deferred = []
for number, identity, reason in [
    (5, 'PHerc1667Cr1Fr3', 'Shares PHerc1667 already used in upstream/local development; current labels/IR/mask and 65-TIFF listing exist, but inspected evidence does not establish this release annotation-depth/optical provenance contract.'),
    (6, 'PHerc51Cr4Fr8', 'Additional parent group and current labels/IR/mask and 65-TIFF listing exist; inspected evidence does not establish annotation provenance or exact render depth/spacing. 65 filenames alone do not establish the depth convention.'),
]:
    deferred.append(dict(id=identity, legacy_alias=f'Frag{number}', role='deferred', reason=reason,
        claim='Not declared absent or universally unusable; outside this benchmark until the specific source gaps close.'))

ledger = json.loads((AUDIT/'request-ledger.json').read_text())
assert len(ledger['attempts']) == 42
assert all(row['http_status'] == 200 and row['status'] == 'complete' for row in ledger['attempts'])
license_record = record(AUDIT/'legacy-LICENSE.current.txt')
assert license_record['sha256'] == '9dc08a92194617d4ae68fb96d4fd0da99d1c7207f821f2a3750b0034eac8d154'
paper = LAB/'reports/eleventh-experiment/optical-reference/annotation-provenance/educelab-arxiv-v4-complete.html'
contract = dict(
    schema_version=1, catalog=record(catalog), paper=record(paper),
    published_excerpts=record(paper.parent/'paragraph-excerpts.json'),
    historical_loader=record(paper.parent/'inkid-dataset.py'),
    existing_model_exposure=record(LAB/'reports/seventh-experiment/model-exposure-ledger.json'),
    license=license_record, eligible_candidates=candidates, deferred_candidates=deferred,
    existing_seen_diagnostic=dict(id='PHercParis1Fr39', alias='Frag4', parent_papyrus='Paris1',
        role='Seen diagnostic only; excluded from calibration/training across aliases'),
    independence='Three new physical fragments, two parent papyri; evaluation fragments share one evaluation parent, with a separate calibration parent. No independent-parent statistical claim.',
    metadata_requests=dict(attempts=42, response_body_bytes=ledger['response_body_bytes'],
        all_http_200_complete=True, retries=0, redirects=0, images_or_array_payloads=0,
        ledger=record(AUDIT/'request-ledger.json')),
    model_calls=0, network_source_files=[record(p) for p in sorted(AUDIT.iterdir()) if p.is_file()],
)
destination = ROOT/'SOURCE_CONTRACT.json'
assert not destination.exists()
destination.write_text(json.dumps(contract, indent=2, sort_keys=True)+'\n')
print(json.dumps(record(destination)))
