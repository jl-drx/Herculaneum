"""Read-only reconciliation of all frozen phases within phase30 and fresh target values."""
import base64,hashlib,json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def pin(p):
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p.resolve()),size_bytes=p.stat().st_size,sha256=h)


def main():
    freezes=['SOURCE_FREEZE.json','SOFTWARE_TAG_GO.json','RENDERER_SOURCE_GO.json','HISTORICAL_REPLAY_GO.json','FRESH_CONFIRMATION_GO.json']
    for name in freezes:
        for r in json.loads((HERE/name).read_text())['files']:assert pin(Path(r['path']))==r
    ledger_names=['run-v1/REQUEST_LEDGER.json','software-tags-v1/REQUEST_LEDGER.json','renderer-source-v1/LEDGER.json','fresh-confirmation-v1/REQUEST_LEDGER.json']
    request_count=reserved=returned=0
    for name in ledger_names:
        d=json.loads((HERE/name).read_text());request_count+=len(d['requests']);reserved+=d['reserved_body_bytes'];returned+=d['returned_body_bytes']
        assert all(r['success'] for r in d['requests'])
        for r in d['requests']:
            for key in ('file','response','source'):
                if key in r:assert pin(Path(r[key]['path']))==r[key]
            if 'git_blob_sha1' in r:
                data=Path(r['source']['path']).read_bytes()
                assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==r['git_blob_sha1']
    assert (request_count,returned,reserved)==(100,56586037,69403600)
    original=json.loads((HERE/'run-v1/CORRESPONDENCE.json').read_text())
    assert original['status']=='MAPPING_NOT_QUALIFIED'
    observed=np.load(HERE/'run-v1/PUBLISHED_VALUES.npy',allow_pickle=False)
    replay=np.load(HERE/'historical-replay-v1/REPLAY_VALUES.npy',allow_pickle=False)
    assert observed.shape==replay.shape==(7,3,9) and np.array_equal(observed,replay)
    fresh=json.loads((HERE/'fresh-confirmation-v1/CONFIRMATION_RESULT.json').read_text())
    prediction_pin=json.loads((HERE/'FRESH_CONFIRMATION_GO.json').read_text())['files'][3]
    assert prediction_pin['path'].endswith('/FRESH_PREDICTIONS.json')
    assert prediction_pin['sha256']=='e42095d722466e5f6adfcde66c8b412d0aacd0ccbcfd3a025fb154e672acdad3'
    expected=json.loads((HERE/'FRESH_PREDICTIONS.json').read_text())['predictions']
    values=[int.from_bytes((HERE/f'fresh-confirmation-v1/pixel-{i}.bin').read_bytes(),'little') for i in range(3)]
    assert values==fresh['observed']==[p['historical_uint16'] for p in expected]==[31274,13069,44471]
    assert [abs(v-p['correct_trilinear_uint16']) for v,p in zip(values,expected)]==[282,120,5]
    assert fresh['status']=='FRESH_LAYER_MATCH'
    software=[(HERE/f'software-tags-v1/layer-{n:02d}.bin').read_bytes() for n in (0,32,64)]
    software.append((HERE/'fresh-confirmation-v1/software.bin').read_bytes())
    assert all(s==b'volume-cartographer 2.26.0\0' for s in software)
    ref=json.loads((HERE/'renderer-source-v1/version-ref.response.json').read_text())
    assert next(r for r in ref if r['ref']=='refs/tags/v2.26.0')['object']['sha']=='7640ba6c8e41f39a7aef63e9780cd13c296b2d15'
    return dict(status='PASS',network_requests=0,new_decodes=0,independent_agent_review=False,
        combined_acquisition_requests=request_count,returned_body_bytes=returned,reserved_body_bytes=reserved,
        primary_mapping_result_preserved='MAPPING_NOT_QUALIFIED',post_hoc_exact_pairs=189,
        prospectively_frozen_fresh_layer_pairs=3,all_fresh_exact=True,
        no_ink_accuracy_claim=True,scope='Saved-byte/source identity and result reconciliation; scalar decoder check retained separately')


if __name__=='__main__':print(json.dumps(main(),indent=2,sort_keys=True))
