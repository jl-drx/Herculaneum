"""Root-owned scalar verification from retained native bytes; no acquisition/model calls."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import resource
import time
import numpy as np
import numcodecs

HERE=Path(__file__).resolve().parent
RUN=HERE/'run-v1'


def pin(path):
    with path.open('rb') as stream:h=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=str(path.resolve()),size_bytes=path.stat().st_size,sha256=h)


def main():
    started=time.monotonic()
    freeze=json.loads((HERE/'SOURCE_FREEZE.json').read_text())
    for r in freeze['files']:assert pin(Path(r['path']))==r
    plan=json.loads((HERE/'SAMPLING_PLAN.json').read_text())
    receipt=json.loads((RUN/'RUN_RECEIPT.json').read_text())
    for key in ('source_freeze','sampling_plan','source_manifest','correspondence','native_values','published_values','ledger'):
        r=receipt[key];assert pin(Path(r['path']))==r
    ledger=json.loads((RUN/'REQUEST_LEDGER.json').read_text())
    assert len(ledger['requests'])==80
    assert sum(r['status']==200 for r in ledger['requests'])==17
    assert sum(r['status']==206 for r in ledger['requests'])==63
    for r in ledger['requests']:
        assert r['success'] and pin(Path(r['file']['path']))==r['file']
    assert sum(r['returned_body_bytes'] for r in ledger['requests'])==ledger['returned_body_bytes']==56276149
    assert sum(r['reserved_body_bytes'] for r in ledger['requests'])==ledger['reserved_body_bytes']==58917546
    raw=json.loads(Path(plan['geometry']['path']).read_text())
    sites=[s for s in raw['geometry']['samples'] if s['potential_native_site'] and s['complete_mapped_neighborhood']]
    # Independently recreate point order one coordinate at a time.
    expected=[]
    for shift in ((0,0,0),(17,13,11)):
        for mode in ('linear','floor','round'):
            for site in sites:
                for offset in (-32,0,32):
                    for r in site['records_row_major']:
                        q=[r['xyz'][d]+offset*r['normal'][d]+shift[d] for d in range(3)]
                        if mode=='floor':q=[math.floor(v) for v in q]
                        if mode=='round':q=[math.floor(v+.5) for v in q]
                        expected.append(q)
    points=np.load(HERE/'POINTS.npy',allow_pickle=False)
    np.testing.assert_array_equal(points,np.array(expected))
    # Independent scalar corner enumeration. A chunk is decoded once and released.
    contributions={};weight_sum=[0.]*len(expected)
    for i,q in enumerate(expected):
        low=[math.floor(v) for v in q];frac=[q[j]-low[j] for j in range(3)]
        for delta in itertools.product((0,1),repeat=3):
            w=math.prod(frac[j] if delta[j] else 1-frac[j] for j in range(3))
            if not w:continue
            xyz=[low[j]+delta[j] for j in range(3)]
            zyx=tuple(reversed(xyz));key=tuple(v//128 for v in zyx);local=tuple(v%128 for v in zyx)
            contributions.setdefault(key,[]).append((i,local,w));weight_sum[i]+=w
    assert sorted(contributions)==[tuple(k) for k in plan['native_plan']['keys_zyx']]
    assert max(abs(v-1) for v in weight_sum)<1e-12
    scalar=[0.]*len(expected)
    for key,entries in sorted(contributions.items()):
        file=RUN/'source'/'0'/('/'.join(map(str,key)))
        payload=file.read_bytes()
        decoded=numcodecs.Blosc(cname='zstd').decode(payload)
        assert len(decoded)==4194304
        chunk=np.frombuffer(decoded,dtype='<u2').reshape(128,128,128)
        for i,local,w in entries:scalar[i]+=int(chunk[local])*w
        del chunk,decoded,payload
    native=np.load(RUN/'NATIVE_VALUES.npy',allow_pickle=False)
    error=float(np.max(np.abs(native-scalar)))
    assert error<=1e-10
    # Rebuild the TIFF values from their individually pinned ranges and scalar byte offsets.
    published=np.load(RUN/'PUBLISHED_VALUES.npy',allow_pickle=False)
    for li,layer in enumerate(plan['layers']):
        for si,site in enumerate(sites):
            cy,cx=site['center_yx']
            values=[]
            for row in range(3):
                idx=si*3+row;start=8+2*((cy-1+row)*6205+cx-1)
                assert layer['ranges'][idx]==[start,start+5]
                b=(RUN/'tiff-ranges'/f"{layer['layer']:02d}-{idx:02d}.bin").read_bytes()
                values.extend(int.from_bytes(b[j:j+2],'little') for j in (0,2,4))
            np.testing.assert_array_equal(published[si,li],values)
    blocks=np.array(scalar).reshape(6,7,3,9)
    decision=json.loads((RUN/'CORRESPONDENCE.json').read_text())
    errors=[]
    for kernel in range(3):
        for sign in (1,-1):
            e=np.abs(blocks[kernel,:,::sign,:]-published)
            errors.append(e)
    calibration=[sum(e[:3].ravel())/81 for e in errors]
    selected=min(range(6),key=lambda i:calibration[i])
    assert selected==decision['selected_index']==0
    selected_errors=errors[selected]
    assert abs(sum(selected_errors[3:].ravel())/108-decision['selected_check_mae'])<1e-10
    assert abs(float(selected_errors[3:].max())-decision['selected_check_max_error'])<1e-10
    assert decision['status']=='MAPPING_NOT_QUALIFIED'
    assert decision['criteria']==dict(calibration_mae=False,calibration_max_error=False,
        every_check_site_mae=False,every_check_site_max_error=False,identified_from_calibration=True,
        informative_check_sites=True,wrong_coordinate_absolute=True,wrong_coordinate_relative=True)
    return dict(status='PASS',scope='Retained byte identities, scalar point/range reconstruction and independent scalar interpolation',
        network_requests=0,model_forwards=0,independent_agent_review=False,native_chunks_decoded=14,
        additional_decoded_voxels=14*128**3,scalar_sample_count=1134,max_sample_difference=error,
        selected_recipe='trilinear:+1',mapping_qualified=False,
        selected_check_mae_by_layer=selected_errors[3:].mean(axis=(0,2)).tolist(),
        selected_check_p95_abs_error=float(np.percentile(selected_errors[3:],95)),
        selected_check_pairs_within_two_counts=int((selected_errors[3:]<=2).sum()),
        elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)


if __name__=='__main__':
    result=main();out=HERE/'SCALAR_VERIFICATION.json'
    with out.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(result,indent=2,sort_keys=True))
