"""Source-informed algebraic explanation only; no execution of upstream code.

Behavior inspected in educelab/volume-cartographer v2.26.0, Volume.cpp lines174–187.
The upstream source and GPLv3 license are retained in renderer-source-v1.
"""
import hashlib,json,math,resource,time
from pathlib import Path
import numpy as np
import numcodecs

HERE=Path(__file__).resolve().parent


def pin(p):
    b=p.read_bytes();return dict(path=str(p.resolve()),size_bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def main():
    freeze=json.loads((HERE/'HISTORICAL_REPLAY_GO.json').read_text())
    for r in freeze['files']:assert pin(Path(r['path']))==r
    # Asymmetric cell: only I100=100 and I110=300 nonzero; fractions(.25,.5,.75).
    exact=.25*.5*.25*100+.25*.5*.25*300
    historical=exact+.25*.5*.25*(100-300)
    assert exact==12.5 and historical==6.25 and int(np.rint(historical))==6
    start=time.monotonic()
    root=HERE/'run-v1';manifest=json.loads((root/'SOURCE_MANIFEST.json').read_text())
    points=np.load(HERE/'POINTS.npy',allow_pickle=False)[:189]
    base=np.load(root/'NATIVE_VALUES.npy',allow_pickle=False)[:189]
    truth=np.load(root/'PUBLISHED_VALUES.npy',allow_pickle=False).reshape(-1)
    correction=np.zeros(189);groups={}
    for i,q in enumerate(points):
        lo=np.floor(q).astype(int);f=q-lo;w=float(f[0]*f[1]*(1-f[2]))
        if not w:continue
        for sign,corner in ((1,(lo[2],lo[1],lo[0]+1)),(-1,(lo[2],lo[1]+1,lo[0]+1))):
            key=tuple(int(v)//128 for v in corner);local=tuple(int(v)%128 for v in corner)
            groups.setdefault(key,[]).append((i,local,sign*w))
    assert len(groups)<=14
    for key,entries in sorted(groups.items()):
        name='0/'+'/'.join(map(str,key));p=root/'source'/name;b=p.read_bytes();r=manifest['objects'][name]
        assert len(b)==r['size_bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
        raw=numcodecs.Blosc(cname='zstd').decode(b);assert len(raw)==4194304
        a=np.frombuffer(raw,dtype='<u2').reshape(128,128,128)
        for i,local,w in entries:correction[i]+=w*int(a[local])
        del a,raw,b
    unrounded=base+correction
    replay=np.rint(unrounded);assert ((replay>=0)&(replay<=65535)).all()
    residual=replay-truth
    out=HERE/'historical-replay-v1';out.mkdir(exist_ok=False)
    np.save(out/'REPLAY_VALUES.npy',replay.reshape(7,3,9).astype('<u2'),allow_pickle=False)
    np.save(out/'RESIDUALS.npy',residual.reshape(7,3,9),allow_pickle=False)
    result=dict(status='EXPLANATORY_REPLAY_COMPLETE',post_hoc=True,fresh_confirmation=False,
        original_mapping_verdict_unchanged=True,pairs=189,exact_matches=int((residual==0).sum()),
        mean_abs_error=float(abs(residual).mean()),maximum_abs_error=float(abs(residual).max()),
        minimum_distance_to_half_integer=float(np.min(abs((unrounded-np.floor(unrounded))-.5))),
        decoded_cached_chunks=len(groups),decoded_source_work_bytes=len(groups)*4194304,network_requests=0,
        asymmetric_phantom=dict(correct_value=12.5,historical_unrounded=6.25,historical_rounded=6),
        elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        outputs=[pin(out/'REPLAY_VALUES.npy'),pin(out/'RESIDUALS.npy')])
    with (out/'REPLAY_RESULT.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':main()
