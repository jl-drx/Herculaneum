"""Freeze fresh-layer CT values from cached data before any target-layer acquisition."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import numcodecs

HERE=Path(__file__).resolve().parent


def pin(p):
    b=p.read_bytes();return dict(path=str(p.resolve()),size_bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def main():
    plan=json.loads((HERE/'SAMPLING_PLAN.json').read_text())
    g=json.loads(Path(plan['geometry']['path']).read_text())
    sites=[s for s in g['geometry']['samples'] if s['potential_native_site'] and s['complete_mapped_neighborhood']][3:6]
    assert [s['center_yx'] for s in sites]==[[4426,4963],[7081,1240],[7081,3102]]
    manifest=json.loads((HERE/'run-v1/SOURCE_MANIFEST.json').read_text())
    points=[];requests={};corners=[{} for _ in sites]
    import itertools
    for i,s in enumerate(sites):
        r=s['records_row_major'][4];q=[r['xyz'][d]-16*r['normal'][d] for d in range(3)]
        points.append(q);lo=[math.floor(v) for v in q]
        for delta in itertools.product((0,1),repeat=3):
            xyz=[lo[d]+delta[d] for d in range(3)];zyx=xyz[::-1]
            key=tuple(v//128 for v in zyx);loc=tuple(v%128 for v in zyx)
            requests.setdefault(key,[]).append((i,delta,loc))
    assert len(requests)<=3
    source_pins=[]
    for key,items in sorted(requests.items()):
        name='0/'+'/'.join(map(str,key));r=manifest['objects'][name];p=HERE/'run-v1/source'/name
        b=p.read_bytes();assert len(b)==r['size_bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
        source_pins.append(pin(p));raw=numcodecs.Blosc(cname='zstd').decode(b)
        a=np.frombuffer(raw,dtype='<u2').reshape(128,128,128)
        for i,delta,loc in items:corners[i][delta]=int(a[loc])
        del a,raw,b
    predictions=[]
    for i,(s,q) in enumerate(zip(sites,points)):
        f=[v-math.floor(v) for v in q];c=corners[i]
        standard=sum(value*math.prod(f[d] if delta[d] else 1-f[d] for d in range(3)) for delta,value in c.items())
        historical=standard+f[0]*f[1]*(1-f[2])*(c[(1,0,0)]-c[(1,1,0)])
        y,x=s['center_yx'];offset=8+2*(y*6205+x)
        predictions.append(dict(raster_yx=[y,x],native_xyz=q,byte_range=[offset,offset+1],
            historical_unrounded=historical,historical_uint16=int(np.rint(historical)),
            correct_trilinear_unrounded=standard,correct_trilinear_uint16=int(np.rint(standard)),
            distance_to_half_integer=abs((historical-math.floor(historical))-.5)))
    assert min(p['distance_to_half_integer'] for p in predictions)>1e-8
    result=dict(status='PREDICTIONS_FROZEN_BEFORE_TARGET_READ',target_layer=16,predictions=predictions,
        source_chunks=source_pins,decoded_cached_chunks=len(requests),decoded_source_work_bytes=len(requests)*4194304,
        target_layer_bytes_read=0,source_algorithm='correct trilinear + dx*dy*(1-dz)*(I100-I110), rounded nearest-even',
        reference_manifest=pin(HERE/'run-v1/SOURCE_MANIFEST.json'))
    with (HERE/'FRESH_PREDICTIONS.json').open('x') as out:json.dump(result,out,indent=2,sort_keys=True);out.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
