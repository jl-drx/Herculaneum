"""Reconstruct source values independently: scalar PPM, byte-addressed voxels, nested lerps."""
import math, resource, struct, time
import numpy as np
import numcodecs
from packet_common import *

started=time.monotonic()
require(not (HERE/'NATIVE_VERIFICATION.json').exists(),'Write-once verifier')
go=json.loads((HERE/'NATIVE_GO.json').read_text())
for p in go['files']: verify(p)
plan=json.loads((HERE/'NATIVE_PLAN.json').read_text())
complete=json.loads((HERE/'native-v1/COMPLETE.json').read_text())
for key in ['GO','arrays','manifest','ledger']: verify(complete[key])
arrays=np.load(complete['arrays']['path'],allow_pickle=False)
manifest=json.loads(Path(complete['manifest']['path']).read_text())
ledger=json.loads(Path(complete['ledger']['path']).read_text())
require(len(ledger['requests'])==43 and ledger['redirects']==ledger['retries']==0 and ledger['verified_TLS'],'Transport outcome')
require(all(r['success'] and r['status']==200 for r in ledger['requests']),'All exact objects received')
require([r['url'] for r in ledger['requests']]==plan['exact_urls'],'Frozen URL order')
require(sum(r['returned_body_bytes'] for r in ledger['requests'])==complete['returned_body_bytes']==ledger['returned_body_bytes'],'Body accounting')
require(sum(r['reserved_body_bytes'] for r in ledger['requests'])==complete['reserved_body_bytes']==ledger['reserved_body_bytes'],'Reservation accounting')
for r in ledger['requests']: verify(r['file'])
source=HERE/'native-v1/source'
for name,p in manifest['metadata'].items():
    require(pin(source/name)==dict(path=str(source/name),**p),'Metadata checksum')
meta=json.loads((source/'0/.zarray').read_text())
require(meta['shape']==[8855,2260,6300] and meta['chunks']==[128,128,128] and meta['dtype']=='<u2' and meta['order']=='C','Byte-address assumptions')
require(meta['compressor']==dict(id='blosc',cname='zstd',clevel=3,shuffle=0,blocksize=0) and meta['filters'] is None,'Codec assumptions')
gplan=json.loads((HERE/'GEOMETRY_PLAN.json').read_text())
records=list(struct.iter_unpack('<6d',(HERE/'geometry-v1/records.bin').read_bytes()))
points=[]; descriptors=[]; groups={}
for i,site in enumerate(gplan['sites']):
    record=records[9*i+4]
    for layer in (range(1,64) if site['local_yx']==[193,193] else [32]):
        xyz=tuple(record[k]+(layer-32)*record[k+3] for k in range(3))
        descriptors.append(dict(context=site['context'],local_yx=site['local_yx'],global_yx=site['global_yx'],layer=layer,XYZ=list(xyz)))
        points.append(xyz)
require(descriptors==plan['points'] and np.array_equal(points,arrays['points_XYZ']),'Direct PPM point reconstruction')
corners=[[None]*8 for _ in points]; fractions=[]
for i,xyz in enumerate(points):
    lo=[math.floor(v) for v in xyz]; fractions.append([xyz[k]-lo[k] for k in range(3)])
    for dz in [0,1]:
        for dy in [0,1]:
            for dx in [0,1]:
                x,y,z=lo[0]+dx,lo[1]+dy,lo[2]+dz
                name=f'0/{z//128}/{y//128}/{x//128}'
                offset=2*((z%128)*128*128+(y%128)*128+x%128)
                groups.setdefault(name,[]).append((i,4*dz+2*dy+dx,offset))
require(set(groups)==set(manifest['objects']) and len(groups)==40,'Exact second-decode support')
codec=numcodecs.Blosc(cname='zstd',clevel=3,shuffle=0,blocksize=0)
for name,entries in groups.items():
    require(pin(source/name)==dict(path=str(source/name),**manifest['objects'][name]),'Encoded object checksum')
    body=(source/name).read_bytes()
    version,codecversion,flags,typesize,nbytes,blocksize,cbytes=struct.unpack('<BBBBIII',body[:16])
    require((version,codecversion,typesize,nbytes,cbytes)==(2,1,2,4194304,len(body)),'Blosc sizes')
    require(not flags&5 and 0<blocksize<=nbytes and blocksize%2==0 and numcodecs.blosc.cbuffer_complib(body)=='Zstd','Blosc codec')
    if flags&2: require(len(body)==nbytes+16,'Memcpy bound')
    decoded=bytearray(4194304); codec.decode(body,out=decoded)
    for i,j,offset in entries: corners[i][j]=struct.unpack_from('<H',decoded,offset)[0]
    del body,decoded
require(np.array_equal(corners,arrays['corners']) and np.array_equal(fractions,arrays['fractions']),'Every integer corner and fraction')
raw={c['id']:np.load(verify(c['raw']),mmap_mode='r',allow_pickle=False) for c in plan['contexts']}
targets=[]; historical=[]; standard=[]
for p,cs,fs in zip(descriptors,corners,fractions):
    fx,fy,fz=fs
    c00=cs[0]*(1-fx)+cs[1]*fx
    c10=cs[2]*(1-fx)+cs[3]*fx
    c10old=cs[2]*(1-fx)+cs[1]*fx
    c01=cs[4]*(1-fx)+cs[5]*fx
    c11=cs[6]*(1-fx)+cs[7]*fx
    upper=c01*(1-fy)+c11*fy
    standard.append((c00*(1-fy)+c10*fy)*(1-fz)+upper*fz)
    historical.append((c00*(1-fy)+c10old*fy)*(1-fz)+upper*fz)
    y,x=p['local_yx']; targets.append(int(raw[p['context']][p['layer']-1,y,x]))
maxdiff={}
for name,values in [('standard',standard),('historical',historical)]:
    maxdiff[name]=float(np.max(abs(np.array(values)-arrays[name])))
    require(maxdiff[name]<1e-9,'Independent nested interpolation')
    require(np.array_equal([round(v) for v in values],arrays['rounded_'+name]),'Python nearest-even rounding')
require(np.array_equal(targets,arrays['target']),'All published targets')
residual=[round(v)-t for v,t in zip(historical,targets)]
require(np.array_equal(residual,arrays['residual']) and not any(residual),'Every historical value matches exactly')
for c in complete['contexts']:
    indices=[i for i,p in enumerate(descriptors) if p['context']==c['id']]
    errors=[abs(round(standard[i])-targets[i]) for i in indices]
    require(len(indices)==c['pairs']==c['historical_exact_matches']==71,'All 71 regional pairs')
    require(errors.count(0)==c['standard_exact_matches'] and max(errors)==c['standard_max_error'] and sum(errors)/len(errors)==c['standard_MAE'],'Reference-only standard results')
rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
require(time.monotonic()-started<120 and rss<512*1024**2,'Verifier resource bounds')
receipt=dict(status='PASS',created_at_utc=now(),verifier=pin(__file__),worker_complete=pin(HERE/'native-v1/COMPLETE.json'),
    scalar_PPM_points=284,integer_corners_checked=2272,second_chunk_decodes=40,decoded_bytes=167772160,
    published_targets_checked=284,historical_exact_matches=284,maximum_unrounded_difference=maxdiff,
    no_independent_ink_or_optical_ground_truth=True,new_source_requests=0,new_model_forwards=0,
    elapsed_seconds=time.monotonic()-started,peak_RSS_bytes=rss)
save(HERE/'NATIVE_VERIFICATION.json',receipt);print(json.dumps(receipt,indent=2))
