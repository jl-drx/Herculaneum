"""Bounded native chunks, one decode each, fixed historical/source comparison."""
import resource,sys,time
import numpy as np
from packet_common import *
from native_math import CORNERS,evaluate
sys.path.insert(0,str(REPORTS/'thirtieth-experiment'))
from run_correspondence import Acquisition
from local_zarr_reader import LocalReader

class RegionalAcquisition(Acquisition):
    def resources(self):
        lim=self.plan['limits']
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=lim['max_rss_bytes'],'Regional RSS cap')
        require(sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<=lim['max_phase_bytes'],'Regional disk cap')
        require(time.monotonic()-self.start<=lim['total_seconds'],'Regional acquisition deadline')
    def get(self,url,*args,**kwargs):
        require(url in self.plan['exact_urls'],'Only frozen exact native URLs');return super().get(url,*args,**kwargs)

def main():
    started=time.monotonic();go=json.loads((HERE/'NATIVE_GO.json').read_text())
    for p in go['files']:verify(p)
    plan=json.loads((HERE/'NATIVE_PLAN.json').read_text());out=HERE/'native-v1';out.mkdir(exist_ok=False);source=out/'source';source.mkdir()
    acq=RegionalAcquisition(out,plan);manifest=dict(metadata={},objects={})
    try:
        for m in plan['metadata']:
            dest=source/m['name'];acq.get(m['url'],dest,cap=65536,etag=m['etag'],expected_body=m['body'])
            manifest['metadata'][m['name']]={k:pin(dest)[k] for k in ['size_bytes','sha256']}
        for key in plan['keys_zyx']:
            name='0/'+'/'.join(map(str,key));dest=source/name;acq.get(plan['native_base_url']+name,dest,cap=4194320)
            manifest['objects'][name]={k:pin(dest)[k] for k in ['size_bytes','sha256']}
        save(out/'SOURCE_MANIFEST.json',manifest);reader=LocalReader(source,manifest)
        points=np.array([p['XYZ'] for p in plan['points']],np.float64);floor=np.floor(points).astype(np.int64);fraction=points-floor
        groups={};corners=np.zeros((len(points),8),np.float64);seen=np.zeros(corners.shape,bool)
        for i,lo in enumerate(floor):
            for j,bits in enumerate(CORNERS):
                xyz=lo+bits;key=tuple(int(xyz[k])//128 for k in [2,1,0]);local=tuple(int(xyz[k])%128 for k in [2,1,0]);groups.setdefault(key,[]).append((i,j,local))
        require(sorted(groups)==[tuple(k) for k in plan['keys_zyx']],'Exact decode support')
        for key,entries in sorted(groups.items()):
            chunk=reader(key);require(chunk is not None,'Missing source chunk')
            for i,j,local in entries:require(not seen[i,j],'Duplicate corner');corners[i,j]=int(chunk[local]);seen[i,j]=True
            del chunk
        require(seen.all() and reader.stats['decoded']==40,'Complete one-pass corner gather')
        predicted=evaluate(corners,fraction)
        raw={c['id']:np.load(verify(c['raw']),mmap_mode='r',allow_pickle=False) for c in plan['contexts']}
        target=np.array([int(raw[p['context']][p['layer']-1,*p['local_yx']]) for p in plan['points']],np.uint16)
        residual=predicted['rounded_historical'].astype(np.int64)-target.astype(np.int64)
        path=out/'COMPARISON_ARRAYS.npz';np.savez(path,points_XYZ=points,corners=corners,fractions=fraction,target=target,residual=residual,**predicted)
        per_context=[]
        for c in plan['contexts']:
            mask=np.array([p['context']==c['id'] for p in plan['points']]);d=residual[mask];sd=predicted['rounded_standard'][mask].astype(np.int64)-target[mask]
            per_context.append(dict(id=c['id'],pairs=int(mask.sum()),historical_exact_matches=int((d==0).sum()),historical_MAE=float(abs(d).mean()),historical_max_error=int(abs(d).max()),
                standard_exact_matches=int((sd==0).sum()),standard_MAE=float(abs(sd).mean()),standard_max_error=int(abs(sd).max())))
        for p in go['files']:verify(p)
        acq.resources()
        receipt=dict(status='COMPLETED',scientific_correspondence='PASS_ALL_EXACT' if np.all(residual==0) else 'FAIL_EXACT_MATCH_CRITERION',
            created_at_utc=now(),GO=pin(HERE/'NATIVE_GO.json'),arrays=pin(path),manifest=pin(out/'SOURCE_MANIFEST.json'),ledger=pin(out/'REQUEST_LEDGER.json'),
            pairs=len(target),historical_exact_matches=int((residual==0).sum()),contexts=per_context,
            minimum_distance_to_half_integer=float(np.min(abs((predicted['historical']-np.floor(predicted['historical']))-.5))),
            reader_stats=reader.stats,requests=len(acq.ledger['requests']),reserved_body_bytes=acq.ledger['reserved_body_bytes'],returned_body_bytes=acq.ledger['returned_body_bytes'],
            elapsed_seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,new_model_forwards=0,GPU_spend_GBP=0)
        save(out/'COMPLETE.json',receipt);print(json.dumps(receipt,indent=2))
    except Exception as e:
        save(out/'FAILURE.json',dict(status='STOP',error=type(e).__name__+': '+str(e),ledger=pin(out/'REQUEST_LEDGER.json')));raise
if __name__=='__main__':main()
