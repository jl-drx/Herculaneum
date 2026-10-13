"""Replay from frozen chunk bytes and geometric query fields. Never downloads."""
import argparse,datetime,hashlib,json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d,map_coordinates
from structural_scores import landscapes,summarise,fixtures
B=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def assemble(plan,site,kind,led):
    x0,y0,x1,y1=site['source_xy_half_open'][kind]
    meta=plan['volumes'][kind]['array'];nz=meta['shape'][0];shape=tuple(meta['chunks'])
    data=np.full((nz,y1-y0,x1-x0),np.nan,dtype=np.float64);covered=np.zeros(data.shape[1:],bool)
    for r in plan['requests']:
        if r['site']!=site['site'] or r['acquisition']!=kind:continue
        p=B/'chunks'/r['name'];record=led[r['name']]
        assert record['status']==200 and record['bytes']==r['exact_bytes'] and p.stat().st_size==r['exact_bytes']
        assert sha(p)==record['sha256'],'Chunk hash mismatch'
        a=np.fromfile(p,dtype=np.uint8).reshape(shape)
        _,yy,xx=r['chunk_zyx'];ox=xx*shape[2];oy=yy*shape[1]
        xa,xb=max(x0,ox),min(x1,ox+shape[2]);ya,yb=max(y0,oy),min(y1,oy+shape[1])
        data[:,ya-y0:yb-y0,xa-x0:xb-x0]=a[:,ya-oy:yb-oy,xa-ox:xb-ox]
        covered[ya-y0:yb-y0,xa-x0:xb-x0]=True
    assert covered.all() and np.isfinite(data).all(),'Missing source support'
    return data

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=B/'results');args=ap.parse_args();out=args.output
    assert not out.exists(),'Refusing overwrite'
    plan=json.loads((B/'REQUEST_PLAN.json').read_text());ledger=json.loads((B/'REQUEST_LEDGER.json').read_text())
    assert ledger['complete'] and ledger['plan_sha256']==sha(B/'REQUEST_PLAN.json')
    for f in plan['frozen_files']:assert sha(B/f['path'])==f['sha256']
    assert fixtures()['status']=='PASS'
    records={r['name']:r for r in ledger['requests']};assert len(records)==len(plan['requests'])
    geo=B.parent/'geometry-review/pullback';g=json.loads((geo/'RESULTS.json').read_text());bases=json.loads((geo/'LOCAL_BASES.json').read_text())
    out.mkdir(parents=True);summaries=[]
    for site,gr,bas in zip(plan['sites'],g['sites'],bases['sites'],strict=True):
        sid=site['site'];assert sid==gr['site']==bas['site'];sd=out/sid;sd.mkdir()
        query=geo/sid/'queries_uvk.npy';assert sha(query)==site['queries_sha256'];q=np.load(query)
        valid=geo/sid/'support.npy';assert sha(valid)==site['support_sha256'];assert np.load(valid).all()
        f=assemble(plan,site,'fine',records);n=assemble(plan,site,'native',records)
        filtered=gaussian_filter1d(f,sigma=1.5,axis=0,radius=5,mode='constant',cval=np.nan)
        x0,y0,_,_=site['source_xy_half_open']['fine']
        coords=np.stack((q[...,2],q[...,1]-y0,q[...,0]-x0))
        lo=np.floor(coords.reshape(3,-1).min(1)).astype(int);hi=np.floor(coords.reshape(3,-1).max(1)).astype(int)+1
        assert (lo>=0).all() and (hi<np.array(f.shape)).all() and lo[0]>=5 and hi[0]<=103
        mapped=map_coordinates(filtered,coords,order=1,mode='constant',cval=np.nan,prefilter=False)
        assert np.isfinite(mapped).all()
        native=n[4:24];scores=None
        try:
            scores=landscapes(native,mapped)
            summary=summarise(scores,bas['core_basis_columns_depth_y_x_nativeXYZ'],[s['basis_columns_depth_y_x_nativeXYZ'] for s in bas['supports']])
        except ValueError as error:
            summary=dict(structural_screen_pass=False,score_failure=str(error))
        summary.update(site=sid,native_mean=float(native.mean()),fine_remapped_mean=float(mapped.mean()),native_std=float(native.std()),fine_remapped_std=float(mapped.std()),native_zero_fraction=float((native==0).mean()),fine_source_zero_fraction=float((f==0).mean()),all_source_and_query_support_complete=True,meaning='CT structural correspondence only; not ink truth, calibrated registration probability, or a whole-core/same-sheet certificate.')
        np.save(sd/'fine_remapped.npy',mapped);np.save(sd/'native_full.npy',n);np.save(sd/'fine_full.npy',f)
        if scores is not None:np.save(sd/'landscapes.npy',scores)
        (sd/'RESULTS.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');summaries.append(summary)
        print(sid,'fit',summary.get('fit_translation_zyx'),'baseline',summary.get('baseline_mean_fit_ncc'),'fitNCC',summary.get('fit_mean_ncc'),'screen',summary['structural_screen_pass'],flush=True)
    result=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=sha(B/'REQUEST_PLAN.json'),analysis_code_sha256=sha(Path(__file__)),sites=summaries)
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    files=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]
    (out/'OUTPUT_FILES.json').write_text(json.dumps(files,indent=2)+'\n')
if __name__=='__main__':main()
