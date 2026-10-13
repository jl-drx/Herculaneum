#!/usr/bin/env python3
"""Independent direct-window Pearson replay, no authored scoring imports/FFT."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import hashlib,json,itertools
import numpy as np
R=Path(__file__).resolve().parent;B=R.parents[1];D=B/'ct-study/results'
XY=((48,48),(208,48),(48,208),(208,208),(128,48),(48,128),(208,128),(128,208))

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def landscape(native,fine,cx,cy):
    # Explicittemplate coordinates; fine windowzero corresponds to shift-6.
    template=np.asarray(native[10:18,cy-8:cy+24,cx-8:cx+24],dtype=np.float64)
    assert template.shape==(8,32,32)
    ref=template.reshape(-1);ref=ref-np.mean(ref);den=np.sqrt(np.sum(ref*ref))
    assert den>0
    ans=np.empty((13,13,13))
    for zi,dz in enumerate(range(-6,7)):
        search=np.asarray(fine[6+dz:14+dz,cy-14:cy+30,cx-14:cx+30])
        assert search.shape==(8,44,44)
        # Keep depthwhole, enumerate all13x13lateral windows; no FFT orintegral image.
        patches=np.lib.stride_tricks.sliding_window_view(search,(32,32),axis=(1,2))
        values=np.moveaxis(patches,0,2).reshape(169,-1).copy()
        assert values.shape==(169,8192)
        values-=np.mean(values,axis=1,keepdims=True)
        denominators=np.sqrt(np.sum(values*values,axis=1))*den
        assert (denominators>0).all()
        ans[zi]=((values@ref)/denominators).reshape(13,13)
    return ans,{'native_template_mean':float(template.mean()),'native_template_std':float(template.std()),'native_zero_fraction':float(np.mean(template==0))}

def summarise(scores,core,bases):
    fitcube=np.mean(scores[:4],axis=0);idx=np.unravel_index(np.argmax(fitcube),fitcube.shape);fit=np.array(idx)-6
    rows=[]
    for i,(score,basis) in enumerate(zip(scores,bases)):
        bestidx=np.unravel_index(np.argmax(score),score.shape);best=np.array(bestidx)-6
        n=basis[:,0]/np.linalg.norm(basis[:,0]);candidates=[];normal_candidates=[]
        for index in itertools.product(range(13),repeat=3):
            shift=np.array(index)-6;d=basis@(shift-fit);distance=float(np.linalg.norm(d));nd=float(np.dot(d,n))
            if distance>2+1e-6:candidates.append((float(score[index]),tuple(int(s) for s in shift)))
            if abs(nd)>2+1e-6:normal_candidates.append((float(score[index]),tuple(int(s) for s in shift)))
        competitor=max(candidates,key=lambda a:a[0]);nc=max(normal_candidates,key=lambda a:a[0])
        d=basis@(best-fit);dn=float(np.dot(d,n));dt=float(np.linalg.norm(d-dn*n));fv=float(score[idx])
        rows.append({'index':i,'role':'fit' if i<4 else 'check','centre_core_xy':list(XY[i]),'zero_score':float(score[6,6,6]),'at_fit_score':fv,'independent_best_zyx':best.tolist(),'independent_best_score':float(score[bestidx]),'independent_peak_on_boundary':bool(np.any(np.abs(best)==6)),'check_fit_normal_delta_native_vox':dn,'check_fit_tangential_delta_native_vox':dt,'check_fit_distance_native_vox':float(np.linalg.norm(d)),'distinct_competitor_zyx':list(competitor[1]),'distinct_competitor_score':competitor[0],'margin':fv-competitor[0],'normal_competitor_zyx':list(nc[1]),'normal_competitor_score':nc[0],'normal_competitor_margin':fv-nc[0]})
    checks=rows[4:];t=[e['check_fit_tangential_delta_native_vox'] for e in checks];norm=[abs(e['check_fit_normal_delta_native_vox']) for e in checks]
    criteria={'fit_not_boundary':bool(np.max(np.abs(fit))<6),'checks_not_boundary':not any(e['independent_peak_on_boundary'] for e in checks),'all_check_ncc_at_least_05':all(e['at_fit_score']>=.5 for e in checks),'all_check_margin_at_least_005':all(e['margin']>=.05 for e in checks),'median_tangent_delta_at_most_05':bool(np.median(t)<=.5+1e-6),'max_tangent_delta_at_most_1':max(t)<=1+1e-6,'max_abs_normal_delta_at_most_1':max(norm)<=1+1e-6}
    return {'fit_translation_zyx':fit.tolist(),'fit_mean_ncc':float(fitcube[idx]),'baseline_mean_fit_ncc':float(fitcube[6,6,6]),'fit_translation_native_xyz':(core@fit).tolist(),'fit_translation_micrometres':(core@fit*9.362).tolist(),'criteria':criteria,'structural_screen_pass':all(criteria.values()),'supports':rows}

def compare(a,b,path='',diffs=None):
    if diffs is None:diffs=[]
    if isinstance(a,dict):
        for k,v in a.items():
            if k=='normal_competitor_margin':continue
            compare(v,b[k],path+'/'+k,diffs)
    elif isinstance(a,list):
        assert len(a)==len(b),(path,'length')
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+f'/{i}',diffs)
    elif isinstance(a,(float,np.floating)):
        error=abs(a-b);diffs.append(error);assert error<1e-9,(path,a,b)
    else:assert a==b,(path,a,b)
    return diffs

def main():
    verified=[]
    for e in json.loads((D/'OUTPUT_FILES.json').read_text()):
        p=D/e['path'];assert p.stat().st_size==e['bytes'] and sha(p)==e['sha256'];verified.append(e)
    local=json.loads((B/'geometry-review/pullback/LOCAL_BASES.json').read_text());results=[]
    for s in local['sites']:
        sid=s['site'];directory=D/sid;native=np.load(directory/'native_full.npy');fine=np.load(directory/'fine_remapped.npy');saved=np.load(directory/'landscapes.npy');authored=json.loads((directory/'RESULTS.json').read_text())
        calculations=[landscape(native,fine,x,y) for x,y in XY];scores=np.stack([a for a,b in calculations]);diagnostics=[b for a,b in calculations]
        difference=np.abs(scores-saved);assert np.isfinite(scores).all() and difference.max()<1e-9
        result=summarise(scores,np.asarray(s['core_basis_columns_depth_y_x_nativeXYZ']),[np.asarray(e['basis_columns_depth_y_x_nativeXYZ']) for e in s['supports']])
        metric_errors=compare(result,authored)
        result.update(site=sid,maximum_landscape_difference=float(difference.max()),maximum_summary_difference=max(metric_errors),support_intensity_diagnostics=diagnostics)
        np.save(R/f'{sid}-independent-landscapes.npy',scores);results.append(result)
        print(sid,'fit',result['fit_translation_zyx'],'ncc',result['fit_mean_ncc'],'maxdiff',float(difference.max()),'screen',result['structural_screen_pass'],flush=True)
    output={'status':'PASS','verified_outputs':verified,'scores_independently_recomputed':3*8*13**3,'method':'Directspatialwindowmean/varianceanddotproduct; noFFT,integralsums,authoredscoringimports. Allsummariesrecomputedwithscalarphysicalcompetitorloopandfinal1e-6tolerance.','sites':results,'limitations':['Replays saved CT-derived arrays; independentrawchunkassembly/resampling isoutside thischeck.','Noinkorindependentphysicalregistrationtruth; fixedscreen failures preserved.']}
    (R/'REAL_DATA_REPLAY.json').write_text(json.dumps(output,indent=2)+'\n')
if __name__=='__main__':main()
