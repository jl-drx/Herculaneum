"""Independent paired-score metrics; no models, checkpoints or file I/O."""
from __future__ import annotations
import numpy as np

class Reject(ValueError): pass
def need(ok, code):
    if not ok: raise Reject(code)

def selected(labels, scores, primary, evaluated):
    need(labels.shape==scores.shape==primary.shape==evaluated.shape,'support_shape')
    need(primary.dtype==np.bool_ and evaluated.dtype==np.bool_,'support_bool')
    need(bool(primary.any()),'primary_empty')
    # Exact inherited support: never discard unevaluated or nonfinite primary pixels.
    need(bool(evaluated[primary].all()),'unscored_primary')
    y=labels[primary]; s=scores[primary]
    need(bool(np.isfinite(s).all()),'primary_nonfinite')
    need(bool(np.isin(y,[0,1]).all()),'nonbinary_label')
    need(bool((y==0).any() and (y==1).any()),'single_class')
    return y.astype(np.uint8),s

def ranking(y, score):
    y=np.asarray(y); score=np.asarray(score)
    need(y.ndim==score.ndim==1 and len(y)==len(score) and len(y)>0,'ranking_shape')
    need(bool(np.isfinite(score).all() and np.isin(y,[0,1]).all()),'ranking_values')
    positives=int((y==1).sum()); negatives=int((y==0).sum())
    need(positives>0 and negatives>0,'ranking_single_class')
    # Group equal scores independently of author ordering or rank assignment.
    unique, inverse=np.unique(score,return_inverse=True)
    p=np.bincount(inverse,weights=(y==1).astype(np.float64),minlength=len(unique))
    n=np.bincount(inverse,weights=(y==0).astype(np.float64),minlength=len(unique))
    negatives_below=np.cumsum(n)-n
    concordance=np.sum(p*(negatives_below+0.5*n),dtype=np.float64)
    precision=np.cumsum(p[::-1]) / np.cumsum((p+n)[::-1])
    ap=np.sum((p[::-1]/positives)*precision,dtype=np.float64)
    return dict(auc=float(concordance/(positives*negatives)),ap=float(ap),
        unique_scores=len(unique),positive=positives,negative=negatives,n=len(y))

def representations(raw):
    need(raw.dtype==np.float32,'raw_float32')
    need(bool(np.isfinite(raw).all() and ((raw>=0)&(raw<=1)).all()),'raw_range')
    encoded=(np.clip(raw,0,1)*np.float32(255)).astype(np.uint8)
    decoded=np.clip((encoded.astype(np.float32)-np.float32(64))/np.float32(128),0,1)
    return dict(raw=raw,uint8=encoded,decoded=decoded)

def model_metrics(labels, raw, primary, evaluated):
    y,s=selected(labels,raw,primary,evaluated)
    reprs=representations(s)
    result={key:ranking(y,value) for key,value in reprs.items()}
    result['decoded_zero_fraction']=float(np.count_nonzero(reprs['decoded']==0)/len(y))
    result['decoded_one_fraction']=float(np.count_nonzero(reprs['decoded']==1)/len(y))
    return result

def disagreement(native,seed42,primary):
    need(native.shape==seed42.shape==primary.shape and primary.dtype==np.bool_ and primary.any(),'paired_support')
    delta=native[primary].astype(np.float64)-seed42[primary].astype(np.float64)
    need(np.isfinite(delta).all(),'paired_finiteness')
    absolute=np.abs(delta)
    return dict(mean_signed=float(delta.mean()),mean_absolute=float(absolute.mean()),
        p95_absolute=float(np.quantile(absolute,.95,method='linear')),max_absolute=float(absolute.max()))

def parent_means(cases):
    groups={}
    for parent,count in [('Paris1',2),('Paris2',4)]:
        members=[c for c in cases if c['parent_papyrus']==parent]
        need(len(members)==count,'parent_context_count')
        entry=dict(context_ids=[c['id'] for c in members],context_count=count,models={})
        for model in ('seed42','native'):
            entry['models'][model]={representation:{metric:float(np.mean([c['models'][model][representation][metric] for c in members]))
                for metric in ('auc','ap')} for representation in ('raw','uint8','decoded')}
        entry['native_minus_seed42']={representation:{metric:entry['models']['native'][representation][metric]-entry['models']['seed42'][representation][metric]
            for metric in ('auc','ap')} for representation in ('raw','uint8','decoded')}
        groups[parent]=entry
    return groups

def scalar_close(actual, expected, path='', tolerance=1e-12):
    """Compare required numeric projection without hiding missing or changed fields."""
    if isinstance(expected,dict):
        need(isinstance(actual,dict),path+':mapping')
        for key,value in expected.items():
            need(key in actual,path+':missing:'+key); scalar_close(actual[key],value,path+'/'+key,tolerance)
    elif isinstance(expected,list):
        need(isinstance(actual,list) and len(actual)==len(expected),path+':list')
        for i,(a,b) in enumerate(zip(actual,expected,strict=True)): scalar_close(a,b,path+'/'+str(i),tolerance)
    elif isinstance(expected,float):
        need(isinstance(actual,(int,float)) and np.isfinite(actual) and abs(float(actual)-expected)<=tolerance,path+':numeric')
    else: need(actual==expected,path+':exact')
