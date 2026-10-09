"""Offline descriptive analysis of retained scores, with existing cutoffs only."""
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import socket
import numpy as np

LAB=Path('/workspace/herculaneum-lab')
HERE=Path(__file__).resolve().parent
OLD=LAB/'reports/eighth-experiment/implementation'
NINTH=LAB/'reports/ninth-experiment/implementation'
ARMS=['frozen_seed42','A','B','C','A_smooth','B_smooth']
RANK=LAB/'reports/sixth-experiment/independent-metric-review/review_metrics.py'
RANK_SHA='e3cad20abca6f4daa82c34f25c3237ba4626fbda43c90cd36e0735f4923c0452'
QUANTILES=[.01,.05,.25,.5,.75,.95,.99]
QNAMES=['q01','q05','q25','median','q75','q95','q99']

def require(value,message):
    if not value:raise ValueError(message)

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def module(path):
    spec=importlib.util.spec_from_file_location('retained_independent_rank',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def correlation(x,y):
    x=x.astype(np.float64);y=y.astype(np.float64);x-=x.mean();y-=y.mean();denom=math.sqrt(float(np.dot(x,x))*float(np.dot(y,y)))
    return float(np.dot(x,y))/denom if denom else None

def ranks(scores):
    _,inverse,counts=np.unique(scores,return_inverse=True,return_counts=True)
    starts=np.cumsum(counts)-counts
    return (starts+(counts-1)/2)[inverse].astype(np.float64)

def distribution(scores):
    q=dict(zip(QNAMES,map(float,np.quantile(scores,QUANTILES))))
    return dict(count=int(scores.size),mean=float(np.mean(scores.astype(np.float64))),std=float(np.std(scores.astype(np.float64))),
                minimum=float(scores.min()),maximum=float(scores.max()),quantiles=q,iqr=q['q75']-q['q25'],central_98_width=q['q99']-q['q01'])

def operating(y,s,t):
    predicted=s.astype(np.float64)>=t;p,q=int(y.sum()),int((~y).sum());tp,fp=int((y&predicted).sum()),int((~y&predicted).sum())
    return dict(threshold=float(t),operator='float64(score) >= threshold',tp=tp,fp=fp,tn=q-fp,fn=p-tp,
                sensitivity=tp/p,specificity=(q-fp)/q,precision=tp/(tp+fp) if tp+fp else None,
                specificity_95_met_exact=fp<=q//20,negative_fraction_strictly_below_cutoff=(q-fp)/q,
                positive_fraction_below_cutoff=(p-tp)/p,negative_ties_at_cutoff=int((s[~y].astype(np.float64)==t).sum()))

def main():
    out=HERE/'result';require(not out.exists(),'Fresh score-analysis output required')
    pins={}
    def pin(path,item=None):
        p=Path(path).resolve();r=dict(path=str(p),size_bytes=p.stat().st_size,sha256=sha(p))
        if item is not None:require(r['size_bytes']==item['size_bytes'] and r['sha256']==item['sha256'],'Retained artifact differs: '+p.name)
        require(str(p) not in pins or pins[str(p)]==r,'Conflicting identity');pins[str(p)]=r;return p
    def read(path,item=None):return json.loads(pin(path,item).read_text())
    def load(item):return np.load(pin(item['path'],item),allow_pickle=False)
    pin(__file__);pin(RANK);require(sha(RANK)==RANK_SHA,'Independent rank source changed');ranking=module(RANK).ranked_metrics
    p8=read(OLD/'final-plan/training-plan.json');p9=read(NINTH/'final-plan/control-plan.json')
    require(sha(OLD/'final-plan/training-plan.json')=='eadb8406f594446e62840366c280358ded25b93874bd04e07b3e0e2482ee96f3' and sha(NINTH/'final-plan/control-plan.json')=='9d72d3177d5aa68cd5d59db6f316ed1a5c335f442b6c5c2439f45bfdaebe29cd','Prior frozen plans differ')
    require(p9['development']==p8['development'] and p9['diagnostic']==p8['fresh'],'Common source contract differs')
    reports={'eighth':read(OLD/'analysis/metrics.json'),'ninth':read(NINTH/'analysis/metrics.json')}
    require(sha(OLD/'analysis/metrics.json')=='ce8354f3a023b105b5e02aeabd4fbf512915ec81d3b76e7b999b2e16a9ddb89b' and sha(NINTH/'analysis/metrics.json')=='5b70146072474d00998bd82dc164c0cf0e6cedf4a570418dedaeb5066f11cad0','Prior frozen metrics differ')
    for path,digest in [(LAB/'reports/eighth-experiment/metric-review/result/independent-verification.json','9ec0ca5f0f0db7352afdaa2d3b31254ef7ed7a14794b81a20e3ad7789fe0207b'),(LAB/'reports/ninth-experiment/review/result/independent-verification.json','0cb4bf51046ea300d080bdd242526939c40604b34900f09960bb027798cd7a39')]:
        require(read(path)['status']=='PASS' and sha(path)==digest,'Independent historical proof differs')
    workers={arm:read((NINTH if arm.endswith('_smooth') else OLD)/'run'/arm/'result.json') for arm in ARMS}
    locks={arm:read(w['threshold_lock']['path'],w['threshold_lock']) for arm,w in workers.items()}
    for arm,w in workers.items():
        saved=reports['ninth' if arm.endswith('_smooth') else 'eighth']['model_results'][arm]
        pin(saved['path'],saved);require(w['status']=='PASS' and locks[arm]['arm']==arm,'Model role/status differs')
        require(w['initial_state_sha256']=='21c186000b9b3093835e185342668acd0d86147a719c37513e937924b77ea228','Initializer changed')
    frozen_cutoff=locks['frozen_seed42']['primary'];require(frozen_cutoff==0.7558947205543518,'Exact fixed reference cutoff differs')
    cases=[('calibration',c) for c in p8['development']]+[('diagnostic',c) for c in p8['fresh']]
    result={};rows=[];max_metric_error=0.
    for role,case in cases:
        arrays={k:load(v) for k,v in case['inputs'].items()};y=arrays['inklabels']>0;shape=y.shape
        require(arrays['inklabels'].dtype==np.uint8 and set(np.unique(arrays['inklabels']))<={0,1},'Label encoding differs')
        part=arrays['validation_mask']>0 if case['evaluation_participation_role']=='public_validation' else arrays['supervision_mask']>0
        require(np.array_equal(part,arrays['evaluation_participation_mask']>0),'Participation differs')
        mask=np.zeros(shape,bool);mask[64:-64,64:-64]=True;mask&=part&arrays['acquisition_observed_mask'];fixed=int(mask.sum());maps={}
        for arm,w in workers.items():
            field='development' if role=='calibration' else 'diagnostic' if arm.endswith('_smooth') else 'fresh'
            item=w[field][case['id']];s,e=load(item['scores']),load(item['evaluated']);require(s.dtype==np.float32 and e.dtype==bool and s.shape==e.shape==shape,'Map encoding differs')
            require(np.all((s[np.isfinite(s)]>=0)&(s[np.isfinite(s)]<=1)),'Score range differs');maps[arm]=s;mask&=e&np.isfinite(s)
        require(int(mask.sum())==fixed,'All-six common support changed from fixed domain');labels=y[mask];require(min(int(labels.sum()),int((~labels).sum()))>=256,'Class support changed')
        context=dict(role=role,physical_group=case['physical_group'],bbox_source_xywh=case['data']['bbox_source_xywh'],evaluation_participation_role=case['evaluation_participation_role'],
                     pixel_count=fixed,positive_count=int(labels.sum()),negative_count=int((~labels).sum()),models={})
        baseline=maps['frozen_seed42'][mask];base_d={name:distribution(baseline[m]) for name,m in [('positive',labels),('negative',~labels)]}
        for arm,scoremap in maps.items():
            values=scoremap[mask];rank=ranking(labels,values);dist={name:distribution(values[m]) for name,m in [('positive',labels),('negative',~labels)]};shift={}
            for name,m in [('positive',labels),('negative',~labels)]:
                delta=values[m].astype(np.float64)-baseline[m].astype(np.float64)
                shift[name]=dict(paired_score_shift_quantiles=dict(zip(QNAMES,map(float,np.quantile(delta,QUANTILES)))),
                    iqr_ratio_to_frozen=dist[name]['iqr']/base_d[name]['iqr'] if base_d[name]['iqr'] else None,
                    central_98_width_ratio_to_frozen=dist[name]['central_98_width']/base_d[name]['central_98_width'] if base_d[name]['central_98_width'] else None,
                    pearson_to_frozen=correlation(values[m],baseline[m]),spearman_to_frozen=correlation(ranks(values[m]),ranks(baseline[m])))
            ops={name:operating(labels,values,t) for name,t in [('midpoint',.5),('same_frozen_numeric_cutoff',frozen_cutoff),('own_primary',locks[arm]['primary']),('own_secondary',locks[arm]['secondary'])]}
            if role=='diagnostic':
                saved=reports['ninth']['diagnostic'][case['id']]['models'][arm] if arm.endswith('_smooth') else reports['eighth']['fresh'][case['id']]['models'][arm]
                for key,v in [('auc',rank['roc_auc']),('ap',rank['average_precision'])]:
                    error=abs(v-saved[key]);max_metric_error=max(max_metric_error,error);require(error<=1e-12,'Prior ranking metric differs')
                for rule in ('primary','secondary'):
                    stored=saved['operating'][rule];actual=ops['own_'+rule]
                    require({k:actual[k] for k in ('tp','fp','tn','fn')}==stored['confusion'],'Prior operating counts differ')
                    require(actual['threshold']==stored['threshold'],'Prior cutoff differs')
            else:
                saved=reports['ninth']['calibration'][arm][case['id']] if arm.endswith('_smooth') else reports['eighth']['calibration'][arm][case['id']]
                source=saved if arm.endswith('_smooth') else dict(auc=saved['ranking']['roc_auc'],ap=saved['ranking']['average_precision'])
                for key,v in [('auc',rank['roc_auc']),('ap',rank['average_precision'])]:
                    error=abs(v-source[key]);max_metric_error=max(max_metric_error,error);require(error<=1e-12,'Prior calibration ranking differs')
                require(ops['own_primary']['specificity_95_met_exact'],'Locked primary calibration guarantee differs')
            gap=dist['positive']['quantiles']['median']-dist['negative']['quantiles']['median'];base_gap=base_d['positive']['quantiles']['median']-base_d['negative']['quantiles']['median']
            model=dict(ranking=rank,class_score_distribution=dist,class_comparison_to_frozen=shift,median_class_gap=gap,median_gap_ratio_to_frozen=gap/base_gap if base_gap else None,
                       overall_pearson_to_frozen=correlation(values,baseline),overall_spearman_to_frozen=correlation(ranks(values),ranks(baseline)),operating=ops,
                       own_primary_minus_midpoint_sensitivity=ops['own_primary']['sensitivity']-ops['midpoint']['sensitivity'],own_primary_minus_secondary_sensitivity=ops['own_primary']['sensitivity']-ops['own_secondary']['sensitivity'])
            if role=='calibration':model['already_locked_local_cutoff']=locks[arm]['development'][case['id']]['selection']['threshold']
            context['models'][arm]=model
            rows.append(dict(role=role,case=case['id'],model=arm,pixels=fixed,positive=int(labels.sum()),negative=int((~labels).sum()),auc=rank['roc_auc'],ap=rank['average_precision'],
                             positive_q05=dist['positive']['quantiles']['q05'],positive_median=dist['positive']['quantiles']['median'],positive_q95=dist['positive']['quantiles']['q95'],
                             negative_q05=dist['negative']['quantiles']['q05'],negative_median=dist['negative']['quantiles']['median'],negative_q95=dist['negative']['quantiles']['q95'],median_class_gap=gap,
                             positive_iqr_ratio=shift['positive']['iqr_ratio_to_frozen'],negative_iqr_ratio=shift['negative']['iqr_ratio_to_frozen'],spearman_to_frozen=model['overall_spearman_to_frozen'],
                             midpoint_sensitivity=ops['midpoint']['sensitivity'],midpoint_specificity=ops['midpoint']['specificity'],own_primary=ops['own_primary']['threshold'],own_primary_sensitivity=ops['own_primary']['sensitivity'],own_primary_specificity=ops['own_primary']['specificity'],
                             own_secondary=ops['own_secondary']['threshold'],own_secondary_sensitivity=ops['own_secondary']['sensitivity'],same_frozen_numeric_cutoff_sensitivity=ops['same_frozen_numeric_cutoff']['sensitivity'],same_frozen_numeric_cutoff_specificity=ops['same_frozen_numeric_cutoff']['specificity']))
        result[case['id']]=context
    diagnostic=[v for v in result.values() if v['role']=='diagnostic'];macro={}
    for arm in ARMS:
        m=[c['models'][arm] for c in diagnostic]
        macro[arm]=dict(auc=math.fsum(x['ranking']['roc_auc'] for x in m)/2,ap=math.fsum(x['ranking']['average_precision'] for x in m)/2,
                       median_positive=math.fsum(x['class_score_distribution']['positive']['quantiles']['median'] for x in m)/2,
                       median_negative=math.fsum(x['class_score_distribution']['negative']['quantiles']['median'] for x in m)/2,
                       positive_iqr_ratio=math.fsum(x['class_comparison_to_frozen']['positive']['iqr_ratio_to_frozen'] for x in m)/2,
                       negative_iqr_ratio=math.fsum(x['class_comparison_to_frozen']['negative']['iqr_ratio_to_frozen'] for x in m)/2,
                       spearman_to_frozen=math.fsum(x['overall_spearman_to_frozen'] for x in m)/2,
                       operating={rule:dict(sensitivity=math.fsum(x['operating'][rule]['sensitivity'] for x in m)/2,specificity=math.fsum(x['operating'][rule]['specificity'] for x in m)/2) for rule in ('midpoint','same_frozen_numeric_cutoff','own_primary','own_secondary')})
    for path,item in pins.items():require(Path(path).stat().st_size==item['size_bytes'] and sha(path)==item['sha256'],'Retained input changed during analysis')
    out.mkdir(parents=True)
    with (out/'score-distributions.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(status='PASS',code_sha256=sha(__file__),independent_rank_sha256=RANK_SHA,verified_unique_records=len(pins),verified_inputs_before_and_after=list(pins.values()),
                 existing_thresholds={a:dict(primary=l['primary'],secondary=l['secondary'],local_calibration={i:v['selection']['threshold'] for i,v in l['development'].items()}) for a,l in locks.items()},
                 contexts=result,equal_context_diagnostic_macro=macro,max_prior_metric_difference=max_metric_error,new_thresholds_selected=0,network_requests=0,model_calls=0,
                 interpretation='Descriptive retained score shifts/ranking/operating-point transfer; common fixed supplied-annotation domains, one specimen for diagnostics, no feature evidence or causal representation-damage inference')
    target=out/'score-diagnostics.json';target.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(status='PASS',path=str(target),sha256=sha(target),records=len(pins),max_prior_metric_difference=max_metric_error)))

if __name__=='__main__':
    def blocked(*a,**k):raise RuntimeError('Score analysis attempted network access')
    socket.socket.connect=blocked;socket.socket.connect_ex=blocked;socket.socket.sendto=blocked;socket.create_connection=blocked;socket.getaddrinfo=blocked
    main()
