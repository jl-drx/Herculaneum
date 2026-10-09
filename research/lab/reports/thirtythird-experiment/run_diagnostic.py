"""One offline execution of the frozen diagnostic family; no model imports."""
import resource,signal,socket,time
from PIL import Image
from diagnostic import *

def offline(*args,**kwargs):raise RuntimeError('Network disabled in cached diagnostic')
socket.create_connection=offline;socket.socket.connect=offline;socket.socket.connect_ex=offline

def ranges(rows,role):
    defined=[r['metrics'][role]['ap'] for r in rows if r['metrics'][role]['ap'] is not None]
    zero=next(r['metrics'][role]['ap'] for r in rows if r['shift']==[0,0])
    return dict(defined_shifts=len(defined),zero_AP=zero,min_AP=min(defined) if defined else None,
                max_AP=max(defined) if defined else None,AP_span=max(defined)-min(defined) if defined else None,
                min_delta_vs_same_support_rule_zero=min(defined)-zero if defined and zero is not None else None,
                max_delta_vs_same_support_rule_zero=max(defined)-zero if defined and zero is not None else None)

def main():
    started=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in()).throw(TimeoutError('120second diagnostic cap')));signal.alarm(120)
    check_freeze();plan=json.loads((HERE/'PLAN.json').read_text());out=HERE/'run-v1';out.mkdir(exist_ok=False)
    summaries=[];all_metrics=[]
    try:
        with Image.open(verify(plan['optical'])) as ir:
            require(ir.mode=='RGBA' and ir.size==(6205,8853),'Optical source contract')
            for c in plan['contexts']:
                labels=np.load(verify(c['labels128']),allow_pickle=False)
                core=np.load(verify(c['core64']),allow_pickle=False)
                require(np.array_equal(core,labels[32:96,32:96]),'Core registration changed')
                old_arms=np.load(verify(c['prior_scores']),allow_pickle=False)
                require(old_arms.shape==(3,2,64,64),'Cached score shape')
                shifted,common,stable=support(labels);dist,bands=boundary_bands(labels)
                y,x,h,w=c['source_bbox_yxhw'];rgba=np.asarray(ir.crop((x,y,x+w,y+h)))
                require(rgba.shape==(384,384,4) and np.array_equal(rgba[...,0],rgba[...,1]) and np.array_equal(rgba[...,0],rgba[...,2]),'Grayscale source required')
                sums=rgba[...,0].reshape(128,3,128,3).sum(axis=(1,3),dtype=np.uint32)
                optical=1-sums.astype(np.float64)/2295.0
                opaque=(rgba[...,3].reshape(128,3,128,3)==255).all(axis=(1,3))[32:96,32:96]
                require(opaque[(shifted<2).any(axis=0)].all(),'Nonopaque optical support; do not silently shrink')
                predictions=dict(baseline=old_arms[1,0],candidate=old_arms[1,1],optical_darkness=optical[32:96,32:96])
                original=np.ones(core.shape,bool);known=core<2
                original_metrics={k:measure(v,core,original) for k,v in predictions.items()}
                for role in ('baseline','candidate'):
                    prior=c['prior_clean'][role]
                    require(original_metrics[role]['counts']==prior['counts'],'Original support counts changed')
                    for key in ('ap','auc'):require(abs(original_metrics[role][key]-prior[key])<1e-12,'Original phase32 score changed')
                common_rows=[];moving_rows=[]
                for sl,(dy,dx) in zip(shifted,SHIFTS,strict=True):
                    moving=sl<2
                    common_rows.append(dict(shift=[dy,dx],changed_labels_on_common=int(np.count_nonzero(common&(sl!=core))),
                        metrics={k:measure(v,sl,common) for k,v in predictions.items()}))
                    moving_rows.append(dict(shift=[dy,dx],support_added_vs_original=int(np.count_nonzero(moving&~known)),
                        support_removed_vs_original=int(np.count_nonzero(known&~moving)),
                        changed_labels_on_shared_known=int(np.count_nonzero(known&moving&(sl!=core))),
                        metrics={k:measure(v,sl,original) for k,v in predictions.items()}))
                band_rows=[dict(band=name,metrics={k:measure(v,core,bands[name]) for k,v in predictions.items()}) for name in BANDS]
                stable_metrics={k:measure(v,core,stable) for k,v in predictions.items()}
                arr=out/(c['id']+'-arrays.npz')
                np.savez(arr,shifted_labels=shifted,common=common,stable=stable,distance=dist,
                         band_masks=np.stack([bands[n] for n in BANDS]),optical128=optical,
                         predictions=np.stack([predictions[k] for k in PREDICTORS]))
                item=dict(id=c['id'],parent='PHerc51',source_context=c,arrays=pin(arr),predictor_order=PREDICTORS,
                    shifts=[list(s) for s in SHIFTS],band_order=BANDS,original=original_metrics,
                    common_shifts=common_rows,moving_support_shifts=moving_rows,bands=band_rows,stable=stable_metrics,
                    support=dict(original_known=int(known.sum()),common_known=int(common.sum()),
                         removed_for_common=int(np.count_nonzero(known&~common)),class_stable_common=int(stable.sum()),
                         potentially_changeable_common=int(np.count_nonzero(common&~stable)),
                         optical_opaque_on_any_scored_pixel=True),
                    common_shift_ranges={k:ranges(common_rows,k) for k in PREDICTORS},
                    moving_support_shift_ranges={k:ranges(moving_rows,k) for k in PREDICTORS},
                    common_shift_test_informative_about_labels=bool(np.any(common&~stable)))
                save(out/(c['id']+'.json'),item);summaries.append(item)
                all_metrics+=list(original_metrics.values())+list(stable_metrics.values())
                all_metrics+=[m for rows in [common_rows,moving_rows,band_rows] for r in rows for m in r['metrics'].values()]
                require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=1024**3,'RSS cap exceeded')
                print(json.dumps(dict(context=c['id'],support=item['support'],baseline_common_shift_range=item['common_shift_ranges']['baseline'],
                     baseline_moving_shift_range=item['moving_support_shift_ranges']['baseline'],
                     original_optical=original_metrics['optical_darkness'],bands=[dict(band=r['band'],baseline=r['metrics']['baseline'],optical=r['metrics']['optical_darkness']) for r in band_rows])),flush=True)
        require(len(all_metrics)==276,'Frozen metric family size')
        require(sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<64*1024**2,'Phase disk exceeded')
        check_freeze()
        result=dict(status='PASS',created_at_utc=now(),GO=pin(HERE/'GO.json'),contexts=summaries,
                    metric_records=len(all_metrics),defined_metric_records=sum(m['status']=='defined' for m in all_metrics),
                    undefined_metric_records=sum(m['status']!='defined' for m in all_metrics),
                    elapsed_seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    source_primary_result=plan['prior_results'],original_primary_unchanged=True,
                    new_model_forwards=0,new_source_requests=0,GPU_spend_GBP=0,
                    scope='Posthoc reference sensitivity on four previously scored contexts; no chosen correction/independent truth')
        save(out/'COMPLETE.json',result)
    except Exception as e:
        save(out/'FAILURE.json',dict(status='STOP',error=type(e).__name__+': '+str(e)));raise

if __name__=='__main__':main()
