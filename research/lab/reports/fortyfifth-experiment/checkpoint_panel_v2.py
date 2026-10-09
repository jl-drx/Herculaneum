"""Fixed 2 seeds x 2 released endpoints on the reused eight-region panel."""
import sys,json,time,hashlib,gc
from pathlib import Path
P=Path(__file__).resolve().parent
OLD=P.parent/'thirtyseventh-experiment'
sys.path.insert(0,str(OLD))
import study_common as common
from run_evaluation import Capture
from verify_evaluation import independent_rank
import numpy as np
import torch

def main():
    out=P/'checkpoint-panel';go=json.loads((out/'GO_V2.json').read_text())
    for pin in go['pins']:common.verify(pin)
    p=common.plan();base=json.loads(common.verify(p['training_plan']).read_text())
    models=json.loads((out/'ACQUISITION_PLAN.json').read_text())['models']
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    helper=common.module(common.verify(base['helper']),'panel_model_helper');runtime=helper.load_runtime()
    start=time.monotonic();results=[]
    for rec in models:
        # Construct using the already qualified final checkpoint; then strict-load the
        # actual pinned temporal endpoint. Do not lie to the old 74999-only helper.
        final=next(m for m in models if m['seed']==rec['seed'] and m['step']==75000)
        helper_record=final | {'label':f'seed{rec["seed"]}'}
        model,meta=helper.build_model(Path(final['path']),helper_record,runtime)
        if rec['step']!=75000:
            payload=torch.load(rec['path'],map_location='cpu',weights_only=True)
            common.require(payload['step']==rec['embedded_step'],'Actual saved step')
            helper.validate_authored_config(payload['config'])
            config=runtime['InkConfig'].from_mapping(payload['config'])
            common.require(config.model==runtime['InkConfig'].from_mapping(meta['embedded_config']).model,'Same architecture config')
            container,state=runtime['reference'].selected_state(payload)
            common.require(container=='model','Same non-EMA weight container')
            helper.verify_storage_contract(state,model.state_dict(),torch)
            common.require(all(torch.isfinite(v).all() for v in state.values()),'Finite released state')
            model.load_state_dict(state,strict=True);del payload,state
        model.eval();gc.collect();rows=[];dest=out/rec['label'];dest.mkdir(exist_ok=False)
        with torch.inference_mode():
            for c in p['cases']:
                x,y,m=common.load_case(p,c);capture=Capture(model,c['baseline_metadata'].get('captured_tiles'))
                scores,metadata,evaluated=runtime['reference'].infer_surface(capture,x,c['layer_start'],'forward',runtime['normalize_robust'],None)
                common.require(len(capture.logits)==25 and evaluated[m].all() and np.isfinite(scores[m]).all(),'Complete unchanged panel')
                np.save(dest/(c['id']+'-scores.npy'),scores);np.save(dest/(c['id']+'-logits.npy'),np.stack(capture.logits))
                metrics=independent_rank(y[m],scores[m].astype(np.float64))
                original=np.load(common.verify(c['baseline_metadata']['scores']))
                baseline=independent_rank(y[m],original[m].astype(np.float64))
                if rec['label']=='seed42-75000':common.require(np.array_equal(scores,original),'Exact eight-region baseline replay')
                rows.append(dict(id=c['id'],parent=c['parent'],metrics=metrics,baseline_metrics=baseline,delta={k:metrics[k]-baseline[k] for k in metrics},input_hashes=capture.hashes,scores=common.pin(dest/(c['id']+'-scores.npy')),logits=common.pin(dest/(c['id']+'-logits.npy'))))
        parents={parent:{k:float(np.mean([r['metrics'][k] for r in rows if r['parent']==parent])) for k in ['average_precision','roc_auc']} for parent in sorted(set(r['parent'] for r in rows))}
        result=dict(checkpoint=rec,regions=rows,parents=parents,Paris_equal_parent={k:(parents['Paris1'][k]+parents['Paris2'][k])/2 for k in ['average_precision','roc_auc']},new_forwards=200)
        common.save(dest/'RESULTS.json',result);results.append(result);print(json.dumps(dict(arm=rec['label'],parents=parents)),flush=True)
        del model;gc.collect()
    common.save(out/'RESULTS.json',dict(results=results,new_forwards=800,elapsed_seconds=time.monotonic()-start,interpretation='Descriptive checkpoint variability on reused development regions; two seeds do not estimate a reliable training-noise distribution. Temporal checkpoints are correlated, not extra seeds.'))

if __name__=='__main__':main()
