"""Separate mask-only panel following the archived union-panel admission failure."""
import argparse, json, shutil
from pathlib import Path
import numpy as np
import intake
from sampling import context,spread


def main():
    p=argparse.ArgumentParser();p.add_argument('--initial',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    a=p.parse_args();initial,root=a.initial,a.work
    assert not root.exists();root.mkdir(parents=True)
    original=intake.verify_freeze(initial);old=intake.load(initial/'SELECTION.json')
    oldinputs=intake.load(initial/'INPUTS_COMPLETE.json')
    failure=[c for c in oldinputs['cases'] if c['id'].startswith('pherc0841')]
    assert len(failure)==3 and all(sum(r['positive_pixels'] for r in c['cores'])==0 for c in failure)
    # Copy only the original immutable metadata/implementation, never ink/CT here.
    for name in intake.load(initial/'FREEZE.json')['pins']:
        dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(initial/name,dest)
    shutil.copyfile(initial/'FREEZE.json',root/'FREEZE.json')
    here=Path(__file__).parent
    for name in ['SUPERVISION_PANEL.md','select_supervision.py']:
        shutil.copyfile(here/name,root/'implementation'/name)
    go={'utc':intake.stamp(),'original_selection_sha256':intake.sha(initial/'SELECTION.json'),
        'original_input_receipt_sha256':intake.sha(initial/'INPUTS_COMPLETE.json'),
        'implementation':{n:intake.pin(here/n) for n in ['SUPERVISION_PANEL.md','select_supervision.py']},
        'prior_model_inference':0,'project_reference_labels_already_exposed':True,
        'selector_uses_ink_values':False,'change':'Supervision alone for both selection and primary score support'}
    intake.write(root/'SUPERVISION_GO.json',go)
    cases=[];requests=[];supportdir=root/'support';supportdir.mkdir()
    for c in original['cases']:
        oldcase=next(r for r in old['cases'] if r['id']==c['id'])
        with np.load(initial/'support'/(c['id']+'.npz')) as cached:
            arrays={k:cached[k] for k in cached.files}
        s=arrays['supervision'];h,w=s.shape;arrays['participation']=s
        np.savez_compressed(supportdir/(c['id']+'.npz'),**arrays)
        rows=[]
        for y in range(0,h-127,128):
            for x in range(0,w-127,128):
                n=int(s[y:y+128,x:x+128].sum());ctx=context((h,w),c['native']['spacing_zyx_um'][0],x,y)
                rows.append(dict(id=f'{c["id"]}-x{x:05}-y{y:05}',x=x,y=y,supported_pixels=n,
                                 complete_context=ctx is not None,eligible=n>=4096 and ctx is not None))
        eligible=[r for r in rows if r['eligible']];selected=spread(eligible);cores=[];needed=set()
        intake.csvfile(supportdir/(c['id']+'-all-cores.csv'),rows)
        for rank,r in enumerate(selected):
            ctx=context((h,w),c['native']['spacing_zyx_um'][0],r['x'],r['y'])
            cores.append(dict(r,selection_order=rank,**ctx));needed.update(map(tuple,ctx['native_chunks_zyx']))
        for z,cy,cx in sorted(needed):
            key=f'{z}/{cy}/{cx}';ident=f'{c["id"]}-z{z}-y{cy}-x{cx}'
            requests.append(dict(id=ident,case=c['id'],chunk_zyx=[z,cy,cx],url=intake.BASE+c['native']['source_path']+'0/'+key,
                expected_bytes=c['native']['shape_zyx'][0]*128*128,relative_path=f'ct/{c["id"]}/0/{key}'))
        cases.append(dict(id=c['id'],sample=c['sample'],segment=c['segment'],native=c['native'],
                 mask_summary=dict(support_policy='supervision_only',support_pixels=int(s.sum()),support_sha256=intake.ash(s)),
                 eligible_cores=len(eligible),cores=cores,source_chunk_count=len(needed),
                 coordinates_equal_original=[r['core_xywh'] for r in cores]==[r['core_xywh'] for r in oldcase['cores']]))
        print(json.dumps({'case':c['id'],'eligible':len(eligible),'chosen':[(r['x'],r['y']) for r in selected]}),flush=True)
    intake.write(root/'SELECTION.json',dict(status='FROZEN_BEFORE_NEW_COHORT_CT_REUSE_OR_MODEL_INFERENCE',utc=intake.stamp(),
        freeze_sha256=intake.sha(root/'FREEZE.json'),supervision_go_sha256=intake.sha(root/'SUPERVISION_GO.json'),
        cases=cases,selected_cores=24,model_forwards=0,selector_ink_input=False,prior_reference_class_counts_exposed=True))
    expected=sum(r['expected_bytes'] for r in requests)
    intake.write(root/'CT_PLAN.json',dict(selection_sha256=intake.sha(root/'SELECTION.json'),requests=requests,
        expected_bytes=expected,body_cap_bytes=expected+8*1024**2))
    # Only after sealing selection: verified cache reuse and inventory reuse.
    shutil.copytree(initial/'ct-inventory',root/'ct-inventory')
    cache=[]
    for r in requests+[i for c in original['cases'] for i in c['arrays']]:
        source=initial/r['relative_path'];rec=initial/'receipts'/(r['id']+'.json')
        if not rec.exists():continue
        receipt=intake.load(rec);assert intake.pin(source)=={k:receipt[k] for k in ['bytes','sha256']}
        dest=root/r['relative_path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
        dr=root/'receipts'/rec.name;dr.parent.mkdir(exist_ok=True);shutil.copyfile(rec,dr)
        cache.append(dict(file=r['relative_path'],**intake.pin(dest),receipt=rec.name))
    intake.write(root/'CACHE_REUSE.json',dict(original_selection_sha256=go['original_selection_sha256'],
        reused_after_selection_sha256=intake.sha(root/'SELECTION.json'),source_scope='Initial union panel',files=cache))
    shutil.copytree(initial/'errors',root/'inherited-initial-errors')


if __name__=='__main__':main()
