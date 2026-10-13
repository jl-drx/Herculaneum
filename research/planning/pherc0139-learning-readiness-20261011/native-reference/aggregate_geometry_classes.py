"""Post-selection counts only; cannot change mask or geometry-selected core IDs."""
import pathlib,json,hashlib
P=pathlib.Path(__file__).resolve().parent
sources=['RESULTS.json','ct/support/RESULTS.json','geometry/RESULTS.json']
a,ct,g=[json.loads((P/f).read_text()) for f in sources]
rows={r['core_id']:r for s in a['sites'] for r in s['rows']}
ctrows={r['core_id']:r for s in ct['sites'] for r in s['rows']}
assert set(rows)==set(ctrows) and g['ink_labels_used_for_selection'] is False
out=[]
def summarize(ids):
 assert len(ids)==len(set(ids))
 rr=[rows[x] for x in ids]
 return {'cores':len(rr),'supervised_pixels':sum(r['supervised_pixels'] for r in rr),'ink_supervised':sum(r['ink_supervised'] for r in rr),'nonink_supervised':sum(r['nonink_supervised'] for r in rr),'cores_both_classes':sum(r['ink_supervised']>0 and r['nonink_supervised']>0 for r in rr),'cores_without_ink':sum(r['ink_supervised']==0 for r in rr),'cores_without_nonink':sum(r['nonink_supervised']==0 for r in rr),'supervised_zero_ct_pixels':sum(ctrows[x]['supervised_zero_ct_pixels'] for x in ids)}
for f in g['folds']:
 t=f['retained_training_core_ids'];e=f['original_evaluation_core_ids'];assert not set(t)&set(e)
 train=summarize(t);assert train['supervised_pixels']==f['retained_training_supervised_pixels']
 out.append({'held_out':f['held_out'],'geometry_admitted':f['fold_geometry_admitted'],'training_after_geometry_exclusions':train,'evaluation_unchanged':summarize(e),'class_counts_used_for_selection':False})
result={'scope':'Counts after mask-only cohort and geometry-only training exclusions; no label-based selection or model training','inputs_sha256':{f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in sources},'folds':out}
(P/'GEOMETRY_FOLD_CLASS_COUNTS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(out,indent=2))
