"""Bind the fixed central model crop and conservative 3x3 labels; no CT/model."""
from pathlib import Path
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent

def pin(p):return dict(path=str(p.resolve()),size_bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def main():
    frozen=json.loads((HERE/'CPU_PANEL_GO.json').read_text())
    for item in frozen['files']:assert pin(Path(item['path']))==item
    panel=json.loads((HERE/'qc-v1/PANEL.json').read_text())
    out=HERE/'cpu-panel-v1';out.mkdir(exist_ok=False)
    contexts=[]
    for tile,item in zip(panel['tiles'],panel['label_arrays'],strict=True):
        assert pin(Path(item['path']))==item
        source=np.load(item['path'],allow_pickle=False)
        crop=source[64:448,64:448]
        grouped=crop.reshape(128,3,128,3)
        low=grouped.min(axis=(1,3));high=grouped.max(axis=(1,3))
        labels=np.where((low==high)&(high<2),low,2).astype(np.uint8)
        core=labels[32:96,32:96].copy()
        # Separate scalar 3x3 implementation; do not import or repeat grouping axes.
        scalar=np.full((128,128),2,np.uint8)
        for y in range(128):
            for x in range(128):
                values=crop[y*3:y*3+3,x*3:x*3+3].ravel().tolist()
                if all(v==0 for v in values):scalar[y,x]=0
                elif all(v==1 for v in values):scalar[y,x]=1
        assert np.array_equal(labels,scalar)
        np.save(out/(tile['id']+'-labels128.npy'),labels,allow_pickle=False)
        np.save(out/(tile['id']+'-core64.npy'),core,allow_pickle=False)
        counts=np.bincount(core.ravel(),minlength=3).tolist()
        contexts.append(dict(id=tile['id'],parent='PHerc51',source_bbox_yxhw=[tile['y']+64,tile['x']+64,384,384],
            source_score_bbox_yxhw=[tile['y']+160,tile['x']+160,192,192],
            labels128=pin(out/(tile['id']+'-labels128.npy')),core64=pin(out/(tile['id']+'-core64.npy')),
            core_counts_negative_positive_unknown=counts,core_both_classes=counts[0]>0 and counts[1]>0,
            discarded_mixed_or_unknown_3x3_blocks=int(np.count_nonzero(labels==2))))
    result=dict(status='LABEL_CONTRACT_BOUND' if all(c['core_both_classes'] for c in contexts) else 'STOP_SINGLE_CLASS_CORE',
        experiment_specification=pin(HERE/'CPU_STUDY_SPEC.md'),source_panel=pin(HERE/'qc-v1/PANEL.json'),
        candidate='fixed_three_window_probability_average',contexts=contexts,parent_count=1,
        source_planes_inclusive=[1,63],effective_group_centers=list(range(2,63,3)),
        unique_window_starts=list(range(5)),evaluation_displacements=[-1,0,1],
        maximum_model_forwards=20,model_forwards_executed=0,model_execution_released=False,
        practical_minimum_robust_AP_gain=.01,practical_maximum_clean_AP_loss=.01,
        scalar_label_checks=4*128*128,pending=['per-TIFF metadata/identity','bounded multipart acquisition implementation and freeze','source payload verification','model input and runtime freeze'])
    with (out/'CPU_PANEL.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='contexts'},indent=2))
    for c in contexts:print(c['id'],c['core_counts_negative_positive_unknown'])

if __name__=='__main__':main()
