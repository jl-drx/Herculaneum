"""Summarize frozen regions as detector behavior, never accuracy or text."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

LAB = Path('/workspace/herculaneum-lab')
REPORT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(LAB/'config/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def summary(values):
    return {'count':int(values.size),'minimum':float(values.min()) if values.size else None,
            'maximum':float(values.max()) if values.size else None,
            'mean':float(values.mean()) if values.size else None,
            'std':float(values.std()) if values.size else None,
            'percentiles_1_50_99':[float(v) for v in np.percentile(values,[1,50,99])] if values.size else None,
            'above_midpoint_count':int((values>0.5).sum()),
            'above_midpoint_fraction':float((values>0.5).mean()) if values.size else None}


def main():
    plan_path=REPORT/'independent-selection-plan.json'
    plan=json.loads(plan_path.read_text())
    assert sha(plan_path)=='32741ba9d330940f077129e07e1d6e33f844824fa9d1b589de15592643ff01c2'
    for path,expected in plan['inputs'].items():
        assert sha(path)==expected
    full_surface=np.load(LAB/'data/experiments/pherc0800-surface980/surface.npy',mmap_mode='r',allow_pickle=False)
    geometry=np.load(LAB/'data/experiments/pherc0800-surface-geometry/full-surface-valid-mask.npy',mmap_mode='r',allow_pickle=False)
    # Prediction maps are first loaded here, after selections were frozen and
    # all independent-context inference commands completed.
    full_output=LAB/'reports/first-experiment/ink9um-full-center-forward'
    full_metadata=json.loads((full_output/'metadata.json').read_text())
    assert full_metadata['source']['sha256']==sha(LAB/'data/experiments/pherc0800-surface980/surface.npy')
    baseline={}
    for label in ('seed42','seed43'):
        entry=full_metadata['outputs'][label]
        path=full_output/entry['path']
        assert sha(path)==entry['sha256']
        baseline[label]=np.load(path,mmap_mode='r',allow_pickle=False)
    display_values=np.concatenate([full_surface[layer][geometry] for layer in (11,15,19)])
    ct_low,ct_high=[float(value) for value in np.percentile(display_values,[1,99])]
    records=[]
    for selection in plan['selected']:
        name=selection['id']
        data=LAB/'data/experiments'/('pherc0800-'+name)
        output=REPORT/name
        provenance=json.loads((data/'provenance.json').read_text())
        assert provenance['selection_plan_sha256']==sha(plan_path)
        assert sha(data/'surface.npy')==provenance['array_sha256']
        assert sha(data/'valid-mask.npy')==provenance['valid_mask_sha256']
        surface=np.load(data/'surface.npy',mmap_mode='r',allow_pickle=False)
        mask=np.load(data/'valid-mask.npy',mmap_mode='r',allow_pickle=False)
        metadata=json.loads((output/'metadata.json').read_text())
        scores={label:np.load(output/metadata['outputs'][label]['path'],allow_pickle=False) for label in ('seed42','seed43')}
        evaluated={label:np.load(output/metadata['outputs'][label+'-evaluated-mask']['path'],allow_pickle=False) for label in ('seed42','seed43')}
        for label in ('seed42','seed43'):
            assert sha(output/metadata['outputs'][label]['path'])==metadata['outputs'][label]['sha256']
            assert metadata['models'][0 if label=='seed42' else 1]['strict_state_match']
        observed=np.any(surface[7:24]!=0,axis=0)
        central=np.zeros((256,256),dtype=bool)
        central[64:192,64:192]=True
        common=central & mask & observed & evaluated['seed42'] & evaluated['seed43']
        assert np.isfinite(scores['seed42'][common]).all() and np.isfinite(scores['seed43'][common]).all()
        x,y,w,h=selection['bbox_source_xywh']
        record={'id':name,'bbox_source_xywh':selection['bbox_source_xywh'],
                'selection_geometry_valid_fraction':selection['geometry_valid_fraction'],
                'selection_observed_fraction_among_geometry_valid':selection['observed_fraction_among_geometry_valid'],
                'central_source_xywh':[x+64,y+64,128,128],
                'central_common_pixel_count':int(common.sum()),'central_possible_pixel_count':16384,
                'seed_summaries':{},'context_consistency_vs_full_field':{},
                'inference_seconds_by_seed':{entry['label']:entry['inference_seconds'] for entry in metadata['models']}}
        for label in ('seed42','seed43'):
            record['seed_summaries'][label]=summary(scores[label][common])
            full_crop=baseline[label][y:y+h,x:x+w]
            comparable=common & np.isfinite(full_crop)
            difference=np.abs(scores[label][comparable]-full_crop[comparable])
            record['context_consistency_vs_full_field'][label]={'pixel_count':int(comparable.sum()),
              'maximum_absolute_difference':float(difference.max()) if difference.size else None,
              'mean_absolute_difference':float(difference.mean()) if difference.size else None,
              'bitwise_equal_on_common_pixels':bool(np.array_equal(scores[label][comparable],full_crop[comparable]))}
        joint=np.minimum(scores['seed42'],scores['seed43'])
        record['minimum_seed_score_summary']=summary(joint[common])
        record['absolute_seed_difference']=summary(np.abs(scores['seed42'][common]-scores['seed43'][common]))
        record['central_geometry_valid_pixels']=int((central&mask).sum())
        record['central_geometry_observed_pixels']=int((central&mask&observed).sum())
        if common.sum()>1 and all(scores[label][common].std()>0 for label in ('seed42','seed43')):
            record['pearson_seed_score_correlation']=float(np.corrcoef(scores['seed42'][common],scores['seed43'][common])[0,1])
        records.append(record)
        fig,axes=plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
        fig.suptitle(f'{name}: frozen geometry/data-only context; source XY {x},{y}\nPreviously scored field; no ground truth, letter reading or accuracy claim',fontsize=11)
        extent=[x,x+w,y+h,y]
        cmap=plt.get_cmap('gray').copy()
        cmap.set_bad('white')
        for axis,layer in zip(axes[0],(11,15,19)):
            view=np.where(mask,surface[layer],np.nan)
            axis.imshow(view,cmap=cmap,vmin=ct_low,vmax=ct_high,extent=extent)
            axis.set_title(f'Observed layer {layer}; shared CT display limits')
        for axis,values,title in zip(axes[1],(scores['seed42'],scores['seed43'],joint),('Seed42','Seed43','Minimum seed score')):
            axis.imshow(np.clip((values-.25)/.5,0,1),cmap=cmap,vmin=0,vmax=1,extent=extent)
            axis.set_title(title+'; display (p−.25)/.5')
        for axis in axes.flat:
            axis.add_patch(Rectangle((x+64,y+64),128,128,fill=False,edgecolor='cyan',linewidth=1))
            axis.set_xlabel('Source x');axis.set_ylabel('Source y')
        figure_path=REPORT/(name+'-review.png')
        if figure_path.exists():raise SystemExit('Preserving existing plot: '+str(figure_path))
        fig.savefig(figure_path,dpi=150);plt.close(fig)
    fig,axis=plt.subplots(figsize=(7,7),constrained_layout=True)
    axis.imshow(np.where(geometry,full_surface[15],np.nan),cmap='gray',vmin=ct_low,vmax=ct_high)
    for bbox,label in zip(plan['criteria']['excluded_bboxes_source_xywh'],('Excluded pilot','Excluded selected hotspot')):
        x,y,w,h=bbox
        axis.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='red',linewidth=1,linestyle='--'))
        axis.text(x,y,label,color='red',fontsize=8)
    for selection in plan['selected']:
        x,y,w,h=selection['bbox_source_xywh']
        axis.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='cyan',linewidth=1.5))
        axis.text(x+5,y+20,selection['id'],color='cyan',fontsize=9)
    axis.set_title('Frozen geometry/data-only selections on a previously scored field\nObserved layer15; cyan contexts, red excluded boxes')
    axis.set_xlabel('Native source x');axis.set_ylabel('Native source y')
    overview=REPORT/'independent-selection-overview.png'
    if overview.exists():raise SystemExit('Preserving existing overview')
    fig.savefig(overview,dpi=150);plt.close(fig)
    result={'selection_plan_sha256':sha(plan_path),'selection_plan_frozen_at_utc':plan['frozen_at_utc'],
            'summary_domain':'central128 AND geometry-valid AND both seeds evaluated AND any observed raw selected-depth data',
            'raw_midpoint_threshold':.5,'threshold_is_not_accuracy_or_authenticity':True,
            'shared_source_display_limits':{'lower':ct_low,'upper':ct_high,'derivation':'global1/99percentiles of geometry-valid full-field pixels in layers11,15,19'},
            'model_display':'clip((p-.25)/.5,0,1), same[0,1]scale in every model panel; all summaries use raw unscaled scores',
            'contexts':records,'analysis_code_sha256':sha(__file__)}
    path=REPORT/'independent-regions-summary.json'
    if path.exists():raise SystemExit('Preserving existing summary')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
