"""Different arithmetic plus a declared boundary-isolation follow-up."""
import argparse
import json
from pathlib import Path
import numpy as np
import nrrd
from scipy import ndimage


def direct_scores(g, p):
    total=int(np.count_nonzero(g));miss=int(np.count_nonzero((g>0)&(p==0)))
    merge=split=0;mixed=split_ids=0
    for label in np.unique(p[g>0]):
        if label==0:continue
        values=g[p==label];values=values[values>0]
        _,counts=np.unique(values,return_counts=True)
        merge+=int(counts.sum()-counts.max());mixed+=int(len(counts)>1)
    for label in np.unique(g):
        if label==0:continue
        values=p[g==label];values=values[values>0]
        if len(values):
            _,counts=np.unique(values,return_counts=True)
            split+=int(counts.sum()-counts.max());split_ids+=int(len(counts)>1)
    return {'reference_voxels':total,'abstained_reference_fraction':miss/total,
            'merge_minority_fraction_of_reference':merge/total,
            'fragmented_survivor_fraction_of_reference':split/total,
            'mixed_prediction_instances':mixed,'split_reference_instances':split_ids}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--primary',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    names=['00768-02152-03536','00768-02408-03536']
    masks=[nrrd.read(str(a.inputs/(n+'-mask.raw')),index_order='F')[0] for n in names]
    truth=np.concatenate(masks,axis=1)
    p,count=ndimage.label(truth>0)
    observed=direct_scores(truth,p)
    expected=json.loads(a.primary.read_text())['scores']['6-connected']
    for key,value in observed.items():assert abs(value-expected[key])<1e-12,key
    del p
    local=[];separate=[];offset=0
    for name,g in zip(names,masks):
        p,count=ndimage.label(g>0)
        local.append({'cube':name,**direct_scores(g,p),'components':int(count)})
        p[p>0]+=offset;offset+=count;separate.append(p)
    blockwise=direct_scores(truth,np.concatenate(separate,axis=1))
    out={'status':'PASS','primary_arithmetic_matches':True,'joined':observed,
         'per_cube':local,'cross_file_connectivity_disabled':blockwise,
         'followup_scope':'Post-primary development control: test whether identity mixing is only caused by joining the two files. All original labelled voxels retained; no image-based reselection.',
         'limits':'Connected-components/reference diagnostic only. Not an independently annotated native topology benchmark or a model improvement.'}
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
