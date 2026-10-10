"""Localize completed-composite differences to recorded row annotations."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
from PIL import Image

ap=argparse.ArgumentParser()
ap.add_argument('--lab',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
a=ap.parse_args()
p=a.lab/'reports/fortysixth-experiment'
old=a.lab/'reports/fortyfifth-experiment/source-evidence'
stem='concat_w047-w048_R5B2_z9500-11000'
script=old/'pherc343/ink/scripts/make_submission_image.py'
spec=importlib.util.spec_from_file_location('original_composer',script)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
images=[];masks=[];sides=[]
for root,meta in [(old,old/'pherc343/outputs'),(p/'work/image',p/'work/image')]:
    side=json.loads((meta/(stem+'.json')).read_text());sides.append(side)
    b=side['outputs']['image']['data_panel']
    im=np.array(Image.open(root/(stem+'.png')).convert('RGB'))
    images.append(im[b['y']:b['y']+b['height'],b['x']:b['x']+b['width']])
    rows=side['rows'];mask=np.zeros((b['height'],b['width']),np.uint8)
    assert rows['dash']=='solid'
    mod.draw_rows(mask,rows['rows_area_px'],0,0,255,rows['thickness_px'],None,aa=True)
    masks.append(mask>0)
valid=np.load(p/'work/valid.npy')[105:2560,380:6021][::-1,::-1]
d=np.abs(images[0].astype(np.int16)-images[1].astype(np.int16))
union=(masks[0]|masks[1])&valid
remaining=valid&~union
maximum=d[valid].max()
peak=(d.max(2)==maximum)&valid
report={'status':'PASS','annotation_source_equal':sides[0]['rows']['sha256']==sides[1]['rows']['sha256'],
        'derived_paths_equal':sides[0]['rows']['rows_area_px']==sides[1]['rows']['rows_area_px'],
        'maximum_difference':int(maximum),'all_maxima_on_recorded_row_coverage':bool(union[peak].all()),
        'union_row_coverage_valid_pixels':int(union.sum()),
        'outside_recorded_row_coverage':{'n':int(remaining.sum()),'rgb_mae':float(d[remaining].mean()),'max':int(d[remaining].max())},
        'scope':'Post-hoc diagnosis of the maximum RGB difference; original full-panel endpoint unchanged. Original row rasterizer, saved sidecar paths, no model or composition run.'}
with a.out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(report,indent=2))
