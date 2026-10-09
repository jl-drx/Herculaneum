"""Independent PNG crop, explicit Gaussian and rank-statistic reconstruction."""
import gc
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import resource
import sys
import time
import numpy as np
from PIL import Image,__version__ as pillow_version
from independent_math import block_average,gaussian_reflect,ranking
from reconstruct_ct import PINS,read_pin,read_array,digest,json_dump
from reconstruct_ppm import NEWPINS

P=Path('/workspace/herculaneum-lab/reports/seventeenth-experiment')
RESULT_SHA='4083f0b3100eb35a512655252b0d3c37a2c0239ce683daf26ee9ddd5aead2b27'


def main():
    start=time.monotonic();cpu=time.process_time();resource.setrlimit(resource.RLIMIT_CPU,(1200,1200));resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,12*1024**3))
    for p,s in {**PINS,**NEWPINS}.items():assert digest(p.read_bytes())==s,str(p)
    authorpath=P/'implementation/ir-run/ir-diagnostics.json';assert digest(authorpath.read_bytes())==RESULT_SHA
    author=json.loads(authorpath.read_text());assert author['status']=='PASS'
    desc=json.loads(next(p for p in PINS if p.name=='pooled-descriptor.json').read_text())
    header=json.loads((P/'acquisition/header-run/result.json').read_text())
    objects={o['object_id']:o for o in header['objects']if o['object_id'].endswith('-ir')}
    inventory={o['object_id']:o for o in header['fixed_inventory']['objects']}
    by_case={c['id']:c for c in author['cases']}
    output=P/'independent-review/ir-reconstruction';output.mkdir(exist_ok=False)
    runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,pillow=pillow_version,
                 scipy_installed=importlib.metadata.version('scipy'),scipy_functions_imported=False,author_helpers_imported=False)
    json_dump(output/'start-receipt.json',dict(runtime=runtime,input_pins={str(k):v for k,v in {**PINS,**NEWPINS}.items()},
        author_result_sha256=RESULT_SHA,source_sha256={q.name:digest(q.read_bytes())for q in[Path(__file__),Path(__file__).with_name('independent_math.py')]}))
    rows=[];images=[]
    Image.MAX_IMAGE_PIXELS=150000000
    for oid,obj in objects.items():
        body=read_pin(obj['file']);expected=inventory[oid]['shape_yx']
        with Image.open(io.BytesIO(body))as im:
            assert im.format=='PNG'and [im.height,im.width]==expected and im.height*im.width<=150000000
            assert im.mode in ['L','RGB','RGBA','P']
            im.load()
            if im.mode in ['RGBA','P']or 'transparency'in im.info:
                assert im.convert('RGBA').getchannel('A').getextrema()==(255,255)
            images.append(dict(object_id=oid,mode=im.mode,shape_yx=expected,fully_opaque=True,source_sha256=obj['file']['sha256']))
            for case in [c for c in desc['cases']if c['id'].split('-')[0]==oid.split('-')[0]]:
                ac=by_case[case['id']];x,y,w,h=case['bbox_source_xywh'];assert ac['bbox_source_xywh']==[x,y,w,h]
                gray_native=np.asarray(im.crop((x,y,x+w,y+h)).convert('L'),dtype=np.uint8)
                gray=block_average(gray_native);dark=255-gray
                blurred=gaussian_reflect(gray_native,sigma=16,radius=64)
                local=block_average(blurred-gray_native.astype(np.float64))
                primary=read_array(case['arrays']['primary_mask']);labels=read_array(case['arrays']['labels'])
                assert primary.dtype==np.bool_ and labels.dtype==np.uint8 and primary.shape==labels.shape==(384,384)
                comparisons={};metrics={}
                for key,array in [('gray',gray),('darkness',dark),('local_darkness',local)]:
                    saved=read_array(ac['arrays'][key]);assert saved.dtype==np.float64 and saved.shape==(384,384)
                    difference=np.abs(saved-array);maxdiff=float(difference.max())
                    assert np.allclose(saved,array,atol=1e-9,rtol=1e-10), (case['id'],key,maxdiff)
                    comparisons[key]=dict(voxels=int(saved.size),max_absolute_difference=maxdiff,exact_equal=bool(np.array_equal(saved,array)))
                    if key!='gray':
                        r=ranking(array[primary],labels[primary]);expected_r=ac[key]
                        assert r['count']==expected_r['count']==ac['primary_support_count']
                        assert r['positive_count']==expected_r['positive']and r['negative_count']==expected_r['negative']
                        for own,k in [('roc_auc','auroc'),('average_precision','average_precision')]:
                            assert (r[own]is None and expected_r[k]is None)or math.isclose(r[own],expected_r[k],rel_tol=0,abs_tol=1e-12),(case['id'],key,k,r[own],expected_r[k])
                        groups=len(np.unique(array[primary]));assert groups==expected_r['distinct_score_groups'],(case['id'],key,groups,expected_r['distinct_score_groups'])
                        r['distinct_score_groups']=groups;metrics[key]=r
                rows.append(dict(id=case['id'],array_comparisons=comparisons,metrics=metrics));print(case['id'],'PASS',flush=True)
        del body;gc.collect()
    r=dict(status='PASS',images=images,cases=rows,runtime=runtime,array_tolerance=dict(atol=1e-9,rtol=1e-10),ranking_atol=1e-12,
           array_maps=18,array_values=18*384*384,wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,
           peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,author_result_sha256=RESULT_SHA,network_calls=0,model_calls=0)
    assert r['wall_seconds']<=600 and r['cpu_seconds']<=1200 and r['peak_rss_bytes']<=3*1024**3
    json_dump(output/'independent-ir-verification.json',r)
    print(json.dumps({k:v for k,v in r.items()if k not in ['cases','images']},indent=2))

if __name__=='__main__':main()
