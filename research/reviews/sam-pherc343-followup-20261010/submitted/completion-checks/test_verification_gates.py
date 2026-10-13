#!/usr/bin/env python3
"""Synthetic-only tests; no real maps, checkpoints, model forwards or resampling."""
import copy, json, tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import verify_final_maps as v

results=[]
def test(label, operation, rejection=False):
    try:
        operation()
    except (ValueError, KeyError, FileNotFoundError) as exc:
        if not rejection:
            raise
        results.append({'case':label,'result':'PASS','expected_rejection':str(exc)})
    else:
        if rejection:
            raise AssertionError(label+': invalid fixture accepted')
        results.append({'case':label,'result':'PASS'})

with tempfile.TemporaryDirectory(prefix='pherc343-synthetic-') as d:
    d=Path(d)
    a=np.array([[0,1,127],[128,254,255]],dtype=np.uint8)
    ck={'sha256':'1'*64,'bytes':123}
    for ext in ['png','tif']:
        f=d/('synthetic.'+ext)
        Image.fromarray(a).save(f)
        r={'status':'PASS','model':'hecate','phase':'plain','shape':[2,3], 'output':{'path':str(f),'bytes':f.stat().st_size,'sha256':v.sha(f)},'decoded_sha256':v.decoded(a),'checkpoint':ck}
        test(ext+': valid uint8 receipt accepted', lambda:v.validate_final_file(f,r,'hecate','plain',(2,3),ck))
        for key in ['encoded','decoded','shape','bytes','partial','checkpoint']:
            bad=copy.deepcopy(r)
            if key=='encoded': bad['output']['sha256']='0'*64
            if key=='decoded': bad['decoded_sha256']='0'*64
            if key=='shape': bad['shape']=[3,2]
            if key=='bytes': bad['output']['bytes']+=1
            if key=='partial': bad['status']='PARTIAL'; bad['saved_rows']=1
            if key=='checkpoint': bad['checkpoint']['sha256']='2'*64
            test(ext+': reject '+key, lambda:v.validate_final_file(f,bad,'hecate','plain',(2,3),ck),True)
        test(ext+': reject short prefix image as full output', lambda:v.validate_final_file(f,r,'hecate','plain',(3,3),ck),True)
    f=d/'wrong-dtype.png'
    Image.fromarray(a.astype(np.uint16)).save(f)
    test('reject uint16 image',lambda:v.validate_final_file(f,r,'hecate','plain',(2,3),ck),True)
    complete={'status':'PASS','maps':[{'model':m,'phase':p,'status':'PASS'} for m in v.MODELS for p in ['plain','padded']]}
    test('accept eight unique final receipt identities',lambda:v.validate_completion(complete))
    partial=copy.deepcopy(complete); partial['maps'].pop()
    test('reject seven-map completion',lambda:v.validate_completion(partial),True)
    prefix={'status':'PASS','generation':'synthetic','saved_rows':2304,'decoded_sha256':'0'*64}
    test('reject checkpoint manifest even if all rows claimed',lambda:v.validate_completion(prefix),True)
    dup=copy.deepcopy(complete); dup['maps'][-1]=dup['maps'][0]
    test('reject duplicate model-phase pair',lambda:v.validate_completion(dup),True)

print(json.dumps({'status':'PASS','scope':'Synthetic tests of shared validation functions only; no completed Hecate outputs accessed; no model execution; no OpenCV merge/resampling test.', 'cases':results},indent=2))
