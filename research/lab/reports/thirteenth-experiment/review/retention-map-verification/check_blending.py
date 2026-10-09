"""Independent NumPy-only retained-logit blending; no inference or model imports."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLAN = ROOT/'retention/final-plan/retention-plan.json'
INDEX = ROOT/'retention/run/results.json'
PLAN_SHA = 'eb75f743a99d9707219689db1893df40947a7467b98d8c326b527c9f5b51f47a'
INDEX_SHA = '2d8d87916a6e4b6bed04dd0437e9f2a7c02c91729de657b6735df65305c22e8e'
TOLERANCE = 2e-6
ARMS = ['A_preserve', 'B_preserve']
CASES = ['pherc0814-46527-fresh-rank01', 'pherc0814-46527-fresh-rank14', 'frag4-context1', 'frag4-context2']
GRID = [[x,y,128,128] for y in range(0,257,64) for x in range(0,257,64)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).resolve()
    require(path.suffix.lower() not in {'.pth','.pt','.ckpt','.bin','.safetensors'}, 'No checkpoint reads allowed')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=digest)


def weights():
    # Mathematical nonperiodic Hann, evaluated independently in float64.
    w = (1.-np.cos(2.*np.pi*np.arange(128,dtype=np.float64)/127.))/2.
    outer = np.outer(w,w)
    outer /= max(float(outer.max()),np.finfo(np.float32).eps)
    return np.maximum(outer,.001).astype(np.float32)


def sigmoid(logits):
    z = np.asarray(logits,dtype=np.float64)
    return np.exp(-np.logaddexp(0.,-z)).astype(np.float32)


def reconstruct(logits):
    require(logits.dtype == np.float32 and logits.shape == (25,128,128) and np.isfinite(logits).all(), 'Finite FP32 tile logits required')
    w = weights()
    sums = np.zeros((384,384),dtype=np.float32)
    denominator = np.zeros_like(sums)
    probabilities = sigmoid(logits)
    for bbox,p in zip(GRID,probabilities):
        x,y,width,height = bbox
        sums[y:y+height,x:x+width] += p*w
        denominator[y:y+height,x:x+width] += w
    out = np.zeros_like(sums)
    np.divide(sums,denominator,out=out,where=denominator>0)
    return out,denominator>0


def fixtures():
    zero = np.zeros((25,128,128),dtype=np.float32)
    score,mask = reconstruct(zero)
    require(mask.all() and np.array_equal(score,np.full((384,384),.5,dtype=np.float32)), 'Zero-logit blend or coverage differs')
    yy,xx = np.indices((128,128),dtype=np.float64)
    varied = np.stack([(i-12)/5.+xx/97.-yy/71. for i in range(25)]).astype(np.float32)
    mixed,valid = reconstruct(varied)
    w = weights()
    points = [(0,0),(127,127),(128,128),(191,255),(320,320),(383,383)]
    for y,x in points:
        numerator = 0.; denominator = 0.
        for i,(left,top,width,height) in enumerate(GRID):
            if left <= x < left+width and top <= y < top+height:
                z = float(varied[i,y-top,x-left])
                p = float(np.exp(-np.logaddexp(0.,-z)))
                weight = float(w[y-top,x-left])
                numerator += p*weight; denominator += weight
        require(abs(float(mixed[y,x])-numerator/denominator)<3e-7, 'Independent pointwise blending differs')
    require(valid.all() and w.dtype==np.float32 and float(w.min())>0, 'Positive floor/coverage differs')
    require(np.array_equal(sigmoid(np.array([-1000.,0.,1000.],dtype=np.float32)),np.array([0.,.5,1.],dtype=np.float32)), 'Stable sigmoid extremes differ')
    return dict(status='PASS',zero_logit_exact_map_cases=1,independent_pointwise_cases=len(points),sigmoid_extreme_cases=1,actual_retained_array_reads=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--authorized-retained-read',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    require(not args.output.exists() and 'torch' not in sys.modules,'Fresh output and no Torch required')
    source = record(__file__)
    tests = fixtures()
    if args.self_test:
        args.output.write_text(json.dumps(dict(status='PASS',source=source,fixtures=tests,model_calls=0,checkpoint_reads=0),indent=2,sort_keys=True)+'\n')
        return
    require(args.authorized_retained_read,'Completed-run authorization required')
    method = json.loads((HERE/'METHOD.json').read_text())
    require(method['checker']==source and method['maximum_absolute_score_error']==TOLERANCE,'Method/source changed after predeclaration')
    pins = {}
    def verify(item):
        actual = record(item['path'])
        require(actual==item,'File identity differs: '+str(item['path']))
        pins[item['path']] = actual
        return Path(item['path'])
    def read(item):
        return json.loads(verify(item).read_text())
    pp,ip = record(PLAN),record(INDEX)
    require(pp['sha256']==PLAN_SHA and ip['sha256']==INDEX_SHA,'Root entry pins differ')
    plan,index = read(pp),read(ip)
    require(index['status']=='PASS' and index['plan']==pp and index['eligible_arms']==ARMS and index['tile_forward_count']==200 and index['retries']==0,'Run scope differs')
    require(plan['eligible_arms']==ARMS and [c['id'] for c in plan['cases']]==CASES,'Frozen scope differs')
    require(plan['inference']['direction']=='forward' and plan['inference']['mask_passed_to_adapter'] is False,'Inference geometry differs')
    checks = []; total_tiles = 0
    for arm in ARMS:
        worker = read(index['models'][arm])
        require(worker['status']=='PASS' and worker['arm']==arm and worker['plan']==pp and set(worker['runs'])==set(CASES),'Worker lineage differs')
        require(worker['state_sha256']==plan['models'][arm]['state_sha256'] and worker['model_state_unchanged'] is True,'Worker state receipt differs')
        for case in plan['cases']:
            item = worker['runs'][case['id']]
            tiles = item['inference']['tiles']
            require(len(tiles)==25 and all(t['status']=='scored' for t in tiles),'Scored tile accounting differs')
            require([t['bbox_roi_xywh'] for t in tiles]==GRID==[t['bbox_roi_xywh'] for t in item['captured_tiles']],'Tile placement/order differs')
            require(item['adapter_direction']=='forward' and item['adapter_layer_start']==case['layer_start'],'Depth metadata differs')
            arrays = {name:np.load(verify(item[name]),allow_pickle=False) for name in ['scores','evaluated','logits']}
            actual,evaluated,logits = arrays['scores'],arrays['evaluated'],arrays['logits']
            require(actual.dtype==np.float32 and actual.shape==(384,384) and np.isfinite(actual).all() and np.all((actual>=0)&(actual<=1)),'Saved score contract differs')
            require(evaluated.dtype==np.bool_ and evaluated.shape==(384,384),'Saved coverage contract differs')
            rebuilt,coverage = reconstruct(logits)
            require(np.array_equal(evaluated,coverage),'Evaluated mask differs exactly')
            error = np.abs(rebuilt.astype(np.float64)-actual.astype(np.float64))
            maximum = float(error.max())
            require(maximum<=TOLERANCE,'Independent blending error exceeds predeclared bound')
            equality = rebuilt.view(np.uint32)==actual.view(np.uint32)
            checks.append(dict(arm=arm,context=case['id'],family=case['family'],tiles=25,
                maximum_absolute_score_error=maximum,mean_absolute_score_error=float(error.mean()),
                root_mean_square_score_error=float(np.sqrt(np.mean(error*error))),score_bitwise_equal=bool(equality.all()),
                bitwise_equal_pixels=int(equality.sum()),pixels=int(equality.size),evaluated_mask_exact=True,
                evaluated_pixels=int(coverage.sum()),source_records={name:item[name] for name in ['scores','evaluated','logits']}))
            total_tiles += len(logits)
    require(len(checks)==8 and total_tiles==200,'Complete map scope differs')
    for item in pins.values():
        require(record(item['path'])==item,'Input changed during reconstruction')
    require(record(__file__)==source and 'torch' not in sys.modules,'Checker/source scope changed')
    result = dict(status='PASS',method=record(HERE/'METHOD.json'),source=source,plan=pp,run_index=ip,
        fixtures=tests,maps_verified=len(checks),tiles_verified=total_tiles,map_checks=checks,
        maximum_absolute_score_error=max(c['maximum_absolute_score_error'] for c in checks),
        all_score_maps_bitwise_equal=all(c['score_bitwise_equal'] for c in checks),all_evaluated_masks_exact=True,
        maximum_absolute_score_error_limit=TOLERANCE,verified_inputs=list(pins.values()),
        limitation='Independent float64 mathematical sigmoid/Hann kernels, rounded to FP32 before FP32 accumulation; original Torch transcendental kernels are not replayed. Exact score bit identity is reported, not assumed.',
        model_calls=0,Torch_imports=0,checkpoint_reads=0,network_requests=0)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(status='PASS',maps=8,tiles=200,maximum_absolute_score_error=result['maximum_absolute_score_error'],score_bitwise_equal=result['all_score_maps_bitwise_equal'],receipt=record(args.output))))


if __name__=='__main__':
    main()
