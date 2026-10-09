"""Generated-array and toy-model fixtures; never constructs a real factory."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
import torch
from parity_sources import (INPUT, LAB, pin, load_module, historical_inference, namespace,
                            inactive_stub, STUB_COUNTS, config_guards)
from parity_inference import (synthetic_volume, selected_indices, historical_case,
                              current_case, RecordingModel)
from run_parity import state_receipt, write_json


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls=0
    def forward(self, x):
        self.calls+=1
        return {'ink': (x[:,:,0]*.15 + x[:,:,8]*.25 - x[:,:,16]*.1).contiguous()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    rows=[]
    def check(name,condition):
        if not condition:
            raise AssertionError(name)
        rows.append({'name':name,'status':'PASS'})
    def rejects(name,fn):
        try:
            fn()
        except (ValueError,RuntimeError):
            rows.append({'name':name,'status':'PASS'})
            return
        raise AssertionError(name)
    torch.set_num_threads(4)
    closure=json.loads((INPUT/'HISTORICAL_FACTORY_CLOSURE.json').read_text())
    historical=historical_inference(closure['historical_inference_source_for_reviewed_pure_extraction'])
    for name in ('vesuvius','vesuvius.image_proc','vesuvius.image_proc.intensity'):
        namespace(name)
    oldnorm=load_module('vesuvius.image_proc.intensity.normalization',closure['normalization_module_for_separate_input_parity']).normalize_robust
    currentnorm=load_module('phase22_current_normalization_fixture',pin(LAB/'vendor/villa-ink-baseline/vesuvius/image_proc/intensity/normalization.py')).normalize_robust
    baseline=load_module('phase22_baseline_fixture',pin(LAB/'run_ink_baseline.py'))
    for a,b,expected in [(0,28,list(range(6,23))),(0,21,list(range(2,19))),
        (0,20,list(range(2,19))),(3,23,list(range(5,22))),
        (5,25,list(range(7,24))),(8,27,list(range(9,26))),
        (0,17,list(range(17))),(0,16,list(range(16)))]:
        check('center_%d_%d'%(a,b),historical.center_crop_layer_indices(np.arange(a,b),output_depth=17).tolist()==expected)
    rejects('invalid_depth',lambda: historical.center_crop_layer_indices(np.arange(20),output_depth=0))
    reverse={'interval':[0,28],'reverse':True}
    check('reverse_after_selection',selected_indices(reverse).tolist()==list(range(22,5,-1)))
    exact=synthetic_volume(3,4,5)
    check('root_uint8_scalar_formula',exact.dtype==np.uint8 and all(
        int(exact[z,y,x])==1+((17*z+3*y+5*x+(y*x)%29+(z*y)%23)%251)
        for z in range(3) for y in range(4) for x in range(5)))
    old_input=np.arange(33,dtype=np.float32);new_input=old_input.copy();old_before=old_input.copy()
    a=oldnorm(old_input);b=currentnorm(new_input)
    check('normalization_values_equal',a.tobytes()==b.tobytes())
    check('historical_float32_may_alias',np.shares_memory(a,old_input) and not np.array_equal(old_input,old_before))
    check('current_float32_copies',not np.shares_memory(b,new_input) and np.array_equal(new_input,old_before))
    for value in (0.,17.,255.):
        check('constant_normalization_'+str(value),np.array_equal(oldnorm(np.full((3,5,5),value,np.float32)),np.zeros((3,5,5),np.float32)))
    # Reader exercises depth and spatial padding before normalizing, using exact source.
    vol=np.arange(7*4*5,dtype=np.uint8).reshape(7,4,5)
    reader=historical.OmeZarrPatchReader(input_path='<synthetic-memory>',resolution='0',depth_axis_first=True,
        height=4,width=5,layer_indices=np.arange(7),output_depth=17,preprocessing='tifxyz_robust')
    reader._array=vol
    padded=reader.read(-1,-2,8,9)
    check('depth_pad_center',np.array_equal(padded[1:5,2:7,5:12],vol.transpose(1,2,0)))
    expected=np.zeros((8,9,17),np.float32);expected[1:5,2:7,5:12]=vol.transpose(1,2,0)
    check('spatial_and_depth_pad_zero',np.array_equal(padded,expected))
    check('padding_normalized_in_scope',np.array_equal(oldnorm(np.moveaxis(padded.copy(),-1,0)),currentnorm(np.moveaxis(expected,-1,0))))
    rejects('insufficient_output_depth',lambda:historical.OmeZarrPatchReader(input_path='x',resolution='0',depth_axis_first=True,
        height=4,width=5,layer_indices=np.arange(7),output_depth=6))
    indices=historical.center_crop_layer_indices(np.arange(28),output_depth=17)
    bad_center=np.arange(5,22)
    check('wrong_even_center_detected',not np.array_equal(indices,bad_center))
    for name,shape,zero,rev,fw in [('toy-single',[28,128,128],False,False,1),
        ('toy-reverse',[28,128,128],False,True,1),('toy-tiled',[28,192,192],False,False,4),
        ('toy-zero',[28,128,128],True,False,0)]:
        case={'name':name,'shape':shape,'interval':[0,28],'reverse':rev,'zero':zero,'forwards':fw}
        volume=synthetic_volume(*shape,zero=zero);before=volume.copy()
        ma=Toy().eval();mb=Toy().eval()
        aa,am=historical_case(ma,oldnorm,historical,case,volume)
        bb,bm=current_case(mb,currentnorm,baseline,case,volume)
        check(name+'_exact_forwards',ma.calls==mb.calls==fw)
        check(name+'_input_unchanged',np.array_equal(volume,before))
        for key in ('selected_indices','normalized','logits','probabilities','hann','blocks',
                    'probability_sum','weight_sum','raw_score','uint8','author_clipped_score','evaluated'):
            check(name+'_'+key+'_exact',aa[key].tobytes()==bb[key].tobytes())
        if zero:
            check('zero_skip_unevaluated',not aa['evaluated'].any() and not aa['raw_score'].any())
        if fw==4:
            check('overlap_counts_nonuniform',aa['weight_sum'][64,64]>1 and aa['weight_sum'][0,0]==np.float32(.001))
    probs=np.array([[-.5,0,.25,.5,1,1.5]],np.float32)
    encoded=list(historical.iter_probability_tiles(probs.copy(),np.ones_like(probs),(1,6)))[0]
    check('quantized_clipped_endpoints',encoded.tolist()==[[0,0,63,127,255,255]])
    decoded=np.clip((encoded.astype(np.float32)-64)/128,0,1)
    check('author_decoded_endpoints',decoded.tolist()==[[0,0,0,63/128,1,1]])
    class AliasToy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.a=torch.nn.Parameter(torch.tensor([1.,-0.]));self.b=self.a
    model=AliasToy();state={k:v.clone() for k,v in model.state_dict().items()}
    schema={'tensor_schema':[{'name':k,'shape':[2]} for k in ('a','b')],
        'expected_factory_alias_groups':[{'names':['a','b']}], 'retained_factory_unique_storage_groups':1,
        'expected_named_tensor_count':2,'expected_named_numel':4,'retained_factory_unique_parameters':2}
    receipt=state_receipt(model,state,schema)
    check('toy_alias_and_values_positive',receipt['unique_groups']==1 and receipt['alias_pairs']==[['a','b']])
    mutated={k:v.clone() for k,v in state.items()};mutated['b'][1]=0.
    rejects('alias_signed_zero_mismatch',lambda:state_receipt(model,mutated,schema))
    rejects('missing_key',lambda:state_receipt(model,{'a':state['a']},schema))
    rejects('unexpected_key',lambda:state_receipt(model,{**state,'extra':state['a']},schema))
    wrong=copy.deepcopy(schema);wrong['tensor_schema'][0]['shape']=[1,2]
    rejects('wrong_shape',lambda:state_receipt(model,state,wrong))
    wrong=copy.deepcopy(schema);wrong['expected_named_numel']=5
    rejects('wrong_element_count',lambda:state_receipt(model,state,wrong))
    wrong=copy.deepcopy(schema);wrong['expected_factory_alias_groups']=[]
    rejects('wrong_alias_partition',lambda:state_receipt(model,state,wrong))
    first=receipt['semantic_value_sha256']
    with torch.no_grad():model.a[0]=2.
    new={k:v.clone() for k,v in model.state_dict().items()}
    check('state_hash_changes',state_receipt(model,new,schema)['semantic_value_sha256']!=first)
    inactive_stub('phase22_fixture_inactive',['BlockedClass','blocked_function'])
    stub=sys.modules['phase22_fixture_inactive']
    rejects('inactive_constructor_fails',lambda:stub.BlockedClass())
    rejects('inactive_function_fails',lambda:stub.blocked_function())
    rejects('inactive_instancecheck_fails',lambda:isinstance(object(),stub.BlockedClass))
    check('stub_counts_recorded',sum(STUB_COUNTS['runtime_calls'].values())==3)
    output=dict(status='PASS',checks=rows,check_count=len(rows),real_factory_constructions=0,
        real_checkpoint_loads=0,real_model_forwards=0,toy_model_forwards=12,scientific_array_reads=0)
    write_json(args.output,output)
    print(json.dumps({'status':'PASS','checks':len(rows),'output':args.output}))


if __name__=='__main__':main()
