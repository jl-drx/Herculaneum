"""Independent PHerc0139 schedule/support audit; no CT, logits or model reads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
sys.dont_write_bytecode=True
import numpy as np

HERE=Path(__file__).resolve().parent
REPORTS=HERE.parent.parent
ENTRY=REPORTS/'twentieth-experiment/implementation/final-plan/experiment-plan.json'
ENTRY_SHA='58bb5c64ad64db19995c87c84d1e6a1ce6585d380afeaa7ef347ac82b138d78a'
IDS=['pherc0139-w016-rank01','pherc0139-w016-rank02']


def denied(*args,**kwargs):
    raise RuntimeError('Network forbidden in support review')


socket.create_connection=denied
socket.socket.connect=denied
socket.socket.connect_ex=denied


def require(condition,message):
    if not condition:raise ValueError(message)


def record(path):
    path=Path(path).resolve(strict=True)
    require(path.is_relative_to(REPORTS) and path.suffix in {'.json','.jsonl','.npy','.py','.md'},
            'Only local report/metadata/support-array files permitted')
    return dict(path=str(path),size_bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class Reader:
    def __init__(self):self.pins={};self.array_reads=0
    def verify(self,pin):
        actual=record(pin['path'])
        require(actual['sha256']==pin['sha256'] and actual['size_bytes']==pin.get('size_bytes',pin.get('byte_length')),
                'Pinned source changed: '+pin['path'])
        self.pins[actual['path']]=actual
        return Path(actual['path'])
    def json(self,pin):return json.loads(self.verify(pin).read_text())
    def jsonl(self,pin):return [json.loads(line) for line in self.verify(pin).read_text().splitlines()]
    def array(self,pin):
        path=self.verify(pin)
        require(path.suffix=='.npy' and any(name in path.parts for name in IDS), 'Only named PHerc0139 support arrays permitted')
        require(path.name in {'acquisition-observed-mask.npy','source-observed-nonzero-mask.npy','supervision_mask.npy','validation_mask.npy'},
                'CT, labels, teacher/student logits and other arrays forbidden')
        value=np.load(path,allow_pickle=False)
        require(value.shape==(384,384) and (value.dtype==np.bool_ or value.dtype==np.uint8) and
                np.isin(value,[0,1]).all(), 'Support encoding differs')
        self.array_reads+=1
        return value.astype(bool)
    def finish(self):
        for item in self.pins.values():require(record(item['path'])==item,'Source changed during review')
        require('torch' not in sys.modules,'Torch forbidden in support review')
        return dict(input_pins=list(self.pins.values()),support_array_reads=self.array_reads,
                    CT_array_reads=0,inklabel_array_reads=0,teacher_student_logit_reads=0,
                    Paris_array_reads=0,checkpoint_reads=0,model_calls=0,network_calls=0)


def d4(array,code):
    require(type(code) is int and 0<=code<8 and array.shape[-2]==array.shape[-1], 'Square D4 input required')
    size=array.shape[-1]
    row,column=np.indices((size,size))
    other=column if code<4 else size-1-column
    indices=((row,other),(other,size-1-row),(size-1-row,size-1-other),(size-1-other,row))[code%4]
    return np.ascontiguousarray(array[...,indices[0],indices[1]])


def fingerprint(value):
    return dict(shape=list(value.shape),dtype=str(value.dtype),
                raw_bytes_sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())


def synthetic_checks():
    source=np.arange(1,10).reshape(3,3)
    expected=[[[1,2,3],[4,5,6],[7,8,9]],[[3,6,9],[2,5,8],[1,4,7]],
              [[9,8,7],[6,5,4],[3,2,1]],[[7,4,1],[8,5,2],[9,6,3]],
              [[3,2,1],[6,5,4],[9,8,7]],[[9,6,3],[8,5,2],[7,4,1]],
              [[7,8,9],[4,5,6],[1,2,3]],[[1,4,7],[2,5,8],[3,6,9]]]
    for code,answer in enumerate(expected):require(np.array_equal(d4(source,code),answer),'Golden D4 differs')
    observed=np.asarray([1,1,1,1,0],dtype=bool)
    supervision=np.asarray([1,0,0,1,1],dtype=bool)
    retired=np.asarray([0,1,0,1,0],dtype=bool)
    manual=observed&supervision&~retired
    unknown=observed&~supervision&~retired
    require(np.array_equal(manual,[1,0,0,0,0]) and np.array_equal(unknown,[0,0,1,0,0]),
            'Support partition fixture differs')
    require(np.array_equal(manual|(observed&retired)|unknown,observed), 'Support partition incomplete')
    exposure=np.asarray([0,1,2,3,4])
    require(int((exposure>0).sum())==4 and int(exposure.sum())==10 and int(exposure.sum()-(exposure>0).sum())==6,
            'Unique/repeated exposure distinction differs')
    return dict(status='PASS',golden_D4_cases=8,partition_with_overlap_case=1,repeated_exposure_case=1)


def support_counts(mask,exposure):
    active=mask&(exposure>0)
    unique=int(active.sum());total=int(exposure[mask].sum())
    return dict(context_pixels=int(mask.sum()),scheduled_unique_pixels=unique,scheduled_exposures=total,
                repeated_exposures_beyond_first=total-unique,
                repeat_histogram={str(int(value)):int((active&(exposure==value)).sum())
                                  for value in np.unique(exposure[active])})


def audit(reader):
    entry_pin=record(ENTRY);require(entry_pin['sha256']==ENTRY_SHA,'Phase 20 plan anchor differs')
    phase20=reader.json(entry_pin)
    base=reader.json(phase20['training_base_plan']);schedule=reader.json(base['schedule'])
    require([case['id'] for case in base['adaptation']]==IDS and
            all(case['physical_group']=='PHerc0139' for case in base['adaptation']), 'Training parent scope differs')
    require([step['step'] for step in schedule]==list(range(1,33)), '32-step schedule differs')
    histories={arm:reader.jsonl(item['steps']) for arm,item in base['historical'].items()}
    require(set(histories)=={'A_preserve','B_preserve'} and all(len(rows)==32 for rows in histories.values()),
            'Historical step receipts differ')
    sources=[]
    for case in base['adaptation']:
        provenance=reader.json(case['provenance'])
        require(provenance['sample_id']=='PHerc0139' and provenance['crop_id']==case['id'] and
                provenance['geometry_validity']==case['data']['geometry_validity']==dict(status='not_acquired',mask_file=None),
                'Geometry support prerequisite changed; separate review required')
        values={name:reader.array(case['inputs'][name]) for name in
                ('acquisition_observed_mask','source_observed_nonzero_mask','supervision_mask','validation_mask')}
        o,n,s,v=[values[key] for key in ('acquisition_observed_mask','source_observed_nonzero_mask','supervision_mask','validation_mask')]
        categories=dict(acquisition_observed=o,source_nonzero_diagnostic=n,manual=o&s&~v,
                        retired_validation=o&v,annotation_unknown=o&~s&~v,outside_current_manual=o&~(s&~v),
                        supervision_validation_overlap=o&s&v)
        require(np.array_equal(categories['manual']|categories['retired_validation']|categories['annotation_unknown'],o),
                'Observed-participation partition differs')
        require(not(categories['manual']&categories['retired_validation']).any() and
                not(categories['manual']&categories['annotation_unknown']).any() and
                not(categories['retired_validation']&categories['annotation_unknown']).any(), 'Partition overlap')
        sources.append(dict(id=case['id'],categories=categories,provenance=case['provenance']))
    array_names=('manual','acquisition_observed','retired_validation','annotation_unknown')
    batches={name:np.zeros((32,2,1,128,128),dtype=bool) for name in array_names}
    source_indices=np.empty((32,2,1,128,128),dtype=np.int32)
    source_canvas=np.arange(384*384,dtype=np.int32).reshape(384,384)
    exposures=[np.zeros((384,384),dtype=np.uint16) for _ in range(2)]
    rows=[];frequencies=[{} for _ in range(2)];augmentations={str(code):0 for code in range(8)}
    for index,step in enumerate(schedule):
        require([patch['region'] for patch in step['patches']]==[0,1], 'Patch region order differs')
        row=dict(step=index+1,patches=[])
        for region,patch in enumerate(step['patches']):
            x,y,width,height=patch['bbox'];code=patch['augmentation']
            require(width==height==128 and patch['bbox'] in base['exposure'][region]['eligible_tiles'] and
                    0<=x<=256 and 0<=y<=256, 'Undeclared patch coordinates')
            exposures[region][y:y+height,x:x+width]+=1
            tile=','.join(map(str,patch['bbox']));frequencies[region][tile]=frequencies[region].get(tile,0)+1
            augmentations[str(code)]+=1
            source_indices[index,region,0]=d4(source_canvas[y:y+height,x:x+width],code)
            for name in array_names:
                batches[name][index,region,0]=d4(sources[region]['categories'][name][y:y+height,x:x+width],code)
            counts={name:int(value[index,region].sum()) for name,value in batches.items()}
            for history in histories.values():
                require(history[index]['step']==step['step'] and history[index]['patches']==step['patches'],
                        'Historical patch/augmentation identity differs')
                require(sum(history[index]['exposures'][region]['manual'].values())==counts['manual'],
                        'Historical manual exposure count differs')
            require(counts['manual']+counts['retired_validation']+counts['annotation_unknown']==counts['acquisition_observed'],
                    'Patch partition does not cover acquired outputs')
            row['patches'].append(dict(region=region,bbox=patch['bbox'],augmentation=code,counts=counts))
        row['counts']={name:sum(patch['counts'][name] for patch in row['patches']) for name in array_names}
        row['manual_fraction_of_observed']=row['counts']['manual']/row['counts']['acquisition_observed']
        rows.append(row)
    regions=[]
    for index,source in enumerate(sources):
        counts={name:support_counts(mask,exposures[index]) for name,mask in source['categories'].items()}
        require(counts['manual']['scheduled_unique_pixels']==base['exposure'][index]['unique_manual_pixels'],
                'Prior unique manual exposure differs')
        regions.append(dict(id=source['id'],counts=counts,exposure_map=fingerprint(exposures[index]),
                            tile_frequencies=frequencies[index],geometry_validity='unknown',
                            geometry_valid_context_pixels=None,geometry_valid_scheduled_unique_pixels=None,
                            geometry_valid_exposures=None))
    totals={name:{field:sum(region['counts'][name][field] for region in regions)
                  for field in ('context_pixels','scheduled_unique_pixels','scheduled_exposures','repeated_exposures_beyond_first')}
            for name in sources[0]['categories']}
    require(totals['acquisition_observed']['scheduled_exposures']==32*2*128*128,
            'Acquired scheduled output count differs')
    arrays={name:fingerprint(array) for name,array in batches.items()}
    arrays['source_index']=fingerprint(source_indices)
    return dict(status='PASS',broader_valid_support_ready=False,geometry_valid_counts=None,
                evidence_scope='PHerc0139 source support and exposure accounting only',regions=regions,totals=totals,
                batches=rows,batch_array_fingerprints=arrays,augmentation_frequencies=augmentations,
                manual_fraction_of_observed_exposures=totals['manual']['scheduled_exposures']/totals['acquisition_observed']['scheduled_exposures'],
                outside_current_manual_includes_retired_validation=True,
                missing_annotation_is_background=False,acquisition_or_nonzero_is_geometric_validity=False),batches,source_indices,exposures


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    require(args.output.resolve().is_relative_to(HERE),'Output must stay in owned review directory')
    fixtures=synthetic_checks();reader=Reader()
    answer,_,_,_=audit(reader)
    for filename in ('assemble_panel_sources.py','prepare_source_plan.py','select_panel_labels.py'):
        reader.verify(record(REPORTS/'seventh-experiment/panel-sources'/filename))
    answer.update(synthetic_checks=fixtures,integrity=reader.finish(),source=record(__file__),
                  numerical_contract=record(HERE/'NUMERICAL_CONTRACT.md'))
    with args.output.open('x') as stream:json.dump(answer,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status='PASS',broader_valid_support_ready=False,report=record(args.output)),sort_keys=True))


if __name__=='__main__':main()
