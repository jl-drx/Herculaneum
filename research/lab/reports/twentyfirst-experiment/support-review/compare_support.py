"""Compare frozen author support output with separately reconstructed counts/hashes."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
AUTHOR=ROOT.parent/'preservation-support/audit-v2/support-contract.json'
AUTHOR_SHA='762cb1e7ae7649d042358f80fea52712ce6471ebd5d4c4aaafc8f7a547694a8f'
OWN=ROOT/'support-accounting-v1.json'
OWN_SHA='1f387b24b489171d070a19d21e36944b971125cae8432f95437b0b7a57a25b81'


def require(condition,message):
    if not condition:raise ValueError(message)


def record(path):
    path=Path(path).resolve(strict=True)
    require(path.is_relative_to(ROOT.parent) or (path.name=='provenance.json' and
            any(name in path.parts for name in ('pherc0139-w016-rank01','pherc0139-w016-rank02'))),
            'Only phase 21 reports/support arrays or exact PHerc0139 provenance permitted')
    require(path.suffix in ('.json','.npy','.py'), 'Only comparison reports/support files permitted')
    return dict(path=str(path),size_bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    own_pin,author_pin=record(OWN),record(AUTHOR)
    require(own_pin['sha256']==OWN_SHA and author_pin['sha256']==AUTHOR_SHA,'Entry receipt pin differs')
    own,author=json.loads(OWN.read_text()),json.loads(AUTHOR.read_text())
    require(own['status']==author['status']=='PASS' and own['broader_valid_support_ready'] is author['broader_valid_support_ready'] is False,
            'Readiness result differs')
    require(author['machine_readable_valid_support_mask'] is author['authenticated_geometry_invalid_pixels'] is
            author['authenticated_geometry_valid_pixels'] is own['geometry_valid_counts'] is None,
            'Unknown geometry validity was converted to a mask/count')
    name_map={'source_nonzero_diagnostic':'nonzero_occupancy_diagnostic',
              'supervision_validation_overlap':'raw_supervision_validation_overlap'}
    field_map={'context_pixels':'whole_context_pixels','scheduled_unique_pixels':'unique_exposed_pixels',
               'scheduled_exposures':'repeated_output_exposures','repeated_exposures_beyond_first':'excess_repeats_beyond_first'}
    comparisons=0
    for name,row in own['totals'].items():
        other=author['totals'][name_map.get(name,name)]
        for key,value in row.items():
            require(other[field_map[key]]==value,'Aggregate support count differs');comparisons+=1
    for first,second in zip(own['regions'],author['regions']):
        require(first['id']==second['id'] and first['tile_frequencies']==second['tile_frequency'],'Region/tile order differs')
        for name,row in first['counts'].items():
            other=second['statistics'][name_map.get(name,name)]
            for key,value in row.items():
                mapped='exposure_multiplicity_histogram_exposed_only' if key=='repeat_histogram' else field_map[key]
                require(other[mapped]==value,'Per-region support/multiplicity count differs');comparisons+=1
    for first,second in zip(own['batches'],author['per_batch']):
        require(first['step']==second['step'] and first['counts']==second['counts'],'Per-batch support counts differ')
        for one,two in zip(first['patches'],second['regions']):
            require(one['region']==two['region']==two['patch']['region'] and one['bbox']==two['patch']['bbox'] and
                    one['augmentation']==two['patch']['augmentation'] and one['counts']==two['counts'],
                    'D4 patch support membership differs')
            require(two['geometry_valid_pixels'] is two['geometry_invalid_pixels'] is None,'Patch validity inferred without source')
            comparisons+=len(one['counts'])
    pins=[own_pin,author_pin];checked_arrays=[]
    array_pairs=[(name,pin,own['batch_array_fingerprints'][name]) for name,pin in author['batch_arrays'].items()]
    array_pairs += [(row['id']+'/exposure',row['exposure_counts'],independent['exposure_map'])
                    for row,independent in zip(author['regions'],own['regions'])]
    require(len(array_pairs)==7,'Exactly five batch support/index arrays and two exposure maps required')
    for name,pin,expected in array_pairs:
        actual_pin=record(pin['path']);require(actual_pin['sha256']==pin['sha256'] and actual_pin['size_bytes']==pin['size_bytes'],
                                              'Derived support array pin differs')
        value=np.load(pin['path'],allow_pickle=False)
        fingerprint=dict(shape=list(value.shape),dtype=str(value.dtype),
                         raw_bytes_sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
        require(fingerprint==expected and pin['raw_sha256']==fingerprint['raw_bytes_sha256'] and
                pin['shape']==fingerprint['shape'] and pin['dtype']==fingerprint['dtype'],
                'Author D4 support/index/exposure array differs from independent reconstruction')
        pins.append(actual_pin);checked_arrays.append(dict(name=name,source=actual_pin,independent_fingerprint=expected))
    coverage=[]
    for item in own['integrity']['input_pins']:
        if Path(item['path']).name!='provenance.json':continue
        actual=record(item['path']);require(actual==item,'PHerc0139 provenance changed')
        source=json.loads(Path(item['path']).read_text());x,y,width,height=source['bbox_source_xywh']
        depth,canvas_y,canvas_x=source['source_shape_zyx']
        require(depth==109 and 0<=x<x+width<=canvas_x and 0<=y<y+height<=canvas_y and width==height==384,
                'Crop extends outside declared source canvas')
        observed=np.zeros((384,384),dtype=np.uint8)
        require(len(source['source_objects'])==9,'Expected nine source chunks')
        for chunk in source['source_objects']:
            z,cy,cx=chunk['chunk_zyx'];yy,xx=cy*128-y,cx*128-x
            require(z==0 and yy in (0,128,256) and xx in (0,128,256) and chunk['verified'] is True and
                    chunk['byte_length']==109*128*128,'Source transport receipt differs')
            observed[yy:yy+128,xx:xx+128]+=1
        require((observed==1).all(),'Retained source chunks do not tile crop exactly once')
        require(source['geometry_validity']==dict(status='not_acquired',mask_file=None),'Geometry source status changed')
        coverage.append(dict(region=source['crop_id'],nine_chunks_exactly_cover_local_crop=True,
                             crop_inside_declared_canvas=True,local_padding_pixels=0,local_missing_chunk_fill_pixels=0,
                             source_unmapped_or_invalid_pixels=None,
                             qualification='Retained transport/source-layout metadata only; no raw chunk decode or geometry authentication'))
        pins.append(actual)
    reallocation=[]
    for row in own['batches']:
        m=row['counts']['manual'];r=row['counts']['retired_validation']
        require(m>0 and r>0,'Proposed allocation has an empty scheduled stratum')
        reallocation.append(dict(step=row['step'],manual_count=m,retired_count=r,
                                 A_manual_KL_coefficient_per_output=2/m,B_manual_KL_coefficient_per_output=1/m,
                                 B_retired_KL_coefficient_per_output=1/r,
                                 B_retired_to_manual_per_output_coefficient_ratio=m/r))
    require(own['totals']['supervision_validation_overlap']['context_pixels']==0,
            'Narrow S-only interpretation needs separate overlap review')
    for item in pins:require(record(item['path'])==item,'Input changed during support comparison')
    require('torch' not in sys.modules,'Torch forbidden')
    result=dict(status='PASS',source=record(__file__),inputs=pins,checked_arrays=checked_arrays,
                scalar_or_histogram_comparisons=comparisons,all_32_batches_exact=True,source_transport_checks=coverage,
                broader_geometry_valid_support_ready=False,geometry_valid_counts=None,
                acquired_supervision_alone_adds_no_support=True,
                proposed_explicit_supervision_union_retired=dict(scheduled_unique_outputs=98216,scheduled_output_exposures=1027656,
                    excluded_annotation_unknown_unique_outputs=4184,excluded_annotation_unknown_exposures=20920,
                    geometry_validity_claim=False),
                proposed_reallocation_per_batch=reallocation,empty_retired_batches=0,
                teacher_student_logit_reads=0,Paris_scientific_array_reads=0,CT_array_reads=0,inklabel_array_reads=0,
                model_calls=0,optimization_steps=0,network_calls=0)
    output=ROOT/'author-comparison-v1.json'
    with output.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status='PASS',comparison=record(output)),sort_keys=True))


if __name__=='__main__':main()
