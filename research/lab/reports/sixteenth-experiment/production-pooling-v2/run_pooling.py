"""Root-run offline verification and exact CT pooling; publish descriptor only after all gates."""
import argparse
import gc
import json
import os
from pathlib import Path
import resource
import signal
import socket
import sys
import time


def offline(*args,**kwargs):raise RuntimeError('Network disabled in offline CT pooling')
socket.create_connection=offline;socket.socket.connect=offline;socket.socket.connect_ex=offline
sys.dont_write_bytecode=True
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='4'

import numpy as np
from pooling_kernels import assemble_group,pool_triplet
from pooling_common import (HERE,CASE_IDS,GROUPS,load_plan,record,identity,verify,bounded_bytes,read_json,write_json,
    module,selected_cases,validate_inventory,validate_batch,storage_bytes,require,persisted_layout_matches)
from pooling_provenance import (JSON_PLAN_CAP,PRODUCER_ROOT,shared_elapsed,required_root_coverage,
    validate_ready_pixel_plan,validate_global_receipts,validate_terminal_chain)


def rss():return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    plan,pin=load_plan(args.plan,execute=True);limits=plan['limits'];signal.alarm(limits['wall_seconds'])
    resource.setrlimit(resource.RLIMIT_CPU,(limits['cpu_seconds'],)*2);resource.setrlimit(resource.RLIMIT_AS,(limits['address_space_bytes'],)*2)
    resource.setrlimit(resource.RLIMIT_FSIZE,(limits['output_bytes'],)*2);started=time.perf_counter()
    require(list(sys.version_info[:3])==plan['sources']['runtime']['python'] and np.__version__==plan['sources']['runtime']['numpy'],'Pooling runtime differs')
    args.output_dir.mkdir(parents=True,exist_ok=False);roots=plan['production_roots']+[str(HERE),str(args.output_dir)]
    require(storage_bytes(roots)<=limits['combined_storage_bytes'],'Combined production storage exceeds2GiB')
    raw=read_json(plan['raw_provenance']);cases,fragments=selected_cases(plan['sources']);case_map={c['case_id']:c for c in cases}
    production_epoch=raw['production_started_epoch_seconds'];shared_elapsed(production_epoch)
    contracts,schedule=validate_inventory(raw,cases,fragments);parser_source=module(plan['sources']['tiff_parser'],'pooling_retained_TIFF_parser')
    input_pins={};verified_batches={};encoded_bytes=0
    def remember(item,cap=None):
        canonical=identity(item)
        require(canonical['path'] not in input_pins or input_pins[canonical['path']]==canonical,'Conflicting input pins')
        data=bounded_bytes(canonical,cap) if cap is not None else None
        if cap is None:verify(canonical)
        input_pins[canonical['path']]=canonical;return data
    def read_tracked(item,cap=8*1024**2):return json.loads(remember(item,cap))
    read_tracked.remember=remember
    remember(plan['raw_provenance']);pixel=read_tracked(raw['plan'],JSON_PLAN_CAP)
    producer_freeze=read_json(plan['sources']['producer_code_freeze'])
    for item in producer_freeze['code']:verify(item)
    metadata=validate_ready_pixel_plan(raw,pixel,plan['sources'],producer_freeze,read_tracked)
    required_root_coverage(roots,[plan['raw_provenance']['path'],raw['plan']['path'],pixel['metadata_result']['path'],pixel['metadata_plan']['path']])
    requests,entries,terminals=validate_global_receipts(raw,pixel,read_tracked(raw['request_ledger']),read_tracked(raw['terminal_receipts']))
    for contract in contracts.values():
        metadata=contract['raw_metadata'];header=remember(metadata['header'],8);count=remember(metadata['ifd_count'],2);ifd=remember(metadata['ifd'],16384)
        require(len(header)==8 and len(count)==2 and count==ifd[:2],'Metadata prefix/count identity differs')
        layout=parser_source.parse_tiff(header,ifd,contract['object_total_bytes'],[contract['height'],contract['width']])
        require(persisted_layout_matches(layout,contract),'Raw metadata reconstruction differs from per-file layout')
    for case in cases:
        for item in case['files'].values():remember(item)
    # Entire complete source package is verified before any sample decode/frombuffer.
    for key,index in schedule.items():
        batch=read_json(index['batch'],1024**2);remember(index['batch'])
        contract=contracts[(index['fragment'],index['plane_index'])];body=remember(batch['multipart'],1024**2)
        identifier=index['request_id'];require(identity(batch['terminal_receipt'])==identity(terminals[identifier]),'Batch terminal differs from global index')
        terminal=read_tracked(batch['terminal_receipt']);config=read_tracked(terminal['request']);transport=read_tracked(terminal['transport_ledger'])
        validate_terminal_chain(batch,requests[identifier],entries[identifier],terminal,config,transport,raw['plan'],contract)
        require(config['code']==pixel['code'],'Child executed producer source differs')
        required_root_coverage(roots,[index['batch']['path'],batch['multipart']['path'],batch['terminal_receipt']['path'],terminal['request']['path'],terminal['transport_ledger']['path']])
        validate_batch(index,batch,contract,case_map[index['case_id']],body)
        verified_batches[key]=identity(index['batch']);encoded_bytes+=len(body)
        require(rss()<=limits['rss_bytes'],'Pre-decode verification RSS exceeds acceptance')
        shared_elapsed(production_epoch)
    require(len(verified_batches)==1530 and encoded_bytes<=1.25*1024**3,'Complete bounded pixel bodies required')
    write_json(args.output_dir/'raw-verification.json',dict(status='PASS',plan=pin,raw_provenance=plan['raw_provenance'],
        object_count=153,batch_count=1530,useful_sample_bytes=812187648,encoded_multipart_bytes=encoded_bytes,
        all_raw_verified_before_numeric_decode=True,inputs=list(input_pins.values())))
    preprocessing=read_json(plan['sources']['inference_contract'])['preprocessing'];outputs=[]
    for case in cases:
        volume=np.empty((17,384,384),np.float32);y,x,_,_=case['native_bbox_yxhw'];fragment=case['case_id'].split('-')[0]
        for group_index,plane_start in enumerate(range(7,58,3)):
            planes=np.empty((3,1152,1152),np.uint32)
            for depth,plane in enumerate(range(plane_start,plane_start+3)):
                relative_y=0;contract=contracts[(fragment,plane)]
                for row_count in GROUPS:
                    key=(case['case_id'],plane,y+relative_y);index=schedule[key];batch=read_json(verified_batches[key],1024**2)
                    body=bounded_bytes(batch['multipart'],1024**2);validate_batch(index,batch,contract,case,body)
                    planes[depth,relative_y:relative_y+row_count]=assemble_group(body,batch['rows'],contract,x,y+relative_y,row_count,1152)
                    relative_y+=row_count
            volume[group_index]=pool_triplet(planes);del planes,body,batch;gc.collect()
            require(rss()<=limits['rss_bytes'],'Numeric pooling RSS exceeds acceptance')
            shared_elapsed(production_epoch)
        require(volume.dtype.isnative and volume.flags.c_contiguous and np.isfinite(volume).all()
            and np.all((volume>=0)&(volume<=255)),'Final native FP32 volume differs')
        directory=args.output_dir/case['case_id'];directory.mkdir();np.save(directory/'surface.npy',volume,allow_pickle=False)
        names=dict(labels='labels.npy',primary_mask='primary-mask.npy',known_mask='known-mask.npy',surface_mask='surface-mask.npy')
        arrays={key:identity(case['files'][name]) for key,name in names.items()}
        arrays['surface']=dict(record(directory/'surface.npy'),shape=[17,384,384],dtype='float32',byte_order=sys.byteorder)
        outputs.append(dict(id=case['case_id'],native_bbox_yxhw=case['native_bbox_yxhw'],bbox_source_xywh=case['bbox_source_xywh'],
            fragment=case['fragment'],role=case['role'],parent_papyrus=case['parent_papyrus'],arrays=arrays));del volume;gc.collect()
    for item in input_pins.values():verify(item)
    load_plan(args.plan,execute=True)
    require(rss()<=limits['rss_bytes'] and time.perf_counter()-started<=limits['wall_seconds'],'Final pooling resource acceptance failed')
    require(sum(p.stat().st_size for p in args.output_dir.rglob('*') if p.is_file())<=limits['output_bytes'],'Pooling output cap exceeded')
    shared_elapsed(production_epoch)
    descriptor=dict(status='pooled_reference_complete_no_inference',preprocessing=preprocessing,cases=outputs,
        raw_provenance=plan['raw_provenance'],pixel_plan=raw['plan'],pooling_plan=pin,production_schema=plan['sources']['schema'],
        production_design=plan['sources']['production_design'],pooling_code_freeze=plan['code_freeze'],pooling_source_go=plan['source_go'],
        pooling_runtime=plan['sources']['runtime'],raw_verification=record(args.output_dir/'raw-verification.json'),
        metadata_object_count=153,pixel_batch_count=1530,independent_CT_verification_pending=True,model_calls=0)
    descriptor.update(metadata_provenance=pixel['metadata_result'],production_started_epoch_seconds=production_epoch,
        production_elapsed_seconds=shared_elapsed(production_epoch))
    # Stage final bytes, count every written output, then publish canonical descriptor last.
    staged_descriptor=args.output_dir/'descriptor.staged';write_json(staged_descriptor,descriptor)
    descriptor_record=dict(record(staged_descriptor),path=str((args.output_dir/'pooled-descriptor.json').resolve()))
    result=dict(status='PASS',plan=pin,descriptor=descriptor_record,cases=outputs,
        wall_seconds=time.perf_counter()-started,peak_rss_bytes=rss(),combined_production_bytes=storage_bytes(roots),
        production_started_epoch_seconds=production_epoch,production_elapsed_seconds=shared_elapsed(production_epoch),
        all_inputs_unchanged_before_after=True,all_raw_verified_before_numeric_decode=True,model_calls=0,network_calls=0,checkpoint_reads=0)
    staged_result=args.output_dir/'result.staged';write_json(staged_result,result)
    require(storage_bytes(roots)<=limits['combined_storage_bytes'],'Actual final M/P/C storage exceeds2GiB')
    require(storage_bytes([args.output_dir])<=limits['output_bytes'] and rss()<=limits['rss_bytes']
        and time.perf_counter()-started<=limits['wall_seconds'],'Actual final output/resource cap exceeded')
    shared_elapsed(production_epoch)
    staged_result.replace(args.output_dir/'result.json');staged_descriptor.replace(args.output_dir/'pooled-descriptor.json');signal.alarm(0)
    print(json.dumps(dict(status='PASS',descriptor=result['descriptor'],result=record(args.output_dir/'result.json'))))


if __name__=='__main__':main()
