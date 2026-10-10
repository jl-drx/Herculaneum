"""Both-endian saved-JSON layout fixtures and optionally authorized metadata-only replay."""
import argparse
import copy
import json
from pathlib import Path
import struct

from pooling_common import (HERE,record,read_json,bounded_bytes,write_json,sources,module,
    persisted_layout_matches,require)


def synthetic_tiff(order):
    fields=[(256,4,9),(257,4,6),(258,3,16),(259,3,1),(262,3,1),(273,4,256),
        (274,3,1),(277,3,1),(278,4,6),(279,4,108),(339,3,1)]
    header=(b'II' if order=='<' else b'MM')+struct.pack(order+'HI',42,8)
    entries=[]
    for tag,kind,value in fields:
        inline=struct.pack(order+'H',value)+b'\0\0' if kind==3 else struct.pack(order+'I',value)
        entries.append(struct.pack(order+'HHI',tag,kind,1)+inline)
    return header,struct.pack(order+'H',len(fields))+b''.join(entries)+struct.pack(order+'I',0)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--real-metadata',type=Path);args=parser.parse_args();source,_=sources()
    retained_parser=module(source['tiff_parser'],'layout_fixture_retained_parser');passed=[]
    for order in ('<','>'):
        header,ifd=synthetic_tiff(order);layout=retained_parser.parse_tiff(header,ifd,364,[6,9])
        saved=json.loads(json.dumps(layout));require(layout['tags']!=saved['tags'],'Fixture must expose integer/JSON string key mismatch')
        require(persisted_layout_matches(layout,saved),'Both-endian JSON representation match failed');passed.append(order+'saved JSON exact match')
        for key,value in [('type',4),('count',2),('value_or_offset',99),('value_bytes_hex','00000000'),('storage_bytes',99)]:
            bad=copy.deepcopy(saved);bad['tags']['258'][key]=value
            require(not persisted_layout_matches(layout,bad),'Tampered tag accepted');passed.append(order+'reject tag '+key)
        bad=dict(saved,strip_offset=257);require(not persisted_layout_matches(layout,bad),'Non-tag layout mutation accepted')
        passed.append(order+'reject non-tag strip offset')
    objects=[];inputs=[]
    if args.real_metadata:
        pin=record(args.real_metadata);raw=read_json(pin);inputs.append(pin)
        require(len(raw['tiff_contracts'])==153,'Complete metadata inventory required')
        for contract in raw['tiff_contracts']:
            m=contract['raw_metadata'];header=bounded_bytes(m['header'],8);count=bounded_bytes(m['ifd_count'],2);ifd=bounded_bytes(m['ifd'],16384)
            require(len(header)==8 and len(count)==2 and count==ifd[:2],'Raw metadata prefix differs')
            layout=retained_parser.parse_tiff(header,ifd,contract['object_total_bytes'],[contract['height'],contract['width']])
            require(persisted_layout_matches(layout,contract),'Authorized metadata-only exact check failed')
            objects.append(dict(object_id=contract['object_id'],dtype=contract['dtype'],strip_offset=contract['strip_offset'],all_layout_values_exact=True))
            inputs.extend(record(m[k]['path']) for k in ('header','ifd_count','ifd'))
    args.output_dir.mkdir(parents=True,exist_ok=False);path=args.output_dir/'fixture-receipt.json'
    write_json(path,dict(status='PASS',synthetic_checks=passed,count=len(passed),metadata_objects=objects,metadata_only_inputs=inputs,
        code=[record(p) for p in sorted(HERE.glob('*.py'))],CT_sample_decodes=0,network_calls=0,model_calls=0,checkpoint_reads=0))
    print(json.dumps(dict(status='PASS',receipt=record(path),synthetic_checks=len(passed),metadata_objects=len(objects))))


if __name__=='__main__':main()
