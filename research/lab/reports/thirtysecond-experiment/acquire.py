"""Separate metadata and frozen-pixel stages; publish completion only after all reads."""
import argparse, struct, resource, time
import numpy as np
from common import *
from transport import Client

def metadata(plan):
    root=HERE/'metadata-v1';require(not root.exists(),'Metadata already attempted')
    client=Client(root,plan);objects=[]
    for layer,url in enumerate(plan['urls'],1):
        header,e=client.get(url,[[0,7]],f'{layer:02d}-header.bin')
        identity=e['identity'];head=tiff_header(header.read_bytes());offset=head['ifd_offset']
        require(8<=offset<identity['total_bytes']-2,'Invalid IFD offset')
        count,_=client.get(url,[[offset,offset+1]],f'{layer:02d}-count.bin',identity)
        n=struct.unpack(head['struct_endian']+'H',count.read_bytes())[0]
        require(0<n<=128,'IFD count bound');size=2+12*n+4
        ifd,_=client.get(url,[[offset,offset+size-1]],f'{layer:02d}-ifd.bin',identity)
        layout=parse_tiff(header.read_bytes(),ifd.read_bytes(),identity['total_bytes'],[8853,6205])
        tag=layout['tags'][305]
        require(tag['type']==2 and tag['count']==27,'Historical Software field changed')
        a=tag['value_or_offset'];software,_=client.get(url,[[a,a+26]],f'{layer:02d}-software.bin',identity)
        require(software.read_bytes()==b'volume-cartographer 2.26.0\x00','Historical Software differs')
        objects.append(dict(layer=layer,url=url,identity=identity,layout=layout,
                            metadata_files=[pin(p) for p in [header,count,ifd,software]]))
        if layer%9==0:print(json.dumps(dict(metadata_layers=layer,requests=len(client.ledger['requests']))),flush=True)
    require(len(objects)==63,'Missing objects')
    save(root/'OBJECTS.json',dict(status='PASS',objects=objects,ledger=pin(root/'LEDGER.json')))

def pixels(plan):
    freeze=check_freeze('PIXEL_GO.json')
    source=json.loads(verify(freeze['objects']).read_text());pixel_plan=json.loads(verify(freeze['pixel_plan']).read_text())
    ledger=json.loads((HERE/'metadata-v1/LEDGER.json').read_text())
    root=HERE/'pixels-v1';require(not root.exists(),'Pixels already attempted')
    client=Client(root,plan,ledger=ledger);out=HERE/'raw-v1';out.mkdir(exist_ok=False)
    arrays={c['id']:np.lib.format.open_memmap(out/(c['id']+'.npy'),mode='w+',dtype='<u2',shape=(63,384,384)) for c in plan['contexts']}
    rows_per_layer=np.zeros((63,4,384),dtype=bool)
    for obj,batches in zip(source['objects'],pixel_plan['layers'],strict=True):
        z=obj['layer'];require(z==batches['layer'],'Layer order changed')
        for group in batches['batches']:
            ranges=[r['range'] for r in group['rows']]
            body,entry=client.get(obj['url'],ranges,f'{z:02d}-{group["index"]}.bin',obj['identity'])
            data=body.read_bytes();parts={tuple(p['range_inclusive']):p for p in entry['parts']}
            for row in group['rows']:
                part=parts[tuple(row['range'])];start=part['payload_offset'];size=part['payload_bytes']
                require(size==768,'Crop row size')
                values=np.frombuffer(data,dtype=obj['layout']['dtype'],count=384,offset=start)
                ci=row['context_index'];rr=row['local_row'];require(not rows_per_layer[z-1,ci,rr],'Duplicate row')
                arrays[plan['contexts'][ci]['id']][z-1,rr]=values;rows_per_layer[z-1,ci,rr]=True
        if z%3==0:print(json.dumps(dict(pixel_layers=z,requests=len(client.ledger['requests']),returned_bytes=client.ledger['returned_body_bytes'])),flush=True)
    require(rows_per_layer.all(),'Incomplete crop support')
    for a in arrays.values():a.flush()
    records={k:pin(out/(k+'.npy')) for k in arrays}
    save(root/'COMPLETE.json',dict(status='PASS',created_at_utc=now(),raw_arrays=records,
         rows_verified=int(rows_per_layer.sum()),useful_CT_bytes=74317824,
         requests=len(client.ledger['requests']),returned_body_bytes=client.ledger['returned_body_bytes'],
         reserved_body_bytes=client.ledger['reserved_body_bytes'],active_acquisition_seconds=client.ledger['active_acquisition_seconds'],
         peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,ledger=pin(root/'LEDGER.json'),pixel_freeze=pin(HERE/'PIXEL_GO.json')))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['metadata','pixels']);args=parser.parse_args()
    check_freeze('ACQUISITION_GO.json');plan=json.loads((HERE/'PLAN.json').read_text())
    try:(metadata if args.stage=='metadata' else pixels)(plan)
    except Exception as e:
        save(HERE/(args.stage+'-FAILURE.json'),dict(status='STOP',created_at_utc=now(),error=type(e).__name__+': '+str(e)))
        raise

if __name__=='__main__':main()
