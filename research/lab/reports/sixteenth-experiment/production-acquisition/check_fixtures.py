"""Synthetic binary/HTTP fixtures only; never open a network connection or source CT array."""
import argparse
from email.message import Message
import hashlib
import http.client
import json
from pathlib import Path
import socket
import tempfile
from production_common import *
from prepare_plan import pixel_requests,metadata_inventory
from range_http import RangeClient
from run_acquisition import Controller,batch_record
import struct
import time

URL='https://dl.ash2txt.org/fragments/Frag1/synthetic.volpkg/working/54keV_exposed_surface/surface_volume/07.tif'
TOTAL=2000000
RANGES=[[1000+i*3000,1000+i*3000+2303] for i in range(256)]
BOUNDARY='bounded-test'


def multipart(ranges=RANGES,total=TOTAL,reverse=False,prefix=b'',boundary=BOUNDARY):
    chunks=[]
    selected=list(reversed(ranges)) if reverse else ranges
    for index,(a,b) in enumerate(selected):
        payload=(bytes([index%251])*20+b'\r\n--bounded-test\r\n'+bytes([255-index%251])*2304)[:b-a+1]
        chunks.append(b'--'+boundary.encode()+b'\r\nContent-Type: application/octet-stream\r\nContent-Range: bytes '+f'{a}-{b}/{total}'.encode()+b'\r\n\r\n'+payload+b'\r\n')
    return prefix+b''.join(chunks)+b'--'+boundary.encode()+b'--\r\n'


class Response:
    def __init__(self,body,status=206,headers=None,incomplete=None):
        self.body,self.status,self.position,self.incomplete=body,status,0,incomplete
        self.read_calls=0
        self.headers=Message()
        for k,v in (headers or {}).items():
            self.headers[k]=str(v)
    def read(self,n):
        self.read_calls+=1
        if self.incomplete is not None:
            partial,self.incomplete=self.incomplete,None
            raise http.client.IncompleteRead(partial,len(self.body))
        result=self.body[self.position:self.position+n]
        self.position+=len(result)
        return result
    def close(self):
        pass


class Opener:
    def __init__(self,response):
        self.response,self.calls=response,[]
    def open(self,request,timeout):
        self.calls.append(request)
        return self.response


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    root=Path(args.output_dir).resolve()
    require(not root.exists(),'Fresh fixture output required')
    root.mkdir(parents=True)
    socket.socket.connect=lambda *_: (_ for _ in ()).throw(AssertionError('Network forbidden in fixtures'))
    results=[]
    def passed(name,condition=True):
        require(condition,'Synthetic fixture failed:'+name)
        results.append({'name':name,'status':'PASS'})
    def rejected(name,callback):
        try:
            callback()
        except ContractError:
            passed(name)
            return
        raise AssertionError('Expected rejection:'+name)
    body=multipart()
    parsed=parse_multipart(body,'multipart/byteranges; boundary='+BOUNDARY,RANGES,TOTAL)
    passed('256 binary payloads survive delimiter collisions',len(parsed)==256 and all(
        hashlib.sha256(body[p['payload_offset']:p['payload_offset']+p['payload_bytes']]).hexdigest()==p['payload_sha256']
        for p in parsed))
    reverse=multipart(reverse=True,prefix=b'\r\n')
    parsed_reverse=parse_multipart(reverse,'multipart/byteranges; boundary='+BOUNDARY,RANGES,TOTAL)
    passed('reordered parts and one leading CRLF',parsed_reverse[0]['range_inclusive']==RANGES[-1])
    quoted=multipart(boundary='a=b')
    passed('quoted MIME special boundary',len(parse_multipart(quoted,'multipart/byteranges; boundary="a=b"',RANGES,TOTAL))==256)
    rejected('unquoted MIME special boundary',lambda:parse_multipart(quoted,'multipart/byteranges; boundary=a=b',RANGES,TOTAL))
    rejected('255 requested parts',lambda:parse_multipart(multipart(RANGES[:-1]),'multipart/byteranges; boundary='+BOUNDARY,RANGES[:-1],TOTAL))
    rejected('257 requested parts',lambda:parse_multipart(body,'multipart/byteranges; boundary='+BOUNDARY,RANGES+[RANGES[0]],TOTAL))
    rejected('missing returned part',lambda:parse_multipart(multipart(RANGES[:-1]),'multipart/byteranges; boundary='+BOUNDARY,RANGES,TOTAL))
    rejected('duplicate returned part',lambda:parse_multipart(multipart(RANGES[:-1]+[RANGES[0]]),'multipart/byteranges; boundary='+BOUNDARY,RANGES,TOTAL))
    rejected('unrequested/coalesced part',lambda:parse_multipart(multipart([[RANGES[0][0],RANGES[1][1]]]+RANGES[2:]),'multipart/byteranges; boundary='+BOUNDARY,RANGES,TOTAL))
    rejected('object total changed',lambda:parse_multipart(multipart(total=TOTAL+1),'multipart/byteranges; boundary='+BOUNDARY,RANGES,TOTAL))
    rejected('overlapping requested spans',lambda:parse_multipart(body,'multipart/byteranges; boundary='+BOUNDARY,[RANGES[0]]+[[RANGES[0][0]+1,RANGES[0][1]+1]]+RANGES[2:],TOTAL))
    rejected('truncated binary body',lambda:parse_multipart(body[:-10],'multipart/byteranges; boundary='+BOUNDARY,RANGES,TOTAL))
    rejected('trailing data',lambda:parse_multipart(body+b'x','multipart/byteranges; boundary='+BOUNDARY,RANGES,TOTAL))
    rejected('arbitrary preamble',lambda:parse_multipart(b'ignored'+body,'multipart/byteranges; boundary='+BOUNDARY,RANGES,TOTAL))
    rejected('two leading CRLF',lambda:parse_multipart(b'\r\n\r\n'+body,'multipart/byteranges; boundary='+BOUNDARY,RANGES,TOTAL))
    rejected('part header above512',lambda:parse_multipart(body.replace(b'Content-Type: application/octet-stream',b'Content-Type: '+b'x'*520,1),'multipart/byteranges; boundary='+BOUNDARY,RANGES,TOTAL))
    rejected('duplicate MIME header',lambda:parse_multipart(body.replace(b'Content-Type:',b'Content-Length: 2304\r\nContent-Length: 2304\r\nContent-Type:',1),'multipart/byteranges; boundary='+BOUNDARY,RANGES,TOTAL))
    rejected('framing exceeds allowance',lambda:parse_multipart(body+b'x'*(768*256+1024),'multipart/byteranges; boundary='+BOUNDARY,RANGES,TOTAL))
    limits=LIMITS['pixels']
    pin={URL:{'etag':'"stable"','object_total_bytes':TOTAL}}
    count=0
    def client(response):
        nonlocal count
        count+=1
        opener=Opener(response)
        return RangeClient(root/('http-'+str(count)),[URL],limits,{'synthetic':True},pins=dict(pin),opener=opener),opener
    single_headers={'ETag':'"stable"','Content-Range':f'bytes 1000-1007/{TOTAL}','Content-Length':'8'}
    c,op=client(Response(b'abcdefgh',headers=single_headers))
    path,rec=c.get(URL,[[1000,1007]],'single.bin')
    passed('single success exact charged bytes and IfMatch',path.read_bytes()==b'abcdefgh' and c.ledger['charged_response_bytes']==8 and op.calls[0].get_header('If-match')=='"stable"')
    resp=Response(b'x'*100,status=200,headers=single_headers)
    c,_=client(resp)
    rejected('HTTP200 rejected before body',lambda:c.get(URL,[[1000,1007]],'single.bin'))
    passed('HTTP200 full failure reservation no consumed body',resp.read_calls==0 and c.ledger['charged_response_bytes']==9 and c.ledger['retained_body_bytes']==0)
    resp=Response(b'abcdefgh',headers={**single_headers,'ETag':'"changed"'})
    c,_=client(resp)
    rejected('stale ETag before body',lambda:c.get(URL,[[1000,1007]],'single.bin'))
    passed('stale ETag no body',resp.read_calls==0)
    resp=Response(b'abcdefgh',headers=single_headers)
    resp.headers['ETag']='"other"'
    c,_=client(resp)
    rejected('duplicate identity header',lambda:c.get(URL,[[1000,1007]],'single.bin'))
    resp=Response(b'abcdefgh',headers={**single_headers,'Content-Range':f'bytes 1000-1007/{TOTAL+1}'})
    c,_=client(resp)
    rejected('changed total before body',lambda:c.get(URL,[[1000,1007]],'single.bin'))
    resp=Response(b'abcdefgh',headers=single_headers,incomplete=b'abc')
    c,_=client(resp)
    rejected('partial failed read preserved',lambda:c.get(URL,[[1000,1007]],'single.bin'))
    passed('partial charge separated from retained',c.ledger['charged_response_bytes']==9 and c.ledger['retained_body_bytes']==3 and (c.root/'raw/single.bin.partial').read_bytes()==b'abc')
    resp=Response(b'abcdefgh',headers={k:v for k,v in single_headers.items()if k!='Content-Length'})
    c,_=client(resp)
    path,rec=c.get(URL,[[1000,1007]],'single.bin')
    passed('unknown length bounded EOF',path.read_bytes()==b'abcdefgh' and rec['complete_eof'])
    resp=Response(b'abcdefghi',headers={k:v for k,v in single_headers.items()if k!='Content-Length'})
    c,_=client(resp)
    rejected('unknown length sentinel overflow',lambda:c.get(URL,[[1000,1007]],'single.bin'))
    c,_=client(Response(body,headers={'ETag':'"stable"','Content-Type':'multipart/byteranges; boundary='+BOUNDARY,'Content-Length':len(body)}))
    path,rec=c.get(URL,RANGES,'multi.bin',mode='multipart')
    passed('256 multipart HTTP payload accepted',len(rec['parts'])==256 and c.ledger['charged_response_bytes']==len(body))
    c,_=client(Response(body,headers={'ETag':'"stable"','Content-Type':'multipart/byteranges; boundary='+BOUNDARY,'Content-Length':len(body)}))
    rejected('8part network count forbidden',lambda:c.get(URL,RANGES[:8],'multi.bin',mode='multipart'))
    passed('bad count consumes no attempt',len(c.ledger['requests'])==0)
    short_ranges=RANGES[:128]
    short_body=multipart(short_ranges)
    passed('128 exact production parts',len(parse_multipart(short_body,'multipart/byteranges; boundary='+BOUNDARY,short_ranges,TOTAL))==128)
    rejected('127 requested production parts',lambda:parse_multipart(short_body,'multipart/byteranges; boundary='+BOUNDARY,short_ranges[:-1],TOTAL))
    def synthetic_tiff(order):
        width,height,strip=1234,2311,260 if order=='>'else 8
        strip_bytes=width*height*2;offset=strip+strip_bytes+32
        tags=[(256,4,width),(257,4,height),(258,3,16),(262,3,1),(273,4,strip),(278,4,height),(279,4,strip_bytes)]
        table=struct.pack(order+'H',len(tags))
        for tag,kind,value in tags:
            value_bytes=struct.pack(order+('H'if kind==3 else'I'),value)+(b'\xa5\x5a'if kind==3 else b'')
            table+=struct.pack(order+'HHI',tag,kind,1)+value_bytes
        table+=struct.pack(order+'I',0)
        prefix=(b'MM'if order=='>'else b'II')+struct.pack(order+'HI',42,offset)
        return prefix,table,offset+len(table),[height,width]
    for order in ('<','>'):
        prefix,table,total,shape=synthetic_tiff(order)
        parsed_tiff=parse_tiff(prefix,table,total,shape)
        passed('asymmetric TIFF '+order+' inline SHORT',parsed_tiff['dtype']==order+'u2'and parsed_tiff['strip_bytes']==shape[0]*shape[1]*2)
        rejected('wrong magic TIFF '+order,lambda prefix=prefix,table=table,total=total,shape=shape:parse_tiff(prefix[:2]+b'\x00\x00'+prefix[4:],table,total,shape))
    cases=[];objects=[];contracts=[]
    dimensions=[(6330,8181),(9506,14830),(5249,7606)]
    bboxes=[[[2304,1152,1152,1152],[0,6912,1152,1152]],[[1152,1152,1152,1152],[8064,12672,1152,1152]],[[2304,1152,1152,1152],[3456,5760,1152,1152]]]
    for fi,(width,height)in enumerate(dimensions,1):
        name=f'frag{fi}'
        for ci,bbox in enumerate(bboxes[fi-1],1):
            cases.append({'case_id':f'{name}-context{ci}','bbox_source_xywh':bbox})
        for z in range(7,58):
            url=URL[:-6]+f'{z:02d}.tif';identity=f'{name}-z{z:02d}'
            objects.append({'object_id':identity,'fragment':name,'plane_index':z,'url':url,'shape_yx':[height,width]})
            contracts.append({'object_id':identity,'url':url,'width':width,'height':height,'strip_offset':260 if fi==1 else 8,
                              'strip_bytes':width*height*2,'object_total_bytes':width*height*2+1000,'etag':'"stable"'})
    inventory=pixel_requests(cases,objects,contracts)
    passed('complete1530 fixed IDs and useful/reserved totals',len(inventory)==1530 and sum(r['reserved_cap_bytes']for r in inventory)==EXPECTED_RESERVED_BYTES)
    passed('fixed rowgroups cover each context/plane exactly',all([r['row_count']for r in inventory[i:i+5]]==[256,256,256,256,128]and
        [r['native_row_start']-inventory[i]['native_row_start']for r in inventory[i:i+5]]==[0,256,512,768,1024]for i in range(0,len(inventory),5)))
    passed('exact source addressing both strip conventions',all(r['ranges'][0][0]==next(c['strip_offset']+2*(r['native_row_start']*c['width']+r['native_x_start'])for c in contracts if c['object_id']==r['object_id'])for r in inventory))
    passed('metadata450 fixed logical IDs excluding carried07',len(metadata_inventory(objects))==450 and all('-z07-'not in r['request_id']for r in metadata_inventory(objects)))
    rejected('missing metadata cannot derive pixels',lambda:pixel_requests(cases,objects,contracts[:-1]))
    sample=inventory[0];part_rows=[]
    for i,span in reversed(list(enumerate(sample['ranges']))):
        part_rows.append({'range_inclusive':span,'payload_offset':i*2400,'payload_bytes':2304,'payload_sha256':'a'*64,'headers':{}})
    routed=batch_record(sample,{'etag':'"stable"','object_total_bytes':contracts[0]['object_total_bytes'],'content_type':'multipart/byteranges; boundary=x','body':{'synthetic':True}}, {'parts':part_rows})
    passed('production rows route independent of physical part order',[r['native_y']for r in routed['rows']]==list(range(sample['native_row_start'],sample['native_row_start']+256)))
    # Fake child processes expose durable scheduler state and never launch a real process.
    synthetic_objects=objects[1:5]
    fake_plan={'stage':'metadata','fixed_inputs':{'objects':synthetic_objects,'carried_contracts':[]},
               'requests':metadata_inventory(synthetic_objects),'code':[record(ROOT/n)for n in CODE_NAMES]}
    def child_success(config_path,bad_binding=False):
        config=json.loads(config_path.read_text());request=config['request'];folder=config_path.parent
        response=folder/'response';(response/'raw').mkdir(parents=True,exist_ok=True)
        payload=b'II'+struct.pack('<HI',42,8);body_path=response/'raw/body.bin';body_path.write_bytes(payload)
        http={'success':True,'url':request['url'],'ranges_inclusive':request['ranges'],'mode':'single',
              'http_status':206,'file':record(body_path),'charged_response_bytes':8,'retained_body_bytes':8,
              'reserved_cap_bytes':9,'etag':'"stable"','object_total_bytes':TOTAL,'request_headers':{}}
        ledger={'requests':[http],'automatic_retries':0,'redirects_followed':0,'verified_tls':True}
        save(response/'request-ledger.json',ledger)
        terminal={'status':'PASS','request_id':('changed'if bad_binding else request['request_id']),
                  'request':record(config_path),'body':record(body_path),'transport_ledger':record(response/'request-ledger.json'),
                  'status_code':206,'etag':'"stable"','object_total_bytes':TOTAL,'charged_response_bytes':8,'retained_body_bytes':8}
        save(folder/'terminal.json',terminal)
    class FakeProcess:
        def __init__(self,factory,index):
            self.factory,self.index,self.pid,self.returncode=factory,index,50000+index,None
        def poll(self):
            if self.factory.count>=4:
                self.returncode=1 if self.index==2 else 0
            return self.returncode
        def wait(self,timeout=None):
            self.returncode=self.returncode if self.returncode is not None else -15
            return self.returncode
    class Factory:
        count=0
        def __call__(self,command,**kwargs):
            self.count+=1
            config_path=Path(command[command.index('--request')+1])
            ledger=json.loads((config_path.parent.parent.parent/'request-ledger.json').read_text())
            require(ledger['requests'][-1]['status']=='RESERVED','Reservation must be durable before spawn')
            if self.count!=2:
                child_success(config_path)
            return FakeProcess(self,self.count)
    factory=Factory();controller=Controller(root/'scheduler-failure',fake_plan,{'synthetic':True},spawn=factory)
    controller.complete=lambda *_: None
    rejected('completed failure stops new scheduling',controller.run)
    passed('four maximum and no completed RESERVED entries',factory.count==4 and not controller.active and
        len(controller.ledger['requests'])==4 and all(e['status']in ('PASS','STOP')for e in controller.ledger['requests']))
    factory=Factory();bad_controller=Controller(root/'scheduler-binding',fake_plan,{'synthetic':True},spawn=factory)
    job=single_job_for_fixture={'request_id':fake_plan['requests'][0]['request_id'],'object_id':synthetic_objects[0]['object_id'],
                               'phase':'header','url':synthetic_objects[0]['url'],'mode':'single','ranges':[[0,7]],'pin':None,'reserved_cap_bytes':9}
    bad_controller.launch(job);active_job=next(iter(bad_controller.active.values()));active_job['process'].returncode=0
    child_success(active_job['folder']/'request.json',bad_binding=True)
    rejected('parent bad binding becomes STOP',lambda:bad_controller.settle(active_job))
    passed('bad binding keeps full charge and child evidence',active_job['entry']['status']=='STOP'and
        bad_controller.ledger['charged_response_bytes']==9 and 'parent_failure'in active_job['entry']and(active_job['folder']/'terminal.json').exists())
    bad_controller.active.clear()
    def spawn_failure(*a,**k):
        raise OSError('synthetic spawn failure')
    failed_spawn=Controller(root/'scheduler-spawn',fake_plan,{'synthetic':True},spawn=spawn_failure)
    try:failed_spawn.run()
    except OSError:pass
    else:raise AssertionError('Expected synthetic spawn failure')
    passed('spawn failure terminal and full reservation',len(failed_spawn.ledger['requests'])==1 and
        failed_spawn.ledger['requests'][0]['status']=='STOP'and failed_spawn.ledger['charged_response_bytes']==9)
    deadline=Controller(root/'scheduler-deadline',fake_plan,{'synthetic':True},spawn=spawn_failure)
    deadline.deadline=time.monotonic()-1
    rejected('expired deadline cannot spawn',deadline.run)
    passed('expired deadline has no requests or active children',len(deadline.ledger['requests'])==0 and not deadline.active)
    receipt=dict(status='PASS',schema_version=1,synthetic_only=True,cases=len(results),checks=results,
                 code=[record(ROOT/n)for n in CODE_NAMES],network_requests=0,numeric_pixel_decodes=0,
                 actual_source_arrays_read=0,model_calls=0)
    save(root/'fixture-receipt.json',receipt)
    print(json.dumps({'status':'PASS','cases':len(results),'receipt':record(root/'fixture-receipt.json')}))


if __name__=='__main__':
    main()
