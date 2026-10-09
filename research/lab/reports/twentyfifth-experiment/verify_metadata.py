"""Offline second-pass audit of retained requests; no model or network imports."""
from pathlib import Path
import hashlib
import json
import struct
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent
checks=0


def require(ok):
    global checks
    checks+=1
    if not ok:raise AssertionError(f'Check {checks}')


def pin(path):
    path=Path(path)
    return dict(path=str(path.resolve()),size_bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def ledger(name):
    log=json.loads((ROOT/name/'request-ledger.json').read_text())
    reads={}
    for r in log['requests']:
        require(r['success'])
        if r['method']=='HEAD':
            require(r['returned_bytes']==0 and r['status']==200)
            continue
        require(r['status']==206 and r['returned_bytes']==r['length'])
        require(pin(r['body']['path'])==r['body'])
        require(r['headers']['ETag']==['"656fa2a5-521e712"'])
        require(r['headers']['Content-Range']==[f"bytes {r['start']}-{r['start']+r['length']-1}/86107922"])
        require(r['headers']['Content-Length']==[str(r['length'])])
        key=(r['start'],r['length']);require(key not in reads)
        reads[key]=Path(r['body']['path']).read_bytes()
    require(sum(r['returned_bytes'] for r in log['requests'])==log['retained_body_bytes'])
    require(sum(r['reserved_bytes'] for r in log['requests'])==log['reserved_body_bytes'])
    return log,reads


def main():
    psd,reads=ledger('metadata-run-v1');exif,xreads=ledger('exif-run-v1')
    require(len(psd['requests'])==56 and len(exif['requests'])==5)
    require(sum(r['method']=='HEAD' for r in psd['requests'])==1)
    require(all(r['method']=='GET' for r in exif['requests']))
    expected={(0,30),(30,4)}
    h=reads[(0,30)]
    require(h[:12]==b'8BPS\0\1'+b'\0'*6)
    require(struct.unpack('>HIIHHI',h[12:])==(1,8853,6205,8,1,0))
    end=34+int.from_bytes(reads[(30,4)],'big');require(end==22494)
    pos=34;resources=[];xmp=None;exif_extent=None
    while pos<end:
        first=reads[(pos,7)];expected.add((pos,7));require(first[:4]==b'8BIM')
        ident=int.from_bytes(first[4:6],'big');n=first[6]
        tail_length=((n+2)//2)*2-1+4
        tail=reads[(pos+7,tail_length)];expected.add((pos+7,tail_length))
        size=int.from_bytes(tail[-4:],'big');start=pos+7+tail_length
        following=start+size+(size&1);require(following<=end)
        resources.append((ident,start,size))
        if ident==1060:
            expected.add((start,size));xmp=reads[(start,size)]
        elif ident==1058:exif_extent=(start,size)
        else:
            for at,length in reads:require(at+length<=start or at>=start+size)
        pos=following
    require(expected==set(reads) and pos==end and len(resources)==26)
    require(1028 not in [r[0] for r in resources])
    require(exif_extent==(22104,390))
    # No PSD layer-section read or thumbnail payload read occurred.
    require(all(at+length<=end for at,length in reads))
    require(xmp is not None and b'<!DOCTYPE' not in xmp and b'<!ENTITY' not in xmp)
    xml=ET.fromstring(xmp.decode('utf-8'))
    ns={'xmp':'http://ns.adobe.com/xap/1.0/','dc':'http://purl.org/dc/elements/1.1/',
        'mm':'http://ns.adobe.com/xap/1.0/mm/','event':'http://ns.adobe.com/xap/1.0/sType/ResourceEvent#',
        'rdf':'http://www.w3.org/1999/02/22-rdf-syntax-ns#'}
    def field(name):
        found=xml.findall('.//'+name,ns)
        return [''.join(e.itertext()).strip() for e in found]
    fields={name:field(name) for name in ['xmp:CreatorTool','xmp:CreateDate','xmp:ModifyDate',
        'xmp:MetadataDate','dc:creator','dc:description','mm:DocumentID','mm:OriginalDocumentID']}
    history=[]
    for item in xml.findall('.//mm:History/rdf:Seq/rdf:li',ns):
        history.append({child.tag:child.text for child in item})
    require(fields['dc:creator']==[] and fields['xmp:CreatorTool']==['Adobe Photoshop 25.2 (Macintosh)'])
    base,size=exif_extent
    eh=xreads[(base,8)];require(eh[:4]==b'II*\0')
    offset=int.from_bytes(eh[4:8],'little');require(offset==8)
    n=int.from_bytes(xreads[(base+offset,2)],'little');require(n==14)
    table=xreads[(base+offset+2,12*n+4)]
    expected_x={(base,8),(base+offset,2),(base+offset+2,12*n+4)}
    tags={};texts={}
    selected={270,305,306,315,33432,40093}
    for i in range(n):
        raw=table[i*12:i*12+12];tag,kind,count,value=struct.unpack('<HHII',raw)
        require(tag not in tags);tags[tag]=dict(type=kind,count=count,value_or_offset=value)
        if tag in selected:
            require(kind==2 and 4<count<=256 and value>=offset+2+12*n+4 and value+count<=size)
            expected_x.add((base+value,count));s=xreads[(base+value,count)]
            require(s.endswith(b'\0'));texts[str(tag)]=s[:-1].decode('ascii')
    require(expected_x==set(xreads))
    require(set(texts)=={'305','306'} and 315 not in tags and 40093 not in tags)
    require(tags[258]['value_or_offset']==16)
    require(all(base<=at and at+length<=base+size for at,length in xreads))
    require(int.from_bytes(table[-4:],'little')==296)  # Retained pointer, not followed.
    for name in ['METADATA_GO.json','EXIF_GO.json']:
        go=json.loads((ROOT/name).read_text())
        for record in go['frozen_files']:require(pin(record['path'])==record)
    result=dict(status='PASS',assertions=checks,review='Root offline second-pass audit; not an independent agent review',
        requests=61,returned_body_bytes=psd['retained_body_bytes']+exif['retained_body_bytes'],
        reserved_body_bytes=psd['reserved_body_bytes']+exif['reserved_body_bytes'],
        psd_resources_enumerated=len(resources),xmp_fields=fields,xmp_history=history,
        exif_selected_text=texts,exif_first_directory_tags=tags,
        encoding_discrepancy='PSD header depth 8 versus EXIF BitsPerSample 16; do not use embedded EXIF to override current PSD storage.',
        resource_payloads_unread='All PSD resources except XMP and the bounded EXIF first-directory/text ranges. No thumbnail/layer/composite data decoded or candidate evaluation.',
        iptc_resource_present=False,stage_C_open=False,benchmark_admission=False,network_calls_by_verifier=0,
        inputs=[pin(ROOT/'metadata-run-v1/PSD_METADATA.json'),pin(ROOT/'metadata-run-v1/request-ledger.json'),
                pin(ROOT/'exif-run-v1/EXIF_METADATA.json'),pin(ROOT/'exif-run-v1/request-ledger.json')])
    destination=ROOT/'METADATA_VERIFICATION.json';require(not destination.exists())
    destination.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ['status','assertions','requests','returned_body_bytes','reserved_body_bytes']}))


if __name__=='__main__':main()
