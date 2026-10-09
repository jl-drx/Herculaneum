"""Read only EXIF first-directory metadata and selected bounded provenance strings."""
import json
from pathlib import Path
import struct
import sys
import psd_metadata as transport

TAGS={270:'ImageDescription',305:'Software',306:'DateTime',315:'Artist',33432:'Copyright',40093:'XPAuthor'}


def inspect_exif(get, size):
    require=transport.require
    h=get(0,8,'exif-header')
    require(h[:2] in (b'II',b'MM'),'EXIF byte order')
    endian='<' if h[:2]==b'II' else '>'
    u16=lambda b:struct.unpack(endian+'H',b)[0]
    u32=lambda b:struct.unpack(endian+'I',b)[0]
    require(u16(h[2:4])==42,'EXIF TIFF magic')
    offset=u32(h[4:8]);require(8<=offset<=size-2,'EXIF directory offset')
    count=u16(get(offset,2,'exif-ifd-count'))
    require(count<=24 and offset+2+12*count+4<=size,'EXIF directory extent')
    start=offset+2;end=start+12*count+4
    entries=get(start,12*count+4,'exif-ifd-entries')
    result=dict(byte_order=h[:2].decode(),directory_offset=offset,entries=[],selected_fields={},
                next_ifd_offset=u32(entries[-4:]),next_ifd_followed=False)
    for i in range(count):
        e=entries[i*12:(i+1)*12];tag=u16(e[:2]);kind=u16(e[2:4]);n=u32(e[4:8])
        record=dict(tag=tag,type=kind,count=n,value_or_offset_hex=e[8:12].hex(),payload_read=False)
        if tag in TAGS:
            name=TAGS[tag];require(name not in result['selected_fields'],'Duplicate provenance tag')
            require(kind==(1 if name=='XPAuthor' else 2) and 0<n<=256,'Provenance string type/cap')
            if n<=4:data=e[8:8+n]
            else:
                at=u32(e[8:12]);require(8<=at and at+n<=size,'String outside EXIF resource')
                require(at+n<=offset or at>=end,'String overlaps directory')
                data=get(at,n,'exif-text')
            if name=='XPAuthor':
                require(n%2==0,'XPAuthor UTF16 length');value=data.decode('utf-16-le').rstrip('\0')
            else:
                require(data[-1:]==b'\0','ASCII termination');value=data[:-1].decode('ascii')
            result['selected_fields'][name]=value;record['payload_read']=True
        result['entries'].append(record)
    result['status']='PASS';return result


def main():
    go=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2])
    transport.require(go['status']=='GO','EXIF GO required')
    for item in go['frozen_files']:transport.require(transport.pin(item['path'])==item,'Frozen file changed')
    transport.MAX_ATTEMPTS=12;transport.MAX_RESERVED=4096
    c=transport.Client(out);c.total=go['object_bytes'];c.etag=go['etag']
    c.log['identity_source']=go['identity_source'];c.flush()
    try:
        result=inspect_exif(lambda at,n,kind:c.request(go['resource_offset']+at,n,kind),go['resource_bytes'])
        for item in go['frozen_files']:transport.require(transport.pin(item['path'])==item,'Frozen file changed')
        result.update(go=transport.pin(sys.argv[1]),http_attempts=len(c.log['requests']),returned_body_bytes=c.log['retained_body_bytes'])
        transport.save(out/'EXIF_METADATA.json',result)
        print(json.dumps(result))
    except Exception as error:
        transport.save(out/'STOP.json',dict(status='STOP',reason=type(error).__name__+': '+str(error)))
        raise


if __name__=='__main__':main()
