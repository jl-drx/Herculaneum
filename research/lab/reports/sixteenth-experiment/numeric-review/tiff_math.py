"""Independent classic-TIFF metadata arithmetic; no network or pixel decoding."""
import struct

TYPE_BYTES={1:1,2:1,3:2,4:4,5:8,6:1,7:1,8:2,9:4,10:8,11:4,12:8}
DEFAULTS={259:1,266:1,274:1,277:1,278:2**32-1,284:1,317:1,339:1}


def require(condition,message):
    if not condition:raise ValueError(message)


def overlaps(a,b):return a[0]<b[1] and b[0]<a[1]


def inspect(header,ifd,expected_shape,total):
    require(isinstance(header,bytes) and len(header)==8 and header[:4]==b'II*\0',
            'Classic little-endian8-byte header required')
    require(type(total) is int and total>8 and len(expected_shape)==2 and
            all(type(n) is int and n>0 for n in expected_shape),'Positive object/canvas size required')
    origin=int.from_bytes(header[4:8],'little')
    require(origin>=8 and origin+2<=total and len(ifd)>=6,'First IFD bounds invalid')
    count=int.from_bytes(ifd[:2],'little')
    require(0<count<=128 and len(ifd)==2+12*count+4 and origin+len(ifd)<=total,
            'Exact bounded IFD bytes required')
    require(ifd[-4:]==b'\0'*4,'Multiple TIFF images unsupported')
    tags={};external=[]
    for index in range(count):
        raw=ifd[2+12*index:14+12*index]
        tag,kind,number=struct.unpack('<HHI',raw[:8])
        require(tag not in tags and kind in TYPE_BYTES and number>0,'Duplicate/ambiguous TIFF entry')
        value=int.from_bytes(raw[8:12],'little');extent=TYPE_BYTES[kind]*number
        tags[tag]=dict(type=kind,count=number,value_or_offset=value,value_bytes_hex=raw[8:12].hex())
        if extent>4:
            require(value>=8 and value+extent<=total,'External TIFF metadata outside object')
            external.append(dict(tag=tag,start=value,stop_exclusive=value+extent))
    require(not ({320,322,323,324,325,330,338,347,513,514}&tags.keys()),'Tiled/colored/JPEG/subsampled TIFF unsupported')
    applied={};values={}
    def scalar(tag):
        if tag not in tags:
            require(tag in DEFAULTS,'Required TIFF scalar absent')
            applied[tag]=DEFAULTS[tag];values[tag]=DEFAULTS[tag];return DEFAULTS[tag]
        t=tags[tag]
        require(t['count']==1 and t['type'] in (3,4),'Inline SHORT/LONG scalar required')
        raw=bytes.fromhex(t['value_bytes_hex']);result=int.from_bytes(raw[:2] if t['type']==3 else raw,'little')
        values[tag]=result;return result
    h,w=expected_shape
    require(scalar(256)==w and scalar(257)==h,'TIFF canvas differs')
    for tag,wanted in {258:16,259:1,262:1,266:1,274:1,277:1,284:1,317:1,339:1}.items():
        require(scalar(tag)==wanted,'Unsupported TIFF sample/layout convention')
    require(scalar(278)>=h,'Multiple strips unsupported')
    offset,length=scalar(273),scalar(279)
    require(length==2*w*h and offset>=8 and offset+length<=total,'Exact uint16 strip bounds invalid')
    ranges=[(0,8),(origin,origin+len(ifd)),(offset,offset+length)]
    require(not overlaps(ranges[1],ranges[2]),'TIFF IFD overlaps pixels')
    for e in external:
        require(not any(overlaps((e['start'],e['stop_exclusive']),r) for r in ranges),
                'External TIFF metadata overlaps header/IFD/pixels')
    return dict(shape_yx=list(expected_shape),ifd_offset=origin,ifd_length=len(ifd),strip_offset=offset,
        strip_bytes=length,object_total_bytes=total,dtype='<u2',orientation=1,photometric='min-is-black',
        tags={str(k):v for k,v in tags.items()},scalar_values={str(k):v for k,v in values.items()},
        specification_defaults_applied={str(k):v for k,v in applied.items()},external_metadata_extents=external)


def fixture_bytes(changes=None,drop=(),extra=(),next_pointer=0):
    values={256:9,257:6,258:16,259:1,262:1,266:1,273:256,274:1,
            277:1,278:6,279:108,284:1,317:1,339:1}
    values.update(changes or {})
    rows=[struct.pack('<HHII',tag,4,1,value) for tag,value in sorted(values.items()) if tag not in drop]
    rows+=list(extra)
    ifd=struct.pack('<H',len(rows))+b''.join(rows)+struct.pack('<I',next_pointer)
    return b'II*\0'+struct.pack('<I',8),ifd,(6,9),364


def fixtures():
    full=inspect(*fixture_bytes())
    require(full['strip_bytes']==108 and not full['specification_defaults_applied'],'Complete TIFF fixture differs')
    defaults=inspect(*fixture_bytes(drop=tuple(DEFAULTS)))
    require(defaults['specification_defaults_applied']=={str(k):v for k,v in DEFAULTS.items()},
            'Recorded TIFF specification defaults differ')
    external=inspect(*fixture_bytes(extra=(struct.pack('<HHII',270,2,5,230),)))
    require(external['external_metadata_extents']==[dict(tag=270,start=230,stop_exclusive=235)],
            'External metadata extent fixture differs')
    bad=[]
    for tag,value in [(256,10),(257,7),(258,8),(259,5),(262,0),(274,2),(277,2),(278,5),(279,109),(273,16),(339,2)]:
        bad.append(fixture_bytes({tag:value}))
    bad += [fixture_bytes(next_pointer=8),fixture_bytes(extra=(struct.pack('<HHII',256,4,1,9),)),
            fixture_bytes(extra=(struct.pack('<HHII',322,4,1,9),)),
            fixture_bytes(extra=(struct.pack('<HHII',270,2,5,260),)),
            fixture_bytes(extra=(struct.pack('<HHII',270,2,5,10),)),
            fixture_bytes(extra=(struct.pack('<HHII',270,2,5,363),))]
    for args in bad:
        try:inspect(*args)
        except ValueError:pass
        else:raise ValueError('Unsupported/overlapping TIFF fixture accepted')
    h,i,s,t=fixture_bytes()
    for header,ifd in [(b'MM\0*'+h[4:],i),(h,i[:-1]),(h[:7],i)]:
        try:inspect(header,ifd,s,t)
        except ValueError:pass
        else:raise ValueError('Truncated/endian TIFF fixture accepted')
    return dict(status='PASS',accepted_layout_cases=3,rejected_layout_cases=len(bad)+3,
                actual_TIFF_or_CT_objects_read=0,network_calls=0,model_calls=0)
