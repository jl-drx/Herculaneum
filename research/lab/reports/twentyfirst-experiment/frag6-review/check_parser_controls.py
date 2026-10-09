"""Independent synthetic byte fixtures; no HTTP or scientific payload reads."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'frag6-stage-a'
sys.path.insert(0, str(SOURCE))
from common import ContractError
import parsers


def pin(path):
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def ppm(fields=None, comments=(), tail=b'\xff'):
    fields = fields or [('width', '11'), ('height', '13'), ('dim', '6'),
                        ('ordered', 'true'), ('type', 'double'), ('version', '1.0')]
    raw = ''.join(x+'\n' for x in comments).encode() + ''.join(k+': '+v+'\n' for k,v in fields).encode() + b'<>\n'
    assert len(raw) < 1024
    return raw + (tail * 1024)[:1024-len(raw)], len(raw)+11*13*48


def tiff(order='<', description=b'synthetic renderer declaration\0', omit=(),
         replacements=None, extra=(), next_ifd=0):
    """Construct metadata independently; no image bytes are allocated."""
    spec = [(256,4,1,11), (257,4,1,13), (258,3,1,16), (259,3,1,1),
            (262,3,1,1), (273,4,1,0), (277,3,1,1), (278,4,1,13), (279,4,1,286)]
    spec = [x for x in spec if x[0] not in omit]
    if description is not None:
        spec.append((270,2,len(description),0))
    spec.extend(extra)
    table_end = 8+2+12*len(spec)+4
    description_offset = table_end+2
    strip_offset = description_offset+(len(description) if description is not None and len(description)>4 else 0)+16
    total = strip_offset+286+32
    replacements = replacements or {}
    rows=[]
    for tag, kind, number, value in spec:
        if tag == 273: value=strip_offset
        if tag == 270: value=description_offset
        if tag in replacements: kind,number,value=replacements[tag]
        if tag==270 and kind==2 and number<=4 and tag not in replacements:
            raw=description.ljust(4,b'\0')
        elif kind==3 and number==1:
            raw=struct.pack(order+'H',value)+b'\0\0'
        else:
            raw=struct.pack(order+'I',value)
        rows.append(struct.pack(order+'HHI',tag,kind,number)+raw)
    table=struct.pack(order+'H',len(rows))+b''.join(rows)+struct.pack(order+'I',next_ifd)
    header=(b'II' if order=='<' else b'MM')+struct.pack(order+'HI',42,8)
    return header, table, total, dict(strip_offset=strip_offset, description_offset=description_offset)


def parse_tiff(fixture):
    header,table,total,_=fixture
    h=parsers.tiff_header(header,total)
    count=parsers.tiff_count(table[:2],h,total)
    return parsers.tiff_ifd(table,h,count,total,(11,13))


def main():
    checks=[]
    def check(name, result):
        assert result, name
        checks.append(name)
    def rejects(name, callback):
        try: callback()
        except ContractError: checks.append(name)
        else: raise AssertionError(name)

    data,total=ppm(comments=('# Synthetic source declaration only',))
    parsed=parsers.ppm_prefix(data,total)
    check('PPM canvas and record extent', (parsed['width'],parsed['height'],parsed['record_bytes'])==(11,13,48))
    check('PPM exact UTF-8 header preservation', data[:parsed['data_offset']].decode()==parsed['header_text'])
    check('PPM invalid UTF-8 payload tail remains uninterpreted', parsed['numerical_payload_decoded'] is False and parsed['numerical_byte_order'] is None)
    check('PPM comments do not authenticate source volume', parsed['source_volume_binding'] is None and parsed['plane_offset_rule'] is None)
    fields=[('width','11'),('height','13'),('dim','6'),('ordered','true'),('type','double'),('version','1')]
    p,t=ppm(fields)
    check('PPM supported version 1', parsers.ppm_prefix(p,t)['fields']['version']=='1')
    for name,key,bad in [('small width','width','10'),('leading-zero width','width','011'),
                         ('unordered','ordered','false'),('wrong dimension','dim','3'),
                         ('wrong type','type','float'),('unsupported version','version','2')]:
        altered=[(k,bad if k==key else v) for k,v in fields]
        p,t=ppm(altered)
        rejects('PPM '+name,lambda p=p,t=t:parsers.ppm_prefix(p,t))
    for name,altered in [('duplicate field',fields+[('width','11')]),
                         ('unknown field',fields+[('byte_order','little')]),
                         ('missing field',fields[:-1])]:
        p,t=ppm(altered);rejects('PPM '+name,lambda p=p,t=t:parsers.ppm_prefix(p,t))
    rejects('PPM changed total',lambda:parsers.ppm_prefix(data,total+1))
    rejects('PPM wrong prefix length',lambda:parsers.ppm_prefix(data[:-1],total))
    rejects('PPM absent terminator',lambda:parsers.ppm_prefix(data.replace(b'<>\n',b'xx\n'),total))
    rejects('PPM CRLF rejected',lambda:parsers.ppm_prefix(data.replace(b'\n',b'\r',1),total))
    rejects('PPM invalid UTF-8 header rejected',lambda:parsers.ppm_prefix(b'\xff'+data[1:],total))

    for order in ('<','>'):
        result=parse_tiff(tiff(order))
        check('TIFF '+order+' independent byte order',result['byte_order']==('little' if order=='<' else 'big'))
        check('TIFF '+order+' one unsigned 16-bit strip',result['dtype']=='uint16' and result['strip_bytes']==286)
        check('TIFF '+order+' external description span',result['description'] is None and len(result['description_range'])==2)
        inline=parse_tiff(tiff(order,b'abc\0'))
        check('TIFF '+order+' inline ASCII description',inline['description']['literal_ascii']=='abc\0' and inline['description_range'] is None)
        absent=parse_tiff(tiff(order,None))
        check('TIFF '+order+' explicit absent description',absent['description'] is None and absent['description_range'] is None)
    default=parse_tiff(tiff(omit=(259,277,278)))
    check('TIFF complete declared defaults',default['baseline_defaults_applied']=={'259':1,'266':1,'274':1,'277':1,'284':1,'317':1,'339':1,'278':4294967295})
    check('TIFF default RowsPerStrip covers complete image',default['rows_per_strip']==4294967295)
    for tag in (258,259,262,266,274,277,284,317,339):
        replacement={tag:(4,1,16 if tag==258 else 1)}
        extra=() if tag in (258,259,262,277) else ((tag,3,1,1),)
        rejects('TIFF SHORT-only tag '+str(tag)+' rejects LONG',lambda replacement=replacement,extra=extra:parse_tiff(tiff(replacements=replacement,extra=extra)))
    for tag,value in [(258,8),(259,5),(262,0),(277,3),(278,1),(279,284)]:
        kind=3 if tag in (258,259,262,277) else 4
        rejects('TIFF unsupported scalar '+str(tag),lambda tag=tag,kind=kind,value=value:parse_tiff(tiff(replacements={tag:(kind,1,value)})))
    rejects('TIFF duplicate tag',lambda:parse_tiff(tiff(extra=((256,4,1,11),))))
    rejects('TIFF tiled layout rejected',lambda:parse_tiff(tiff(extra=((322,4,1,11),))))
    rejects('TIFF extra samples rejected',lambda:parse_tiff(tiff(extra=((338,3,1,1),))))
    rejects('TIFF next-IFD cycle rejected',lambda:parse_tiff(tiff(next_ifd=8)))
    rejects('TIFF out-of-object external metadata',lambda:parse_tiff(tiff(extra=((65000,5,1,2**32-8),))))
    rejects('TIFF external metadata overlaps header',lambda:parse_tiff(tiff(extra=((65000,5,1,4),))))
    rejects('TIFF external metadata overlaps IFD',lambda:parse_tiff(tiff(extra=((65000,5,1,8),))))
    fixture=tiff();info=fixture[3]
    rejects('TIFF description overlaps image strip',lambda:parse_tiff(tiff(replacements={270:(2,31,info['strip_offset'])})))
    rejects('TIFF strip overlaps header',lambda:parse_tiff(tiff(replacements={273:(4,1,4)})))
    rejects('TIFF strip overlaps IFD',lambda:parse_tiff(tiff(replacements={273:(4,1,8)})))
    rejects('TIFF strip extends outside object',lambda:parse_tiff(tiff(replacements={273:(4,1,2**32-1)})))
    rejects('TIFF oversized description',lambda:parse_tiff(tiff(description=b'x'*65536+b'\0')))
    rejects('TIFF unterminated inline description',lambda:parse_tiff(tiff(description=b'bad!')))
    rejects('TIFF unsupported field type',lambda:parse_tiff(tiff(extra=((65000,13,1,100),))))
    rejects('TIFF zero field count',lambda:parse_tiff(tiff(extra=((65000,4,0,0),))))
    header,table,total,_=tiff()
    rejects('BigTIFF magic rejected',lambda:parsers.tiff_header(header[:2]+struct.pack('<H',43)+header[4:],total))
    rejects('TIFF invalid endian marker',lambda:parsers.tiff_header(b'XX'+header[2:],total))
    rejects('TIFF IFD pointer below header',lambda:parsers.tiff_header(header[:4]+struct.pack('<I',4),total))
    rejects('TIFF IFD pointer outside object',lambda:parsers.tiff_header(header[:4]+struct.pack('<I',total),total))
    h=parsers.tiff_header(header,total)
    rejects('TIFF more than 128 entries',lambda:parsers.tiff_count(struct.pack('<H',129),h,total))
    rejects('TIFF zero entries',lambda:parsers.tiff_count(b'\0\0',h,total))
    count=parsers.tiff_count(table[:2],h,total)
    rejects('TIFF changing count between requests',lambda:parsers.tiff_ifd(struct.pack('<H',count['entries']-1)+table[2:],h,count,total,(11,13)))
    rejects('TIFF shape differs from PPM',lambda:parsers.tiff_ifd(table,h,count,total,(13,11)))
    rejects('ASCII description non-ASCII',lambda:parsers.ascii_description(b'\xff\0'))
    rejects('ASCII description no terminal NUL',lambda:parsers.ascii_description(b'bad'))
    result=dict(status='PASS',checks=checks,check_count=len(checks),
                source_pins=[pin(SOURCE/name) for name in ('common.py','parsers.py')],
                fixture_source=pin(Path(__file__).resolve()),live_requests=0,
                scientific_arrays_read=0,numerical_ppm_records_decoded=0,image_pixels_decoded=0)
    with (HERE/'parser-controls-v1.json').open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status='PASS',check_count=len(checks))))


if __name__=='__main__': main()
