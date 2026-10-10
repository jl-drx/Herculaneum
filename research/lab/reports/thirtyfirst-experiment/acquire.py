"""Three bounded official PNG reads; reuses qualified immutable HTTP transport."""
from pathlib import Path
import hashlib, json, struct, sys, zlib
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'thirtieth-experiment'))
import run_correspondence as transport
transport.HERE = HERE  # disk-budget root only; no retained source is modified


def png_info(path):
    raw = path.read_bytes()
    if raw[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('PNG signature')
    pos, chunks, ihdr = 8, [], None
    while pos < len(raw):
        n = struct.unpack('>I', raw[pos:pos+4])[0]
        kind = raw[pos+4:pos+8]
        data = raw[pos+8:pos+8+n]
        crc = raw[pos+8+n:pos+12+n]
        if len(crc) != 4 or zlib.crc32(kind+data) != struct.unpack('>I', crc)[0]:
            raise ValueError('PNG CRC/framing')
        if not chunks and kind != b'IHDR':
            raise ValueError('IHDR must be first')
        chunks.append(dict(type=kind.decode('ascii'), length=n))
        if kind == b'IHDR':
            if ihdr is not None or n != 13:
                raise ValueError('Duplicate/invalid IHDR')
            w,h,depth,color,compression,filtering,interlace = struct.unpack('>IIBBBBB', data)
            ihdr = dict(width=w,height=h,bit_depth=depth,color_type=color,compression=compression,
                        filtering=filtering,interlace=interlace)
            if (w,h) != (6205,8853) or depth not in (1,2,4,8) or color not in (0,2,3,4,6):
                raise ValueError('Unplanned PNG encoding/canvas')
        pos += n+12
        if kind == b'IEND':
            if n or pos != len(raw):
                raise ValueError('Invalid PNG end/trailing bytes')
            break
    if not chunks or chunks[-1]['type'] != 'IEND':
        raise ValueError('Missing IEND')
    return dict(header=ihdr,chunks=chunks,body=transport.pin(path))


def main():
    freeze=json.loads((HERE/'GO.json').read_text())
    for item in freeze['files']:
        transport.require(transport.pin(Path(item['path'])) == item,'Frozen file changed')
    plan=json.loads((HERE/'ACQUISITION_PLAN.json').read_text())
    run=HERE/'acquisition-v1';run.mkdir(exist_ok=False)
    acq=transport.Acquisition(run,plan)
    objects={}
    try:
        for obj in plan['objects']:
            dest=run/obj['name']
            acq.get(obj['url'],dest,cap=obj['max_bytes'])
            objects[obj['name']]=png_info(dest)
        transport.save(run/'PNG_OBJECTS.json',objects)
    except Exception as e:
        transport.save(run/'FAILURE.json',dict(error=type(e).__name__+': '+str(e)))
        raise
    print(json.dumps({k:v['header'] for k,v in objects.items()},indent=2))


if __name__=='__main__': main()
