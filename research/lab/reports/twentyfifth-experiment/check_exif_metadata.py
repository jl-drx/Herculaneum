import json
from pathlib import Path
import struct
import sys
from exif_metadata import inspect_exif


def fixture(endian='<'):
    marker=b'II' if endian=='<' else b'MM'
    # Selected ASCII artist, skipped JPEG offset and subordinate EXIF pointer.
    entries=[struct.pack(endian+'HHII',315,2,7,50),struct.pack(endian+'HHII',513,4,1,57),
             struct.pack(endian+'HHII',34665,4,1,57)]
    return marker+struct.pack(endian+'HIH',42,8,3)+b''.join(entries)+struct.pack(endian+'I',57)+b'Author\0'+b'POISON_IMAGE'


def parse(data):
    requests=[]
    def get(at,n,kind):
        assert at+n<=57,'Forbidden image/subordinate payload'
        requests.append((at,n));return data[at:at+n]
    return inspect_exif(get,len(data))


checks=[]
for endian in ('<','>'):
    r=parse(fixture(endian));assert r['selected_fields']=={'Artist':'Author'} and not r['next_ifd_followed']
    checks.append('Selected ASCII and skip thumbnail/subordinate/next IFD '+endian)
for name,at,new in [('outside directory',4,struct.pack('<I',999)),('oversized directory',8,struct.pack('<H',25)),
                    ('outside string',18,struct.pack('<I',999)),('overlap string',18,struct.pack('<I',10)),
                    ('string cap',14,struct.pack('<I',257)),('wrong type',12,struct.pack('<H',4)),
                    ('nonterminated ASCII',56,b'x')]:
    d=bytearray(fixture());d[at:at+len(new)]=new
    try:parse(bytes(d))
    except ValueError:checks.append(name)
    else:raise AssertionError(name)
p=Path(sys.argv[1]);assert not p.exists();p.write_text(json.dumps(dict(status='PASS',count=len(checks),checks=checks,network_calls=0),indent=2)+'\n')
print(json.dumps(dict(status='PASS',count=len(checks))))
