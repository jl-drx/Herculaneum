"""Independent length-directed multipart byte verification; no pixel decoding."""
import re


def require(condition,message):
    if not condition:raise ValueError(message)


def boundary_from_type(content_type):
    require(isinstance(content_type,str),'Multipart content type required')
    match=re.fullmatch(r'\s*multipart/byteranges\s*;\s*boundary=(?:"([^"\r\n]+)"|([^;\s]+))\s*',
                      content_type,re.IGNORECASE)
    require(match is not None,'Single multipart boundary required')
    value=match[1] if match[1] is not None else match[2]
    if match[1] is None:
        require(re.fullmatch(r"[A-Za-z0-9'+_.-]+",value) is not None,'Unquoted MIME boundary must be a token')
    require(1<=len(value)<=70 and re.fullmatch(r"[A-Za-z0-9'()+_,./:=? -]+",value) is not None and
            not value.endswith(' '),'Invalid MIME boundary')
    return value.encode('ascii')


def parse(body,content_type,ranges,total,maximum_body=262144,maximum_parts=8):
    require(isinstance(body,bytes) and 0<len(body)<=maximum_body and type(total) is int and total>0,
            'Bounded body and positive stable object size required')
    require(0<len(ranges)<=maximum_parts and all(len(r)==2 and all(type(v) is int for v in r) and
            0<=r[0]<=r[1]<total for r in ranges),'Declared bounded source ranges required')
    wanted={tuple(r) for r in ranges};require(len(wanted)==len(ranges),'Duplicate requested source range')
    ordered=sorted(wanted)
    require(all(a[1]<b[0] for a,b in zip(ordered,ordered[1:])),'Overlapping requested ranges')
    delimiter=b'--'+boundary_from_type(content_type);position=2 if body.startswith(b'\r\n') else 0;parts={};order=[]
    while True:
        require(body[position:position+len(delimiter)]==delimiter,'Exact MIME delimiter required')
        position+=len(delimiter)
        if body[position:position+4]==b'--\r\n':
            require(position+4==len(body) and set(parts)==wanted,'Final delimiter/count/trailing data differs')
            break
        require(body[position:position+2]==b'\r\n' and len(parts)<len(wanted),'MIME delimiter CRLF/count differs')
        position+=2
        end=body.find(b'\r\n\r\n',position,min(len(body),position+516))
        require(end>=position and end-position<=512,'Bounded complete MIME headers required')
        raw=body[position:end]
        try:lines=raw.decode('ascii').split('\r\n')
        except UnicodeDecodeError:raise ValueError('ASCII MIME headers required') from None
        require(1<=len(lines)<=16 and all(line and not line[0].isspace() and ':' in line for line in lines),
                'Bounded unfolded MIME headers required')
        headers={}
        for line in lines:
            name,value=line.split(':',1);name=name.lower()
            require(re.fullmatch(r'[a-z0-9-]+',name) is not None and name not in headers,'Duplicate/invalid MIME header')
            headers[name]=value.strip()
        require(set(headers)<={'content-type','content-range','content-length'} and
                all(all(32<=ord(c)<=126 or c=='\t' for c in value) for value in headers.values()),
                'Unsupported/control-character MIME headers')
        require('content-transfer-encoding' not in headers and 'content-encoding' not in headers,
                'Encoded MIME part unsupported')
        match=re.fullmatch(r'bytes ([0-9]+)-([0-9]+)/([0-9]+)',headers.get('content-range',''))
        require(match is not None,'Exact MIME Content-Range required')
        start,stop,object_size=map(int,match.groups());key=(start,stop)
        require(key in wanted and key not in parts and object_size==total,'Unexpected/duplicate part or source total')
        length=stop-start+1
        if 'content-length' in headers:
            require(headers['content-length'].isdigit() and int(headers['content-length'])==length,
                    'MIME declared length differs')
        position=end+4
        require(position+length+2<=len(body),'Truncated MIME payload')
        parts[key]=body[position:position+length];order.append(key);position+=length
        require(body[position:position+2]==b'\r\n','MIME payload trailing CRLF differs')
        position+=2
    require(len(body)-sum(len(value) for value in parts.values())<=8192,'MIME framing exceeds bounded probe contract')
    return dict(parts=parts,response_range_order=order,boundary=delimiter[2:].decode('ascii'))


def fixture_body(parts,boundary='fixed',alter_headers=None,tail=b''):
    body=b''
    for index,(span,payload,total) in enumerate(parts):
        header=f'Content-Type: application/octet-stream\r\nContent-Range: bytes {span[0]}-{span[1]}/{total}\r\n'
        if alter_headers is not None:header=alter_headers(index,header)
        body+=b'--'+boundary.encode()+b'\r\n'+header.encode()+b'\r\n'+payload+b'\r\n'
    return body+b'--'+boundary.encode()+b'--\r\n'+tail


def fixtures():
    first=b'\x00\r\n--fixed\r\nnot-a-header\xff';second=b'other\x00data'
    spans=[(10,10+len(first)-1),(100,100+len(second)-1)];total=1000
    parts=[(spans[0],first,total),(spans[1],second,total)]
    for order in (parts,list(reversed(parts))):
        result=parse(fixture_body(order),'multipart/byteranges; boundary="fixed"',spans,total)
        require(result['parts']=={spans[0]:first,spans[1]:second},'Binary delimiter collision/order fixture differs')
    failed=[(fixture_body(parts[:1]),spans,total),
            (fixture_body([parts[0],parts[0]]),spans,total),
            (fixture_body(parts)+b'extra',spans,total),
            (fixture_body(parts),spans,total+1),
            (fixture_body(parts),[spans[0],spans[0]],total),
            (fixture_body(parts)[:-1],spans,total),
            (fixture_body(parts).replace(b'\r\n',b'\n',1),spans,total),
            (fixture_body(parts,alter_headers=lambda i,h:h+'Content-Range: bytes10-20/1000\r\n'),spans,total),
            (fixture_body(parts,alter_headers=lambda i,h:h+'Content-Transfer-Encoding: binary\r\n'),spans,total),
            (fixture_body(parts,alter_headers=lambda i,h:h+'Content-Length: 999\r\n'),spans,total),
            (fixture_body(parts,alter_headers=lambda i,h:h+' folded: bad\r\n'),spans,total),
            (fixture_body(parts,alter_headers=lambda i,h:h+'X-Large: '+('x'*4096)+'\r\n'),spans,total)]
    for body,requested,size in failed:
        try:parse(body,'multipart/byteranges; boundary=fixed',requested,size)
        except ValueError:pass
        else:raise ValueError('Malformed/ambiguous MIME fixture accepted')
    for content in ('application/octet-stream','multipart/byteranges; boundary=fixed; boundary=other',
                    'multipart/byteranges; boundary=unquoted:invalid',
                    'multipart/byteranges; boundary="bad\r\nboundary"'):
        try:boundary_from_type(content)
        except ValueError:pass
        else:raise ValueError('Invalid boundary fixture accepted')
    return dict(status='PASS',accepted_binary_collision_order_cases=2,rejected_multipart_cases=len(failed),
                rejected_boundary_cases=4,actual_CT_objects_read=0,numeric_CT_decodes=0,network_calls=0,model_calls=0)
