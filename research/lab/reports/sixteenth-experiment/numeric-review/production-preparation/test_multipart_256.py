"""Synthetic-only independent128/256-part fixtures; no transport/image interpretation."""
import hashlib
import json
from pathlib import Path
import sys
from multipart_256 import parse,fixture_body,require


def data(n):
    # All source rows are2304bytes; embedded delimiter is ordinary raw data.
    payloads=[(bytes([i%256])*100+b'\r\n--fixed\r\nContent-Range: collision\0'+bytes([(i+3)%256])*2304)[:2304]
              for i in range(n)]
    spans=[(260+(1152+i)*6330*2+2304*2,260+(1152+i)*6330*2+2304*2+2303)for i in range(n)]
    return spans,[(span,payload,103571720)for span,payload in zip(spans,payloads)]


def fixtures():
    accepted=rejected=0
    for n in (128,256):
        spans,parts=data(n);wanted={span:payload for span,payload,_ in parts}
        for order in (parts,list(reversed(parts))):
            for prefix in (b'',b'\r\n'):
                encoded=prefix+fixture_body(order)
                actual=parse(encoded,'multipart/byteranges; boundary=fixed',spans,103571720)
                require(actual['parts']==wanted,'Large binary collision/order/initial-CRLF fixture differs')
                require([m['range_inclusive']for m in actual['part_metadata']]==[list(span) for span, _, _ in order],
                        'Independent extracted-position range ordering differs')
                for m in actual['part_metadata']:
                    require(m['payload_bytes']==2304 and encoded[m['payload_offset']:m['payload_offset']+2304]==wanted[tuple(m['range_inclusive'])],
                            'Independent exact byte-position fixture differs')
                accepted+=1
        wrong_parts=list(parts);wrong_parts[17]=wrong_parts[16]
        bad=[(fixture_body(parts[:-1]),spans,103571720),
             (fixture_body(wrong_parts),spans,103571720),
             (fixture_body(parts)+b'extra',spans,103571720),
             (fixture_body(parts),spans,103571721),
             (fixture_body(parts)[:-1],spans,103571720),
             (fixture_body(parts),spans[:-1],103571720),
             (fixture_body(parts),[spans[0],*spans[:-1]],103571720),
             (fixture_body(parts,alter_headers=lambda i,h:h+'Content-Transfer-Encoding: binary\r\n'),spans,103571720),
             (fixture_body(parts,alter_headers=lambda i,h:h+'Content-Length: 2303\r\n'),spans,103571720),
             (fixture_body(parts,alter_headers=lambda i,h:h+'X-Large: '+('x'*513)+'\r\n'),spans,103571720)]
        for body,ranges,total in bad:
            try:parse(body,'multipart/byteranges; boundary=fixed',ranges,total)
            except ValueError:rejected+=1
            else:raise ValueError('Malformed large multipart fixture accepted')
        body=fixture_body(parts)
        try:parse(body,'multipart/byteranges; boundary=fixed',spans,103571720,maximum_body=len(body)-1)
        except ValueError:rejected+=1
        else:raise ValueError('Encoded-body cap fixture accepted')
    spans,parts=data(256)
    try:parse(fixture_body(parts),'multipart/byteranges; boundary=fixed',spans,103571720,maximum_parts=128)
    except ValueError:rejected+=1
    else:raise ValueError('Frozen maximum-part cap ignored')
    return dict(status='PASS',accepted_cases=accepted,rejected_cases=rejected,part_counts=[128,256],
        framing_formula='768*part_count+1024',maximum_part_headers_bytes=512,
        actual_raw_CT_bodies_read=0,numeric_CT_decodes=0,network_calls=0,model_calls=0)


if __name__=='__main__':
    output=Path(sys.argv[1]);require(not output.exists(),'Fresh synthetic receipt required')
    x=fixtures();x['sources']=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                             for p in [Path(__file__).resolve(),Path(__file__).resolve().with_name('multipart_256.py')]]
    output.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(status='PASS',path=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest())))
