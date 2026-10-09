"""Pure classic TIFF structural parsing; no image/coordinate-value decoding."""
import math,struct
class ContractError(ValueError): pass
def require(x,s):
 if not x:raise ContractError(s)
TYPE_BYTES={1:1,2:1,3:2,4:4,5:8,6:1,7:1,8:2,9:4,10:8,11:4,12:8,13:4}
def extent(start,length,total):
 require(type(start) is int and type(length) is int and 0<=start<total and 0<length<=total-start,'invalid metadata extent')
 return [start,start+length]
def disjoint(a,b):return a[1]<=b[0] or b[1]<=a[0]
def header(body,total):
 require(len(body)==8,'classic header length');require(body[:2] in (b'II',b'MM'),'unknown TIFF byte order')
 order='<' if body[:2]==b'II' else '>'
 magic,offset=struct.unpack(order+'HI',body[2:]);require(magic==42,'only classic TIFF supported');require(offset>=8,'IFD overlaps header');extent(offset,2,total)
 return {'byte_order':order,'first_ifd_offset':offset,'total_bytes':total}
def count(body,head):
 require(len(body)==2,'IFD count length');n=struct.unpack(head['byte_order']+'H',body)[0];require(1<=n<=256,'IFD entry cap')
 extent(head['first_ifd_offset'],2+12*n+4,head['total_bytes']);return n
def parse_ifd(body,head,n):
 require(len(body)==12*n+4,'IFD body length');order=head['byte_order'];total=head['total_bytes'];tags={};regions=[[0,8],extent(head['first_ifd_offset'],2+12*n+4,total)]
 for i in range(n):
  raw=body[12*i:12*(i+1)];tag,typ,c=struct.unpack(order+'HHI',raw[:8]);require(tag not in tags,'duplicate TIFF tag');require(typ in TYPE_BYTES and c>0,'unsupported TIFF type/count')
  length=c*TYPE_BYTES[typ];require(length<=total,'TIFF metadata length exceeds object');inline=length<=4
  entry={'tag':tag,'type':typ,'count':c,'length':length,'inline':inline,'value_field_hex':raw[8:].hex()}
  if not inline:
   offset=struct.unpack(order+'I',raw[8:])[0];r=extent(offset,length,total);require(all(disjoint(r,q) for q in regions),'overlapping structural metadata');regions.append(r);entry['extent']=r
  tags[tag]=entry
 require(struct.unpack(order+'I',body[-4:])[0]==0,'multipage TIFF unsupported')
 return {'header':head,'tags':tags,'metadata_extents':regions}
def integer_values(entry,order,external=None):
 require(entry['type'] in (3,4),'required integer TIFF tag type')
 if entry['inline']:
  require(external is None,'unexpected external tag body');data=bytes.fromhex(entry['value_field_hex'])[:entry['length']]
 else:
  require(external is not None and len(external)==entry['length'],'external metadata length');data=external
 return list(struct.unpack(order+str(entry['count'])+('H' if entry['type']==3 else 'I'),data))
def scalar(doc,tag,default=None):
 entry=doc['tags'].get(tag)
 if entry is None:
  require(default is not None,'missing required tag '+str(tag));return default
 require(entry['inline'] and entry['count']==1,'required scalar tag not inline singleton')
 return integer_values(entry,doc['header']['byte_order'])[0]
def layout(doc):
 t=doc['tags'];w=scalar(doc,256);h=scalar(doc,257);require(w>0 and h>0,'empty dimensions')
 bits=scalar(doc,258);samples=scalar(doc,277,1);planar=scalar(doc,284,1);orientation=scalar(doc,274,1)
 require(samples==1 and planar==1,'coordinate TIFF must have one contiguous sample');require(bits in (8,16,32,64),'unsupported coordinate bits');require(orientation==1,'nondefault TIFF orientation unsupported')
 form=scalar(doc,339,1);require(form in (1,2,3),'unsupported sample format');compression=scalar(doc,259,1)
 require(compression>0,'invalid compression');has_strip=273 in t or 279 in t;has_tile=324 in t or 325 in t;require(has_strip!=has_tile,'ambiguous/missing strip or tile storage')
 if has_strip:
  require(273 in t and 279 in t,'missing strip layout tag');rps=scalar(doc,278,2**32-1);require(rps>0,'zero rows per strip');expected=(h+rps-1)//rps;offset_tag,count_tag=273,279;storage={'kind':'strips','rows_per_strip':rps}
 else:
  require(324 in t and 325 in t,'missing tile layout tag');tw=scalar(doc,322);th=scalar(doc,323);require(tw>0 and th>0,'zero tile dimensions');expected=((w+tw-1)//tw)*((h+th-1)//th);offset_tag,count_tag=324,325;storage={'kind':'tiles','tile_width':tw,'tile_height':th}
 for tag in (offset_tag,count_tag):
  e=t[tag];require(e['type'] in (3,4) and e['count']==expected,'storage metadata cardinality/type');require(e['length']<=32768,'storage metadata byte cap')
 return {'width':w,'height':h,'bits_per_sample':bits,'samples_per_pixel':samples,'sample_format':form,'compression':compression,'orientation':orientation,'planar_configuration':planar,'byte_order':doc['header']['byte_order'],'storage':storage,'storage_count':expected,'offset_tag':offset_tag,'bytecount_tag':count_tag,'photometric_interpretation':scalar(doc,262,1),'coordinate_values_decoded':False}
def finish(doc,lay,external):
 order=doc['header']['byte_order'];ot,ct=lay['offset_tag'],lay['bytecount_tag'];offsets=integer_values(doc['tags'][ot],order,external.get(ot));counts=integer_values(doc['tags'][ct],order,external.get(ct));require(len(offsets)==len(counts)==lay['storage_count'],'storage array lengths')
 payload=[]
 for o,c in zip(offsets,counts):
  r=extent(o,c,doc['header']['total_bytes']);require(all(disjoint(r,m) for m in doc['metadata_extents']),'storage overlaps metadata');payload.append(r)
 ordered=sorted(payload);require(all(disjoint(a,b) for a,b in zip(ordered,ordered[1:])),'overlapping image storage')
 if lay['compression']==1:
  require(lay['bits_per_sample']%8==0,'uncompressed sample alignment');bps=lay['bits_per_sample']//8
  if lay['storage']['kind']=='strips':
   rows=lay['storage']['rows_per_strip'];expected=[min(rows,lay['height']-i*rows)*lay['width']*bps for i in range(len(counts))]
  else:expected=[lay['storage']['tile_width']*lay['storage']['tile_height']*bps]*len(counts)
  require(counts==expected,'uncompressed storage byte counts')
 return {**lay,'storage_offsets':offsets,'storage_byte_counts':counts,'metadata_extents':doc['metadata_extents'],'object_size_bytes':doc['header']['total_bytes'],'tags':doc['tags']}
def validate_http(response,obj,lo,hi):
 require(response.status==206,'expected206; no full-object fallback')
 for name in ('Content-Range','ETag','Content-Length','Content-Encoding'):
  require(len(response.headers.get_all(name,[]))<=1,'duplicate '+name)
 require(response.headers.get('Content-Range')==f'bytes {lo}-{hi}/{obj["object_size_bytes"]}','wrong Content-Range')
 require(response.headers.get('ETag')==obj['strong_etag'] and not obj['strong_etag'].startswith('W/'),'changed/weak ETag')
 require(response.headers.get('Content-Encoding','identity').lower()=='identity','nonidentity encoding')
 if response.headers.get('Content-Length') is not None:require(response.headers['Content-Length']==str(hi-lo+1),'wrong range length')
