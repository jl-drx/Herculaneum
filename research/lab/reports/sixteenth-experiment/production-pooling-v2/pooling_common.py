"""Offline file/provenance guards; never decode CT during plan construction."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from pooling_kernels import require,row_range

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
SOURCE_SHA='d06438cbf6f74749fd01b5019a10c309b31bbd784be6859d0c217411e63e7d3a'
DESIGN_SHA='5c942839223b69b8667da841db82babc15dbf62eaed0bf37005d76130e5315d8'
SCHEMA_SHA='70c810cbfc6bc1ed174f9c3d7c5654c6e3e8f4223ec6ea101aea7dd373be2f5c'
GROUPS=[256,256,256,256,128]
CASE_IDS=[f'frag{f}-context{c}' for f in [1,2,3] for c in [1,2]]
LIMITS=dict(wall_seconds=120,cpu_seconds=360,threads=4,rss_bytes=3*1024**3,
    address_space_bytes=12*1024**3,combined_storage_bytes=2*1024**3,output_bytes=96*1024**2)


def identity(item):
    sizes=[item[k] for k in ['size_bytes','bytes','byte_length'] if k in item]
    require(sizes and all(type(x) is int and x==sizes[0] for x in sizes),'File size identity differs')
    return dict(path=str(Path(item['path']).resolve()),size_bytes=sizes[0],sha256=item['sha256'])


def record(path):
    path=Path(path).resolve(strict=True)
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=str(path),size_bytes=path.stat().st_size,sha256=digest)


def verify(item):
    expected=identity(item);require(record(expected['path'])==expected,'File pin changed: '+expected['path'])
    return Path(expected['path'])


def bounded_bytes(item,cap):
    expected=identity(item);require(0<=expected['size_bytes']<=cap,'File exceeds bounded input size')
    path=Path(expected['path']);require(path.stat().st_size==expected['size_bytes'],'Input file length changed')
    with path.open('rb') as stream:data=stream.read(cap+1)
    require(len(data)==expected['size_bytes'] and hashlib.sha256(data).hexdigest()==expected['sha256'],'Input file bytes changed')
    return data


def read_json(item,cap=8*1024**2):return json.loads(bounded_bytes(item,cap))
def write_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.pending')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');temporary.replace(path)


def module(item,name):
    spec=importlib.util.spec_from_file_location(name,verify(item));value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value


def sources():
    pin=record(HERE/'SOURCE_PINS.json');require(pin['sha256']==SOURCE_SHA,'Pooling source pins differ')
    source=read_json(pin)
    require(source['production_design']['sha256']==DESIGN_SHA and source['schema']['sha256']==SCHEMA_SHA,'Frozen design/schema differ')
    for item in source.values():
        if isinstance(item,dict) and 'path' in item:verify(item)
    return source,pin


def selected_cases(source):
    benchmark=read_json(source['benchmark']);selected={c['case_id']:c for c in benchmark['selected_contexts']}
    require(set(selected)==set(CASE_IDS),'Literal fifteenth context inventory differs')
    catalog=read_json(benchmark['source_contract'])
    fragments={c['legacy_alias'].lower():c for c in catalog['eligible_candidates']}
    return [selected[key] for key in CASE_IDS],fragments


def validate_inventory(raw,cases,fragments):
    require(raw['schema_version']==1 and raw['status']=='PASS' and raw['stage']=='pixels','Production raw acquisition incomplete')
    require(len(raw['tiff_contracts'])==153 and len(raw['batches'])==1530,'Complete153-object/1530-batch inventory required')
    contracts={};expected={(f,p) for f in ['frag1','frag2','frag3'] for p in range(7,58)}
    for c in raw['tiff_contracts']:
        key=(c['fragment'],c['plane_index']);require(key in expected and key not in contracts,'Missing/duplicate/unexpected TIFF contract')
        origin=fragments[key[0]];require(c['url']==origin['surface_volume_url']+f'{key[1]:02d}.tif'
            and [c['height'],c['width']]==origin['shape_yx'],'TIFF URL/canvas differs from frozen source')
        require((c['dtype'],c['byte_order'],c['struct_endian']) in [('<u2','little','<'),('>u2','big','>')]
            and c['orientation']==1 and c['photometric']=='min-is-black','TIFF numeric convention differs')
        require(c['strip_bytes']==2*c['height']*c['width'] and 8<=c['strip_offset']
            and c['strip_offset']+c['strip_bytes']<=c['object_total_bytes'],'TIFF strip bounds differ')
        require(isinstance(c['etag'],str) and c['etag'].startswith('"') and c['etag'].endswith('"') and not c['etag'].startswith('W/'),'Strong object ETag required')
        contracts[key]=c
    require(set(contracts)==expected,'Incomplete per-object contract inventory')
    rows={c['case_id']:c for c in cases};schedule={};ids=set()
    for b in raw['batches']:
        require(b['case_id'] in rows and type(b['plane_index']) is int,'Unknown batch context/layer')
        case=rows[b['case_id']];y,x,height,width=case['native_bbox_yxhw'];fragment=b['case_id'].split('-')[0]
        allowed={y:256,y+256:256,y+512:256,y+768:256,y+1024:128}
        key=(b['case_id'],b['plane_index'],b['native_row_start'])
        require(b['fragment']==fragment and 7<=b['plane_index']<=57 and b['native_row_start'] in allowed
            and b['row_count']==allowed[b['native_row_start']] and key not in schedule
            and b['request_id'] not in ids,'Batch gap/duplicate/route differs')
        require(height==width==1152,'Frozen native crop size differs');ids.add(b['request_id']);schedule[key]=b
    require(len(schedule)==6*51*5,'Complete row schedule required')
    return contracts,schedule


def validate_batch(index,batch,contract,case,body):
    for key in ['request_id','case_id','fragment','plane_index','native_row_start','row_count']:
        require(batch[key]==index[key],'Batch routing differs: '+key)
    y,x,_,width=case['native_bbox_yxhw'];count=batch['row_count'];start=batch['native_row_start']
    require(batch['source_url']==contract['url'] and batch['etag']==contract['etag']
        and batch['object_total_bytes']==contract['object_total_bytes'] and batch['status_code']==206
        and batch['native_x_start']==x and batch['native_crop_width']==width,'Batch source identity/status/crop differs')
    require(isinstance(batch['content_type'],str) and batch['content_type'].split(';',1)[0].strip().lower()=='multipart/byteranges','Multipart type required')
    require(len(body)<=count*(2304+768)+1024<=1024**2 and len(batch['parts'])==len(batch['rows'])==count,'Multipart bounds/coverage differ')
    ranges={tuple(row_range(contract,x,yy,width)):yy for yy in range(start,start+count)};parts={};intervals=[]
    for p in batch['parts']:
        span=tuple(p['range_inclusive']);offset,size=p['payload_offset'],p['payload_bytes']
        require(span in ranges and span not in parts and type(offset) is int and type(size) is int
            and 0<=offset and size==2304 and offset+size<=len(body),'Invalid/duplicate/missing source part')
        require(all(offset+size<=a or b<=offset for a,b in intervals),'Overlapping encoded payload intervals')
        intervals.append((offset,offset+size));headers=p['headers']
        require(set(headers)<={'content-range','content-type','content-length'}
            and headers.get('content-range')==f'bytes {span[0]}-{span[1]}/{contract["object_total_bytes"]}','Part source header differs')
        if 'content-length' in headers:require(headers['content-length']==str(size),'Part Content-Length differs')
        require(hashlib.sha256(memoryview(body)[offset:offset+size]).hexdigest()==p['payload_sha256'],'Part SHA differs');parts[span]=p
    require(set(parts)==set(ranges) and {r['native_y'] for r in batch['rows']}==set(ranges.values()),'Incomplete native rows')
    for r in batch['rows']:
        require(r['range_inclusive']==row_range(contract,x,r['native_y'],width),'Native row/source range differs')
        p=parts[tuple(r['range_inclusive'])]
        require(all(r[k]==p[k] for k in ['range_inclusive','payload_offset','payload_bytes','payload_sha256']),'Row/physical part mapping differs')


def storage_bytes(roots):
    files={}
    for root in roots:
        for path in Path(root).resolve().rglob('*'):
            if path.is_file():files[str(path.resolve())]=path.stat().st_size
    return sum(files.values())


def validate_source_go(plan):
    go=read_json(plan['source_go'])
    require(go['status']=='GO' and identity(go['code_freeze'])==identity(plan['code_freeze'])
        and identity(go['production_design'])==identity(plan['sources']['production_design'])
        and identity(go['schema'])==identity(plan['sources']['schema']),'Pooling source GO does not bind exact package/design/schema')


def load_plan(path,execute=True):
    pin=record(path);plan=read_json(pin);source,source_pin=sources()
    require(plan['kind']=='offline-Paris-CT-pooling-v1' and plan['sources']==source and plan['source_pins']==source_pin
        and plan['limits']==LIMITS,'Pooling plan/source scope differs')
    if execute:
        require(plan['ready_for_execution'] is True and plan['source_go'] is not None and plan['raw_provenance'] is not None,'Pooling plan not ready')
        validate_source_go(plan);freeze=read_json(plan['code_freeze']);require(freeze['no_further_edits'] is True,'Pooling code not sealed')
        for item in freeze['files']:
            verify(item)
            if Path(item['path']).suffix=='.py':require(plan['code'][Path(item['path']).name]==identity(item),'Pooling code differs from seal')
        require(set(plan['code'])=={p.name for p in HERE.glob('*.py')},'Executable source set differs')
        for name,item in plan['code'].items():require(verify(item)==HERE/name,'Pooling code path differs')
        require(str((HERE.parent/'production-acquisition').resolve()) in plan['production_roots'],
            'Complete producer storage root required')
        require(str((HERE.parent/'production-pooling').resolve()) in plan['production_roots'],
            'Original stopped pooling package/history must be counted')
    return plan,pin


def persisted_layout_matches(layout,contract):
    """Project only integer TIFF tag keys to their persisted JSON representation."""
    require(all(type(k) is int for k in layout['tags']),'Reconstructed TIFF tag IDs must be integers')
    projected=dict(layout,tags={str(k):v for k,v in layout['tags'].items()})
    return all(contract.get(k)==v for k,v in projected.items())
