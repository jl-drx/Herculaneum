"""Independent source-array and scalar PPM reconstruction; no network/model."""
import math,resource,struct,time
import numpy as np
from PIL import Image
from packet_common import *
started=time.monotonic();require(not (HERE/'PACKET_GEOMETRY_VERIFICATION.json').exists(),'Write-once verification')
for name in ['PACKET_GO.json','GEOMETRY_GO.json']:
    for p in json.loads((HERE/name).read_text())['files']:verify(p)
packet=json.loads((HERE/'packet-v1/COMPLETE.json').read_text());plan=json.loads((HERE/'PACKET_PLAN.json').read_text())
values=optical_values=section_values=0
with Image.open(verify(plan['optical'])) as ir:
    for record in packet['records']:
        c=record['source'];raw=np.load(verify(c['raw']),allow_pickle=False);arr=np.load(verify(record['arrays']),allow_pickle=False)
        pooled=np.zeros((63,128,128),np.uint32)
        for dy in range(3):
            for dx in range(3):pooled+=raw[:,dy::3,dx::3]
        require(np.array_equal(pooled.astype(np.float64)/9,arr['CT_mean3']),'Every displayed CT mean');values+=pooled.size
        yy,xx,h,w=c['source_bbox_yxhw'];opt=np.asarray(ir.crop((xx,yy,xx+w,yy+h)))
        require(np.array_equal(opt,arr['optical_RGBA']),'Exact optical crop');optical_values+=opt.size
        for key,fixedrow in [('row193',True),('column193',False)]:
            independent=np.stack([raw[z,193,:] if fixedrow else raw[z,:,193] for z in range(63)])
            require(np.array_equal(independent,arr[key]),'Exact source cross-section');section_values+=independent.size
        verify(record['montage']);arr.close()
verify(packet['comparison'])
gplan=json.loads((HERE/'GEOMETRY_PLAN.json').read_text());g=json.loads((HERE/'geometry-v1/COMPLETE.json').read_text());ledger=json.loads(verify(g['ledger']).read_text())
require(len(ledger['requests'])==1 and ledger['requests'][0]['success'],'One completed geometry request')
entry=ledger['requests'][0];body=verify(entry['file']).read_bytes();require(entry['identity']['etag']==gplan['object']['etag'],'Pinned geometry identity')
parts={tuple(p['range_inclusive']):p for p in entry['parts']};assembled=[]
for site in gplan['sites']:
    y,x=site['global_yx']
    for dy in [-1,0,1]:
        start=73+((y+dy)*6205+x-1)*48;span=(start,start+143);require(span in parts,'Independently addressed PPM row')
        p=parts[span];payload=body[p['payload_offset']:p['payload_offset']+144]
        require(hashlib.sha256(payload).hexdigest()==p['payload_sha256'],'Payload checksum');assembled.append(payload)
raw=b''.join(assembled);require(raw==verify(g['raw_records']).read_bytes(),'Scalar-address record reconstruction')
records=list(struct.iter_unpack('<6d',raw));require(len(records)==324,'324 records')
prior=json.loads((REPORTS/'twentyfirst-experiment/frag6-stage-b/run-v1/acquisition/SPARSE_GEOMETRY.json').read_text());prior_map={}
for site in prior['geometry']['samples']:
    y,x=site['center_yx']
    for index,r in enumerate(site['records_row_major']):prior_map[(y+index//3-1,x+index%3-1)]=r
maxerr=0.;overlap=0;checked=0

def compare(a,b):
    global maxerr,checked
    if a is None or b is None:require(a is b,'Undefined geometry differs');return
    d=abs(float(a)-float(b));maxerr=max(maxerr,d);checked+=1;require(d<=1e-10,'Scalar geometry differs')
for i,site in enumerate(g['sites']):
    q=records[9*i:9*i+9];r=site['geometry'];mapped=[any(v!=0 for v in rec) for rec in q]
    require(sum(mapped)==r['mapped_records'] and all(mapped)==r['complete'],'Mapped record counts')
    for z,rec in enumerate(q):
        coord=(site['global_yx'][0]+z//3-1,site['global_yx'][1]+z%3-1)
        if coord in prior_map:
            old=prior_map[coord];oldvalues=tuple((old['xyz'] or [0,0,0])+old['normal'])
            require(struct.pack('<6d',*rec)==struct.pack('<6d',*oldvalues),'Cached geometry overlap differs');overlap+=1
        if mapped[z]:require(all(math.isfinite(v) for v in rec) and abs(math.hypot(*rec[3:])-1)<=1e-5,'Mapped scalar invariants')
    compare(max(abs(math.hypot(*rec[3:])-1) for rec,m in zip(q,mapped) if m),r['normal_norm_max_error'])
    if mapped[4]:
        xyz=q[4][:3];normal=q[4][3:];require(list(xyz)==r['center_XYZ'] and list(normal)==r['center_normal'],'Center records')
        valid=[all(0<=xyz[k]+offset*normal[k]<=bound for k,bound in enumerate([6299,2259,8854])) for offset in range(-31,32)]
        require(valid==r['layer_validity'] and sum(valid)==r['valid_layer_count'],'Scalar ray bounds')
    if all(mapped):
        dx=[(q[5][k]-q[3][k])/2 for k in range(3)];dy=[(q[7][k]-q[1][k])/2 for k in range(3)]
        require(dx==r['tangent_x'] and dy==r['tangent_y'],'Scalar finite differences')
        nx=math.hypot(*dx);ny=math.hypot(*dy);dot=sum(a*b for a,b in zip(dx,dy));cross=[dx[1]*dy[2]-dx[2]*dy[1],dx[2]*dy[0]-dx[0]*dy[2],dx[0]*dy[1]-dx[1]*dy[0]];area=math.hypot(*cross)
        eigdisc=math.sqrt((nx*nx-ny*ny)**2+4*dot*dot);sv=[math.sqrt((nx*nx+ny*ny+sgn*eigdisc)/2) for sgn in [1,-1]]
        expected=dict(tangent_x_norm=nx,tangent_y_norm=ny,tangent_cosine=dot/(nx*ny),area_scale=area,condition_number=sv[0]/sv[1],
            normal_dot_unit_tangent_x=sum(a*b for a,b in zip(normal,dx))/nx,normal_dot_unit_tangent_y=sum(a*b for a,b in zip(normal,dy))/ny,
            normal_dot_unit_raster_cross=sum(a*b for a,b in zip(normal,cross))/area)
        for key,val in expected.items():compare(val,r[key])
        for a,b in zip(sv,r['singular_values']):compare(a,b)
require(time.monotonic()-started<120,'Verification wall bound');rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;require(rss<1024**3,'Verification memory bound')
receipt=dict(status='PASS',created_at_utc=now(),verifier=pin(__file__),packet=pin(HERE/'packet-v1/COMPLETE.json'),geometry=pin(HERE/'geometry-v1/COMPLETE.json'),
    displayed_CT_mean_values=values,optical_channel_values=optical_values,cross_section_values=section_values,PPM_records=324,PPM_sites=36,
    scalar_geometry_values=checked,maximum_geometry_difference=maxerr,overlapping_prior_records=overlap,
    elapsed_seconds=time.monotonic()-started,peak_RSS_bytes=rss,new_source_requests=0,new_model_forwards=0,independent_agent=False)
save(HERE/'PACKET_GEOMETRY_VERIFICATION.json',receipt);print(json.dumps(receipt,indent=2))
