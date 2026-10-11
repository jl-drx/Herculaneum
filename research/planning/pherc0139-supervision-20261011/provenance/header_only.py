import urllib.request,struct,json,hashlib,pathlib,datetime
R=pathlib.Path(__file__).parent
cat=json.load(open(R/'CATALOG_TARGET_EXTRACT.json'));a=next(a for a in cat['data'] if a['type']=='ink-detection')
url='https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/'+a['origins'][0]['path']; log=[]
with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=60) as r:
 hdr=dict(r.headers); length=int(r.headers['Content-Length']);etag=r.headers['ETag'];log.append({'method':'HEAD','url':url,'status':r.status,'headers':hdr})
def getrange(off,n,label):
 req=urllib.request.Request(url,headers={'Range':f'bytes={off}-{off+n-1}','If-Match':etag})
 with urllib.request.urlopen(req,timeout=60) as r:
  assert r.status==206,r.status;b=r.read(n+1);assert len(b)==n,(off,n,len(b))
  path=R/f'teacher-metadata-{len(log):03}-{label}.bin';path.write_bytes(b)
  log.append({'method':'GET','url':url,'range':[off,off+n],'status':r.status,'headers':dict(r.headers),'sha256':hashlib.sha256(b).hexdigest(),'file':str(path)})
 return b
h=getrange(0,8,'header');endian='<' if h[:2]==b'II' else '>';typ=struct.unpack(endian+'H',h[2:4])[0];assert typ==42,typ
off=struct.unpack(endian+'I',h[4:8])[0];n=struct.unpack(endian+'H',getrange(off,2,'ifd-count'))[0];data=getrange(off+2,12*n+4,'ifd');tags=[]
sizes={1:1,2:1,3:2,4:4,5:8,6:1,7:1,8:2,9:4,10:8,11:4,12:8,16:8,17:8,18:8};fmt={1:'B',3:'H',4:'I',6:'b',8:'h',9:'i',11:'f',12:'d',16:'Q',17:'q',18:'Q'}
for i in range(n):
 raw=data[12*i:12*i+12];tag,t,cnt,ptr=struct.unpack(endian+'HHII',raw);num=cnt*sizes[t];v=raw[8:8+num] if num<=4 else getrange(ptr,num,f'tag-{tag}')
 if t==2:val=v.rstrip(b'\0').decode(errors='replace')
 elif t in fmt:val=list(struct.unpack(endian+str(cnt)+fmt[t],v))
 else:val=v.hex()
 tags.append({'tag':tag,'type':t,'count':cnt,'value':val,'out_of_line_offset':ptr if num>4 else None})
json.dump({'url':url,'length':length,'etag':etag,'tiff_type':typ,'ifd_offset':off,'tags':tags,'next_ifd_offset':struct.unpack(endian+'I',data[-4:])[0],'no_image_data_read':True},open(R/'TEACHER_TIFF_HEADER.json','w'),indent=2)
json.dump(log,open(R/'TEACHER_HEADER_REQUESTS.json','w'),indent=2)
print('Metadata requests',len(log),'total bytes',sum(x['range'][1]-x['range'][0] for x in log if 'range' in x))
print(json.dumps({str(t['tag']):t['value'] for t in tags if t['tag'] not in [273,279,324,325]},indent=2))
