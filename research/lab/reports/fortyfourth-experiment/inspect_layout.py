"""Choose one small sample per parent from metadata, then inspect fixed headers."""
import csv,io,struct
import tifffile
from intake_http import *
def main():
 cases=json.loads((P/'headers/case-manifest-hop0.bin').read_text())['samples'];manifest=list(csv.DictReader((P.parent/'fortythird-experiment/source-evidence/profilometer-manifest.csv').open()));by_path={r['raw_height_txt']:r for r in manifest};tree=json.loads((P/'headers/data-tree-hop0.bin').read_text());files={r['path']:r for r in tree if r['type']=='file'}
 chosen=[]
 for parent in sorted({r['papyrus_id'] for r in cases}):
  c=min([r for r in cases if r['papyrus_id']==parent],key=lambda r:(int(by_path[r['raw_height_txt']]['pixel_count']),r['case_id']))
  chosen.append(dict(**c,width=int(by_path[c['raw_height_txt']]['width_px']),height=int(by_path[c['raw_height_txt']]['height_px']),files={k:files[c[k]] for k in ['raw_height_txt','normalized_height_tif','brightfield_tif','label_png']}))
 save(P/'HEADER_SELECTION.json',dict(created_at_utc=now(),rule='Smallest manifest pixel count per papyrus, then case ID; before header or image inspection',cases=chosen,source_pins=[pin(P/'headers/case-manifest-hop0.bin'),pin(P/'headers/data-tree-hop0.bin'),pin(P.parent/'fortythird-experiment/source-evidence/profilometer-manifest.csv')]))
 rows=[]
 for c in chosen:
  for kind in ['raw_height_txt','normalized_height_tif','brightfield_tif','label_png']:
   url=f'https://huggingface.co/datasets/scrollprize/profilometer/resolve/{REV}/'+urllib.parse.quote(c[kind],safe='/');tag=c['case_id']+'-'+kind
   try:
    path,r=fetch(tag,url,(0,63 if kind=='label_png' else 65535));need(r['object_bytes']==c['files'][kind]['size'],'Header total vs pinned tree')
    body=path.read_bytes();meta=dict(case=c['case_id'],kind=kind,body=pin(path),object_bytes=r['object_bytes'])
    if kind.endswith('tif'):
     with tifffile.TiffFile(io.BytesIO(body)) as t:
      page=t.pages[0];meta.update(shape=list(page.shape),dtype=str(page.dtype),byteorder=t.byteorder,compression=int(page.compression),photometric=int(page.photometric),planarconfig=int(page.planarconfig),rowsperstrip=page.rowsperstrip,offsets=list(page.dataoffsets),bytecounts=list(page.databytecounts),orientation=page.tags['Orientation'].value if 'Orientation' in page.tags else None)
    elif kind=='label_png':
     need(body[:8]==b'\x89PNG\r\n\x1a\n','PNG signature');w,h,depth,color,comp,filt,inter=struct.unpack('>IIBBBBB',body[16:29]);meta.update(width=w,height=h,bit_depth=depth,color_type=color,interlace=inter)
    else:
     meta.update(first_text=body[:600].decode('utf-8-sig'),newlines_in_prefix=body.count(b'\n'))
    rows.append(meta);print(json.dumps({k:v for k,v in meta.items() if k not in ['offsets','bytecounts','body']},ensure_ascii=False),flush=True)
   except Exception as e:rows.append(dict(case=c['case_id'],kind=kind,error=type(e).__name__+': '+str(e)[:250]));print(json.dumps(rows[-1]),flush=True)
 save(P/'LAYOUTS.json',rows)
if __name__=='__main__':main()
