"""Pure bounded row decoding and exact pooling; no filesystem, transport or model code."""
import hashlib
import numpy as np


def require(condition,message):
    if not condition:raise ValueError(message)


def row_range(contract,x,y,width):
    require(all(type(v) is int for v in [x,y,width,contract['width'],contract['height'],contract['strip_offset'],contract['strip_bytes'],contract['object_total_bytes']]),'Integer TIFF coordinates required')
    require(0<=x and width>0 and x+width<=contract['width'] and 0<=y<contract['height'],'Row escapes TIFF canvas')
    start=contract['strip_offset']+2*(y*contract['width']+x);stop=start+2*width
    require(contract['strip_offset']<=start<stop<=contract['strip_offset']+contract['strip_bytes']
        and stop<=contract['object_total_bytes'],'Row escapes strip/object bounds')
    return [start,stop-1]


def decode_row(body,row,expected_range,dtype,width):
    require(dtype in ['<u2','>u2'],'Declared endian-aware unsigned16 dtype required')
    offset,size=row['payload_offset'],row['payload_bytes']
    require(type(offset) is int and type(size) is int and 0<=offset and size==2*width and offset+size<=len(body),'Payload length/offset differs')
    require(row['range_inclusive']==expected_range and expected_range[1]-expected_range[0]+1==size,'Source row range differs')
    payload=memoryview(body)[offset:offset+size]
    require(hashlib.sha256(payload).hexdigest()==row['payload_sha256'],'Row payload SHA differs')
    values=np.frombuffer(payload,dtype=np.dtype(dtype))
    require(values.shape==(width,),'Decoded row shape differs')
    return values.astype(np.uint32,copy=True)


def assemble_group(body,rows,contract,x,y,row_count,width):
    require(type(row_count) is int and row_count>0 and len(rows)==row_count,'Row-count contract differs')
    require(len({row['native_y'] for row in rows})==row_count,'Duplicate native row')
    expected=set(range(y,y+row_count));require({row['native_y'] for row in rows}==expected,'Missing/unexpected native rows')
    output=np.empty((row_count,width),np.uint32);intervals=[]
    for row in rows:
        native_y=row['native_y'];require(type(native_y) is int,'Integer native row required')
        interval=(row['payload_offset'],row['payload_offset']+row['payload_bytes'])
        require(all(interval[1]<=a or b<=interval[0] for a,b in intervals),'Overlapping body payloads')
        intervals.append(interval)
        output[native_y-y]=decode_row(body,row,row_range(contract,x,native_y,width),contract['dtype'],width)
    require(output.flags.c_contiguous and output.dtype.isnative,'Native contiguous row assembly required')
    return output


def pool_triplet(planes):
    require(isinstance(planes,np.ndarray) and planes.ndim==3 and planes.shape[0]==3
        and planes.dtype.kind=='u' and planes.dtype.itemsize in [2,4],'Three unsigned16/32 source planes required')
    _,height,width=planes.shape
    require(height>0 and width>0 and height%3==width%3==0,'XYZ factor3 alignment differs')
    require(np.all(planes<=65535),'Unsigned16 sample range exceeded')
    native=planes.astype(np.uint32,copy=False)
    summed=native.reshape(3,height//3,3,width//3,3).sum(axis=(0,2,4),dtype=np.uint32)
    require(summed.dtype==np.uint32 and np.all(summed<=27*65535),'Exact uint32 sum bound differs')
    # One float64 division, followed by one native float32 conversion; no quantization.
    output=(summed.astype(np.float64)/np.float64(6939)).astype(np.float32)
    require(output.dtype.isnative and output.flags.c_contiguous and np.isfinite(output).all()
        and np.all((output>=0)&(output<=255)),'Native FP32 pooling output differs')
    return output
