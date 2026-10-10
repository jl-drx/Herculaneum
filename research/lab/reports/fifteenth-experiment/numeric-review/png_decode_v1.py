"""Independent PNG structure/encoding checks; no network or author imports."""
from io import BytesIO
from pathlib import Path
import hashlib
import struct
import zlib
import numpy as np
from PIL import Image
from integral_support import require

MAX_PIXELS=150000000


def structure(body,shape):
    require(body[:8]==b'\x89PNG\r\n\x1a\n','PNG signature differs')
    position=8; chunks=[];header=None
    while position<len(body):
        require(position+12<=len(body),'Truncated chunk framing')
        size=int.from_bytes(body[position:position+4],'big')
        kind=body[position+4:position+8];finish=position+size+12
        require(finish<=len(body),'Truncated PNG chunk payload')
        payload=body[position+8:finish-4]
        require(zlib.crc32(payload,zlib.crc32(kind))&0xffffffff==int.from_bytes(body[finish-4:finish],'big'),
                'PNG CRC mismatch')
        if not chunks:
            require(kind==b'IHDR' and size==13,'IHDR must be first')
            width,height,depth,color,compression,filtering,interlace=struct.unpack('>IIBBBBB',payload)
            require((height,width)==tuple(shape) and 0<height*width<=MAX_PIXELS,'Declared PNG canvas differs/exceeds cap')
            require(color in (0,3) and compression==filtering==0 and interlace in (0,1),'PNG type differs')
            header=dict(shape_yx=[height,width],png_bit_depth=depth,png_color_type=color)
        require(kind not in (b'tRNS',b'acTL'),'Transparent or animated image is outside binary contract')
        chunks.append(kind);position=finish
        if kind==b'IEND':
            require(size==0 and position==len(body),'IEND/trailing bytes invalid')
            break
    require(chunks.count(b'IHDR')==chunks.count(b'IEND')==1 and b'IDAT' in chunks,'PNG incomplete')
    return header


def decode_bytes(body,shape):
    facts=structure(body,shape)
    Image.MAX_IMAGE_PIXELS=MAX_PIXELS
    with Image.open(BytesIO(body)) as image:
        require(image.mode in ('1','L','P') and 'transparency' not in image.info,'Mode/transparency differs')
        mode=image.mode;palette=image.getpalette('RGB') if mode=='P' else None
        image.verify()
    with Image.open(BytesIO(body)) as image:
        image.load();raw=np.asarray(image)
        require(raw.shape==tuple(shape) and raw.dtype in (np.dtype('uint8'),np.dtype('bool')),'Decoded canvas/dtype differs')
        count=np.zeros(256,np.int64)
        for first in range(0,shape[0],128):
            block=raw[first:first+128]
            # Small bands bound bincount's integer-conversion scratch allocation.
            count+=np.bincount(block.ravel().astype(np.uint8,copy=False),minlength=256)
        used=np.flatnonzero(count).tolist()
        require(set(used)<={0,1} or set(used)<={0,255},'Raw values are not permitted binary encoding')
        rgb={}
        if mode=='P':
            require(palette is not None,'Palette missing')
            for index in used:
                color=palette[3*index:3*index+3]
                require(color==([0,0,0] if index==0 else [255,255,255]),'Used palette is ambiguous/inverted')
                rgb[str(index)]=color
        canonical=np.not_equal(raw,0).astype(np.uint8)
        facts.update(mode=mode,raw_dtype=str(raw.dtype),raw_value_counts={str(v):int(count[v]) for v in used},
            used_palette_rgb=rgb,canonicalization='raw nonzero -> uint8 1; no polarity inversion',
            encoded_bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
    return canonical.view(np.bool_),facts


def decode(path,shape):
    return decode_bytes(Path(path).read_bytes(),shape)


def fixtures():
    source=np.zeros((41,47),np.uint8);source[17:29,11:33]=1
    cases=0
    for mode,values in [('1',source.astype(np.bool_)),('L',source),('L',source*255),('P',source)]:
        image=Image.fromarray(values)
        if mode=='P':
            image=image.convert('P');palette=[0]*768;palette[3:6]=[255]*3;image.putpalette(palette)
        buffer=BytesIO();image.save(buffer,format='PNG')
        field,facts=decode_bytes(buffer.getvalue(),source.shape)
        require(np.array_equal(field,source!=0) and facts['mode']==mode,'Synthetic binary encoding differs')
        cases+=1
    image=Image.fromarray(source*2);buffer=BytesIO();image.save(buffer,format='PNG')
    try:decode_bytes(buffer.getvalue(),source.shape)
    except ValueError:pass
    else:raise ValueError('Nonbinary PNG accepted')
    image=Image.fromarray(source).convert('P');image.putpalette([255]*768);buffer=BytesIO();image.save(buffer,format='PNG')
    try:decode_bytes(buffer.getvalue(),source.shape)
    except ValueError:pass
    else:raise ValueError('Inverted/ambiguous palette accepted')
    body=bytearray(buffer.getvalue());body[-5]^=1
    try:structure(bytes(body),source.shape)
    except ValueError:pass
    else:raise ValueError('Bad CRC accepted')
    return dict(status='PASS',binary_encoding_cases=cases,rejected_nonbinary_cases=1,
                rejected_palette_cases=1,rejected_CRC_cases=1,actual_real_image_decodes=0)
