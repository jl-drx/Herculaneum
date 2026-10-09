"""Independent bounded integral support; no SciPy or author selector imports."""
import numpy as np


def require(condition,message):
    if not condition: raise ValueError(message)


def square_counts(field,start,stop,radius=16):
    """Centered zero-padded counts on a short full-width output-row band.

    Prefix sums cover at most (output_rows+2r)*(width+2r), not the native canvas.
    Signed int32 is safe under the explicitly checked band-size bound.
    """
    require(field.ndim==2 and field.dtype==np.bool_ and 0<=start<stop<=field.shape[0], 'Binary native band required')
    h,w=field.shape; size=2*radius+1
    padded=np.zeros((stop-start+2*radius,w+2*radius),np.uint8)
    y0,y1=max(0,start-radius),min(h,stop+radius)
    padded[y0-start+radius:y1-start+radius,radius:radius+w]=field[y0:y1]
    require(padded.size<np.iinfo(np.int32).max,'Band prefix exceeds int32 bound')
    integral=np.zeros((padded.shape[0]+1,padded.shape[1]+1),np.int32)
    integral[1:,1:]=np.cumsum(np.cumsum(padded,axis=1,dtype=np.int32),axis=0,dtype=np.int32)
    return (integral[size:,size:]-integral[:-size,size:]-integral[size:,:-size]+integral[:-size,:-size])


def native_known(labels,surface,band_rows=256):
    """Return coarse sampled known support plus exact full-native known count."""
    require(labels.shape==surface.shape and labels.ndim==2 and labels.dtype==surface.dtype==np.bool_, 'Aligned binary fields required')
    require(type(band_rows) is int and 1<=band_rows<=1024,'Bounded band size required')
    coarse=np.zeros(labels[1::3,1::3].shape,np.bool_)
    native_count=0
    for start in range(0,labels.shape[0],band_rows):
        stop=min(start+band_rows,labels.shape[0])
        surface_count=square_counts(surface,start,stop)
        fully_inside=surface_count==1089
        del surface_count
        label_count=square_counts(labels,start,stop)
        known=fully_inside & ((label_count==0)|(label_count==1089))
        del label_count,fully_inside
        native_count+=int(np.count_nonzero(known))
        first=start+((1-start)%3)
        if first<stop:
            sampled=known[first-start:stop-start:3,1::3]
            row=(first-1)//3
            coarse[row:row+sampled.shape[0]]=sampled
        del known
    return coarse,native_count


def candidates(labels,surface,coarse_known):
    """All complete native1152 contexts, source-aligned and row-major."""
    require(labels.dtype==surface.dtype==np.bool_ and labels.shape==surface.shape and
            coarse_known.dtype==np.bool_ and coarse_known.shape==labels[1::3,1::3].shape,'Field contracts changed')
    rows=[]
    for y in range(0,labels.shape[0]-1152+1,1152):
        for x in range(0,labels.shape[1]-1152+1,1152):
            yy,xx=y//3,x//3
            coarse_labels=labels[y+1:y+1152:3,x+1:x+1152:3]
            known=coarse_known[yy:yy+384,xx:xx+384]
            primary=known[32:352,32:352]
            positive=int(np.count_nonzero(primary&coarse_labels[32:352,32:352]))
            negative=int(np.count_nonzero(primary&~coarse_labels[32:352,32:352]))
            inside=int(np.count_nonzero(surface[y:y+1152,x:x+1152]))
            rows.append(dict(x=x,y=y,inside_count=inside,primary_count=positive+negative,
                positive=positive,negative=negative,
                coarse_surface_count=int(np.count_nonzero(surface[y+1:y+1152:3,x+1:x+1152:3])),
                coarse_known_count=int(np.count_nonzero(known)),
                eligible=10*inside>=9*1152*1152 and positive>=512 and negative>=8192))
    return rows


def choose(rows):
    eligible=[r for r in rows if r['eligible']]
    require(eligible,'No eligible first context')
    first=eligible[0]
    separated=[r for r in eligible if r['y']>=first['y']+2304]
    require(separated,'No eligible separated second context')
    return first,separated[-1]


def context_arrays(labels,surface,coarse_known,selected):
    x,y=selected['x'],selected['y'];yy,xx=y//3,x//3
    label=np.ascontiguousarray(labels[y+1:y+1152:3,x+1:x+1152:3],dtype=np.uint8)
    mask=np.ascontiguousarray(surface[y+1:y+1152:3,x+1:x+1152:3],dtype=np.bool_)
    known=np.ascontiguousarray(coarse_known[yy:yy+384,xx:xx+384],dtype=np.bool_)
    primary=np.zeros((384,384),np.bool_);primary[32:352,32:352]=known[32:352,32:352]
    require(np.all(~known|mask),'Known support escapes surface')
    return dict(labels=label,surface_mask=mask,known_mask=known,primary_mask=primary)


def fixtures():
    # Brute force is used only on tiny synthetic rasters, independently of prefixes.
    rng=np.random.default_rng(150)
    a=rng.integers(0,2,(43,49),dtype=np.uint8).astype(np.bool_)
    expected=np.zeros(a.shape,np.int32)
    for y in range(a.shape[0]):
        for x in range(a.shape[1]):expected[y,x]=int(a[max(0,y-16):y+17,max(0,x-16):x+17].sum())
    band_cases=0
    for start,stop in [(0,1),(0,17),(7,36),(42,43)]:
        require(np.array_equal(square_counts(a,start,stop),expected[start:stop]),'Integral/brute-force square count differs')
        band_cases+=1
    labels=np.zeros((67,71),np.bool_);surface=np.ones_like(labels);labels[20:55,23:58]=True;surface[33,44]=False
    full=np.zeros_like(labels)
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            if y<16 or x<16 or y+16>=labels.shape[0] or x+16>=labels.shape[1]:continue
            m=surface[y-16:y+17,x-16:x+17];v=labels[y-16:y+17,x-16:x+17]
            full[y,x]=bool(m.all() and (v.all() or not v.any()))
    for band in [1,16,17,32,256]:
        coarse,count=native_known(labels,surface,band)
        require(np.array_equal(coarse,full[1::3,1::3]) and count==int(full.sum()),'Known/coarse/band-boundary fixture differs')
    rows=[dict(x=0,y=y,eligible=True) for y in [0,1152,2304,3456]]
    first,last=choose(rows);require(first['y']==0 and last['y']==3456,'First/last separated choice differs')
    rows[-1]['eligible']=False
    require(choose(rows)[1]['y']==2304,'Exact2304 spacing boundary differs')
    for insufficient in [[],[dict(x=0,y=0,eligible=True)],[dict(x=0,y=0,eligible=True),dict(x=0,y=1152,eligible=True)]]:
        try:choose(insufficient)
        except ValueError:pass
        else:raise ValueError('Incomplete panel fixture accepted')
    n=1152*1152;ceil90=(9*n+9)//10
    require(10*ceil90>=9*n and 10*(ceil90-1)<9*n,'Exact90-percent integer boundary differs')
    # Separate candidate/extraction fixture supplies coarse known support directly;
    # the native-known operation is already checked against brute force above.
    labels=np.zeros((3456,2304),np.bool_);surface=np.ones_like(labels)
    for x in (0,1152):labels[1::3,x+97:x+289:3]=True
    surface[:6,:6]=False
    known=np.ones((1152,768),np.bool_);known[:2,:2]=False;known[40,50]=False
    rows=candidates(labels,surface,known)
    require([(r['y'],r['x']) for r in rows]==[(0,0),(0,1152),(1152,0),(1152,1152),(2304,0),(2304,1152)],
            'Complete row-major candidate grid differs')
    for index,row in enumerate(rows):
        special=index==0
        require(row['inside_count']==n-(36 if special else 0) and row['primary_count']==102400-(1 if special else 0) and
                row['positive']==20480-(1 if special else 0) and row['negative']==81920 and row['eligible'] is True and
                row['coarse_surface_count']==147456-(4 if special else 0) and
                row['coarse_known_count']==147456-(5 if special else 0),'Literal candidate count fixture differs')
    arrays=context_arrays(labels,surface,known,rows[0])
    require(arrays['labels'].dtype==np.uint8 and int(arrays['labels'].sum())==24576 and
            int(arrays['surface_mask'].sum())==147452 and int(arrays['known_mask'].sum())==147451 and
            int(arrays['primary_mask'].sum())==102399 and not arrays['primary_mask'][31].any() and
            not arrays['primary_mask'][352].any(),'Literal center/border/canonical array fixture differs')
    require(choose(rows)==(rows[0],rows[-1]),'Literal row-major first/last extraction fixture differs')
    return dict(status='PASS',square_sum_cases=band_cases,known_sampling_band_cases=5,
                first_last_spacing_cases=2,incomplete_panel_cases=3,exact90_boundary_cases=1,
                literal_candidate_grid_cases=1,literal_context_extraction_cases=1,actual_real_array_reads=0)
