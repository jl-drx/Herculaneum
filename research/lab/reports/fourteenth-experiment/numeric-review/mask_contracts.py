"""Canonical development and diagnostic masks rebuilt from original inputs."""
import numpy as np
from numeric_kernels import require

DEVELOPMENT_IDS = ['pherc1667-w029','pherc0814-46527-rank01','pherc0814-46527-rank07']
NEGATIVE_COUNTS = dict(zip(DEVELOPMENT_IDS,[9378,40400,14644]))
SOURCE_BBOXES = dict(zip(DEVELOPMENT_IDS,[[6912,3200,256,256],[256,1664,384,384],[640,1536,384,384]]))
DIAGNOSTIC_IDS = ['pherc0814-46527-fresh-rank01','pherc0814-46527-fresh-rank14','frag4-context1','frag4-context2']
DIAGNOSTIC_COUNTS = dict(zip(DIAGNOSTIC_IDS,[(49033,19545,29488),(21328,4767,16561),
                                           (91444,3818,87626),(77851,15252,62599)]))


def binary(value, shape, label):
    require(value.shape == shape and value.dtype in (np.dtype('uint8'),np.dtype('bool')) and
            np.isin(value,[0,1]).all(), 'Binary shape/encoding differs: ' + label)


def calibration_mask(labels, validation, observed, margin=64):
    require(labels.ndim == 2 and min(labels.shape) > 2*margin and margin >= 0, 'Invalid calibration canvas/margin')
    binary(labels,labels.shape,'labels'); binary(validation,labels.shape,'validation')
    require(observed.shape == labels.shape and observed.dtype == np.bool_, 'Observed mask contract differs')
    interior = np.zeros(labels.shape,np.bool_)
    if margin: interior[margin:-margin,margin:-margin] = True
    else: interior[:] = True
    return (validation > 0) & observed & interior


def development(review, case):
    require(case['id'] in DEVELOPMENT_IDS and case['evaluation_participation_role']=='public_validation' and
            case['data']['bbox_source_xywh']==SOURCE_BBOXES[case['id']], 'Literal development identity/role/bbox differs')
    shape = (256,256) if case['id']==DEVELOPMENT_IDS[0] else (384,384)
    source = case['inputs']
    labels = review.array(source['inklabels'],shape=shape)
    validation = review.array(source['validation_mask'],shape=shape)
    observed = review.array(source['acquisition_observed_mask'],shape=shape,dtype=np.bool_)
    participation = review.array(source['evaluation_participation_mask'],shape=shape)
    binary(participation,shape,'participation')
    require(observed.all() and np.array_equal(participation>0,validation>0), 'Original participation/observation changed')
    primary = calibration_mask(labels,validation,observed)
    require(int((primary & (labels==0)).sum()) == NEGATIVE_COUNTS[case['id']],
            'Original negative count changed; no support shrinking allowed')
    surface = review.array(source['surface'],shape=(21,*shape),dtype=np.uint8)
    return labels.astype(np.bool_),primary,surface


def diagnostic(review, case):
    require(case['id'] in DIAGNOSTIC_IDS, 'Unexpected diagnostic context')
    legacy = case['legacy_case']
    shape = (384,384)
    if case['family']=='pooled21':
        require(case['id'] in DIAGNOSTIC_IDS[:2] and case['physical_group']=='PHerc0814' and
                case['layer_start']==2 and legacy['evaluation_participation_role']=='supplied_supervision',
                'Pooled diagnostic role/depth differs')
        source = legacy['inputs']
        labels = review.array(source['inklabels'],shape=shape)
        supervision = review.array(source['supervision_mask'],shape=shape)
        participation = review.array(source['evaluation_participation_mask'],shape=shape)
        observed = review.array(source['acquisition_observed_mask'],shape=shape,dtype=np.bool_)
        binary(supervision,shape,'supervision'); binary(participation,shape,'participation')
        require(observed.all() and np.array_equal(participation>0,supervision>0), 'Pooled diagnostic participation changed')
        interior = np.zeros(shape,np.bool_); interior[64:-64,64:-64] = True
        primary = (supervision>0) & observed & interior
        review.same(case['primary_mask'],legacy['data']['arrays']['primary_annotation_mask'],'Pooled original mask')
    else:
        require(case['family']=='frag4_float17' and case['id'] in DIAGNOSTIC_IDS[2:] and
                case['physical_group']=='PHercParis1Fr39' and case['layer_start']==0, 'Frag4 identity/depth differs')
        source = legacy['arrays']
        labels = review.array(source['labels'],shape=shape)
        known = review.array(source['known_mask'],shape=shape,dtype=np.bool_)
        surface = review.array(source['surface_mask'],shape=shape,dtype=np.bool_)
        require(np.all(~known | surface), 'Known Frag4 support escapes surface')
        interior = np.zeros(shape,np.bool_); interior[32:-32,32:-32] = True
        primary = known & interior
        review.same(case['primary_mask'],source['primary_mask'],'Frag4 original mask')
    binary(labels,shape,'diagnostic labels')
    saved = review.array(case['primary_mask'],shape=shape,dtype=np.bool_)
    require(np.array_equal(saved,primary),'Canonical diagnostic mask differs')
    labels = labels.astype(np.bool_)
    counts = (int(primary.sum()),int((primary&labels).sum()),int((primary&~labels).sum()))
    require(counts==DIAGNOSTIC_COUNTS[case['id']] and case['canonical_counts']==
            dict(fixed_primary_count=counts[0],positive=counts[1],negative=counts[2]),'Canonical diagnostic counts differ')
    return labels,primary


def fixtures():
    labels = np.zeros((6,6),np.uint8); labels[2:4,2:4] = 1
    validation = np.ones((6,6),np.uint8)
    observed = np.ones((6,6),np.bool_); observed[2,2] = False
    expected = np.zeros((6,6),np.bool_); expected[1:5,1:5] = True; expected[2,2] = False
    require(np.array_equal(calibration_mask(labels,validation,observed,1),expected),
            'Validation/acquisition/interior mask fixture failed')
    require(int((expected & (labels==0)).sum())==12 and int((expected & (labels==1)).sum())==3,
            'Positive/negative support fixture failed')
    zero_occupancy = np.zeros((6,6),np.bool_)
    require(np.array_equal(calibration_mask(labels,validation,observed,1),expected) and
            not (expected&zero_occupancy).any(), 'Occupancy is incorrectly required for calibration participation')
    return dict(status='PASS',canonical_mask_cases=1,canonical_class_count_cases=1,
                occupancy_exclusion_cases=1,actual_retained_array_reads=0)
