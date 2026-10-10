"""Independent sixteenth cutoff/ranking/confusion arithmetic; no author imports."""
from fractions import Fraction
import math
import numpy as np


def require(value, message):
    if not value:
        raise ValueError(message)


def float32_bits(value):
    cast = np.float32(value)
    require(math.isfinite(value) and float(cast) == float(value), 'Exact finite float32 value required')
    return int(cast.view(np.uint32))


def select_negatives(negative_scores):
    """Find the crossing group by unique-score multiplicity, not sorted index m."""
    scores = np.asarray(negative_scores)
    require(scores.ndim == 1 and scores.size > 0 and scores.dtype == np.float32,
            'Nonempty one-dimensional float32 negative scores required')
    require(np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all(), 'Negative scores invalid')
    maximum = scores.size // 20
    levels, multiplicity = np.unique(scores, return_counts=True)
    cumulative = 0
    for level, count in zip(levels[::-1], multiplicity[::-1]):
        cumulative += int(count)
        if cumulative > maximum:
            boundary = np.float32(level)
            ties = int(count)
            break
    threshold32 = np.nextafter(boundary, np.float32(np.inf), dtype=np.float32)
    threshold = float(threshold32)
    false_positive = int((scores.astype(np.float64) >= threshold).sum())
    require(false_positive <= maximum and 20 * false_positive <= scores.size, 'Tie exclusion fails FP bound')
    return dict(target_specificity_numerator=95,target_specificity_denominator=100,
        negative_count=int(scores.size),maximum_false_positives=int(maximum),
        excluded_negative_boundary=float(boundary),threshold=threshold,
        threshold_construction='nextafter(float32 kth-descending negative boundary, float32 +inf)',
        threshold_operator='float64(score) >= threshold',boundary_tie_count=ties,
        development_false_positives=false_positive,development_specificity=(scores.size-false_positive)/scores.size,
        scope='Chosen only from development negatives; never refitted on reserved scores or labels.')


def lock_values(regional):
    require(set(regional)=={'frag3-context1','frag3-context2'},'Two frozen Frag3 calibration contexts required')
    return {'primary':max(row['threshold'] for row in regional.values())}


def ranking(labels, scores):
    """Grouped pair concordance for AUC; group precision mass for AP."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    require(labels.ndim == 1 and scores.shape == labels.shape and labels.size > 0 and
            np.isin(labels,[0,1]).all() and np.isfinite(scores).all(), 'Binary aligned finite rank inputs required')
    truth = labels.astype(np.bool_)
    positive = int(truth.sum()); negative = int(labels.size-positive)
    require(positive > 0 and negative > 0 and labels.size <= 384*384, 'Both bounded classes required')
    values, reverse_index = np.unique(scores.astype(np.float64), return_inverse=True)
    counts = np.bincount(reverse_index, minlength=values.size).astype(np.int64)
    group_pos = np.bincount(reverse_index, weights=truth.astype(np.int64), minlength=values.size).astype(np.int64)
    group_neg = counts-group_pos
    negative_below = np.cumsum(group_neg)-group_neg
    concordance_twice = int(np.dot(group_pos, 2*negative_below+group_neg))
    auc = concordance_twice/(2*positive*negative)
    descending_pos = group_pos[::-1]
    cumulative_positive = np.cumsum(descending_pos)
    cumulative_count = np.cumsum(counts[::-1])
    ap = math.fsum(int(p)*int(tp)/int(n) for p,tp,n in zip(descending_pos,cumulative_positive,cumulative_count))/positive
    return {'auc':auc,'ap':ap,'positive':positive,'negative':negative,'pixels':int(labels.size)}


def operating(labels, scores, threshold):
    labels, scores = np.asarray(labels), np.asarray(scores)
    require(labels.ndim == 1 and labels.shape == scores.shape and np.isin(labels,[0,1]).all() and
            np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all() and math.isfinite(threshold),
            'Valid aligned labels/scores and finite cutoff required')
    truth = labels.astype(np.bool_)
    classified = scores.astype(np.float64) >= float(threshold)
    positive, negative = int(truth.sum()), int((~truth).sum())
    tp, fp = int((truth & classified).sum()), int((~truth & classified).sum())
    tn, fn = negative-fp, positive-tp
    return dict(threshold=float(threshold),threshold_operator='float64(score) >= threshold',
        pixel_count=int(labels.size),supplied_positive_pixels=positive,supplied_negative_pixels=negative,
        confusion=dict(tp=tp,fp=fp,tn=tn,fn=fn),sensitivity=tp/positive if positive else None,
        specificity=tn/negative if negative else None,precision=tp/(tp+fp) if tp+fp else None,
        balanced_accuracy=(tp/positive+tn/negative)/2 if positive and negative else None)


RULES=('retained_fourteenth','frag3_only')
ENDPOINTS=('sensitivity','specificity','precision','f1','predicted_positive_fraction','balanced_accuracy')


def derived_operating(labels,scores,threshold):
    row=operating(labels,scores,threshold);c=row['confusion']
    denominator=2*c['tp']+c['fp']+c['fn']
    row.update(f1=2*c['tp']/denominator if denominator else None,
        predicted_positive_fraction=(c['tp']+c['fp'])/row['pixel_count'],
        zero_predicted_positives=c['tp']+c['fp']==0,
        specificity_at_least_95=20*c['fp']<=c['fp']+c['tn'])
    return row


def mean(values):
    require(len(values)>0,'Nonempty fixed macro components required')
    return math.fsum(values)/len(values) if all(v is not None for v in values) else None


def macro(rows):
    return dict(**{key:mean([r[key] for r in rows]) for key in ('auc','ap','supplied_positive_prevalence')},
        operating={rule:{key:mean([r['operating'][rule][key] for r in rows]) for key in ENDPOINTS} for rule in RULES})


def delta(candidate,baseline):
    return dict(auc=candidate['auc']-baseline['auc'],ap=candidate['ap']-baseline['ap'],
        operating={rule:{key:candidate['operating'][rule][key]-baseline['operating'][rule][key]
            if candidate['operating'][rule][key] is not None and baseline['operating'][rule][key] is not None else None
            for key in ENDPOINTS} for rule in RULES})


def transfer_screen(rows):
    require(len(rows)==4,'Four fixed Paris2 evaluation contexts required')
    checked={name:20*row['confusion']['fp']<=row['confusion']['fp']+row['confusion']['tn']
             for name,row in rows.items()}
    return dict(per_context=checked,all_four=all(checked.values()))


def fixtures():
    cases=[(np.ones(20,np.float32),0,20),
           (np.array([.9]*2+[.2]*18,np.float32),0,2),
           (np.array([.99]*2+[.5]*10+[.1]*28,np.float32),2,10),
           (np.array([0],np.float32),0,1)]
    for values,fp,ties in cases:
        selected=select_negatives(values)
        require(selected['development_false_positives']==fp and selected['boundary_tie_count']==ties,
                'Conservative tied cutoff fixture differs')
    require(float32_bits(select_negatives(cases[0][0])['threshold'])==0x3f800001 and
            float32_bits(select_negatives(cases[-1][0])['threshold'])==1,'Exact cutoff bit fixture differs')
    require(lock_values({'frag3-context1':dict(threshold=.2),'frag3-context2':dict(threshold=.8)})==dict(primary=.8),
            'Two-context max lock differs')
    y=np.array([0,1,0,1],bool);s=np.array([.2,.4,.4,.8],np.float32)
    value=ranking(y,s);require(value['auc']==.875 and abs(value['ap']-5/6)<1e-12,'Tied ranking fixture differs')
    tied=ranking(y,np.ones(4,np.float32));require(tied['auc']==tied['ap']==.5,'All-tied ranking differs')
    threshold=float(np.nextafter(np.float32(.4),np.float32(np.inf),dtype=np.float32))
    row=derived_operating(y,s,threshold)
    require(row['confusion']==dict(tp=1,fp=0,tn=2,fn=1) and row['f1']==2/3 and
            row['predicted_positive_fraction']==.25 and row['precision']==1.,'Extended confusion fixture differs')
    zero=derived_operating(y,s,2.)
    require(zero['precision'] is None and zero['f1']==0 and zero['zero_predicted_positives'] and
            zero['predicted_positive_fraction']==0 and zero['specificity_at_least_95'],
            'All-negative operating fixture differs')
    def m(auc,ap,operating_row):
        return dict(auc=auc,ap=ap,supplied_positive_prevalence=.5,
                    operating={rule:operating_row for rule in RULES})
    a,b=m(.2,.1,zero),m(.8,.7,row)
    result=macro([a,b])
    require(result['auc']==.5 and result['operating']['frag3_only']['precision'] is None and
            result['operating']['frag3_only']['f1']==1/3,'None-preserving equal-context macro differs')
    def r(fp,n):return dict(confusion=dict(tp=1,fn=1,fp=fp,tn=n-fp))
    pass_rows={f'case{i}':r(1,20) for i in range(4)}
    require(transfer_screen(pass_rows)['all_four'],'Inclusive95 specificity fixture differs')
    pass_rows['case3']=r(1,19)
    require(not transfer_screen(pass_rows)['all_four'],'All-four exact specificity failure differs')
    difference=delta(b,a)
    require(difference['operating']['frag3_only']['precision'] is None and
            abs(difference['auc']-.6)<1e-12,'None-preserving candidate difference differs')
    for invalid in (np.array([],np.float32),np.array([np.nan],np.float32),np.array([.5],np.float64)):
        try:select_negatives(invalid)
        except ValueError:pass
        else:raise ValueError('Invalid cutoff fixture accepted')
    return dict(status='PASS',cutoff_cases=4,exact_cutoff_bit_cases=2,max_lock_cases=1,ranking_cases=2,
        operating_cases=2,macro_cases=1,integer_transfer_screen_cases=2,difference_cases=1,
        rejected_selector_cases=3,actual_saved_predictions_read=0,network_calls=0,model_calls=0)
