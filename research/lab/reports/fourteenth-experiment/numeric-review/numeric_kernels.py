"""Independent threshold/ranking/confusion/recovery math; NumPy + stdlib only."""
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
    require(set(regional) == {'pherc1667-w029','pherc0814-46527-rank01','pherc0814-46527-rank07'},
            'Exactly the three original development cases required')
    return {'primary':max(v['threshold'] for v in regional.values()),
            'secondary':regional['pherc1667-w029']['threshold']}


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


def exact_sensitivity(row):
    c = row['confusion']
    require(c['tp']+c['fn'] > 0, 'Positive support required for sensitivity screen')
    return Fraction(c['tp'], c['tp']+c['fn'])


def fraction_record(value):
    return {'numerator':value.numerator,'denominator':value.denominator}


def published_fraction(value):
    return dict(fraction_record(value),value=float(value))


def published_screen(current, baseline):
    checked = recovery_screen(current,baseline)
    specificity, contexts = {}, {}
    for name,row in current.items():
        c = row['confusion']
        delta = exact_sensitivity(row)-exact_sensitivity(baseline[name])
        specificity[name] = dict(false_positives=c['fp'],negative_count=c['fp']+c['tn'],
                                passed=checked['context_checks'][name]['specificity_at_least_95'])
        contexts[name] = dict(delta=published_fraction(delta),minimum=published_fraction(Fraction(-1,50)),
                             passed=checked['context_checks'][name]['sensitivity_loss_at_most_002'])
    current_mean = sum((exact_sensitivity(r) for r in current.values()),Fraction())/2
    baseline_mean = sum((exact_sensitivity(r) for r in baseline.values()),Fraction())/2
    return dict(specificity_by_context=specificity,per_context_sensitivity_checks=contexts,
        mean_sensitivity_comparison=dict(candidate=published_fraction(current_mean),baseline=published_fraction(baseline_mean),
                                        delta=published_fraction(current_mean-baseline_mean),
                                        passed=checked['mean_sensitivity_at_least_baseline']),
        passed=checked['passed'],automatic_promotion=False)


def recovery_screen(current, baseline):
    require(set(current) == set(baseline) and len(current) == 2, 'Two fixed contexts per specimen required')
    context_checks = {}
    for case in current:
        c = current[case]['confusion']
        b = baseline[case]['confusion']
        require(c['tp']+c['fn'] == b['tp']+b['fn'] and c['fp']+c['tn'] == b['fp']+b['tn'],
                'Canonical class supports differ across models')
        context_checks[case] = dict(specificity_at_least_95=20*c['fp'] <= c['fp']+c['tn'],
            sensitivity_loss_at_most_002=exact_sensitivity(current[case])-exact_sensitivity(baseline[case]) >= -Fraction(1,50))
    current_mean = sum((exact_sensitivity(r) for r in current.values()), Fraction())/2
    baseline_mean = sum((exact_sensitivity(r) for r in baseline.values()), Fraction())/2
    mean_check = current_mean >= baseline_mean
    return dict(context_checks=context_checks,mean_sensitivity_at_least_baseline=mean_check,
                current_mean_sensitivity=fraction_record(current_mean),baseline_mean_sensitivity=fraction_record(baseline_mean),
                passed=mean_check and all(all(c.values()) for c in context_checks.values()))


def fixtures():
    cutoff_cases = [
        (np.array([.9]*2+[.2]*18,np.float32),0,2),
        (np.array([.99]*2+[.5]*10+[.1]*28,np.float32),2,10),
        (np.ones(20,np.float32),0,20),
        (np.array([0],np.float32),0,1),
        (np.arange(19,dtype=np.float32)/20,0,1),
        (np.full(40,.3,np.float32),0,40)]
    for scores, expected_fp, expected_ties in cutoff_cases:
        result = select_negatives(scores)
        require(result['development_false_positives']==expected_fp and result['boundary_tie_count']==expected_ties,
                'Crossing-tie fixture failed')
        float32_bits(result['threshold'])
    require(float32_bits(select_negatives(np.ones(20,np.float32))['threshold']) == 0x3f800001,
            'Finite above-one cutoff bit fixture failed')
    require(float32_bits(select_negatives(np.array([0],np.float32))['threshold']) == 1,
            'Smallest positive subnormal cutoff bit fixture failed')
    for invalid in (np.array([],np.float32), np.array([.5],np.float64),
                    np.array([np.nan],np.float32), np.array([np.inf],np.float32),
                    np.array([-1],np.float32), np.array([1.1],np.float32)):
        try: select_negatives(invalid)
        except ValueError: pass
        else: raise ValueError('Invalid selector input fixture was accepted')
    names = ['pherc1667-w029','pherc0814-46527-rank01','pherc0814-46527-rank07']
    locks = lock_values({n: {'threshold': v} for n,v in zip(names,[.2,.8,.5])})
    require(locks==dict(primary=.8,secondary=.2),'Primary maximum/w029-secondary fixture failed')
    y = np.array([0,1,0,1],bool)
    s = np.array([.2,.4,.4,.8],np.float32)
    rank = ranking(y,s)
    require(rank['auc']==.875 and abs(rank['ap']-5/6)<1e-12,'Tied rank fixture failed')
    require(ranking(y,np.ones(4,np.float32))['auc']==.5 and ranking(y,np.ones(4,np.float32))['ap']==.5,
            'All-tied ranking fixture failed')
    cutoff = float(np.nextafter(np.float32(.4),np.float32(np.inf),dtype=np.float32))
    result = operating(y,s,cutoff)
    require(result['confusion']==dict(tp=1,fp=0,tn=2,fn=1),'Float64 cutoff comparison fixture failed')
    require(operating(y,s,2.)['precision'] is None,'Zero-prediction precision fixture failed')
    def row(tp, positive=100, fp=5, negative=100):
        return {'confusion':dict(tp=tp,fn=positive-tp,fp=fp,tn=negative-fp)}
    baseline = {'a':row(50),'b':row(50)}
    require(recovery_screen({'a':row(48),'b':row(52)},baseline)['passed'],
            'Exact 0.02 loss/equal-mean/specificity boundary fixture failed')
    require(not recovery_screen({'a':row(47),'b':row(53)},baseline)['passed'],
            'Individual sensitivity-loss failure fixture failed')
    require(not recovery_screen({'a':row(49),'b':row(50)},baseline)['passed'],
            'Equal-context mean sensitivity failure fixture failed')
    require(not recovery_screen({'a':row(50,fp=6),'b':row(50)},baseline)['passed'],
            'Exact specificity failure fixture failed')
    boundary_baseline = {'a':row(10,positive=50),'b':row(10,positive=50)}
    require(9/50-1/5 < -.02 and recovery_screen({'a':row(9,positive=50),'b':row(11,positive=50)},
            boundary_baseline)['passed'], 'Nonbinary 0.02 exact rational boundary fixture failed')
    for n, expected in [(19,False),(20,True)]:
        bg = {'a':row(50,fp=0,negative=n),'b':row(50,fp=0,negative=n)}
        cg = {'a':row(50,fp=1,negative=n),'b':row(50,fp=1,negative=n)}
        require(recovery_screen(cg,bg)['passed'] == expected, '19/20 exact specificity integer fixture failed')
    macro_bg = {'a':row(50,positive=100),'b':row(1,positive=2)}
    macro_cg = {'a':row(49,positive=100),'b':row(2,positive=2)}
    macro = recovery_screen(macro_cg,macro_bg)
    require(macro['passed'] and macro['current_mean_sensitivity'] == fraction_record(Fraction(149,200)) and
            Fraction(49+2,100+2) == Fraction(50+1,100+2), 'Equal-context versus pooled sensitivity fixture failed')
    return dict(status='PASS',crossing_tie_cases=6,exact_cutoff_bit_cases=2,invalid_selector_cases=6,
                primary_secondary_lock_cases=1,tie_ranking_cases=2,operating_cases=2,exact_recovery_screen_cases=8,
                actual_retained_array_reads=0)
