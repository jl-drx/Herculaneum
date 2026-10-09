"""Independent loss and augmentation mathematics; no author or model imports."""
import math
import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bit_equal(left, right):
    require(left.dtype == right.dtype == np.float32 and left.shape == right.shape,
            'Bitwise comparisons require aligned float32 arrays')
    return np.array_equal(left.view(np.uint32), right.view(np.uint32))


def average(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.size > 0 and np.isfinite(values).all(), 'Finite nonempty mean required')
    return math.fsum(float(value) for value in values.ravel()) / values.size


def d4(array, code):
    require(type(code) is int and 0 <= code < 8 and array.shape[-2] == array.shape[-1],
            'D4 needs a square spatial raster and an integer code in 0..7')
    size = array.shape[-1]
    row, column = np.indices((size, size))
    transformed_column = column if code < 4 else size - 1 - column
    source = ((row, transformed_column), (transformed_column, size - 1 - row),
              (size - 1 - row, size - 1 - transformed_column),
              (size - 1 - transformed_column, row))[code % 4]
    return np.ascontiguousarray(array[..., source[0], source[1]])


def support_losses(student, teacher, hard_target):
    student, teacher, target = [np.asarray(value, dtype=np.float64)
                                for value in (student, teacher, hard_target)]
    require(student.shape == teacher.shape == target.shape and student.size > 0,
            'Loss support must be aligned and nonempty')
    require(np.isfinite(student).all() and np.isfinite(teacher).all(), 'Nonfinite logits')
    require(np.isin(target, [0, 1]).all(), 'Hard support targets must be binary')
    log_p = -np.logaddexp(0., -student)
    log_not_p = -np.logaddexp(0., student)
    log_q = -np.logaddexp(0., -teacher)
    log_not_q = -np.logaddexp(0., teacher)
    q, not_q = np.exp(log_q), np.exp(log_not_q)
    soft = .25 + .5 * target
    return dict(smoothed_bce=average(-soft * log_p - (1. - soft) * log_not_p),
                hard_bce=average(-target * log_p - (1. - target) * log_not_p),
                kl=average(q * (log_q - log_p) + not_q * (log_not_q - log_not_p)),
                probability_drift=average(np.abs(np.exp(log_p) - q)),
                count=int(target.size), positive=int((target == 1).sum()),
                negative=int((target == 0).sum()))


def manual_record(student, teacher, mask, hard_target):
    require(mask.dtype == np.bool_ and mask.shape == hard_target.shape == student.shape == teacher.shape,
            'Manual participant shape/dtype mismatch')
    terms = support_losses(student[mask], teacher[mask], hard_target[mask])
    return dict(base=terms['smoothed_bce'], full=terms['smoothed_bce'] + terms['kl'],
                kl=terms['kl'], hard_bce=terms['hard_bce'],
                probability_drift_manual=terms['probability_drift'])


def loss_record(student, teacher, batch, pseudo_coefficient):
    require(student.shape == teacher.shape == (2, 1, 128, 128), 'Logit shape differs')
    require(pseudo_coefficient in (0., .25), 'Undeclared pseudo coefficient')
    terms = {}
    for name in ('manual', 'pseudo'):
        mask, target = batch[name + '_mask'], batch[name + '_hard_target']
        require(mask.dtype == np.bool_ and mask.shape == target.shape == student.shape,
                'Mask and target shapes/dtypes differ')
        terms[name] = support_losses(student[mask], teacher[mask], target[mask])
    require(not (batch['manual_mask'] & batch['pseudo_mask']).any(), 'Support overlap')
    manual, pseudo = terms['manual'], terms['pseudo']
    balanced = []
    for region in range(2):
        mask, target = batch['manual_mask'][region], batch['manual_hard_target'][region]
        for label in (0, 1):
            selected = mask & (target == label)
            balanced.append(support_losses(student[region][selected], teacher[region][selected],
                                           target[selected])['smoothed_bce'])
    base = manual['smoothed_bce'] + pseudo_coefficient * pseudo['smoothed_bce']
    return dict(loss_U=manual['smoothed_bce'], loss_P=pseudo['smoothed_bce'],
                loss_R=average(balanced), loss_D_M=manual['kl'], loss_D_P=pseudo['kl'],
                base=base, full=base + manual['kl'] + pseudo_coefficient * pseudo['kl'],
                hard_base=manual['hard_bce'] + pseudo_coefficient * pseudo['hard_bce'],
                probability_drift_manual=manual['probability_drift'],
                probability_drift_pseudo=pseudo['probability_drift'])


def equal_batch_means(rows):
    require(len(rows) == 32 and all(set(row) == set(rows[0]) for row in rows),
            'Exactly 32 identically structured batch records required')
    return {key: math.fsum(float(row[key]) for row in rows) / 32 for key in rows[0]}


def rank_metrics(labels, scores):
    """Average pairwise AUROC and grouped-threshold step-function AP."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    require(labels.ndim == scores.ndim == 1 and labels.shape == scores.shape,
            'Rank metrics require aligned flat arrays')
    require(np.isin(labels, [0, 1]).all() and np.isfinite(scores).all(),
            'Rank metrics require binary labels and finite scores')
    count, positives = labels.size, int(labels.sum())
    negatives = count - positives
    if count == 0:
        return dict(count=0, positive=0, negative=0, auc=None, ap=None)
    order = np.argsort(scores, kind='stable')
    sorted_scores, sorted_labels = scores[order], labels[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    ends = np.r_[starts[1:], count]
    group_positive = np.add.reduceat(sorted_labels.astype(np.int64), starts)
    group_negative = ends - starts - group_positive
    negative_before = np.cumsum(group_negative) - group_negative
    favorable_pairs = math.fsum(float(p) * (float(n) + .5 * float(tie))
                                for p, n, tie in zip(group_positive, negative_before, group_negative))
    auc = favorable_pairs / (positives * negatives) if positives and negatives else None
    cumulative_positive = np.cumsum(group_positive[::-1])
    cumulative_total = np.cumsum((ends - starts)[::-1])
    ap = math.fsum(float(p) * float(tp) / float(total)
                   for p, tp, total in zip(group_positive[::-1], cumulative_positive, cumulative_total)) / positives if positives else None
    return dict(count=int(count), positive=positives, negative=negatives, auc=auc, ap=ap)


def threshold_metrics(labels, scores, cutoff):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    require(labels.ndim == scores.ndim == 1 and labels.shape == scores.shape and
            np.isin(labels, [0, 1]).all() and np.isfinite(scores).all() and math.isfinite(cutoff),
            'Threshold metrics require aligned binary labels and finite scores/cutoff')
    predicted, positive = scores >= cutoff, labels == 1
    tp, fp = int((predicted & positive).sum()), int((predicted & ~positive).sum())
    fn, tn = int((~predicted & positive).sum()), int((~predicted & ~positive).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall,
                specificity=specificity, f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                iou=tp / (tp + fp + fn) if tp + fp + fn else None,
                predicted_positive=tp + fp, cutoff=float(cutoff))


def fixture_checks():
    raster = np.arange(1, 10).reshape(3, 3)
    expected = [[[1,2,3],[4,5,6],[7,8,9]], [[3,6,9],[2,5,8],[1,4,7]],
                [[9,8,7],[6,5,4],[3,2,1]], [[7,4,1],[8,5,2],[9,6,3]],
                [[3,2,1],[6,5,4],[9,8,7]], [[9,6,3],[8,5,2],[7,4,1]],
                [[7,8,9],[4,5,6],[1,2,3]], [[1,4,7],[2,5,8],[3,6,9]]]
    for code, truth in enumerate(expected):
        require(np.array_equal(d4(raster, code), truth), 'Golden D4 raster mismatch')
    target = np.array([0., 1., 0., 1.])
    teacher = np.array([-1000., -2., 3., 1000.])
    identity = support_losses(teacher, teacher, target)
    require(identity['kl'] == identity['probability_drift'] == 0., 'Teacher identity mismatch')
    extreme = support_losses(-teacher, teacher, target)
    require(all(math.isfinite(value) for value in extreme.values()), 'Unstable extreme logits')
    require(extreme['kl'] >= 0, 'Negative KL')
    rows, columns = np.indices((128, 128))
    manual = np.broadcast_to(columns < 48, (2, 1, 128, 128)).copy()
    pseudo = np.broadcast_to(columns >= 80, (2, 1, 128, 128)).copy()
    hard = np.broadcast_to((rows % 2).astype(np.float32), (2, 1, 128, 128)).copy()
    batch = dict(manual_mask=manual, pseudo_mask=pseudo,
                 manual_hard_target=hard, pseudo_hard_target=hard)
    zeros = np.zeros((2, 1, 128, 128), dtype=np.float32)
    unknown_changed = zeros.copy()
    unknown_changed[~(manual | pseudo)] = 987.
    for coefficient in (0., .25):
        record = loss_record(zeros, zeros, batch, coefficient)
        require(record == loss_record(unknown_changed, zeros, batch, coefficient),
                'Unknown output locations influenced the losses')
        require(record['base'] == record['full'] == (1. + coefficient) * math.log(2.),
                'Separate-support weighting mismatch')
        require(record['loss_U'] == record['loss_P'] == record['loss_R'] == math.log(2.),
                'Zero-logit BCE mismatch')
    require(equal_batch_means([{'x': 0.}] * 16 + [{'x': 2.}] * 16) == {'x': 1.},
            'Equal-batch weighting mismatch')
    require(not bit_equal(np.array([0.], dtype=np.float32), np.array([-0.], dtype=np.float32)),
            'Bit comparison lost signed zero')
    try:
        support_losses(np.array([]), np.array([]), np.array([]))
    except ValueError:
        pass
    else:
        raise ValueError('Empty support accepted')
    metric_labels, metric_scores = np.array([1, 0, 1, 0]), np.array([.5, .5, .9, .1])
    ranked = rank_metrics(metric_labels, metric_scores)
    require(ranked['auc'] == .875 and abs(ranked['ap'] - 5 / 6) < 1e-15,
            'Grouped tie-aware ranking mismatch')
    tied = rank_metrics(metric_labels, np.ones(4))
    require(tied['auc'] == tied['ap'] == .5, 'All-tied ranking mismatch')
    require(rank_metrics(np.array([0, 0]), np.array([1., 2.]))['ap'] is None and
            rank_metrics(np.array([1, 1]), np.array([1., 2.]))['auc'] is None and
            rank_metrics(np.array([1, 1]), np.array([1., 2.]))['ap'] == 1.,
            'Single-class ranking convention mismatch')
    thresholded = threshold_metrics(metric_labels, metric_scores, .5)
    require([thresholded[name] for name in ('tp', 'fp', 'fn', 'tn')] == [2, 1, 0, 1],
            'Inclusive cutoff convention mismatch')
    return dict(status='PASS', golden_d4_cases=8, teacher_identity_cases=1,
                extreme_logit_cases=1, separate_support_weighting_cases=2,
                unknown_support_exclusion_cases=2, equal_batch_cases=1,
                signed_zero_cases=1, empty_support_rejection_cases=1,
                ranking_tie_cases=2, ranking_single_class_cases=3, inclusive_threshold_cases=1,
                scientific_arrays_read=0, model_imports=0)
