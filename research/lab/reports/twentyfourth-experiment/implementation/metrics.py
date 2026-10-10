"""Descriptive exact-tie rank metrics; no threshold classification or fitting."""
import math
import numpy as np
from common import require


def rank_metrics(labels,scores):
    labels=np.asarray(labels);scores=np.asarray(scores,dtype=np.float64)
    require(labels.dtype==np.bool_ and labels.ndim==scores.ndim==1 and labels.shape==scores.shape,'Rank metric inputs differ')
    require(np.isfinite(scores).all() and labels.size>0,'Empty/nonfinite rank metrics')
    n=int(labels.size);p=int(labels.sum());negative=n-p
    require(p>0 and negative>0,'Both supplied reference classes required')
    order=np.argsort(-scores,kind='stable');s=scores[order];y=labels[order]
    ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),n-1]
    tp=np.cumsum(y,dtype=np.int64)[ends];fp=ends+1-tp
    tpr=np.r_[0.,tp/p];fpr=np.r_[0.,fp/negative]
    roc=float(np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])*.5))
    ap=float(np.sum(np.diff(tpr)*(tp/(ends+1))))
    return dict(roc_auc=roc,average_precision=ap,unique_scores=int(ends.size))


def paired_difference(native,baseline):
    a=np.asarray(native,dtype=np.float64);b=np.asarray(baseline,dtype=np.float64)
    require(a.ndim==b.ndim==1 and a.shape==b.shape and a.size>0 and np.isfinite(a).all() and np.isfinite(b).all(),'Paired scores differ')
    d=a-b;absolute=np.abs(d)
    return dict(mean_signed=float(np.mean(d)),mean_absolute=float(np.mean(absolute)),
        p95_absolute=float(np.quantile(absolute,.95,method='linear')),max_absolute=float(np.max(absolute)))


def raw_delta(native,baseline):
    return {k:native[k]-baseline[k] for k in ('roc_auc','average_precision')}


def checkpoint_changes(models):
    return {role:raw_delta(models[role]['reverse'],models[role]['forward']) for role in ('seed42','native')}


def difference_of_changes(models):
    changes=checkpoint_changes(models)
    return raw_delta(changes['native'],changes['seed42'])


def parent_summary(contexts,parent):
    ids=[key for key,row in contexts.items() if row['parent_papyrus']==parent]
    require(len(ids)==({'Paris1':2,'Paris2':4}[parent]),'Physical parent context count differs')
    models={role:{direction:{key:math.fsum(contexts[c]['models'][role][direction][key] for c in ids)/len(ids)
        for key in ('roc_auc','average_precision')} for direction in ('forward','reverse')} for role in ('seed42','native')}
    for role in models: models[role]['reverse_minus_forward']=raw_delta(models[role]['reverse'],models[role]['forward'])
    return dict(context_ids=ids,weighting='equal context within named physical parent',models=models,
        difference_of_changes=difference_of_changes(models))
