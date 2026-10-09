"""Replay fixed label/support draws only: quantify the frozen objective's class weighting."""
from learn_resolution import *
data=prepare();results=[]
for held in range(3):
 ids=[i for i in range(3) if i!=held];truth=np.concatenate([data[i]['label'][data[i]['support']] for i in ids]);natural=float(truth.mean());alpha=(1-natural)/natural;rng=np.random.default_rng(SEED+held);fractions=[];positives=0;count=0;h=__import__('hashlib').sha256()
 for step in range(STEPS):
  _,y,m=draw_batch(data,ids,'native',rng);yy=y.numpy();mm=m.numpy();pos=int(yy[mm].sum());n=int(mm.sum());fractions.append(pos/n);positives+=pos;count+=n;h.update(yy.tobytes());h.update(mm.tobytes())
 sampled=float(np.mean(fractions));optimal=alpha*sampled/(alpha*sampled+1-sampled);results.append(dict(held_out=data[held]['case'],training_cases=[data[i]['case'] for i in ids],updates_replayed=STEPS,valid_pixel_presentations=count,positive_presentations=positives,natural_support_prevalence=natural,mean_batch_sampled_prevalence=sampled,fixed_pos_weight=alpha,optimal_constant_for_sampled_weighted_BCE=optimal,label_support_sequence_sha256=h.hexdigest()))
save(P/'SAMPLING_AUDIT.json',dict(created_at_utc=now(),status='DIAGNOSTIC_ONLY_NO_PROTOCOL_CHANGE',results=results,conclusion='Positive-centered sampling and natural-prevalence BCE weighting both emphasize foreground. Raw sigmoid values are not calibrated natural-prevalence posteriors; fixed0.5 Dice and Brier must be read with this limitation. No threshold or loss changed after outputs.',scientific_model_forwards=0,optimizer_updates=0));print(json.dumps(results,indent=2))
