"""Independent fixtures for the numerical operations that could confound comparison."""
from learn_resolution import *
from scipy.ndimage import map_coordinates
# Compare 4× averaging + align_corners=False interpolation against explicit coordinates.
a=np.arange(32*32,dtype=np.float32).reshape(32,32)**1.1
blocks=np.array([[a[y:y+4,x:x+4].mean() for x in range(0,32,4)] for y in range(0,32,4)])
grid=np.meshgrid((np.arange(32)+.5)/4-.5,(np.arange(32)+.5)/4-.5,indexing='ij');ref=map_coordinates(blocks,grid,order=1,mode='nearest');need(np.allclose(degrade(a),ref,atol=.001),'Independent degradation interpolation')
need(np.array_equal(degrade(np.ones((32,32),np.float32)),np.ones((32,32),np.float32)),'Constant conservation')
# Whole4×4 block invalid, and bilinear neighbors invalid as specified.
v=np.ones((32,32),np.float32);v[8,8]=0;cv=v.reshape(8,4,8,4).all((1,3)).astype(np.float32);interp=F.interpolate(torch.from_numpy(cv)[None,None],size=(32,32),mode='bilinear',align_corners=False)[0,0].numpy();need((interp[6:14,6:14]<1).all() and (interp[:6]==1).all(),'Interpolation footprint support')
# Tie-aware AP fixture: positives and negatives share scores, AP is not trapezoidal PR area.
y=np.array([[1,0,1,0]],np.float32);s=np.array([[.8,.8,.2,.2]],np.float32);r=metrics(y,s,np.ones_like(y,bool));need(abs(r['ap']-.5)<1e-12 and abs(r['auroc']-.5)<1e-12,'Tied AP/AUROC')
# Independent all-pairs AUROC and threshold-by-threshold AP on ties.
rng=np.random.default_rng(19)
for _ in range(20):
 truth=rng.integers(0,2,30);score=rng.integers(0,5,30);ap,auc=ranking_metrics(truth,score);p=score[truth==1];n=score[truth==0];ref_auc=((p[:,None]>n).sum()+.5*(p[:,None]==n).sum())/(len(p)*len(n));ref_ap=0.;old=0
 for cutoff in sorted(set(score),reverse=True):
  hit=score>=cutoff;tp=int(truth[hit].sum());ref_ap+=(tp-old)/len(p)*tp/hit.sum();old=tp
 need(abs(ap-ref_ap)<1e-12 and abs(auc-ref_auc)<1e-12,'Independent metric arithmetic')
# Raw coordinate delimiter preserves a missing interior; trailing delimiter is not a column.
from acquire_crops import fields
need(fields(b'1.0;2.0;nan;3.0;\r\n')==[b'1.0',b'2.0',b'nan',b'3.0'],'Raw missingness and delimiter')
# Demonstrate model output/central support and deterministic paired init.
torch.manual_seed(SEED);one=Net();torch.manual_seed(SEED);two=Net();need(all(torch.equal(a,b) for a,b in zip(one.parameters(),two.parameters())),'Identical initialization');need(one(torch.zeros(2,1,192,192)).shape==(2,1,192,192),'Output context')
# Actual data contracts and identical stochastic minibatches across paired arms.
data=prepare();a1=draw_batch(data,[0,1],'native',np.random.default_rng(SEED));a4=draw_batch(data,[0,1],'coarse4',np.random.default_rng(SEED));need(torch.equal(a1[1],a4[1]) and torch.equal(a1[2],a4[2]),'Identical sampled labels/support/augmentations');need(not torch.equal(a1[0],a4[0]),'Distinct observation quality')
need(len({d['parent'] for d in data})==3,'Three physical parents');need(all(np.isfinite(d['images'][q]).all() for d in data for q in QUALITIES),'Finite context only')
save(P/'LEARNING_FIXTURES.json',dict(status='PASS',checks=11,synthetic_model_forwards=1,scientific_model_forwards=0,training_updates=0,parameters=sum(p.numel() for p in one.parameters()),cases=[dict(case=d['case'],support=int(d['support'].sum()),positive=int(d['label'][d['support']].sum())) for d in data]));print('PASS 11 learning-contract checks')
