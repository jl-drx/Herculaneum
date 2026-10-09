"""Record the completed root visual review; creates no pixel annotations."""
from common import *
observations = {
 'pair01': 'Broad sheet edge and compact bright features; several fine laminae in cross-section. Large structures have coarse counterparts, but local physical landmarks were not measured.',
 'pair02': 'Folded laminae and an air/sheet boundary near the nominal surface plane. Fine and coarse structure is visible; no stable set of independent local correspondences was established.',
 'pair03': 'Bent laminar sheet with a curved crest. White wedges in the review are outside acquired coverage, not anatomical features. No registration fit was admitted.',
 'pair04': 'Repeated laminae and compact bright features. The repeated layer pattern makes local correspondence ambiguous.',
 'pair05': 'Separated sheet bands and folds, with a compact bright feature. Parts of the nominal plane pass through an air gap or sheet edge. Broad correspondence does not resolve local ambiguity.',
 'pair06': 'Many repeated thin laminae and a large bright feature. A small view-coverage wedge is missing data. No registration fit was admitted.'
}
figs=json.loads((HERE/'review/FIGURES.json').read_text())
rows=[]
for row in figs['figures']:
 verify(row['figure']);verify(row['views'])
 rows.append(dict(id=row['id'],figure=row['figure'],root_visually_reviewed=True,observation=observations[row['id']],verified_ink_positive_pixels=0,verified_nonink_papyrus_pixels=0,all_material_identity_ambiguous=True,exposure='Native and coarse CT visually inspected for structure and possible reference evidence; not an untouched test'))
result=dict(status='STOP_LEARNING_FOR_THIS_PANEL',created_at_utc=now(),reviewer='root',review_scope='Six retained multi-plane figures; structural screening by the assistant, not expert annotation or volumetric ink ground truth',rows=rows,reference_established=False,independent_ink_positive_pixels=0,independent_nonink_papyrus_pixels=0,public_label_bodies_read=0,detector_predictions_used=0,physical_registration_established=False,native_learning_admitted=False,synthetic_coarse_learning_admitted=False,real_coarse_learning_admitted=False,new_model_calls=0,GPU_spend_GBP=0,reasons=['No defensible independent ink-positive and non-ink papyrus labels established in this panel','Four local correlation screens failed; two windows failed the frozen blur-coverage guard','Correlation peak disagreements are proxy uncertainty, not measured physical alignment errors'],material_cautions=['Bright features are not identified as ink or a particular chemical inclusion','Air is not a negative ink label on a writing surface','Quiet-looking papyrus is not verified non-ink','No surface recentering is justified'],next_task='Reference-first positive-control packet: obtain documented ink-positive/non-ink support with known acquisition and physical coordinates before selecting another CT panel')
start=json.loads((HERE/'GEOMETRY_GO.json').read_text())['created_at_utc']
elapsed=(datetime.fromisoformat(result['created_at_utc'])-datetime.fromisoformat(start)).total_seconds()
need(elapsed<=45*60,'Admission completed inside frozen time budget')
result['seconds_since_acquisition_GO']=elapsed
save(HERE/'REFERENCE_REVIEW.json',result)
print(json.dumps(dict(status=result['status'],seconds_since_acquisition_GO=elapsed,reviewed_figures=len(rows))))
