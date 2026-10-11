# Next actual learning comparison

This is a design recommendation informed by the completed controls and intake,
not a claim that training ran or a frozen model/optimizer protocol.

## Question

Can a fresh full-depth native-CT learner predict the released native annotations
on an excluded physical winding better than a simple native-input baseline?
Answer this before making a CT-uncertainty method the main intervention. The
continuous fine-teacher baseline remains the first paired-supervision arm once
adequate training transfer support exists. Generic distillation already has
prior art, including Hecate; neither target averaging nor this baseline is a
novel model contribution by itself.

Use all admitted mask-selected cores from w035/w039/w044, retaining class-poor
cores rather than replacing them. Three leave-one-winding-out folds, with the
conservative cross-winding context exclusions supplied by the geometry audit.
Do not mix aliases, neighboring-depth renders or overlapping contexts between
training and evaluation. These are three portions of one scroll, not three
independent specimens. Do not initialize from released weights trained on these
annotations or call the evaluation pristine public-data generalization.

## Baseline and learner

Compare a training-prevalence constant predictor and a low-capacity learner of
the native28-plane vector against a modest complete spatial CNN retaining all28
planes. No saved prediction fusion, residual depth adapter, teacher input at
inference, position coordinates, or artificial change of depth window. Freeze
architecture, seeds, loss, sampling, normalization, optimization ceiling and
stopping rule before looking at any model's held-out predictions.

Use the same supervised/native-valid pixels for every learner. Supervision zero
means ignored, even when a stored ink label is nonzero there. Fit normalization
and any decision threshold on training data only. Keep unweighted masked loss
and a documented sampling distribution; avoid silently combining positive
oversampling with positive loss weights. Report effective exposure, not only
update counts.

## Stop and decide

First use training-only curves to establish whether the chosen representation
can fit and whether optimization has saturated; CPU duration is not a monetary
gate. Set a finite maximum and plateau rule prospectively. If the ceiling is
reached while fit is still improving materially, the result is budget-limited,
not evidence that native CT lacks signal. If fit saturates badly, one declared
training-only capacity/optimizer diagnosis is appropriate; repeated tiny
continuations are not.

Evaluate one frozen endpoint in each excluded winding, across paired
initializations, against the same native annotations. Report annotation AP/AUC,
training and evaluation loss, class prevalence, each region's results and blank
supervised-core response. Do not estimate uncertainty by treating millions of
pixels as independent, or import the old Paris +0.01 gate as a measured noise
floor here. Freeze a practical advancement margin in the training protocol once
runtime, optimization feasibility and endpoint aggregation are specified.

A consistent cross-winding gain over the simple baseline, with adequate fit and
preserved context exclusions, would support investing in the matched ordinary
soft / CT-averaged / smoothing comparison on broader qualified training tissue.
Fit without transfer would instead prioritize coverage or reference mismatch.
A low-capacity null would not establish absence of recoverable native signal.

The present operator study provides no reason to add an uncertainty architecture
now: local score averaging is almost the score at the local mean coordinate on
these four supports. It does not prove such averaging could never improve
learning elsewhere. Fine/native registration is a future paired-target issue;
it does not block this native-frame annotation-agreement baseline.

Paid GPU work remains deferred. This CPU study is not an automatic GPU spend
authorization or an arbitrary CPU AP hurdle for all future GPU research.
