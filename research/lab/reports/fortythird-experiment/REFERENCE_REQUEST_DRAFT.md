# Unsent request: Paris4 native ink reference coordinates

Suggested recipient: Giorgio Angelotti or W. Brent Seales, the materials contacts named in arXiv:2606.29085v1. No message has been sent. This is a new request about the Paris4 visible-ink reference, separate from the earlier Frag6 rendering question.

Subject: Small coordinate-linked Paris4 reference for an acquisition-quality control

Hello,

We are developing Plinius, a small research programme on ink detection from coarse Herculaneum CT. Before training a high-resolution teacher, we want to compare learning from native CT and synthetically degraded observations of the same papyrus, with independent reference labels and physically separated regions.

Your PHercParis4 example in Figure 2 and Extended Data Figure 3 of arXiv:2606.29085v1 seems ideal. We found the public 2.4 µm reconstruction `20260411134726` and the released `20260428123845-v3-78k-fullsup` volumetric ink prediction. We understand that the dense prediction follows DINO-guided/intersection supervision and self-distillation, so we would not use it as independent test truth.

Could you point us to an existing release containing the exact native-volume coordinates of those figure regions and the original expert-selected ink voxels? If that is not already public, a small packet would be enough:

- Volume ID, level, XYZ/ZYX convention and a bounding box or slice coordinates for the visible deposits, plus the correct surface/wrap where applicable.
- Expert-supported ink-positive and non-ink-papyrus points or small masks, with ambiguous material kept unknown. A handful of separated examples is sufficient for initial reference QC; a learning study would need enough separately reserved regions after that.
- A short note identifying which labels came from direct CT inspection and which had model assistance, and any known placement/material uncertainty.

We can stream the public CT ourselves; no whole-volume transfer or new dense annotation is requested. Existing figure scripts or annotation exports would be welcome if they already carry this information. Please also identify the applicable reuse/attribution terms for the reference annotations.

Our present small-panel result is that broad structures correspond across scans, while local registration remains ambiguous and no independent ink reference was established. This does not challenge the published visible-ink result; we are trying to select its documented region correctly and avoid evaluating a descendant model against its own supervision.

Thank you.

## Why this is the minimum useful request

The inspected public catalog links CT, meshes and model predictions, but it does not identify the expert seed coordinates or a separately qualified negative set. A coordinate pointer can localize the positive-control region without another blind CT sample. Native/synthetic learning would not require old-to-new scan registration; the real coarse arm would still need that later. Do not send this draft or imply institutional affiliation on the user's behalf without explicit authorization and confirmed sender identity.
