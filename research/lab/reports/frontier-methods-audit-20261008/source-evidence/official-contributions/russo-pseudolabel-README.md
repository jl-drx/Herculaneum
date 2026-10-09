---
license: mit
tags:
  - vesuvius-challenge
  - ink-detection
---

# Dense pseudo-labels for the 9 um ink corpus

288.7 M supervised pixels across 7 aligned segments, against 5.8 M in the
manual labels those segments ship with. Same layout and same zarr parameters as
`scrollprize` `ink_9um/labels`, so it drops into a training config by changing
one path.

Download and unpack:

```
huggingface-cli download domenicor046/ink9um-dense-labels \
  ink9um-dense-pseudolabels.tar --local-dir .
tar -xf ink9um-dense-pseudolabels.tar
```

That gives `pseudo_labels/aligned-scrollprizeorg-21slices/<segment>/` holding
`<segment>_inklabels.zarr`, `<segment>_supervision_mask.zarr`, and
`<segment>_validation_mask.zarr` where one exists.

## How the labels were made

The canonical 2.4 um ink model is run on each segment's public 2.4 um surface
volume and pooled to the exact 9.6 um raster the 9 um model trains on, then
thresholded per segment at a value calibrated only on that segment's manual
supervision region. Manual labels win wherever manual supervision exists.

- `inklabels` = manual label inside manual supervision, else `teacher >= t*`
- `supervision_mask` = render-valid AND NOT validation_mask
- content on the middle slice only, which is the only label slice training reads
- values 0/255 uint8

Per-segment thresholds: w016 0.15, w017 0.33, w028 0.27, w029 0.25, 0814 0.53,
1667-w028 0.45, 1667-w029 0.36.

| segment | manual px | pseudo px | multiplier | canvas covered |
|---|---|---|---|---|
| pherc0139-w016 | 418,602 | 42,533,230 | 101.6x | 83.9% |
| pherc0139-w017 | 719,008 | 42,835,355 | 59.6x | 85.0% |
| pherc0139-w028 | 1,756,535 | 41,464,705 | 23.6x | 88.8% |
| pherc0139-w029 | 394,114 | 41,188,859 | 104.5x | 88.2% |
| pherc0814-46527 | 428,993 | 4,175,514 | 9.7x | 56.7% |
| pherc1667-w028 | 844,780 | 58,012,015 | 68.7x | 81.8% |
| pherc1667-w029 | 1,212,915 | 58,508,474 | 48.2x | 78.7% |
| **total** | **5,774,947** | **288,718,152** | **50.0x** | **83.2%** |

## Important caveats

**The three held-out validation regions are excluded from `supervision_mask` by
construction**, verified as exactly zero overlapping pixels per segment. Keep
the `validation_mask.zarr` files in place if you train on this: without them the
trainer silently produces no validation metric and no best checkpoint, and still
exits 0.

**w016's labels are the weakest of the seven.** Its threshold of 0.15 was the
most permissive and marks 28 percent of its canvas as ink. A model trained on
all seven renders w016's held-out letters as merged blobs, while excluding w016
brings them back. If you use this dataset, consider dropping w016 or
recalibrating it.

**These are model-derived labels, not human annotation.** They are noisier than
the manual labels, which is part of why the training recipe's
`bce_label_smoothing` of 0.5 is load-bearing.

## Provenance and licence

The scripts and documentation are MIT. The labels are derived from Vesuvius
Challenge data and models (`scrollprize/ink_9um`, `scrollprize/ink_canonical_2um`,
and the open-data S3 bucket) and remain subject to the terms of those sources.

Full results, figures, and the code that built this: [https://github.com/DomRusso2/ink9um-dense](https://github.com/DomRusso2/ink9um-dense)
