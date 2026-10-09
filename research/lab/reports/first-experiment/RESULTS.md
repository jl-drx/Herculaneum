# First PHerc0800 experiment

The pipeline now runs on real scan-derived surface data. It produced a shared model hotspot, but **no defensible letter reading or translation**. The candidate response is diffuse, and its output depends on depth direction. This is a useful baseline and an unresolved image-analysis lead, not a text discovery.

## What was run

Live metadata identifies PHerc0800 volume `20250521135224` and six released surface segments. This experiment used the smallest, `20251029010146`, whose native array contains 31 depth layers and a 980×980 canvas at 8.640 µm sampling. The full canvas spans about 8.47 mm per side; the actual valid papyrus occupies only part of that rectangle.

The downloaded native surface used 43 stored chunks (21,839,872 bytes); 21 unstored chunks were reconstructed with the array's documented Zarr v2 `fill_value=0`. Only explicit S3 `NoSuchKey` responses receive that treatment. Source hashes, headers, array metadata and the reconstruction records are preserved in the [surface provenance](../../data/experiments/pherc0800-surface980/provenance.json).

The geometry check retained the corresponding X/Y/Z coordinate TIFFs. Of 960,400 canvas pixels, 497,600 have complete valid bilinear mesh mappings. These form one connected area. Another 20,500 of those pixels have zero values across the entire observed stack; the signal screen excludes them. Two small coarse CT reads, including one containing the original mapped ROI, independently verify data access and localization.

Both official `ink_9um` final-step checkpoints were downloaded from a fixed revision and matched their advertised full SHA-256 hashes. Their actual embedded configurations reconstructed the same 34,546,498-parameter model, with strict matching of every state key. CPU execution used 17-layer 128×128 patches, upstream patchwise robust-MAD normalization, stride 64, and Hann blending. Each seed scored 173 tiles and skipped 52 with no valid geometry; all valid pixels received evaluated coverage. The full-segment runs took about 31 and 28 seconds of inference for the two seeds. Float32 outputs remain available; no image display transformation changes those stored scores.

## Result and controls

The initial 2.21 mm square pilot was mostly near the model's smoothed background target of 0.25. Widening to the full small segment revealed some shared higher responses. A diagnostic screen using both seed scores above 0.5 and a 64-pixel margin from invalid/zero-data areas found 3,200 pixels in four connected components. This threshold screens candidates; it is not a measured ink classifier threshold, accuracy score, or proof of writing.

The highest jointly scored interior point was native surface **(x=652, y=563)**, with scores approximately 0.703 and 0.712. Its 256×256 context at `(x=512, y=448)` has valid mesh mappings throughout. The crop was selected after viewing detector results and therefore cannot serve as an independent held-out accuracy test.

The two models share a broad response in that context, but the source shows strong material texture, irregular gaps and bright points. A [separate assistant visual review](visual-evidence-review.md) supports no coherent letterform from the panels; this is not a specialist papyrological assessment. Agreement between these seeds is a stability check; they share the same stated training corpus.

Fixed-coordinate controls were run on the selected context:

| Seed42 depth window | Mean absolute score change from forward baseline | Pattern correlation |
| --- | ---: | ---: |
| Layers 5–21, forward | 0.063 | 0.818 |
| Layers 9–25, forward | 0.045 | 0.945 |
| Layers 23–7, reversed | 0.118 | −0.158 |

These are model-behavior measurements, not ink accuracy. The control crop's central region exactly reproduced the corresponding full-surface seed42 output, confirming that the smaller diagnostic did not change its contributing model context.

## Native CT consistency

The full export metadata supplied the actual rendering command and historical code revision `530584554c031c982ba3aa243e1ea39c4bef9818`. It specifies flipped surface normals, 31 slices, and centered offsets −15 through +15 at a one-voxel step. The historical renderer confirms the half-pixel coordinate convention used in the geometry check.

At the hotspot `(652,563)` and two neighboring surface pixels, samples from two native CT chunks exactly reproduced **93 of 93** stored uint8 surface values. No offsets or transforms were fitted. Reversing the depth order instead gave mean absolute differences of about 25–33 intensity counts. The [native CT consistency report](native-ct-consistency.md) records the code and source provenance and the limits of this comparison.

This establishes the local rendering geometry and depth order at the tested points. It does not establish ink, globally validate every surface pixel, or resolve how the trained model's depth convention transfers to this scroll. The broad model response remains untranscribed.

## Review the evidence

- [Full surface, two seed maps and their difference](full-surface-baseline.png)
- [Candidate source layers beside model outputs](candidate-context.png)
- [Original pilot depth layers](observed-surface-layers.png)
- [Mapped coarse CT localization](linked-ct-localization.png)
- [Machine-readable control measurements](candidate-depth-controls.json)
- [Native CT consistency](native-ct-consistency.md)

The model's stated training scrolls differ from PHerc0800, but this is not independent checkpoint-provenance verification. There are no trusted ink labels or vetted text for this region. Training resolution also differs from the target by roughly 8–10%. These limits prevent precision/recall, authenticated-ink, transcription-accuracy, or prize-qualification claims.

The next scientific gate is separating ink evidence from the material texture that produces these model responses. Compare a matched reference region and additional independently selected surface regions, preserving the verified depth convention, before changing models or producing a reading. Diplomatic transcription and translation remain empty until the image evidence supports characters.
