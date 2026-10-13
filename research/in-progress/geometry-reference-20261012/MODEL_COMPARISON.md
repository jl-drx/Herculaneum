# Next surface-field execution contract

Source audit only; no model weights or prediction payloads were acquired.
Pinned Socratic revision: `7f86cbf8d38155b37fdd39449c6775ec50344b6c`.
The four fetched source files and their exact receipts are in `model-source.zip`.

The twelve retained CT contexts **already have the declared F0 inference
footprint**: uint8 ZYX, 192³ context, central 128³ output, 32-voxel halo. The
published `grid_inference.py` takes precisely that centre crop; the released
benchmark `cube_infer.py` uses it for both m7 and F0. PHerc0800's 8.64 µm and
PHerc0814's 9.362 µm spacing are within the recipe's accepted 7.5–10 µm range.
This is a source/input contract match, not a successful neural execution.

The shipped normalization clips raw CT to [0,212], then uses fixed mean
87.54424285888672 and standard deviation 47.74376678466797. These are surface
model parameters, not the ink models' MAD normalization. The shipped F0 recipe
uses CUDA BF16, eight mirror transforms and threshold 0.30. Its repair/tracing
postprocessors were pinned at 0.35 and are not automatically qualified at 0.30.
The public m7 thresholded field is 0.20 with a different chunk/grid history.

Keep three contrasts separate:

1. **Weights under a shared execution:** same 192³ input, preprocessing,
   precision, mirror policy and output crop for m7/F0. Retain raw probabilities.
   The same eight forwards can retain the unmirrored output and eight-way mean,
   avoiding a redundant extra no-TTA run when the implementation guarantees this.
2. **Matched operating point:** a predeclared common threshold (the existing
   0.35 comparison is a suitable starting control) and the same tracer/seeds.
   A threshold selected on the new references would be development tuning and
   need a separate confirmation endpoint.
3. **Published practical pipelines:** each release's own threshold/TTA and
   compatible postprocessor. Differences are pipeline differences, not an
   isolated training-method effect. The public m7 binary mask cannot substitute
   for a matched raw-probability baseline or support threshold calibration.

For twelve contexts, two models and eight transforms, the shared execution is
192 forward passes, plus synthetic qualification and measured warmup. This
counts work; it is not a runtime or price forecast. Local CPU source review
does not establish that the CUDA kernels, BF16 trajectory, model loading or
memory fit are correct. A separately approved 24 GB GPU timing qualification
would measure those before a larger job. No new paid compute is authorized.

Before that request, pin the actual weight hashes and CUDA image, qualify exact
checkpoint loading and restart, freeze the input/probability artifact manifest,
and establish working private artifact delivery and independent shutdown. Use
the existing £10 total cap across retries and overheads. Do not launch a geometry
accuracy comparison until its references distinguish branch identity and recto
placement; a successful runtime alone cannot supply those references.
