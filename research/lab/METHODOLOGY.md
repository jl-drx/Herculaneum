# Initial scientific workflow for review

## Scope

Start with a small, documented, already flattened surface image and a matching ink annotation from an authoritative public challenge asset. Download only the minimum crop or fragment needed once its identity, terms, dimensions, and annotation coordinates are checked. This lets a CPU environment establish image handling, provenance, and evaluation before attempting full scroll segmentation or model training. If the only available annotations are incomplete or contain no reliable negative pixels, treat the task as evidence review and use only supported metrics.

A rendered image alone may not retain the volumetric signal needed for genuine ink detection. If ink information is provided through a surface-aligned volume stack, use a bounded local crop with documented layer indices and labels; do not mistake a papyrus texture rendering for an ink-sensitive input. An asset identifier and verified source are prerequisites to interpreting its pixels.

## Small CPU setup

Use an isolated Python environment with NumPy, SciPy, Pillow, and Matplotlib. Add `tifffile` only when the selected asset needs TIFF decoding. A small project can use `unittest` for integrity checks without introducing a test dependency. Defer PyTorch, CUDA, large volumes, and training until the initial asset and scientific target justify them. Record Python and package versions, input SHA-256 hashes, exact commands, random seeds, and machine resource limits. Route Matplotlib's cache to a writable location through `MPLCONFIGDIR`.

The host currently imports NumPy 2.3.5, SciPy 1.17.0, Pillow 12.3.0, pandas 2.2.3, Matplotlib 3.10.8, and PyYAML 6.0.3. `tifffile`, Zarr, PyTorch, scikit-image, OpenCV, and pytest were unavailable during inspection. This describes the host, not a new environment's resolved dependencies.

## Stages and review gates

1. **Asset verification.** Record the source, exact asset identity, checksum, citation, access terms, scroll/fragment/segment identifiers, coordinate system, scale if supplied, annotation version, and retrieval time. Keep source images immutable. Inspect dimensions, bit depth, channels/layers, orientation, finite values, mask alignment, and a representative crop. Document unknown metadata rather than guessing it.
2. **Surface extraction and segmentation.** For already flattened assets, preserve the producer's mapping and rendering provenance. For raw CT data, identifying a papyrus surface, tracking it, constructing a mesh, and flattening it are separate research work. Inspect extraction artifacts and adjacent layers before claiming downstream ink. A full segmentation workflow is outside the initial CPU proof of operation.
3. **Ink evidence.** Generate unaltered image views, documented contrast views, and coordinate-aligned label overlays. A threshold or handcrafted filter can establish a baseline and exercise evaluation; its output is a candidate signal, not proof of writing. Compare suspected strokes with neighboring fiber texture, seams, cracks, rendering artifacts, and neighboring layers where available. Preserve original intensity data for each candidate.
4. **Reading.** Transcribe only characters supported by the displayed evidence. Give each candidate reading its image coordinates, alternatives, reviewer, and confidence rationale. Keep unknown spans explicit. Seek independent review of letter shapes and stroke continuity before using vocabulary or expected content to choose between alternatives.
5. **Language and interpretation.** Establish Greek, Latin, or an unresolved language from evidence. Store diplomatic transcription separately from normalized spelling, word division, expansions, restored text, and translation. Distinguish visible text from supplied letters and conjectures. Dictionary and language-model suggestions are hypotheses; neither establishes a letter in the image. Keep translations traceable to an explicitly selected transcription version.

## Low-cost validation

- **Pipeline integrity:** use tiny synthetic fixtures with known arrays to check crop coordinate round-trips, mask/image registration, unchanged source bytes, output determinism, schema validation, and metric edge cases. Synthetic strokes validate software, never recovery of ancient writing.
- **Authentic evidence:** on a small annotated real asset, show paired original/contrast/annotation/proposal views. Measure precision and recall, intersection-over-union, and optionally precision-recall AUC only over known labeled pixels. Mark unknown pixels separately; do not silently convert unlabeled areas to background. Annotations express the available reference, not complete truth.
- **Decision thresholds:** choose thresholds on development data and freeze them before a held-out evaluation. Report the chosen threshold, labeled positive/negative counts, ignored pixels, and confidence limits or the limitations of the sample size. Include a simple baseline and disclose that a tiny crop cannot establish generalization.
- **Reading quality:** use independent transcription of a limited, blinded region when feasible. Report agreements and alternatives. Evaluate against a vetted reading only where such a reading actually exists. A plausible translation or high language-model score is not validation of ink recovery.

## Leakage controls

Split by physical provenance, not by random pixels. Keep crops, overlapping tiles, adjacent layers, alternate renderings, and augmented variants of the same physical area together. Separate scrolls/fragments or sufficiently separated physical regions where the dataset permits it, and document any weaker split. Exclude held-out labels from threshold choice, feature selection, preprocessing tuning, and model selection. Record public prior readings and any pretraining exposure where known; public challenge material may not offer a pristine blind benchmark. Review the current program's rules before deciding which public labels or existing models can enter a submission. Do not claim challenge eligibility from a local smoke test.

## Minimal evidence record

```json
{
  "asset": {
    "id": "source-defined identifier",
    "source_url": "authoritative asset URL",
    "sha256": "verified full-file digest",
    "retrieved_at": "ISO-8601 timestamp",
    "terms_url": "source-defined terms URL",
    "scroll_id": null,
    "fragment_id": null,
    "segment_id": null,
    "annotation_version": null,
    "coordinate_system": "documented source coordinate convention"
  },
  "region": {
    "id": "local region identifier",
    "bbox_xywh": [0, 0, 1, 1],
    "layer_indices": null,
    "source_to_display_transform": "recorded crop and display transform",
    "processing_run_id": "immutable run record"
  },
  "reading": {
    "status": "unread",
    "language": null,
    "diplomatic_text": null,
    "alternatives": [],
    "uncertain_spans": [],
    "supplied_spans": [],
    "normalized_text": null,
    "translation": null,
    "reviewers": [],
    "evidence_notes": []
  }
}
```

Use a separate record per candidate or line. Uncertain and supplied spans should carry indices into the diplomatic transcription, image evidence coordinates where applicable, a rationale, and reviewer attribution. Indices must declare whether they count Unicode code points or another unit; use UTF-8 storage and preserve the original code points. Do not generate placeholder ancient text that could be mistaken for an observed reading.

## First deliverable

A reproducible command that validates a small verified asset, writes provenance and crop metadata, produces a reviewable image/annotation panel, and reports an appropriately limited baseline. Until an authentic asset is selected, deliver only the pipeline fixture and clearly label it as synthetic. The next scientific milestone is independently supported letter evidence on a real surface, followed by transcription and translation; environment setup is a prerequisite, not evidence that a scroll has been decoded.
