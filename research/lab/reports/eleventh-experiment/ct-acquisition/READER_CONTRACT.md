# Bounded Frag4 CT range reader

`reader.py` is ready for a root-reviewed frozen plan. Preparation used synthetic TIFF objects and an in-process HTTP-response fixture only. No real CT acquisition or model call was performed by the reader owner.

The CLI accepts `--plan`, `--plan-sha256` and a **fresh** `--output-dir`. The plan must pin this reader and all scientific/selection inputs. Recommended pins include `RESEARCH_DESIGN.md`, the frozen `reference-selection/selection-result.json`, optical-depth sampling contract, asset inventory and TIFF structural observations. All pins are verified before requests and after pooling; a changed input stops with a retained record.

```json
{
  "schema_version": 1,
  "ready": true,
  "input_pins": [{"path": "ABSOLUTE_PATH", "sha256": "SHA256", "bytes": 0}],
  "source_inventory": {"path": "ABSOLUTE_PATH", "sha256": "SHA256", "bytes": 0},
  "source_layers": [{"index": 7, "url": "EXACT_INVENTORY_URL"}],
  "expected_tiff": {"width": 6064, "height": 9791},
  "crops": [
    {"case_id": "frag4-context1", "bbox_source_xywh": [1152, 0, 1152, 1152]},
    {"case_id": "frag4-context2", "bbox_source_xywh": [3456, 8064, 1152, 1152]}
  ],
  "pooling": {
    "factor_zyx": [3, 3, 3],
    "formula": "float 32(sum_uint32_27/(27*257))",
    "quantization": "none"
  },
  "limits": {
    "maximum_http_attempts": 260,
    "maximum_charged_response_bytes": 1717986918,
    "maximum_per_range_bytes": 14680064,
    "maximum_peak_rss_bytes": 1073741824
  }
}
```

This is a schema illustration, not an executable plan: provide all 51 ascending layer records7 through57 and real input pins. Every layer URL must match the retained65-layer asset inventory exactly. The reader does not infer URLs from a filename template. It supports smaller crops for its synthetic fixture; the actual root plan fixes1152² crops and17×384² outputs.

Each layer independently requires three byte-range GETs: the 8-byte classic TIFF header,2-byte IFD entry count and exact IFD table. The first response establishes the object's strong quoted ETag and total size; every subsequent range uses `If-Match` and checks both again. The reader accepts only strict206 responses with exact Content-Range/Content-Length and identity encoding. It uses verified TLS and disables redirects; there are no automatic retries. An HTTP200 full-object response, source change, truncation or unsupported TIFF convention stops the acquisition.

The IFD must describe little-endian unsigned16-bit, uncompressed, min-is-black grayscale in a single contiguous strip. Dimensions, strip byte count and bounds are validated per file; multi-image IFD chains, tiles, extra samples, predictors, incompatible orientations or sample formats are rejected. Absent Orientation/SampleFormat/PlanarConfiguration/FillOrder/Predictor use their TIFF baseline defaults, which are explicit in the reader source; absent tags remain identifiable in its retained tag list. The initial layer32 structural observation does not substitute for checking the other 50 files.

Only the required full-width row bands are downloaded, with overlapping or adjacent spans merged. For the two declared disjoint spans, each band is1152×6064×2 =13,971,456bytes. Across51 layers the raw CT bands total **1,425,088,512bytes**. Three metadata requests plus two bands per layer require **255GETs**, within260; maximum 128-entry IFDs add at most 79,152metadata bytes. An individual merged band must fit the 14 MiB cap.

Every attempted request reserves its full frozen response cap **before** opening the connection. A failed request retains that charge, including when its declared length exceeds the cap; the partial stored byte count and SHA are separate. Partial bytes returned through `IncompleteRead` are preserved. Responses are streamed to disk in blocks of at most64KiB. All successful headers, IFDs and full-width bands remain under `ranges/`, with paths, ranges, SHA256, object sizes and ETags in `request-ledger.json`. These records describe charged/retained response bytes, not total wire traffic. A failure leaves additive `provenance.json` with `status:STOP`; no resumption, automatic retry or silent replacement occurs.

The decoder seeks directly to the selected X crop on each retained row; it does not decode an entire full-width band image. For each output pixel, nine XY uint16 samples from each of three consecutive depth planes are summed exactly in uint32. The sum of 27values is divided by`27*257` in float64, then cast once to float 32, preserving fractional intensities without uint8 quantization. Source groups are[7,8,9] through[55,56,57], with native depth centers8,11,…,56; output depth8 is native32. XY centers are`crop_origin+1+3*output_index`. The paper-declared pooled spacing is9.72µm in all three axes.

Working allocations are bounded by two pooled float 32 outputs, two uint32 accumulators and one decoded native crop, with raw bands streamed to disk. Peak RSS is checked against 1 GiB during acquisition and recorded; this is an acceptance bound, not a claim that Linux enforces a hard RSS allocation limit. The successful synthetic full-pipeline fixture peaked at 32,890,880bytes. Raw retained disk bytes cannot exceed charged response bytes; pooled outputs add20,054,016 payload bytes for the actual two 17×384² arrays, plus NPY headers and small JSON records.

The fixture receipt records eight passing tests, including all 51 synthetic layers and independent nested27-sample expectations for asymmetric crops, source ETag changes, failed/partial ranges, unsupported TIFF conventions, budget prechecks and no quantization. Socket connect/connect_ex tripwires were active. It verifies the arithmetic and acquisition contract; it does not validate real source pixels, optical ink labels, papyrus placement or physical independence.

Root execution after code/plan freeze:

```sh
PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/herculaneum/bin/python \
  /workspace/herculaneum-lab/reports/eleventh-experiment/ct-acquisition/reader.py \
  --plan /ABSOLUTE/FROZEN/PLAN.json --plan-sha256 ACTUAL_PLAN_SHA256 \
  --output-dir /ABSOLUTE/FRESH/ACQUISITION
```

Keep earlier records and this frozen source unchanged. Any subsequent correction requires a coordinated pause and an additive version before execution.
