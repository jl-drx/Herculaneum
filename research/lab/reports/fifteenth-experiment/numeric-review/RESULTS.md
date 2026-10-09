The single authorized independent saved-output verification passed. All six PNG encodings, CRCs, raw binary values and used palette entries matched the pinned source receipts. The independent banded integral reconstruction agreed on all 155 candidates, all 79 eligibility decisions and every value in the 24 selected arrays. First/last selection, native spacing, source-center mappings, calibration/evaluation roles, parent papyri and seen/deferred exclusions also matched.

The checker completed in 4.416157 seconds with peak RSS 750,723,072 bytes (0.699 GiB), below the 3 GiB cap. Address space was capped at 12 GiB. Acquisition retained 652,579 bytes across six successful, recorded HTTP200 requests without redirects or retries. Completed selection files occupy 3,643,383 bytes and acquisition files 664,893 bytes, within the frozen caps.

| Fragment | Role / parent | Grid candidates | Eligible | Context 1 native x,y | Context 2 native x,y |
|---|---|---:|---:|---|---|
| frag1 | evaluation / Paris2 | 35 | 11 | 2304,1152 | 0,6912 |
| frag2 | evaluation / Paris2 | 96 | 57 | 1152,1152 | 8064,12672 |
| frag3 | calibration / Paris1 | 24 | 11 | 2304,1152 | 3456,5760 |

Each native box is 1152×1152. Its four retained arrays are 384×384, sampled at native centers +1+3i; labels are uint8 and the three masks are bool. The primary region uses the frozen 32-pixel border and centered native 33×33 homogeneous-label/full-surface support. Exact integer 90% surface gates were reconstructed without SciPy or author selector imports.

Result receipt: `completed-review/independent-verification.json`, 116,483 bytes, SHA256 `839391e588f4e2f7a246dc3958f9f55cf9b6799f0d6e91cb1be36872690734d4`. The pre-execution source/fixture freeze remains unchanged at `code-freeze.json`, SHA256 `23613ecdf5709b64af0c64b1a714f2f120174e9bd1ff3d8fc67de21872dffce1`.

The checker decoded only the six retained label/mask PNGs. It performed no CT decode, model call, checkpoint read, training, prediction, calibration or network call. All recorded input/source pins remained unchanged. There was one actual-output checker invocation and no retry.

This confirms deterministic label-supported panel construction. It does not establish CT readiness or model performance. The six contexts represent three physical fragments and two parent papyri: both evaluation fragments share Paris2, while calibration Paris1 already contains seen Frag4. Optical registration residuals and complete upstream training exposure remain unknown. Transport and author execution claims are verified as pinned source/recorded receipts, not independently replayed network or worker execution.
