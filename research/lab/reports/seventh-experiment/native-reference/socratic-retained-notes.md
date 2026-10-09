Retained Socratic evidence for seventh-experiment native reference

Audit scope: local retained source documents and metadata only, reviewed on 2026-10-08. This review made zero network requests, downloaded no model/dataset/label objects, loaded no models and ran no inference. All statements about data registration, training and scores below are attributed to the pinned author documents; this review did not independently verify their underlying arrays or transforms. Only this new notes file was written.

The Socratic retained sources do not supply an ink-labelled native reference panel. They describe **3D papyrus-surface segmentation**. Their human and fine-teacher labels supervise sheet geometry. Their cross-resolution records can identify acquisition and registration leads for a later ink-panel audit, but cannot establish ink presence, ink authenticity or ink accuracy.

The principal source is Hob3rMallow/socratic_method at immutable commit `7f86cbf8d38155b37fdd39449c6775ec50344b6c`. The retained community inference README is yupengtang/vesuvius-scroll-inference at `893a075ef59b0e62f546b0f34ed6f6b7a27950af`. Retained bodies listed at the end matched every corresponding recorded SHA-256 and byte count; their source metadata records HTTP 200 and complete bodies. The retained recursive source tree reports `truncated: false`.

Label semantics and supervision

- The model-card template calls F0 a “3D papyrus-surface segmentation student,” with one CT channel and two-class voxel logits; class-1 softmax is the foreground surface probability. It is not a 2D ink-probability model. The community README likewise calls all three compared models papyrus surface detectors, and distinguishes HercUNet's medial sheet crest from M7's sheet face.
- The executable recipe describes `stitched-wide15-antialias-soft-native-plus-hard-human-v1`: 16,200 total rows, 15,432 train and 768 telemetry-validation rows; 10,072 native soft rows and 6,128 hard human passthrough rows across the corpus. These are row counts, not unique physical samples or independent acquisition counts.
- Human record families include PHerc0139 w023–w032, w037–w041 and w047–w059 at 2.399 µm; PHerc0343P at 2.215 µm; PHerc1667 at 2.399 µm; and PHercMANBp. The recipe distinguishes `registered-real` from `same-scan-pyramid`. Those names are scope/lineage claims, not transforms or error measurements.
- Frozen v14p2 benchmark: 689 validation rows, 192³ CT patches, PHerc0814 256 and PHerc1451 433. The benchmark README says targets come from the official ~2.4 µm fine teacher, registered to ~9 µm scans, hardened at q ≥ 0.5, with label 2 ignored. It claims 442 rows are byte-for-byte crops of public native acquisitions at recorded origins, while 247 are 4× downsampled fine scans. The synthetic rows therefore cannot count as native coarse acquisition validation.
- PHerc0500P2's 320 human-labelled rows are a surface benchmark from `voxel_all_sources_patches_5856`. The author says released M7 trained on these labels; F0 initialises from and is anchored to M7, while HercUNet starts from M7. The source explicitly treats these scores as absolute only. They are not an unexposed human ink holdout.

Native/fine acquisition leads and limits

The strongest new native/fine lead is **PHerc1203**. The registry labels it F0-unseen and says it is “the only eligible scroll with a co-registered 2.403 um scan.” It supplies the recorded native coarse CT URI:

` s3://vesuvius-challenge-open-data/PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr `

The registry also records native shape [18977, 6844, 6844] and umbilicus [3462, 3379], attributed to project census/grid manifests. An umbilicus is not a cross-acquisition registration. No exact fine-volume URI, fine/coarse scan identities in a paired manifest, affine/deformation field, coordinate convention, valid-overlap mask, landmark residuals or measured local registration uncertainty for PHerc1203 are present in the retained reviewed documents. Consequently this is a metadata lead, not a ready paired reference or an independently verified native/fine registration. This review did not access the coarse URI.

Other coarse-volume leads are present, with materially different provenance:

| Physical scroll | Retained coarse CT URI | Registry source | F0 relationship |
|---|---|---|---|
| PHerc0139 | `s3://vesuvius-challenge-open-data/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr` | recorded | training and project rehearsal |
| PHerc0814 | `s3://vesuvius-challenge-open-data/PHerc0814/volumes/20250804134230-9.362um-1.2m-113keV-masked.zarr` | pattern | validation and release selection |
| PHerc1451 | `s3://vesuvius-challenge-open-data/PHerc1451/volumes/20250521151225-8.640um-1.2m-116keV-masked.zarr` | recorded | validation and release selection |
| PHerc1667 | `https://data.aws.ash2txt.org/samples/PHerc1667/volumes/20231117161658-7.910um-53keV-masked.zarr` | legacy | training |

Registry `recorded` means an exact URI appeared in a project manifest/config/script; `pattern` means constructed from the documented naming pattern and an upstream timestamp, “not seen verbatim”; `legacy` means an older EduceLab acquisition outside the cited open-data bucket. The registry purpose explicitly says fields are advisory, opened-volume metadata wins, and voxel pitch is “an operator assertion, not a registration measurement.” Public accessibility, dimensions and scale transforms of these URLs were not tested here. A path bearing `predictions/surfaces` identifies a model prediction, not native CT, ink labels or human truth.

The recipe lists claimed genuine cross-acquisition families (e.g. PHerc0139 2.399→9.362 µm; PHerc0332/PHerc1667 2.399→7.91 µm; PHerc0814 2.399→9.362 µm; PHerc0009B 8.64 µm; PHerc0841 9.366 µm; PHerc1451 8.64 µm) alongside synthetic same-scan pyramid families. It gives aggregate manifest hashes, not these paired manifests' contents. The 44,773,698-byte training manifest and 20,344,063-byte frozen validation manifest were not retained as data in this folder and were not fetched during this review. Claimed byte equality of benchmark CT constrains acquisition fidelity; it does not establish target registration precision or an independent ink label.

The retained tree identifies follow-up source locations `docs/data.md`, `src/crossres_pred/voxel/official_corpus.py`, `manual_labels.py`, `registered_mirror.py`, `registration.py`, `registration_evidence.py` and `registration_fit.py`. Their tree metadata provides paths and Git blob IDs, not code contents or actual pair registration evidence. Those files were not fetched or executed here.

Teacher lineage, exposure and interpretation

F0 uses the released ~9.362 µm M7 surface model for initialisation and a frozen M7 KL anchor (weight 0.5), alongside soft fine-teacher CE and Dice (weights 1 each). The model-card template identifies the fine teacher as the Villa fine teacher. The retained recipe pins M7 input checkpoint SHA-256 `17465b77591b794638e671f1a9f79c4cf1e79821f302e6fc235e3725e5da7d7e`; no fine-teacher binary identity or complete upstream fine-teacher physical training roster is established by these retained documents. Inference uses a single F0 student, with no explicit teacher blend; this does not remove the training lineage.

The twelve F0 training scrolls are PHerc0009B, PHerc0139, PHerc0332, PHerc0343P, PHerc0841, PHerc1299, PHerc1667, PHercMAN5, PHercMANB, PHercMANBp, PHercParis3 and PHercParis4. PHerc0814 and PHerc1451 are telemetry-validation and frozen benchmark scrolls. The recipe uses their frozen metrics for checkpoint selection and the operating-point choice, so “not trained on” does not mean unopened or independent of all model selection. The author's PHerc1447 eligibility/frontier gate is another development exposure. PHerc0800 is called F0-unseen in the registry, but appears in the later author random-cube geometry audit; neither designation supplies human ink truth or proves absence from M7/fine-teacher ancestry. Pair aliases from one scroll must remain one physical family when judging heldout status.

Licences and research/prize scope

The pinned repository LICENSE is Apache-2.0; CURRENT_MODEL.json declares Apache-2.0 for F0 and the model-card template explicitly releases the weights under it. Template front matter uses `license: other`, so the prose and repository licence are the precise retained statements. The source explicitly says upstream Vesuvius data, M7 and Villa fine-teacher artifacts keep their own terms: Apache-2.0 does not assign a licence to those underlying CT/label objects. The community inference wrapper says MIT for its own code and separate licences for the models. Exact target dataset/label licence and version remain unresolved by these retained sources.

The retained upstream `src/crossres_pred/policy.py` supports distinct `research` and `prize-safe` profiles and requires prize-safe configurations to forbid PHerc1203 higher-resolution data. This is evidence of the upstream project's prize eligibility separation; it is not a fresh session permission request or a licence finding. The user's research-first goal can track PHerc1203 as a research lead, while any prize claim must preserve that provenance and independently check current official rules.

Practical conclusion: no newly qualifying native ink-labelled panel is established here. PHerc1203 is a native/fine metadata lead requiring exact fine URL, measured registration and separate ink-label provenance before any panel can be reviewed. The Socratic human surface rows and registered fine-teacher surface targets are useful geometry evidence but cannot serve as human ink ground truth or independent ink accuracy validation.

Verified retained source inventory

All paths below are under `reports/frontier-methods-audit-20261008/raw-sources/socratic/` in the local lab. Matching `.metadata.json` files retain pinned source URLs and original HTTP/hash records. The source bodies and metadata were rechecked for unchanged hashes after writing these notes.

| Retained source body | Bytes | SHA-256 |
|---|---:|---|
| socratic-readme-08b115728d6d.body | 16713 | `664ffef2fb507e678c5c52a3f0c4a61ea68ee87f2c0fd7922c59c076cc8e2aac` |
| socratic-f0-recipe-38637e386640.body | 15909 | `f44b186e5dc05eba0494e27f0d92344a74770ce11ed745f307fc518e709b3364` |
| socratic-benchmark-504195347fc0.body | 21133 | `ecb22b4a3562c99c7f6451ff0fc153f94e63a7536974d80eadca047cd11fb752` |
| socratic-scroll-registry-9cd780bd7633.body | 26077 | `53bcc53e149b3e0811a02a9c17c045bdf4455640d8cca59c8e7001a151eda848` |
| socratic-model-card-template-2547de67a6ee.body | 6634 | `fc6be3e87752931c9a6df574cc6f61d5a1a932ff5deeebd950952d4e300be46a` |
| socratic-current-model-cb6e0fd6e598.body | 2856 | `76dbc01cfff712ba91269ebabc6497f3ee814f49633d358cfc0245547c0e7880` |
| socratic-license-542b41eb41f6.body | 11358 | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| socratic-runtime-policy-433d3b2f80b3.body | 4953 | `5e478d91ea324ebb26801f684332ab914f37b2a33ca407a16341c27c8ce0c81f` |
| socratic-lead-readme-6eb51c69d337.body | 10571 | `34df5222ed3dd58bbb3366b5c458edffbb29fd4c4af784288d79132a96df938a` |
| socratic-tree-8dd84e4f76b5.body | 270624 | `12fa19d50695f6fa5e28197d6ee59528b18ce9564891123a2ebe8012ce4d03b4` |

These local content hashes establish the retained documents reviewed, not the truth of author claims or integrity of unacquired data/model objects.
