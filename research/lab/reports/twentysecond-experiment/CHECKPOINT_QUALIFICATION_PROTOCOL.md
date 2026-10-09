# Native checkpoint qualification: integrity before model use

This protocol qualifies one immutable public artifact; it does not compare ink accuracy, optimize a model, select a checkpoint or consume the reserved evaluation pixels. The user authorized continuing the prior recommendation. Root retains scientific judgment. Existing seed42 remains unchanged.

## Stage 1: complete artifact and explicit alias equality

The only artifact is `Nieuwlaar/ink9um-dense-native`, revision `b894a0493ab2291cc8dc3247020c0d7cb0967a58`, path `weights/dense_native_016000.safetensors`. Expected length is272,767,904 bytes and full SHA256 is`d3d95dc41dcb24ed76b84b5e12ca5dae25d7fb34edf1dfbb93e0186f91e6fc6d`. The source author documents these stripped weights as a bare safetensors state dictionary and supplies an MIT model-card declaration. No conversion to a new checkpoint is needed.

The previously verified prefix/header must match exactly: prefix SHA256`b94f491fdf06f1bf3f8592ca9c5c3f575fa9ca8464f76296559b94a752f1271f`, 66,192-byte header SHA256`b58a17c22694c0083156de30f02cb58838777eb4716fdb56e76a95d5b2962a24`, data begins at66,200. The saved strong ETag is a transport validator, not the expected whole-file SHA.

Freeze reviewed source, synthetic fixtures and a machine-readable plan before retrieval. Use one logical exact full-file Range GET, with manual HTTPS redirects on the two already observed hosts (`huggingface.co`, `us.aws.cdn.hf.co`); at most four attempts, two redirects, 280MiB observed response bodies, 180seconds absolute per request and240seconds external wall. Require exact206/Content-Range/Content-Length, matching strong ETag, identity encoding, no Transfer-Encoding, verified TLS and no fallback. Reserve budgets durably before opening; account for partial failures. Signed destination URLs stay transient. Redirect/error bodies are bounded to4,096bytes and retained only by hash/count. No automatic retries, credential use, URL logging or full-response exception dumps.

Hash while retaining one immutable local file. Do not interpret tensor values before size, complete SHA and retained header checks pass. The offline inspector must reject duplicate header keys, malformed shapes/dtypes, holes/overlaps, unexpected metadata/schema and offsets outside the exact contiguous tensor extent.

Required exact outcomes are508F32 named tensors, 68,175,426 named elements and272,701,704 tensor-data bytes. Check all named values for finiteness. For every244 factory alias-name pair, require bitwise equality, including signed zero; numeric approximate equality is insufficient. Failure must be reported before any model load. The conditional factory partition has264 groups, 34,546,498 unique parameters and138,185,992 bytes. The safetensors intervals remain separate storage even if their values agree; report that distinction.

A narrow independent reviewer checks the code/plan before execution and verifies the complete artifact through an independently structured parser or the installed safetensors library. Freeze independent acceptance rules before values are read. Exact schema, hash, finite-value and alias equality checks need no tuned floating tolerance. Do not widen a criterion after observing a failure.

Stage1 writes at most512MiB including the full artifact and all its evidence. Offline inspection is CPU-only, at most4GiB measured RSS, 8GiB address space, 120CPU seconds and180seconds external wall. Full phase22 output remains below768MiB. No model forward or optimizer update belongs to this stage.

## Stage 2: source-bound strict loading and synthetic compatibility

Proceed only after Stage1 passes and the exact historical native inference/model source has been acquired and reviewed. Before execution, freeze a supplemental test specifying authored model configuration, source construction, depth/window/normalization/padding/aggregation rules, synthetic inputs, measured quantities, acceptance tolerances, forward counts and resource limits. The same schema does not establish the same forward function or preprocessing.

Use exact508-key strict loading, with no missing/unexpected keys, no dtype conversion and no ignored alias disagreement. Verify resulting factory alias structure and all loaded parameter values. Preserve the supplied safetensors artifact; do not create an interchangeable `.pth` merely to work around an inference CLI. Offline runtime checks must block network access. No upstream installer or unreviewed entry point is invoked.

Any parity claim must name the two actual implementations compared and the coverage exercised. Where only source-equivalent extraction can run in this CPU environment, report that qualification rather than claiming historical Torch2.4.1/CUDA reproduction. Synthetic forwards can demonstrate construction, shape and numerical implementation agreement; they cannot demonstrate model accuracy, valid papyrus, decipherment or transfer to PHerc0800.

No paid GPU resource, scientific crop scoring, new threshold, gradient, teacher target, candidate selection or deployment follows automatically. Failures stop the affected stage and remain evidence. Root will use the completed qualification and independent benchmark provenance to define the next research experiment.
