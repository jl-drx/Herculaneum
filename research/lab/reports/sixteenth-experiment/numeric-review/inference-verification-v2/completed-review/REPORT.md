# Independent saved-output verification V2: PASS

One root-authorized corrected run passed without a retry. It verified all 18 maps and 450 raw tile-logit captures, exact evaluated masks, all 450 normalized C-order FP32 input SHA256 values, exact cutoff/boundary bits, and all reported confusion, ranking, operating, macro and transfer-screen values.

Maximum absolute blended-score error was 2.980232238769531e-07 against the frozen 2e-06 limit. The 9,128 arithmetic scalar comparisons had maximum absolute error 2.220446049250313e-16 against the frozen 1e-12 limit. No bitwise score equality is claimed.

Retained receipts passed six-worker, twelve-event, two-phase validation, including the cutoff barrier before evaluation, 150 calibration and 300 evaluation forwards, six constructions, six initializer strict loads and four candidate strict replacements (ten strict-load calls). These operation/state counts are validated source and worker receipt claims; this checker does not replay or independently authenticate the model.

The checker read 84 pinned saved arrays and made no model calls, Torch imports, checkpoint reads or network requests. All scientific input pins, root entry pins and frozen reviewer sources remained unchanged. Wall time was 1.770132609 seconds, child CPU time 2.171349 seconds, and peak child RSS 59,813,888 bytes.

V2 changes only worker output-dir lexical resolution under the exact pinned retrospective root context. All other command arguments and numeric kernels/checks are unchanged. The original checker and original STOP evidence remain sealed and unchanged. The retrospective context is not an independent launch-time trace; scientific interpretation and promotion decisions belong to root.

`PASS_RECEIPT.json` and `verification.json` hold exact file identities and detailed checks. `EXECUTION.json`, stdout and stderr preserve the single execution. `MANIFEST.json` binds the completed review evidence.
