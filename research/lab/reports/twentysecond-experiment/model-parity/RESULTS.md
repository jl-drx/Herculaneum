# Native checkpoint: strict loading and synthetic CPU compatibility passed

The single authorized `run-v1` completed on its first attempt. Both the historical Villa factory (`3ea17f54…`) and current focused factory (`e0bbb8b4…`) strictly loaded the qualified native safetensors with all 508 keys, exact FP32 values, 264 actual storage groups, 244 duplicate-name pairs and 34,546,498 unique parameters. Source hashes and every loaded state value remained unchanged after inference. This producer execution result was subsequently confirmed by the separately frozen independent checker.

Both factories performed exactly eleven full model forwards: seven single-tile depth/direction cases, four tiles on the 192×192 synthetic canvas, and zero forwards on the all-zero skip case. There were two constructions and two strict loads, with no warmups, retries, gradients, training, scientific images or acquisition requests. All nonzero tiles received contiguous `[1,1,17,128,128]` FP32 inputs. The zero case remained unevaluated with zero output.

Independent verification found byte-identical historical/current normalized inputs, schedules, evaluated masks and outputs: **180,224 logits, 180,224 probabilities and 167,936 raw blended score values** matched exactly. Every frozen tolerance passed; there were zero cross-factory uint8 or author-decoded discrepancies. Independent sigmoid reconstruction differed by at most `5.960464477539063e-8`, and the separately constructed Hann map by at most `1.7881393432617188e-7`, within the checker’s frozen arithmetic tolerances. Reconstructed sums, weights and raw scores were exact. The verifier made no model calls.

| Measurement | Historical factory | Current factory |
|---|---:|---:|
| Worker wall time | 2.624 s | 3.059 s |
| Process CPU time | 8.941 s | 9.583 s |
| Measured peak RSS | 785,960,960 bytes | 826,798,080 bytes |
| Model modules | 393 | 395 |
| Full-model forwards | 11 | 11 |

The sequential controller completed in **8.465 seconds**. Both resolved to the same six-stage 2D backbone with features `[32,64,128,256,320,320]`, 32 fused input channels and depth 17. The current factory contains two additional `ModuleDict` containers named `guide_skip_projectors` and `guide_stage_tokenbooks`; their presence did not change the state schema or tested outputs. Each historical inactive architecture export was retrieved once during import, with zero constructor, function or instance-check calls. Network and `torch.load` counters stayed zero.

The common pre/post semantic value hash is `83698399e8467f728e6cde2eed998603b82f0fa298f14f2533dcdbabe08b9b58`. Independent verification is recorded in `../parity-review/verification-v1.json`, SHA256 `973cf4f841e7c46dff30c03398a44fe21eb91a63825951ac5867870737677da4`.

This qualifies exact strict loading and the named synthetic comparison under Python 3.12.14, NumPy 2.5.3, Torch 2.12.0+cpu and safetensors 0.8.0. Historical inference uses explicitly selected unchanged source definitions over in-memory arrays; the current comparator uses the retained baseline inference adapter. It does not reproduce historical Torch 2.4.1/CUDA, the complete old CLI, untested preprocessing branches or real-data accuracy. The author’s clipped uint8 decoding remains a distinct score, not a calibrated probability. Seed42 remains unchanged; no GPU was rented.

All frozen source, the 94 passing preparation checks, two corrected preparation failures, both workers’ pre/post state receipts and every saved inference array are retained. No actual-run failure or scientific retry occurred.
