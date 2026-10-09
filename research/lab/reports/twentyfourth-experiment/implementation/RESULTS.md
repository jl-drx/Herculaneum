# Paired normalized-depth reversal: execution and verification passed

The first authorized attempt completed with two sequential workers, two constructions, two strict loads and **300 full-model forwards**, exactly 150 per checkpoint. The adapter read and normalized the original forward patches. Each call verified that original tensor’s cached hash, flipped only normalized depth axis 2 to order 16…0, verified the prospective reversed hash, and checked both tensors for mutation afterward. Raw reversed-volume normalization was not used for inference.

All original primary support remained evaluated: **520,794 pixels**, comprising 86,535 supplied positives and 434,259 supplied negatives across six reused contexts. Forward controls were reused from the sealed seed42 and native runs. There were no new forward-control model calls, warmups, gradients, training steps, threshold classifications, quantized metrics or direction selection.

Raw metric changes are reverse minus forward. The final column is the native change minus the seed42 change; it is a descriptive contrast, not a model-selection rule.

| Context | Seed42 ΔAUC / ΔAP | Native ΔAUC / ΔAP | Difference of changes, AUC / AP |
|---|---:|---:|---:|
| frag3-context1 | −0.416318 / −0.186475 | −0.321797 / −0.256592 | +0.094521 / −0.070117 |
| frag3-context2 | −0.174884 / −0.116931 | +0.141831 / +0.018844 | +0.316715 / +0.135775 |
| frag1-context1 | −0.214990 / −0.048436 | −0.134575 / −0.029818 | +0.080415 / +0.018618 |
| frag1-context2 | −0.329490 / −0.298621 | −0.250360 / −0.435302 | +0.079130 / −0.136681 |
| frag2-context1 | −0.107067 / −0.072650 | −0.088429 / −0.086600 | +0.018638 / −0.013951 |
| frag2-context2 | −0.070522 / −0.166569 | −0.088562 / −0.219904 | −0.018041 / −0.053335 |

The separate equal-context parent summaries preserve all four checkpoint/direction combinations:

| Parent / checkpoint | Forward AUC / AP | Reverse AUC / AP | Reverse−forward AUC / AP |
|---|---:|---:|---:|
| Paris1 / seed42 | 0.784788 / 0.240483 | 0.489187 / 0.088780 | −0.295601 / −0.151703 |
| Paris1 / native | 0.630308 / 0.203584 | 0.540326 / 0.084710 | −0.089983 / −0.118874 |
| Paris2 / seed42 | 0.657642 / 0.384286 | 0.477125 / 0.237717 | −0.180517 / −0.146569 |
| Paris2 / native | 0.705446 / 0.487602 | 0.564964 / 0.294696 | −0.140481 / −0.192906 |

The parent differences of changes are +0.205618 AUC / +0.032829 AP for Paris1 and +0.040036 AUC / −0.046337 AP for Paris2. Full metrics for every context, checkpoint and direction, plus signed, absolute, linear-p95 and maximum probability drift, remain in `analysis-v1/metrics.json`. Mean absolute reverse-versus-forward probability drift ranges from 0.094140 to 0.158859 for seed42 and from 0.108375 to 0.225311 for native.

Each model retained exactly 508 FP32 named state tensors, 244 alias pairs, 264 actual storage groups and 34,546,498 unique parameters. Every loaded byte, source pin and model state remained unchanged. Seed42 used one safe `weights_only=True` CPU deserialization of its pinned original `model` container; native used one safetensors decode. Unreviewed `torch.load` and network counters remained zero.

Seed42’s original semantic hash stayed `21c186000b9b3093835e185342668acd0d86147a719c37513e937924b77ea228`; its per-name-value-hash receipt is `e9c9f52e7a54001c0d49bba5b737bee2421c2454c2f4a62b8fc9afc31cf59591`. These are different hash constructions, not different model states. Native’s per-name-value-hash receipt stayed `83698399e8467f728e6cde2eed998603b82f0fa298f14f2533dcdbabe08b9b58`, matching its forward control.

| Runtime measurement | Seed42 | Native |
|---|---:|---:|
| Worker wall time | 14.399 s | 14.464 s |
| Process CPU time | 50.959 s | 51.325 s |
| Measured peak RSS | 709,980,160 bytes | 837,009,408 bytes |
| Full-model forwards | 150 | 150 |

The controller completed in **31.421 seconds**, with both workers within the frozen limits. All 49 preparation checks passed on their first attempt. No scientific retry occurred.

The frozen independent checker passed on its first attempt. It verified all 300 original/reversed input pairs and calls, both models’ 1,016 named state values and 488 aliases before and after, the original support, and unchanged inputs/sources. Every metric, drift statistic and change matched within the fixed `1e-12` tolerance; reversed sums, weights and raw scores reconstructed exactly. The checker replayed no model and changed no source or tolerance after observing outputs.

Receipts:

- Seed42 result SHA256: `df04cc4dbb5da815c3fa2a767edc895cbe1d5ea50f15b1fd4f88eb861747ffd0`.
- Native result SHA256: `a35cc5927cfd1811627232034aeef0b7e56221e70504550d3ea3075c6485e25d`.
- Metrics SHA256: `8c424831c0bbe78aef529b27278c3609639046e6d263c5385af4bc1cd37a921c`.
- Independent verification: `../independent-review/verification-v1.json`, SHA256 `70b58255c623f4cc7ce7ca65a92f45b92405e8c5d713ebb57b0229399fdb23a5`.

These results describe responses to an exact normalized-depth permutation on repeatedly reused development inputs. They do not establish the correct physical orientation, independent accuracy, a causal training benefit or a production correction. No model or direction decision is encoded. All earlier evidence remains unchanged; there was no acquisition request, checkpoint copy or GPU rental.
