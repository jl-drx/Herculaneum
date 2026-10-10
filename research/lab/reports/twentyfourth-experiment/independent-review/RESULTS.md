# Independent paired depth-order verification

**PASS on the first verification attempt using the frozen checker.** This qualifies the paired execution and descriptive calculations, not a preferred input direction or a performance improvement.

All 300 model calls and normalized input pairs were verified: 150 unique original patches, each evaluated by both models after the exact normalized depth permutation. Original normalized fingerprints and reversed fingerprints matched their prospective contract. Adapter read order remained forward; only normalized tensor depth was reversed. The original 520,794 primary pixels were preserved, with 86,535 positive and 434,259 negative supplied-reference labels. No pixels were dropped to obtain agreement.

Independent checkpoint inspection matched all 1,016 named state hashes and 488 alias pairs across both models before and after inference. Seed42's original semantic identity and config/step matched its qualified artifact; native matched its qualified artifact and prior forward-run identity. Source, state and input identities remained unchanged. The checker performed no model construction or forward.

All raw ROC-AUC/AP, probability drift, reverse-minus-forward changes, differences of changes and separate parent means matched the author within `1e-12`. Reverse probability sums, weights and raw scores reconstructed bitwise exactly. Independent sigmoid maximum error was `5.960464477539063e-8`; analytic Hann error was `1.7881393432617188e-7`; retained forward-control reconstruction error was at most `2.980232238769531e-7`. These satisfy the arithmetic tolerance frozen before outputs.

| Parent | Model | Forward AUC | Reverse AUC | Forward AP | Reverse AP |
|---|---|---:|---:|---:|---:|
| Paris1 | seed42 | 0.784788 | 0.489187 | 0.240483 | 0.088780 |
| Paris1 | native | 0.630308 | 0.540326 | 0.203584 | 0.084710 |
| Paris2 | seed42 | 0.657642 | 0.477125 | 0.384286 | 0.237717 |
| Paris2 | native | 0.705446 | 0.564964 | 0.487602 | 0.294696 |

The following are reverse-minus-forward changes on every context:

| Context | seed42 AUC change | seed42 AP change | Native AUC change | Native AP change |
|---|---:|---:|---:|---:|
| frag3-context1 | −0.416318 | −0.186475 | −0.321797 | −0.256592 |
| frag3-context2 | −0.174884 | −0.116931 | +0.141831 | +0.018844 |
| frag1-context1 | −0.214990 | −0.048436 | −0.134575 | −0.029818 |
| frag1-context2 | −0.329490 | −0.298621 | −0.250360 | −0.435302 |
| frag2-context1 | −0.107067 | −0.072650 | −0.088429 | −0.086600 |
| frag2-context2 | −0.070522 | −0.166569 | −0.088562 | −0.219904 |

The complete independent metrics and drift statistics are in [verification-v1.json](verification-v1.json), SHA256 `70b58255c623f4cc7ce7ca65a92f45b92405e8c5d713ebb57b0229399fdb23a5`. Verification used 2.983 seconds wall, 4.085 seconds CPU and 351,408,128 bytes peak RSS. No verification retry, source change, model replay or tolerance adjustment occurred after native outputs.

Preparation history is retained, including the prospective normalization noncommutation example and the corrected test-class placement described in [ACCEPTANCE_RULES.md](ACCEPTANCE_RULES.md). No actual scientific execution failed during this verification.

These are counterfactual responses on six repeatedly reused development contexts. They do not identify the correct physical normal, justify direction selection, establish independent accuracy, or isolate a causal training effect. Scientific interpretation remains with root. No GPU or other paid resource was used.
