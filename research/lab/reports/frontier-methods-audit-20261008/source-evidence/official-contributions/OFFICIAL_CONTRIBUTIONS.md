# Official contribution identities and author links

Source audit, 8 October 2026. The official primary source is the August 2026
Progress Prizes section of [the pinned Vesuvius Challenge winners page](https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/scrollprize.org/docs/15_winners.md).
Its retained body is `15_winners.md`, SHA256
`b228ffe51269410c6eeec88750568bdc0d7505f18f5505ab6946bfe519fa852e`,
115,259 bytes. The winners page supplies prize amounts, not experiment compute costs.

| Requested contributor | What the official page establishes | Author source located |
|---|---|---|
| Domenico Russo | $1,000; benchmarks all 14 released 9 micrometre ink checkpoints and retrains from dense teacher labels rather than sparse hand-labelled strips (lines 57-58). | Official entry has no repository or model link. Nieuwlaar's README calls the dense contribution KLAVIS / DomRusso2 / ink9um-dense; that is a secondary author statement, not independent identity proof. |
| Erwin / Eriwn Nieuwlaar | Official spelling **Erwin Nieuwlaar**; $500 for native 113 keV dense pseudo-labels and repair of a blank-output Scroll 1 checkpoint (lines 80-82). | [GitHub pin](https://github.com/Nieuwlaar/ink9um-dense-native/tree/217e7521670db49cde97872ea6a8bdfb80c449f1); README links [author model](https://huggingface.co/Nieuwlaar/ink9um-dense-native) and [repaired detector](https://huggingface.co/Nieuwlaar/resnet50_7.9um_scroll1_frags-repaired). |
| Paulo Sergio Camillo | Official name and @pscamillo; $1,000 for real depth-window training experiments (lines 64-66). | [GitHub pin](https://github.com/pscamillo/ink9-depth-window/tree/999b912c1ce92b8f5ab63ef2cc90ea017dc6ad04); detailed audit in CAMILLO_AUDIT.md. |
| The Socratic Method | This name is not present in the inspected pinned winners/community pages. The 9 micrometre surface-model prize entry credits **Ben Kyles / @hari_seldon and collaborators**, $2,500, and describes distillation from 2.4 micrometre predictions (lines 52-54). | Official entry links [ScrollFiesta](https://github.com/Hob3rMallow/scrollfiesta_public). No identity equivalence to The Socratic Method is established here. Detailed surface-method review belongs to the scientific-workflow agent. |

## Nieuwlaar README handoff

Pinned README retained as `nieuwlaar-README.md`, SHA256
`dd10ac91900277d3bbe3a5a40035e19f8bc2198e7536527bb8db72f9d97daf0e`,
29,334 bytes. Current main-ref metadata reports Last-Modified 26 August 2026.
The parent agent owns further code/model analysis; no model or code payloads
beyond the README were fetched here.

- Author reports a matched treatment/control costing about five hours on two
  cloud RTX 4090s, roughly $4 compute (README lines 305-307). This is an author
  report, not a quote for our environment or an independently reproduced cost.
- Benchmark main references are public model maps, not independent human ground
  truth. Teacher/reference-family overlap is explicitly acknowledged; vs-78-keV
  gains are called an upper bound. Better AUC is explicitly not yet readable text.
- Dated correction **26 August 2026** says a fixed raw-darkness veto removed
  50-65% of known ink; it is now report-only unless explicitly enabled. Avoid
  adopting the original sign assumption as a general physical ink test.

Every response body in this agent's research is counted in `sources.json`:
**303,543 bytes / 22 requests**, all metadata/documentation/source-code reads.
No figure assets, weights, scans, installs, publication or outreach occurred.
