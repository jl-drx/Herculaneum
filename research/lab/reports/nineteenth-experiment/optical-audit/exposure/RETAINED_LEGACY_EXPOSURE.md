# Retained legacy optical exposure audit

Frag6 is a **candidate with unresolved historical exposure**, not a confirmed independent evaluation set. This read-only local audit found no affirmative Frag6/PHerc51 training entry in the retained legacy sources examined. The omission cannot establish that no upstream model or teacher used it.

The scope was narrowed during work to legacy optical-fragment/Ink-ID evidence; the sibling upstream exposure audit owns detailed seed42, Russo, Nieuwlaar and teacher lineage. No network requests, scientific array reads, checkpoint decoding, inference or training occurred here. All outputs are additive under this directory.

## Identity and direct evidence

- The retained phase 15 `frag6-config.json` says exactly `{"materialthickness":150.0,"name":"PHerc51Cr4Fr8","version":7}`. `PHerc0051` is a normalized parent identifier inferred from this name, not an additional independently verified alias mapping.
- The counterpart Frag5 config names `PHerc1667Cr1Fr3`. Its PHerc1667 parent is already explicitly represented in seed42/Russo training membership. This excludes a *new-parent* claim for Frag5, without proving that the precise exposed fragment pixels were used.
- The phase 18 data audit retained Frag6 only as a deferred metadata candidate. It records unresolved annotation/render-depth and upstream/local exposure provenance. This audit does not change those image geometry or labeling judgments.

## What the retained EduceLab publication actually establishes

The complete retained EduceLab arXiv v4 HTML describes a dataset containing “16 scans of 6 fragments from 4 scrolls.” Its demonstrated Ink-ID result, however, uses **four Paris fragment surfaces**, split into eight top/bottom regions for cross-validation. A model trained on all four surfaces is used for hidden-layer predictions. The four named fragments are Paris 1 fragments 34 and 39, and Paris 2 fragments 47 and 143.

The dataset overview additionally lists `P.Herc.Paris. Objet 59` and `P.Herc. s.n. (Cass. 110)` and explicitly warns that “Some names are temporarily obfuscated.” Neither extra identity is linked to `PHerc51Cr4Fr8` by the inspected source. The phrase “six fragments” must **not** be equated with today's numbered Frag1–Frag6 directories.

The paper's training command refers to `fragments_all.txt` and Ink-ID commit `58b1519`; its prose specifies the four-fragment experiment. The separately retained Ink-ID repository tree at `a01075223745f0d5f1c114933f03b784eaff4b3a` does not include a `fragments_all.txt` content record. A source tree listing or generic dataset loader is not a full historical run manifest. This limits any exposure conclusion, even though the paper's described experiment does not name the PHerc51 parent.

Exact normalized excerpts and the original HTML SHA-256 are retained in `paper-excerpts.json`.

## Other local source checks

A retained 2 µm ResNet training metadata file names eight segment entries, with base-path labels `0139`, `343P`, `500p2a` and `9b`; none is explicitly `Frag6`, `PHerc51Cr4Fr8` or `PHerc0051`. This is an observation about that one metadata snapshot. Alias equivalence, checkpoint lineage, auxiliary pretraining, and complete historical execution are not established by it. In particular, no unknown numeric segment identifier is silently treated as proof of independence.

As a cross-check only, the retained phase 7 ledger derived from embedded seed42, seed43 and Russo-pair configurations lists 29 representations per model and parent keys `0139`, `0814`, `1667`, `Paris4`. Its six PHerc1667 segments establish parent-level exposure for Frag5. That ledger is not a complete teacher-family or historical optical-fragment exposure audit; detailed upstream conclusions belong to the sibling audit.

## Reproducible bounded search and decision

`source-pins.json` enumerates **15 text sources totaling 731,316 bytes**, their exact paths/SHA-256 hashes, and named-alias patterns/results. The only named Frag6/PHerc51 matches in that bounded set are its volume-package identity and our prior metadata audit notes. Source-file contents, rather than filenames, were searched. The helper also preserves selected paper excerpts without downloading or viewing scientific payloads.

Broader local filename/text discovery informed this curated evidence set, but it does not constitute an exhaustive search of every historical publication, release, training run or alias. No inference is made from missing search hits about uninspected or unavailable material.

**Decision:** keep Frag6 eligible for further provenance work, with upstream exposure marked **unknown / not affirmatively found in the inspected legacy records**. Admission as a fresh confirmatory parent still requires resolved aliases, annotation/input provenance and the candidate model's complete relevant training/initialization/teacher lineage. This audit neither admits Frag6 nor rejects it as contaminated.
