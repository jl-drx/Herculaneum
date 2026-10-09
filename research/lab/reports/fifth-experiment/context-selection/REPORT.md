# Post hoc surrounding-context geometry outcome

**Stop: no fully geometry-valid 768 × 768 crop meets the frozen rule.** No new acquisition or inference was performed, and no acquisition plan was emitted.

The fixed follow-up uses segment `20251028220955`, native source shape `[31, 2100, 2100]`, at 8.64 µm. A candidate must have an origin on the 128-pixel grid, remain inside the source, contain the entire fourth-experiment bbox `[1024, 384, 384, 384]`, and place both fixed points `(1278, 504)` and `(1100, 628)` at least 128 pixels from each crop edge. The entire 768-square geometry mask must be valid. Ranking is maximum valid 128-margin interior count, minimum distance from the crop center to the fourth field center, then ascending Y and X. Failure does not relax these requirements.

The primary selector checked all 121 inside-source grid origins. Nine satisfy containment and point margins; none satisfies full geometry validity. Selection took 0.264 seconds, including verification of 19 frozen inputs before and after geometry evaluation. Only geometry masks, XYZ coordinates and metadata were evaluated. Previously observed source chunks were hashed solely to verify cache identities; intensity values and detector maps were not inspected.

| Source origin X,Y | Valid pixels / 589,824 | Invalid pixels | Valid interior / 262,144 |
| --- | ---: | ---: | ---: |
| 640,0 | 417,664 | 172,160 | 230,960 |
| 768,0 | 426,224 | 163,600 | 239,520 |
| 896,0 | 399,664 | 190,160 | 215,392 |
| 640,128 | 504,208 | 85,616 | 256,176 |
| 768,128 | 512,768 | 77,056 | 261,056 |
| 896,128 | 486,208 | 103,616 | 245,360 |
| 640,256 | 558,512 | 31,312 | 262,144 |
| 768,256 | 567,072 | 22,752 | 262,144 |
| 896,256 | 540,512 | 49,312 | 260,208 |

The separately declared **failure diagnostic** identifies `[768, 256, 768, 768]` as the best-covered rejected box. Its central 512-square is fully valid, but the minimum global geometry-boundary clearance within that interior is only 41.4005 pixels. Fully valid interior pixels therefore do not imply that every model tile contributing to them has valid complete support. This box is not selected for acquisition.

For that diagnostic box, valid neighboring XYZ displacements range from 0.9631 to 1.0354 base voxels horizontally and 0.9649 to 1.0476 vertically. All valid coordinates and tested differences are finite. No zero neighbor displacement or central finite-difference cross-product norm at or below `1e-8` was found. These local continuity checks do not reconstruct archived renderer normals, establish written-surface placement, or resolve complete through-depth interpolation and accumulation support.

The unchanged rule was specified by the root agent before any candidate inspection. A secondary reviewer inspected candidate geometry before the on-disk methodology was frozen; the primary selector received no secondary candidate outcomes before that freeze. The methodology records this timing plainly. This is a post hoc surrounding-context proposal based on the fourth field and previously identified response locations. It is not held-out validation or a new independently selected coverage test.

`result/coverage-outcome.json` and `result/candidate-scores.json` preserve the actual outcome. `prior-observed-chunk-cache.json` records all nine previously observed HTTP-200 chunks and their verified hashes. A qualifying 768-square would require 36 raw chunks (18,284,544 bytes), with nine reusable and at most 27 new chunks (13,713,408 bytes). The original fourth adapter is hard-coded to 384-square and a 6 MiB response cap; any future 768 acquisition needs a separately frozen cache-aware adapter and the native CT mapping gate. Those operational prerequisites do not override this failed geometry gate.

Any alternative crop size or partial-geometry inference design requires a separately declared methodology. No ink, letterform, transcription or translation is established by this geometry outcome.
