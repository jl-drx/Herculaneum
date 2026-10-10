# Post hoc explanation from the identified renderer source

The version-tagged Volume.cpp contains a specific interpolation-index defect: c10 mixes I(x0,y1,z0) with I(x1,y0,z0), where trilinear interpolation requires I(x1,y1,z0) for the second corner. The line generator calls interpolateAt at pt+normal×offset and includes both endpoints. This is a source-derived hypothesis, discovered after the first fixed CT comparison failed; not an extra member retroactively added to that comparison.

Run one explanatory replay on the already exposed189CT pairs, with the unchanged positive-normal, layer−32 coordinate recipe. For each sample, compute

    historical = round_even(correct_trilinear + dx*dy*(1-dz)*(I100-I110))

using the already retained native chunks. This algebra describes the observed source-index defect; no native intensity fitting, offset fitting, learned parameters, new data or execution of downloaded C++ is involved. Record distance to half-integer rounding boundaries; unknown historic cvRound/platform details remain a limitation. Test one asymmetric eight-corner cell with an exact hand-calculated value before using the retained CT. Preserve GPLv3 upstream source/license notices in renderer-source-v1; this research-only compatibility calculation changes no repository code or baseline sampler.

Primary report: all189raw residuals, number of exact uint16 matches, maximum residual, calibration/check labels inherited only for bookkeeping. Previously inspected check data are exposed: an exact replay is explanatory, not fresh confirmation, and does not change the original MAPPING_NOT_QUALIFIED verdict. No comparison to ink labels, optical accuracy claim, model change or novelty claim.

Resource extension for this source-informed offline follow-up: at most14additional decodes of the same cached chunks,56MiB extra decoded work, no new network requests. The original experiment and scalar verification used14+14decodes,112MiB; report all passes separately rather than silently treating repeated decompression as free. Disk/RSS and all financial limits remain unchanged. A further fresh-data confirmation, if warranted, needs its own prospective coordinate/byte contract.
