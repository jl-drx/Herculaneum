# Pre-execution clarifications

2026-10-08 UTC, before numerical producer execution. These disambiguate the frozen design without fitting data.

Infrared order is: decode published registered image; take the exact1152x1152 native crop; convert crop to gray; apply the fixed Gaussian within that crop with reflect boundaries; then compute3x3 block means. Primary pixels are at least96native pixels from the crop boundary, beyond the64pixel Gaussian radius, so this boundary convention does not change primary-support association results. Display pixels near the border may depend on the explicit reflect convention.

PPM signed normal agreement is dot(cross(du,dv),n)/(norm(cross(du,dv))*norm(n)); either zero or nonfinite norm yields null. Unit tangent agreement likewise divides by both tangent norms and yields null when undefined. No anatomy is inferred from the sign.

Native endpoint wording in the design refers to exact values0and65535, and pooled0and255, not each plane's observed extrema. Labels use their declared binary encoding and may not be inferred from positive prevalence.
