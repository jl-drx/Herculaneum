# Metadata-only future CT cost estimate

No new CT access or pixel decoding was performed. This note is planning arithmetic, not an acquisition budget or authorization. Exact inputs and calculations are retained in `ct-cost-estimate.json`.

The successful eleventh acquisition retained 1,425,096,672 raw range bytes in 255 HTTP206 attempts. Pixel bands used 1,425,088,512 bytes; 153 header/IFD requests used 8,160 bytes. It fetched 51 planes 7..57 and two row bands across the full 6064-pixel width, Y[0,1152) and [8064,9216). Cropping X locally left 270,729,216 useful native-crop bytes: transfer was 5.2639 times useful pixels. Overlapping or adjacent Y spans are merged before requesting.

Assuming two disjoint 1152-row contexts on each of Frag1/2/3, their declared widths 6330/9506/5249 imply 4,955,143,680 pixel-band bytes (4.615 GiB), before headers or derived outputs. Actual contexts are not yet selected and TIFF layout/range/ETag support remains unverified. Frag2's 21,901,824-byte band exceeds the old 14 MiB per-range cap. Splitting that band's Y extent can preserve the cap at the cost of more requests; this needs a separate reviewed CT plan. The label-only stage fetches none of these bytes.
