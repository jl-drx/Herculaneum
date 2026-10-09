# Independent Stage B technical review contract

This review concerns one bounded PPM geometry request only. It will use public source text, the retained Stage A structural JSON, and synthetic fixtures. No real numerical PPM payload, CT pixel, optical image, annotation, model or network service will be accessed by the reviewer before root's separate acquisition GO. Earlier evidence remains immutable.

The independent span oracle uses width 6,205, height 8,853 and payload offset 73, as retained in Stage A. For each row-major lattice index `(i,j)` in `0..8`, its center is `y=((i+1)*(H-1))//10`, `x=((j+1)*(W-1))//10`. Three row spans cover the full 3×3 neighborhood. Each span contains three six-double records, 144 bytes; the fixed schedule has 243 parts, 729 records and 34,992 useful bytes. No overlap, replacement, score or coverage-driven adaptation is permitted.

Transport checks will cover exact source URL, strong ETag, object length, inclusive ranges, 206 status, identity encoding, singleton framing headers, one multipart boundary, length-based binary framing, exact part membership and count, complete termination, and no redirects, fallback or retry. Fake transports will test durable reservation, partial retention and absolute deadlines without making network calls. The full 222,640-byte body reservation must include every application-level read, including any excess-byte probe. The response-body ceiling is not a total wire-traffic claim.

The independent numerical fixture uses a known plane:

`p(x,y) = (0.2*x, 0.1*y, 500 + 0.01*x + 0.02*y)`

Its derivatives are `(0.2,0,0.01)` and `(0,0.1,0.02)`. Their cross product is `(-0.001,-0.004,0.02)`; the synthetic stored normal is this vector divided by its Euclidean norm. Fixtures include little- and big-endian encodings, exact six-zero sentinels, nonfinite fields, nonunit normals, all-unmapped data and byte-palindromic normal doubles for a deliberately ambiguous interpretation. Conditional byte-order admission must require exactly one interpretation with mapped data and every nonsentinel finite/unit-normal within the frozen `1e-5` threshold.

Sentinels remain at their original scheduled sites, including incomplete fixed native neighborhoods. The nine later sites are the original lattice centers with i,j in {1,4,7}; fewer than six complete neighborhoods blocks native comparison without replacement. No Stage C execution is authorized by a Stage B technical pass.

Derivative, area/Jacobian, tangent-normal and candidate-volume bounds are consistency diagnostics. They do not authenticate a source UUID, physical distance, inward/outward orientation, anatomical layer, ink depth or label alignment. Any geometry-only technical GO must preserve those limitations and all unresolved Stage A render fields.

The reviewer will independently pin source and plan identities before fixtures, preserve failures, and report exact tested versions. Root owns scientific interpretation and any live acquisition decision. Resource checks are bounded by the prescribed 256 MiB measured RSS, 1 GiB address space and 32 MiB artifacts; the executable request has a 60-second absolute deadline and the external controller an 1,800-second ceiling.
