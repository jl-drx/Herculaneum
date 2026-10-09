"""Independent raw/depth feature construction and historical batch fingerprints."""
import hashlib
import json
import numpy as np
from review_math import d4, require

ARMS = ('manual_double', 'manual_retired')


def features(image, arm):
    require(arm in ARMS and image.ndim == 5 and image.shape[1:3] == (1, 17)
            and image.dtype == np.float32 and np.isfinite(image).all(), 'Feature input contract differs')
    batch, _, depth, height, width = image.shape
    output = np.empty((batch, 3, depth, height, width), dtype=np.float32)
    for index in range(depth):
        output[:, 0, index] = image[:, 0, index]
        low, high = max(0, index - 1), min(depth - 1, index + 1)
        difference = image[:, 0, high] - image[:, 0, low]
        output[:, 1, index] = difference if index in (0, depth - 1) else difference * np.float32(.5)
        output[:, 2, index] = np.float32(index - 8) / np.float32(8)
    require(np.isfinite(output).all(), 'Nonfinite feature output')
    return output


def normalize_patch(image):
    output = np.array(image, dtype=np.float32, copy=True)
    require(output.size > 0 and np.isfinite(output).all(), 'Finite nonempty normalization input required')
    original_std = float(np.std(output.ravel()))
    low, high = (float(np.percentile(output.ravel(), p)) for p in (1., 99.))
    np.clip(output, low, high, out=output)
    median = float(np.median(output))
    scale = 1.4826 * float(np.median(np.abs(output - median)))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = next((value for value in (abs(original_std), abs(high - low) / 2.)
                      if np.isfinite(value) and value >= 1e-6), 1.)
    output -= median
    output /= scale
    np.nan_to_num(output, copy=False, nan=0., posinf=0., neginf=0.)
    return output


def fingerprint(values):
    digest = hashlib.sha256()
    dtypes = {np.dtype('float32'): 'torch.float32', np.dtype('bool'): 'torch.bool'}
    for name, value in sorted(values.items()):
        array = np.ascontiguousarray(value)
        require(array.dtype in dtypes, 'Unexpected fingerprint dtype')
        digest.update(json.dumps([name, list(array.shape), dtypes[array.dtype]],
                                 separators=(',', ':')).encode() + b'\0')
        digest.update(array.tobytes())
    return digest.hexdigest()


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def fixture_checks():
    row, column = np.indices((5, 5))
    ramp = np.empty((2, 1, 17, 5, 5), dtype=np.float32)
    for sample in range(2):
        for index in range(17):
            ramp[sample, 0, index] = sample * 1000 + index * index + row * 5 + column
    depth = features(ramp, 'manual_retired')
    control = features(ramp, 'manual_double')
    require(np.array_equal(depth[:, 0], ramp[:, 0]) and np.array_equal(control, depth),
            'Raw feature channel changed')
    for index in range(17):
        derivative = 1 if index == 0 else 31 if index == 16 else 2 * index
        require((depth[:, 1, index] == derivative).all(), 'Endpoint/central derivative mismatch')
        require((depth[:, 2, index] == (index - 8) / 8).all(), 'Signed nominal coordinate mismatch')
    for arm in ARMS:
        for code in range(8):
            require(np.array_equal(features(d4(ramp, code), arm), d4(features(ramp, arm), code)),
                    'D4/feature commutation mismatch')
    constant = np.full_like(ramp, 17.)
    require((features(constant, 'manual_retired')[:, 1] == 0).all(), 'Constant-depth difference mismatch')
    require((normalize_patch(np.full((2, 3, 3), 17, dtype=np.uint8)) == 0).all(),
            'Constant normalization mismatch')
    original = np.arange(5, dtype=np.uint8)
    expected = (np.array([.04, 1., 2., 3., 3.96], dtype=np.float32) - np.float32(2)) / np.float32(1.4826)
    actual = normalize_patch(original)
    require(np.array_equal(actual.view(np.uint32), expected.view(np.uint32)) and
            np.array_equal(original, np.arange(5, dtype=np.uint8)), 'MAD normalization/source immutability mismatch')
    return dict(status='PASS', matched_explicit_feature_cases=1, depth_difference_coordinate_cases=17,
                d4_commutation_cases=16, constant_difference_cases=1,
                normalization_cases=2, scientific_array_reads=0)
