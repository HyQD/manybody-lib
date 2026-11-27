"""Memory-aware accumulation and projection of owning rank-six sigma arrays."""

from __future__ import annotations

from functools import lru_cache
from itertools import permutations

import numpy as np
from numba import njit, prange, types


# Do not pay JIT or parallel dispatch overhead for small tensor spaces.
_FUSED_MIN_ELEMENTS = 65_536
_S3 = tuple(permutations(range(3)))
_TARGET_TYPE = types.Array(types.complex128, 6, "C")
_WEIGHT_TYPE = types.Array(types.complex128, 1, "C", readonly=True)


def _weighted_rank6(target, x0, x1, x2, x3, x4, x5, weights, count):
    """Fuse up to six weighted transpose views into one destination pass."""

    for ab in prange(target.shape[0] * target.shape[1]):
        a = ab // target.shape[1]
        b = ab % target.shape[1]
        for c in range(target.shape[2]):
            for i in range(target.shape[3]):
                for j in range(target.shape[4]):
                    for k in range(target.shape[5]):
                        value = target[a, b, c, i, j, k]
                        value += weights[0] * x0[a, b, c, i, j, k]
                        if count > 1:
                            value += weights[1] * x1[a, b, c, i, j, k]
                        if count > 2:
                            value += weights[2] * x2[a, b, c, i, j, k]
                        if count > 3:
                            value += weights[3] * x3[a, b, c, i, j, k]
                        if count > 4:
                            value += weights[4] * x4[a, b, c, i, j, k]
                        if count > 5:
                            value += weights[5] * x5[a, b, c, i, j, k]
                        target[a, b, c, i, j, k] = value


@lru_cache(maxsize=4)
def _compiled_weighted_rank6(dtype: np.dtype, strided_target: bool = False):
    # An arbitrary-strided readonly source signature handles every output
    # permutation without compiling a separate specialization per layout.
    source_dtype = types.complex128 if dtype == np.complex128 else types.float64
    source = types.Array(source_dtype, 6, "A", readonly=True)
    target = types.Array(types.complex128, 6, "A") if strided_target else _TARGET_TYPE
    signature = types.void(
        target, *((source,) * 6), _WEIGHT_TYPE, types.int64
    )
    return njit(signature, parallel=True, nogil=True)(_weighted_rank6)


def _numpy_accumulate(target, views, weights):
    """Use NumPy for small/unsupported arrays, bounding large scratch space."""

    def update(block, view, weight, scratch=None):
        if weight == 1:
            np.add(block, view, out=block)
        elif weight == -1:
            np.subtract(block, view, out=block)
        elif scratch is None:
            block += weight * view
        else:
            np.multiply(view, weight, out=scratch)
            np.add(block, scratch, out=block)

    if target.ndim != 6 or target.size < _FUSED_MIN_ELEMENTS:
        for view, weight in zip(views, weights, strict=True):
            update(target, view, weight)
        return

    # Keep a single virtual-pair tile in scratch, never a full rank-six
    # temporary. This also works for extended precision and unusual strides.
    scratch = np.empty(target.shape[2:], dtype=target.dtype)
    for a in range(target.shape[0]):
        for b in range(target.shape[1]):
            block = target[a, b]
            for view, weight in zip(views, weights, strict=True):
                update(block, view[a, b], weight, scratch)


def accumulate_permutations(
    target, source, contributions, *, allow_strided_target=False
):
    """Accumulate scaled permutations without full-size scaling temporaries.

    ``contributions`` is a sequence of (axes, coefficient) pairs, all with
    the same target. The source is not modified and must not alias the target.
    Opt into a strided destination only for known nonoverlapping basic slices
    of an owning array (as in the blocked GEMM evaluator).
    """

    if not contributions or not target.size:
        return
    views = [
        np.broadcast_to(source.transpose(axes), target.shape)
        for axes, _ in contributions
    ]
    weights = [weight for _, weight in contributions]

    if (
        target.ndim == source.ndim == 6
        and target.size >= _FUSED_MIN_ELEMENTS
        and target.dtype == np.complex128
        and (target.flags.c_contiguous or allow_strided_target)
        and target.flags.aligned
        and source.dtype in (np.dtype(np.complex128), np.dtype(np.float64))
        and source.flags.aligned
    ):
        fused = _compiled_weighted_rank6(source.dtype, not target.flags.c_contiguous)
        # The generated groups have at most six destinations. Chunk any
        # larger groups while still eliminating full-array temporaries.
        for start in range(0, len(views), 6):
            chunk = views[start : start + 6]
            count = len(chunk)
            padded = chunk + [chunk[0]] * (6 - count)
            coefficients = np.asarray(weights[start : start + 6], dtype=np.complex128)
            fused(target, *padded, coefficients, count)
    else:
        _numpy_accumulate(target, views, weights)


def _project_rank6(target):
    """Project disjoint occupied-index orbits on each virtual triple in place."""

    for ab in prange(target.shape[0] * target.shape[1]):
        a = ab // target.shape[1]
        b = ab % target.shape[1]
        for c in range(target.shape[2]):
            for i in range(target.shape[3]):
                for j in range(i, target.shape[4]):
                    for k in range(j, target.shape[5]):
                        # Gather every original value before overwriting any
                        # orbit member, including multiplicities when indices
                        # coincide. The orbits are disjoint.
                        x0 = target[a, b, c, i, j, k]
                        x1 = target[a, b, c, i, k, j]
                        x2 = target[a, b, c, j, i, k]
                        x3 = target[a, b, c, j, k, i]
                        x4 = target[a, b, c, k, i, j]
                        x5 = target[a, b, c, k, j, i]
                        mean = (x0 + x1 + x2 + x3 + x4 + x5) / 6.0
                        target[a, b, c, i, j, k] = x0 - mean
                        target[a, b, c, i, k, j] = x1 - mean
                        target[a, b, c, j, i, k] = x2 - mean
                        target[a, b, c, j, k, i] = x3 - mean
                        target[a, b, c, k, i, j] = x4 - mean
                        target[a, b, c, k, j, i] = x5 - mean


@lru_cache(maxsize=1)
def _compiled_projection():
    return njit(types.void(_TARGET_TYPE), parallel=True, nogil=True)(_project_rank6)


def project_triples_inplace(target: np.ndarray) -> None:
    """Remove the Matthews triples null component from an owning sigma array."""

    if not target.size:
        return
    if (
        target.size >= _FUSED_MIN_ELEMENTS
        and target.dtype == np.complex128
        and target.flags.c_contiguous
        and target.flags.aligned
    ):
        _compiled_projection()(target)
        return

    # Orbit-at-a-time NumPy fallback needs only a virtual-triple scratch
    # array, rather than several additional full triples tensors.
    scratch = np.empty(target.shape[:3], dtype=target.dtype)
    for i in range(target.shape[3]):
        for j in range(i, target.shape[4]):
            for k in range(j, target.shape[5]):
                indices = (i, j, k)
                orbit = tuple(tuple(indices[p] for p in perm) for perm in _S3)
                scratch.fill(0)
                for p, q, r in orbit:
                    scratch += target[:, :, :, p, q, r]
                scratch /= 6.0
                for p, q, r in dict.fromkeys(orbit):
                    target[:, :, :, p, q, r] -= scratch
