"""Compare the reference and optimized closed-shell triples sigma vectors.

Run from the repository root, for example::

    OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 taskset -c 0,2,4,6 \
        python -m dev.benchmark_closed_shell_triples_sigma --nocc 5 --nvir 19

CPU affinity is selected externally so the script also works on other CPUs.
The timings exclude input construction, correctness checks, JIT compilation,
and warm-ups.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
import os
import statistics
import time

import numpy as np

from manybody_lib.configuration_interaction.closed_shell import cidt, cisdt


def _complex_random(rng, shape):
    result = rng.standard_normal(shape).astype(np.complex128)
    result.imag = rng.standard_normal(shape)
    return result


def _inputs(nocc, nvir):
    rng = np.random.default_rng(20261127)
    norb = nocc + nvir
    f = _complex_random(rng, (norb, norb))
    f = 0.5 * (f + f.T.conj())
    u = _complex_random(rng, (norb,) * 4)
    u = 0.25 * (
        u
        + u.transpose(1, 0, 3, 2)
        + u.transpose(2, 3, 0, 1).conj()
        + u.transpose(3, 2, 1, 0).conj()
    )
    c1 = _complex_random(rng, (nvir, nocc))
    c2 = _complex_random(rng, (nvir,) * 2 + (nocc,) * 2)
    c2 = 0.5 * (c2 + c2.transpose(1, 0, 3, 2))
    c3 = _complex_random(rng, (nvir,) * 3 + (nocc,) * 3)
    c3 = cisdt.project_physical_triples(cisdt.symmetrize_triples_columns(c3))
    common = (-4.2, f, u, 0.7 - 0.2j)
    slices = (slice(0, nocc), slice(nocc, norb))
    return {
        "cidt": (*common, c2, c3, *slices),
        "cisdt": (*common, c1, c2, c3, *slices),
    }


def _benchmark(module, args, repeats, warmups, gemm_only, gemm_block_size):
    names = (
        "reference", "optimized", "symmetry_optimized", "combined_integral",
        "optimized_accumulation", "gemm", "blocked_gemm",
    )
    point4 = dict(
        assume_symmetry=True, combine_integrals=True, optimize_accumulation=True
    )
    functions = (
        module.compute_sigma,
        module.compute_sigma_optimized,
        partial(module.compute_sigma_optimized, assume_symmetry=True),
        partial(
            module.compute_sigma_optimized,
            assume_symmetry=True,
            combine_integrals=True,
        ),
        partial(module.compute_sigma_optimized, **point4),
        partial(module.compute_sigma_optimized, **point4, use_gemm=True),
        partial(
            module.compute_sigma_optimized, **point4,
            use_gemm=True, gemm_block_size=gemm_block_size,
        ),
    )
    if gemm_only:
        names, functions = names[-3:], functions[-3:]

    # Verify numerical equivalence on the exact benchmark inputs.
    expected = module.compute_sigma(*args)
    errors = {}
    for name, function in zip(names, functions, strict=True):
        if name == "reference":
            continue
        actual = function(*args)
        relative_errors = []
        for original, optimized in zip(expected, actual, strict=True):
            np.testing.assert_allclose(optimized, original, rtol=1e-11, atol=1e-10)
            relative_errors.append(
                float(np.max(np.abs(optimized - original)))
                / max(1.0, float(np.max(np.abs(original))))
            )
        errors[name] = relative_errors
        del actual, original, optimized
    del expected

    for _ in range(warmups):
        for function in functions:
            function(*args)

    timings = [[] for _ in functions]
    for repeat in range(repeats):
        # Rotate the order to reduce systematic warm-cache/frequency bias.
        order = tuple((repeat + i) % len(functions) for i in range(len(functions)))
        for index in order:
            started = time.perf_counter()
            result = functions[index](*args)
            timings[index].append(time.perf_counter() - started)
            del result

    medians = {name: statistics.median(values) for name, values in zip(names, timings)}
    return {
        "seconds": dict(zip(names, timings, strict=True)),
        "median_seconds": medians,
        "speedup_baseline": names[0],
        "speedups": {
            name: medians[names[0]] / value for name, value in medians.items()
        },
        "max_relative_errors_vs_reference": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nocc", type=int, required=True)
    parser.add_argument("--nvir", type=int, required=True)
    parser.add_argument("--method", choices=("both", "cidt", "cisdt"), default="both")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument(
        "--gemm-only", action="store_true", help="time point 4 and both GEMM variants"
    )
    parser.add_argument("--gemm-block-size", type=int, default=4)
    args = parser.parse_args()
    if (
        min(args.nocc, args.nvir, args.repeats, args.gemm_block_size) < 1
        or args.warmups < 0
    ):
        parser.error("orbital dimensions, block size, and repeats must be positive")

    inputs = _inputs(args.nocc, args.nvir)
    methods = ("cidt", "cisdt") if args.method == "both" else (args.method,)
    modules = {"cidt": cidt, "cisdt": cisdt}
    print(
        json.dumps(
            {
                "nocc": args.nocc,
                "nvir": args.nvir,
                "affinity": sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None,
                "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
                "numba_num_threads": os.environ.get("NUMBA_NUM_THREADS"),
                "numpy_version": np.__version__,
                "gemm_block_size": args.gemm_block_size,
            }
        ),
        flush=True,
    )
    for method in methods:
        result = _benchmark(
            modules[method], inputs[method], args.repeats, args.warmups,
            args.gemm_only, args.gemm_block_size,
        )
        print(json.dumps({"method": method, **result}), flush=True)


if __name__ == "__main__":
    main()
