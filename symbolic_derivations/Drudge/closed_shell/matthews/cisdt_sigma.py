"""Derive closed-shell CISDT sigma vectors in the Matthews triples manifold.

Run with

    conda run --no-capture-output -n drudge python cisdt_sigma.py

The equations printed to stdout are metric-resolved.  Diagnostic term counts
and timings are written to stderr.
"""

from common import (
    DerivationOptions,
    build_model,
    derive_sigma_vectors,
    make_spark_context,
    print_align,
    print_conventions,
)


def main() -> None:
    options = DerivationOptions.from_environment()
    context = make_spark_context(options, "closed-shell-matthews-cisdt-sigma")
    try:
        model = build_model(context, options)
        sigma, _raw_covariant = derive_sigma_vectors(model)
        print_conventions()
        print("% Metric-resolved right sigma vector")
        print_align(
            model.dr,
            (
                (r"\sigma_{0}", sigma[0]),
                (r"\sigma^{a}_{i}", sigma[1]),
                (r"\sigma^{ab}_{ij}", sigma[2]),
                (r"\sigma^{abc}_{ijk}", sigma[3]),
            ),
        )
    finally:
        context.stop()


if __name__ == "__main__":
    main()
