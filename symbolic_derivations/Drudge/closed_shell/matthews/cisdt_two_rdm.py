"""Derive the nonzero closed-shell CISDT 2-RDM blocks.

The triples use Matthews' non-orthogonal orbital manifold.  The left and
right coefficients are independent and the RDM is not divided by the norm.
"""

from common import (
    DerivationOptions,
    build_model,
    derive_two_rdm,
    make_spark_context,
    print_align,
    print_conventions,
)


def main() -> None:
    options = DerivationOptions.from_environment()
    context = make_spark_context(options, "closed-shell-matthews-cisdt-2rdm")
    try:
        model = build_model(context, options)
        blocks = derive_two_rdm(model)
        print_conventions()
        print("% Nonzero unique two-body reduced-density-matrix blocks")
        print_align(model.dr, blocks)
    finally:
        context.stop()


if __name__ == "__main__":
    main()
