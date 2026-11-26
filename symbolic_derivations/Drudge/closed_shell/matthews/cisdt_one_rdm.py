"""Derive the closed-shell CISDT norm and 1-RDM in the Matthews manifold.

The left orbital coefficients are algebraically independent.  For a Hermitian
bra use ``Ctilde_n = conjugate(C_n)`` in the same zero-null triples gauge.  No
normalization denominator is applied.
"""

from common import (
    DerivationOptions,
    build_model,
    derive_norm,
    derive_one_rdm,
    make_spark_context,
    print_align,
    print_conventions,
)


def main() -> None:
    options = DerivationOptions.from_environment()
    context = make_spark_context(options, "closed-shell-matthews-cisdt-1rdm")
    try:
        model = build_model(context, options)
        norm = derive_norm(model)
        blocks = derive_one_rdm(model)
        print_conventions()
        print("% Norm")
        print_align(
            model.dr,
            ((r"\langle\widetilde{\Psi}\vert\Psi\rangle", norm),),
        )
        print()
        print("% Nonzero one-body reduced-density-matrix blocks")
        print_align(model.dr, blocks)
    finally:
        context.stop()


if __name__ == "__main__":
    main()
