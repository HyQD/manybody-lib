# Matthews-manifold closed-shell CISDT derivations

These scripts derive the closed-shell CISDT right sigma vector, norm, 1-RDM,
and 2-RDM in the non-orthogonal orbital (skeleton) triples manifold used by
Matthews.  They are independent of the HJO doubles-only scripts in the sibling
`biorthogonal/` directory.

The wavefunction is

\[
|\Psi\rangle=\left(C_0+C_i^aE_{ai}
+\frac12C_{ij}^{ab}E_{ai}E_{bj}
+\frac16C_{ijk}^{abc}E_{ai}E_{bj}E_{ck}\right)|\Phi_0\rangle .
\]

The amplitudes have simultaneous column symmetry.  At triples rank the six
orbital functions have a rank-five overlap metric.  The generated sigma uses

\[
\sigma_n=M_n^+\rho_n,
\qquad
\rho_n=\langle\Phi_0|E_{ia}E_{jb}\cdots H|\Psi\rangle,
\]

and fixes the wavefunction-null trivial-S3 part of `C3` to zero.  No RDM is
divided by the norm.

Run the scripts in the dedicated environment:

```console
cd symbolic_derivations/Drudge/closed_shell/matthews
SPARK_LOCAL_IP=127.0.0.1 conda run --no-capture-output -n drudge \
  python cisdt_sigma.py
SPARK_LOCAL_IP=127.0.0.1 conda run --no-capture-output -n drudge \
  python cisdt_one_rdm.py
SPARK_LOCAL_IP=127.0.0.1 conda run --no-capture-output -n drudge \
  python cisdt_two_rdm.py
conda run --no-capture-output -n drudge python validate_metric.py
```

`DRUDGE_WORKERS` and `DRUDGE_PARTITIONS` default to four and can be set
independently.

The diagonal triples Hamiltonian block is the expensive exception.  It is a
one-time generated-equation task and should not be placed in the ordinary
test suite.  `validate_metric.py` checks the singular-manifold algebra
independently and completes in well under a second.

## Why the scripts are sectorized

The Hamiltonian changes excitation rank by at most two, a one-body RDM block
has one fixed particle-hole grade, and a two-body block has one fixed grade
between -2 and +2.  The scripts therefore derive only compatible bra/ket
rank pairs and simplify every contribution before summing them.  This avoids
the large Spark expression graph created by expanding a complete CISDT bra
and ket at once.  The sigma, 1-RDM, and 2-RDM are separate entry points so a
single derivation does not retain all three graphs.

The diagonal triples Hamiltonian block is additionally split into the 14
particle-hole-resolved terms of Drudge's normal-ordered Hamiltonian.  Each is
Wick contracted and simplified before the results are combined.

Drudge remains useful here because `RestrictedPartHoleDrudge` already knows
the closed-shell reference contractions of the spin-free `E_pq` algebra.
SymPy's `physics.secondquant` is much faster for spin orbitals, but it does
not directly provide this non-orthogonal restricted excitation metric.  A
future faster replacement should therefore be either a small dedicated
spin-free Wick/string engine or a spin-orbital generator followed by a
verified spin-integration and metric transformation; merely translating the
current expressions to generic SymPy would move the difficult part into
custom code.
