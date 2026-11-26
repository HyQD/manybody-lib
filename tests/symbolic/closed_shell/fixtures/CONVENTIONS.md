# Closed-shell symbolic fixture conventions

## Indices

- `i, j, k, l, ...`: spatial orbitals doubly occupied in the closed-shell
  reference.
- `a, b, c, d, ...`: virtual spatial orbitals.
- `p, q, r, s, ...`: general spatial orbitals.
- Greek labels are alpha or beta spin.

Repeated internal indices are summed.  Orbitals are orthonormal but are not
assumed canonical.  The fixture algebra does not assume that orbitals,
integrals, or CI coefficients are real.

## Spin-free operators

The unitary-group generator and two-electron substitution operator are

\[
E_{pq}=\sum_\sigma a^\dagger_{p\sigma}a_{q\sigma},
\qquad
e_{pqrs}=E_{pr}E_{qs}-\delta_{qr}E_{ps}.
\]

Equivalently,

\[
e_{pqrs}=\sum_{\sigma\tau}
a^\dagger_{p\sigma}a^\dagger_{q\tau}
a_{s\tau}a_{r\sigma}.
\]

The defining commutators are

\[
[E_{pq},E_{rs}]=\delta_{qr}E_{ps}-\delta_{ps}E_{rq},
\]

and

\[
[e_{pqrs},E_{tu}]
=\delta_{rt}e_{pqus}+\delta_{st}e_{pqru}
-\delta_{pu}e_{tqrs}-\delta_{qu}e_{ptrs}.
\]

## Spatial-orbital integrals and Hamiltonian

Spatial two-electron integrals use physicists' ordering,

\[
u^{pq}_{rs}
=\iint \phi_p^*(1)\phi_q^*(2)\,w(1,2)\,
\phi_r(1)\phi_s(2)\,\mathrm d1\,\mathrm d2.
\]

They are not antisymmetrized.  For a symmetric two-particle interaction,

\[
u^{pq}_{rs}=u^{qp}_{sr},
\qquad
u^{pq}_{rs}=\left(u^{rs}_{pq}\right)^*.
\]

The exchange combination used only for compact presentation is

\[
L^{pq}_{rs}=2u^{pq}_{rs}-u^{pq}_{sr}.
\]

The Hamiltonian is

\[
H=\sum_{pq}h^p_qE_{pq}
+\frac12\sum_{pqrs}u^{pq}_{rs}e_{pqrs},
\]

or, normal ordered relative to the closed-shell reference,

\[
H=E_{\mathrm{ref}}
+\sum_{pq}f^p_q\{E_{pq}\}
+\frac12\sum_{pqrs}u^{pq}_{rs}\{e_{pqrs}\}.
\]

## Spin-orbital fixtures

Spin-orbital indices refer to individual spin orbitals.  Their two-electron
tensor is antisymmetrized,

\[
v_{pqrs}=\langle pq\Vert rs\rangle,
\]

and enters the normal-ordered Hamiltonian with the factor `1/4`.  This tensor
must not be confused with the non-antisymmetrized spatial tensor `u` above.

## HJO state and projectors

The right singles and redundant doubles are

\[
X_i^a=E_{ai},
\qquad
X_{ij}^{ab}=E_{ai}E_{bj}.
\]

The HJO dual operators are

\[
Y_a^i=\frac12E_{ia},
\qquad
Y_{ab}^{ij}=\frac13E_{jb}E_{ia}+\frac16E_{ib}E_{ja}.
\]

Doubles coefficients have simultaneous column symmetry,

\[
C^{ab}_{ij}=C^{ba}_{ji},
\]

not separate occupied- and virtual-index symmetry.

## Matthews triples

The right CISDT state is

\[
|\Psi\rangle=\left(
C_0+C_i^aE_{ai}
+\frac12C_{ij}^{ab}E_{ai}E_{bj}
+\frac16C_{ijk}^{abc}E_{ai}E_{bj}E_{ck}
\right)|\Phi_0\rangle.
\]

Triples coefficients also have simultaneous column symmetry.  The six
relative occupied-index permutations have a rank-five Gram matrix.  Sigma is
reported in the Moore--Penrose gauge,

\[
\sigma_3=M_3^+\rho_3,
\]

with the null component set to zero.

## Reduced density matrices and normalization

The spatial spin-summed RDMs are

\[
\gamma^q_p=\langle\widetilde\Psi|E_{pq}|\Psi\rangle,
\qquad
\Gamma^{rs}_{pq}=\langle\widetilde\Psi|e_{pqrs}|\Psi\rangle.
\]

Left and right coefficients are independent in the closed-shell fixtures.
No RDM fixture is divided by
\(\langle\widetilde\Psi|\Psi\rangle\).  A Hermitian bra is an explicit
substitution performed after derivation.

## Canonical versus presentation forms

Golden algebra is stored fully expanded.  The `L` tensor and permutation
operators are presentation abbreviations only.  Any future factored result
must be expanded and compared with the canonical fixture.
