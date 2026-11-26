# Closed-shell symbolic algebra implementation plan

**Status:** active; Milestones 0--2 completed

**Date:** 2026-11-26

**Primary objective:** add a reusable closed-shell symbolic operator engine to
`manybody-lib`, validate it first with truncated CI equations, and preserve a
direct route to single-reference coupled-cluster theory.

## 1. Motivation

The repository already contains:

- numerical closed-shell and spin-orbital CI implementations;
- numerical closed-shell CC infrastructure;
- fast spin-orbital SymPy derivations;
- closed-shell Drudge derivations for HJO and Matthews projection manifolds.

The Drudge scripts are useful references, but their Spark representation has
high startup, serialization, and canonicalization costs for the relatively
small operator products needed by truncated CI.  The spin-orbital SymPy
scripts demonstrate that an in-process symbolic implementation can be much
faster.

The proposed feature is a small, typed, spin-free algebra system specialized
for a closed-shell determinant reference.  It should generate equations; it
must not become a runtime dependency of the numerical CI or CC evaluators.

## 2. Goals

1. Represent and manipulate the spin-free substitution operators

   \[
   E_{pq}=\sum_\sigma a^\dagger_{p\sigma}a_{q\sigma},
   \qquad
   e_{pqrs}=E_{pr}E_{qs}-\delta_{qr}E_{ps}.
   \]

2. Implement the closed commutator rules

   \[
   [E_{pq},E_{rs}]
   =\delta_{qr}E_{ps}-\delta_{ps}E_{rq},
   \]

   and

   \[
   [e_{pqrs},E_{tu}]
   =\delta_{rt}e_{pqus}
   +\delta_{st}e_{pqru}
   -\delta_{pu}e_{tqrs}
   -\delta_{qu}e_{ptrs}.
   \]

3. Evaluate closed-shell reference expectation values without expanding all
   operators into explicit alpha and beta spin orbitals.

4. Derive right sigma vectors, norms, 1-RDMs, and 2-RDMs for closed-shell
   CIS, CID, CISD, and CISDT.

5. Keep left and right coefficients independent.  Complex amplitudes and
   complex orbitals must be supported without implicit reality assumptions.

6. Support both the HJO biorthogonal singles/doubles formulation and the
   Matthews triples manifold as layers above the common operator algebra.

7. Retain operator-valued expressions so that connected BCH expansions for
   CCS, CCD, and CCSD can be added without redesigning the core.

8. Produce deterministic expanded and permutation-factored LaTeX, together
   with a machine-readable contraction representation suitable for later
   numerical code generation.

## 3. Non-goals for the first release

- Multireference vacua or generalized normal ordering.
- Relativistic or explicitly spin-dependent operators.
- Arbitrary-rank automatic coupled-cluster theory.
- Distributed execution.
- Replacing SymPy as a general computer algebra system.
- Generating optimized production `einsum` code in the first milestone.
- Removing the existing SymPy or Drudge derivations before the new engine has
  reproduced them independently.

## 4. Design principles

### 4.1 Separate algebra from parametrization

The operator core should know about indices, deltas, `E`, `e`, products,
commutators, and the reference.  It should not know what CISD, HJO, Matthews,
or CCSD means.

Method and manifold modules will construct states and projectors from the
generic operators.  This permits several projection manifolds to share one
contraction engine.

### 4.2 Use a lightweight internal representation

During contraction, use immutable Python records, tuples, exact integer or
rational coefficients, and dictionaries keyed by canonical terms.  Do not
build a large SymPy `Add` expression after every commutator.

Convert to SymPy at a controlled boundary for additional simplification,
inspection, LaTeX, and eventual code generation.

### 4.3 Preserve operator-valued results

CI matrix elements can be reduced directly to scalars, but CC requires nested
operator commutators.  The reduction API must therefore support both:

```text
operator expression -> operator expression
operator expression -> reference expectation value
```

### 4.4 Make canonicalization deterministic and testable

Dummy renaming, factor ordering, and tensor symmetries must produce a unique
term independent of construction order and Python hash order.  Presentation
with permutation operators is a reversible postprocessing step, not part of
the algebraic canonical form.

### 4.5 Track index spaces explicitly

Every index carries one of the following domains:

- occupied;
- virtual;
- general;
- optional spin, used only by an independent validation backend.

Substitution and delta elimination must never change an index domain
silently.  Free and dummy indices are distinct types or carry distinct
metadata.

## 5. Proposed repository layout

```text
manybody_lib/
└── symbolic/
    ├── __init__.py
    └── closed_shell/
        ├── __init__.py
        ├── indices.py
        ├── tensors.py
        ├── operators.py
        ├── expressions.py
        ├── commutators.py
        ├── reference.py
        ├── canonicalize.py
        ├── contraction.py
        ├── sympy_bridge.py
        ├── printers.py
        ├── manifolds/
        │   ├── __init__.py
        │   ├── hjo.py
        │   └── matthews.py
        └── equations/
            ├── __init__.py
            ├── ci.py
            ├── rdm.py
            ├── bch.py
            └── coupled_cluster.py

tests/
└── symbolic/
    └── closed_shell/

symbolic_derivations/
└── closed_shell/
    ├── generate_ci.py
    └── generate_cc.py
```

The exact module split may be consolidated during implementation, but the
boundaries between core algebra, reference rules, manifolds, equation
generation, and presentation should remain.

## 6. Core data model

### 6.1 Indices

An immutable `Index` should contain at least:

- a display name;
- its orbital domain;
- whether it is free or dummy;
- a stable ordering key.

Index equality must include the domain.  An occupied `i` and virtual `i` must
not compare equal merely because their printed labels match.

### 6.2 Scalar tensor factors

Represent amplitudes, integrals, Fock matrices, and deltas as typed tensor
factors with:

- a label;
- an ordered tuple of indices;
- declared monoterm permutation symmetries;
- optional complex conjugation metadata.

Do not assume tensors are real.  Hermitian relations involving complex
conjugation should only be applied when explicitly requested.

### 6.3 Operators

The minimum atomic operator set is:

- `OneBodyOperator(p, q)` for \(E_{pq}\);
- `TwoBodyOperator(p, q, r, s)` for \(e_{pqrs}\);
- an ordered `OperatorWord`;
- sums of coefficient-weighted words.

Keep `e` atomic during commutation.  Its definition in terms of two `E`
operators should remain available for validation and terminal expansion, but
should not be the default contraction path.

### 6.4 Terms and expressions

A term consists of:

- an exact scalar coefficient;
- an ordered tuple of scalar tensor factors;
- an operator word;
- explicit dummy-index declarations;
- optional particle-hole grade and connected-component metadata.

An expression is a mapping from canonical term keys to accumulated exact
coefficients.  Zero terms should disappear as soon as coefficients combine.

## 7. Algebra and contraction algorithms

### 7.1 Product and commutator expansion

Implement the derivation rules

\[
[A,BC]=[A,B]C+B[A,C],
\qquad
[AB,C]=A[B,C]+[A,C]B,
\]

using the atomic commutator table.  Memoize reductions by canonical operator
word and index-domain signature.

The initial implementation should include property tests for antisymmetry,
linearity, the Leibniz rules, and selected Jacobi identities.

### 7.2 Reference reduction

Use closed-shell termination rules such as

\[
E_{ia}|\Phi_0\rangle=0,
\qquad
E_{ab}|\Phi_0\rangle=0,
\qquad
E_{ij}|\Phi_0\rangle=2\delta_{ij}|\Phi_0\rangle,
\]

and the corresponding left-reference rules.  Reduce a structured operator
word by commuting a reference-annihilating operator toward the appropriate
side, emitting delta-weighted branches and memoizing each reduced word.

As an independent oracle, provide a small validation path that expands `E`
operators over alpha and beta spin and evaluates the result using ordinary
fermionic contractions.  This need not be the production path.

### 7.3 Delta processing

Delta elimination must:

1. reject incompatible occupied/virtual identifications;
2. prefer free indices over dummy indices;
3. apply substitutions simultaneously;
4. preserve tensor slot ordering;
5. recanonicalize the affected term;
6. be deterministic when several equivalent substitutions exist.

### 7.4 Canonicalization

Canonicalization should proceed in small, explicit stages:

1. resolve and eliminate deltas;
2. canonicalize each tensor under its declared slot symmetry;
3. sort commuting scalar factors;
4. canonicalize dummy names separately by orbital domain;
5. canonicalize simultaneous excitation-column permutations;
6. combine equivalent terms.

For doubles and triples amplitudes, distinguish simultaneous column symmetry
from independent occupied and virtual symmetry.  In particular,

\[
C^{abc}_{ijk}
=C^{\pi(abc)}_{\pi(ijk)}
\]

does not permit independent permutations of only `abc` or only `ijk`.

Begin with direct enumeration of the small symmetry groups used through
triples.  Evaluate SymPy's tensor canonicalizer only if profiling shows that a
general Butler--Portugal implementation is advantageous.

### 7.5 Permutation presentation

After obtaining a canonical expanded expression, a separate reversible pass
may factor term orbits into operators such as

\[
P^{ab}_{ij},
\qquad
P(ab),
\qquad
P(ij),
\qquad
P(ijk\mid a/bc).
\]

Every factored result must be expanded automatically and compared with the
raw canonical expression before it is printed or stored.

## 8. Projection manifolds

### 8.1 HJO singles and doubles

Implement the HJO excitation and dual projection operators independently of
the contraction core.  Validate the overlap relations directly before using
the projectors in sigma or RDM derivations.

The module should expose:

- right excitation operators;
- covariant projectors;
- biorthogonal dual projectors;
- amplitude symmetries;
- conversion between a Hermitian bra and independent left coefficients.

### 8.2 Matthews triples

The Matthews module should construct the triples Gram matrix from the common
operator engine rather than treating a hard-coded inverse as the sole source
of truth.

It should provide:

- the redundant triples excitation basis;
- simultaneous column symmetry;
- the covariant triples projector;
- an exact rank and null-space calculation;
- the Moore--Penrose inverse on the physical subspace;
- a documented gauge for the null component;
- permutation operators used only for presentation.

The analytic group-algebra form of the metric inverse should be checked
against the explicitly constructed finite matrix in exact arithmetic.

## 9. Equation-generation layer

### 9.1 CI states

Represent a truncated CI state as a collection of excitation-rank sectors,
not as one pre-expanded expression.  Each sector contains its prefactor,
amplitude tensor, and excitation operator.

The generator should exploit selection rules before contraction:

- a two-body Hamiltonian changes excitation rank by at most two;
- a one-body RDM block has a fixed particle-hole grade;
- a two-body RDM block has a grade between -2 and +2.

Only compatible bra/operator/ket sectors should be constructed.

### 9.2 Sigma vectors

Generate the raw covariant projections

\[
\rho_\mu=\langle\widetilde\Phi_\mu|H|\Psi\rangle.
\]

Metric resolution is a separate manifold operation.  For a nonorthogonal or
singular manifold,

\[
\sigma=M^+\rho
\]

should be applied explicitly with its gauge documented.

### 9.3 Density matrices

Keep the left and right states independent and derive unnormalized RDMs by
default:

\[
\gamma^q_p
=\langle\widetilde\Psi|E_{pq}|\Psi\rangle,
\qquad
\Gamma^{rs}_{pq}
=\langle\widetilde\Psi|e_{pqrs}|\Psi\rangle.
\]

Normalization, if desired, must be an explicit caller operation.  This
preserves norm drift for real-time calculations.

## 10. Coupled-cluster extension path

The core must support operator-valued expressions required by

\[
\bar H=e^{-T}H_Ne^T
=H_N+[H_N,T]
+\frac{1}{2}[[H_N,T],T]
+\frac{1}{6}[[[H_N,T],T],T]
+\frac{1}{24}[[[[H_N,T],T],T],T].
\]

For a two-body Hamiltonian and pure excitation `T`, the connected BCH series
terminates after four nested commutators.  Since `T` is built from products of
`E`, the atomic `[E,E]` and `[e,E]` rules are sufficient for standard
single-reference CC similarity transformations.

CC support additionally needs:

- reference-normal-ordered expressions;
- connected-component tracking;
- construction of connected commutators without generating disconnected
  terms that later cancel;
- operator-rank and particle-hole-grade bounds;
- BCH depth and factorial bookkeeping;
- right amplitude and left Lambda projectors;
- similarity transformation of 1-RDM and 2-RDM operators;
- optional intermediate discovery and factorization.

CCS and CCD should be used as small operator-valued tests before CCSD.
CCSDT is a later performance milestone, not a first-release requirement.

## 11. SymPy boundary and dependencies

The core should not subclass large SymPy expression types unless a prototype
shows a clear benefit.  The proposed boundary is:

```text
internal canonical expression
    -> SymPy expression
    -> simplification / LaTeX / code generation
```

Use SymPy initially for:

- exact `Rational` conversion;
- `IndexedBase` or tensor output;
- Kronecker-delta output;
- complex conjugation representation;
- LaTeX printing;
- common-subexpression elimination and later code generation.

SymPy is not currently a runtime dependency of `manybody-lib`.  Add it under a
separate optional dependency group, for example `symbolic`, unless later
requirements show that the numerical package also needs it.  Drudge and
PySpark should remain development/reference dependencies only.

## 12. Public API direction

The initial public API should be small and functional.  Exact names are to be
settled during the prototype, but it should support operations conceptually
equivalent to:

```text
commutator(left, right)
canonicalize(expression)
reference_expectation(expression, reference)
derive_ci_sigma(state, hamiltonian, manifold)
derive_rdm(state, operator_rank, blocks=None)
similarity_transform(operator, cluster, max_depth=4, connected=True)
project(expression, manifold, ranks)
to_sympy(expression)
to_latex(expression, factor_permutations=True)
```

Internal classes should remain private until their invariants and serialization
format are stable.

## 13. Implementation milestones

### Milestone 0: freeze reference fixtures (completed 2026-11-26)

Deliverables:

- archive canonical expanded CIS, CID, and CISD equations from Drudge;
- archive spin-orbital SymPy results used for spin-summation checks;
- record the current CISDT Matthews metric and completed RDM blocks;
- define index and integral conventions in one test document;
- add a small independent determinant/string numerical oracle for tests.

Exit criteria:

- fixtures are deterministic and reviewed;
- every fixture records its conventions and source generator.

Completion record:

- canonical expanded HJO CIS, CID, and CISD fixtures are stored under
  `tests/symbolic/closed_shell/fixtures/` and reproduced byte-for-byte by a
  second complete Drudge run;
- expanded spin-orbital SymPy CIS, CID, and CISD sigma, norm, 1-RDM, and
  2-RDM fixtures are byte-for-byte reproducible;
- exact Matthews S2/S3 metric matrices, pseudoinverses, projectors, and
  validation status are recorded;
- all four Matthews CISDT 1-RDM blocks and ten independent 2-RDM blocks are
  archived as sorted canonical Drudge terms;
- `CONVENTIONS.md` records the index, integral, Hamiltonian, state, manifold,
  and normalization conventions;
- a pure-Python determinant oracle independently verifies the definitions of
  `e`, `[E,E]`, `[e,E]`, and the closed-shell reference termination rules;
- the complete repository test suite passes with 34 tests.

### Milestone 1: internal representation and canonicalization (completed 2026-11-26)

Deliverables:

- typed indices and domains;
- scalar tensors, deltas, `E`, `e`, words, terms, and expressions;
- deterministic dummy renaming;
- tensor and simultaneous-column symmetries;
- exact term collection;
- initial SymPy and LaTeX conversion.

Exit criteria:

- canonicalization is idempotent;
- shuffled construction orders yield identical expressions;
- incompatible domain deltas vanish;
- complex conjugation is preserved rather than simplified implicitly.

Completion record:

- the dependency-free core under `manybody_lib/symbolic/closed_shell/`
  provides immutable typed indices, exact Gaussian-rational coefficients,
  scalar tensors, deltas, orbital dimensions, atomic `E` and `e` operators,
  ordered operator words, terms, and collected expressions;
- delta graphs are reduced with explicit orbital-domain validation, free-index
  constraints are retained, and closed dummy-only components produce symbolic
  orbital dimensions;
- canonicalization jointly handles tensor symmetry orbits, commuting-factor
  ordering, deterministic domain-specific dummy renaming, exact coefficient
  collection, simultaneous excitation-column permutations, and
  antisymmetry-induced zeros;
- the optional `symbolic` dependency group exposes exact SymPy conversion,
  including noncommuting operator products and unevaluated complex
  conjugation, while direct deterministic LaTeX output uses stacked tensor
  indices;
- focused canonicalization, core-model, and presentation tests pass, including
  idempotence, renamed/shuffled constructions, incompatible deltas, explicit
  conjugation, and a permutation stress check;
- the complete repository suite passes with 51 tests.  Three warnings from
  pre-existing invalid escape sequences in legacy closed-shell CI docstrings
  remain outside this milestone's scope.

### Milestone 2: commutators and reference contractions (completed 2026-11-26)

Deliverables:

- atomic `[E,E]` and `[e,E]` rules;
- product commutators;
- memoized reference expectation values;
- explicit-spin validation backend for short words.

Exit criteria:

- commutator antisymmetry and selected Jacobi tests pass;
- all operator words required through CISD agree with the explicit-spin
  backend;
- repeated evaluation hits the cache and returns deterministic results.

Completion record:

- the defining `[E, E]` and `[e, E]` rules are implemented with exact delta
  factors, reverse-order antisymmetry, distributive expression products, and
  both Leibniz rules for operator words; `[e, e]` remains an exact unreduced
  word difference because no additional atomic rule is required;
- reference contraction terminally expands `e` into its defining `E` products
  and reduces the resulting words with closed-shell left/right termination
  rules; unresolved general indices are rejected until a caller selects their
  occupied/virtual blocks;
- term-scoped dummy indices are protected across cached word contractions and
  restored before whole-term canonicalization, while sums eliminated entirely
  by a contraction produce the appropriate orbital-dimension factors;
- terminal expansion, one-body word reduction, and complete operator-word
  expectation values are independently memoized with public cache inspection
  and reset helpers;
- a dependency-free explicit-spin bit-string backend evaluates short `E`/`e`
  words in exact integer arithmetic and is cross-checked against the frozen
  independent determinant oracle;
- antisymmetry, linearity, both Leibniz rules, a one-body Jacobi identity,
  exact unreduced `[e, e]`, dummy-scope preservation, and cache reuse pass;
- 3,276 reference/singles/doubles bra--ket words containing the identity, all
  representative one-body blocks, and all representative two-body blocks
  agree exactly with explicit alpha/beta evaluation;
- the complete repository suite passes with 67 tests.  The same three
  pre-existing legacy CI docstring escape warnings remain outside this
  milestone's scope.

### Milestone 3: CIS, CID, and CISD sigma vectors

Deliverables:

- rank-sector CI state builder;
- Hamiltonian builder under documented integral conventions;
- HJO projectors and metric checks;
- right sigma-vector generation;
- expanded and permutation-factored LaTeX.

Exit criteria:

- exact agreement with the current Drudge equations after canonicalization;
- numerical agreement with the existing closed-shell implementations for
  random complex tensors;
- spin-orbital/closed-shell agreement for a restricted reference.

### Milestone 4: CI norms and RDMs

Deliverables:

- independent left/right state representation;
- norms and every nonzero 1-RDM and 2-RDM block through CISD;
- optional explicit normalization at the API boundary only.

Exit criteria:

- exact agreement with existing symbolic fixtures;
- particle-number and RDM contraction identities pass;
- Hermiticity holds after the explicit Hermitian-bra substitution;
- RDM and sigma evaluations give the same energy.

### Milestone 5: Matthews CISDT

Deliverables:

- triples state and projector;
- exact Gram matrix, rank, null space, and pseudoinverse;
- CISDT sigma, norm, 1-RDM, and 2-RDM generation;
- reversible Matthews permutation factoring.

Exit criteria:

- metric identities pass in exact arithmetic;
- results are invariant under a null-gauge shift of the triples coefficients;
- all RDM blocks agree with the completed Drudge references;
- remaining Drudge triples-sigma sectors are completed or checked against an
  independent explicit-spin derivation;
- peak memory and runtime are recorded for every rank/block.

### Milestone 6: connected BCH prototype

Deliverables:

- operator-valued connected commutators;
- excitation grade and connected-component pruning;
- BCH construction through fourth order;
- CCS and CCD energy/amplitude equations.

Exit criteria:

- disconnected terms are never materialized in the connected mode;
- CCS/CCD equations agree with independent references;
- BCH termination is demonstrated structurally, not imposed only by a caller
  cutoff.

### Milestone 7: CCSD and CC RDMs

Deliverables:

- CCSD energy and right amplitude equations;
- Lambda equations from a CC Lagrangian;
- CCSD 1-RDM and 2-RDM;
- optional intermediate factoring and contraction export.

Exit criteria:

- numerical comparison with PySCF or another independent CCSD implementation;
- left/right stationarity tests pass;
- RDM energy and derivative identities pass;
- generated contractions can reproduce the existing numerical closed-shell
  CCSD results.

## 14. Validation strategy

### 14.1 Algebraic unit tests

- atomic commutator identities;
- product-rule identities;
- selected Jacobi identities;
- `e` expanded into `E E - delta E` versus atomic `e`;
- left/right reference termination rules;
- delta-domain rejection;
- canonicalization idempotence;
- dummy-renaming invariance;
- amplitude and integral slot symmetries.

### 14.2 Golden symbolic tests

Compare canonical expanded expressions, not only formatted LaTeX.  Golden
files should be small, human-readable, and versioned with their conventions.
Permutation-factored output must be re-expanded before comparison.

### 14.3 Independent numerical tests

Evaluate generated expressions for deterministic random complex tensors and
compare with:

- an independent determinant/string evaluator;
- the spin-orbital CI implementation after restricted spin embedding;
- existing numerical closed-shell CI routines;
- PySCF for small RHF-based integration tests where applicable.

### 14.4 Structural identities

For an unnormalized state with norm \(\mathcal N\), test

\[
\sum_p\gamma^p_p=N\mathcal N,
\qquad
\sum_q\Gamma^{rq}_{pq}=(N-1)\gamma^r_p.
\]

Also test Hermiticity under the appropriate left/right substitution, RDM
permutation symmetries, energy reconstruction from the RDMs, and invariance
under Matthews null-space transformations.

## 15. Performance strategy

Performance should be measured from the first prototype, but correctness and
determinism take priority until CISD is reproduced.

Track for every benchmark:

- raw and canonical term counts;
- cache hits and misses;
- time in commutation, delta elimination, and canonicalization;
- peak resident memory;
- SymPy conversion and printing time separately from derivation time.

Initial benchmark set:

1. representative operator words of lengths 2 through 8;
2. all CIS/CID/CISD sigma ranks;
3. all CISD 1-RDM and 2-RDM blocks;
4. the CISDT triples-to-triples Hamiltonian block;
5. CCS/CCD nested commutators when available.

Optimization order:

1. memoize canonical operator-word reductions;
2. prune by particle-hole grade and rank before multiplication;
3. combine terms immediately in hash maps;
4. cache symmetry orbits and dummy-renaming templates;
5. profile before considering SymEngine, C++, Rust, or parallel execution.

Do not introduce Spark.  The expressions are small enough that a deterministic
single-process engine should be the baseline.  Establish hard runtime budgets
only after Milestones 2 and 3 provide representative measurements.

## 16. Migration of existing derivations

1. Keep all current Drudge and SymPy scripts unchanged while the new engine is
   being validated.
2. Use their canonical expanded results as fixtures, not their presentation
   formatting as an API requirement.
3. Add thin scripts under `symbolic_derivations/closed_shell/` that invoke the
   installed `manybody_lib.symbolic.closed_shell` package.
4. Regenerate the closed-shell LaTeX document from the new engine and compare
   its expanded equations with the existing document.
5. Mark Drudge scripts as reference implementations only after CISDT has been
   independently validated.
6. Remove no reference implementation until at least one release has shipped
   with both generators available.

## 17. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Fermionic sign or factor-of-two errors | Explicit-spin oracle, exact low-rank Gram matrices, and numerical string tests |
| Occupied/virtual dummy corruption | Typed domains included in equality and canonical keys |
| Over-applying amplitude symmetry | Separate simultaneous-column symmetry from independent slot symmetry |
| Matthews singular metric hidden by canonicalization | Construct and test the Gram matrix before applying any pseudoinverse |
| Expression explosion in CC | Connected commutators, grade/rank pruning, and immediate term collection |
| SymPy becomes the performance bottleneck | Keep a lightweight internal representation and convert only at boundaries |
| Pretty permutation output changes the algebra | Re-expand and compare every factored expression |
| Complex-valued calculations are simplified as real | No implicit conjugation or reality assumptions |
| Public API freezes too early | Keep internal classes private through the CISD milestones |

## 18. Pull-request sequence

Prefer small reviewable changes:

1. conventions, fixtures, and test oracle;
2. index/tensor/operator representation;
3. canonicalization and SymPy bridge;
4. commutators and reference contractions;
5. HJO CIS/CID/CISD sigma equations;
6. CISD norms and RDMs;
7. Matthews metric and CISDT;
8. connected BCH, CCS, and CCD;
9. CCSD and CC density matrices.

Each pull request should include exact tests, numerical tests where applicable,
benchmark deltas, and a short note about which mathematical conventions it
introduces or relies on.

## 19. First implementation decision

Before writing the full engine, build one disposable prototype that compares
two reduction strategies on the complete set of operator words needed by
CISD:

1. memoized spin-free commutator recursion;
2. explicit spin expansion followed by fermionic Wick contraction.

Compare correctness, canonical term counts, runtime, and implementation
complexity.  The expected production choice is spin-free recursion with the
explicit-spin path retained as a test oracle, but this should be established
by measurement.

## 20. Definition of the first feature-complete release

The closed-shell symbolic feature is ready for an initial release when:

- CIS, CID, and CISD sigma vectors, norms, 1-RDMs, and 2-RDMs are reproduced;
- HJO projectors and metrics are validated independently;
- complex independent left/right amplitudes are supported;
- all outputs have deterministic canonical forms;
- factored LaTeX re-expands to the canonical expressions;
- independent algebraic and numerical validation passes;
- the symbolic dependency is isolated from the numerical runtime package;
- a benchmark report shows that the local engine materially improves on the
  Drudge workflow for the supported derivations;
- the operator-valued API has passed a CCS or CCD BCH proof of concept.
