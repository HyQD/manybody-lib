# Closed-shell symbolic reference fixtures

These files freeze the independent symbolic results available before the
native `manybody-lib` closed-shell engine is implemented.  They are test
oracles, not proof by themselves that an equation is correct.

Every JSON fixture contains:

- a schema version;
- the mathematical method and representation;
- source paths and SHA-256 hashes;
- generator and dependency versions;
- deterministic canonical expressions or sorted Drudge terms;
- human-readable LaTeX where practical.

The common mathematical conventions are documented in
[`CONVENTIONS.md`](CONVENTIONS.md).

## Fixture families

- `hjo_{cis,cid,cisd}.json`: expanded HJO closed-shell expressions generated
  with Drudge.  Exchange-integral and permutation presentation passes are
  disabled so these fixtures contain the canonical expanded algebra.
- `spin_orbital_{cis,cid,cisd}.json`: expanded spin-orbital SymPy sigma, norm,
  1-RDM, and 2-RDM expressions.  Identity-bearing SymPy `Dummy` objects are
  replaced by stable typed symbols during serialization.
- `matthews_metric.json`: exact S2/S3 overlap metrics, pseudoinverses, and
  null/physical projectors.
- `matthews_cisdt_{one,two}_rdm.json`: completed unnormalized Matthews CISDT
  RDM blocks from Drudge.
- `matthews_status.json`: records the validation boundary, including the
  deliberately incomplete triples-to-triples sigma run.

Golden fixtures are never rewritten by an ordinary test run.  Regeneration
is an explicit developer operation in the dedicated `drudge` or
`scipy-1.18.0` environment, followed by review of the JSON diff.

## Generators

From the repository root:

```console
SPARK_LOCAL_IP=127.0.0.1 conda run --no-capture-output -n drudge \
  python tests/symbolic/closed_shell/fixtures/generate_hjo_fixtures.py

conda run --no-capture-output -n scipy-1.18.0 \
  python tests/symbolic/closed_shell/fixtures/generate_spin_orbital_fixtures.py

SPARK_LOCAL_IP=127.0.0.1 conda run --no-capture-output -n drudge \
  python tests/symbolic/closed_shell/fixtures/generate_matthews_fixtures.py \
  --all
```

The Matthews RDM generation is intentionally not part of the ordinary test
suite: the two-body blocks take several minutes with Drudge/Spark.
