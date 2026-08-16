# numcalckit

The [NumCalc](https://github.com/Any2HRTF/Mesh2HRTF) solver contract as a Python
package: read and write `NC.inp` input decks, read `be.out/` results, and run and
track the solver.

Depends on numpy and nothing else. It knows the file formats and the binary — not
meshes, config files, or any particular modelling workflow. Those live in
[bemkit](../bemkit).

```python
from numcalckit import Deck, BC, Job, run_jobs, load_bins

deck = Deck(title="sphere", frequencies=np.array([250.0, 500.0]),
            n_total_nodes=..., n_total_elements=...,
            boundary=[BC(0, 111, "VELO", 1.0)])
deck.write(nc_dir / "NC.inp")
run_jobs([Job("sphere", nc_dir)])
p = load_bins(nc_dir, "pBoundary", len(deck.frequencies))
```

`Deck.read(path)` parses a deck back, and `Deck.read(deck.write(p)) == deck`
round-trips — the property that keeps the format coverage honest as fields are
added. The behaviour of NumCalc that these formats encode, including the parts
that fail silently, is documented in
[`docs/numcalc-feature-coverage.md`](../../docs/numcalc-feature-coverage.md).
