# sparql-relax-rs

Diagnoses and repairs broken SPARQL queries over an [Oxigraph](https://github.com/oxigraph/oxigraph)
store. Based on the ideas in `sparql_prune.py`/`sparql_relax.py`, but:

- **Diagnosis** is ablation-style (like `sparql_prune`): for each BGP triple with a concrete
  predicate, remove it, re-run the rest of the query, and check whether the triple's predicate
  actually holds for any of the resulting bindings. If not, that triple is flagged as the culprit.
- **Relaxation** searches the graph's *actual* edges: a bounded breadth-first search from the
  culprit's bound endpoints, trying both a forward (`<p>`) and inverse (`^<p>`) step at each hop,
  to find a real connecting path — rather than substituting predicates from a fixed/frequent
  candidate list. The discovered path is spliced into the query as a SPARQL property path and the
  fix is verified by re-running the modified query.

The query is rewritten as a typed algebra tree (via `spargebra`, the same parser Oxigraph itself
uses) and re-serialized to text, rather than via regex text substitution.

## Usage

For a detailed walkthrough, see the [tutorial.ipynb](../tutorial.ipynb).

```python
from sparql_relax import diagnose, diagnose_and_relax

data = open("model.ttl").read()
query = """
PREFIX ex: <urn:example#>
SELECT ?sensor WHERE {
    ex:building223 ex:hasSensor ?sensor .
    ?sensor a ex:TempSensor .
}
"""

diagnosis = diagnose(data, query)
for culprit in diagnosis.culprits:
    print("broken triple:", culprit.triple)

report = diagnose_and_relax(data, query)
for result in report.results:
    if result.fixed:
        print(result.triple, "->", result.path_text)
        print(result.relaxed_query)
```

Once a query is confirmed working (`diagnose`/`diagnose_and_relax` report no culprits), fetch its
actual results with `query`, which supports any SPARQL form — `SELECT`, `ASK`, `CONSTRUCT`,
`DESCRIBE` — rather than the row counts and samples diagnosis reports:

```python
from sparql_relax import query

result = query(data, "PREFIX ex: <urn:example#> SELECT ?sensor WHERE { ?sensor a ex:TempSensor }")
for row in result.bindings:
    print(row["sensor"].value)
```

For repeated queries against the same graph, build a `Store` once and call its `diagnose`/
`diagnose_and_relax`/`query` methods instead of the module-level functions, which each reparse
`data` from scratch on every call.

## Development

```sh
maturin develop --release
```
