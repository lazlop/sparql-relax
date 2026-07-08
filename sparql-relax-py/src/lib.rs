use oxigraph::io::{RdfFormat, RdfParser};
use oxigraph::store::Store;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use sparql_relax_core::{diagnose as core_diagnose, diagnose_and_relax as core_relax};
use std::fmt::Display;

fn to_py_err(err: impl Display) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn resolve_format(name: &str) -> PyResult<RdfFormat> {
    match name.to_ascii_lowercase().as_str() {
        "ttl" | "turtle" => Ok(RdfFormat::Turtle),
        "nt" | "ntriples" => Ok(RdfFormat::NTriples),
        "nq" | "nquads" => Ok(RdfFormat::NQuads),
        "rdf" | "xml" | "owl" | "rdfxml" => Ok(RdfFormat::RdfXml),
        "trig" => Ok(RdfFormat::TriG),
        other => Err(PyValueError::new_err(format!("unrecognized RDF format {other:?}"))),
    }
}

fn load_store(data: &str, format: &str) -> PyResult<Store> {
    let store = Store::new().map_err(to_py_err)?;
    let fmt = resolve_format(format)?;
    store.load_from_slice(RdfParser::from_format(fmt), data).map_err(to_py_err)?;
    Ok(store)
}

type CulpritTuple = (Vec<String>, usize);
type FilterCulpritTuple = (String, usize);
type RelaxedTripleTuple = (String, Option<String>);
type RelaxResultTuple = (usize, Vec<RelaxedTripleTuple>, Option<String>, usize);

#[pymodule]
mod _sparql_relax_rs {
    use super::*;

    /// Diagnoses `query` against the RDF graph in `data` (parsed as `format`).
    ///
    /// For each basic-graph-pattern triple with a concrete predicate, removes
    /// it and re-runs the rest of the query; if the triple's predicate never
    /// actually holds for any of the resulting bindings, it's reported as a
    /// culprit likely responsible for the original query's empty/wrong result.
    ///
    /// `depth` controls how many triples may be removed *together* while
    /// searching for a culprit: single triples are always tried first (depth
    /// 1); if none of those unblock the query, pairs are tried (depth 2),
    /// then triples of three (depth 3), and so on up to `depth`. The search
    /// stops escalating as soon as some combination at the current size
    /// unblocks the query, or once the combination size would exceed the
    /// number of candidate triples. Defaults to 3, which keeps the
    /// (combinatorial) search cheap while still catching the common case of
    /// two or three jointly-broken triples.
    ///
    /// Also applies the same one-at-a-time ablation logic to `FILTER`
    /// expressions (both plain `FILTER(...)` and the condition on an
    /// `OPTIONAL`): remove one, re-run the rest, and flag it if that
    /// strictly grows the result set. Filters are only ever reported, never
    /// relaxed, and `depth` does not apply to them.
    ///
    /// This only identifies *which* triple(s)/filter(s) are broken — it
    /// does no variable-binding work, so it's cheap even on large result
    /// sets. Use `diagnose_and_relax` to also resolve what a culprit's
    /// variables are bound to and search for a fix.
    ///
    /// Returns `(original_row_count, culprits, filter_culprits)`. Each
    /// culprit is `(triples, depth)`: `triples` is a list of triple texts in
    /// the combination (just one unless `depth > 1` was needed), and `depth`
    /// is the combination size at which it was found. Each filter culprit is
    /// `(expression_text, row_count_without_filter)`.
    #[pyfunction]
    #[pyo3(signature = (data, query, format="turtle", depth=3))]
    fn diagnose(
        data: &str,
        query: &str,
        format: &str,
        depth: usize,
    ) -> PyResult<(usize, Vec<CulpritTuple>, Vec<FilterCulpritTuple>)> {
        let store = load_store(data, format)?;
        let diagnosis = core_diagnose(query, &store, depth).map_err(to_py_err)?;
        let culprits = diagnosis
            .culprits
            .into_iter()
            .map(|c| (c.triples.iter().map(ToString::to_string).collect(), c.depth))
            .collect();
        let filter_culprits = diagnosis
            .filter_culprits
            .into_iter()
            .map(|f| (f.expression.to_string(), f.row_count_without_filter))
            .collect();
        Ok((diagnosis.original_row_count, culprits, filter_culprits))
    }

    /// Diagnoses `query` and, for each culprit combination found, searches
    /// for a real forward/inverse path (over the graph's actual edges)
    /// connecting each of its triples' bound endpoints, then splices *all*
    /// of them into the query at once and re-runs it to confirm the fix. A
    /// combination is only relaxed as a whole: if any one of its triples has
    /// no discoverable path, no relaxed query is built for it (the others
    /// being fixable wouldn't help, since they were only broken *together*).
    ///
    /// `ablation_depth` is passed through to `diagnose` to control how many
    /// triples may be jointly removed while searching for a culprit
    /// (default 3, same as `diagnose`'s `depth`).
    ///
    /// `FILTER` culprits found during diagnosis are included as-is (no
    /// relaxation is attempted for them).
    ///
    /// Different sampled bound pairs for the same triple can need genuinely
    /// different real paths (e.g. one entity reached via a 2-hop path,
    /// another via an unrelated 1-hop path); rather than picking just one,
    /// every distinct path found for that triple is combined into a single
    /// SPARQL alternation (`|`) so the fix recovers all of them.
    /// `sample_limit` controls how many bound pairs are considered per
    /// triple (default 5 — a representative sample rather than every row) —
    /// pass `None` to consider every distinct pair instead of stopping
    /// early.
    ///
    /// `max_depth` bounds the forward/inverse path search itself. Left as
    /// `None` (the default), it adapts to how much of a triple resolved:
    /// depth 2 when both its subject and object are bound (a concrete,
    /// target-bounded search), or the shallower depth 1 when only one side
    /// is bound (an undirected exploration with no fixed goal, so more
    /// expensive per level). Pass an explicit integer to override both
    /// cases uniformly.
    ///
    /// Returns `(original_row_count, results, filter_results)`. Each result
    /// is `(found_at_depth, triples, relaxed_query, row_count)`: `triples`
    /// is a list of `(triple_text, path_text)` for every triple in the
    /// combination (`path_text` is `None` when no connecting path was found
    /// within `max_depth` hops); `relaxed_query` is the combined fix, or
    /// `None` if any triple in the combination had no path. Each filter
    /// result is `(expression_text, row_count_without_filter)`.
    #[pyfunction]
    #[pyo3(signature = (data, query, format="turtle", ablation_depth=3, max_depth=None, sample_limit=5))]
    fn diagnose_and_relax(
        data: &str,
        query: &str,
        format: &str,
        ablation_depth: usize,
        max_depth: Option<usize>,
        sample_limit: Option<usize>,
    ) -> PyResult<(usize, Vec<RelaxResultTuple>, Vec<FilterCulpritTuple>)> {
        let store = load_store(data, format)?;
        let report = core_relax(query, &store, ablation_depth, max_depth, sample_limit).map_err(to_py_err)?;
        let results = report
            .results
            .into_iter()
            .map(|r| {
                let triples = r.triples.into_iter().map(|t| (t.triple_text, t.path_text)).collect();
                (r.found_at_depth, triples, r.relaxed_query, r.row_count)
            })
            .collect();
        let filter_results = report
            .filter_results
            .into_iter()
            .map(|f| (f.expression_text, f.row_count_without_filter))
            .collect();
        Ok((report.original_row_count, results, filter_results))
    }
}
