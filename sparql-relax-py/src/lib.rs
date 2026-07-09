use oxigraph::io::{RdfFormat, RdfParser};
use oxigraph::store::Store;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use sparql_relax_core::{NamespaceScope, diagnose as core_diagnose, diagnose_and_relax as core_relax};
use std::fmt::Display;
use std::time::Duration;

fn to_py_err(err: impl Display) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Default value for `allowed_namespaces` when the Python caller doesn't
/// pass one: restricted to `DEFAULT_RELAX_NAMESPACES`. Passing `None`
/// explicitly (rather than omitting the argument) opts out to unrestricted
/// search instead.
fn default_relax_namespaces() -> Option<Vec<String>> {
    Some(sparql_relax_core::DEFAULT_RELAX_NAMESPACES.iter().map(|ns| ns.to_string()).collect())
}

fn namespace_scope(allowed_namespaces: Option<Vec<String>>) -> NamespaceScope {
    match allowed_namespaces {
        Some(namespaces) => NamespaceScope::Only(namespaces),
        None => NamespaceScope::Unrestricted,
    }
}

/// Default value for `timeout` when the Python caller doesn't pass one:
/// `DEFAULT_RELAX_TIMEOUT`, in seconds. Passing `None` explicitly opts out
/// to unbounded relaxation instead.
fn default_relax_timeout() -> Option<f64> {
    Some(sparql_relax_core::DEFAULT_RELAX_TIMEOUT.as_secs_f64())
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
type RelaxResultTuple = (usize, Vec<RelaxedTripleTuple>, Option<String>, usize, String, usize);

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
        py: Python<'_>,
        data: &str,
        query: &str,
        format: &str,
        depth: usize,
    ) -> PyResult<(usize, Vec<CulpritTuple>, Vec<FilterCulpritTuple>)> {
        let store = load_store(data, format)?;
        // Releases the GIL for the (potentially long-running, internally
        // multi-threaded) search: without this, a Python-side timeout
        // wrapper around this call couldn't actually regain control until
        // the search finished on its own, since no other Python thread
        // could run while this one held the GIL.
        let diagnosis = py.detach(|| core_diagnose(query, &store, depth)).map_err(to_py_err)?;
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
    /// `None` (the default), it uses depth 2. A triple whose other side
    /// isn't bound anywhere else in the query is skipped entirely (there's
    /// no specific target to search for, only a single anchor with no fixed
    /// goal — not worth searching for a suggestion with nothing to verify
    /// it against).
    ///
    /// `result_limit` caps how many rows a relaxed query's `LIMIT` allows
    /// (default 50,000 — a relaxed path, especially an alternation of
    /// several distinct paths, can match far more broadly than the original
    /// triple did); only ever tightens a `LIMIT` already present in the
    /// original query, never loosens it. Pass `None` to leave it unbounded.
    ///
    /// `allowed_namespaces` restricts path search to predicates whose IRI
    /// starts with one of these prefixes; a real edge outside every listed
    /// namespace is invisible to the search even if it would otherwise
    /// connect the two endpoints. Defaults to `DEFAULT_RELAX_NAMESPACES`
    /// (Brick, ASHRAE 223P, RDFS, QUDT). Pass `None` explicitly for no
    /// restriction (any real predicate found in the graph is fair game).
    ///
    /// `timeout` (seconds) bounds the SPARQL query work needed to relax
    /// *each* culprit combination (resolving endpoints, verifying a
    /// candidate fix) — not diagnosis, and not path search itself, which
    /// never touches the query engine. A combination that can't finish
    /// within its budget falls back to `pruned_query` rather than hanging
    /// or failing the whole call. Defaults to 5.0 seconds; pass `None` to
    /// leave it unbounded.
    ///
    /// Returns `(original_row_count, results, filter_results)`. Each result
    /// is `(found_at_depth, triples, relaxed_query, row_count, pruned_query,
    /// pruned_row_count)`: `triples` is a list of `(triple_text, path_text)`
    /// for every triple in the combination (`path_text` is `None` when no
    /// connecting path was found within `max_depth` hops); `relaxed_query`
    /// is the combined fix, or `None` if any triple in the combination had
    /// no path (including one abandoned because `timeout` was exceeded).
    /// `pruned_query` is the original query with every triple in the
    /// combination simply removed (no path substitution) — not a real fix,
    /// but always present and (outside the rare case where `timeout` cuts
    /// off even its own verification) guaranteed non-empty, so it's there
    /// as a fallback when `relaxed_query` is `None` or still returns
    /// nothing. Each filter result is `(expression_text,
    /// row_count_without_filter)`.
    #[pyfunction]
    #[pyo3(signature = (
        data, query, format="turtle", ablation_depth=3, max_depth=None, sample_limit=5, result_limit=50_000,
        allowed_namespaces=default_relax_namespaces(), timeout=default_relax_timeout()
    ))]
    fn diagnose_and_relax(
        py: Python<'_>,
        data: &str,
        query: &str,
        format: &str,
        ablation_depth: usize,
        max_depth: Option<usize>,
        sample_limit: Option<usize>,
        result_limit: Option<usize>,
        allowed_namespaces: Option<Vec<String>>,
        timeout: Option<f64>,
    ) -> PyResult<(usize, Vec<RelaxResultTuple>, Vec<FilterCulpritTuple>)> {
        let store = load_store(data, format)?;
        let scope = namespace_scope(allowed_namespaces);
        let timeout = match timeout {
            Some(seconds) if seconds.is_finite() && seconds >= 0.0 => Some(Duration::from_secs_f64(seconds)),
            Some(_) => return Err(PyValueError::new_err("timeout must be a non-negative, finite number of seconds")),
            None => None,
        };
        // See the comment on `diagnose` above: releasing the GIL here is
        // what lets a Python-side timeout wrapper actually regain control if
        // a particular query needs an expensive reduced-query evaluation
        // (e.g. removing a triple leaves the rest of the query essentially
        // unconstrained).
        let report = py
            .detach(|| core_relax(query, &store, ablation_depth, max_depth, sample_limit, result_limit, scope, timeout))
            .map_err(to_py_err)?;
        let results = report
            .results
            .into_iter()
            .map(|r| {
                let triples = r.triples.into_iter().map(|t| (t.triple_text, t.path_text)).collect();
                (r.found_at_depth, triples, r.relaxed_query, r.row_count, r.pruned_query, r.pruned_row_count)
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
