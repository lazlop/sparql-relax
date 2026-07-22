use oxigraph::io::{RdfFormat, RdfParser};
use oxigraph::store::Store;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use sparql_relax_core::{
    FanoutIndex, NamespaceScope, QueryOutcome, RdfTerm, diagnose as core_diagnose,
    diagnose_and_connect_with_fanout_index as core_connect, pruned_query_text as core_pruned_query_text, query as core_query,
};
use std::fmt::Display;
use std::time::Duration;

fn to_py_err(err: impl Display) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Default value for `allowed_namespaces` when the Python caller doesn't
/// pass one: restricted to `DEFAULT_CONNECT_NAMESPACES`. Passing `None`
/// explicitly (rather than omitting the argument) opts out to unrestricted
/// search instead.
fn default_connect_namespaces() -> Option<Vec<String>> {
    Some(sparql_relax_core::DEFAULT_CONNECT_NAMESPACES.iter().map(|ns| ns.to_string()).collect())
}

fn namespace_scope(allowed_namespaces: Option<Vec<String>>) -> NamespaceScope {
    match allowed_namespaces {
        Some(namespaces) => NamespaceScope::Only(namespaces),
        None => NamespaceScope::Unrestricted,
    }
}

/// Default value for `timeout` when the Python caller doesn't pass one:
/// `DEFAULT_CONNECT_TIMEOUT`, in seconds. Passing `None` explicitly opts out
/// to unbounded connection instead.
fn default_connect_timeout() -> Option<f64> {
    Some(sparql_relax_core::DEFAULT_CONNECT_TIMEOUT.as_secs_f64())
}

/// Default value for `diagnose_timeout`/`timeout` (on `diagnose_and_connect`
/// and `diagnose` respectively) when the Python caller doesn't pass one:
/// `DEFAULT_ABLATION_TIMEOUT`, in seconds. Passing `None` explicitly opts
/// out to unbounded diagnosis instead.
fn default_ablation_timeout() -> Option<f64> {
    Some(sparql_relax_core::DEFAULT_ABLATION_TIMEOUT.as_secs_f64())
}

/// Default value for `timeout` on `query`/`Store.query` when the Python
/// caller doesn't pass one: `DEFAULT_QUERY_TIMEOUT`, in seconds. Passing
/// `None` explicitly opts out to unbounded execution instead.
fn default_query_timeout() -> Option<f64> {
    Some(sparql_relax_core::DEFAULT_QUERY_TIMEOUT.as_secs_f64())
}

/// Converts a `timeout` argument (seconds) to a `Duration`, rejecting
/// negative/non-finite values rather than letting `Duration::from_secs_f64`
/// panic on them.
fn parse_timeout_seconds(timeout: Option<f64>) -> PyResult<Option<Duration>> {
    match timeout {
        Some(seconds) if seconds.is_finite() && seconds >= 0.0 => Ok(Some(Duration::from_secs_f64(seconds))),
        Some(_) => Err(PyValueError::new_err("timeout must be a non-negative, finite number of seconds")),
        None => Ok(None),
    }
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
type ConnectedTripleTuple = (String, Option<String>);
type ConnectResultTuple = (usize, Vec<ConnectedTripleTuple>, Option<String>, usize, String, usize);
// `CartesianRiskCombo` has the exact same (triples, depth) shape as `Culprit`,
// so it reuses `CulpritTuple` rather than a duplicate type — the two are
// kept apart at the Python layer by which list they end up in, not by shape.
type DiagnoseTuples = (usize, Vec<CulpritTuple>, Vec<FilterCulpritTuple>, Vec<CulpritTuple>);
type ConnectTuples = (usize, Vec<ConnectResultTuple>, Vec<FilterCulpritTuple>, Vec<CulpritTuple>);

/// An RDF term as `(kind, value, datatype, language)`, where `kind` is
/// `"uri"`, `"bnode"`, or `"literal"` — the same three-way split as the
/// SPARQL 1.1 Query Results JSON Format's `type` field, so callers already
/// familiar with that shape don't need to learn a new one. `datatype`/
/// `language` are only ever set when `kind` is `"literal"`.
type TermTuple = (String, String, Option<String>, Option<String>);

/// The result of `query`/`Store.query`, tagged by SPARQL query form since a
/// single query can only ever produce one of these three shapes:
/// `(form, boolean_result, variables, rows, triples)` where `form` is
/// `"boolean"`, `"solutions"`, or `"graph"` and only the field(s) matching
/// that form are set (the rest are `None`).
type QueryTuple = (String, Option<bool>, Option<Vec<String>>, Option<Vec<Vec<Option<TermTuple>>>>, Option<Vec<(TermTuple, TermTuple, TermTuple)>>);

fn term_tuple(term: RdfTerm) -> TermTuple {
    match term {
        RdfTerm::Iri(value) => ("uri".to_string(), value, None, None),
        RdfTerm::BlankNode(value) => ("bnode".to_string(), value, None, None),
        RdfTerm::Literal { value, datatype, language } => ("literal".to_string(), value, Some(datatype), language),
    }
}

/// Runs `query` against an already-loaded `store` and converts the result to
/// the plain tuple the Python layer returns. Shared by the free `query`
/// pyfunction (which loads a throwaway `store` first) and `RdfStore::query`
/// (which reuses one built once, up front) so the two entry points can't
/// drift apart.
fn query_tuples(store: &Store, query: &str, row_limit: Option<usize>, timeout: Option<Duration>) -> Result<QueryTuple, sparql_relax_core::RelaxError> {
    match core_query(query, store, row_limit, timeout)? {
        QueryOutcome::Boolean(b) => Ok(("boolean".to_string(), Some(b), None, None, None)),
        QueryOutcome::Solutions { variables, rows } => {
            let rows = rows.into_iter().map(|row| row.into_iter().map(|t| t.map(term_tuple)).collect()).collect();
            Ok(("solutions".to_string(), None, Some(variables), Some(rows), None))
        }
        QueryOutcome::Graph(triples) => {
            let triples =
                triples.into_iter().map(|t| (term_tuple(t.subject), term_tuple(t.predicate), term_tuple(t.object))).collect();
            Ok(("graph".to_string(), None, None, None, Some(triples)))
        }
    }
}

/// Runs `diagnose` against an already-loaded `store` and converts the result
/// to the plain tuples the Python layer returns. Shared by the free
/// `diagnose` pyfunction (which loads a throwaway `store` first) and
/// `RdfStore::diagnose` (which reuses one built once, up front) so the two
/// entry points can't drift apart.
fn diagnose_tuples(
    store: &Store,
    query: &str,
    depth: usize,
    timeout: Option<Duration>,
    ignore_cartesian_risk: bool,
) -> Result<DiagnoseTuples, sparql_relax_core::RelaxError> {
    let diagnosis = core_diagnose(query, store, depth, timeout, ignore_cartesian_risk)?;
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
    let cartesian_risks = diagnosis
        .cartesian_risks
        .into_iter()
        .map(|c| (c.triples.iter().map(ToString::to_string).collect(), c.depth))
        .collect();
    Ok((diagnosis.original_row_count, culprits, filter_culprits, cartesian_risks))
}

/// Same as [`diagnose_tuples`], but for `diagnose_and_connect` — shared by the
/// free `diagnose_and_connect` pyfunction and `RdfStore::diagnose_and_connect`.
///
/// `fanout_index` is [`FanoutIndex`]'s one-time, whole-graph index of each
/// predicate's typical fan-out, used by path search to reject a candidate
/// hop whose specific endpoint is an unusually shared "hub" value for its
/// predicate (see `sparql-relax-core::fanout`'s module docs). The free
/// `diagnose_and_connect` pyfunction builds one fresh per call (it already
/// builds a throwaway `Store` per call too); `RdfStore` builds it once
/// alongside its `inner` store and reuses it for every call.
#[allow(clippy::too_many_arguments)]
fn diagnose_and_connect_tuples(
    store: &Store,
    fanout_index: &FanoutIndex,
    query: &str,
    ablation_depth: usize,
    max_depth: Option<usize>,
    sample_limit: Option<usize>,
    result_limit: Option<usize>,
    scope: NamespaceScope,
    timeout: Option<Duration>,
    diagnose_timeout: Option<Duration>,
    ignore_cartesian_risk: bool,
    find_all_paths: bool,
) -> Result<ConnectTuples, sparql_relax_core::RelaxError> {
    let report = core_connect(
        query,
        store,
        fanout_index,
        ablation_depth,
        max_depth,
        sample_limit,
        result_limit,
        scope,
        timeout,
        diagnose_timeout,
        ignore_cartesian_risk,
        find_all_paths,
    )?;
    let results = report
        .results
        .into_iter()
        .map(|r| {
            let triples = r.triples.into_iter().map(|t| (t.triple_text, t.path_text)).collect();
            (r.found_at_depth, triples, r.connected_query, r.row_count, r.pruned_query, r.pruned_row_count)
        })
        .collect();
    let filter_results = report
        .filter_results
        .into_iter()
        .map(|f| (f.expression_text, f.row_count_without_filter))
        .collect();
    let cartesian_risks = report
        .cartesian_risks
        .into_iter()
        .map(|c| (c.triples.iter().map(ToString::to_string).collect(), c.depth))
        .collect();
    Ok((report.original_row_count, results, filter_results, cartesian_risks))
}

#[pymodule]
mod _sparql_relax {
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
    /// connected, and `depth` does not apply to them.
    ///
    /// This only identifies *which* triple(s)/filter(s) are broken — it
    /// does no variable-binding work, so it's cheap even on large result
    /// sets. Use `diagnose_and_connect` to also resolve what a culprit's
    /// variables are bound to and search for a fix.
    ///
    /// `timeout` (seconds) bounds every SPARQL query this call runs — the
    /// original query itself, and every triple-combo/filter ablation check,
    /// all sharing one deadline rather than a fresh budget per check — so
    /// the *whole* call is bounded by roughly `timeout` regardless of how
    /// many candidates there are to work through. A check that can't finish
    /// in time is treated as "not a culprit"; if the *original* query can't
    /// even be evaluated within the budget, this raises rather than
    /// returning a result, since there's nothing meaningful to diagnose
    /// without it. Defaults to 5.0 seconds; pass `None` to leave it
    /// unbounded — but note that abandoning this call from the Python side
    /// (e.g. `future.result(timeout=...)`) does *not* stop it from running
    /// to completion in the background, since Python threads can't be
    /// force-cancelled; a real, enforced `timeout` here is what actually
    /// bounds the work and lets a slow row's search release its threads
    /// instead of piling up behind an abandoned future.
    ///
    /// Returns `(original_row_count, culprits, filter_culprits,
    /// cartesian_risks)`. Each culprit is `(triples, depth)`: `triples` is a
    /// list of triple texts in the combination (just one unless `depth > 1`
    /// was needed), and `depth` is the combination size at which it was
    /// found. Each filter culprit is `(expression_text,
    /// row_count_without_filter)`.
    ///
    /// `cartesian_risks` is shaped exactly like `culprits` (`(triples,
    /// depth)`), but means something different: each entry is a combination
    /// whose reduced pattern (with it removed) was never evaluated against
    /// `data` at all, because doing so would force a cartesian product —
    /// some of the remaining triples' variables never overlap, even
    /// transitively, with the rest, which is exactly the shape that can make
    /// a query engine materialize a full N×M cross product before yielding a
    /// single row, `timeout` notwithstanding. This is *not* a claim that the
    /// combination is or isn't a genuine culprit — it's surfaced separately
    /// so a query this call declined to check doesn't look identical to one
    /// it actually ruled out, unless `ignore_cartesian_risk` is set (see
    /// below), in which case `cartesian_risks` always comes back empty.
    ///
    /// `ignore_cartesian_risk` disables that guard entirely: every
    /// combination is actually evaluated against `data`, so a combination
    /// that would otherwise have been skipped can be confirmed a genuine
    /// culprit instead. Defaults to `False`, preserving the guard. Passing
    /// `True` means opting out of the protection it applies — a
    /// disconnected BGP can make the query engine materialize a full N×M
    /// cross product before yielding a single row, regardless of `timeout`
    /// — a measured case elsewhere in this project sat for over 200 seconds
    /// and permanently occupied a shared worker thread until the whole
    /// process was killed (see `eval/run_eval.py`'s process-level watchdog
    /// for why that backstop lives at the process level, not inside this
    /// call). Only set this once you've independently judged the risk worth
    /// taking for this specific query/graph, ideally from a process you can
    /// afford to kill outright if a check gets stuck.
    ///
    /// Runs `query` (any SPARQL query form — `SELECT`, `ASK`, `CONSTRUCT`,
    /// `DESCRIBE`) against the RDF graph in `data` (parsed as `format`) and
    /// returns its actual results — unlike `diagnose`/`diagnose_and_connect`,
    /// which only explain and fix a query that returns nothing, this is the
    /// ordinary case of running one that already works.
    ///
    /// `row_limit` caps how many rows a `SELECT`/`CONSTRUCT`/`DESCRIBE`
    /// result may return, applied as a `LIMIT` on the query itself before
    /// evaluation — only ever tightening a `LIMIT` already present in the
    /// query, never loosening it — so an oversized result is bounded during
    /// evaluation rather than computed in full and then truncated. Has no
    /// effect on `ASK`, which only ever returns one boolean. Defaults to
    /// `None` (unbounded).
    ///
    /// `timeout` (seconds) bounds evaluation; a query that doesn't finish in
    /// time raises rather than continuing to run unobserved in the
    /// background (Python threads can't be force-cancelled, so an
    /// internally-enforced timeout here is what actually stops the work —
    /// see `diagnose`'s docs for why that matters even when the caller has
    /// its own external timeout). Defaults to 10.0 seconds; pass `None` to
    /// leave it unbounded.
    ///
    /// Returns `(form, boolean_result, variables, rows, triples)`, tagged by
    /// `form`:
    /// - `"boolean"`: `boolean_result` is the `ASK` result.
    /// - `"solutions"`: `variables` is the `SELECT` column order; each row in
    ///   `rows` aligns to it position-for-position, with `None` wherever
    ///   that variable was left unbound in that row.
    /// - `"graph"`: `triples` is the `CONSTRUCT`/`DESCRIBE` result graph, as
    ///   `(subject, predicate, object)` tuples.
    ///
    /// Every term (in `rows` or `triples`) is `(kind, value, datatype,
    /// language)`: `kind` is `"uri"`, `"bnode"`, or `"literal"` (the same
    /// three-way split as the SPARQL 1.1 Query Results JSON Format's `type`
    /// field); `datatype`/`language` are only set when `kind` is
    /// `"literal"`.
    ///
    /// Builds a throwaway store from `data` on every call — for more than
    /// one query against the same graph, build a `Store` once instead and
    /// call its `query` method, which reuses it.
    #[pyfunction]
    #[pyo3(signature = (data, query, format="turtle", row_limit=None, timeout=default_query_timeout()))]
    fn query(py: Python<'_>, data: &str, query: &str, format: &str, row_limit: Option<usize>, timeout: Option<f64>) -> PyResult<QueryTuple> {
        let store = load_store(data, format)?;
        let timeout = parse_timeout_seconds(timeout)?;
        // See the comment on `diagnose` below about releasing the GIL.
        py.detach(|| query_tuples(&store, query, row_limit, timeout)).map_err(to_py_err)
    }

    /// The SPARQL text of `query` with every triple in `triples` removed
    /// from its basic graph pattern — no path substitution, just ablation.
    /// `triples` should be triple texts from a `Culprit`/`CartesianRiskCombo`
    /// (e.g. `diagnosis.culprits[i].triples` or `diagnosis.cartesian_risks[i].triples`)
    /// already obtained for this same `query`; each is matched back to the
    /// query's actual BGP triples by an exact text match, and this raises if
    /// any isn't found there.
    ///
    /// Unlike every other function here, this takes no RDF graph at all and
    /// runs nothing against one — it's a pure syntactic transform, useful
    /// for scoring what a confirmed culprit combination's removal alone gets
    /// you (e.g. value-set F1 against ground truth) without needing a real
    /// path-substituted fix built for it too.
    #[pyfunction]
    fn pruned_query(query: &str, triples: Vec<String>) -> PyResult<String> {
        core_pruned_query_text(query, &triples).map_err(to_py_err)
    }

    /// Builds a throwaway store from `data` on every call — for more than
    /// one query against the same graph, build a `Store` once instead and
    /// call its `diagnose` method, which reuses it.
    #[pyfunction]
    #[pyo3(signature = (data, query, format="turtle", depth=3, timeout=default_ablation_timeout(), ignore_cartesian_risk=false))]
    fn diagnose(
        py: Python<'_>,
        data: &str,
        query: &str,
        format: &str,
        depth: usize,
        timeout: Option<f64>,
        ignore_cartesian_risk: bool,
    ) -> PyResult<DiagnoseTuples> {
        let store = load_store(data, format)?;
        let timeout = parse_timeout_seconds(timeout)?;
        // Releases the GIL for the (potentially long-running, internally
        // multi-threaded) search: without this, a Python-side timeout
        // wrapper around this call couldn't actually regain control until
        // the search finished on its own, since no other Python thread
        // could run while this one held the GIL.
        py.detach(|| diagnose_tuples(&store, query, depth, timeout, ignore_cartesian_risk)).map_err(to_py_err)
    }

    /// Diagnoses `query` and, for each culprit combination found, searches
    /// for a real forward/inverse path (over the graph's actual edges)
    /// connecting each of its triples' bound endpoints, then splices *all*
    /// of them into the query at once and re-runs it to confirm the fix. A
    /// combination is only connected as a whole: if any one of its triples has
    /// no discoverable path, no connected query is built for it (the others
    /// being fixable wouldn't help, since they were only broken *together*).
    ///
    /// `ablation_depth` is passed through to `diagnose` to control how many
    /// triples may be jointly removed while searching for a culprit
    /// (default 3, same as `diagnose`'s `depth`).
    ///
    /// `FILTER` culprits found during diagnosis are included as-is (no
    /// connection is attempted for them).
    ///
    /// Different sampled bound pairs for the same triple can need genuinely
    /// different real paths (e.g. one entity reached via a 2-hop path,
    /// another via an unrelated 1-hop path). By default (`find_all_paths=False`),
    /// path search stops as soon as it finds a connecting path — the
    /// shortest one reachable within `max_depth` across all sampled
    /// endpoints — rather than searching every sampled endpoint for every
    /// distinct path it might individually need. Pass `find_all_paths=True`
    /// to search exhaustively instead: every distinct path found for that
    /// triple is then combined into a single SPARQL alternation (`|`) so the
    /// fix recovers all of them, not just the first found.
    ///
    /// `sample_limit` controls how many bound pairs are considered per
    /// triple (default 500 — a cartesian-risk combination evaluated with
    /// `ignore_cartesian_risk` cross-joins its endpoints, and the reduced
    /// query's row order tends to exhaust one side's matches before moving
    /// to the next, so a small sample can miss the pairing that actually
    /// connects) — pass `None` to consider every distinct pair instead of
    /// stopping early.
    ///
    /// `max_depth` bounds the forward/inverse path search itself. Left as
    /// `None` (the default), it uses depth 2. A triple whose other side
    /// isn't bound anywhere else in the query is skipped entirely (there's
    /// no specific target to search for, only a single anchor with no fixed
    /// goal — not worth searching for a suggestion with nothing to verify
    /// it against).
    ///
    /// `result_limit` caps how many rows a connected query's `LIMIT` allows
    /// (default 50,000 — a connected path, especially an alternation of
    /// several distinct paths, can match far more broadly than the original
    /// triple did); only ever tightens a `LIMIT` already present in the
    /// original query, never loosens it. Pass `None` to leave it unbounded.
    ///
    /// `allowed_namespaces` restricts path search to predicates whose IRI
    /// starts with one of these prefixes; a real edge outside every listed
    /// namespace is invisible to the search even if it would otherwise
    /// connect the two endpoints. Defaults to `DEFAULT_CONNECT_NAMESPACES`
    /// (Brick, ASHRAE 223P, RDFS, QUDT). Pass `None` explicitly for no
    /// restriction (any real predicate found in the graph is fair game).
    ///
    /// `timeout` (seconds) bounds all the work needed to connect *each*
    /// culprit combination — resolving endpoints, the path search itself,
    /// and verifying a candidate fix — not diagnosis, which has its own
    /// separate budget (see `diagnose_timeout`). A combination that can't
    /// finish within its budget falls back to `pruned_query` rather than
    /// hanging or failing the whole call. Defaults to 5.0 seconds; pass
    /// `None` to leave it unbounded.
    ///
    /// `diagnose_timeout` (seconds) is passed straight through to
    /// `diagnose`'s own `timeout` — see its docs for what it bounds and why
    /// an internally-enforced timeout matters even when the caller has its
    /// own external one. Independent of `timeout` above: diagnosis runs
    /// once, before any connection work starts, so the two budgets don't
    /// interact. Also defaults to 5.0 seconds.
    ///
    /// `ignore_cartesian_risk` disables diagnosis's disconnected-pattern
    /// guard for this call: a combination that would otherwise be reported
    /// as a `cartesian_risks` entry and never connected is instead actually
    /// evaluated against `data`, and connected like any other culprit if
    /// confirmed. Defaults to `False`, preserving the guard. Passing `True`
    /// means opting out of the protection that guard applies — a
    /// disconnected BGP can make the query engine materialize a full N×M
    /// cross product before yielding a single row, regardless of `timeout` —
    /// a measured case elsewhere in this project sat for over 200 seconds
    /// and permanently occupied a shared worker thread until the whole
    /// process was killed (see `eval/run_eval.py`'s process-level watchdog
    /// for why that backstop lives at the process level, not inside this
    /// call). Only set this once you've independently judged the risk worth
    /// taking for this specific query/graph, ideally from a process you can
    /// afford to kill outright if a check gets stuck.
    ///
    /// Returns `(original_row_count, results, filter_results,
    /// cartesian_risks)`. Each result is `(found_at_depth, triples,
    /// connected_query, row_count, pruned_query, pruned_row_count)`: `triples`
    /// is a list of `(triple_text, path_text)` for every triple in the
    /// combination (`path_text` is `None` when no connecting path was found
    /// within `max_depth` hops); `connected_query` is the combined fix — every
    /// triple that found a path spliced in, every triple that didn't simply
    /// dropped — or `None` only if *no* triple in the combination had a path
    /// (including one abandoned because `timeout` was exceeded).
    /// `pruned_query` is the original query with every triple in the
    /// combination simply removed (no path substitution) — not a real fix,
    /// but always present and (outside the rare case where `timeout` cuts
    /// off even its own verification) guaranteed non-empty, so it's there
    /// as a fallback when `connected_query` is `None` or still returns
    /// nothing. Each filter result is `(expression_text,
    /// row_count_without_filter)`. `cartesian_risks` is shaped exactly like
    /// `diagnose`'s own (`(triples, depth)`) and means the same thing:
    /// combinations skipped rather than connected because their reduced
    /// pattern was disconnected — always empty when `ignore_cartesian_risk`
    /// was set.
    ///
    /// Builds a throwaway store from `data` on every call — for more than
    /// one query against the same graph, build a `Store` once instead and
    /// call its `diagnose_and_connect` method, which reuses it.
    #[pyfunction]
    #[pyo3(signature = (
        data, query, format="turtle", ablation_depth=3, max_depth=None, sample_limit=500, result_limit=50_000,
        allowed_namespaces=default_connect_namespaces(), timeout=default_connect_timeout(),
        diagnose_timeout=default_ablation_timeout(), ignore_cartesian_risk=false, find_all_paths=false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn diagnose_and_connect(
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
        diagnose_timeout: Option<f64>,
        ignore_cartesian_risk: bool,
        find_all_paths: bool,
    ) -> PyResult<ConnectTuples> {
        let store = load_store(data, format)?;
        let scope = namespace_scope(allowed_namespaces);
        let timeout = parse_timeout_seconds(timeout)?;
        let diagnose_timeout = parse_timeout_seconds(diagnose_timeout)?;
        // See the comment on `diagnose` above: releasing the GIL here is
        // what lets a Python-side timeout wrapper actually regain control if
        // a particular query needs an expensive reduced-query evaluation
        // (e.g. removing a triple leaves the rest of the query essentially
        // unconstrained).
        py.detach(|| {
            let fanout_index = FanoutIndex::build(&store);
            diagnose_and_connect_tuples(
                &store,
                &fanout_index,
                query,
                ablation_depth,
                max_depth,
                sample_limit,
                result_limit,
                scope,
                timeout,
                diagnose_timeout,
                ignore_cartesian_risk,
                find_all_paths,
            )
        })
        .map_err(to_py_err)
    }

    /// An RDF graph loaded once and held for repeated `diagnose`/
    /// `diagnose_and_connect` calls against it.
    ///
    /// The free `diagnose`/`diagnose_and_connect` functions each parse `data`
    /// and build a fresh in-memory store from scratch on every call — fine
    /// for a one-off query, but wasteful for the common case of running many
    /// queries against the same graph (e.g. evaluating a batch of generated
    /// queries against one building's data), where that parse-and-index
    /// work is identical and pointless to repeat. Build a `Store` once and
    /// call its methods instead; they carry the same `depth`/`timeout`/etc.
    /// parameters as their free-function counterparts, just without
    /// `data`/`format`, which were already fixed when the `Store` was built.
    ///
    /// Also builds its `FanoutIndex` (see `sparql-relax-core::fanout`) once
    /// here, for the same reason: it's a one-time, whole-graph scan, and
    /// `diagnose_and_connect`'s path search reuses it on every call rather
    /// than re-scanning the graph per query.
    #[pyclass(name = "Store")]
    struct RdfStore {
        inner: Store,
        fanout_index: FanoutIndex,
    }

    #[pymethods]
    impl RdfStore {
        #[new]
        #[pyo3(signature = (data, format="turtle"))]
        fn new(data: &str, format: &str) -> PyResult<Self> {
            let inner = load_store(data, format)?;
            let fanout_index = FanoutIndex::build(&inner);
            Ok(Self { inner, fanout_index })
        }

        #[pyo3(signature = (query, depth=3, timeout=default_ablation_timeout(), ignore_cartesian_risk=false))]
        fn diagnose(
            &self,
            py: Python<'_>,
            query: &str,
            depth: usize,
            timeout: Option<f64>,
            ignore_cartesian_risk: bool,
        ) -> PyResult<DiagnoseTuples> {
            let timeout = parse_timeout_seconds(timeout)?;
            py.detach(|| diagnose_tuples(&self.inner, query, depth, timeout, ignore_cartesian_risk)).map_err(to_py_err)
        }

        #[pyo3(signature = (
            query, ablation_depth=3, max_depth=None, sample_limit=500, result_limit=50_000,
            allowed_namespaces=default_connect_namespaces(), timeout=default_connect_timeout(),
            diagnose_timeout=default_ablation_timeout(), ignore_cartesian_risk=false, find_all_paths=false
        ))]
        #[allow(clippy::too_many_arguments)]
        fn diagnose_and_connect(
            &self,
            py: Python<'_>,
            query: &str,
            ablation_depth: usize,
            max_depth: Option<usize>,
            sample_limit: Option<usize>,
            result_limit: Option<usize>,
            allowed_namespaces: Option<Vec<String>>,
            timeout: Option<f64>,
            diagnose_timeout: Option<f64>,
            ignore_cartesian_risk: bool,
            find_all_paths: bool,
        ) -> PyResult<ConnectTuples> {
            let scope = namespace_scope(allowed_namespaces);
            let timeout = parse_timeout_seconds(timeout)?;
            let diagnose_timeout = parse_timeout_seconds(diagnose_timeout)?;
            py.detach(|| {
                diagnose_and_connect_tuples(
                    &self.inner,
                    &self.fanout_index,
                    query,
                    ablation_depth,
                    max_depth,
                    sample_limit,
                    result_limit,
                    scope,
                    timeout,
                    diagnose_timeout,
                    ignore_cartesian_risk,
                    find_all_paths,
                )
            })
            .map_err(to_py_err)
        }

        #[pyo3(signature = (query, row_limit=None, timeout=default_query_timeout()))]
        fn query(&self, py: Python<'_>, query: &str, row_limit: Option<usize>, timeout: Option<f64>) -> PyResult<QueryTuple> {
            let timeout = parse_timeout_seconds(timeout)?;
            py.detach(|| query_tuples(&self.inner, query, row_limit, timeout)).map_err(to_py_err)
        }
    }
}
