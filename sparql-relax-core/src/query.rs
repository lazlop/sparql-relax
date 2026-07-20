//! Direct execution of an arbitrary SPARQL query (any form — `SELECT`,
//! `ASK`, `CONSTRUCT`, `DESCRIBE`).
//!
//! [`crate::diagnose`] and [`crate::relax`] exist to explain and fix a query
//! that returns nothing (or wrongly); this module is the ordinary case of
//! running one that already works and wanting its actual results. It has no
//! knowledge of ablation or path search — just parse, run, convert.

use crate::algebra::{pattern_of, with_limit, with_pattern};
use crate::error::{RelaxError, Result};
use oxigraph::model::{Term, Triple};
use oxigraph::sparql::{CancellationToken, QueryEvaluationError, QueryResults, SparqlEvaluator};
use oxigraph::store::Store;
use spargebra::Query;
use spargebra::SparqlParser;
use std::thread;
use std::time::{Duration, Instant};

/// A single RDF term, decomposed into plain data rather than exposing
/// Oxigraph's own [`Term`] to callers outside this crate — keeps language
/// bindings (e.g. the Python layer) independent of the store crate's types.
#[derive(Debug, Clone, PartialEq)]
pub enum RdfTerm {
    Iri(String),
    BlankNode(String),
    Literal { value: String, datatype: String, language: Option<String> },
}

impl From<Term> for RdfTerm {
    fn from(term: Term) -> Self {
        match term {
            Term::NamedNode(n) => RdfTerm::Iri(n.into_string()),
            Term::BlankNode(b) => RdfTerm::BlankNode(b.into_string()),
            Term::Literal(l) => {
                let language = l.language().map(ToString::to_string);
                let datatype = l.datatype().as_str().to_string();
                RdfTerm::Literal { value: l.value().to_string(), datatype, language }
            }
            // RDF-star quoted triples only exist behind the `rdf-12`
            // feature (not enabled for this crate); this arm is here so the
            // conversion still compiles unconditionally rather than growing
            // a fourth `RdfTerm` variant every caller would need to handle
            // for a case this tool's graphs never produce. If the feature
            // is ever enabled, a quoted triple is rendered as its SPARQL
            // text form instead of being modeled structurally.
            #[allow(unreachable_patterns)]
            other => RdfTerm::Iri(other.to_string()),
        }
    }
}

/// One triple in a `CONSTRUCT`/`DESCRIBE` result graph.
#[derive(Debug, Clone, PartialEq)]
pub struct ResultTriple {
    pub subject: RdfTerm,
    pub predicate: RdfTerm,
    pub object: RdfTerm,
}

impl From<Triple> for ResultTriple {
    fn from(triple: Triple) -> Self {
        ResultTriple { subject: Term::from(triple.subject).into(), predicate: Term::from(triple.predicate).into(), object: triple.object.into() }
    }
}

/// Default `timeout` for [`query_default`]: generous relative to
/// [`crate::diagnose::DEFAULT_ABLATION_TIMEOUT`], since a query run through
/// this path is normally one diagnosis has already confirmed works and the
/// caller now wants full results for, not a cheap yes/no ablation check.
pub const DEFAULT_QUERY_TIMEOUT: Duration = Duration::from_secs(10);

/// The result of running an arbitrary SPARQL query, shaped by its form.
#[derive(Debug, Clone, PartialEq)]
pub enum QueryOutcome {
    /// `SELECT` results. `variables` gives the projected column order;
    /// each row in `rows` aligns to it position-for-position, with `None`
    /// wherever that variable was left unbound in that particular row.
    Solutions { variables: Vec<String>, rows: Vec<Vec<Option<RdfTerm>>> },
    /// An `ASK` query's single boolean result.
    Boolean(bool),
    /// `CONSTRUCT`/`DESCRIBE` results: the graph they produced.
    Graph(Vec<ResultTriple>),
}

/// Runs `query_text` (any SPARQL query form) against `store`.
///
/// `row_limit` caps how many rows a `SELECT`/`CONSTRUCT`/`DESCRIBE` result
/// may return. It's applied as a `LIMIT` on the query itself before
/// evaluation — only ever tightening a `LIMIT` already present, never
/// loosening it (see [`crate::algebra::with_limit`]) — so an oversized
/// result is bounded during evaluation rather than computed in full and
/// truncated afterward. Has no effect on `ASK`, which only ever returns one
/// boolean. Pass `None` to leave it unbounded.
///
/// `timeout` (seconds) bounds evaluation; a query that doesn't finish in
/// time returns [`RelaxError::QueryTimeout`] rather than continuing to run
/// unobserved. This differs from [`crate::diagnose::diagnose`]'s ablation
/// checks, where "ran out of time" is a reasonable "not a culprit" default
/// to fall back on — here, a direct query timing out *is* the answer the
/// caller needs to see, so it's surfaced as an error rather than swallowed.
/// Pass `None` to leave it unbounded.
pub fn query(query_text: &str, store: &Store, row_limit: Option<usize>, timeout: Option<Duration>) -> Result<QueryOutcome> {
    let parsed = SparqlParser::new().parse_query(query_text)?;
    let parsed = match (row_limit, &parsed) {
        (Some(_), Query::Ask { .. }) | (None, _) => parsed,
        (Some(limit), _) => with_pattern(&parsed, with_limit(pattern_of(&parsed).clone(), limit)),
    };

    let Some(deadline) = timeout.map(|t| Instant::now() + t) else {
        return execute(parsed, store, None);
    };
    match deadline_token(deadline) {
        Some(token) => execute(parsed, store, Some(token)),
        // Deadline already in the past: nothing left to run.
        None => Err(RelaxError::QueryTimeout),
    }
}

/// Runs `query_text` against `store` with no `row_limit` and
/// [`DEFAULT_QUERY_TIMEOUT`]. A convenient default for callers who don't
/// need to tune either.
pub fn query_default(query_text: &str, store: &Store) -> Result<QueryOutcome> {
    query(query_text, store, None, Some(DEFAULT_QUERY_TIMEOUT))
}

fn execute(parsed: Query, store: &Store, cancellation_token: Option<CancellationToken>) -> Result<QueryOutcome> {
    let mut evaluator = SparqlEvaluator::new();
    if let Some(token) = cancellation_token {
        evaluator = evaluator.with_cancellation_token(token);
    }
    let results = match evaluator.for_query(parsed).on_store(store).execute() {
        Ok(results) => results,
        Err(QueryEvaluationError::Cancelled) => return Err(RelaxError::QueryTimeout),
        Err(e) => return Err(RelaxError::Evaluation(e)),
    };
    match results {
        QueryResults::Boolean(b) => Ok(QueryOutcome::Boolean(b)),
        QueryResults::Solutions(iter) => {
            let variables: Vec<String> = iter.variables().iter().map(|v| v.as_str().to_string()).collect();
            let mut rows = Vec::new();
            for solution in iter {
                let solution = match solution {
                    Ok(s) => s,
                    Err(QueryEvaluationError::Cancelled) => return Err(RelaxError::QueryTimeout),
                    Err(e) => return Err(RelaxError::Evaluation(e)),
                };
                rows.push(solution.values().iter().map(|v| v.clone().map(RdfTerm::from)).collect());
            }
            Ok(QueryOutcome::Solutions { variables, rows })
        }
        QueryResults::Graph(iter) => {
            let mut triples = Vec::new();
            for triple in iter {
                let triple = match triple {
                    Ok(t) => t,
                    Err(QueryEvaluationError::Cancelled) => return Err(RelaxError::QueryTimeout),
                    Err(e) => return Err(RelaxError::Evaluation(e)),
                };
                triples.push(ResultTriple::from(triple));
            }
            Ok(QueryOutcome::Graph(triples))
        }
    }
}

/// A cancellation token a background timer will cancel once `deadline`
/// passes, or `None` if `deadline` is already in the past. Backed by
/// Oxigraph's own `CancellationToken`, checked by the query engine on every
/// underlying quad lookup — a real mid-evaluation abort, not just a wrapper
/// that gives up waiting while the query keeps running in the background.
///
/// Deliberately not shared with [`crate::diagnose`]'s own copy of this
/// helper: that one backs `SharedDeadline`, which amortizes one timer
/// across many ablation checks sharing a single budget — a concern this
/// module, which only ever runs one query per call, doesn't have.
fn deadline_token(deadline: Instant) -> Option<CancellationToken> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        return None;
    }
    let cancellation_token = CancellationToken::new();
    let timer_token = cancellation_token.clone();
    thread::spawn(move || {
        thread::sleep(remaining);
        timer_token.cancel();
    });
    Some(cancellation_token)
}
