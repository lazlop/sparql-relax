use oxigraph::sparql::QueryEvaluationError;
use spargebra::SparqlSyntaxError;

#[derive(thiserror::Error, Debug)]
pub enum RelaxError {
    #[error("SPARQL syntax error: {0}")]
    Syntax(#[from] SparqlSyntaxError),
    #[error("SPARQL evaluation error: {0}")]
    Evaluation(#[from] QueryEvaluationError),
    #[error("only SELECT queries are supported, got: {0}")]
    UnsupportedQueryForm(&'static str),
    #[error("query has no basic graph pattern triples to diagnose")]
    NoTriples,
    #[error("culprit triple {0:?} could not be located while relaxing the query it was diagnosed from")]
    CulpritNotFound(String),
    #[error("diagnosis timed out before the original query could even be evaluated")]
    Timeout,
    #[error("query timed out")]
    QueryTimeout,
}

pub type Result<T> = std::result::Result<T, RelaxError>;
