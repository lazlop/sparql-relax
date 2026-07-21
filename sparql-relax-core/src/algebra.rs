//! Structural rewriting of a parsed SPARQL query's algebra tree.
//!
//! Unlike the Python `sparql_prune`/`sparql_relax`, which regenerate SPARQL
//! text via regex substitution on the original query string, this module
//! rewrites `spargebra`'s typed [`GraphPattern`] tree directly and lets its
//! own `Display` impl regenerate valid SPARQL text — no text-level pattern
//! matching, so it can't be confused by formatting differences.

use spargebra::Query;
use spargebra::algebra::{Expression, GraphPattern, PropertyPathExpression};
use spargebra::term::{NamedNodePattern, TermPattern, TriplePattern};
use std::collections::HashSet;

/// Every triple pattern appearing in a `Bgp` node anywhere in `pattern`.
/// Property-path triples (`GraphPattern::Path`) are not included: they are
/// not plain `TriplePattern`s and are left untouched by this tool.
pub fn collect_bgp_triples(pattern: &GraphPattern) -> Vec<TriplePattern> {
    let mut out = Vec::new();
    collect_rec(pattern, &mut out);
    out
}

fn collect_rec(pattern: &GraphPattern, out: &mut Vec<TriplePattern>) {
    use GraphPattern::*;
    match pattern {
        Bgp { patterns } => out.extend(patterns.iter().cloned()),
        Path { .. } | Values { .. } => {}
        Join { left, right } | Union { left, right } | Minus { left, right } | Lateral { left, right } => {
            collect_rec(left, out);
            collect_rec(right, out);
        }
        LeftJoin { left, right, .. } => {
            collect_rec(left, out);
            collect_rec(right, out);
        }
        Filter { inner, .. }
        | Graph { inner, .. }
        | Extend { inner, .. }
        | OrderBy { inner, .. }
        | Project { inner, .. }
        | Distinct { inner }
        | Reduced { inner }
        | Slice { inner, .. }
        | Group { inner, .. }
        | Service { inner, .. } => collect_rec(inner, out),
    }
}

/// Every `FILTER` expression appearing anywhere in `pattern`: both plain
/// `GraphPattern::Filter` nodes and the optional expression attached to an
/// `OPTIONAL` (`LeftJoin`), e.g. `OPTIONAL { ?s ex:p ?o FILTER(?o > 5) }`.
pub fn collect_filters(pattern: &GraphPattern) -> Vec<Expression> {
    let mut out = Vec::new();
    collect_filters_rec(pattern, &mut out);
    out
}

fn collect_filters_rec(pattern: &GraphPattern, out: &mut Vec<Expression>) {
    use GraphPattern::*;
    match pattern {
        Filter { expr, inner } => {
            out.push(expr.clone());
            collect_filters_rec(inner, out);
        }
        Bgp { .. } | Path { .. } | Values { .. } => {}
        Join { left, right } | Union { left, right } | Minus { left, right } | Lateral { left, right } => {
            collect_filters_rec(left, out);
            collect_filters_rec(right, out);
        }
        LeftJoin { left, right, expression } => {
            if let Some(expr) = expression {
                out.push(expr.clone());
            }
            collect_filters_rec(left, out);
            collect_filters_rec(right, out);
        }
        Graph { inner, .. }
        | Extend { inner, .. }
        | OrderBy { inner, .. }
        | Project { inner, .. }
        | Distinct { inner }
        | Reduced { inner }
        | Slice { inner, .. }
        | Group { inner, .. }
        | Service { inner, .. } => collect_filters_rec(inner, out),
    }
}

/// Removes the first occurrence of `target` from `pattern`, wherever it
/// appears as a `Filter` node's expression or a `LeftJoin`'s optional
/// expression. Returns `None` if `target` isn't present anywhere.
pub fn remove_filter(pattern: &GraphPattern, target: &Expression) -> Option<GraphPattern> {
    use GraphPattern::*;
    match pattern {
        Filter { expr, inner } if expr == target => Some(inner.as_ref().clone()),
        Filter { expr, inner } => {
            remove_filter(inner, target).map(|i| Filter { expr: expr.clone(), inner: Box::new(i) })
        }
        Bgp { .. } | Path { .. } | Values { .. } => None,
        Join { left, right } => remove_filter_binary(left, right, target, |l, r| Join { left: Box::new(l), right: Box::new(r) }),
        Union { left, right } => remove_filter_binary(left, right, target, |l, r| Union { left: Box::new(l), right: Box::new(r) }),
        Minus { left, right } => remove_filter_binary(left, right, target, |l, r| Minus { left: Box::new(l), right: Box::new(r) }),
        Lateral { left, right } => remove_filter_binary(left, right, target, |l, r| Lateral { left: Box::new(l), right: Box::new(r) }),
        LeftJoin { left, right, expression } if expression.as_ref() == Some(target) => Some(LeftJoin {
            left: left.clone(),
            right: right.clone(),
            expression: None,
        }),
        LeftJoin { left, right, expression } => {
            if let Some(l) = remove_filter(left, target) {
                return Some(LeftJoin { left: Box::new(l), right: right.clone(), expression: expression.clone() });
            }
            remove_filter(right, target)
                .map(|r| LeftJoin { left: left.clone(), right: Box::new(r), expression: expression.clone() })
        }
        Graph { name, inner } => remove_filter(inner, target).map(|i| Graph { name: name.clone(), inner: Box::new(i) }),
        Extend { inner, variable, expression } => remove_filter(inner, target)
            .map(|i| Extend { inner: Box::new(i), variable: variable.clone(), expression: expression.clone() }),
        OrderBy { inner, expression } => {
            remove_filter(inner, target).map(|i| OrderBy { inner: Box::new(i), expression: expression.clone() })
        }
        Project { inner, variables } => {
            remove_filter(inner, target).map(|i| Project { inner: Box::new(i), variables: variables.clone() })
        }
        Distinct { inner } => remove_filter(inner, target).map(|i| Distinct { inner: Box::new(i) }),
        Reduced { inner } => remove_filter(inner, target).map(|i| Reduced { inner: Box::new(i) }),
        Slice { inner, start, length } => {
            remove_filter(inner, target).map(|i| Slice { inner: Box::new(i), start: *start, length: *length })
        }
        Group { inner, variables, aggregates } => remove_filter(inner, target)
            .map(|i| Group { inner: Box::new(i), variables: variables.clone(), aggregates: aggregates.clone() }),
        Service { name, inner, silent } => remove_filter(inner, target)
            .map(|i| Service { name: name.clone(), inner: Box::new(i), silent: *silent }),
    }
}

fn remove_filter_binary(
    left: &GraphPattern,
    right: &GraphPattern,
    target: &Expression,
    rebuild: impl Fn(GraphPattern, GraphPattern) -> GraphPattern,
) -> Option<GraphPattern> {
    if let Some(l) = remove_filter(left, target) {
        return Some(rebuild(l, right.clone()));
    }
    remove_filter(right, target).map(|r| rebuild(left.clone(), r))
}

/// Result of trying to rewrite one occurrence of a target triple somewhere
/// under a `GraphPattern` subtree.
enum Rewrite {
    /// The target triple does not appear anywhere in this subtree.
    NotFound,
    /// The target triple was found and rewritten. `None` means the whole
    /// subtree collapsed to the trivial "one empty solution" identity
    /// (e.g. a single-triple `Bgp` with its only triple removed).
    Found(Option<GraphPattern>),
}

/// Walks `pattern` looking for `target` inside a `Bgp` node; `on_match` is
/// called with that `Bgp`'s triple list and the index of `target` within it,
/// and decides what replaces the `Bgp` node (removal vs. path substitution
/// are the only two callers, differing only in this leaf behavior).
fn rewrite_rec(
    pattern: &GraphPattern,
    target: &TriplePattern,
    on_match: &impl Fn(&[TriplePattern], usize) -> Option<GraphPattern>,
) -> Rewrite {
    use GraphPattern::*;
    match pattern {
        Bgp { patterns } => match patterns.iter().position(|t| t == target) {
            Some(idx) => Rewrite::Found(on_match(patterns, idx)),
            None => Rewrite::NotFound,
        },
        Path { .. } | Values { .. } => Rewrite::NotFound,
        Join { left, right } => rewrite_binary(left, right, target, on_match, |l, r| Join {
            left: Box::new(l),
            right: Box::new(r),
        }),
        Union { left, right } => rewrite_binary(left, right, target, on_match, |l, r| Union {
            left: Box::new(l),
            right: Box::new(r),
        }),
        Minus { left, right } => rewrite_binary(left, right, target, on_match, |l, r| Minus {
            left: Box::new(l),
            right: Box::new(r),
        }),
        Lateral { left, right } => rewrite_binary(left, right, target, on_match, |l, r| Lateral {
            left: Box::new(l),
            right: Box::new(r),
        }),
        LeftJoin { left, right, expression } => {
            rewrite_binary(left, right, target, on_match, |l, r| LeftJoin {
                left: Box::new(l),
                right: Box::new(r),
                expression: expression.clone(),
            })
        }
        Filter { expr, inner } => {
            rewrite_unary(inner, target, on_match, |i| Filter { expr: expr.clone(), inner: Box::new(i) })
        }
        Graph { name, inner } => {
            rewrite_unary(inner, target, on_match, |i| Graph { name: name.clone(), inner: Box::new(i) })
        }
        Extend { inner, variable, expression } => rewrite_unary(inner, target, on_match, |i| Extend {
            inner: Box::new(i),
            variable: variable.clone(),
            expression: expression.clone(),
        }),
        OrderBy { inner, expression } => {
            rewrite_unary(inner, target, on_match, |i| OrderBy { inner: Box::new(i), expression: expression.clone() })
        }
        Project { inner, variables } => {
            rewrite_unary(inner, target, on_match, |i| Project { inner: Box::new(i), variables: variables.clone() })
        }
        Distinct { inner } => rewrite_unary(inner, target, on_match, |i| Distinct { inner: Box::new(i) }),
        Reduced { inner } => rewrite_unary(inner, target, on_match, |i| Reduced { inner: Box::new(i) }),
        Slice { inner, start, length } => rewrite_unary(inner, target, on_match, |i| Slice {
            inner: Box::new(i),
            start: *start,
            length: *length,
        }),
        Group { inner, variables, aggregates } => rewrite_unary(inner, target, on_match, |i| Group {
            inner: Box::new(i),
            variables: variables.clone(),
            aggregates: aggregates.clone(),
        }),
        Service { name, inner, silent } => rewrite_unary(inner, target, on_match, |i| Service {
            name: name.clone(),
            inner: Box::new(i),
            silent: *silent,
        }),
    }
}

fn rewrite_binary(
    left: &GraphPattern,
    right: &GraphPattern,
    target: &TriplePattern,
    on_match: &impl Fn(&[TriplePattern], usize) -> Option<GraphPattern>,
    rebuild: impl Fn(GraphPattern, GraphPattern) -> GraphPattern,
) -> Rewrite {
    match rewrite_rec(left, target, on_match) {
        Rewrite::Found(new_left) => {
            return Rewrite::Found(Some(match new_left {
                None => right.clone(),
                Some(l) => rebuild(l, right.clone()),
            }));
        }
        Rewrite::NotFound => {}
    }
    match rewrite_rec(right, target, on_match) {
        Rewrite::Found(new_right) => Rewrite::Found(Some(match new_right {
            None => left.clone(),
            Some(r) => rebuild(left.clone(), r),
        })),
        Rewrite::NotFound => Rewrite::NotFound,
    }
}

fn rewrite_unary(
    inner: &GraphPattern,
    target: &TriplePattern,
    on_match: &impl Fn(&[TriplePattern], usize) -> Option<GraphPattern>,
    rebuild: impl Fn(GraphPattern) -> GraphPattern,
) -> Rewrite {
    match rewrite_rec(inner, target, on_match) {
        Rewrite::Found(None) => Rewrite::Found(None),
        Rewrite::Found(Some(new_inner)) => Rewrite::Found(Some(rebuild(new_inner))),
        Rewrite::NotFound => Rewrite::NotFound,
    }
}

/// Removes `target` from `pattern` wherever it appears in a `Bgp`. Returns
/// `None` if `target` isn't present anywhere.
pub fn remove_triple(pattern: &GraphPattern, target: &TriplePattern) -> Option<GraphPattern> {
    let on_match = |patterns: &[TriplePattern], idx: usize| -> Option<GraphPattern> {
        let mut remaining = patterns.to_vec();
        remaining.remove(idx);
        if remaining.is_empty() { None } else { Some(GraphPattern::Bgp { patterns: remaining }) }
    };
    match rewrite_rec(pattern, target, &on_match) {
        Rewrite::Found(inner) => Some(inner.unwrap_or(GraphPattern::Bgp { patterns: Vec::new() })),
        Rewrite::NotFound => None,
    }
}

/// Replaces `target` in `pattern` with a `Path` node using `path` in place
/// of its predicate. Returns `None` if `target` isn't present anywhere.
pub fn replace_triple_with_path(
    pattern: &GraphPattern,
    target: &TriplePattern,
    path: PropertyPathExpression,
) -> Option<GraphPattern> {
    let on_match = |patterns: &[TriplePattern], idx: usize| -> Option<GraphPattern> {
        let mut remaining = patterns.to_vec();
        remaining.remove(idx);
        let path_node = GraphPattern::Path {
            subject: target.subject.clone(),
            path: path.clone(),
            object: target.object.clone(),
        };
        Some(if remaining.is_empty() {
            path_node
        } else {
            GraphPattern::Join { left: Box::new(GraphPattern::Bgp { patterns: remaining }), right: Box::new(path_node) }
        })
    };
    match rewrite_rec(pattern, target, &on_match) {
        Rewrite::Found(inner) => inner,
        Rewrite::NotFound => None,
    }
}

/// Rebuilds a `Query` with the same form (`SELECT`/`ASK`/...), dataset, and
/// base IRI as `query`, but with its top-level pattern replaced.
pub fn with_pattern(query: &Query, new_pattern: GraphPattern) -> Query {
    match query {
        Query::Select { dataset, base_iri, .. } => {
            Query::Select { dataset: dataset.clone(), pattern: new_pattern, base_iri: base_iri.clone() }
        }
        Query::Construct { template, dataset, base_iri, .. } => Query::Construct {
            template: template.clone(),
            dataset: dataset.clone(),
            pattern: new_pattern,
            base_iri: base_iri.clone(),
        },
        Query::Describe { dataset, base_iri, .. } => {
            Query::Describe { dataset: dataset.clone(), pattern: new_pattern, base_iri: base_iri.clone() }
        }
        Query::Ask { dataset, base_iri, .. } => {
            Query::Ask { dataset: dataset.clone(), pattern: new_pattern, base_iri: base_iri.clone() }
        }
    }
}

/// Wraps `pattern` with a `LIMIT` of at most `limit` results. If `pattern`
/// already has an outer `Slice` (an explicit `LIMIT`/`OFFSET` in the
/// original query), only tightens it — never loosens an already-smaller
/// limit — rather than nesting a redundant second `Slice`.
pub fn with_limit(pattern: GraphPattern, limit: usize) -> GraphPattern {
    if let GraphPattern::Slice { inner, start, length } = pattern {
        let tightened = Some(length.map_or(limit, |l| l.min(limit)));
        return GraphPattern::Slice { inner, start, length: tightened };
    }
    GraphPattern::Slice { inner: Box::new(pattern), start: 0, length: Some(limit) }
}

/// Builds an `ASK` query with `pattern` as its body, carrying over `query`'s
/// dataset and base IRI regardless of `query`'s own form. Used where only
/// existence of a solution matters (not the actual bindings): `ASK`
/// evaluation short-circuits on the first matching solution, which is
/// strictly cheaper than a `SELECT` that has to be told to stop separately.
pub fn ask_query(query: &Query, pattern: GraphPattern) -> Query {
    match query {
        Query::Select { dataset, base_iri, .. }
        | Query::Construct { dataset, base_iri, .. }
        | Query::Describe { dataset, base_iri, .. }
        | Query::Ask { dataset, base_iri, .. } => {
            Query::Ask { dataset: dataset.clone(), pattern, base_iri: base_iri.clone() }
        }
    }
}

/// The top-level `GraphPattern` of any query form.
pub fn pattern_of(query: &Query) -> &GraphPattern {
    match query {
        Query::Select { pattern, .. }
        | Query::Construct { pattern, .. }
        | Query::Describe { pattern, .. }
        | Query::Ask { pattern, .. } => pattern,
    }
}

/// Every variable/blank-node identifier `triple` mentions, as join keys
/// (`?name` for a variable, `_:name` for a blank node — prefixed so a
/// variable and a blank node that happen to share a spelling are never
/// mistaken for the same join key). A concrete term (IRI/literal) on any
/// side contributes nothing: it can't be joined *on*, only matched.
fn triple_join_keys(triple: &TriplePattern) -> impl Iterator<Item = String> {
    fn term_key(term: &TermPattern) -> Option<String> {
        match term {
            TermPattern::Variable(v) => Some(format!("?{}", v.as_str())),
            TermPattern::BlankNode(b) => Some(format!("_:{}", b.as_str())),
            _ => None,
        }
    }
    let subject = term_key(&triple.subject);
    let predicate = match &triple.predicate {
        NamedNodePattern::Variable(v) => Some(format!("?{}", v.as_str())),
        NamedNodePattern::NamedNode(_) => None,
    };
    let object = term_key(&triple.object);
    subject.into_iter().chain(predicate).chain(object)
}

/// Whether `triples`, evaluated together as a single basic graph pattern,
/// would force a cartesian product — i.e. its variable/blank-node-sharing
/// graph has more than one connected component, so at least one triple's
/// join keys never overlap, even transitively through a chain of other
/// triples, with another's. This is exactly the shape that can make a query
/// engine materialize a full N×M cross product before yielding a single row
/// (see the `diagnose`/`relax` module docs on why an internally-enforced
/// timeout doesn't reliably bound that).
///
/// A triple with no variables or blank nodes at all (every side concrete)
/// never joins with anything and only ever filters existence rather than
/// multiplying results, so it's ignored entirely rather than counted as a
/// second, disconnected component on its own. `triples.len() < 2` is always
/// `false` for the same reason — nothing to be disconnected *from* yet.
pub(crate) fn has_cartesian_join(triples: &[TriplePattern]) -> bool {
    let joinable: Vec<&TriplePattern> = triples.iter().filter(|t| triple_join_keys(t).next().is_some()).collect();
    if joinable.len() < 2 {
        return false;
    }

    let mut parent: Vec<usize> = (0..joinable.len()).collect();
    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }
    fn union(parent: &mut [usize], a: usize, b: usize) {
        let (ra, rb) = (find(parent, a), find(parent, b));
        if ra != rb {
            parent[ra] = rb;
        }
    }

    let keys: Vec<HashSet<String>> = joinable.iter().map(|t| triple_join_keys(t).collect()).collect();
    for i in 0..joinable.len() {
        for j in (i + 1)..joinable.len() {
            if keys[i].intersection(&keys[j]).next().is_some() {
                union(&mut parent, i, j);
            }
        }
    }

    let root0 = find(&mut parent, 0);
    (1..joinable.len()).any(|i| find(&mut parent, i) != root0)
}
