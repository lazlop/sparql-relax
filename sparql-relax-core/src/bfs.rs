//! Real graph-path search: rather than trying a curated list of candidate
//! predicates (as the Python `sparql_relax` does), this walks Oxigraph's
//! actual triples outward from a bound endpoint, trying both a real
//! outgoing edge (forward, `<p>`) and a real incoming edge (inverse, `^<p>`)
//! at every step, to find a sequence of hops that actually connects two
//! bound nodes. An optional namespace allowlist (`allowed_namespaces`) can
//! further restrict which real edges are eligible to walk at all — a
//! predicate outside every listed namespace is invisible to the search,
//! even if it would otherwise connect the two nodes.

use oxigraph::model::{GraphNameRef, NamedNode, NamedOrBlankNode, Term};
use oxigraph::store::Store;
use spargebra::algebra::PropertyPathExpression;
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Hop {
    Forward(NamedNode),
    Inverse(NamedNode),
}

/// Bounded-depth breadth-first search from `start` to `goal` over the
/// store's real edges, in either direction. Returns the shortest hop
/// sequence found (`Some(vec![])` if `start == goal` already), or `None`
/// if no path exists within `max_depth` hops. `allowed_namespaces` restricts
/// which predicates are eligible hops (see [`neighbors`]); `None` means any
/// real predicate is fair game.
pub fn find_path(store: &Store, start: &Term, goal: &Term, max_depth: usize, allowed_namespaces: Option<&[String]>) -> Option<Vec<Hop>> {
    if start == goal {
        return Some(Vec::new());
    }

    let mut visited: HashSet<Term> = HashSet::from([start.clone()]);
    let mut frontier: Vec<(Term, Vec<Hop>)> = vec![(start.clone(), Vec::new())];

    for _ in 0..max_depth {
        let mut next_frontier = Vec::new();
        for (node, path) in &frontier {
            for (hop, neighbor) in neighbors(store, node, allowed_namespaces) {
                if &neighbor == goal {
                    let mut full = path.clone();
                    full.push(hop);
                    return Some(full);
                }
                if visited.insert(neighbor.clone()) {
                    let mut full = path.clone();
                    full.push(hop);
                    next_frontier.push((neighbor, full));
                }
            }
        }
        if next_frontier.is_empty() {
            return None;
        }
        frontier = next_frontier;
    }
    None
}

/// Verifies that following `hops` from `start` (step by step, over the
/// store's real edges) actually lands on `goal`. Used to check whether a
/// hop sequence discovered for one bound pair generalizes to another.
pub fn path_holds(store: &Store, start: &Term, goal: &Term, hops: &[Hop]) -> bool {
    let mut current = start.clone();
    for hop in hops {
        let mut stepped = false;
        for (candidate_hop, neighbor) in neighbors(store, &current, None) {
            if &candidate_hop == hop {
                current = neighbor;
                stepped = true;
                break;
            }
        }
        if !stepped {
            return false;
        }
    }
    current == *goal
}

/// Every real forward/inverse edge out of `node`, optionally restricted to
/// predicates under one of `allowed_namespaces`' prefixes (`None` allows
/// any predicate).
fn neighbors(store: &Store, node: &Term, allowed_namespaces: Option<&[String]>) -> Vec<(Hop, Term)> {
    let mut out = Vec::new();

    if let Some(subject) = as_subject(node) {
        for quad in store.quads_for_pattern(Some(subject.as_ref()), None, None, Some(GraphNameRef::DefaultGraph)).flatten() {
            if predicate_allowed(&quad.predicate, allowed_namespaces) {
                out.push((Hop::Forward(quad.predicate), quad.object));
            }
        }
    }

    for quad in store.quads_for_pattern(None, None, Some(node.as_ref()), Some(GraphNameRef::DefaultGraph)).flatten() {
        if predicate_allowed(&quad.predicate, allowed_namespaces) {
            out.push((Hop::Inverse(quad.predicate), Term::from(quad.subject)));
        }
    }

    out
}

fn predicate_allowed(predicate: &NamedNode, allowed_namespaces: Option<&[String]>) -> bool {
    match allowed_namespaces {
        None => true,
        Some(namespaces) => namespaces.iter().any(|ns| predicate.as_str().starts_with(ns.as_str())),
    }
}

fn as_subject(term: &Term) -> Option<NamedOrBlankNode> {
    match term {
        Term::NamedNode(n) => Some(NamedOrBlankNode::NamedNode(n.clone())),
        Term::BlankNode(b) => Some(NamedOrBlankNode::BlankNode(b.clone())),
        Term::Literal(_) => None,
    }
}

/// Folds a hop sequence into a `PropertyPathExpression`, e.g.
/// `[Forward(p1), Inverse(p2)]` becomes `<p1>/^<p2>`. Returns `None` for an
/// empty sequence (start already equals goal; no path is needed).
pub fn path_to_property_path(hops: &[Hop]) -> Option<PropertyPathExpression> {
    let mut iter = hops.iter();
    let mut expr = hop_to_expr(iter.next()?);
    for hop in iter {
        expr = PropertyPathExpression::Sequence(Box::new(expr), Box::new(hop_to_expr(hop)));
    }
    Some(expr)
}

fn hop_to_expr(hop: &Hop) -> PropertyPathExpression {
    match hop {
        Hop::Forward(p) => PropertyPathExpression::NamedNode(p.clone()),
        Hop::Inverse(p) => PropertyPathExpression::Reverse(Box::new(PropertyPathExpression::NamedNode(p.clone()))),
    }
}
