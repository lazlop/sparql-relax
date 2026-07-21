//! Real graph-path search: rather than trying a curated list of candidate
//! predicates (as the Python `sparql_relax` does), this walks Oxigraph's
//! actual triples outward from a bound endpoint, trying both a real
//! outgoing edge (forward, `<p>`) and a real incoming edge (inverse, `^<p>`)
//! at every step, to find a sequence of hops that actually connects two
//! bound nodes. An optional namespace allowlist (`allowed_namespaces`) can
//! further restrict which real edges are eligible to walk at all — a
//! predicate outside every listed namespace is invisible to the search,
//! even if it would otherwise connect the two nodes.
//!
//! This never touches the SPARQL query engine (it's direct `Store` quad
//! lookups), so it can't use Oxigraph's `CancellationToken` the way
//! [`crate::diagnose`]/[`crate::relax`]'s query executions do. A `deadline`
//! is checked by hand instead — every [`DEADLINE_CHECK_INTERVAL`] edges
//! examined, not just once per frontier node — cheap relative to the quad
//! lookups themselves, but frequent enough that even a single high-fan-out
//! hub node (common in real building graphs — a `building` entity wired to
//! hundreds or thousands of sensors, say) can't run past its budget
//! unnoticed the way it could if the check only happened once *before* that
//! one node's (potentially huge) edge set was walked.

use oxigraph::model::{GraphNameRef, NamedNode, NamedOrBlankNode, Term};
use oxigraph::store::Store;
use spargebra::algebra::PropertyPathExpression;
use std::collections::HashSet;
use std::time::Instant;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Hop {
    Forward(NamedNode),
    Inverse(NamedNode),
}

/// How many edges [`find_path`] examines between deadline checks. Small
/// enough that a hub node with a huge fan-out still gets cut off close to
/// the deadline rather than only after its entire edge set is walked; large
/// enough that `Instant::now()` isn't called on every single edge.
const DEADLINE_CHECK_INTERVAL: usize = 256;

/// Bounded-depth breadth-first search from `start` to `goal` over the
/// store's real edges, in either direction. Returns the shortest hop
/// sequence found (`Some(vec![])` if `start == goal` already), or `None`
/// if no path exists within `max_depth` hops — or if `deadline` passes
/// before the search finishes; a search cut off partway through hasn't
/// proven there's no path, only that none was found *yet*, so this stays
/// consistent with the `max_depth`-exhausted case rather than erroring
/// (see the module docs on why `deadline` is checked by hand here).
/// `allowed_namespaces` restricts which predicates are eligible hops (see
/// [`neighbors`]); `None` means any real predicate is fair game.
pub fn find_path(
    store: &Store,
    start: &Term,
    goal: &Term,
    max_depth: usize,
    allowed_namespaces: Option<&[String]>,
    deadline: Option<Instant>,
) -> Option<Vec<Hop>> {
    if start == goal {
        return Some(Vec::new());
    }

    let mut visited: HashSet<Term> = HashSet::from([start.clone()]);
    let mut frontier: Vec<(Term, Vec<Hop>)> = vec![(start.clone(), Vec::new())];
    let mut edges_since_check = 0usize;

    for _ in 0..max_depth {
        let mut next_frontier = Vec::new();
        for (node, path) in &frontier {
            // Checked once before expanding *this* node too (not just
            // periodically inside its edge loop below) — otherwise an
            // already-past deadline could go unnoticed on a graph small
            // enough that no single node's edges ever reach
            // `DEADLINE_CHECK_INTERVAL`.
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return None;
            }
            for (hop, neighbor) in neighbors(store, node, allowed_namespaces) {
                edges_since_check += 1;
                if edges_since_check >= DEADLINE_CHECK_INTERVAL {
                    edges_since_check = 0;
                    if deadline.is_some_and(|d| Instant::now() >= d) {
                        return None;
                    }
                }
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
        // Collected via `find_map` (rather than a `for` loop assigning
        // `current` directly) so the iterator — which borrows `current` —
        // is dropped before `current` is reassigned below.
        let Some(next) = neighbors(store, &current, None).find_map(|(candidate_hop, neighbor)| (&candidate_hop == hop).then_some(neighbor))
        else {
            return false;
        };
        current = next;
    }
    current == *goal
}

/// Every real forward/inverse edge out of `node`, optionally restricted to
/// predicates under one of `allowed_namespaces`' prefixes (`None` allows
/// any predicate).
///
/// Lazy rather than collected into a `Vec` up front: [`find_path`] checks
/// its deadline as it consumes this iterator, so a hub node with a huge
/// fan-out can still be cut off partway through instead of the whole edge
/// set being materialized (and the store scanned in full) before the first
/// check would even happen.
fn neighbors<'a>(store: &'a Store, node: &'a Term, allowed_namespaces: Option<&'a [String]>) -> impl Iterator<Item = (Hop, Term)> + 'a {
    let forward = as_subject(node).into_iter().flat_map(move |subject| {
        store
            .quads_for_pattern(Some(subject.as_ref()), None, None, Some(GraphNameRef::DefaultGraph))
            .flatten()
            .filter(move |quad| predicate_allowed(&quad.predicate, allowed_namespaces))
            .map(|quad| (Hop::Forward(quad.predicate), quad.object))
    });

    let inverse = store
        .quads_for_pattern(None, None, Some(node.as_ref()), Some(GraphNameRef::DefaultGraph))
        .flatten()
        .filter(move |quad| predicate_allowed(&quad.predicate, allowed_namespaces))
        .map(|quad| (Hop::Inverse(quad.predicate), Term::from(quad.subject)));

    forward.chain(inverse)
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
