//! Real graph-path search: rather than trying a curated list of candidate
//! predicates (as the Python `sparql_relax` does), this walks Oxigraph's
//! actual triples outward from a bound endpoint, trying both a real
//! outgoing edge (forward, `<p>`) and a real incoming edge (inverse, `^<p>`)
//! at every step, to find a sequence of hops that actually connects two
//! bound nodes.

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
/// if no path exists within `max_depth` hops.
pub fn find_path(store: &Store, start: &Term, goal: &Term, max_depth: usize) -> Option<Vec<Hop>> {
    if start == goal {
        return Some(Vec::new());
    }

    let mut visited: HashSet<Term> = HashSet::from([start.clone()]);
    let mut frontier: Vec<(Term, Vec<Hop>)> = vec![(start.clone(), Vec::new())];

    for _ in 0..max_depth {
        let mut next_frontier = Vec::new();
        for (node, path) in &frontier {
            for (hop, neighbor) in neighbors(store, node) {
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

/// Caps how many distinct hop-sequence shapes [`explore_from`] returns, so
/// exploring outward from a well-connected node without a fixed goal can't
/// blow up.
const MAX_UNDIRECTED_CANDIDATES: usize = 20;

/// Explores outward from `anchor` over the store's real edges (forward and
/// inverse) with no fixed goal, up to `max_depth` hops, and returns the
/// shortest hop sequence reaching each distinct node found (standard BFS
/// dedup by node, not by hop-sequence text — without this, a real inverse
/// edge back toward an ancestor, e.g. `<p>` then `^<p>`, would otherwise
/// show up as a spurious "alternative" that doesn't actually lead anywhere
/// new). Used when a broken triple's other side isn't bound anywhere else
/// in the query, so there's no specific target to search for: these are
/// offered as suggested alternatives related to `anchor`, not verified
/// fixes.
pub fn explore_from(store: &Store, anchor: &Term, max_depth: usize) -> Vec<Vec<Hop>> {
    let mut results: Vec<Vec<Hop>> = Vec::new();
    let mut expanded: HashSet<Term> = HashSet::from([anchor.clone()]);
    let mut frontier: Vec<(Term, Vec<Hop>)> = vec![(anchor.clone(), Vec::new())];

    'depth: for _ in 0..max_depth {
        let mut next_frontier = Vec::new();
        for (node, path) in &frontier {
            for (hop, neighbor) in neighbors(store, node) {
                if !expanded.insert(neighbor.clone()) {
                    continue; // already reached via an earlier, shorter-or-equal path
                }
                let mut new_path = path.clone();
                new_path.push(hop);
                results.push(new_path.clone());
                if results.len() >= MAX_UNDIRECTED_CANDIDATES {
                    break 'depth;
                }
                next_frontier.push((neighbor, new_path));
            }
        }
        if next_frontier.is_empty() {
            break;
        }
        frontier = next_frontier;
    }

    results.sort_by_key(Vec::len);
    results
}

/// Reverses a hop sequence and flips each step's direction, e.g.
/// `[Forward(p1), Inverse(p2)]` becomes `[Forward(p2), Inverse(p1)]`. Used
/// when a path was searched outward from an object anchor (no subject
/// available) but needs to be phrased as subject → object.
pub fn reverse_hops(hops: &[Hop]) -> Vec<Hop> {
    hops.iter()
        .rev()
        .map(|hop| match hop {
            Hop::Forward(p) => Hop::Inverse(p.clone()),
            Hop::Inverse(p) => Hop::Forward(p.clone()),
        })
        .collect()
}

/// Verifies that following `hops` from `start` (step by step, over the
/// store's real edges) actually lands on `goal`. Used to check whether a
/// hop sequence discovered for one bound pair generalizes to another.
pub fn path_holds(store: &Store, start: &Term, goal: &Term, hops: &[Hop]) -> bool {
    let mut current = start.clone();
    for hop in hops {
        let mut stepped = false;
        for (candidate_hop, neighbor) in neighbors(store, &current) {
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

fn neighbors(store: &Store, node: &Term) -> Vec<(Hop, Term)> {
    let mut out = Vec::new();

    if let Some(subject) = as_subject(node) {
        for quad in store.quads_for_pattern(Some(subject.as_ref()), None, None, Some(GraphNameRef::DefaultGraph)).flatten() {
            out.push((Hop::Forward(quad.predicate), quad.object));
        }
    }

    for quad in store.quads_for_pattern(None, None, Some(node.as_ref()), Some(GraphNameRef::DefaultGraph)).flatten() {
        out.push((Hop::Inverse(quad.predicate), Term::from(quad.subject)));
    }

    out
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
