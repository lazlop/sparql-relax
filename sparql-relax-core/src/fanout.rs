//! Per-predicate, per-direction fan-out index: for each predicate a graph
//! actually uses, how many distinct neighbors is *typical* to have via it,
//! in each direction — used by [`crate::bfs::find_path`] to reject a
//! candidate hop whose *specific* endpoint has an unusually large number of
//! neighbors via that predicate.
//!
//! The motivating failure mode (found by tracing real regressions in this
//! tool's building-automation eval set): a path search bouncing out to a
//! shared "hub" value and back — e.g. two unrelated sensors that happen to
//! share a generic Brick tag, or two unrelated properties that happen to
//! share a QUDT quantity kind — can "connect" almost any two entities of a
//! given kind, which is indistinguishable from noise even though every
//! individual edge involved is real. A single global cutoff on fan-out
//! doesn't separate this from genuinely useful connections, though: a
//! predicate like `feeds`/`isFedBy` (a near-bijective supply-chain
//! hierarchy) can have the *same* raw fan-out as a dangerous hop through
//! `hasAssociatedTag`, yet tracing a real supply chain is exactly the kind
//! of fix this tool exists to find. What actually separates them is
//! *relative* to each predicate's own usage: `feeds`/`isFedBy` are used
//! consistently (every endpoint's fan-out sits at or near that predicate's
//! own typical value), while `hasAssociatedTag`/`hasQuantityKind`-style
//! predicates are dominated by rare, specific values with a long tail of
//! generic "hub" values shared by everything. Rejecting a hop whose
//! specific endpoint's fan-out sits in that tail — above the *75th
//! percentile of its own predicate's* fan-out, not some fixed number
//! shared across every predicate — catches the hub case while leaving
//! structural, near-uniform relations untouched.
//!
//! The percentile alone breaks down for a low-cardinality predicate — a
//! boolean like `writable` has exactly two possible values, so with only
//! two (predicate, direction) groups to rank, the 75th percentile always
//! lands on whichever of the two has the larger count, and `>` never fires
//! against its own value. A majority-share backstop catches this: a value
//! that alone accounts for more than half of a predicate's total usage in
//! that direction is a hub regardless of how many other values exist to
//! rank it against (see [`collect_thresholds`]).
//!
//! Built once per [`Store`] (like the store itself) rather than recomputed
//! per search: it's a single pass over every triple, and the resulting
//! index is read-only for the lifetime of the store, so every
//! `diagnose_and_connect`/`find_path` call against it reuses the same one
//! rather than re-scanning the whole graph per query.

use oxigraph::model::{GraphNameRef, NamedNode, Term};
use oxigraph::store::Store;
use std::collections::HashMap;

/// Which side of a `(subject, predicate, object)` triple a hop travels
/// away from: [`Direction::Forward`] leaves the subject (following `<p>`
/// toward the object), [`Direction::Inverse`] leaves the object (following
/// `^<p>` toward the subject) — see [`crate::bfs::Hop`], which this
/// mirrors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    Forward,
    Inverse,
}

/// The 75th percentile of local fan-out for every `(predicate, direction)`
/// pair a [`Store`] actually contains, computed once (see [`FanoutIndex::build`])
/// and reused for every hop [`crate::bfs::find_path`] considers against it.
pub struct FanoutIndex {
    thresholds: HashMap<(NamedNode, Direction), usize>,
}

impl FanoutIndex {
    /// Scans every default-graph triple in `store` once, grouping by
    /// `(predicate, direction)` and, within each group, by the endpoint a
    /// hop in that direction would leave from (the subject for `Forward`,
    /// the object for `Inverse`) — counting each endpoint's *distinct*
    /// neighbors, not its raw triple count, so a handful of duplicate
    /// statements can't inflate a single endpoint's apparent fan-out. The
    /// 75th percentile of those per-endpoint counts becomes that
    /// `(predicate, direction)` pair's threshold.
    ///
    /// A graph with on the order of a hundred distinct predicates (typical
    /// for the building-automation ontologies this tool targets) costs a
    /// single linear pass to index this way, comparable to the store's own
    /// one-time parse — see this module's docs on why that's worth paying
    /// once rather than folding into every search.
    pub fn build(store: &Store) -> Self {
        let mut forward_neighbors: HashMap<(NamedNode, Term), HashSetTerm> = HashMap::new();
        let mut inverse_neighbors: HashMap<(NamedNode, Term), HashSetTerm> = HashMap::new();

        for quad in store.quads_for_pattern(None, None, None, Some(GraphNameRef::DefaultGraph)).flatten() {
            let subject = Term::from(quad.subject);
            forward_neighbors.entry((quad.predicate.clone(), subject.clone())).or_default().insert(quad.object.clone());
            inverse_neighbors.entry((quad.predicate, quad.object)).or_default().insert(subject);
        }

        let mut thresholds = HashMap::new();
        collect_thresholds(forward_neighbors, Direction::Forward, &mut thresholds);
        collect_thresholds(inverse_neighbors, Direction::Inverse, &mut thresholds);
        Self { thresholds }
    }

    /// Whether `count` — an actual endpoint's local fan-out via `predicate`
    /// in `direction` — exceeds that *specific predicate's own* 75th
    /// percentile fan-out (see [`FanoutIndex::build`]), i.e. whether this
    /// particular hop is unusually promiscuous for this predicate
    /// specifically, not against some threshold shared across every
    /// predicate. A predicate this index never saw (nothing to compare
    /// against) is never rejected.
    pub fn exceeds_threshold(&self, predicate: &NamedNode, direction: Direction, count: usize) -> bool {
        self.thresholds.get(&(predicate.clone(), direction)).is_some_and(|&threshold| count > threshold)
    }
}

type HashSetTerm = std::collections::HashSet<Term>;

/// For each predicate, folds its per-endpoint fan-out counts down to one
/// threshold: the smaller of the 75th-percentile count (the normal case,
/// see the module docs) and half the predicate's total usage in this
/// direction — a single endpoint value accounting for a majority of every
/// occurrence of the predicate is a hub on its own terms, independent of
/// how many *other* distinct values exist to rank it against. Without this
/// second term, a low-cardinality predicate (a boolean, an enum with a
/// handful of values) can't produce a useful percentile at all: with only
/// two values to rank, the larger one *is* the 75th percentile, so it can
/// never exceed its own threshold.
///
/// The majority-share term only applies with at least two distinct endpoint
/// values for the predicate: with only one, that lone value *is* the
/// predicate's entire usage in this direction, and "it accounts for a
/// majority of the total" is trivially true of any single-use predicate
/// (the common case in a small or sparsely-connected graph) rather than a
/// sign of anything hub-like — there's nothing else to have been
/// disproportionate relative to.
fn collect_thresholds(groups: HashMap<(NamedNode, Term), HashSetTerm>, direction: Direction, out: &mut HashMap<(NamedNode, Direction), usize>) {
    let mut counts_by_predicate: HashMap<NamedNode, Vec<usize>> = HashMap::new();
    for ((predicate, _endpoint), neighbors) in groups {
        counts_by_predicate.entry(predicate).or_default().push(neighbors.len());
    }
    for (predicate, counts) in counts_by_predicate {
        let threshold = if counts.len() >= 2 {
            let total: usize = counts.iter().sum();
            percentile_75(counts).min(total / 2)
        } else {
            percentile_75(counts)
        };
        out.insert((predicate, direction), threshold);
    }
}

/// The 75th percentile of `counts` (nearest-rank method): sorts, then takes
/// the smallest value at or past the 75%-of-the-way point. Doesn't need to
/// match any particular statistics library's interpolation exactly — this
/// is a threshold for a heuristic filter, not a reported statistic.
fn percentile_75(mut counts: Vec<usize>) -> usize {
    counts.sort_unstable();
    let n = counts.len();
    let idx = ((n as f64) * 0.75).ceil() as usize;
    counts[idx.saturating_sub(1).min(n - 1)]
}
