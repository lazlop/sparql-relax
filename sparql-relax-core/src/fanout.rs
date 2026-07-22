//! Graph-wide fan-out cap: how many distinct neighbors is *typical* for a
//! node in this specific graph to have, via any one predicate/direction —
//! used by [`crate::bfs::find_path`] to reject a candidate hop whose
//! specific endpoint has an unusually large number of neighbors via the
//! predicate being walked.
//!
//! The motivating failure mode (found by tracing real regressions in this
//! tool's building-automation eval set): a path search bouncing out to a
//! shared "hub" value and back — e.g. two unrelated sensors that happen to
//! share a generic Brick tag, or two unrelated properties that happen to
//! share a QUDT quantity kind — can "connect" almost any two entities of a
//! given kind, which is indistinguishable from noise even though every
//! individual edge involved is real.
//!
//! The cap is a single number — the [`FANOUT_PERCENTILE`] percentile of
//! every node's own total degree in this graph (distinct
//! `(predicate, neighbor)` pairs, summed over both directions) — rather
//! than one computed separately per predicate. A per-predicate version was
//! tried first, but it has its own failure mode: a predicate that's
//! genuinely, uniformly high-fan-out everywhere it's used (or one used so
//! rarely that there's barely any data to rank against) never looks
//! unusual *relative to itself*, so nothing about it ever exceeds its own
//! threshold — e.g. `feeds`/`isFedBy`, where the overwhelming majority of
//! subjects feed exactly one thing and only a handful of genuine
//! supply-hub nodes feed many, ends up with a per-predicate threshold of 1
//! regardless of percentile chosen, rejecting the very supply-chain hops
//! this tool exists to help trace. Measuring against *this graph's* overall
//! connectivity instead sidesteps that: a hop is compared to how connected
//! nodes in this graph typically are at all, not to how one specific
//! predicate happens to be used.
//!
//! Still computed fresh per [`Store`] (not a fixed constant baked into the
//! tool) so it generalizes to graphs this tool hasn't seen, rather than
//! being tuned to one dataset's raw fan-out numbers — a sparse graph and a
//! densely-connected one each get their own cap, reflecting their own
//! typical connectivity.
//!
//! Built once per [`Store`] (like the store itself) rather than recomputed
//! per search: it's a single pass over every triple, and the resulting
//! index is read-only for the lifetime of the store, so every
//! `diagnose_and_connect`/`find_path` call against it reuses the same one
//! rather than re-scanning the whole graph per query.

use crate::bfs::predicate_allowed;
use oxigraph::model::{NamedNode, Term};
use oxigraph::store::Store;
use std::collections::{HashMap, HashSet};

/// Percentile of graph-wide node degree above which a specific endpoint is
/// treated as a hub and excluded from path search — see the module docs and
/// [`percentile`].
///
/// Chosen empirically against this project's building-automation eval set
/// (`eval/buildings`). A lower value (0.90) was tried first, but it measurably
/// regressed real connections in the smaller/sparser sample buildings: e.g.
/// `TUC_building` has only 552 relevant nodes, so its 90th-percentile degree
/// is just 3, while `brick:hasPoint` there is legitimately, uniformly
/// fan-out-10 (every one of ~19 pieces of equipment has exactly 10 points) —
/// completely normal structure, not a hub, but the cap couldn't tell the
/// difference at that percentile and rejected it. 0.98 clears that (and the
/// analogous case in `dflexlibs_multizone`, the other sparse sample building)
/// while staying far below every genuine hub value measured across all four
/// buildings (the smallest of which — a Brick tag shared by dozens of
/// entities — is still several times larger than the resulting cap in every
/// sample building).
///
/// A graph with fewer than `1 / (1 - FANOUT_PERCENTILE)` relevant nodes (50,
/// at the current value) loses this protection entirely: the nearest-rank
/// percentile degenerates to the single largest degree in the graph, so
/// nothing can ever exceed it (the same degeneracy the old per-predicate
/// design hit for a low-cardinality predicate, just triggered here by a
/// small overall population instead). None of this project's four sample
/// buildings are anywhere near that small (the smallest has 223 relevant
/// nodes), so this is a real limit worth documenting, not one currently
/// biting any graph this tool is actually run against.
pub(crate) const FANOUT_PERCENTILE: f64 = 0.98;

/// The [`FANOUT_PERCENTILE`] percentile of node degree across an entire
/// graph, computed once (see [`FanoutIndex::build`]) and reused for every
/// hop [`crate::bfs::find_path`] considers against it.
pub struct FanoutIndex {
    threshold: usize,
}

impl FanoutIndex {
    /// Scans every triple in `store` once — every graph, not just the
    /// default one (matching [`crate::bfs::neighbors`], which this index is
    /// built to be checked against; see that module's docs for why) —
    /// skipping any whose predicate `allowed_namespaces` excludes (see
    /// [`crate::bfs::predicate_allowed`], the same filter path search itself
    /// applies, so a predicate invisible to the search never contributes to
    /// what "typical connectivity" means for it either) and computes each
    /// node's total degree: the number of distinct `(predicate, neighbor)`
    /// pairs it participates in, as either subject or object. A node reached
    /// only as a literal object is not counted (a literal can never be a
    /// subject, so it has no "typical connectivity" of its own to rank
    /// against others') — but a literal endpoint can still be *checked*
    /// against the resulting threshold at search time, same as any other
    /// endpoint.
    ///
    /// The [`FANOUT_PERCENTILE`] percentile of that degree distribution
    /// becomes the single threshold every hop in this graph is checked
    /// against, regardless of which predicate or direction it's on — see
    /// the module docs for why this is graph-wide rather than per-predicate.
    ///
    /// A graph with on the order of a hundred distinct predicates (typical
    /// for the building-automation ontologies this tool targets) costs a
    /// single linear pass to index this way, comparable to the store's own
    /// one-time parse — see this module's docs on why that's worth paying
    /// once rather than folding into every search.
    pub fn build(store: &Store, allowed_namespaces: Option<&[String]>) -> Self {
        let mut out_neighbors: HashMap<Term, HashSet<(NamedNode, Term)>> = HashMap::new();
        let mut in_neighbors: HashMap<Term, HashSet<(NamedNode, Term)>> = HashMap::new();

        for quad in store.quads_for_pattern(None, None, None, None).flatten() {
            if !predicate_allowed(&quad.predicate, allowed_namespaces) {
                continue;
            }
            let subject = Term::from(quad.subject);
            let object = quad.object;
            out_neighbors.entry(subject.clone()).or_default().insert((quad.predicate.clone(), object.clone()));
            if matches!(object, Term::NamedNode(_) | Term::BlankNode(_)) {
                in_neighbors.entry(object).or_default().insert((quad.predicate, subject));
            }
        }

        let mut nodes: HashSet<&Term> = out_neighbors.keys().collect();
        nodes.extend(in_neighbors.keys());

        let degrees: Vec<usize> = nodes
            .into_iter()
            .map(|n| out_neighbors.get(n).map_or(0, HashSet::len) + in_neighbors.get(n).map_or(0, HashSet::len))
            .collect();

        // An empty (or fully-filtered-out) graph has no basis to rank
        // anything against; nothing should ever be rejected as a hub in
        // that case, hence `usize::MAX` rather than `0` (which would reject
        // every single edge).
        let threshold = if degrees.is_empty() { usize::MAX } else { percentile(degrees) };
        Self { threshold }
    }

    /// Whether `count` — an actual endpoint's local fan-out via one specific
    /// predicate and direction — exceeds this graph's [`FANOUT_PERCENTILE`]
    /// node degree (see [`FanoutIndex::build`]), i.e. whether this
    /// particular hop is unusually promiscuous for *this graph*, not
    /// against some fixed number shared across every graph this tool is
    /// ever run against.
    pub fn exceeds_threshold(&self, count: usize) -> bool {
        count > self.threshold
    }
}

/// The [`FANOUT_PERCENTILE`] percentile of `counts` (nearest-rank method):
/// sorts, then takes the smallest value at or past that fraction of the way
/// through. Doesn't need to match any particular statistics library's
/// interpolation exactly — this is a threshold for a heuristic filter, not a
/// reported statistic.
fn percentile(mut counts: Vec<usize>) -> usize {
    counts.sort_unstable();
    let n = counts.len();
    let idx = ((n as f64) * FANOUT_PERCENTILE).ceil() as usize;
    counts[idx.saturating_sub(1).min(n - 1)]
}
