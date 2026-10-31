// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Well graph of the network solve.

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZero;

use kiddo::{ImmutableKdTree, SquaredEuclidean};
use petgraph::unionfind::UnionFind;

use crate::config::CorrelateGraphConfig;
use crate::error::{SxError, SxResult};
use crate::types::{WellHeader, WellId};
use crate::util::median;

/// Distance used for wells with identical surface coordinates, m.
const SAME_SPOT_M: f64 = 1.0;

/// One undirected edge of the well graph.
#[derive(Clone, Debug)]
pub struct GraphEdge {
    /// Lower id.
    pub a: WellId,
    /// Higher id.
    pub b: WellId,
    /// Surface distance, m (1.0 for identical coordinates).
    pub length_m: f64,
    /// Edge is in the minimum spanning tree of its component.
    pub in_mst: bool,
}

/// Undirected well graph used by pair DTW and the network solve.
#[derive(Clone, Debug)]
pub struct WellGraph {
    /// Ascending.
    pub wells: Vec<WellId>,
    /// Sorted by (a, b), a < b, no duplicates.
    pub edges: Vec<GraphEdge>,
    /// `L` = median edge length, m (1.0 when there are no edges).
    pub median_length_m: f64,
}

impl WellGraph {
    /// Index into `edges` of the edge {a, b} in either order.
    ///
    /// `None` when there is no such edge (including `a == b`). NaN rule: not
    /// applicable. Errors: none. Complexity: `O(log edges)`.
    pub fn edge_index(&self, a: WellId, b: WellId) -> Option<usize> {
        let key = (a.min(b), a.max(b));
        self.edges.binary_search_by(|e| (e.a, e.b).cmp(&key)).ok()
    }

    /// Graph neighbors of `w` with edge length (m), sorted by (length, WellId).
    ///
    /// Empty when `w` has no edge or is not in the graph. NaN rule: not
    /// applicable (lengths are finite). Errors: none. Complexity:
    /// `O(edges + d log d)`, `d` = degree.
    pub fn neighbors_by_distance(&self, w: WellId) -> Vec<(WellId, f64)> {
        let mut out: Vec<(WellId, f64)> = self
            .edges
            .iter()
            .filter_map(|e| {
                if e.a == w {
                    Some((e.b, e.length_m))
                } else if e.b == w {
                    Some((e.a, e.length_m))
                } else {
                    None
                }
            })
            .collect();
        out.sort_by(|x, y| x.1.total_cmp(&y.1).then(x.0.cmp(&y.0)));
        out
    }

    /// `w_edge = quality × exp(−length_m / L)`, f64.
    ///
    /// `length_m` in m, `quality` unitless. NaN rule: propagate. Errors: none.
    /// Complexity: `O(1)`.
    pub fn edge_weight(&self, length_m: f64, quality: f32) -> f64 {
        f64::from(quality) * (-length_m / self.median_length_m).exp()
    }
}

/// Check `k_graph ≥ 1` and `max_edge_m > 0` ([`SxError::Config`]).
pub(crate) fn check_graph_config(cfg: &CorrelateGraphConfig) -> SxResult<()> {
    if cfg.k_graph < 1 {
        return Err(SxError::Config(
            "correlate.graph.k_graph must be >= 1".into(),
        ));
    }
    if cfg.max_edge_m.is_nan() || cfg.max_edge_m <= 0.0 {
        return Err(SxError::Config(
            "correlate.graph.max_edge_m must be > 0".into(),
        ));
    }
    Ok(())
}

/// Build the well graph from surface coordinates.
///
/// Edges are the union of: every pair inside a group of identical
/// coordinates; all three edges of every Delaunay triangle of the group
/// representatives (lowest id), each duplicate also taking its
/// representative's Delaunay edges; and each well's `k_graph` nearest other
/// wells by (distance, WellId). Edges longer than `max_edge_m` are dropped
/// unless they are in the minimum spanning tree (Kruskal by (length, a, b)) of
/// their component.
///
/// **Choice:** k-th distance ties: the k-d tree is asked for the
/// `k + 1` nearest wells, doubling the count until the farthest returned
/// well is strictly farther than the k-th candidate (or every well is
/// returned); the candidates are then sorted by (squared distance, WellId)
/// and the first `k` kept, which equals a brute-force sort.
///
/// Units: `x_m`, `y_m` and lengths in m. NaN rule: error. Errors:
/// [`SxError::EmptyInput`] (no headers), [`SxError::InvalidInput`]
/// (non-finite coordinate, ids not strictly ascending), [`SxError::Config`]
/// (`k_graph < 1`, `max_edge_m ≤ 0`). Complexity:
/// `O(n log n + n k log n)`.
pub fn build_well_graph(headers: &[WellHeader], cfg: &CorrelateGraphConfig) -> SxResult<WellGraph> {
    check_graph_config(cfg)?;
    if headers.is_empty() {
        return Err(SxError::EmptyInput("well graph: no headers".into()));
    }
    for h in headers {
        if !h.x_m.is_finite() || !h.y_m.is_finite() {
            return Err(SxError::InvalidInput(format!(
                "well {}: surface coordinates must be finite",
                h.id.0
            )));
        }
    }
    if headers
        .windows(2)
        .any(|p| matches!(p, [x, y] if x.id >= y.id))
    {
        return Err(SxError::InvalidInput(
            "well graph: headers must be sorted by id, unique".into(),
        ));
    }
    let n = headers.len();
    let xy: Vec<(f64, f64)> = headers.iter().map(|h| (h.x_m, h.y_m)).collect();
    let mut pairs: BTreeSet<(usize, usize)> = BTreeSet::new();

    // Groups of identical coordinates; `-0.0 + 0.0 == +0.0`.
    let mut groups: BTreeMap<(u64, u64), Vec<usize>> = BTreeMap::new();
    for (i, &(x, y)) in xy.iter().enumerate() {
        groups
            .entry(((x + 0.0).to_bits(), (y + 0.0).to_bits()))
            .or_default()
            .push(i);
    }
    let mut members: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for group in groups.values() {
        let Some(&rep) = group.iter().min() else {
            continue;
        };
        for (gi, &a) in group.iter().enumerate() {
            for &b in group.iter().skip(gi + 1) {
                pairs.insert((a.min(b), a.max(b)));
            }
        }
        members.insert(rep, group.clone());
    }
    let reps: Vec<usize> = members.keys().copied().collect();

    // Delaunay on representatives, shifted by the minimum x and y.
    let min_x = xy.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let min_y = xy.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    if reps.len() >= 3 {
        let points: Vec<delaunator::Point> = reps
            .iter()
            .filter_map(|&r| xy.get(r))
            .map(|&(x, y)| delaunator::Point {
                x: x - min_x,
                y: y - min_y,
            })
            .collect();
        let tri = delaunator::triangulate(&points);
        let mut add_rep_edge = |p: usize, q: usize| {
            let (Some(&ra), Some(&rb)) = (reps.get(p), reps.get(q)) else {
                return;
            };
            if ra == rb {
                return;
            }
            for (r, other) in [(ra, rb), (rb, ra)] {
                for &m in members.get(&r).map_or(&[][..], Vec::as_slice) {
                    if m != other {
                        pairs.insert((m.min(other), m.max(other)));
                    }
                }
            }
        };
        for &[p, q, r] in tri.triangles.as_chunks::<3>().0 {
            add_rep_edge(p, q);
            add_rep_edge(q, r);
            add_rep_edge(r, p);
        }
    }

    // Nearest wells (all wells, duplicates included).
    let k = cfg.k_graph.min(n - 1);
    if k > 0 {
        let entries: Vec<[f64; 2]> = xy.iter().map(|&(x, y)| [x, y]).collect();
        let tree: ImmutableKdTree<f64, 2> = ImmutableKdTree::new_from_slice(&entries)
            .map_err(|e| SxError::Numerical(format!("well graph: k-d tree build failed: {e}")))?;
        for (i, &(xi, yi)) in xy.iter().enumerate() {
            let sq = |j: usize| {
                xy.get(j).map_or(f64::INFINITY, |&(xj, yj)| {
                    let (dx, dy) = (xj - xi, yj - yi);
                    dx * dx + dy * dy
                })
            };
            let mut want = k + 1;
            let cands: Vec<(f64, usize)> = loop {
                let q = NonZero::new(want.min(n)).unwrap_or(NonZero::<usize>::MIN);
                let found = tree
                    .query(&[xi, yi])
                    .nearest_n::<SquaredEuclidean<f64>>(q)
                    .execute();
                let mut c: Vec<(f64, usize)> = found
                    .iter()
                    .map(|r| r.item as usize)
                    .filter(|&j| j != i)
                    .map(|j| (sq(j), j))
                    .collect();
                c.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
                let kth = c.get(k - 1).map_or(f64::INFINITY, |p| p.0);
                let far = c.last().map_or(f64::NEG_INFINITY, |p| p.0);
                if found.len() >= n || want >= n || far > kth {
                    break c;
                }
                want = want.saturating_mul(2);
            };
            for &(_, j) in cands.iter().take(k) {
                pairs.insert((i.min(j), i.max(j)));
            }
        }
    }

    // Lengths, MST per component, length filter.
    let mut edges: Vec<(f64, usize, usize, bool)> = pairs
        .iter()
        .filter_map(|&(a, b)| {
            let (&(xa, ya), &(xb, yb)) = (xy.get(a)?, xy.get(b)?);
            let (dx, dy) = (xb - xa, yb - ya);
            let d = (dx * dx + dy * dy).sqrt();
            Some((if d == 0.0 { SAME_SPOT_M } else { d }, a, b, false))
        })
        .collect();
    edges.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));
    let mut uf = UnionFind::<usize>::new(n);
    for e in &mut edges {
        e.3 = uf.union(e.1, e.2);
    }
    let mut kept: Vec<GraphEdge> = edges
        .into_iter()
        .filter(|e| e.3 || e.0 <= cfg.max_edge_m)
        .filter_map(|(len, a, b, in_mst)| {
            Some(GraphEdge {
                a: headers.get(a)?.id,
                b: headers.get(b)?.id,
                length_m: len,
                in_mst,
            })
        })
        .collect();
    kept.sort_by_key(|e| (e.a, e.b));
    let lengths: Vec<f64> = kept.iter().map(|e| e.length_m).collect();
    let median_length_m = if lengths.is_empty() {
        1.0
    } else {
        median(&lengths)
    };
    Ok(WellGraph {
        wells: headers.iter().map(|h| h.id).collect(),
        edges: kept,
        median_length_m,
    })
}
