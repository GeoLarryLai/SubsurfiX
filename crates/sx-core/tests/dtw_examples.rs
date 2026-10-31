// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Tests: worked DTW examples, property tests, brute force,
//! mode behavior and the `#[ignore]` timing test.
//!
//! The capped-cost and pooled-scale examples test private helpers
//! and live next to them in `src/dtw/cost.rs`.

mod dtw_examples {
    use std::collections::BTreeMap;

    use proptest::prelude::*;
    use sx_core::dtw::{
        band_half_width, banded_dtw, banded_dtw_dense, full_path, interval_path, interval_plan,
        map_a_to_b, map_b_to_a, md_to_k, pairs_from_md, row_mean_map, window_path, IntervalParams,
        IntervalSide, MatchMethod, SpanParams, SpanSide,
    };
    use sx_core::types::{GridRange, LogKind, WellId, WellLogs};
    use sx_core::SxError;

    // ------------------------------------------------------------------ helpers

    const INF: f64 = f64::INFINITY;

    fn well(id: i32, k_start: i32, len: usize, curves: Vec<(LogKind, Vec<f32>)>) -> WellLogs {
        let mut w = WellLogs::new(WellId(id), GridRange::new(k_start, len).unwrap());
        for (log, v) in curves {
            assert_eq!(v.len(), len);
            w.curves.insert(log, v);
        }
        w
    }

    /// Deterministic pseudo-random numbers in `[0, 1)`.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    /// The reference 9.1 matrix; cells outside the band hold a sentinel that must
    /// never be read.
    fn reference_matrix() -> Vec<f64> {
        let x = 99.0;
        vec![
            1.0, 1.0, x, x, //
            2.0, 3.0, 0.0, x, //
            x, 0.0, 3.0, 2.0, //
            x, x, 1.0, 1.0,
        ]
    }

    /// Dense reference DP with the reference recursion and tie order.
    fn reference(cost: &[f64], n: usize, w: usize) -> Option<(Vec<(usize, usize)>, f64)> {
        let inb = |i: usize, j: usize| i.abs_diff(j) <= w;
        let mut d = vec![INF; n * n];
        let mut step = vec![0u8; n * n];
        for i in 0..n {
            for j in 0..n {
                if !inb(i, j) {
                    continue;
                }
                let c = cost[i * n + j];
                if i == 0 && j == 0 {
                    d[0] = c;
                    continue;
                }
                let diag = if i > 0 && j > 0 {
                    d[(i - 1) * n + j - 1]
                } else {
                    INF
                };
                let left = if j > 0 { d[i * n + j - 1] } else { INF };
                let up = if i > 0 { d[(i - 1) * n + j] } else { INF };
                let (mut best, mut s) = (diag, 1u8);
                if left < best {
                    best = left;
                    s = 2;
                }
                if up < best {
                    best = up;
                    s = 3;
                }
                if best.is_finite() {
                    d[i * n + j] = c + best;
                    step[i * n + j] = s;
                }
            }
        }
        let total = d[n * n - 1];
        if !total.is_finite() {
            return None;
        }
        let (mut i, mut j) = (n - 1, n - 1);
        let mut cells = vec![(i, j)];
        while (i, j) != (0, 0) {
            match step[i * n + j] {
                1 => {
                    i -= 1;
                    j -= 1;
                }
                2 => j -= 1,
                _ => i -= 1,
            }
            cells.push((i, j));
        }
        cells.reverse();
        Some((cells, total))
    }

    /// Minimum cost over all band-respecting paths.
    fn brute_min(cost: &[f64], n: usize, w: usize) -> f64 {
        fn go(cost: &[f64], n: usize, w: usize, i: usize, j: usize, acc: f64, best: &mut f64) {
            if i.abs_diff(j) > w {
                return;
            }
            let acc = acc + cost[i * n + j];
            if (i, j) == (n - 1, n - 1) {
                *best = best.min(acc);
                return;
            }
            if i + 1 < n && j + 1 < n {
                go(cost, n, w, i + 1, j + 1, acc, best);
            }
            if j + 1 < n {
                go(cost, n, w, i, j + 1, acc, best);
            }
            if i + 1 < n {
                go(cost, n, w, i + 1, j, acc, best);
            }
        }
        let mut best = INF;
        go(cost, n, w, 0, 0, 0.0, &mut best);
        best
    }

    // ---------------------------------------------------------

    #[test]
    fn worked_recursion_and_tie_break() {
        let path = banded_dtw_dense(&reference_matrix(), 4, 4, 1).unwrap();
        assert_eq!(path.cells, vec![(0, 0), (1, 0), (2, 1), (3, 2), (3, 3)]);
        assert_eq!(path.total_cost, 5.0);
        assert_eq!(path.mean_cost(), 1.0);
        assert_eq!(row_mean_map(&path, 4), vec![0.0, 0.0, 1.0, 3.0]);

        let md = [
            (5000.0, 6000.0),
            (5001.0, 6000.0),
            (5002.0, 6002.0),
            (5003.0, 6006.0),
        ];
        assert_eq!(
            pairs_from_md(&md).unwrap(),
            vec![
                (10000, 12000),
                (10002, 12000),
                (10004, 12004),
                (10006, 12012)
            ]
        );
    }

    #[test]
    fn worked_d_table_and_call_order() {
        // Accumulated D of the worked example is verified through total cost of prefixes:
        // end cell (3,3) -> 5; and the closure is called once per in-band cell in
        // row-major order.
        let m = reference_matrix();
        let mut calls = Vec::new();
        let path = banded_dtw(4, 4, 1, |i, j| {
            calls.push((i, j));
            m[i * 4 + j]
        })
        .unwrap();
        assert_eq!(path.total_cost, 5.0);
        let want: Vec<(usize, usize)> = (0..4)
            .flat_map(|i| (0..4).map(move |j| (i, j)))
            .filter(|&(i, j): &(usize, usize)| i.abs_diff(j) <= 1)
            .collect();
        assert_eq!(calls, want);
        // Sub-problems read off the D table: top-left 2x2 block, w = 1.
        let sub = banded_dtw_dense(&[1.0, 1.0, 2.0, 3.0], 2, 2, 1).unwrap();
        assert_eq!(sub.total_cost, 4.0); // D[1][1]
        assert_eq!(sub.cells, vec![(0, 0), (1, 1)]);
    }

    #[test]
    fn band_table() {
        let rows = [
            (24, 0.12, 2),
            (46, 0.12, 5),
            (128, 0.12, 14),
            (400, 0.12, 47),
            (2000, 0.12, 239),
            (400, 0.25, 99),
            (2000, 0.25, 499),
            (512, 0.25, 127),
        ];
        for (n, f, w) in rows {
            assert_eq!(band_half_width(n, f).unwrap(), w, "N={n} frac={f}");
        }
    }

    #[test]
    fn interval_plan_examples() {
        let p = IntervalParams::default();
        let plan = interval_plan(13, 2, &p);
        assert_eq!((plan.n_nodes, plan.min_shared), (46, 11));
        let plan = interval_plan(4, 0, &p);
        assert_eq!((plan.n_nodes, plan.min_shared), (128, 32));
        let plan = interval_plan(30, 0, &p);
        assert_eq!((plan.n_nodes, plan.min_shared), (24, 8));
    }

    // ------------------------------------------------------------

    #[test]
    fn worked_grid_pairs_and_sampling() {
        let a_nodes = [5000.0, 5000.5, 5001.0, 5001.5, 5002.0];
        let b_nodes = [6000.3, 6001.0, 6001.7, 6002.4, 6003.1];
        let cells = [(0, 0), (1, 0), (2, 1), (3, 2), (3, 3), (4, 4)];
        let md: Vec<(f64, f64)> = cells
            .iter()
            .map(|&(i, j)| (a_nodes[i], b_nodes[j]))
            .collect();
        let path = pairs_from_md(&md).unwrap();
        assert_eq!(
            path,
            vec![
                (10000, 12001),
                (10001, 12001),
                (10002, 12002),
                (10003, 12003),
                (10003, 12005),
                (10004, 12006)
            ]
        );
        assert_eq!(map_a_to_b(&path, 10003.0), Some(12003.0));
        assert_eq!(map_a_to_b(&path, 10003.5), Some(12004.5));
        assert_eq!(map_b_to_a(&path, 12004.0), Some(10003.0));
        assert_eq!(map_a_to_b(&path, 9999.0), None);
        // more sampling rules
        assert_eq!(map_a_to_b(&path, 10000.0), Some(12001.0));
        assert_eq!(map_a_to_b(&path, 10004.0), Some(12006.0));
        assert_eq!(map_a_to_b(&path, 10004.5), None);
        assert_eq!(map_a_to_b(&path, f64::NAN), None);
        assert_eq!(map_b_to_a(&path, 12001.0), Some(10000.0));
        assert_eq!(map_b_to_a(&path, 12011.0), None);
        assert_eq!(map_b_to_a(&path, f64::NAN), None);
    }

    #[test]
    fn grid_pair_dedupe_and_errors() {
        let got = pairs_from_md(&[(0.0, 0.0), (0.1, 0.1), (0.2, 0.3), (0.6, 0.6)]).unwrap();
        // 0.1 -> 0, 0.2 -> 0, 0.3 -> 1 (0.3/0.5+0.5 = 1.1), 0.6 -> 1
        assert_eq!(got, vec![(0, 0), (0, 1), (1, 1)]);
        assert!(matches!(
            pairs_from_md(&[(f64::NAN, 1.0)]),
            Err(SxError::InvalidInput(_))
        ));
        assert_eq!(md_to_k(10.25).unwrap(), 21);
    }

    // ------------------------------------------------------------ engine errors

    #[test]
    fn engine_errors() {
        assert!(matches!(
            banded_dtw(5, 9, 2, |_, _| 1.0),
            Err(SxError::Infeasible(_))
        ));
        assert!(matches!(
            banded_dtw(3, 3, 1, |i, j| if (i, j) == (1, 1) {
                f64::NAN
            } else {
                1.0
            }),
            Err(SxError::InvalidInput(_))
        ));
        assert!(matches!(
            banded_dtw(3, 3, 1, |_, _| -1.0),
            Err(SxError::InvalidInput(_))
        ));
        assert!(matches!(
            banded_dtw(3, 3, 1, |_, _| f64::INFINITY),
            Err(SxError::InvalidInput(_))
        ));
        assert!(matches!(
            banded_dtw(0, 3, 1, |_, _| 1.0),
            Err(SxError::EmptyInput(_))
        ));
        assert!(matches!(
            banded_dtw(3, 0, 1, |_, _| 1.0),
            Err(SxError::EmptyInput(_))
        ));
        assert!(matches!(
            banded_dtw_dense(&[1.0; 5], 2, 3, 2),
            Err(SxError::Shape(_))
        ));
        assert!(matches!(
            band_half_width(10, 0.0),
            Err(SxError::InvalidInput(_))
        ));
        assert!(matches!(
            band_half_width(10, f64::NAN),
            Err(SxError::InvalidInput(_))
        ));
        assert!(matches!(
            band_half_width(0, 0.1),
            Err(SxError::InvalidInput(_))
        ));
    }

    #[test]
    fn rectangular_inside_band_works() {
        let path = banded_dtw(4, 6, 3, |_, _| 1.0).unwrap();
        assert_eq!(path.cells.first(), Some(&(0, 0)));
        assert_eq!(path.cells.last(), Some(&(3, 5)));
        assert!(path.cells.windows(2).all(|w| {
            let (di, dj) = (w[1].0 - w[0].0, w[1].1 - w[0].1);
            (di, dj) == (1, 1) || (di, dj) == (0, 1) || (di, dj) == (1, 0)
        }));
    }

    #[test]
    fn all_equal_costs_give_pure_diagonal() {
        for n in 1..=8 {
            for w in 0..n {
                let path = banded_dtw(n, n, w, |_, _| 2.5).unwrap();
                let diag: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();
                assert_eq!(path.cells, diag, "n={n} w={w}");
                assert_eq!(path.total_cost, 2.5 * n as f64);
            }
        }
    }

    #[test]
    fn brute_force_and_tie_order_small() {
        let mut rng = Lcg(12345);
        for n in 2..=6usize {
            for w in 1..n {
                for _ in 0..60 {
                    // small integers give many ties
                    let cost: Vec<f64> = (0..n * n).map(|_| (rng.next() * 3.0).floor()).collect();
                    let got = banded_dtw_dense(&cost, n, n, w).unwrap();
                    assert_eq!(got.total_cost, brute_min(&cost, n, w), "n={n} w={w}");
                    let (cells, total) = reference(&cost, n, w).unwrap();
                    assert_eq!(got.cells, cells, "n={n} w={w} cost={cost:?}");
                    assert_eq!(got.total_cost, total);
                }
            }
        }
    }

    fn square_case() -> impl Strategy<Value = (usize, f64, Vec<f64>)> {
        (2usize..=60, 0.05f64..=1.0).prop_flat_map(|(n, bf)| {
            (
                Just(n),
                Just(bf),
                proptest::collection::vec(0.0f64..10.0, n * n),
            )
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn path_properties((n, bf, cost) in square_case()) {
            let w = band_half_width(n, bf).unwrap();
            let path = banded_dtw_dense(&cost, n, n, w).unwrap();
            prop_assert_eq!(path.cells.first().copied(), Some((0, 0)));
            prop_assert_eq!(path.cells.last().copied(), Some((n - 1, n - 1)));
            let mut sum = 0.0;
            for (idx, &(i, j)) in path.cells.iter().enumerate() {
                prop_assert!(i.abs_diff(j) <= w);
                sum += cost[i * n + j];
                if idx > 0 {
                    let (pi, pj) = path.cells[idx - 1];
                    let step = (i - pi.min(i), j - pj.min(j));
                    prop_assert!(i >= pi && j >= pj);
                    prop_assert!(step == (1, 1) || step == (0, 1) || step == (1, 0));
                }
            }
            prop_assert!((path.total_cost - sum).abs() <= 1e-12 * sum.abs().max(1.0));
            let (cells, total) = reference(&cost, n, w).unwrap();
            prop_assert_eq!(path.cells, cells);
            prop_assert_eq!(path.total_cost, total);
        }
    }

    // ----------------------------------------------------------- interval mode

    /// Interval 0 of a well whose layer coordinate is `k / 200` on `k = 0..=200`.
    fn interval_well(id: i32, gr: Vec<f32>) -> (WellLogs, Vec<f64>) {
        let len = gr.len();
        let coord: Vec<f64> = (0..len).map(|k| k as f64 / 200.0).collect();
        (well(id, 0, len, vec![(LogKind::Gr, gr)]), coord)
    }

    fn side<'a>(
        w: &'a WellLogs,
        coord: &'a [f64],
        top: f64,
        base: f64,
        conf: f32,
        pad: bool,
    ) -> IntervalSide<'a> {
        IntervalSide {
            logs: w,
            layer_coord: coord,
            top_md_ft: top,
            base_md_ft: base,
            confidence: conf,
            is_pad: pad,
        }
    }

    fn diagonal_pairs(
        n: usize,
        top_a: f64,
        base_a: f64,
        top_b: f64,
        base_b: f64,
    ) -> Vec<(i32, i32)> {
        let last = (n - 1) as f64;
        let md: Vec<(f64, f64)> = (0..n)
            .map(|m| {
                let f = m as f64;
                let (ma, mb) = if m == n - 1 {
                    (base_a, base_b)
                } else {
                    (
                        top_a + (base_a - top_a) * f / last,
                        top_b + (base_b - top_b) * f / last,
                    )
                };
                (ma, mb)
            })
            .collect();
        pairs_from_md(&md).unwrap()
    }

    #[test]
    fn interval_pad_is_linear() {
        let p = IntervalParams::default();
        let plan = interval_plan(13, 2, &p);
        let gr: Vec<f32> = (0..=200).map(|k| k as f32 * 0.5).collect();
        let (wa, ca) = interval_well(1, gr.clone());
        let (wb, cb) = interval_well(2, gr);
        for (pad_a, pad_b) in [(true, false), (false, true), (true, true)] {
            let a = side(&wa, &ca, 0.0, 100.0, 0.8, pad_a);
            let b = side(&wb, &cb, 1000.0, 1120.0, 0.5, pad_b);
            let m = interval_path(&a, &b, 0, plan, &p).unwrap();
            assert_eq!(m.method, MatchMethod::Linear);
            assert_eq!(m.n_nodes, 46);
            assert!(m.mean_cost.is_nan() && m.path.cost.is_nan());
            assert_eq!(m.path.path, diagonal_pairs(46, 0.0, 100.0, 1000.0, 1120.0));
            assert_eq!(m.path.interval, Some(0));
            assert_eq!((m.path.a, m.path.b), (WellId(1), WellId(2)));
            let want = 0.45 * (0.8f64 * 0.5).sqrt();
            assert!((f64::from(m.path.quality) - want).abs() < 1e-6);
            assert!((m.span_ratio - 1.2).abs() < 1e-12);
        }
    }

    #[test]
    fn interval_constant_curve_is_linear() {
        let p = IntervalParams::default();
        let plan = interval_plan(13, 2, &p);
        let (wa, ca) = interval_well(1, vec![5.0; 201]);
        let (wb, cb) = interval_well(2, (0..=200).map(|k| k as f32).collect());
        let a = side(&wa, &ca, 0.0, 100.0, 0.6, false);
        let b = side(&wb, &cb, 0.0, 100.0, 0.6, false);
        let m = interval_path(&a, &b, 0, plan, &p).unwrap();
        assert_eq!(m.method, MatchMethod::Linear);
        assert_eq!(m.path.path, diagonal_pairs(46, 0.0, 100.0, 0.0, 100.0));
    }

    #[test]
    fn interval_too_few_shared_nodes_is_linear() {
        let p = IntervalParams::default();
        // The curve covers u <= 0.2 only: 10 finite nodes of 46 (>= 8, < 11).
        let mut gr: Vec<f32> = (0..=200).map(|k| k as f32).collect();
        for v in gr.iter_mut().skip(41) {
            *v = f32::NAN;
        }
        let (wa, ca) = interval_well(1, gr.clone());
        let (wb, cb) = interval_well(2, gr);
        let a = side(&wa, &ca, 0.0, 100.0, 0.6, false);
        let b = side(&wb, &cb, 0.0, 100.0, 0.6, false);
        let plan = interval_plan(13, 2, &p);
        let m = interval_path(&a, &b, 0, plan, &p).unwrap();
        assert_eq!(m.method, MatchMethod::Linear);
        // With a low requirement the same wells use DTW.
        let loose = sx_core::dtw::IntervalPlan {
            n_nodes: 46,
            min_shared: 8,
        };
        let m = interval_path(&a, &b, 0, loose, &p).unwrap();
        assert_eq!(m.method, MatchMethod::Dtw);
    }

    #[test]
    fn interval_identical_increasing_curve_is_dtw_diagonal() {
        let p = IntervalParams::default();
        let plan = interval_plan(13, 2, &p);
        let gr: Vec<f32> = (0..=200).map(|k| (k as f32).powf(1.3)).collect();
        let (wa, ca) = interval_well(1, gr.clone());
        let (wb, cb) = interval_well(2, gr);
        let a = side(&wa, &ca, 0.0, 100.0, 0.8, false);
        let b = side(&wb, &cb, 0.0, 100.0, 0.5, false);
        let m = interval_path(&a, &b, 0, plan, &p).unwrap();
        assert_eq!(m.method, MatchMethod::Dtw);
        assert_eq!(m.mean_cost, 0.0);
        assert_eq!(m.path.cost, 0.0);
        assert_eq!(m.path.path, diagonal_pairs(46, 0.0, 100.0, 0.0, 100.0));
        let want = (0.8f64 * 0.5).sqrt();
        assert!((f64::from(m.path.quality) - want).abs() < 1e-6);
        assert_eq!(m.span_ratio, 1.0);
    }

    #[test]
    fn interval_input_errors() {
        let p = IntervalParams::default();
        let plan = interval_plan(13, 2, &p);
        let gr: Vec<f32> = (0..=200).map(|k| k as f32).collect();
        let (wa, ca) = interval_well(1, gr.clone());
        let (wb, cb) = interval_well(2, gr);
        let bad = side(&wa, &ca, f64::NAN, 100.0, 0.8, false);
        let ok = side(&wb, &cb, 0.0, 100.0, 0.8, false);
        assert!(matches!(
            interval_path(&bad, &ok, 0, plan, &p),
            Err(SxError::InvalidInput(_))
        ));
        let bad_conf = side(&wa, &ca, 0.0, 100.0, f32::NAN, false);
        assert!(matches!(
            interval_path(&bad_conf, &ok, 0, plan, &p),
            Err(SxError::InvalidInput(_))
        ));
        let short = &ca[..100];
        let bad_len = side(&wa, short, 0.0, 100.0, 0.8, false);
        assert!(matches!(
            interval_path(&bad_len, &ok, 0, plan, &p),
            Err(SxError::Shape(_))
        ));
    }

    // ----------------------------------------------------- full / window modes

    /// A well with GR and RHOB on MD `lo_ft .. hi_ft`, with an optional GR spike.
    fn span_well(id: i32, lo_ft: f64, hi_ft: f64, spike_md: Option<f64>) -> WellLogs {
        let k0 = (lo_ft * 2.0) as i32;
        let len = ((hi_ft - lo_ft) * 2.0) as usize + 1;
        let mut gr = Vec::with_capacity(len);
        let mut rhob = Vec::with_capacity(len);
        for i in 0..len {
            let md = f64::from(k0 + i as i32) * 0.5;
            let rel = md - lo_ft;
            let mut g = 60.0 + 0.5 * (rel * 0.05).sin();
            if let Some(s) = spike_md {
                g += 120.0 * (-((md - s) / 3.0).powi(2)).exp();
            }
            gr.push(g as f32);
            rhob.push((2.4 + 0.002 * (rel * 0.05).cos()) as f32);
        }
        well(id, k0, len, vec![(LogKind::Gr, gr), (LogKind::Rhob, rhob)])
    }

    #[test]
    fn full_identical_wells_diagonal_quality_one() {
        let p = SpanParams::default();
        let wa = span_well(1, 900.0, 1200.0, None);
        let wb = span_well(2, 900.0, 1200.0, None);
        let a = SpanSide {
            logs: &wa,
            top_md_ft: 1000.0,
            base_md_ft: 1100.0,
        };
        let b = SpanSide {
            logs: &wb,
            top_md_ft: 1000.0,
            base_md_ft: 1100.0,
        };
        let path = full_path(&a, &b, 3, &p).unwrap();
        assert_eq!(path.interval, Some(3));
        assert_eq!((path.a, path.b), (WellId(1), WellId(2)));
        assert!(path.path.iter().all(|&(x, y)| x == y));
        assert_eq!(path.path.first(), Some(&(2000, 2000)));
        assert_eq!(path.path.last(), Some(&(2200, 2200)));
        assert_eq!(path.path.len(), 201);
        assert_eq!(path.quality, 1.0);
        assert_eq!(path.cost, 0.0);
    }

    fn spike_pair() -> (WellLogs, WellLogs) {
        // Same thickness (100 ft); spike at fraction 0.30 of a and 0.40 of b.
        (
            span_well(1, 900.0, 1200.0, Some(1030.0)),
            span_well(2, 1900.0, 2200.0, Some(2040.0)),
        )
    }

    fn assert_spike_aligned(path: &[(i32, i32)]) {
        let k_a = 2060.0; // MD 1030 ft
        let k_b = 4080.0; // MD 2040 ft
        let got = map_a_to_b(path, k_a).expect("spike inside the path");
        assert!((got - k_b).abs() <= 1.0, "spike maps to {got}, want {k_b}");
    }

    #[test]
    fn full_spike_alignment() {
        let (wa, wb) = spike_pair();
        let a = SpanSide {
            logs: &wa,
            top_md_ft: 1000.0,
            base_md_ft: 1100.0,
        };
        let b = SpanSide {
            logs: &wb,
            top_md_ft: 2000.0,
            base_md_ft: 2100.0,
        };
        let path = full_path(&a, &b, 0, &SpanParams::default()).unwrap();
        assert_eq!(path.path.first(), Some(&(2000, 4000)));
        assert_eq!(path.path.last(), Some(&(2200, 4200)));
        assert_spike_aligned(&path.path);
        let back = map_b_to_a(&path.path, 4080.0).unwrap();
        assert!((back - 2060.0).abs() <= 1.0);
    }

    #[test]
    fn window_spike_alignment_and_interval_none() {
        let (wa, wb) = spike_pair();
        let a = SpanSide {
            logs: &wa,
            top_md_ft: 1000.0,
            base_md_ft: 1100.0,
        };
        let b = SpanSide {
            logs: &wb,
            top_md_ft: 2000.0,
            base_md_ft: 2100.0,
        };
        let path = window_path(&a, &b, &SpanParams::window_default()).unwrap();
        assert_eq!(path.interval, None);
        assert_spike_aligned(&path.path);
    }

    #[test]
    fn span_errors() {
        let p = SpanParams::default();
        let wa = well(
            1,
            0,
            400,
            vec![(LogKind::Gr, (0..400).map(|k| (k % 17) as f32).collect())],
        );
        let wb = well(
            2,
            0,
            400,
            vec![(
                LogKind::Rhob,
                (0..400).map(|k| 2.0 + (k % 13) as f32 * 0.01).collect(),
            )],
        );
        let mk = mk_span;
        // no shared log
        assert!(matches!(
            full_path(&mk(&wa, 10.0, 100.0), &mk(&wb, 10.0, 100.0), 0, &p),
            Err(SxError::EmptyInput(_))
        ));
        // bad thickness
        assert!(matches!(
            full_path(&mk(&wa, 100.0, 100.0), &mk(&wa, 10.0, 100.0), 0, &p),
            Err(SxError::InvalidInput(_))
        ));
        assert!(matches!(
            window_path(&mk(&wa, f64::NAN, 100.0), &mk(&wa, 10.0, 100.0), &p),
            Err(SxError::InvalidInput(_))
        ));
        assert!(matches!(
            window_path(&mk(&wa, 10.0, 5.0), &mk(&wa, 10.0, 100.0), &p),
            Err(SxError::InvalidInput(_))
        ));
    }

    #[test]
    fn deterministic_bit_identical() {
        let (wa, wb) = spike_pair();
        let a = SpanSide {
            logs: &wa,
            top_md_ft: 1000.0,
            base_md_ft: 1100.0,
        };
        let b = SpanSide {
            logs: &wb,
            top_md_ft: 2000.0,
            base_md_ft: 2100.0,
        };
        let p = SpanParams::default();
        let one = full_path(&a, &b, 1, &p).unwrap();
        let two = full_path(&a, &b, 1, &p).unwrap();
        assert_eq!(one, two);
        assert_eq!(one.cost.to_bits(), two.cost.to_bits());
        let w1 = window_path(&a, &b, &SpanParams::window_default()).unwrap();
        let w2 = window_path(&a, &b, &SpanParams::window_default()).unwrap();
        assert_eq!(w1, w2);
    }

    fn mk_span(w: &WellLogs, top: f64, base: f64) -> SpanSide<'_> {
        SpanSide {
            logs: w,
            top_md_ft: top,
            base_md_ft: base,
        }
    }

    // ------------------------------------------------------------------ timing

    fn six_log_well(id: i32, len: usize, seed: u64) -> WellLogs {
        let mut rng = Lcg(seed);
        let mut curves: BTreeMap<LogKind, Vec<f32>> = BTreeMap::new();
        for (idx, log) in LogKind::ALL.into_iter().enumerate() {
            let mut v = Vec::with_capacity(len);
            let mut x = 0.0f64;
            for k in 0..len {
                x += rng.next() - 0.5;
                let base = match log {
                    LogKind::Resd => {
                        10.0 * (1.0 + 0.3 * (k as f64 * 0.01 * (idx + 1) as f64).sin())
                    }
                    _ => {
                        50.0 + 5.0 * idx as f64 + 20.0 * (k as f64 * 0.01 * (idx + 1) as f64).sin()
                    }
                };
                let val = if matches!(log, LogKind::Resd) {
                    (base + x * 0.1).abs() + 0.1
                } else {
                    base + x
                };
                v.push(val as f32);
            }
            curves.insert(log, v);
        }
        let mut w = WellLogs::new(WellId(id), GridRange::new(0, len).unwrap());
        w.curves = curves;
        w
    }

    fn best_of<F: FnMut()>(reps: usize, mut f: F) -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let t = std::time::Instant::now();
            f();
            best = best.min(t.elapsed().as_secs_f64());
        }
        best * 1e3
    }

    #[test]
    #[ignore]
    fn dtw_timing() {
        // banded_dtw N=2000 w=239 with stored costs
        let n = 2000;
        let mut rng = Lcg(7);
        let stored: Vec<f32> = (0..n * n).map(|_| rng.next() as f32).collect();
        let ms = best_of(10, || {
            let p = banded_dtw(n, n, 239, |i, j| f64::from(stored[i * n + j])).unwrap();
            std::hint::black_box(p);
        });
        println!("banded_dtw N=2000 w=239: {ms:.3} ms (budget 10)");

        // interval_path N=46, 6 logs, 2000-sample wells
        let wa = six_log_well(1, 2000, 1);
        let wb = six_log_well(2, 2000, 2);
        let coord: Vec<f64> = (0..2000).map(|k| k as f64 / 2000.0).collect();
        let p = IntervalParams::default();
        let plan = interval_plan(13, 2, &p);
        let a = side(&wa, &coord, 0.0, 1000.0, 0.8, false);
        let b = side(&wb, &coord, 0.0, 1000.0, 0.7, false);
        let ms = best_of(200, || {
            let m = interval_path(&a, &b, 0, plan, &p).unwrap();
            assert_eq!(m.method, MatchMethod::Dtw);
            std::hint::black_box(m);
        });
        println!("interval_path N=46, 6 logs, 2000 samples: {ms:.4} ms (budget 0.3)");

        // full / window
        let wa = six_log_well(1, 2400, 3);
        let wb = six_log_well(2, 2400, 4);
        let mk = mk_span;
        let (a, b) = (mk(&wa, 100.0, 299.5), mk(&wb, 100.0, 299.5)); // N = 400
        let sp = SpanParams::default();
        let ms = best_of(20, || {
            std::hint::black_box(full_path(&a, &b, 0, &sp).unwrap());
        });
        println!("full_path N=400, 6 logs: {ms:.3} ms (budget 3)");
        let (a, b) = (mk(&wa, 100.0, 1100.0), mk(&wb, 100.0, 1100.0)); // N = 2000
        let ms = best_of(10, || {
            std::hint::black_box(full_path(&a, &b, 0, &sp).unwrap());
        });
        println!("full_path N=2000, 6 logs: {ms:.3} ms (budget 40)");
        let wp = SpanParams::window_default();
        let ms = best_of(10, || {
            std::hint::black_box(window_path(&a, &b, &wp).unwrap());
        });
        println!("window_path N=2000, 6 logs: {ms:.3} ms (budget 80)");
    }
}
