// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Tests: a small graph example and a tiny end-to-end run on three
//! synthetic wells.
//!
//! The reference solve example and the monotonicity examples test private
//! helpers and live next to them in `src/correlate/solve.rs`.

mod correlate_solve {
    use sx_core::config::{
        CorrelateGraphConfig, CorrelateSolveConfig, CorrelateTopsConfig, DtwConfig,
    };
    use sx_core::correlate::graph::build_well_graph;
    use sx_core::correlate::solve::{
        layer_time_batch, md_at_layer_time, solve_report_batch, ReportRowKind,
    };
    use sx_core::correlate::tops::CheckState;
    use sx_core::correlate::{run_correlation, CorrelationConfig, CorrelationInputs};
    use sx_core::io::parquet::schema_tag;
    use sx_core::types::{
        GridRange, IntervalRow, IntervalState, LogKind, TopPick, TopSet, TopSource, WellGeometry,
        WellHeader, WellId, WellLogs,
    };

    fn close(got: f64, want: f64, tol: f64) {
        assert!(
            (got - want).abs() <= tol + tol * want.abs(),
            "got {got}, want {want}"
        );
    }

    fn header(id: i32, x_m: f64, y_m: f64) -> WellHeader {
        WellHeader {
            id: WellId(id),
            name: format!("W{id}"),
            x_m,
            y_m,
            kb_elev_ft: 1000.0,
        }
    }

    /// Graph example.
    #[test]
    fn correlate_graph_example() {
        let headers = vec![
            header(1, 0.0, 0.0),
            header(2, 1000.0, 0.0),
            header(3, 0.0, 1000.0),
            header(4, 1000.0, 1000.0),
            header(5, 20000.0, 0.0),
        ];
        let cfg = CorrelateGraphConfig {
            k_graph: 6,
            max_edge_m: 8000.0,
        };
        let g = build_well_graph(&headers, &cfg).unwrap();
        let got: Vec<(i32, i32, bool)> = g.edges.iter().map(|e| (e.a.0, e.b.0, e.in_mst)).collect();
        assert_eq!(
            got,
            vec![
                (1, 2, true),
                (1, 3, true),
                (1, 4, false),
                (2, 3, false),
                (2, 4, true),
                (2, 5, true),
                (3, 4, false),
            ]
        );
        close(g.edges[2].length_m, 2f64.sqrt() * 1000.0, 1e-9);
        close(g.edges[5].length_m, 19000.0, 1e-9);
        close(g.median_length_m, 1000.0, 1e-9);
        assert_eq!(g.edge_index(WellId(5), WellId(2)), Some(5));
        assert_eq!(g.edge_index(WellId(1), WellId(5)), None);
        let nb: Vec<i32> = g
            .neighbors_by_distance(WellId(2))
            .iter()
            .map(|n| n.0 .0)
            .collect();
        assert_eq!(nb, vec![1, 4, 3, 5]);
        let q = f64::from(0.9f32);
        close(g.edge_weight(1000.0, 0.9), q * (-1f64).exp(), 1e-9);
        close(g.edge_weight(1000.0, 0.9), 0.3310915, 1e-6);
        close(g.edge_weight(19000.0, 0.9), q * (-19f64).exp(), 1e-9);
        close(g.edge_weight(19000.0, 0.9), 5.04e-9, 1e-2);
    }

    /// One well on grid 900–1200 ft: tops A and B, pads at 950 and 1150 ft,
    /// GR from a warped true layer time.
    struct Synth {
        header: WellHeader,
        geometry: WellGeometry,
        logs: WellLogs,
        rows: Vec<IntervalRow>,
        picks: Vec<TopPick>,
    }

    fn synth_well(id: i32, x_m: f64, y_m: f64, top_a: f64, top_b: f64, warp: f64) -> Synth {
        let grid = GridRange::new(1800, 601).unwrap();
        let bounds = [950.0, top_a, top_b, 1150.0];
        let true_time = |md: f64| -> f64 {
            for i in 0..3 {
                let (t, b) = (bounds[i], bounds[i + 1]);
                if md >= t && md <= b {
                    let f = (md - t) / (b - t);
                    return i as f64 + f + 0.08 * warp * (std::f64::consts::PI * f).sin();
                }
            }
            f64::NAN
        };
        let md: Vec<f64> = (0..grid.len).map(|s| grid.md_ft(s)).collect();
        let gr: Vec<f32> = md
            .iter()
            .map(|&m| {
                let t = true_time(m.clamp(950.0, 1150.0));
                let tau = std::f64::consts::TAU;
                (75.0 + 25.0 * (tau * 5.0 * t).sin() + 10.0 * (tau * 13.0 * t).sin()) as f32
            })
            .collect();
        let mut logs = WellLogs::new(WellId(id), grid);
        logs.curves.insert(LogKind::Gr, gr);
        logs.provenance.insert(LogKind::Gr, vec![0u8; grid.len]);
        let geometry = WellGeometry {
            id: WellId(id),
            grid,
            tvd_ft: md.clone(),
            elev_ft: md.iter().map(|m| 1000.0 - m).collect(),
            x_m: vec![x_m; grid.len],
            y_m: vec![y_m; grid.len],
        };
        let states = [
            IntervalState::Pad,
            IntervalState::Observed,
            IntervalState::Pad,
        ];
        let rows = (0..3)
            .map(|i| IntervalRow {
                well: WellId(id),
                interval: i as u16,
                top_md_ft: bounds[i],
                base_md_ft: bounds[i + 1],
                state: states[i],
            })
            .collect();
        let picks = vec![
            TopPick {
                well: WellId(id),
                top: 0,
                md_ft: top_a,
                source: TopSource::User,
            },
            TopPick {
                well: WellId(id),
                top: 1,
                md_ft: top_b,
                source: TopSource::User,
            },
        ];
        Synth {
            header: header(id, x_m, y_m),
            geometry,
            logs,
            rows,
            picks,
        }
    }

    /// Tiny end-to-end run: Pass 1, stub tops check, Pass 2 (= Pass 1).
    #[test]
    fn correlate_end_to_end_three_wells() {
        let wells = [
            synth_well(1, 0.0, 0.0, 1000.0, 1100.0, 0.0),
            synth_well(2, 1000.0, 0.0, 1010.0, 1105.0, 0.5),
            synth_well(3, 0.0, 1000.0, 995.0, 1090.0, -0.5),
        ];
        let headers: Vec<WellHeader> = wells.iter().map(|w| w.header.clone()).collect();
        let geometry: Vec<WellGeometry> = wells.iter().map(|w| w.geometry.clone()).collect();
        let logs: Vec<WellLogs> = wells.iter().map(|w| w.logs.clone()).collect();
        let intervals: Vec<IntervalRow> = wells.iter().flat_map(|w| w.rows.clone()).collect();
        let picks: Vec<TopPick> = wells.iter().flat_map(|w| w.picks.clone()).collect();
        let tops = TopSet::new(vec!["A".into(), "B".into()]).unwrap();
        let inp = CorrelationInputs {
            headers: &headers,
            geometry: &geometry,
            logs: &logs,
            tops: &tops,
            intervals: &intervals,
            picks: &picks,
        };
        let (graph_cfg, solve_cfg, tops_cfg, dtw_cfg) = (
            CorrelateGraphConfig::default(),
            CorrelateSolveConfig::default(),
            CorrelateTopsConfig::default(),
            DtwConfig::default(),
        );
        let cfg = CorrelationConfig {
            graph: &graph_cfg,
            solve: &solve_cfg,
            tops: &tops_cfg,
            dtw: &dtw_cfg,
        };
        let out = run_correlation(&inp, &cfg).unwrap();

        // Graph: one triangle, three edges.
        assert_eq!(out.graph.edges.len(), 3);
        assert_eq!(out.boundaries_pass1, out.boundaries_pass2);
        assert_eq!(out.boundaries_pass1[1].source[1], TopSource::User);
        assert_eq!(out.tops_check.rows.len(), 6);
        // Three wells give each well two voters, below min_votes = 3: nothing is
        // checked fully and nothing moves.
        assert!(out
            .tops_check
            .rows
            .iter()
            .all(|r| r.state == CheckState::TooFewVotes && !r.moved));

        // Layer time: full grid, NaN outside the pads, monotone, pinned at user tops.
        assert_eq!(out.layer_time.len(), 3);
        for (lt, w) in out.layer_time.iter().zip(&wells) {
            assert_eq!(lt.layer_time.len(), lt.grid.len);
            let mut prev = f64::NEG_INFINITY;
            for (s, &t) in lt.layer_time.iter().enumerate() {
                let md = lt.grid.md_ft(s);
                if (950.0..=1150.0).contains(&md) {
                    assert!(t.is_finite(), "well {} md {md}", lt.well.0);
                    assert!(t >= prev, "well {} md {md}: {t} < {prev}", lt.well.0);
                    assert!((0.0..=3.0).contains(&t));
                    prev = t;
                } else {
                    assert!(t.is_nan());
                }
            }
            let at = |md: f64| lt.layer_time[lt.grid.index_of_k((md * 2.0) as i32).unwrap()];
            assert_eq!(at(w.rows[1].top_md_ft), 1.0);
            assert_eq!(at(w.rows[1].base_md_ft), 2.0);
            close(
                md_at_layer_time(&lt.grid, &lt.layer_time, 1.0),
                w.rows[1].top_md_ft,
                1e-9,
            );
        }

        // Report: per pass 3 interval rows + 3 edge rows each; Pass 2 = Pass 1.
        assert_eq!(out.solve_report.len(), 24);
        let (p1, p2) = out.solve_report.split_at(12);
        for (a, b) in p1.iter().zip(p2) {
            assert_eq!((a.pass, b.pass), (1, 2));
            assert_eq!(a.kind, b.kind);
            assert_eq!(
                (a.interval, a.n_rows, a.n_unknowns),
                (b.interval, b.n_rows, b.n_unknowns)
            );
            assert_eq!(a.median_residual.to_bits(), b.median_residual.to_bits());
            if a.kind == ReportRowKind::Edge {
                assert!(a.path_recomputed && !b.path_recomputed);
            } else {
                assert_eq!(a.n_dtw_failed, 0);
                assert!(a.n_rows > 0 && a.n_unknowns > 0);
                assert!(a.unconstrained.is_empty());
            }
        }

        // Tables.
        let lt = layer_time_batch(&out.layer_time).unwrap();
        assert_eq!(lt.num_rows(), 3 * 601);
        assert_eq!(schema_tag(&lt), Some("subsurfix.layer_time/1"));
        let rep = solve_report_batch(&out.solve_report).unwrap();
        assert_eq!(rep.num_rows(), 24);
        assert_eq!(rep.num_columns(), 17);
        assert_eq!(schema_tag(&rep), Some("subsurfix.solve_report/1"));
    }
}
