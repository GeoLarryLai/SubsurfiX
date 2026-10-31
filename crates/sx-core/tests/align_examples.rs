// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Tests: worked rough-alignment examples and a
//! few edge cases.
//!
//! The node-plan and pair-quality example tests the private
//! pair-map assembly and lives next to it in `src/align/pairs.rs`.

mod align_examples {
    use arrow::datatypes::DataType;
    use arrow::record_batch::RecordBatch;
    use sx_core::align::{
        build_breaks, canonical_order, coord_at_md, elev_at_md, interval_of_coord, interval_table,
        intervals_batch, layer_coord, layer_coord_batch, md_at_elev, pairs_rough_batch,
        rough_align, tops_estimated_batch, AlignInput, AlignParams, BoundaryRow, BoundaryStatus,
        ConfidenceParams, CoordSource, PairStatus, RoughAlignment, UserPick,
    };
    use sx_core::dtw::IntervalParams;
    use sx_core::io::parquet::schema_tag;
    use sx_core::types::{
        GridRange, IntervalState, LogKind, TopSource, WellGeometry, WellHeader, WellId, WellLogs,
    };
    use sx_core::{SxError, SxResult};

    // ------------------------------------------------------------------ helpers

    fn header(id: i32, x_m: f64, y_m: f64) -> WellHeader {
        WellHeader {
            id: WellId(id),
            name: format!("w{id}"),
            x_m,
            y_m,
            kb_elev_ft: 1200.0,
        }
    }

    /// Vertical well: no geometry samples, so elevation = KB − MD.
    fn vertical(id: i32) -> WellGeometry {
        WellGeometry {
            id: WellId(id),
            grid: GridRange::new(0, 0).unwrap(),
            tvd_ft: vec![],
            elev_ft: vec![],
            x_m: vec![],
            y_m: vec![],
        }
    }

    fn logs(id: i32, k_start: i32, len: usize, curves: Vec<(LogKind, Vec<f32>)>) -> WellLogs {
        let mut w = WellLogs::new(WellId(id), GridRange::new(k_start, len).unwrap());
        for (log, v) in curves {
            w.provenance.insert(log, vec![0; v.len()]);
            w.curves.insert(log, v);
        }
        w
    }

    fn pick(id: i32, name: &str, md_ft: f64) -> UserPick {
        UserPick {
            well: WellId(id),
            name: name.into(),
            md_ft,
        }
    }

    #[derive(Clone)]
    struct Wells {
        headers: Vec<WellHeader>,
        geometry: Vec<WellGeometry>,
        logs: Vec<WellLogs>,
        picks: Vec<UserPick>,
    }

    impl Wells {
        /// Vertical wells with KB 1,200 ft (MD = 1,200 − elevation), no curves.
        fn new(xy: &[(i32, f64, f64)]) -> Self {
            Self {
                headers: xy.iter().map(|&(id, x, y)| header(id, x, y)).collect(),
                geometry: xy.iter().map(|&(id, _, _)| vertical(id)).collect(),
                logs: xy
                    .iter()
                    .map(|&(id, _, _)| logs(id, 0, 0, vec![]))
                    .collect(),
                picks: vec![],
            }
        }

        fn run_with(&self, p: &AlignParams) -> SxResult<RoughAlignment> {
            let input = AlignInput {
                headers: &self.headers,
                geometry: &self.geometry,
                logs: &self.logs,
                picks: &self.picks,
                order: &[],
            };
            rough_align(&input, p, &IntervalParams::default())
        }

        fn run(&self) -> SxResult<RoughAlignment> {
            self.run_with(&AlignParams::default())
        }
    }

    fn boundary(r: &RoughAlignment, well: i32, b: u16) -> &BoundaryRow {
        r.boundaries
            .iter()
            .find(|x| x.well == WellId(well) && x.boundary == b)
            .unwrap()
    }

    fn interval_state(r: &RoughAlignment, well: i32, i: u16) -> IntervalState {
        r.intervals
            .iter()
            .find(|x| x.row.well == WellId(well) && x.row.interval == i)
            .unwrap()
            .row
            .state
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a.is_nan() && b.is_nan()) || (a - b).abs() <= tol
    }

    /// Inputs: well 1 has T03 above T02; wells 2 and 3 fix the
    /// canonical order T01, T02, T03. Well 1's logs grid spans 4,800–5,250 ft.
    fn example_14_3() -> Wells {
        let mut w = Wells::new(&[(1, 0.0, 0.0), (2, 1000.0, 0.0), (3, 2000.0, 0.0)]);
        w.logs[0] = logs(1, 9600, 901, vec![]);
        w.picks = vec![
            pick(1, "T01", 5000.0),
            pick(1, "T02", 5100.0),
            pick(1, "T03", 5050.0),
            pick(2, "T01", 5000.0),
            pick(2, "T02", 5050.0),
            pick(2, "T03", 5100.0),
            pick(3, "T01", 5000.0),
            pick(3, "T02", 5050.0),
            pick(3, "T03", 5100.0),
        ];
        w
    }

    // ---------------------------------------------------------- worked examples

    /// Three donors, weighted lower median in elevation.
    #[test]
    fn align_example_14_1_estimated_top_in_elevation() {
        let mut w = Wells::new(&[
            (1, 0.0, 0.0),
            (2, 3000.0, 0.0),
            (3, 0.0, 1000.0),
            (4, -2000.0, 0.0),
        ]);
        w.picks = vec![
            pick(1, "T01", 4200.0),
            pick(2, "T01", 4200.0),
            pick(2, "T03", 5200.0), // −4,000 ft
            pick(3, "T01", 4200.0),
            pick(3, "T03", 5250.0), // −4,050 ft
            pick(4, "T01", 4200.0),
            pick(4, "T03", 5300.0), // −4,100 ft
        ];
        let r = w.run().unwrap();
        assert_eq!(r.top_set.names, vec!["T01", "T03"]);
        let b = boundary(&r, 1, 2);
        assert_eq!(b.elev_ft, -4050.0);
        assert_eq!(b.md_ft, 5250.0);
        assert_eq!(b.source, Some(TopSource::Estimated));
        assert_eq!(b.status, BoundaryStatus::Kept);
        assert_eq!(b.n_donors, 3);
        assert_eq!(r.coords[0].source, CoordSource::WithEstimates);
        assert_eq!(r.coords[1].source, CoordSource::Own);
    }

    /// Break table, interval table and coordinate rows.
    #[test]
    fn align_example_14_2_break_table_and_coordinate() {
        let nan = f64::NAN;
        let md = [4800.0, 5000.0, 5100.0, nan, 5200.0, 5200.0, 5400.0];
        let b = build_breaks(&md);
        assert_eq!(b.md_ft, vec![4800.0, 5000.0, 5100.0, 5200.0, 5400.0]);
        assert_eq!(b.coord_above, vec![0, 1, 2, 4, 6]);
        assert_eq!(b.coord_below, vec![0, 1, 2, 5, 6]);
        assert_eq!(b.kept, vec![true, true, true, false, true, true, true]);
        assert!(b.has_coord());

        let (user, pad) = (Some(TopSource::User), Some(TopSource::Pad));
        let src = [pad, user, user, None, user, user, pad];
        let rows = interval_table(WellId(1), &md, &src, &ConfidenceParams::default()).unwrap();
        let states: Vec<IntervalState> = rows.iter().map(|r| r.row.state).collect();
        use IntervalState::*;
        assert_eq!(states, vec![Pad, Observed, Unknown, Unknown, Pinched, Pad]);
        let conf: Vec<f32> = rows.iter().map(|r| r.confidence).collect();
        assert_eq!(conf, vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0]);
        assert!(rows[2].thickness_ft.is_nan() && rows[3].thickness_ft.is_nan());
        assert_eq!(rows[4].thickness_ft, 0.0);

        let grid = GridRange::new(9599, 1602).unwrap(); // 4,799.5 .. 5,400 ft
        let u_all = layer_coord(grid, &b);
        for (md_ft, u, interval) in [
            (4799.5, nan, -1),
            (4900.0, 0.5, 0),
            (5025.0, 1.25, 1),
            (5150.0, 3.0, 3),
            (5200.0, 5.0, 5),
            (5300.0, 5.5, 5),
            (5400.0, 6.0, 5),
        ] {
            let got = coord_at_md(&b, md_ft);
            assert!(close(got, u, 1e-12), "u({md_ft}) = {got}, want {u}");
            let k = (md_ft / 0.5) as i32;
            let from_grid = u_all[grid.index_of_k(k).unwrap()];
            assert!(close(from_grid, u, 1e-12), "grid u({md_ft}) = {from_grid}");
            assert_eq!(interval_of_coord(got, 5), interval, "interval at {md_ft}");
        }
    }

    /// Order conflict.
    #[test]
    fn align_example_14_3_order_conflict() {
        let r = example_14_3().run().unwrap();
        assert_eq!(r.top_set.names, vec!["T01", "T02", "T03"]);
        let t02 = boundary(&r, 1, 2);
        assert_eq!(t02.status, BoundaryStatus::OrderConflict);
        assert_eq!(t02.md_ft, 5100.0);
        assert_eq!(t02.source, Some(TopSource::User));
        assert_eq!(boundary(&r, 1, 3).status, BoundaryStatus::Kept);
        assert_eq!(boundary(&r, 1, 4).md_ft, 5250.0);
        assert_eq!(interval_state(&r, 1, 1), IntervalState::Unknown);
        assert_eq!(interval_state(&r, 1, 2), IntervalState::Unknown);
        let c = &r.coords[0];
        let u = |k: i32| c.layer_coord[c.grid.index_of_k(k).unwrap()];
        assert_eq!(u(10000), 1.0);
        assert!(close(u(10050), 2.0, 1e-12));
        assert_eq!(u(10100), 3.0);
        assert_eq!(c.source, CoordSource::Own);
    }

    /// Depth helpers on a deviated well.
    #[test]
    fn align_depth_helpers_on_deviated_well() {
        let h = WellHeader {
            kb_elev_ft: 1000.0,
            ..header(1, 0.0, 0.0)
        };
        let g = WellGeometry {
            id: WellId(1),
            grid: GridRange::new(2000, 3).unwrap(), // md 1000, 1000.5, 1001
            tvd_ft: vec![1000.0, 1000.4, 1000.8],
            elev_ft: vec![0.0, -0.4, -0.8],
            x_m: vec![0.0; 3],
            y_m: vec![0.0; 3],
        };
        assert!(close(elev_at_md(&h, &g, 1000.25), -0.2, 1e-9));
        assert!(close(md_at_elev(&h, &g, -0.6), 1000.75, 1e-9));
        assert!(close(elev_at_md(&h, &g, 1002.0), -1.8, 1e-9));
        assert!(close(md_at_elev(&h, &g, 10.0), 990.0, 1e-9));
        for md in [999.0, 1000.0, 1000.25, 1000.5, 1000.9, 1003.0] {
            let back = md_at_elev(&h, &g, elev_at_md(&h, &g, md));
            assert!(close(back, md, 1e-9), "md {md} back {back}");
        }
        assert!(elev_at_md(&h, &g, f64::NAN).is_nan());
    }

    // --------------------------------------------------------------- edge cases

    #[test]
    fn align_input_and_config_errors() {
        let mut w = Wells::new(&[(1, 0.0, 0.0), (2, 1000.0, 0.0)]);
        w.picks = vec![
            pick(1, "A", 5000.0),
            pick(2, "A", 5000.0),
            pick(2, "B", 5100.0),
        ];
        assert!(w.run().is_ok());

        let empty = Wells::new(&[]);
        assert!(matches!(empty.run(), Err(SxError::EmptyInput(_))));

        let mut bad = w.clone();
        bad.geometry.pop();
        assert!(matches!(bad.run(), Err(SxError::Shape(_))));

        let mut bad = w.clone();
        bad.headers.swap(0, 1);
        bad.geometry.swap(0, 1);
        bad.logs.swap(0, 1);
        assert!(matches!(bad.run(), Err(SxError::InvalidInput(_))));

        let mut bad = w.clone();
        bad.logs[1].id = WellId(9);
        assert!(matches!(bad.run(), Err(SxError::InvalidInput(_))));

        let mut bad = w.clone();
        bad.picks = vec![pick(1, "A", f64::NAN), pick(9, "B", 5000.0)];
        assert!(matches!(bad.run(), Err(SxError::EmptyInput(_))));

        for p in [
            AlignParams {
                pad_ft: 0.0,
                ..AlignParams::default()
            },
            AlignParams {
                study_top: "Z".into(),
                ..AlignParams::default()
            },
            AlignParams {
                study_top: "B".into(),
                study_base: "A".into(),
                ..AlignParams::default()
            },
        ] {
            assert!(matches!(w.run_with(&p), Err(SxError::Config(_))), "{p:?}");
        }
    }

    #[test]
    fn align_one_top_pads_and_no_coordinate() {
        let (user, pad) = (Some(TopSource::User), Some(TopSource::Pad));
        let rows = interval_table(
            WellId(1),
            &[4800.0, 5000.0, 5200.0],
            &[pad, user, pad],
            &ConfidenceParams::default(),
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r.row.state == IntervalState::Pad));
        assert!(rows.iter().all(|r| r.confidence == 0.5));

        let b = build_breaks(&[f64::NAN, 5000.0, f64::NAN]);
        assert!(!b.has_coord());
        let grid = GridRange::new(9990, 21).unwrap();
        assert!(layer_coord(grid, &b).iter().all(|u| u.is_nan()));
        assert_eq!(interval_of_coord(f64::NAN, 1), -1);
    }

    #[test]
    fn align_canonical_order_ties_and_drops() {
        let e = |n: &str, v: f64| (n.to_string(), v);
        let order = canonical_order(&[
            e("B", -100.0),
            e("A", -100.0),
            e("C", -50.0),
            e("C", -150.0), // even count: median −100, ties with A and B
            e("D", f64::NAN),
            e("E", -300.0),
        ]);
        assert_eq!(order, vec!["A", "B", "C", "E"]);
    }

    #[test]
    fn align_tables_have_sheet_columns() {
        let r = example_14_3().run().unwrap();
        let list =
            |t: DataType| DataType::List(arrow::datatypes::Field::new("item", t, true).into());
        let check = |b: &RecordBatch, tag: &str, rows: usize, cols: &[(&str, DataType)]| {
            assert_eq!(schema_tag(b), Some(tag));
            assert_eq!(b.num_rows(), rows, "{tag}");
            let got: Vec<(String, DataType)> = b
                .schema_ref()
                .fields()
                .iter()
                .map(|f| (f.name().clone(), f.data_type().clone()))
                .collect();
            let want: Vec<(String, DataType)> = cols
                .iter()
                .map(|(n, t)| (n.to_string(), t.clone()))
                .collect();
            assert_eq!(got, want, "{tag}");
        };
        use DataType::{Float32, Float64, Int32, UInt8, Utf8};
        check(
            &intervals_batch(&r).unwrap(),
            "subsurfix.intervals/1",
            3 * 4,
            &[
                ("well_id", Int32),
                ("interval", Int32),
                ("top_name", Utf8),
                ("base_name", Utf8),
                ("top_md_ft", Float64),
                ("base_md_ft", Float64),
                ("thickness_ft", Float64),
                ("state", UInt8),
                ("confidence", Float32),
            ],
        );
        check(
            &tops_estimated_batch(&r).unwrap(),
            "subsurfix.tops_estimated/1",
            3 * 5,
            &[
                ("well_id", Int32),
                ("boundary", Int32),
                ("top_name", Utf8),
                ("md_ft", Float64),
                ("elev_ft", Float64),
                ("source", UInt8),
                ("status", UInt8),
                ("n_donors", Int32),
            ],
        );
        check(
            &pairs_rough_batch(&r).unwrap(),
            "subsurfix.pairs_rough/1",
            0,
            &[
                ("target_id", Int32),
                ("donor_id", Int32),
                ("rank", Int32),
                ("distance_m", Float64),
                ("status", UInt8),
                ("quality", Float32),
                ("span_ratio", Float64),
                ("total_cost", Float64),
                ("mean_cost", Float64),
                ("n_dtw_intervals", Int32),
                ("n_linear_intervals", Int32),
                ("intervals", list(Int32)),
                ("interval_quality", list(Float32)),
                ("interval_method", list(UInt8)),
                ("k_target", list(Int32)),
                ("k_donor", list(Int32)),
            ],
        );
        check(
            &layer_coord_batch(&r).unwrap(),
            "subsurfix.layer_coord/1",
            901,
            &[
                ("well_id", Int32),
                ("k", Int32),
                ("md_ft", Float64),
                ("layer_coord", Float64),
                ("interval", Int32),
                ("interval_state", UInt8),
                ("coord_source", UInt8),
            ],
        );
    }

    /// No interval keeps a channel, so every accepted
    /// interval is linear.
    #[test]
    fn align_constant_curves_give_linear_interval_map() {
        let mut w = Wells::new(&[(1, 0.0, 0.0), (2, 1000.0, 0.0)]);
        for (i, id) in [1, 2].into_iter().enumerate() {
            w.logs[i] = logs(id, 9000, 2001, vec![(LogKind::Gr, vec![50.0; 2001])]);
        }
        w.picks = vec![
            pick(1, "T01", 4900.0),
            pick(1, "T02", 5100.0),
            pick(2, "T01", 4900.0),
            pick(2, "T02", 5100.0),
        ];
        let r = w.run().unwrap();
        let ids: Vec<(i32, i32, u16)> = r
            .pairs
            .iter()
            .map(|m| (m.path.a.0, m.path.b.0, m.rank))
            .collect();
        assert_eq!(ids, vec![(1, 2, 1), (2, 1, 1)]);
        for m in &r.pairs {
            assert_eq!(m.status, PairStatus::IntervalMap);
            assert_eq!((m.n_dtw_intervals, m.n_linear_intervals), (0, 3));
            assert_eq!(m.intervals, vec![0, 1, 2]);
            assert_eq!(m.interval_method, vec![1, 1, 1]);
            assert!((f64::from(m.path.quality) - 0.45).abs() < 1e-6);
            assert!(m.path.cost.is_nan() && m.mean_cost.is_nan());
            assert_eq!(m.span_ratio, 1.0);
            assert_eq!(m.path.path.first(), Some(&(9400, 9400)));
            assert_eq!(m.path.path.last(), Some(&(10600, 10600)));
            assert!(m
                .path
                .path
                .windows(2)
                .all(|p| p[1].0 > p[0].0 && p[1].1 >= p[0].1));
        }
    }

    /// A well with no picks and too few donors has no coordinate, so its
    /// pairs use depth mode.
    #[test]
    fn align_well_without_coordinate_uses_depth_mode() {
        let mut w = Wells::new(&[(1, 0.0, 0.0), (2, 1000.0, 0.0)]);
        let gr: Vec<f32> = (0..2001)
            .map(|i| 60.0 + 30.0 * (i as f32 * 0.02).sin())
            .collect();
        for (i, id) in [1, 2].into_iter().enumerate() {
            w.logs[i] = logs(id, 9000, 2001, vec![(LogKind::Gr, gr.clone())]);
        }
        w.picks = vec![pick(1, "T01", 4900.0), pick(1, "T02", 5100.0)];
        let r = w.run().unwrap();
        assert_eq!(r.coords[0].source, CoordSource::Own);
        assert_eq!(r.coords[1].source, CoordSource::None);
        assert!(r.coords[1].interval.iter().all(|&i| i == -1));
        for b in 0..4 {
            assert_eq!(boundary(&r, 2, b).status, BoundaryStatus::Missing);
        }
        assert_eq!(r.pairs.len(), 2);
        for m in &r.pairs {
            assert_eq!(m.status, PairStatus::DepthMap);
            assert_eq!(m.path.quality, 0.2);
            assert_eq!(m.span_ratio, 1.0);
            assert!(m.intervals.is_empty() && m.path.interval.is_none());
            assert!(m.path.cost.is_finite() && m.mean_cost.is_finite());
            assert_eq!(m.path.path.first(), Some(&(9000, 9000)));
            assert_eq!(m.path.path.last(), Some(&(11000, 11000)));
            assert!(m
                .path
                .path
                .windows(2)
                .all(|p| p[1].0 >= p[0].0 && p[1].1 >= p[0].1));
        }
    }
}
