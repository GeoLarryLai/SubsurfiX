// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Integration tests for the QC rule set: examples from the
//! committed fixtures, whole-stage behavior on synthetic fleets, the bit
//! invariant, and the schemas of the three products.

mod qc_rules {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use proptest::prelude::*;
    use serde_json::Value;
    use sx_core::config::QcConfig;
    use sx_core::io::csv::batch_to_csv_string;
    use sx_core::io::parquet::{read_batch_expect, schema_tag, write_batch};
    use sx_core::qc::{
        flags_batch, jsd_scores, kl_knn_1d, logs_batch, plan_rescue, plateau_mask, qc_report_batch,
        repeat_mask, ring_test, run_qc_stage, short_run_mask, tail_mask, Baseline, ChannelAction,
        PlateauParams, QcInput, QcStageOutput, RescueReason, LOGS_QC_TAG, QC_CHANNEL, QC_FLAGS_TAG,
        QC_OUT_OF_BOUNDS, QC_PLATEAU, QC_REPEAT, QC_TAIL, SIGMA_MEAN,
    };
    use sx_core::qc::{ChannelStats, QcFlags};
    use sx_core::types::{GridRange, LogKind, WellHeader, WellId, WellLogs};
    use sx_core::util::rng::{rng_for, standard_normal};
    use sx_core::SxError;

    // ------------------------------------------------------------------ helpers

    fn fixture(name: &str) -> Value {
        let path: PathBuf = [
            env!("CARGO_MANIFEST_DIR"),
            "..",
            "..",
            "tests",
            "fixtures",
            "qc",
            name,
        ]
        .iter()
        .collect();
        let text = std::fs::read_to_string(&path).expect("fixture readable");
        serde_json::from_str(&text).expect("fixture parses")
    }

    fn f64s(v: &Value) -> Vec<f64> {
        v.as_array()
            .expect("array")
            .iter()
            .map(|x| x.as_f64().unwrap_or(f64::NAN))
            .collect()
    }

    fn usizes(v: &Value) -> Vec<usize> {
        v.as_array()
            .expect("array")
            .iter()
            .map(|x| x.as_u64().expect("uint") as usize)
            .collect()
    }

    fn idx(mask: &[bool]) -> Vec<usize> {
        mask.iter()
            .enumerate()
            .filter(|(_, &m)| m)
            .map(|(i, _)| i)
            .collect()
    }

    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs().max(1e-300)
    }

    // ------------------------------------------------------- examples

    #[test]
    fn example_1_tail_pad() {
        let f = fixture("tail_pad.json");
        let v = f64s(&f["values"]);
        let pct = f64s(&f["pct"]);
        let m = tail_mask(
            &v,
            [pct[0], pct[1]],
            f["pad_frac"].as_f64().unwrap(),
            f["min_count"].as_u64().unwrap() as usize,
        );
        assert_eq!(idx(&m), usizes(&f["masked"]));
        assert!(rel(f["pad"].as_f64().unwrap(), 3.98) < 1e-9);
        // a plain clip (pad 0) would also drop 50.0
        let plain = tail_mask(&v, [pct[0], pct[1]], 0.0, 5);
        assert!(plain[0] && plain[400]);
    }

    #[test]
    fn example_2_repeat_inside_drift() {
        let f = fixture("repeat_drift.json");
        let v = f64s(&f["values"]);
        let m = repeat_mask(
            &v,
            f["min_len"].as_u64().unwrap() as usize,
            f["rel_tol"].as_f64().unwrap(),
            f["min_count"].as_u64().unwrap() as usize,
        );
        assert_eq!(idx(&m), usizes(&f["masked"]));
        assert_eq!(idx(&m), (4..=27).collect::<Vec<_>>());
    }

    #[test]
    fn example_3_short_runs() {
        let f = fixture("short_runs.json");
        for case in f["cases"].as_array().unwrap() {
            let v = f64s(&case["values"]);
            let m = short_run_mask(&v, case["min_len"].as_u64().unwrap() as usize);
            assert_eq!(idx(&m), usizes(&case["masked"]));
        }
    }

    fn stat_row(id: i32, mean: f64) -> ChannelStats {
        ChannelStats {
            well: WellId(id),
            log: LogKind::Gr,
            n: 1000,
            mean,
            median: mean,
            std: 5.0,
            p_low: mean - 8.0,
            p_high: mean + 8.0,
        }
    }

    #[test]
    fn example_4_ring_deviation() {
        let f = fixture("ring_dev.json");
        let near = f64s(&f["near_means"]);
        let well_mean = f["well_mean"].as_f64().unwrap();
        let mut rows = vec![stat_row(0, well_mean)];
        let mut xy = BTreeMap::new();
        xy.insert(WellId(0), (0.0, 0.0));
        for (i, m) in near.iter().enumerate() {
            rows.push(stat_row(i as i32 + 1, *m));
            xy.insert(WellId(i as i32 + 1), (10.0 * (i as f64 + 1.0), 0.0));
        }
        let out = ring_test(LogKind::Gr, &rows, &xy, &QcConfig::default());
        let w0 = &out[0];
        assert_eq!(w0.well, WellId(0));
        assert_eq!(w0.baseline, Baseline::Local);
        assert!(rel(w0.max_dev, f["dev_mean"].as_f64().unwrap()) < 1e-9);
        assert!(rel(w0.max_dev, 4.04693850117649) < 1e-9);
        assert_eq!(w0.bits & SIGMA_MEAN, SIGMA_MEAN);
    }

    /// 101 evenly spaced samples with median `m` and cv about 0.35.
    fn level_channel(m: f64) -> Vec<f64> {
        (0..=100)
            .map(|i| m * (0.4 + 1.2 * f64::from(i) / 100.0))
            .collect()
    }

    #[test]
    fn example_5_unit_rescue() {
        let f = fixture("rescue.json");
        let factor = f["factor"].as_f64().unwrap();
        let mut chans: Vec<(WellId, Vec<f64>)> = Vec::new();
        let mut xy = BTreeMap::new();
        chans.push((WellId(1), level_channel(f["A"]["median"].as_f64().unwrap())));
        xy.insert(WellId(1), (0.0, 0.0));
        chans.push((WellId(2), level_channel(f["B"]["median"].as_f64().unwrap())));
        xy.insert(WellId(2), (0.0, -0.5));
        for (j, m) in f64s(&f["normal_medians"]).iter().enumerate() {
            let id = WellId(10 + j as i32);
            chans.push((id, level_channel(*m)));
            xy.insert(id, (1.0 + j as f64, 0.0));
        }
        for (j, m) in f64s(&f["far_normal_medians"]).iter().enumerate() {
            let id = WellId(20 + j as i32);
            chans.push((id, level_channel(*m)));
            xy.insert(id, (100.0 + j as f64, 0.0));
        }
        let view: Vec<(WellId, &[f64])> = chans.iter().map(|(i, v)| (*i, v.as_slice())).collect();
        let rows = plan_rescue(LogKind::Gr, Some(factor), &view, &xy, &QcConfig::default());
        let a = &rows[0];
        let b = &rows[1];
        assert_eq!(a.reason, RescueReason::Rescued);
        assert_eq!(b.reason, RescueReason::FactorOff);
        assert_eq!(a.n_ref, 6);
        assert!(rel(a.ref_median, f["A"]["ref_median"].as_f64().unwrap()) < 1e-9);
        assert!(
            rel(
                a.median * factor / a.ref_median,
                f["A"]["conv"].as_f64().unwrap()
            ) < 1e-9
        );
        assert!(
            rel(
                b.median * factor / b.ref_median,
                f["B"]["conv"].as_f64().unwrap()
            ) < 1e-9
        );
        assert!(rel(a.level_ratio, f["A"]["level_ratio"].as_f64().unwrap()) < 1e-9);
        assert!((a.cv - f["A"]["cv"].as_f64().unwrap()).abs() < 0.02);
        assert!(rows[2..]
            .iter()
            .all(|r| r.reason == RescueReason::NotCandidate));
    }

    #[test]
    fn example_6_knn_kl() {
        let f = fixture("kl_knn.json");
        let x = f64s(&f["x"]);
        let y = f64s(&f["y"]);
        let k = f["k"].as_u64().unwrap() as usize;
        let floor = f["floor"].as_f64().unwrap();
        let kxy = kl_knn_1d(&x, &y, k, floor);
        let kyx = kl_knn_1d(&y, &x, k, floor);
        assert!(rel(kxy, f["kl_xy"].as_f64().unwrap()) < 1e-9);
        assert!(rel(kyx, f["kl_yx"].as_f64().unwrap()) < 1e-9);
        assert!(rel(0.5 * (kxy + kyx), f["pair"].as_f64().unwrap()) < 1e-9);
        assert!(rel(kxy, 0.48842127705209) < 1e-9);
        assert!(rel(kyx, -0.31812394675845) < 1e-9);
        assert!(rel(0.5 * (kxy + kyx), 0.08514866514682) < 1e-9);
    }

    // --------------------------------------------------------- rule acceptance

    #[test]
    fn repeat_boundary_and_ramp() {
        let ramp = |i: usize| 10.0 + 5.0 * i as f64;
        let mut v: Vec<f64> = (0..60).map(ramp).collect();
        for x in v.iter_mut().skip(10).take(20) {
            *x = 100.0;
        }
        assert_eq!(
            idx(&repeat_mask(&v, 20, 1e-3, 5)),
            (10..30).collect::<Vec<_>>()
        );
        let mut w: Vec<f64> = (0..60).map(ramp).collect();
        for x in w.iter_mut().skip(10).take(19) {
            *x = 100.0;
        }
        assert!(idx(&repeat_mask(&w, 20, 1e-3, 5)).is_empty());
        // slow ramp: every step (0.04) is below tol (about 0.057), the 20-sample range is not
        let slow: Vec<f64> = (0..300).map(|i| 50.0 + 0.04 * i as f64).collect();
        assert!(idx(&repeat_mask(&slow, 20, 1e-3, 5)).is_empty());
    }

    #[test]
    fn plateau_vs_noise() {
        let params = PlateauParams {
            window: 20,
            window_min: 10,
            min_len: 20,
            min_count: 60,
            std_share: 0.02,
        };
        let mut rng = rng_for(1, "test.plateau", 0);
        let mut v: Vec<f64> = (0..300).map(|_| 10.0 * standard_normal(&mut rng)).collect();
        assert!(idx(&plateau_mask(&v, &params)).is_empty());
        for (j, x) in v.iter_mut().enumerate().skip(100).take(40) {
            *x = 0.001 * (j % 3) as f64;
        }
        let got = idx(&plateau_mask(&v, &params));
        assert!(!got.is_empty());
        assert!(got.iter().all(|&i| (100..140).contains(&i)));
    }

    // ---------------------------------------------------------- stage fixtures

    const N_SAMPLES: usize = 600;

    fn level(log: LogKind, z: f64) -> f64 {
        match log {
            LogKind::Gr => 60.0 * (0.3 * z).exp(),
            LogKind::Rhob => 2.5 + 0.1 * z,
            LogKind::Nphi => 0.10 + 0.03 * z,
            LogKind::Dtc => 80.0 + 10.0 * z,
            LogKind::Resd => 10f64.powf(1.0 + 0.3 * z),
            LogKind::Pef => 3.0 + 0.5 * z,
        }
    }

    /// A 4 x 3 fleet of white-noise wells, `N_SAMPLES` samples each.
    fn fleet(n_wells: usize) -> (Vec<WellHeader>, Vec<WellLogs>) {
        let mut headers = Vec::new();
        let mut logs = Vec::new();
        for i in 0..n_wells {
            let id = WellId(i as i32 + 1);
            headers.push(WellHeader {
                id,
                name: format!("W{:02}", i + 1),
                x_m: (i % 4) as f64 * 2000.0,
                y_m: (i / 4) as f64 * 2000.0,
                kb_elev_ft: 100.0,
            });
            let grid = GridRange::new(10 * i as i32, N_SAMPLES).unwrap();
            let mut w = WellLogs::new(id, grid);
            for log in LogKind::ALL {
                let mut rng = rng_for(7, log.name(), id.0);
                let c: Vec<f32> = (0..N_SAMPLES)
                    .map(|_| level(log, standard_normal(&mut rng)) as f32)
                    .collect();
                w.provenance.insert(
                    log,
                    c.iter()
                        .map(|v| if v.is_finite() { 0 } else { 255 })
                        .collect(),
                );
                w.curves.insert(log, c);
            }
            logs.push(w);
        }
        (headers, logs)
    }

    fn well_mut(logs: &mut [WellLogs], id: i32) -> &mut WellLogs {
        logs.iter_mut().find(|w| w.id == WellId(id)).unwrap()
    }

    fn run(headers: &[WellHeader], logs: &[WellLogs]) -> QcStageOutput {
        run_qc_stage(
            &QcInput {
                wells: headers,
                logs,
                aux: &[],
            },
            &QcConfig::default(),
        )
        .unwrap()
    }

    fn out_well(out: &QcStageOutput, id: i32) -> (&WellLogs, &QcFlags) {
        (
            out.logs.iter().find(|w| w.id == WellId(id)).unwrap(),
            out.flags.iter().find(|w| w.id == WellId(id)).unwrap(),
        )
    }

    fn report_row(out: &QcStageOutput, id: i32, log: LogKind) -> &sx_core::qc::QcChannelRow {
        out.report
            .iter()
            .find(|r| r.well_id == WellId(id) && r.log == log)
            .unwrap()
    }

    fn debug_of(out: &QcStageOutput) -> String {
        format!("{out:?}")
    }

    fn check_invariants(input: &[WellLogs], out: &QcStageOutput) {
        for (inp, (ol, of)) in input.iter().zip(out.logs.iter().zip(out.flags.iter())) {
            assert_eq!(inp.id, ol.id);
            for log in LogKind::ALL {
                let raw = inp.curves.get(&log);
                let qc = &ol.curves[&log];
                let bits = &of.bits[&log];
                assert_eq!(qc.len(), inp.grid.len);
                assert_eq!(bits.len(), inp.grid.len);
                for (i, (&q, &b)) in qc.iter().zip(bits.iter()).enumerate() {
                    assert!(b.count_ones() <= 1, "more than one bit at {i}: {b}");
                    let raw_nan = raw.is_none_or(|r| r[i].is_nan());
                    if q.is_nan() && b == 0 {
                        assert!(raw_nan, "NaN with no bit but raw finite at {i}");
                    }
                    if raw_nan {
                        assert!(q.is_nan() && b == 0, "raw NaN must stay NaN, no bit");
                    }
                    if b != 0 {
                        assert!(q.is_nan(), "blanked sample must be NaN");
                    }
                }
                let row = out
                    .report
                    .iter()
                    .find(|r| r.well_id == inp.id && r.log == log)
                    .unwrap();
                let sum = row.n_out_of_bounds
                    + row.n_tail
                    + row.n_repeat
                    + row.n_plateau
                    + row.n_short_run
                    + row.n_channel;
                assert_eq!(row.n_raw - row.n_qc, sum, "well {:?} {:?}", inp.id, log);
            }
        }
    }

    #[test]
    fn stage_planted_artifacts() {
        let (headers, mut logs) = fleet(12);
        // out-of-bounds RHOB value and a GR tail spike
        well_mut(&mut logs, 3)
            .curves
            .get_mut(&LogKind::Rhob)
            .unwrap()[50] = 9.0;
        well_mut(&mut logs, 3).curves.get_mut(&LogKind::Gr).unwrap()[100] = 480.0;
        // 30-sample exact repeat in DTC
        for v in &mut well_mut(&mut logs, 5)
            .curves
            .get_mut(&LogKind::Dtc)
            .unwrap()[200..230]
        {
            *v = 90.0;
        }
        // quiet plateau in PEF (alternating +-0.003: not a repeat, std far below 2% of channel std)
        for (j, v) in well_mut(&mut logs, 7)
            .curves
            .get_mut(&LogKind::Pef)
            .unwrap()
            .iter_mut()
            .enumerate()
            .skip(300)
            .take(50)
        {
            *v = if j % 2 == 0 { 3.003 } else { 2.997 };
        }
        let out = run(&headers, &logs);
        check_invariants(&logs, &out);

        let (ol3, of3) = out_well(&out, 3);
        assert_eq!(of3.bits[&LogKind::Rhob][50], QC_OUT_OF_BOUNDS);
        assert!(ol3.curves[&LogKind::Rhob][50].is_nan());
        assert_eq!(of3.bits[&LogKind::Gr][100], QC_TAIL);
        assert_eq!(ol3.provenance[&LogKind::Gr][100], 255);
        assert_eq!(ol3.provenance[&LogKind::Gr][101], 0);

        let (_, of5) = out_well(&out, 5);
        assert!(of5.bits[&LogKind::Dtc][200..230]
            .iter()
            .all(|&b| b == QC_REPEAT));

        let (_, of7) = out_well(&out, 7);
        let pl = &of7.bits[&LogKind::Pef];
        assert!(
            pl[310..340].iter().all(|&b| b == QC_PLATEAU),
            "{:?}",
            &pl[300..350]
        );
        assert!(pl[..290].iter().all(|&b| b != QC_PLATEAU));

        assert_eq!(report_row(&out, 3, LogKind::Rhob).n_out_of_bounds, 1);
        assert_eq!(report_row(&out, 5, LogKind::Dtc).n_repeat, 30);
    }

    #[test]
    fn stage_rescues_inverted_unit() {
        let (headers, mut logs) = fleet(12);
        let original: Vec<f32> = well_mut(&mut logs, 4).curves[&LogKind::Gr].clone();
        for v in well_mut(&mut logs, 4).curves.get_mut(&LogKind::Gr).unwrap() {
            *v = (f64::from(*v) / 14.0) as f32;
        }
        let stored = well_mut(&mut logs, 4).curves[&LogKind::Gr].clone();
        let out = run(&headers, &logs);
        check_invariants(&logs, &out);
        let row = report_row(&out, 4, LogKind::Gr);
        assert_eq!(row.rescue_reason, RescueReason::Rescued);
        assert_eq!(row.rescue_factor, 14.0);
        assert!(row.n_qc > 500);
        let (ol, _) = out_well(&out, 4);
        let q = &ol.curves[&LogKind::Gr];
        let mut ratios = Vec::new();
        for (i, &v) in q.iter().enumerate() {
            if v.is_finite() {
                let want = 14.0 * f64::from(stored[i]);
                assert!(rel(f64::from(v), want) < 1e-5);
                ratios.push((f64::from(v), f64::from(original[i])));
            }
        }
        // the median returns to the level of the unconverted fleet
        let mut vals: Vec<f64> = ratios.iter().map(|r| r.0).collect();
        vals.sort_by(f64::total_cmp);
        let med = vals[vals.len() / 2];
        assert!((med / 60.0 - 1.0).abs() < 0.2, "median {med}");
        // other wells are not rescued
        for r in out
            .report
            .iter()
            .filter(|r| r.log == LogKind::Gr && r.well_id != WellId(4))
        {
            assert_ne!(r.rescue_reason, RescueReason::Rescued);
        }
    }

    #[test]
    fn stage_blanks_unit_mismatch() {
        let (headers, mut logs) = fleet(12);
        // NPHI stored in percent: 150 samples that survive the bounds (a tight
        // low-porosity zone), the rest far above 1.0.
        let w = well_mut(&mut logs, 6);
        let c = w.curves.get_mut(&LogKind::Nphi).unwrap();
        let mut rng = rng_for(3, "pct", 0);
        for (i, v) in c.iter_mut().enumerate() {
            let z = standard_normal(&mut rng);
            *v = if i < 150 {
                (0.9 + 0.03 * z) as f32
            } else {
                (9.0 + 3.0 * z.abs()) as f32
            };
        }
        let out = run(&headers, &logs);
        check_invariants(&logs, &out);
        let row = report_row(&out, 6, LogKind::Nphi);
        assert!(row.n_out_of_bounds >= 440);
        assert_eq!(row.channel_action, ChannelAction::UnitMismatch);
        assert_eq!(row.n_qc, 0);
        assert!(row.n_channel > 0);
        assert!(row.sigma_bits & sx_core::qc::SIGMA_UNIT != 0);
        assert!(row.unit_ratio > 5.0);
        let (ol, _) = out_well(&out, 6);
        assert!(ol.curves[&LogKind::Nphi].iter().all(|v| v.is_nan()));
        // nobody else was blanked as unit mismatch
        assert!(out
            .report
            .iter()
            .filter(|r| r.well_id != WellId(6))
            .all(|r| r.channel_action != ChannelAction::UnitMismatch));
    }

    #[test]
    fn stage_constant_and_missing_logs() {
        let (headers, mut logs) = fleet(8);
        // constant RHOB in well 2, PEF absent in well 1
        well_mut(&mut logs, 2)
            .curves
            .get_mut(&LogKind::Rhob)
            .unwrap()
            .fill(2.65);
        well_mut(&mut logs, 1).curves.remove(&LogKind::Pef);
        well_mut(&mut logs, 1).provenance.remove(&LogKind::Pef);
        let out = run(&headers, &logs);
        let row = report_row(&out, 2, LogKind::Rhob);
        assert_eq!(row.channel_action, ChannelAction::Constant);
        assert_eq!(row.n_channel, N_SAMPLES);
        assert_eq!(row.n_qc, 0);
        assert_eq!(row.stats_n, 0);
        let (o1, f1) = out_well(&out, 1);
        assert!(o1.curves[&LogKind::Pef].iter().all(|v| v.is_nan()));
        assert!(f1.bits[&LogKind::Pef].iter().all(|&b| b == 0));
        assert_eq!(report_row(&out, 1, LogKind::Pef).n_raw, 0);
        // every well x log has a report row, all six logs everywhere
        assert_eq!(out.report.len(), 8 * 6);
        for f in &out.flags {
            assert_eq!(f.bits.len(), 6);
        }
    }

    #[test]
    fn stage_low_variance_blanked() {
        let (headers, mut logs) = fleet(12);
        // GR of well 9 nearly flat (std ~ 0.4 vs ~19), not an exact repeat or plateau
        let c = well_mut(&mut logs, 9).curves.get_mut(&LogKind::Gr).unwrap();
        let mut rng = rng_for(5, "flat", 0);
        for v in c.iter_mut() {
            *v = (60.0 + 0.4 * standard_normal(&mut rng)) as f32;
        }
        let out = run(&headers, &logs);
        check_invariants(&logs, &out);
        let row = report_row(&out, 9, LogKind::Gr);
        assert_eq!(row.channel_action, ChannelAction::LowVariance);
        assert_eq!(row.n_qc, 0);
        // low-variance stats row is dropped before the ring test
        assert_eq!(row.stats_n, 0);
        assert_eq!(row.sigma_baseline, Baseline::None);
    }

    /// Closed-caliper switch: "reject" keeps every closed-caliper RHOB sample
    /// out; "rescue" (default) shifts by the per-well bias and keeps samples
    /// inside the neighbor envelope, listed with raw values.
    #[test]
    fn stage_closed_caliper_reject_vs_rescue() {
        let (headers, mut logs) = fleet(12);
        let w3 = well_mut(&mut logs, 3);
        let n = w3.grid.len;
        let grid = w3.grid;
        let r = w3.curves.get_mut(&LogKind::Rhob).unwrap();
        r[..300].iter_mut().for_each(|v| *v -= 0.2);
        let mut cal = vec![8.5f32; n];
        cal[..300].iter_mut().for_each(|v| *v = 2.0);
        let aux = vec![sx_core::types::WellAux {
            id: WellId(3),
            grid,
            cal_in: cal,
            bs_in: vec![f32::NAN; n],
            lspd_ftmin: (0..n).map(|i| if i < 300 { 90.0 } else { 30.0 }).collect(),
            temp_c: vec![f32::NAN; n],
            drho: vec![0.02f32; n],
        }];
        let input = QcInput {
            wells: &headers,
            logs: &logs,
            aux: &aux,
        };
        let mut cfg = QcConfig::default();
        assert_eq!(cfg.closed_caliper.mode, "rescue");
        cfg.closed_caliper.min_overlap = 100;
        cfg.closed_caliper.neighbor_min_bins = 30;
        cfg.closed_caliper.mode = "reject".into();
        let rej = run_qc_stage(&input, &cfg).unwrap();
        check_invariants(&logs, &rej);
        let row = report_row(&rej, 3, LogKind::Rhob);
        assert_eq!((row.n_bad_hole, row.n_closed_cal_rescued), (300, 0));
        assert!(rej.closed_caliper_samples.is_empty());
        assert_eq!(rej.closed_caliper_runs.len(), 1);
        assert_eq!(rej.closed_caliper_runs[0].status, "reject_mode");
        assert!(rej.closed_caliper_runs[0]
            .note
            .starts_with("fast pass, arm closed"));

        cfg.closed_caliper.mode = "rescue".into();
        let res = run_qc_stage(&input, &cfg).unwrap();
        check_invariants(&logs, &res);
        let row = report_row(&res, 3, LogKind::Rhob);
        assert!(
            row.n_closed_cal_rescued > 30,
            "{}",
            row.n_closed_cal_rescued
        );
        assert_eq!(row.n_bad_hole + row.n_closed_cal_rescued, 300);
        assert!(
            (row.closed_cal_bias_gcc + 0.2).abs() < 0.05,
            "{}",
            row.closed_cal_bias_gcc
        );
        let run = &res.closed_caliper_runs[0];
        assert_eq!(
            (run.status, run.n_rescued),
            ("rescued", row.n_closed_cal_rescued)
        );
        assert_eq!(res.closed_caliper_samples.len(), run.n_rescued);
        let (l3, f3) = out_well(&res, 3);
        let raw = &logs.iter().find(|w| w.id == WellId(3)).unwrap().curves[&LogKind::Rhob];
        for s in &res.closed_caliper_samples {
            let i = (s.k - grid.k_start) as usize;
            assert_eq!(s.rhob_raw_gcc, f64::from(raw[i]));
            assert!((f64::from(l3.curves[&LogKind::Rhob][i]) - s.rhob_qc_gcc).abs() < 1e-6);
            assert_eq!(f3.bits[&LogKind::Rhob][i], 0);
            assert!(s.rhob_qc_gcc >= s.env_lo && s.rhob_qc_gcc <= s.env_hi);
        }
        // PEF is never rescued.
        assert_eq!(report_row(&res, 3, LogKind::Pef).n_closed_cal_rescued, 0);
    }

    #[test]
    fn stage_bad_hole_range_and_implausible_median() {
        let (headers, mut logs) = fleet(12);
        // Well 3: caliper closed on the first 100 samples, washout on 200..250,
        // a bad density correction on 300..310.
        let w3 = well_mut(&mut logs, 3);
        let n = w3.grid.len;
        let mut cal = vec![8.5f32; n];
        cal[..100].iter_mut().for_each(|v| *v = 1.0);
        cal[200..250].iter_mut().for_each(|v| *v = 12.0);
        let mut drho = vec![0.02f32; n];
        drho[300..310].iter_mut().for_each(|v| *v = 0.3);
        let aux = vec![sx_core::types::WellAux {
            id: WellId(3),
            grid: w3.grid,
            cal_in: cal,
            bs_in: vec![f32::NAN; n],
            lspd_ftmin: vec![f32::NAN; n],
            temp_c: vec![f32::NAN; n],
            drho,
        }];
        // Well 3 RHOB: two implausible samples.
        let r = w3.curves.get_mut(&LogKind::Rhob).unwrap();
        r[500] = 1.3;
        r[501] = 3.3;
        // Well 7 PEF median far too high (but within physical bounds).
        let p = well_mut(&mut logs, 7)
            .curves
            .get_mut(&LogKind::Pef)
            .unwrap();
        p.iter_mut().for_each(|v| *v += 3.3);
        let out = run_qc_stage(
            &QcInput {
                wells: &headers,
                logs: &logs,
                aux: &aux,
            },
            &QcConfig {
                closed_caliper: sx_core::config::QcClosedCalConfig {
                    mode: "reject".into(),
                    ..Default::default()
                },
                ..QcConfig::default()
            },
        )
        .unwrap();
        check_invariants(&logs, &out);
        let rho = report_row(&out, 3, LogKind::Rhob);
        assert_eq!(rho.n_bad_hole, 160);
        // Isolated extremes go to the tail rule first; either way they are gone.
        let (l3, _) = out_well(&out, 3);
        assert!(l3.curves[&LogKind::Rhob][500].is_nan() && l3.curves[&LogKind::Rhob][501].is_nan());
        let pef3 = report_row(&out, 3, LogKind::Pef);
        assert_eq!(pef3.n_bad_hole, 160);
        // GR is not a bad-hole log.
        assert_eq!(report_row(&out, 3, LogKind::Gr).n_bad_hole, 0);
        let pef7 = report_row(&out, 7, LogKind::Pef);
        assert_eq!(pef7.channel_action, ChannelAction::ImplausibleLevel);
        assert_eq!(pef7.n_qc, 0);
        // Switched off: no bad-hole or range masking.
        let mut cfg = QcConfig::default();
        cfg.badhole_logs.clear();
        cfg.plausible_range.clear();
        cfg.plausible_median.clear();
        let off = run_qc_stage(
            &QcInput {
                wells: &headers,
                logs: &logs,
                aux: &aux,
            },
            &cfg,
        )
        .unwrap();
        assert_eq!(report_row(&off, 3, LogKind::Rhob).n_bad_hole, 0);
        assert_ne!(
            report_row(&off, 7, LogKind::Pef).channel_action,
            ChannelAction::ImplausibleLevel
        );
    }

    #[test]
    fn stage_low_density_kept_in_gauge_flagged_without_caliper() {
        let (headers, mut logs) = fleet(12);
        // A coal-like bed (RHOB ~1.35-1.5) on 300..330 of wells 3 and 5; well 3
        // has an in-gauge caliper, well 5 has no caliper / DRHO at all.
        let coal = |r: &mut Vec<f32>| {
            for (k, v) in r[300..330].iter_mut().enumerate() {
                *v = 1.35 + 0.005 * ((k * 7) % 31) as f32;
            }
        };
        coal(
            well_mut(&mut logs, 3)
                .curves
                .get_mut(&LogKind::Rhob)
                .unwrap(),
        );
        coal(
            well_mut(&mut logs, 5)
                .curves
                .get_mut(&LogKind::Rhob)
                .unwrap(),
        );
        let g = well_mut(&mut logs, 3).grid;
        let aux = vec![sx_core::types::WellAux {
            id: WellId(3),
            grid: g,
            cal_in: vec![8.5; g.len],
            bs_in: vec![f32::NAN; g.len],
            lspd_ftmin: vec![f32::NAN; g.len],
            temp_c: vec![f32::NAN; g.len],
            drho: vec![0.02; g.len],
        }];
        let out = run_qc_stage(
            &QcInput {
                wells: &headers,
                logs: &logs,
                aux: &aux,
            },
            &QcConfig::default(),
        )
        .unwrap();
        check_invariants(&logs, &out);
        for id in [3, 5] {
            let (l, _) = out_well(&out, id);
            let kept = l.curves[&LogKind::Rhob][300..330]
                .iter()
                .filter(|v| v.is_finite())
                .count();
            assert!(kept >= 24, "well {id}: only {kept} coal samples kept");
        }
        assert_eq!(
            report_row(&out, 3, LogKind::Rhob).n_low_density_unverified,
            0
        );
        let u5 = report_row(&out, 5, LogKind::Rhob).n_low_density_unverified;
        assert!((24..=30).contains(&u5), "{u5}");
        // Per-sample list matches the counts, only well 5, all below the cut.
        let rows = &out.low_density_unverified;
        assert_eq!(rows.len(), u5);
        assert!(rows
            .iter()
            .all(|r| r.0 == WellId(5) && f64::from(r.2) < 1.65));
        assert!(rows.windows(2).all(|p| p[0].1 < p[1].1));
        assert_eq!(sx_core::qc::low_density_batch(rows).unwrap().num_rows(), u5);
        // Flat legacy cut (low_density off, range 1.65): the coal is dropped.
        let mut cfg = QcConfig {
            low_density_rhob: 0.0,
            ..QcConfig::default()
        };
        cfg.plausible_range.insert("RHOB".into(), [1.65, 3.1]);
        let flat = run_qc_stage(
            &QcInput {
                wells: &headers,
                logs: &logs,
                aux: &aux,
            },
            &cfg,
        )
        .unwrap();
        let (l, _) = out_well(&flat, 3);
        assert!(l.curves[&LogKind::Rhob][300..330]
            .iter()
            .all(|v| v.is_nan()));
    }

    #[test]
    fn stage_is_deterministic_and_order_free() {
        let (headers, mut logs) = fleet(10);
        well_mut(&mut logs, 2).curves.get_mut(&LogKind::Gr).unwrap()[10] = 499.0;
        let a = run(&headers, &logs);
        let b = run(&headers, &logs);
        assert_eq!(debug_of(&a), debug_of(&b));
        let mut rev = logs.clone();
        rev.reverse();
        let mut rev_h = headers.clone();
        rev_h.reverse();
        let c = run(&rev_h, &rev);
        assert_eq!(debug_of(&a), debug_of(&c));
        // sorted by WellId
        assert!(a.logs.windows(2).all(|p| p[0].id < p[1].id));
        assert!(a
            .report
            .windows(2)
            .all(|p| (p[0].well_id, p[0].log) < (p[1].well_id, p[1].log)));
    }

    #[test]
    fn stage_errors() {
        let (headers, logs) = fleet(4);
        let out = run_qc_stage(
            &QcInput {
                wells: &headers,
                logs: &[],
                aux: &[],
            },
            &QcConfig::default(),
        )
        .unwrap();
        assert!(out.logs.is_empty() && out.flags.is_empty() && out.report.is_empty());

        let bad = QcConfig {
            tail_pct: [99.0, 1.0],
            ..QcConfig::default()
        };
        assert!(matches!(
            run_qc_stage(
                &QcInput {
                    wells: &headers,
                    logs: &logs,
                    aux: &[],
                },
                &bad
            ),
            Err(SxError::Config(_))
        ));
        let mut bad_factors = QcConfig::default();
        bad_factors.rescue_factors.insert("GR".into(), 1.0);
        assert!(matches!(
            run_qc_stage(
                &QcInput {
                    wells: &headers,
                    logs: &logs,
                    aux: &[],
                },
                &bad_factors
            ),
            Err(SxError::Config(_))
        ));
        let mut unknown = QcConfig::default();
        unknown.rescue_factors.insert("FOO".into(), 2.0);
        assert!(matches!(
            run_qc_stage(
                &QcInput {
                    wells: &headers,
                    logs: &logs,
                    aux: &[],
                },
                &unknown
            ),
            Err(SxError::Config(_))
        ));
        let zero = QcConfig {
            run_min_len: 0,
            ..QcConfig::default()
        };
        assert!(matches!(
            run_qc_stage(
                &QcInput {
                    wells: &headers,
                    logs: &logs,
                    aux: &[],
                },
                &zero
            ),
            Err(SxError::Config(_))
        ));

        // shape mismatch
        let mut short = logs.clone();
        short[0].curves.get_mut(&LogKind::Gr).unwrap().pop();
        assert!(matches!(
            run_qc_stage(
                &QcInput {
                    wells: &headers,
                    logs: &short,
                    aux: &[],
                },
                &QcConfig::default()
            ),
            Err(SxError::Shape(_))
        ));
        // missing header
        assert!(matches!(
            run_qc_stage(
                &QcInput {
                    wells: &headers[1..],
                    logs: &logs,
                    aux: &[],
                },
                &QcConfig::default()
            ),
            Err(SxError::Shape(_))
        ));
        // duplicate well id
        let mut dup = logs.clone();
        dup.push(logs[0].clone());
        assert!(matches!(
            run_qc_stage(
                &QcInput {
                    wells: &headers,
                    logs: &dup,
                    aux: &[],
                },
                &QcConfig::default()
            ),
            Err(SxError::InvalidInput(_))
        ));
    }

    // ---------------------------------------------------------------- divergence

    #[test]
    fn divergence_single_outlier_any_order() {
        let mut chans: Vec<(WellId, Vec<f64>)> = Vec::new();
        let mut xy = BTreeMap::new();
        for i in 0..20usize {
            let mut rng = rng_for(11, "div", i as i32);
            let (mu, sd) = if i == 7 { (4.0, 0.4) } else { (0.0, 1.0) };
            chans.push((
                WellId(i as i32),
                (0..400)
                    .map(|_| mu + sd * standard_normal(&mut rng))
                    .collect(),
            ));
            xy.insert(
                WellId(i as i32),
                ((i % 5) as f64 * 1000.0, (i / 5) as f64 * 1000.0),
            );
        }
        let cfg = QcConfig::default();
        let rows = jsd_scores(LogKind::Gr, &chans, &xy, &cfg, 0);
        let out: Vec<i32> = rows
            .iter()
            .filter(|r| r.outlier)
            .map(|r| r.well.0)
            .collect();
        assert_eq!(out, vec![7]);
        let mut rev = chans.clone();
        rev.reverse();
        assert_eq!(rows, jsd_scores(LogKind::Gr, &rev, &xy, &cfg, 0));
        // subsampling is seed-stable
        let small = QcConfig {
            jsd_max_count: 120,
            ..QcConfig::default()
        };
        let s1 = jsd_scores(LogKind::Gr, &chans, &xy, &small, 3);
        let s2 = jsd_scores(LogKind::Gr, &chans, &xy, &small, 3);
        assert_eq!(s1, s2);
    }

    // ------------------------------------------------------------------- schemas

    const REPORT_COLUMNS: [(&str, &str); 34] = [
        ("well_id", "Int32"),
        ("well_name", "Utf8"),
        ("log", "Utf8"),
        ("n_raw", "Int64"),
        ("n_out_of_bounds", "Int64"),
        ("n_tail", "Int64"),
        ("n_repeat", "Int64"),
        ("n_plateau", "Int64"),
        ("n_short_run", "Int64"),
        ("n_channel", "Int64"),
        ("n_bad_hole", "Int64"),
        ("n_implausible", "Int64"),
        ("n_low_density_unverified", "Int64"),
        ("n_closed_cal_rescued", "Int64"),
        ("closed_cal_bias_gcc", "Float64"),
        ("n_qc", "Int64"),
        ("channel_action", "Utf8"),
        ("rescue_reason", "Utf8"),
        ("rescue_factor", "Float64"),
        ("rescue_level_ratio", "Float64"),
        ("rescue_cv", "Float64"),
        ("rescue_ref_median_lin", "Float64"),
        ("rescue_n_ref", "Int64"),
        ("stats_n", "Int64"),
        ("mean_native", "Float64"),
        ("median_native", "Float64"),
        ("std_native", "Float64"),
        ("p_low_native", "Float64"),
        ("p_high_native", "Float64"),
        ("sigma_bits", "UInt8"),
        ("sigma_max_dev", "Float64"),
        ("sigma_cluster_dev", "Float64"),
        ("sigma_baseline", "Utf8"),
        ("unit_ratio", "Float64"),
    ];

    #[test]
    fn schemas_and_row_order() {
        let (headers, mut logs) = fleet(6);
        well_mut(&mut logs, 2).curves.get_mut(&LogKind::Gr).unwrap()[5] = f32::NAN;
        let out = run(&headers, &logs);

        // logs_qc: same rows, order and columns as logs_raw
        let raw = logs_batch(&logs, "subsurfix.logs_raw/1").unwrap();
        let qc = logs_batch(&out.logs, LOGS_QC_TAG).unwrap();
        assert_eq!(schema_tag(&qc), Some("subsurfix.logs_qc/1"));
        assert_eq!(qc.num_rows(), raw.num_rows());
        let names = |b: &arrow::record_batch::RecordBatch| -> Vec<(String, String)> {
            b.schema()
                .fields()
                .iter()
                .map(|f| (f.name().clone(), f.data_type().to_string()))
                .collect()
        };
        assert_eq!(names(&qc), names(&raw));
        let want: Vec<(String, String)> = [
            ("well_id", "Int32"),
            ("k", "Int32"),
            ("md_ft", "Float64"),
            ("gr_api", "Float32"),
            ("rhob_gcc", "Float32"),
            ("nphi_frac", "Float32"),
            ("dtc_usft", "Float32"),
            ("resd_ohmm", "Float32"),
            ("pef_bfrac", "Float32"),
        ]
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect();
        assert_eq!(names(&qc), want);
        for c in 0..3 {
            assert_eq!(qc.column(c), raw.column(c));
        }

        // qc_flags
        let fl = flags_batch(&out.flags, QC_FLAGS_TAG).unwrap();
        assert_eq!(schema_tag(&fl), Some("subsurfix.qc_flags/1"));
        assert_eq!(fl.num_rows(), raw.num_rows());
        let want: Vec<(String, String)> = [
            ("well_id", "Int32"),
            ("k", "Int32"),
            ("md_ft", "Float64"),
            ("gr_qc_bits", "UInt8"),
            ("rhob_qc_bits", "UInt8"),
            ("nphi_qc_bits", "UInt8"),
            ("dtc_qc_bits", "UInt8"),
            ("resd_qc_bits", "UInt8"),
            ("pef_qc_bits", "UInt8"),
        ]
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect();
        assert_eq!(names(&fl), want);
        for c in 0..3 {
            assert_eq!(fl.column(c), raw.column(c));
        }

        // qc_report
        let rb = qc_report_batch(&out.report).unwrap();
        assert_eq!(rb.num_rows(), 6 * 6);
        let got: Vec<(String, String)> = names(&rb);
        let want: Vec<(String, String)> = REPORT_COLUMNS
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect();
        assert_eq!(got, want);
        let csv = batch_to_csv_string(&rb).unwrap();
        let mut lines = csv.lines();
        let header: Vec<&str> = lines.next().unwrap().split(',').collect();
        assert_eq!(header.len(), 34);
        assert_eq!(header[0], "well_id");
        assert_eq!(header[33], "unit_ratio");
        let first = lines.next().unwrap();
        assert!(first.starts_with("1,W01,GR,"), "{first}");
        assert!(first.contains(",none,"));

        // Parquet round trip keeps exactly the schema tag
        let dir = tempfile::tempdir().unwrap();
        for (name, batch, tag) in [
            ("logs_qc.parquet", &qc, "subsurfix.logs_qc/1"),
            ("qc_flags.parquet", &fl, "subsurfix.qc_flags/1"),
        ] {
            let p = dir.path().join(name);
            write_batch(&p, batch).unwrap();
            let back = read_batch_expect(&p, tag).unwrap();
            assert_eq!(back.schema().metadata().len(), 1);
            assert_eq!(schema_tag(&back), Some(tag));
            assert_eq!(back.num_rows(), batch.num_rows());
        }

        // flags batch errors
        let mut bad = out.flags.clone();
        bad[0].bits.get_mut(&LogKind::Gr).unwrap().pop();
        assert!(matches!(
            flags_batch(&bad, QC_FLAGS_TAG),
            Err(SxError::Shape(_))
        ));
        let mut dup = out.flags.clone();
        dup.push(out.flags[0].clone());
        assert!(matches!(
            flags_batch(&dup, QC_FLAGS_TAG),
            Err(SxError::InvalidInput(_))
        ));
    }

    #[test]
    fn channel_bit_is_channel() {
        let (headers, mut logs) = fleet(6);
        well_mut(&mut logs, 1)
            .curves
            .get_mut(&LogKind::Dtc)
            .unwrap()
            .fill(77.0);
        let out = run(&headers, &logs);
        let (_, f) = out_well(&out, 1);
        assert!(f.bits[&LogKind::Dtc].iter().all(|&b| b == QC_CHANNEL));
    }

    // ------------------------------------------------------------------ proptest

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(24))]

        /// Bit invariant: at most one bit per sample; NaN with bits 0 exactly when
        /// NaN in the input; `n_raw - n_qc` equals the sum of the `n_*` counts.
        #[test]
        fn bit_invariant(seed in 0u64..10_000, n_wells in 3usize..9, nan_pct in 0u32..40, len in 40usize..260) {
            let mut headers = Vec::new();
            let mut logs = Vec::new();
            for i in 0..n_wells {
                let id = WellId(i as i32 + 1);
                headers.push(WellHeader {
                    id,
                    name: format!("P{i}"),
                    x_m: (i % 3) as f64 * 700.0,
                    y_m: (i / 3) as f64 * 700.0,
                    kb_elev_ft: 0.0,
                });
                let mut w = WellLogs::new(id, GridRange::new(i as i32 * 3, len).unwrap());
                for log in LogKind::ALL {
                    let mut rng = rng_for(seed, log.name(), id.0);
                    let c: Vec<f32> = (0..len)
                        .map(|j| {
                            let u = sx_core::util::rng::uniform_f64(&mut rng);
                            let z = standard_normal(&mut rng);
                            if u * 100.0 < f64::from(nan_pct) {
                                f32::NAN
                            } else if u > 0.97 {
                                (level(log, z) * 40.0) as f32 // wild values
                            } else if j % 31 < 22 && u > 0.5 && u < 0.9 {
                                level(log, 0.0) as f32 // repeats
                            } else {
                                level(log, z) as f32
                            }
                        })
                        .collect();
                    w.provenance.insert(log, vec![0; len]);
                    w.curves.insert(log, c);
                }
                logs.push(w);
            }
            let out = run(&headers, &logs);
            check_invariants(&logs, &out);
        }
    }
}
