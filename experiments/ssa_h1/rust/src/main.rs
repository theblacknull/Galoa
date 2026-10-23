//! SSA H=1 Gold experiment — **Rust only** (ssa_rs).
//!
//! Architecture: fit → signal group by cumsum contribution → L/K forecast H=1.
//! Metrics: dir_acc, pnl_pct, pnl_over_maxdd. No baselines.
//!
//! ```text
//! cargo run --release --manifest-path experiments/ssa_h1/rust/Cargo.toml -- \
//!   --data data/xauusd_m1_prior5d_b.bin
//! ```

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;
use serde::Serialize;
use ssa_rs::Ssa;

#[derive(Clone, Copy)]
enum ForecastMode {
    L,
    K,
}

#[derive(Clone)]
struct Cfg {
    win: usize,
    l: usize,
    energy_pct: f64,
    mode: ForecastMode,
    stride: usize,
}

#[derive(Serialize)]
struct Row {
    forecast: String,
    win: usize,
    l: usize,
    energy_pct: f64,
    dir_acc: f64,
    pnl_pct: f64,
    pnl_over_maxdd: Option<f64>,
    max_dd_pct: f64,
    n: usize,
    mean_n_comp: f64,
    elapsed_sec: f64,
    error: Option<String>,
}

fn load_galoa_closes(path: &PathBuf) -> Result<Vec<f64>, String> {
    let buf = fs::read(path).map_err(|e| e.to_string())?;
    if buf.len() < 16 || &buf[..8] != b"GALOAP1\0" {
        return Err("bad GALOAP1 magic".into());
    }
    let n = u64::from_le_bytes(buf[8..16].try_into().unwrap()) as usize;
    let off = 16 + n * 8;
    if buf.len() < off + n * 8 {
        return Err("truncated GALOAP1".into());
    }
    let mut closes = Vec::with_capacity(n);
    for i in 0..n {
        let b: [u8; 8] = buf[off + i * 8..off + i * 8 + 8].try_into().unwrap();
        closes.push(f64::from_le_bytes(b));
    }
    Ok(closes)
}

fn log_returns(c: &[f64]) -> Vec<f64> {
    c.windows(2).map(|w| (w[1] / w[0]).ln()).collect()
}

fn signal_components(cumsum: &[f64], d: usize, energy_pct: f64) -> Vec<usize> {
    if cumsum.is_empty() || d == 0 {
        return vec![0];
    }
    let mut k = cumsum
        .iter()
        .position(|&v| v >= energy_pct)
        .map(|i| i + 1)
        .unwrap_or(d);
    k = k.max(1).min(d.saturating_sub(1).max(1));
    (0..k).collect()
}

fn eval_three(scores: &[f64], next_r: &[f64], c0: &[f64], c1: &[f64]) -> (f64, f64, f64, Option<f64>) {
    let mut hits = 0usize;
    let mut nz = 0usize;
    for (s, r) in scores.iter().zip(next_r.iter()) {
        if *r == 0.0 {
            continue;
        }
        nz += 1;
        let side = if *s < 0.0 { -1.0 } else { 1.0 };
        if side == r.signum() {
            hits += 1;
        }
    }
    let dir = if nz == 0 { 0.0 } else { hits as f64 / nz as f64 };

    let mut eq = 10_000.0_f64;
    let mut peak = eq;
    let mut max_dd = 0.0_f64;
    for ((s, a), b) in scores.iter().zip(c0.iter()).zip(c1.iter()) {
        if *a <= 0.0 {
            continue;
        }
        let side = if *s < 0.0 { -1.0 } else { 1.0 };
        eq *= 1.0 + side * (b / a - 1.0);
        peak = peak.max(eq);
        if peak > 0.0 {
            max_dd = max_dd.max((peak - eq) / peak);
        }
    }
    let pnl = (eq / 10_000.0 - 1.0) * 100.0;
    let mdd = max_dd * 100.0;
    let ratio = if mdd > 1e-12 { Some(pnl / mdd) } else { None };
    (dir, pnl, mdd, ratio)
}

fn run_cfg(closes: &[f64], rets: &[f64], cfg: &Cfg) -> Row {
    let t0 = Instant::now();
    let mode_s = match cfg.mode {
        ForecastMode::L => "L",
        ForecastMode::K => "K",
    };
    let i0 = cfg.win - 1;
    let i1 = rets.len().saturating_sub(2);
    if i1 <= i0 {
        return Row {
            forecast: mode_s.into(),
            win: cfg.win,
            l: cfg.l,
            energy_pct: cfg.energy_pct,
            dir_acc: 0.0,
            pnl_pct: 0.0,
            pnl_over_maxdd: None,
            max_dd_pct: 0.0,
            n: 0,
            mean_n_comp: 0.0,
            elapsed_sec: t0.elapsed().as_secs_f64(),
            error: Some("series too short".into()),
        };
    }

    let mut scores = Vec::new();
    let mut next_rs = Vec::new();
    let mut c0s = Vec::new();
    let mut c1s = Vec::new();
    let mut n_comp_sum = 0.0;
    let mut n_ok = 0usize;

    let ssa = Ssa::new();
    let mut i = i0;
    while i <= i1 {
        if (i - i0) % cfg.stride != 0 {
            i += 1;
            continue;
        }
        let start = i + 1 - cfg.win;
        let seg = &rets[start..=i];
        if !seg.iter().all(|v| v.is_finite()) || seg.iter().all(|v| *v == 0.0) {
            i += 1;
            continue;
        }
        match ssa.fit(seg, Some(cfg.l)) {
            Ok(model) => {
                let idx = signal_components(&model.cumsum_contr, model.d, cfg.energy_pct);
                let fc = match cfg.mode {
                    ForecastMode::L => model.l_forecast(1, &idx, false),
                    ForecastMode::K => model.k_forecast(1, &idx, false),
                };
                match fc {
                    Ok(m) => {
                        let sc = m[(0, 0)];
                        if sc.is_finite() {
                            scores.push(sc);
                            next_rs.push(rets[i + 1]);
                            c0s.push(closes[i + 1]);
                            c1s.push(closes[i + 2]);
                            n_comp_sum += idx.len() as f64;
                            n_ok += 1;
                        }
                    }
                    Err(_) => {}
                }
            }
            Err(_) => {}
        }
        i += 1;
    }

    if scores.is_empty() {
        return Row {
            forecast: mode_s.into(),
            win: cfg.win,
            l: cfg.l,
            energy_pct: cfg.energy_pct,
            dir_acc: 0.0,
            pnl_pct: 0.0,
            pnl_over_maxdd: None,
            max_dd_pct: 0.0,
            n: 0,
            mean_n_comp: 0.0,
            elapsed_sec: t0.elapsed().as_secs_f64(),
            error: Some("no_scores".into()),
        };
    }
    let (dir, pnl, mdd, ratio) = eval_three(&scores, &next_rs, &c0s, &c1s);
    Row {
        forecast: mode_s.into(),
        win: cfg.win,
        l: cfg.l,
        energy_pct: cfg.energy_pct,
        dir_acc: dir,
        pnl_pct: pnl,
        pnl_over_maxdd: ratio,
        max_dd_pct: mdd,
        n: scores.len(),
        mean_n_comp: n_comp_sum / n_ok as f64,
        elapsed_sec: t0.elapsed().as_secs_f64(),
        error: None,
    }
}

fn main() {
    let mut data = PathBuf::from("data/xauusd_m1_prior5d_b.bin");
    let mut out = PathBuf::from("experiments/ssa_h1/ssa_prior5d_b_rust.json");
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => {
                i += 1;
                data = PathBuf::from(&args[i]);
            }
            "--out" => {
                i += 1;
                out = PathBuf::from(&args[i]);
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let wall0 = Instant::now();
    let closes = load_galoa_closes(&data).unwrap_or_else(|e| {
        eprintln!("load failed: {e}");
        std::process::exit(1);
    });
    let rets = log_returns(&closes);
    println!(
        "SSA Rust H=1 | bars={} returns={} data={}",
        closes.len(),
        rets.len(),
        data.display()
    );

    let grid: Vec<Cfg> = [
        (64usize, 16usize, 80.0, ForecastMode::L),
        (64, 16, 90.0, ForecastMode::L),
        (96, 24, 85.0, ForecastMode::L),
        (96, 32, 85.0, ForecastMode::L),
        (96, 32, 85.0, ForecastMode::K),
        (128, 32, 85.0, ForecastMode::L),
        (128, 40, 90.0, ForecastMode::L),
        (128, 40, 85.0, ForecastMode::K),
    ]
    .into_iter()
    .map(|(win, l, energy_pct, mode)| Cfg {
        win,
        l,
        energy_pct,
        mode,
        stride: 3,
    })
    .collect();

    let rows: Vec<Row> = grid
        .par_iter()
        .map(|cfg| run_cfg(&closes, &rets, cfg))
        .collect();

    for r in &rows {
        println!(
            "fc={} win={} L={} E={:.0}% dir={:.4} pnl={:+.2}% pnl/dd={:?} k_avg={:.1} n={} t={:.1}s err={:?}",
            r.forecast,
            r.win,
            r.l,
            r.energy_pct,
            r.dir_acc,
            r.pnl_pct,
            r.pnl_over_maxdd,
            r.mean_n_comp,
            r.n,
            r.elapsed_sec,
            r.error
        );
    }

    let best = rows
        .iter()
        .filter(|r| r.error.is_none())
        .max_by(|a, b| {
            let ka = (
                a.pnl_over_maxdd.unwrap_or(f64::NEG_INFINITY),
                a.dir_acc,
                a.pnl_pct,
            );
            let kb = (
                b.pnl_over_maxdd.unwrap_or(f64::NEG_INFINITY),
                b.dir_acc,
                b.pnl_pct,
            );
            ka.partial_cmp(&kb).unwrap()
        });

    if let Some(b) = best {
        println!(
            "\nBEST fc={} win={} L={} E={:.0}% | dir={:.2}% | PnL={:+.2}% | PnL/MaxDD={:.2}",
            b.forecast,
            b.win,
            b.l,
            b.energy_pct,
            100.0 * b.dir_acc,
            b.pnl_pct,
            b.pnl_over_maxdd.unwrap_or(f64::NAN)
        );
    }

    let payload = serde_json::json!({
        "runtime": "rust+ssa_rs",
        "architecture": "fit -> cumsum contribution signal group -> L/K forecast H=1",
        "data": data.to_string_lossy(),
        "metrics_only": ["dir_acc", "pnl_pct", "pnl_over_maxdd"],
        "rows": rows,
        "best": best,
        "wall_sec": wall0.elapsed().as_secs_f64(),
    });
    if let Some(parent) = out.parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(&out, serde_json::to_string_pretty(&payload).unwrap()).unwrap();
    println!("wall={:.1}s wrote {}", wall0.elapsed().as_secs_f64(), out.display());
}
