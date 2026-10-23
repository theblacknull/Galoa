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
    let need = off.checked_add(n.saturating_mul(8)).ok_or("GALOAP1 size overflow")?;
    if buf.len() < need {
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
    c.windows(2)
        .map(|w| {
            if w[0] > 0.0 && w[1] > 0.0 {
                (w[1] / w[0]).ln()
            } else {
                f64::NAN
            }
        })
        .collect()
}

/// Indices of leading eigentriples whose cumsum contribution reaches `energy_pct` (0–100).
fn signal_components(cumsum: &[f64], d: usize, energy_pct: f64) -> Vec<usize> {
    if d == 0 {
        return Vec::new();
    }
    let k = cumsum
        .iter()
        .position(|&v| v >= energy_pct)
        .map(|i| i + 1)
        .unwrap_or(d)
        .clamp(1, d);
    (0..k).collect()
}

fn side_of(score: f64) -> f64 {
    if score > 0.0 {
        1.0
    } else if score < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn eval_three(scores: &[f64], next_r: &[f64], c0: &[f64], c1: &[f64]) -> (f64, f64, f64, Option<f64>) {
    let mut hits = 0usize;
    let mut nz = 0usize;
    for (s, r) in scores.iter().zip(next_r.iter()) {
        let side = side_of(*s);
        if side == 0.0 || *r == 0.0 {
            continue;
        }
        nz += 1;
        if side == r.signum() {
            hits += 1;
        }
    }
    let dir = if nz == 0 { 0.0 } else { hits as f64 / nz as f64 };

    let mut eq = 10_000.0_f64;
    let mut peak = eq;
    let mut max_dd = 0.0_f64;
    for ((s, a), b) in scores.iter().zip(c0.iter()).zip(c1.iter()) {
        if *a <= 0.0 || *b <= 0.0 {
            continue;
        }
        let side = side_of(*s);
        if side == 0.0 {
            continue;
        }
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

fn empty_row(mode_s: &str, cfg: &Cfg, t0: Instant, error: &str) -> Row {
    Row {
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
        error: Some(error.into()),
    }
}

fn run_cfg(closes: &[f64], rets: &[f64], cfg: &Cfg) -> Row {
    let t0 = Instant::now();
    let mode_s = match cfg.mode {
        ForecastMode::L => "L",
        ForecastMode::K => "K",
    };

    // ssa_rs: 2 <= L <= floor((N+1)/2) with N = win.
    let max_l = (cfg.win + 1) / 2;
    if cfg.l < 2 || cfg.l > max_l {
        return empty_row(mode_s, cfg, t0, "invalid L for win");
    }
    if cfg.stride == 0 {
        return empty_row(mode_s, cfg, t0, "stride must be >= 1");
    }

    let i0 = cfg.win - 1;
    let i1 = rets.len().saturating_sub(2);
    if i1 <= i0 {
        return empty_row(mode_s, cfg, t0, "series too short");
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
        if let Ok(model) = ssa.fit(seg, Some(cfg.l)) {
            let idx = signal_components(&model.cumsum_contr, model.d, cfg.energy_pct);
            if idx.is_empty() {
                i += 1;
                continue;
            }
            let fc = match cfg.mode {
                ForecastMode::L => model.l_forecast(1, &idx, false),
                ForecastMode::K => model.k_forecast(1, &idx, false),
            };
            if let Ok(m) = fc {
                let sc = m[(0, 0)];
                if sc.is_finite() {
                    // rets[i+1] = ln(c[i+2]/c[i+1]); trade close[i+1] → close[i+2]
                    scores.push(sc);
                    next_rs.push(rets[i + 1]);
                    c0s.push(closes[i + 1]);
                    c1s.push(closes[i + 2]);
                    n_comp_sum += idx.len() as f64;
                    n_ok += 1;
                }
            }
        }
        i += 1;
    }

    if scores.is_empty() || n_ok == 0 {
        return empty_row(mode_s, cfg, t0, "no_scores");
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

fn parse_args() -> Result<(PathBuf, PathBuf), String> {
    let mut data = PathBuf::from("data/xauusd_m1_prior5d_b.bin");
    let mut out = PathBuf::from("experiments/ssa_h1/ssa_prior5d_b_rust.json");
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => {
                i += 1;
                let v = args.get(i).ok_or("--data needs a path")?;
                data = PathBuf::from(v);
            }
            "--out" => {
                i += 1;
                let v = args.get(i).ok_or("--out needs a path")?;
                out = PathBuf::from(v);
            }
            other => return Err(format!("unknown arg {other}")),
        }
        i += 1;
    }
    Ok((data, out))
}

fn main() {
    let (data, out) = parse_args().unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2);
    });

    let wall0 = Instant::now();
    let closes = load_galoa_closes(&data).unwrap_or_else(|e| {
        eprintln!("load failed: {e}");
        std::process::exit(1);
    });
    if closes.len() < 3 {
        eprintln!("need at least 3 closes");
        std::process::exit(1);
    }
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
            a.pnl_over_maxdd
                .unwrap_or(f64::NEG_INFINITY)
                .total_cmp(&b.pnl_over_maxdd.unwrap_or(f64::NEG_INFINITY))
                .then(a.dir_acc.total_cmp(&b.dir_acc))
                .then(a.pnl_pct.total_cmp(&b.pnl_pct))
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
    println!(
        "wall={:.1}s wrote {}",
        wall0.elapsed().as_secs_f64(),
        out.display()
    );
}
