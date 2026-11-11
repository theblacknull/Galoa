//! SSA multi-horizon Gold experiment — **Rust only** (ssa_rs).
//!
//! Architecture: fit → signal group by cumsum contribution → L/K forecast H steps.
//! Score = sum of H forecasted log-returns; realized = sum rets[i+1..=i+H].
//! PnL: hold close[i+1] → close[i+1+H]. Metrics: dir_acc, pnl_pct, pnl_over_maxdd.
//!
//! ```text
//! # default grid: H∈{5,15,30} × 8 (win,L,E,mode), stride=3
//! cargo run --release --manifest-path experiments/ssa_h1/rust/Cargo.toml -- \
//!   --data data/xauusd_m1_15d.bin --out experiments/ssa_h1/ssa_15d_hmulti.json
//!
//! # locked OOS
//! cargo run --release --manifest-path experiments/ssa_h1/rust/Cargo.toml -- \
//!   --data data/xauusd_m1_prior15d.bin --out experiments/ssa_h1/ssa_prior15d_hmulti_locked.json \
//!   --forecast L --win 64 --l 16 --energy 90 --horizon 15
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
    horizon: usize,
}

#[derive(Serialize)]
struct Row {
    forecast: String,
    win: usize,
    l: usize,
    energy_pct: f64,
    horizon: usize,
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
        horizon: cfg.horizon,
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
    let h = cfg.horizon;

    // ssa_rs: 2 <= L <= floor((N+1)/2) with N = win.
    let max_l = (cfg.win + 1) / 2;
    if cfg.l < 2 || cfg.l > max_l {
        return empty_row(mode_s, cfg, t0, "invalid L for win");
    }
    if cfg.stride == 0 {
        return empty_row(mode_s, cfg, t0, "stride must be >= 1");
    }
    if h == 0 {
        return empty_row(mode_s, cfg, t0, "horizon must be >= 1");
    }

    let i0 = cfg.win - 1;
    // Need rets[i+H] and closes[i+1+H]; i1 = rets.len()-1-H
    let i1 = rets.len().saturating_sub(1 + h);
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
                ForecastMode::L => model.l_forecast(h, &idx, false),
                ForecastMode::K => model.k_forecast(h, &idx, false),
            };
            if let Ok(m) = fc {
                // Score = sum of H forecasted log-returns
                let mut sc = 0.0;
                let mut ok = true;
                for step in 0..h {
                    let v = m[(step, 0)];
                    if !v.is_finite() {
                        ok = false;
                        break;
                    }
                    sc += v;
                }
                if ok {
                    // realized = sum rets[i+1..=i+H] = ln(c[i+1+H]/c[i+1])
                    let mut realized = 0.0;
                    for t in 1..=h {
                        let r = rets[i + t];
                        if !r.is_finite() {
                            ok = false;
                            break;
                        }
                        realized += r;
                    }
                    if ok {
                        // trade close[i+1] → close[i+1+H]
                        scores.push(sc);
                        next_rs.push(realized);
                        c0s.push(closes[i + 1]);
                        c1s.push(closes[i + 1 + h]);
                        n_comp_sum += idx.len() as f64;
                        n_ok += 1;
                    }
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
        horizon: h,
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

struct Args {
    data: PathBuf,
    out: PathBuf,
    /// If set, run only this config (locked OOS); else default multi-H grid.
    locked: Option<Cfg>,
}

fn parse_args() -> Result<Args, String> {
    let mut data = PathBuf::from("data/xauusd_m1_15d.bin");
    let mut out = PathBuf::from("experiments/ssa_h1/ssa_15d_hmulti.json");
    let mut forecast: Option<ForecastMode> = None;
    let mut win: Option<usize> = None;
    let mut l: Option<usize> = None;
    let mut energy: Option<f64> = None;
    let mut horizon: Option<usize> = None;
    let mut stride: usize = 3;
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
            "--forecast" => {
                i += 1;
                let v = args.get(i).ok_or("--forecast needs L or K")?;
                forecast = Some(match v.as_str() {
                    "L" | "l" => ForecastMode::L,
                    "K" | "k" => ForecastMode::K,
                    other => return Err(format!("--forecast must be L or K, got {other}")),
                });
            }
            "--win" => {
                i += 1;
                let v = args.get(i).ok_or("--win needs an int")?;
                win = Some(v.parse().map_err(|_| "--win must be int")?);
            }
            "--l" | "--L" => {
                i += 1;
                let v = args.get(i).ok_or("--l needs an int")?;
                l = Some(v.parse().map_err(|_| "--l must be int")?);
            }
            "--energy" => {
                i += 1;
                let v = args.get(i).ok_or("--energy needs a float")?;
                energy = Some(v.parse().map_err(|_| "--energy must be float")?);
            }
            "--horizon" | "-H" => {
                i += 1;
                let v = args.get(i).ok_or("--horizon needs an int")?;
                horizon = Some(v.parse().map_err(|_| "--horizon must be int")?);
            }
            "--stride" => {
                i += 1;
                let v = args.get(i).ok_or("--stride needs an int")?;
                stride = v.parse().map_err(|_| "--stride must be int")?;
            }
            other => return Err(format!("unknown arg {other}")),
        }
        i += 1;
    }

    let locked = match (forecast, win, l, energy, horizon) {
        (None, None, None, None, None) => None,
        (Some(mode), Some(win), Some(l), Some(energy_pct), Some(horizon)) => Some(Cfg {
            win,
            l,
            energy_pct,
            mode,
            stride,
            horizon,
        }),
        _ => {
            return Err(
                "locked run needs all of --forecast --win --l --energy --horizon (or omit all for grid)"
                    .into(),
            )
        }
    };

    Ok(Args { data, out, locked })
}

/// Base (win, L, energy%, mode) configs — same 8 as H=1.
fn base_configs() -> [(usize, usize, f64, ForecastMode); 8] {
    [
        (64usize, 16usize, 80.0, ForecastMode::L),
        (64, 16, 90.0, ForecastMode::L),
        (96, 24, 85.0, ForecastMode::L),
        (96, 32, 85.0, ForecastMode::L),
        (96, 32, 85.0, ForecastMode::K),
        (128, 32, 85.0, ForecastMode::L),
        (128, 40, 90.0, ForecastMode::L),
        (128, 40, 85.0, ForecastMode::K),
    ]
}

/// Default grid: horizons {5,15,30} × 8 configs, stride=3 → 24 rows.
fn default_grid(stride: usize) -> Vec<Cfg> {
    const HORIZONS: [usize; 3] = [5, 15, 30];
    let mut out = Vec::with_capacity(HORIZONS.len() * 8);
    for &horizon in &HORIZONS {
        for &(win, l, energy_pct, mode) in &base_configs() {
            out.push(Cfg {
                win,
                l,
                energy_pct,
                mode,
                stride,
                horizon,
            });
        }
    }
    out
}

fn main() {
    let args = parse_args().unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2);
    });
    let data = args.data;
    let out = args.out;

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
        "SSA Rust multi-H | bars={} returns={} data={}",
        closes.len(),
        rets.len(),
        data.display()
    );

    let grid: Vec<Cfg> = match args.locked {
        Some(cfg) => {
            println!(
                "LOCKED fc={} win={} L={} E={:.0}% H={} stride={}",
                match cfg.mode {
                    ForecastMode::L => "L",
                    ForecastMode::K => "K",
                },
                cfg.win,
                cfg.l,
                cfg.energy_pct,
                cfg.horizon,
                cfg.stride
            );
            vec![cfg]
        }
        None => {
            let g = default_grid(3);
            println!("GRID {} configs (H∈{{5,15,30}} × 8), stride=3", g.len());
            g
        }
    };

    let rows: Vec<Row> = grid
        .par_iter()
        .map(|cfg| run_cfg(&closes, &rets, cfg))
        .collect();

    for r in &rows {
        println!(
            "fc={} win={} L={} E={:.0}% H={} dir={:.4} pnl={:+.2}% pnl/dd={:?} k_avg={:.1} n={} t={:.1}s err={:?}",
            r.forecast,
            r.win,
            r.l,
            r.energy_pct,
            r.horizon,
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
            "\nBEST fc={} win={} L={} E={:.0}% H={} | dir={:.2}% | PnL={:+.2}% | PnL/MaxDD={:.2}",
            b.forecast,
            b.win,
            b.l,
            b.energy_pct,
            b.horizon,
            100.0 * b.dir_acc,
            b.pnl_pct,
            b.pnl_over_maxdd.unwrap_or(f64::NAN)
        );
    }

    let locked = rows.len() == 1;
    let payload = serde_json::json!({
        "runtime": "rust+ssa_rs",
        "architecture": "fit -> cumsum contribution signal group -> L/K forecast H steps; score=sum of H forecasted log-returns; PnL hold close[i+1]->close[i+1+H]",
        "data": data.to_string_lossy(),
        "locked_config": locked,
        "horizons": if locked { serde_json::json!([rows[0].horizon]) } else { serde_json::json!([5, 15, 30]) },
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
