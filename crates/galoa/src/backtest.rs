//! Trading-style USD equity backtest on Gold M1 (ours — not paper metrics).
//!
//! Papers (RAFT / SpecReTF) forecast **original series values** then StandardScaler
//! for train/eval and report MSE/MAE. This module converts directional forecasts into
//! a simple CFD-style PnL for illustration only on a tiny ~5d slice.

use anyhow::{bail, Context, Result};
use galoa_core::{SpecReTfParams, SPECRETF_EMA_ALPHA, SPECRETF_HOP, SPECRETF_K, SPECRETF_N_FFT};
use galoa_index::{
    run_specretf_fusion_signals, EvalConfig, FusionTrainConfig, SignalMode,
};
use galoa_ingest::{load_series, DEFAULT_GOLD_SERIES};
use galoa_raft::{
    run_raft_signals, HeadKind, RaftEvalConfig, RaftParams, SimilarityKind, DEFAULT_HORIZON,
    DEFAULT_LOOKBACK, DEFAULT_TAU, DEFAULT_TOP_M,
};
use serde::Serialize;
use std::path::{Path, PathBuf};

const EPS: f64 = 1e-12;

#[derive(Clone, Debug)]
struct Signal {
    end: usize,
    close: f64,
    pred_h: f64,
}

impl Signal {
    fn from_spec(s: &galoa_index::ForecastSignal) -> Self {
        Self {
            end: s.end,
            close: s.close,
            pred_h: s.pred_h,
        }
    }
    fn from_raft(s: &galoa_raft::ForecastSignal) -> Self {
        Self {
            end: s.end,
            close: s.close,
            pred_h: s.pred_h,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BacktestOpts {
    pub data: PathBuf,
    pub equity0: f64,
    pub window: usize,
    pub horizon: usize,
    pub k: usize,
    pub n_fft: usize,
    pub hop: usize,
    pub ema_alpha: f64,
    /// Round-trip cost as fraction of notional (e.g. 0.0002 = 2 bps).
    pub cost_rt_bps: f64,
    pub lookback_raft: usize,
    pub top_m_raft: usize,
    pub tau: f64,
    pub min_history: usize,
    /// Optional JSON array of `{end, close, pred_h}` from paper-faithful RAFT MLP.
    pub raft_mlp_signals: Option<PathBuf>,
    pub metrics_out: Option<PathBuf>,
}

impl Default for BacktestOpts {
    fn default() -> Self {
        Self {
            data: PathBuf::from(DEFAULT_GOLD_SERIES),
            equity0: 10_000.0,
            window: galoa_index::DEFAULT_WINDOW,
            horizon: DEFAULT_HORIZON,
            k: SPECRETF_K,
            n_fft: SPECRETF_N_FFT,
            hop: SPECRETF_HOP,
            ema_alpha: SPECRETF_EMA_ALPHA,
            cost_rt_bps: 2.0,
            lookback_raft: DEFAULT_LOOKBACK,
            top_m_raft: DEFAULT_TOP_M,
            tau: DEFAULT_TAU,
            min_history: 128,
            raft_mlp_signals: None,
            metrics_out: None,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct StrategyResult {
    pub name: String,
    pub with_cost: bool,
    pub final_equity: f64,
    pub total_return_pct: f64,
    pub max_drawdown_pct: f64,
    pub n_trades: usize,
    pub win_rate: f64,
    pub n_decisions: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct BacktestReport {
    pub note: String,
    pub paper_vs_ours: String,
    pub data: String,
    pub n_bars: usize,
    pub equity0: f64,
    pub horizon: usize,
    pub decision_rule: String,
    pub position_rule: String,
    pub cost_rule: String,
    pub cost_rt_bps: f64,
    pub params: serde_json::Value,
    pub strategies: Vec<StrategyResult>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    Flat = 0,
    Long = 1,
    Short = -1,
}

fn side_from_pred(close: f64, pred_h: f64) -> Side {
    let d = pred_h - close;
    if d > EPS {
        Side::Long
    } else if d < -EPS {
        Side::Short
    } else {
        Side::Flat
    }
}

/// Non-overlapping H-bar holds: pick first signal, skip until exit bar, etc.
fn thin_nonoverlapping(sigs: &[Signal], horizon: usize) -> Vec<Signal> {
    let mut out = Vec::new();
    let mut next_ok = 0usize;
    for s in sigs {
        if s.end < next_ok {
            continue;
        }
        out.push(s.clone());
        next_ok = s.end + horizon;
    }
    out
}

fn simulate(
    name: &str,
    closes: &[f64],
    equity0: f64,
    horizon: usize,
    signals: &[Signal],
    cost_rt: f64,
    with_cost: bool,
) -> StrategyResult {
    let thin = thin_nonoverlapping(signals, horizon);
    let mut equity = equity0;
    let mut peak = equity0;
    let mut max_dd = 0.0;
    let mut n_trades = 0usize;
    let mut wins = 0usize;

    for s in &thin {
        let exit_i = s.end + horizon - 1;
        if exit_i >= closes.len() {
            break;
        }
        let entry = s.close;
        let exit = closes[exit_i];
        if entry <= 0.0 {
            continue;
        }
        let side = side_from_pred(entry, s.pred_h);
        if side == Side::Flat {
            continue;
        }
        let ret = (exit / entry) - 1.0;
        let signed = (side as i8 as f64) * ret;
        let cost = if with_cost { cost_rt } else { 0.0 };
        let pnl_ret = signed - cost;
        equity *= 1.0 + pnl_ret;
        n_trades += 1;
        if pnl_ret > 0.0 {
            wins += 1;
        }
        if equity > peak {
            peak = equity;
        }
        if peak > 0.0 {
            let dd = (peak - equity) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }
    }

    StrategyResult {
        name: name.to_string(),
        with_cost,
        final_equity: equity,
        total_return_pct: 100.0 * (equity / equity0 - 1.0),
        max_drawdown_pct: 100.0 * max_dd,
        n_trades,
        win_rate: if n_trades == 0 {
            0.0
        } else {
            wins as f64 / n_trades as f64
        },
        n_decisions: thin.len(),
    }
}

fn buy_and_hold(
    name: &str,
    closes: &[f64],
    equity0: f64,
    start_end: usize,
    _horizon: usize,
    cost_rt: f64,
    with_cost: bool,
) -> StrategyResult {
    // Enter at first decision close; exit at last bar. One long trade.
    if closes.is_empty() || start_end == 0 || start_end > closes.len() {
        return StrategyResult {
            name: name.into(),
            with_cost,
            final_equity: equity0,
            total_return_pct: 0.0,
            max_drawdown_pct: 0.0,
            n_trades: 0,
            win_rate: 0.0,
            n_decisions: 0,
        };
    }
    let entry_i = start_end - 1;
    let exit_i = closes.len() - 1;
    let entry = closes[entry_i];
    let exit = closes[exit_i];
    if entry <= 0.0 {
        return always_flat_result(name, equity0, 0, with_cost);
    }
    let cost = if with_cost { cost_rt } else { 0.0 };
    let mut peak = equity0;
    let mut max_dd = 0.0;
    for i in entry_i..=exit_i {
        let eq_i = equity0 * (closes[i] / entry);
        if eq_i > peak {
            peak = eq_i;
        }
        if peak > 0.0 {
            let dd = (peak - eq_i) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }
    }
    let equity = equity0 * (exit / entry) * (1.0 - cost);
    StrategyResult {
        name: name.into(),
        with_cost,
        final_equity: equity,
        total_return_pct: 100.0 * (equity / equity0 - 1.0),
        max_drawdown_pct: 100.0 * max_dd,
        n_trades: 1,
        win_rate: if equity > equity0 { 1.0 } else { 0.0 },
        n_decisions: 1,
    }
}

fn always_long_signals(sigs: &[Signal]) -> Vec<Signal> {
    sigs.iter()
        .map(|s| Signal {
            end: s.end,
            close: s.close,
            pred_h: s.close + 1.0, // force long
        })
        .collect()
}

fn always_flat_result(name: &str, equity0: f64, n_decisions: usize, with_cost: bool) -> StrategyResult {
    StrategyResult {
        name: name.into(),
        with_cost,
        final_equity: equity0,
        total_return_pct: 0.0,
        max_drawdown_pct: 0.0,
        n_trades: 0,
        win_rate: 0.0,
        n_decisions,
    }
}

fn load_external_signals(path: &Path) -> Result<Vec<Signal>> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&raw)?;
    let arr = if let Some(a) = v.as_array() {
        a.clone()
    } else if let Some(a) = v.get("signals").and_then(|x| x.as_array()) {
        a.clone()
    } else {
        bail!("expected JSON array or {{signals:[...]}} in {}", path.display());
    };
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let end = item
            .get("end")
            .and_then(|x| x.as_u64())
            .ok_or_else(|| anyhow::anyhow!("signal missing end"))? as usize;
        let close = item
            .get("close")
            .and_then(|x| x.as_f64())
            .ok_or_else(|| anyhow::anyhow!("signal missing close"))?;
        let pred_h = item
            .get("pred_h")
            .and_then(|x| x.as_f64())
            .ok_or_else(|| anyhow::anyhow!("signal missing pred_h"))?;
        out.push(Signal { end, close, pred_h });
    }
    Ok(out)
}

pub fn run_backtest(opts: &BacktestOpts) -> Result<BacktestReport> {
    let events = load_series(&opts.data).with_context(|| format!("load {}", opts.data.display()))?;
    if events.is_empty() {
        bail!("empty series");
    }
    let closes: Vec<f64> = events.iter().map(|e| e.close).collect();
    let h = opts.horizon;
    let cost_rt = opts.cost_rt_bps / 10_000.0;

    let spec_cfg = EvalConfig {
        window: opts.window,
        horizon: h,
        k: opts.k,
        stride: 1,
        min_history_windows: opts.min_history,
        signal: SignalMode::RawClose,
        params: SpecReTfParams {
            n_fft: opts.n_fft,
            hop: opts.hop,
            ema_alpha: opts.ema_alpha,
            ..Default::default()
        },
    };
    let (fusion_raw, _retrieval_raw) =
        run_specretf_fusion_signals(&closes, &spec_cfg, &FusionTrainConfig::default())
            .context("SpecReTF fusion signals")?;
    let fusion_sigs: Vec<Signal> = fusion_raw.iter().map(Signal::from_spec).collect();

    let raft_cfg = RaftEvalConfig {
        params: RaftParams {
            lookback: opts.lookback_raft,
            horizon: h,
            top_m: opts.top_m_raft,
            tau: opts.tau,
            periods: vec![1, 2, 4],
            similarity: SimilarityKind::Pearson,
            library_stride: 1,
        },
        stride: 1,
        min_history: opts.min_history,
        head: HeadKind::Ridge,
        ..Default::default()
    };
    let raft_sigs: Vec<Signal> = run_raft_signals(&closes, &raft_cfg)
        .context("RAFT ridge signals")?
        .iter()
        .map(Signal::from_raft)
        .collect();

    let first_end = fusion_sigs
        .first()
        .map(|s| s.end)
        .or_else(|| raft_sigs.first().map(|s| s.end))
        .unwrap_or(opts.window + opts.min_history);

    let mut strategies = Vec::new();
    for with_cost in [false, true] {
        strategies.push(buy_and_hold(
            "buy_and_hold",
            &closes,
            opts.equity0,
            first_end,
            h,
            cost_rt,
            with_cost,
        ));
        strategies.push(always_flat_result(
            "naive_always_flat",
            opts.equity0,
            thin_nonoverlapping(&fusion_sigs, h).len(),
            with_cost,
        ));
        let long_sigs = always_long_signals(&fusion_sigs);
        strategies.push(simulate(
            "naive_always_long",
            &closes,
            opts.equity0,
            h,
            &long_sigs,
            cost_rt,
            with_cost,
        ));
        strategies.push(simulate(
            "specretf_fusion",
            &closes,
            opts.equity0,
            h,
            &fusion_sigs,
            cost_rt,
            with_cost,
        ));
        strategies.push(simulate(
            "raft_ridge",
            &closes,
            opts.equity0,
            h,
            &raft_sigs,
            cost_rt,
            with_cost,
        ));
        if let Some(ref path) = opts.raft_mlp_signals {
            let mlp = load_external_signals(path)?;
            strategies.push(simulate(
                "raft_paper_mlp",
                &closes,
                opts.equity0,
                h,
                &mlp,
                cost_rt,
                with_cost,
            ));
        }
    }

    let report = BacktestReport {
        note: "Illustrative only: ~5 calendar days of M1 (~6k bars) is far too short for trading claims.".into(),
        paper_vs_ours: "RAFT/SpecReTF papers use original series values (+ StandardScaler for train/eval) and report MSE/MAE — not USD PnL. This backtest is Galoa's own trading-style wrapper.".into(),
        data: opts.data.display().to_string(),
        n_bars: closes.len(),
        equity0: opts.equity0,
        horizon: h,
        decision_rule: "Every non-overlapping H bars: long if pred_H > close, short if pred_H < close, flat if equal.".into(),
        position_rule: "CFD-like 1x equity: equity *= 1 + side * (exit/entry - 1) [minus round-trip cost when enabled].".into(),
        cost_rule: format!(
            "Round-trip {} bps of notional charged once per opened trade (with_cost=true rows).",
            opts.cost_rt_bps
        ),
        cost_rt_bps: opts.cost_rt_bps,
        params: serde_json::json!({
            "specretf": {
                "window": opts.window,
                "horizon": h,
                "k": opts.k,
                "n_fft": opts.n_fft,
                "hop": opts.hop,
                "ema_alpha": opts.ema_alpha,
                "signal": "raw_close",
                "min_history": opts.min_history,
            },
            "raft_ridge": {
                "lookback": opts.lookback_raft,
                "horizon": h,
                "top_m": opts.top_m_raft,
                "tau": opts.tau,
                "periods": [1, 2, 4],
                "head": "ridge",
            },
        }),
        strategies,
    };

    if let Some(ref out) = opts.metrics_out {
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(&report)?;
        std::fs::write(out, json).with_context(|| format!("write {}", out.display()))?;
    }
    Ok(report)
}

pub fn print_report(r: &BacktestReport) {
    println!("Galoa Gold M1 trading-style backtest (ours — not paper PnL)");
    println!("data={}  bars={}  equity0={:.2}  H={}", r.data, r.n_bars, r.equity0, r.horizon);
    println!("{}", r.paper_vs_ours);
    println!("{}", r.note);
    println!();
    println!("Decision: {}", r.decision_rule);
    println!("Position: {}", r.position_rule);
    println!("Cost:     {}", r.cost_rule);
    println!();
    println!(
        "{:<22} {:>6} {:>12} {:>10} {:>10} {:>8} {:>8}",
        "strategy", "cost?", "final_$", "ret_%", "maxDD_%", "trades", "win%"
    );
    for s in &r.strategies {
        println!(
            "{:<22} {:>6} {:>12.2} {:>10.3} {:>10.3} {:>8} {:>8.1}",
            s.name,
            if s.with_cost { "yes" } else { "no" },
            s.final_equity,
            s.total_return_pct,
            s.max_drawdown_pct,
            s.n_trades,
            100.0 * s.win_rate
        );
    }
}
