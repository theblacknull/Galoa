//! Offline RAFT walk-forward evaluation (always FIRE) + SpecReTF side-by-side.

use anyhow::{bail, Context, Result};
use galoa_core::{SpecReTfParams, SPECRETF_EMA_ALPHA, SPECRETF_HOP, SPECRETF_N_FFT};
use galoa_index::{run_specretf_eval, EvalConfig as SpecEvalConfig};
use galoa_ingest::{load_series, DEFAULT_GOLD_SERIES};
use galoa_raft::{
    run_raft_eval, HeadKind, RaftEvalConfig, RaftEvalReport, RaftParams, SimilarityKind,
    DEFAULT_HORIZON, DEFAULT_LOOKBACK, DEFAULT_TAU, DEFAULT_TOP_M,
};
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct EvalRaftOpts {
    pub data: PathBuf,
    pub lookback: usize,
    pub horizon: usize,
    pub top_m: usize,
    pub tau: f64,
    pub periods: Vec<usize>,
    pub similarity: SimilarityKind,
    pub head: HeadKind,
    pub stride: usize,
    pub min_history: usize,
    pub with_specretf: bool,
    pub metrics_out: Option<PathBuf>,
}

impl Default for EvalRaftOpts {
    fn default() -> Self {
        Self {
            data: PathBuf::from(DEFAULT_GOLD_SERIES),
            lookback: DEFAULT_LOOKBACK,
            horizon: DEFAULT_HORIZON,
            top_m: DEFAULT_TOP_M,
            tau: DEFAULT_TAU,
            periods: vec![1, 2, 4],
            similarity: SimilarityKind::Pearson,
            head: HeadKind::Identity,
            stride: 1,
            min_history: 128,
            with_specretf: true,
            metrics_out: None,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CombinedReport {
    pub raft: RaftEvalReport,
    pub specretf: Option<galoa_index::EvalReport>,
}

pub fn run_eval(opts: &EvalRaftOpts) -> Result<CombinedReport> {
    let events = load_series(&opts.data).with_context(|| format!("load {}", opts.data.display()))?;
    if events.is_empty() {
        bail!("empty series");
    }
    let closes: Vec<f64> = events.iter().map(|e| e.close).collect();

    let cfg = RaftEvalConfig {
        params: RaftParams {
            lookback: opts.lookback,
            horizon: opts.horizon,
            top_m: opts.top_m,
            tau: opts.tau,
            periods: opts.periods.clone(),
            similarity: opts.similarity,
            library_stride: 1,
        },
        stride: opts.stride,
        min_history: opts.min_history,
        head: opts.head,
        ..Default::default()
    };
    let raft = run_raft_eval(&closes, &cfg).context("RAFT eval")?;

    let specretf = if opts.with_specretf {
        let scfg = SpecEvalConfig {
            window: opts.lookback,
            horizon: opts.horizon,
            k: opts.top_m,
            stride: opts.stride,
            min_history_windows: opts.min_history,
            params: SpecReTfParams {
                n_fft: SPECRETF_N_FFT,
                hop: SPECRETF_HOP,
                ema_alpha: SPECRETF_EMA_ALPHA,
                ..Default::default()
            },
        };
        Some(run_specretf_eval(&closes, &scfg).context("SpecReTF side-by-side")?)
    } else {
        None
    };

    let report = CombinedReport { raft, specretf };
    if let Some(ref out) = opts.metrics_out {
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(&report)?;
        std::fs::write(out, json).with_context(|| format!("write {}", out.display()))?;
    }
    Ok(report)
}

pub fn print_report(r: &CombinedReport, data: &Path) {
    let raft = &r.raft;
    println!("Galoa RAFT offline eval (always FIRE)");
    println!(
        "data={}  bars={}  L={} F={} m={}  τ={}  periods={:?}  sim={}  head={}  stride={}",
        data.display(),
        raft.n_bars,
        raft.lookback,
        raft.horizon,
        raft.top_m,
        raft.tau,
        raft.periods,
        raft.similarity,
        raft.head,
        raft.stride
    );
    println!();
    println!(
        "{:<18} {:>8} {:>12} {:>12} {:>12} {:>12} {:>10} {:>10}",
        "method", "n", "MSE_close", "MAE_close", "MSE_ret", "MAE_ret", "dir_H", "dir_step"
    );

    // Prefer a merged view: raft methods + specretf primary (skip specretf's own baselines if present).
    for m in &raft.methods {
        print_row(m.name.as_str(), m.n_queries, m.mse_close, m.mae_close, m.mse_return, m.mae_return, m.dir_hit_h, m.dir_hit_step);
    }
    if let Some(ref sp) = r.specretf {
        if let Some(m) = sp.methods.iter().find(|m| m.name == "specretf") {
            print_row(
                "specretf",
                m.n_queries,
                m.mse_close,
                m.mae_close,
                m.mse_return,
                m.mae_return,
                m.dir_hit_h,
                m.dir_hit_step,
            );
        }
    }
    println!();
    println!(
        "Note: ~5d M1 (~{} bars) is short; paper L∈{{96..720}} — we use L={} for query count. Identity head = retrieval stage; ridge = simplified linear.",
        raft.n_bars, raft.lookback
    );
}

fn print_row(
    name: &str,
    n: usize,
    mse_c: f64,
    mae_c: f64,
    mse_r: f64,
    mae_r: f64,
    dir_h: f64,
    dir_step: f64,
) {
    println!(
        "{:<18} {:>8} {:>12.6} {:>12.6} {:>12.8} {:>12.8} {:>10.4} {:>10.4}",
        name, n, mse_c, mae_c, mse_r, mae_r, dir_h, dir_step
    );
}

pub fn parse_similarity(s: &str) -> Result<SimilarityKind> {
    match s.to_ascii_lowercase().as_str() {
        "pearson" | "corr" | "r" => Ok(SimilarityKind::Pearson),
        "euclidean" | "l2" | "neg_l2" => Ok(SimilarityKind::Euclidean),
        other => bail!("unknown similarity '{other}' (use pearson|euclidean)"),
    }
}

pub fn parse_head(s: &str) -> Result<HeadKind> {
    match s.to_ascii_lowercase().as_str() {
        "identity" | "retrieval" | "none" => Ok(HeadKind::Identity),
        "ridge" | "linear" => Ok(HeadKind::Ridge),
        other => bail!("unknown head '{other}' (use identity|ridge)"),
    }
}

pub fn parse_periods(s: &str) -> Result<Vec<usize>> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let p: usize = part.trim().parse().with_context(|| format!("period '{part}'"))?;
        if p == 0 {
            bail!("period must be > 0");
        }
        out.push(p);
    }
    if out.is_empty() {
        bail!("need at least one period");
    }
    Ok(out)
}
