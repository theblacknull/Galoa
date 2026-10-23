//! Offline SpecReTF walk-forward evaluation (always FIRE).

use anyhow::{bail, Context, Result};
use galoa_core::{SpecReTfParams, SPECRETF_EMA_ALPHA, SPECRETF_HOP, SPECRETF_N_FFT};
use galoa_index::{run_specretf_eval, EvalConfig, EvalReport};
use galoa_ingest::{load_series, DEFAULT_GOLD_SERIES};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct EvalOpts {
    pub data: PathBuf,
    pub window: usize,
    pub horizon: usize,
    pub k: usize,
    pub n_fft: usize,
    pub hop: usize,
    pub ema_alpha: f64,
    pub stride: usize,
    pub min_history: usize,
    pub metrics_out: Option<PathBuf>,
}

impl Default for EvalOpts {
    fn default() -> Self {
        Self {
            data: PathBuf::from(DEFAULT_GOLD_SERIES),
            window: galoa_index::DEFAULT_WINDOW,
            horizon: galoa_index::DEFAULT_HORIZON,
            k: galoa_index::DEFAULT_K,
            n_fft: SPECRETF_N_FFT,
            hop: SPECRETF_HOP,
            ema_alpha: SPECRETF_EMA_ALPHA,
            stride: 1,
            min_history: 128,
            metrics_out: None,
        }
    }
}

pub fn run_eval(opts: &EvalOpts) -> Result<EvalReport> {
    let events = load_series(&opts.data).with_context(|| format!("load {}", opts.data.display()))?;
    if events.is_empty() {
        bail!("empty series");
    }
    let closes: Vec<f64> = events.iter().map(|e| e.close).collect();
    let cfg = EvalConfig {
        window: opts.window,
        horizon: opts.horizon,
        k: opts.k,
        stride: opts.stride,
        min_history_windows: opts.min_history,
        params: SpecReTfParams {
            n_fft: opts.n_fft,
            hop: opts.hop,
            ema_alpha: opts.ema_alpha,
            ..Default::default()
        },
    };
    let report = run_specretf_eval(&closes, &cfg)?;
    if let Some(ref out) = opts.metrics_out {
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(&report)?;
        std::fs::write(out, json).with_context(|| format!("write {}", out.display()))?;
    }
    Ok(report)
}

pub fn print_report(r: &EvalReport, data: &Path) {
    println!("Galoa SpecReTF offline eval (always FIRE)");
    println!(
        "data={}  bars={}  W={} H={} K={}  stride={}  STFT n_fft={} hop={}  ema_α={}",
        data.display(),
        r.n_bars,
        r.window,
        r.horizon,
        r.k,
        r.stride,
        r.n_fft,
        r.hop,
        r.ema_alpha
    );
    println!();
    println!(
        "{:<18} {:>8} {:>12} {:>12} {:>12} {:>12} {:>10} {:>10} {:>10}",
        "method", "n", "MSE_close", "MAE_close", "MSE_ret", "MAE_ret", "dir_H", "dir_step", "band"
    );
    for m in &r.methods {
        let band = m
            .band_coverage
            .map(|b| format!("{:.3}", b))
            .unwrap_or_else(|| "-".into());
        println!(
            "{:<18} {:>8} {:>12.6} {:>12.6} {:>12.8} {:>12.8} {:>10.4} {:>10.4} {:>10}",
            m.name,
            m.n_queries,
            m.mse_close,
            m.mae_close,
            m.mse_return,
            m.mae_return,
            m.dir_hit_h,
            m.dir_hit_step,
            band
        );
    }
    println!();
    println!(
        "Note: ~5d M1 (~{} bars) is short for strong claims; numbers are diagnostic only.",
        r.n_bars
    );
}
