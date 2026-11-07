//! Offline SpecReTF walk-forward evaluation (always FIRE).

use anyhow::{bail, Context, Result};
use galoa_core::{
    SpecReTfParams, SPECRETF_EMA_ALPHA, SPECRETF_HOP, SPECRETF_K, SPECRETF_N_FFT,
};
use galoa_index::{
    run_specretf_eval, run_specretf_fusion_eval, run_specretf_sweep, EvalConfig, EvalReport,
    FusionTrainConfig, SignalMode, SweepReport,
};
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
    pub signal: SignalMode,
    pub stride: usize,
    pub min_history: usize,
    /// If true: naive | retrieval_only | direct_only | fusion_full (eq. 11 spirit).
    pub fusion: bool,
    pub metrics_out: Option<PathBuf>,
}

impl Default for EvalOpts {
    fn default() -> Self {
        Self {
            data: PathBuf::from(DEFAULT_GOLD_SERIES),
            window: galoa_index::DEFAULT_WINDOW,
            horizon: galoa_index::DEFAULT_HORIZON,
            k: SPECRETF_K,
            n_fft: SPECRETF_N_FFT,
            hop: SPECRETF_HOP,
            ema_alpha: SPECRETF_EMA_ALPHA,
            signal: SignalMode::LogReturns,
            stride: 1,
            min_history: 128,
            fusion: false,
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
        signal: opts.signal,
        params: SpecReTfParams {
            n_fft: opts.n_fft,
            hop: opts.hop,
            ema_alpha: opts.ema_alpha,
            ..Default::default()
        },
    };
    let report = if opts.fusion {
        run_specretf_fusion_eval(&closes, &cfg, &FusionTrainConfig::default())?
    } else {
        run_specretf_eval(&closes, &cfg)?
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

#[derive(Clone, Debug)]
pub struct SweepOpts {
    pub data: PathBuf,
    pub windows: Vec<usize>,
    pub horizon: usize,
    pub ks: Vec<usize>,
    pub n_ffts: Vec<usize>,
    pub alphas: Vec<f64>,
    pub signals: Vec<SignalMode>,
    pub stride: usize,
    pub min_history: usize,
    pub metrics_out: Option<PathBuf>,
}

impl Default for SweepOpts {
    fn default() -> Self {
        Self {
            data: PathBuf::from(DEFAULT_GOLD_SERIES),
            windows: vec![64],
            horizon: galoa_index::DEFAULT_HORIZON,
            ks: vec![5, 10, 20],
            n_ffts: vec![8, 16, 32],
            alphas: vec![0.01, 0.05, 0.1, 0.2],
            signals: vec![SignalMode::LogReturns, SignalMode::RawClose],
            stride: 16,
            min_history: 128,
            metrics_out: None,
        }
    }
}

pub fn run_sweep(opts: &SweepOpts) -> Result<SweepReport> {
    let events = load_series(&opts.data).with_context(|| format!("load {}", opts.data.display()))?;
    if events.is_empty() {
        bail!("empty series");
    }
    let closes: Vec<f64> = events.iter().map(|e| e.close).collect();
    let report = run_specretf_sweep(
        &closes,
        &opts.windows,
        opts.horizon,
        &opts.ks,
        &opts.n_ffts,
        &opts.alphas,
        &opts.signals,
        opts.stride,
        opts.min_history,
    )?;
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
    let mode = if r.methods.iter().any(|m| m.name == "fusion_full") {
        "fusion (eq.11 spirit: retrieval + direct ridge + Linear concat)"
    } else {
        "retrieval-only (eq.10)"
    };
    println!("Galoa SpecReTF offline eval (always FIRE) — {mode}");
    println!(
        "data={}  bars={}  W={} H={} K={}  stride={}  STFT n_fft={} hop={}  ema_α={}  signal={}",
        data.display(),
        r.n_bars,
        r.window,
        r.horizon,
        r.k,
        r.stride,
        r.n_fft,
        r.hop,
        r.ema_alpha,
        r.signal.as_str()
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
    let t = &r.timings;
    println!(
        "timing: feature_build={:.3}s  retrieval={:.3}s  total={:.3}s",
        t.feature_build_secs, t.retrieval_secs, t.total_secs
    );
    if let Some(naive) = t.naive_full_stft_estimate_secs {
        let speedup = if t.total_secs > 0.0 {
            naive / t.total_secs
        } else {
            0.0
        };
        println!(
            "timing: naive_full_stft_estimate={:.1}s  (~{:.1}× vs cached total; sampled)",
            naive, speedup
        );
    }
    println!();
    println!(
        "Note: ~5d M1 (~{} bars) is short for strong claims; numbers are diagnostic only.",
        r.n_bars
    );
}

pub fn print_sweep(r: &SweepReport, data: &Path) {
    println!(
        "Galoa SpecReTF hyperparam sweep (always FIRE)  data={}  bars={}  rows={}",
        data.display(),
        r.n_bars,
        r.rows.len()
    );
    println!(
        "{:<11} {:>4} {:>3} {:>3} {:>5} {:>6} {:>5} {:>10} {:>10} {:>8} {:>10} {:>10}",
        "signal", "W", "M", "K", "α", "frames", "n", "MSE_spec", "MSE_naive", "ΔMSE%", "dir_H", "MAE_spec"
    );
    let mut ranked: Vec<_> = r.rows.iter().collect();
    ranked.sort_by(|a, b| {
        a.specretf
            .mse_close
            .partial_cmp(&b.specretf.mse_close)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for row in ranked.iter().take(20) {
        let delta = if row.naive_last.mse_close > 0.0 {
            100.0 * (row.specretf.mse_close - row.naive_last.mse_close) / row.naive_last.mse_close
        } else {
            0.0
        };
        println!(
            "{:<11} {:>4} {:>3} {:>3} {:>5.2} {:>6} {:>5} {:>10.4} {:>10.4} {:>7.1}% {:>10.4} {:>10.4}",
            row.signal.as_str(),
            row.window,
            row.n_fft,
            row.k,
            row.ema_alpha,
            row.n_frames,
            row.specretf.n_queries,
            row.specretf.mse_close,
            row.naive_last.mse_close,
            delta,
            row.specretf.dir_hit_h,
            row.specretf.mae_close
        );
    }
    if ranked.len() > 20 {
        println!("… {} more rows (sorted by SpecReTF MSE_close; top 20 shown)", ranked.len() - 20);
    }
}
