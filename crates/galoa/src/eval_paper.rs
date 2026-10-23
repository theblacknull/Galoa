//! CLI wrapper: SpecReTF retrieval-only on Autoformer/Informer datasets.

use anyhow::{bail, Context, Result};
use galoa_core::SpecReTfParams;
use galoa_index::{run_etth_multi_horizon, PaperEvalConfig, PaperEvalReport};
use galoa_ingest::{etth_borders, load_multivariate_csv, percent_borders};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct PaperEvalOpts {
    pub data: PathBuf,
    pub dataset: String,
    pub lookback: usize,
    pub horizons: Vec<usize>,
    pub k: usize,
    pub n_fft: usize,
    pub hop: usize,
    pub ema_alpha: f64,
    pub library_stride: usize,
    pub query_stride: usize,
    pub metrics_out: Option<PathBuf>,
}

impl Default for PaperEvalOpts {
    fn default() -> Self {
        Self {
            data: PathBuf::from("data/ETT/ETTh1.csv"),
            dataset: "ETTh1".into(),
            lookback: 720,
            horizons: vec![96],
            k: 10,
            n_fft: 32,
            hop: 16,
            ema_alpha: 0.1,
            library_stride: 4,
            query_stride: 24,
            metrics_out: None,
        }
    }
}

pub fn run_paper(opts: &PaperEvalOpts) -> Result<PaperEvalReport> {
    let series = load_multivariate_csv(&opts.data)
        .with_context(|| format!("load {}", opts.data.display()))?;
    if series.n_steps() == 0 {
        bail!("empty series");
    }

    let base = PaperEvalConfig {
        lookback: opts.lookback,
        horizon: opts.horizons.first().copied().unwrap_or(96),
        k: opts.k,
        params: SpecReTfParams {
            n_fft: opts.n_fft,
            hop: opts.hop,
            ema_alpha: opts.ema_alpha,
        },
        library_stride: opts.library_stride,
        query_stride: opts.query_stride,
    };

    let name = opts.dataset.to_lowercase();
    let report = if name.starts_with("etth") {
        run_etth_multi_horizon(&opts.dataset, &series.values, &opts.horizons, &base)?
    } else if name.starts_with("exchange") {
        // Percent split like Autoformer custom datasets.
        let borders = percent_borders(series.n_steps(), opts.lookback);
        let (train_lo, train_end) = borders[0];
        let _ = train_lo;
        let (test_lo, test_hi) = borders[2];
        let mut horizons = Vec::new();
        for &h in &opts.horizons {
            let mut cfg = base.clone();
            cfg.horizon = h;
            horizons.push(galoa_index::run_paper_eval(
                &opts.dataset,
                &series.values,
                train_end,
                test_lo,
                test_hi,
                &cfg,
            )?);
        }
        avg_report(opts, series.n_channels(), horizons)
    } else {
        // Fall back to ETTh borders (caller must ensure length).
        let _b = etth_borders(opts.lookback);
        run_etth_multi_horizon(&opts.dataset, &series.values, &opts.horizons, &base)?
    };

    if let Some(ref out) = opts.metrics_out {
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(out, serde_json::to_string_pretty(&report)?)
            .with_context(|| format!("write {}", out.display()))?;
    }
    Ok(report)
}

fn avg_report(
    opts: &PaperEvalOpts,
    n_channels: usize,
    horizons: Vec<galoa_index::PaperHorizonReport>,
) -> PaperEvalReport {
    fn avg_mse(horizons: &[galoa_index::PaperHorizonReport], name: &str) -> Option<f64> {
        if horizons.is_empty() {
            return None;
        }
        let mut s = 0.0;
        for r in horizons {
            let m = r.methods.iter().find(|m| m.name == name)?;
            s += m.mse;
        }
        Some(s / horizons.len() as f64)
    }
    let avg_mse_retrieval = avg_mse(&horizons, "retrieval_only_specretf");
    let avg_mse_naive = avg_mse(&horizons, "naive_last");
    let avg_mse_td_pearson = avg_mse(&horizons, "retrieval_only_pearson");
    PaperEvalReport {
        dataset: opts.dataset.clone(),
        lookback: opts.lookback,
        k: opts.k,
        n_fft: opts.n_fft,
        hop: opts.hop,
        ema_alpha: opts.ema_alpha,
        library_stride: opts.library_stride,
        query_stride: opts.query_stride,
        n_channels,
        signal: "standard_scaler_raw",
        protocol: "train_library_test_query_retrieval_only",
        horizons,
        avg_mse_retrieval,
        avg_mse_naive,
        avg_mse_td_pearson,
    }
}

pub fn print_report(r: &PaperEvalReport, data: &Path) {
    println!("Galoa SpecReTF paper-protocol eval (retrieval-only, no neural fusion)");
    println!(
        "data={}  dataset={}  L={}  K={}  STFT M={} B={}  α={}  lib_stride={}  q_stride={}",
        data.display(),
        r.dataset,
        r.lookback,
        r.k,
        r.n_fft,
        r.hop,
        r.ema_alpha,
        r.library_stride,
        r.query_stride
    );
    println!(
        "signal={}  protocol={}  channels={}",
        r.signal, r.protocol, r.n_channels
    );
    println!();
    for h in &r.horizons {
        println!(
            "--- H={}  train_windows={}  test_queries={} ---",
            h.horizon, h.n_train_windows, h.n_test_queries
        );
        println!(
            "{:<28} {:>8} {:>12} {:>12}",
            "method", "n", "MSE", "MAE"
        );
        for m in &h.methods {
            println!(
                "{:<28} {:>8} {:>12.6} {:>12.6}",
                m.name, m.n_queries, m.mse, m.mae
            );
        }
        println!();
    }
    if let (Some(a), Some(b), Some(c)) = (
        r.avg_mse_retrieval,
        r.avg_mse_naive,
        r.avg_mse_td_pearson,
    ) {
        println!(
            "avg MSE over horizons:  specretf_retrieval={a:.6}  naive={b:.6}  pearson_retrieval={c:.6}"
        );
    }
    println!();
    println!(
        "Paper Table 1 (full SpecReTF w/ fusion) is NOT directly comparable — expect a gap."
    );
}
