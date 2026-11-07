//! SpecReTF **paper-protocol** eval on Autoformer/Informer datasets.
//!
//! Differences vs Gold `run_specretf_eval`:
//! - STFT on **StandardScaler** raw values (paper), not log-returns (FX adaptation)
//! - Multivariate: per-channel similarity, trivial mean (Appendix G)
//! - Library from **train** split only; queries on **test** (no expanding causal leak)
//! - Softmax futures in scaled space (**no** last-close level shift)
//! - Retrieval-only `Y_retrieval` (eq. 10) — **no** Linear fusion / Ŷ_direct (eq. 11)
//!
//! Expect MSE gap vs paper Table 1 SpecReTF / RAFT: those include trained fusion.

use galoa_core::{
    softmax_ensemble_forecast, spectral_descriptor, spectral_similarity, SpecReTfError,
    SpecReTfParams, SpectralDescriptor,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PaperEvalError {
    #[error("window / horizon / k / stride must be > 0")]
    InvalidParams,
    #[error("series too short for L={lookback} H={horizon} (n={n})")]
    TooShort {
        n: usize,
        lookback: usize,
        horizon: usize,
    },
    #[error("empty train library (check borders / strides)")]
    EmptyLibrary,
    #[error(transparent)]
    Spec(#[from] SpecReTfError),
}

#[derive(Clone, Debug)]
pub struct PaperEvalConfig {
    pub lookback: usize,
    pub horizon: usize,
    pub k: usize,
    pub params: SpecReTfParams,
    /// Subsample train library starts (1 = dense).
    pub library_stride: usize,
    /// Subsample test query starts (1 = every Autoformer sample).
    pub query_stride: usize,
}

impl Default for PaperEvalConfig {
    fn default() -> Self {
        Self {
            lookback: 720,
            horizon: 96,
            k: 10,
            params: SpecReTfParams {
                n_fft: 32,
                hop: 16,
                ema_alpha: 0.1,
            },
            library_stride: 1,
            query_stride: 1,
        }
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct PaperMethodMetrics {
    pub name: String,
    pub n_queries: usize,
    pub mse: f64,
    pub mae: f64,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct PaperHorizonReport {
    pub horizon: usize,
    pub n_train_windows: usize,
    pub n_test_queries: usize,
    pub methods: Vec<PaperMethodMetrics>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct PaperEvalReport {
    pub dataset: String,
    pub lookback: usize,
    pub k: usize,
    pub n_fft: usize,
    pub hop: usize,
    pub ema_alpha: f64,
    pub library_stride: usize,
    pub query_stride: usize,
    pub n_channels: usize,
    pub signal: &'static str,
    pub protocol: &'static str,
    pub horizons: Vec<PaperHorizonReport>,
    /// Mean MSE over reported horizons (SpecReTF retrieval-only).
    pub avg_mse_retrieval: Option<f64>,
    pub avg_mse_naive: Option<f64>,
    pub avg_mse_td_pearson: Option<f64>,
}

struct Scaler {
    mean: Vec<f64>,
    std: Vec<f64>,
}

impl Scaler {
    fn fit(data: &[Vec<f64>]) -> Self {
        let c = data.first().map(|r| r.len()).unwrap_or(0);
        let n = data.len().max(1) as f64;
        let mut mean = vec![0.0; c];
        let mut std = vec![0.0; c];
        for row in data {
            for (j, &v) in row.iter().enumerate() {
                mean[j] += v;
            }
        }
        for m in &mut mean {
            *m /= n;
        }
        for row in data {
            for (j, &v) in row.iter().enumerate() {
                let d = v - mean[j];
                std[j] += d * d;
            }
        }
        for s in &mut std {
            *s = (*s / n).sqrt().max(1e-8);
        }
        Self { mean, std }
    }

    fn transform_row(&self, row: &[f64]) -> Vec<f64> {
        row.iter()
            .enumerate()
            .map(|(j, &v)| (v - self.mean[j]) / self.std[j])
            .collect()
    }
}

struct LibEntry {
    /// Per-channel spectral descriptors for the lookback window.
    descs: Vec<SpectralDescriptor>,
    /// Future H×C in scaled space (row-major: t then channel flattened as H rows of C).
    future: Vec<Vec<f64>>,
    /// Flattened lookback for Pearson baseline (C concatenated or mean channel — we use
    /// channel-mean of Pearson; store per-channel windows).
    windows: Vec<Vec<f64>>,
}

/// Run paper-protocol retrieval-only eval.
///
/// `values`: raw multivariate rows `T × C`.
/// `train_end`: exclusive end of train segment used to fit scaler + library (e.g. 8640).
/// `test_lo` / `test_hi`: exclusive-hi slice of the series for test queries
///   (Autoformer: lo = val_end - L, hi = test_end).
pub fn run_paper_eval(
    dataset: &str,
    values: &[Vec<f64>],
    train_end: usize,
    test_lo: usize,
    test_hi: usize,
    cfg: &PaperEvalConfig,
) -> Result<PaperHorizonReport, PaperEvalError> {
    if cfg.lookback == 0
        || cfg.horizon == 0
        || cfg.k == 0
        || cfg.library_stride == 0
        || cfg.query_stride == 0
    {
        return Err(PaperEvalError::InvalidParams);
    }
    if values.len() < cfg.lookback + cfg.horizon || train_end < cfg.lookback + cfg.horizon {
        return Err(PaperEvalError::TooShort {
            n: values.len(),
            lookback: cfg.lookback,
            horizon: cfg.horizon,
        });
    }
    let n_c = values[0].len();
    let scaler = Scaler::fit(&values[..train_end]);
    let scaled: Vec<Vec<f64>> = values.iter().map(|r| scaler.transform_row(r)).collect();

    // Channel-major for STFT.
    let mut ch: Vec<Vec<f64>> = vec![vec![0.0; scaled.len()]; n_c];
    for (t, row) in scaled.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            ch[c][t] = v;
        }
    }

    let last_lib_start = train_end.saturating_sub(cfg.lookback + cfg.horizon);
    let mut library = Vec::new();
    let mut start = 0usize;
    while start <= last_lib_start {
        match build_entry(&ch, &scaled, start, cfg.lookback, cfg.horizon, &cfg.params) {
            Ok(e) => library.push(e),
            Err(PaperEvalError::Spec(_)) => {}
            Err(e) => return Err(e),
        }
        start += cfg.library_stride;
    }
    if library.is_empty() {
        return Err(PaperEvalError::EmptyLibrary);
    }

    // Test queries: Autoformer Dataset_ETT samples start indices in
    // [0, len(slice)-L-H] mapped to absolute = test_lo + i.
    let slice_len = test_hi.saturating_sub(test_lo);
    if slice_len < cfg.lookback + cfg.horizon {
        return Err(PaperEvalError::TooShort {
            n: slice_len,
            lookback: cfg.lookback,
            horizon: cfg.horizon,
        });
    }
    let n_possible = slice_len - cfg.lookback - cfg.horizon + 1;

    let mut sse_ret = 0.0;
    let mut sae_ret = 0.0;
    let mut sse_naive = 0.0;
    let mut sae_naive = 0.0;
    let mut sse_td = 0.0;
    let mut sae_td = 0.0;
    let mut n_q = 0usize;

    let mut q = 0usize;
    while q < n_possible {
        let abs = test_lo + q;
        let q_descs: Result<Vec<_>, _> = (0..n_c)
            .map(|c| spectral_descriptor(&ch[c][abs..abs + cfg.lookback], &cfg.params))
            .collect();
        let Ok(q_descs) = q_descs else {
            q += cfg.query_stride;
            continue;
        };
        let q_windows: Vec<Vec<f64>> = (0..n_c)
            .map(|c| ch[c][abs..abs + cfg.lookback].to_vec())
            .collect();
        let actual: Vec<Vec<f64>> = (0..cfg.horizon)
            .map(|h| scaled[abs + cfg.lookback + h].clone())
            .collect();
        let last_obs = scaled[abs + cfg.lookback - 1].clone();

        // --- SpecReTF retrieval ---
        let mut scored: Vec<(f64, usize)> = Vec::with_capacity(library.len());
        for (i, entry) in library.iter().enumerate() {
            let mut s_sum = 0.0;
            let mut ok = true;
            for c in 0..n_c {
                match spectral_similarity(&q_descs[c], &entry.descs[c], &cfg.params) {
                    Ok(s) => s_sum += s,
                    Err(_) => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                scored.push((s_sum / n_c as f64, i));
            }
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let take = cfg.k.min(scored.len());
        let pred_ret = if take == 0 {
            vec![last_obs.clone(); cfg.horizon]
        } else {
            let mut trajs = Vec::with_capacity(take);
            let mut scores = Vec::with_capacity(take);
            for (s, idx) in &scored[..take] {
                trajs.push(flatten_future(&library[*idx].future));
                scores.push(*s);
            }
            let ens = softmax_ensemble_forecast(&trajs, &scores);
            unflatten_future(&ens.points.iter().map(|p| p.mean).collect::<Vec<_>>(), n_c)
        };

        // --- Naive last-value ---
        let pred_naive = vec![last_obs.clone(); cfg.horizon];

        // --- Time-domain Pearson kNN (RAFT-like retrieval-only contrast) ---
        let mut td_scored: Vec<(f64, usize)> = Vec::with_capacity(library.len());
        for (i, entry) in library.iter().enumerate() {
            let mut r_sum = 0.0;
            for c in 0..n_c {
                r_sum += pearson(&q_windows[c], &entry.windows[c]);
            }
            td_scored.push((r_sum / n_c as f64, i));
        }
        td_scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let td_take = cfg.k.min(td_scored.len());
        let pred_td = if td_take == 0 {
            vec![last_obs.clone(); cfg.horizon]
        } else {
            // Softmax on Pearson scores (same aggregation form as SpecReTF eq.10).
            let mut trajs = Vec::with_capacity(td_take);
            let mut scores = Vec::with_capacity(td_take);
            for (s, idx) in &td_scored[..td_take] {
                trajs.push(flatten_future(&library[*idx].future));
                scores.push(*s);
            }
            let ens = softmax_ensemble_forecast(&trajs, &scores);
            unflatten_future(&ens.points.iter().map(|p| p.mean).collect::<Vec<_>>(), n_c)
        };

        accumulate_mse_mae(&pred_ret, &actual, &mut sse_ret, &mut sae_ret);
        accumulate_mse_mae(&pred_naive, &actual, &mut sse_naive, &mut sae_naive);
        accumulate_mse_mae(&pred_td, &actual, &mut sse_td, &mut sae_td);
        n_q += 1;
        q += cfg.query_stride;
    }

    let denom = (n_q.max(1) * cfg.horizon * n_c) as f64;
    let _ = dataset;
    Ok(PaperHorizonReport {
        horizon: cfg.horizon,
        n_train_windows: library.len(),
        n_test_queries: n_q,
        methods: vec![
            PaperMethodMetrics {
                name: "retrieval_only_specretf".into(),
                n_queries: n_q,
                mse: sse_ret / denom,
                mae: sae_ret / denom,
            },
            PaperMethodMetrics {
                name: "naive_last".into(),
                n_queries: n_q,
                mse: sse_naive / denom,
                mae: sae_naive / denom,
            },
            PaperMethodMetrics {
                name: "retrieval_only_pearson".into(),
                n_queries: n_q,
                mse: sse_td / denom,
                mae: sae_td / denom,
            },
        ],
    })
}

fn build_entry(
    ch: &[Vec<f64>],
    scaled: &[Vec<f64>],
    start: usize,
    lookback: usize,
    horizon: usize,
    params: &SpecReTfParams,
) -> Result<LibEntry, PaperEvalError> {
    let n_c = ch.len();
    let mut descs = Vec::with_capacity(n_c);
    let mut windows = Vec::with_capacity(n_c);
    for c in 0..n_c {
        let w = &ch[c][start..start + lookback];
        descs.push(spectral_descriptor(w, params)?);
        windows.push(w.to_vec());
    }
    let future: Vec<Vec<f64>> = (0..horizon)
        .map(|h| scaled[start + lookback + h].clone())
        .collect();
    Ok(LibEntry {
        descs,
        future,
        windows,
    })
}

fn flatten_future(future: &[Vec<f64>]) -> Vec<f64> {
    // One trajectory as H consecutive means? Softmax ensemble expects Vec of length H
    // for univariate. For multivariate we flatten H*C into length H*C and treat as
    // "horizon" = H*C, then unflatten. Equivalent to weighting each channel step.
    future.iter().flatten().copied().collect()
}

fn unflatten_future(flat: &[f64], n_c: usize) -> Vec<Vec<f64>> {
    if n_c == 0 {
        return vec![];
    }
    flat.chunks(n_c).map(|c| c.to_vec()).collect()
}

fn accumulate_mse_mae(
    pred: &[Vec<f64>],
    actual: &[Vec<f64>],
    sse: &mut f64,
    sae: &mut f64,
) {
    let h = pred.len().min(actual.len());
    for t in 0..h {
        let c = pred[t].len().min(actual[t].len());
        for j in 0..c {
            let e = pred[t][j] - actual[t][j];
            *sse += e * e;
            *sae += e.abs();
        }
    }
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let ma = a[..n].iter().sum::<f64>() / n as f64;
    let mb = b[..n].iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for i in 0..n {
        let xa = a[i] - ma;
        let xb = b[i] - mb;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    let den = (da * db).sqrt();
    if den < 1e-12 {
        0.0
    } else {
        num / den
    }
}

/// Convenience: ETTh borders + multi-horizon average.
pub fn run_etth_multi_horizon(
    dataset: &str,
    values: &[Vec<f64>],
    horizons: &[usize],
    base: &PaperEvalConfig,
) -> Result<PaperEvalReport, PaperEvalError> {
    let train_end = 12 * 30 * 24;
    let val_end = train_end + 4 * 30 * 24;
    let test_end = (val_end + 4 * 30 * 24).min(values.len());
    let test_lo = val_end.saturating_sub(base.lookback);
    let test_hi = test_end;

    let mut reports = Vec::new();
    for &h in horizons {
        let mut cfg = base.clone();
        cfg.horizon = h;
        reports.push(run_paper_eval(
            dataset, values, train_end, test_lo, test_hi, &cfg,
        )?);
    }

    fn avg_mse(reports: &[PaperHorizonReport], name: &str) -> Option<f64> {
        if reports.is_empty() {
            return None;
        }
        let mut s = 0.0;
        for r in reports {
            let m = r.methods.iter().find(|m| m.name == name)?;
            s += m.mse;
        }
        Some(s / reports.len() as f64)
    }
    let avg_mse_retrieval = avg_mse(&reports, "retrieval_only_specretf");
    let avg_mse_naive = avg_mse(&reports, "naive_last");
    let avg_mse_td_pearson = avg_mse(&reports, "retrieval_only_pearson");

    Ok(PaperEvalReport {
        dataset: dataset.to_string(),
        lookback: base.lookback,
        k: base.k,
        n_fft: base.params.n_fft,
        hop: base.params.hop,
        ema_alpha: base.params.ema_alpha,
        library_stride: base.library_stride,
        query_stride: base.query_stride,
        n_channels: values.first().map(|r| r.len()).unwrap_or(0),
        signal: "standard_scaler_raw",
        protocol: "train_library_test_query_retrieval_only",
        horizons: reports,
        avg_mse_retrieval,
        avg_mse_naive,
        avg_mse_td_pearson,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_synthetic_multivariate() {
        // Tiny synthetic: L=32 H=4, 2 channels
        let mut values = Vec::new();
        for t in 0..200 {
            let x = (t as f64 * 0.1).sin();
            let y = (t as f64 * 0.07).cos();
            values.push(vec![x, y]);
        }
        let cfg = PaperEvalConfig {
            lookback: 32,
            horizon: 4,
            k: 3,
            params: SpecReTfParams {
                n_fft: 16,
                hop: 8,
                ema_alpha: 0.1,
            },
            library_stride: 2,
            query_stride: 4,
        };
        let train_end = 120;
        let test_lo = 120 - 32;
        let test_hi = 200;
        let r = run_paper_eval("synth", &values, train_end, test_lo, test_hi, &cfg).unwrap();
        assert!(r.n_test_queries > 0);
        assert_eq!(r.methods.len(), 3);
    }
}
