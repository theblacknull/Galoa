//! SpecReTF-style spectral retrieval index + always-FIRE offline forecast.
//!
//! Phase-1 eval policy: **never abstain** — every query with enough history emits
//! a top-K aggregate forecast (weighted mean / median / P10 / P90).

use galoa_core::{
    log_returns, softmax_ensemble_forecast, spectral_descriptor, spectral_similarity, z_normalize,
    SpecReTfError, SpecReTfParams, SpectralDescriptor, WeightedForecast,
};
use thiserror::Error;

use crate::{DEFAULT_HORIZON, DEFAULT_WINDOW};

/// What series SpecReTF STFT sees (paper uses raw values; Galoa FX often uses log-returns).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalMode {
    /// `ln(c[t+1]/c[t])` over the W-close window (length W−1). Scale-invariant for FX.
    #[default]
    LogReturns,
    /// Raw close prices of length W — closer to SpecReTF paper STFT input.
    RawClose,
}

impl SignalMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LogReturns => "log_returns",
            Self::RawClose => "raw_close",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "log_returns" | "log-returns" | "returns" | "ret" => Some(Self::LogReturns),
            "raw_close" | "raw-close" | "close" | "raw" | "price" => Some(Self::RawClose),
            _ => None,
        }
    }
}

#[derive(Debug, Error)]
pub enum SpecReTfIndexError {
    #[error("need at least window+horizon samples, got {got} (W={window}, H={horizon})")]
    TooShort {
        got: usize,
        window: usize,
        horizon: usize,
    },
    #[error("window / horizon / k must be > 0")]
    InvalidParams,
    #[error(transparent)]
    Spec(#[from] SpecReTfError),
    #[error(transparent)]
    Returns(#[from] galoa_core::ReturnsError),
}

#[derive(Clone, Debug)]
pub struct SpecReTfEntry {
    pub end_idx: usize,
    pub descriptor: SpectralDescriptor,
    /// Absolute next-H closes after the window.
    pub future_abs: Vec<f64>,
    /// Last close of the window (for level-align).
    pub last_close: f64,
}

/// Precomputed SpecReTF library over a close series.
#[derive(Clone, Debug)]
pub struct SpecReTfIndex {
    pub window: usize,
    pub horizon: usize,
    pub params: SpecReTfParams,
    pub signal: SignalMode,
    pub closes: Vec<f64>,
    pub entries: Vec<SpecReTfEntry>,
}

#[derive(Clone, Debug)]
pub struct SpecNeighbor {
    pub end_idx: usize,
    pub similarity: f64,
}

#[derive(Clone, Debug)]
pub struct SpecForecastResult {
    pub neighbors: Vec<SpecNeighbor>,
    pub forecast: WeightedForecast,
    pub query_last: f64,
    pub query_end: usize,
}

impl SpecReTfIndex {
    pub fn build(
        closes: &[f64],
        window: usize,
        horizon: usize,
        params: SpecReTfParams,
    ) -> Result<Self, SpecReTfIndexError> {
        Self::build_with_signal(closes, window, horizon, params, SignalMode::default())
    }

    pub fn build_with_signal(
        closes: &[f64],
        window: usize,
        horizon: usize,
        params: SpecReTfParams,
        signal: SignalMode,
    ) -> Result<Self, SpecReTfIndexError> {
        if window == 0 || horizon == 0 {
            return Err(SpecReTfIndexError::InvalidParams);
        }
        let need = window + horizon;
        if closes.len() < need {
            return Err(SpecReTfIndexError::TooShort {
                got: closes.len(),
                window,
                horizon,
            });
        }
        let min_w = match signal {
            // log-returns shorten by 1 → need W ≥ M+1
            SignalMode::LogReturns => params.n_fft + 1,
            SignalMode::RawClose => params.n_fft,
        };
        if window < min_w {
            return Err(SpecReTfIndexError::TooShort {
                got: window,
                window: min_w,
                horizon,
            });
        }

        let mut entries = Vec::new();
        let last_start = closes.len() - need;
        for start in 0..=last_start {
            let w = &closes[start..start + window];
            match descriptor_from_closes(w, &params, signal) {
                Ok(descriptor) => {
                    entries.push(SpecReTfEntry {
                        end_idx: start + window,
                        descriptor,
                        future_abs: closes[start + window..start + window + horizon].to_vec(),
                        last_close: closes[start + window - 1],
                    });
                }
                Err(SpecReTfIndexError::Returns(_) | SpecReTfIndexError::Spec(_)) => continue,
                Err(e) => return Err(e),
            }
        }
        Ok(Self {
            window,
            horizon,
            params,
            signal,
            closes: closes.to_vec(),
            entries,
        })
    }

    /// Always-FIRE forecast: top-K by SpecReTF similarity, exclude overlapping windows.
    pub fn forecast_at(
        &self,
        query_end: usize,
        k: usize,
    ) -> Result<SpecForecastResult, SpecReTfIndexError> {
        if k == 0 || query_end < self.window || query_end > self.closes.len() {
            return Err(SpecReTfIndexError::InvalidParams);
        }
        let q_start = query_end - self.window;
        let query_window = &self.closes[q_start..query_end];
        let q_desc = descriptor_from_closes(query_window, &self.params, self.signal)?;
        let q_last = self.closes[query_end - 1];

        let mut scored: Vec<(f64, usize)> = Vec::with_capacity(self.entries.len());
        for (i, entry) in self.entries.iter().enumerate() {
            if overlaps(entry.end_idx, self.window, self.horizon, query_end, self.window) {
                continue;
            }
            // Causal: only use library windows that finished before the query window.
            if entry.end_idx > q_start {
                continue;
            }
            let sim = spectral_similarity(&q_desc, &entry.descriptor, &self.params)?;
            scored.push((sim, i));
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let take = k.min(scored.len());

        let neighbors: Vec<SpecNeighbor> = scored[..take]
            .iter()
            .map(|(sim, idx)| SpecNeighbor {
                end_idx: self.entries[*idx].end_idx,
                similarity: *sim,
            })
            .collect();

        // SpecReTF eq. (10): Y_retrieval = Σ exp(S_k) Y_k / Σ exp(S_i)
        let (trajectories, scores) = if take == 0 {
            (vec![vec![q_last; self.horizon]], vec![0.0])
        } else {
            let mut trajs = Vec::with_capacity(take);
            let mut ss = Vec::with_capacity(take);
            for (sim, idx) in &scored[..take] {
                let entry = &self.entries[*idx];
                let shift = q_last - entry.last_close;
                trajs.push(entry.future_abs.iter().map(|p| p + shift).collect());
                ss.push(*sim); // raw S_k — softmax applied inside ensemble
            }
            (trajs, ss)
        };

        let forecast = softmax_ensemble_forecast(&trajectories, &scores);
        Ok(SpecForecastResult {
            neighbors,
            forecast,
            query_last: q_last,
            query_end,
        })
    }
}

fn overlaps(entry_end: usize, window: usize, horizon: usize, query_end: usize, q_window: usize) -> bool {
    let e_start = entry_end.saturating_sub(window);
    let e_end = entry_end + horizon;
    let q_start = query_end.saturating_sub(q_window);
    e_start < query_end && e_end > q_start
}

fn descriptor_from_closes(
    closes: &[f64],
    params: &SpecReTfParams,
    signal: SignalMode,
) -> Result<SpectralDescriptor, SpecReTfIndexError> {
    match signal {
        SignalMode::LogReturns => {
            let rets = log_returns(closes)?;
            Ok(spectral_descriptor(&rets, params)?)
        }
        SignalMode::RawClose => Ok(spectral_descriptor(closes, params)?),
    }
}

// ─── Offline walk-forward eval ───────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct EvalConfig {
    pub window: usize,
    pub horizon: usize,
    pub k: usize,
    pub params: SpecReTfParams,
    pub signal: SignalMode,
    /// Subsample query stride (1 = every bar). Larger → faster smoke runs.
    pub stride: usize,
    /// Minimum library entries before first query (expanding warm-up).
    pub min_history_windows: usize,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            window: DEFAULT_WINDOW,
            horizon: DEFAULT_HORIZON,
            k: galoa_core::SPECRETF_K,
            params: SpecReTfParams::default(),
            // Gold M1 sweep: raw close STFT beat log-returns vs naive (diagnostic only).
            signal: SignalMode::RawClose,
            stride: 1,
            min_history_windows: 64,
        }
    }
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct MethodMetrics {
    pub name: String,
    pub n_queries: usize,
    pub mse_close: f64,
    pub mae_close: f64,
    pub mse_return: f64,
    pub mae_return: f64,
    /// Sign(Δclose over H) hit-rate vs actual.
    pub dir_hit_h: f64,
    /// Mean per-step sign(Δclose step) hit-rate.
    pub dir_hit_step: f64,
    /// Fraction of actual H-closes inside [P10, P90] (SpecReTF / kNN only).
    pub band_coverage: Option<f64>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct EvalReport {
    pub n_bars: usize,
    pub window: usize,
    pub horizon: usize,
    pub k: usize,
    pub stride: usize,
    pub n_fft: usize,
    pub hop: usize,
    pub ema_alpha: f64,
    pub signal: SignalMode,
    pub methods: Vec<MethodMetrics>,
}

/// One hyperparameter grid cell (SpecReTF vs baselines).
#[derive(Clone, Debug, serde::Serialize)]
pub struct SweepRow {
    pub window: usize,
    pub horizon: usize,
    pub k: usize,
    pub n_fft: usize,
    pub hop: usize,
    pub ema_alpha: f64,
    pub signal: SignalMode,
    pub stride: usize,
    pub n_frames: usize,
    pub specretf: MethodMetrics,
    pub naive_last: MethodMetrics,
    pub td_znorm_l2_knn: MethodMetrics,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct SweepReport {
    pub n_bars: usize,
    pub protocol: String,
    pub rows: Vec<SweepRow>,
}

#[derive(Default)]
struct Acc {
    n: usize,
    sse_close: f64,
    sae_close: f64,
    sse_ret: f64,
    sae_ret: f64,
    dir_h_ok: usize,
    dir_step_ok: usize,
    dir_step_n: usize,
    band_in: usize,
    band_n: usize,
}

impl Acc {
    fn finish(self, name: &str, with_band: bool) -> MethodMetrics {
        let n = self.n.max(1) as f64;
        MethodMetrics {
            name: name.to_string(),
            n_queries: self.n,
            mse_close: self.sse_close / n,
            mae_close: self.sae_close / n,
            mse_return: self.sse_ret / n,
            mae_return: self.sae_ret / n,
            dir_hit_h: self.dir_h_ok as f64 / n,
            dir_hit_step: if self.dir_step_n == 0 {
                0.0
            } else {
                self.dir_step_ok as f64 / self.dir_step_n as f64
            },
            band_coverage: if with_band && self.band_n > 0 {
                Some(self.band_in as f64 / self.band_n as f64)
            } else {
                None
            },
        }
    }
}

/// Walk-forward SpecReTF eval + naive + time-domain z-norm L2 kNN baselines.
///
/// Always forecasts (no abstain). Expanding causal library: neighbours must end
/// before the query window starts.
pub fn run_specretf_eval(closes: &[f64], cfg: &EvalConfig) -> Result<EvalReport, SpecReTfIndexError> {
    if cfg.window == 0 || cfg.horizon == 0 || cfg.k == 0 || cfg.stride == 0 {
        return Err(SpecReTfIndexError::InvalidParams);
    }
    let need = cfg.window + cfg.horizon;
    if closes.len() < need + cfg.min_history_windows {
        return Err(SpecReTfIndexError::TooShort {
            got: closes.len(),
            window: cfg.window,
            horizon: cfg.horizon,
        });
    }

    let index = SpecReTfIndex::build_with_signal(
        closes,
        cfg.window,
        cfg.horizon,
        cfg.params.clone(),
        cfg.signal,
    )?;

    // Precompute z-norm vectors for time-domain baseline (same window starts as index).
    let mut z_feats: Vec<Option<Vec<f64>>> = Vec::with_capacity(index.entries.len());
    for e in &index.entries {
        let start = e.end_idx - cfg.window;
        let w = &closes[start..e.end_idx];
        z_feats.push(z_normalize(w).ok().map(|(z, _)| z));
    }

    let (acc_spec, acc_naive, acc_td) = eval_accums(closes, &index, &z_feats, cfg)?;

    Ok(EvalReport {
        n_bars: closes.len(),
        window: cfg.window,
        horizon: cfg.horizon,
        k: cfg.k,
        stride: cfg.stride,
        n_fft: cfg.params.n_fft,
        hop: cfg.params.hop,
        ema_alpha: cfg.params.ema_alpha,
        signal: cfg.signal,
        methods: vec![
            acc_spec.finish("specretf", true),
            acc_naive.finish("naive_last", false),
            acc_td.finish("td_znorm_l2_knn", false),
        ],
    })
}

fn eval_accums(
    closes: &[f64],
    index: &SpecReTfIndex,
    z_feats: &[Option<Vec<f64>>],
    cfg: &EvalConfig,
) -> Result<(Acc, Acc, Acc), SpecReTfIndexError> {
    let mut acc_spec = Acc::default();
    let mut acc_naive = Acc::default();
    let mut acc_td = Acc::default();

    let first_end = cfg.window + cfg.min_history_windows;
    let last_end = closes.len() - cfg.horizon;
    let mut end = first_end;
    while end <= last_end {
        let q_last = closes[end - 1];
        let actual: Vec<f64> = closes[end..end + cfg.horizon].to_vec();

        let spec = index.forecast_at(end, cfg.k)?;
        accumulate_pred(
            &mut acc_spec,
            q_last,
            &actual,
            &pred_means(&spec.forecast),
            Some(&spec.forecast),
        );

        let naive_pred = vec![q_last; cfg.horizon];
        accumulate_pred(&mut acc_naive, q_last, &actual, &naive_pred, None);

        let td_pred = time_domain_knn_forecast(
            closes,
            index,
            z_feats,
            end,
            cfg.window,
            cfg.horizon,
            cfg.k,
        );
        accumulate_pred(&mut acc_td, q_last, &actual, &td_pred, None);

        end += cfg.stride;
    }
    Ok((acc_spec, acc_naive, acc_td))
}

/// Paper-like hyperparameter grid. Rebuilds SpecReTF index when `(W,M,α,signal)` change;
/// reuses index across `K` values.
pub fn run_specretf_sweep(
    closes: &[f64],
    windows: &[usize],
    horizon: usize,
    ks: &[usize],
    n_ffts: &[usize],
    alphas: &[f64],
    signals: &[SignalMode],
    stride: usize,
    min_history: usize,
) -> Result<SweepReport, SpecReTfIndexError> {
    if windows.is_empty()
        || ks.is_empty()
        || n_ffts.is_empty()
        || alphas.is_empty()
        || signals.is_empty()
        || horizon == 0
        || stride == 0
    {
        return Err(SpecReTfIndexError::InvalidParams);
    }

    let mut rows = Vec::new();
    for &window in windows {
        for &signal in signals {
            for &n_fft in n_ffts {
                let hop = n_fft / 2;
                if hop == 0 {
                    continue;
                }
                for &ema_alpha in alphas {
                    let params = SpecReTfParams {
                        n_fft,
                        hop,
                        ema_alpha,
                    };
                    let index = match SpecReTfIndex::build_with_signal(
                        closes,
                        window,
                        horizon,
                        params.clone(),
                        signal,
                    ) {
                        Ok(idx) => idx,
                        Err(SpecReTfIndexError::TooShort { .. }) => continue,
                        Err(e) => return Err(e),
                    };
                    let mut z_feats: Vec<Option<Vec<f64>>> =
                        Vec::with_capacity(index.entries.len());
                    for e in &index.entries {
                        let start = e.end_idx - window;
                        let w = &closes[start..e.end_idx];
                        z_feats.push(z_normalize(w).ok().map(|(z, _)| z));
                    }
                    let signal_len = match signal {
                        SignalMode::LogReturns => window.saturating_sub(1),
                        SignalMode::RawClose => window,
                    };
                    let n_frames = params.num_frames(signal_len);

                    for &k in ks {
                        let cfg = EvalConfig {
                            window,
                            horizon,
                            k,
                            params: params.clone(),
                            signal,
                            stride,
                            min_history_windows: min_history,
                        };
                        if closes.len() < window + horizon + min_history {
                            continue;
                        }
                        let (acc_spec, acc_naive, acc_td) =
                            eval_accums(closes, &index, &z_feats, &cfg)?;
                        rows.push(SweepRow {
                            window,
                            horizon,
                            k,
                            n_fft,
                            hop,
                            ema_alpha,
                            signal,
                            stride,
                            n_frames,
                            specretf: acc_spec.finish("specretf", true),
                            naive_last: acc_naive.finish("naive_last", false),
                            td_znorm_l2_knn: acc_td.finish("td_znorm_l2_knn", false),
                        });
                    }
                }
            }
        }
    }

    Ok(SweepReport {
        n_bars: closes.len(),
        protocol: "walk_forward_expanding_causal_always_fire".into(),
        rows,
    })
}

fn pred_means(f: &WeightedForecast) -> Vec<f64> {
    f.points.iter().map(|p| p.mean).collect()
}

fn accumulate_pred(
    acc: &mut Acc,
    q_last: f64,
    actual: &[f64],
    pred: &[f64],
    band: Option<&WeightedForecast>,
) {
    let h = actual.len().min(pred.len());
    if h == 0 {
        return;
    }
    acc.n += 1;

    // Per-query means over the H steps (finish() averages across queries).
    let mut sse_c = 0.0;
    let mut sae_c = 0.0;
    let mut sse_r = 0.0;
    let mut sae_r = 0.0;
    for i in 0..h {
        let e = pred[i] - actual[i];
        sse_c += e * e;
        sae_c += e.abs();
        let a_ret = if q_last > 0.0 && actual[i] > 0.0 {
            (actual[i] / q_last).ln()
        } else {
            0.0
        };
        let p_ret = if q_last > 0.0 && pred[i] > 0.0 {
            (pred[i] / q_last).ln()
        } else {
            0.0
        };
        let er = p_ret - a_ret;
        sse_r += er * er;
        sae_r += er.abs();

        let prev_a = if i == 0 { q_last } else { actual[i - 1] };
        let prev_p = if i == 0 { q_last } else { pred[i - 1] };
        let da = actual[i] - prev_a;
        let dp = pred[i] - prev_p;
        if same_direction(da, dp) {
            acc.dir_step_ok += 1;
        }
        acc.dir_step_n += 1;
    }
    let hf = h as f64;
    acc.sse_close += sse_c / hf;
    acc.sae_close += sae_c / hf;
    acc.sse_ret += sse_r / hf;
    acc.sae_ret += sae_r / hf;

    let da_h = actual[h - 1] - q_last;
    let dp_h = pred[h - 1] - q_last;
    if same_direction(da_h, dp_h) {
        acc.dir_h_ok += 1;
    }

    if let Some(fc) = band {
        for i in 0..h.min(fc.points.len()) {
            let lo = fc.points[i].p10;
            let hi = fc.points[i].p90;
            if actual[i] >= lo && actual[i] <= hi {
                acc.band_in += 1;
            }
            acc.band_n += 1;
        }
    }
}

/// Time-domain baseline: z-norm L2 kNN, level-align futures, unweighted mean.
fn time_domain_knn_forecast(
    closes: &[f64],
    index: &SpecReTfIndex,
    z_feats: &[Option<Vec<f64>>],
    query_end: usize,
    window: usize,
    horizon: usize,
    k: usize,
) -> Vec<f64> {
    let q_start = query_end - window;
    let q_last = closes[query_end - 1];
    let q_win = &closes[q_start..query_end];
    let Ok((q_z, _)) = z_normalize(q_win) else {
        return vec![q_last; horizon];
    };

    let mut scored: Vec<(f64, usize)> = Vec::new();
    for (i, entry) in index.entries.iter().enumerate() {
        if entry.end_idx > q_start {
            continue;
        }
        if overlaps(entry.end_idx, window, horizon, query_end, window) {
            continue;
        }
        let Some(ref z) = z_feats[i] else {
            continue;
        };
        let dist = l2(&q_z, z);
        scored.push((dist, i));
    }
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let take = k.min(scored.len());
    if take == 0 {
        return vec![q_last; horizon];
    }
    let mut means = vec![0.0; horizon];
    for (_, idx) in &scored[..take] {
        let entry = &index.entries[*idx];
        let shift = q_last - entry.last_close;
        for t in 0..horizon.min(entry.future_abs.len()) {
            means[t] += entry.future_abs[t] + shift;
        }
    }
    for m in &mut means {
        *m /= take as f64;
    }
    means
}

fn l2(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Ternary sign match. Note: Rust `0.0.signum() == 1.0`, so we must not use `signum`
/// for flat forecasts (naive last-value would look like “always up”).
fn same_direction(da: f64, dp: f64) -> bool {
    sign3(da) == sign3(dp)
}

fn sign3(x: f64) -> i8 {
    const EPS: f64 = 1e-12;
    if x > EPS {
        1
    } else if x < -EPS {
        -1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planted_twin_ranks_first() {
        let w = 32;
        let h = 4;
        let params = SpecReTfParams {
            n_fft: 16,
            hop: 8,
            ema_alpha: 0.1,
            ..Default::default()
        };
        let mut closes: Vec<f64> = (0..80).map(|i| 100.0 + (i as f64) * 0.01).collect();
        let pattern: Vec<f64> = (0..w)
            .map(|i| {
                let t = i as f64;
                200.0 + (t * 0.4).sin() * 2.0 + t * 0.05
            })
            .collect();
        let future: Vec<f64> = (0..h)
            .map(|i| pattern[w - 1] + (i as f64 + 1.0) * 0.15)
            .collect();
        closes.extend_from_slice(&pattern);
        closes.extend_from_slice(&future);
        for i in 0..40 {
            closes.push(150.0 + ((i % 7) as f64) * 0.2);
        }
        // Scaled twin (same log-returns)
        let scale = 1.2;
        let q: Vec<f64> = pattern.iter().map(|p| p * scale).collect();
        closes.extend_from_slice(&q);
        // Need horizon room after query for index build; append dummy future for index
        for i in 0..h {
            closes.push(q[w - 1] + (i as f64 + 1.0) * 0.1);
        }

        let index = SpecReTfIndex::build(&closes, w, h, params).unwrap();
        let query_end = closes.len() - h;
        let result = index.forecast_at(query_end, 5).unwrap();
        assert!(!result.neighbors.is_empty());
        // Paper eq. (8) is unnormalized: identical windows score
        // S ≈ 2 · (1 − (1−α)^W_frames), not 1.0. For M=16 hop=8 on 31 returns,
        // W_frames=2 → S ≈ 2·(1−0.9²)=0.38. Assert near that self-score.
        let top = result.neighbors[0].similarity;
        assert!(
            (0.35..0.45).contains(&top),
            "planted twin should near self-score ≈0.38, got {top}"
        );
        assert_eq!(result.forecast.points.len(), h);
    }

    #[test]
    fn eval_smoke_runs() {
        let mut closes: Vec<f64> = Vec::new();
        let mut px = 2000.0;
        for i in 0..400 {
            px *= 1.0 + 0.0002 * ((i as f64) * 0.17).sin();
            closes.push(px);
        }
        let cfg = EvalConfig {
            window: 32,
            horizon: 4,
            k: 5,
            stride: 4,
            min_history_windows: 20,
            signal: SignalMode::LogReturns,
            params: SpecReTfParams {
                n_fft: 16,
                hop: 8,
                ..Default::default()
            },
        };
        let report = run_specretf_eval(&closes, &cfg).unwrap();
        assert_eq!(report.methods.len(), 3);
        assert!(report.methods[0].n_queries > 0);
    }
}
