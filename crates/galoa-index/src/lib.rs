//! Spectral window index: STFT features + exact kNN + selective FIRE gates.
//!
//! Also: SpecReTF-style retrieval (JS amp + phase + EMA) for phase-1 offline eval
//! (`specretf` module — always FIRE).
//!
//! Time-domain z-norm L2 matching is not a product path (eval baseline only).

mod fusion;
mod paper_eval;
mod specretf;

pub use fusion::{
    run_specretf_fusion_eval, run_specretf_fusion_signals, ForecastSignal, FusionTrainConfig,
    RidgeMap,
};
pub use paper_eval::{
    run_etth_multi_horizon, run_paper_eval, PaperEvalConfig, PaperEvalError, PaperEvalReport,
    PaperHorizonReport, PaperMethodMetrics,
};
pub use specretf::{
    run_specretf_eval, run_specretf_sweep, EvalConfig, EvalReport, MethodMetrics, SignalMode,
    SpecForecastResult, SpecNeighbor, SpecReTfEntry, SpecReTfIndex, SpecReTfIndexError, SweepReport,
    SweepRow,
};

use galoa_core::{
    ensemble_forecast, evaluate_gates, log_returns, stft_magnitude_features, AbstainReason,
    Forecast, GateConfig, MatchDecision, PriceEvent, ReturnsError, StftError, StftParams,
};
use thiserror::Error;

pub const DEFAULT_WINDOW: usize = 64;
/// Product default: next-bar only (`H=1`). Papers often use long multi-step `H`;
/// Galoa forecasts a single next close / log-return, so futures `Y_k` are length 1.
pub const DEFAULT_HORIZON: usize = 1;
pub const DEFAULT_K: usize = 10;

#[derive(Debug, Error)]
pub enum IndexError {
    #[error("need at least window+horizon samples, got {got} (W={window}, H={horizon})")]
    TooShort {
        got: usize,
        window: usize,
        horizon: usize,
    },
    #[error("window / horizon / k must be > 0")]
    InvalidParams,
    #[error(transparent)]
    Returns(#[from] ReturnsError),
    #[error(transparent)]
    Stft(#[from] StftError),
    #[error("no valid spectral analogues found")]
    NoAnalogues,
}

/// One indexed historical window: STFT feature vector + absolute next-H closes.
#[derive(Clone, Debug)]
pub struct IndexedWindow {
    /// End index in the close series (exclusive end of the W window).
    pub end_idx: usize,
    /// L2-normalized STFT magnitude feature vector.
    pub features: Vec<f64>,
    /// Absolute next-H closes after the window. Product default `H=1` ⇒ length 1.
    pub future_abs: Vec<f64>,
}

/// Exact kNN over STFT feature vectors (spectral-only matching).
#[derive(Clone, Debug)]
pub struct WindowIndex {
    pub window: usize,
    pub horizon: usize,
    pub stft: StftParams,
    pub gates: GateConfig,
    /// Raw closes used to build the index (resampled series).
    pub closes: Vec<f64>,
    pub entries: Vec<IndexedWindow>,
}

impl WindowIndex {
    /// Build spectral index from a close series.
    /// Each entry uses closes[start..start+W] → log-returns → STFT F, and
    /// closes[start+W..start+W+H] as the absolute future.
    pub fn build(closes: &[f64], window: usize, horizon: usize) -> Result<Self, IndexError> {
        Self::build_with(closes, window, horizon, StftParams::default(), GateConfig::default())
    }

    pub fn build_with(
        closes: &[f64],
        window: usize,
        horizon: usize,
        stft: StftParams,
        gates: GateConfig,
    ) -> Result<Self, IndexError> {
        if window == 0 || horizon == 0 {
            return Err(IndexError::InvalidParams);
        }
        let need = window + horizon;
        if closes.len() < need {
            return Err(IndexError::TooShort {
                got: closes.len(),
                window,
                horizon,
            });
        }
        // STFT needs ≥ n_fft returns → window closes ≥ n_fft + 1
        if window < stft.n_fft + 1 {
            return Err(IndexError::TooShort {
                got: window,
                window: stft.n_fft + 1,
                horizon,
            });
        }

        let mut entries = Vec::new();
        let last_start = closes.len() - need;
        for start in 0..=last_start {
            let w = &closes[start..start + window];
            match window_features(w, &stft) {
                Ok(features) => {
                    let future_abs = closes[start + window..start + window + horizon].to_vec();
                    entries.push(IndexedWindow {
                        end_idx: start + window,
                        features,
                        future_abs,
                    });
                }
                Err(IndexError::Returns(_) | IndexError::Stft(_)) => continue,
                Err(e) => return Err(e),
            }
        }
        if entries.is_empty() {
            return Err(IndexError::NoAnalogues);
        }
        Ok(Self {
            window,
            horizon,
            stft,
            gates,
            closes: closes.to_vec(),
            entries,
        })
    }

    pub fn from_events(
        events: &[PriceEvent],
        window: usize,
        horizon: usize,
    ) -> Result<Self, IndexError> {
        let closes: Vec<f64> = events.iter().map(|e| e.close).collect();
        Self::build(&closes, window, horizon)
    }

    pub fn from_events_with(
        events: &[PriceEvent],
        window: usize,
        horizon: usize,
        stft: StftParams,
        gates: GateConfig,
    ) -> Result<Self, IndexError> {
        let closes: Vec<f64> = events.iter().map(|e| e.close).collect();
        Self::build_with(&closes, window, horizon, stft, gates)
    }

    /// Query with a close window of length `self.window`.
    /// Excludes historical windows overlapping the query when `exclude_end` is set.
    pub fn forecast_from_window(
        &self,
        query_window: &[f64],
        k: usize,
        exclude_end: Option<usize>,
    ) -> Result<ForecastResult, IndexError> {
        if k == 0 || query_window.len() != self.window {
            return Err(IndexError::InvalidParams);
        }
        let q_feat = window_features(query_window, &self.stft)?;

        let mut scored: Vec<(f64, usize)> = Vec::with_capacity(self.entries.len());
        for (i, entry) in self.entries.iter().enumerate() {
            if let Some(q_end) = exclude_end {
                let e_start = entry.end_idx.saturating_sub(self.window);
                let e_end = entry.end_idx + self.horizon;
                let q_start = q_end.saturating_sub(self.window);
                if e_start < q_end && e_end > q_start {
                    continue;
                }
            }
            let dist = l2_distance(&q_feat, &entry.features);
            scored.push((dist, i));
        }
        if scored.is_empty() {
            return Ok(ForecastResult::abstain(
                AbstainReason::NoAnalogues,
                *query_window.last().unwrap(),
            ));
        }
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let take = k.min(scored.len());

        let neighbors: Vec<Neighbor> = scored[..take]
            .iter()
            .map(|(dist, idx)| {
                let entry = &self.entries[*idx];
                Neighbor {
                    end_idx: entry.end_idx,
                    distance: *dist,
                }
            })
            .collect();

        let q_last = *query_window.last().unwrap();
        let distances: Vec<f64> = neighbors.iter().map(|n| n.distance).collect();
        let h_rets: Vec<f64> = scored[..take]
            .iter()
            .map(|(_, idx)| {
                let entry = &self.entries[*idx];
                let a_last = self.closes[entry.end_idx - 1];
                let h_last = *entry.future_abs.last().unwrap_or(&a_last);
                if a_last > 0.0 && h_last > 0.0 {
                    (h_last / a_last).ln()
                } else {
                    0.0
                }
            })
            .collect();

        let (decision, reason, support, sigma) =
            evaluate_gates(&distances, &h_rets, &self.gates);

        if decision == MatchDecision::Abstain {
            return Ok(ForecastResult {
                decision: MatchDecision::Abstain,
                abstain_reason: reason,
                support,
                outcome_sigma: sigma,
                forecast: None,
                neighbors,
                query_last: q_last,
            });
        }

        // FIRE: level-align analogue futures to query last close, then ensemble.
        let mut trajectories = Vec::with_capacity(support.max(1));
        for (dist, idx) in &scored[..take] {
            if *dist > self.gates.d_max {
                continue;
            }
            let entry = &self.entries[*idx];
            let a_last = self.closes[entry.end_idx - 1];
            let shift = q_last - a_last;
            let traj: Vec<f64> = entry.future_abs.iter().map(|p| p + shift).collect();
            trajectories.push(traj);
        }
        let forecast = ensemble_forecast(&trajectories);
        Ok(ForecastResult {
            decision: MatchDecision::Fire,
            abstain_reason: None,
            support,
            outcome_sigma: sigma,
            forecast: Some(forecast),
            neighbors,
            query_last: q_last,
        })
    }

    /// Forecast using the last `window` closes, excluding overlapping history.
    pub fn forecast_latest(&self, k: usize) -> Result<ForecastResult, IndexError> {
        if self.closes.len() < self.window {
            return Err(IndexError::TooShort {
                got: self.closes.len(),
                window: self.window,
                horizon: self.horizon,
            });
        }
        let end = self.closes.len();
        let query = &self.closes[end - self.window..end];
        self.forecast_from_window(query, k, Some(end))
    }

    /// Scan every valid query end index (with history exclusion) and tally coverage.
    pub fn scan_coverage(&self, k: usize) -> Result<CoverageStats, IndexError> {
        if k == 0 {
            return Err(IndexError::InvalidParams);
        }
        let min_end = self.window;
        let max_end = self.closes.len();
        if max_end < min_end {
            return Err(IndexError::TooShort {
                got: self.closes.len(),
                window: self.window,
                horizon: self.horizon,
            });
        }

        let mut fire = 0usize;
        let mut abstain = 0usize;
        let mut low_support = 0usize;
        let mut high_dispersion = 0usize;
        let mut no_analogues = 0usize;

        // Only query ends that leave room for a prior library (skip early cold start
        // where almost everything overlaps). Start after window+horizon so non-overlap exists.
        let start_end = (self.window + self.horizon).min(max_end);
        for end in start_end..=max_end {
            let query = &self.closes[end - self.window..end];
            let r = self.forecast_from_window(query, k, Some(end))?;
            match r.decision {
                MatchDecision::Fire => fire += 1,
                MatchDecision::Abstain => {
                    abstain += 1;
                    match r.abstain_reason {
                        Some(AbstainReason::LowSupport) => low_support += 1,
                        Some(AbstainReason::HighDispersion) => high_dispersion += 1,
                        Some(AbstainReason::NoAnalogues) | None => no_analogues += 1,
                    }
                }
            }
        }
        let queries = fire + abstain;
        Ok(CoverageStats {
            queries,
            fire,
            abstain,
            low_support,
            high_dispersion,
            no_analogues,
            coverage: if queries == 0 {
                0.0
            } else {
                fire as f64 / queries as f64
            },
        })
    }
}

fn window_features(closes: &[f64], stft: &StftParams) -> Result<Vec<f64>, IndexError> {
    let rets = log_returns(closes)?;
    Ok(stft_magnitude_features(&rets, stft)?)
}

#[derive(Clone, Debug)]
pub struct Neighbor {
    pub end_idx: usize,
    pub distance: f64,
}

#[derive(Clone, Debug)]
pub struct ForecastResult {
    pub decision: MatchDecision,
    pub abstain_reason: Option<AbstainReason>,
    pub support: usize,
    pub outcome_sigma: f64,
    pub forecast: Option<Forecast>,
    pub neighbors: Vec<Neighbor>,
    pub query_last: f64,
}

impl ForecastResult {
    fn abstain(reason: AbstainReason, query_last: f64) -> Self {
        Self {
            decision: MatchDecision::Abstain,
            abstain_reason: Some(reason),
            support: 0,
            outcome_sigma: f64::NAN,
            forecast: None,
            neighbors: vec![],
            query_last,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CoverageStats {
    pub queries: usize,
    pub fire: usize,
    pub abstain: usize,
    pub low_support: usize,
    pub high_dispersion: usize,
    pub no_analogues: usize,
    /// `fire / queries`
    pub coverage: f64,
}

fn l2_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planted_spectral_twin_fires_or_matches() {
        // Multiplicative scale preserves log-returns → STFT distance ~0.
        let w = 32;
        let h = 4;
        let stft = StftParams {
            n_fft: 16,
            hop: 4,
        };
        let gates = GateConfig {
            d_max: 0.5,
            n_min: 1,
            sigma_max: 1.0,
        };

        let mut closes = Vec::new();
        for i in 0..40 {
            closes.push(100.0 + (i as f64) * 0.02);
        }
        // Planted pattern: smooth sine in price → structured returns
        let pattern_base: Vec<f64> = (0..w)
            .map(|i| {
                let t = i as f64;
                200.0 + (t * 0.4).sin() * 2.0 + t * 0.05
            })
            .collect();
        let future: Vec<f64> = (0..h)
            .map(|i| pattern_base[w - 1] + (i as f64 + 1.0) * 0.1)
            .collect();
        closes.extend_from_slice(&pattern_base);
        closes.extend_from_slice(&future);
        for i in 0..50 {
            closes.push(180.0 + ((i % 11) as f64) * 0.15);
        }
        // Same log-returns at a different absolute level (×1.15)
        let scale = 1.15;
        let q: Vec<f64> = pattern_base.iter().map(|p| p * scale).collect();
        closes.extend_from_slice(&q);

        let index = WindowIndex::build_with(&closes, w, h, stft, gates).unwrap();
        let result = index.forecast_latest(5).unwrap();
        assert!(
            result.neighbors[0].distance < 1e-9,
            "expected near-zero spectral distance, got {}",
            result.neighbors[0].distance
        );
        assert_eq!(result.decision, MatchDecision::Fire);
        let step1 = result.forecast.as_ref().unwrap().points[0].median;
        // Level-align by last-close shift: future[0] + (q_last - a_last)
        let q_last = q[w - 1];
        let a_last = pattern_base[w - 1];
        let expected = future[0] + (q_last - a_last);
        assert!((step1 - expected).abs() < 0.2, "got {step1} expected {expected}");
    }

    #[test]
    fn gate_abstain_smoke() {
        let w = 32;
        let h = 4;
        let stft = StftParams {
            n_fft: 16,
            hop: 4,
        };
        // Impossible proximity → always abstain
        let gates = GateConfig {
            d_max: 1e-12,
            n_min: 5,
            sigma_max: 1e-12,
        };
        let mut closes: Vec<f64> = (0..200).map(|i| 100.0 + (i as f64).sin() * 0.5).collect();
        for i in 0..w {
            closes.push(120.0 + i as f64 * 0.01);
        }
        let index = WindowIndex::build_with(&closes, w, h, stft, gates).unwrap();
        let result = index.forecast_latest(10).unwrap();
        assert_eq!(result.decision, MatchDecision::Abstain);
        assert!(result.forecast.is_none());
        assert!(result.abstain_reason.is_some());
    }
}
