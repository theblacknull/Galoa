//! In-memory sliding-window index with exact brute-force kNN.
//!
//! MVP prefers a working exact search over ANN crate friction. Swap to
//! usearch / hnsw_rs later when series length justifies it.

use galoa_core::{ensemble_forecast, z_normalize, Forecast, PriceEvent, ZNormError};
use thiserror::Error;

pub const DEFAULT_WINDOW: usize = 64;
pub const DEFAULT_HORIZON: usize = 16;
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
    ZNorm(#[from] ZNormError),
    #[error("no valid analogues found (flat windows?)")]
    NoAnalogues,
}

/// One indexed historical window: z-normalized shape + absolute next-H closes.
#[derive(Clone, Debug)]
pub struct IndexedWindow {
    /// End index in the close series (exclusive end of the W window).
    pub end_idx: usize,
    pub z_window: Vec<f64>,
    /// Absolute price trajectory for the H bars after the window.
    pub future_abs: Vec<f64>,
}

/// Exact kNN over z-normalized windows.
#[derive(Clone, Debug)]
pub struct WindowIndex {
    pub window: usize,
    pub horizon: usize,
    /// Raw closes used to build the index (resampled series).
    pub closes: Vec<f64>,
    pub entries: Vec<IndexedWindow>,
}

impl WindowIndex {
    /// Build index from a resampled close series. Each entry i uses
    /// closes[i..i+W] as the pattern and closes[i+W..i+W+H] as the future.
    pub fn build(closes: &[f64], window: usize, horizon: usize) -> Result<Self, IndexError> {
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

        let mut entries = Vec::new();
        let last_start = closes.len() - need;
        for start in 0..=last_start {
            let w = &closes[start..start + window];
            match z_normalize(w) {
                Ok((z_window, _)) => {
                    let future_abs = closes[start + window..start + window + horizon].to_vec();
                    entries.push(IndexedWindow {
                        end_idx: start + window,
                        z_window,
                        future_abs,
                    });
                }
                Err(ZNormError::ZeroStd) => continue,
                Err(e) => return Err(e.into()),
            }
        }
        if entries.is_empty() {
            return Err(IndexError::NoAnalogues);
        }
        Ok(Self {
            window,
            horizon,
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

    /// Query with the latest window of length `self.window` from `query_closes`.
    /// Excludes self-matches that overlap the query region when `exclude_end` is set
    /// (end index of the query window in the same series).
    pub fn forecast_from_window(
        &self,
        query_window: &[f64],
        k: usize,
        exclude_end: Option<usize>,
    ) -> Result<ForecastResult, IndexError> {
        if k == 0 || query_window.len() != self.window {
            return Err(IndexError::InvalidParams);
        }
        let (z_query, params) = z_normalize(query_window)?;

        let mut scored: Vec<(f64, usize)> = Vec::with_capacity(self.entries.len());
        for (i, entry) in self.entries.iter().enumerate() {
            if let Some(q_end) = exclude_end {
                // Skip windows that overlap the query window [q_end-W, q_end).
                let e_start = entry.end_idx.saturating_sub(self.window);
                let e_end = entry.end_idx + self.horizon;
                let q_start = q_end.saturating_sub(self.window);
                if e_start < q_end && e_end > q_start {
                    continue;
                }
            }
            let dist = l2_distance(&z_query, &entry.z_window);
            scored.push((dist, i));
        }
        if scored.is_empty() {
            return Err(IndexError::NoAnalogues);
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

        // Map each analogue's absolute future into query price space via z of the
        // analogue future relative to the analogue window, then denorm with query params.
        // Simpler MVP: scale futures by aligning last close of analogue window to query last.
        let q_last = *query_window.last().unwrap();
        let mut trajectories = Vec::with_capacity(take);
        for (_, idx) in &scored[..take] {
            let entry = &self.entries[*idx];
            let a_start = entry.end_idx - self.window;
            let a_last = self.closes[entry.end_idx - 1];
            let shift = q_last - a_last;
            let traj: Vec<f64> = entry.future_abs.iter().map(|p| p + shift).collect();
            let _ = a_start; // kept for clarity / future overlap logic
            trajectories.push(traj);
        }

        let forecast = ensemble_forecast(&trajectories);
        Ok(ForecastResult {
            forecast,
            neighbors,
            query_mean: params.mean,
            query_std: params.std,
            query_last: q_last,
        })
    }

    /// Forecast using the last `window` closes of the indexed series, excluding
    /// overlapping historical windows so we do not cheat with self-matches.
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
}

#[derive(Clone, Debug)]
pub struct Neighbor {
    pub end_idx: usize,
    pub distance: f64,
}

#[derive(Clone, Debug)]
pub struct ForecastResult {
    pub forecast: Forecast,
    pub neighbors: Vec<Neighbor>,
    pub query_mean: f64,
    pub query_std: f64,
    pub query_last: f64,
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
    fn finds_planted_similar_segment() {
        // Build a series where the last W bars repeat an earlier pattern with known future.
        let w = 8;
        let h = 4;
        let mut closes = Vec::new();
        // noise head
        for i in 0..20 {
            closes.push(100.0 + (i as f64) * 0.01);
        }
        let pattern: Vec<f64> = (0..w).map(|i| 200.0 + (i as f64) * 0.5).collect();
        let future: Vec<f64> = (0..h).map(|i| 204.0 + (i as f64) * 0.25).collect();
        closes.extend_from_slice(&pattern);
        closes.extend_from_slice(&future);
        // filler
        for i in 0..30 {
            closes.push(150.0 + ((i % 7) as f64) * 0.1);
        }
        // query = same pattern shape at different level
        let q_pattern: Vec<f64> = pattern.iter().map(|p| p + 50.0).collect();
        closes.extend_from_slice(&q_pattern);

        let index = WindowIndex::build(&closes, w, h).unwrap();
        let result = index.forecast_latest(3).unwrap();
        assert_eq!(result.forecast.horizon, h);
        assert!(result.neighbors[0].distance < 1e-6);
        // After level shift, median step-1 ≈ (204+50)
        let step1 = result.forecast.points[0].median;
        assert!((step1 - 254.0).abs() < 0.5, "got {step1}");
    }
}
