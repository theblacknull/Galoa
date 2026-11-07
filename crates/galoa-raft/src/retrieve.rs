//! RAFT retrieval module (paper §3.2 eqs. 1–5).
//!
//! Sliding patches of lookback `L` (keys) with following `F` futures (values),
//! offset-subtract last step, top-`m` by similarity, temperature-softmax weights,
//! weighted sum of offset-subtracted values.

use crate::downsample::{avg_pool, repeat_upsample};
use crate::similarity::SimilarityKind;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RaftError {
    #[error("need at least L+F samples, got {got} (L={lookback}, F={horizon})")]
    TooShort {
        got: usize,
        lookback: usize,
        horizon: usize,
    },
    #[error("lookback / horizon / top_m / period must be > 0")]
    InvalidParams,
    #[error("period {period} does not fit lookback={lookback} or horizon={horizon}")]
    PeriodTooLarge {
        period: usize,
        lookback: usize,
        horizon: usize,
    },
}

/// Default temperature τ from paper §4.1 / App. B.
pub const DEFAULT_TAU: f64 = 0.1;
/// Default top-m (paper grid includes 1,5,10,20; 10 matches SpecReTF k for parity).
pub const DEFAULT_TOP_M: usize = 10;
/// Paper multi-period set `{1,2,4}`.
pub const DEFAULT_PERIODS: [usize; 3] = [1, 2, 4];
/// Short Gold M1 defaults (aligned with SpecReTF W/H; paper L∈{96..720} too long for 5d).
pub const DEFAULT_LOOKBACK: usize = 64;
/// Product default: next-bar only (`F=1`). Shares Galoa's SpecReTF horizon choice;
/// paper RAFT often uses long `F` (e.g. 96). Futures / value patches are length 1.
pub const DEFAULT_HORIZON: usize = 1;

#[derive(Clone, Debug)]
pub struct RaftParams {
    pub lookback: usize,
    pub horizon: usize,
    pub top_m: usize,
    pub tau: f64,
    pub periods: Vec<usize>,
    pub similarity: SimilarityKind,
    /// Library stride (paper default 1; raise for speed).
    pub library_stride: usize,
}

impl Default for RaftParams {
    fn default() -> Self {
        Self {
            lookback: DEFAULT_LOOKBACK,
            horizon: DEFAULT_HORIZON,
            top_m: DEFAULT_TOP_M,
            tau: DEFAULT_TAU,
            periods: DEFAULT_PERIODS.to_vec(),
            similarity: SimilarityKind::Pearson,
            library_stride: 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RaftNeighbor {
    pub start_idx: usize,
    pub similarity: f64,
    pub weight: f64,
}

#[derive(Clone, Debug)]
pub struct PeriodRetrieval {
    pub period: usize,
    /// Offset-subtracted weighted value patch, length ⌊F/p⌋.
    pub value_hat: Vec<f64>,
    /// Upsampled to F (repeat_interleave identity of g^(p)).
    pub value_up: Vec<f64>,
    pub neighbors: Vec<RaftNeighbor>,
}

#[derive(Clone, Debug)]
pub struct RaftRetrievalResult {
    pub query_end: usize,
    pub query_last: f64,
    /// Offset-subtracted query lookback.
    pub query_hat: Vec<f64>,
    pub periods: Vec<PeriodRetrieval>,
    /// Multi-period aggregate: mean of upsampled period values (identity Σ g^(p)).
    pub retrieval_agg: Vec<f64>,
}

/// Offset subtraction (eq. 1): subtract last step from every step.
pub fn offset_subtract(patch: &[f64]) -> Vec<f64> {
    let last = *patch.last().unwrap_or(&0.0);
    patch.iter().map(|v| v - last).collect()
}

/// Softmax over scores with temperature τ (eq. 4, top-m only).
pub fn softmax_tau(scores: &[f64], tau: f64) -> Vec<f64> {
    if scores.is_empty() {
        return Vec::new();
    }
    let t = if tau.abs() < 1e-15 { 1e-15 } else { tau };
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut exps: Vec<f64> = scores.iter().map(|s| ((s - max) / t).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum < 1e-300 {
        let u = 1.0 / scores.len() as f64;
        return vec![u; scores.len()];
    }
    for e in &mut exps {
        *e /= sum;
    }
    exps
}

/// Single-period retrieval on a (already possibly downsampled) series.
///
/// `closes` is the series at this period's resolution. Query is the last `L_p`
/// points ending at `query_end` (exclusive). Candidates are causal: key end ≤
/// query start, and key+value must fit.
pub fn retrieve_period(
    closes: &[f64],
    query_end: usize,
    lookback: usize,
    horizon: usize,
    top_m: usize,
    tau: f64,
    similarity: SimilarityKind,
    library_stride: usize,
) -> Result<(Vec<f64>, Vec<RaftNeighbor>), RaftError> {
    if lookback == 0 || horizon == 0 || top_m == 0 || library_stride == 0 {
        return Err(RaftError::InvalidParams);
    }
    if query_end < lookback || closes.len() < lookback + horizon {
        return Err(RaftError::TooShort {
            got: closes.len(),
            lookback,
            horizon,
        });
    }
    let q_start = query_end - lookback;
    let query = &closes[q_start..query_end];
    let q_hat = offset_subtract(query);

    // Key [start, start+L), value [start+L, start+L+F) must not overlap query
    // [q_start, query_end): require start+L+F ≤ q_start.
    let last_ok = q_start.saturating_sub(lookback + horizon);
    if q_start < lookback + horizon {
        return Ok((vec![0.0; horizon], Vec::new()));
    }

    let mut scored: Vec<(f64, usize)> = Vec::new();
    let mut start = 0usize;
    while start <= last_ok {
        let key = &closes[start..start + lookback];
        let k_hat = offset_subtract(key);
        let rho = similarity.score(&q_hat, &k_hat);
        scored.push((rho, start));
        start += library_stride;
    }

    if scored.is_empty() {
        return Ok((vec![0.0; horizon], Vec::new()));
    }

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let take = top_m.min(scored.len());
    let top = &scored[..take];
    let scores: Vec<f64> = top.iter().map(|(s, _)| *s).collect();
    let weights = softmax_tau(&scores, tau);

    let mut v_hat = vec![0.0; horizon];
    let mut neighbors = Vec::with_capacity(take);
    for (i, &(rho, s)) in top.iter().enumerate() {
        let w = weights[i];
        let key = &closes[s..s + lookback];
        let key_last = key[lookback - 1];
        let val = &closes[s + lookback..s + lookback + horizon];
        // Level-align futures to key end (start of horizon). Paper §3.2 also
        // demeans value patches by their *own* last step for the trained head;
        // for retrieval-only forecast that forces ˆv_F≈0, so we use key-last
        // alignment (NLinear / SpecReTF-style) here — see algorithm notes.
        for t in 0..horizon {
            v_hat[t] += w * (val[t] - key_last);
        }
        neighbors.push(RaftNeighbor {
            start_idx: s,
            similarity: rho,
            weight: w,
        });
    }
    Ok((v_hat, neighbors))
}

/// Multi-period RAFT retrieval (paper §3.3). Operates on raw univariate closes.
///
/// For each period `p`, query/key/value patches are average-pooled independently
/// (per-patch), not by re-indexing a globally pooled series — matches paper text.
pub fn retrieve_multiperiod(
    closes: &[f64],
    query_end: usize,
    params: &RaftParams,
) -> Result<RaftRetrievalResult, RaftError> {
    let l = params.lookback;
    let f = params.horizon;
    if l == 0 || f == 0 || params.top_m == 0 || params.periods.is_empty() {
        return Err(RaftError::InvalidParams);
    }
    if query_end < l || closes.len() < l {
        return Err(RaftError::TooShort {
            got: closes.len().min(query_end),
            lookback: l,
            horizon: f,
        });
    }
    for &p in &params.periods {
        if p == 0 || l / p < 2 || f / p < 1 {
            return Err(RaftError::PeriodTooLarge {
                period: p,
                lookback: l,
                horizon: f,
            });
        }
    }

    let q_last = closes[query_end - 1];
    let query = &closes[query_end - l..query_end];
    let query_hat = offset_subtract(query);
    let q_start = query_end - l;

    let mut period_results = Vec::with_capacity(params.periods.len());
    let mut agg = vec![0.0; f];

    for &p in &params.periods {
        let l_p = l / p;
        let f_p = f / p;
        let q_ds = avg_pool(query, p);
        debug_assert_eq!(q_ds.len(), l_p);
        let q_hat_p = offset_subtract(&q_ds);

        // Causal + non-overlap in *original* index space.
        let last_ok = q_start.saturating_sub(l + f);
        let mut scored: Vec<(f64, usize)> = Vec::new();
        if q_start >= l + f {
            let mut start = 0usize;
            let stride = params.library_stride.max(1);
            while start <= last_ok {
                let key_ds = avg_pool(&closes[start..start + l], p);
                let k_hat = offset_subtract(&key_ds);
                let rho = params.similarity.score(&q_hat_p, &k_hat);
                scored.push((rho, start));
                start += stride;
            }
        }

        let (v_hat, neighbors) = if scored.is_empty() {
            (vec![0.0; f_p], Vec::new())
        } else {
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let take = params.top_m.min(scored.len());
            let top = &scored[..take];
            let scores: Vec<f64> = top.iter().map(|(s, _)| *s).collect();
            let weights = softmax_tau(&scores, params.tau);
            let mut v_hat = vec![0.0; f_p];
            let mut neighbors = Vec::with_capacity(take);
            for (i, &(rho, s)) in top.iter().enumerate() {
                let w = weights[i];
                let key_last = closes[s + l - 1];
                let val_ds = avg_pool(&closes[s + l..s + l + f], p);
                // Level-align in downsampled space using pooled key last ≈ mean of
                // last p raw closes; use raw key_last projected: subtract key_last
                // from each raw future then pool — equivalent enough for MVP:
                // pool(val) - pool(key_last repeated) = pool(val) - key_last.
                for t in 0..f_p {
                    v_hat[t] += w * (val_ds[t] - key_last);
                }
                neighbors.push(RaftNeighbor {
                    start_idx: s,
                    similarity: rho,
                    weight: w,
                });
            }
            (v_hat, neighbors)
        };

        let value_up = repeat_upsample(&v_hat, p, f);
        for t in 0..f {
            agg[t] += value_up[t];
        }
        period_results.push(PeriodRetrieval {
            period: p,
            value_hat: v_hat,
            value_up,
            neighbors,
        });
    }

    let n_p = params.periods.len() as f64;
    for a in &mut agg {
        *a /= n_p;
    }

    Ok(RaftRetrievalResult {
        query_end,
        query_last: q_last,
        query_hat,
        periods: period_results,
        retrieval_agg: agg,
    })
}

/// Retrieval-only forecast: restore offset (eq. 8 with identity head on retrieval).
///
/// Matches paper retrieval stage; skips trained `f`/`g`/`h` (see `head` module).
pub fn forecast_retrieval_only(result: &RaftRetrievalResult) -> Vec<f64> {
    result
        .retrieval_agg
        .iter()
        .map(|v| v + result.query_last)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planted_pattern_retrieves() {
        let l = 16;
        let f = 4;
        let mut closes = Vec::new();
        // Noise prefix
        for i in 0..80 {
            closes.push(100.0 + (i as f64) * 0.01);
        }
        // Planted pattern + rising future
        let pattern: Vec<f64> = (0..l)
            .map(|i| 200.0 + (i as f64 * 0.5).sin() * 3.0 + i as f64 * 0.1)
            .collect();
        let future: Vec<f64> = (0..f)
            .map(|i| pattern[l - 1] + (i as f64 + 1.0) * 0.5)
            .collect();
        closes.extend_from_slice(&pattern);
        closes.extend_from_slice(&future);
        // More noise
        for i in 0..40 {
            closes.push(150.0 + (i as f64) * 0.05);
        }
        // Query ≈ same shape, different offset
        let q: Vec<f64> = pattern.iter().map(|p| p + 50.0).collect();
        closes.extend_from_slice(&q);
        for i in 0..f {
            closes.push(q[l - 1] + (i as f64 + 1.0) * 0.1);
        }

        let params = RaftParams {
            lookback: l,
            horizon: f,
            top_m: 1,
            periods: vec![1],
            ..Default::default()
        };
        let query_end = closes.len() - f;
        let r = retrieve_multiperiod(&closes, query_end, &params).unwrap();
        assert!(!r.periods[0].neighbors.is_empty());
        // Top neighbour should be near the planted start
        let top = &r.periods[0].neighbors[0];
        assert!(
            top.similarity > 0.99,
            "expected near-perfect Pearson, got {}",
            top.similarity
        );
        let pred = forecast_retrieval_only(&r);
        assert_eq!(pred.len(), f);
        // Planted future rises +0.5/step; after level-align, prediction should not be flat-naive.
        let rise = pred[f - 1] - pred[0];
        assert!(
            rise > 0.5,
            "expected rising retrieval forecast, got Δ={rise}, pred={pred:?}"
        );
    }

    #[test]
    fn softmax_peaks_on_max() {
        let w = softmax_tau(&[0.1, 0.9, 0.2], 0.1);
        assert!(w[1] > 0.9);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }
}
