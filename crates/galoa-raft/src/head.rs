//! Forecast head on top of RAFT retrieval (paper §3.3 eqs. 6–8).
//!
//! Paper: linear `f: L→F`, `g^(p): ⌊F/p⌋→F`, `h: 2F→F`, then add offset `x^L`.
//! On ~6k Gold M1 bars a full SGD train is unstable / leakage-prone under
//! walk-forward, so we ship:
//! 1. **identity** — retrieval aggregate + offset restore (no train)
//! 2. **ridge** — expanding-window closed-form ridge on `[ˆx_tail∥ṽ]` → `ŷ`
//!    (minimal linear stand-in for `h∘(f⊕g)`; documented as simplified)

use crate::retrieve::RaftRetrievalResult;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum HeadKind {
    /// `y = x^L + mean_p upsample(ṽ^(p))` — retrieval stage only.
    #[default]
    Identity,
    /// Ridge on concat(last-F of ˆx, ṽ_agg) → ŷ, then + x^L.
    Ridge,
}

#[derive(Clone, Debug)]
pub struct RidgeHead {
    /// Weights: input dim = 2F, output dim = F (row-major: out × in).
    pub w: Vec<f64>,
    pub bias: Vec<f64>,
    pub horizon: usize,
    pub fitted: bool,
}

impl RidgeHead {
    pub fn new(horizon: usize) -> Self {
        let dim_in = 2 * horizon;
        // Init near identity on retrieval half: ŷ ≈ ṽ.
        let mut w = vec![0.0; horizon * dim_in];
        for t in 0..horizon {
            // weight on retrieval features (second half)
            w[t * dim_in + horizon + t] = 1.0;
        }
        Self {
            w,
            bias: vec![0.0; horizon],
            horizon,
            fitted: false,
        }
    }

    pub fn features(result: &RaftRetrievalResult) -> Vec<f64> {
        let f = result.retrieval_agg.len();
        let mut x = Vec::with_capacity(2 * f);
        // f(ˆx): take last F of offset lookback (truncation of Linear L→F).
        let q = &result.query_hat;
        if q.len() >= f {
            x.extend_from_slice(&q[q.len() - f..]);
        } else {
            x.extend(std::iter::repeat(0.0).take(f - q.len()));
            x.extend_from_slice(q);
        }
        x.extend_from_slice(&result.retrieval_agg);
        x
    }

    pub fn predict(&self, result: &RaftRetrievalResult) -> Vec<f64> {
        let f = self.horizon;
        let x = Self::features(result);
        let dim_in = 2 * f;
        let mut y = vec![0.0; f];
        for t in 0..f {
            let mut s = self.bias[t];
            for j in 0..dim_in {
                s += self.w[t * dim_in + j] * x[j];
            }
            y[t] = s + result.query_last;
        }
        y
    }

    /// Fit multivariate ridge: each horizon step independent.
    /// `xs`: n × 2F, `ys`: n × F (offset-space targets: actual - query_last).
    pub fn fit_ridge(&mut self, xs: &[Vec<f64>], ys: &[Vec<f64>], lambda: f64) {
        let f = self.horizon;
        let dim_in = 2 * f;
        if xs.is_empty() || xs.len() != ys.len() {
            return;
        }
        let n = xs.len();
        // For each output t, solve (XᵀX + λI) β = Xᵀy_t with intercept via
        // augmented column of ones.
        let d = dim_in + 1;
        for t in 0..f {
            let mut xtx = vec![0.0; d * d];
            let mut xty = vec![0.0; d];
            for i in 0..n {
                let xi = &xs[i];
                let yi = ys[i][t];
                // features with bias
                for a in 0..dim_in {
                    xty[a] += xi[a] * yi;
                    for b in 0..dim_in {
                        xtx[a * d + b] += xi[a] * xi[b];
                    }
                    xtx[a * d + dim_in] += xi[a];
                    xtx[dim_in * d + a] += xi[a];
                }
                xtx[dim_in * d + dim_in] += 1.0;
                xty[dim_in] += yi;
            }
            for a in 0..dim_in {
                xtx[a * d + a] += lambda;
            }
            // small ridge on bias too
            xtx[dim_in * d + dim_in] += lambda * 1e-3;

            if let Some(beta) = solve_linear_system(&xtx, &xty, d) {
                for j in 0..dim_in {
                    self.w[t * dim_in + j] = beta[j];
                }
                self.bias[t] = beta[dim_in];
            }
        }
        self.fitted = true;
    }
}

/// Gaussian elimination with partial pivoting for A (d×d row-major) x = b.
fn solve_linear_system(a_in: &[f64], b_in: &[f64], d: usize) -> Option<Vec<f64>> {
    let mut a = a_in.to_vec();
    let mut b = b_in.to_vec();
    for col in 0..d {
        let mut piv = col;
        let mut best = a[col * d + col].abs();
        for r in (col + 1)..d {
            let v = a[r * d + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-14 {
            return None;
        }
        if piv != col {
            for c in 0..d {
                a.swap(col * d + c, piv * d + c);
            }
            b.swap(col, piv);
        }
        let diag = a[col * d + col];
        for r in (col + 1)..d {
            let factor = a[r * d + col] / diag;
            for c in col..d {
                a[r * d + c] -= factor * a[col * d + c];
            }
            b[r] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; d];
    for i in (0..d).rev() {
        let mut s = b[i];
        for j in (i + 1)..d {
            s -= a[i * d + j] * x[j];
        }
        let diag = a[i * d + i];
        if diag.abs() < 1e-14 {
            return None;
        }
        x[i] = s / diag;
    }
    Some(x)
}

pub fn forecast_with_head(result: &RaftRetrievalResult, head: HeadKind, ridge: Option<&RidgeHead>) -> Vec<f64> {
    match head {
        HeadKind::Identity => result
            .retrieval_agg
            .iter()
            .map(|v| v + result.query_last)
            .collect(),
        HeadKind::Ridge => {
            if let Some(r) = ridge {
                r.predict(result)
            } else {
                result
                    .retrieval_agg
                    .iter()
                    .map(|v| v + result.query_last)
                    .collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieve::{retrieve_multiperiod, RaftParams};

    #[test]
    fn identity_matches_agg_plus_offset() {
        let mut closes: Vec<f64> = (0..200).map(|i| 100.0 + (i as f64) * 0.1).collect();
        for i in 0..50 {
            closes.push(120.0 + ((i as f64) * 0.3).sin());
        }
        let params = RaftParams {
            lookback: 16,
            horizon: 4,
            periods: vec![1, 2],
            top_m: 5,
            ..Default::default()
        };
        let end = closes.len() - 4;
        let r = retrieve_multiperiod(&closes, end, &params).unwrap();
        let pred = forecast_with_head(&r, HeadKind::Identity, None);
        assert_eq!(pred.len(), 4);
        for t in 0..4 {
            assert!((pred[t] - (r.retrieval_agg[t] + r.query_last)).abs() < 1e-12);
        }
    }
}
