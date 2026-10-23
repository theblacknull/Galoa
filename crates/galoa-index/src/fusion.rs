//! SpecReTF full-ish MVP: retrieval (eq. 10) + direct + Linear fusion (eq. 11 spirit).
//!
//! Paper (arXiv:2606.19412 §3.3):
//!   Y_retrieval = softmax-weighted neighbour futures
//!   Ŷ_retrieval = Linear(Y_retrieval)     — we use **identity** projection
//!   Ŷ_direct    = Linear(Q)               — expanding **ridge** W→H (offset space)
//!   Ŷ_final     = Linear(Concat(Ŷ_retrieval, Ŷ_direct)) — expanding **ridge** 2H→H
//!
//! Walk-forward: fit only on past (query_end, features, actual) pairs — no leakage.

use crate::specretf::{
    accumulate_pred, pred_means, Acc, EvalConfig, EvalReport, SpecReTfIndex, SpecReTfIndexError,
};
/// Ridge / fusion hyperparams for walk-forward training.
#[derive(Clone, Debug)]
pub struct FusionTrainConfig {
    pub ridge_lambda: f64,
    pub ridge_retrain_every: usize,
    pub ridge_min_train: usize,
}

impl Default for FusionTrainConfig {
    fn default() -> Self {
        Self {
            ridge_lambda: 1.0,
            ridge_retrain_every: 64,
            ridge_min_train: 64,
        }
    }
}

/// Closed-form ridge: `dim_in → horizon` (each horizon step independent + bias).
#[derive(Clone, Debug)]
pub struct RidgeMap {
    pub w: Vec<f64>,
    pub bias: Vec<f64>,
    pub dim_in: usize,
    pub horizon: usize,
    pub fitted: bool,
}

impl RidgeMap {
    pub fn new(dim_in: usize, horizon: usize) -> Self {
        // Near-zero init; until fitted, predict returns zeros (naive in offset space).
        Self {
            w: vec![0.0; horizon * dim_in],
            bias: vec![0.0; horizon],
            dim_in,
            horizon,
            fitted: false,
        }
    }

    /// Init fusion so Ŷ ≈ Ŷ_retrieval (first half of concat) before any fit.
    pub fn identity_on_first_half(horizon: usize) -> Self {
        let dim_in = 2 * horizon;
        let mut w = vec![0.0; horizon * dim_in];
        for t in 0..horizon {
            w[t * dim_in + t] = 1.0;
        }
        Self {
            w,
            bias: vec![0.0; horizon],
            dim_in,
            horizon,
            fitted: false,
        }
    }

    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; self.horizon];
        let n = self.dim_in.min(x.len());
        for t in 0..self.horizon {
            let mut s = self.bias[t];
            for j in 0..n {
                s += self.w[t * self.dim_in + j] * x[j];
            }
            y[t] = s;
        }
        y
    }

    /// Fit: `xs` n×dim_in, `ys` n×horizon (offset-space targets).
    pub fn fit_ridge(&mut self, xs: &[Vec<f64>], ys: &[Vec<f64>], lambda: f64) {
        let h = self.horizon;
        let dim_in = self.dim_in;
        if xs.is_empty() || xs.len() != ys.len() {
            return;
        }
        let n = xs.len();
        let d = dim_in + 1;
        for t in 0..h {
            let mut xtx = vec![0.0; d * d];
            let mut xty = vec![0.0; d];
            for i in 0..n {
                let xi = &xs[i];
                let yi = ys[i][t];
                for a in 0..dim_in {
                    let xa = if a < xi.len() { xi[a] } else { 0.0 };
                    xty[a] += xa * yi;
                    for b in 0..dim_in {
                        let xb = if b < xi.len() { xi[b] } else { 0.0 };
                        xtx[a * d + b] += xa * xb;
                    }
                    xtx[a * d + dim_in] += xa;
                    xtx[dim_in * d + a] += xa;
                }
                xtx[dim_in * d + dim_in] += 1.0;
                xty[dim_in] += yi;
            }
            for a in 0..dim_in {
                xtx[a * d + a] += lambda;
            }
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

/// Offset lookback features for Ŷ_direct (paper Linear `L→H`).
///
/// Uses the **last `horizon` bars** of the query (truncation of `L→H`, same spirit as
/// RAFT's `f(ˆx)` tail). Full-`L` ridge on ~6k noisy M1 bars is ill-conditioned and
/// collapses; document as shallow stand-in, not SGD-trained TimesNet/Linear.
fn query_direct_features(closes: &[f64], query_end: usize, window: usize, horizon: usize) -> Vec<f64> {
    let q_last = closes[query_end - 1];
    let start = query_end - window;
    let win = &closes[start..query_end];
    let take = horizon.min(win.len());
    win[win.len() - take..]
        .iter()
        .map(|c| c - q_last)
        .collect()
}

fn to_absolute(offset: &[f64], q_last: f64) -> Vec<f64> {
    offset.iter().map(|v| v + q_last).collect()
}

/// Walk-forward SpecReTF with direct + fusion heads (eq. 11 spirit).
///
/// Methods reported: `naive_last`, `retrieval_only`, `direct_only`, `fusion_full`.
pub fn run_specretf_fusion_eval(
    closes: &[f64],
    cfg: &EvalConfig,
    train: &FusionTrainConfig,
) -> Result<EvalReport, SpecReTfIndexError> {
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

    let h = cfg.horizon;

    let mut direct = RidgeMap::new(h, h); // last-H of Q → H (truncated Linear L→H)
    let mut fusion = RidgeMap::identity_on_first_half(h);

    let mut train_direct_x: Vec<Vec<f64>> = Vec::new();
    let mut train_direct_y: Vec<Vec<f64>> = Vec::new();
    let mut train_fusion_x: Vec<Vec<f64>> = Vec::new();
    let mut train_fusion_y: Vec<Vec<f64>> = Vec::new();

    let mut acc_naive = Acc::default();
    let mut acc_ret = Acc::default();
    let mut acc_dir = Acc::default();
    let mut acc_fus = Acc::default();

    let first_end = cfg.window + cfg.min_history_windows;
    let last_end = closes.len() - cfg.horizon;
    let mut end = first_end;
    let mut query_i = 0usize;

    while end <= last_end {
        let q_last = closes[end - 1];
        let actual: Vec<f64> = closes[end..end + h].to_vec();
        let y_off: Vec<f64> = actual.iter().map(|a| a - q_last).collect();

        // Retrain on past only (before this query's prediction).
        if query_i > 0
            && query_i % train.ridge_retrain_every == 0
            && train_direct_x.len() >= train.ridge_min_train
        {
            direct.fit_ridge(&train_direct_x, &train_direct_y, train.ridge_lambda);
            fusion.fit_ridge(&train_fusion_x, &train_fusion_y, train.ridge_lambda);
        }

        let spec = index.forecast_at(end, cfg.k)?;
        let y_ret_abs = pred_means(&spec.forecast);
        let y_ret_off: Vec<f64> = y_ret_abs.iter().map(|v| v - q_last).collect();

        let q_feat = query_direct_features(closes, end, cfg.window, h);
        let y_dir_off = direct.predict(&q_feat);
        let y_dir_abs = to_absolute(&y_dir_off, q_last);

        let mut fus_feat = Vec::with_capacity(2 * h);
        fus_feat.extend_from_slice(&y_ret_off);
        fus_feat.extend_from_slice(&y_dir_off);
        let y_fus_off = fusion.predict(&fus_feat);
        let y_fus_abs = to_absolute(&y_fus_off, q_last);

        let naive = vec![q_last; h];
        accumulate_pred(&mut acc_naive, q_last, &actual, &naive, None);
        accumulate_pred(
            &mut acc_ret,
            q_last,
            &actual,
            &y_ret_abs,
            Some(&spec.forecast),
        );
        accumulate_pred(&mut acc_dir, q_last, &actual, &y_dir_abs, None);
        accumulate_pred(&mut acc_fus, q_last, &actual, &y_fus_abs, None);

        // Store for *future* fits (this query's outcome known after pred — no leakage).
        train_direct_x.push(q_feat);
        train_direct_y.push(y_off.clone());
        train_fusion_x.push(fus_feat);
        train_fusion_y.push(y_off);

        query_i += 1;
        end += cfg.stride;
    }

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
            acc_naive.finish("naive_last", false),
            acc_ret.finish("retrieval_only", true),
            acc_dir.finish("direct_only", false),
            acc_fus.finish("fusion_full", false),
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::specretf::{EvalConfig, SignalMode};
    use galoa_core::SpecReTfParams;

    #[test]
    fn fusion_eval_smoke() {
        let mut closes: Vec<f64> = Vec::new();
        let mut px = 2000.0;
        for i in 0..500 {
            px *= 1.0 + 0.0002 * ((i as f64) * 0.17).sin();
            closes.push(px);
        }
        let cfg = EvalConfig {
            window: 32,
            horizon: 4,
            k: 5,
            stride: 4,
            min_history_windows: 40,
            signal: SignalMode::RawClose,
            params: SpecReTfParams {
                n_fft: 16,
                hop: 8,
                ..Default::default()
            },
        };
        let train = FusionTrainConfig {
            ridge_min_train: 16,
            ridge_retrain_every: 8,
            ..Default::default()
        };
        let report = run_specretf_fusion_eval(&closes, &cfg, &train).unwrap();
        assert_eq!(report.methods.len(), 4);
        assert!(report.methods.iter().all(|m| m.n_queries > 0));
        assert!(report.methods.iter().any(|m| m.name == "fusion_full"));
    }

    #[test]
    fn ridge_identity_init_preserves_first_half() {
        let h = 3;
        let m = RidgeMap::identity_on_first_half(h);
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let y = m.predict(&x);
        assert!((y[0] - 1.0).abs() < 1e-12);
        assert!((y[1] - 2.0).abs() < 1e-12);
        assert!((y[2] - 3.0).abs() < 1e-12);
    }
}
