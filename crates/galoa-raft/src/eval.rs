//! Walk-forward always-FIRE offline eval for RAFT (same protocol spirit as SpecReTF).

use crate::head::{forecast_with_head, HeadKind, RidgeHead};
use crate::retrieve::{retrieve_multiperiod, RaftError, RaftParams};
use crate::similarity::SimilarityKind;
use galoa_core::z_normalize;
use serde::Serialize;

#[derive(Clone, Debug)]
pub struct RaftEvalConfig {
    pub params: RaftParams,
    pub stride: usize,
    pub min_history: usize,
    pub head: HeadKind,
    /// Ridge re-fit every this many queries (expanding history only).
    pub ridge_retrain_every: usize,
    pub ridge_lambda: f64,
    /// Min samples before first ridge fit.
    pub ridge_min_train: usize,
}

impl Default for RaftEvalConfig {
    fn default() -> Self {
        Self {
            params: RaftParams::default(),
            stride: 1,
            min_history: 128,
            head: HeadKind::Identity,
            ridge_retrain_every: 64,
            ridge_lambda: 1.0,
            ridge_min_train: 64,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct MethodMetrics {
    pub name: String,
    pub n_queries: usize,
    pub mse_close: f64,
    pub mae_close: f64,
    pub mse_return: f64,
    pub mae_return: f64,
    pub dir_hit_h: f64,
    pub dir_hit_step: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct RaftEvalReport {
    pub n_bars: usize,
    pub lookback: usize,
    pub horizon: usize,
    pub top_m: usize,
    pub tau: f64,
    pub periods: Vec<usize>,
    pub similarity: String,
    pub head: String,
    pub stride: usize,
    pub methods: Vec<MethodMetrics>,
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
}

impl Acc {
    fn finish(self, name: &str) -> MethodMetrics {
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
        }
    }
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

fn same_direction(da: f64, dp: f64) -> bool {
    sign3(da) == sign3(dp)
}

fn accumulate(acc: &mut Acc, q_last: f64, actual: &[f64], pred: &[f64]) {
    let h = actual.len().min(pred.len());
    if h == 0 {
        return;
    }
    acc.n += 1;
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
        if same_direction(actual[i] - prev_a, pred[i] - prev_p) {
            acc.dir_step_ok += 1;
        }
        acc.dir_step_n += 1;
    }
    let hf = h as f64;
    acc.sse_close += sse_c / hf;
    acc.sae_close += sae_c / hf;
    acc.sse_ret += sse_r / hf;
    acc.sae_ret += sae_r / hf;

    if same_direction(actual[h - 1] - q_last, pred[h - 1] - q_last) {
        acc.dir_h_ok += 1;
    }
}

fn l2(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Time-domain z-norm L2 kNN baseline (same spirit as SpecReTF eval).
fn td_znorm_forecast(
    closes: &[f64],
    query_end: usize,
    window: usize,
    horizon: usize,
    k: usize,
) -> Vec<f64> {
    let q_start = query_end - window;
    let q_last = closes[query_end - 1];
    let Ok((q_z, _)) = z_normalize(&closes[q_start..query_end]) else {
        return vec![q_last; horizon];
    };
    let last_ok = q_start.saturating_sub(window + horizon);
    let mut scored: Vec<(f64, usize)> = Vec::new();
    let mut start = 0usize;
    while start <= last_ok {
        if let Ok((z, _)) = z_normalize(&closes[start..start + window]) {
            scored.push((l2(&q_z, &z), start));
        }
        start += 1;
    }
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let take = k.min(scored.len());
    if take == 0 {
        return vec![q_last; horizon];
    }
    let mut means = vec![0.0; horizon];
    for (_, s) in &scored[..take] {
        let key_last = closes[s + window - 1];
        let shift = q_last - key_last;
        for t in 0..horizon {
            means[t] += closes[s + window + t] + shift;
        }
    }
    for m in &mut means {
        *m /= take as f64;
    }
    means
}

/// Run walk-forward RAFT + naive + td_znorm baselines.
pub fn run_raft_eval(closes: &[f64], cfg: &RaftEvalConfig) -> Result<RaftEvalReport, RaftError> {
    let l = cfg.params.lookback;
    let h = cfg.params.horizon;
    let k = cfg.params.top_m;
    if l == 0 || h == 0 || k == 0 || cfg.stride == 0 {
        return Err(RaftError::InvalidParams);
    }
    let need = l + h + cfg.min_history;
    if closes.len() < need {
        return Err(RaftError::TooShort {
            got: closes.len(),
            lookback: l,
            horizon: h,
        });
    }

    let mut acc_raft = Acc::default();
    let mut acc_naive = Acc::default();
    let mut acc_td = Acc::default();

    let mut ridge = RidgeHead::new(h);
    let mut train_x: Vec<Vec<f64>> = Vec::new();
    let mut train_y: Vec<Vec<f64>> = Vec::new();
    let mut query_i = 0usize;

    let first_end = l + cfg.min_history;
    let last_end = closes.len() - h;
    let mut end = first_end;
    while end <= last_end {
        let q_last = closes[end - 1];
        let actual: Vec<f64> = closes[end..end + h].to_vec();

        let result = retrieve_multiperiod(closes, end, &cfg.params)?;

        if cfg.head == HeadKind::Ridge {
            // Fit on past (query_end, features, offset-target) collected earlier only.
            if query_i > 0
                && query_i % cfg.ridge_retrain_every == 0
                && train_x.len() >= cfg.ridge_min_train
            {
                ridge.fit_ridge(&train_x, &train_y, cfg.ridge_lambda);
            }
        }

        let pred = forecast_with_head(&result, cfg.head, Some(&ridge));
        accumulate(&mut acc_raft, q_last, &actual, &pred);

        // Store training pair for *future* ridge fits (no leakage into this pred).
        if cfg.head == HeadKind::Ridge {
            let feat = RidgeHead::features(&result);
            let y_off: Vec<f64> = actual.iter().map(|a| a - q_last).collect();
            train_x.push(feat);
            train_y.push(y_off);
        }

        let naive = vec![q_last; h];
        accumulate(&mut acc_naive, q_last, &actual, &naive);

        let td = td_znorm_forecast(closes, end, l, h, k);
        accumulate(&mut acc_td, q_last, &actual, &td);

        query_i += 1;
        end += cfg.stride;
    }

    let sim_name = match cfg.params.similarity {
        SimilarityKind::Pearson => "pearson",
        SimilarityKind::Euclidean => "euclidean",
    };
    let head_name = match cfg.head {
        HeadKind::Identity => "identity",
        HeadKind::Ridge => "ridge",
    };
    let raft_name = format!("raft_{head_name}");

    Ok(RaftEvalReport {
        n_bars: closes.len(),
        lookback: l,
        horizon: h,
        top_m: k,
        tau: cfg.params.tau,
        periods: cfg.params.periods.clone(),
        similarity: sim_name.into(),
        head: head_name.into(),
        stride: cfg.stride,
        methods: vec![
            acc_raft.finish(&raft_name),
            acc_naive.finish("naive_last"),
            acc_td.finish("td_znorm_l2"),
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_smoke() {
        let mut closes = Vec::new();
        let mut px = 2000.0;
        for i in 0..500 {
            px *= 1.0 + 0.0003 * ((i as f64) * 0.19).sin();
            closes.push(px);
        }
        let cfg = RaftEvalConfig {
            params: RaftParams {
                lookback: 32,
                horizon: 4,
                top_m: 5,
                periods: vec![1, 2],
                ..Default::default()
            },
            stride: 4,
            min_history: 40,
            ..Default::default()
        };
        let report = run_raft_eval(&closes, &cfg).unwrap();
        assert_eq!(report.methods.len(), 3);
        assert!(report.methods[0].n_queries > 0);
    }
}
