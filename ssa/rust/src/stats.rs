//! Statistical tools (port of `stat_tools.py`).

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::error::{Result, SsaError};

/// Model-free bootstrap prediction intervals (Hyndman & Athanasopoulos,
/// FPP3 section 5.5), port of `bootstrap_prediction_intervals`.
///
/// Draws `n_sims` paths `forecasts[h] + e`, `e` sampled with replacement from
/// `residuals`, and returns the `alpha/2` and `1 - alpha/2` percentiles of
/// every horizon (numpy "linear" percentile).  `seed = None` uses OS entropy
/// (Python uses the global numpy RNG), so draws differ from Python's.
pub fn bootstrap_prediction_intervals(
    residuals: &[f64],
    forecasts: &[f64],
    alpha: f64,
    n_sims: usize,
    seed: Option<u64>,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if residuals.is_empty() {
        return Err(SsaError::InvalidArgument("residuals must be non-empty".into()));
    }
    if n_sims == 0 {
        return Err(SsaError::InvalidArgument("N_sims must be >= 1".into()));
    }
    if !(0.0..=1.0).contains(&alpha) {
        return Err(SsaError::InvalidArgument("alpha must lie in [0, 1]".into()));
    }
    let h = forecasts.len();
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    // Same layout as Python: sims[sim, horizon], draws in row-major order.
    let mut by_horizon = vec![vec![0.0; n_sims]; h];
    for sim in 0..n_sims {
        for (hz, col) in by_horizon.iter_mut().enumerate() {
            let e = residuals[rng.gen_range(0..residuals.len())];
            col[sim] = forecasts[hz] + e;
        }
    }
    let q_lo = alpha / 2.0 * 100.0;
    let q_hi = (1.0 - alpha / 2.0) * 100.0;
    let res: Vec<(f64, f64)> = by_horizon
        .into_par_iter()
        .map(|mut col| {
            col.sort_by(f64::total_cmp);
            (percentile_sorted(&col, q_lo), percentile_sorted(&col, q_hi))
        })
        .collect();
    Ok(res.into_iter().unzip())
}

/// numpy `percentile(..., method="linear")` on sorted data.
pub fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if sorted.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let virt = q / 100.0 * (n - 1) as f64;
    let lo = (virt.floor() as usize).min(n - 1);
    let hi = (lo + 1).min(n - 1);
    let t = virt - lo as f64;
    let (a, b) = (sorted[lo], sorted[hi]);
    // numpy `_lerp`: switches formula at t >= 0.5 for monotonicity.
    let diff = b - a;
    if t >= 0.5 { b - diff * (1.0 - t) } else { a + diff * t }
}

/// numpy `percentile(..., method="linear")` of unsorted data.
pub fn percentile(data: &[f64], q: f64) -> f64 {
    let mut v = data.to_vec();
    v.sort_by(f64::total_cmp);
    percentile_sorted(&v, q)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_matches_numpy() {
        let x = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        // np.percentile(x, [0, 2.5, 25, 50, 97.5, 100])
        let want = [1.0, 1.0, 1.75, 3.5, 8.475, 9.0];
        for (q, w) in [0.0, 2.5, 25.0, 50.0, 97.5, 100.0].iter().zip(want) {
            assert!((percentile(&x, *q) - w).abs() < 1e-12, "q={q}");
        }
    }

    #[test]
    fn bootstrap_constant_residual() {
        let (lo, hi) = bootstrap_prediction_intervals(&[0.5], &[1.0, 2.0], 0.05, 100, Some(1)).unwrap();
        assert_eq!(lo, vec![1.5, 2.5]);
        assert_eq!(hi, vec![1.5, 2.5]);
    }
}
