use serde::{Deserialize, Serialize};

/// One step of an ensemble forecast in price space.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ForecastPoint {
    /// Steps ahead from the query window end (1..=H).
    pub step: usize,
    pub median: f64,
    pub p10: f64,
    pub p90: f64,
}

/// Ensemble forecast over horizon H.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Forecast {
    pub horizon: usize,
    pub k_used: usize,
    pub points: Vec<ForecastPoint>,
}

/// One step with optional similarity-weighted mean (SpecReTF aggregate).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WeightedForecastPoint {
    pub step: usize,
    /// Similarity-weighted mean of neighbour closes (level-aligned).
    pub mean: f64,
    pub median: f64,
    pub p10: f64,
    pub p90: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WeightedForecast {
    pub horizon: usize,
    pub k_used: usize,
    pub points: Vec<WeightedForecastPoint>,
}

/// Build median / P10 / P90 bands from `k` analogue trajectories of length `H`
/// in **price space** (already denormalized to the query window scale).
///
/// Each row of `trajectories` is one analogue's next-H closes.
pub fn ensemble_forecast(trajectories: &[Vec<f64>]) -> Forecast {
    let k_used = trajectories.len();
    if k_used == 0 {
        return Forecast {
            horizon: 0,
            k_used: 0,
            points: vec![],
        };
    }
    let horizon = trajectories.iter().map(|t| t.len()).min().unwrap_or(0);
    let mut points = Vec::with_capacity(horizon);
    for step in 0..horizon {
        let mut vals: Vec<f64> = trajectories.iter().map(|t| t[step]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        points.push(ForecastPoint {
            step: step + 1,
            median: percentile_sorted(&vals, 0.50),
            p10: percentile_sorted(&vals, 0.10),
            p90: percentile_sorted(&vals, 0.90),
        });
    }
    Forecast {
        horizon,
        k_used,
        points,
    }
}

/// Weighted mean + median/P10/P90 from neighbour trajectories.
///
/// `weights` align with `trajectories` rows (e.g. SpecReTF similarities). Non-positive
/// weights are clamped to a tiny epsilon so every neighbour still contributes.
pub fn weighted_ensemble_forecast(trajectories: &[Vec<f64>], weights: &[f64]) -> WeightedForecast {
    let k_used = trajectories.len().min(weights.len());
    if k_used == 0 {
        return WeightedForecast {
            horizon: 0,
            k_used: 0,
            points: vec![],
        };
    }
    let horizon = trajectories[..k_used]
        .iter()
        .map(|t| t.len())
        .min()
        .unwrap_or(0);
    let wsum: f64 = weights[..k_used].iter().map(|w| w.max(1e-12)).sum();
    let mut points = Vec::with_capacity(horizon);
    for step in 0..horizon {
        let mut vals: Vec<f64> = trajectories[..k_used].iter().map(|t| t[step]).collect();
        let mean = if wsum > 0.0 {
            trajectories[..k_used]
                .iter()
                .zip(weights[..k_used].iter())
                .map(|(t, w)| t[step] * w.max(1e-12))
                .sum::<f64>()
                / wsum
        } else {
            vals.iter().sum::<f64>() / vals.len() as f64
        };
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        points.push(WeightedForecastPoint {
            step: step + 1,
            mean,
            median: percentile_sorted(&vals, 0.50),
            p10: percentile_sorted(&vals, 0.10),
            p90: percentile_sorted(&vals, 0.90),
        });
    }
    WeightedForecast {
        horizon,
        k_used,
        points,
    }
}

/// Linear-interpolated percentile on a sorted ascending slice.
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    debug_assert!((0.0..=1.0).contains(&p));
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let n = sorted.len() as f64;
    let idx = p * (n - 1.0);
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let w = idx - lo as f64;
        sorted[lo] * (1.0 - w) + sorted[hi] * w
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn median_and_bands() {
        let traj = vec![
            vec![1.0, 10.0],
            vec![2.0, 20.0],
            vec![3.0, 30.0],
            vec![4.0, 40.0],
            vec![5.0, 50.0],
        ];
        let f = ensemble_forecast(&traj);
        assert_eq!(f.horizon, 2);
        assert_eq!(f.k_used, 5);
        assert_abs_diff_eq!(f.points[0].median, 3.0, epsilon = 1e-9);
        assert!(f.points[0].p10 < f.points[0].median);
        assert!(f.points[0].p90 > f.points[0].median);
    }
}
