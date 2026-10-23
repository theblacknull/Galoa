//! Demoted utility — **not** used by the spectral matching path.
//! Kept for experiments / legacy helpers only.

use thiserror::Error;

/// Mean and std used to z-normalize a window (needed to lift forecasts back).
#[derive(Clone, Debug, PartialEq)]
pub struct ZNormParams {
    pub mean: f64,
    pub std: f64,
}

#[derive(Debug, Error)]
pub enum ZNormError {
    #[error("window is empty")]
    Empty,
    #[error("window std is zero (flat series)")]
    ZeroStd,
}

/// Z-normalize `window` in place conceptually: returns normalized values + params.
/// Uses sample std with Bessel's correction (n-1) when n > 1.
pub fn z_normalize(window: &[f64]) -> Result<(Vec<f64>, ZNormParams), ZNormError> {
    if window.is_empty() {
        return Err(ZNormError::Empty);
    }
    let n = window.len() as f64;
    let mean = window.iter().sum::<f64>() / n;
    let var = if window.len() == 1 {
        0.0
    } else {
        window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
    };
    let std = var.sqrt();
    if std == 0.0 || !std.is_finite() {
        return Err(ZNormError::ZeroStd);
    }
    let normalized: Vec<f64> = window.iter().map(|x| (x - mean) / std).collect();
    Ok((normalized, ZNormParams { mean, std }))
}

/// Apply stored params to lift a z-space trajectory back to price space relative
/// to the query window's mean/std (shape analogue → absolute levels of query).
pub fn denormalize(z_values: &[f64], params: &ZNormParams) -> Vec<f64> {
    z_values
        .iter()
        .map(|z| z * params.std + params.mean)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn z_norm_unit_variance() {
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (z, p) = z_normalize(&w).unwrap();
        let mean_z: f64 = z.iter().sum::<f64>() / z.len() as f64;
        assert_abs_diff_eq!(mean_z, 0.0, epsilon = 1e-12);
        assert!(p.std > 0.0);
        let back = denormalize(&z, &p);
        for (a, b) in w.iter().zip(back.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-12);
        }
    }

    #[test]
    fn flat_window_errors() {
        assert!(matches!(
            z_normalize(&[3.0, 3.0, 3.0]),
            Err(ZNormError::ZeroStd)
        ));
    }
}
