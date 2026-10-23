use thiserror::Error;

#[derive(Debug, Error)]
pub enum ReturnsError {
    #[error("need at least 2 closes for log-returns, got {0}")]
    TooShort(usize),
    #[error("non-positive close at index {0}")]
    NonPositive(usize),
}

/// Log-returns `r[i] = ln(close[i+1] / close[i])` for `i in 0..n-1`.
pub fn log_returns(closes: &[f64]) -> Result<Vec<f64>, ReturnsError> {
    if closes.len() < 2 {
        return Err(ReturnsError::TooShort(closes.len()));
    }
    let mut out = Vec::with_capacity(closes.len() - 1);
    for i in 0..closes.len() - 1 {
        let a = closes[i];
        let b = closes[i + 1];
        if !(a > 0.0 && b > 0.0) {
            return Err(ReturnsError::NonPositive(i));
        }
        out.push((b / a).ln());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn simple_log_returns() {
        let closes = vec![100.0, 110.0, 99.0];
        let r = log_returns(&closes).unwrap();
        assert_eq!(r.len(), 2);
        assert_abs_diff_eq!(r[0], (1.1f64).ln(), epsilon = 1e-12);
        assert_abs_diff_eq!(r[1], (99.0 / 110.0f64).ln(), epsilon = 1e-12);
    }
}
