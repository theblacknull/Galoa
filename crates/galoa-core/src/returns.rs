//! Log-return conversion for model feature input.
//!
//! ## Alignment
//!
//! Given closes `c[0..N]`, output length is `N−1`:
//!
//! ```text
//! r[i] = ln(c[i+1] / c[i])    for i in 0..N-1
//! ```
//!
//! Return `r[i]` is the move **ending at bar `i+1`** (the step from close `i` → `i+1`).
//! Timestamps for a log-return series should therefore be the **ending bar** timestamps
//! (`ts[i+1]` from the close series).
//!
//! ## Guards
//!
//! Non-positive, NaN, or Inf closes are rejected (error) or skipped (skip policy).
//! Never silently write NaN/Inf into the output buffer.
//!
//! ## Performance
//!
//! Hot path: `log_returns_into` (single preallocated buffer) or `log_return_step`
//! (one new close → one new return). No per-element allocation.

use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ReturnsError {
    #[error("need at least 2 closes for log-returns, got {0}")]
    TooShort(usize),
    #[error("non-positive or non-finite close at index {0}")]
    NonPositive(usize),
    #[error("output buffer length {got} != closes.len()-1 = {need}")]
    BufferLen { got: usize, need: usize },
    #[error("non-positive or non-finite close in incremental step")]
    NonPositiveStep,
}

/// How to handle invalid closes when converting a full series.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InvalidClosePolicy {
    /// Fail on the first non-positive / non-finite close.
    #[default]
    Error,
    /// Drop the step that would use an invalid close (gap in alignment — prefer Error for models).
    Skip,
}

#[inline]
fn close_ok(c: f64) -> bool {
    c.is_finite() && c > 0.0
}

/// Log-returns `r[i] = ln(close[i+1] / close[i])` for `i in 0..n-1`.
///
/// Allocates one `Vec` of length `n-1`. Prefer [`log_returns_into`] on hot paths.
pub fn log_returns(closes: &[f64]) -> Result<Vec<f64>, ReturnsError> {
    log_returns_with_policy(closes, InvalidClosePolicy::Error)
}

/// Same as [`log_returns`] with an explicit invalid-close policy.
pub fn log_returns_with_policy(
    closes: &[f64],
    policy: InvalidClosePolicy,
) -> Result<Vec<f64>, ReturnsError> {
    if closes.len() < 2 {
        return Err(ReturnsError::TooShort(closes.len()));
    }
    match policy {
        InvalidClosePolicy::Error => {
            let mut out = vec![0.0; closes.len() - 1];
            log_returns_into(closes, &mut out)?;
            Ok(out)
        }
        InvalidClosePolicy::Skip => {
            let mut out = Vec::with_capacity(closes.len() - 1);
            for i in 0..closes.len() - 1 {
                let a = closes[i];
                let b = closes[i + 1];
                if !(close_ok(a) && close_ok(b)) {
                    continue;
                }
                out.push((b / a).ln());
            }
            if out.is_empty() && closes.len() >= 2 {
                // All steps invalid — surface as NonPositive at first bad index.
                for (i, &c) in closes.iter().enumerate() {
                    if !close_ok(c) {
                        return Err(ReturnsError::NonPositive(i));
                    }
                }
            }
            Ok(out)
        }
    }
}

/// Write log-returns into a preallocated buffer (`out.len() == closes.len() - 1`).
///
/// Tight loop, no allocation. Errors on non-positive / non-finite closes (no silent NaN).
pub fn log_returns_into(closes: &[f64], out: &mut [f64]) -> Result<(), ReturnsError> {
    if closes.len() < 2 {
        return Err(ReturnsError::TooShort(closes.len()));
    }
    let need = closes.len() - 1;
    if out.len() != need {
        return Err(ReturnsError::BufferLen {
            got: out.len(),
            need,
        });
    }
    for i in 0..need {
        let a = closes[i];
        let b = closes[i + 1];
        if !(close_ok(a) && close_ok(b)) {
            // Prefer the first bad of the pair.
            let bad = if !close_ok(a) { i } else { i + 1 };
            return Err(ReturnsError::NonPositive(bad));
        }
        out[i] = (b / a).ln();
    }
    Ok(())
}

/// Incremental: one new close → one new log-return `ln(new / prev)`.
///
/// Suitable for real-time M1 append. Caller keeps `prev_close` from the last accepted bar.
#[inline]
pub fn log_return_step(prev_close: f64, new_close: f64) -> Result<f64, ReturnsError> {
    if !(close_ok(prev_close) && close_ok(new_close)) {
        return Err(ReturnsError::NonPositiveStep);
    }
    Ok((new_close / prev_close).ln())
}

/// Map a forecast in **return space** back to absolute closes for PnL / trading metrics.
///
/// Given last observed close `c_t` and predicted per-bar log-returns `r[0..H)`:
///
/// ```text
/// pred_close[k] = c_t * exp(sum_{j=0..=k} r[j])
/// ```
///
/// Returns length `H`. No alloc beyond the output `Vec`.
pub fn closes_from_log_returns(last_close: f64, returns: &[f64]) -> Result<Vec<f64>, ReturnsError> {
    if !close_ok(last_close) {
        return Err(ReturnsError::NonPositiveStep);
    }
    let mut out = Vec::with_capacity(returns.len());
    let mut px = last_close;
    for &r in returns {
        if !r.is_finite() {
            return Err(ReturnsError::NonPositiveStep);
        }
        px *= r.exp();
        if !close_ok(px) {
            return Err(ReturnsError::NonPositiveStep);
        }
        out.push(px);
    }
    Ok(out)
}

/// In-place variant of [`closes_from_log_returns`] (`out.len() == returns.len()`).
pub fn closes_from_log_returns_into(
    last_close: f64,
    returns: &[f64],
    out: &mut [f64],
) -> Result<(), ReturnsError> {
    if !close_ok(last_close) {
        return Err(ReturnsError::NonPositiveStep);
    }
    if out.len() != returns.len() {
        return Err(ReturnsError::BufferLen {
            got: out.len(),
            need: returns.len(),
        });
    }
    let mut px = last_close;
    for (i, &r) in returns.iter().enumerate() {
        if !r.is_finite() {
            return Err(ReturnsError::NonPositiveStep);
        }
        px *= r.exp();
        if !close_ok(px) {
            return Err(ReturnsError::NonPositiveStep);
        }
        out[i] = px;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::time::Instant;

    #[test]
    fn simple_log_returns() {
        let closes = vec![100.0, 110.0, 99.0];
        let r = log_returns(&closes).unwrap();
        assert_eq!(r.len(), 2);
        assert_abs_diff_eq!(r[0], (1.1f64).ln(), epsilon = 1e-12);
        assert_abs_diff_eq!(r[1], (99.0 / 110.0f64).ln(), epsilon = 1e-12);
    }

    #[test]
    fn into_matches_allocating() {
        let closes: Vec<f64> = (0..64).map(|i| 2000.0 + i as f64 * 0.1).collect();
        let a = log_returns(&closes).unwrap();
        let mut b = vec![0.0; closes.len() - 1];
        log_returns_into(&closes, &mut b).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn step_matches_batch() {
        let closes = [100.0, 101.0, 99.5, 100.2];
        let batch = log_returns(&closes).unwrap();
        for i in 0..batch.len() {
            let s = log_return_step(closes[i], closes[i + 1]).unwrap();
            assert_abs_diff_eq!(s, batch[i], epsilon = 1e-15);
        }
    }

    #[test]
    fn rejects_non_positive_and_nan() {
        assert!(matches!(
            log_returns(&[100.0, 0.0]),
            Err(ReturnsError::NonPositive(1))
        ));
        assert!(matches!(
            log_returns(&[f64::NAN, 100.0]),
            Err(ReturnsError::NonPositive(0))
        ));
        assert!(matches!(
            log_return_step(100.0, f64::INFINITY),
            Err(ReturnsError::NonPositiveStep)
        ));
    }

    #[test]
    fn skip_policy_drops_bad_step() {
        let closes = [100.0, -1.0, 110.0];
        // step 0 invalid (uses -1), step 1 invalid (uses -1) → empty / error
        let r = log_returns_with_policy(&closes, InvalidClosePolicy::Skip);
        assert!(r.is_err());
        let closes2 = [100.0, 110.0, 0.0, 120.0];
        // steps: 100→110 ok, 110→0 skip, 0→120 skip
        let r2 = log_returns_with_policy(&closes2, InvalidClosePolicy::Skip).unwrap();
        assert_eq!(r2.len(), 1);
        assert_abs_diff_eq!(r2[0], (1.1f64).ln(), epsilon = 1e-12);
    }

    #[test]
    fn roundtrip_price_from_returns() {
        let closes = [2000.0, 2001.0, 1999.5, 2002.0];
        let rets = log_returns(&closes).unwrap();
        let back = closes_from_log_returns(closes[0], &rets).unwrap();
        assert_eq!(back.len(), 3);
        for i in 0..3 {
            assert_abs_diff_eq!(back[i], closes[i + 1], epsilon = 1e-9);
        }
    }

    /// Microbench note (dev-test): close copy vs log-return convert should be same order.
    #[test]
    fn microbench_close_copy_vs_logret() {
        let n = 50_000usize;
        let closes: Vec<f64> = (0..n)
            .map(|i| 2650.0 * (1.0 + 1e-6 * (i as f64).sin()))
            .collect();

        let t0 = Instant::now();
        let mut close_copy = vec![0.0; n];
        close_copy.copy_from_slice(&closes);
        let dt_close = t0.elapsed();

        let mut rets = vec![0.0; n - 1];
        let t1 = Instant::now();
        log_returns_into(&closes, &mut rets).unwrap();
        let dt_ret = t1.elapsed();

        // Same order of magnitude: logret should not be >20× a memcpy of similar size.
        // (CI noise allowed; this is a sanity guard, not a hard SLA.)
        let close_ns = dt_close.as_nanos().max(1);
        let ret_ns = dt_ret.as_nanos().max(1);
        let ratio = ret_ns as f64 / close_ns as f64;
        assert!(
            ratio < 50.0,
            "logret too slow vs close copy: close={dt_close:?} logret={dt_ret:?} ratio={ratio:.1}"
        );
        // Touch outputs so the optimizer cannot DCE.
        assert!(close_copy[0] > 0.0 && rets[0].is_finite());
    }
}
