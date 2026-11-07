//! Selective FIRE gates — abstain by default unless proximity, support, and
//! outcome dispersion all pass.

use serde::{Deserialize, Serialize};

/// Gate thresholds (configurable; starters tuned for L2-normalized STFT features
/// on Gold M1 log-returns).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GateConfig {
    /// Max L2 distance in feature space for a neighbour to count as support.
    pub d_max: f64,
    /// Min number of neighbours within `d_max` required to FIRE.
    pub n_min: usize,
    /// Max std-dev of H-step log-returns across supporting analogues.
    pub sigma_max: f64,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            // Unit-norm STFT vectors: chordal L2 in ~[0, 2].
            // On XAUUSD M1 5d, ~0.48 yields low-single-digit FIRE coverage (selective).
            d_max: 0.48,
            n_min: 3,
            // ~ few bps of H-step return dispersion on XAUUSD M1 analogues.
            sigma_max: 0.0025,
        }
    }
}

/// Why a query abstained (if it did).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AbstainReason {
    /// Fewer than `n_min` neighbours within `d_max`.
    LowSupport,
    /// Outcome dispersion (σ of H-step log-return) above `sigma_max`.
    HighDispersion,
    /// No usable analogues after overlap exclusion / flat windows.
    NoAnalogues,
}

/// Match decision after gates.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchDecision {
    Fire,
    Abstain,
}

/// Evaluate selective gates given distances (ascending) and per-neighbour
/// H-step log-returns (aligned with the same neighbour order).
///
/// Support = neighbours with `distance <= d_max`. Dispersion = sample std of
/// those neighbours' H-step log-returns. FIRE only if support ≥ `n_min` and
/// σ ≤ `sigma_max`.
pub fn evaluate_gates(
    distances: &[f64],
    h_step_log_returns: &[f64],
    cfg: &GateConfig,
) -> (MatchDecision, Option<AbstainReason>, usize, f64) {
    debug_assert_eq!(distances.len(), h_step_log_returns.len());

    if distances.is_empty() {
        return (
            MatchDecision::Abstain,
            Some(AbstainReason::NoAnalogues),
            0,
            f64::NAN,
        );
    }
    let mut support_rets = Vec::new();
    for (d, r) in distances.iter().zip(h_step_log_returns.iter()) {
        if *d <= cfg.d_max {
            support_rets.push(*r);
        }
    }
    let support = support_rets.len();
    if support < cfg.n_min {
        let sigma = sample_std(&support_rets);
        return (
            MatchDecision::Abstain,
            Some(AbstainReason::LowSupport),
            support,
            sigma,
        );
    }
    let sigma = sample_std(&support_rets);
    if !sigma.is_finite() || sigma > cfg.sigma_max {
        return (
            MatchDecision::Abstain,
            Some(AbstainReason::HighDispersion),
            support,
            sigma,
        );
    }
    (MatchDecision::Fire, None, support, sigma)
}

fn sample_std(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    if xs.len() < 2 {
        return 0.0;
    }
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    var.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fire_when_tight_and_stable() {
        let cfg = GateConfig {
            d_max: 0.4,
            n_min: 3,
            sigma_max: 0.01,
        };
        let dists = vec![0.1, 0.2, 0.3, 0.5];
        let rets = vec![0.001, 0.0012, 0.0008, 0.05];
        let (dec, reason, support, sigma) = evaluate_gates(&dists, &rets, &cfg);
        assert_eq!(dec, MatchDecision::Fire);
        assert!(reason.is_none());
        assert_eq!(support, 3);
        assert!(sigma < cfg.sigma_max);
    }

    #[test]
    fn abstain_low_support() {
        let cfg = GateConfig {
            d_max: 0.15,
            n_min: 3,
            sigma_max: 0.01,
        };
        let dists = vec![0.1, 0.2, 0.3];
        let rets = vec![0.001, 0.001, 0.001];
        let (dec, reason, support, _) = evaluate_gates(&dists, &rets, &cfg);
        assert_eq!(dec, MatchDecision::Abstain);
        assert_eq!(reason, Some(AbstainReason::LowSupport));
        assert_eq!(support, 1);
    }

    #[test]
    fn abstain_high_dispersion() {
        let cfg = GateConfig {
            d_max: 0.5,
            n_min: 2,
            sigma_max: 0.001,
        };
        let dists = vec![0.1, 0.2];
        let rets = vec![-0.01, 0.01];
        let (dec, reason, _, _) = evaluate_gates(&dists, &rets, &cfg);
        assert_eq!(dec, MatchDecision::Abstain);
        assert_eq!(reason, Some(AbstainReason::HighDispersion));
    }
}
