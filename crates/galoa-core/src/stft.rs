//! STFT magnitude features on log-return windows (spectral matching path).
//!
//! Defaults for `W=64` closes → 63 returns:
//! - `n_fft = 16`, `hop = 4`, Hann window
//! - real FFT → `(n_fft/2 + 1)` magnitude bins per frame
//! - frames concatenated → feature vector `F`, then L2-normalized
//!
//! Mel-filterbank is deferred (optional later).

use rustfft::{num_complex::Complex, FftPlanner};
use thiserror::Error;

/// Default FFT size for MVP STFT on ~W-length return windows.
pub const DEFAULT_N_FFT: usize = 16;
/// Default hop between STFT frames.
pub const DEFAULT_HOP: usize = 4;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StftParams {
    pub n_fft: usize,
    pub hop: usize,
}

impl Default for StftParams {
    fn default() -> Self {
        Self {
            n_fft: DEFAULT_N_FFT,
            hop: DEFAULT_HOP,
        }
    }
}

impl StftParams {
    /// Magnitude bins per frame (onesided real spectrum).
    pub fn bins_per_frame(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// Number of STFT frames for a signal of length `len`.
    pub fn num_frames(&self, len: usize) -> usize {
        if len < self.n_fft || self.hop == 0 {
            return 0;
        }
        1 + (len - self.n_fft) / self.hop
    }

    pub fn feature_dim(&self, signal_len: usize) -> usize {
        self.num_frames(signal_len) * self.bins_per_frame()
    }
}

#[derive(Debug, Error)]
pub enum StftError {
    #[error("n_fft and hop must be > 0")]
    InvalidParams,
    #[error("signal shorter than n_fft ({got} < {n_fft})")]
    TooShort { got: usize, n_fft: usize },
}

/// Hann window of length `n`.
fn hann(n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|i| {
            let x = std::f64::consts::PI * 2.0 * i as f64 / (n - 1) as f64;
            0.5 * (1.0 - x.cos())
        })
        .collect()
}

/// STFT magnitude feature vector for `signal` (typically log-returns).
/// Frames are Hann-windowed, FFT'd; onesided magnitudes are concatenated and
/// L2-normalized so exact kNN uses comparable chordal distances.
pub fn stft_magnitude_features(
    signal: &[f64],
    params: &StftParams,
) -> Result<Vec<f64>, StftError> {
    if params.n_fft == 0 || params.hop == 0 {
        return Err(StftError::InvalidParams);
    }
    if signal.len() < params.n_fft {
        return Err(StftError::TooShort {
            got: signal.len(),
            n_fft: params.n_fft,
        });
    }

    let n_frames = params.num_frames(signal.len());
    let bins = params.bins_per_frame();
    let win = hann(params.n_fft);

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(params.n_fft);

    let mut features = Vec::with_capacity(n_frames * bins);
    let mut buf = vec![Complex::new(0.0, 0.0); params.n_fft];

    for f in 0..n_frames {
        let start = f * params.hop;
        for i in 0..params.n_fft {
            buf[i] = Complex::new(signal[start + i] * win[i], 0.0);
        }
        fft.process(&mut buf);
        for b in 0..bins {
            features.push(buf[b].norm());
        }
    }

    l2_normalize_in_place(&mut features);
    Ok(features)
}

fn l2_normalize_in_place(v: &mut [f64]) {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 && norm.is_finite() {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn feature_dim_and_unit_norm() {
        let params = StftParams::default();
        // 63 returns from W=64 closes
        let signal: Vec<f64> = (0..63)
            .map(|i| ((i as f64) * 0.3).sin() * 0.001)
            .collect();
        let f = stft_magnitude_features(&signal, &params).unwrap();
        assert_eq!(f.len(), params.feature_dim(signal.len()));
        assert!(f.len() > 0);
        let norm: f64 = f.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn identical_signals_match() {
        let params = StftParams::default();
        let a: Vec<f64> = (0..48).map(|i| (i as f64 * 0.2).cos() * 0.002).collect();
        let fa = stft_magnitude_features(&a, &params).unwrap();
        let fb = stft_magnitude_features(&a, &params).unwrap();
        let dist: f64 = fa
            .iter()
            .zip(fb.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(dist < 1e-12);
    }

    #[test]
    fn too_short_errors() {
        let params = StftParams {
            n_fft: 16,
            hop: 4,
        };
        assert!(matches!(
            stft_magnitude_features(&[0.0; 8], &params),
            Err(StftError::TooShort { .. })
        ));
    }
}
