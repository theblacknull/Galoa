//! SpecReTF-style spectral similarity on log-return windows.
//!
//! Pipeline (retrieval core only — no neural fusion):
//! 1. STFT (Hann, ~50% overlap) → per-frame amplitude + phase
//! 2. Amplitude → probability distribution → Jensen–Shannon vs candidate
//! 3. Phase → mean cosine of Δφ
//! 4. Combine into a per-frame score; optional EMA recency across frames
//!
//! Defaults for `W=64` closes → 63 returns: `n_fft=16`, `hop=8` (50% overlap).

use rustfft::{num_complex::Complex, FftPlanner};
use thiserror::Error;

/// SpecReTF-default FFT size (small frames, paper-like {8,16,32}).
pub const SPECRETF_N_FFT: usize = 16;
/// 50% overlap hop for `SPECRETF_N_FFT`.
pub const SPECRETF_HOP: usize = 8;
/// Default EMA recency α (recent frames weighted more).
pub const SPECRETF_EMA_ALPHA: f64 = 0.1;
/// Relative weight on amplitude similarity vs phase (amp dominates per SpecReTF ablation).
pub const SPECRETF_AMP_WEIGHT: f64 = 0.7;
pub const SPECRETF_PHASE_WEIGHT: f64 = 0.3;

const LN2: f64 = std::f64::consts::LN_2;
const EPS: f64 = 1e-12;

#[derive(Clone, Debug, PartialEq)]
pub struct SpecReTfParams {
    pub n_fft: usize,
    pub hop: usize,
    /// EMA α ∈ (0,1]; higher → more weight on recent frames. `0` = uniform mean.
    pub ema_alpha: f64,
    pub amp_weight: f64,
    pub phase_weight: f64,
}

impl Default for SpecReTfParams {
    fn default() -> Self {
        Self {
            n_fft: SPECRETF_N_FFT,
            hop: SPECRETF_HOP,
            ema_alpha: SPECRETF_EMA_ALPHA,
            amp_weight: SPECRETF_AMP_WEIGHT,
            phase_weight: SPECRETF_PHASE_WEIGHT,
        }
    }
}

impl SpecReTfParams {
    pub fn bins_per_frame(&self) -> usize {
        self.n_fft / 2 + 1
    }

    pub fn num_frames(&self, signal_len: usize) -> usize {
        if len_ok(signal_len, self.n_fft, self.hop) {
            1 + (signal_len - self.n_fft) / self.hop
        } else {
            0
        }
    }
}

fn len_ok(len: usize, n_fft: usize, hop: usize) -> bool {
    n_fft > 0 && hop > 0 && len >= n_fft
}

#[derive(Debug, Error)]
pub enum SpecReTfError {
    #[error("n_fft and hop must be > 0")]
    InvalidParams,
    #[error("signal shorter than n_fft ({got} < {n_fft})")]
    TooShort { got: usize, n_fft: usize },
    #[error("spectral descriptor frame count mismatch")]
    FrameMismatch,
}

/// One STFT frame: amplitude as a probability distribution + phase radians.
#[derive(Clone, Debug)]
pub struct SpectralFrame {
    /// Onesided magnitude bins normalized to sum ≈ 1.
    pub amp_dist: Vec<f64>,
    /// Onesided phase (atan2) per bin.
    pub phase: Vec<f64>,
}

/// Precomputed SpecReTF descriptor for one window of log-returns.
#[derive(Clone, Debug)]
pub struct SpectralDescriptor {
    pub frames: Vec<SpectralFrame>,
}

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

/// Build a SpecReTF spectral descriptor from a log-return signal.
pub fn spectral_descriptor(
    signal: &[f64],
    params: &SpecReTfParams,
) -> Result<SpectralDescriptor, SpecReTfError> {
    if params.n_fft == 0 || params.hop == 0 {
        return Err(SpecReTfError::InvalidParams);
    }
    if signal.len() < params.n_fft {
        return Err(SpecReTfError::TooShort {
            got: signal.len(),
            n_fft: params.n_fft,
        });
    }

    let n_frames = params.num_frames(signal.len());
    let bins = params.bins_per_frame();
    let win = hann(params.n_fft);

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(params.n_fft);
    let mut buf = vec![Complex::new(0.0, 0.0); params.n_fft];
    let mut frames = Vec::with_capacity(n_frames);

    for f in 0..n_frames {
        let start = f * params.hop;
        for i in 0..params.n_fft {
            buf[i] = Complex::new(signal[start + i] * win[i], 0.0);
        }
        fft.process(&mut buf);

        let mut amp = Vec::with_capacity(bins);
        let mut phase = Vec::with_capacity(bins);
        for b in 0..bins {
            amp.push(buf[b].norm());
            phase.push(buf[b].arg());
        }
        normalize_distribution(&mut amp);
        frames.push(SpectralFrame {
            amp_dist: amp,
            phase,
        });
    }

    Ok(SpectralDescriptor { frames })
}

fn normalize_distribution(v: &mut [f64]) {
    let sum: f64 = v.iter().sum();
    if sum > EPS && sum.is_finite() {
        for x in v.iter_mut() {
            *x /= sum;
        }
    } else {
        let u = 1.0 / v.len().max(1) as f64;
        for x in v.iter_mut() {
            *x = u;
        }
    }
}

/// Jensen–Shannon divergence (natural log). Range ≈ [0, ln 2].
pub fn js_divergence(p: &[f64], q: &[f64]) -> f64 {
    debug_assert_eq!(p.len(), q.len());
    let n = p.len().min(q.len());
    let mut js = 0.0;
    for i in 0..n {
        let pi = p[i].max(EPS);
        let qi = q[i].max(EPS);
        let m = 0.5 * (pi + qi);
        js += 0.5 * pi * (pi / m).ln() + 0.5 * qi * (qi / m).ln();
    }
    js.max(0.0)
}

/// Mean cosine of phase difference: `(1/B) Σ cos(φ_p − φ_q)` ∈ [-1, 1].
pub fn phase_cosine(p: &[f64], q: &[f64]) -> f64 {
    let n = p.len().min(q.len());
    if n == 0 {
        return 0.0;
    }
    let s: f64 = (0..n).map(|i| (p[i] - q[i]).cos()).sum();
    s / n as f64
}

/// Per-frame similarity in roughly [0, 1] (amp JS + phase cosine).
pub fn frame_similarity(a: &SpectralFrame, b: &SpectralFrame, params: &SpecReTfParams) -> f64 {
    let js = js_divergence(&a.amp_dist, &b.amp_dist);
    // Map JS ∈ [0, ln2] → similarity ∈ [0, 1]
    let amp_sim = (1.0 - (js / LN2).clamp(0.0, 1.0)).clamp(0.0, 1.0);
    let phase_sim = ((phase_cosine(&a.phase, &b.phase) + 1.0) * 0.5).clamp(0.0, 1.0);
    let w_sum = params.amp_weight + params.phase_weight;
    if w_sum <= 0.0 {
        return 0.5 * amp_sim + 0.5 * phase_sim;
    }
    (params.amp_weight * amp_sim + params.phase_weight * phase_sim) / w_sum
}

/// SpecReTF window similarity with optional EMA recency (recent frames matter more).
///
/// With `ema_alpha == 0`: uniform mean of frame similarities.
/// Else: normalized exponential weights `α(1−α)^{F−1−f}` from oldest (f=0) to newest.
pub fn spectral_similarity(
    query: &SpectralDescriptor,
    cand: &SpectralDescriptor,
    params: &SpecReTfParams,
) -> Result<f64, SpecReTfError> {
    let n = query.frames.len().min(cand.frames.len());
    if n == 0 {
        return Err(SpecReTfError::FrameMismatch);
    }
    if query.frames.len() != cand.frames.len() {
        // Still compare overlapping prefix; callers should use same STFT geometry.
    }

    let mut sims = Vec::with_capacity(n);
    for i in 0..n {
        sims.push(frame_similarity(&query.frames[i], &cand.frames[i], params));
    }

    let alpha = params.ema_alpha;
    if alpha <= 0.0 || !alpha.is_finite() {
        let mean = sims.iter().sum::<f64>() / n as f64;
        return Ok(mean);
    }
    // EMA-style recency: newest frame gets highest weight.
    let one_m = (1.0 - alpha).clamp(0.0, 1.0);
    let mut weight_sum = 0.0;
    let mut score = 0.0;
    for (f, sim) in sims.iter().enumerate() {
        // f=0 oldest → weight α * (1-α)^{n-1}; f=n-1 newest → α
        let age_from_end = (n - 1 - f) as i32;
        let w = alpha * one_m.powi(age_from_end);
        score += w * sim;
        weight_sum += w;
    }
    if weight_sum > EPS {
        Ok(score / weight_sum)
    } else {
        Ok(sims.iter().sum::<f64>() / n as f64)
    }
}

/// Distance for ranking: `1 − similarity` (lower is better).
pub fn spectral_distance(
    query: &SpectralDescriptor,
    cand: &SpectralDescriptor,
    params: &SpecReTfParams,
) -> Result<f64, SpecReTfError> {
    Ok(1.0 - spectral_similarity(query, cand, params)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn identical_descriptors_near_one() {
        let params = SpecReTfParams::default();
        let signal: Vec<f64> = (0..63)
            .map(|i| ((i as f64) * 0.35).sin() * 0.001 + ((i as f64) * 0.1).cos() * 0.0005)
            .collect();
        let d = spectral_descriptor(&signal, &params).unwrap();
        let sim = spectral_similarity(&d, &d, &params).unwrap();
        assert!(sim > 0.99, "self-sim={sim}");
        assert!(spectral_distance(&d, &d, &params).unwrap() < 0.01);
    }

    #[test]
    fn js_zero_on_same() {
        let p = vec![0.1, 0.2, 0.3, 0.4];
        assert_abs_diff_eq!(js_divergence(&p, &p), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn phase_cosine_aligned() {
        let p = vec![0.0, 1.0, -1.0];
        assert_abs_diff_eq!(phase_cosine(&p, &p), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn frame_count_50pct_overlap() {
        let params = SpecReTfParams::default();
        // 63 returns, n_fft=16 hop=8 → 1 + (63-16)/8 = 6
        assert_eq!(params.num_frames(63), 6);
        let signal = vec![0.001; 63];
        let d = spectral_descriptor(&signal, &params).unwrap();
        assert_eq!(d.frames.len(), 6);
        assert_eq!(d.frames[0].amp_dist.len(), 9);
    }

    #[test]
    fn ema_prefers_recent_when_last_matches() {
        let params = SpecReTfParams {
            ema_alpha: 0.5,
            ..Default::default()
        };
        let mut q = SpectralDescriptor {
            frames: vec![
                SpectralFrame {
                    amp_dist: vec![1.0, 0.0, 0.0],
                    phase: vec![0.0, 0.0, 0.0],
                },
                SpectralFrame {
                    amp_dist: vec![0.0, 1.0, 0.0],
                    phase: vec![0.0, 0.0, 0.0],
                },
            ],
        };
        // Normalize
        for f in &mut q.frames {
            normalize_distribution(&mut f.amp_dist);
        }
        let mut match_last = q.clone();
        match_last.frames[0].amp_dist = vec![0.0, 0.0, 1.0];
        normalize_distribution(&mut match_last.frames[0].amp_dist);

        let mut match_first = q.clone();
        match_first.frames[1].amp_dist = vec![0.0, 0.0, 1.0];
        normalize_distribution(&mut match_first.frames[1].amp_dist);

        let s_last = spectral_similarity(&q, &match_last, &params).unwrap();
        let s_first = spectral_similarity(&q, &match_first, &params).unwrap();
        assert!(
            s_last > s_first,
            "recent match should score higher: last={s_last} first={s_first}"
        );
    }
}
