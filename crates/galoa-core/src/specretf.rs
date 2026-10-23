//! SpecReTF frequency-aware similarity (arXiv:2606.19412 §3.2).
//!
//! Paper equations implemented literally (retrieval core only; no neural fusion):
//!
//! 1. STFT with **Hanning** window, hop `B ≈ M/2` (50% overlap). Each frame yields
//!    an **M-dimensional** complex spectrum (Appendix G).
//! 2. Amplitude → probability: `p(f) = A(f) / Σ_f A(f)`  (eq. 2)
//! 3. `d_JS = JSD(p_Q ‖ p_X)` with **log₂** so `d_JS ∈ [0,1]`, then
//!    `s_amp = 1 − d_JS`  (eq. 3–4)
//! 4. Phase: `ΔΦ = (1/M) Σ_f (Φ_Q(f) − Φ_X(f))`, `s_phase = cos(ΔΦ)`  (eq. 5–6)
//!    — cosine of the **mean** phase difference, not mean of cosines.
//! 5. Frame score: `s^w = s_amp^w + s_phase^w`  (eq. 7) — unweighted sum
//! 6. Recency: `S = Σ_{w=1}^{W} α (1−α)^{W−w} s^w`  (eq. 8) — **not** renormalized
//!
//! Defaults (Table 5 / impl details): `M=window_size∈{8,16,32}`, α∈{0.01…0.2}, K~10.

use rustfft::{num_complex::Complex, FftPlanner};
use thiserror::Error;

/// SpecReTF default STFT frame length `M` (Table 5 `window_size`).
/// Tuned on Gold M1 5d: M=32 beat 8/16 for MSE vs naive (still no edge claim).
pub const SPECRETF_N_FFT: usize = 32;
/// 50% overlap hop `B = M/2` (impl details + Appendix G).
pub const SPECRETF_HOP: usize = 16;
/// Default recency decay α (Table 5 / Fig. 3 mid-range).
pub const SPECRETF_EMA_ALPHA: f64 = 0.1;
/// Default top-K for SpecReTF eval (Gold M1 sweep preferred 20 over 5/10).
pub const SPECRETF_K: usize = 20;

const EPS: f64 = 1e-12;
const LN2: f64 = std::f64::consts::LN_2;

#[derive(Clone, Debug, PartialEq)]
pub struct SpecReTfParams {
    /// STFT frame length `M` (paper `window_size`).
    pub n_fft: usize,
    /// Hop `B`; paper uses ≈ `M/2` (50% overlap).
    pub hop: usize,
    /// Recency decay α ∈ (0,1) in eq. (8). `≤0` → uniform mean (ablation “w/o recency”).
    pub ema_alpha: f64,
}

impl Default for SpecReTfParams {
    fn default() -> Self {
        Self {
            n_fft: SPECRETF_N_FFT,
            hop: SPECRETF_HOP,
            ema_alpha: SPECRETF_EMA_ALPHA,
        }
    }
}

impl SpecReTfParams {
    /// Full FFT length = number of frequency bins per frame (Appendix G: M-dim spectrum).
    pub fn bins_per_frame(&self) -> usize {
        self.n_fft
    }

    pub fn num_frames(&self, signal_len: usize) -> usize {
        if self.n_fft == 0 || self.hop == 0 || signal_len < self.n_fft {
            0
        } else {
            1 + (signal_len - self.n_fft) / self.hop
        }
    }
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

/// One STFT frame: amplitude as a probability distribution + phase (radians).
#[derive(Clone, Debug)]
pub struct SpectralFrame {
    /// Length-`M` amplitude bins normalized to sum ≈ 1 (eq. 2).
    pub amp_dist: Vec<f64>,
    /// Length-`M` phase `atan2(Im, Re)` per FFT bin.
    pub phase: Vec<f64>,
}

/// Precomputed SpecReTF descriptor for one length-`L` signal window.
#[derive(Clone, Debug)]
pub struct SpectralDescriptor {
    pub frames: Vec<SpectralFrame>,
}

/// Symmetric Hann (Hanning) window of length `n` — same family as paper “Hanning”.
///
/// Formula: `0.5 (1 − cos(2π i / (n−1)))`, endpoints zero. Paper §4.1 only says
/// “Hanning … 50% overlap”; does not specify periodic vs symmetric. PyTorch’s
/// default `hann_window(periodic=True)` differs slightly — treated as ambiguous,
/// not switched here without an explicit paper claim.
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

/// Build SpecReTF spectral descriptor via STFT (eq. 1–2).
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
    let m = params.n_fft;
    let win = hann(m);

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(m);
    let mut buf = vec![Complex::new(0.0, 0.0); m];
    let mut frames = Vec::with_capacity(n_frames);

    for f in 0..n_frames {
        let start = f * params.hop;
        for i in 0..m {
            buf[i] = Complex::new(signal[start + i] * win[i], 0.0);
        }
        fft.process(&mut buf);

        // Full M-bin spectrum (Appendix G), not onesided.
        let mut amp = Vec::with_capacity(m);
        let mut phase = Vec::with_capacity(m);
        for b in 0..m {
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

/// Jensen–Shannon divergence with **log₂** (so `JSD ∈ [0, 1]` and eq. 4 holds).
///
/// `JSD(P‖Q) = ½ KL₂(P‖M) + ½ KL₂(Q‖M)`, `M = ½(P+Q)`.
pub fn js_divergence(p: &[f64], q: &[f64]) -> f64 {
    debug_assert_eq!(p.len(), q.len());
    let n = p.len().min(q.len());
    let mut js = 0.0;
    for i in 0..n {
        let pi = p[i].max(EPS);
        let qi = q[i].max(EPS);
        let m = 0.5 * (pi + qi);
        // log2(x) = ln(x)/ln(2)
        js += 0.5 * pi * ((pi / m).ln() / LN2) + 0.5 * qi * ((qi / m).ln() / LN2);
    }
    js.max(0.0).min(1.0)
}

/// Phase coherence (eq. 5–6): `cos( mean_f (Φ_Q − Φ_X) )` ∈ [-1, 1].
pub fn phase_cosine(p: &[f64], q: &[f64]) -> f64 {
    let n = p.len().min(q.len());
    if n == 0 {
        return 0.0;
    }
    let mean_delta: f64 = (0..n).map(|i| p[i] - q[i]).sum::<f64>() / n as f64;
    mean_delta.cos()
}

/// Frame-level composite score (eq. 4 + 6 + 7): `s = (1 − JSD) + cos(ΔΦ)`.
pub fn frame_similarity(a: &SpectralFrame, b: &SpectralFrame, _params: &SpecReTfParams) -> f64 {
    let d_js = js_divergence(&a.amp_dist, &b.amp_dist);
    let s_amp = 1.0 - d_js;
    let s_phase = phase_cosine(&a.phase, &b.phase);
    s_amp + s_phase
}

/// Window similarity `S(Q, X_k)` (eq. 8).
///
/// `S = Σ_{w=1}^{W} α (1−α)^{W−w} s^w` with `w=1` oldest, `w=W` newest.
/// **Not** divided by Σ weights — absolute scale feeds softmax in eq. (10).
/// If `ema_alpha ≤ 0`, uses uniform mean of frame scores (ablation “w/o recency”).
pub fn spectral_similarity(
    query: &SpectralDescriptor,
    cand: &SpectralDescriptor,
    params: &SpecReTfParams,
) -> Result<f64, SpecReTfError> {
    let n = query.frames.len().min(cand.frames.len());
    if n == 0 {
        return Err(SpecReTfError::FrameMismatch);
    }

    let mut sims = Vec::with_capacity(n);
    for i in 0..n {
        sims.push(frame_similarity(&query.frames[i], &cand.frames[i], params));
    }

    let alpha = params.ema_alpha;
    if alpha <= 0.0 || !alpha.is_finite() {
        return Ok(sims.iter().sum::<f64>() / n as f64);
    }

    let one_m = (1.0 - alpha).clamp(0.0, 1.0);
    let mut score = 0.0;
    // Paper indexing: w=1..W with w=W newest → weight α (1−α)^{0}
    // Our index f=0..n-1 oldest→newest → exponent W−w = n−1−f
    for (f, sim) in sims.iter().enumerate() {
        let exponent = (n - 1 - f) as i32;
        let w = alpha * one_m.powi(exponent);
        score += w * sim;
    }
    Ok(score)
}

/// Convenience: larger similarity is better; distance = −S for min-heap style ranking.
pub fn spectral_distance(
    query: &SpectralDescriptor,
    cand: &SpectralDescriptor,
    params: &SpecReTfParams,
) -> Result<f64, SpecReTfError> {
    Ok(-spectral_similarity(query, cand, params)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn identical_self_score_matches_ema_of_two() {
        // Identical → s_amp=1, s_phase=1, s^w=2; S = 2 Σ α(1−α)^{W−w}
        let params = SpecReTfParams::default();
        let signal: Vec<f64> = (0..63)
            .map(|i| ((i as f64) * 0.35).sin() * 0.001 + ((i as f64) * 0.1).cos() * 0.0005)
            .collect();
        let d = spectral_descriptor(&signal, &params).unwrap();
        let w = d.frames.len();
        assert!(w > 0);
        let weight_sum: f64 = (0..w)
            .map(|f| {
                let exp = (w - 1 - f) as i32;
                params.ema_alpha * (1.0 - params.ema_alpha).powi(exp)
            })
            .sum();
        let expected = 2.0 * weight_sum;
        let sim = spectral_similarity(&d, &d, &params).unwrap();
        assert_abs_diff_eq!(sim, expected, epsilon = 1e-9);
    }

    #[test]
    fn js_log2_zero_on_same_and_unit_max_bound() {
        let p = vec![0.1, 0.2, 0.3, 0.4];
        assert_abs_diff_eq!(js_divergence(&p, &p), 0.0, epsilon = 1e-12);
        // Disjoint supports → JSD → 1 with log2
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let mut aa = a.clone();
        let mut bb = b.clone();
        normalize_distribution(&mut aa);
        normalize_distribution(&mut bb);
        let js = js_divergence(&aa, &bb);
        assert!(js > 0.9, "disjoint JS should be near 1, got {js}");
    }

    #[test]
    fn phase_is_cos_of_mean_delta_not_mean_of_cos() {
        // Δ = [π/2, π/2] → mean=π/2 → cos=0
        // mean(cos) would also be 0 here; use asymmetric case:
        // Δ = [0, π] → mean=π/2 → cos(mean)=0; mean(cos)=(1 + −1)/2=0 — still same
        // Δ = [π/3, π/3, 0] → mean=2π/9 → cos(2π/9) ≠ mean of cosines
        let p = vec![0.0, 0.0, 0.0];
        let q = vec![-std::f64::consts::PI / 3.0, -std::f64::consts::PI / 3.0, 0.0];
        let paper = phase_cosine(&p, &q);
        let mean_delta = (std::f64::consts::PI / 3.0 + std::f64::consts::PI / 3.0 + 0.0) / 3.0;
        assert_abs_diff_eq!(paper, mean_delta.cos(), epsilon = 1e-12);
        let mean_of_cos = ((std::f64::consts::PI / 3.0).cos()
            + (std::f64::consts::PI / 3.0).cos()
            + 0.0f64.cos())
            / 3.0;
        assert!(
            (paper - mean_of_cos).abs() > 1e-6,
            "paper cos(mean) must differ from mean(cos)"
        );
    }

    #[test]
    fn frame_count_50pct_overlap_full_m_bins() {
        // Explicit M=16 (not SPECRETF_N_FFT default) so the arithmetic stays readable.
        let params = SpecReTfParams {
            n_fft: 16,
            hop: 8,
            ema_alpha: 0.1,
        };
        // 63 samples, M=16 hop=8 → 1 + (63-16)/8 = 6
        assert_eq!(params.num_frames(63), 6);
        let signal = vec![0.001; 63];
        let d = spectral_descriptor(&signal, &params).unwrap();
        assert_eq!(d.frames.len(), 6);
        assert_eq!(d.frames[0].amp_dist.len(), 16); // full M, not onesided 9
        // Default M=32 hop=16 on same length → 1 + (63-32)/16 = 2
        assert_eq!(SpecReTfParams::default().num_frames(63), 2);
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

    #[test]
    fn frame_score_is_sum_not_average() {
        let a = SpectralFrame {
            amp_dist: vec![1.0, 0.0],
            phase: vec![0.0, 0.0],
        };
        let b = a.clone();
        let s = frame_similarity(&a, &b, &SpecReTfParams::default());
        // s_amp=1, s_phase=1 → 2
        assert_abs_diff_eq!(s, 2.0, epsilon = 1e-12);
    }
}
