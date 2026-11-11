//! Hankelisation (anti-diagonal averaging) and fast rank-one hankelisation.
//!
//! The Python code materialises every elementary matrix `sigma_i u_i v_i^T`
//! (`d * L * K * S` doubles) and averages its anti-diagonals with a Python
//! loop.  The anti-diagonal *sums* of a rank-one matrix `a b^T` are exactly the
//! linear convolution `a * b`, so the same numbers can be obtained without
//! ever forming the matrix: directly in `O(L K)` for small sizes or with real
//! FFTs in `O(N log N)` otherwise.

use std::sync::Arc;

use faer::MatRef;
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

/// Anti-diagonal averaging of an arbitrary `(R, C)` matrix, returning
/// `R + C - 1` values (port of `BaseSSA.diagonal_averaging`).
///
/// Element `n` is the mean of all `X[i, j]` with `i + j == n`.
pub fn diagonal_averaging(x: MatRef<'_, f64>) -> Vec<f64> {
    let (r, c) = (x.nrows(), x.ncols());
    if r == 0 || c == 0 {
        // numpy: range(-R+1, C) yields R+C-1 offsets (none when both are 0)
        // and every diagonal is empty, so each mean is NaN.
        return vec![f64::NAN; (r + c).saturating_sub(1)];
    }
    let mut sums = vec![0.0; r + c - 1];
    for j in 0..c {
        let col = x.col(j);
        for i in 0..r {
            sums[i + j] += col[i];
        }
    }
    for (n, s) in sums.iter_mut().enumerate() {
        *s /= anti_diagonal_len(n, r, c) as f64;
    }
    sums
}

/// Number of elements of anti-diagonal `n` in an `(r, c)` matrix.
#[inline]
pub fn anti_diagonal_len(n: usize, r: usize, c: usize) -> usize {
    let total = r + c - 1;
    (n + 1).min(r).min(c).min(total - n)
}

/// Hankel weights `w_n = min(n + 1, L, K, N - n)` (port of
/// `BaseSSA.construct_hankel_weights`).
pub fn hankel_weights(n: usize, l: usize, k: usize) -> Vec<usize> {
    (0..n).map(|i| (i + 1).min(l).min(k).min(n - i)).collect()
}

/// Smallest even 5-smooth integer `>= n` (fast FFT length).
pub fn next_fast_len(n: usize) -> usize {
    let mut m = n.max(2);
    loop {
        if m.is_multiple_of(2) {
            let mut r = m;
            for p in [2, 3, 5] {
                while r.is_multiple_of(p) {
                    r /= p;
                }
            }
            if r == 1 {
                return m;
            }
        }
        m += 1;
    }
}

enum Mode {
    Direct,
    Fft { p: usize, r2c: Arc<dyn RealToComplex<f64>>, c2r: Arc<dyn ComplexToReal<f64>> },
}

/// Computes anti-diagonal sums of sums of rank-one `(L, K)` matrices.
pub struct Convolver {
    l: usize,
    k: usize,
    mode: Mode,
}

/// Spectrum (or raw copy in direct mode) of a zero-padded vector.
pub enum Spectrum {
    Direct(Vec<f64>),
    Fft(Vec<Complex<f64>>),
}

impl Convolver {
    /// Picks direct or FFT convolution from a simple cost model.
    pub fn new(l: usize, k: usize) -> Self {
        let n = l + k - 1;
        let p = next_fast_len(n);
        let direct_cost = (l * k) as f64;
        let fft_cost = 6.0 * p as f64 * (p as f64).log2();
        if direct_cost <= fft_cost {
            Self::direct(l, k)
        } else {
            Self::fft(l, k)
        }
    }

    pub fn direct(l: usize, k: usize) -> Self {
        Self { l, k, mode: Mode::Direct }
    }

    pub fn fft(l: usize, k: usize) -> Self {
        let p = next_fast_len(l + k - 1);
        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(p);
        let c2r = planner.plan_fft_inverse(p);
        Self { l, k, mode: Mode::Fft { p, r2c, c2r } }
    }

    pub fn is_fft(&self) -> bool {
        matches!(self.mode, Mode::Fft { .. })
    }

    /// Output length `L + K - 1`.
    pub fn out_len(&self) -> usize {
        self.l + self.k - 1
    }

    /// Transforms a vector of length `L` (left factor) or `K` (right factor),
    /// multiplied by `scale`.
    pub fn spectrum(&self, x: impl Iterator<Item = f64>, scale: f64) -> Spectrum {
        match &self.mode {
            Mode::Direct => Spectrum::Direct(x.map(|v| v * scale).collect()),
            Mode::Fft { p, r2c, .. } => {
                let mut buf = vec![0.0; *p];
                for (b, v) in buf.iter_mut().zip(x) {
                    *b = v * scale;
                }
                let mut out = r2c.make_output_vec();
                r2c.process(&mut buf, &mut out).expect("FFT length mismatch");
                Spectrum::Fft(out)
            }
        }
    }

    /// New zeroed accumulator.
    pub fn accumulator(&self) -> Accumulator {
        match &self.mode {
            Mode::Direct => Accumulator::Direct(vec![0.0; self.out_len()]),
            Mode::Fft { p, .. } => Accumulator::Fft(vec![Complex::new(0.0, 0.0); p / 2 + 1]),
        }
    }

    /// `acc += a (*) b` (linear convolution, i.e. anti-diagonal sums of `a b^T`).
    pub fn add_product(&self, acc: &mut Accumulator, a: &Spectrum, b: &Spectrum) {
        match (acc, a, b) {
            (Accumulator::Direct(y), Spectrum::Direct(a), Spectrum::Direct(b)) => {
                for (i, &ai) in a.iter().enumerate() {
                    if ai == 0.0 {
                        continue;
                    }
                    for (yj, &bj) in y[i..i + b.len()].iter_mut().zip(b) {
                        *yj += ai * bj;
                    }
                }
            }
            (Accumulator::Fft(y), Spectrum::Fft(a), Spectrum::Fft(b)) => {
                for ((yj, aj), bj) in y.iter_mut().zip(a).zip(b) {
                    *yj += aj * bj;
                }
            }
            _ => unreachable!("mixed convolution modes"),
        }
    }

    /// Returns the `L + K - 1` anti-diagonal sums held by the accumulator.
    pub fn finish(&self, acc: Accumulator) -> Vec<f64> {
        match (acc, &self.mode) {
            (Accumulator::Direct(y), _) => y,
            (Accumulator::Fft(mut y), Mode::Fft { p, c2r, .. }) => {
                // Spectra of real signals: DC (and Nyquist) bins are real.
                y[0].im = 0.0;
                if p % 2 == 0 {
                    let last = y.len() - 1;
                    y[last].im = 0.0;
                }
                let mut out = c2r.make_output_vec();
                c2r.process(&mut y, &mut out).expect("FFT length mismatch");
                let inv = 1.0 / *p as f64;
                out.truncate(self.out_len());
                out.iter_mut().for_each(|v| *v *= inv);
                out
            }
            _ => unreachable!("mixed convolution modes"),
        }
    }
}

/// Frequency- or time-domain accumulator.
pub enum Accumulator {
    Direct(Vec<f64>),
    Fft(Vec<Complex<f64>>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    #[test]
    fn fast_len() {
        assert_eq!(next_fast_len(1), 2);
        assert_eq!(next_fast_len(7), 8);
        assert_eq!(next_fast_len(11), 12);
        assert_eq!(next_fast_len(1999), 2000);
        assert_eq!(next_fast_len(2001), 2048);
        assert_eq!(next_fast_len(2049), 2160);
    }

    #[test]
    fn diag_avg_matches_definition() {
        let x = Mat::from_fn(3, 5, |i, j| (i * 7 + j * 3) as f64 + 0.5 * (i as f64).sin());
        let y = diagonal_averaging(x.as_ref());
        assert_eq!(y.len(), 7);
        for n in 0..7 {
            let mut s = 0.0;
            let mut c = 0;
            for i in 0..3 {
                for j in 0..5 {
                    if i + j == n {
                        s += x[(i, j)];
                        c += 1;
                    }
                }
            }
            assert!((y[n] - s / c as f64).abs() < 1e-14);
        }
    }

    #[test]
    fn direct_and_fft_agree() {
        for (l, k) in [(2, 2), (5, 9), (30, 31), (100, 401)] {
            let a: Vec<f64> = (0..l).map(|i| (i as f64 * 0.37).sin()).collect();
            let b: Vec<f64> = (0..k).map(|i| (i as f64 * 0.11).cos() + 0.3).collect();
            let mut out = Vec::new();
            for conv in [Convolver::direct(l, k), Convolver::fft(l, k)] {
                let sa = conv.spectrum(a.iter().copied(), 2.0);
                let sb = conv.spectrum(b.iter().copied(), 1.0);
                let mut acc = conv.accumulator();
                conv.add_product(&mut acc, &sa, &sb);
                out.push(conv.finish(acc));
            }
            assert_eq!(out[0].len(), l + k - 1);
            for (x, y) in out[0].iter().zip(&out[1]) {
                assert!((x - y).abs() < 1e-11, "{x} vs {y}");
            }
        }
    }
}
