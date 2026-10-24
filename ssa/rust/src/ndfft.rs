//! n-dimensional FFT helpers for nD-SSA: correlations and convolutions of
//! column-major arrays on a (possibly periodic) grid.

use std::sync::Arc;

use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use crate::hankel::next_fast_len;

pub type C64 = Complex<f64>;

/// Column-major strides of `dims`.
pub fn strides(dims: &[usize]) -> Vec<usize> {
    let mut s = vec![1; dims.len()];
    for i in 1..dims.len() {
        s[i] = s[i - 1] * dims[i - 1];
    }
    s
}

/// Multi-index of linear index `idx` in `dims` (column-major).
pub fn unravel(mut idx: usize, dims: &[usize]) -> Vec<usize> {
    dims.iter()
        .map(|&d| {
            let r = idx % d;
            idx /= d;
            r
        })
        .collect()
}

/// Grid position (linear index in a `p`-shaped grid) of every element of a
/// `dims`-shaped array, in column-major order of the array.
pub fn positions(dims: &[usize], p: &[usize]) -> Vec<usize> {
    let n: usize = dims.iter().product();
    let sp = strides(p);
    let mut out = Vec::with_capacity(n);
    let mut idx = vec![0usize; dims.len()];
    let mut pos = 0usize;
    for _ in 0..n {
        out.push(pos);
        for d in 0..dims.len() {
            idx[d] += 1;
            pos += sp[d];
            if idx[d] < dims[d] {
                break;
            }
            pos -= idx[d] * sp[d];
            idx[d] = 0;
        }
    }
    out
}

/// FFT grid of size `p` (per dimension).  The first axis uses a real FFT
/// (half spectrum of `p0 / 2 + 1` rows), the others complex FFTs, so a
/// spectrum has `(p0/2 + 1) * p1 * ...` entries (column-major).
pub struct Grid {
    pub p: Vec<usize>,
    /// Shape of the half spectrum.
    pub sdims: Vec<usize>,
    r2c: Arc<dyn RealToComplex<f64>>,
    c2r: Arc<dyn ComplexToReal<f64>>,
    fwd: Vec<Arc<dyn Fft<f64>>>,
    inv: Vec<Arc<dyn Fft<f64>>>,
}

const PAR_MIN: usize = 1 << 16;

impl Grid {
    /// Grid for a field of size `n`: periodic dimensions use exactly `n`,
    /// the others a fast length `>= n` (no wrap-around for linear
    /// correlations/convolutions that stay inside the field).
    pub fn for_field(n: &[usize], circular: &[bool]) -> Self {
        let p: Vec<usize> = n.iter().zip(circular).map(|(&v, &c)| if c { v } else { next_fast_len(v) }).collect();
        Self::new(p)
    }

    pub fn new(p: Vec<usize>) -> Self {
        let mut rp = RealFftPlanner::<f64>::new();
        let r2c = rp.plan_fft_forward(p[0]);
        let c2r = rp.plan_fft_inverse(p[0]);
        let mut planner = FftPlanner::<f64>::new();
        let fwd = p.iter().map(|&d| planner.plan_fft_forward(d)).collect();
        let inv = p.iter().map(|&d| planner.plan_fft_inverse(d)).collect();
        let mut sdims = p.clone();
        sdims[0] = p[0] / 2 + 1;
        Self { p, sdims, r2c, c2r, fwd, inv }
    }

    /// Number of grid points.
    pub fn len(&self) -> usize {
        self.p.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Complex FFT along axes `1..` of a half spectrum (in place).
    fn other_axes(&self, buf: &mut [C64], inverse: bool) {
        use rayon::prelude::*;
        let st = strides(&self.sdims);
        let total: usize = self.sdims.iter().product();
        for d in 1..self.p.len() {
            let (n, s) = (self.p[d], st[d]);
            if n == 1 {
                continue;
            }
            let plan = if inverse { &self.inv[d] } else { &self.fwd[d] };
            let block = s * n;
            // `s` interleaved lines per block: gather them into one
            // contiguous buffer and transform all at once (rustfft batches
            // over consecutive chunks), with buffers reused per thread
            let init = || (vec![C64::new(0.0, 0.0); block], vec![C64::new(0.0, 0.0); plan.get_inplace_scratch_len()]);
            let work = |(lines, scratch): &mut (Vec<C64>, Vec<C64>), blk: &mut [C64]| {
                // tiled transposes (contiguous reads of `blk` rows)
                const T: usize = 16;
                for o0 in (0..s).step_by(T) {
                    let o1 = (o0 + T).min(s);
                    for i in 0..n {
                        let row = &blk[i * s + o0..i * s + o1];
                        for (q, v) in row.iter().enumerate() {
                            lines[(o0 + q) * n + i] = *v;
                        }
                    }
                }
                plan.process_with_scratch(lines, scratch);
                for o0 in (0..s).step_by(T) {
                    let o1 = (o0 + T).min(s);
                    for i in 0..n {
                        let row = &mut blk[i * s + o0..i * s + o1];
                        for (q, v) in row.iter_mut().enumerate() {
                            *v = lines[(o0 + q) * n + i];
                        }
                    }
                }
            };
            if total >= PAR_MIN && total / block > 1 {
                buf.par_chunks_mut(block).for_each_init(init, work);
            } else {
                let mut st = init();
                buf.chunks_mut(block).for_each(|b| work(&mut st, b));
            }
        }
    }

    /// Forward transform of a real grid buffer (length `len()`).
    pub fn forward(&self, real: &mut [f64]) -> Vec<C64> {
        use rayon::prelude::*;
        let (p0, h0) = (self.p[0], self.sdims[0]);
        let mut out = vec![C64::new(0.0, 0.0); h0 * (self.len() / p0)];
        let work = |sc: &mut Vec<C64>, (src, dst): (&mut [f64], &mut [C64])| {
            self.r2c.process_with_scratch(src, dst, sc).expect("fft");
        };
        if self.len() >= PAR_MIN {
            real.par_chunks_mut(p0).zip(out.par_chunks_mut(h0)).for_each_init(|| self.r2c.make_scratch_vec(), work);
        } else {
            let mut sc = self.r2c.make_scratch_vec();
            real.chunks_mut(p0).zip(out.chunks_mut(h0)).for_each(|x| work(&mut sc, x));
        }
        self.other_axes(&mut out, false);
        out
    }

    /// Inverse transform to a real grid buffer (unnormalised).
    pub fn inverse(&self, mut spec: Vec<C64>) -> Vec<f64> {
        use rayon::prelude::*;
        self.other_axes(&mut spec, true);
        let (p0, h0) = (self.p[0], self.sdims[0]);
        let mut out = vec![0.0; self.len()];
        let work = |sc: &mut Vec<C64>, (src, dst): (&mut [C64], &mut [f64])| {
            src[0].im = 0.0;
            if p0 % 2 == 0 {
                src[h0 - 1].im = 0.0;
            }
            self.c2r.process_with_scratch(src, dst, sc).expect("fft");
        };
        if self.len() >= PAR_MIN {
            spec.par_chunks_mut(h0).zip(out.par_chunks_mut(p0)).for_each_init(|| self.c2r.make_scratch_vec(), work);
        } else {
            let mut sc = self.c2r.make_scratch_vec();
            spec.chunks_mut(h0).zip(out.chunks_mut(p0)).for_each(|x| work(&mut sc, x));
        }
        out
    }

    /// Embeds a real array of size `dims` at the origin of a real grid buffer.
    pub fn embed(&self, a: &[f64], dims: &[usize]) -> Vec<f64> {
        let mut buf = vec![0.0; self.len()];
        for (v, pos) in a.iter().zip(positions(dims, &self.p)) {
            buf[pos] = *v;
        }
        buf
    }

    /// Half spectrum of a real array of size `dims`.
    pub fn spectrum(&self, a: &[f64], dims: &[usize]) -> Vec<C64> {
        let mut b = self.embed(a, dims);
        self.forward(&mut b)
    }

    /// The first `dims` entries of a real grid buffer, scaled by `1 / len`.
    pub fn extract(&self, buf: &[f64], dims: &[usize]) -> Vec<f64> {
        let s = 1.0 / self.len() as f64;
        positions(dims, &self.p).into_iter().map(|pos| buf[pos] * s).collect()
    }

    /// `r[m] = sum_i f[m + i] a[i]` (periodic on the grid) for `m < out_dims`,
    /// given the spectrum `fspec` of `f`.
    pub fn correlate(&self, fspec: &[C64], a: &[f64], a_dims: &[usize], out_dims: &[usize]) -> Vec<f64> {
        let mut s = self.spectrum(a, a_dims);
        for (z, x) in s.iter_mut().zip(fspec) {
            *z = x * z.conj();
        }
        let r = self.inverse(s);
        self.extract(&r, out_dims)
    }

    /// `out[n] = sum_{l + k = n} a[l] b[k]` (periodic on the grid) for `n < out_dims`.
    pub fn convolve_spectra(&self, mut sa: Vec<C64>, sb: &[C64], out_dims: &[usize]) -> Vec<f64> {
        for (z, x) in sa.iter_mut().zip(sb) {
            *z *= x;
        }
        let r = self.inverse(sa);
        self.extract(&r, out_dims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correlate_and_convolve_3d() {
        let n = [5, 4, 3];
        let f: Vec<f64> = (0..60).map(|i| ((i * 7 % 11) as f64) - 5.0).collect();
        let a_dims = [2, 2, 2];
        let a: Vec<f64> = (0..8).map(|i| i as f64 * 0.5 - 1.0).collect();
        for circ in [[false, false, false], [true, false, true]] {
            let g = Grid::for_field(&n, &circ);
            let fs = g.spectrum(&f, &n);
            let out_dims: Vec<usize> = (0..3).map(|d| if circ[d] { n[d] } else { n[d] - a_dims[d] + 1 }).collect();
            let r = g.correlate(&fs, &a, &a_dims, &out_dims);
            for (idx, &val) in r.iter().enumerate() {
                let m = unravel(idx, &out_dims);
                let mut want = 0.0;
                for (j, &av) in a.iter().enumerate() {
                    let i = unravel(j, &a_dims);
                    let p: Vec<usize> = (0..3).map(|d| (m[d] + i[d]) % n[d]).collect();
                    want += f[p[0] + 5 * p[1] + 20 * p[2]] * av;
                }
                assert!((val - want).abs() < 1e-10, "{idx}: {val} vs {want}");
            }
        }
    }
}
