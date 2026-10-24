//! Matrix-free trajectory operators.
//!
//! The trajectory matrix of a series is Hankel, so `X v` and `X^T u` are
//! correlations of the series with the vector and cost `O(N log N)` with FFT
//! instead of `O(L K)`.  The matrix itself is never formed, which is what
//! makes truncated (Lanczos) decompositions of very long series possible.

use std::sync::Arc;

use faer::MatRef;
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::{Fft, FftPlanner};

use crate::hankel::next_fast_len;

/// A linear operator `R^n -> R^m` with its transpose.
pub trait LinOp: Sync {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    /// `y = A x`, `x.len() == ncols`, `y.len() == nrows`.
    fn matvec(&self, x: &[f64], y: &mut [f64]);
    /// `y = A^T x`, `x.len() == nrows`, `y.len() == ncols`.
    fn rmatvec(&self, x: &[f64], y: &mut [f64]);
}

/// Dense matrix as a [`LinOp`].
pub struct DenseOp<'a>(pub MatRef<'a, f64>);

impl LinOp for DenseOp<'_> {
    fn nrows(&self) -> usize {
        self.0.nrows()
    }
    fn ncols(&self) -> usize {
        self.0.ncols()
    }
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        y.iter_mut().for_each(|v| *v = 0.0);
        for (j, &xj) in x.iter().enumerate() {
            if xj != 0.0 {
                for (yi, a) in y.iter_mut().zip(self.0.col(j).iter()) {
                    *yi += a * xj;
                }
            }
        }
    }
    fn rmatvec(&self, x: &[f64], y: &mut [f64]) {
        for (j, yj) in y.iter_mut().enumerate() {
            *yj = self.0.col(j).iter().zip(x).map(|(a, b)| a * b).sum();
        }
    }
}

/// Per-thread FFT work buffers (avoid allocations in matrix-vector products).
#[derive(Default)]
struct Scratch {
    buf: Vec<f64>,
    spec: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
}

impl Scratch {
    fn get(&mut self, p: usize, sl: usize) -> (&mut [f64], &mut [Complex<f64>], &mut [Complex<f64>]) {
        if self.buf.len() != p {
            self.buf = vec![0.0; p];
            self.spec = vec![Complex::new(0.0, 0.0); p / 2 + 1];
        }
        if self.scratch.len() < sl {
            self.scratch = vec![Complex::new(0.0, 0.0); sl];
        }
        (&mut self.buf, &mut self.spec, &mut self.scratch[..sl])
    }
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch::default());
}

/// Weighted (M)SSA trajectory matrix `[sqrt(w_1) H_1 | ... | sqrt(w_S) H_S]`
/// of shape `(L, K*S)` applied through FFT.
///
/// For long series the factor index range `0..K` is split into segments
/// processed in parallel: segment `[j0, j0 + c)` only involves
/// `x[j0 .. j0 + c + L - 1)`, whose spectrum is precomputed, so every product
/// becomes a set of independent, shorter FFT correlations (exact, no
/// approximation).
pub struct HankelOp {
    l: usize,
    k: usize,
    nch: usize,
    p: usize,
    /// `(j0, c)` of each segment.
    segs: Vec<(usize, usize)>,
    /// Spectra per `(channel, segment)`, index `ch * segs.len() + g`.
    spectra: Vec<Vec<Complex<f64>>>,
    r2c: Arc<dyn RealToComplex<f64>>,
    c2r: Arc<dyn ComplexToReal<f64>>,
}

impl HankelOp {
    /// `x_s` is `(N, S)`, `weights` has length `S`.
    pub fn new(x_s: MatRef<'_, f64>, l: usize, weights: &[f64]) -> Self {
        let n = x_s.nrows();
        let k = n - l + 1;
        let cores = (rayon::current_num_threads() / 2).max(1);
        // split only when the series is long enough to amortise the threads;
        // keep segments at least L/2 long so the extra FFT work stays bounded
        let nseg = if n >= 20_000 && cores > 1 { cores.min(k.div_ceil((l / 2).max(1))).max(1) } else { 1 };
        let c = k.div_ceil(nseg);
        let segs: Vec<(usize, usize)> = (0..nseg).map(|g| (g * c, c.min(k - (g * c).min(k)))).filter(|s| s.1 > 0).collect();
        let p = next_fast_len(c + l - 1);
        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(p);
        let c2r = planner.plan_fft_inverse(p);
        let mut spectra = Vec::new();
        for ch in 0..x_s.ncols() {
            let sw = weights[ch].sqrt();
            for &(j0, cl) in &segs {
                let mut buf = vec![0.0; p];
                let len = cl + l - 1;
                for (b, t) in buf.iter_mut().zip(j0..j0 + len) {
                    *b = sw * x_s[(t, ch)];
                }
                let mut out = r2c.make_output_vec();
                r2c.process(&mut buf, &mut out).expect("fft");
                spectra.push(out);
            }
        }
        Self { l, k, nch: x_s.ncols(), p, segs, spectra, r2c, c2r }
    }

    /// `r[m] = sum_i seg[m + i] a[i]` for `m < out.len()` (no wrap because
    /// `m + i < c + L - 1 <= p`).
    fn correlate(&self, idx: usize, a: &[f64], out: &mut [f64]) {
        SCRATCH.with(|cell| {
            let mut sc = cell.borrow_mut();
            let (buf, spec, scratch) = sc.get(self.p, self.r2c.get_scratch_len().max(self.c2r.get_scratch_len()));
            buf[..a.len()].copy_from_slice(a);
            buf[a.len()..].iter_mut().for_each(|v| *v = 0.0);
            self.r2c.process_with_scratch(buf, spec, scratch).expect("fft");
            for (z, x) in spec.iter_mut().zip(&self.spectra[idx]) {
                *z = x * z.conj();
            }
            spec[0].im = 0.0;
            if self.p % 2 == 0 {
                let last = spec.len() - 1;
                spec[last].im = 0.0;
            }
            self.c2r.process_with_scratch(spec, buf, scratch).expect("fft");
            let inv = 1.0 / self.p as f64;
            for (o, r) in out.iter_mut().zip(buf.iter()) {
                *o = r * inv;
            }
        })
    }
}

impl LinOp for HankelOp {
    fn nrows(&self) -> usize {
        self.l
    }
    fn ncols(&self) -> usize {
        self.k * self.nch
    }
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        use rayon::prelude::*;
        let ns = self.segs.len();
        let work = |idx: usize| {
            let (ch, g) = (idx / ns, idx % ns);
            let (j0, c) = self.segs[g];
            let mut tmp = vec![0.0; self.l];
            self.correlate(idx, &x[ch * self.k + j0..ch * self.k + j0 + c], &mut tmp);
            tmp
        };
        let parts: Vec<Vec<f64>> = if ns > 1 { (0..self.nch * ns).into_par_iter().map(work).collect() } else { (0..self.nch).map(work).collect() };
        y.iter_mut().for_each(|v| *v = 0.0);
        for p in parts {
            y.iter_mut().zip(&p).for_each(|(a, b)| *a += b);
        }
    }
    fn rmatvec(&self, x: &[f64], y: &mut [f64]) {
        use rayon::prelude::*;
        let ns = self.segs.len();
        let k = self.k;
        // disjoint output slices per (channel, segment)
        let mut slices: Vec<(usize, &mut [f64])> = Vec::with_capacity(self.nch * ns);
        let mut rest = y;
        for ch in 0..self.nch {
            for g in 0..ns {
                let c = self.segs[g].1;
                let (head, tail) = rest.split_at_mut(c);
                slices.push((ch * ns + g, head));
                rest = tail;
            }
        }
        debug_assert!(rest.is_empty());
        let _ = k;
        if ns > 1 {
            slices.into_par_iter().for_each(|(idx, out)| self.correlate(idx, x, out));
        } else {
            for (idx, out) in slices {
                self.correlate(idx, x, out);
            }
        }
    }
}

/// Hankel-block-Hankel trajectory operator of 2D-SSA.
///
/// Field `F` is `(N1, N2)`, window `(L1, L2)`; columns of the trajectory
/// matrix are vectorised `L1 x L2` windows (first index fastest, as in R),
/// ordered by the window position `(k1, k2)`, `k1` fastest.  Products are
/// 2D correlations computed with a real FFT along the first axis and a
/// complex FFT along the second.
pub struct HbhOp {
    n: (usize, usize),
    l: (usize, usize),
    k: (usize, usize),
    p: (usize, usize),
    /// Half spectrum `(p0/2 + 1) x p1` of the zero-padded field, column-major.
    spec: Vec<Complex<f64>>,
    r2c: Arc<dyn RealToComplex<f64>>,
    c2r: Arc<dyn ComplexToReal<f64>>,
    f1: Arc<dyn Fft<f64>>,
    i1: Arc<dyn Fft<f64>>,
}

impl HbhOp {
    /// `field` is `(N1, N2)`.
    pub fn new(field: MatRef<'_, f64>, l: (usize, usize)) -> Self {
        let n = (field.nrows(), field.ncols());
        let k = (n.0 - l.0 + 1, n.1 - l.1 + 1);
        let p = (next_fast_len(n.0), next_fast_len(n.1));
        let mut rp = RealFftPlanner::<f64>::new();
        let (r2c, c2r) = (rp.plan_fft_forward(p.0), rp.plan_fft_inverse(p.0));
        let mut cp = FftPlanner::<f64>::new();
        let (f1, i1) = (cp.plan_fft_forward(p.1), cp.plan_fft_inverse(p.1));
        let mut op = Self { n, l, k, p, spec: Vec::new(), r2c, c2r, f1, i1 };
        let a: Vec<f64> = (0..n.1).flat_map(|j| (0..n.0).map(move |i| (i, j))).map(|(i, j)| field[(i, j)]).collect();
        op.spec = op.forward(&a, n);
        op
    }

    pub fn window(&self) -> (usize, usize) {
        self.l
    }
    pub fn factor_dims(&self) -> (usize, usize) {
        self.k
    }
    pub fn field_dims(&self) -> (usize, usize) {
        self.n
    }

    /// Half 2D spectrum of `a` (`dims`, column-major) zero-padded to `p`.
    fn forward(&self, a: &[f64], dims: (usize, usize)) -> Vec<Complex<f64>> {
        let (p0, p1) = self.p;
        let h = p0 / 2 + 1;
        let mut out = vec![Complex::new(0.0, 0.0); h * p1];
        let mut col = vec![0.0; p0];
        let mut sc = self.r2c.make_scratch_vec();
        for j in 0..dims.1 {
            col[..dims.0].copy_from_slice(&a[j * dims.0..(j + 1) * dims.0]);
            col[dims.0..].iter_mut().for_each(|v| *v = 0.0);
            self.r2c.process_with_scratch(&mut col, &mut out[j * h..(j + 1) * h], &mut sc).expect("fft");
        }
        self.axis1(&mut out, false);
        out
    }

    fn axis1(&self, buf: &mut [Complex<f64>], inverse: bool) {
        let (h, p1) = (self.p.0 / 2 + 1, self.p.1);
        let f = if inverse { &self.i1 } else { &self.f1 };
        let mut row = vec![Complex::new(0.0, 0.0); p1];
        let mut sc = vec![Complex::new(0.0, 0.0); f.get_inplace_scratch_len()];
        for i in 0..h {
            for j in 0..p1 {
                row[j] = buf[i + j * h];
            }
            f.process_with_scratch(&mut row, &mut sc);
            for j in 0..p1 {
                buf[i + j * h] = row[j];
            }
        }
    }

    /// `r[m1, m2] = sum_{i1,i2} F[m1+i1, m2+i2] a[i1, i2]` for `m < out_dims`.
    fn correlate(&self, a: &[f64], a_dims: (usize, usize), out: &mut [f64], out_dims: (usize, usize)) {
        let (p0, p1) = self.p;
        let h = p0 / 2 + 1;
        let mut s = self.forward(a, a_dims);
        for (z, x) in s.iter_mut().zip(&self.spec) {
            *z = x * z.conj();
        }
        self.axis1(&mut s, true);
        let mut col = vec![0.0; p0];
        let mut sc = self.c2r.make_scratch_vec();
        let inv = 1.0 / (p0 * p1) as f64;
        for j in 0..out_dims.1 {
            let c = &mut s[j * h..(j + 1) * h];
            c[0].im = 0.0;
            if p0 % 2 == 0 {
                c[h - 1].im = 0.0;
            }
            self.c2r.process_with_scratch(c, &mut col, &mut sc).expect("fft");
            for i in 0..out_dims.0 {
                out[i + j * out_dims.0] = col[i] * inv;
            }
        }
    }
}

impl LinOp for HbhOp {
    fn nrows(&self) -> usize {
        self.l.0 * self.l.1
    }
    fn ncols(&self) -> usize {
        self.k.0 * self.k.1
    }
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        self.correlate(x, self.k, y, self.l);
    }
    fn rmatvec(&self, x: &[f64], y: &mut [f64]) {
        self.correlate(x, self.l, y, self.k);
    }
}


/// Symmetric Toeplitz matrix `T_ij = c(|i - j|)` applied by FFT through a
/// circulant embedding of size `>= 2L - 1`.
pub struct ToeplitzOp {
    l: usize,
    p: usize,
    spec: Vec<Complex<f64>>,
    r2c: Arc<dyn RealToComplex<f64>>,
    c2r: Arc<dyn ComplexToReal<f64>>,
}

impl ToeplitzOp {
    pub fn new(c: &[f64]) -> Self {
        let l = c.len();
        let p = next_fast_len(2 * l - 1);
        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(p);
        let c2r = planner.plan_fft_inverse(p);
        let mut circ = vec![0.0; p];
        circ[..l].copy_from_slice(c);
        for i in 1..l {
            circ[p - i] = c[i];
        }
        let mut spec = r2c.make_output_vec();
        r2c.process(&mut circ, &mut spec).expect("fft");
        Self { l, p, spec, r2c, c2r }
    }
}

impl LinOp for ToeplitzOp {
    fn nrows(&self) -> usize {
        self.l
    }
    fn ncols(&self) -> usize {
        self.l
    }
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        let mut buf = vec![0.0; self.p];
        buf[..self.l].copy_from_slice(x);
        let mut s = self.r2c.make_output_vec();
        self.r2c.process(&mut buf, &mut s).expect("fft");
        for (a, b) in s.iter_mut().zip(&self.spec) {
            *a *= b;
        }
        s[0].im = 0.0;
        if self.p % 2 == 0 {
            let last = s.len() - 1;
            s[last].im = 0.0;
        }
        self.c2r.process(&mut s, &mut buf).expect("fft");
        let inv = 1.0 / self.p as f64;
        for (o, v) in y.iter_mut().zip(&buf) {
            *o = v * inv;
        }
    }
    fn rmatvec(&self, x: &[f64], y: &mut [f64]) {
        self.matvec(x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    fn check(op: &dyn LinOp, dense: &Mat<f64>) {
        let x: Vec<f64> = (0..op.ncols()).map(|i| ((i * 7 % 11) as f64 - 5.0) * 0.3).collect();
        let u: Vec<f64> = (0..op.nrows()).map(|i| ((i * 5 % 13) as f64 - 6.0) * 0.2).collect();
        let mut y = vec![0.0; op.nrows()];
        let mut z = vec![0.0; op.ncols()];
        op.matvec(&x, &mut y);
        op.rmatvec(&u, &mut z);
        let d = DenseOp(dense.as_ref());
        let mut y2 = vec![0.0; op.nrows()];
        let mut z2 = vec![0.0; op.ncols()];
        d.matvec(&x, &mut y2);
        d.rmatvec(&u, &mut z2);
        for (a, b) in y.iter().zip(&y2).chain(z.iter().zip(&z2)) {
            assert!((a - b).abs() < 1e-10, "{a} vs {b}");
        }
    }

    #[test]
    fn hankel_op_matches_dense() {
        let x = Mat::from_fn(37, 3, |t, s| ((t * (s + 2)) as f64 * 0.31).sin() + s as f64);
        let w = [1.0, 0.5, 2.0];
        let op = HankelOp::new(x.as_ref(), 11, &w);
        let dense = crate::ssa::construct_trajectory_matrix(x.as_ref(), 11, &w);
        check(&op, &dense);
    }

    #[test]
    fn toeplitz_op_and_symmetric_lanczos() {
        let c: Vec<f64> = (0..40).map(|i| (-(i as f64) * 0.1).exp() * (i as f64 * 0.7).cos()).collect();
        let op = ToeplitzOp::new(&c);
        let dense = Mat::from_fn(40, 40, |i, j| c[i.abs_diff(j)]);
        check(&op, &dense);
        let (lam, vecs) = crate::lanczos::lanczos_eigh(&op, 5, 1e-12, 1).unwrap();
        let (ld, _) = crate::la::eigh(dense.as_ref()).unwrap();
        for i in 0..5 {
            assert!((lam[i] - ld[39 - i]).abs() < 1e-10 * ld[39].abs(), "{} vs {}", lam[i], ld[39 - i]);
            let mut y = vec![0.0; 40];
            let vi: Vec<f64> = vecs.col(i).iter().copied().collect();
            op.matvec(&vi, &mut y);
            for (a, b) in y.iter().zip(&vi) {
                assert!((a - lam[i] * b).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn segmented_hankel_op_matches_dense() {
        // long enough to trigger the parallel segmented path
        let x = Mat::from_fn(30_011, 2, |t, s| ((t * (s + 3)) as f64 * 0.0123).sin() + ((t * 7919) % 101) as f64 / 101.0);
        let w = [1.0, 0.7];
        for l in [7, 4000, 15_006] {
            let op = HankelOp::new(x.as_ref(), l, &w);
            let k = 30_011 - l + 1;
            let v: Vec<f64> = (0..2 * k).map(|i| ((i * 13 % 17) as f64 - 8.0) * 0.1).collect();
            let u: Vec<f64> = (0..l).map(|i| ((i * 5 % 11) as f64 - 5.0) * 0.2).collect();
            let mut y = vec![0.0; l];
            let mut z = vec![0.0; 2 * k];
            op.matvec(&v, &mut y);
            op.rmatvec(&u, &mut z);
            // direct sums at a few positions
            for &i in &[0, l / 2, l - 1] {
                let want: f64 = (0..2).map(|s| w[s].sqrt() * (0..k).map(|j| x[(i + j, s)] * v[s * k + j]).sum::<f64>()).sum();
                assert!((y[i] - want).abs() < 1e-8 * want.abs().max(1.0), "matvec {i}: {} vs {want}", y[i]);
            }
            for &(s, j) in &[(0, 0), (0, k - 1), (1, k / 3), (1, k - 1)] {
                let want: f64 = w[s].sqrt() * (0..l).map(|i| x[(i + j, s)] * u[i]).sum::<f64>();
                assert!((z[s * k + j] - want).abs() < 1e-8 * want.abs().max(1.0), "rmatvec {s},{j}");
            }
        }
    }

    #[test]
    fn hbh_op_matches_dense() {
        let f = Mat::from_fn(9, 12, |i, j| ((i * 3 + j * 5) as f64 * 0.17).cos() + 0.1 * i as f64);
        let l = (4, 5);
        let op = HbhOp::new(f.as_ref(), l);
        let (k0, k1) = op.factor_dims();
        let dense = Mat::from_fn(l.0 * l.1, k0 * k1, |r, c| {
            let (i1, i2) = (r % l.0, r / l.0);
            let (j1, j2) = (c % k0, c / k0);
            f[(i1 + j1, i2 + j2)]
        });
        check(&op, &dense);
    }
}

/// Column `j` of an operator (Rssa `hcols` / `hbhcols` / `tcols`).
pub fn op_col(op: &dyn LinOp, j: usize) -> Vec<f64> {
    let mut e = vec![0.0; op.ncols()];
    e[j] = 1.0;
    let mut y = vec![0.0; op.nrows()];
    op.matvec(&e, &mut y);
    y
}

/// Row `i` of an operator (Rssa `hrows` / `hbhrows` / `trows`).
pub fn op_row(op: &dyn LinOp, i: usize) -> Vec<f64> {
    let mut e = vec![0.0; op.nrows()];
    e[i] = 1.0;
    let mut y = vec![0.0; op.ncols()];
    op.rmatvec(&e, &mut y);
    y
}

/// `A x` or `A^T x` (Rssa `hmatmul(transposed = )`, `hbhmatmul`, `tmatmul`).
pub fn op_mul(op: &dyn LinOp, x: &[f64], transposed: bool) -> Vec<f64> {
    if transposed {
        let mut y = vec![0.0; op.ncols()];
        op.rmatvec(x, &mut y);
        y
    } else {
        let mut y = vec![0.0; op.nrows()];
        op.matvec(x, &mut y);
        y
    }
}
