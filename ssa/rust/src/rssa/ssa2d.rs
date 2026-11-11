//! 2D-SSA (Rssa `kind = "2d-ssa"`, rectangular window): decomposition of a
//! field `(N1, N2)` with a window `(L1, L2)` through its Hankel-block-Hankel
//! trajectory matrix, reconstruction, w-correlations and 2D-ESPRIT.

use std::sync::Arc;

use faer::Mat;
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use super::common::{wcor_matrix, Group};
use super::ossa::eigen_with_vectors;
use super::parest::{roots2pars, shift_matrix_rows, Params, SolveMethod};
use crate::error::{Result, SsaError};
use crate::hankel::{hankel_weights, next_fast_len};
use crate::hmat::HbhOp;
use crate::matrix::{select_cols, Matrix};
use crate::ssa::Complex64;

/// Decomposition used by [`Ssa2d::fit`].
#[derive(Debug, Clone, Copy)]
pub enum Method2d {
    /// Exact SVD of the dense trajectory matrix.
    Full,
    /// Leading `k` triples by Lanczos on the FFT operator.
    Lanczos(usize),
}

/// Fitted 2D-SSA.
#[derive(Debug, Clone)]
pub struct Ssa2d {
    pub field: Matrix,
    pub l: (usize, usize),
    pub k: (usize, usize),
    /// `(L1*L2, d)` eigenvectors (vectorised windows, first index fastest).
    pub u: Matrix,
    pub sigma: Vec<f64>,
    /// `(K1*K2, d)` factor vectors.
    pub v: Matrix,
    pub d: usize,
}

impl Ssa2d {
    pub fn fit(field: &Matrix, l: (usize, usize), method: Method2d) -> Result<Self> {
        let n = (field.nrows(), field.ncols());
        if l.0 == 0 || l.1 == 0 || l.0 > n.0 || l.1 > n.1 {
            return Err(SsaError::InvalidArgument("invalid window size".into()));
        }
        if field.col_iter().any(|c| c.iter().any(|v| !v.is_finite())) {
            return Err(SsaError::LinAlg("field contains NaN or infinity".into()));
        }
        let k = (n.0 - l.0 + 1, n.1 - l.1 + 1);
        let (u, sigma, v) = match method {
            Method2d::Full => {
                let x = Mat::from_fn(l.0 * l.1, k.0 * k.1, |r, c| field[(r % l.0 + c % k.0, r / l.0 + c / k.0)]);
                crate::la::thin_svd(x.as_ref())?
            }
            Method2d::Lanczos(kk) => {
                let op = HbhOp::new(field.as_ref(), l);
                crate::lanczos::lanczos_svd(&op, kk, 1e-12, 0x5eed)?
            }
        };
        let d = sigma.len();
        Ok(Self { field: field.clone(), l, k, u, sigma, v, d })
    }

    fn weights(&self) -> Matrix {
        let (n0, n1) = (self.field.nrows(), self.field.ncols());
        let w0 = hankel_weights(n0, self.l.0, self.k.0);
        let w1 = hankel_weights(n1, self.l.1, self.k.1);
        Mat::from_fn(n0, n1, |i, j| (w0[i] * w1[j]) as f64)
    }

    /// Reconstructed field of a group.
    pub fn reconstruct(&self, group: &[usize]) -> Result<Matrix> {
        if let Some(&i) = group.iter().find(|&&i| i >= self.d) {
            return Err(SsaError::IndexOutOfRange { index: i, len: self.d });
        }
        let (n0, n1) = (self.field.nrows(), self.field.ncols());
        let conv = Conv2::new((n0, n1));
        let mut acc = vec![Complex::new(0.0, 0.0); conv.p.0 * conv.p.1];
        for &i in group {
            let a = conv.spectrum(&self.u.col(i).iter().map(|v| v * self.sigma[i]).collect::<Vec<_>>(), self.l);
            let b = conv.spectrum(&self.v.col(i).iter().copied().collect::<Vec<_>>(), self.k);
            for ((z, x), y) in acc.iter_mut().zip(&a).zip(&b) {
                *z += x * y;
            }
        }
        let sums = conv.inverse(acc, (n0, n1));
        let w = self.weights();
        Ok(Mat::from_fn(n0, n1, |i, j| sums[(i, j)] / w[(i, j)]))
    }

    /// Signed w-correlations between groups.
    pub fn wcor(&self, groups: &[Group]) -> Result<Matrix> {
        let fs: Vec<Matrix> = groups.iter().map(|g| self.reconstruct(g)).collect::<Result<_>>()?;
        let w = self.weights();
        let n = w.nrows() * w.ncols();
        let flat = |m: &Matrix, t: usize| m[(t % m.nrows(), t / m.nrows())];
        let x = Mat::from_fn(n, fs.len(), |t, j| flat(&fs[j], t));
        let wv: Vec<f64> = (0..n).map(|t| flat(&w, t)).collect();
        Ok(wcor_matrix(&x, &wv))
    }

    /// 2D-ESPRIT (Rssa `parestimate` for 2D-SSA, column subspace, "diag"
    /// pairing with `beta = 8`): roots along each dimension.
    pub fn parestimate(&self, group: &[usize], method: SolveMethod) -> Result<(Params, Params)> {
        if let Some(&i) = group.iter().find(|&&i| i >= self.d) {
            return Err(SsaError::IndexOutOfRange { index: i, len: self.d });
        }
        let u = select_cols(self.u.as_ref(), group);
        let (l0, l1) = self.l;
        let mut left1 = Vec::new();
        let mut right1 = Vec::new();
        let mut left2 = Vec::new();
        let mut right2 = Vec::new();
        for i2 in 0..l1 {
            for i1 in 0..l0 {
                let p = i1 + i2 * l0;
                if i1 + 1 < l0 {
                    left1.push(p);
                    right1.push(p + 1);
                }
                if i2 + 1 < l1 {
                    left2.push(p);
                    right2.push(p + l0);
                }
            }
        }
        let z1 = shift_matrix_rows(&u, &left1, &right1, method)?;
        let z2 = shift_matrix_rows(&u, &left2, &right2, method)?;
        // beta = 8 -> c(1 - 8, 8)
        let z = Mat::from_fn(z1.nrows(), z1.ncols(), |i, j| -7.0 * z1[(i, j)] + 8.0 * z2[(i, j)]);
        let (_, t) = eigen_with_vectors(&z)?;
        let roots = |zk: &Matrix| -> Result<Vec<Complex64>> {
            let r = zk.nrows();
            let zc = Mat::from_fn(r, r, |i, j| Complex64::new(zk[(i, j)], 0.0));
            let zt = mm_c(&zc, &t);
            // diag(T^{-1} Zk T): solve T X = Zk T
            let lu = t.partial_piv_lu();
            let x = faer::linalg::solvers::Solve::solve(&lu, zt.as_ref());
            Ok((0..r).map(|i| x[(i, i)]).collect())
        };
        Ok((roots2pars(roots(&z1)?), roots2pars(roots(&z2)?)))
    }

    /// Dense trajectory matrix (for tests and small problems).
    pub fn trajectory(&self) -> Matrix {
        let (l, k) = (self.l, self.k);
        Mat::from_fn(l.0 * l.1, k.0 * k.1, |r, c| self.field[(r % l.0 + c % k.0, r / l.0 + c / k.0)])
    }

    /// `||X||_F^2`.
    pub fn total_energy(&self) -> f64 {
        let w = self.weights();
        (0..w.nrows()).flat_map(|i| (0..w.ncols()).map(move |j| (i, j))).map(|(i, j)| w[(i, j)] * self.field[(i, j)].powi(2)).sum()
    }
}

fn mm_c(a: &Mat<Complex64>, b: &Mat<Complex64>) -> Mat<Complex64> {
    Mat::from_fn(a.nrows(), b.ncols(), |i, j| (0..a.ncols()).map(|k| a[(i, k)] * b[(k, j)]).sum())
}

/// 2D linear convolution through FFT.
struct Conv2 {
    p: (usize, usize),
    f0: Arc<dyn Fft<f64>>,
    i0: Arc<dyn Fft<f64>>,
    f1: Arc<dyn Fft<f64>>,
    i1: Arc<dyn Fft<f64>>,
}

impl Conv2 {
    fn new(n: (usize, usize)) -> Self {
        let p = (next_fast_len(n.0), next_fast_len(n.1));
        let mut pl = FftPlanner::<f64>::new();
        Self { p, f0: pl.plan_fft_forward(p.0), i0: pl.plan_fft_inverse(p.0), f1: pl.plan_fft_forward(p.1), i1: pl.plan_fft_inverse(p.1) }
    }
    fn fft2(&self, buf: &mut [Complex<f64>], inverse: bool) {
        let (p0, p1) = self.p;
        let (a, b) = if inverse { (&self.i0, &self.i1) } else { (&self.f0, &self.f1) };
        for col in buf.chunks_mut(p0) {
            a.process(col);
        }
        let mut row = vec![Complex::new(0.0, 0.0); p1];
        for i in 0..p0 {
            for j in 0..p1 {
                row[j] = buf[i + j * p0];
            }
            b.process(&mut row);
            for j in 0..p1 {
                buf[i + j * p0] = row[j];
            }
        }
    }
    fn spectrum(&self, a: &[f64], dims: (usize, usize)) -> Vec<Complex<f64>> {
        let mut buf = vec![Complex::new(0.0, 0.0); self.p.0 * self.p.1];
        for j in 0..dims.1 {
            for i in 0..dims.0 {
                buf[i + j * self.p.0] = Complex::new(a[i + j * dims.0], 0.0);
            }
        }
        self.fft2(&mut buf, false);
        buf
    }
    fn inverse(&self, mut buf: Vec<Complex<f64>>, out: (usize, usize)) -> Matrix {
        self.fft2(&mut buf, true);
        let s = 1.0 / (self.p.0 * self.p.1) as f64;
        Mat::from_fn(out.0, out.1, |i, j| buf[i + j * self.p.0].re * s)
    }
}

