//! SVD strategies (port of `svd_strategies.py`).
//!
//! All strategies return a thin decomposition `X = U diag(sigma) V^T` with
//! `sigma` sorted in non-increasing order.  Singular vectors are only defined
//! up to sign (as in LAPACK); `RandomizedSvd` applies the sklearn sign rule.

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::error::{Result, SsaError};
use crate::matrix::{mm, Matrix};

/// Result of a decomposition: `U` is `(L, d)`, `sigma` has length `d`,
/// `V` is `(K*S, d)` (i.e. `Vt.T` of the Python code).
#[derive(Debug, Clone)]
pub struct Decomposition {
    pub u: Matrix,
    pub sigma: Vec<f64>,
    pub v: Matrix,
}

/// What a strategy decomposes: the (weighted, multichannel) trajectory
/// matrix of `x_s` with window `l`.  It can be materialised
/// ([`Trajectory::dense`]) or applied matrix-free ([`Trajectory::operator`]).
pub struct Trajectory<'a> {
    pub x_s: MatRef<'a, f64>,
    pub l: usize,
    pub weights: &'a [f64],
}

impl Trajectory<'_> {
    pub fn n(&self) -> usize {
        self.x_s.nrows()
    }
    pub fn k(&self) -> usize {
        self.x_s.nrows() - self.l + 1
    }
    /// Dense `(L, K*S)` trajectory matrix.
    pub fn dense(&self) -> Matrix {
        crate::ssa::construct_trajectory_matrix(self.x_s, self.l, self.weights)
    }
    /// FFT-based matrix-free trajectory operator.
    pub fn operator(&self) -> crate::hmat::HankelOp {
        crate::hmat::HankelOp::new(self.x_s, self.l, self.weights)
    }
    /// `||X||_F^2` computed from the series (`sum_s w_s sum_t c_t x_t^2`).
    pub fn frob_norm2(&self) -> f64 {
        let (n, l, k) = (self.n(), self.l, self.k());
        (0..self.x_s.ncols())
            .map(|s| {
                self.weights[s]
                    * (0..n)
                        .map(|t| crate::hankel::anti_diagonal_len(t, l, k) as f64 * self.x_s[(t, s)].powi(2))
                        .sum::<f64>()
            })
            .sum()
    }
}

/// Strategy used to decompose the trajectory matrix.
pub trait SvdStrategy: Send + Sync {
    fn decompose(&self, t: &Trajectory<'_>) -> Result<Decomposition>;
}

/// Strategies that work on an explicit matrix.
pub trait DenseSvd {
    fn decompose_matrix(&self, x: MatRef<'_, f64>) -> Result<Decomposition>;
}

macro_rules! dense_strategy {
    ($t:ty) => {
        impl SvdStrategy for $t {
            fn decompose(&self, t: &Trajectory<'_>) -> Result<Decomposition> {
                self.decompose_matrix(t.dense().as_ref())
            }
        }
    };
}
dense_strategy!(FullSvd);
dense_strategy!(EigenSvd);
dense_strategy!(RandomizedSvd);

/// Exact thin SVD (`numpy.linalg.svd(X, full_matrices=False)`).
#[derive(Debug, Clone, Copy, Default)]
pub struct FullSvd;

impl DenseSvd for FullSvd {
    fn decompose_matrix(&self, x: MatRef<'_, f64>) -> Result<Decomposition> {
        check_finite(x)?;
        thin_svd(x)
    }
}

/// Thin SVD `A = U diag(s) V^T` (`min(m, n)` triplets, descending).
/// faer reduces strongly rectangular matrices with a QR first (R-SVD, like
/// LAPACK `gesdd`); small problems run sequentially (see [`crate::la`]).
pub fn thin_svd(a: MatRef<'_, f64>) -> Result<Decomposition> {
    let (u, sigma, v) = crate::la::thin_svd(a)?;
    Ok(Decomposition { u, sigma, v })
}

/// Exact decomposition through the eigen-decomposition of the lag-covariance
/// matrix `X X^T` (the "eigen" method of Rssa).  Much faster than `FullSvd`
/// when `L << K*S`.  Not present in the Python package (extension).
///
/// `sigma_i` is computed as `||X^T u_i||` and `v_i = X^T u_i / sigma_i`, so the
/// elementary matrices `sigma_i u_i v_i^T = u_i u_i^T X` always sum exactly to
/// `X`.  Singular values below roughly `sqrt(eps) * sigma_max` are less
/// accurate than with `FullSvd`.
#[derive(Debug, Clone, Copy, Default)]
pub struct EigenSvd;

impl DenseSvd for EigenSvd {
    fn decompose_matrix(&self, x: MatRef<'_, f64>) -> Result<Decomposition> {
        check_finite(x)?;
        let (l, n) = (x.nrows(), x.ncols());
        if l > n {
            // The lag-covariance trick needs L <= K*S; fall back to exact SVD.
            return FullSvd.decompose_matrix(x);
        }
        let mut c = Mat::<f64>::zeros(l, l);
        matmul(c.as_mut(), Accum::Replace, x, x.transpose(), 1.0, faer::get_global_parallelism());
        let (_, evecs) = crate::la::eigh(c.as_ref())?;
        // eigenvalues come in ascending order -> reverse
        let u = Mat::from_fn(l, l, |i, k| evecs[(i, l - 1 - k)]);
        let mut v = mm(x.transpose(), u.as_ref()); // X^T U, (n, l)
        let mut sigma = vec![0.0; l];
        for k in 0..l {
            let s = v.col(k).norm_l2();
            sigma[k] = s;
            if s > 0.0 {
                for i in 0..n {
                    v[(i, k)] /= s;
                }
            }
        }
        // Rounding may break the order for (nearly) equal values: stable re-sort.
        let mut idx: Vec<usize> = (0..l).collect();
        idx.sort_by(|&a, &b| sigma[b].total_cmp(&sigma[a]));
        if idx.iter().enumerate().all(|(k, &j)| k == j) {
            return Ok(Decomposition { u, sigma, v });
        }
        Ok(Decomposition {
            u: Mat::from_fn(l, l, |i, k| u[(i, idx[k])]),
            sigma: idx.iter().map(|&j| sigma[j]).collect(),
            v: Mat::from_fn(n, l, |i, k| v[(i, idx[k])]),
        })
    }
}

/// Number of power iterations (`n_iter` of sklearn).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerIterations {
    /// sklearn `"auto"`: 7 if `n_components < 0.1 * min(shape)` else 4.
    Auto,
    Fixed(usize),
}

/// Approximate truncated SVD, port of `sklearn.utils.extmath.randomized_svd`
/// (Halko et al., 2009) with the defaults used by py-ssa-lib
/// (`transpose="auto"`, `flip_sign=True`).
///
/// Differences: power iterations are normalised with QR instead of LU when
/// `n_iter > 2` (same subspace, numerically at least as stable) and the random
/// stream differs from numpy's, so results agree with sklearn only up to the
/// randomized approximation error.
#[derive(Debug, Clone)]
pub struct RandomizedSvd {
    pub n_components: usize,
    pub n_oversamples: usize,
    pub n_iter: PowerIterations,
    pub random_state: Option<u64>,
}

impl RandomizedSvd {
    /// sklearn defaults: `n_oversamples = 10`, `n_iter = "auto"`, `random_state = None`.
    pub fn new(n_components: usize) -> Self {
        Self { n_components, n_oversamples: 10, n_iter: PowerIterations::Auto, random_state: None }
    }
    pub fn with_oversamples(mut self, p: usize) -> Self {
        self.n_oversamples = p;
        self
    }
    pub fn with_n_iter(mut self, n_iter: PowerIterations) -> Self {
        self.n_iter = n_iter;
        self
    }
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

fn thin_q(a: &Matrix) -> Matrix {
    a.qr().compute_thin_Q()
}

impl DenseSvd for RandomizedSvd {
    fn decompose_matrix(&self, x: MatRef<'_, f64>) -> Result<Decomposition> {
        check_finite(x)?;
        if self.n_components < 1 {
            return Err(SsaError::InvalidArgument("n_components must be >= 1".into()));
        }
        let (n_samples, n_features) = (x.nrows(), x.ncols());
        let n_iter = match self.n_iter {
            PowerIterations::Auto => {
                if (self.n_components as f64) < 0.1 * n_samples.min(n_features) as f64 {
                    7
                } else {
                    4
                }
            }
            PowerIterations::Fixed(k) => k,
        };
        let transpose = n_samples < n_features;
        let a = if transpose { x.transpose() } else { x };
        let n = a.ncols();
        let size = self.n_components + self.n_oversamples;

        let mut rng = match self.random_state {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let mut q = Mat::from_fn(n, size, |_, _| StandardNormal.sample(&mut rng));
        let normalize = n_iter > 2; // sklearn: "none" normaliser for n_iter <= 2
        for _ in 0..n_iter {
            let aq = mm(a, q.as_ref());
            let aq = if normalize { thin_q(&aq) } else { aq };
            let atq = mm(a.transpose(), aq.as_ref());
            q = if normalize { thin_q(&atq) } else { atq };
        }
        let q = thin_q(&mm(a, q.as_ref())); // (m, r)
        let b = mm(q.transpose(), a); // (r, n)
        let svd = thin_svd(b.as_ref())?;
        let uu = mm(q.as_ref(), svd.u.as_ref()); // (m, r)
        let vv = svd.v.as_ref(); // (n, r)
        let k = self.n_components.min(svd.sigma.len());
        let sigma: Vec<f64> = svd.sigma[..k].to_vec();
        // back to the original orientation: X = U_x diag(s) V_x^T
        let (mut ux, mut vx) = if transpose {
            (vv.subcols(0, k).to_owned(), uu.as_ref().subcols(0, k).to_owned())
        } else {
            (uu.as_ref().subcols(0, k).to_owned(), vv.subcols(0, k).to_owned())
        };
        flip_signs(&mut ux, &mut vx);
        Ok(Decomposition { u: ux, sigma, v: vx })
    }
}

/// sklearn `svd_flip` as applied by `randomized_svd`: the entry of largest
/// magnitude of each column of the final `U` is made positive (first index on
/// ties; a zero column is zeroed like `np.sign(0) == 0`).
pub fn flip_signs(u: &mut Matrix, v: &mut Matrix) {
    for j in 0..u.ncols() {
        let mut best = 0;
        let mut best_abs = -1.0;
        for i in 0..u.nrows() {
            let a = u[(i, j)].abs();
            if a > best_abs {
                best_abs = a;
                best = i;
            }
        }
        let s = if u.nrows() == 0 { 1.0 } else { u[(best, j)].signum() * (u[(best, j)] != 0.0) as i32 as f64 };
        if s != 1.0 {
            for i in 0..u.nrows() {
                u[(i, j)] *= s;
            }
            for i in 0..v.nrows() {
                v[(i, j)] *= s;
            }
        }
    }
}

fn check_finite(x: MatRef<'_, f64>) -> Result<()> {
    for j in 0..x.ncols() {
        for i in 0..x.nrows() {
            if !x[(i, j)].is_finite() {
                return Err(SsaError::LinAlg(
                    "SVD did not converge: trajectory matrix contains NaN or infinity".into(),
                ));
            }
        }
    }
    Ok(())
}

/// Truncated decomposition: the leading `k` eigentriples by Lanczos
/// bidiagonalisation applied to the FFT trajectory operator (Rssa
/// `svd.method = "nutrlan"/"propack"`).  The trajectory matrix is never
/// formed: memory `O((L + K) m)` and time `O(m N log N)` for `m` steps.
///
/// Complements [`FullSvd`] (still the default): converged triples are
/// accurate to `tol * sigma_1` in residual norm, but only `k` of them are
/// computed, so `d = k` and reconstructions use those `k` components only.
#[derive(Debug, Clone, Copy)]
pub struct LanczosSvd {
    pub k: usize,
    /// Relative residual tolerance (default `1e-12`).
    pub tol: f64,
    /// Seed of the random starting vector (results do not depend on it
    /// beyond the tolerance).
    pub seed: u64,
}

impl LanczosSvd {
    pub fn new(k: usize) -> Self {
        Self { k, tol: 1e-12, seed: 0x5eed }
    }
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
}

impl SvdStrategy for LanczosSvd {
    fn decompose(&self, t: &Trajectory<'_>) -> Result<Decomposition> {
        check_finite(t.x_s)?;
        if self.k == 0 {
            return Err(SsaError::InvalidArgument("k must be >= 1".into()));
        }
        let op = t.operator();
        let (u, sigma, v) = crate::lanczos::lanczos_svd(&op, self.k, self.tol, self.seed)?;
        Ok(Decomposition { u, sigma, v })
    }
}

/// Toeplitz SSA (Rssa `kind = "toeplitz-ssa"`): `U` are the eigenvectors of
/// the Toeplitz lag-covariance matrix `R_ij = c(|i-j|)`,
/// `c(m) = sum_t x_t x_{t+m} / (N - m)`; then `Z = X^T U`, `sigma = ||Z||`,
/// `V = Z / sigma`, triples sorted by `sigma`.  Univariate series only.
///
/// `neig` keeps the leading `neig` eigenvectors (by eigenvalue) before the
/// sort by `sigma`, as Rssa does (`None`: all `L`).
#[derive(Debug, Clone, Copy, Default)]
pub struct ToeplitzSvd {
    pub neig: Option<usize>,
}

/// Rssa `Lcor`: `c(m) = sum_{t} x_t x_{t+m} / (N - m)`, `m < L`.
pub fn lag_correlations(x: &[f64], l: usize) -> Vec<f64> {
    let n = x.len();
    if (n as f64) * (l as f64) < 2.0e5 {
        return (0..l)
            .map(|m| x[..n - m].iter().zip(&x[m..]).map(|(a, b)| a * b).sum::<f64>() / (n - m) as f64)
            .collect();
    }
    // autocorrelation through FFT (as Rssa's C code)
    let p = crate::hankel::next_fast_len(n + l - 1);
    let mut planner = realfft::RealFftPlanner::<f64>::new();
    let (f, b) = (planner.plan_fft_forward(p), planner.plan_fft_inverse(p));
    let mut buf = vec![0.0; p];
    buf[..n].copy_from_slice(x);
    let mut spec = f.make_output_vec();
    f.process(&mut buf, &mut spec).expect("fft");
    for z in spec.iter_mut() {
        *z = realfft::num_complex::Complex::new(z.norm_sqr(), 0.0);
    }
    let mut out = b.make_output_vec();
    b.process(&mut spec, &mut out).expect("fft");
    (0..l).map(|m| out[m] / p as f64 / (n - m) as f64).collect()
}

impl SvdStrategy for ToeplitzSvd {
    fn decompose(&self, t: &Trajectory<'_>) -> Result<Decomposition> {
        check_finite(t.x_s)?;
        if t.x_s.ncols() != 1 {
            return Err(SsaError::InvalidArgument("Toeplitz SSA is defined for one series".into()));
        }
        let l = t.l;
        let sw = t.weights[0].sqrt();
        let x: Vec<f64> = t.x_s.col(0).iter().map(|v| v * sw).collect();
        let c = lag_correlations(&x, l);
        let neig = self.neig.unwrap_or(l).min(l);
        // Rssa "auto": full eigen-decomposition below L = 500, otherwise the
        // leading eigenvectors by Lanczos on the FFT Toeplitz operator.
        let u = if l < 500 || neig * 2 > l {
            let tm = Mat::from_fn(l, l, |i, j| c[i.abs_diff(j)]);
            let (_, evecs) = crate::la::eigh(tm.as_ref())?;
            Mat::from_fn(l, neig, |i, j| evecs[(i, l - 1 - j)])
        } else {
            let op = crate::hmat::ToeplitzOp::new(&c);
            crate::lanczos::lanczos_eigh(&op, neig, 1e-12, 0x5eed)?.1
        };
        let op = t.operator();
        let ncols = crate::hmat::LinOp::ncols(&op);
        let mut v = Mat::<f64>::zeros(ncols, neig);
        let mut sigma = vec![0.0; neig];
        let mut buf = vec![0.0; ncols];
        for j in 0..neig {
            let uj: Vec<f64> = u.col(j).iter().copied().collect();
            crate::hmat::LinOp::rmatvec(&op, &uj, &mut buf);
            let s = buf.iter().map(|z| z * z).sum::<f64>().sqrt();
            sigma[j] = s;
            for (i, z) in buf.iter().enumerate() {
                v[(i, j)] = if s > 0.0 { z / s } else { 0.0 };
            }
        }
        let mut idx: Vec<usize> = (0..neig).collect();
        idx.sort_by(|&a, &b| sigma[b].total_cmp(&sigma[a]));
        Ok(Decomposition {
            u: Mat::from_fn(l, neig, |i, k| u[(i, idx[k])]),
            sigma: idx.iter().map(|&j| sigma[j]).collect(),
            v: Mat::from_fn(ncols, neig, |i, k| v[(i, idx[k])]),
        })
    }
}
