//! SSA / MSSA (port of `ssa_collection.py`).
//!
//! [`Ssa`] and [`Mssa`] are the unfitted estimators (Python classes `SSA` and
//! `MSSA`); `fit` returns an [`SsaModel`] that holds every attribute the
//! Python object gets after `fit` (`X_s`, `N`, `S`, `L`, `K`, `weights`,
//! `X_ss`, `U`, `Sigma`, `V`, `d`, `rel_contribution`, `cumsum_contr`) and
//! implements all of its methods.
//!
//! Main implementation differences (results are the same up to rounding):
//! * elementary matrices `X_elem` are not stored (the Python array has
//!   `d * L * K * S` doubles); they are produced on demand by
//!   [`SsaModel::elementary_matrix`] / [`SsaModel::elementary_matrices`];
//! * reconstructions use rank-one anti-diagonal sums (direct or FFT
//!   convolution) instead of materialising and averaging matrices;
//! * the weighted-correlation Gram matrix is a single GEMM;
//! * per-component work runs in parallel (rayon).

use faer::{Mat, MatRef};
use rayon::prelude::*;

use crate::error::{Result, SsaError};
use crate::hankel::{self, Accumulator, Convolver, Spectrum};
use crate::matrix::{mm, select_cols, Matrix};
use crate::svd::{FullSvd, SvdStrategy};
use crate::weights::{UniformWeights, WeightingStrategy};

/// Complex number type returned by ESPRIT.
pub type Complex64 = realfft::num_complex::Complex<f64>;

/// Shared configuration of [`Ssa`] and [`Mssa`] (Python `BaseSSA.__init__`).
pub struct Estimator {
    pub decomposition_strategy: Box<dyn SvdStrategy>,
    pub weighting_strategy: Box<dyn WeightingStrategy>,
}

impl Default for Estimator {
    fn default() -> Self {
        Self { decomposition_strategy: Box::new(FullSvd), weighting_strategy: Box::new(UniformWeights) }
    }
}

impl Estimator {
    /// Python `BaseSSA.fit` on an already prepared `(N, S)` matrix.
    pub fn fit_matrix(&self, x_s: Matrix, l: Option<usize>) -> Result<SsaModel> {
        let (n, s) = (x_s.nrows(), x_s.ncols());
        if s == 0 {
            return Err(SsaError::InvalidArgument("need at least one time series".into()));
        }
        let max_l = n.div_ceil(2);
        let l = l.unwrap_or(max_l);
        if !(2 <= l && l <= max_l) {
            return Err(SsaError::InvalidArgument(
                "L must satisfy 2 <= L <= floor((N + 1) / 2).".into(),
            ));
        }
        let weights = self.weighting_strategy.compute_weights(&x_s)?;
        validate_weights(&weights, s)?;

        let traj = crate::svd::Trajectory { x_s: x_s.as_ref(), l, weights: &weights };
        let dec = self.decomposition_strategy.decompose(&traj)?;
        let total_energy = traj.frob_norm2();
        Ok(SsaModel::from_parts(x_s, l, weights, dec, total_energy))
    }
}

fn validate_weights(w: &[f64], s: usize) -> Result<()> {
    if w.len() != s {
        return Err(SsaError::InvalidArgument(format!(
            "expected {s} channel weights, got {}",
            w.len()
        )));
    }
    if let Some(bad) = w.iter().find(|v| !(v.is_finite() && **v > 0.0)) {
        return Err(SsaError::InvalidArgument(format!(
            "channel weights must be strictly positive and finite, got {bad}"
        )));
    }
    Ok(())
}

macro_rules! estimator_builder {
    ($name:ident) => {
        impl $name {
            /// Default strategies: `FullSvd` and `UniformWeights`.
            pub fn new() -> Self {
                Self(Estimator::default())
            }
            pub fn with_decomposition(mut self, s: impl SvdStrategy + 'static) -> Self {
                self.0.decomposition_strategy = Box::new(s);
                self
            }
            pub fn with_weighting(mut self, w: impl WeightingStrategy + 'static) -> Self {
                self.0.weighting_strategy = Box::new(w);
                self
            }
            pub fn decomposition_strategy(&self) -> &dyn SvdStrategy {
                self.0.decomposition_strategy.as_ref()
            }
            pub fn weighting_strategy(&self) -> &dyn WeightingStrategy {
                self.0.weighting_strategy.as_ref()
            }
        }
        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

/// Univariate SSA (Python class `SSA`).
pub struct Ssa(pub Estimator);
estimator_builder!(Ssa);

impl Ssa {
    /// Fits one series; `l = None` uses `floor((N + 1) / 2)`.
    pub fn fit(&self, x: &[f64], l: Option<usize>) -> Result<SsaModel> {
        self.0.fit_matrix(Mat::from_fn(x.len(), 1, |i, _| x[i]), l)
    }
}

/// Multivariate SSA (Python class `MSSA`).
pub struct Mssa(pub Estimator);
estimator_builder!(Mssa);

impl Mssa {
    /// Fits an `(N, S)` matrix whose columns are the series.
    pub fn fit(&self, x: MatRef<'_, f64>, l: Option<usize>) -> Result<SsaModel> {
        self.0.fit_matrix(x.to_owned(), l)
    }
}

/// Weighted trajectory matrix `[sqrt(w_1) H_1 | ... | sqrt(w_S) H_S]`, shape
/// `(L, K*S)` (Python `construct_trajectory_matrix`).
pub fn construct_trajectory_matrix(x_s: MatRef<'_, f64>, l: usize, weights: &[f64]) -> Matrix {
    let (n, s) = (x_s.nrows(), x_s.ncols());
    let k = n - l + 1;
    let mut out = Mat::<f64>::zeros(l, k * s);
    for ch in 0..s {
        let sw = weights[ch].sqrt();
        let col = x_s.col(ch);
        for j in 0..k {
            let dst = out.col_mut(ch * k + j);
            for (i, d) in dst.iter_mut().enumerate() {
                *d = sw * col[i + j];
            }
        }
    }
    out
}

/// Fitted SSA/MSSA model.
#[derive(Debug, Clone)]
pub struct SsaModel {
    /// Input data, `(N, S)` (`X_s`).
    pub x_s: Matrix,
    pub n: usize,
    pub s: usize,
    pub l: usize,
    pub k: usize,
    /// Channel weights, length `S`.
    pub weights: Vec<f64>,
    /// Left singular vectors `(L, d)` (`U`).
    pub u: Matrix,
    /// Singular values (`Sigma`).
    pub sigma: Vec<f64>,
    /// Right singular vectors `(K*S, d)` (`V`, i.e. `Vt.T`).
    pub v: Matrix,
    pub d: usize,
    /// `sigma^2 / ||X_ss||_F^2 * 100` (`rel_contribution`).
    pub rel_contribution: Vec<f64>,
    /// Cumulative sum of `rel_contribution` (`cumsum_contr`).
    pub cumsum_contr: Vec<f64>,
}

impl SsaModel {
    /// Assembles a model from a decomposition of the trajectory matrix of
    /// `x_s` (`total_energy = ||X||_F^2`).
    pub fn from_parts(
        x_s: Matrix,
        l: usize,
        weights: Vec<f64>,
        dec: crate::svd::Decomposition,
        total_energy: f64,
    ) -> Self {
        let (n, s) = (x_s.nrows(), x_s.ncols());
        let d = dec.sigma.len();
        let rel_contribution: Vec<f64> = dec.sigma.iter().map(|v| v * v / total_energy * 100.0).collect();
        let cumsum_contr = rel_contribution
            .iter()
            .scan(0.0, |acc, v| {
                *acc += v;
                Some(*acc)
            })
            .collect();
        SsaModel {
            x_s,
            n,
            s,
            l,
            k: n - l + 1,
            weights,
            u: dec.u,
            sigma: dec.sigma,
            v: dec.v,
            d,
            rel_contribution,
            cumsum_contr,
        }
    }

    /// Weighted trajectory matrix `(L, K*S)` (`X_ss`), built on demand.
    pub fn x_ss(&self) -> Matrix {
        construct_trajectory_matrix(self.x_s.as_ref(), self.l, &self.weights)
    }

    /// Matrix-free trajectory operator (FFT).
    pub fn trajectory_operator(&self) -> crate::hmat::HankelOp {
        crate::hmat::HankelOp::new(self.x_s.as_ref(), self.l, &self.weights)
    }

    /// `||X||_F^2`.
    pub fn total_energy(&self) -> f64 {
        crate::svd::Trajectory { x_s: self.x_s.as_ref(), l: self.l, weights: &self.weights }.frob_norm2()
    }

    /// `Vt` of the Python object, `(d, K*S)`.
    pub fn vt(&self) -> Matrix {
        self.v.transpose().to_owned()
    }

    /// Errors if any index is `>= d`.
    pub fn check_indices(&self, idx: &[usize]) -> Result<()> {
        match idx.iter().find(|&&i| i >= self.d) {
            Some(&i) => Err(SsaError::IndexOutOfRange { index: i, len: self.d }),
            None => Ok(()),
        }
    }

    /// Elementary matrix `sigma_i u_i v_i^T`, shape `(L, K*S)` (`X_elem[i]`).
    pub fn elementary_matrix(&self, i: usize) -> Result<Matrix> {
        self.check_indices(&[i])?;
        let (u, v, s) = (self.u.col(i), self.v.col(i), self.sigma[i]);
        Ok(Mat::from_fn(self.l, self.k * self.s, |r, c| s * u[r] * v[c]))
    }

    /// All elementary matrices (Python `elementary_matrix()` / `X_elem`).
    /// Needs `d * L * K * S` doubles; prefer the on-demand methods.
    pub fn elementary_matrices(&self) -> Vec<Matrix> {
        (0..self.d).into_par_iter().map(|i| self.elementary_matrix(i).unwrap()).collect()
    }

    /// Hankel weights `min(n+1, L, K, N-n)` (`construct_hankel_weights`).
    pub fn construct_hankel_weights(&self) -> Vec<usize> {
        hankel::hankel_weights(self.n, self.l, self.k)
    }

    fn convolver(&self) -> Convolver {
        Convolver::new(self.l, self.k)
    }

    /// Reconstructs the sum of the given components with `conv`, `(N, S)`.
    fn reconstruct_with(&self, conv: &Convolver, idx: &[usize]) -> Matrix {
        let left: Vec<Spectrum> = idx
            .iter()
            .map(|&i| conv.spectrum(self.u.col(i).iter().copied(), self.sigma[i]))
            .collect();
        let mut out = Mat::<f64>::zeros(self.n, self.s);
        let cols: Vec<Vec<f64>> = (0..self.s)
            .into_par_iter()
            .map(|ch| {
                let mut acc: Accumulator = conv.accumulator();
                for (a, &i) in left.iter().zip(idx) {
                    let vb = self.v.col(i).subrows(ch * self.k, self.k);
                    let b = conv.spectrum(vb.iter().copied(), 1.0);
                    conv.add_product(&mut acc, a, &b);
                }
                let mut y = conv.finish(acc);
                let inv_sw = 1.0 / self.weights[ch].sqrt();
                for (t, val) in y.iter_mut().enumerate() {
                    *val = *val / hankel::anti_diagonal_len(t, self.l, self.k) as f64 * inv_sw;
                }
                y
            })
            .collect();
        for (ch, y) in cols.into_iter().enumerate() {
            for (t, val) in y.into_iter().enumerate() {
                out[(t, ch)] = val;
            }
        }
        out
    }

    /// Reconstructed series of the sum of the selected components, `(N, S)`
    /// (`reconstruct_components`).  Duplicated indices are summed twice, as
    /// with numpy fancy indexing.
    pub fn reconstruct_components(&self, idx: &[usize]) -> Result<Matrix> {
        self.check_indices(idx)?;
        Ok(self.reconstruct_with(&self.convolver(), idx))
    }

    /// Every elementary reconstructed component: `d` matrices of shape
    /// `(N, S)` (`_reconstruct_elementary_components`, shape `(d, N, S)`).
    pub fn reconstruct_elementary_components(&self) -> Vec<Matrix> {
        let conv = self.convolver();
        // parallelise over components; inner channel loop stays sequential.
        (0..self.d)
            .into_par_iter()
            .map(|i| {
                let a = conv.spectrum(self.u.col(i).iter().copied(), self.sigma[i]);
                let mut out = Mat::<f64>::zeros(self.n, self.s);
                for ch in 0..self.s {
                    let vb = self.v.col(i).subrows(ch * self.k, self.k);
                    let b = conv.spectrum(vb.iter().copied(), 1.0);
                    let mut acc = conv.accumulator();
                    conv.add_product(&mut acc, &a, &b);
                    let y = conv.finish(acc);
                    let inv_sw = 1.0 / self.weights[ch].sqrt();
                    for (t, val) in y.into_iter().enumerate() {
                        out[(t, ch)] =
                            val / hankel::anti_diagonal_len(t, self.l, self.k) as f64 * inv_sw;
                    }
                }
                out
            })
            .collect()
    }

    /// Weighted correlation matrix `(d, d)` (`compute_weighted_correlation_matrix`).
    pub fn compute_weighted_correlation_matrix(&self) -> Matrix {
        let comps = self.reconstruct_elementary_components();
        let hw = self.construct_hankel_weights();
        let (n, s, d) = (self.n, self.s, self.d);
        // F[i, (t, ch)] = comp_i[t, ch] * sqrt(hw_t * w_ch);  gram = F F^T.
        let mut f = Mat::<f64>::zeros(d, n * s);
        for (i, c) in comps.iter().enumerate() {
            for ch in 0..s {
                for t in 0..n {
                    f[(i, ch * n + t)] = c[(t, ch)] * (hw[t] as f64 * self.weights[ch]).sqrt();
                }
            }
        }
        let gram = mm(f.as_ref(), f.transpose());
        let norms: Vec<f64> = (0..d).map(|i| gram[(i, i)].sqrt()).collect();
        Mat::from_fn(d, d, |i, j| {
            let denom = norms[i] * norms[j];
            if i == j {
                1.0
            } else if denom == 0.0 {
                // A component that is exactly zero (sigma == 0) is orthogonal to
                // everything; Python would produce NaN from 0/0 here.
                0.0
            } else {
                (gram[(i, j)] / denom).abs()
            }
        })
    }

    /// Shift-invariance ESPRIT estimate (`estimate_ESPRIT`): returns
    /// `(mu, rho, omega)` sorted by `|omega|`.
    pub fn estimate_esprit(&self, idx: &[usize]) -> Result<(Vec<Complex64>, Vec<f64>, Vec<f64>)> {
        self.check_indices(idx)?;
        let r = idx.len();
        if r == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }
        let p = select_cols(self.u.as_ref(), idx);
        let upper = p.as_ref().subrows(0, self.l - 1);
        let lower = p.as_ref().subrows(1, self.l - 1);
        let phi = lstsq(upper, lower)?;
        let ev = crate::la::eigenvalues(phi.as_ref())?;
        if ev.iter().any(|(re, im)| !re.is_finite() || !im.is_finite()) {
            return Err(SsaError::LinAlg("Array must not contain infs or NaNs".into()));
        }
        let mut mu: Vec<Complex64> = ev
            .into_iter()
            .map(|(re, im)| Complex64::new(re, if im == 0.0 { 0.0 } else { im }))
            .collect();
        // LAPACK convention for conjugate pairs: positive imaginary part first.
        let mut j = 0;
        while j + 1 < mu.len() {
            if mu[j].im != 0.0 && mu[j + 1].re == mu[j].re && mu[j + 1].im == -mu[j].im {
                if mu[j].im < 0.0 {
                    mu.swap(j, j + 1);
                }
                j += 2;
            } else {
                j += 1;
            }
        }
        // stable sort by |angle|
        mu.sort_by(|a, b| a.im.atan2(a.re).abs().total_cmp(&b.im.atan2(b.re).abs()));
        let rho = mu.iter().map(|z| z.norm()).collect();
        let omega = mu.iter().map(|z| z.im.atan2(z.re)).collect();
        Ok((mu, rho, omega))
    }

    /// Linear recurrence coefficients, length `L - 1` (`estimate_LRR`).
    pub fn estimate_lrr(&self, idx: &[usize]) -> Result<Vec<f64>> {
        self.check_indices(idx)?;
        let l = self.l;
        let pi: Vec<f64> = idx.iter().map(|&i| self.u[(l - 1, i)]).collect();
        let nu2: f64 = pi.iter().map(|v| v * v).sum();
        let denom = 1.0 - nu2;
        Ok((0..l - 1)
            .map(|row| {
                idx.iter().zip(&pi).map(|(&i, p)| self.u[(row, i)] * p).sum::<f64>() / denom
            })
            .collect())
    }

    /// Recurrent (L-) forecast (`L_forecast`).  Returns `(N + steps, S)` when
    /// `return_full`, else `(steps, S)`.
    pub fn l_forecast(&self, steps: usize, idx: &[usize], return_full: bool) -> Result<Matrix> {
        let r = self.estimate_lrr(idx)?;
        let rec = self.reconstruct_components(idx)?;
        let (n, l) = (self.n, self.l);
        let mut y = Mat::<f64>::zeros(n + steps, self.s);
        y.as_mut().subrows_mut(0, n).copy_from(rec.as_ref());
        for m in 0..steps {
            let start = n - l + m + 1;
            for ch in 0..self.s {
                let col = y.col(ch);
                let mut acc = 0.0;
                for (t, coef) in r.iter().enumerate() {
                    acc += col[start + t] * coef;
                }
                y[(n + m, ch)] = acc;
            }
        }
        Ok(finish_forecast(y, n, steps, return_full))
    }

    /// Vector (K-) forecast (`K_forecast`).
    pub fn k_forecast(&self, steps: usize, idx: &[usize], return_full: bool) -> Result<Matrix> {
        self.check_indices(idx)?;
        let (n, k, s) = (self.n, self.k, self.s);
        let vsel = select_cols(self.v.as_ref(), idx); // (K*S, r)
        let r = idx.len();
        // W: last row of every channel block; Q: the remaining rows in order.
        let w = Mat::from_fn(s, r, |ch, j| vsel[(ch * k + k - 1, j)]);
        let q = Mat::from_fn(s * (k - 1), r, |row, j| {
            let (ch, t) = (row / (k - 1), row % (k - 1));
            vsel[(ch * k + t, j)]
        });
        let wwt = mm(w.as_ref(), w.transpose());
        let a = Mat::from_fn(s, s, |i, j| if i == j { 1.0 } else { 0.0 } - wwt[(i, j)]);
        let mut op = mm(w.as_ref(), q.transpose()); // (S, S*(K-1))
        solve_in_place(a, &mut op)?;

        let sqrt_w: Vec<f64> = self.weights.iter().map(|w| w.sqrt()).collect();
        let rec = self.reconstruct_components(idx)?;
        let mut y = Mat::<f64>::zeros(n + steps, s);
        for ch in 0..s {
            for t in 0..n {
                y[(t, ch)] = rec[(t, ch)] * sqrt_w[ch];
            }
        }
        let mut z = vec![0.0; s * (k - 1)];
        for m in 0..steps {
            let start = n - k + m + 1;
            for ch in 0..s {
                for t in 0..k - 1 {
                    z[ch * (k - 1) + t] = y[(start + t, ch)];
                }
            }
            for i in 0..s {
                let mut acc = 0.0;
                for (j, zj) in z.iter().enumerate() {
                    acc += op[(i, j)] * zj;
                }
                y[(n + m, i)] = acc;
            }
        }
        for ch in 0..s {
            for t in 0..n + steps {
                y[(t, ch)] /= sqrt_w[ch];
            }
        }
        Ok(finish_forecast(y, n, steps, return_full))
    }
}

fn finish_forecast(y: Matrix, n: usize, steps: usize, return_full: bool) -> Matrix {
    if return_full {
        y
    } else {
        // NOTE: Python's `y_pred[-0:]` returns the whole array when steps == 0;
        // here the (empty) forecast part is returned instead.
        y.as_ref().subrows(n, steps).to_owned()
    }
}

/// Solves `A X = B` in place (`B <- X`) by LU with partial pivoting, like
/// LAPACK `gesv` behind `numpy.linalg.solve`: an exactly zero pivot raises
/// `LinAlg("Singular matrix")`.  `A` is the small `(S, S)` MSSA matrix.
fn solve_in_place(mut a: Matrix, b: &mut Matrix) -> Result<()> {
    let n = a.nrows();
    for c in 0..n {
        let p = (c..n)
            .max_by(|&i, &j| a[(i, c)].abs().total_cmp(&a[(j, c)].abs()))
            .unwrap();
        if a[(p, c)] == 0.0 {
            return Err(SsaError::LinAlg("Singular matrix".into()));
        }
        if p != c {
            for j in 0..n {
                let t = a[(c, j)];
                a[(c, j)] = a[(p, j)];
                a[(p, j)] = t;
            }
            for j in 0..b.ncols() {
                let t = b[(c, j)];
                b[(c, j)] = b[(p, j)];
                b[(p, j)] = t;
            }
        }
        for i in c + 1..n {
            let f = a[(i, c)] / a[(c, c)];
            if f != 0.0 {
                for j in c..n {
                    a[(i, j)] -= f * a[(c, j)];
                }
                for j in 0..b.ncols() {
                    b[(i, j)] -= f * b[(c, j)];
                }
            }
        }
    }
    for j in 0..b.ncols() {
        for i in (0..n).rev() {
            let mut acc = b[(i, j)];
            for k in i + 1..n {
                acc -= a[(i, k)] * b[(k, j)];
            }
            b[(i, j)] = acc / a[(i, i)];
        }
    }
    Ok(())
}

/// Minimum-norm least squares `argmin ||A X - B||` with numpy's default
/// cutoff `rcond = eps * max(M, N)` (as `numpy.linalg.lstsq(rcond=None)`).
pub fn lstsq(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Result<Matrix> {
    let svd = crate::svd::thin_svd(a)?;
    let s = &svd.sigma;
    let smax = s.iter().copied().fold(0.0, f64::max);
    let cutoff = f64::EPSILON * a.nrows().max(a.ncols()) as f64 * smax;
    let mut utb = mm(svd.u.transpose(), b); // (k, nrhs)
    for i in 0..utb.nrows() {
        let inv = if s[i] > cutoff { 1.0 / s[i] } else { 0.0 };
        for j in 0..utb.ncols() {
            utb[(i, j)] *= inv;
        }
    }
    Ok(mm(svd.v.as_ref(), utb.as_ref()))
}
