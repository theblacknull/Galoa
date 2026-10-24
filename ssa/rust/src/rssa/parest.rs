//! Parameter estimation (Rssa `parestimate`): ESPRIT with LS or TLS shift
//! equations, the "pairs" method, and conversion of roots to periods,
//! frequencies, moduli and rates.

use faer::Mat;

use super::common::{mad, median};
use crate::error::{Result, SsaError};
use crate::matrix::{select_cols, Matrix};
use crate::ssa::{Complex64, SsaModel};

/// Shift-equation solver (Rssa `solve.method`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveMethod {
    Ls,
    Tls,
}

/// Subspace used by ESPRIT (Rssa `subspace`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subspace {
    Column,
    Row,
}

/// Rssa `roots2pars`.
#[derive(Debug, Clone, PartialEq)]
pub struct Params {
    pub roots: Vec<Complex64>,
    pub periods: Vec<f64>,
    pub frequencies: Vec<f64>,
    pub moduli: Vec<f64>,
    pub rates: Vec<f64>,
}

pub fn roots2pars(roots: Vec<Complex64>) -> Params {
    let arg: Vec<f64> = roots.iter().map(|z| z.im.atan2(z.re)).collect();
    Params {
        periods: arg
            .iter()
            .map(|a| if a.abs() < f64::EPSILON.sqrt() { f64::INFINITY } else { 2.0 * std::f64::consts::PI / a })
            .collect(),
        frequencies: arg.iter().map(|a| a / (2.0 * std::f64::consts::PI)).collect(),
        moduli: roots.iter().map(|z| z.norm()).collect(),
        rates: roots.iter().map(|z| z.norm().ln()).collect(),
        roots,
    }
}

/// Total least squares `A X = B` (Rssa `tls.solve`).
pub fn tls_solve(a: &Matrix, b: &Matrix) -> Result<Matrix> {
    let r = a.ncols();
    let ab = Mat::from_fn(a.nrows(), 2 * r, |i, j| if j < r { a[(i, j)] } else { b[(i, j - r)] });
    let (_, _, v) = crate::la::thin_svd(ab.as_ref())?;
    if v.ncols() < 2 * r {
        return Err(SsaError::LinAlg("not enough rows for TLS".into()));
    }
    // solve t(V11) X = t(V21)
    let v11t = Mat::from_fn(r, r, |i, j| v[(j, i)]);
    let v21t = Mat::from_fn(r, r, |i, j| v[(r + j, i)]);
    crate::ssa::lstsq(v11t.as_ref(), v21t.as_ref())
}

/// Shift matrix `Z` with `U[right] ~ U[left] Z` (Rssa `.shift.matrix`).
pub fn shift_matrix_rows(u: &Matrix, left: &[usize], right: &[usize], method: SolveMethod) -> Result<Matrix> {
    let a = Mat::from_fn(left.len(), u.ncols(), |i, j| u[(left[i], j)]);
    let b = Mat::from_fn(right.len(), u.ncols(), |i, j| u[(right[i], j)]);
    match method {
        SolveMethod::Ls => crate::ssa::lstsq(a.as_ref(), b.as_ref()),
        SolveMethod::Tls => tls_solve(&a, &b),
    }
}

/// Eigenvalues of a real matrix in R `eigen()` order (decreasing modulus).
pub fn eigen_r_order(z: &Matrix) -> Result<Vec<Complex64>> {
    let mut ev: Vec<Complex64> =
        crate::la::eigenvalues(z.as_ref())?.into_iter().map(|(a, b)| Complex64::new(a, if b == 0.0 { 0.0 } else { b })).collect();
    // R: decreasing Mod; conjugate pairs keep (+im, -im) order
    ev.sort_by(|a, b| b.norm().total_cmp(&a.norm()).then(b.im.total_cmp(&a.im)));
    Ok(ev)
}

impl SsaModel {
    /// ESPRIT estimate of the signal roots of a group (Rssa
    /// `parestimate(method = "esprit")`).  Roots are in R `eigen` order
    /// (decreasing modulus); `normalize` puts them on the unit circle.
    pub fn parestimate_esprit(&self, group: &[usize], subspace: Subspace, method: SolveMethod, normalize: bool) -> Result<Params> {
        self.check_indices(group)?;
        let (span, block, nblocks) = match subspace {
            Subspace::Column => (select_cols(self.u.as_ref(), group), self.l, 1),
            Subspace::Row => (select_cols(self.v.as_ref(), group), self.k, self.s),
        };
        let mut left = Vec::new();
        let mut right = Vec::new();
        for b in 0..nblocks {
            for i in 0..block - 1 {
                left.push(b * block + i);
                right.push(b * block + i + 1);
            }
        }
        let z = shift_matrix_rows(&span, &left, &right, method)?;
        let mut roots = eigen_r_order(&z)?;
        if normalize {
            roots.iter_mut().for_each(|r| *r /= r.norm());
        }
        Ok(roots2pars(roots))
    }

    /// "pairs" estimate for a group of two components (Rssa
    /// `parestimate(method = "pairs")`).  Returns the parameters and the
    /// relative MAD of the per-lag period estimates (Rssa warns above 0.3).
    pub fn parestimate_pairs(&self, group: &[usize], subspace: Subspace, normalize: bool) -> Result<(Params, f64)> {
        if group.len() != 2 {
            return Err(SsaError::InvalidArgument("can estimate for pair of eigenvectors only using `pairs' method".into()));
        }
        self.check_indices(group)?;
        let span = match subspace {
            Subspace::Column => select_cols(self.u.as_ref(), group),
            Subspace::Row => {
                if self.s > 1 {
                    return Err(SsaError::InvalidArgument("row space `pairs' is not implemented for MSSA".into()));
                }
                select_cols(self.v.as_ref(), group)
            }
        };
        let n = span.nrows();
        // U1 = diff(U[-1, ]), U2 = diff(U[-n, ])
        let mut scos = Vec::with_capacity(n - 2);
        for i in 0..n - 2 {
            let u1 = [span[(i + 2, 0)] - span[(i + 1, 0)], span[(i + 2, 1)] - span[(i + 1, 1)]];
            let u2 = [span[(i + 1, 0)] - span[(i, 0)], span[(i + 1, 1)] - span[(i, 1)]];
            let d = u1[0] * u2[0] + u1[1] * u2[1];
            scos.push(d / (u1[0].hypot(u1[1])) / (u2[0].hypot(u2[1])));
        }
        let per: Vec<f64> = scos.iter().map(|c| 2.0 * std::f64::consts::PI / c.acos()).collect();
        let mres = mad(&per) / median(&per);
        let ang = median(&scos).acos();
        let mut r = Complex64::new(ang.cos(), ang.sin());
        if normalize {
            r /= r.norm();
        }
        Ok((roots2pars(vec![r]), mres))
    }
}
