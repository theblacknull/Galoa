//! Truncated SVD by Golub–Kahan–Lanczos bidiagonalisation.
//!
//! Only matrix-vector products with the operator are needed, so together
//! with [`crate::hmat::HankelOp`] the leading `k` eigentriples of a series of
//! length `N` cost `O(m N log N + m^2 (L + K))` for `m` Lanczos steps,
//! instead of `O(L^2 K)` for a full SVD of the dense trajectory matrix.
//!
//! One-sided full reorthogonalisation (Simon & Zha, 2000): the basis of the
//! *shorter* side is reorthogonalised against all previous vectors
//! (classical Gram–Schmidt twice, as two matrix-vector products), which
//! keeps the bidiagonal matrix exact to machine precision; the singular
//! vectors of the longer side are then recomputed exactly as
//! `A^T u_i / sigma_i` (or `A v_i / sigma_i`), as TRLan/Rssa do.  The basis is
//! extended until the residual bound of every wanted triple,
//! `beta_m |e_m^T p_i|`, falls below `tol * sigma_1`.

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, Par};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::error::{Result, SsaError};
use crate::hmat::LinOp;

/// Growing orthonormal basis stored column-major.
struct Basis {
    data: Mat<f64>,
    len: usize,
}

impl Basis {
    fn new(n: usize, cap: usize) -> Self {
        Self { data: Mat::zeros(n, cap.max(1)), len: 0 }
    }
    fn push(&mut self, v: &[f64]) {
        if self.len == self.data.ncols() {
            let n = self.data.nrows();
            let mut bigger = Mat::<f64>::zeros(n, (2 * self.len).max(8));
            bigger.as_mut().subcols_mut(0, self.len).copy_from(self.data.as_ref().subcols(0, self.len));
            self.data = bigger;
        }
        self.data.col_mut(self.len).iter_mut().zip(v).for_each(|(d, s)| *d = *s);
        self.len += 1;
    }
    /// x <- x - B (B^T x), applied twice.
    fn reorth(&self, x: &mut [f64]) {
        if self.len == 0 {
            return;
        }
        let n = x.len();
        let b = self.data.as_ref().subcols(0, self.len);
        // memory-bound GEMV: sequential is fastest and leaves the cores to the FFTs
        let par = Par::Seq;
        for _ in 0..2 {
            let xv = faer::MatRef::from_column_major_slice(x, n, 1);
            let mut c = Mat::<f64>::zeros(self.len, 1);
            matmul(c.as_mut(), Accum::Replace, b.transpose(), xv, 1.0, par);
            let mut xm = Mat::<f64>::from_fn(n, 1, |i, _| x[i]);
            matmul(xm.as_mut(), Accum::Add, b, c.as_ref(), -1.0, par);
            x.iter_mut().zip(xm.col(0).iter()).for_each(|(d, s)| *d = *s);
        }
    }
    /// B[:, :m] @ p  -> (n, k)
    fn times(&self, p: &Mat<f64>, m: usize, k: usize) -> Mat<f64> {
        let mut out = Mat::<f64>::zeros(self.data.nrows(), k);
        matmul(
            out.as_mut(),
            Accum::Replace,
            self.data.as_ref().subcols(0, m),
            p.as_ref().submatrix(0, 0, m, k),
            1.0,
            faer::get_global_parallelism(),
        );
        out
    }
}

/// Leading `k` singular triplets of `op`: `(U (m, k), sigma, V (n, k))`.
pub fn lanczos_svd(op: &dyn LinOp, k: usize, tol: f64, seed: u64) -> Result<(Mat<f64>, Vec<f64>, Mat<f64>)> {
    if op.nrows() > op.ncols() {
        let (v, s, u) = lanczos_core(&Transposed(op), k, tol, seed)?;
        return Ok((u, s, v));
    }
    lanczos_core(op, k, tol, seed)
}

struct Transposed<'a>(&'a dyn LinOp);

impl LinOp for Transposed<'_> {
    fn nrows(&self) -> usize {
        self.0.ncols()
    }
    fn ncols(&self) -> usize {
        self.0.nrows()
    }
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        self.0.rmatvec(x, y)
    }
    fn rmatvec(&self, x: &[f64], y: &mut [f64]) {
        self.0.matvec(x, y)
    }
}

/// Lanczos with reorthogonalisation of the left (shorter, `m <= n`) side.
fn lanczos_core(op: &dyn LinOp, k: usize, tol: f64, seed: u64) -> Result<(Mat<f64>, Vec<f64>, Mat<f64>)> {
    let (m, n) = (op.nrows(), op.ncols());
    let kmax = m.min(n);
    if k == 0 || kmax == 0 {
        return Ok((Mat::zeros(m, 0), Vec::new(), Mat::zeros(n, 0)));
    }
    let k = k.min(kmax);
    let mut rng = StdRng::seed_from_u64(seed);
    let cap = (2 * k + 20).min(kmax);
    let mut us = Basis::new(m, cap);
    let mut alpha: Vec<f64> = Vec::new();
    let mut beta: Vec<f64> = Vec::new();

    let empty = Basis::new(0, 1);
    let mut v = random_unit(n, &mut rng, &empty);
    let mut steps = 0usize;
    let mut next_check = (k + 10).min(kmax);
    let mut beta_prev = 0.0;
    let mut u_prev: Option<Vec<f64>> = None;
    let mut scale_est = 0.0f64;
    loop {
        // u_j = A v_j - beta_{j-1} u_{j-1}
        let mut u = vec![0.0; m];
        op.matvec(&v, &mut u);
        if let Some(prev) = &u_prev {
            axpy(-beta_prev, prev, &mut u);
        }
        us.reorth(&mut u);
        let mut a = norm(&u);
        scale_est = scale_est.max(a);
        if a <= f64::EPSILON * 1e3 * scale_est.max(f64::MIN_POSITIVE) {
            u = random_unit(m, &mut rng, &us);
            a = 0.0;
        } else {
            scale(&mut u, 1.0 / a);
        }
        us.push(&u);
        alpha.push(a);
        steps += 1;

        // v_{j+1} = A^T u_j - alpha_j v_j
        let mut vn = vec![0.0; n];
        op.rmatvec(&u, &mut vn);
        axpy(-a, &v, &mut vn);
        let mut b = norm(&vn);
        scale_est = scale_est.max(b);
        let breakdown = b <= f64::EPSILON * 1e3 * scale_est.max(f64::MIN_POSITIVE);
        if breakdown && steps < kmax {
            vn = random_unit(n, &mut rng, &empty);
            b = 0.0;
        } else if !breakdown {
            scale(&mut vn, 1.0 / b);
        }
        beta.push(b);
        beta_prev = b;
        u_prev = Some(u);
        v = vn;

        if steps >= next_check || steps == kmax {
            let (p, s, q) = bidiag_svd(&alpha, &beta[..steps - 1])?;
            let bm = beta[steps - 1];
            let ok = steps == kmax || (0..k).all(|i| bm * p[(steps - 1, i)].abs() <= tol * s[0].max(f64::MIN_POSITIVE));
            if ok {
                if std::env::var("LANCZOS_DEBUG").is_ok() {
                    eprintln!("lanczos steps={steps}");
                }
                let um = us.times(&p, steps, k);
                let _ = &q;
                // right vectors recomputed exactly: v_i = A^T u_i / ||A^T u_i||
                let cols: Vec<(Vec<f64>, f64)> = {
                    use rayon::prelude::*;
                    (0..k)
                        .into_par_iter()
                        .map(|i| {
                            let ui: Vec<f64> = um.col(i).iter().copied().collect();
                            let mut z = vec![0.0; n];
                            op.rmatvec(&ui, &mut z);
                            let nz = norm(&z);
                            (z, nz)
                        })
                        .collect()
                };
                let vm = Mat::from_fn(n, k, |r, c| if cols[c].1 > 0.0 { cols[c].0[r] / cols[c].1 } else { 0.0 });
                return Ok((um, s[..k].to_vec(), vm));
            }
            next_check = (steps + (steps / 10).max(5)).min(kmax);
        }
    }
}

/// SVD of the upper bidiagonal matrix with diagonal `a` and superdiagonal `b`.
fn bidiag_svd(a: &[f64], b: &[f64]) -> Result<(Mat<f64>, Vec<f64>, Mat<f64>)> {
    let s = a.len();
    let bm = Mat::from_fn(s, s, |i, j| if i == j { a[i] } else if j == i + 1 { b[i] } else { 0.0 });
    let (p, sig, q) = crate::la::thin_svd(bm.as_ref())?;
    if sig.iter().any(|v| !v.is_finite()) {
        return Err(SsaError::LinAlg("Lanczos bidiagonal SVD failed".into()));
    }
    Ok((p, sig, q))
}

fn random_unit(n: usize, rng: &mut StdRng, basis: &Basis) -> Vec<f64> {
    let mut v: Vec<f64> = (0..n).map(|_| StandardNormal.sample(rng)).collect();
    basis.reorth(&mut v);
    let nv = norm(&v);
    scale(&mut v, 1.0 / nv);
    v
}

#[inline]
fn norm(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}
#[inline]
fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += alpha * xi;
    }
}
#[inline]
fn scale(x: &mut [f64], a: f64) {
    x.iter_mut().for_each(|v| *v *= a);
}


/// Leading `k` eigenpairs (largest algebraic eigenvalues, descending) of a
/// symmetric operator by Lanczos with full reorthogonalisation (Rssa
/// `trlan.eigen`).  Converged when `beta_m |s_{m,i}| <= tol * max |lambda|`.
pub fn lanczos_eigh(op: &dyn LinOp, k: usize, tol: f64, seed: u64) -> Result<(Vec<f64>, Mat<f64>)> {
    let n = op.nrows();
    if k == 0 || n == 0 {
        return Ok((Vec::new(), Mat::zeros(n, 0)));
    }
    let k = k.min(n);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut qs = Basis::new(n, (2 * k + 20).min(n));
    let empty = Basis::new(0, 1);
    let mut q = random_unit(n, &mut rng, &empty);
    let mut alpha: Vec<f64> = Vec::new();
    let mut beta: Vec<f64> = Vec::new();
    let mut q_prev: Option<Vec<f64>> = None;
    let mut next_check = (k + 10).min(n);
    let mut scale_est = 0.0f64;
    loop {
        qs.push(&q);
        let mut w = vec![0.0; n];
        op.matvec(&q, &mut w);
        if let (Some(p), Some(b)) = (&q_prev, beta.last()) {
            axpy(-b, p, &mut w);
        }
        let a: f64 = q.iter().zip(&w).map(|(x, y)| x * y).sum();
        axpy(-a, &q, &mut w);
        qs.reorth(&mut w);
        alpha.push(a);
        let mut b = norm(&w);
        scale_est = scale_est.max(a.abs()).max(b);
        let steps = alpha.len();
        if b <= f64::EPSILON * 1e3 * scale_est.max(f64::MIN_POSITIVE) && steps < n {
            w = random_unit(n, &mut rng, &qs);
            b = 0.0;
        } else if steps < n {
            scale(&mut w, 1.0 / b);
        }
        beta.push(b);
        if steps >= next_check || steps == n {
            let t = Mat::from_fn(steps, steps, |i, j| {
                if i == j {
                    alpha[i]
                } else if i + 1 == j {
                    beta[i]
                } else if j + 1 == i {
                    beta[j]
                } else {
                    0.0
                }
            });
            let (lam, s) = crate::la::eigh(t.as_ref())?; // ascending
            let lmax = lam.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let top: Vec<usize> = (0..k).map(|i| steps - 1 - i).collect();
            let ok = steps == n || top.iter().all(|&c| (beta[steps - 1] * s[(steps - 1, c)]).abs() <= tol * lmax.max(f64::MIN_POSITIVE));
            if ok {
                let p = Mat::from_fn(steps, k, |i, j| s[(i, top[j])]);
                let vecs = qs.times(&p, steps, k);
                return Ok((top.iter().map(|&c| lam[c]).collect(), vecs));
            }
            next_check = (steps + (steps / 10).max(5)).min(n);
        }
        q_prev = Some(std::mem::replace(&mut q, w));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hmat::{DenseOp, HankelOp};

    #[test]
    fn matches_dense_svd() {
        let x: Vec<f64> = (0..800).map(|t| (t as f64 * 0.05).sin() * 3.0 + (t as f64 * 0.31).cos() + ((t * 7919 % 97) as f64 / 97.0)).collect();
        let xm = Mat::from_fn(800, 1, |i, _| x[i]);
        let l = 300;
        let op = HankelOp::new(xm.as_ref(), l, &[1.0]);
        let (u, s, v) = lanczos_svd(&op, 10, 1e-12, 1).unwrap();
        let dense = crate::ssa::construct_trajectory_matrix(xm.as_ref(), l, &[1.0]);
        let (ud, sd, vd) = crate::la::thin_svd(dense.as_ref()).unwrap();
        for i in 0..10 {
            assert!((s[i] - sd[i]).abs() <= 1e-10 * sd[0], "sigma {i}: {} vs {}", s[i], sd[i]);
            let c: f64 = (0..l).map(|r| u[(r, i)] * ud[(r, i)]).sum();
            let cv: f64 = (0..v.nrows()).map(|r| v[(r, i)] * vd[(r, i)]).sum();
            assert!((c.abs() - 1.0).abs() < 1e-8 && (cv.abs() - 1.0).abs() < 1e-8, "vector {i}: {c} {cv}");
        }
        let small = Mat::from_fn(20, 30, |i, j| ((i + 1) * (j + 2)) as f64);
        let (_, s2, _) = lanczos_svd(&DenseOp(small.as_ref()), 3, 1e-12, 2).unwrap();
        let (_, sd2, _) = crate::la::thin_svd(small.as_ref()).unwrap();
        assert!((s2[0] - sd2[0]).abs() < 1e-9 * sd2[0]);
        assert!(s2[1].abs() < 1e-9 * sd2[0]);
    }
}
