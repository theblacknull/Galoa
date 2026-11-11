//! Nested oblique decompositions (Rssa `iossa`, `fossa`, `eossa`, `owcor`).
//!
//! Each method rotates the eigentriples of a set of components so that
//! mixed components separate; the rotated triples replace the original
//! ones in a new model, and the method returns the resulting groups.

use faer::Mat;

use super::common::{auto_strategy, consecutive_groups, cov2cor, hclust_complete_cut, pseudo_inverse, refit, split_by_label, Group};
use super::parest::{eigen_r_order, shift_matrix_rows, SolveMethod};
use crate::error::{Result, SsaError};
use crate::hmat::LinOp;
use crate::matrix::{mm, select_cols, Matrix};
use crate::ssa::{Complex64, SsaModel};

/// Result of an oblique decomposition.
#[derive(Debug, Clone)]
pub struct OssaResult {
    /// Model whose triples at `set` were replaced.
    pub model: SsaModel,
    /// Groups found / refined (component indices of `model`).
    pub groups: Vec<Group>,
    /// Components touched by the method (Rssa `ossa.set`).
    pub set: Vec<usize>,
    pub iterations: usize,
    pub converged: bool,
}

/// Rssa `orthogonalize(Y, Z, sigma, side = "bi")`: SVD of `Y diag(s) Z^T`.
pub fn orthogonalize(y: &Matrix, z: &Matrix, sigma: &[f64]) -> Result<(Vec<f64>, Matrix, Matrix)> {
    let qy = y.qr();
    let qz = z.qr();
    let (ry, rz) = (qy.thin_R().to_owned(), qz.thin_R().to_owned());
    let r = sigma.len();
    let srz = Mat::from_fn(r, r, |i, j| sigma[i] * rz[(j, i)]);
    let m = mm(ry.as_ref(), srz.as_ref());
    let (u, d, v) = crate::la::thin_svd(m.as_ref())?;
    Ok((d, mm(qy.compute_thin_Q().as_ref(), u.as_ref()), mm(qz.compute_thin_Q().as_ref(), v.as_ref())))
}

fn sorted_set(groups: &[Group]) -> Vec<usize> {
    let mut idx: Vec<usize> = groups.iter().flatten().copied().collect();
    idx.sort_unstable();
    idx.dedup();
    idx
}

impl SsaModel {
    /// Rssa `.get.orth.triples`.
    fn orth_triples(&self, idx: &[usize], orth: bool) -> Result<(Vec<f64>, Matrix, Matrix)> {
        self.check_indices(idx)?;
        let sigma: Vec<f64> = idx.iter().map(|&i| self.sigma[i]).collect();
        let u = select_cols(self.u.as_ref(), idx);
        let v = select_cols(self.v.as_ref(), idx);
        if orth { orthogonalize(&u, &v, &sigma) } else { Ok((sigma, u, v)) }
    }

    /// Copy of the model with triples at `idx` replaced by the columns of
    /// `y`, `z` (normalised; Rssa `.save.oblique.decomposition`).
    pub fn with_oblique_triples(&self, nosigma: &[f64], y: &Matrix, z: &Matrix, idx: &[usize]) -> SsaModel {
        let mut m = self.clone();
        for (c, &i) in idx.iter().enumerate() {
            let yn = y.col(c).norm_l2();
            let zn = z.col(c).norm_l2();
            m.sigma[i] = nosigma[c] * yn * zn;
            for r in 0..m.u.nrows() {
                m.u[(r, i)] = y[(r, c)] / yn;
            }
            for r in 0..m.v.nrows() {
                m.v[(r, i)] = z[(r, c)] / zn;
            }
        }
        let e = m.total_energy();
        m.rel_contribution = m.sigma.iter().map(|s| s * s / e * 100.0).collect();
        m.cumsum_contr = m.rel_contribution.iter().scan(0.0, |a, v| { *a += v; Some(*a) }).collect();
        m
    }

    /// Filter-adjusted O-SSA / DerivSSA (Rssa `fossa`) with a 1D filter
    /// applied to the factor vectors (default `[-1, 1]`, i.e. differences),
    /// `gamma = inf` and `normalize = true` by default.
    pub fn fossa(&self, nested_groups: &[Group], filter: &[f64], gamma: f64, normalize: bool) -> Result<OssaResult> {
        let idx = sorted_set(nested_groups);
        let (os, u, v) = self.orth_triples(&idx, true)?;
        let r = idx.len();
        let zz = Mat::from_fn(v.nrows(), r, |i, j| v[(i, j)] * os[j]);
        let y = if normalize { v.clone() } else { zz.clone() };
        let (k, s, fl) = (self.k, self.s, filter.len());
        let kk = k - fl + 1;
        let fy = Mat::from_fn(kk * s, r, |row, j| {
            let (c, t) = (row / kk, row % kk);
            (0..fl).map(|q| filter[q] * y[(c * k + t + q, j)]).sum()
        });
        let newv = if gamma.is_infinite() {
            fy
        } else {
            Mat::from_fn(y.nrows() + fy.nrows(), r, |i, j| if i < y.nrows() { y[(i, j)] } else { gamma * fy[(i - y.nrows(), j)] })
        };
        let g = mm(newv.transpose(), newv.as_ref());
        let (_, ev) = crate::la::eigh(g.as_ref())?;
        let e = Mat::from_fn(r, r, |i, j| ev[(i, r - 1 - j)]);
        let (un, zn) = if normalize {
            let us = Mat::from_fn(u.nrows(), r, |i, j| u[(i, j)] * os[j]);
            (mm(us.as_ref(), e.as_ref()), mm(v.as_ref(), e.as_ref()))
        } else {
            (mm(u.as_ref(), e.as_ref()), mm(zz.as_ref(), e.as_ref()))
        };
        let model = self.with_oblique_triples(&vec![1.0; r], &un, &zn, &idx);
        Ok(OssaResult { model, groups: nested_groups.to_vec(), set: idx, iterations: 0, converged: true })
    }

    /// Iterative O-SSA (Rssa `iossa`, defaults `tol = 1e-5`, `kappa = 2`,
    /// `maxiter = 100`, `norm = sqrt(mean(x^2))`, `kappa.balance = 0.5`).
    pub fn iossa(&self, nested_groups: &[Group], tol: f64, kappa: Option<f64>, maxiter: usize, kappa_balance: f64) -> Result<OssaResult> {
        let ng: Vec<Group> = nested_groups.iter().map(|g| { let mut h = g.clone(); h.dedup(); h }).collect();
        let total: usize = ng.iter().map(|g| g.len()).sum();
        let idx = sorted_set(&ng);
        if idx.len() != total {
            return Err(SsaError::InvalidArgument("Intersected nested groups".into()));
        }
        let mut fs: Vec<Matrix> = ng.iter().map(|g| self.reconstruct_components(g)).collect::<Result<_>>()?;
        let (osigma, u, v) = self.orth_triples(&idx, true)?;
        let ranks: Vec<usize> = ng.iter().map(|g| g.len()).collect();
        let groups_pos: Vec<Group> = if kappa.is_some() {
            consecutive_groups(&ranks)
        } else {
            ng.iter().map(|g| g.iter().map(|c| idx.iter().position(|x| x == c).unwrap()).collect()).collect()
        };
        let mut converged = false;
        let mut iterations = 0;
        let (mut sigma, mut yy, mut zz) = (osigma.clone(), u.clone(), v.clone());
        for iter in 1..=maxiter {
            iterations = iter;
            let mut osig: Vec<Vec<f64>> = Vec::new();
            let mut us: Vec<Matrix> = Vec::new();
            let mut vs: Vec<Matrix> = Vec::new();
            for (i, f) in fs.iter().enumerate() {
                let strat = auto_strategy(self.l, ranks[i]);
                let cur = refit(strat.as_ref(), f.clone(), self.l, self.weights.clone())?;
                let (s, uu, vv) = cur.orth_triples(&(0..ranks[i]).collect::<Vec<_>>(), true)?;
                osig.push(s);
                us.push(uu);
                vs.push(vv);
            }
            if let Some(kp) = kappa {
                for i in 0..ng.len().saturating_sub(1) {
                    let div = osig[i][ranks[i] - 1] / osig[i + 1][0];
                    if div < kp {
                        let mul = kp / div;
                        osig[i + 1].iter_mut().for_each(|s| *s /= mul);
                        let (a, b) = (mul.powf(kappa_balance), mul.powf(1.0 - kappa_balance));
                        us[i + 1].col_iter_mut().for_each(|c| c.iter_mut().for_each(|x| *x *= a));
                        vs[i + 1].col_iter_mut().for_each(|c| c.iter_mut().for_each(|x| *x *= b));
                    }
                }
            }
            let bas_u = hcat(&us);
            let bas_v = hcat(&vs);
            let (s2, y2, z2) = svd2lrsvd(&osigma, &u, &v, &bas_u, &bas_v)?;
            sigma = s2;
            yy = y2;
            zz = z2;
            // reconstruct the nested groups from the new triples only
            let tmp = SsaModel { u: yy.clone(), v: zz.clone(), sigma: sigma.clone(), d: sigma.len(), ..self.clone() };
            let fs_new: Vec<Matrix> = groups_pos.iter().map(|g| tmp.reconstruct_components(g)).collect::<Result<_>>()?;
            let deltas: Vec<f64> = fs_new
                .iter()
                .zip(&fs)
                .map(|(a, b)| {
                    let (fa, fb) = (super::common::flatten(a), super::common::flatten(b));
                    (fa.iter().zip(&fb).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / fa.len() as f64).sqrt()
                })
                .collect();
            fs = fs_new;
            if deltas.iter().copied().fold(0.0, f64::max) < tol {
                converged = true;
                break;
            }
        }
        let model = self.with_oblique_triples(&sigma, &yy, &zz, &idx);
        let groups = groups_pos.iter().map(|g| g.iter().map(|&p| idx[p]).collect()).collect();
        Ok(OssaResult { model, groups, set: idx, iterations, converged })
    }

    /// ESPRIT-based O-SSA (Rssa `eossa`, column subspace): rotates the set
    /// to the (real) ESPRIT basis and clusters the roots into `k` groups.
    pub fn eossa(&self, nested_groups: &[Group], k: usize, method: SolveMethod) -> Result<OssaResult> {
        let idx = sorted_set(nested_groups);
        let (os, u, v) = self.orth_triples(&idx, false)?;
        let r = idx.len();
        let vs = Mat::from_fn(v.nrows(), r, |i, j| v[(i, j)] * os[j]);
        let left: Vec<usize> = (0..self.l - 1).collect();
        let right: Vec<usize> = (1..self.l).collect();
        let z = shift_matrix_rows(&u, &left, &right, method)?;
        let (roots, vecs) = eigen_with_vectors(&z)?;
        // order by |Arg| (stable)
        let mut ord: Vec<usize> = (0..r).collect();
        ord.sort_by(|&a, &b| roots[a].im.atan2(roots[a].re).abs().total_cmp(&roots[b].im.atan2(roots[b].re).abs()));
        let roots: Vec<Complex64> = ord.iter().map(|&i| roots[i]).collect();
        let maxk = roots.iter().filter(|z| z.im >= -f64::EPSILON).count();
        if k > maxk {
            return Err(SsaError::InvalidArgument(format!(
                "k exceeds the number of different ESPRIT roots with non-negative imaginary parts ({maxk})"
            )));
        }
        let pts: Vec<(f64, f64)> = roots.iter().map(|z| (z.re, z.im.abs())).collect();
        let d = Mat::from_fn(r, r, |i, j| ((pts[i].0 - pts[j].0).powi(2) + (pts[i].1 - pts[j].1).powi(2)).sqrt());
        let labels = hclust_complete_cut(&d, k);
        let pos_groups = split_by_label(&(0..r).collect::<Vec<_>>(), &labels);
        let mut c = Mat::<f64>::zeros(r, r);
        for g in &pos_groups {
            let m = Mat::from_fn(r, 2 * g.len(), |i, j| {
                let z = vecs[(i, ord[g[j % g.len()]])];
                if j < g.len() { z.re } else { z.im }
            });
            let (lu, _, _) = crate::la::thin_svd(m.as_ref())?;
            for (jj, &col) in g.iter().enumerate() {
                for i in 0..r {
                    c[(i, col)] = lu[(i, jj)];
                }
            }
        }
        let un = mm(u.as_ref(), c.as_ref());
        // V C^{-T}
        let cinv_vt = crate::ssa::lstsq(c.as_ref(), vs.transpose())?; // (r, KS)
        let vn = cinv_vt.transpose().to_owned();
        let model = self.with_oblique_triples(&vec![1.0; r], &un, &vn, &idx);
        let groups = pos_groups.iter().map(|g| g.iter().map(|&p| idx[p]).collect()).collect();
        Ok(OssaResult { model, groups, set: idx, iterations: 0, converged: true })
    }

    /// Oblique w-correlations (Rssa `owcor`) of groups drawn from `set`.
    pub fn owcor(&self, groups: &[Group], set: &[usize]) -> Result<Matrix> {
        let lm = pseudo_inverse(&select_cols(self.u.as_ref(), set))?; // (r, L)
        let rm = pseudo_inverse(&select_cols(self.v.as_ref(), set))?; // (r, KS)
        let r = set.len();
        let mut mx = Mat::<f64>::zeros(r * r, groups.len());
        for (gi, g) in groups.iter().enumerate() {
            let f = self.reconstruct_components(g)?;
            let op = crate::hmat::HankelOp::new(f.as_ref(), self.l, &self.weights);
            // T = TF RM^T  (L, r)
            let mut t = Mat::<f64>::zeros(self.l, r);
            let mut buf = vec![0.0; self.l];
            for j in 0..r {
                let row: Vec<f64> = (0..rm.ncols()).map(|c| rm[(j, c)]).collect();
                op.matvec(&row, &mut buf);
                for i in 0..self.l {
                    t[(i, j)] = buf[i];
                }
            }
            let prod = mm(lm.as_ref(), t.as_ref());
            for j in 0..r {
                for i in 0..r {
                    mx[(i + j * r, gi)] = prod[(i, j)];
                }
            }
        }
        Ok(cov2cor(&mm(mx.transpose(), mx.as_ref())))
    }
}

/// Rssa `svd2LRsvd(..., need.project = TRUE, fast = TRUE)`.
pub fn svd2lrsvd(d: &[f64], u: &Matrix, v: &Matrix, bas_l: &Matrix, bas_r: &Matrix) -> Result<(Vec<f64>, Matrix, Matrix)> {
    let r = d.len();
    let ubl = mm(u.transpose(), bas_l.as_ref());
    let vbr = mm(v.transpose(), bas_r.as_ref());
    let scaled = Mat::from_fn(r, r, |i, j| vbr[(i, j)] / d[i]);
    let m = mm(ubl.transpose(), scaled.as_ref());
    let (du, ds, dv) = crate::la::thin_svd(m.as_ref())?;
    let sig: Vec<f64> = ds.iter().rev().map(|x| 1.0 / x).collect();
    let du = Mat::from_fn(r, r, |i, j| du[(i, r - 1 - j)]);
    let dv = Mat::from_fn(r, r, |i, j| dv[(i, r - 1 - j)]);
    let bl = mm(u.as_ref(), ubl.as_ref());
    let br = mm(v.as_ref(), vbr.as_ref());
    Ok((sig, mm(bl.as_ref(), du.as_ref()), mm(br.as_ref(), dv.as_ref())))
}

pub fn hcat(ms: &[Matrix]) -> Matrix {
    let rows = ms[0].nrows();
    let cols: usize = ms.iter().map(|m| m.ncols()).sum();
    let mut out = Mat::<f64>::zeros(rows, cols);
    let mut c0 = 0;
    for m in ms {
        for j in 0..m.ncols() {
            for i in 0..rows {
                out[(i, c0 + j)] = m[(i, j)];
            }
        }
        c0 += m.ncols();
    }
    out
}

/// Eigenvalues (R order) and matching eigenvectors of a real matrix.
pub fn eigen_with_vectors(z: &Matrix) -> Result<(Vec<Complex64>, Mat<Complex64>)> {
    let e = z.eigen().map_err(|e| SsaError::LinAlg(format!("eigen decomposition failed: {e:?}")))?;
    let vals: Vec<Complex64> = e.S().column_vector().iter().copied().collect();
    let target = eigen_r_order(z)?;
    let n = vals.len();
    let mut used = vec![false; n];
    let mut order = Vec::with_capacity(n);
    for t in &target {
        let j = (0..n).filter(|&j| !used[j]).min_by(|&a, &b| (vals[a] - t).norm().total_cmp(&(vals[b] - t).norm())).unwrap();
        used[j] = true;
        order.push(j);
    }
    let vecs = Mat::from_fn(n, n, |i, j| e.U()[(i, order[j])]);
    Ok((order.iter().map(|&j| vals[j]).collect(), vecs))
}
