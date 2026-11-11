//! Nested oblique decompositions for shaped / nD-SSA and MSSA of series of
//! different lengths (Rssa `fossa`, `iossa`, `eossa`, `owcor` on `nd.ssa`).

use faer::Mat;

use super::common::{consecutive_groups, hclust_complete_cut, pseudo_inverse, split_by_label, Group};
use super::nd::{NdMethod, NdSsa};
use super::ossa::{eigen_with_vectors, hcat, orthogonalize, svd2lrsvd};
use super::parest::{shift_matrix_rows, SolveMethod, Subspace};
use crate::error::{Result, SsaError};
use crate::matrix::{mm, select_cols, Matrix};
use crate::ssa::Complex64;

/// Result of an oblique decomposition of an [`NdSsa`].
#[derive(Debug, Clone)]
pub struct NdOssaResult {
    pub model: NdSsa,
    pub groups: Vec<Group>,
    pub set: Vec<usize>,
    pub iterations: usize,
    pub converged: bool,
}

fn sorted_set(groups: &[Group]) -> Vec<usize> {
    let mut idx: Vec<usize> = groups.iter().flatten().copied().collect();
    idx.sort_unstable();
    idx.dedup();
    idx
}

impl NdSsa {
    fn orth_triples(&self, idx: &[usize], orth: bool) -> Result<(Vec<f64>, Matrix, Matrix)> {
        self.check(idx)?;
        let sigma: Vec<f64> = idx.iter().map(|&i| self.sigma[i]).collect();
        let u = select_cols(self.u.as_ref(), idx);
        let v = select_cols(self.v.as_ref(), idx);
        if orth { orthogonalize(&u, &v, &sigma) } else { Ok((sigma, u, v)) }
    }

    /// Copy with the triples at `idx` replaced (Rssa `.save.oblique.decomposition`).
    pub fn with_oblique_triples(&self, nosigma: &[f64], y: &Matrix, z: &Matrix, idx: &[usize]) -> NdSsa {
        let mut m = self.clone();
        for (c, &i) in idx.iter().enumerate() {
            let (yn, zn) = (y.col(c).norm_l2(), z.col(c).norm_l2());
            m.sigma[i] = nosigma[c] * yn * zn;
            for r in 0..m.u.nrows() {
                m.u[(r, i)] = y[(r, c)] / yn;
            }
            for r in 0..m.v.nrows() {
                m.v[(r, i)] = z[(r, c)] / zn;
            }
        }
        m
    }

    /// Dimensions along which factor vectors are filtered / shifted
    /// (one for MSSA-type fields, all otherwise).
    fn eff_dims(&self) -> Vec<usize> {
        if self.spec.mssa {
            vec![0]
        } else {
            (0..self.n.len()).filter(|&d| self.l[d] > 1 || self.k[d] > 1).collect()
        }
    }

    /// F-OSSA / DerivSSA (Rssa `fossa`): `filter` applied along every
    /// dimension of the factor vectors.
    pub fn fossa(&self, nested_groups: &[Group], filter: &[f64], gamma: f64, normalize: bool) -> Result<NdOssaResult> {
        let idx = sorted_set(nested_groups);
        let (os, u, v) = self.orth_triples(&idx, true)?;
        let r = idx.len();
        let zz = Mat::from_fn(v.nrows(), r, |i, j| v[(i, j)] * os[j]);
        let y = if normalize { v.clone() } else { zz.clone() };
        let fy = self.filter_factors(&y, filter, &self.eff_dims());
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
        Ok(NdOssaResult { model, groups: nested_groups.to_vec(), set: idx, iterations: 0, converged: true })
    }

    /// Iterative O-SSA (Rssa `iossa`).
    pub fn iossa(&self, nested_groups: &[Group], tol: f64, kappa: Option<f64>, maxiter: usize, kappa_balance: f64) -> Result<NdOssaResult> {
        let ng: Vec<Group> = nested_groups.iter().map(|g| { let mut h = g.clone(); h.dedup(); h }).collect();
        let idx = sorted_set(&ng);
        if idx.len() != ng.iter().map(|g| g.len()).sum::<usize>() {
            return Err(SsaError::InvalidArgument("Intersected nested groups".into()));
        }
        let mask: Vec<bool> = self.weights.iter().map(|&w| w > 0).collect();
        let mut fs: Vec<Vec<f64>> = ng.iter().map(|g| self.reconstruct(g)).collect::<Result<_>>()?;
        let (osigma, u, v) = self.orth_triples(&idx, true)?;
        let ranks: Vec<usize> = ng.iter().map(|g| g.len()).collect();
        let pos: Vec<Group> = if kappa.is_some() {
            consecutive_groups(&ranks)
        } else {
            ng.iter().map(|g| g.iter().map(|c| idx.iter().position(|x| x == c).unwrap()).collect()).collect()
        };
        let (mut sigma, mut yy, mut zz) = (osigma.clone(), u.clone(), v.clone());
        let (mut iterations, mut converged) = (0, false);
        for iter in 1..=maxiter {
            iterations = iter;
            let mut osig = Vec::new();
            let mut us = Vec::new();
            let mut vs = Vec::new();
            for (i, f) in fs.iter().enumerate() {
                let filled: Vec<f64> = f.iter().map(|v| if v.is_finite() { *v } else { 0.0 }).collect();
                let cur = self.refit_with(&filled, NdMethod::Auto(ranks[i]))?;
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
            let (s2, y2, z2) = svd2lrsvd(&osigma, &u, &v, &hcat(&us), &hcat(&vs))?;
            sigma = s2;
            yy = y2;
            zz = z2;
            let mut tmp = self.clone();
            tmp.u = yy.clone();
            tmp.v = zz.clone();
            tmp.sigma = sigma.clone();
            tmp.d = sigma.len();
            let fs_new: Vec<Vec<f64>> = pos.iter().map(|g| tmp.reconstruct(g)).collect::<Result<_>>()?;
            let delta = fs_new
                .iter()
                .zip(&fs)
                .map(|(a, b)| {
                    let (s, c) = a.iter().zip(b).zip(&mask).filter(|(_, m)| **m).fold((0.0, 0usize), |(s, c), ((x, y), _)| (s + (x - y).powi(2), c + 1));
                    (s / c as f64).sqrt()
                })
                .fold(0.0, f64::max);
            fs = fs_new;
            if delta < tol {
                converged = true;
                break;
            }
        }
        let model = self.with_oblique_triples(&sigma, &yy, &zz, &idx);
        let groups = pos.iter().map(|g| g.iter().map(|&p| idx[p]).collect()).collect();
        Ok(NdOssaResult { model, groups, set: idx, iterations, converged })
    }

    /// ESPRIT-based O-SSA (Rssa `eossa`, column subspace).
    pub fn eossa(&self, nested_groups: &[Group], k: usize, method: SolveMethod, dimensions: Option<&[usize]>) -> Result<NdOssaResult> {
        let idx = sorted_set(nested_groups);
        let (os, u, v) = self.orth_triples(&idx, false)?;
        let r = idx.len();
        let vs = Mat::from_fn(v.nrows(), r, |i, j| v[(i, j)] * os[j]);
        let all = self.eff_dims();
        let dims = dimensions.unwrap_or(&all);
        let zs: Vec<Matrix> = dims
            .iter()
            .map(|&d| {
                let (l, rr) = self.shift_pairs(Subspace::Column, d);
                shift_matrix_rows(&u, &l, &rr, method)
            })
            .collect::<Result<_>>()?;
        let z = if zs.len() == 1 {
            zs[0].clone()
        } else {
            let nb = zs.len() - 1;
            let mut b: Vec<f64> = (1..=nb).map(|i| 8f64.powi(i as i32)).collect();
            b.insert(0, 1.0 - b.iter().sum::<f64>());
            Mat::from_fn(r, r, |i, j| zs.iter().zip(&b).map(|(m, w)| w * m[(i, j)]).sum())
        };
        let (roots, vecs) = eigen_with_vectors(&z)?;
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
        let pg = split_by_label(&(0..r).collect::<Vec<_>>(), &hclust_complete_cut(&d, k));
        let mut c = Mat::<f64>::zeros(r, r);
        for g in &pg {
            let m = Mat::from_fn(r, 2 * g.len(), |i, j| {
                let zz = vecs[(i, ord[g[j % g.len()]])];
                if j < g.len() { zz.re } else { zz.im }
            });
            let (lu, _, _) = crate::la::thin_svd(m.as_ref())?;
            for (jj, &col) in g.iter().enumerate() {
                for i in 0..r {
                    c[(i, col)] = lu[(i, jj)];
                }
            }
        }
        let un = mm(u.as_ref(), c.as_ref());
        let vn = crate::ssa::lstsq(c.as_ref(), vs.transpose())?.transpose().to_owned();
        let model = self.with_oblique_triples(&vec![1.0; r], &un, &vn, &idx);
        let groups = pg.iter().map(|g| g.iter().map(|&p| idx[p]).collect()).collect();
        Ok(NdOssaResult { model, groups, set: idx, iterations: 0, converged: true })
    }

    /// Oblique w-correlations (Rssa `owcor`).
    pub fn owcor(&self, groups: &[Group], set: &[usize]) -> Result<Matrix> {
        let lm = pseudo_inverse(&select_cols(self.u.as_ref(), set))?;
        let rm = pseudo_inverse(&select_cols(self.v.as_ref(), set))?;
        let r = set.len();
        let mut mx = Mat::<f64>::zeros(r * r, groups.len());
        for (gi, g) in groups.iter().enumerate() {
            let f = self.reconstruct(g)?;
            let t = self.traj_times(&f, &rm);
            let prod = mm(lm.as_ref(), t.as_ref());
            for j in 0..r {
                for i in 0..r {
                    mx[(i + j * r, gi)] = prod[(i, j)];
                }
            }
        }
        Ok(super::nd::owcor_matrix(&mx))
    }
}

