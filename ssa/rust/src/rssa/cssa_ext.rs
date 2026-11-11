//! Complex SSA, continued: norms and contributions, gap filling
//! (`gapfill`, `igapfill` for `kind = "cssa"`) and the nested oblique
//! decompositions `iossa`, `fossa`, `eossa`, `owcor`.
//!
//! Rssa runs the oblique methods on a CSSA object through the real-valued
//! code and loses the imaginary parts; here every step is done in complex
//! arithmetic (`X = sum sigma_i u_i v_i^H`).

use faer::Mat;

use super::common::{consecutive_groups, hclust_complete_cut, split_by_label, Group};
use super::cssa::{clstsq, is_na, Cssa};
use super::gaps::{classify_gaps, GapPos};
use crate::error::{Result, SsaError};
use crate::ssa::Complex64;

type CMat = Mat<Complex64>;

fn c0() -> Complex64 {
    Complex64::new(0.0, 0.0)
}

fn sorted_set(groups: &[Group]) -> Vec<usize> {
    let mut idx: Vec<usize> = groups.iter().flatten().copied().collect();
    idx.sort_unstable();
    idx.dedup();
    idx
}

fn cols_of(m: &CMat, idx: &[usize]) -> CMat {
    CMat::from_fn(m.nrows(), idx.len(), |i, j| m[(i, idx[j])])
}

/// `A^H B`.
fn hmul(a: &CMat, b: &CMat) -> CMat {
    a.adjoint() * b
}

fn csvd(a: &CMat) -> Result<(CMat, Vec<f64>, CMat)> {
    let s = a.thin_svd().map_err(|e| SsaError::LinAlg(format!("SVD did not converge: {e:?}")))?;
    Ok((s.U().to_owned(), s.S().column_vector().iter().map(|c| c.re).collect(), s.V().to_owned()))
}

/// Complex `orthogonalize(Y, Z, sigma)`: SVD of `Y diag(s) Z^H`.
fn orthogonalize(y: &CMat, z: &CMat, sigma: &[f64]) -> Result<(Vec<f64>, CMat, CMat)> {
    let (qy, qz) = (y.qr(), z.qr());
    let (ry, rz) = (qy.thin_R().to_owned(), qz.thin_R().to_owned());
    let r = sigma.len();
    let m = &ry * CMat::from_fn(r, r, |i, j| rz[(j, i)].conj() * sigma[i]);
    let (u, d, v) = csvd(&m)?;
    Ok((d, qy.compute_thin_Q() * u, qz.compute_thin_Q() * v))
}

/// Complex LRR of a basis (Rssa `lrr.default`): forward predicts the last
/// row from the others, reverse the first row.
fn lrr_of(u: &CMat, reverse: bool) -> Result<Vec<Complex64>> {
    let q = u.qr().compute_thin_Q();
    let l = q.nrows();
    let at = if reverse { 0 } else { l - 1 };
    let lpf: Vec<Complex64> = (0..l).map(|i| (0..q.ncols()).map(|c| q[(i, c)].conj() * q[(at, c)]).sum()).collect();
    let div = Complex64::new(1.0, 0.0) - lpf[at];
    if div.norm() < f64::EPSILON.sqrt() {
        return Err(SsaError::LinAlg("Verticality coefficient equals to 1".into()));
    }
    Ok((0..l).filter(|&i| i != at).map(|i| lpf[i] / div).collect())
}

/// `len` new values: forward continues after `seg`, reverse precedes it
/// (returned in time order).
fn apply_clrr(seg: &[Complex64], lrr: &[Complex64], len: usize, reverse: bool) -> Vec<Complex64> {
    let r = lrr.len();
    if reverse {
        let mut f: Vec<Complex64> = seg[..r].to_vec();
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            let v: Complex64 = f[..r].iter().zip(lrr).map(|(a, b)| a * b).sum();
            f.insert(0, v);
            out.push(v);
        }
        out.reverse();
        out
    } else {
        let mut f: Vec<Complex64> = seg[seg.len() - r..].to_vec();
        for _ in 0..len {
            let n = f.len();
            let v: Complex64 = f[n - r..].iter().zip(lrr).map(|(a, b)| a * b).sum();
            f.push(v);
        }
        f[r..].to_vec()
    }
}

/// Result of a complex oblique decomposition.
#[derive(Debug, Clone)]
pub struct CssaOssaResult {
    pub model: Cssa,
    pub groups: Vec<Group>,
    pub set: Vec<usize>,
    pub iterations: usize,
    pub converged: bool,
}

impl Cssa {
    /// Total energy `||X||_F^2` over the kept lag vectors.
    pub fn total_energy(&self) -> f64 {
        self.cols.iter().map(|&j| self.z[j..j + self.l].iter().map(|c| c.norm_sqr()).sum::<f64>()).sum()
    }

    /// Hankel-weighted norm of the series (Rssa `wnorm` for CSSA).
    pub fn wnorm(&self) -> f64 {
        self.z.iter().zip(&self.cover).map(|(z, &c)| if c > 0 { z.norm_sqr() * c as f64 } else { 0.0 }).sum::<f64>().sqrt()
    }

    /// Shares of `sigma_i^2` in the total energy (Rssa `contributions`).
    pub fn contributions(&self, idx: &[usize]) -> Result<Vec<f64>> {
        self.check(idx)?;
        let e = self.total_energy();
        Ok(idx.iter().map(|&i| self.sigma[i].powi(2) / e).collect())
    }

    /// Series minus the reconstruction of `group` (Rssa `residuals`).
    pub fn residuals(&self, group: &[usize]) -> Result<Vec<Complex64>> {
        let r = self.reconstruct(group)?;
        Ok(self.z.iter().zip(&r).map(|(a, b)| a - b).collect())
    }

    /// Copy with the triples at `idx` replaced by the (normalised) columns of
    /// `y`, `z` times `nosigma`.
    pub fn with_oblique_triples(&self, nosigma: &[f64], y: &CMat, z: &CMat, idx: &[usize]) -> Cssa {
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

    fn orth_triples(&self, idx: &[usize], orth: bool) -> Result<(Vec<f64>, CMat, CMat)> {
        self.check(idx)?;
        let s: Vec<f64> = idx.iter().map(|&i| self.sigma[i]).collect();
        let (u, v) = (cols_of(&self.u, idx), cols_of(&self.v, idx));
        if orth { orthogonalize(&u, &v, &s) } else { Ok((s, u, v)) }
    }

    /// Sequential (`simultaneous = false`) or simultaneous gap filling
    /// (Rssa `gapfill` for CSSA).  Needs a model fitted on a series with NA.
    pub fn gapfill(&self, group: &[usize], reconstructed_base: bool, simultaneous: bool) -> Result<Vec<Complex64>> {
        self.check(group)?;
        let f = if reconstructed_base { self.reconstruct(group)? } else { self.z.clone() };
        let ug = cols_of(&self.u, group);
        if simultaneous {
            return self.gapfill_sim(&f, &ug);
        }
        let (blrr, flrr) = (lrr_of(&ug, true)?, lrr_of(&ug, false)?);
        let (l, n) = (self.l, self.n);
        let mask: Vec<f64> = f.iter().map(|c| if is_na(c) { f64::NAN } else { 0.0 }).collect();
        let mut res = f.clone();
        for gap in classify_gaps(&mask, l) {
            let len = gap.right - gap.left + 1;
            let back = || apply_clrr(&f[gap.right + 1..(gap.right + 1 + l).min(n)], &blrr, len, true);
            let fwd = || apply_clrr(&f[gap.left.saturating_sub(l)..gap.left], &flrr, len, false);
            let fill: Vec<Complex64> = match gap.pos {
                GapPos::Left => back(),
                GapPos::Right => fwd(),
                GapPos::Internal => {
                    let (b, w) = (back(), fwd());
                    (0..len)
                        .map(|i| {
                            let a = if len == 1 { 0.0 } else { i as f64 / (len - 1) as f64 };
                            b[i] * a + w[i] * (1.0 - a)
                        })
                        .collect()
                }
            };
            res[gap.left..=gap.right].copy_from_slice(&fill);
        }
        Ok(res)
    }

    fn gapfill_sim(&self, f: &[Complex64], ug: &CMat) -> Result<Vec<Complex64>> {
        let (l, k, n) = (self.l, self.k, self.n);
        let r = ug.ncols();
        let nan = Complex64::new(f64::NAN, f64::NAN);
        let mut sums = vec![c0(); n];
        let mut cnt = vec![0usize; n];
        for j in 0..k {
            let col: Vec<Complex64> = f[j..j + l].to_vec();
            let midx: Vec<usize> = (0..l).filter(|&i| is_na(&col[i])).collect();
            let fidx: Vec<usize> = (0..l).filter(|&i| !is_na(&col[i])).collect();
            let mut out = col.clone();
            if midx.is_empty() {
                let c: Vec<Complex64> = (0..r).map(|q| (0..l).map(|i| ug[(i, q)].conj() * col[i]).sum()).collect();
                for i in 0..l {
                    out[i] = (0..r).map(|q| ug[(i, q)] * c[q]).sum();
                }
            } else if !fidx.is_empty() {
                // oblique projections (Rssa pi.project.complete / missing)
                let w = CMat::from_fn(midx.len(), r, |a, b| ug[(midx[a], b)]);
                let v = CMat::from_fn(fidx.len(), r, |a, b| ug[(fidx[a], b)]);
                let vf = CMat::from_fn(fidx.len(), 1, |a, _| col[fidx[a]]);
                let wwh = &w * w.adjoint();
                let a = CMat::from_fn(midx.len(), midx.len(), |p, q| if p == q { Complex64::new(1.0, 0.0) } else { c0() } - wwh[(p, q)]);
                let (_, sa, _) = csvd(&a)?;
                let smax = sa.iter().copied().fold(0.0, f64::max);
                if sa.iter().any(|&s| s <= 1e-7 * smax) {
                    continue;
                }
                let vhv = hmul(&v, &vf); // (r, 1)
                let vwh = &v * w.adjoint(); // (f, m)
                let sc = clstsq(&a, &hmul(&vwh, &vf))?;
                let sm = clstsq(&a, &(&w * &vhv))?;
                let resf = &(&v * &vhv) + &(&vwh * &sc);
                for (q, &i) in fidx.iter().enumerate() {
                    out[i] = resf[(q, 0)];
                }
                for (q, &i) in midx.iter().enumerate() {
                    out[i] = sm[(q, 0)];
                }
            } else {
                continue;
            }
            for i in 0..l {
                sums[i + j] += out[i];
                cnt[i + j] += 1;
            }
        }
        Ok(sums.into_iter().zip(cnt).map(|(s, c)| if c == 0 { nan } else { s / c as f64 }).collect())
    }

    /// Iterative gap filling (Rssa `igapfill` for CSSA, `base =
    /// "original"`): gaps start at `fill` (default: mean of the observed
    /// values) and take the rank-`max(groups)` reconstruction until the
    /// largest change is below `tol` or `maxiter` iterations (0 = no limit).
    pub fn igapfill(&self, groups: &[Group], fill: Option<Complex64>, tol: f64, maxiter: usize) -> Result<Vec<Vec<Complex64>>> {
        let na: Vec<usize> = (0..self.n).filter(|&i| is_na(&self.z[i])).collect();
        let top = groups.iter().flatten().copied().max().unwrap_or(0);
        let ug: Vec<usize> = (0..=top).collect();
        let obs: Vec<&Complex64> = self.z.iter().filter(|c| !is_na(c)).collect();
        let fill = fill.unwrap_or(obs.iter().copied().sum::<Complex64>() / obs.len() as f64);
        let mut f = self.z.clone();
        for &i in &na {
            f[i] = fill;
        }
        let mut it = 0;
        loop {
            let r = Cssa::fit(&f, self.l)?.reconstruct(&ug)?;
            let mut rf = f.clone();
            for &i in &na {
                rf[i] = r[i];
            }
            let rss = f.iter().zip(&rf).map(|(a, b)| (a - b).norm_sqr()).fold(0.0, f64::max).sqrt();
            it += 1;
            if (maxiter > 0 && it >= maxiter) || rss < tol {
                break;
            }
            f = rf;
        }
        let s = Cssa::fit(&f, self.l)?;
        groups
            .iter()
            .map(|g| {
                let r = s.reconstruct(g)?;
                let mut out = self.z.clone();
                for &i in &na {
                    out[i] = r[i];
                }
                Ok(out)
            })
            .collect()
    }

    /// Filter-adjusted O-SSA (Rssa `fossa`) in complex arithmetic: the
    /// factor vectors are filtered (default `[-1, 1]`) and the set is
    /// rotated by the eigenvectors of the Hermitian Gram matrix.
    pub fn fossa(&self, nested_groups: &[Group], filter: &[f64], gamma: f64, normalize: bool) -> Result<CssaOssaResult> {
        self.require_full()?;
        let idx = sorted_set(nested_groups);
        let (os, u, v) = self.orth_triples(&idx, true)?;
        let r = idx.len();
        let zz = CMat::from_fn(v.nrows(), r, |i, j| v[(i, j)] * os[j]);
        let y = if normalize { v.clone() } else { zz.clone() };
        let (k, fl) = (self.k, filter.len());
        let kk = k - fl + 1;
        let fy = CMat::from_fn(kk, r, |t, j| (0..fl).map(|q| y[(t + q, j)] * filter[q]).sum());
        let newv = if gamma.is_infinite() {
            fy
        } else {
            CMat::from_fn(k + kk, r, |i, j| if i < k { y[(i, j)] } else { fy[(i - k, j)] * gamma })
        };
        let g = hmul(&newv, &newv);
        let evd = g.self_adjoint_eigen(faer::Side::Lower).map_err(|e| SsaError::LinAlg(format!("{e:?}")))?;
        let ev = evd.U();
        let e = CMat::from_fn(r, r, |i, j| ev[(i, r - 1 - j)]);
        let (un, zn) = if normalize {
            (CMat::from_fn(u.nrows(), r, |i, j| u[(i, j)] * os[j]) * &e, &v * &e)
        } else {
            (&u * &e, &zz * &e)
        };
        let model = self.with_oblique_triples(&vec![1.0; r], &un, &zn, &idx);
        Ok(CssaOssaResult { model, groups: nested_groups.to_vec(), set: idx, iterations: 0, converged: true })
    }

    /// ESPRIT-based O-SSA (Rssa `eossa`) in complex arithmetic: the set is
    /// rotated to the eigenvectors of the shift matrix and the roots are
    /// clustered into `k` groups (complex roots are not merged with their
    /// conjugates: a complex series has one root per component).
    pub fn eossa(&self, nested_groups: &[Group], k: usize) -> Result<CssaOssaResult> {
        self.require_full()?;
        let idx = sorted_set(nested_groups);
        let (os, u, v) = self.orth_triples(&idx, false)?;
        let r = idx.len();
        if k == 0 || k > r {
            return Err(SsaError::InvalidArgument(format!("k must be between 1 and {r}")));
        }
        let l = self.l;
        let up = CMat::from_fn(l - 1, r, |i, j| u[(i, j)]);
        let lo = CMat::from_fn(l - 1, r, |i, j| u[(i + 1, j)]);
        let z = clstsq(&up, &lo)?;
        let evd = z.eigen().map_err(|e| SsaError::LinAlg(format!("{e:?}")))?;
        let roots: Vec<Complex64> = evd.S().column_vector().iter().copied().collect();
        let vecs = evd.U().to_owned();
        let mut ord: Vec<usize> = (0..r).collect();
        ord.sort_by(|&a, &b| roots[a].arg().abs().total_cmp(&roots[b].arg().abs()));
        let d = Mat::from_fn(r, r, |i, j| (roots[ord[i]] - roots[ord[j]]).norm());
        let labels = hclust_complete_cut(&d, k);
        let pos_groups = split_by_label(&(0..r).collect::<Vec<_>>(), &labels);
        let mut c = CMat::from_fn(r, r, |_, _| c0());
        for g in &pos_groups {
            let m = CMat::from_fn(r, g.len(), |i, j| vecs[(i, ord[g[j]])]);
            let q = m.qr().compute_thin_Q();
            for (jj, &col) in g.iter().enumerate() {
                for i in 0..r {
                    c[(i, col)] = q[(i, jj)];
                }
            }
        }
        // X_set = U S V^H = (U C) (V S C^{-H})^H
        let un = &u * &c;
        let vs = CMat::from_fn(v.nrows(), r, |i, j| v[(i, j)] * os[j]);
        let cinv = clstsq(&c, &CMat::identity(r, r))?;
        let vn = &vs * cinv.adjoint();
        let model = self.with_oblique_triples(&vec![1.0; r], &un, &vn, &idx);
        let groups = pos_groups.iter().map(|g| g.iter().map(|&p| idx[p]).collect()).collect();
        Ok(CssaOssaResult { model, groups, set: idx, iterations: 0, converged: true })
    }

    /// Iterative O-SSA (Rssa `iossa`) in complex arithmetic; defaults in
    /// Rssa are `tol = 1e-5`, `kappa = 2`, `maxiter = 100`,
    /// `kappa_balance = 0.5`.
    pub fn iossa(&self, nested_groups: &[Group], tol: f64, kappa: Option<f64>, maxiter: usize, kappa_balance: f64) -> Result<CssaOssaResult> {
        self.require_full()?;
        let ng: Vec<Group> = nested_groups.iter().map(|g| { let mut h = g.clone(); h.dedup(); h }).collect();
        let total: usize = ng.iter().map(|g| g.len()).sum();
        let idx = sorted_set(&ng);
        if idx.len() != total {
            return Err(SsaError::InvalidArgument("Intersected nested groups".into()));
        }
        let mut fs: Vec<Vec<Complex64>> = ng.iter().map(|g| self.reconstruct(g)).collect::<Result<_>>()?;
        let (osigma, u, v) = self.orth_triples(&idx, true)?;
        let ranks: Vec<usize> = ng.iter().map(|g| g.len()).collect();
        let groups_pos: Vec<Group> = if kappa.is_some() {
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
                let cur = Cssa::fit(f, self.l)?;
                let (s, a, b) = cur.orth_triples(&(0..ranks[i]).collect::<Vec<_>>(), true)?;
                osig.push(s);
                us.push(a);
                vs.push(b);
            }
            if let Some(kp) = kappa {
                for i in 0..ng.len().saturating_sub(1) {
                    let div = osig[i][ranks[i] - 1] / osig[i + 1][0];
                    if div < kp {
                        let mul = kp / div;
                        osig[i + 1].iter_mut().for_each(|s| *s /= mul);
                        let (a, b) = (mul.powf(kappa_balance), mul.powf(1.0 - kappa_balance));
                        us[i + 1] = &us[i + 1] * faer::Scale(Complex64::new(a, 0.0));
                        vs[i + 1] = &vs[i + 1] * faer::Scale(Complex64::new(b, 0.0));
                    }
                }
            }
            let (s2, y2, z2) = svd2lrsvd(&osigma, &u, &v, &hcat(&us), &hcat(&vs))?;
            sigma = s2;
            yy = y2;
            zz = z2;
            let tmp = Cssa { u: yy.clone(), v: zz.clone(), sigma: sigma.clone(), d: sigma.len(), ..self.clone() };
            let fs_new: Vec<Vec<Complex64>> = groups_pos.iter().map(|g| tmp.reconstruct(g)).collect::<Result<_>>()?;
            let delta = fs_new
                .iter()
                .zip(&fs)
                .map(|(a, b)| (a.iter().zip(b).map(|(x, y)| (x - y).norm_sqr()).sum::<f64>() / a.len() as f64).sqrt())
                .fold(0.0, f64::max);
            fs = fs_new;
            if delta < tol {
                converged = true;
                break;
            }
        }
        let model = self.with_oblique_triples(&sigma, &yy, &zz, &idx);
        let groups = groups_pos.iter().map(|g| g.iter().map(|&p| idx[p]).collect()).collect();
        Ok(CssaOssaResult { model, groups, set: idx, iterations, converged })
    }

    /// Oblique w-correlations (moduli) of groups drawn from `set` (Rssa
    /// `owcor`).
    pub fn owcor(&self, groups: &[Group], set: &[usize]) -> Result<Mat<f64>> {
        self.require_full()?;
        self.check(set)?;
        let pinv = |m: &CMat| clstsq(m, &CMat::identity(m.nrows(), m.nrows()));
        let lm = pinv(&cols_of(&self.u, set))?; // (r, L)
        let rm = pinv(&cols_of(&self.v, set))?; // (r, K)
        let r = set.len();
        let mut mx = CMat::from_fn(r * r, groups.len(), |_, _| c0());
        for (gi, g) in groups.iter().enumerate() {
            let f = self.reconstruct(g)?;
            let h = CMat::from_fn(self.l, self.k, |i, j| f[i + j]);
            // LM H RM^H : coefficients of H in the (u_i v_j^H) basis
            let prod = &(&lm * &h) * rm.adjoint();
            for j in 0..r {
                for i in 0..r {
                    mx[(i + j * r, gi)] = prod[(i, j)];
                }
            }
        }
        let g = hmul(&mx, &mx);
        let m = groups.len();
        Ok(Mat::from_fn(m, m, |i, j| if i == j { 1.0 } else { (g[(i, j)].norm() / (g[(i, i)].re * g[(j, j)].re).sqrt()).min(1.0) }))
    }
}

/// Complex `svd2LRsvd`: SVD of `U diag(d) V^H` in the bases `bas_l`, `bas_r`.
fn svd2lrsvd(d: &[f64], u: &CMat, v: &CMat, bas_l: &CMat, bas_r: &CMat) -> Result<(Vec<f64>, CMat, CMat)> {
    let r = d.len();
    let ubl = hmul(u, bas_l);
    let vbr = hmul(v, bas_r);
    let scaled = CMat::from_fn(r, r, |i, j| vbr[(i, j)] / d[i]);
    let m = hmul(&ubl, &scaled);
    let (du, ds, dv) = csvd(&m)?;
    let sig: Vec<f64> = ds.iter().rev().map(|x| 1.0 / x).collect();
    let du = CMat::from_fn(r, r, |i, j| du[(i, r - 1 - j)]);
    let dv = CMat::from_fn(r, r, |i, j| dv[(i, r - 1 - j)]);
    Ok((sig, &(u * &ubl) * &du, &(v * &vbr) * &dv))
}

fn hcat(ms: &[CMat]) -> CMat {
    let rows = ms[0].nrows();
    let cols: usize = ms.iter().map(|m| m.ncols()).sum();
    let mut out = CMat::from_fn(rows, cols, |_, _| c0());
    let mut c = 0;
    for m in ms {
        for j in 0..m.ncols() {
            for i in 0..rows {
                out[(i, c + j)] = m[(i, j)];
            }
        }
        c += m.ncols();
    }
    out
}
