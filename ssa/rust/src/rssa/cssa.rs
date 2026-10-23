//! Complex SSA (Rssa `kind = "cssa"`): SSA of a complex-valued series,
//! e.g. a pair of real series `x + i y` analysed jointly.

use faer::Mat;

use crate::error::{Result, SsaError};
use crate::ssa::Complex64;

type CMat = Mat<Complex64>;

/// Fitted Complex SSA.  A series with missing values (`NaN` in either
/// part) gives a shaped decomposition over the complete lag vectors only
/// (Rssa builds the same for `kind = "cssa"` with `NA`).
#[derive(Debug, Clone)]
pub struct Cssa {
    pub z: Vec<Complex64>,
    pub n: usize,
    pub l: usize,
    pub k: usize,
    /// `(L, d)` left singular vectors.
    pub u: CMat,
    pub sigma: Vec<f64>,
    /// `(K', d)` right singular vectors over the kept lag vectors
    /// (`H = U diag(sigma) V^H`).
    pub v: CMat,
    pub d: usize,
    /// Indices of the kept lag vectors (all `0..K` without gaps).
    pub cols: Vec<usize>,
    /// Number of kept lag vectors covering each point (0 = not covered).
    pub cover: Vec<usize>,
}

pub(crate) fn is_na(c: &Complex64) -> bool {
    !c.re.is_finite() || !c.im.is_finite()
}

impl Cssa {
    pub fn fit(z: &[Complex64], l: usize) -> Result<Self> {
        let n = z.len();
        if l < 2 || l > n {
            return Err(SsaError::InvalidArgument("invalid window length".into()));
        }
        let k = n - l + 1;
        let cols: Vec<usize> = if z.iter().any(is_na) {
            // lag vector j is complete iff no NA in z[j..j+L]
            let mut bad = 0usize;
            let mut out = Vec::new();
            for (t, c) in z.iter().enumerate() {
                if is_na(c) {
                    bad += 1;
                }
                if t >= l && is_na(&z[t - l]) {
                    bad -= 1;
                }
                if t + 1 >= l && bad == 0 {
                    out.push(t + 1 - l);
                }
            }
            out
        } else {
            (0..k).collect()
        };
        if cols.is_empty() {
            return Err(SsaError::InvalidArgument("no complete lag vectors".into()));
        }
        let h = CMat::from_fn(l, cols.len(), |i, j| z[i + cols[j]]);
        let svd = h.thin_svd().map_err(|e| SsaError::LinAlg(format!("SVD did not converge: {e:?}")))?;
        let sigma: Vec<f64> = svd.S().column_vector().iter().map(|c| c.re).collect();
        let d = sigma.len();
        let mut cover = vec![0usize; n];
        for &j in &cols {
            cover[j..j + l].iter_mut().for_each(|c| *c += 1);
        }
        Ok(Self { z: z.to_vec(), n, l, k, u: svd.U().to_owned(), sigma, v: svd.V().to_owned(), d, cols, cover })
    }

    /// Whether every lag vector is kept (no gaps).
    pub fn is_full(&self) -> bool {
        self.cols.len() == self.k
    }

    pub(crate) fn check(&self, group: &[usize]) -> Result<()> {
        match group.iter().find(|&&i| i >= self.d) {
            Some(&i) => Err(SsaError::IndexOutOfRange { index: i, len: self.d }),
            None => Ok(()),
        }
    }

    pub(crate) fn require_full(&self) -> Result<()> {
        if self.is_full() { Ok(()) } else { Err(SsaError::InvalidArgument("not available for a series with gaps".into())) }
    }

    /// Hankelised sum of `sigma_i u_i v_i^H` over `group` (FFT convolution
    /// of every term, one inverse transform).
    pub(crate) fn hankel_sums(&self, group: &[usize]) -> Vec<Complex64> {
        use rayon::prelude::*;
        use rustfft::FftPlanner;
        let (l, k, n) = (self.l, self.k, self.n);
        let p = crate::hankel::next_fast_len(n);
        let mut planner = FftPlanner::<f64>::new();
        let fwd = planner.plan_fft_forward(p);
        let inv = planner.plan_fft_inverse(p);
        let zero = Complex64::new(0.0, 0.0);
        let parts: Vec<Vec<Complex64>> = group
            .par_iter()
            .map(|&c| {
                let mut a = vec![zero; p];
                let mut b = vec![zero; p];
                for i in 0..l {
                    a[i] = self.u[(i, c)] * self.sigma[c];
                }
                for (r, &j) in self.cols.iter().enumerate() {
                    b[j] = self.v[(r, c)].conj();
                }
                fwd.process(&mut a);
                fwd.process(&mut b);
                a.iter_mut().zip(&b).for_each(|(x, y)| *x *= y);
                a
            })
            .collect();
        let mut acc = vec![zero; p];
        for q in parts {
            acc.iter_mut().zip(&q).for_each(|(x, y)| *x += y);
        }
        inv.process(&mut acc);
        let _ = k;
        acc.truncate(n);
        acc.iter_mut().for_each(|v| *v /= p as f64);
        acc
    }

    /// Reconstruction: anti-diagonal means of `sum sigma_i u_i v_i^H`
    /// (`NaN` at points no kept lag vector covers).
    pub fn reconstruct(&self, group: &[usize]) -> Result<Vec<Complex64>> {
        self.check(group)?;
        let nan = Complex64::new(f64::NAN, f64::NAN);
        Ok(self.hankel_sums(group).into_iter().zip(&self.cover).map(|(v, &c)| if c == 0 { nan } else { v / c as f64 }).collect())
    }

    /// Moduli of the complex weighted correlations between groups.
    pub fn wcor(&self, groups: &[Vec<usize>]) -> Result<Mat<f64>> {
        let fs: Vec<Vec<Complex64>> = groups.iter().map(|g| self.reconstruct(g)).collect::<Result<_>>()?;
        let w: Vec<f64> = self.cover.iter().map(|&v| v as f64).collect();
        let m = fs.len();
        let cov = |a: usize, b: usize| -> Complex64 { (0..self.n).filter(|&t| w[t] > 0.0).map(|t| fs[a][t].conj() * fs[b][t] * w[t]).sum() };
        let diag: Vec<f64> = (0..m).map(|i| cov(i, i).re).collect();
        Ok(Mat::from_fn(m, m, |i, j| if i == j { 1.0 } else { (cov(i, j).norm() / (diag[i] * diag[j]).sqrt()).min(1.0) }))
    }

    /// Complex LRR of a group (Rssa `lrr.cssa`).
    pub fn lrr(&self, group: &[usize]) -> Result<Vec<Complex64>> {
        self.check(group)?;
        let l = self.l;
        let lpf: Vec<Complex64> = (0..l).map(|i| group.iter().map(|&c| self.u[(i, c)].conj() * self.u[(l - 1, c)]).sum()).collect();
        let div = Complex64::new(1.0, 0.0) - lpf[l - 1];
        if div.norm() < f64::EPSILON.sqrt() {
            return Err(SsaError::LinAlg("Verticality coefficient equals to 1".into()));
        }
        Ok(lpf[..l - 1].iter().map(|v| v / div).collect())
    }

    /// Recurrent forecast of the reconstructed group (Rssa `rforecast`).
    pub fn rforecast(&self, group: &[usize], len: usize) -> Result<Vec<Complex64>> {
        self.require_full()?;
        let lrr = self.lrr(group)?;
        let mut f = self.reconstruct(group)?;
        let r = lrr.len();
        for _ in 0..len {
            let n = f.len();
            let s: Complex64 = f[n - r..].iter().zip(&lrr).map(|(a, b)| a * b).sum();
            f.push(s);
        }
        Ok(f[self.n..].to_vec())
    }
}

/// Complex least squares `A X = B` through the SVD pseudo-inverse (numpy /
/// R `qr.solve` give the same solution for full column rank).
pub fn clstsq(a: &CMat, b: &CMat) -> Result<CMat> {
    // thin full-rank systems (the ESPRIT case): sequential Gram-Schmidt QR
    // with reorthogonalisation, avoiding parallel-kernel start-up for a few
    // columns; rank-deficient systems fall through to the SVD
    if a.nrows() >= a.ncols() && a.ncols() > 0 && a.ncols() <= 32 {
        if let Some(x) = small_qr_solve(a, b) {
            return Ok(x);
        }
    }
    let svd = a.thin_svd().map_err(|e| SsaError::LinAlg(format!("SVD did not converge: {e:?}")))?;
    let s: Vec<f64> = svd.S().column_vector().iter().map(|c| c.re).collect();
    let smax = s.iter().copied().fold(0.0, f64::max);
    let cut = f64::EPSILON * a.nrows().max(a.ncols()) as f64 * smax;
    let (u, v) = (svd.U(), svd.V());
    let k = s.len();
    // X = V diag(1/s) U^H B
    let uhb = CMat::from_fn(k, b.ncols(), |i, j| (0..a.nrows()).map(|t| u[(t, i)].conj() * b[(t, j)]).sum::<Complex64>() * if s[i] > cut { 1.0 / s[i] } else { 0.0 });
    Ok(CMat::from_fn(a.ncols(), b.ncols(), |i, j| (0..k).map(|t| v[(i, t)] * uhb[(t, j)]).sum()))
}

/// `A X = B` by classical Gram-Schmidt applied twice; `None` when a column
/// is numerically dependent.
fn small_qr_solve(a: &CMat, b: &CMat) -> Option<CMat> {
    let (m, r) = (a.nrows(), a.ncols());
    let zero = Complex64::new(0.0, 0.0);
    let mut q: Vec<Vec<Complex64>> = Vec::with_capacity(r);
    let mut rm = vec![vec![zero; r]; r];
    let scale = (0..r).map(|j| a.col(j).norm_l2()).fold(0.0, f64::max);
    for j in 0..r {
        let mut v: Vec<Complex64> = (0..m).map(|i| a[(i, j)]).collect();
        for _ in 0..2 {
            for (k, qk) in q.iter().enumerate() {
                let c: Complex64 = qk.iter().zip(&v).map(|(x, y)| x.conj() * y).sum();
                rm[k][j] += c;
                v.iter_mut().zip(qk).for_each(|(y, x)| *y -= c * x);
            }
        }
        let nv = v.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        if nv <= 1e-10 * scale {
            return None;
        }
        rm[j][j] = Complex64::new(nv, 0.0);
        v.iter_mut().for_each(|c| *c /= nv);
        q.push(v);
    }
    // R X = Q^H B
    let mut x = CMat::from_fn(r, b.ncols(), |i, c| q[i].iter().enumerate().map(|(t, qv)| qv.conj() * b[(t, c)]).sum());
    for c in 0..b.ncols() {
        for i in (0..r).rev() {
            let mut s = x[(i, c)];
            for k in i + 1..r {
                s -= rm[i][k] * x[(k, c)];
            }
            x[(i, c)] = s / rm[i][i];
        }
    }
    Some(x)
}

impl Cssa {
    /// ESPRIT roots of a group (Rssa `parestimate` for CSSA), R `eigen` order.
    pub fn parestimate(&self, group: &[usize], tls: bool) -> Result<super::parest::Params> {
        self.check(group)?;
        let r = group.len();
        let l = self.l;
        let up = CMat::from_fn(l - 1, r, |i, j| self.u[(i, group[j])]);
        let lo = CMat::from_fn(l - 1, r, |i, j| self.u[(i + 1, group[j])]);
        let z = if tls {
            let ab = CMat::from_fn(l - 1, 2 * r, |i, j| if j < r { up[(i, j)] } else { lo[(i, j - r)] });
            let svd = ab.thin_svd().map_err(|e| SsaError::LinAlg(format!("{e:?}")))?;
            let v = svd.V();
            let v11t = CMat::from_fn(r, r, |i, j| v[(j, i)]);
            let v21t = CMat::from_fn(r, r, |i, j| v[(r + j, i)]);
            let x = clstsq(&v11t, &v21t)?;
            CMat::from_fn(r, r, |i, j| x[(i, j)].conj())
        } else {
            clstsq(&up, &lo)?
        };
        let mut roots: Vec<Complex64> = z.eigenvalues().map_err(|e| SsaError::LinAlg(format!("{e:?}")))?;
        roots.sort_by(|a, b| b.norm().total_cmp(&a.norm()));
        Ok(super::parest::roots2pars(roots))
    }

    /// Vector forecast (Rssa `vforecast` for CSSA).
    pub fn vforecast(&self, group: &[usize], len: usize) -> Result<Vec<Complex64>> {
        self.check(group)?;
        self.require_full()?;
        let mut g: Vec<usize> = Vec::new();
        for &i in group {
            if !g.contains(&i) {
                g.push(i);
            }
        }
        let (l, k, r) = (self.l, self.k, g.len());
        let up = CMat::from_fn(l - 1, r, |i, j| self.u[(i, g[j])]);
        let lo = CMat::from_fn(l - 1, r, |i, j| self.u[(i + 1, g[j])]);
        let s = clstsq(&up, &lo)?;
        let p = CMat::from_fn(r, r, |i, j| s[(i, j)].conj());
        let rows = k + len + l - 1;
        let mut z = CMat::from_fn(rows, r, |_, _| Complex64::new(0.0, 0.0));
        for t in 0..k {
            for j in 0..r {
                z[(t, j)] = self.v[(t, g[j])] * self.sigma[g[j]];
            }
        }
        for t in k..rows {
            for i in 0..r {
                z[(t, i)] = (0..r).map(|j| p[(i, j)] * z[(t - 1, j)]).sum();
            }
        }
        // anti-diagonal means of U Z^H (FFT convolution per component)
        let n2 = l + rows - 1;
        let sums = {
            use rustfft::FftPlanner;
            let p = crate::hankel::next_fast_len(n2);
            let mut planner = FftPlanner::<f64>::new();
            let (fwd, inv) = (planner.plan_fft_forward(p), planner.plan_fft_inverse(p));
            let zero = Complex64::new(0.0, 0.0);
            let mut acc = vec![zero; p];
            for j in 0..r {
                let mut a = vec![zero; p];
                let mut b = vec![zero; p];
                for i in 0..l {
                    a[i] = self.u[(i, g[j])];
                }
                for t in 0..rows {
                    b[t] = z[(t, j)].conj();
                }
                fwd.process(&mut a);
                fwd.process(&mut b);
                acc.iter_mut().zip(a.iter().zip(&b)).for_each(|(c, (x, y))| *c += x * y);
            }
            inv.process(&mut acc);
            acc.truncate(n2);
            acc.iter_mut().for_each(|v| *v /= p as f64);
            acc
        };
        Ok((self.n..self.n + len).map(|t| sums[t] / crate::hankel::anti_diagonal_len(t, l, rows) as f64).collect())
    }

    /// Cadzow iterations (Rssa `cadzow` for CSSA).
    pub fn cadzow(&self, rank: usize, correct: bool, tol: f64, maxiter: usize) -> Result<Vec<Complex64>> {
        self.require_full()?;
        let g: Vec<usize> = (0..rank).collect();
        let mut f = self.reconstruct(&g)?;
        let mut it = 0;
        loop {
            let s = Cssa::fit(&f, self.l)?;
            let rf = s.reconstruct(&g)?;
            it += 1;
            let dist = f.iter().zip(&rf).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max);
            if (maxiter > 0 && it >= maxiter) || dist < tol {
                break;
            }
            f = rf;
        }
        if correct {
            let w: Vec<f64> = crate::hankel::hankel_weights(self.n, self.l, self.k).into_iter().map(|v| v as f64).collect();
            // Rssa .series.winnerprod: sum(w * F1 * F2) (no conjugation)
            let num: Complex64 = self.z.iter().zip(&f).zip(&w).map(|((a, b), w)| a * b * *w).sum();
            let den: Complex64 = f.iter().zip(&w).map(|(b, w)| b * b * *w).sum();
            let a = num / den;
            f.iter_mut().for_each(|v| *v *= a);
        }
        Ok(f)
    }
}
