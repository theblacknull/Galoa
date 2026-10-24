//! Forecasting as in Rssa (`lrr`, `roots`, `rforecast`, `vforecast`,
//! `bforecast`), including reverse forecasting and both MSSA directions.

use faer::Mat;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use super::common::{auto_strategy, quantile7, refit};
use crate::error::{Result, SsaError};
use crate::matrix::{mm, select_cols, Matrix};
use crate::ssa::{Complex64, SsaModel};

/// Linear recurrence relation from an `(L, r)` basis (Rssa `lrr.default`).
///
/// With `orthonormalize` the basis is replaced by `Q` of its QR first.
/// Forward: `x_n = sum_i a_i x_{n-L+i}` (coefficients for the first `L-1`
/// rows); reverse: coefficients of the last `L-1` rows predicting the first.
pub fn lrr_from_basis(u: &Matrix, reverse: bool, orthonormalize: bool) -> Result<Vec<f64>> {
    let l = u.nrows();
    if u.ncols() == 0 {
        return Ok(vec![0.0; l - 1]);
    }
    let q = if orthonormalize { u.qr().compute_thin_Q() } else { u.clone() };
    let idx = if reverse { 0 } else { l - 1 };
    let lpf: Vec<f64> = (0..l).map(|i| (0..q.ncols()).map(|c| q[(i, c)] * q[(idx, c)]).sum()).collect();
    let divider = 1.0 - lpf[idx];
    if divider.abs() < f64::EPSILON.sqrt() {
        return Err(SsaError::LinAlg("Verticality coefficient equals to 1".into()));
    }
    Ok((0..l).filter(|&i| i != idx).map(|i| lpf[i] / divider).collect())
}

/// Roots of the characteristic polynomial of an LRR (eigenvalues of the
/// companion matrix), sorted by decreasing modulus (Rssa `roots.lrr`).
pub fn lrr_roots(lrr: &[f64]) -> Result<Vec<Complex64>> {
    let n = lrr.len();
    let c = Mat::from_fn(n, n, |i, j| if j == n - 1 { lrr[i] } else if i == j + 1 { 1.0 } else { 0.0 });
    let mut r: Vec<Complex64> = crate::la::eigenvalues(c.as_ref())?.into_iter().map(|(a, b)| Complex64::new(a, b)).collect();
    r.sort_by(|a, b| b.norm().total_cmp(&a.norm()));
    Ok(r)
}

/// Rssa `apply.lrr`: continues `f` by `len` values (or prepends them when
/// `reverse`).  Returns only the new values.
pub fn apply_lrr(f: &[f64], lrr: &[f64], len: usize, reverse: bool, drift: &[f64]) -> Result<Vec<f64>> {
    let r = lrr.len();
    if r > f.len() {
        return Err(SsaError::InvalidArgument("Wrong length of LRR".into()));
    }
    let dr = |i: usize| if drift.is_empty() { 0.0 } else { drift[i % drift.len()] };
    if !reverse {
        let mut v = f.to_vec();
        for i in 0..len {
            let n = v.len();
            let s: f64 = v[n - r..].iter().zip(lrr).map(|(a, b)| a * b).sum();
            v.push(s + dr(i));
        }
        Ok(v[f.len()..].to_vec())
    } else {
        // positions 0..len are new, filled from the end towards the start
        let mut v = vec![0.0; len];
        v.extend_from_slice(f);
        for i in 0..len {
            let pos = len - i - 1;
            let s: f64 = v[pos + 1..pos + 1 + r].iter().zip(lrr).map(|(a, b)| a * b).sum();
            v[pos] = s + dr(pos);
        }
        Ok(v[..len].to_vec())
    }
}

/// Base series for forecasting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Base {
    Reconstructed,
    Original,
}

/// MSSA forecasting direction (Rssa `direction`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// One LRR from `U`, applied to every series.
    Column,
    /// Joint forecast from `V` (Python `K_forecast` for unit weights).
    Row,
}

impl SsaModel {
    fn base_series(&self, group: &[usize], base: Base) -> Result<Matrix> {
        Ok(match base {
            Base::Reconstructed => self.reconstruct_components(group)?,
            Base::Original => self.x_s.clone(),
        })
    }

    /// LRR of a group (Rssa `lrr(x, groups, reverse)`; basis used as is).
    pub fn lrr(&self, group: &[usize], reverse: bool) -> Result<Vec<f64>> {
        self.check_indices(group)?;
        lrr_from_basis(&select_cols(self.u.as_ref(), group), reverse, false)
    }

    /// Recurrent forecast (Rssa `rforecast`): `(len, S)` new values.
    /// MSSA uses `direction`; `reverse` is available for the column direction.
    pub fn rforecast(&self, group: &[usize], len: usize, base: Base, direction: Direction, reverse: bool) -> Result<Matrix> {
        let f = self.base_series(group, base)?;
        if self.s == 1 || direction == Direction::Column {
            let lrr = self.lrr(group, reverse)?;
            let cols: Vec<Vec<f64>> = (0..self.s)
                .map(|c| apply_lrr(&f.col(c).iter().copied().collect::<Vec<_>>(), &lrr, len, reverse, &[]))
                .collect::<Result<_>>()?;
            return Ok(Mat::from_fn(len, self.s, |i, c| cols[c][i]));
        }
        if reverse {
            return Err(SsaError::InvalidArgument("reverse row forecast is not available".into()));
        }
        // row direction (Rssa rforecast.mssa, direction = "row")
        let (k, s) = (self.k, self.s);
        let v = select_cols(self.v.as_ref(), group);
        let r = group.len();
        let w = Mat::from_fn(s, r, |c, j| v[(c * k + k - 1, j)]);
        let q = Mat::from_fn(s * (k - 1), r, |row, j| v[((row / (k - 1)) * k + row % (k - 1), j)]);
        let wwt = mm(w.as_ref(), w.transpose());
        let a = Mat::from_fn(s, s, |i, j| if i == j { 1.0 } else { 0.0 } - wwt[(i, j)]);
        let wtq = mm(w.as_ref(), q.transpose());
        let mut z: Vec<f64> = (0..s).flat_map(|c| (self.n - (k - 1)..self.n).map(move |t| (c, t))).map(|(c, t)| f[(t, c)]).collect();
        let mut out = Mat::<f64>::zeros(len, s);
        for step in 0..len {
            let rhs = Mat::from_fn(s, 1, |i, _| (0..z.len()).map(|j| wtq[(i, j)] * z[j]).sum());
            let cr = crate::ssa::lstsq(a.as_ref(), rhs.as_ref())?;
            for c in 0..s {
                out[(step, c)] = cr[(c, 0)];
                let blk = &mut z[c * (k - 1)..(c + 1) * (k - 1)];
                blk.rotate_left(1);
                blk[k - 2] = cr[(c, 0)];
            }
        }
        Ok(out)
    }

    /// Vector forecast (Rssa `vforecast`): `(len, S)` new values.
    pub fn vforecast(&self, group: &[usize], len: usize, direction: Direction) -> Result<Matrix> {
        let mut g: Vec<usize> = Vec::new();
        for &i in group {
            if !g.contains(&i) {
                g.push(i);
            }
        }
        self.check_indices(&g)?;
        let (l, k, n) = (self.l, self.k, self.n);
        let r = g.len();
        let u = select_cols(self.u.as_ref(), &g);
        let v = select_cols(self.v.as_ref(), &g);
        let sig: Vec<f64> = g.iter().map(|&i| self.sigma[i]).collect();
        if self.s == 1 || direction == Direction::Column {
            let p = if self.s == 1 {
                shift_matrix(&u)?
            } else {
                // Rssa vforecast.mssa, column direction
                let uh = u.as_ref().subrows(0, l - 1);
                let ut = u.as_ref().subrows(1, l - 1);
                let pi: Vec<f64> = (0..r).map(|j| u[(l - 1, j)]).collect();
                let t = mm(uh.transpose(), ut);
                let nu2: f64 = pi.iter().map(|x| x * x).sum();
                Mat::from_fn(r, r, |i, j| t[(i, j)] + pi[i] * (0..r).map(|m| pi[m] * t[(m, j)]).sum::<f64>() / (1.0 - nu2))
            };
            let mut out = Mat::<f64>::zeros(len, self.s);
            for c in 0..self.s {
                let rows = k + len + l - 1;
                let mut z = Mat::<f64>::zeros(rows, r);
                for t in 0..k {
                    for j in 0..r {
                        z[(t, j)] = sig[j] * v[(c * k + t, j)];
                    }
                }
                for t in k..rows {
                    for i in 0..r {
                        z[(t, i)] = (0..r).map(|j| p[(i, j)] * z[(t - 1, j)]).sum();
                    }
                }
                let y = hankelize_sum(&u, &z);
                for i in 0..len {
                    out[(i, c)] = y[n + i];
                }
            }
            return Ok(out);
        }
        // row direction (Rssa vforecast.mssa, direction = "row")
        let s = self.s;
        let last = |c: usize| c * k + k - 1;
        let pi = Mat::from_fn(s, r, |c, j| v[(last(c), j)]);
        let vh = Mat::from_fn(s * (k - 1), r, |row, j| v[((row / (k - 1)) * k + row % (k - 1), j)]);
        let vt = Mat::from_fn(s * (k - 1), r, |row, j| v[((row / (k - 1)) * k + row % (k - 1) + 1, j)]);
        let t = mm(vh.transpose(), vt.as_ref());
        let ippt = Mat::from_fn(s, s, |i, j| if i == j { 1.0 } else { 0.0 } - (0..r).map(|m| pi[(i, m)] * pi[(j, m)]).sum::<f64>());
        let pit = mm(pi.as_ref(), t.as_ref()); // (S, r)
        let sol = crate::ssa::lstsq(ippt.as_ref(), pit.as_ref())?;
        let corr = mm(pi.transpose(), sol.as_ref());
        let p = Mat::from_fn(r, r, |i, j| t[(i, j)] + corr[(i, j)]);
        let rows = l + len + k - 1;
        let mut z = Mat::<f64>::zeros(rows, r);
        for t in 0..l {
            for j in 0..r {
                z[(t, j)] = sig[j] * u[(t, j)];
            }
        }
        for tt in l..rows {
            for i in 0..r {
                z[(tt, i)] = (0..r).map(|j| p[(i, j)] * z[(tt - 1, j)]).sum();
            }
        }
        let mut out = Mat::<f64>::zeros(len, s);
        for c in 0..s {
            let vb = Mat::from_fn(k, r, |t, j| v[(c * k + t, j)]);
            let y = hankelize_sum(&z, &vb);
            for i in 0..len {
                out[(i, c)] = y[n + i];
            }
        }
        Ok(out)
    }

    /// Bootstrap forecast (Rssa `bforecast`): returns `(mean, lower, upper)`
    /// for the first series, each of length `len`.  `R` replicates of
    /// `reconstruction + resampled residuals` are re-decomposed and
    /// forecast; `prediction` adds residual quantiles (prediction interval).
    #[allow(clippy::too_many_arguments)]
    pub fn bforecast(
        &self,
        group: &[usize],
        len: usize,
        r: usize,
        level: f64,
        vector: bool,
        prediction: bool,
        seed: u64,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        if self.s != 1 {
            return Err(SsaError::InvalidArgument("bforecast is defined for one series".into()));
        }
        let rec = self.reconstruct_components(group)?;
        let rec: Vec<f64> = rec.col(0).iter().copied().collect();
        let res: Vec<f64> = self.x_s.col(0).iter().zip(&rec).map(|(a, b)| a - b).collect();
        let neig = group.iter().copied().max().unwrap_or(0) + 1;
        let paths: Vec<Vec<f64>> = (0..r)
            .into_par_iter()
            .map(|b| {
                let mut rng = StdRng::seed_from_u64(seed.wrapping_add(b as u64));
                let f: Vec<f64> = rec.iter().map(|v| v + res[rng.gen_range(0..res.len())]).collect();
                let strat = auto_strategy(self.l, neig);
                let m = refit(strat.as_ref(), Mat::from_fn(f.len(), 1, |i, _| f[i]), self.l, self.weights.clone())?;
                let fc = if vector {
                    m.vforecast(group, len, Direction::Column)?
                } else {
                    m.rforecast(group, len, Base::Reconstructed, Direction::Column, false)?
                };
                Ok(fc.col(0).iter().copied().collect())
            })
            .collect::<Result<_>>()?;
        let (pl, pu) = ((1.0 - level) / 2.0, (1.0 + level) / 2.0);
        let (rl, ru) = if prediction { (quantile7(&res, pl), quantile7(&res, pu)) } else { (0.0, 0.0) };
        let mut mean = vec![0.0; len];
        let mut lo = vec![0.0; len];
        let mut hi = vec![0.0; len];
        for h in 0..len {
            let col: Vec<f64> = paths.iter().map(|p| p[h]).collect();
            mean[h] = col.iter().sum::<f64>() / r as f64;
            lo[h] = quantile7(&col, pl) + rl;
            hi[h] = quantile7(&col, pu) + ru;
        }
        Ok((mean, lo, hi))
    }
}

/// Rssa `.shift.matrix` (LS): `P` with `U[-1,] ~ U[-L,] P`, transposed the
/// way `vforecast` uses it (`z_j = P z_{j-1}` with `P = t(solution)`).
fn shift_matrix(u: &Matrix) -> Result<Matrix> {
    let l = u.nrows();
    let a = u.as_ref().subrows(0, l - 1).to_owned();
    let b = u.as_ref().subrows(1, l - 1).to_owned();
    let sol = crate::ssa::lstsq(a.as_ref(), b.as_ref())?; // (r, r)
    // R: Z[j, ] <- P %*% Z[j-1, ] with P = Conj(shift) -> z_j = P z_{j-1}
    Ok(sol)
}

/// `sum_j hankelize(a[:, j], b[:, j])`: anti-diagonal means of `A B^T`.
pub(crate) fn hankelize_sum(a: &Matrix, b: &Matrix) -> Vec<f64> {
    let (l, k) = (a.nrows(), b.nrows());
    let conv = crate::hankel::Convolver::new(l, k);
    let mut acc = conv.accumulator();
    for j in 0..a.ncols() {
        let sa = conv.spectrum(a.col(j).iter().copied(), 1.0);
        let sb = conv.spectrum(b.col(j).iter().copied(), 1.0);
        conv.add_product(&mut acc, &sa, &sb);
    }
    let mut y = conv.finish(acc);
    for (t, v) in y.iter_mut().enumerate() {
        *v /= crate::hankel::anti_diagonal_len(t, l, k) as f64;
    }
    y
}
