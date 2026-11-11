//! Cadzow iterations and gap filling (Rssa `cadzow`, shaped 1D-SSA,
//! `gapfill`, `igapfill`, `summarize.gaps`, `clplot`).

use faer::Mat;

use super::common::{auto_strategy, flatten, refit, series_weights, wcor_matrix, Group};
use super::forecast::{apply_lrr, lrr_from_basis};
use crate::error::{Result, SsaError};
use crate::hankel::Convolver;
use crate::matrix::{column, mm, select_cols, Matrix};
use crate::ssa::SsaModel;

impl SsaModel {
    /// Cadzow iterations (Rssa `cadzow`): alternates rank-`rank` projection
    /// and hankelisation until `max|F - F'| < tol` or `maxiter` iterations
    /// (`maxiter = 0`: no limit).  With `correct` the result is rescaled by
    /// the weighted LS factor `<X, F> / <F, F>`.  Returns `(N, S)`.
    pub fn cadzow(&self, rank: usize, correct: bool, tol: f64, maxiter: usize) -> Result<Matrix> {
        let grp: Vec<usize> = (0..rank).collect();
        let mut f = self.reconstruct_components(&grp)?;
        let strat = auto_strategy(self.l, rank);
        let mut it = 0;
        loop {
            let s = refit(strat.as_ref(), f.clone(), self.l, self.weights.clone())?;
            let rf = s.reconstruct_components(&grp)?;
            it += 1;
            let dist = flatten(&f).iter().zip(flatten(&rf)).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            if (maxiter > 0 && it >= maxiter) || dist < tol {
                break;
            }
            f = rf;
        }
        if correct {
            let w = series_weights(self);
            let (x, ff) = (flatten(&self.x_s), flatten(&f));
            let num: f64 = w.iter().zip(&x).zip(&ff).map(|((w, a), b)| w * a * b).sum();
            let den: f64 = w.iter().zip(&ff).map(|(w, b)| w * b * b).sum();
            let alpha = num / den;
            for v in f.col_iter_mut().flat_map(|c| c.iter_mut().collect::<Vec<_>>()) {
                *v *= alpha;
            }
        }
        Ok(f)
    }
}

/// Position of a gap relative to the series ends (Rssa `classify.gaps`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GapPos {
    Left,
    Right,
    Internal,
}

/// A run of missing values (0-based, inclusive bounds).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gap {
    pub left: usize,
    pub right: usize,
    pub pos: GapPos,
    /// `true` if every value in `[left, right]` is missing ("dense").
    pub dense: bool,
}

/// Groups missing values closer than `L` into gaps (Rssa `classify.gaps`).
pub fn classify_gaps(x: &[f64], l: usize) -> Vec<Gap> {
    let n = x.len();
    let na: Vec<usize> = (0..n).filter(|&i| x[i].is_nan()).collect();
    let mut gaps = Vec::new();
    let mut i = 0;
    while i < na.len() {
        let start = na[i];
        let mut j = i;
        while j + 1 < na.len() && na[j + 1] - na[j] <= l {
            j += 1;
        }
        let end = na[j];
        // R uses 1-based: l < L -> left;  r >= N - L + 1 -> right
        let pos = if start + 1 < l {
            GapPos::Left
        } else if end + 1 >= n - l + 1 {
            GapPos::Right
        } else {
            GapPos::Internal
        };
        gaps.push(Gap { left: start, right: end, pos, dense: j - i == end - start });
        i = j + 1;
    }
    gaps
}

/// Rssa `summarize.gaps` for one window length.
pub fn summarize_gaps(x: &[f64], l: usize) -> Vec<Gap> {
    classify_gaps(x, l)
}

/// Percentage of complete lag vectors for every `L = 2..=(N+1)/2`
/// (the curve drawn by Rssa `clplot`).
pub fn complete_lag_fraction(x: &[f64]) -> Vec<(usize, f64)> {
    let n = x.len();
    (2..=n.div_ceil(2))
        .map(|l| {
            let k = n - l + 1;
            let ok = (0..k).filter(|&j| x[j..j + l].iter().all(|v| !v.is_nan())).count();
            (l, ok as f64 / k as f64 * 100.0)
        })
        .collect()
}

/// Shaped 1D-SSA of a series with missing values (Rssa `ssa()` of a series
/// with `NA`): only complete lag vectors enter the trajectory matrix and
/// every element is averaged over the windows that cover it.
#[derive(Debug, Clone)]
pub struct ShapedSsa {
    pub x: Vec<f64>,
    pub n: usize,
    pub l: usize,
    pub k: usize,
    /// Complete lag vectors (`fmask`).
    pub fmask: Vec<bool>,
    /// Number of complete windows covering each element (0 = not covered).
    pub weights: Vec<usize>,
    pub u: Matrix,
    pub sigma: Vec<f64>,
    /// `(K_complete, d)` factor vectors.
    pub v: Matrix,
    pub d: usize,
}

impl ShapedSsa {
    /// Decomposes `x` (`NaN` = missing) with window `l`.
    pub fn fit(x: &[f64], l: usize) -> Result<Self> {
        let n = x.len();
        if l < 2 || l > n {
            return Err(SsaError::InvalidArgument("invalid window length".into()));
        }
        let k = n - l + 1;
        let fmask: Vec<bool> = (0..k).map(|j| x[j..j + l].iter().all(|v| !v.is_nan())).collect();
        let cols: Vec<usize> = (0..k).filter(|&j| fmask[j]).collect();
        if cols.is_empty() {
            return Err(SsaError::InvalidArgument("Nothing to decompose: the given field shape is empty".into()));
        }
        let mut weights = vec![0usize; n];
        for &j in &cols {
            for w in &mut weights[j..j + l] {
                *w += 1;
            }
        }
        let traj = Mat::from_fn(l, cols.len(), |i, c| x[cols[c] + i]);
        let (u, sigma, v) = crate::la::thin_svd(traj.as_ref())?;
        let d = sigma.len();
        Ok(Self { x: x.to_vec(), n, l, k, fmask, weights, u, sigma, v, d })
    }

    fn full_v(&self, i: usize) -> Vec<f64> {
        let mut out = vec![0.0; self.k];
        let mut r = 0;
        for (j, &m) in self.fmask.iter().enumerate() {
            if m {
                out[j] = self.v[(r, i)];
                r += 1;
            }
        }
        out
    }

    /// Reconstruction of a group; `NaN` where no complete window covers.
    pub fn reconstruct(&self, group: &[usize]) -> Result<Vec<f64>> {
        if let Some(&i) = group.iter().find(|&&i| i >= self.d) {
            return Err(SsaError::IndexOutOfRange { index: i, len: self.d });
        }
        let conv = Convolver::new(self.l, self.k);
        let mut acc = conv.accumulator();
        for &i in group {
            let a = conv.spectrum(self.u.col(i).iter().copied(), self.sigma[i]);
            let b = conv.spectrum(self.full_v(i).into_iter(), 1.0);
            conv.add_product(&mut acc, &a, &b);
        }
        let y = conv.finish(acc);
        Ok(y.iter().zip(&self.weights).map(|(v, &w)| if w == 0 { f64::NAN } else { v / w as f64 }).collect())
    }

    /// Signed w-correlations of groups over covered elements.
    pub fn wcor(&self, groups: &[Group]) -> Result<Matrix> {
        let keep: Vec<usize> = (0..self.n).filter(|&t| self.weights[t] > 0).collect();
        let cols: Vec<Vec<f64>> = groups.iter().map(|g| self.reconstruct(g)).collect::<Result<_>>()?;
        let fs = Mat::from_fn(keep.len(), cols.len(), |t, j| cols[j][keep[t]]);
        let w: Vec<f64> = keep.iter().map(|&t| self.weights[t] as f64).collect();
        Ok(wcor_matrix(&fs, &w))
    }

    /// Sequential gap filling by forward/backward recurrent forecasts
    /// (Rssa `gapfill`, `method = "sequential"`).  Internal gaps mix the two
    /// forecasts with weights `alpha = seq(0, 1, length = len)`.
    pub fn gapfill(&self, group: &[usize], reconstructed_base: bool) -> Result<Vec<f64>> {
        let f = if reconstructed_base { self.reconstruct(group)? } else { self.x.clone() };
        let ug = select_cols(self.u.as_ref(), group);
        let blrr = lrr_from_basis(&ug, true, true)?;
        let flrr = lrr_from_basis(&ug, false, true)?;
        let mut res = f.clone();
        let l = self.l;
        for gap in classify_gaps(&f, l) {
            let len = gap.right - gap.left + 1;
            let back = || -> Result<Vec<f64>> {
                let seg = &f[gap.right + 1..(gap.right + 1 + l).min(self.n)];
                apply_lrr(seg, &blrr, len, true, &[])
            };
            let fwd = || -> Result<Vec<f64>> {
                let seg = &f[gap.left.saturating_sub(l)..gap.left];
                apply_lrr(seg, &flrr, len, false, &[])
            };
            let fill = match gap.pos {
                GapPos::Left => back()?,
                GapPos::Right => fwd()?,
                GapPos::Internal => {
                    let (b, fw) = (back()?, fwd()?);
                    (0..len)
                        .map(|i| {
                            let a = if len == 1 { 0.0 } else { i as f64 / (len - 1) as f64 };
                            a * b[i] + (1.0 - a) * fw[i]
                        })
                        .collect()
                }
            };
            res[gap.left..=gap.right].copy_from_slice(&fill);
        }
        Ok(res)
    }
}

/// Iterative gap filling (Rssa `igapfill`, `base = "original"`): missing
/// values start at `fill` (default: mean of the observed values) and are
/// replaced by the rank-`max(groups)` reconstruction until
/// `sqrt(max((F - F')^2)) < tol` or `maxiter` iterations (0 = no limit).
/// Returns, for each group, the series with its gaps filled by that group.
pub fn igapfill(x: &[f64], l: usize, groups: &[Group], fill: Option<f64>, tol: f64, maxiter: usize) -> Result<Vec<Vec<f64>>> {
    let na: Vec<usize> = (0..x.len()).filter(|&i| x[i].is_nan()).collect();
    let ug: Vec<usize> = (0..=groups.iter().flatten().copied().max().unwrap_or(0)).collect();
    let obs: Vec<f64> = x.iter().copied().filter(|v| !v.is_nan()).collect();
    let fill = fill.unwrap_or(obs.iter().sum::<f64>() / obs.len() as f64);
    let mut f = x.to_vec();
    for &i in &na {
        f[i] = fill;
    }
    let strat = auto_strategy(l, ug.len());
    let mut it = 0;
    loop {
        let s = refit(strat.as_ref(), column(&f), l, vec![1.0])?;
        let r = s.reconstruct_components(&ug)?;
        let mut rf = f.clone();
        for &i in &na {
            rf[i] = r[(i, 0)];
        }
        let rss = f.iter().zip(&rf).map(|(a, b)| (a - b).powi(2)).fold(0.0, f64::max).sqrt();
        it += 1;
        if (maxiter > 0 && it >= maxiter) || rss < tol {
            break;
        }
        f = rf;
    }
    let s = refit(strat.as_ref(), column(&f), l, vec![1.0])?;
    groups
        .iter()
        .map(|g| {
            let r = s.reconstruct_components(g)?;
            let mut out = x.to_vec();
            for &i in &na {
                out[i] = r[(i, 0)];
            }
            Ok(out)
        })
        .collect()
}

/// Least-squares solve of a square system with R `qr` rank check
/// (`None` when numerically singular, as Rssa's `pi.project.*` skip it).
fn solve_checked(a: &Matrix, b: &Matrix) -> Option<Matrix> {
    let (_, s, _) = crate::la::thin_svd(a.as_ref()).ok()?;
    let smax = s.iter().copied().fold(0.0, f64::max);
    if s.iter().any(|&v| v <= 1e-7 * smax) {
        return None;
    }
    crate::ssa::lstsq(a.as_ref(), b.as_ref()).ok()
}

impl ShapedSsa {
    /// Simultaneous gap filling (Rssa `gapfill(method = "simultaneous")`):
    /// complete lag vectors are projected on the signal subspace, incomplete
    /// ones are completed by the oblique projections `pi.project.complete` /
    /// `pi.project.missing`, then the matrix is hankelised.
    pub fn gapfill_simultaneous(&self, group: &[usize], reconstructed_base: bool) -> Result<Vec<f64>> {
        let f = if reconstructed_base { self.reconstruct(group)? } else { self.x.clone() };
        let (l, k) = (self.l, self.k);
        let ug = select_cols(self.u.as_ref(), group);
        let r = ug.ncols();
        let mut xr = Mat::from_fn(l, k, |i, j| f[i + j]);
        let utu = |col: &[f64]| -> Vec<f64> {
            let c: Vec<f64> = (0..r).map(|j| (0..l).map(|i| ug[(i, j)] * col[i]).sum()).collect();
            (0..l).map(|i| (0..r).map(|j| ug[(i, j)] * c[j]).sum()).collect()
        };
        for j in 0..k {
            let col: Vec<f64> = (0..l).map(|i| xr[(i, j)]).collect();
            let midx: Vec<usize> = (0..l).filter(|&i| col[i].is_nan()).collect();
            if midx.is_empty() {
                let p = utu(&col);
                for i in 0..l {
                    xr[(i, j)] = p[i];
                }
                continue;
            }
            let fidx: Vec<usize> = (0..l).filter(|&i| !col[i].is_nan()).collect();
            if fidx.is_empty() {
                continue;
            }
            let w = Mat::from_fn(midx.len(), r, |a, b| ug[(midx[a], b)]);
            let v = Mat::from_fn(fidx.len(), r, |a, b| ug[(fidx[a], b)]);
            let vf = Mat::from_fn(fidx.len(), 1, |a, _| col[fidx[a]]);
            let wwt = mm(w.as_ref(), w.transpose());
            let a = Mat::from_fn(midx.len(), midx.len(), |p, q| if p == q { 1.0 } else { 0.0 } - wwt[(p, q)]);
            let vtw = mm(v.as_ref(), w.transpose()); // (f, m)
            let vtv = mm(v.transpose(), vf.as_ref()); // (r, 1)
            // complete part: V V^T v + VtW (A^{-1} t(VtW) v)
            let rhs_c = mm(vtw.transpose(), vf.as_ref());
            let rhs_m = mm(w.as_ref(), vtv.as_ref());
            let (Some(sc), Some(sm)) = (solve_checked(&a, &rhs_c), solve_checked(&a, &rhs_m)) else { continue };
            let resf = &mm(v.as_ref(), vtv.as_ref()) + &mm(vtw.as_ref(), sc.as_ref());
            for (a_, &i) in fidx.iter().enumerate() {
                xr[(i, j)] = resf[(a_, 0)];
            }
            for (a_, &i) in midx.iter().enumerate() {
                xr[(i, j)] = sm[(a_, 0)];
            }
        }
        Ok(crate::hankel::diagonal_averaging(xr.as_ref()))
    }
}
