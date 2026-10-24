//! Remaining Rssa utilities: `hmatr`, `high.rank.rate`, `Cond`, the I-OSSA
//! summary, `forecast` / `predict` with intervals, decomposition
//! continuation, `residuals` and the `nu` / `nv` / `nsigma` accessors.

use faer::Mat;
use rayon::prelude::*;

use super::common::{auto_strategy, flatten, pseudo_inverse, refit, wcor_matrix, Group};
use super::forecast::{Base, Direction};
use super::ossa::OssaResult;
use crate::error::{Result, SsaError};
use crate::matrix::{column, select_cols, Matrix};
use crate::ssa::SsaModel;
use crate::svd::LanczosSvd;

/// Heterogeneity matrix (Rssa `hmatr`): entry `(j, i)` measures how badly
/// the `neig`-dimensional SSA subspace of `F[i ..= i + B]` approximates the
/// lag vectors of the test interval `F[j .. j + T)`.  Defaults in Rssa:
/// `B = T = N / 4`, `L = B / 2`, `neig = 10`.  Returns `(N - T, N - B)`.
pub fn hmatr(f: &[f64], b: usize, t: usize, l: usize, neig: usize) -> Result<Matrix> {
    let n = f.len();
    if l < 2 || l > b || t < l || b >= n || t >= n {
        return Err(SsaError::InvalidArgument("invalid hmatr parameters".into()));
    }
    let k = n - l + 1;
    // squared norms of the lag vectors and their cumulative sums
    let row2: Vec<f64> = (0..k).map(|j| f[j..j + l].iter().map(|v| v * v).sum()).collect();
    let mut cth2 = vec![0.0; k + 1];
    for j in 0..k {
        cth2[j + 1] = cth2[j] + row2[j];
    }
    let nt = n - t;
    let w = t - l + 1;
    let denom: Vec<f64> = (0..nt).map(|j| cth2[j + w] - cth2[j]).collect();
    let cols: Vec<Vec<f64>> = (0..n - b)
        .into_par_iter()
        .map(|idx| {
            let fb = &f[idx..=idx + b];
            let m = crate::Ssa::new().fit(fb, Some(l))?;
            let ne = neig.min(m.d);
            let u = select_cols(m.u.as_ref(), &(0..ne).collect::<Vec<_>>());
            let proj: Vec<f64> = (0..k)
                .map(|j| (0..ne).map(|c| (0..l).map(|i| f[j + i] * u[(i, c)]).sum::<f64>().powi(2)).sum())
                .collect();
            let mut cx = vec![0.0; k + 1];
            for j in 0..k {
                cx[j + 1] = cx[j] + proj[j];
            }
            Ok((0..nt).map(|j| 1.0 - (cx[j + w] - cx[j]) / denom[j]).collect())
        })
        .collect::<Result<_>>()?;
    Ok(Mat::from_fn(nt, n - b, |i, j| cols[j][i]))
}

/// Condition number `sigma_max / sigma_min` (Rssa `Cond`).
pub fn cond(a: &Matrix) -> Result<f64> {
    let (_, s, _) = crate::la::thin_svd(a.as_ref())?;
    Ok(s[0] / s[s.len() - 1])
}

/// Interval type of [`SsaModel::forecast`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interval {
    None,
    Confidence,
    Prediction,
}

/// Rssa `forecast` object.
#[derive(Debug, Clone)]
pub struct Forecast {
    pub mean: Vec<f64>,
    pub lower: Option<Vec<f64>>,
    pub upper: Option<Vec<f64>>,
    pub fitted: Vec<f64>,
    pub residuals: Vec<f64>,
    pub level: f64,
}

/// I-OSSA diagnostics (Rssa `iossa.result`).
#[derive(Debug, Clone)]
pub struct IossaSummary {
    pub cond: (f64, f64),
    pub owcor: Matrix,
    pub wcor: Matrix,
    pub initial_wcor: Matrix,
    pub tau: Vec<f64>,
    pub initial_tau: Vec<f64>,
}

impl SsaModel {
    /// Number of eigentriples (Rssa `nsigma`, `nu`, `nv`).
    pub fn nsigma(&self) -> usize {
        self.d
    }

    /// `F - reconstruction(group)` (Rssa `residuals`).
    pub fn residuals(&self, group: &[usize]) -> Result<Matrix> {
        let r = self.reconstruct_components(group)?;
        Ok(&self.x_s - &r)
    }

    /// Continues a truncated decomposition up to `neig` triples (Rssa
    /// `decompose(..., force.continue = TRUE)`).
    pub fn continue_decomposition(&self, neig: usize) -> Result<SsaModel> {
        if neig <= self.d {
            return Ok(self.clone());
        }
        refit(&LanczosSvd::new(neig), self.x_s.clone(), self.l, self.weights.clone())
    }

    /// `1 - sum(sigma[..rank]^2) / ||X(F)||^2` of a series with the same
    /// window (Rssa `high.rank.rate`).
    pub fn high_rank_rate(&self, f: &Matrix, rank: usize) -> Result<f64> {
        let s = refit(auto_strategy(self.l, rank).as_ref(), f.clone(), self.l, self.weights.clone())?;
        let e = s.total_energy();
        Ok(1.0 - s.sigma[..rank.min(s.d)].iter().map(|v| v * v).sum::<f64>() / e)
    }

    /// Diagnostics of an I-OSSA result (Rssa `.make.ossa.result`).
    pub fn iossa_summary(&self, res: &OssaResult, initial_groups: &[Group]) -> Result<IossaSummary> {
        let m = &res.model;
        let ibl = pseudo_inverse(&select_cols(m.u.as_ref(), &res.set))?;
        let ibr = pseudo_inverse(&select_cols(m.v.as_ref(), &res.set))?;
        let fs: Vec<Matrix> = res.groups.iter().map(|g| m.reconstruct_components(g)).collect::<Result<_>>()?;
        let init: Vec<Matrix> = initial_groups.iter().map(|g| self.reconstruct_components(g)).collect::<Result<_>>()?;
        let w = super::common::series_weights(self);
        let wc = |v: &[Matrix]| {
            let cols: Vec<Vec<f64>> = v.iter().map(flatten).collect();
            wcor_matrix(&Mat::from_fn(cols[0].len(), cols.len(), |t, j| cols[j][t]), &w)
        };
        let tau = fs.iter().zip(&res.groups).map(|(f, g)| self.high_rank_rate(f, g.len())).collect::<Result<_>>()?;
        let itau = init.iter().zip(initial_groups).map(|(f, g)| self.high_rank_rate(f, g.len())).collect::<Result<_>>()?;
        Ok(IossaSummary {
            cond: (cond(&ibl)?, cond(&ibr)?),
            owcor: m.owcor(&res.groups, &res.set)?,
            wcor: wc(&fs),
            initial_wcor: wc(&init),
            tau,
            initial_tau: itau,
        })
    }

    /// Rssa `forecast` / `predict` for one series: point forecast of a group
    /// (recurrent or vector), optionally with bootstrap confidence or
    /// prediction intervals (`r` replicates, `only.intervals = TRUE`).
    #[allow(clippy::too_many_arguments)]
    pub fn forecast(&self, group: &[usize], h: usize, vector: bool, interval: Interval, level: f64, r: usize, seed: u64) -> Result<Forecast> {
        if self.s != 1 {
            return Err(SsaError::InvalidArgument("forecast is defined for one series".into()));
        }
        let point = if vector {
            self.vforecast(group, h, Direction::Column)?
        } else {
            self.rforecast(group, h, Base::Reconstructed, Direction::Column, false)?
        };
        let mean: Vec<f64> = point.col(0).iter().copied().collect();
        let fitted: Vec<f64> = self.reconstruct_components(group)?.col(0).iter().copied().collect();
        let residuals = self.x_s.col(0).iter().zip(&fitted).map(|(a, b)| a - b).collect();
        let (lower, upper) = match interval {
            Interval::None => (None, None),
            _ => {
                let (_, lo, hi) = self.bforecast(group, h, r, level, vector, interval == Interval::Prediction, seed)?;
                (Some(lo), Some(hi))
            }
        };
        Ok(Forecast { mean, lower, upper, fitted, residuals, level })
    }
}

/// Convenience: SSA model of one series with the Rssa default decomposition.
pub fn ssa_auto(x: &[f64], l: usize, neig: usize) -> Result<SsaModel> {
    refit(auto_strategy(l, neig).as_ref(), column(x), l, vec![1.0])
}

/// Methods available for each kind of decomposition (Rssa
/// `ssa.capabilities`).  Kinds: `"1d-ssa"`, `"toeplitz-ssa"`, `"mssa"`,
/// `"cssa"`, `"2d-ssa"`, `"nd-ssa"`, `"pssa"`, `"wossa"`, `"shaped"`.
pub fn capabilities(kind: &str) -> Vec<&'static str> {
    let common = ["decompose", "reconstruct", "wcor", "wnorm", "contributions", "residuals", "cadzow", "iossa", "fossa", "eossa", "owcor"];
    let mut v: Vec<&'static str> = common.to_vec();
    let extra: &[&'static str] = match kind {
        "1d-ssa" | "toeplitz-ssa" => &["lrr", "roots", "rforecast", "vforecast", "bforecast", "forecast", "predict", "parestimate", "gapfill", "igapfill", "grouping.auto.pgram", "grouping.auto.wcor", "hmatr", "plot"],
        "mssa" => &["lrr", "rforecast", "vforecast", "predict", "parestimate", "gapfill", "igapfill", "grouping.auto.wcor", "plot"],
        "cssa" => &["lrr", "rforecast", "vforecast", "parestimate", "gapfill", "igapfill", "plot"],
        "2d-ssa" | "nd-ssa" => &["parestimate", "igapfill", "grouping.auto.wcor", "plot"],
        "pssa" => &["lrr", "rforecast", "vforecast", "parestimate", "plot"],
        "wossa" => &["lrr", "rforecast", "vforecast", "plot"],
        "shaped" => &["gapfill", "igapfill", "plot"],
        _ => &[],
    };
    v.extend_from_slice(extra);
    v
}
