//! Rssa-style correlations and automatic grouping
//! (`wcor`, `wnorm`, `contributions`, `frobenius.cor`,
//! `grouping.auto.wcor`, `grouping.auto.pgram`).

use faer::Mat;
use realfft::RealFftPlanner;

use super::common::{cov2cor, flatten, hclust_complete_cut, series_weights, split_by_label, wcor_matrix, Group};
use crate::error::Result;
use crate::matrix::{select_cols, Matrix};
use crate::ssa::SsaModel;

/// Basis for periodogram grouping (Rssa `base`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PgramBase {
    Series,
    Eigen,
    Factor,
}

/// How contributions are integrated over a frequency band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PgramMethod {
    Constant,
    Linear,
}

/// Frequency bands of [`SsaModel::grouping_auto_pgram`] (Rssa `freq.bins`).
#[derive(Debug, Clone)]
pub enum FreqBins {
    /// `n >= 2` equal bands on `[0, 0.5]`.
    Equal(usize),
    /// Upper edges; band `i` is `[upper[i-1], upper[i])`, the first starts at `-inf`.
    Upper(Vec<f64>),
    /// Explicit `[lower, upper)` bands.
    Ranges(Vec<(f64, f64)>),
}

/// Result of [`SsaModel::grouping_auto_pgram`].
#[derive(Debug, Clone)]
pub struct PgramGrouping {
    pub groups: Vec<Group>,
    /// `(components, bins)` relative contributions.
    pub contributions: Matrix,
    pub splitting: bool,
}

impl SsaModel {
    /// Signed weighted correlations between reconstructed groups (Rssa
    /// `wcor`).  `groups = None` uses one group per component.
    pub fn wcor(&self, groups: Option<&[Group]>) -> Result<Matrix> {
        let single: Vec<Group>;
        let groups = match groups {
            Some(g) => g,
            None => {
                single = (0..self.d).map(|i| vec![i]).collect();
                &single
            }
        };
        use rayon::prelude::*;
        let cols: Vec<Vec<f64>> = groups.par_iter().map(|g| self.reconstruct_components(g).map(|m| flatten(&m))).collect::<Result<_>>()?;
        let fs = Mat::from_fn(self.n * self.s, cols.len(), |t, j| cols[j][t]);
        Ok(wcor_matrix(&fs, &series_weights(self)))
    }

    /// Weighted norm of the series (Rssa `wnorm`), equal to `||X||_F`.
    pub fn wnorm(&self) -> f64 {
        self.total_energy().sqrt()
    }

    /// `sigma_i^2 / ||X||_F^2` (Rssa `contributions`, fractions not %).
    pub fn contributions(&self) -> Vec<f64> {
        let e = self.total_energy();
        self.sigma.iter().map(|s| s * s / e).collect()
    }

    /// Frobenius correlations of the group matrices (Rssa `frobenius.cor`).
    pub fn frobenius_cor(&self, groups: &[Group]) -> Result<Matrix> {
        let mut idx: Vec<usize> = groups.iter().flatten().copied().collect();
        idx.sort_unstable();
        idx.dedup();
        self.check_indices(&idx)?;
        let u = select_cols(self.u.as_ref(), &idx);
        let v = select_cols(self.v.as_ref(), &idx);
        let utu = crate::matrix::mm(u.transpose(), u.as_ref());
        let vtv = crate::matrix::mm(v.transpose(), v.as_ref());
        let r = idx.len();
        let cov = Mat::from_fn(r, r, |i, j| utu[(i, j)] * vtv[(i, j)] * self.sigma[idx[i]] * self.sigma[idx[j]]);
        let pos = |c: usize| idx.iter().position(|&x| x == c).unwrap();
        let g = groups.len();
        let gc = Mat::from_fn(g, g, |a, b| {
            groups[a].iter().map(|&i| groups[b].iter().map(|&j| cov[(pos(i), pos(j))]).sum::<f64>()).sum()
        });
        Ok(cov2cor(&gc))
    }

    /// Hierarchical clustering of components by w-correlation (Rssa
    /// `grouping.auto.wcor`, complete linkage on `(1 - w) / 2`).
    pub fn grouping_auto_wcor(&self, components: &[usize], nclust: usize) -> Result<Vec<Group>> {
        let mut comps = components.to_vec();
        comps.sort_unstable();
        comps.dedup();
        let gs: Vec<Group> = comps.iter().map(|&i| vec![i]).collect();
        let w = self.wcor(Some(&gs))?;
        let d = Mat::from_fn(w.nrows(), w.ncols(), |i, j| (1.0 - w[(i, j)]) / 2.0);
        let labels = hclust_complete_cut(&d, nclust);
        Ok(split_by_label(&comps, &labels))
    }

    /// Periodogram-based grouping (Rssa `grouping.auto.pgram`).
    ///
    /// `threshold`: if all `<= 0` every component goes to the band with the
    /// largest contribution ("splitting"), otherwise each band keeps the
    /// components whose contribution is `>= threshold` ("independent").
    pub fn grouping_auto_pgram(
        &self,
        components: &[usize],
        base: PgramBase,
        freq_bins: &FreqBins,
        threshold: &[f64],
        method: PgramMethod,
        drop_empty: bool,
    ) -> Result<PgramGrouping> {
        let mut comps = components.to_vec();
        comps.sort_unstable();
        comps.dedup();
        let fs: Matrix = match base {
            PgramBase::Eigen => select_cols(self.u.as_ref(), &comps),
            PgramBase::Factor => select_cols(self.v.as_ref(), &comps),
            PgramBase::Series => {
                let cols: Vec<Vec<f64>> = comps.iter().map(|&i| self.reconstruct_components(&[i]).map(|m| flatten(&m))).collect::<Result<_>>()?;
                Mat::from_fn(self.n * self.s, cols.len(), |t, j| cols[j][t])
            }
        };
        let (spec, freq) = pgram(&fs);
        let (lower, upper): (Vec<f64>, Vec<f64>) = match freq_bins {
            FreqBins::Equal(b) => {
                let up: Vec<f64> = (1..=*b).map(|i| 0.5 * i as f64 / *b as f64).collect();
                let lo = std::iter::once(f64::NEG_INFINITY).chain(up[..b - 1].iter().copied()).collect();
                (lo, up)
            }
            FreqBins::Upper(up) => {
                let lo = std::iter::once(f64::NEG_INFINITY).chain(up[..up.len().saturating_sub(1)].iter().copied()).collect();
                (lo, up.clone())
            }
            FreqBins::Ranges(r) => r.iter().copied().unzip(),
        };
        let nres = threshold.len().max(upper.len());
        let rep = |v: &[f64], i: usize| v[i % v.len()];
        let n = fs.nrows();
        let norms: Vec<f64> = (0..fs.ncols()).map(|j| (0..spec.nrows()).map(|f| spec[(f, j)]).sum()).collect();
        let contributions = Mat::from_fn(comps.len(), nres, |c, b| {
            let (lo, up) = (rep(&lower, b), rep(&upper, b));
            let v = match method {
                PgramMethod::Constant => (0..spec.nrows()).filter(|&f| freq[f] < up && freq[f] >= lo).map(|f| spec[(f, c)]).sum::<f64>(),
                PgramMethod::Linear => cumspec(&spec, &freq, n, c, up) - cumspec(&spec, &freq, n, c, lo),
            };
            v / norms[c]
        });
        let splitting = threshold.iter().all(|&t| t <= 0.0);
        let mut groups: Vec<Group> = vec![Vec::new(); nres];
        for (c, &comp) in comps.iter().enumerate() {
            if splitting {
                let mut best = 0;
                for b in 1..nres {
                    if contributions[(c, b)] > contributions[(c, best)] {
                        best = b;
                    }
                }
                groups[best].push(comp);
            } else {
                for (b, g) in groups.iter_mut().enumerate() {
                    if contributions[(c, b)] >= rep(threshold, b) {
                        g.push(comp);
                    }
                }
            }
        }
        if drop_empty {
            groups.retain(|g| !g.is_empty());
        }
        Ok(PgramGrouping { groups, contributions, splitting })
    }
}

/// Rssa `pgram`: one-sided periodogram of every column; frequencies `j/n`.
pub fn pgram(x: &Matrix) -> (Matrix, Vec<f64>) {
    let n = x.nrows();
    let nf = n / 2 + 1;
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let mut spec = Mat::<f64>::zeros(nf, x.ncols());
    for j in 0..x.ncols() {
        let mut buf: Vec<f64> = x.col(j).iter().copied().collect();
        let mut out = fft.make_output_vec();
        fft.process(&mut buf, &mut out).expect("fft");
        for f in 0..nf {
            let mut p = out[f].norm_sqr();
            let interior = if n % 2 == 0 { f > 0 && f < nf - 1 } else { f > 0 };
            if interior {
                p *= 2.0;
            }
            spec[(f, j)] = p;
        }
    }
    let freq = (0..nf).map(|f| f as f64 / n as f64).collect();
    (spec, freq)
}

/// Piecewise-linear cumulative spectrum (Rssa `cumspecfuns`, `rule = 2`).
fn cumspec(spec: &Matrix, freq: &[f64], n: usize, c: usize, x: f64) -> f64 {
    let nf = freq.len();
    // knots: 0, freq[-N] + 1/(2n), 0.5 ; values: 0, cumsum(spec)
    let mut kx = vec![0.0];
    kx.extend(freq[..nf - 1].iter().map(|f| f + 1.0 / (2.0 * n as f64)));
    kx.push(0.5);
    let mut ky = vec![0.0];
    let mut acc = 0.0;
    for f in 0..nf {
        acc += spec[(f, c)];
        ky.push(acc);
    }
    if x <= kx[0] {
        return ky[0];
    }
    if x >= kx[kx.len() - 1] {
        return ky[ky.len() - 1];
    }
    let i = kx.iter().rposition(|&k| k <= x).unwrap();
    let t = (x - kx[i]) / (kx[i + 1] - kx[i]);
    ky[i] + t * (ky[i + 1] - ky[i])
}
