//! MSSA of series of different lengths and/or with missing values (Rssa
//! `ssa(list(...), kind = "mssa")`).  Leading and trailing `NaN` of each
//! series are dropped (R `na.omit` on a series list); internal `NaN` make
//! the decomposition shaped.

use faer::Mat;

use super::common::Group;
use super::forecast::{apply_lrr, hankelize_sum, lrr_from_basis, Base, Direction};
use super::gaps::{classify_gaps, GapPos};
use super::nd::{NdMethod, NdSsa, NdSsaBuilder};
use crate::error::{Result, SsaError};
use crate::matrix::{mm, select_cols, Matrix};

/// Fitted MSSA of a list of series.
#[derive(Debug, Clone)]
pub struct MssaList {
    pub nd: NdSsa,
    /// Trimmed series (no leading / trailing `NaN`).
    pub series: Vec<Vec<f64>>,
    /// Number of leading / trailing values removed from each input series.
    pub trimmed: Vec<(usize, usize)>,
    pub n: Vec<usize>,
    pub l: usize,
    pub k: Vec<usize>,
}

impl MssaList {
    /// `l = None` uses `(min(N) + 1) / 2`.
    pub fn fit(series: &[Vec<f64>], l: Option<usize>, method: NdMethod) -> Result<Self> {
        if series.is_empty() {
            return Err(SsaError::InvalidArgument("need at least one series".into()));
        }
        let mut trimmed = Vec::new();
        let mut ss = Vec::new();
        for x in series {
            let st = x.iter().position(|v| !v.is_nan()).ok_or_else(|| SsaError::InvalidArgument("all times contain an NA".into()))?;
            let en = x.iter().rposition(|v| !v.is_nan()).unwrap();
            trimmed.push((st, x.len() - 1 - en));
            ss.push(x[st..=en].to_vec());
        }
        let n: Vec<usize> = ss.iter().map(|s| s.len()).collect();
        let nmin = *n.iter().min().unwrap();
        let l = l.unwrap_or(nmin.div_ceil(2));
        if l < 1 || l > nmin {
            return Err(SsaError::InvalidArgument("invalid window length".into()));
        }
        let nmax = *n.iter().max().unwrap();
        let nch = ss.len();
        let field: Vec<f64> = (0..nch).flat_map(|c| (0..nmax).map(move |t| (c, t))).map(|(c, t)| if t < n[c] { ss[c][t] } else { f64::NAN }).collect();
        let mut b = NdSsaBuilder::new(field, vec![nmax, nch]).window(vec![l, 1]).method(method);
        b.mssa = true;
        let nr = &n;
        b.domain = Some((0..nch).flat_map(|c| (0..nmax).map(move |t| t < nr[c])).collect());
        let nd = b.fit()?;
        let k = n.iter().map(|&ni| ni - l + 1).collect();
        Ok(Self { nd, series: ss, trimmed, n, l, k })
    }

    fn split(&self, field: &[f64]) -> Vec<Vec<f64>> {
        let nmax = self.nd.n[0];
        (0..self.n.len()).map(|c| field[c * nmax..c * nmax + self.n[c]].to_vec()).collect()
    }

    /// Restores the trimmed `NaN` around each series.
    pub fn pad(&self, s: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        s.into_iter()
            .zip(&self.trimmed)
            .map(|(v, &(a, b))| std::iter::repeat_n(f64::NAN, a).chain(v).chain(std::iter::repeat_n(f64::NAN, b)).collect())
            .collect()
    }

    /// Reconstruction of a group, one vector per (trimmed) series.
    pub fn reconstruct(&self, group: &[usize]) -> Result<Vec<Vec<f64>>> {
        Ok(self.split(&self.nd.reconstruct(group)?))
    }

    pub fn wcor(&self, groups: &[Group]) -> Result<Matrix> {
        self.nd.wcor(groups)
    }

    fn shaped(&self) -> bool {
        self.series.iter().any(|s| s.iter().any(|v| v.is_nan()))
    }

    /// Rows of `V` belonging to series `c` (positions `0..K_c`).
    fn v_rows(&self) -> Vec<std::ops::Range<usize>> {
        let mut start = 0;
        self.k
            .iter()
            .map(|&kc| {
                let r = start..start + kc;
                start += kc;
                r
            })
            .collect()
    }

    fn base(&self, group: &[usize], base: Base) -> Result<Vec<Vec<f64>>> {
        Ok(match base {
            Base::Reconstructed => self.reconstruct(group)?,
            Base::Original => self.series.clone(),
        })
    }

    /// Recurrent forecast (Rssa `rforecast.mssa`): `len` new values per series.
    pub fn rforecast(&self, group: &[usize], len: usize, base: Base, direction: Direction) -> Result<Vec<Vec<f64>>> {
        if self.shaped() {
            return Err(SsaError::InvalidArgument("recurrent forecasting is not implemented for shaped MSSA".into()));
        }
        self.nd.check(group)?;
        let f = self.base(group, base)?;
        match direction {
            Direction::Column => {
                let lrr = lrr_from_basis(&select_cols(self.nd.u.as_ref(), group), false, false)?;
                f.iter().map(|s| apply_lrr(s, &lrr, len, false, &[])).collect()
            }
            Direction::Row => {
                let v = select_cols(self.nd.v.as_ref(), group);
                let rows = self.v_rows();
                let s = self.n.len();
                let r = group.len();
                let last: Vec<usize> = rows.iter().map(|rg| rg.end - 1).collect();
                let others: Vec<usize> = rows.iter().flat_map(|rg| rg.start..rg.end - 1).collect();
                let w = Mat::from_fn(s, r, |c, j| v[(last[c], j)]);
                let q = Mat::from_fn(others.len(), r, |i, j| v[(others[i], j)]);
                let wwt = mm(w.as_ref(), w.transpose());
                let a = Mat::from_fn(s, s, |i, j| if i == j { 1.0 } else { 0.0 } - wwt[(i, j)]);
                let wtq = mm(w.as_ref(), q.transpose());
                let mut z: Vec<f64> = (0..s).flat_map(|c| f[c][self.n[c] - (self.k[c] - 1)..].to_vec()).collect();
                let mut out = vec![Vec::with_capacity(len); s];
                for _ in 0..len {
                    let rhs = Mat::from_fn(s, 1, |i, _| (0..z.len()).map(|j| wtq[(i, j)] * z[j]).sum());
                    let cr = crate::ssa::lstsq(a.as_ref(), rhs.as_ref())?;
                    let mut off = 0;
                    for c in 0..s {
                        let km = self.k[c] - 1;
                        out[c].push(cr[(c, 0)]);
                        let blk = &mut z[off..off + km];
                        blk.rotate_left(1);
                        blk[km - 1] = cr[(c, 0)];
                        off += km;
                    }
                }
                Ok(out)
            }
        }
    }

    /// Vector forecast (Rssa `vforecast.mssa`).
    pub fn vforecast(&self, group: &[usize], len: usize, direction: Direction) -> Result<Vec<Vec<f64>>> {
        if self.shaped() {
            return Err(SsaError::InvalidArgument("vector forecasting is not implemented for shaped MSSA".into()));
        }
        let mut g: Vec<usize> = Vec::new();
        for &i in group {
            if !g.contains(&i) {
                g.push(i);
            }
        }
        self.nd.check(&g)?;
        let (l, r) = (self.l, g.len());
        let u = select_cols(self.nd.u.as_ref(), &g);
        let v = select_cols(self.nd.v.as_ref(), &g);
        let sig: Vec<f64> = g.iter().map(|&i| self.nd.sigma[i]).collect();
        let rows = self.v_rows();
        match direction {
            Direction::Column => {
                let uh = u.as_ref().subrows(0, l - 1);
                let ut = u.as_ref().subrows(1, l - 1);
                let pi: Vec<f64> = (0..r).map(|j| u[(l - 1, j)]).collect();
                let t = mm(uh.transpose(), ut);
                let nu2: f64 = pi.iter().map(|x| x * x).sum();
                let p = Mat::from_fn(r, r, |i, j| t[(i, j)] + pi[i] * (0..r).map(|m| pi[m] * t[(m, j)]).sum::<f64>() / (1.0 - nu2));
                rows.iter()
                    .enumerate()
                    .map(|(c, rg)| {
                        let kc = self.k[c];
                        let nrow = kc + len + l - 1;
                        let mut z = Mat::<f64>::zeros(nrow, r);
                        for t in 0..kc {
                            for j in 0..r {
                                z[(t, j)] = sig[j] * v[(rg.start + t, j)];
                            }
                        }
                        for t in kc..nrow {
                            for i in 0..r {
                                z[(t, i)] = (0..r).map(|j| p[(i, j)] * z[(t - 1, j)]).sum();
                            }
                        }
                        let y = hankelize_sum(&u, &z);
                        Ok(y[self.n[c]..self.n[c] + len].to_vec())
                    })
                    .collect()
            }
            Direction::Row => {
                let s = self.n.len();
                let last: Vec<usize> = rows.iter().map(|rg| rg.end - 1).collect();
                let head: Vec<usize> = rows.iter().flat_map(|rg| rg.start..rg.end - 1).collect();
                let tail: Vec<usize> = rows.iter().flat_map(|rg| rg.start + 1..rg.end).collect();
                let pi = Mat::from_fn(s, r, |c, j| v[(last[c], j)]);
                let vh = Mat::from_fn(head.len(), r, |i, j| v[(head[i], j)]);
                let vt = Mat::from_fn(tail.len(), r, |i, j| v[(tail[i], j)]);
                let t = mm(vh.transpose(), vt.as_ref());
                let ippt = Mat::from_fn(s, s, |i, j| if i == j { 1.0 } else { 0.0 } - (0..r).map(|m| pi[(i, m)] * pi[(j, m)]).sum::<f64>());
                let sol = crate::ssa::lstsq(ippt.as_ref(), mm(pi.as_ref(), t.as_ref()).as_ref())?;
                let corr = mm(pi.transpose(), sol.as_ref());
                let p = Mat::from_fn(r, r, |i, j| t[(i, j)] + corr[(i, j)]);
                let kmax = *self.k.iter().max().unwrap();
                let nrow = l + len + kmax - 1;
                let mut z = Mat::<f64>::zeros(nrow, r);
                for tt in 0..l {
                    for j in 0..r {
                        z[(tt, j)] = sig[j] * u[(tt, j)];
                    }
                }
                for tt in l..nrow {
                    for i in 0..r {
                        z[(tt, i)] = (0..r).map(|j| p[(i, j)] * z[(tt - 1, j)]).sum();
                    }
                }
                rows.iter()
                    .enumerate()
                    .map(|(c, rg)| {
                        let kc = self.k[c];
                        let zc = z.as_ref().subrows(0, l + len + kc - 1).to_owned();
                        let vb = Mat::from_fn(kc, r, |t, j| v[(rg.start + t, j)]);
                        let y = hankelize_sum(&zc, &vb);
                        Ok(y[self.n[c]..self.n[c] + len].to_vec())
                    })
                    .collect()
            }
        }
    }

    /// Gap filling by forward/backward LRR forecasts (Rssa `gapfill.mssa`).
    pub fn gapfill(&self, group: &[usize], reconstructed_base: bool) -> Result<Vec<Vec<f64>>> {
        self.nd.check(group)?;
        let f = if reconstructed_base { self.reconstruct(group)? } else { self.series.clone() };
        let ug = select_cols(self.nd.u.as_ref(), group);
        let blrr = lrr_from_basis(&ug, true, true)?;
        let flrr = lrr_from_basis(&ug, false, true)?;
        let l = self.l;
        f.iter()
            .map(|s| {
                let mut res = s.clone();
                for gap in classify_gaps(s, l) {
                    let len = gap.right - gap.left + 1;
                    let back = || apply_lrr(&s[gap.right + 1..(gap.right + 1 + l).min(s.len())], &blrr, len, true, &[]);
                    let fwd = || apply_lrr(&s[gap.left.saturating_sub(l)..gap.left], &flrr, len, false, &[]);
                    let fill = match gap.pos {
                        GapPos::Left => back()?,
                        GapPos::Right => fwd()?,
                        GapPos::Internal => {
                            let (b, w) = (back()?, fwd()?);
                            (0..len)
                                .map(|i| {
                                    let a = if len == 1 { 0.0 } else { i as f64 / (len - 1) as f64 };
                                    a * b[i] + (1.0 - a) * w[i]
                                })
                                .collect()
                        }
                    };
                    res[gap.left..=gap.right].copy_from_slice(&fill);
                }
                Ok(res)
            })
            .collect()
    }

    /// Iterative gap filling (Rssa `igapfill.mssa`).
    pub fn igapfill(&self, groups: &[Group], fill: Option<f64>, tol: f64, maxiter: usize) -> Result<Vec<Vec<Vec<f64>>>> {
        // Rssa fills every series with the mean of the first one
        let fill = fill.or_else(|| {
            let o: Vec<f64> = self.series[0].iter().copied().filter(|v| !v.is_nan()).collect();
            Some(o.iter().sum::<f64>() / o.len() as f64)
        });
        let outs = self.nd.igapfill(groups, fill, tol, maxiter)?;
        Ok(outs.iter().map(|f| self.split(f)).collect())
    }
}
