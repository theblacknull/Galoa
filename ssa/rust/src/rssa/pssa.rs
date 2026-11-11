//! Projection SSA (Rssa `column.projector` / `row.projector`) and weighted
//! oblique SSA (Rssa `column.oblique` / `row.oblique`, diagonal weights).

use faer::Mat;

use super::common::{wcor_matrix, Group};
use super::forecast::{apply_lrr, hankelize_sum, lrr_from_basis};
use crate::error::{Result, SsaError};
use crate::hankel::{anti_diagonal_len, Convolver};
use crate::matrix::{column, mm, select_cols, Matrix};
use crate::ssa::{construct_trajectory_matrix, SsaModel};
use crate::svd::Decomposition;

/// Projector specification (Rssa: `"none"`, `"centering"`, `"linear"`, ...
/// or an explicit basis).
#[derive(Debug, Clone)]
pub enum Projector {
    None,
    /// Polynomials of degree `< d`: 1 = constant/centering, 2 = linear,
    /// 3 = quadratic, 4 = cubic.
    Degree(usize),
    /// Explicit basis (columns), orthonormalised by QR.
    Basis(Matrix),
}

/// Orthonormal basis of polynomials of degree `< d` on `1..=n` (span of
/// Rssa `orthopoly`).
pub fn orthopoly(d: usize, n: usize) -> Matrix {
    if d == 0 {
        return Mat::zeros(n, 0);
    }
    let c = (n as f64 + 1.0) / 2.0;
    let raw = Mat::from_fn(n, d, |i, j| (i as f64 + 1.0 - c).powi(j as i32));
    raw.qr().compute_thin_Q()
}

fn projector_basis(p: &Projector, n: usize) -> Matrix {
    match p {
        Projector::None => Mat::zeros(n, 0),
        Projector::Degree(d) => orthopoly(*d, n),
        Projector::Basis(b) => b.qr().compute_thin_Q(),
    }
}

/// `(I - P_L) X (I - P_R)` applied matrix-free (Rssa `.phmat`).
pub struct ProjectedOp {
    h: crate::hmat::HankelOp,
    lu: Matrix,
    rv: Matrix,
}

fn project_out(b: &Matrix, x: &mut [f64]) {
    for j in 0..b.ncols() {
        let d: f64 = b.col(j).iter().zip(x.iter()).map(|(a, c)| a * c).sum();
        x.iter_mut().zip(b.col(j).iter()).for_each(|(v, a)| *v -= d * a);
    }
}

impl crate::hmat::LinOp for ProjectedOp {
    fn nrows(&self) -> usize {
        self.h.nrows()
    }
    fn ncols(&self) -> usize {
        self.h.ncols()
    }
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        let mut xv = x.to_vec();
        project_out(&self.rv, &mut xv);
        self.h.matvec(&xv, y);
        project_out(&self.lu, y);
    }
    fn rmatvec(&self, x: &[f64], y: &mut [f64]) {
        let mut xv = x.to_vec();
        project_out(&self.lu, &mut xv);
        self.h.rmatvec(&xv, y);
        project_out(&self.rv, y);
    }
}

/// Fitted Projection SSA: an ordinary model whose first `n_pr + n_pl`
/// triples are the special (projection) triples.
#[derive(Debug, Clone)]
pub struct Pssa {
    pub model: SsaModel,
    /// Number of row-projection ("right") special triples.
    pub n_pr: usize,
    /// Number of column-projection ("left") special triples.
    pub n_pl: usize,
}

impl Pssa {
    /// Projection SSA of a series.  `neig = None` keeps every non-special
    /// triple of the projected trajectory matrix.
    pub fn fit(x: &[f64], l: usize, column_projector: &Projector, row_projector: &Projector, neig: Option<usize>) -> Result<Self> {
        let n = x.len();
        if l < 2 || l > n {
            return Err(SsaError::InvalidArgument("invalid window length".into()));
        }
        let k = n - l + 1;
        let xs = column(x);
        let lu = projector_basis(column_projector, l);
        let rv = projector_basis(row_projector, k);
        if lu.nrows() != l || rv.nrows() != k {
            return Err(SsaError::InvalidArgument("projector size does not match the trajectory matrix".into()));
        }
        let h = crate::hmat::HankelOp::new(xs.as_ref(), l, &[1.0]);
        use crate::hmat::LinOp;
        // special triples: RU = X RV, LV = X^T LU - RV (RU^T LU)
        let mut ru = Mat::<f64>::zeros(l, rv.ncols());
        let mut buf = vec![0.0; l];
        for j in 0..rv.ncols() {
            let c: Vec<f64> = rv.col(j).iter().copied().collect();
            h.matvec(&c, &mut buf);
            for i in 0..l {
                ru[(i, j)] = buf[i];
            }
        }
        let rtl = mm(ru.transpose(), lu.as_ref()); // (pR, pL)
        let mut lv = Mat::<f64>::zeros(k, lu.ncols());
        let mut bufk = vec![0.0; k];
        for j in 0..lu.ncols() {
            let c: Vec<f64> = lu.col(j).iter().copied().collect();
            h.rmatvec(&c, &mut bufk);
            for i in 0..k {
                lv[(i, j)] = bufk[i];
            }
        }
        let corr = mm(rv.as_ref(), rtl.as_ref());
        for i in 0..k {
            for j in 0..lv.ncols() {
                lv[(i, j)] -= corr[(i, j)];
            }
        }
        let rs: Vec<f64> = (0..ru.ncols()).map(|j| ru.col(j).norm_l2()).collect();
        let ls: Vec<f64> = (0..lv.ncols()).map(|j| lv.col(j).norm_l2()).collect();
        for j in 0..ru.ncols() {
            for i in 0..l {
                ru[(i, j)] /= rs[j];
            }
        }
        for j in 0..lv.ncols() {
            for i in 0..k {
                lv[(i, j)] /= ls[j];
            }
        }
        let maxe = (l - lu.ncols()).min(k - rv.ncols());
        // Rssa "auto": dense below L = 500 (or without neig), else Lanczos on
        // the projected FFT operator.
        let (su, ss, sv) = match neig {
            Some(ne) if l >= 500 && 2 * ne <= maxe => {
                let op = ProjectedOp { h, lu: lu.clone(), rv: rv.clone() };
                crate::lanczos::lanczos_svd(&op, ne, 1e-12, 0x5eed)?
            }
            _ => {
                let traj = construct_trajectory_matrix(xs.as_ref(), l, &[1.0]);
                let mut p = traj;
                let xr = mm(p.as_ref(), rv.as_ref());
                let sub = mm(xr.as_ref(), rv.transpose());
                p = &p - &sub;
                let lx = mm(lu.transpose(), p.as_ref());
                let sub = mm(lu.as_ref(), lx.as_ref());
                p = &p - &sub;
                crate::la::thin_svd(p.as_ref())?
            }
        };
        let ne = neig.unwrap_or(maxe).min(maxe).min(ss.len());
        let (npr, npl) = (rv.ncols(), lu.ncols());
        let d = npr + npl + ne;
        let u = Mat::from_fn(l, d, |i, j| if j < npr { ru[(i, j)] } else if j < npr + npl { lu[(i, j - npr)] } else { su[(i, j - npr - npl)] });
        let v = Mat::from_fn(k, d, |i, j| if j < npr { rv[(i, j)] } else if j < npr + npl { lv[(i, j - npr)] } else { sv[(i, j - npr - npl)] });
        let sigma: Vec<f64> = rs.iter().chain(&ls).chain(&ss[..ne]).copied().collect();
        let e = crate::svd::Trajectory { x_s: xs.as_ref(), l, weights: &[1.0] }.frob_norm2();
        let model = SsaModel::from_parts(xs, l, vec![1.0], Decomposition { u, sigma, v }, e);
        Ok(Self { model, n_pr: npr, n_pl: npl })
    }

    /// Recurrent forecast of Projection SSA (Rssa `rforecast.pssa.1d.ssa`):
    /// the LRR of the non-row-special part plus the drift generated by the
    /// row-special part.  `original` uses the series itself as base (and all
    /// row-special triples for the drift); `reverse` forecasts backwards.
    pub fn rforecast_ext(&self, group: &[usize], len: usize, original: bool, reverse: bool) -> Result<Vec<f64>> {
        let m = &self.model;
        m.check_indices(group)?;
        let all_right: Vec<usize> = (0..self.n_pr).collect();
        let right: Vec<usize> = if original { all_right } else { group.iter().copied().filter(|&i| i < self.n_pr).collect() };
        let nonright: Vec<usize> = group.iter().copied().filter(|&i| i >= self.n_pr).collect();
        let base: Vec<f64> = if original {
            m.x_s.col(0).iter().copied().collect()
        } else {
            m.reconstruct_components(group)?.col(0).iter().copied().collect()
        };
        let lf = lrr_from_basis(&select_cols(m.u.as_ref(), &nonright), reverse, false)?;
        let ur = select_cols(m.u.as_ref(), &right);
        let vr = select_cols(m.v.as_ref(), &right);
        let drift = if right.is_empty() {
            vec![0.0; len]
        } else {
            let rlf = lrr_from_basis(&vr, reverse, false)?;
            // drift = ((c(-lf, 1) %*% U_r) * sigma_r) %*% t(V_r)   (reverse: c(1, -lf))
            let a: Vec<f64> = if reverse {
                std::iter::once(1.0).chain(lf.iter().map(|v| -v)).collect()
            } else {
                lf.iter().map(|v| -v).chain(std::iter::once(1.0)).collect()
            };
            let coef: Vec<f64> = (0..right.len()).map(|j| (0..m.l).map(|i| a[i] * ur[(i, j)]).sum::<f64>() * m.sigma[right[j]]).collect();
            let ds: Vec<f64> = (0..m.k).map(|t| (0..right.len()).map(|j| coef[j] * vr[(t, j)]).sum()).collect();
            apply_lrr(&ds, &rlf, len, reverse, &[])?
        };
        apply_lrr(&base, &lf, len, reverse, &drift)
    }

    /// Recurrent forecast from the reconstructed group (Rssa default).
    pub fn rforecast(&self, group: &[usize], len: usize) -> Result<Vec<f64>> {
        self.rforecast_ext(group, len, false, false)
    }

    /// Vector forecast of Projection SSA (Rssa `vforecast.pssa.1d.ssa`).
    pub fn vforecast(&self, group: &[usize], len: usize) -> Result<Vec<f64>> {
        let m = &self.model;
        m.check_indices(group)?;
        let mut g: Vec<usize> = Vec::new();
        for &i in group {
            if !g.contains(&i) {
                g.push(i);
            }
        }
        let right: Vec<usize> = g.iter().copied().filter(|&i| i < self.n_pr).collect();
        let nonright: Vec<usize> = g.iter().copied().filter(|&i| i >= self.n_pr).collect();
        let (l, k) = (m.l, m.k);
        let (rr, rn) = (right.len(), nonright.len());
        let ur = select_cols(m.u.as_ref(), &right);
        let un = select_cols(m.u.as_ref(), &nonright);
        let zr = Mat::from_fn(k, rr, |i, j| m.v[(i, right[j])] * m.sigma[right[j]]);
        let zn = Mat::from_fn(k, rn, |i, j| m.v[(i, nonright[j])] * m.sigma[nonright[j]]);
        let ls = |a: &Matrix, n: usize| -> Result<Matrix> {
            crate::ssa::lstsq(a.as_ref().subrows(0, n - 1), a.as_ref().subrows(1, n - 1))
        };
        // Pright = t(shift(Z_right)), Pnonright = shift(U_nonright)
        let pright = if rr > 0 { ls(&zr, k)?.transpose().to_owned() } else { Mat::zeros(0, 0) };
        let pnon = if rn > 0 { ls(&un, l)? } else { Mat::zeros(0, 0) };
        let pp = if rn > 0 && rr > 0 {
            let rhs = Mat::from_fn(l - 1, rr, |i, j| ur[(i + 1, j)] - (0..rr).map(|q| ur[(i, q)] * pright[(q, j)]).sum::<f64>());
            crate::ssa::lstsq(un.as_ref().subrows(0, l - 1), rhs.as_ref())?
        } else {
            Mat::zeros(rn, rr)
        };
        let r = rr + rn;
        let u = Mat::from_fn(l, r, |i, j| if j < rr { ur[(i, j)] } else { un[(i, j - rr)] });
        let p = Mat::from_fn(r, r, |i, j| match (i < rr, j < rr) {
            (true, true) => pright[(i, j)],
            (true, false) => 0.0,
            (false, true) => pp[(i - rr, j)],
            (false, false) => pnon[(i - rr, j - rr)],
        });
        let rows = k + len + l - 1;
        let mut z = Mat::<f64>::zeros(rows, r);
        for t in 0..k {
            for j in 0..r {
                z[(t, j)] = if j < rr { zr[(t, j)] } else { zn[(t, j - rr)] };
            }
        }
        for t in k..rows {
            for i in 0..r {
                z[(t, i)] = (0..r).map(|j| p[(i, j)] * z[(t - 1, j)]).sum();
            }
        }
        let y = hankelize_sum(&u, &z);
        Ok(y[m.n..m.n + len].to_vec())
    }
}

/// Weighted oblique SSA with diagonal column/row weights (Rssa
/// `column.oblique` / `row.oblique` given as vectors).
#[derive(Debug, Clone)]
pub struct Wossa {
    pub x: Vec<f64>,
    pub l: usize,
    pub k: usize,
    pub column_weights: Vec<f64>,
    pub row_weights: Vec<f64>,
    pub u: Matrix,
    pub sigma: Vec<f64>,
    pub v: Matrix,
    /// Singular values of the weighted matrix (Rssa `osigma`).
    pub osigma: Vec<f64>,
    /// Anti-diagonal sums of `c r^T` (reconstruction weights).
    wsum: Vec<f64>,
}

impl Wossa {
    pub fn fit(x: &[f64], l: usize, column_weights: &[f64], row_weights: &[f64]) -> Result<Self> {
        let n = x.len();
        let k = n - l + 1;
        if column_weights.len() != l || row_weights.len() != k {
            return Err(SsaError::InvalidArgument("oblique weights must have lengths L and K".into()));
        }
        let a: Vec<f64> = column_weights.iter().map(|v| v.sqrt()).collect();
        let b: Vec<f64> = row_weights.iter().map(|v| v.sqrt()).collect();
        let inv = |v: f64| if v.abs() < 1e-6 { 0.0 } else { 1.0 / v };
        let w = Mat::from_fn(l, k, |i, j| a[i] * x[i + j] * b[j]);
        let (ou, os, ov) = crate::la::thin_svd(w.as_ref())?;
        let d = os.len();
        let mut u = Mat::from_fn(l, d, |i, j| inv(a[i]) * ou[(i, j)]);
        let mut v = Mat::from_fn(k, d, |i, j| inv(b[i]) * ov[(i, j)]);
        let mut sigma = os.clone();
        for j in 0..d {
            let (su, sv) = (u.col(j).norm_l2(), v.col(j).norm_l2());
            sigma[j] *= su * sv;
            for i in 0..l {
                u[(i, j)] /= su;
            }
            for i in 0..k {
                v[(i, j)] /= sv;
            }
        }
        let cm = Mat::from_fn(l, 1, |i, _| column_weights[i]);
        let rm = Mat::from_fn(k, 1, |i, _| row_weights[i]);
        let wsum: Vec<f64> = hankelize_sum(&cm, &rm).iter().enumerate().map(|(t, v)| v * anti_diagonal_len(t, l, k) as f64).collect();
        Ok(Self { x: x.to_vec(), l, k, column_weights: column_weights.to_vec(), row_weights: row_weights.to_vec(), u, sigma, v, osigma: os, wsum })
    }

    /// Reconstruction with oblique (weighted) hankelisation.
    pub fn reconstruct(&self, group: &[usize]) -> Result<Vec<f64>> {
        if let Some(&i) = group.iter().find(|&&i| i >= self.sigma.len()) {
            return Err(SsaError::IndexOutOfRange { index: i, len: self.sigma.len() });
        }
        let conv = Convolver::new(self.l, self.k);
        let mut acc = conv.accumulator();
        for &i in group {
            let a = conv.spectrum((0..self.l).map(|r| self.u[(r, i)] * self.column_weights[r]), self.sigma[i]);
            let b = conv.spectrum((0..self.k).map(|r| self.v[(r, i)] * self.row_weights[r]), 1.0);
            conv.add_product(&mut acc, &a, &b);
        }
        Ok(conv.finish(acc).iter().zip(&self.wsum).map(|(v, w)| v / w).collect())
    }

    /// Signed w-correlations with the oblique weights.
    pub fn wcor(&self, groups: &[Group]) -> Result<Matrix> {
        let cols: Vec<Vec<f64>> = groups.iter().map(|g| self.reconstruct(g)).collect::<Result<_>>()?;
        let fs = Mat::from_fn(self.x.len(), cols.len(), |t, j| cols[j][t]);
        Ok(wcor_matrix(&fs, &self.wsum))
    }
}

impl Wossa {
    /// LRR of a group (orthonormalised basis, as Rssa `lrr` for an oblique
    /// decomposition).
    pub fn lrr(&self, group: &[usize], reverse: bool) -> Result<Vec<f64>> {
        if let Some(&i) = group.iter().find(|&&i| i >= self.sigma.len()) {
            return Err(SsaError::IndexOutOfRange { index: i, len: self.sigma.len() });
        }
        lrr_from_basis(&select_cols(self.u.as_ref(), group), reverse, true)
    }

    /// Recurrent forecast from the (oblique) reconstruction of `group`.
    pub fn rforecast(&self, group: &[usize], len: usize) -> Result<Vec<f64>> {
        let f = self.reconstruct(group)?;
        apply_lrr(&f, &self.lrr(group, false)?, len, false, &[])
    }

    /// Vector forecast (Rssa `vforecast.1d.ssa` on W-OSSA): factor vectors
    /// continued by the shift matrix of `U`, then oblique hankelisation.
    pub fn vforecast(&self, group: &[usize], len: usize) -> Result<Vec<f64>> {
        let mut g: Vec<usize> = Vec::new();
        for &i in group {
            if i >= self.sigma.len() {
                return Err(SsaError::IndexOutOfRange { index: i, len: self.sigma.len() });
            }
            if !g.contains(&i) {
                g.push(i);
            }
        }
        let (l, k, r) = (self.l, self.k, g.len());
        let u = select_cols(self.u.as_ref(), &g);
        let p = crate::ssa::lstsq(u.as_ref().subrows(0, l - 1), u.as_ref().subrows(1, l - 1))?;
        let rows = k + len + l - 1;
        let mut z = Mat::<f64>::zeros(rows, r);
        for t in 0..k {
            for j in 0..r {
                z[(t, j)] = self.sigma[g[j]] * self.v[(t, g[j])];
            }
        }
        for t in k..rows {
            for i in 0..r {
                z[(t, i)] = (0..r).map(|j| p[(i, j)] * z[(t - 1, j)]).sum();
            }
        }
        let y = hankelize_sum(&u, &z);
        Ok(y[self.x.len()..self.x.len() + len].to_vec())
    }
}

/// W-OSSA on top of Projection SSA for a series (Rssa `ssa(x, L,
/// column.projector, row.projector, column.oblique, row.oblique)`, which
/// Rssa itself cannot run).  Special triples come first (see
/// [`super::nd::NdSsa::nspecial`]).
pub fn wossa_pssa(
    x: &[f64],
    l: usize,
    column_projector: Projector,
    row_projector: Projector,
    column_weights: Option<Vec<f64>>,
    row_weights: Option<Vec<f64>>,
) -> Result<super::nd::NdSsa> {
    super::nd::NdSsaBuilder::new(x.to_vec(), vec![x.len()])
        .window(vec![l])
        .projectors(column_projector, row_projector)
        .oblique(column_weights, row_weights)
        .method(super::nd::NdMethod::Full)
        .fit()
}
