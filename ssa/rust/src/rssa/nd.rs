//! General shaped / circular nD-SSA (Rssa `kind = "1d-ssa"`, `"2d-ssa"`,
//! `"nd-ssa"` with `mask`, `wmask`, `circular`, projectors and oblique
//! weights; also the engine behind MSSA of series of different lengths).
//!
//! A field `F` of shape `N = (N1, ..., Nr)` (column-major) is analysed with
//! a window of shape `L` restricted to `wmask`.  A window position `k` is used
//! ("factor mask" `fmask`) when every window cell falls on an observed
//! element; periodic dimensions wrap around.  Element `n` of the field is
//! covered by `weights[n]` windows.  The trajectory matrix has one row per
//! window cell and one column per used position, and is applied through
//! nD FFT correlations (never formed unless an exact dense SVD is asked).

use faer::Mat;
use rayon::prelude::*;

use super::common::{cov2cor, wcor_matrix, Group};
use super::parest::{roots2pars, shift_matrix_rows, Params, SolveMethod, Subspace};
use super::pssa::{orthopoly, Projector};
use crate::error::{Result, SsaError};
use crate::hmat::LinOp;
use crate::matrix::{mm, select_cols, Matrix};
use crate::ndfft::{strides, unravel, Grid, C64};
use crate::ssa::Complex64;

/// Decomposition method (Rssa `svd.method`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NdMethod {
    /// Rssa "auto": exact below 500 trajectory rows, else Lanczos(`neig`).
    Auto(usize),
    /// Exact dense SVD (all triples).
    Full,
    /// Leading `k` triples by Lanczos on the FFT operator.
    Lanczos(usize),
}

/// Pairing of the per-dimension ESPRIT roots (Rssa `pairing.method`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pairing {
    Diag,
    Memp,
}

/// Window shape helpers (Rssa `circle()` / `triangle()` in `wmask`).
pub fn ball_mask(r: usize, rank: usize) -> (Vec<usize>, Vec<bool>) {
    let side = 2 * r - 1;
    let dims = vec![side; rank];
    let n: usize = dims.iter().product();
    let m = (0..n)
        .map(|i| {
            let mi = unravel(i, &dims);
            let d2: i64 = mi.iter().map(|&v| (v as i64 + 1 - r as i64).pow(2)).sum();
            d2 <= ((r as i64) - 1).pow(2)
        })
        .collect();
    (dims, m)
}

/// Rssa `.simplex.mask`: cells with `sum(index) <= side + rank - 1` (1-based).
pub fn simplex_mask(side: usize, rank: usize) -> (Vec<usize>, Vec<bool>) {
    let dims = vec![side; rank];
    let n: usize = dims.iter().product();
    let m = (0..n).map(|i| unravel(i, &dims).iter().map(|v| v + 1).sum::<usize>() <= side + rank - 1).collect();
    (dims, m)
}

/// Builder of an [`NdSsa`].
#[derive(Debug, Clone)]
pub struct NdSsaBuilder {
    pub n: Vec<usize>,
    pub field: Vec<f64>,
    pub l: Option<Vec<usize>>,
    pub wmask: Option<Vec<bool>>,
    pub mask: Option<Vec<bool>>,
    pub circular: Vec<bool>,
    /// Explicit factor mask (dims `K`) and weights (dims `N`), as Rssa MSSA
    /// builds them; computed from the masks when `None`.
    pub fmask: Option<Vec<bool>>,
    pub weights: Option<Vec<usize>>,
    pub column_projector: Projector,
    pub row_projector: Projector,
    pub column_oblique: Option<Vec<f64>>,
    pub row_oblique: Option<Vec<f64>>,
    /// Toeplitz SSA (rank 1 only); `Some(neig)` like [`crate::ToeplitzSvd`].
    pub toeplitz: Option<Option<usize>>,
    pub method: NdMethod,
    /// Field is a set of series stacked as columns (MSSA layout, window `(L, 1)`).
    pub mssa: bool,
    /// Elements that belong to the data (`NaN` outside it is padding, not a gap).
    pub domain: Option<Vec<bool>>,
}

impl NdSsaBuilder {
    /// `field` is column-major with shape `n`; `NaN` marks missing values.
    pub fn new(field: Vec<f64>, n: Vec<usize>) -> Self {
        let r = n.len();
        Self {
            n,
            field,
            l: None,
            wmask: None,
            mask: None,
            circular: vec![false; r],
            fmask: None,
            weights: None,
            column_projector: Projector::None,
            row_projector: Projector::None,
            column_oblique: None,
            row_oblique: None,
            toeplitz: None,
            method: NdMethod::Auto(50),
            mssa: false,
            domain: None,
        }
    }
    pub fn window(mut self, l: Vec<usize>) -> Self {
        self.l = Some(l);
        self
    }
    /// Shaped window `wmask` of shape `dims` (sets the window size).
    pub fn wmask(mut self, dims: Vec<usize>, wmask: Vec<bool>) -> Self {
        self.l = Some(dims);
        self.wmask = Some(wmask);
        self
    }
    pub fn mask(mut self, mask: Vec<bool>) -> Self {
        self.mask = Some(mask);
        self
    }
    pub fn circular(mut self, c: Vec<bool>) -> Self {
        self.circular = c;
        self
    }
    pub fn projectors(mut self, column: Projector, row: Projector) -> Self {
        self.column_projector = column;
        self.row_projector = row;
        self
    }
    pub fn oblique(mut self, column: Option<Vec<f64>>, row: Option<Vec<f64>>) -> Self {
        self.column_oblique = column;
        self.row_oblique = row;
        self
    }
    pub fn toeplitz(mut self, neig: Option<usize>) -> Self {
        self.toeplitz = Some(neig);
        self
    }
    pub fn method(mut self, m: NdMethod) -> Self {
        self.method = m;
        self
    }
    pub fn fit(self) -> Result<NdSsa> {
        NdSsa::fit(self)
    }
}

/// Fitted shaped / circular nD-SSA.
#[derive(Debug, Clone)]
pub struct NdSsa {
    pub spec: NdSsaBuilder,
    pub n: Vec<usize>,
    pub l: Vec<usize>,
    pub k: Vec<usize>,
    pub circular: Vec<bool>,
    /// Observed-and-masked elements.
    pub mask: Vec<bool>,
    pub wmask: Vec<bool>,
    pub fmask: Vec<bool>,
    pub weights: Vec<usize>,
    /// Linear indices (in `L` dims) of the window cells, and (in `K` dims) of
    /// the used window positions.
    pub wpos: Vec<usize>,
    pub fpos: Vec<usize>,
    /// Field with non-masked values replaced by the masked mean.
    pub filled: Vec<f64>,
    pub u: Matrix,
    pub sigma: Vec<f64>,
    pub v: Matrix,
    pub d: usize,
    /// Special (projection) triples: row-projection then column-projection.
    pub n_pr: usize,
    pub n_pl: usize,
    /// Oblique weights `(column (L dims), row (K dims), anti-diagonal sums)`.
    oblique: Option<(Vec<f64>, Vec<f64>, Vec<f64>)>,
    /// Grid positions of the window cells, used window positions and field
    /// elements (FFT grid of the field).
    gw: Vec<usize>,
    gf: Vec<usize>,
    gn: Vec<usize>,
}

/// Matrix-free trajectory operator of an [`NdSsa`].
pub struct NdOp<'a> {
    pub m: &'a NdSsa,
    grid: &'a Grid,
    fspec: &'a [C64],
}

impl NdOp<'_> {
    /// `out[r] = sum_i f[g_out[r] + i] a[i]` with `a` scattered at `g_in`.
    fn corr(&self, x: &[f64], g_in: &[usize], g_out: &[usize], y: &mut [f64]) {
        let mut buf = vec![0.0; self.grid.len()];
        for (v, &p) in x.iter().zip(g_in) {
            buf[p] = *v;
        }
        let mut spec = self.grid.forward(&mut buf);
        for (z, f) in spec.iter_mut().zip(self.fspec) {
            *z = f * z.conj();
        }
        let r = self.grid.inverse(spec);
        let s = 1.0 / self.grid.len() as f64;
        for (o, &p) in y.iter_mut().zip(g_out) {
            *o = r[p] * s;
        }
    }
}

impl LinOp for NdOp<'_> {
    fn nrows(&self) -> usize {
        self.m.wpos.len()
    }
    fn ncols(&self) -> usize {
        self.m.fpos.len()
    }
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        self.corr(x, &self.m.gf, &self.m.gw, y);
    }
    fn rmatvec(&self, x: &[f64], y: &mut [f64]) {
        self.corr(x, &self.m.gw, &self.m.gf, y);
    }
}

/// `(I - P_L) A (I - P_R)` for any operator.
struct Projected<'a> {
    a: &'a dyn LinOp,
    lu: &'a Matrix,
    rv: &'a Matrix,
}

fn project_out(b: &Matrix, x: &mut [f64]) {
    for j in 0..b.ncols() {
        let d: f64 = b.col(j).iter().zip(x.iter()).map(|(a, c)| a * c).sum();
        x.iter_mut().zip(b.col(j).iter()).for_each(|(v, a)| *v -= d * a);
    }
}

impl LinOp for Projected<'_> {
    fn nrows(&self) -> usize {
        self.a.nrows()
    }
    fn ncols(&self) -> usize {
        self.a.ncols()
    }
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        let mut xv = x.to_vec();
        project_out(self.rv, &mut xv);
        self.a.matvec(&xv, y);
        project_out(self.lu, y);
    }
    fn rmatvec(&self, x: &[f64], y: &mut [f64]) {
        let mut xv = x.to_vec();
        project_out(self.lu, &mut xv);
        self.a.rmatvec(&xv, y);
        project_out(self.rv, y);
    }
}

/// `diag(a) A diag(b)`.
/// PSSA special triples: `RU = X RV`, `LV = X^T LU - RV (RU^T LU)`,
/// normalised (norms returned as the singular values).
fn special_triples(base: &dyn LinOp, lu: &Matrix, rv: &Matrix, rows: usize, cols: usize) -> (Matrix, Vec<f64>, Matrix, Vec<f64>) {
    let mut ru = Mat::<f64>::zeros(rows, rv.ncols());
    for j in 0..rv.ncols() {
        let c: Vec<f64> = rv.col(j).iter().copied().collect();
        let mut y = vec![0.0; rows];
        base.matvec(&c, &mut y);
        ru.col_mut(j).iter_mut().zip(&y).for_each(|(a, b)| *a = *b);
    }
    let rtl = mm(ru.transpose(), lu.as_ref());
    let mut lv = Mat::<f64>::zeros(cols, lu.ncols());
    for j in 0..lu.ncols() {
        let c: Vec<f64> = lu.col(j).iter().copied().collect();
        let mut z = vec![0.0; cols];
        base.rmatvec(&c, &mut z);
        lv.col_mut(j).iter_mut().zip(&z).for_each(|(a, b)| *a = *b);
    }
    let corr = mm(rv.as_ref(), rtl.as_ref());
    lv = &lv - &corr;
    let rs: Vec<f64> = (0..ru.ncols()).map(|j| ru.col(j).norm_l2()).collect();
    let ls: Vec<f64> = (0..lv.ncols()).map(|j| lv.col(j).norm_l2()).collect();
    for j in 0..ru.ncols() {
        ru.col_mut(j).iter_mut().for_each(|x| *x /= rs[j]);
    }
    for j in 0..lv.ncols() {
        lv.col_mut(j).iter_mut().for_each(|x| *x /= ls[j]);
    }
    (ru, rs, lv, ls)
}

struct Scaled<'a> {
    a: &'a dyn LinOp,
    left: &'a [f64],
    right: &'a [f64],
}

impl LinOp for Scaled<'_> {
    fn nrows(&self) -> usize {
        self.a.nrows()
    }
    fn ncols(&self) -> usize {
        self.a.ncols()
    }
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        let xv: Vec<f64> = x.iter().zip(self.right).map(|(a, b)| a * b).collect();
        self.a.matvec(&xv, y);
        y.iter_mut().zip(self.left).for_each(|(v, a)| *v *= a);
    }
    fn rmatvec(&self, x: &[f64], y: &mut [f64]) {
        let xv: Vec<f64> = x.iter().zip(self.left).map(|(a, b)| a * b).collect();
        self.a.rmatvec(&xv, y);
        y.iter_mut().zip(self.right).for_each(|(v, a)| *v *= a);
    }
}

/// Dense matrix of an operator (columns `A e_j`), parallel.
fn densify(op: &dyn LinOp) -> Matrix {
    let (m, n) = (op.nrows(), op.ncols());
    if m <= n {
        let rows: Vec<Vec<f64>> = (0..m)
            .into_par_iter()
            .map(|i| {
                let mut e = vec![0.0; m];
                e[i] = 1.0;
                let mut z = vec![0.0; n];
                op.rmatvec(&e, &mut z);
                z
            })
            .collect();
        Mat::from_fn(m, n, |i, j| rows[i][j])
    } else {
        let cols: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|j| {
                let mut e = vec![0.0; n];
                e[j] = 1.0;
                let mut y = vec![0.0; m];
                op.matvec(&e, &mut y);
                y
            })
            .collect();
        Mat::from_fn(m, n, |i, j| cols[j][i])
    }
}

/// Rssa `svd.method = "eigen"`: eigenvectors of `X X^T` (small `L x L`),
/// `sigma_i = ||X^T u_i||`, `v_i = X^T u_i / sigma_i`; leading `k` triples.
fn gram_triples(op: &dyn LinOp, k: usize) -> Result<(Matrix, Vec<f64>, Matrix)> {
    gram_dense(densify(op), k)
}

/// Leading `k` triples of a dense matrix through the eigenvectors of
/// `X X^T` (Rssa `svd.method = "eigen"`).
fn gram_dense(x: Matrix, k: usize) -> Result<(Matrix, Vec<f64>, Matrix)> {
    let g = mm(x.as_ref(), x.transpose());
    let (_, ev) = crate::la::eigh(g.as_ref())?;
    let l = x.nrows();
    let k = k.min(l);
    let u = Mat::from_fn(l, k, |i, j| ev[(i, l - 1 - j)]);
    let mut z = mm(x.transpose(), u.as_ref());
    let mut sigma = vec![0.0; k];
    for j in 0..k {
        let s = z.col(j).norm_l2();
        sigma[j] = s;
        if s > 0.0 {
            z.col_mut(j).iter_mut().for_each(|v| *v /= s);
        }
    }
    let mut idx: Vec<usize> = (0..k).collect();
    idx.sort_by(|&a, &b| sigma[b].total_cmp(&sigma[a]));
    Ok((
        Mat::from_fn(l, k, |i, j| u[(i, idx[j])]),
        idx.iter().map(|&j| sigma[j]).collect(),
        Mat::from_fn(z.nrows(), k, |i, j| z[(i, idx[j])]),
    ))
}

fn decompose_op(op: &dyn LinOp, method: NdMethod, neig_cap: usize) -> Result<(Matrix, Vec<f64>, Matrix)> {
    if let NdMethod::Auto(k) = method {
        if op.nrows() < 500 && op.nrows() <= op.ncols() {
            let (u, s, v) = gram_triples(op, k.min(neig_cap))?;
            return Ok((u, s, v));
        }
    }
    let rows = op.nrows().min(op.ncols());
    let lanczos = match method {
        NdMethod::Full => None,
        NdMethod::Lanczos(k) => Some(k),
        NdMethod::Auto(k) => if op.nrows() < 500 || 2 * k > rows { None } else { Some(k) },
    };
    let (u, s, v) = match lanczos {
        Some(k) => crate::lanczos::lanczos_svd(op, k.min(rows), 1e-12, 0x5eed)?,
        None => {
            if op.nrows() as f64 * op.ncols() as f64 > 2.0e8 {
                return Err(SsaError::InvalidArgument(
                    "trajectory matrix too large for a dense SVD; use NdMethod::Lanczos".into(),
                ));
            }
            crate::la::thin_svd(densify(op).as_ref())?
        }
    };
    let keep = s.len().min(neig_cap);
    Ok((u.as_ref().subcols(0, keep).to_owned(), s[..keep].to_vec(), v.as_ref().subcols(0, keep).to_owned()))
}

fn basis_rows(p: &Projector, full: usize, dims_rank: usize, keep: &[usize]) -> Result<Matrix> {
    let b = match p {
        Projector::None => return Ok(Mat::zeros(keep.len(), 0)),
        Projector::Degree(d) => {
            if dims_rank > 1 && *d > 1 {
                return Err(SsaError::InvalidArgument("Polynomial projection is not implemented for multidimensional SSA".into()));
            }
            orthopoly(*d, full)
        }
        Projector::Basis(m) => m.clone(),
    };
    if b.nrows() != full {
        return Err(SsaError::InvalidArgument("projector size does not match the trajectory matrix".into()));
    }
    let sub = Mat::from_fn(keep.len(), b.ncols(), |i, j| b[(keep[i], j)]);
    Ok(sub.qr().compute_thin_Q())
}

impl NdSsa {
    pub fn fit(spec: NdSsaBuilder) -> Result<Self> {
        let n = spec.n.clone();
        let rank = n.len();
        let total: usize = n.iter().product();
        if spec.field.len() != total || rank == 0 {
            return Err(SsaError::InvalidArgument("field length does not match its shape".into()));
        }
        let l = spec.l.clone().unwrap_or_else(|| n.iter().map(|d| d.div_ceil(2)).collect());
        if l.len() != rank || l.iter().zip(&n).any(|(a, b)| *a == 0 || a > b) {
            return Err(SsaError::InvalidArgument("invalid window size".into()));
        }
        let circular: Vec<bool> = (0..rank).map(|i| spec.circular[i % spec.circular.len().max(1)]).collect();
        let k: Vec<usize> = (0..rank).map(|i| if circular[i] { n[i] } else { n[i] - l[i] + 1 }).collect();
        let ln: usize = l.iter().product();
        let kn: usize = k.iter().product();
        let wmask = spec.wmask.clone().unwrap_or_else(|| vec![true; ln]);
        if wmask.len() != ln {
            return Err(SsaError::InvalidArgument("wmask size does not match the window".into()));
        }
        let mut mask: Vec<bool> = spec.field.iter().map(|v| !v.is_nan()).collect();
        if let Some(m) = &spec.mask {
            for (a, b) in mask.iter_mut().zip(m) {
                *a &= *b;
            }
        }
        let obs: Vec<f64> = spec.field.iter().zip(&mask).filter(|(_, m)| **m).map(|(v, _)| *v).collect();
        if obs.is_empty() {
            return Err(SsaError::InvalidArgument("Nothing to decompose: the given field shape is empty".into()));
        }
        let mean = obs.iter().sum::<f64>() / obs.len() as f64;
        let filled: Vec<f64> = spec.field.iter().zip(&mask).map(|(v, m)| if *m { *v } else { mean }).collect();
        let grid = Grid::for_field(&n, &circular);
        let wsize = wmask.iter().filter(|b| **b).count() as f64;
        let wf: Vec<f64> = wmask.iter().map(|&b| b as i32 as f64).collect();
        let fmask = match &spec.fmask {
            Some(f) => f.clone(),
            None => {
                let mf: Vec<f64> = mask.iter().map(|&b| b as i32 as f64).collect();
                let ms = grid.spectrum(&mf, &n);
                grid.correlate(&ms, &wf, &l, &k).iter().map(|c| (c - wsize).abs() < 0.5).collect()
            }
        };
        if fmask.len() != kn {
            return Err(SsaError::InvalidArgument("fmask size does not match the factor dimensions".into()));
        }
        let weights = match &spec.weights {
            Some(w) => w.clone(),
            None => {
                let ff: Vec<f64> = fmask.iter().map(|&b| b as i32 as f64).collect();
                let sf = grid.spectrum(&ff, &k);
                let sw = grid.spectrum(&wf, &l);
                grid.convolve_spectra(sf, &sw, &n).iter().map(|v| v.round().max(0.0) as usize).collect()
            }
        };
        let wpos: Vec<usize> = (0..ln).filter(|&i| wmask[i]).collect();
        let fpos: Vec<usize> = (0..kn).filter(|&i| fmask[i]).collect();
        if fpos.is_empty() {
            return Err(SsaError::InvalidArgument("Nothing to decompose: no complete window position".into()));
        }
        let mut m = NdSsa {
            spec: spec.clone(),
            n: n.clone(),
            l: l.clone(),
            k: k.clone(),
            circular,
            mask,
            wmask,
            fmask,
            weights,
            wpos,
            fpos,
            filled,
            u: Mat::zeros(0, 0),
            sigma: Vec::new(),
            v: Mat::zeros(0, 0),
            d: 0,
            n_pr: 0,
            n_pl: 0,
            oblique: None,
            gw: Vec::new(),
            gf: Vec::new(),
            gn: Vec::new(),
        };
        {
            let pl = crate::ndfft::positions(&m.l, &grid.p);
            let pk = crate::ndfft::positions(&m.k, &grid.p);
            m.gw = m.wpos.iter().map(|&i| pl[i]).collect();
            m.gf = m.fpos.iter().map(|&i| pk[i]).collect();
            m.gn = crate::ndfft::positions(&m.n, &grid.p);
        }
        let fspec = grid.spectrum(&m.filled, &n);
        m.decompose(&grid, &fspec)?;
        Ok(m)
    }

    fn op<'a>(&'a self, grid: &'a Grid, fspec: &'a [C64]) -> NdOp<'a> {
        NdOp { m: self, grid, fspec }
    }

    fn decompose(&mut self, grid: &Grid, fspec: &[C64]) -> Result<()> {
        let spec = self.spec.clone();
        let rank = self.n.len();
        let (rows, cols) = (self.wpos.len(), self.fpos.len());
        let base = self.op(grid, fspec);
        if let Some(neig) = spec.toeplitz {
            if rank != 1 {
                return Err(SsaError::InvalidArgument("Toeplitz SSA is defined for one series".into()));
            }
            if rows != self.l[0] || cols != self.k[0] {
                return Err(SsaError::InvalidArgument("gaps are not allowed in Toeplitz SSA".into()));
            }
            let (u, sigma, v) = toeplitz_triples(&self.filled, self.l[0], self.circular[0], neig, &base)?;
            self.set(u, sigma, v);
            return Ok(());
        }
        if spec.column_oblique.is_some() || spec.row_oblique.is_some() {
            if rows != self.l.iter().product::<usize>() || cols != self.k.iter().product::<usize>() {
                return Err(SsaError::InvalidArgument("Weighted-oblique decomposition is not implemented for shaped SSA".into()));
            }
            let c = spec.column_oblique.clone().unwrap_or_else(|| vec![1.0; rows]);
            let r = spec.row_oblique.clone().unwrap_or_else(|| vec![1.0; cols]);
            if c.len() != rows || r.len() != cols {
                return Err(SsaError::InvalidArgument("oblique weights must have prod(L) and prod(K) entries".into()));
            }
            let a: Vec<f64> = c.iter().map(|v| v.sqrt()).collect();
            let b: Vec<f64> = r.iter().map(|v| v.sqrt()).collect();
            let inv = |v: f64| if v.abs() < 1e-6 { 0.0 } else { 1.0 / v };
            // W-OSSA on top of projections: special triples as in PSSA, the
            // projected rest decomposed with the oblique weights (Rssa
            // intends this for `column.projector` + `column.oblique` but
            // recurses forever).
            let lu = basis_rows(&spec.column_projector, self.l.iter().product(), rank, &self.wpos)?;
            let rv = basis_rows(&spec.row_projector, self.k.iter().product(), rank, &self.fpos)?;
            let (npr, npl) = (rv.ncols(), lu.ncols());
            let (ru, rs, lv, ls) = special_triples(&base, &lu, &rv, rows, cols);
            let pop = Projected { a: &base, lu: &lu, rv: &rv };
            let inner: &dyn LinOp = if npr + npl > 0 { &pop } else { &base };
            let wop = Scaled { a: inner, left: &a, right: &b };
            let maxe = if npr + npl > 0 { (rows - npl).min(cols - npr) } else { usize::MAX };
            let (ou, os, ov) = decompose_op(&wop, spec.method, maxe)?;
            let d = os.len();
            let mut u = Mat::from_fn(rows, d, |i, j| inv(a[i]) * ou[(i, j)]);
            let mut v = Mat::from_fn(cols, d, |i, j| inv(b[i]) * ov[(i, j)]);
            let mut sigma = os.clone();
            for j in 0..d {
                let (su, sv) = (u.col(j).norm_l2(), v.col(j).norm_l2());
                sigma[j] *= su * sv;
                u.col_mut(j).iter_mut().for_each(|x| *x /= su);
                v.col_mut(j).iter_mut().for_each(|x| *x /= sv);
            }
            let (u, sigma, v) = if npr + npl > 0 {
                let dd = npr + npl + d;
                let uu = Mat::from_fn(rows, dd, |i, j| if j < npr { ru[(i, j)] } else if j < npr + npl { lu[(i, j - npr)] } else { u[(i, j - npr - npl)] });
                let vv = Mat::from_fn(cols, dd, |i, j| if j < npr { rv[(i, j)] } else if j < npr + npl { lv[(i, j - npr)] } else { v[(i, j - npr - npl)] });
                let ss: Vec<f64> = rs.iter().chain(&ls).chain(&sigma).copied().collect();
                (uu, ss, vv)
            } else {
                (u, sigma, v)
            };
            self.n_pr = npr;
            self.n_pl = npl;
            let wsum = grid.convolve_spectra(grid.spectrum(&c, &self.l), &grid.spectrum(&r, &self.k), &self.n);
            self.oblique = Some((c, r, wsum));
            self.set(u, sigma, v);
            return Ok(());
        }
        let lu = basis_rows(&spec.column_projector, self.l.iter().product(), rank, &self.wpos)?;
        let rv = basis_rows(&spec.row_projector, self.k.iter().product(), rank, &self.fpos)?;
        if lu.ncols() == 0 && rv.ncols() == 0 {
            let (u, s, v) = decompose_op(&base, spec.method, usize::MAX)?;
            self.set(u, s, v);
            return Ok(());
        }
        // Projection SSA: special triples then the projected matrix
        let (ru, rs, lv, ls) = special_triples(&base, &lu, &rv, rows, cols);
        let pop = Projected { a: &base, lu: &lu, rv: &rv };
        let maxe = (rows - lu.ncols()).min(cols - rv.ncols());
        let (su, ss, sv) = decompose_op(&pop, spec.method, maxe)?;
        let (npr, npl) = (rv.ncols(), lu.ncols());
        let d = npr + npl + ss.len();
        let u = Mat::from_fn(rows, d, |i, j| if j < npr { ru[(i, j)] } else if j < npr + npl { lu[(i, j - npr)] } else { su[(i, j - npr - npl)] });
        let v = Mat::from_fn(cols, d, |i, j| if j < npr { rv[(i, j)] } else if j < npr + npl { lv[(i, j - npr)] } else { sv[(i, j - npr - npl)] });
        let sigma: Vec<f64> = rs.iter().chain(&ls).chain(&ss).copied().collect();
        self.n_pr = npr;
        self.n_pl = npl;
        self.set(u, sigma, v);
        Ok(())
    }

    fn set(&mut self, u: Matrix, sigma: Vec<f64>, v: Matrix) {
        self.d = sigma.len();
        self.u = u;
        self.sigma = sigma;
        self.v = v;
    }

    pub fn check(&self, group: &[usize]) -> Result<()> {
        match group.iter().find(|&&i| i >= self.d) {
            Some(&i) => Err(SsaError::IndexOutOfRange { index: i, len: self.d }),
            None => Ok(()),
        }
    }

    /// Number of special (projection) triples (Rssa `nspecial`).
    pub fn nspecial(&self) -> usize {
        self.n_pr + self.n_pl
    }

    /// Reconstructed field of a group (`NaN` where not covered).
    pub fn reconstruct(&self, group: &[usize]) -> Result<Vec<f64>> {
        self.check(group)?;
        let grid = Grid::for_field(&self.n, &self.circular);
        let parts: Vec<Vec<C64>> = group
            .par_iter()
            .map(|&i| {
                let mut a = vec![0.0; grid.len()];
                let mut b = vec![0.0; grid.len()];
                for (r, &p) in self.gw.iter().enumerate() {
                    let w = self.oblique.as_ref().map_or(1.0, |o| o.0[self.wpos[r]]);
                    a[p] = self.u[(r, i)] * self.sigma[i] * w;
                }
                for (c, &p) in self.gf.iter().enumerate() {
                    let w = self.oblique.as_ref().map_or(1.0, |o| o.1[self.fpos[c]]);
                    b[p] = self.v[(c, i)] * w;
                }
                let mut sa = grid.forward(&mut a);
                let sb = grid.forward(&mut b);
                sa.iter_mut().zip(&sb).for_each(|(x, y)| *x *= y);
                sa
            })
            .collect();
        let mut acc = vec![C64::new(0.0, 0.0); grid.sdims.iter().product()];
        for p in parts {
            acc.iter_mut().zip(&p).for_each(|(a, b)| *a += b);
        }
        let r = grid.inverse(acc);
        let sc = 1.0 / grid.len() as f64;
        let sums: Vec<f64> = self.gn.iter().map(|&p| r[p] * sc).collect();
        Ok(match &self.oblique {
            Some((_, _, wsum)) => sums.iter().zip(wsum).zip(&self.weights).map(|((s, w), &c)| if c == 0 { f64::NAN } else { s / w }).collect(),
            None => sums.iter().zip(&self.weights).map(|(s, &w)| if w == 0 { f64::NAN } else { s / w as f64 }).collect(),
        })
    }

    /// Residuals `F - reconstruction(group)` on covered elements.
    pub fn residuals(&self, group: &[usize]) -> Result<Vec<f64>> {
        let r = self.reconstruct(group)?;
        Ok(self.spec.field.iter().zip(&r).map(|(a, b)| a - b).collect())
    }

    /// Elementwise weights used by `wcor` / `wnorm`.
    pub fn hweights(&self) -> Vec<f64> {
        match &self.oblique {
            Some((_, _, wsum)) => wsum.clone(),
            None => self.weights.iter().map(|&w| w as f64).collect(),
        }
    }

    /// Signed w-correlations of groups (Rssa `wcor`).
    pub fn wcor(&self, groups: &[Group]) -> Result<Matrix> {
        let w = self.hweights();
        let keep: Vec<usize> = (0..w.len()).filter(|&t| self.weights[t] > 0).collect();
        let cols: Vec<Vec<f64>> = groups.iter().map(|g| self.reconstruct(g)).collect::<Result<_>>()?;
        let fs = Mat::from_fn(keep.len(), cols.len(), |t, j| cols[j][keep[t]]);
        let wk: Vec<f64> = keep.iter().map(|&t| w[t]).collect();
        Ok(wcor_matrix(&fs, &wk))
    }

    /// Weighted norm of the field (Rssa `wnorm`).
    pub fn wnorm(&self) -> f64 {
        let w = self.hweights();
        (0..w.len()).filter(|&t| self.weights[t] > 0).map(|t| w[t] * self.filled[t].powi(2)).sum::<f64>().sqrt()
    }

    /// `sigma^2 / wnorm^2` (Rssa `contributions`).
    pub fn contributions(&self) -> Vec<f64> {
        let e = self.wnorm().powi(2);
        self.sigma.iter().map(|s| s * s / e).collect()
    }

    /// Same configuration on new field values (non-masked entries are kept).
    pub fn refit_with(&self, values: &[f64], method: NdMethod) -> Result<NdSsa> {
        let mut spec = self.spec.clone();
        spec.field = self.spec.field.iter().zip(values).zip(&self.mask).map(|((o, v), m)| if *m { *v } else { *o }).collect();
        // the same shape must be kept: reuse masks
        spec.fmask = Some(self.fmask.clone());
        spec.weights = Some(self.weights.clone());
        spec.method = method;
        NdSsa::fit(spec)
    }

    /// Shift pairs of the window (or factor) cells along dimension `d`
    /// (Rssa `.shifted.matrix.masks`).
    pub fn shift_pairs(&self, subspace: Subspace, d: usize) -> (Vec<usize>, Vec<usize>) {
        let (dims, mask, circ) = match subspace {
            Subspace::Column => (&self.l, &self.wmask, false),
            Subspace::Row => (&self.k, &self.fmask, self.circular[d]),
        };
        let st = strides(dims);
        let mut ord = vec![usize::MAX; mask.len()];
        let mut c = 0;
        for (i, &m) in mask.iter().enumerate() {
            if m {
                ord[i] = c;
                c += 1;
            }
        }
        let (mut left, mut right) = (Vec::new(), Vec::new());
        for (i, &m) in mask.iter().enumerate() {
            if !m {
                continue;
            }
            let mi = unravel(i, dims);
            let next = if mi[d] + 1 < dims[d] {
                Some(i + st[d])
            } else if circ {
                Some(i + st[d] - dims[d] * st[d])
            } else {
                None
            };
            if let Some(j) = next {
                if mask[j] {
                    left.push(ord[i]);
                    right.push(ord[j]);
                }
            }
        }
        (left, right)
    }

    /// nD-ESPRIT (Rssa `parestimate` for nD-SSA): roots per dimension,
    /// paired with the "diag" or "memp" method (`beta = 8`).
    pub fn parestimate(
        &self,
        group: &[usize],
        subspace: Subspace,
        dimensions: Option<&[usize]>,
        method: SolveMethod,
        pairing: Pairing,
        normalize: Option<&[bool]>,
    ) -> Result<Vec<Params>> {
        self.check(group)?;
        let all: Vec<usize> = (0..self.n.len()).collect();
        let dims = dimensions.unwrap_or(&all);
        let span = match subspace {
            Subspace::Column => select_cols(self.u.as_ref(), group),
            Subspace::Row => select_cols(self.v.as_ref(), group),
        };
        let zs: Vec<Matrix> = dims
            .iter()
            .map(|&d| {
                let (l, r) = self.shift_pairs(subspace, d);
                shift_matrix_rows(&span, &l, &r, method)
            })
            .collect::<Result<_>>()?;
        let mut roots = pair_roots(&zs, pairing, 8.0)?;
        let toe = self.spec.toeplitz.is_some();
        for (i, &d) in dims.iter().enumerate() {
            let norm = normalize.map_or(self.circular[d] || toe, |v| v[i % v.len()]);
            if norm {
                roots[i].iter_mut().for_each(|z| *z /= z.norm());
            }
        }
        Ok(roots.into_iter().map(roots2pars).collect())
    }

    /// Cadzow iterations (Rssa `cadzow` for nD/shaped SSA).
    pub fn cadzow(&self, rank: usize, correct: bool, tol: f64, maxiter: usize) -> Result<Vec<f64>> {
        let g: Vec<usize> = (0..rank).collect();
        let mut f = self.reconstruct(&g)?;
        let mut it = 0;
        loop {
            let s = self.refit_with(&f, NdMethod::Auto(rank))?;
            let rf = s.reconstruct(&g)?;
            it += 1;
            let dist = f
                .iter()
                .zip(&rf)
                .zip(&self.weights)
                .filter(|(_, w)| **w > 0)
                .map(|((a, b), _)| (a - b).abs())
                .fold(0.0, f64::max);
            if (maxiter > 0 && it >= maxiter) || dist < tol {
                break;
            }
            f = rf;
        }
        if correct {
            let w = self.hweights();
            let (mut num, mut den) = (0.0, 0.0);
            for t in 0..f.len() {
                if self.weights[t] > 0 {
                    num += w[t] * self.filled[t] * f[t];
                    den += w[t] * f[t] * f[t];
                }
            }
            let a = num / den;
            f.iter_mut().for_each(|v| *v *= a);
        }
        Ok(f)
    }

    /// Iterative gap filling (Rssa `igapfill` for nD/shaped SSA).
    pub fn igapfill(&self, groups: &[Group], fill: Option<f64>, tol: f64, maxiter: usize) -> Result<Vec<Vec<f64>>> {
        let inside = |i: usize| self.spec.domain.as_ref().is_none_or(|d| d[i]);
        let na: Vec<usize> = (0..self.spec.field.len()).filter(|&i| self.spec.field[i].is_nan() && inside(i)).collect();
        let ug: Vec<usize> = (0..=groups.iter().flatten().copied().max().unwrap_or(0)).collect();
        let obs: Vec<f64> = self.spec.field.iter().copied().filter(|v| !v.is_nan()).collect();
        let fill = fill.unwrap_or(obs.iter().sum::<f64>() / obs.len() as f64);
        let mut f = self.spec.field.clone();
        let _ = &inside;
        for &i in &na {
            f[i] = fill;
        }
        let refit = |vals: &[f64]| -> Result<NdSsa> {
            let mut spec = self.spec.clone();
            spec.field = vals.to_vec();
            spec.method = NdMethod::Auto(ug.len());
            NdSsa::fit(spec)
        };
        let mut it = 0;
        loop {
            let s = refit(&f)?;
            let r = s.reconstruct(&ug)?;
            let mut rf = f.clone();
            for &i in &na {
                rf[i] = r[i];
            }
            let rss = f.iter().zip(&rf).map(|(a, b)| (a - b).powi(2)).fold(0.0, f64::max).sqrt();
            it += 1;
            if (maxiter > 0 && it >= maxiter) || rss < tol {
                break;
            }
            f = rf;
        }
        let s = refit(&f)?;
        groups
            .iter()
            .map(|g| {
                let r = s.reconstruct(g)?;
                let mut out = self.spec.field.clone();
                for &i in &na {
                    out[i] = r[i];
                }
                Ok(out)
            })
            .collect()
    }

    /// Weighted correlation-free helper: trajectory product `X(F) B^T` of a
    /// field (used by `owcor`).
    pub fn traj_times(&self, values: &[f64], bt: &Matrix) -> Matrix {
        let f: Vec<f64> = values.iter().zip(&self.mask).map(|(v, m)| if *m && v.is_finite() { *v } else { 0.0 }).collect();
        let grid = Grid::for_field(&self.n, &self.circular);
        let fs = grid.spectrum(&f, &self.n);
        let op = self.op(&grid, &fs);
        let mut out = Mat::<f64>::zeros(self.wpos.len(), bt.nrows());
        for j in 0..bt.nrows() {
            let row: Vec<f64> = (0..bt.ncols()).map(|c| bt[(j, c)]).collect();
            let mut y = vec![0.0; self.wpos.len()];
            op.matvec(&row, &mut y);
            out.col_mut(j).iter_mut().zip(&y).for_each(|(a, b)| *a = *b);
        }
        out
    }

    /// Filtered factor vectors (Rssa `.filter.vectors`): for each dimension,
    /// `sum_q filter[q] y[k + q e_d]` over positions whose whole filter
    /// window lies in `fmask`; blocks are stacked dimension after dimension.
    pub fn filter_factors(&self, y: &Matrix, filter: &[f64], dims: &[usize]) -> Matrix {
        let kd = &self.k;
        let st = strides(kd);
        let kn: usize = kd.iter().product();
        let mut ord = vec![usize::MAX; kn];
        for (c, &p) in self.fpos.iter().enumerate() {
            ord[p] = c;
        }
        let mut rows: Vec<Vec<usize>> = Vec::new(); // per output row: ordinals of the filter cells
        for &d in dims {
            for &p in &self.fpos {
                let mi = unravel(p, kd);
                let mut cells = Vec::with_capacity(filter.len());
                let mut ok = true;
                for q in 0..filter.len() {
                    let pos = mi[d] + q;
                    let pos = if pos < kd[d] {
                        pos
                    } else if self.circular[d] {
                        pos - kd[d]
                    } else {
                        ok = false;
                        break;
                    };
                    let lin = p + pos * st[d] - mi[d] * st[d];
                    if ord[lin] == usize::MAX {
                        ok = false;
                        break;
                    }
                    cells.push(ord[lin]);
                }
                if ok {
                    rows.push(cells);
                }
            }
        }
        Mat::from_fn(rows.len(), y.ncols(), |i, j| rows[i].iter().zip(filter).map(|(&c, w)| w * y[(c, j)]).sum())
    }
}

/// Toeplitz triples on a full rank-1 window (non-circular or circular).
fn toeplitz_triples(x: &[f64], l: usize, circular: bool, neig: Option<usize>, op: &dyn LinOp) -> Result<(Matrix, Vec<f64>, Matrix)> {
    let n = x.len();
    let c: Vec<f64> = if circular {
        (0..l).map(|m| (0..n).map(|t| x[t] * x[(t + m) % n]).sum::<f64>() / n as f64).collect()
    } else {
        crate::svd::lag_correlations(x, l)
    };
    let neig = neig.unwrap_or(l).min(l);
    let u = if l < 500 || neig * 2 > l {
        let tm = Mat::from_fn(l, l, |i, j| c[i.abs_diff(j)]);
        let (_, ev) = crate::la::eigh(tm.as_ref())?;
        Mat::from_fn(l, neig, |i, j| ev[(i, l - 1 - j)])
    } else {
        crate::lanczos::lanczos_eigh(&crate::hmat::ToeplitzOp::new(&c), neig, 1e-12, 0x5eed)?.1
    };
    let cols = op.ncols();
    let zs: Vec<(Vec<f64>, f64)> = (0..neig)
        .into_par_iter()
        .map(|j| {
            let uj: Vec<f64> = u.col(j).iter().copied().collect();
            let mut z = vec![0.0; cols];
            op.rmatvec(&uj, &mut z);
            let s = z.iter().map(|a| a * a).sum::<f64>().sqrt();
            (z, s)
        })
        .collect();
    let mut idx: Vec<usize> = (0..neig).collect();
    idx.sort_by(|&a, &b| zs[b].1.total_cmp(&zs[a].1));
    Ok((
        Mat::from_fn(l, neig, |i, k| u[(i, idx[k])]),
        idx.iter().map(|&j| zs[j].1).collect(),
        Mat::from_fn(cols, neig, |i, k| {
            let (z, s) = &zs[idx[k]];
            if *s > 0.0 { z[i] / s } else { 0.0 }
        }),
    ))
}

/// Joint roots of several shift matrices (Rssa `.est.exp.2desprit` /
/// `.est.exp.memp.new` with `beta`).
pub fn pair_roots(zs: &[Matrix], pairing: Pairing, beta: f64) -> Result<Vec<Vec<Complex64>>> {
    let r = zs[0].nrows();
    if zs.len() == 1 {
        return Ok(vec![super::parest::eigen_r_order(&zs[0])?]);
    }
    let nb = zs.len() - 1;
    let mut b: Vec<f64> = (1..=nb).map(|i| beta.powi(i as i32)).collect();
    b.insert(0, 1.0 - b.iter().sum::<f64>());
    let z = Mat::from_fn(r, r, |i, j| zs.iter().zip(&b).map(|(m, w)| w * m[(i, j)]).sum());
    let (_, t) = super::ossa::eigen_with_vectors(&z)?;
    let lu = t.partial_piv_lu();
    match pairing {
        Pairing::Diag => zs
            .iter()
            .map(|zk| {
                let zc = Mat::from_fn(r, r, |i, j| Complex64::new(zk[(i, j)], 0.0));
                let zt = cmm(&zc, &t);
                let x = faer::linalg::solvers::Solve::solve(&lu, zt.as_ref());
                Ok((0..r).map(|i| x[(i, i)]).collect())
            })
            .collect(),
        Pairing::Memp => zs
            .iter()
            .map(|zk| {
                let (vals, vecs) = super::ossa::eigen_with_vectors(zk)?;
                let x = faer::linalg::solvers::Solve::solve(&lu, vecs.as_ref()); // T^{-1} T_k
                // simple.assignment of t(|x|): rows = eigenvalues of Z_k? (R: mx = t(abs(...)))
                let mut mx = Mat::from_fn(r, r, |i, j| x[(j, i)].norm());
                let mut res = vec![0usize; r];
                for _ in 0..r {
                    let (mut bi, mut bj, mut best) = (0, 0, f64::NEG_INFINITY);
                    for j in 0..r {
                        for i in 0..r {
                            if mx[(i, j)] > best {
                                best = mx[(i, j)];
                                bi = i;
                                bj = j;
                            }
                        }
                    }
                    res[bi] = bj;
                    for c in 0..r {
                        mx[(bi, c)] = f64::NEG_INFINITY;
                        mx[(c, bj)] = f64::NEG_INFINITY;
                    }
                }
                Ok(res.iter().map(|&j| vals[j]).collect())
            })
            .collect(),
    }
}

fn cmm(a: &Mat<Complex64>, b: &Mat<Complex64>) -> Mat<Complex64> {
    Mat::from_fn(a.nrows(), b.ncols(), |i, j| (0..a.ncols()).map(|k| a[(i, k)] * b[(k, j)]).sum())
}

/// Oblique w-correlations helper shared with the MSSA-list engine.
pub fn owcor_matrix(mx: &Matrix) -> Matrix {
    cov2cor(&mm(mx.transpose(), mx.as_ref()))
}

impl NdSsa {
    fn is_shaped(&self) -> bool {
        self.wpos.len() != self.l.iter().product::<usize>()
            || self.fpos.len() != self.k.iter().product::<usize>()
            || self.circular.iter().any(|c| *c)
    }

    fn require_1d(&self) -> Result<()> {
        if self.n.len() != 1 || self.wpos.len() != self.l[0] {
            return Err(SsaError::InvalidArgument("forecasting needs a one-dimensional series with a full window".into()));
        }
        Ok(())
    }

    /// LRR of a group (Rssa `lrr`; projected bases containing row-special
    /// triples are orthonormalised as Rssa's `.colspan.pssa` does).
    pub fn lrr(&self, group: &[usize], reverse: bool) -> Result<Vec<f64>> {
        self.require_1d()?;
        self.check(group)?;
        let orth = group.iter().any(|&i| i < self.n_pr);
        super::forecast::lrr_from_basis(&select_cols(self.u.as_ref(), group), reverse, orth)
    }

    /// Recurrent forecast (Rssa `rforecast`; allowed for circular series).
    pub fn rforecast(&self, group: &[usize], len: usize, reconstructed_base: bool, reverse: bool) -> Result<Vec<f64>> {
        self.require_1d()?;
        if self.is_shaped() && !self.circular[0] {
            return Err(SsaError::InvalidArgument("recurrent forecasting is not implemented for shaped SSA".into()));
        }
        let f = if reconstructed_base { self.reconstruct(group)? } else { self.spec.field.clone() };
        let lrr = self.lrr(group, reverse)?;
        super::forecast::apply_lrr(&f, &lrr, len, reverse, &[])
    }

    /// Vector forecast (Rssa `vforecast`; not for shaped or circular SSA).
    pub fn vforecast(&self, group: &[usize], len: usize) -> Result<Vec<f64>> {
        self.require_1d()?;
        if self.is_shaped() {
            return Err(SsaError::InvalidArgument("vector forecasting is not implemented for shaped or circular SSA".into()));
        }
        self.check(group)?;
        let mut g: Vec<usize> = Vec::new();
        for &i in group {
            if !g.contains(&i) {
                g.push(i);
            }
        }
        let (l, k) = (self.l[0], self.k[0]);
        let r = g.len();
        let u = select_cols(self.u.as_ref(), &g);
        let v = select_cols(self.v.as_ref(), &g);
        let p = crate::ssa::lstsq(u.as_ref().subrows(0, l - 1), u.as_ref().subrows(1, l - 1))?;
        let rows = k + len + l - 1;
        let mut z = Mat::<f64>::zeros(rows, r);
        for t in 0..k {
            for j in 0..r {
                z[(t, j)] = self.sigma[g[j]] * v[(t, j)];
            }
        }
        for t in k..rows {
            for i in 0..r {
                z[(t, i)] = (0..r).map(|j| p[(i, j)] * z[(t - 1, j)]).sum();
            }
        }
        let y = super::forecast::hankelize_sum(&u, &z);
        Ok(y[self.n[0]..self.n[0] + len].to_vec())
    }

    /// Automatic grouping by w-correlation clustering (Rssa `grouping.auto.wcor`).
    pub fn grouping_auto_wcor(&self, components: &[usize], nclust: usize) -> Result<Vec<Group>> {
        let mut comps = components.to_vec();
        comps.sort_unstable();
        comps.dedup();
        let gs: Vec<Group> = comps.iter().map(|&i| vec![i]).collect();
        let w = self.wcor(&gs)?;
        let d = Mat::from_fn(w.nrows(), w.ncols(), |i, j| (1.0 - w[(i, j)]) / 2.0);
        Ok(super::common::split_by_label(&comps, &super::common::hclust_complete_cut(&d, nclust)))
    }
}
