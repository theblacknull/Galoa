//! Size-aware wrappers over faer's low-level kernels.
//!
//! faer's convenience methods always use the global (rayon) parallelism; for
//! small problems (ESPRIT, LRR, forecasting operators ...) waking the thread
//! pool costs far more than the computation itself.  These helpers run small
//! problems sequentially and large ones in parallel.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::{self, ComputeEigenvectors};
use faer::linalg::matmul::matmul;
use faer::linalg::svd::{self, ComputeSvdVectors};
use faer::diag::Diag;
use faer::{Accum, Mat, MatRef, Par};

use crate::error::{Result, SsaError};

/// Below this many floating point operations a kernel runs sequentially.
const PAR_THRESHOLD: f64 = 4.0e6;

/// Parallelism for a kernel performing roughly `flops` operations.
pub fn par_for(flops: f64) -> Par {
    if flops < PAR_THRESHOLD { Par::Seq } else { faer::get_global_parallelism() }
}

/// Parallelism of the SVD for `m * n * min(m, n)` = `work`.
///
/// Measured on faer 0.24 (Ryzen 7 5800H, 8 cores / 16 threads): its SVD is
/// slower with all SMT threads than sequentially up to ~1e8 work (e.g.
/// 100x2010: 6.6 ms sequential vs 142 ms on 16 threads); for larger problems
/// physical cores only (half the logical threads) are fastest.
pub fn svd_par(work: f64) -> Par {
    if work < 2.0e8 {
        Par::Seq
    } else {
        Par::rayon(physical_threads())
    }
}

/// Parallelism of the symmetric eigen-decomposition of an `n x n` matrix.
pub fn eigh_par(n: usize) -> Par {
    if n < 400 { Par::Seq } else { faer::get_global_parallelism() }
}

/// Half the logical threads (physical cores on SMT machines), at least 1;
/// respects `RAYON_NUM_THREADS` / the configured rayon pool.
pub fn physical_threads() -> usize {
    (rayon::current_num_threads() / 2).max(1)
}

/// Eigen-decomposition of a symmetric matrix (lower triangle used):
/// eigenvalues ascending, eigenvectors in the columns of `U`.
pub fn eigh(a: MatRef<'_, f64>) -> Result<(Vec<f64>, Mat<f64>)> {
    let n = a.nrows();
    let par = eigh_par(n);
    let mut s = Diag::<f64>::zeros(n);
    let mut u = Mat::<f64>::zeros(n, n);
    let req = evd::self_adjoint_evd_scratch::<f64>(n, ComputeEigenvectors::Yes, par, Default::default());
    let mut buf = MemBuffer::new(req);
    evd::self_adjoint_evd(a, s.as_mut(), Some(u.as_mut()), par, MemStack::new(&mut buf), Default::default())
        .map_err(|e| SsaError::LinAlg(format!("eigen decomposition failed: {e:?}")))?;
    Ok((s.column_vector().iter().copied().collect(), u))
}

/// `a @ b`.
pub fn mm(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Mat<f64> {
    let mut out = Mat::zeros(a.nrows(), b.ncols());
    let par = par_for(2.0 * a.nrows() as f64 * a.ncols() as f64 * b.ncols() as f64);
    matmul(out.as_mut(), Accum::Replace, a, b, 1.0, par);
    out
}

/// Thin SVD: `(U (m, k), s (k, descending), V (n, k))`, `k = min(m, n)`.
pub fn thin_svd(a: MatRef<'_, f64>) -> Result<(Mat<f64>, Vec<f64>, Mat<f64>)> {
    let (m, n) = (a.nrows(), a.ncols());
    let k = m.min(n);
    if k == 0 {
        return Ok((Mat::zeros(m, 0), Vec::new(), Mat::zeros(n, 0)));
    }
    let par = svd_par(m as f64 * n as f64 * k as f64);
    let mut s = Diag::<f64>::zeros(k);
    let mut u = Mat::<f64>::zeros(m, k);
    let mut v = Mat::<f64>::zeros(n, k);
    let req = svd::svd_scratch::<f64>(m, n, ComputeSvdVectors::Thin, ComputeSvdVectors::Thin, par, Default::default());
    let mut buf = MemBuffer::new(req);
    svd::svd(a, s.as_mut(), Some(u.as_mut()), Some(v.as_mut()), par, MemStack::new(&mut buf), Default::default())
        .map_err(|e| SsaError::LinAlg(format!("SVD did not converge: {e:?}")))?;
    Ok((u, s.column_vector().iter().copied().collect(), v))
}

/// Eigenvalues `(re, im)` of a real square matrix.
pub fn eigenvalues(a: MatRef<'_, f64>) -> Result<Vec<(f64, f64)>> {
    let n = a.nrows();
    if n == 0 {
        return Ok(Vec::new());
    }
    let par = par_for(10.0 * (n as f64).powi(3));
    let mut re = Diag::<f64>::zeros(n);
    let mut im = Diag::<f64>::zeros(n);
    let req = evd::evd_scratch::<f64>(n, ComputeEigenvectors::No, ComputeEigenvectors::No, par, Default::default());
    let mut buf = MemBuffer::new(req);
    evd::evd_real(a, re.as_mut(), im.as_mut(), None, None, par, MemStack::new(&mut buf), Default::default())
        .map_err(|e| SsaError::LinAlg(format!("eigenvalue computation failed: {e:?}")))?;
    Ok(re.column_vector().iter().zip(im.column_vector().iter()).map(|(r, i)| (*r, *i)).collect())
}
