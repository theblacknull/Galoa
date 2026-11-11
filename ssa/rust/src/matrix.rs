//! Thin helpers around `faer` dense matrices (column-major `f64`).

use faer::{Mat, MatRef};

/// Dense column-major `f64` matrix used throughout the crate.
pub type Matrix = Mat<f64>;

/// `a @ b`.
pub fn mm(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Matrix {
    crate::la::mm(a, b)
}

/// Builds a matrix from row-major nested data (handy in tests and examples).
pub fn from_rows(rows: &[Vec<f64>]) -> Matrix {
    let nrows = rows.len();
    let ncols = rows.first().map_or(0, Vec::len);
    assert!(rows.iter().all(|r| r.len() == ncols), "ragged rows");
    Mat::from_fn(nrows, ncols, |i, j| rows[i][j])
}

/// Builds an `(n, 1)` column matrix from a slice.
pub fn column(x: &[f64]) -> Matrix {
    Mat::from_fn(x.len(), 1, |i, _| x[i])
}

/// Returns the matrix as row-major nested vectors.
pub fn to_rows(m: MatRef<'_, f64>) -> Vec<Vec<f64>> {
    (0..m.nrows()).map(|i| (0..m.ncols()).map(|j| m[(i, j)]).collect()).collect()
}

/// Copies column `j` into a new vector.
pub fn col_vec(m: MatRef<'_, f64>, j: usize) -> Vec<f64> {
    (0..m.nrows()).map(|i| m[(i, j)]).collect()
}

/// Selects the given columns (duplicates allowed, numpy fancy-index semantics).
pub fn select_cols(m: MatRef<'_, f64>, idx: &[usize]) -> Matrix {
    Mat::from_fn(m.nrows(), idx.len(), |i, k| m[(i, idx[k])])
}
