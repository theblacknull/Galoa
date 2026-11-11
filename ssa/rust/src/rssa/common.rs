//! Helpers shared by the Rssa ports: signed weighted correlations,
//! hierarchical clustering, pseudo-inverses, R-style quantiles.

use faer::Mat;

use crate::error::{Result, SsaError};
use crate::matrix::Matrix;
use crate::ssa::SsaModel;
use crate::svd::{SvdStrategy, Trajectory};

/// A group of (0-based) component indices.
pub type Group = Vec<usize>;

/// Re-fits SSA/MSSA on new data with the same window and channel weights.
pub fn refit(strategy: &dyn SvdStrategy, x_s: Matrix, l: usize, weights: Vec<f64>) -> Result<SsaModel> {
    let t = Trajectory { x_s: x_s.as_ref(), l, weights: &weights };
    let dec = strategy.decompose(&t)?;
    let e = t.frob_norm2();
    Ok(SsaModel::from_parts(x_s, l, weights, dec, e))
}

/// Default decomposition used inside Rssa iterations: exact for `L < 500`,
/// otherwise the leading `neig` triples by Lanczos (Rssa: `eigen` / `nutrlan`).
pub fn auto_strategy(l: usize, neig: usize) -> Box<dyn SvdStrategy> {
    if l < 500 {
        Box::new(crate::svd::FullSvd)
    } else {
        Box::new(crate::svd::LanczosSvd::new(neig.max(1)))
    }
}

/// Hankel weights of each element of `x_s`, column-major `(N*S)`.
pub fn series_weights(m: &SsaModel) -> Vec<f64> {
    let hw = m.construct_hankel_weights();
    (0..m.s).flat_map(|_| hw.iter().map(|&v| v as f64)).collect()
}

/// Rssa `wcor.default`: signed weighted correlations of the columns of `fs`
/// (`(n, m)`) with weights `w` (length `n`), clamped to `[-1, 1]`.
pub fn wcor_matrix(fs: &Matrix, w: &[f64]) -> Matrix {
    let sw: Vec<f64> = w.iter().map(|v| v.sqrt()).collect();
    let g = Mat::from_fn(fs.nrows(), fs.ncols(), |t, j| sw[t] * fs[(t, j)]);
    cov2cor(&crate::matrix::mm(g.transpose(), g.as_ref()))
}

/// R `cov2cor` with unit diagonal and clamping to `[-1, 1]`.
pub fn cov2cor(cov: &Matrix) -> Matrix {
    let m = cov.nrows();
    let is: Vec<f64> = (0..m).map(|i| (1.0 / cov[(i, i)].abs()).sqrt()).collect();
    Mat::from_fn(m, m, |i, j| if i == j { 1.0 } else { (is[i] * cov[(i, j)] * is[j]).clamp(-1.0, 1.0) })
}

/// Stacks the channels of an `(N, S)` matrix into one column.
pub fn flatten(m: &Matrix) -> Vec<f64> {
    m.col_iter().flat_map(|c| c.iter().copied().collect::<Vec<_>>()).collect()
}

/// Complete-linkage agglomerative clustering cut into `k` clusters
/// (R `cutree(hclust(d, "complete"), k)`).  `dist` is a full symmetric
/// matrix.  Clusters are numbered by first appearance of their members, as
/// `cutree` does; the result holds the cluster number (0-based) of each item.
pub fn hclust_complete_cut(dist: &Matrix, k: usize) -> Vec<usize> {
    let n = dist.nrows();
    let k = k.clamp(1, n.max(1));
    let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut d = dist.clone();
    let mut alive: Vec<bool> = vec![true; n];
    let mut count = n;
    while count > k {
        let (mut bi, mut bj, mut best) = (0, 0, f64::INFINITY);
        for i in 0..n {
            if !alive[i] {
                continue;
            }
            for j in i + 1..n {
                if alive[j] && d[(i, j)] < best {
                    best = d[(i, j)];
                    bi = i;
                    bj = j;
                }
            }
        }
        // merge bj into bi, complete linkage: max distance
        let moved = std::mem::take(&mut clusters[bj]);
        clusters[bi].extend(moved);
        alive[bj] = false;
        for t in 0..n {
            if alive[t] && t != bi {
                let v = d[(bi, t)].max(d[(bj, t)]);
                d[(bi, t)] = v;
                d[(t, bi)] = v;
            }
        }
        count -= 1;
    }
    let mut label = vec![usize::MAX; n];
    let mut next = 0;
    for item in 0..n {
        if label[item] == usize::MAX {
            let c = clusters.iter().position(|c| c.contains(&item)).unwrap();
            for &m in &clusters[c] {
                label[m] = next;
            }
            next += 1;
        }
    }
    label
}

/// Splits `items` by cluster labels, ordered by label (R `split`).
pub fn split_by_label(items: &[usize], labels: &[usize]) -> Vec<Group> {
    let nl = labels.iter().copied().max().map_or(0, |m| m + 1);
    let mut out = vec![Vec::new(); nl];
    for (it, &lb) in items.iter().zip(labels) {
        out[lb].push(*it);
    }
    out
}

/// Moore–Penrose pseudo-inverse of a full-column-rank matrix via QR
/// (Rssa `pseudo.inverse`: `solve(R, t(Q))`).
pub fn pseudo_inverse(a: &Matrix) -> Result<Matrix> {
    let (u, s, v) = crate::la::thin_svd(a.as_ref())?;
    let smax = s.iter().copied().fold(0.0, f64::max);
    if s.iter().any(|&x| x <= f64::EPSILON * smax * a.nrows().max(a.ncols()) as f64) {
        return Err(SsaError::LinAlg("matrix is rank deficient".into()));
    }
    Ok(Mat::from_fn(a.ncols(), a.nrows(), |i, j| (0..s.len()).map(|k| v[(i, k)] * u[(j, k)] / s[k]).sum()))
}

/// Least-squares solution of `A X = B` (R `qr.solve` for full column rank).
pub fn ls_solve(a: &Matrix, b: &Matrix) -> Result<Matrix> {
    crate::ssa::lstsq(a.as_ref(), b.as_ref())
}

/// R `quantile(x, p)` (type 7, identical to numpy's linear rule).
pub fn quantile7(x: &[f64], p: f64) -> f64 {
    crate::stats::percentile(x, p * 100.0)
}

/// R `median`.
pub fn median(x: &[f64]) -> f64 {
    quantile7(x, 0.5)
}

/// R `mad` (constant 1.4826).
pub fn mad(x: &[f64]) -> f64 {
    let m = median(x);
    let dev: Vec<f64> = x.iter().map(|v| (v - m).abs()).collect();
    1.4826 * median(&dev)
}

/// Cumulative ranges `[0..r1), [r1..r1+r2) ...` for group ranks.
pub fn consecutive_groups(ranks: &[usize]) -> Vec<Group> {
    let mut start = 0;
    ranks
        .iter()
        .map(|&r| {
            let g = (start..start + r).collect();
            start += r;
            g
        })
        .collect()
}

/// Full complete-linkage merge tree: `(members_a, members_b, height)` for
/// each of the `n - 1` merges, in merge order (for dendrograms).
pub fn hclust_complete_tree(dist: &Matrix) -> Vec<(Vec<usize>, Vec<usize>, f64)> {
    let n = dist.nrows();
    let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut d = dist.clone();
    let mut alive = vec![true; n];
    let mut merges = Vec::new();
    for _ in 1..n {
        let (mut bi, mut bj, mut best) = (0, 0, f64::INFINITY);
        for i in 0..n {
            if alive[i] {
                for j in i + 1..n {
                    if alive[j] && d[(i, j)] < best {
                        best = d[(i, j)];
                        bi = i;
                        bj = j;
                    }
                }
            }
        }
        merges.push((clusters[bi].clone(), clusters[bj].clone(), best));
        let moved = std::mem::take(&mut clusters[bj]);
        clusters[bi].extend(moved);
        alive[bj] = false;
        for t in 0..n {
            if alive[t] && t != bi {
                let v = d[(bi, t)].max(d[(bj, t)]);
                d[(bi, t)] = v;
                d[(t, bi)] = v;
            }
        }
    }
    merges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hclust_cut() {
        // points on a line: {0, 0.1}, {5, 5.2}, {9}
        let p: [f64; 5] = [0.0, 5.0, 0.1, 9.0, 5.2];
        let d = Mat::from_fn(5, 5, |i, j| (p[i] - p[j]).abs());
        assert_eq!(hclust_complete_cut(&d, 3), vec![0, 1, 0, 2, 1]);
        assert_eq!(hclust_complete_cut(&d, 1), vec![0; 5]);
        assert_eq!(split_by_label(&[10, 11, 12, 13, 14], &[0, 1, 0, 2, 1]), vec![vec![10, 12], vec![11, 14], vec![13]]);
    }

    #[test]
    fn r_stats() {
        let x = [1.0, 2.0, 3.0, 4.0, 100.0];
        assert_eq!(median(&x), 3.0);
        assert!((mad(&x) - 1.4826).abs() < 1e-12);
        assert!((quantile7(&x, 0.25) - 2.0).abs() < 1e-12);
    }
}
