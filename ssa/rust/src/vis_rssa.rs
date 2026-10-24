//! Rssa plot types rendered as SVG (`plot.ssa` with `type = "values"`,
//! `"vectors"`, `"paired"`, `"series"`, `"wcor"`, reconstruction plots,
//! roots / LRR plots, 2D eigenvector images, `plot.hmatr`,
//! `plot.grouping.auto.pgram` and `plot.grouping.auto.wcor` dendrograms).

use faer::Mat;

use crate::error::Result;
use crate::matrix::Matrix;
use crate::rssa::common::{hclust_complete_tree, Group};
use crate::ssa::{Complex64, SsaModel};
use crate::vis::Figure;

fn grid_for(n: usize, ncols: usize, w: f64, h: f64) -> Figure {
    let nc = ncols.min(n).max(1);
    Figure::new(n, nc, (w * nc as f64, h * n.div_ceil(nc) as f64))
}

fn group_name(g: &[usize]) -> String {
    format!("F({})", g.iter().map(|v| (v + 1).to_string()).collect::<Vec<_>>().join(","))
}

/// Component norms on a log scale (Rssa `type = "values"`).
pub fn plot_values(sigma: &[f64], n: Option<usize>) -> Figure {
    let n = n.unwrap_or(sigma.len()).min(sigma.len());
    let mut f = Figure::new(1, 1, (7.0, 4.5));
    let a = &mut f.axes[0];
    let x: Vec<f64> = (1..=n).map(|v| v as f64).collect();
    let y: Vec<f64> = sigma[..n].iter().map(|s| s.max(f64::MIN_POSITIVE).log10()).collect();
    a.line(x.clone(), y.clone());
    a.scatter(x, y);
    a.title = "Component norms".into();
    a.xlabel = "Index".into();
    a.ylabel = "log10(norm)".into();
    a.grid = true;
    f
}

/// Columns of a basis, eigenvectors or factor vectors (Rssa `type = "vectors"`).
/// `contrib` are fractions (Rssa `contributions`) shown in the titles.
pub fn plot_vectors(basis: &Matrix, idx: &[usize], contrib: Option<&[f64]>) -> Figure {
    let mut f = grid_for(idx.len(), 4, 3.0, 2.4);
    for (a, &i) in f.axes.iter_mut().zip(idx) {
        let y: Vec<f64> = basis.col(i).iter().copied().collect();
        a.line((1..=y.len()).map(|v| v as f64).collect(), y);
        a.title = match contrib {
            Some(c) => format!("{} ({:.2}%)", i + 1, 100.0 * c[i]),
            None => format!("{}", i + 1),
        };
        a.hide_ticks = true;
    }
    f
}

/// Scatter plots of vector pairs (Rssa `type = "paired"`).
pub fn plot_paired(basis: &Matrix, pairs: &[(usize, usize)]) -> Figure {
    let mut f = grid_for(pairs.len(), 4, 3.0, 3.0);
    for (a, &(i, j)) in f.axes.iter_mut().zip(pairs) {
        let x: Vec<f64> = basis.col(i).iter().copied().collect();
        let y: Vec<f64> = basis.col(j).iter().copied().collect();
        a.line(x, y);
        a.title = format!("{} vs {}", i + 1, j + 1);
        a.equal_aspect = true;
        a.hide_ticks = true;
    }
    f
}

/// Reconstructed series of groups (Rssa `type = "series"`).
pub fn plot_series(model: &SsaModel, groups: &[Group]) -> Result<Figure> {
    let mut f = grid_for(groups.len(), 1, 8.0, 2.2);
    for (a, g) in f.axes.iter_mut().zip(groups) {
        let r = model.reconstruct_components(g)?;
        for c in 0..r.ncols() {
            let y: Vec<f64> = r.col(c).iter().copied().collect();
            a.line((1..=y.len()).map(|v| v as f64).collect(), y);
        }
        a.title = group_name(g);
        a.grid = true;
    }
    Ok(f)
}

/// Original series, reconstructed groups and residuals
/// (Rssa `plot(reconstruct(...))`).
pub fn plot_reconstruction(model: &SsaModel, groups: &[Group]) -> Result<Figure> {
    let n = groups.len() + 2;
    let mut f = grid_for(n, 1, 8.0, 2.0);
    let x: Vec<f64> = (1..=model.n).map(|v| v as f64).collect();
    let all: Vec<usize> = groups.iter().flatten().copied().collect();
    for c in 0..model.s {
        f.axes[0].line(x.clone(), model.x_s.col(c).iter().copied().collect());
    }
    f.axes[0].title = "Original".into();
    for (a, g) in f.axes[1..=groups.len()].iter_mut().zip(groups) {
        let r = model.reconstruct_components(g)?;
        for c in 0..r.ncols() {
            a.line(x.clone(), r.col(c).iter().copied().collect());
        }
        a.title = group_name(g);
    }
    let res = model.residuals(&all)?;
    for c in 0..res.ncols() {
        f.axes[n - 1].line(x.clone(), res.col(c).iter().copied().collect());
    }
    f.axes[n - 1].title = "Residuals".into();
    for a in &mut f.axes {
        a.grid = true;
    }
    Ok(f)
}

/// W-correlation matrix, shaded by `|w|` (Rssa `plot.wcor.matrix`).
pub fn plot_wcor(w: &Matrix) -> Figure {
    let mut f = Figure::new(1, 1, (6.5, 6.0));
    let a = &mut f.axes[0];
    a.heatmap(Mat::from_fn(w.nrows(), w.ncols(), |i, j| w[(i, j)].abs()), Some(0.0), Some(1.0));
    a.title = "W-correlation matrix".into();
    a.colorbar = Some("|w|".into());
    f
}

/// Roots in the complex plane with the unit circle (Rssa
/// `plot.fdimpars.1d`, `plot(lrr)`).
pub fn plot_roots(roots: &[Complex64], title: &str) -> Figure {
    let mut f = Figure::new(1, 1, (6.0, 6.0));
    let a = &mut f.axes[0];
    let th: Vec<f64> = (0..400).map(|i| 2.0 * std::f64::consts::PI * i as f64 / 399.0).collect();
    a.dashed_line(th.iter().map(|t| t.cos()).collect(), th.iter().map(|t| t.sin()).collect());
    a.scatter(roots.iter().map(|z| z.re).collect(), roots.iter().map(|z| z.im).collect());
    a.axhline(0.0).axvline(0.0);
    let lim = roots.iter().map(|z| 1.1 * z.norm()).fold(1.1, f64::max);
    a.xlim = Some((-lim, lim));
    a.ylim = Some((-lim, lim));
    a.equal_aspect = true;
    a.title = title.into();
    a.xlabel = "Real part".into();
    a.ylabel = "Imaginary part".into();
    f
}

/// 2D eigenvectors as images (Rssa 2D-SSA `type = "vectors"`): column `i`
/// of `basis` reshaped to `dims` (first index fastest); `positions` are the
/// linear indices of the window cells for shaped windows.
pub fn plot_vectors_2d(basis: &Matrix, idx: &[usize], dims: (usize, usize), positions: Option<&[usize]>) -> Figure {
    let mut f = grid_for(idx.len(), 4, 2.8, 2.6);
    for (a, &i) in f.axes.iter_mut().zip(idx) {
        let mut img = Mat::<f64>::from_fn(dims.0, dims.1, |_, _| f64::NAN);
        for r in 0..basis.nrows() {
            let p = positions.map_or(r, |ps| ps[r]);
            img[(p % dims.0, p / dims.0)] = basis[(r, i)];
        }
        a.heatmap(img, None, None);
        a.title = format!("{}", i + 1);
        a.hide_ticks = true;
    }
    f
}

/// Images of 2D fields (Rssa `plot.2d.ssa.reconstruction`).
pub fn plot_fields(fields: &[(String, Matrix)]) -> Figure {
    let mut f = grid_for(fields.len(), 3, 3.5, 3.2);
    for (a, (t, m)) in f.axes.iter_mut().zip(fields) {
        a.heatmap(m.clone(), None, None);
        a.title = t.clone();
        a.hide_ticks = true;
        a.colorbar = Some(String::new());
    }
    f
}

/// Heterogeneity matrix (Rssa `plot.hmatr`).
pub fn plot_hmatr(h: &Matrix) -> Figure {
    let mut f = Figure::new(1, 1, (6.5, 6.0));
    let a = &mut f.axes[0];
    a.heatmap(h.clone(), Some(0.0), Some(1.0));
    a.title = "Heterogeneity Matrix".into();
    a.colorbar = Some(String::new());
    f
}

/// Relative contributions per frequency band (Rssa `plot.grouping.auto.pgram`).
pub fn plot_pgram_contributions(components: &[usize], contributions: &Matrix) -> Figure {
    let mut f = Figure::new(1, 1, (8.0, 4.0));
    let a = &mut f.axes[0];
    let x: Vec<f64> = components.iter().map(|&c| (c + 1) as f64).collect();
    for b in 0..contributions.ncols() {
        let y: Vec<f64> = (0..contributions.nrows()).map(|i| contributions[(i, b)]).collect();
        a.line(x.clone(), y.clone());
        a.scatter(x.clone(), y);
    }
    a.title = "Relative contribution per band".into();
    a.xlabel = "Component".into();
    a.ylabel = "Relative contribution".into();
    a.grid = true;
    f
}

/// Dendrogram of the w-correlation clustering (Rssa `plot.grouping.auto.wcor`).
pub fn plot_wcor_dendrogram(components: &[usize], w: &Matrix) -> Figure {
    let n = components.len();
    let d = Mat::from_fn(n, n, |i, j| (1.0 - w[(i, j)]) / 2.0);
    let merges = hclust_complete_tree(&d);
    let order: Vec<usize> = merges.last().map(|(a, b, _)| a.iter().chain(b).copied().collect()).unwrap_or_else(|| (0..n).collect());
    let mut xpos = vec![0.0; n];
    for (rank, &leaf) in order.iter().enumerate() {
        xpos[leaf] = rank as f64;
    }
    let mut f = Figure::new(1, 1, (8.0, 4.5));
    let a = &mut f.axes[0];
    let mut height = vec![0.0; n];
    for (ma, mb, hgt) in &merges {
        let centre = |m: &[usize]| m.iter().map(|&i| xpos[i]).sum::<f64>() / m.len() as f64;
        let top = |m: &[usize]| m.iter().map(|&i| height[i]).fold(0.0, f64::max);
        let (xa, xb, ha, hb) = (centre(ma), centre(mb), top(ma), top(mb));
        a.line(vec![xa, xa, xb, xb], vec![ha, *hgt, *hgt, hb]);
        for &m in ma.iter().chain(mb) {
            height[m] = *hgt;
        }
    }
    a.title = format!("Clustering of components {}", order.iter().map(|&i| (components[i] + 1).to_string()).collect::<Vec<_>>().join(" "));
    a.ylabel = "(1 - w) / 2".into();
    f
}

/// Share of complete lag vectors against the window length (Rssa `clplot`).
pub fn plot_clplot(x: &[f64]) -> Figure {
    let pts = crate::rssa::gaps::complete_lag_fraction(x);
    let mut f = Figure::new(1, 1, (7.0, 4.5));
    let a = &mut f.axes[0];
    a.line(pts.iter().map(|p| p.0 as f64).collect(), pts.iter().map(|p| p.1).collect());
    a.title = "Complete lag vectors".into();
    a.xlabel = "Window length, L".into();
    a.ylabel = "Percentage of complete lag vectors".into();
    a.grid = true;
    f
}

/// Series with its gaps marked (Rssa `plot.ssa.gaps`): observed values as
/// a line, gap bounds as vertical lines.
pub fn plot_gaps(x: &[f64], l: usize) -> Figure {
    let mut f = Figure::new(1, 1, (8.0, 4.0));
    let a = &mut f.axes[0];
    let (t, y): (Vec<f64>, Vec<f64>) = x.iter().enumerate().filter(|(_, v)| !v.is_nan()).map(|(i, v)| ((i + 1) as f64, *v)).unzip();
    a.scatter(t, y);
    for g in crate::rssa::gaps::summarize_gaps(x, l) {
        a.axvline((g.left + 1) as f64);
        a.axvline((g.right + 1) as f64);
    }
    a.title = "Gaps".into();
    a.xlabel = "Time".into();
    a.grid = true;
    f
}
