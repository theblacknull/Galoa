//! Visualisation (port of `vis_tools.py`).
//!
//! The Python module draws with matplotlib; this port renders the same five
//! figures as standalone SVG documents without external dependencies.  Each
//! `plot_*` function returns a [`Figure`]; call [`Figure::to_svg`] or
//! [`Figure::save`].

use std::fmt::Write as _;
use std::path::Path;

use crate::error::{Result, SsaError};
use crate::matrix::Matrix;
use crate::ssa::SsaModel;

const PALETTE: [&str; 6] = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"];
/// Largest heat-map resolution; bigger matrices are block-averaged.
const MAX_CELLS: usize = 160;

#[derive(Debug, Clone)]
enum Item {
    Line { x: Vec<f64>, y: Vec<f64>, color: String, dashed: bool, width: f64 },
    Scatter { x: Vec<f64>, y: Vec<f64>, color: String },
    Heatmap { data: Matrix, vmin: f64, vmax: f64 },
    HLine(f64),
    VLine(f64),
}

/// One subplot.
#[derive(Debug, Clone, Default)]
pub struct Axes {
    pub title: String,
    pub xlabel: String,
    pub ylabel: String,
    items: Vec<Item>,
    pub xlim: Option<(f64, f64)>,
    pub ylim: Option<(f64, f64)>,
    pub equal_aspect: bool,
    pub grid: bool,
    pub hide_ticks: bool,
    pub colorbar: Option<String>,
}

impl Axes {
    pub fn line(&mut self, x: Vec<f64>, y: Vec<f64>) -> &mut Self {
        let color = PALETTE[self.items.len() % PALETTE.len()].to_string();
        self.items.push(Item::Line { x, y, color, dashed: false, width: 1.5 });
        self
    }
    pub fn dashed_line(&mut self, x: Vec<f64>, y: Vec<f64>) -> &mut Self {
        self.items.push(Item::Line { x, y, color: PALETTE[0].into(), dashed: true, width: 1.0 });
        self
    }
    pub fn scatter(&mut self, x: Vec<f64>, y: Vec<f64>) -> &mut Self {
        self.items.push(Item::Scatter { x, y, color: PALETTE[1].into() });
        self
    }
    pub fn heatmap(&mut self, data: Matrix, vmin: Option<f64>, vmax: Option<f64>) -> &mut Self {
        let (lo, hi) = finite_range(data.col_iter().flat_map(|c| c.iter().copied().collect::<Vec<_>>()));
        self.items.push(Item::Heatmap { data, vmin: vmin.unwrap_or(lo), vmax: vmax.unwrap_or(hi) });
        self
    }
    pub fn axhline(&mut self, y: f64) -> &mut Self {
        self.items.push(Item::HLine(y));
        self
    }
    pub fn axvline(&mut self, x: f64) -> &mut Self {
        self.items.push(Item::VLine(x));
        self
    }
}

/// A grid of subplots rendered to SVG.
#[derive(Debug, Clone)]
pub struct Figure {
    pub width: f64,
    pub height: f64,
    pub ncols: usize,
    pub axes: Vec<Axes>,
}

impl Figure {
    /// `figsize` in inches like matplotlib (100 px per inch).
    pub fn new(nplots: usize, ncols: usize, figsize: (f64, f64)) -> Self {
        let ncols = ncols.min(nplots).max(1);
        Self { width: figsize.0 * 100.0, height: figsize.1 * 100.0, ncols, axes: vec![Axes::default(); nplots] }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        std::fs::write(path.as_ref(), self.to_svg())
            .map_err(|e| SsaError::Io(format!("{}: {e}", path.as_ref().display())))
    }

    pub fn to_svg(&self) -> String {
        let n = self.axes.len().max(1);
        let nrows = n.div_ceil(self.ncols);
        let cw = self.width / self.ncols as f64;
        let ch = self.height / nrows as f64;
        let mut s = String::new();
        let _ = write!(
            s,
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="DejaVu Sans, Arial, sans-serif" font-size="11"><rect width="100%" height="100%" fill="white"/>"#,
            w = self.width,
            h = self.height
        );
        for (i, ax) in self.axes.iter().enumerate() {
            let (r, c) = (i / self.ncols, i % self.ncols);
            render_axes(&mut s, ax, c as f64 * cw, r as f64 * ch, cw, ch, i);
        }
        s.push_str("</svg>\n");
        s
    }
}

fn finite_range(it: impl IntoIterator<Item = f64>) -> (f64, f64) {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for v in it {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if !lo.is_finite() {
        (0.0, 1.0)
    } else if lo == hi {
        (lo - 0.5, hi + 0.5)
    } else {
        (lo, hi)
    }
}

fn nice_ticks(lo: f64, hi: f64) -> Vec<f64> {
    let span = hi - lo;
    if !(span > 0.0) {
        return vec![lo];
    }
    let raw = span / 5.0;
    let mag = 10f64.powf(raw.log10().floor());
    let step = [1.0, 2.0, 2.5, 5.0, 10.0].iter().map(|m| m * mag).find(|s| span / s <= 6.0).unwrap_or(10.0 * mag);
    let mut t = (lo / step).ceil() * step;
    let mut out = Vec::new();
    while t <= hi + step * 1e-9 {
        out.push(if t.abs() < step * 1e-9 { 0.0 } else { t });
        t += step;
    }
    out
}

fn fmt_tick(v: f64) -> String {
    let a = v.abs();
    if a != 0.0 && !(1e-3..1e5).contains(&a) {
        format!("{v:.1e}")
    } else {
        let s = format!("{v:.4}");
        let s = s.trim_end_matches('0').trim_end_matches('.');
        if s == "-0" { "0".into() } else { s.into() }
    }
}

fn esc(t: &str) -> String {
    t.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// Viridis colour map (matplotlib's anchor colours, linear interpolation).
pub fn viridis(t: f64) -> (u8, u8, u8) {
    const A: [(f64, f64, f64); 9] = [
        (68.0, 1.0, 84.0),
        (71.0, 44.0, 122.0),
        (59.0, 81.0, 139.0),
        (44.0, 113.0, 142.0),
        (33.0, 144.0, 141.0),
        (39.0, 173.0, 129.0),
        (92.0, 200.0, 99.0),
        (170.0, 220.0, 50.0),
        (253.0, 231.0, 37.0),
    ];
    let t = if t.is_finite() { t.clamp(0.0, 1.0) } else { 0.0 };
    let x = t * 8.0;
    let i = (x.floor() as usize).min(7);
    let f = x - i as f64;
    let lerp = |a: f64, b: f64| (a + (b - a) * f).round() as u8;
    (lerp(A[i].0, A[i + 1].0), lerp(A[i].1, A[i + 1].1), lerp(A[i].2, A[i + 1].2))
}

#[allow(clippy::too_many_arguments)]
fn render_axes(s: &mut String, ax: &Axes, x0: f64, y0: f64, w: f64, h: f64, id: usize) {
    let cb_space = if ax.colorbar.is_some() { 70.0 } else { 0.0 };
    let (ml, mr, mt, mb) = (58.0, 14.0 + cb_space, 28.0, if ax.xlabel.is_empty() { 26.0 } else { 42.0 });
    let (mut pw, mut ph) = ((w - ml - mr).max(10.0), (h - mt - mb).max(10.0));
    let (mut px, mut py) = (x0 + ml, y0 + mt);

    // data ranges
    let heat = ax.items.iter().find_map(|it| match it {
        Item::Heatmap { data, vmin, vmax } => Some((data, *vmin, *vmax)),
        _ => None,
    });
    let (mut xr, mut yr) = if let Some((d, _, _)) = heat {
        ((-0.5, d.ncols() as f64 - 0.5), (-0.5, d.nrows() as f64 - 0.5))
    } else {
        let xs = ax.items.iter().flat_map(|it| match it {
            Item::Line { x, .. } | Item::Scatter { x, .. } => x.clone(),
            _ => vec![],
        });
        let ys = ax.items.iter().flat_map(|it| match it {
            Item::Line { y, .. } | Item::Scatter { y, .. } => y.clone(),
            _ => vec![],
        });
        let (a, b) = finite_range(xs);
        let (c, d) = finite_range(ys);
        let (px_, py_) = (0.05 * (b - a), 0.05 * (d - c));
        ((a - px_, b + px_), (c - py_, d + py_))
    };
    if let Some(l) = ax.xlim {
        xr = l;
    }
    if let Some(l) = ax.ylim {
        yr = l;
    }
    if ax.equal_aspect {
        let sx = pw / (xr.1 - xr.0);
        let sy = ph / (yr.1 - yr.0);
        if sx > sy {
            let nw = (xr.1 - xr.0) * sy;
            px += (pw - nw) / 2.0;
            pw = nw;
        } else {
            let nh = (yr.1 - yr.0) * sx;
            py += (ph - nh) / 2.0;
            ph = nh;
        }
    }
    let tx = |v: f64| px + (v - xr.0) / (xr.1 - xr.0) * pw;
    // heat maps: row 0 on top (imshow); plots: y up.
    let flip = heat.is_some();
    let ty = |v: f64| if flip { py + (v - yr.0) / (yr.1 - yr.0) * ph } else { py + ph - (v - yr.0) / (yr.1 - yr.0) * ph };

    let _ = write!(s, r#"<defs><clipPath id="c{id}"><rect x="{px:.2}" y="{py:.2}" width="{pw:.2}" height="{ph:.2}"/></clipPath></defs>"#);

    if let Some((d, vmin, vmax)) = heat {
        let (nr, nc) = (d.nrows(), d.ncols());
        let (br, bc) = (nr.div_ceil(MAX_CELLS).max(1), nc.div_ceil(MAX_CELLS).max(1));
        let (gr, gc) = (nr.div_ceil(br), nc.div_ceil(bc));
        let _ = write!(s, r#"<g shape-rendering="crispEdges">"#);
        for i in 0..gr {
            for j in 0..gc {
                let (mut acc, mut cnt) = (0.0, 0);
                for a in i * br..((i + 1) * br).min(nr) {
                    for b in j * bc..((j + 1) * bc).min(nc) {
                        acc += d[(a, b)];
                        cnt += 1;
                    }
                }
                let v = acc / cnt as f64;
                let (r, g, b) = viridis((v - vmin) / (vmax - vmin));
                let (xa, xb) = (tx((j * bc) as f64 - 0.5), tx(((j + 1) * bc).min(nc) as f64 - 0.5));
                let (ya, yb) = (ty((i * br) as f64 - 0.5), ty(((i + 1) * br).min(nr) as f64 - 0.5));
                let _ = write!(
                    s,
                    r#"<rect x="{xa:.2}" y="{ya:.2}" width="{:.2}" height="{:.2}" fill="rgb({r},{g},{b})"/>"#,
                    (xb - xa) + 0.3,
                    (yb - ya) + 0.3
                );
            }
        }
        s.push_str("</g>");
        if let Some(label) = &ax.colorbar {
            let (cx, cw) = (px + pw + 14.0, 14.0);
            let steps = 64;
            for k in 0..steps {
                let (r, g, b) = viridis(1.0 - k as f64 / (steps - 1) as f64);
                let _ = write!(
                    s,
                    r#"<rect x="{cx:.2}" y="{:.2}" width="{cw}" height="{:.2}" fill="rgb({r},{g},{b})"/>"#,
                    py + ph * k as f64 / steps as f64,
                    ph / steps as f64 + 0.5
                );
            }
            let _ = write!(s, r##"<rect x="{cx:.2}" y="{py:.2}" width="{cw}" height="{ph:.2}" fill="none" stroke="#333"/>"##);
            for t in nice_ticks(vmin, vmax) {
                let yy = py + ph - (t - vmin) / (vmax - vmin) * ph;
                let _ = write!(s, r#"<text x="{:.2}" y="{:.2}" font-size="9">{}</text>"#, cx + cw + 3.0, yy + 3.0, fmt_tick(t));
            }
            if !label.is_empty() {
                let (lx, ly) = (cx + cw + 48.0, py + ph / 2.0);
                let _ = write!(s, r#"<text x="{lx:.2}" y="{ly:.2}" text-anchor="middle" transform="rotate(-90 {lx:.2} {ly:.2})">{}</text>"#, esc(label));
            }
        }
    }

    // grid and ticks
    if !ax.hide_ticks {
        let (xt, yt) = if flip {
            let step = |n: usize| ((n as f64 / 6.0).ceil() as usize).max(1);
            let d = heat.unwrap().0;
            (
                (0..d.ncols()).step_by(step(d.ncols())).map(|v| v as f64).collect::<Vec<_>>(),
                (0..d.nrows()).step_by(step(d.nrows())).map(|v| v as f64).collect::<Vec<_>>(),
            )
        } else {
            (nice_ticks(xr.0, xr.1), nice_ticks(yr.0, yr.1))
        };
        for &t in &xt {
            let x = tx(t);
            if ax.grid {
                let _ = write!(s, r##"<line x1="{x:.2}" y1="{py:.2}" x2="{x:.2}" y2="{:.2}" stroke="#000" stroke-opacity="0.3" stroke-width="0.8"/>"##, py + ph);
            }
            let _ = write!(s, r##"<line x1="{x:.2}" y1="{:.2}" x2="{x:.2}" y2="{:.2}" stroke="#000"/><text x="{x:.2}" y="{:.2}" text-anchor="middle" font-size="10">{}</text>"##, py + ph, py + ph + 4.0, py + ph + 15.0, fmt_tick(t));
        }
        for &t in &yt {
            let y = ty(t);
            if ax.grid {
                let _ = write!(s, r##"<line x1="{px:.2}" y1="{y:.2}" x2="{:.2}" y2="{y:.2}" stroke="#000" stroke-opacity="0.3" stroke-width="0.8"/>"##, px + pw);
            }
            let _ = write!(s, r##"<line x1="{:.2}" y1="{y:.2}" x2="{px:.2}" y2="{y:.2}" stroke="#000"/><text x="{:.2}" y="{:.2}" text-anchor="end" font-size="10">{}</text>"##, px - 4.0, px - 6.0, y + 3.5, fmt_tick(t));
        }
    }

    // data
    let _ = write!(s, r#"<g clip-path="url(#c{id})">"#);
    for it in &ax.items {
        match it {
            Item::Line { x, y, color, dashed, width } => {
                let mut d = String::new();
                let mut pen = false;
                for (a, b) in x.iter().zip(y) {
                    if a.is_finite() && b.is_finite() {
                        let _ = write!(d, "{}{:.2},{:.2} ", if pen { "L" } else { "M" }, tx(*a), ty(*b));
                        pen = true;
                    } else {
                        pen = false;
                    }
                }
                let dash = if *dashed { r#" stroke-dasharray="6,4""# } else { "" };
                let _ = write!(s, r#"<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{dash}/>"#);
            }
            Item::Scatter { x, y, color } => {
                for (a, b) in x.iter().zip(y) {
                    if a.is_finite() && b.is_finite() {
                        let _ = write!(s, r#"<circle cx="{:.2}" cy="{:.2}" r="4" fill="{color}"/>"#, tx(*a), ty(*b));
                    }
                }
            }
            Item::HLine(v) => {
                let _ = write!(s, r##"<line x1="{px:.2}" y1="{0:.2}" x2="{1:.2}" y2="{0:.2}" stroke="{2}" stroke-width="0.5"/>"##, ty(*v), px + pw, PALETTE[0]);
            }
            Item::VLine(v) => {
                let _ = write!(s, r##"<line x1="{0:.2}" y1="{py:.2}" x2="{0:.2}" y2="{1:.2}" stroke="{2}" stroke-width="0.5"/>"##, tx(*v), py + ph, PALETTE[0]);
            }
            Item::Heatmap { .. } => {}
        }
    }
    s.push_str("</g>");
    let _ = write!(s, r##"<rect x="{px:.2}" y="{py:.2}" width="{pw:.2}" height="{ph:.2}" fill="none" stroke="#000"/>"##);
    let _ = write!(s, r#"<text x="{:.2}" y="{:.2}" text-anchor="middle" font-size="12">{}</text>"#, px + pw / 2.0, py - 8.0, esc(&ax.title));
    if !ax.xlabel.is_empty() {
        let _ = write!(s, r#"<text x="{:.2}" y="{:.2}" text-anchor="middle">{}</text>"#, px + pw / 2.0, py + ph + 32.0, esc(&ax.xlabel));
    }
    if !ax.ylabel.is_empty() {
        let (lx, ly) = (x0 + 13.0, py + ph / 2.0);
        let _ = write!(s, r#"<text x="{lx:.2}" y="{ly:.2}" text-anchor="middle" transform="rotate(-90 {lx:.2} {ly:.2})">{}</text>"#, esc(&ax.ylabel));
    }
}

fn check(model: &SsaModel, comps: &[usize]) -> Result<()> {
    match comps.iter().find(|&&i| i >= model.d) {
        Some(&i) => Err(SsaError::IndexOutOfRange { index: i, len: model.d }),
        None => Ok(()),
    }
}

/// `0..min(n, d)` (the integer form of the Python `components` argument).
pub fn first_components(model: &SsaModel, n: usize) -> Vec<usize> {
    (0..n.min(model.d)).collect()
}

/// Eigenvectors `U_i` (`plot_eigenvectors`; Python defaults: components
/// `0..5`, `ncols = 2`, `figsize = (5 ncols, 3 ceil(n / ncols))`).
pub fn plot_eigenvectors(model: &SsaModel, components: &[usize], ncols: usize) -> Result<Figure> {
    check(model, components)?;
    let n = components.len();
    let mut fig = Figure::new(n, ncols, (5.0 * ncols as f64, 3.0 * (n as f64 / ncols as f64).ceil()));
    for (ax, &i) in fig.axes.iter_mut().zip(components) {
        let y: Vec<f64> = model.u.col(i).iter().copied().collect();
        ax.line((0..y.len()).map(|v| v as f64).collect(), y);
        ax.title = format!("U_{i} ({:.2}%)", model.rel_contribution[i]);
        ax.xlabel = "Lag".into();
        ax.grid = true;
    }
    Ok(fig)
}

/// Elementary matrices `X_i` as heat maps (`plot_elementary_matrices`;
/// Python defaults: components `0..5`, `ncols = 3`).
pub fn plot_elementary_matrices(model: &SsaModel, components: &[usize], ncols: usize) -> Result<Figure> {
    check(model, components)?;
    let n = components.len();
    let mut fig = Figure::new(n, ncols, (4.0 * ncols as f64, 3.0 * (n as f64 / ncols as f64).ceil()));
    for (ax, &i) in fig.axes.iter_mut().zip(components) {
        ax.heatmap(model.elementary_matrix(i)?, None, None);
        ax.title = format!("X_{i} ({:.2}%)", model.rel_contribution[i]);
        ax.hide_ticks = true;
        ax.colorbar = Some(String::new());
    }
    Ok(fig)
}

/// Weighted-correlation matrix (`plot_weighted_correlation`, figsize (7, 6)).
pub fn plot_weighted_correlation(model: &SsaModel) -> Figure {
    let w = model.compute_weighted_correlation_matrix();
    let mut fig = Figure::new(1, 1, (7.0, 6.0));
    let ax = &mut fig.axes[0];
    ax.heatmap(w, Some(0.0), Some(1.0));
    ax.xlabel = "Component".into();
    ax.ylabel = "Component".into();
    ax.title = "Weighted correlation".into();
    ax.colorbar = Some("W_ij".into());
    fig
}

/// Relative and cumulative contributions (`plot_contributions`, figsize (10, 4)).
pub fn plot_contributions(model: &SsaModel, n_components: Option<usize>) -> Figure {
    let n = n_components.map_or(model.d, |c| c.min(model.d));
    let x: Vec<f64> = (0..n).map(|v| v as f64).collect();
    let mut fig = Figure::new(2, 2, (10.0, 4.0));
    let panels = [(&model.rel_contribution, "Relative contribution"), (&model.cumsum_contr, "Cumulative contribution")];
    for (a, (data, title)) in fig.axes.iter_mut().zip(panels) {
        a.line(x.clone(), data[..n].to_vec());
        a.title = title.into();
        a.xlabel = "Component".into();
        a.ylabel = "Contribution (%)".into();
        a.grid = true;
    }
    fig
}

/// ESPRIT roots in the complex plane with the unit circle
/// (`plot_esprit_roots`, figsize (6, 6)).
pub fn plot_esprit_roots(model: &SsaModel, idx: &[usize]) -> Result<Figure> {
    let (mu, _, _) = model.estimate_esprit(idx)?;
    let mut fig = Figure::new(1, 1, (6.0, 6.0));
    let ax = &mut fig.axes[0];
    let theta: Vec<f64> = (0..400).map(|i| 2.0 * std::f64::consts::PI * i as f64 / 399.0).collect();
    ax.dashed_line(theta.iter().map(|t| t.cos()).collect(), theta.iter().map(|t| t.sin()).collect());
    ax.scatter(mu.iter().map(|z| z.re).collect(), mu.iter().map(|z| z.im).collect());
    ax.axhline(0.0).axvline(0.0);
    ax.xlabel = "Real part".into();
    ax.ylabel = "Imaginary part".into();
    ax.title = "ESPRIT roots".into();
    ax.equal_aspect = true;
    let limit = mu.iter().map(|z| 1.1 * z.norm()).fold(1.1, f64::max);
    ax.xlim = Some((-limit, limit));
    ax.ylim = Some((-limit, limit));
    Ok(fig)
}
