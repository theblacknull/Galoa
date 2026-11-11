//! Timings of ssa_rs on the workloads of `rust/benchmarks/bench_rssa.R`.
//!
//! cargo run --release --example bench_rssa -- out.json

use std::f64::consts::PI;
use std::time::Instant;

use faer::Mat;
use ssa_rs::rssa::gaps::{igapfill, ShapedSsa};
use ssa_rs::rssa::parest::{SolveMethod, Subspace};
use ssa_rs::rssa::pssa::{Projector, Pssa};
use ssa_rs::rssa::ssa2d::{Method2d, Ssa2d};
use ssa_rs::rssa::{Base, Direction};
use ssa_rs::*;

fn series(n: usize, phase: f64) -> Vec<f64> {
    (0..n)
        .map(|t| {
            let tf = t as f64;
            (2.0 * PI * tf / 50.0 + phase).sin() + 0.5 * (2.0 * PI * tf / 17.0).sin() + 0.001 * tf
                + 0.3 * (((t * 7919 + 13) % 1009) as f64 / 1009.0 - 0.5)
        })
        .collect()
}

fn timeit<T>(mut f: impl FnMut() -> T) -> f64 {
    let mut times = Vec::new();
    let end = Instant::now() + std::time::Duration::from_secs(8);
    loop {
        let t0 = Instant::now();
        std::hint::black_box(f());
        times.push(t0.elapsed().as_secs_f64());
        if times.len() >= 7 || Instant::now() > end {
            break;
        }
    }
    times.sort_by(f64::total_cmp);
    let n = times.len();
    if n % 2 == 1 { times[n / 2] } else { 0.5 * (times[n / 2 - 1] + times[n / 2]) }
}

struct Out(Vec<String>);
impl Out {
    fn add(&mut self, case: &str, op: &str, sec: f64) {
        println!("{case:<28} {op:<14} {sec:>10.4} s");
        self.0.push(format!("{{\"case\":\"{case}\",\"op\":\"{op}\",\"sec\":{sec}}}"));
    }
}

fn strategy(l: usize, neig: usize) -> Box<dyn SvdStrategy> {
    // same rule as Rssa's svd.method = "auto": exact below L = 500
    if l < 500 { Box::new(FullSvd) } else { Box::new(LanczosSvd::new(neig)) }
}

fn fit1(x: &[f64], l: usize, neig: usize) -> SsaModel {
    let mut e = Estimator::default();
    e.decomposition_strategy = strategy(l, neig);
    e.fit_matrix(Mat::from_fn(x.len(), 1, |i, _| x[i]), Some(l)).unwrap()
}

fn run1d(out: &mut Out, case: &str, n: usize, l: usize, neig: usize) {
    let x = series(n, 0.0);
    out.add(case, "decompose", timeit(|| fit1(&x, l, neig)));
    let m = fit1(&x, l, neig);
    let g10: Vec<usize> = (0..10).collect();
    let g5: Vec<usize> = (0..5).collect();
    let singles: Vec<Vec<usize>> = (0..20).map(|i| vec![i]).collect();
    out.add(case, "reconstruct", timeit(|| m.reconstruct_components(&g10).unwrap()));
    out.add(case, "wcor20", timeit(|| m.wcor(Some(&singles)).unwrap()));
    out.add(case, "rforecast", timeit(|| m.rforecast(&g5, 100, Base::Reconstructed, Direction::Column, false).unwrap()));
    out.add(case, "vforecast", timeit(|| m.vforecast(&g5, 100, Direction::Column).unwrap()));
    out.add(case, "esprit", timeit(|| m.parestimate_esprit(&g5, Subspace::Column, SolveMethod::Ls, false).unwrap()));
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "rust_times.json".into());
    let mut out = Out(Vec::new());
    run1d(&mut out, "1D N=1000 L=400", 1000, 400, 50);
    run1d(&mut out, "1D N=10000 L=2000", 10000, 2000, 50);
    run1d(&mut out, "1D N=100000 L=10000", 100000, 10000, 50);
    run1d(&mut out, "1D N=500000 L=5000", 500000, 5000, 20);

    // MSSA
    let cols: Vec<Vec<f64>> = (0..5).map(|p| series(2000, p as f64)).collect();
    let xm = Mat::from_fn(2000, 5, |i, j| cols[j][i]);
    let case = "MSSA 5x2000 L=500";
    let fitm = || Mssa::new().with_decomposition(LanczosSvd::new(50)).fit(xm.as_ref(), Some(500)).unwrap();
    out.add(case, "decompose", timeit(fitm));
    let m = fitm();
    let g10: Vec<usize> = (0..10).collect();
    let g5: Vec<usize> = (0..5).collect();
    let singles: Vec<Vec<usize>> = (0..20).map(|i| vec![i]).collect();
    out.add(case, "reconstruct", timeit(|| m.reconstruct_components(&g10).unwrap()));
    out.add(case, "wcor20", timeit(|| m.wcor(Some(&singles)).unwrap()));
    out.add(case, "rforecast", timeit(|| m.rforecast(&g5, 100, Base::Reconstructed, Direction::Row, false).unwrap()));
    out.add(case, "vforecast", timeit(|| m.vforecast(&g5, 100, Direction::Row).unwrap()));

    // 2D-SSA
    let f = Mat::from_fn(200, 200, |i, j| {
        let (a, b) = (i as f64, j as f64);
        (2.0 * PI * a / 20.0).cos() * (2.0 * PI * b / 13.0).cos() + 0.3 * (2.0 * PI * (a + b) / 31.0).sin()
            + 0.2 * (((i * 7919 + j * 104729) % 1009) as f64 / 1009.0 - 0.5)
    });
    let case = "2D 200x200 L=50x50";
    out.add(case, "decompose", timeit(|| Ssa2d::fit(&f, (50, 50), Method2d::Lanczos(20)).unwrap()));
    let s2 = Ssa2d::fit(&f, (50, 50), Method2d::Lanczos(20)).unwrap();
    out.add(case, "reconstruct", timeit(|| s2.reconstruct(&[0, 1, 2, 3, 4, 5]).unwrap()));

    // Toeplitz
    let xt = series(5000, 0.0);
    out.add("Toeplitz N=5000 L=1000", "decompose", timeit(|| {
        Ssa::new().with_decomposition(ToeplitzSvd { neig: Some(20) }).fit(&xt, Some(1000)).unwrap()
    }));

    // iterative / nested
    let x2 = series(2000, 0.0);
    let case = "Iterative N=2000 L=500";
    let m2 = fit1(&x2, 500, 30);
    out.add(case, "cadzow", timeit(|| m2.cadzow(5, true, 1e-6, 20).unwrap()));
    out.add(case, "iossa", timeit(|| m2.iossa(&[vec![1, 2], vec![3, 4]], 1e-5, Some(2.0), 20, 0.5).unwrap()));
    out.add(case, "fossa", timeit(|| m2.fossa(&[vec![1, 2, 3, 4]], &[-1.0, 1.0], f64::INFINITY, true).unwrap()));
    out.add(case, "eossa", timeit(|| m2.eossa(&[vec![1, 2, 3, 4]], 2, SolveMethod::Ls).unwrap()));
    let comps: Vec<usize> = (0..20).collect();
    out.add(case, "grouping.auto", timeit(|| m2.grouping_auto_wcor(&comps, 10).unwrap()));

    let mut xg = x2.clone();
    for i in (299..340).chain(1199..1210) {
        xg[i] = f64::NAN;
    }
    let case = "Gaps N=2000 L=200";
    out.add(case, "decompose", timeit(|| ShapedSsa::fit(&xg, 200).unwrap()));
    let sg = ShapedSsa::fit(&xg, 200).unwrap();
    out.add(case, "gapfill", timeit(|| sg.gapfill(&[0, 1, 2, 3, 4], false).unwrap()));
    out.add(case, "igapfill", timeit(|| igapfill(&xg, 200, &[vec![0, 1, 2, 3, 4]], None, 1e-6, 10).unwrap()));

    out.add("PSSA N=2000 L=500", "decompose", timeit(|| {
        Pssa::fit(&x2, 500, &Projector::Degree(1), &Projector::Degree(1), Some(30)).unwrap()
    }));

    std::fs::write(&path, format!("[{}]\n", out.0.join(",\n"))).unwrap();
    println!("written {path}");
}
