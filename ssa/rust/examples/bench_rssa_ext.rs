//! Timings of ssa_rs on the workloads of `rust/benchmarks/bench_rssa_ext.R`.
//!
//! cargo run --release --example bench_rssa_ext -- out.json

use std::f64::consts::PI;
use std::time::Instant;

use ssa_rs::rssa::cssa::Cssa;
use ssa_rs::rssa::gaps::ShapedSsa;
use ssa_rs::rssa::misc::hmatr;
use ssa_rs::rssa::mssa_list::MssaList;
use ssa_rs::rssa::nd::{ball_mask, NdMethod, NdSsaBuilder, Pairing};
use ssa_rs::rssa::pssa::{Projector, Pssa, Wossa};
use ssa_rs::rssa::{Base, Direction, SolveMethod, Subspace};
use ssa_rs::Complex64;

fn noise(i: usize) -> f64 {
    ((i * 7919 + 13) % 1009) as f64 / 1009.0 - 0.5
}

fn series(n: usize, phase: f64) -> Vec<f64> {
    (0..n)
        .map(|t| {
            let tf = t as f64;
            (2.0 * PI * tf / 50.0 + phase).sin() + 0.5 * (2.0 * PI * tf / 17.0).sin() + 0.001 * tf + 0.3 * noise(t)
        })
        .collect()
}

fn field2(n1: usize, n2: usize) -> Vec<f64> {
    let mut f = vec![0.0; n1 * n2];
    for j in 0..n2 {
        for i in 0..n1 {
            let (a, b) = (i as f64, j as f64);
            f[i + n1 * j] = (2.0 * PI * a / 20.0).cos() * (2.0 * PI * b / 13.0).cos() + 0.3 * (2.0 * PI * (a + b) / 31.0).sin() + 0.2 * noise(i * 131 + j);
        }
    }
    f
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
        println!("{case:<30} {op:<12} {sec:>10.4} s");
        self.0.push(format!("{{\"case\":\"{case}\",\"op\":\"{op}\",\"sec\":{sec}}}"));
    }
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "rust_times_ext.json".into());
    let mut out = Out(Vec::new());
    let g = |n: usize| (0..n).collect::<Vec<usize>>();

    let f = field2(100, 100);
    let c1 = "2D 100x100 circle(10)";
    let (bd, bm) = ball_mask(10, 2);
    let b1 = NdSsaBuilder::new(f.clone(), vec![100, 100]).wmask(bd, bm).method(NdMethod::Auto(20));
    out.add(c1, "decompose", timeit(|| b1.clone().fit().unwrap()));
    let s = b1.clone().fit().unwrap();
    out.add(c1, "reconstruct", timeit(|| s.reconstruct(&g(4)).unwrap()));

    let c2 = "2D 100x100 L=20x20 circular";
    let b2 = NdSsaBuilder::new(f.clone(), vec![100, 100]).window(vec![20, 20]).circular(vec![true, true]).method(NdMethod::Auto(20));
    out.add(c2, "decompose", timeit(|| b2.clone().fit().unwrap()));
    let s = b2.clone().fit().unwrap();
    out.add(c2, "reconstruct", timeit(|| s.reconstruct(&g(4)).unwrap()));

    let mut f3 = vec![0.0; 27000];
    for k in 0..30 {
        for j in 0..30 {
            for i in 0..30 {
                f3[i + 30 * j + 900 * k] =
                    (2.0 * PI * (i as f64 / 7.0 + j as f64 / 5.0 + k as f64 / 6.0)).cos() + 0.1 * noise(i + 30 * j + 900 * k);
            }
        }
    }
    let c3 = "3D 30^3 L=8^3";
    let b3 = NdSsaBuilder::new(f3, vec![30, 30, 30]).window(vec![8, 8, 8]).method(NdMethod::Auto(20));
    out.add(c3, "decompose", timeit(|| b3.clone().fit().unwrap()));
    let s3 = b3.clone().fit().unwrap();
    out.add(c3, "reconstruct", timeit(|| s3.reconstruct(&g(2)).unwrap()));
    out.add(c3, "esprit.memp", timeit(|| s3.parestimate(&g(2), Subspace::Column, None, SolveMethod::Ls, Pairing::Memp, None).unwrap()));

    let x = series(5000, 0.0);
    let c4 = "1D circular N=5000 L=1000";
    let b4 = NdSsaBuilder::new(x.clone(), vec![5000]).window(vec![1000]).circular(vec![true]).method(NdMethod::Auto(20));
    out.add(c4, "decompose", timeit(|| b4.clone().fit().unwrap()));
    let s4 = b4.clone().fit().unwrap();
    out.add(c4, "rforecast", timeit(|| s4.rforecast(&g(4), 100, true, false).unwrap()));

    let c5 = "MSSA list 3000+2500 L=500";
    let xl = vec![series(3000, 0.0), series(2500, 1.0)];
    out.add(c5, "decompose", timeit(|| MssaList::fit(&xl, Some(500), NdMethod::Auto(30)).unwrap()));
    let s5 = MssaList::fit(&xl, Some(500), NdMethod::Auto(30)).unwrap();
    out.add(c5, "rforecast.row", timeit(|| s5.rforecast(&g(4), 50, Base::Reconstructed, Direction::Row).unwrap()));
    out.add(c5, "vforecast.col", timeit(|| s5.vforecast(&g(4), 50, Direction::Column).unwrap()));

    let mut fna = f.clone();
    for j in 39..50 {
        for i in 39..50 {
            fna[i + 100 * j] = f64::NAN;
        }
    }
    let c6 = "2D NA 100x100 L=15x15";
    let b6 = NdSsaBuilder::new(fna, vec![100, 100]).window(vec![15, 15]).method(NdMethod::Auto(20));
    out.add(c6, "decompose", timeit(|| b6.clone().fit().unwrap()));
    let s6 = b6.clone().fit().unwrap();
    out.add(c6, "igapfill", timeit(|| s6.igapfill(&[g(6)], None, 1e-6, 5).unwrap()));

    let mut xg = series(2000, 0.0);
    for i in (499..520).chain(1499..1503) {
        xg[i] = f64::NAN;
    }
    let s7 = ShapedSsa::fit(&xg, 200).unwrap();
    out.add("Gaps N=2000 L=200", "gapfill.sim", timeit(|| s7.gapfill_simultaneous(&g(5), false).unwrap()));

    let (re, im) = (series(2000, 0.0), series(2000, 0.5));
    let z: Vec<Complex64> = re.iter().zip(&im).map(|(a, b)| Complex64::new(*a, *b)).collect();
    let c8 = "CSSA N=2000 L=500";
    out.add(c8, "decompose", timeit(|| Cssa::fit(&z, 500).unwrap()));
    let s8 = Cssa::fit(&z, 500).unwrap();
    out.add(c8, "esprit", timeit(|| s8.parestimate(&g(4), false).unwrap()));
    out.add(c8, "vforecast", timeit(|| s8.vforecast(&g(4), 50).unwrap()));

    let xh = series(500, 0.0);
    out.add("hmatr N=500 L=20", "hmatr", timeit(|| hmatr(&xh, 125, 125, 20, 10).unwrap()));
    let xp = series(2000, 0.0);
    let sp = Pssa::fit(&xp, 500, &Projector::Degree(1), &Projector::Degree(1), Some(30)).unwrap();
    out.add("PSSA N=2000 L=500", "vforecast", timeit(|| sp.vforecast(&g(5), 50).unwrap()));
    out.add("PSSA N=2000 L=500", "rforecast", timeit(|| sp.rforecast(&g(5), 50).unwrap()));

    let xw = series(2000, 0.0);
    let cw: Vec<f64> = (1..=500).map(|i| 1.0 + 0.5 * (2.0 * PI * i as f64 / 500.0).sin()).collect();
    let rw: Vec<f64> = (1..=1501).map(|i| 1.0 + 0.3 * (2.0 * PI * i as f64 / 400.0).cos()).collect();
    let cw9 = "W-OSSA N=2000 L=500";
    out.add(cw9, "decompose", timeit(|| Wossa::fit(&xw, 500, &cw, &rw).unwrap()));
    let sw = Wossa::fit(&xw, 500, &cw, &rw).unwrap();
    out.add(cw9, "rforecast", timeit(|| sw.rforecast(&g(4), 50).unwrap()));
    out.add(cw9, "vforecast", timeit(|| sw.vforecast(&g(4), 50).unwrap()));
    out.add(c8, "reconstruct", timeit(|| s8.reconstruct(&g(4)).unwrap()));

    std::fs::write(&path, format!("[{}]\n", out.0.join(",\n"))).unwrap();
    println!("written {path}");
}
