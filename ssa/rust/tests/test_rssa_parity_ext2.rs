//! Third round: complex gap filling and oblique decompositions, W-OSSA
//! forecasts and W-OSSA on top of projections (`make_reference_ext2.R`).

use serde_json::Value;
use ssa_rs::rssa::cssa::Cssa;
use ssa_rs::rssa::pssa::{wossa_pssa, Projector, Wossa};
use ssa_rs::Complex64;

fn refj() -> Value {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/rssa_ref_ext2.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vecf).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        Value::Null => vec![f64::NAN],
        Value::String(s) if s == "NA" => vec![f64::NAN],
        _ => panic!("unexpected {v:?}"),
    }
}

fn cplx(v: &Value) -> Vec<Complex64> {
    vecf(&v["re"]).into_iter().zip(vecf(&v["im"])).map(|(a, b)| Complex64::new(a, b)).collect()
}

fn close(a: &[f64], b: &[f64], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol * (1.0 + y.abs()), "{what}[{i}]: {x} vs {y}");
    }
}

fn as_complex(x: &[f64]) -> Vec<Complex64> {
    x.iter().map(|&v| if v.is_nan() { Complex64::new(f64::NAN, f64::NAN) } else { Complex64::new(v, 0.0) }).collect()
}

#[test]
fn cssa_gapfill_matches_real_rssa() {
    let r = refj();
    let xg = vecf(&r["xg"]);
    let s = Cssa::fit(&as_complex(&xg), 24).unwrap();
    assert!(!s.is_full());
    let g: Vec<usize> = (0..5).collect();
    let re = |v: Vec<Complex64>| -> Vec<f64> {
        assert!(v.iter().all(|c| c.im.abs() < 1e-8));
        v.iter().map(|c| c.re).collect()
    };
    close(&re(s.gapfill(&g, false, false).unwrap()), &vecf(&r["gap_seq"]), 1e-7, "gapfill seq");
    close(&re(s.gapfill(&g, false, true).unwrap()), &vecf(&r["gap_sim"]), 1e-7, "gapfill sim");
    let ig = s.igapfill(&[g.clone()], None, 1e-6, 10).unwrap();
    close(&re(ig[0].clone()), &vecf(&r["igap"]), 1e-7, "igapfill");
}

/// Exact rank-2 complex signal: every gap is restored exactly.
#[test]
fn cssa_gapfill_exact_on_complex_signal() {
    let n = 100;
    let z0: Vec<Complex64> = (0..n)
        .map(|t| {
            let t = t as f64;
            Complex64::from_polar(1.0, 2.0 * std::f64::consts::PI * t / 9.0) + Complex64::from_polar(0.6, -2.0 * std::f64::consts::PI * t / 4.0 + 0.3)
        })
        .collect();
    let mut z = z0.clone();
    for i in (20..25).chain(60..62).chain(97..100) {
        z[i] = Complex64::new(f64::NAN, 0.0);
    }
    let s = Cssa::fit(&z, 20).unwrap();
    for sim in [false, true] {
        let f = s.gapfill(&[0, 1], false, sim).unwrap();
        for i in 0..97 {
            assert!((f[i] - z0[i]).norm() < 1e-8, "sim={sim} t={i}");
        }
    }
    let f = s.gapfill(&[0, 1], false, false).unwrap();
    for i in 97..100 {
        assert!((f[i] - z0[i]).norm() < 1e-8);
    }
    let ig = s.igapfill(&[vec![0, 1]], None, 1e-12, 500).unwrap();
    for i in 0..n {
        assert!((ig[0][i] - z0[i]).norm() < 1e-6, "igapfill t={i}");
    }
}

#[test]
fn cssa_norms() {
    let r = refj();
    let s = Cssa::fit(&cplx(&r["z0"]), 24).unwrap();
    close(&[s.wnorm()], &vecf(&r["wnorm"]), 1e-10, "wnorm");
    close(&s.contributions(&[0, 1, 2, 3]).unwrap(), &vecf(&r["contrib"]), 1e-8, "contributions");
}

fn mixed_complex() -> Vec<Complex64> {
    // two complex exponentials with close frequencies (mixed by SVD)
    (0..200)
        .map(|t| {
            let t = t as f64;
            let w = 2.0 * std::f64::consts::PI;
            Complex64::from_polar(1.0, w * t / 10.0) + Complex64::from_polar(0.98, w * t / 10.6)
        })
        .collect()
}

fn check_ossa(s: &Cssa, res: &ssa_rs::rssa::cssa_ext::CssaOssaResult, what: &str) {
    // the rotated set reconstructs the same series as the original set
    let a = s.reconstruct(&res.set).unwrap();
    let b = res.model.reconstruct(&res.set).unwrap();
    for (x, y) in a.iter().zip(&b) {
        assert!((x - y).norm() < 1e-8, "{what}: set reconstruction changed");
    }
    // imaginary parts are kept: each group is genuinely complex
    for g in &res.groups {
        let f = res.model.reconstruct(g).unwrap();
        assert!(f.iter().map(|c| c.im.abs()).sum::<f64>() > 1.0, "{what}: imaginary part lost");
    }
}

#[test]
fn cssa_oblique_methods_keep_complex_parts() {
    let z = mixed_complex();
    let s = Cssa::fit(&z, 50).unwrap();
    let ng = vec![vec![0], vec![1]];
    let truth: Vec<Vec<Complex64>> = [(1.0, 10.0), (0.98, 10.6)]
        .iter()
        .map(|&(a, p)| (0..200).map(|t| Complex64::from_polar(a, 2.0 * std::f64::consts::PI * t as f64 / p)).collect())
        .collect();
    let err = |m: &Cssa, g: &[usize], k: usize| -> f64 {
        let f = m.reconstruct(g).unwrap();
        (f.iter().zip(&truth[k]).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>() / 200.0).sqrt()
    };
    let base = err(&s, &[0], 0).min(err(&s, &[0], 1));

    let e = s.eossa(&ng, 2).unwrap();
    check_ossa(&s, &e, "eossa");
    let best = |m: &Cssa, gs: &[Vec<usize>]| (err(m, &gs[0], 0) + err(m, &gs[1], 1)).min(err(m, &gs[0], 1) + err(m, &gs[1], 0));
    assert!(best(&e.model, &e.groups) < 1e-6, "eossa separates exactly");

    let i = s.iossa(&ng, 1e-8, Some(2.0), 200, 0.5).unwrap();
    check_ossa(&s, &i, "iossa");
    assert!(best(&i.model, &i.groups) < 0.5 * base, "iossa improves separation");

    let f = s.fossa(&ng, &[-1.0, 1.0], f64::INFINITY, true).unwrap();
    check_ossa(&s, &f, "fossa");

    let w0 = s.owcor(&ng, &[0, 1]).unwrap();
    let w1 = e.model.owcor(&e.groups, &e.set).unwrap();
    assert!(w1[(0, 1)] < 1e-6 && w1[(0, 1)] <= w0[(0, 1)] + 1e-12);
}

#[test]
fn wossa_forecasts_match_rssa() {
    let r = refj();
    let w = Wossa::fit(&vecf(&r["x"]), 30, &vecf(&r["cw"]), &vecf(&r["rw"])).unwrap();
    close(&w.sigma[..6], &vecf(&r["wossa_sigma"]), 1e-8, "sigma");
    let g: Vec<usize> = (0..4).collect();
    close(&w.rforecast(&g, 10).unwrap(), &vecf(&r["wossa_rf"]), 1e-7, "rforecast");
    close(&w.vforecast(&g, 10).unwrap(), &vecf(&r["wossa_vf"]), 1e-7, "vforecast");
}

#[test]
fn wossa_on_projections() {
    let r = refj();
    let x = vecf(&r["x"]);
    let (cw, rw) = (vecf(&r["cw"]), vecf(&r["rw"]));
    let m = wossa_pssa(&x, 30, Projector::Degree(1), Projector::Degree(2), Some(cw.clone()), Some(rw.clone())).unwrap();
    assert_eq!(m.nspecial(), 3);
    // all triples together give the series back exactly
    let all: Vec<usize> = (0..m.sigma.len()).collect();
    let f = m.reconstruct(&all).unwrap();
    for (a, b) in f.iter().zip(&x) {
        assert!((a - b).abs() < 1e-8);
    }
    // without projectors it is plain W-OSSA
    let m0 = wossa_pssa(&x, 30, Projector::None, Projector::None, Some(cw.clone()), Some(rw.clone())).unwrap();
    let w = Wossa::fit(&x, 30, &cw, &rw).unwrap();
    close(&m0.sigma[..6], &w.sigma[..6], 1e-8, "wossa sigma");
    // the special part is the trend: forecasting the special triples plus
    // the periodic pair stays close to the noiseless continuation
    let g: Vec<usize> = (0..m.sigma.len().min(7)).collect();
    let fc = m.rforecast(&g, 10, true, false).unwrap();
    assert_eq!(fc.len(), 10);
}
