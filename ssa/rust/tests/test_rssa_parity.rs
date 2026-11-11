//! Parity of the Rssa ports against results computed by Rssa 1.1 itself
//! (`rust/benchmarks/make_reference.R` -> `tests/data/rssa_ref.json`).

use faer::Mat;
use serde_json::Value;
use ssa_rs::rssa::common::Group;
use ssa_rs::rssa::cssa::Cssa;
use ssa_rs::rssa::gaps::{classify_gaps, igapfill, GapPos, ShapedSsa};
use ssa_rs::rssa::grouping::{FreqBins, PgramBase, PgramMethod};
use ssa_rs::rssa::pssa::{Projector, Pssa, Wossa};
use ssa_rs::rssa::ssa2d::{Method2d, Ssa2d};
use ssa_rs::rssa::*;
use ssa_rs::*;

fn refj() -> Value {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/rssa_ref.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vecf).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        Value::String(s) if s == "NA" => vec![f64::NAN],
        Value::String(s) => vec![s.trim().parse().unwrap()],
        Value::Object(o) => o.values().flat_map(vecf).collect(),
        _ => panic!("unexpected {v:?}"),
    }
}

fn rows(v: &Value) -> Matrix {
    let a = v.as_array().unwrap();
    let r: Vec<Vec<f64>> = a.iter().map(vecf).collect();
    Mat::from_fn(r.len(), r[0].len(), |i, j| r[i][j])
}

fn cplx(v: &Value) -> Vec<Complex64> {
    let re = vecf(&v["re"]);
    let im = vecf(&v["im"]);
    re.into_iter().zip(im).map(|(a, b)| Complex64::new(a, b)).collect()
}

fn close(a: &[f64], b: &[f64], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    let scale = b.iter().filter(|v| v.is_finite()).fold(1.0f64, |m, v| m.max(v.abs()));
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        if y.is_nan() {
            assert!(x.is_nan(), "{what}[{i}]: expected NaN, got {x}");
            continue;
        }
        assert!((x - y).abs() <= tol * scale, "{what}[{i}]: {x} vs {y} (tol {})", tol * scale);
    }
}

fn close_roots(a: &[Complex64], b: &[Complex64], tol: f64, what: &str) {
    let key = |z: &Complex64| (z.re * 1e6).round() as i64 * 1_000_000_000 + (z.im * 1e6).round() as i64;
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    a.sort_by_key(key);
    b.sort_by_key(key);
    close(&a.iter().flat_map(|z| [z.re, z.im]).collect::<Vec<_>>(), &b.iter().flat_map(|z| [z.re, z.im]).collect::<Vec<_>>(), tol, what);
}

fn col0(m: &Matrix) -> Vec<f64> {
    m.col(0).iter().copied().collect()
}

fn flat(m: &Matrix) -> Vec<f64> {
    m.col_iter().flat_map(|c| c.iter().copied().collect::<Vec<_>>()).collect()
}

fn groups(v: &Value) -> Vec<Group> {
    let it: Vec<&Value> = match v {
        Value::Object(o) => o.values().collect(),
        Value::Array(a) => a.iter().collect(),
        _ => panic!(),
    };
    it.into_iter().map(|g| vecf(g).into_iter().map(|x| x as usize - 1).collect()).collect()
}

fn basic_model(r: &Value) -> SsaModel {
    let x = vecf(&r["x"]);
    Ssa::new().fit(&x, Some(60)).unwrap()
}

const T: f64 = 1e-9;

#[test]
fn basic_1d_ssa() {
    let r = refj();
    let b = &r["basic"];
    let m = basic_model(&r);
    close(&m.sigma[..20], &vecf(&b["sigma"]), T, "sigma");
    let gs: Vec<Group> = vec![vec![0], vec![1, 2], vec![3, 4], (0..7).collect()];
    for (g, want) in gs.iter().zip(b["rec"].as_object().unwrap().values()) {
        close(&col0(&m.reconstruct_components(g).unwrap()), &vecf(want), T, "rec");
    }
    let singles: Vec<Group> = (0..10).map(|i| vec![i]).collect();
    close(&flat(&m.wcor(Some(&singles)).unwrap()), &flat(&rows(&b["wcor"])), 1e-8, "wcor");
    close(&flat(&m.wcor(Some(&gs[..3])).unwrap()), &flat(&rows(&b["wcor_groups"])), 1e-8, "wcor groups");
    close(&flat(&m.frobenius_cor(&gs[..3]).unwrap()), &flat(&rows(&b["frob"])), 1e-8, "frobenius.cor");
    close(&m.contributions()[..10], &vecf(&b["contrib"]), T, "contributions");
    close(&[m.wnorm()], &vecf(&b["wnorm"]), T, "wnorm");
}

#[test]
fn forecasting() {
    let r = refj();
    let b = &r["basic"];
    let m = basic_model(&r);
    let g: Vec<usize> = (0..5).collect();
    close(&m.lrr(&g, false).unwrap(), &vecf(&b["lrr"]), 1e-8, "lrr");
    close(&m.lrr(&g, true).unwrap(), &vecf(&b["lrr_rev"]), 1e-8, "lrr reverse");
    close_roots(&lrr_roots(&m.lrr(&g, false).unwrap()).unwrap(), &cplx(&b["roots"]), 1e-7, "roots");
    let rf = |base, rev| col0(&m.rforecast(&g, 20, base, Direction::Column, rev).unwrap());
    close(&rf(Base::Reconstructed, false), &vecf(&b["rforecast"]), 1e-8, "rforecast");
    close(&rf(Base::Original, false), &vecf(&b["rforecast_orig"]), 1e-8, "rforecast original");
    close(&rf(Base::Reconstructed, true), &vecf(&b["rforecast_rev"]), 1e-8, "rforecast reverse");
    close(&col0(&m.vforecast(&g, 20, Direction::Column).unwrap()), &vecf(&b["vforecast"]), 1e-8, "vforecast");
}

#[test]
fn parestimate() {
    let r = refj();
    let b = &r["basic"];
    let m = basic_model(&r);
    let g = [1, 2, 3, 4];
    let e = m.parestimate_esprit(&g, Subspace::Column, SolveMethod::Ls, false).unwrap();
    close_roots(&e.roots, &cplx(&b["esprit"]), 1e-8, "esprit");
    // R eigen order: decreasing modulus
    let want = cplx(&b["esprit"]);
    close(&e.moduli, &want.iter().map(|z| z.norm()).collect::<Vec<_>>(), 1e-8, "esprit order");
    let mut p = e.periods.clone();
    let mut q = vecf(&b["periods"]);
    p.sort_by(f64::total_cmp);
    q.sort_by(f64::total_cmp);
    close(&p, &q, 1e-7, "periods");
    let t = m.parestimate_esprit(&g, Subspace::Column, SolveMethod::Tls, false).unwrap();
    close_roots(&t.roots, &cplx(&b["esprit_tls"]), 1e-8, "esprit tls");
    let rw = m.parestimate_esprit(&g, Subspace::Row, SolveMethod::Ls, false).unwrap();
    close_roots(&rw.roots, &cplx(&b["esprit_row"]), 1e-8, "esprit row");
    let (pp, _) = m.parestimate_pairs(&[1, 2], Subspace::Column, false).unwrap();
    close_roots(&pp.roots, &cplx(&b["pairs"]), 1e-8, "pairs");
}

#[test]
fn cadzow_and_grouping() {
    let r = refj();
    let b = &r["basic"];
    let m = basic_model(&r);
    close(&col0(&m.cadzow(5, true, 1e-6, 10).unwrap()), &vecf(&b["cadzow"]), 1e-8, "cadzow");
    close(&col0(&m.cadzow(5, false, 1e-6, 10).unwrap()), &vecf(&b["cadzow_nc"]), 1e-8, "cadzow uncorrected");
    let comps: Vec<usize> = (0..10).collect();
    assert_eq!(m.grouping_auto_wcor(&comps, 5).unwrap(), groups(&b["gauto_wcor"]));
    let pg = m.grouping_auto_pgram(&comps, PgramBase::Series, &FreqBins::Equal(4), &[0.0], PgramMethod::Constant, true).unwrap();
    assert_eq!(pg.groups, groups(&b["gauto_pgram"]));
    close(&flat(&pg.contributions), &flat(&rows(&b["gauto_pgram_contrib"])), 1e-9, "pgram contributions");
    // explicit bands, threshold, linear integration
    let bands = FreqBins::Ranges(vec![(0.0, 0.05), (0.05, 0.2)]);
    let pl = m.grouping_auto_pgram(&comps, PgramBase::Series, &bands, &[0.3], PgramMethod::Linear, false).unwrap();
    assert!(!pl.splitting);
    assert_eq!(pl.groups, groups(&b["gauto_pgram_lin"]));
    close(&flat(&pl.contributions), &flat(&rows(&b["gauto_pgram_lin_contrib"])), 1e-9, "pgram linear contributions");
}

#[test]
fn oblique_decompositions() {
    let r = refj();
    let m = basic_model(&r);
    // F-OSSA (DerivSSA)
    let fo = m.fossa(&[vec![1, 2, 3, 4]], &[-1.0, 1.0], f64::INFINITY, true).unwrap();
    close(&fo.model.sigma[1..5], &vecf(&r["fossa"]["sigma"]), 1e-7, "fossa sigma");
    for (i, want) in r["fossa"]["rec"].as_object().unwrap().values().enumerate() {
        close(&col0(&fo.model.reconstruct_components(&[i + 1]).unwrap()), &vecf(want), 1e-7, "fossa rec");
    }
    // I-OSSA
    let io = m.iossa(&[vec![1, 2], vec![3, 4]], 1e-8, Some(2.0), 50, 0.5).unwrap();
    assert!(io.converged);
    assert_eq!(io.groups, groups(&r["iossa"]["groups"]));
    for (g, want) in io.groups.iter().zip(r["iossa"]["rec"].as_object().unwrap().values()) {
        close(&col0(&io.model.reconstruct_components(g).unwrap()), &vecf(want), 1e-6, "iossa rec");
    }
    // E-OSSA
    let eo = m.eossa(&[vec![1, 2, 3, 4]], 2, SolveMethod::Ls).unwrap();
    let mut got = eo.groups.clone();
    let mut want = groups(&r["eossa"]["groups"]);
    got.sort();
    want.sort();
    assert_eq!(got, want);
    let recs: Vec<Vec<f64>> = r["eossa"]["rec"].as_object().unwrap().values().map(vecf).collect();
    for g in &eo.groups {
        let mine = col0(&eo.model.reconstruct_components(g).unwrap());
        let wi = groups(&r["eossa"]["groups"]).iter().position(|w| w == g).unwrap();
        close(&mine, &recs[wi], 1e-7, "eossa rec");
    }
    let ow = eo.model.owcor(&groups(&r["eossa"]["groups"]), &eo.set).unwrap();
    close(&flat(&ow), &flat(&rows(&r["eossa"]["owcor"])), 1e-7, "owcor");
}

#[test]
fn lanczos_vs_nutrlan() {
    let r = refj();
    let lz = &r["lanczos"];
    let x = vecf(&lz["x"]);
    let m = Ssa::new().with_decomposition(LanczosSvd::new(20)).fit(&x, Some(1000)).unwrap();
    assert_eq!(m.d, 20);
    close(&m.sigma, &vecf(&lz["sigma"]), 1e-9, "lanczos sigma");
    close(&col0(&m.reconstruct_components(&[0, 1, 2, 3, 4]).unwrap()), &vecf(&lz["rec"]), 1e-8, "lanczos rec");
    // and against the exact dense SVD
    let e = Ssa::new().fit(&x, Some(1000)).unwrap();
    close(&m.sigma, &e.sigma[..20], 1e-11, "lanczos vs full");
}

#[test]
fn toeplitz() {
    let r = refj();
    let x = vecf(&r["x"]);
    let m = Ssa::new().with_decomposition(ToeplitzSvd { neig: Some(60) }).fit(&x, Some(60)).unwrap();
    close(&m.sigma[..20], &vecf(&r["toeplitz"]["sigma"]), T, "toeplitz sigma");
    let gs: Vec<Group> = vec![vec![0], vec![1, 2], (0..5).collect()];
    for (g, want) in gs.iter().zip(r["toeplitz"]["rec"].as_object().unwrap().values()) {
        close(&col0(&m.reconstruct_components(g).unwrap()), &vecf(want), 1e-8, "toeplitz rec");
    }
}

#[test]
fn mssa() {
    let r = refj();
    let ms = &r["mssa"];
    let x = rows(&ms["x"]);
    let m = Mssa::new().fit(x.as_ref(), Some(50)).unwrap();
    close(&m.sigma[..10], &vecf(&ms["sigma"]), T, "mssa sigma");
    let g = [0, 1, 2];
    close(&flat(&m.reconstruct_components(&g).unwrap()), &vecf(&ms["rec"]), 1e-8, "mssa rec");
    close(&flat(&m.rforecast(&g, 12, Base::Reconstructed, Direction::Row, false).unwrap()), &vecf(&ms["rf_row"]), 1e-8, "rf row");
    close(&flat(&m.rforecast(&g, 12, Base::Reconstructed, Direction::Column, false).unwrap()), &vecf(&ms["rf_col"]), 1e-8, "rf col");
    close(&flat(&m.vforecast(&g, 12, Direction::Row).unwrap()), &vecf(&ms["vf_row"]), 1e-8, "vf row");
    close(&flat(&m.vforecast(&g, 12, Direction::Column).unwrap()), &vecf(&ms["vf_col"]), 1e-8, "vf col");
    // Python K_forecast == Rssa row rforecast (unit weights)
    close(&flat(&m.k_forecast(12, &g, false).unwrap()), &vecf(&ms["rf_row"]), 1e-8, "K_forecast");
}

#[test]
fn complex_ssa() {
    let r = refj();
    let c = &r["cssa"];
    let z = cplx(&c["z"]);
    let m = Cssa::fit(&z, 40).unwrap();
    close(&m.sigma[..10], &vecf(&c["sigma"]), T, "cssa sigma");
    let rec = m.reconstruct(&[0, 1]).unwrap();
    let want = cplx(&c["rec"]);
    close(&rec.iter().flat_map(|z| [z.re, z.im]).collect::<Vec<_>>(), &want.iter().flat_map(|z| [z.re, z.im]).collect::<Vec<_>>(), 1e-8, "cssa rec");
    let rf = m.rforecast(&[0, 1], 10).unwrap();
    let want = cplx(&c["rf"]);
    close(&rf.iter().flat_map(|z| [z.re, z.im]).collect::<Vec<_>>(), &want.iter().flat_map(|z| [z.re, z.im]).collect::<Vec<_>>(), 1e-8, "cssa rforecast");
}

#[test]
fn ssa_2d() {
    let r = refj();
    let s = &r["ssa2d"];
    let f = rows(&s["field"]);
    let l = vecf(&s["L"]);
    let l = (l[0] as usize, l[1] as usize);
    let m = Ssa2d::fit(&f, l, Method2d::Full).unwrap();
    close(&m.sigma[..10], &vecf(&s["sigma"]), T, "2d sigma");
    close(&flat(&m.reconstruct(&[0, 1, 2, 3]).unwrap()), &flat(&rows(&s["rec"])), 1e-8, "2d rec");
    let (p1, p2) = m.parestimate(&[0, 1, 2, 3, 4, 5], SolveMethod::Ls).unwrap();
    close_roots(&p1.roots, &cplx(&s["p1"]), 1e-7, "2d esprit x");
    close_roots(&p2.roots, &cplx(&s["p2"]), 1e-7, "2d esprit y");
    // matrix-free Lanczos gives the same leading triples
    let ml = Ssa2d::fit(&f, l, Method2d::Lanczos(10)).unwrap();
    close(&ml.sigma, &m.sigma[..10], 1e-10, "2d lanczos");
}

#[test]
fn projection_and_oblique() {
    let r = refj();
    let x = vecf(&r["x"]);
    let p = Pssa::fit(&x, 60, &Projector::Degree(1), &Projector::Degree(1), None).unwrap();
    close(&p.model.sigma[..10], &vecf(&r["pssa"]["sigma"]), T, "pssa sigma");
    let gs: Vec<Group> = vec![vec![0], vec![1], (0..6).collect()];
    for (g, want) in gs.iter().zip(r["pssa"]["rec"].as_object().unwrap().values()) {
        close(&col0(&p.model.reconstruct_components(g).unwrap()), &vecf(want), 1e-8, "pssa rec");
    }
    close(&p.rforecast(&(0..6).collect::<Vec<_>>(), 15).unwrap(), &vecf(&r["pssa"]["rf"]), 1e-7, "pssa rforecast");
    let p2 = Pssa::fit(&x, 60, &Projector::Degree(2), &Projector::None, None).unwrap();
    close(&p2.model.sigma[..10], &vecf(&r["pssa2"]["sigma"]), T, "pssa2 sigma");
    close(&col0(&p2.model.reconstruct_components(&[0, 1, 2, 3, 4]).unwrap()), &vecf(&r["pssa2"]["rec"]), 1e-8, "pssa2 rec");

    let w = Wossa::fit(&x, 60, &vecf(&r["wossa"]["cw"]), &vecf(&r["wossa"]["rw"])).unwrap();
    close(&w.sigma[..10], &vecf(&r["wossa"]["sigma"]), T, "wossa sigma");
    let gs: Vec<Group> = vec![vec![0], vec![1, 2]];
    for (g, want) in gs.iter().zip(r["wossa"]["rec"].as_object().unwrap().values()) {
        close(&w.reconstruct(g).unwrap(), &vecf(want), 1e-8, "wossa rec");
    }
}

#[test]
fn gaps() {
    let r = refj();
    let gj = &r["gaps"];
    let x = vecf(&gj["x"]);
    let s = ShapedSsa::fit(&x, 40).unwrap();
    close(&s.sigma[..10], &vecf(&gj["sigma"]), T, "shaped sigma");
    let g: Vec<usize> = (0..5).collect();
    close(&s.reconstruct(&g).unwrap(), &vecf(&gj["rec"]), 1e-8, "shaped rec");
    close(&s.gapfill(&g, false).unwrap(), &vecf(&gj["gapfill"]), 1e-8, "gapfill");
    let ig = igapfill(&x, 40, &[g.clone()], None, 1e-6, 20).unwrap();
    close(&ig[0], &vecf(&gj["igapfill"]), 1e-8, "igapfill");
    let gaps = classify_gaps(&x, 40);
    let want = gj["gaps"].as_array().unwrap();
    assert_eq!(gaps.len(), want.len());
    for (a, w) in gaps.iter().zip(want) {
        assert_eq!(a.left + 1, w["left"].as_u64().unwrap() as usize);
        assert_eq!(a.right + 1, w["right"].as_u64().unwrap() as usize);
        let pos = match a.pos { GapPos::Left => "left", GapPos::Right => "right", GapPos::Internal => "internal" };
        assert_eq!(pos, w["pos"].as_str().unwrap());
        assert_eq!(if a.dense { "dense" } else { "sparse" }, w["kind"].as_str().unwrap());
    }
}

#[test]
fn bootstrap_forecast_is_consistent() {
    let r = refj();
    let m = basic_model(&r);
    let g: Vec<usize> = (0..5).collect();
    let (mean, lo, hi) = m.bforecast(&g, 10, 200, 0.95, false, false, 7).unwrap();
    let point = col0(&m.rforecast(&g, 10, Base::Reconstructed, Direction::Column, false).unwrap());
    for h in 0..10 {
        assert!(lo[h] <= mean[h] && mean[h] <= hi[h]);
        assert!((mean[h] - point[h]).abs() < 0.5 * (hi[h] - lo[h]) + 1e-9);
    }
    let (_, plo, phi) = m.bforecast(&g, 10, 200, 0.95, true, true, 7).unwrap();
    assert!(plo.iter().zip(&phi).all(|(a, b)| a < b));
}

#[test]
fn truncated_toeplitz_and_pssa() {
    let r = refj();
    let t = &r["trunc"];
    let x = vecf(&t["x"]);
    let m = Ssa::new().with_decomposition(ToeplitzSvd { neig: Some(20) }).fit(&x, Some(1000)).unwrap();
    close(&m.sigma, &vecf(&t["ts_sigma"]), 1e-9, "toeplitz lanczos sigma");
    close(&col0(&m.reconstruct_components(&[0, 1, 2, 3]).unwrap()), &vecf(&t["ts_rec"]), 1e-8, "toeplitz lanczos rec");
    let p = Pssa::fit(&x[..2000], 500, &Projector::Degree(1), &Projector::Degree(1), Some(30)).unwrap();
    close(&p.model.sigma[..20], &vecf(&t["ps_sigma"]), 1e-9, "pssa lanczos sigma");
    close(&col0(&p.model.reconstruct_components(&(0..6).collect::<Vec<_>>()).unwrap()), &vecf(&t["ps_rec"]), 1e-8, "pssa lanczos rec");
}
