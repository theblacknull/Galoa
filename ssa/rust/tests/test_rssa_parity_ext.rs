//! Second round of Rssa parity checks (`rust/benchmarks/make_reference_ext.R`):
//! circular / shaped / nD-SSA, MSSA lists, PSSA forecasts, simultaneous
//! gap filling, CSSA extras, hmatr, forecast and I-OSSA diagnostics.

use serde_json::Value;
use ssa_rs::rssa::cssa::Cssa;
use ssa_rs::rssa::gaps::ShapedSsa;
use ssa_rs::rssa::misc::{hmatr, Interval};
use ssa_rs::rssa::mssa_list::MssaList;
use ssa_rs::rssa::nd::{ball_mask, NdMethod, NdSsaBuilder, Pairing};
use ssa_rs::rssa::pssa::{Projector, Pssa};
use ssa_rs::rssa::*;
use ssa_rs::*;

fn refj() -> Value {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/rssa_ref_ext.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn vecf(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(vecf).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        Value::String(s) if s == "NA" => vec![f64::NAN],
        Value::String(s) => vec![s.trim().parse().unwrap()],
        Value::Bool(b) => vec![*b as i32 as f64],
        Value::Object(o) => o.values().flat_map(vecf).collect(),
        _ => panic!("unexpected {v:?}"),
    }
}

fn list(v: &Value) -> Vec<Vec<f64>> {
    match v {
        Value::Array(a) => a.iter().map(vecf).collect(),
        Value::Object(o) => o.values().map(vecf).collect(),
        _ => panic!(),
    }
}

fn cplx(v: &Value) -> Vec<Complex64> {
    vecf(&v["re"]).into_iter().zip(vecf(&v["im"])).map(|(a, b)| Complex64::new(a, b)).collect()
}

fn close(a: &[f64], b: &[f64], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length {} vs {}", a.len(), b.len());
    let scale = b.iter().filter(|v| v.is_finite()).fold(1.0f64, |m, v| m.max(v.abs()));
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        if y.is_nan() {
            assert!(x.is_nan(), "{what}[{i}]: expected NA, got {x}");
            continue;
        }
        assert!((x - y).abs() <= tol * scale, "{what}[{i}]: {x} vs {y}");
    }
}

fn close_c(a: &[Complex64], b: &[Complex64], tol: f64, what: &str) {
    close(&a.iter().flat_map(|z| [z.re, z.im]).collect::<Vec<_>>(), &b.iter().flat_map(|z| [z.re, z.im]).collect::<Vec<_>>(), tol, what);
}

fn close_roots(a: &[Complex64], b: &[Complex64], tol: f64, what: &str) {
    let key = |z: &Complex64| ((z.re * 1e6).round() as i64, (z.im * 1e6).round() as i64);
    let (mut a, mut b) = (a.to_vec(), b.to_vec());
    a.sort_by_key(key);
    b.sort_by_key(key);
    close_c(&a, &b, tol, what);
}

fn g(r: std::ops::Range<usize>) -> Vec<usize> {
    r.collect()
}

#[test]
fn circular_toeplitz_and_shaped_window_1d() {
    let r = refj();
    let x = vecf(&r["x"]);
    let n = x.len();
    let c = NdSsaBuilder::new(x.clone(), vec![n]).window(vec![50]).circular(vec![true]).method(NdMethod::Full).fit().unwrap();
    close(&c.sigma[..10], &vecf(&r["circ"]["sigma"]), 1e-9, "circular sigma");
    close(&c.reconstruct(&g(0..3)).unwrap(), &vecf(&r["circ"]["rec"]), 1e-8, "circular rec");
    close(&c.rforecast(&g(0..3), 8, true, false).unwrap(), &vecf(&r["circ"]["rf"]), 1e-8, "circular rforecast");
    let t = NdSsaBuilder::new(x.clone(), vec![n]).window(vec![50]).circular(vec![true]).toeplitz(Some(50)).fit().unwrap();
    close(&t.sigma[..10], &vecf(&r["tcirc"]["sigma"]), 1e-9, "toeplitz circular sigma");
    close(&t.reconstruct(&g(0..3)).unwrap(), &vecf(&r["tcirc"]["rec"]), 1e-8, "toeplitz circular rec");
    let wm: Vec<bool> = vecf(&r["wm1"]["wmask"]).into_iter().map(|v| v != 0.0).collect();
    let s = NdSsaBuilder::new(x, vec![n]).wmask(vec![wm.len()], wm).method(NdMethod::Full).fit().unwrap();
    close(&s.sigma[..10], &vecf(&r["wm1"]["sigma"]), 1e-9, "shaped window sigma");
    close(&s.reconstruct(&g(0..4)).unwrap(), &vecf(&r["wm1"]["rec"]), 1e-8, "shaped window rec");
}

#[test]
fn ssa_2d_shapes_topologies_and_variants() {
    let r = refj();
    let f = vecf(&r["c2"]["f"]);
    let dims = vec![24, 30];
    let (bd, bm) = ball_mask(4, 2);
    let s = NdSsaBuilder::new(f.clone(), dims.clone()).wmask(bd, bm).method(NdMethod::Full).fit().unwrap();
    close(&s.sigma[..8], &vecf(&r["c2"]["sigma"]), 1e-9, "circle sigma");
    close(&s.reconstruct(&g(0..4)).unwrap(), &vecf(&r["c2"]["rec"]), 1e-8, "circle rec");

    let s = NdSsaBuilder::new(f.clone(), dims.clone()).window(vec![6, 7]).circular(vec![true, false]).method(NdMethod::Full).fit().unwrap();
    close(&s.sigma[..8], &vecf(&r["c2circ"]["sigma"]), 1e-9, "2d circular sigma");
    close(&s.reconstruct(&g(0..4)).unwrap(), &vecf(&r["c2circ"]["rec"]), 1e-8, "2d circular rec");

    let mut fna = f.clone();
    for i in 9..12 {
        for j in 13..16 {
            fna[i + 24 * j] = f64::NAN;
        }
    }
    fna[0] = f64::NAN;
    let s = NdSsaBuilder::new(fna, dims.clone()).window(vec![6, 7]).method(NdMethod::Full).fit().unwrap();
    close(&s.sigma[..8], &vecf(&r["c2na"]["sigma"]), 1e-9, "2d NA sigma");
    close(&s.reconstruct(&g(0..4)).unwrap(), &vecf(&r["c2na"]["rec"]), 1e-8, "2d NA rec");
    let ig = s.igapfill(&[g(0..6)], None, 1e-6, 10).unwrap();
    close(&ig[0], &vecf(&r["c2na"]["igap"]), 1e-8, "2d igapfill");

    let s = NdSsaBuilder::new(f.clone(), dims.clone())
        .window(vec![6, 7])
        .projectors(Projector::Degree(1), Projector::Degree(1))
        .method(NdMethod::Full)
        .fit()
        .unwrap();
    close(&s.sigma[..8], &vecf(&r["c2p"]["sigma"]), 1e-9, "2d pssa sigma");
    close(&s.reconstruct(&g(0..5)).unwrap(), &vecf(&r["c2p"]["rec"]), 1e-8, "2d pssa rec");

    let s = NdSsaBuilder::new(f.clone(), dims.clone())
        .window(vec![6, 7])
        .oblique(Some(vecf(&r["c2w"]["cw"])), Some(vecf(&r["c2w"]["rw"])))
        .method(NdMethod::Full)
        .fit()
        .unwrap();
    close(&s.sigma[..8], &vecf(&r["c2w"]["sigma"]), 1e-9, "2d wossa sigma");
    close(&s.reconstruct(&g(0..4)).unwrap(), &vecf(&r["c2w"]["rec"]), 1e-8, "2d wossa rec");
}

#[test]
fn esprit_2d_and_3d() {
    let r = refj();
    let f = vecf(&r["c2"]["f"]);
    let s = NdSsaBuilder::new(f, vec![24, 30]).window(vec![6, 7]).method(NdMethod::Full).fit().unwrap();
    let e = &r["c2e"];
    let d = s.parestimate(&g(0..4), Subspace::Column, None, SolveMethod::Ls, Pairing::Diag, None).unwrap();
    close_roots(&d[0].roots, &cplx(&e["d1"]), 1e-7, "2d diag x");
    close_roots(&d[1].roots, &cplx(&e["d2"]), 1e-7, "2d diag y");
    let m = s.parestimate(&g(0..4), Subspace::Column, None, SolveMethod::Ls, Pairing::Memp, None).unwrap();
    close_roots(&m[0].roots, &cplx(&e["m1"]), 1e-7, "2d memp x");
    close_roots(&m[1].roots, &cplx(&e["m2"]), 1e-7, "2d memp y");
    let rw = s.parestimate(&g(0..4), Subspace::Row, None, SolveMethod::Ls, Pairing::Diag, None).unwrap();
    close_roots(&rw[0].roots, &cplx(&e["r1"]), 1e-7, "2d row x");
    close_roots(&rw[1].roots, &cplx(&e["r2"]), 1e-7, "2d row y");

    let c3 = &r["c3"];
    let s3 = NdSsaBuilder::new(vecf(&c3["f"]), vec![10, 9, 8]).window(vec![4, 4, 4]).method(NdMethod::Full).fit().unwrap();
    close(&s3.sigma[..6], &vecf(&c3["sigma"]), 1e-9, "3d sigma");
    close(&s3.reconstruct(&g(0..2)).unwrap(), &vecf(&c3["rec"]), 1e-8, "3d rec");
    for (pairing, key) in [(Pairing::Diag, "p"), (Pairing::Memp, "pm")] {
        let p = s3.parestimate(&g(0..2), Subspace::Column, None, SolveMethod::Ls, pairing, None).unwrap();
        let want: Vec<&Value> = match &c3[key] {
            Value::Array(a) => a.iter().collect(),
            Value::Object(o) => o.values().collect(),
            _ => panic!(),
        };
        for (d, w) in p.iter().zip(want) {
            close_roots(&d.roots, &cplx(w), 1e-7, "3d esprit");
        }
    }
}

#[test]
fn oblique_methods_2d() {
    let r = refj();
    let f = vecf(&r["c2"]["f"]);
    let s = NdSsaBuilder::new(f, vec![24, 30]).window(vec![6, 7]).method(NdMethod::Full).fit().unwrap();
    let o = &r["c2o"];
    let fo = s.fossa(&[g(0..4)], &[-1.0, 1.0], f64::INFINITY, true).unwrap();
    close(&fo.model.sigma[..4], &vecf(&o["fsigma"]), 1e-7, "2d fossa sigma");
    for (i, want) in list(&o["frec"]).iter().enumerate() {
        close(&fo.model.reconstruct(&[i]).unwrap(), want, 1e-7, "2d fossa rec");
    }
    let io = s.iossa(&[vec![0, 1], vec![2, 3]], 1e-8, Some(2.0), 30, 0.5).unwrap();
    let ig: Vec<Vec<usize>> = list(&o["igroups"]).iter().map(|v| v.iter().map(|x| *x as usize - 1).collect()).collect();
    assert_eq!(io.groups, ig);
    for (gr, want) in io.groups.iter().zip(list(&o["irec"])) {
        close(&io.model.reconstruct(gr).unwrap(), &want, 1e-6, "2d iossa rec");
    }
    let eo = s.eossa(&[g(0..4)], 2, SolveMethod::Ls, None).unwrap();
    let eg: Vec<Vec<usize>> = list(&o["egroups"]).iter().map(|v| v.iter().map(|x| *x as usize - 1).collect()).collect();
    let erec = list(&o["erec"]);
    for gr in &eo.groups {
        let wi = eg.iter().position(|w| w == gr).expect("eossa group");
        close(&eo.model.reconstruct(gr).unwrap(), &erec[wi], 1e-7, "2d eossa rec");
    }
    close(&s.cadzow(4, true, 1e-6, 5).unwrap(), &vecf(&o["cadzow"]), 1e-8, "2d cadzow");
}

#[test]
fn mssa_of_different_lengths_and_gaps() {
    let r = refj();
    let ml = &r["ml"];
    let m = MssaList::fit(&[vecf(&ml["a"]), vecf(&ml["b"])], Some(40), NdMethod::Full).unwrap();
    close(&m.nd.sigma[..8], &vecf(&ml["sigma"]), 1e-9, "mssa list sigma");
    let rec = m.reconstruct(&g(0..3)).unwrap();
    for (a, b) in rec.iter().zip(list(&ml["rec"])) {
        close(a, &b, 1e-8, "mssa list rec");
    }
    let cases = [
        ("rfr", m.rforecast(&g(0..3), 7, Base::Reconstructed, Direction::Row).unwrap()),
        ("rfc", m.rforecast(&g(0..3), 7, Base::Reconstructed, Direction::Column).unwrap()),
        ("vfr", m.vforecast(&g(0..3), 7, Direction::Row).unwrap()),
        ("vfc", m.vforecast(&g(0..3), 7, Direction::Column).unwrap()),
    ];
    for (key, got) in cases {
        for (a, b) in got.iter().zip(list(&ml[key])) {
            close(a, &b, 1e-8, key);
        }
    }
    let mn = &r["mn"];
    let m = MssaList::fit(&[vecf(&mn["a"]), vecf(&mn["b"])], Some(40), NdMethod::Full).unwrap();
    close(&m.nd.sigma[..8], &vecf(&mn["sigma"]), 1e-9, "mssa NA sigma");
    let gf: Vec<f64> = m.gapfill(&g(0..4), false).unwrap().concat();
    close(&gf, &vecf(&mn["gap"]), 1e-8, "mssa gapfill");
    let ig: Vec<f64> = m.igapfill(&[g(0..4)], None, 1e-6, 10).unwrap()[0].concat();
    close(&ig, &vecf(&mn["igap"]), 1e-8, "mssa igapfill");
}

#[test]
fn pssa_forecasts_and_simultaneous_gapfill() {
    let r = refj();
    let x = vecf(&r["x"]);
    let p = Pssa::fit(&x, 50, &Projector::Degree(1), &Projector::Degree(1), None).unwrap();
    let pf = &r["pf"];
    close(&p.rforecast(&g(0..5), 10).unwrap(), &vecf(&pf["rf"]), 1e-7, "pssa rforecast");
    close(&p.rforecast_ext(&g(0..5), 10, true, false).unwrap(), &vecf(&pf["rfo"]), 1e-7, "pssa rforecast original");
    close(&p.rforecast_ext(&g(0..5), 10, false, true).unwrap(), &vecf(&pf["rfr"]), 1e-7, "pssa rforecast reverse");
    close(&p.vforecast(&g(0..5), 10).unwrap(), &vecf(&pf["vf"]), 1e-7, "pssa vforecast");
    close(&p.vforecast(&g(1..6), 10).unwrap(), &vecf(&pf["vf2"]), 1e-7, "pssa vforecast (no left special)");

    let xg = vecf(&r["gs"]["x"]);
    let s = ShapedSsa::fit(&xg, 30).unwrap();
    close(&s.gapfill_simultaneous(&g(0..5), false).unwrap(), &vecf(&r["gs"]["sim"]), 1e-8, "simultaneous gapfill");
}

#[test]
fn complex_ssa_extras() {
    let r = refj();
    let cs = &r["cs"];
    let m = Cssa::fit(&cplx(&cs["z"]), 40).unwrap();
    close_roots(&m.parestimate(&[0, 1], false).unwrap().roots, &cplx(&cs["esprit"]), 1e-8, "cssa esprit");
    close_c(&m.vforecast(&[0, 1], 6).unwrap(), &cplx(&cs["vf"]), 1e-8, "cssa vforecast");
    close_c(&m.cadzow(2, true, 1e-6, 5).unwrap(), &cplx(&cs["cz"]), 1e-8, "cssa cadzow");
}

#[test]
fn hmatr_forecast_and_iossa_summary() {
    let r = refj();
    let x = vecf(&r["x"]);
    let h = hmatr(&x, 50, 50, 20, 10).unwrap();
    let flat: Vec<f64> = h.col_iter().flat_map(|c| c.iter().copied().collect::<Vec<_>>()).collect();
    close(&flat, &vecf(&r["hm"]), 1e-8, "hmatr");
    let m = Ssa::new().fit(&x, Some(50)).unwrap();
    let fc = m.forecast(&g(0..5), 6, false, Interval::None, 0.95, 0, 0).unwrap();
    close(&fc.mean, &vecf(&r["fc"]), 1e-8, "forecast mean");
    let groups = vec![vec![1, 2], vec![3, 4]];
    let io = m.iossa(&groups, 1e-8, Some(2.0), 50, 0.5).unwrap();
    let sm = m.iossa_summary(&io, &groups).unwrap();
    close(&sm.tau, &vecf(&r["io1"]["tau"]), 1e-6, "iossa tau");
    close(&sm.initial_tau, &vecf(&r["io1"]["itau"]), 1e-8, "iossa initial tau");
    close(&[sm.cond.0, sm.cond.1], &vecf(&r["io1"]["cond"]), 1e-6, "iossa cond");
}

#[test]
fn nd_lanczos_matches_full_svd() {
    let r = refj();
    let f = vecf(&r["c2"]["f"]);
    let (bd, bm) = ball_mask(4, 2);
    let cases = vec![
        NdSsaBuilder::new(f.clone(), vec![24, 30]).window(vec![6, 7]),
        NdSsaBuilder::new(f.clone(), vec![24, 30]).wmask(bd, bm),
        NdSsaBuilder::new(f.clone(), vec![24, 30]).window(vec![6, 7]).circular(vec![true, true]),
        NdSsaBuilder::new(vecf(&r["c3"]["f"]), vec![10, 9, 8]).window(vec![4, 4, 4]),
    ];
    for b in cases {
        let full = b.clone().method(NdMethod::Full).fit().unwrap();
        let lz = b.method(NdMethod::Lanczos(6)).fit().unwrap();
        close(&lz.sigma, &full.sigma[..6], 1e-10, "nd lanczos sigma");
        close(&lz.reconstruct(&g(0..4)).unwrap(), &full.reconstruct(&g(0..4)).unwrap(), 1e-8, "nd lanczos rec");
    }
}

#[test]
fn rssa_plots_render() {
    use ssa_rs::vis_rssa::*;
    let r = refj();
    let x = vecf(&r["x"]);
    let m = Ssa::new().fit(&x, Some(50)).unwrap();
    let w = m.wcor(Some(&(0..8).map(|i| vec![i]).collect::<Vec<_>>())).unwrap();
    let roots = lrr_roots(&m.lrr(&g(0..5), false).unwrap()).unwrap();
    let h = hmatr(&x, 50, 50, 20, 10).unwrap();
    let pg = m
        .grouping_auto_pgram(&g(0..8), ssa_rs::rssa::grouping::PgramBase::Series, &ssa_rs::rssa::grouping::FreqBins::Equal(3), &[0.0], ssa_rs::rssa::grouping::PgramMethod::Constant, true)
        .unwrap();
    let f2 = NdSsaBuilder::new(vecf(&r["c2"]["f"]), vec![24, 30]).window(vec![6, 7]).method(NdMethod::Full).fit().unwrap();
    let rec = f2.reconstruct(&g(0..2)).unwrap();
    let svgs = vec![
        plot_values(&m.sigma, Some(20)).to_svg(),
        plot_vectors(&m.u, &g(0..6), Some(&m.contributions())).to_svg(),
        plot_vectors(&m.v, &g(0..6), None).to_svg(),
        plot_paired(&m.u, &[(0, 1), (2, 3)]).to_svg(),
        plot_series(&m, &[vec![0], vec![1, 2]]).unwrap().to_svg(),
        plot_reconstruction(&m, &[vec![0], vec![1, 2]]).unwrap().to_svg(),
        plot_wcor(&w).to_svg(),
        plot_roots(&roots, "LRR roots").to_svg(),
        plot_vectors_2d(&f2.u, &g(0..4), (6, 7), None).to_svg(),
        plot_fields(&[("F".into(), ssa_rs::Matrix::from_fn(24, 30, |i, j| rec[i + 24 * j]))]).to_svg(),
        plot_hmatr(&h).to_svg(),
        plot_pgram_contributions(&g(0..8), &pg.contributions).to_svg(),
        plot_wcor_dendrogram(&g(0..8), &w).to_svg(),
    ];
    for s in svgs {
        assert!(s.starts_with("<svg") && s.trim_end().ends_with("</svg>"));
        assert!(!s.contains("NaN") && !s.contains("inf"));
    }
}
