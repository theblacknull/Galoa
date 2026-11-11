//! Port of `test/test_ssa_collection.py` plus extra checks of the Rust-only
//! code paths (FFT vs direct hankelisation, EigenSvd, X_elem on demand ...).

use std::f64::consts::PI;

use faer::Mat;
use ssa_rs::hankel::{diagonal_averaging, Convolver};
use ssa_rs::matrix::{column, mm};
use ssa_rs::*;

fn assert_close(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol + tol * y.abs(), "index {i}: {x} vs {y}");
    }
}

fn col(m: &Matrix, j: usize) -> Vec<f64> {
    m.col(j).iter().copied().collect()
}

fn flat(m: &Matrix) -> Vec<f64> {
    m.col_iter().flat_map(|c| c.iter().copied().collect::<Vec<_>>()).collect()
}

fn arange(n: usize) -> Vec<f64> {
    (0..n).map(|v| v as f64).collect()
}

#[test]
fn test_window_length_default_and_bounds() {
    let odd = Ssa::new().fit(&arange(11), None).unwrap();
    assert_eq!((odd.l, odd.k), (6, 6));
    let even = Ssa::new().fit(&arange(12), None).unwrap();
    assert_eq!((even.l, even.k), (6, 7));
    assert_eq!(Ssa::new().fit(&arange(11), Some(6)).unwrap().l, 6);
    assert!(matches!(Ssa::new().fit(&arange(11), Some(1)), Err(SsaError::InvalidArgument(_))));
    assert!(matches!(Ssa::new().fit(&arange(11), Some(7)), Err(SsaError::InvalidArgument(_))));
    assert!(Ssa::new().fit(&arange(1), None).is_err());
    assert!(Ssa::new().fit(&[], None).is_err());
}

#[test]
fn test_input_shapes() {
    let ssa = Ssa::new().fit(&arange(12), Some(4)).unwrap();
    assert_eq!((ssa.x_s.nrows(), ssa.x_s.ncols()), (12, 1));
    let x = Mat::from_fn(12, 2, |i, j| if j == 0 { i as f64 } else { (i * i) as f64 });
    let mssa = Mssa::new().fit(x.as_ref(), Some(4)).unwrap();
    assert_eq!((mssa.x_s.nrows(), mssa.x_s.ncols()), (12, 2));
    assert_eq!((mssa.x_ss().nrows(), mssa.x_ss().ncols()), (4, 18));
    assert!(Mssa::new().fit(Mat::<f64>::zeros(12, 0).as_ref(), Some(4)).is_err());
}

#[test]
fn test_full_reconstruction_ssa() {
    let x: Vec<f64> = (0..40).map(|t| 2.0 + 0.03 * t as f64 + (2.0 * PI * t as f64 / 10.0).sin()).collect();
    let model = Ssa::new().fit(&x, Some(15)).unwrap();
    let all: Vec<usize> = (0..model.d).collect();
    let rec = model.reconstruct_components(&all).unwrap();
    assert_close(&col(&rec, 0), &x, 1e-10);
}

fn three_series(n: usize) -> Matrix {
    Mat::from_fn(n, 3, |t, j| {
        let t = t as f64;
        match j {
            0 => (2.0 * PI * t / 10.0).sin(),
            1 => 0.5 * (2.0 * PI * t / 8.0).cos(),
            _ => 1.0 + 0.02 * t,
        }
    })
}

#[test]
fn test_full_reconstruction_weighted_mssa() {
    let x = three_series(40);
    let model = Mssa::new()
        .with_weighting(PrecomputedWeights::new(vec![1.0, 0.6, 0.2]))
        .fit(x.as_ref(), Some(15))
        .unwrap();
    let all: Vec<usize> = (0..model.d).collect();
    let rec = model.reconstruct_components(&all).unwrap();
    assert_close(&flat(&rec), &flat(&x), 1e-10);
}

#[test]
fn test_uniform_weights_equal_explicit_ones() {
    let x = Mat::from_fn(30, 3, |i, j| ((i * 31 + j * 17) as f64 * 0.7).sin());
    let a = Mssa::new().fit(x.as_ref(), Some(10)).unwrap();
    let b = Mssa::new()
        .with_weighting(PrecomputedWeights::new(vec![1.0; 3]))
        .fit(x.as_ref(), Some(10))
        .unwrap();
    assert_close(&flat(&a.x_ss()), &flat(&b.x_ss()), 1e-15);
    assert_close(&a.sigma, &b.sigma, 1e-12);
}

#[test]
fn test_hankel_weights() {
    let model = Ssa::new().fit(&arange(10), Some(4)).unwrap();
    assert_eq!(model.construct_hankel_weights(), vec![1, 2, 3, 4, 4, 4, 4, 3, 2, 1]);
}

#[test]
fn test_weighted_correlation_properties() {
    let x = Mat::from_fn(48, 3, |t, j| {
        let t = t as f64;
        match j {
            0 => (2.0 * PI * t / 12.0).sin(),
            1 => (2.0 * PI * t / 12.0).cos(),
            _ => 0.02 * t,
        }
    });
    let model = Mssa::new()
        .with_weighting(PrecomputedWeights::new(vec![1.0, 0.7, 0.3]))
        .fit(x.as_ref(), Some(18))
        .unwrap();
    let w = model.compute_weighted_correlation_matrix();
    for i in 0..model.d {
        assert!((w[(i, i)] - 1.0).abs() < 1e-12);
        for j in 0..model.d {
            assert!((w[(i, j)] - w[(j, i)]).abs() < 1e-12);
            assert!(w[(i, j)] >= -1e-12 && w[(i, j)] <= 1.0 + 1e-12);
        }
    }
}

#[test]
fn test_rank_one_exponential_forecast() {
    let (n, steps, g) = (35usize, 5usize, 1.03f64);
    let x: Vec<f64> = (0..n).map(|t| g.powi(t as i32)).collect();
    let truth: Vec<f64> = (n..n + steps).map(|t| g.powi(t as i32)).collect();
    let model = Ssa::new().fit(&x, Some(12)).unwrap();
    let pl = model.l_forecast(steps, &[0], false).unwrap();
    let pk = model.k_forecast(steps, &[0], false).unwrap();
    assert_close(&col(&pl, 0), &truth, 1e-10);
    assert_close(&col(&pk, 0), &truth, 1e-10);
    // full output keeps the reconstruction in front
    let full = model.l_forecast(steps, &[0], true).unwrap();
    assert_eq!(full.nrows(), n + steps);
}

#[test]
fn test_esprit_recovers_sinusoid_frequency() {
    let w0 = 2.0 * PI / 12.0;
    let x: Vec<f64> = (0..72).map(|t| (w0 * t as f64).sin()).collect();
    let model = Ssa::new().fit(&x, Some(30)).unwrap();
    let (mu, rho, omega) = model.estimate_esprit(&[0, 1]).unwrap();
    assert_eq!(mu.len(), 2);
    let mut a: Vec<f64> = omega.iter().map(|o| o.abs()).collect();
    a.sort_by(f64::total_cmp);
    assert_close(&a, &[w0, w0], 1e-10);
    assert_close(&rho, &[1.0, 1.0], 1e-10);
    // conjugate pair, positive imaginary part first (LAPACK order)
    assert!(mu[0].im > 0.0 && mu[1].im < 0.0);
}

#[test]
fn test_weighting_strategies() {
    let x = Mat::<f64>::zeros(10, 3);
    let d = [0.0, 1.0, 2.0];
    let expected: Vec<f64> = d.iter().map(|v: &f64| (-0.5 * (v / 2.0).powi(2)).exp()).collect();
    let dw = DistanceWeights::new(d.to_vec(), 2.0).compute_weights(&x).unwrap();
    assert_close(&dw, &expected, 1e-15);
    let cw = ClusterAwareWeights::new(d.to_vec(), 0, vec![0, 1, 0], 2.0, 0.25).compute_weights(&x).unwrap();
    assert_close(&cw, &[expected[0], expected[1] * 0.25, expected[2]], 1e-15);
    assert_eq!(UniformWeights.compute_weights(&x).unwrap(), vec![1.0; 3]);
}

#[test]
fn test_invalid_weights_rejected() {
    let x = three_series(20);
    for w in [vec![1.0, 1.0], vec![1.0, 0.0, 1.0], vec![1.0, -1.0, 1.0], vec![1.0, f64::NAN, 1.0]] {
        let r = Mssa::new().with_weighting(PrecomputedWeights::new(w)).fit(x.as_ref(), Some(5));
        assert!(matches!(r, Err(SsaError::InvalidArgument(_))));
    }
}

#[test]
fn test_randomized_svd_reports_retained_energy_below_or_equal_100() {
    let x: Vec<f64> = (0..80).map(|i| ((i * 7919 % 101) as f64 / 50.0 - 1.0) * 1.3).collect();
    let model = Ssa::new()
        .with_decomposition(RandomizedSvd::new(5).with_random_state(0))
        .fit(&x, Some(30))
        .unwrap();
    assert_eq!(model.d, 5);
    let last = *model.cumsum_contr.last().unwrap();
    assert!(last > 0.0 && last <= 100.0 + 1e-10);
    // leading singular values match the exact SVD
    let exact = Ssa::new().fit(&x, Some(30)).unwrap();
    assert!((model.sigma[0] - exact.sigma[0]).abs() < 1e-6 * exact.sigma[0]);
    // sklearn sign rule: max-|.| entry of each U column is positive
    for j in 0..model.d {
        let c = col(&model.u, j);
        let m = c.iter().copied().fold(0.0f64, |a, b| if b.abs() > a.abs() { b } else { a });
        assert!(m > 0.0);
    }
}

#[test]
fn test_randomized_svd_is_exact_on_low_rank_data() {
    let x: Vec<f64> = (0..200).map(|t| (0.2 * t as f64).sin() + 0.5 * (0.05 * t as f64).cos()).collect();
    let exact = Ssa::new().fit(&x, Some(60)).unwrap();
    let rnd = Ssa::new().with_decomposition(RandomizedSvd::new(4).with_random_state(3)).fit(&x, Some(60)).unwrap();
    assert_close(&rnd.sigma, &exact.sigma[..4], 1e-9);
    let a = exact.reconstruct_components(&[0, 1, 2, 3]).unwrap();
    let b = rnd.reconstruct_components(&[0, 1, 2, 3]).unwrap();
    assert_close(&flat(&b), &flat(&a), 1e-8);
}

#[test]
fn test_eigen_svd_matches_full_svd() {
    let x = three_series(120);
    let a = Mssa::new().fit(x.as_ref(), Some(40)).unwrap();
    let b = Mssa::new().with_decomposition(EigenSvd).fit(x.as_ref(), Some(40)).unwrap();
    assert_eq!(a.d, b.d);
    assert_close(&b.sigma[..6], &a.sigma[..6], 1e-9);
    let all: Vec<usize> = (0..b.d).collect();
    assert_close(&flat(&b.reconstruct_components(&all).unwrap()), &flat(&x), 1e-10);
    assert_close(
        &flat(&b.reconstruct_components(&[0, 1, 2]).unwrap()),
        &flat(&a.reconstruct_components(&[0, 1, 2]).unwrap()),
        1e-8,
    );
}

/// Reference implementation of the Python algorithm: materialise X_elem and
/// average anti-diagonals block by block.
fn naive_reconstruct(m: &SsaModel, idx: &[usize]) -> Matrix {
    let mut x = Mat::<f64>::zeros(m.l, m.k * m.s);
    for &i in idx {
        let e = m.elementary_matrix(i).unwrap();
        x = &x + &e;
    }
    let mut out = Mat::<f64>::zeros(m.n, m.s);
    for s in 0..m.s {
        let y = diagonal_averaging(x.as_ref().subcols(s * m.k, m.k));
        for (t, v) in y.into_iter().enumerate() {
            out[(t, s)] = v / m.weights[s].sqrt();
        }
    }
    out
}

#[test]
fn test_fast_reconstruction_equals_naive_algorithm() {
    for (n, l) in [(20, 7), (101, 50), (600, 250)] {
        let x = Mat::from_fn(n, 2, |t, j| ((t * (j + 3)) as f64 * 0.13).sin() + 0.001 * (t * t) as f64);
        let m = Mssa::new()
            .with_weighting(PrecomputedWeights::new(vec![0.8, 1.7]))
            .fit(x.as_ref(), Some(l))
            .unwrap();
        for idx in [vec![0], vec![1, 2], vec![0, 0, 3], (0..m.d).collect::<Vec<_>>()] {
            let fast = m.reconstruct_components(&idx).unwrap();
            let slow = naive_reconstruct(&m, &idx);
            assert_close(&flat(&fast), &flat(&slow), 1e-10);
        }
        let comps = m.reconstruct_elementary_components();
        assert_eq!(comps.len(), m.d);
        for i in [0, m.d / 2, m.d - 1] {
            assert_close(&flat(&comps[i]), &flat(&naive_reconstruct(&m, &[i])), 1e-10);
        }
    }
}

#[test]
fn test_convolver_modes_agree_on_model() {
    let x: Vec<f64> = (0..300).map(|t| (t as f64 * 0.31).sin() * (1.0 + 0.002 * t as f64)).collect();
    let m = Ssa::new().fit(&x, Some(120)).unwrap();
    for conv in [Convolver::direct(m.l, m.k), Convolver::fft(m.l, m.k)] {
        let a = conv.spectrum(m.u.col(0).iter().copied(), m.sigma[0]);
        let b = conv.spectrum(m.v.col(0).iter().copied(), 1.0);
        let mut acc = conv.accumulator();
        conv.add_product(&mut acc, &a, &b);
        let y = conv.finish(acc);
        let e = m.elementary_matrix(0).unwrap();
        let mut sums = diagonal_averaging(e.as_ref());
        for (t, v) in sums.iter_mut().enumerate() {
            *v *= ssa_rs::hankel::anti_diagonal_len(t, m.l, m.k) as f64;
        }
        assert_close(&y, &sums, 1e-10);
    }
}

#[test]
fn test_elementary_matrices_sum_to_trajectory() {
    let x: Vec<f64> = (0..30).map(|t| (t as f64).sqrt()).collect();
    let m = Ssa::new().fit(&x, Some(10)).unwrap();
    let mats = m.elementary_matrices();
    assert_eq!(mats.len(), m.d);
    let mut acc = Mat::<f64>::zeros(m.l, m.k);
    for e in &mats {
        acc = &acc + e;
    }
    assert_close(&flat(&acc), &flat(&m.x_ss()), 1e-12);
    // U diag(S) V^T == X_ss
    let us = Mat::from_fn(m.l, m.d, |i, j| m.u[(i, j)] * m.sigma[j]);
    assert_close(&flat(&mm(us.as_ref(), m.vt().as_ref())), &flat(&m.x_ss()), 1e-12);
}

#[test]
fn test_index_errors() {
    let m = Ssa::new().fit(&arange(20), Some(5)).unwrap();
    assert!(matches!(m.reconstruct_components(&[5]), Err(SsaError::IndexOutOfRange { index: 5, len: 5 })));
    assert!(m.elementary_matrix(9).is_err());
    assert!(m.estimate_esprit(&[7]).is_err());
    assert!(m.l_forecast(3, &[6], true).is_err());
    assert!(m.k_forecast(3, &[6], true).is_err());
}

#[test]
fn test_mssa_forecasts_of_shared_sinusoid() {
    // two channels of the same rank-2 signal: both forecasts must be exact
    let n = 90;
    let f = |t: usize, ph: f64| (2.0 * PI * t as f64 / 15.0 + ph).sin();
    let x = Mat::from_fn(n, 2, |t, j| if j == 0 { f(t, 0.0) } else { 2.0 * f(t, 1.0) });
    let m = Mssa::new().fit(x.as_ref(), Some(30)).unwrap();
    let truth: Vec<Vec<f64>> = (0..2).map(|j| (n..n + 10).map(|t| if j == 0 { f(t, 0.0) } else { 2.0 * f(t, 1.0) }).collect()).collect();
    let pl = m.l_forecast(10, &[0, 1], false).unwrap();
    let pk = m.k_forecast(10, &[0, 1], false).unwrap();
    for j in 0..2 {
        assert_close(&col(&pl, j), &truth[j], 1e-9);
        assert_close(&col(&pk, j), &truth[j], 1e-9);
    }
}

#[test]
fn test_zero_step_forecast() {
    let m = Ssa::new().fit(&arange(20), Some(5)).unwrap();
    assert_eq!(m.l_forecast(0, &[0], false).unwrap().nrows(), 0);
    assert_eq!(m.k_forecast(0, &[0], true).unwrap().nrows(), 20);
}

#[test]
fn test_column_helper_and_trajectory() {
    let x = column(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let t = construct_trajectory_matrix(x.as_ref(), 3, &[4.0]);
    // sqrt(4) * hankel
    assert_eq!(t[(0, 0)], 2.0);
    assert_eq!(t[(2, 2)], 10.0);
    assert_eq!(t[(1, 2)], 8.0);
}

#[test]
fn test_bootstrap_intervals_contain_forecast() {
    let res: Vec<f64> = (0..200).map(|i| ((i * 37 % 41) as f64 - 20.0) / 10.0).collect();
    let fc = vec![1.0, 2.0, 3.0];
    let (lo, hi) = bootstrap_prediction_intervals(&res, &fc, 0.05, 2000, Some(7)).unwrap();
    for h in 0..3 {
        assert!(lo[h] < fc[h] && fc[h] < hi[h]);
        assert!(lo[h] >= fc[h] - 2.0 - 1e-12 && hi[h] <= fc[h] + 2.0 + 1e-12);
    }
}

#[test]
fn test_visualisation_produces_svg() {
    let x: Vec<f64> = (0..100).map(|t| (t as f64 * 0.4).sin() + 0.02 * t as f64).collect();
    let m = Ssa::new().fit(&x, Some(30)).unwrap();
    let c = vis::first_components(&m, 5);
    for svg in [
        vis::plot_eigenvectors(&m, &c, 2).unwrap().to_svg(),
        vis::plot_elementary_matrices(&m, &c, 3).unwrap().to_svg(),
        vis::plot_weighted_correlation(&m).to_svg(),
        vis::plot_contributions(&m, Some(10)).to_svg(),
        vis::plot_esprit_roots(&m, &[0, 1, 2]).unwrap().to_svg(),
    ] {
        assert!(svg.starts_with("<svg") && svg.trim_end().ends_with("</svg>"));
        assert!(!svg.contains("NaN"));
    }
    assert!(vis::plot_eigenvectors(&m, &[99], 2).is_err());
}

#[test]
fn test_dataset_loader() {
    let raw = datasets::load_global_average_temperatures(true).unwrap();
    assert_eq!(raw.len(), 239_177);
    let clean = datasets::load_global_average_temperatures(false).unwrap();
    assert!(clean.len() < raw.len());
    assert!(clean.iter().all(|r| r.average_temperature.is_some()));
    let (dates, temps) = datasets::city_series(&clean, "Berlin");
    assert_eq!(dates.len(), temps.len());
    assert!(temps.len() > 1000);
    let (d2, m) = datasets::cities_matrix(&clean, &["Berlin", "Paris"]);
    assert_eq!(m.nrows(), d2.len());
    assert_eq!(m.ncols(), 2);
}
