//! # ssa_rs
//!
//! Rust port of the Python package **py-ssa-lib 2.0.2** (Singular Spectrum
//! Analysis and Multichannel SSA for decomposition, forecasting and
//! signal-root estimation), following the methodology of Golyandina,
//! Nekrutkin & Zhigljavsky.
//!
//! | Python module            | Rust module   |
//! |--------------------------|---------------|
//! | `ssa_collection.py`      | [`ssa`]       |
//! | `svd_strategies.py`      | [`svd`]       |
//! | `weights_strategies.py`  | [`weights`]   |
//! | `stat_tools.py`          | [`stats`]     |
//! | `vis_tools.py`           | [`vis`] (SVG) |
//! | `datasets/`              | [`datasets`]  |
//!
//! ```
//! use ssa_rs::{Ssa, FullSvd};
//! let x: Vec<f64> = (0..100).map(|t| (t as f64 * 0.3).sin() + 0.01 * t as f64).collect();
//! let model = Ssa::new().with_decomposition(FullSvd).fit(&x, Some(40)).unwrap();
//! let trend_and_sine = model.reconstruct_components(&[0, 1, 2]).unwrap();
//! let forecast = model.l_forecast(10, &[0, 1, 2], false).unwrap();
//! assert_eq!(trend_and_sine.nrows(), 100);
//! assert_eq!(forecast.nrows(), 10);
//! ```

pub mod datasets;
pub mod error;
pub mod hankel;
pub mod hmat;
pub mod la;
pub mod lanczos;
pub mod matrix;
pub mod ndfft;
pub mod rssa;
pub mod ssa;
pub mod stats;
pub mod svd;
pub mod vis;
pub mod vis_rssa;
pub mod weights;

pub use error::{Result, SsaError};
pub use hankel::diagonal_averaging;
pub use matrix::Matrix;
pub use ssa::{construct_trajectory_matrix, Complex64, Estimator, Mssa, Ssa, SsaModel};
pub use stats::bootstrap_prediction_intervals;
pub use svd::{
    Decomposition, EigenSvd, FullSvd, LanczosSvd, PowerIterations, RandomizedSvd, SvdStrategy, ToeplitzSvd,
    Trajectory,
};
pub use weights::{
    ClusterAwareWeights, DistanceWeights, PrecomputedWeights, UniformWeights, WeightingStrategy,
};
