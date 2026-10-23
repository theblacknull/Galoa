//! Core types and transforms for Galoa pattern-similarity forecasting.
//!
//! Matching path (spectral only): M1 close → log-returns → STFT magnitude →
//! feature vector F → exact kNN → selective gates → ABSTAIN | FIRE.
//!
//! Time-domain z-norm L2 shape matching is **not** used.

mod forecast;
mod gates;
mod resample;
mod returns;
mod stft;
mod types;
mod znorm;

pub use forecast::{ensemble_forecast, Forecast, ForecastPoint};
pub use gates::{evaluate_gates, AbstainReason, GateConfig, MatchDecision};
pub use resample::{resample_last_close, ResampleError};
pub use returns::{log_returns, ReturnsError};
pub use stft::{
    stft_magnitude_features, StftError, StftParams, DEFAULT_HOP, DEFAULT_N_FFT,
};
pub use types::{PriceEvent, SymbolId, DEFAULT_SYMBOL, XAUUSD};
/// Demoted: kept for utilities/tests only — **not** the matching path.
pub use znorm::{denormalize, z_normalize, ZNormError, ZNormParams};
