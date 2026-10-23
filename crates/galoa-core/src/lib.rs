//! Core types and transforms for Galoa pattern-similarity forecasting.
//!
//! Paths:
//! - **Live / gated MVP:** M1 close → log-returns → STFT magnitude → F → kNN → gates
//! - **SpecReTF offline eval:** STFT frames → JS(amp) + phase cosine + EMA → top-K
//!   aggregate futures (**always FIRE** in phase-1 eval)
//!
//! Time-domain z-norm L2 is **not** a product matching path (baseline contrast only).

mod forecast;
mod gates;
mod resample;
mod returns;
mod specretf;
mod stft;
mod types;
mod znorm;

pub use forecast::{
    ensemble_forecast, weighted_ensemble_forecast, Forecast, ForecastPoint, WeightedForecast,
    WeightedForecastPoint,
};
pub use gates::{evaluate_gates, AbstainReason, GateConfig, MatchDecision};
pub use resample::{resample_last_close, ResampleError};
pub use returns::{log_returns, ReturnsError};
pub use specretf::{
    frame_similarity, js_divergence, phase_cosine, spectral_descriptor, spectral_distance,
    spectral_similarity, SpecReTfError, SpecReTfParams, SpectralDescriptor, SpectralFrame,
    SPECRETF_AMP_WEIGHT, SPECRETF_EMA_ALPHA, SPECRETF_HOP, SPECRETF_N_FFT, SPECRETF_PHASE_WEIGHT,
};
pub use stft::{
    stft_magnitude_features, StftError, StftParams, DEFAULT_HOP, DEFAULT_N_FFT,
};
pub use types::{PriceEvent, SymbolId, DEFAULT_SYMBOL, XAUUSD};
/// Demoted: utilities / offline time-domain baseline only — **not** the matching path.
pub use znorm::{denormalize, z_normalize, ZNormError, ZNormParams};
