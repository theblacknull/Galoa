//! Core types and transforms for Galoa pattern-similarity forecasting.
//!
//! Hot-path contract: one scalar price per event (`close`), multi-cadence via
//! last-close resample, z-normalized windows, analogue ensemble → forecast.

mod forecast;
mod resample;
mod types;
mod znorm;

pub use forecast::{ensemble_forecast, Forecast, ForecastPoint};
pub use resample::{resample_last_close, ResampleError};
pub use types::{PriceEvent, SymbolId, DEFAULT_SYMBOL, XAUUSD};
pub use znorm::{denormalize, z_normalize, ZNormError, ZNormParams};
