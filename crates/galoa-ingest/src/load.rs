//! Load a price series from `.bin` (canonical) or `.csv` (debug).

use crate::binfmt::{load_bin, BinError};
use crate::csv::{load_csv, CsvError};
use galoa_core::PriceEvent;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LoadError {
    #[error(transparent)]
    Bin(#[from] BinError),
    #[error(transparent)]
    Csv(#[from] CsvError),
    #[error("unsupported series extension for {0} (use .bin or .csv)")]
    Unsupported(String),
}

/// Default offline Gold M1 (~5 calendar days) series — MT5 **bid** bar close.
pub const DEFAULT_GOLD_SERIES: &str = "data/xauusd_m1_5d.bin";

/// Load by file extension: `.bin` → `GALOAP1`, `.csv` → `ts_ns,close`.
pub fn load_series(path: impl AsRef<Path>) -> Result<Vec<PriceEvent>, LoadError> {
    let path = path.as_ref();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "bin" => Ok(load_bin(path)?),
        "csv" => Ok(load_csv(path)?),
        _ => Err(LoadError::Unsupported(path.display().to_string())),
    }
}
