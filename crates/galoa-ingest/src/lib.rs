//! Offline ingest: CSV replay and synthetic XAUUSD close series.
//!
//! Live cTrader mapping lives in `galoa-ctrader` (stub until Open API creds).

mod csv;
mod synthetic;

pub use csv::{load_csv, CsvError};
pub use synthetic::{generate_gold_series, write_csv, SyntheticConfig};
