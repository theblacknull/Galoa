//! Offline ingest: binary / CSV Gold replay and optional synthetic series.
//!
//! Live cTrader mapping lives in `galoa-ctrader` (stub until Open API creds).
//! Canonical offline file: `data/xauusd_m1_5d.bin` (MT5 M1 **bid** bar close).

mod binfmt;
mod csv;
mod load;
mod synthetic;

pub use binfmt::{load_bin, parse_bin, write_bin, BinError, MAGIC as BIN_MAGIC};
pub use csv::{load_csv, CsvError};
pub use load::{load_series, LoadError, DEFAULT_GOLD_SERIES};
pub use synthetic::{generate_gold_series, write_csv, SyntheticConfig};
