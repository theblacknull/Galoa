//! Offline ingest: binary / CSV Gold replay and optional synthetic series.
//!
//! Live cTrader mapping lives in `galoa-ctrader` (stub until Open API creds).
//! Canonical offline file: `data/xauusd_m1_5d.bin` (MT5 M1 **bid** bar close).
//! Paper benchmarks: multivariate ETT / Exchange CSV (`ett` module).
//!
//! **Historical (default):** batch log-return via [`closes_to_logret`] /
//! [`convert_close_bin_to_logret`] → `GALOAR1`. Models default to log-return features.
//!
//! **Live trading (later):** [`live`] module — `push_close` / stitch stubs behind
//! [`live::LIVE_INGEST_ENABLED`] (`false`). Eval/backtest CLIs do not call it.

mod binfmt;
mod csv;
mod ett;
pub mod live;
mod load;
mod synthetic;

pub use binfmt::{
    closes_to_logret, convert_close_bin_to_logret, detect_bin_kind, load_bin, load_logret_bin,
    parse_bin, parse_logret_bin, write_bin, write_logret_bin, BinError, BinKind, LogRetSeries,
    MAGIC as BIN_MAGIC, MAGIC_LOGRET,
};
pub use csv::{load_csv, CsvError};
pub use ett::{
    etth_borders, ettm_borders, load_multivariate_csv, percent_borders, EttError, MultivariateSeries,
};
pub use live::{
    live_ingest_enabled, require_live_ingest, IngestBarResult, LogRetStream, ModelInputStore,
    LIVE_INGEST_ENABLED, LIVE_INGEST_FLAG_NAME, MAGIC_LOGRET_APPEND,
};
pub use load::{load_series, LoadError, DEFAULT_GOLD_SERIES};
pub use synthetic::{generate_gold_series, write_csv, SyntheticConfig};
