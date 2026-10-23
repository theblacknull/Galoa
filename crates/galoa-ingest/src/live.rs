//! Live trading ingest scaffolding — **gated off by default**.
//!
//! # Policy
//!
//! | Mode | What runs |
//! |------|-----------|
//! | **Historical eval / backtest (now)** | Batch only: `closes_to_logret` / `convert_close_bin_to_logret`. No broker poll, no gap-fill. |
//! | **Live trading mode (later)** | Enable [`LIVE_INGEST_ENABLED`], then: download missing history → align/stitch continuous block → stream `push_close` → O(1) log-return → append to model store. |
//!
//! This module implements the O(1) `push_close` / append APIs and stubs for
//! history download + stitch. **Eval/backtest CLIs must not call into this
//! module** until live mode is turned on. Gate check: [`require_live_ingest`].
//!
//! Core math is shared with batch: [`galoa_core::log_return_step`].

use crate::binfmt::{closes_to_logret, write_bin, write_logret_bin, BinError, LogRetSeries};
use galoa_core::{log_return_step, PriceEvent, DEFAULT_SYMBOL};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Master gate for live broker → model-store streaming.
///
/// `false` (default): historical testing — batch convert only.
/// Flip to `true` only when enabling live trading mode (later).
pub const LIVE_INGEST_ENABLED: bool = false;

/// Cargo-style name for docs / JSON (`LiveIngest` feature conceptually off).
pub const LIVE_INGEST_FLAG_NAME: &str = "LiveIngest";

/// Returns `Ok(())` only when [`LIVE_INGEST_ENABLED`] is true.
pub fn require_live_ingest() -> Result<(), BinError> {
    if LIVE_INGEST_ENABLED {
        Ok(())
    } else {
        Err(BinError::LiveIngestDisabled)
    }
}

/// Whether live streaming paths may run (compile-time constant today).
#[inline]
pub fn live_ingest_enabled() -> bool {
    LIVE_INGEST_ENABLED
}

/// Streaming append state: keep last close; each new M1 bar → one log-return.
///
/// **Live-mode only.** Prefer [`ModelInputStore`] when also mirroring closes for PnL.
#[derive(Clone, Debug)]
pub struct LogRetStream {
    last_close: Option<f64>,
    pub series: LogRetSeries,
}

impl LogRetStream {
    pub fn new() -> Self {
        Self {
            last_close: None,
            series: LogRetSeries::default(),
        }
    }

    pub fn from_closes(events: &[PriceEvent]) -> Result<Self, BinError> {
        let series = if events.len() >= 2 {
            closes_to_logret(events)?
        } else {
            LogRetSeries::default()
        };
        Ok(Self {
            last_close: events.last().map(|e| e.close),
            series,
        })
    }

    /// O(1): one new close → one log-return append. No full-series recompute.
    pub fn push_close(&mut self, ts_ns: i64, close: f64) -> Result<Option<f64>, BinError> {
        require_live_ingest()?;
        self.push_close_ungated(ts_ns, close)
    }

    /// Library helper used by unit tests and by [`ModelInputStore`] after the gate.
    pub(crate) fn push_close_ungated(
        &mut self,
        ts_ns: i64,
        close: f64,
    ) -> Result<Option<f64>, BinError> {
        match self.last_close {
            None => {
                self.last_close = Some(close);
                Ok(None)
            }
            Some(prev) => {
                let r = log_return_step(prev, close)?;
                self.series.ts_ns.push(ts_ns);
                self.series.values.push(r);
                self.last_close = Some(close);
                Ok(Some(r))
            }
        }
    }

    pub fn last_close(&self) -> Option<f64> {
        self.last_close
    }
}

impl Default for LogRetStream {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of ingesting one M1 close into [`ModelInputStore`].
#[derive(Clone, Debug)]
pub struct IngestBarResult {
    pub log_return: Option<f64>,
    pub close_len: usize,
    pub logret_len: usize,
}

/// Live model-input store: closes (PnL) + log-returns (primary feature).
///
/// Activated only when [`LIVE_INGEST_ENABLED`]. Historical eval loads `GALOAP1`
/// and converts in batch — it does **not** use this store.
#[derive(Clone, Debug)]
pub struct ModelInputStore {
    pub closes: Vec<PriceEvent>,
    pub logrets: LogRetSeries,
    last_close: Option<f64>,
}

impl ModelInputStore {
    pub fn new() -> Self {
        Self {
            closes: Vec::new(),
            logrets: LogRetSeries::default(),
            last_close: None,
        }
    }

    /// Seed from historical closes (batch convert once). Safe without live gate.
    pub fn from_closes(events: &[PriceEvent]) -> Result<Self, BinError> {
        let logrets = if events.len() >= 2 {
            closes_to_logret(events)?
        } else {
            LogRetSeries::default()
        };
        Ok(Self {
            closes: events.to_vec(),
            last_close: events.last().map(|e| e.close),
            logrets,
        })
    }

    pub fn last_close(&self) -> Option<f64> {
        self.last_close
    }

    pub fn feature_series(&self) -> &[f64] {
        &self.logrets.values
    }

    pub fn close_series(&self) -> Vec<f64> {
        self.closes.iter().map(|e| e.close).collect()
    }

    /// Live ingest alias of [`push_close`](Self::push_close). Gated.
    #[inline]
    pub fn ingest_bar(&mut self, ts_ns: i64, close: f64) -> Result<IngestBarResult, BinError> {
        self.push_close(ts_ns, close)
    }

    /// `push_close(ts, close) -> Option`-style result via [`IngestBarResult`].
    ///
    /// **Requires** [`LIVE_INGEST_ENABLED`]. First bar seeds only; later bars O(1).
    pub fn push_close(&mut self, ts_ns: i64, close: f64) -> Result<IngestBarResult, BinError> {
        require_live_ingest()?;
        self.push_close_ungated(ts_ns, close)
    }

    pub(crate) fn push_close_ungated(
        &mut self,
        ts_ns: i64,
        close: f64,
    ) -> Result<IngestBarResult, BinError> {
        self.closes
            .push(PriceEvent::new(DEFAULT_SYMBOL, ts_ns, close));
        let log_return = match self.last_close {
            None => {
                self.last_close = Some(close);
                None
            }
            Some(prev) => {
                let r = log_return_step(prev, close)?;
                self.logrets.ts_ns.push(ts_ns);
                self.logrets.values.push(r);
                self.last_close = Some(close);
                Some(r)
            }
        };
        Ok(IngestBarResult {
            log_return,
            close_len: self.closes.len(),
            logret_len: self.logrets.len(),
        })
    }

    pub fn write_bins(
        &self,
        close_path: impl AsRef<Path>,
        logret_path: impl AsRef<Path>,
    ) -> Result<(), BinError> {
        if !self.closes.is_empty() {
            write_bin(close_path, &self.closes)?;
        }
        if !self.logrets.is_empty() {
            write_logret_bin(logret_path, &self.logrets)?;
        }
        Ok(())
    }
}

impl Default for ModelInputStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Append-log magic `GALOARA\0` (O(1) disk append sidecar for live mode).
pub const MAGIC_LOGRET_APPEND: &[u8; 8] = b"GALOARA\0";

pub fn init_logret_append_log(path: impl AsRef<Path>) -> Result<(), BinError> {
    require_live_ingest()?;
    let mut file = File::create(path)?;
    file.write_all(MAGIC_LOGRET_APPEND)?;
    file.write_all(&0u64.to_le_bytes())?;
    Ok(())
}

pub fn append_logret_record(
    path: impl AsRef<Path>,
    ts_ns: i64,
    value: f64,
) -> Result<(), BinError> {
    require_live_ingest()?;
    append_logret_record_ungated(path, ts_ns, value)
}

pub fn append_store_last_logret(
    store: &ModelInputStore,
    append_log_path: impl AsRef<Path>,
) -> Result<(), BinError> {
    require_live_ingest()?;
    if store.logrets.is_empty() {
        return Err(BinError::Empty);
    }
    let i = store.logrets.len() - 1;
    append_logret_record_ungated(append_log_path, store.logrets.ts_ns[i], store.logrets.values[i])
}

fn append_logret_record_ungated(
    path: impl AsRef<Path>,
    ts_ns: i64,
    value: f64,
) -> Result<(), BinError> {
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)?;
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)?;
    if &magic != MAGIC_LOGRET_APPEND {
        return Err(BinError::BadMagic);
    }
    let mut count_buf = [0u8; 8];
    file.read_exact(&mut count_buf)?;
    let n = u64::from_le_bytes(count_buf).wrapping_add(1);
    file.seek(SeekFrom::Start(8))?;
    file.write_all(&n.to_le_bytes())?;
    file.seek(SeekFrom::End(0))?;
    file.write_all(&ts_ns.to_le_bytes())?;
    file.write_all(&value.to_le_bytes())?;
    Ok(())
}

/// Stub: when live mode starts, download bars missing between local last ts and broker now.
///
/// **Not implemented** in this PR — returns [`BinError::LiveIngestDisabled`] while gated,
/// or a placeholder error once enabled until broker sync lands.
pub fn download_missing_history_stub(
    _symbol: &str,
    _from_ts_ns: i64,
    _to_ts_ns: i64,
) -> Result<Vec<PriceEvent>, BinError> {
    require_live_ingest()?;
    Err(BinError::Io(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "download_missing_history not implemented — enable live trading mode later",
    )))
}

/// Stub: align/stitch downloaded + local closes into one continuous block (no gaps/dupes).
pub fn align_stitch_closes_stub(
    _local: &[PriceEvent],
    _fetched: &[PriceEvent],
) -> Result<Vec<PriceEvent>, BinError> {
    require_live_ingest()?;
    Err(BinError::Io(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "align_stitch_closes not implemented — enable live trading mode later",
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use tempfile::tempdir;

    #[test]
    fn live_gate_off_by_default() {
        assert!(!live_ingest_enabled());
        assert!(matches!(
            require_live_ingest(),
            Err(BinError::LiveIngestDisabled)
        ));
        let mut store = ModelInputStore::new();
        assert!(matches!(
            store.push_close(1, 100.0),
            Err(BinError::LiveIngestDisabled)
        ));
    }

    #[test]
    fn ungated_push_matches_batch() {
        let events = vec![
            PriceEvent::new(DEFAULT_SYMBOL, 1, 2000.0),
            PriceEvent::new(DEFAULT_SYMBOL, 2, 2001.0),
            PriceEvent::new(DEFAULT_SYMBOL, 3, 1999.0),
        ];
        let batch = closes_to_logret(&events).unwrap();
        let mut store = ModelInputStore::new();
        for e in &events {
            store.push_close_ungated(e.ts_ns, e.close).unwrap();
        }
        assert_eq!(store.logrets.values, batch.values);
        assert_abs_diff_eq!(
            store.feature_series()[0],
            (2001.0 / 2000.0f64).ln(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn from_closes_seed_no_gate() {
        let events = vec![
            PriceEvent::new(DEFAULT_SYMBOL, 1, 100.0),
            PriceEvent::new(DEFAULT_SYMBOL, 2, 110.0),
        ];
        let store = ModelInputStore::from_closes(&events).unwrap();
        assert_eq!(store.feature_series().len(), 1);
        let dir = tempdir().unwrap();
        store
            .write_bins(dir.path().join("c.bin"), dir.path().join("r.bin"))
            .unwrap();
    }
}
