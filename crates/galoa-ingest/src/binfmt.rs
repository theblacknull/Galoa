//! Compact little-endian series for fast offline load / batch convert.
//!
//! ## `GALOAP1` — close prices
//! - 8 bytes magic: `GALOAP1\0`
//! - `u64` LE count `n`
//! - `n` × `i64` LE `ts_ns`
//! - `n` × `f64` LE `close`
//!
//! ## `GALOAR1` — log-returns (primary model feature input)
//! - 8 bytes magic: `GALOAR1\0`
//! - `u64` LE count `m` (= N−1 closes)
//! - `m` × `i64` LE `ts_ns` — timestamp of the **ending** bar (bar i+1)
//! - `m` × `f64` LE `r` where `r[i] = ln(close[i+1]/close[i])`
//!
//! Same layout / speed class as `GALOAP1`. Convert uses `galoa_core::log_returns_into`.
//!
//! Live streaming (`push_close`) lives in [`crate::live`] — **gated off** until live trading mode.

use galoa_core::{log_returns_into, PriceEvent, ReturnsError, DEFAULT_SYMBOL};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use thiserror::Error;

/// Close-series magic (`GALOAP1\0`).
pub const MAGIC: &[u8; 8] = b"GALOAP1\0";
/// Log-return series magic (`GALOAR1\0`).
pub const MAGIC_LOGRET: &[u8; 8] = b"GALOAR1\0";

#[derive(Debug, Error)]
pub enum BinError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("bad magic (expected GALOAP1 or GALOAR1)")]
    BadMagic,
    #[error("truncated file (need {need} bytes, have {have})")]
    Truncated { need: usize, have: usize },
    #[error("empty series")]
    Empty,
    #[error("count {0} too large")]
    CountTooLarge(u64),
    #[error(transparent)]
    Returns(#[from] ReturnsError),
    #[error("expected GALOAP1 close series, got GALOAR1")]
    ExpectedClose,
    #[error("expected GALOAR1 log-return series, got GALOAP1")]
    ExpectedLogRet,
    #[error("live ingest is disabled (LiveIngest gate off)")]
    LiveIngestDisabled,
}

/// Detected on-disk series kind from magic bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinKind {
    Close,
    LogReturn,
}

/// Contiguous log-return series (aligned to ending-bar timestamps).
#[derive(Clone, Debug, Default)]
pub struct LogRetSeries {
    pub ts_ns: Vec<i64>,
    pub values: Vec<f64>,
}

impl LogRetSeries {
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

fn parse_header(buf: &[u8]) -> Result<(BinKind, usize), BinError> {
    if buf.len() < 16 {
        return Err(BinError::Truncated {
            need: 16,
            have: buf.len(),
        });
    }
    let kind = if &buf[0..8] == MAGIC {
        BinKind::Close
    } else if &buf[0..8] == MAGIC_LOGRET {
        BinKind::LogReturn
    } else {
        return Err(BinError::BadMagic);
    };
    let n = u64::from_le_bytes(buf[8..16].try_into().unwrap());
    if n == 0 {
        return Err(BinError::Empty);
    }
    if n > (usize::MAX as u64) / 16 {
        return Err(BinError::CountTooLarge(n));
    }
    Ok((kind, n as usize))
}

fn need_bytes(n: usize) -> usize {
    16 + n * 8 + n * 8
}

/// Peek magic without full parse.
pub fn detect_bin_kind(path: impl AsRef<Path>) -> Result<BinKind, BinError> {
    let mut file = File::open(path)?;
    let mut hdr = [0u8; 8];
    file.read_exact(&mut hdr)?;
    if &hdr == MAGIC {
        Ok(BinKind::Close)
    } else if &hdr == MAGIC_LOGRET {
        Ok(BinKind::LogReturn)
    } else {
        Err(BinError::BadMagic)
    }
}

/// Load a `GALOAP1` binary series.
pub fn load_bin(path: impl AsRef<Path>) -> Result<Vec<PriceEvent>, BinError> {
    let mut file = File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    parse_bin(&buf)
}

/// Parse `GALOAP1` bytes into price events (sorted by `ts_ns`).
pub fn parse_bin(buf: &[u8]) -> Result<Vec<PriceEvent>, BinError> {
    let (kind, n) = parse_header(buf)?;
    if kind != BinKind::Close {
        return Err(BinError::ExpectedClose);
    }
    let need = need_bytes(n);
    if buf.len() < need {
        return Err(BinError::Truncated {
            need,
            have: buf.len(),
        });
    }
    let mut events = Vec::with_capacity(n);
    let ts_off = 16;
    let close_off = 16 + n * 8;
    for i in 0..n {
        let ts = i64::from_le_bytes(buf[ts_off + i * 8..ts_off + i * 8 + 8].try_into().unwrap());
        let close =
            f64::from_le_bytes(buf[close_off + i * 8..close_off + i * 8 + 8].try_into().unwrap());
        events.push(PriceEvent::new(DEFAULT_SYMBOL, ts, close));
    }
    events.sort_by_key(|e| e.ts_ns);
    Ok(events)
}

/// Write events as `GALOAP1` binary.
pub fn write_bin(path: impl AsRef<Path>, events: &[PriceEvent]) -> Result<(), BinError> {
    if events.is_empty() {
        return Err(BinError::Empty);
    }
    let mut file = File::create(path)?;
    file.write_all(MAGIC)?;
    file.write_all(&(events.len() as u64).to_le_bytes())?;
    for e in events {
        file.write_all(&e.ts_ns.to_le_bytes())?;
    }
    for e in events {
        file.write_all(&e.close.to_le_bytes())?;
    }
    Ok(())
}

/// Load a `GALOAR1` log-return binary.
pub fn load_logret_bin(path: impl AsRef<Path>) -> Result<LogRetSeries, BinError> {
    let mut file = File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    parse_logret_bin(&buf)
}

/// Parse `GALOAR1` bytes.
pub fn parse_logret_bin(buf: &[u8]) -> Result<LogRetSeries, BinError> {
    let (kind, n) = parse_header(buf)?;
    if kind != BinKind::LogReturn {
        return Err(BinError::ExpectedLogRet);
    }
    let need = need_bytes(n);
    if buf.len() < need {
        return Err(BinError::Truncated {
            need,
            have: buf.len(),
        });
    }
    let mut ts_ns = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);
    let ts_off = 16;
    let val_off = 16 + n * 8;
    for i in 0..n {
        ts_ns.push(i64::from_le_bytes(
            buf[ts_off + i * 8..ts_off + i * 8 + 8].try_into().unwrap(),
        ));
        values.push(f64::from_le_bytes(
            buf[val_off + i * 8..val_off + i * 8 + 8].try_into().unwrap(),
        ));
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| ts_ns[i]);
    if order.windows(2).any(|w| w[0] + 1 != w[1]) {
        let mut ts2 = Vec::with_capacity(n);
        let mut v2 = Vec::with_capacity(n);
        for i in order {
            ts2.push(ts_ns[i]);
            v2.push(values[i]);
        }
        return Ok(LogRetSeries {
            ts_ns: ts2,
            values: v2,
        });
    }
    Ok(LogRetSeries { ts_ns, values })
}

/// Write a `GALOAR1` log-return series.
pub fn write_logret_bin(path: impl AsRef<Path>, series: &LogRetSeries) -> Result<(), BinError> {
    if series.is_empty() || series.ts_ns.len() != series.values.len() {
        return Err(BinError::Empty);
    }
    let mut file = File::create(path)?;
    file.write_all(MAGIC_LOGRET)?;
    file.write_all(&(series.len() as u64).to_le_bytes())?;
    for t in &series.ts_ns {
        file.write_all(&t.to_le_bytes())?;
    }
    for v in &series.values {
        file.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

/// Offline / historical batch: closes → log-returns (in memory).
///
/// Alignment: `values[i] = ln(close[i+1]/close[i])`, `ts_ns[i] = close_ts[i+1]`.
pub fn closes_to_logret(events: &[PriceEvent]) -> Result<LogRetSeries, BinError> {
    if events.len() < 2 {
        return Err(BinError::Returns(ReturnsError::TooShort(events.len())));
    }
    let closes: Vec<f64> = events.iter().map(|e| e.close).collect();
    let mut values = vec![0.0; closes.len() - 1];
    log_returns_into(&closes, &mut values)?;
    let ts_ns: Vec<i64> = events[1..].iter().map(|e| e.ts_ns).collect();
    Ok(LogRetSeries { ts_ns, values })
}

/// Offline converter: `GALOAP1` closes.bin → `GALOAR1` logret.bin.
pub fn convert_close_bin_to_logret(
    from: impl AsRef<Path>,
    to: impl AsRef<Path>,
) -> Result<LogRetSeries, BinError> {
    let events = load_bin(from)?;
    let series = closes_to_logret(&events)?;
    write_logret_bin(to, &series)?;
    Ok(series)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::time::Instant;
    use tempfile::tempdir;

    #[test]
    fn roundtrip_bin() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("x.bin");
        let events = vec![
            PriceEvent::new(DEFAULT_SYMBOL, 1_000_000_000, 2650.5),
            PriceEvent::new(DEFAULT_SYMBOL, 2_000_000_000, 2651.0),
        ];
        write_bin(&path, &events).unwrap();
        let loaded = load_bin(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].ts_ns, 1_000_000_000);
        assert_eq!(loaded[0].close, 2650.5);
        assert_eq!(loaded[1].close, 2651.0);
    }

    #[test]
    fn roundtrip_logret_and_convert() {
        let dir = tempdir().unwrap();
        let close_path = dir.path().join("c.bin");
        let ret_path = dir.path().join("r.bin");
        let events = vec![
            PriceEvent::new(DEFAULT_SYMBOL, 1_000_000_000, 100.0),
            PriceEvent::new(DEFAULT_SYMBOL, 2_000_000_000, 110.0),
            PriceEvent::new(DEFAULT_SYMBOL, 3_000_000_000, 99.0),
        ];
        write_bin(&close_path, &events).unwrap();
        let series = convert_close_bin_to_logret(&close_path, &ret_path).unwrap();
        assert_eq!(series.len(), 2);
        assert_eq!(series.ts_ns[0], 2_000_000_000);
        assert_eq!(series.ts_ns[1], 3_000_000_000);
        assert_abs_diff_eq!(series.values[0], (1.1f64).ln(), epsilon = 1e-12);

        assert_eq!(detect_bin_kind(&ret_path).unwrap(), BinKind::LogReturn);
        let loaded = load_logret_bin(&ret_path).unwrap();
        assert_eq!(loaded.values, series.values);
    }

    #[test]
    fn microbench_load_close_vs_logret_same_order() {
        let dir = tempdir().unwrap();
        let n = 20_000usize;
        let events: Vec<_> = (0..n)
            .map(|i| {
                PriceEvent::new(
                    DEFAULT_SYMBOL,
                    i as i64 * 60_000_000_000,
                    2650.0 * (1.0 + 1e-6 * (i as f64).sin()),
                )
            })
            .collect();
        let close_path = dir.path().join("c.bin");
        let ret_path = dir.path().join("r.bin");
        write_bin(&close_path, &events).unwrap();
        convert_close_bin_to_logret(&close_path, &ret_path).unwrap();

        let t0 = Instant::now();
        let _c = load_bin(&close_path).unwrap();
        let dt_c = t0.elapsed();

        let t1 = Instant::now();
        let _r = load_logret_bin(&ret_path).unwrap();
        let dt_r = t1.elapsed();

        let ratio = dt_r.as_nanos() as f64 / dt_c.as_nanos().max(1) as f64;
        assert!(
            ratio < 5.0,
            "GALOAR1 load not same class as GALOAP1: close={dt_c:?} logret={dt_r:?} ratio={ratio:.2}"
        );
    }
}
