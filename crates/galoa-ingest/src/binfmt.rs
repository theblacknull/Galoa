//! Compact little-endian price series for fast offline load.
//!
//! Layout (`GALOAP1`):
//! - 8 bytes magic: `GALOAP1\0`
//! - `u64` LE count `n`
//! - `n` × `i64` LE `ts_ns`
//! - `n` × `f64` LE `close`

use galoa_core::{PriceEvent, DEFAULT_SYMBOL};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use thiserror::Error;

pub const MAGIC: &[u8; 8] = b"GALOAP1\0";

#[derive(Debug, Error)]
pub enum BinError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("bad magic (expected GALOAP1)")]
    BadMagic,
    #[error("truncated file (need {need} bytes, have {have})")]
    Truncated { need: usize, have: usize },
    #[error("empty series")]
    Empty,
    #[error("count {0} too large")]
    CountTooLarge(u64),
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
    if buf.len() < 16 {
        return Err(BinError::Truncated {
            need: 16,
            have: buf.len(),
        });
    }
    if &buf[0..8] != MAGIC {
        return Err(BinError::BadMagic);
    }
    let n = u64::from_le_bytes(buf[8..16].try_into().unwrap());
    if n == 0 {
        return Err(BinError::Empty);
    }
    if n > (usize::MAX as u64) / 16 {
        return Err(BinError::CountTooLarge(n));
    }
    let n = n as usize;
    let need = 16 + n * 8 + n * 8;
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

#[cfg(test)]
mod tests {
    use super::*;
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
}
