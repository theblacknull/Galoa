use galoa_core::{PriceEvent, DEFAULT_SYMBOL};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CsvError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse error on line {line}: {msg}")]
    Parse { line: usize, msg: String },
    #[error("empty csv")]
    Empty,
}

/// Load `(ts_ns, close)` CSV. Accepts optional header `ts_ns,close` or `timestamp,close`.
/// If timestamps look like unix seconds (< 1e12), they are scaled to nanoseconds.
pub fn load_csv(path: impl AsRef<Path>) -> Result<Vec<PriceEvent>, CsvError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut events = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if idx == 0 && looks_like_header(trimmed) {
            continue;
        }
        let line_no = idx + 1;
        let (ts_raw, close) = parse_row(trimmed).map_err(|msg| CsvError::Parse {
            line: line_no,
            msg,
        })?;
        let ts_ns = if ts_raw.abs() < 1_000_000_000_000 {
            // treat as seconds
            ts_raw * 1_000_000_000
        } else {
            ts_raw
        };
        events.push(PriceEvent::new(DEFAULT_SYMBOL, ts_ns, close));
    }
    if events.is_empty() {
        return Err(CsvError::Empty);
    }
    events.sort_by_key(|e| e.ts_ns);
    Ok(events)
}

fn looks_like_header(s: &str) -> bool {
    let lower = s.to_ascii_lowercase();
    lower.contains("ts") || lower.contains("time") || lower.contains("close")
}

fn parse_row(s: &str) -> Result<(i64, f64), String> {
    let parts: Vec<&str> = s.split(',').map(str::trim).collect();
    if parts.len() < 2 {
        return Err(format!("expected ts,close got {s:?}"));
    }
    let ts: i64 = parts[0]
        .parse()
        .map_err(|e| format!("bad ts {}: {e}", parts[0]))?;
    let close: f64 = parts[1]
        .parse()
        .map_err(|e| format!("bad close {}: {e}", parts[1]))?;
    Ok((ts, close))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn loads_seconds_csv() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.csv");
        let mut f = File::create(&path).unwrap();
        writeln!(f, "ts_ns,close").unwrap();
        writeln!(f, "1700000000,2650.5").unwrap();
        writeln!(f, "1700000001,2651.0").unwrap();
        let ev = load_csv(&path).unwrap();
        assert_eq!(ev.len(), 2);
        assert_eq!(ev[0].ts_ns, 1_700_000_000_000_000_000);
        assert_eq!(ev[0].close, 2650.5);
    }
}
