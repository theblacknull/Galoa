use crate::types::PriceEvent;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ResampleError {
    #[error("dt_ns must be positive")]
    InvalidDt,
    #[error("no events to resample")]
    Empty,
}

/// Resample irregular `(ts, close)` events to a regular grid using **last close**
/// in each bucket of width `dt_ns`.
///
/// Buckets are `[floor(ts/dt)*dt, next)`. Empty buckets are skipped (no forward-fill)
/// so the index only sees observed cadence; callers that need a dense grid can
/// forward-fill separately.
pub fn resample_last_close(events: &[PriceEvent], dt_ns: i64) -> Result<Vec<PriceEvent>, ResampleError> {
    if dt_ns <= 0 {
        return Err(ResampleError::InvalidDt);
    }
    if events.is_empty() {
        return Err(ResampleError::Empty);
    }

    let symbol_id = events[0].symbol_id;
    let mut out = Vec::new();
    let mut cur_bucket: Option<i64> = None;
    let mut last_close: Option<f64> = None;

    for ev in events {
        let bucket = ev.ts_ns.div_euclid(dt_ns) * dt_ns;
        match cur_bucket {
            None => {
                cur_bucket = Some(bucket);
                last_close = Some(ev.close);
            }
            Some(b) if b == bucket => {
                last_close = Some(ev.close);
            }
            Some(b) => {
                if let Some(close) = last_close {
                    out.push(PriceEvent::new(symbol_id, b, close));
                }
                cur_bucket = Some(bucket);
                last_close = Some(ev.close);
            }
        }
    }
    if let (Some(b), Some(close)) = (cur_bucket, last_close) {
        out.push(PriceEvent::new(symbol_id, b, close));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DEFAULT_SYMBOL;

    #[test]
    fn last_close_per_bucket() {
        let dt = 1_000_000_000i64; // 1s
        let events = vec![
            PriceEvent::new(DEFAULT_SYMBOL, 100, 10.0),
            PriceEvent::new(DEFAULT_SYMBOL, 500_000_000, 11.0),
            PriceEvent::new(DEFAULT_SYMBOL, 1_100_000_000, 12.0),
        ];
        let r = resample_last_close(&events, dt).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].ts_ns, 0);
        assert_eq!(r[0].close, 11.0);
        assert_eq!(r[1].ts_ns, dt);
        assert_eq!(r[1].close, 12.0);
    }
}
