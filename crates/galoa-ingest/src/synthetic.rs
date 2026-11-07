use galoa_core::{PriceEvent, DEFAULT_SYMBOL};
use std::io::Write;
use std::path::Path;

/// Parameters for a reproducible synthetic XAUUSD close series.
#[derive(Clone, Debug)]
pub struct SyntheticConfig {
    /// Number of bars to emit.
    pub bars: usize,
    /// Spacing between bars in nanoseconds (default 60s).
    pub dt_ns: i64,
    /// Starting unix seconds.
    pub start_ts_s: i64,
    /// Starting gold-like price.
    pub start_price: f64,
    /// RNG seed for reproducibility.
    pub seed: u64,
    /// Window length used when planting a repeated pattern.
    pub plant_window: usize,
    /// Horizon length planted after the early pattern.
    pub plant_horizon: usize,
}

impl Default for SyntheticConfig {
    fn default() -> Self {
        Self {
            bars: 2_048,
            dt_ns: 60_000_000_000, // 1 minute
            start_ts_s: 1_700_000_000,
            start_price: 2_650.0,
            seed: 42,
            plant_window: 64,
            plant_horizon: 16,
        }
    }
}

fn next_u64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn next_f64(state: &mut u64) -> f64 {
    (next_u64(state) as f64) / (u64::MAX as f64)
}

fn make_pattern(window: usize, base: f64) -> Vec<f64> {
    (0..window)
        .map(|i| {
            let t = i as f64 / window as f64;
            base + 8.0 * (t * std::f64::consts::TAU).sin() + t * 3.0
        })
        .collect()
}

/// Generate a random-walk Gold series with one planted similar segment near the end.
///
/// Early in the series we embed a pattern of length `plant_window` followed by a
/// known continuation of length `plant_horizon`. The same shape (level-shifted) is
/// placed as the final window so kNN can recover the planted future in smoke tests.
pub fn generate_gold_series(cfg: &SyntheticConfig) -> Vec<PriceEvent> {
    let mut rng = cfg.seed;
    let mut closes = Vec::with_capacity(cfg.bars);
    let mut price = cfg.start_price;

    for _ in 0..cfg.bars {
        let shock = (next_f64(&mut rng) - 0.5) * 0.8;
        price = (price + shock).max(100.0);
        closes.push(price);
    }

    let w = cfg.plant_window;
    let h = cfg.plant_horizon;
    let need = w + h + w + 16;
    if cfg.bars >= need {
        let pattern = make_pattern(w, 2_640.0);
        let last = *pattern.last().unwrap();
        let future: Vec<f64> = (0..h).map(|i| last + 0.4 * (i as f64 + 1.0)).collect();

        let early = 128usize.min(cfg.bars / 4);
        for (i, v) in pattern.iter().chain(future.iter()).enumerate() {
            closes[early + i] = *v;
        }

        let level_shift = 12.0;
        let late = cfg.bars - w;
        for (i, v) in pattern.iter().enumerate() {
            closes[late + i] = *v + level_shift;
        }
    }

    closes
        .into_iter()
        .enumerate()
        .map(|(i, close)| {
            PriceEvent::new(
                DEFAULT_SYMBOL,
                cfg.start_ts_s * 1_000_000_000 + i as i64 * cfg.dt_ns,
                close,
            )
        })
        .collect()
}

/// Write series as `ts_ns,close` CSV.
pub fn write_csv(path: impl AsRef<Path>, events: &[PriceEvent]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "ts_ns,close")?;
    for e in events {
        writeln!(f, "{},{}", e.ts_ns, e.close)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_expected_length() {
        let cfg = SyntheticConfig {
            bars: 500,
            ..Default::default()
        };
        let s = generate_gold_series(&cfg);
        assert_eq!(s.len(), 500);
        assert!(s[0].close > 100.0);
        assert!(s.windows(2).all(|w| w[1].ts_ns > w[0].ts_ns));
    }
}
