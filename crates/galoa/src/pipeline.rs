use anyhow::{bail, Context, Result};
use galoa_core::{resample_last_close, XAUUSD};
use galoa_index::{ForecastResult, WindowIndex};
use galoa_ingest::{generate_gold_series, load_series, SyntheticConfig, DEFAULT_GOLD_SERIES};
use serde::Serialize;
use std::path::PathBuf;

#[derive(Serialize)]
pub struct ForecastJson {
    pub symbol: String,
    pub window: usize,
    pub horizon: usize,
    pub k_requested: usize,
    pub k_used: usize,
    pub query_last: f64,
    pub neighbors: Vec<NeighborJson>,
    pub points: Vec<PointJson>,
}

#[derive(Serialize)]
pub struct NeighborJson {
    pub end_idx: usize,
    pub distance: f64,
}

#[derive(Serialize)]
pub struct PointJson {
    pub step: usize,
    pub median: f64,
    pub p10: f64,
    pub p90: f64,
}

pub fn run_forecast(
    data: Option<PathBuf>,
    synthetic: bool,
    window: usize,
    horizon: usize,
    k: usize,
    dt_secs: i64,
) -> Result<ForecastJson> {
    let mut events = if synthetic {
        if let Some(ref p) = data {
            tracing::warn!("--synthetic set; ignoring --data {}", p.display());
        }
        generate_gold_series(&SyntheticConfig::default())
    } else {
        let path = data.unwrap_or_else(|| PathBuf::from(DEFAULT_GOLD_SERIES));
        load_series(&path).with_context(|| format!("load {}", path.display()))?
    };

    if dt_secs > 0 {
        let dt_ns = dt_secs
            .checked_mul(1_000_000_000)
            .context("dt_secs overflow")?;
        events = resample_last_close(&events, dt_ns)?;
    }

    if events.is_empty() {
        bail!("no price events after load/resample");
    }

    let index = WindowIndex::from_events(&events, window, horizon)?;
    let result = index.forecast_latest(k)?;
    Ok(to_json(&result, window, horizon, k))
}

fn to_json(result: &ForecastResult, window: usize, horizon: usize, k: usize) -> ForecastJson {
    ForecastJson {
        symbol: XAUUSD.to_string(),
        window,
        horizon,
        k_requested: k,
        k_used: result.forecast.k_used,
        query_last: result.query_last,
        neighbors: result
            .neighbors
            .iter()
            .map(|n| NeighborJson {
                end_idx: n.end_idx,
                distance: n.distance,
            })
            .collect(),
        points: result
            .forecast
            .points
            .iter()
            .map(|p| PointJson {
                step: p.step,
                median: p.median,
                p10: p.p10,
                p90: p.p90,
            })
            .collect(),
    }
}

pub fn print_human(f: &ForecastJson) {
    println!("Galoa forecast — {} (offline)", f.symbol);
    println!(
        "W={} H={} k_used={}/{}  query_last={:.4}",
        f.window, f.horizon, f.k_used, f.k_requested, f.query_last
    );
    println!("top neighbours (end_idx, L2 z-distance):");
    for (i, n) in f.neighbors.iter().take(5).enumerate() {
        println!("  #{i}: end_idx={} dist={:.6}", n.end_idx, n.distance);
    }
    println!("step  median      p10         p90");
    for p in &f.points {
        println!(
            "{:>4}  {:>10.4}  {:>10.4}  {:>10.4}",
            p.step, p.median, p.p10, p.p90
        );
    }
}
