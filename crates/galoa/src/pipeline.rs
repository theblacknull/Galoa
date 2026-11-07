use anyhow::{bail, Context, Result};
use galoa_core::{
    resample_last_close, AbstainReason, GateConfig, MatchDecision, StftParams, XAUUSD,
    DEFAULT_HOP, DEFAULT_N_FFT,
};
use galoa_index::{CoverageStats, ForecastResult, WindowIndex};
use galoa_ingest::{generate_gold_series, load_series, SyntheticConfig, DEFAULT_GOLD_SERIES};
use serde::Serialize;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct ForecastOpts {
    pub window: usize,
    pub horizon: usize,
    pub k: usize,
    pub dt_secs: i64,
    pub n_fft: usize,
    pub hop: usize,
    pub d_max: f64,
    pub n_min: usize,
    pub sigma_max: f64,
}

impl Default for ForecastOpts {
    fn default() -> Self {
        let g = GateConfig::default();
        Self {
            window: galoa_index::DEFAULT_WINDOW,
            horizon: galoa_index::DEFAULT_HORIZON,
            k: galoa_index::DEFAULT_K,
            dt_secs: 0,
            n_fft: DEFAULT_N_FFT,
            hop: DEFAULT_HOP,
            d_max: g.d_max,
            n_min: g.n_min,
            sigma_max: g.sigma_max,
        }
    }
}

#[derive(Serialize)]
pub struct ForecastJson {
    pub symbol: String,
    pub matching: &'static str,
    pub decision: MatchDecision,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub abstain_reason: Option<AbstainReason>,
    pub support: usize,
    pub outcome_sigma: f64,
    pub window: usize,
    pub horizon: usize,
    pub k_requested: usize,
    pub k_used: usize,
    pub query_last: f64,
    pub stft_n_fft: usize,
    pub stft_hop: usize,
    pub d_max: f64,
    pub n_min: usize,
    pub sigma_max: f64,
    pub neighbors: Vec<NeighborJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub points: Option<Vec<PointJson>>,
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

#[derive(Serialize)]
pub struct CoverageJson {
    pub symbol: String,
    pub matching: &'static str,
    pub queries: usize,
    pub fire: usize,
    pub abstain: usize,
    pub coverage: f64,
    pub low_support: usize,
    pub high_dispersion: usize,
    pub no_analogues: usize,
    pub window: usize,
    pub horizon: usize,
    pub k: usize,
    pub d_max: f64,
    pub n_min: usize,
    pub sigma_max: f64,
}

pub fn run_forecast(
    data: Option<PathBuf>,
    synthetic: bool,
    opts: &ForecastOpts,
) -> Result<ForecastJson> {
    let index = build_index(data, synthetic, opts)?;
    let result = index.forecast_latest(opts.k)?;
    Ok(to_json(&result, opts))
}

pub fn run_scan(
    data: Option<PathBuf>,
    synthetic: bool,
    opts: &ForecastOpts,
) -> Result<CoverageJson> {
    let index = build_index(data, synthetic, opts)?;
    let stats = index.scan_coverage(opts.k)?;
    Ok(coverage_json(&stats, opts))
}

fn build_index(
    data: Option<PathBuf>,
    synthetic: bool,
    opts: &ForecastOpts,
) -> Result<WindowIndex> {
    let mut events = if synthetic {
        if let Some(ref p) = data {
            tracing::warn!("--synthetic set; ignoring --data {}", p.display());
        }
        generate_gold_series(&SyntheticConfig::default())
    } else {
        let path = data.unwrap_or_else(|| PathBuf::from(DEFAULT_GOLD_SERIES));
        load_series(&path).with_context(|| format!("load {}", path.display()))?
    };

    if opts.dt_secs > 0 {
        let dt_ns = opts
            .dt_secs
            .checked_mul(1_000_000_000)
            .context("dt_secs overflow")?;
        events = resample_last_close(&events, dt_ns)?;
    }

    if events.is_empty() {
        bail!("no price events after load/resample");
    }

    let stft = StftParams {
        n_fft: opts.n_fft,
        hop: opts.hop,
    };
    let gates = GateConfig {
        d_max: opts.d_max,
        n_min: opts.n_min,
        sigma_max: opts.sigma_max,
    };
    Ok(WindowIndex::from_events_with(
        &events,
        opts.window,
        opts.horizon,
        stft,
        gates,
    )?)
}

fn to_json(result: &ForecastResult, opts: &ForecastOpts) -> ForecastJson {
    let (k_used, points) = match &result.forecast {
        Some(f) => (
            f.k_used,
            Some(
                f.points
                    .iter()
                    .map(|p| PointJson {
                        step: p.step,
                        median: p.median,
                        p10: p.p10,
                        p90: p.p90,
                    })
                    .collect(),
            ),
        ),
        None => (0, None),
    };
    ForecastJson {
        symbol: XAUUSD.to_string(),
        matching: "spectral_stft",
        decision: result.decision.clone(),
        abstain_reason: result.abstain_reason.clone(),
        support: result.support,
        outcome_sigma: result.outcome_sigma,
        window: opts.window,
        horizon: opts.horizon,
        k_requested: opts.k,
        k_used,
        query_last: result.query_last,
        stft_n_fft: opts.n_fft,
        stft_hop: opts.hop,
        d_max: opts.d_max,
        n_min: opts.n_min,
        sigma_max: opts.sigma_max,
        neighbors: result
            .neighbors
            .iter()
            .map(|n| NeighborJson {
                end_idx: n.end_idx,
                distance: n.distance,
            })
            .collect(),
        points,
    }
}

fn coverage_json(stats: &CoverageStats, opts: &ForecastOpts) -> CoverageJson {
    CoverageJson {
        symbol: XAUUSD.to_string(),
        matching: "spectral_stft",
        queries: stats.queries,
        fire: stats.fire,
        abstain: stats.abstain,
        coverage: stats.coverage,
        low_support: stats.low_support,
        high_dispersion: stats.high_dispersion,
        no_analogues: stats.no_analogues,
        window: opts.window,
        horizon: opts.horizon,
        k: opts.k,
        d_max: opts.d_max,
        n_min: opts.n_min,
        sigma_max: opts.sigma_max,
    }
}

pub fn print_human(f: &ForecastJson) {
    println!("Galoa forecast — {} (spectral STFT only)", f.symbol);
    println!(
        "decision={:?}{}  support={}  σ_outcome={:.6}",
        f.decision,
        f.abstain_reason
            .as_ref()
            .map(|r| format!(" ({r:?})"))
            .unwrap_or_default(),
        f.support,
        f.outcome_sigma
    );
    println!(
        "W={} H={} k_used={}/{}  query_last={:.4}  STFT n_fft={} hop={}",
        f.window, f.horizon, f.k_used, f.k_requested, f.query_last, f.stft_n_fft, f.stft_hop
    );
    println!(
        "gates: d_max={} n_min={} sigma_max={}",
        f.d_max, f.n_min, f.sigma_max
    );
    println!("top neighbours (end_idx, L2 STFT-feature distance):");
    for (i, n) in f.neighbors.iter().take(5).enumerate() {
        println!("  #{i}: end_idx={} dist={:.6}", n.end_idx, n.distance);
    }
    match &f.points {
        Some(points) if f.decision == MatchDecision::Fire => {
            println!("step  median      p10         p90");
            for p in points {
                println!(
                    "{:>4}  {:>10.4}  {:>10.4}  {:>10.4}",
                    p.step, p.median, p.p10, p.p90
                );
            }
        }
        _ => println!("ABSTAIN — no ensemble bands emitted"),
    }
}

pub fn print_coverage(c: &CoverageJson) {
    println!("Galoa scan coverage — {} (spectral STFT)", c.symbol);
    println!(
        "queries={}  FIRE={}  ABSTAIN={}  coverage={:.2}%",
        c.queries,
        c.fire,
        c.abstain,
        c.coverage * 100.0
    );
    println!(
        "abstain breakdown: low_support={} high_dispersion={} no_analogues={}",
        c.low_support, c.high_dispersion, c.no_analogues
    );
    println!(
        "W={} H={} k={}  gates d_max={} n_min={} sigma_max={}",
        c.window, c.horizon, c.k, c.d_max, c.n_min, c.sigma_max
    );
}
