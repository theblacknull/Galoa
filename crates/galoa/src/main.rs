//! Galoa offline MVP: Gold (XAUUSD) M1 bid-close → spectral STFT match → gated forecast.

mod api;
mod pipeline;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use galoa_core::{DEFAULT_HOP, DEFAULT_N_FFT};
use galoa_index::{DEFAULT_HORIZON, DEFAULT_K, DEFAULT_WINDOW};
use galoa_ingest::{generate_gold_series, write_csv, SyntheticConfig, DEFAULT_GOLD_SERIES};
use pipeline::ForecastOpts;
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(
    name = "galoa",
    about = "Galoa — spectral similar-segment price forecast (offline Gold MVP)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Generate synthetic XAUUSD close CSV (dev only; not the primary series).
    Generate {
        /// Output CSV path
        #[arg(long, default_value = "data/xauusd_synthetic.csv")]
        out: PathBuf,
        #[arg(long, default_value_t = 2048)]
        bars: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Run spectral forecast on real MT5 Gold M1 (default) or optional synthetic.
    Forecast {
        /// Path to `.bin` (GALOAP1) or `ts_ns,close` CSV. Default: real 5d M1 bin.
        #[arg(long, default_value = DEFAULT_GOLD_SERIES)]
        data: PathBuf,
        /// Use in-memory synthetic Gold instead of a file (dev only)
        #[arg(long, default_value_t = false)]
        synthetic: bool,
        #[arg(long, default_value_t = DEFAULT_WINDOW)]
        window: usize,
        #[arg(long, default_value_t = DEFAULT_HORIZON)]
        horizon: usize,
        #[arg(long, default_value_t = DEFAULT_K)]
        k: usize,
        /// Resample Δt in seconds (0 = no resample, use series as-is)
        #[arg(long, default_value_t = 0)]
        dt_secs: i64,
        /// STFT FFT size (Hann-windowed frames on log-returns)
        #[arg(long, default_value_t = DEFAULT_N_FFT)]
        n_fft: usize,
        /// STFT hop length
        #[arg(long, default_value_t = DEFAULT_HOP)]
        hop: usize,
        /// Max L2 feature distance for a neighbour to count as support
        #[arg(long, default_value_t = 0.48)]
        d_max: f64,
        /// Min neighbours within d_max to FIRE
        #[arg(long, default_value_t = 3)]
        n_min: usize,
        /// Max std of H-step log-returns across support (outcome dispersion)
        #[arg(long, default_value_t = 0.0025)]
        sigma_max: f64,
        /// Walk the series and report FIRE/ABSTAIN coverage instead of one query
        #[arg(long, default_value_t = false)]
        scan: bool,
        /// Pretty-print JSON to stdout
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Serve `POST /forecast` (and `GET /health`) over axum.
    Serve {
        #[arg(long, default_value = DEFAULT_GOLD_SERIES)]
        data: PathBuf,
        #[arg(long, default_value_t = false)]
        synthetic: bool,
        #[arg(long, default_value_t = DEFAULT_WINDOW)]
        window: usize,
        #[arg(long, default_value_t = DEFAULT_HORIZON)]
        horizon: usize,
        #[arg(long, default_value_t = DEFAULT_K)]
        k: usize,
        #[arg(long, default_value_t = 0)]
        dt_secs: i64,
        #[arg(long, default_value_t = DEFAULT_N_FFT)]
        n_fft: usize,
        #[arg(long, default_value_t = DEFAULT_HOP)]
        hop: usize,
        #[arg(long, default_value_t = 0.48)]
        d_max: f64,
        #[arg(long, default_value_t = 3)]
        n_min: usize,
        #[arg(long, default_value_t = 0.0025)]
        sigma_max: f64,
        #[arg(long, default_value_t = 3000)]
        port: u16,
    },
}

fn opts_from(
    window: usize,
    horizon: usize,
    k: usize,
    dt_secs: i64,
    n_fft: usize,
    hop: usize,
    d_max: f64,
    n_min: usize,
    sigma_max: f64,
) -> ForecastOpts {
    ForecastOpts {
        window,
        horizon,
        k,
        dt_secs,
        n_fft,
        hop,
        d_max,
        n_min,
        sigma_max,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("galoa=info".parse()?))
        .init();

    let cli = Cli::parse();
    match cli.command {
        Commands::Generate { out, bars, seed } => {
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let cfg = SyntheticConfig {
                bars,
                seed,
                ..Default::default()
            };
            let events = generate_gold_series(&cfg);
            write_csv(&out, &events).with_context(|| format!("write {}", out.display()))?;
            println!(
                "Wrote {} synthetic XAUUSD bars to {} (seed={}) — dev only",
                events.len(),
                out.display(),
                seed
            );
        }
        Commands::Forecast {
            data,
            synthetic,
            window,
            horizon,
            k,
            dt_secs,
            n_fft,
            hop,
            d_max,
            n_min,
            sigma_max,
            scan,
            json,
        } => {
            let opts = opts_from(
                window, horizon, k, dt_secs, n_fft, hop, d_max, n_min, sigma_max,
            );
            if scan {
                let cov = pipeline::run_scan(Some(data), synthetic, &opts)?;
                if json {
                    println!("{}", serde_json::to_string_pretty(&cov)?);
                } else {
                    pipeline::print_coverage(&cov);
                }
            } else {
                let result = pipeline::run_forecast(Some(data), synthetic, &opts)?;
                if json {
                    println!("{}", serde_json::to_string_pretty(&result)?);
                } else {
                    pipeline::print_human(&result);
                }
            }
        }
        Commands::Serve {
            data,
            synthetic,
            window,
            horizon,
            k,
            dt_secs,
            n_fft,
            hop,
            d_max,
            n_min,
            sigma_max,
            port,
        } => {
            let opts = opts_from(
                window, horizon, k, dt_secs, n_fft, hop, d_max, n_min, sigma_max,
            );
            api::serve(Some(data), synthetic, opts, port).await?;
        }
    }
    Ok(())
}
