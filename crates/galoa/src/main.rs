//! Galoa offline MVP: Gold (XAUUSD) M1 bid-close replay → similarity forecast.

mod api;
mod pipeline;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use galoa_index::{DEFAULT_HORIZON, DEFAULT_K, DEFAULT_WINDOW};
use galoa_ingest::{generate_gold_series, write_csv, SyntheticConfig, DEFAULT_GOLD_SERIES};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "galoa", about = "Galoa — similar-segment price forecast (offline Gold MVP)")]
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
    /// Run forecast on real MT5 Gold M1 (default) or optional synthetic.
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
        #[arg(long, default_value_t = 3000)]
        port: u16,
    },
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
            json,
        } => {
            let result = pipeline::run_forecast(
                Some(data),
                synthetic,
                window,
                horizon,
                k,
                dt_secs,
            )?;
            if json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                pipeline::print_human(&result);
            }
        }
        Commands::Serve {
            data,
            synthetic,
            window,
            horizon,
            k,
            dt_secs,
            port,
        } => {
            api::serve(Some(data), synthetic, window, horizon, k, dt_secs, port).await?;
        }
    }
    Ok(())
}
