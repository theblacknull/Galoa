//! Galoa offline MVP: Gold (XAUUSD) replay → similarity forecast.

mod api;
mod pipeline;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use galoa_index::{DEFAULT_HORIZON, DEFAULT_K, DEFAULT_WINDOW};
use galoa_ingest::{generate_gold_series, write_csv, SyntheticConfig};
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
    /// Generate synthetic XAUUSD close CSV (random-walk + planted analogues).
    Generate {
        /// Output CSV path
        #[arg(long, default_value = "data/xauusd_synthetic.csv")]
        out: PathBuf,
        #[arg(long, default_value_t = 2048)]
        bars: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Run forecast on a CSV replay (or generate-in-memory if --data omitted with --synthetic).
    Forecast {
        /// Path to `ts_ns,close` CSV
        #[arg(long)]
        data: Option<PathBuf>,
        /// Use in-memory synthetic Gold instead of a file
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
        #[arg(long)]
        data: Option<PathBuf>,
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
                "Wrote {} XAUUSD bars to {} (seed={})",
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
            let result = pipeline::run_forecast(data, synthetic, window, horizon, k, dt_secs)?;
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
            api::serve(data, synthetic, window, horizon, k, dt_secs, port).await?;
        }
    }
    Ok(())
}
