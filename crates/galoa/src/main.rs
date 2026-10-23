//! Galoa offline MVP: Gold (XAUUSD) M1 bid-close → spectral STFT match → gated forecast.
//! Also: SpecReTF (`eval-specretf`) and RAFT (`eval-raft`) always-FIRE offline evals.

mod api;
mod eval_paper;
mod eval_raft;
mod eval_specretf;
mod pipeline;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use galoa_core::{
    DEFAULT_HOP, DEFAULT_N_FFT, SPECRETF_EMA_ALPHA, SPECRETF_HOP, SPECRETF_K, SPECRETF_N_FFT,
};
use galoa_index::{SignalMode, DEFAULT_HORIZON, DEFAULT_K, DEFAULT_WINDOW};
use galoa_ingest::{generate_gold_series, write_csv, SyntheticConfig, DEFAULT_GOLD_SERIES};
use galoa_raft::{DEFAULT_LOOKBACK, DEFAULT_TAU, DEFAULT_TOP_M};
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
    /// SpecReTF retrieval-only on paper datasets (ETT/Exchange; train→test protocol).
    ///
    /// STFT on StandardScaler raw values; no neural fusion (eq. 11). Compare MSE
    /// to paper Table 1 SpecReTF/RAFT with an expected gap.
    EvalPaperSpecretf {
        #[arg(long, default_value = "data/ETT/ETTh1.csv")]
        data: PathBuf,
        #[arg(long, default_value = "ETTh1")]
        dataset: String,
        #[arg(long, default_value_t = 720)]
        lookback: usize,
        /// Comma-separated horizons, e.g. `96` or `96,192,336,720`.
        #[arg(long, default_value = "96")]
        horizons: String,
        #[arg(long, default_value_t = 10)]
        k: usize,
        #[arg(long, default_value_t = 32)]
        n_fft: usize,
        #[arg(long, default_value_t = 16)]
        hop: usize,
        #[arg(long, default_value_t = SPECRETF_EMA_ALPHA)]
        ema_alpha: f64,
        #[arg(long, default_value_t = 4)]
        library_stride: usize,
        #[arg(long, default_value_t = 24)]
        query_stride: usize,
        #[arg(long)]
        metrics_out: Option<PathBuf>,
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// SpecReTF-style walk-forward eval on Gold M1 (always FIRE; no abstain).
    ///
    /// Compares SpecReTF retrieval aggregate vs naive last-value and time-domain
    /// z-norm L2 kNN (baseline contrast only).
    EvalSpecretf {
        #[arg(long, default_value = DEFAULT_GOLD_SERIES)]
        data: PathBuf,
        #[arg(long, default_value_t = DEFAULT_WINDOW)]
        window: usize,
        #[arg(long, default_value_t = DEFAULT_HORIZON)]
        horizon: usize,
        /// Top-K; Gold M1 sweep preferred 20 (gated `forecast` still defaults to 10).
        #[arg(long, default_value_t = SPECRETF_K)]
        k: usize,
        #[arg(long, default_value_t = SPECRETF_N_FFT)]
        n_fft: usize,
        /// Hop; default M/2 = 16 for n_fft=32 (SpecReTF 50% overlap).
        #[arg(long, default_value_t = SPECRETF_HOP)]
        hop: usize,
        #[arg(long, default_value_t = SPECRETF_EMA_ALPHA)]
        ema_alpha: f64,
        /// STFT input: `raw_close` (Gold-tuned / paper-like) or `log_returns`.
        #[arg(long, default_value = "raw_close")]
        signal: String,
        /// Query stride (1 = every eligible bar; larger = faster).
        #[arg(long, default_value_t = 1)]
        stride: usize,
        /// Min prior library windows before first query.
        #[arg(long, default_value_t = 128)]
        min_history: usize,
        /// Write machine-readable metrics JSON (optional).
        #[arg(long)]
        metrics_out: Option<PathBuf>,
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Paper-like SpecReTF hyperparam sweep on Gold M1 (always FIRE).
    EvalSpecretfSweep {
        #[arg(long, default_value = DEFAULT_GOLD_SERIES)]
        data: PathBuf,
        /// Comma-separated lookbacks (default 64; try 128,256,720 if series allows).
        #[arg(long, default_value = "64")]
        windows: String,
        #[arg(long, default_value_t = DEFAULT_HORIZON)]
        horizon: usize,
        #[arg(long, default_value = "5,10,20")]
        ks: String,
        #[arg(long, default_value = "8,16,32")]
        n_ffts: String,
        #[arg(long, default_value = "0.01,0.05,0.1,0.2")]
        alphas: String,
        #[arg(long, default_value = "log_returns,raw_close")]
        signals: String,
        #[arg(long, default_value_t = 16)]
        stride: usize,
        #[arg(long, default_value_t = 128)]
        min_history: usize,
        #[arg(long)]
        metrics_out: Option<PathBuf>,
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// RAFT walk-forward eval on Gold M1 (arXiv:2505.04163; always FIRE).
    ///
    /// Time-domain Pearson retrieval + multi-period `{1,2,4}` + identity/ridge head.
    /// Side-by-side SpecReTF on the same L/F/m protocol unless `--no-specretf`.
    EvalRaft {
        #[arg(long, default_value = DEFAULT_GOLD_SERIES)]
        data: PathBuf,
        /// Lookback L (paper often 96–720; default 64 matches SpecReTF W on short 5d).
        #[arg(long, default_value_t = DEFAULT_LOOKBACK)]
        lookback: usize,
        #[arg(long, default_value_t = DEFAULT_HORIZON)]
        horizon: usize,
        /// Top-m neighbours (paper grid {1,5,10,20}).
        #[arg(long, default_value_t = DEFAULT_TOP_M)]
        top_m: usize,
        #[arg(long, default_value_t = DEFAULT_TAU)]
        tau: f64,
        /// Comma-separated periods (paper `{1,2,4}`).
        #[arg(long, default_value = "1,2,4")]
        periods: String,
        /// `pearson` (paper default) or `euclidean` (neg L2).
        #[arg(long, default_value = "pearson")]
        similarity: String,
        /// `identity` = retrieval+offset only; `ridge` = expanding linear stand-in.
        #[arg(long, default_value = "identity")]
        head: String,
        #[arg(long, default_value_t = 1)]
        stride: usize,
        #[arg(long, default_value_t = 128)]
        min_history: usize,
        /// Skip SpecReTF side-by-side (faster).
        #[arg(long, default_value_t = false)]
        no_specretf: bool,
        #[arg(long)]
        metrics_out: Option<PathBuf>,
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

fn parse_usizes(s: &str) -> Result<Vec<usize>> {
    s.split(',')
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .map(|p| p.parse::<usize>().map_err(|e| anyhow::anyhow!("bad usize `{p}`: {e}")))
        .collect()
}

fn parse_f64s(s: &str) -> Result<Vec<f64>> {
    s.split(',')
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .map(|p| p.parse::<f64>().map_err(|e| anyhow::anyhow!("bad f64 `{p}`: {e}")))
        .collect()
}

fn parse_signals(s: &str) -> Result<Vec<SignalMode>> {
    s.split(',')
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .map(|p| {
            SignalMode::parse(p)
                .ok_or_else(|| anyhow::anyhow!("unknown signal `{p}` (use log_returns|raw_close)"))
        })
        .collect()
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
        Commands::EvalPaperSpecretf {
            data,
            dataset,
            lookback,
            horizons,
            k,
            n_fft,
            hop,
            ema_alpha,
            library_stride,
            query_stride,
            metrics_out,
            json,
        } => {
            let horizons: Vec<usize> = horizons
                .split(',')
                .map(|s| s.trim().parse::<usize>())
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("parse --horizons (comma-separated ints)")?;
            let opts = eval_paper::PaperEvalOpts {
                data: data.clone(),
                dataset,
                lookback,
                horizons,
                k,
                n_fft,
                hop,
                ema_alpha,
                library_stride,
                query_stride,
                metrics_out,
            };
            let report = eval_paper::run_paper(&opts)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                eval_paper::print_report(&report, &data);
            }
        }
        Commands::EvalSpecretf {
            data,
            window,
            horizon,
            k,
            n_fft,
            hop,
            ema_alpha,
            signal,
            stride,
            min_history,
            metrics_out,
            json,
        } => {
            let signal = SignalMode::parse(&signal)
                .ok_or_else(|| anyhow::anyhow!("unknown --signal {signal} (use log_returns|raw_close)"))?;
            let opts = eval_specretf::EvalOpts {
                data: data.clone(),
                window,
                horizon,
                k,
                n_fft,
                hop,
                ema_alpha,
                signal,
                stride,
                min_history,
                metrics_out,
            };
            let report = eval_specretf::run_eval(&opts)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                eval_specretf::print_report(&report, &data);
            }
        }
        Commands::EvalSpecretfSweep {
            data,
            windows,
            horizon,
            ks,
            n_ffts,
            alphas,
            signals,
            stride,
            min_history,
            metrics_out,
            json,
        } => {
            let opts = eval_specretf::SweepOpts {
                data: data.clone(),
                windows: parse_usizes(&windows)?,
                horizon,
                ks: parse_usizes(&ks)?,
                n_ffts: parse_usizes(&n_ffts)?,
                alphas: parse_f64s(&alphas)?,
                signals: parse_signals(&signals)?,
                stride,
                min_history,
                metrics_out,
            };
            let report = eval_specretf::run_sweep(&opts)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                eval_specretf::print_sweep(&report, &data);
            }
        }
        Commands::EvalRaft {
            data,
            lookback,
            horizon,
            top_m,
            tau,
            periods,
            similarity,
            head,
            stride,
            min_history,
            no_specretf,
            metrics_out,
            json,
        } => {
            let opts = eval_raft::EvalRaftOpts {
                data: data.clone(),
                lookback,
                horizon,
                top_m,
                tau,
                periods: eval_raft::parse_periods(&periods)?,
                similarity: eval_raft::parse_similarity(&similarity)?,
                head: eval_raft::parse_head(&head)?,
                stride,
                min_history,
                with_specretf: !no_specretf,
                metrics_out,
            };
            let report = eval_raft::run_eval(&opts)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                eval_raft::print_report(&report, &data);
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
