use crate::pipeline::{run_forecast, ForecastJson, ForecastOpts};
use anyhow::Result;
use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone)]
struct AppState {
    data: Option<PathBuf>,
    synthetic: bool,
    opts: ForecastOpts,
}

#[derive(Deserialize, Default)]
pub struct ForecastRequest {
    pub window: Option<usize>,
    pub horizon: Option<usize>,
    pub k: Option<usize>,
    pub dt_secs: Option<i64>,
    pub n_fft: Option<usize>,
    pub hop: Option<usize>,
    pub d_max: Option<f64>,
    pub n_min: Option<usize>,
    pub sigma_max: Option<f64>,
}

pub async fn serve(
    data: Option<PathBuf>,
    synthetic: bool,
    opts: ForecastOpts,
    port: u16,
) -> Result<()> {
    let state = Arc::new(AppState {
        data,
        synthetic,
        opts,
    });

    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/forecast", post(forecast_handler))
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    tracing::info!("listening on http://{addr}  POST /forecast  GET /health");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn forecast_handler(
    State(state): State<Arc<AppState>>,
    body: Option<Json<ForecastRequest>>,
) -> Result<Json<ForecastJson>, (axum::http::StatusCode, String)> {
    let req = body.map(|j| j.0).unwrap_or_default();
    let mut opts = state.opts.clone();
    if let Some(v) = req.window {
        opts.window = v.max(1);
    }
    if let Some(v) = req.horizon {
        opts.horizon = v.max(1);
    }
    if let Some(v) = req.k {
        opts.k = v.max(1);
    }
    if let Some(v) = req.dt_secs {
        opts.dt_secs = v;
    }
    if let Some(v) = req.n_fft {
        opts.n_fft = v.max(1);
    }
    if let Some(v) = req.hop {
        opts.hop = v.max(1);
    }
    if let Some(v) = req.d_max {
        opts.d_max = v;
    }
    if let Some(v) = req.n_min {
        opts.n_min = v.max(1);
    }
    if let Some(v) = req.sigma_max {
        opts.sigma_max = v;
    }

    run_forecast(state.data.clone(), state.synthetic, &opts)
        .map(Json)
        .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e.to_string()))
}
