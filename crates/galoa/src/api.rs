use crate::pipeline::{run_forecast, ForecastJson};
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
    window: usize,
    horizon: usize,
    k: usize,
    dt_secs: i64,
}

#[derive(Deserialize, Default)]
pub struct ForecastRequest {
    pub window: Option<usize>,
    pub horizon: Option<usize>,
    pub k: Option<usize>,
    pub dt_secs: Option<i64>,
}

pub async fn serve(
    data: Option<PathBuf>,
    synthetic: bool,
    window: usize,
    horizon: usize,
    k: usize,
    dt_secs: i64,
    port: u16,
) -> Result<()> {
    let synthetic = synthetic || data.is_none();
    let state = Arc::new(AppState {
        data,
        synthetic,
        window,
        horizon,
        k,
        dt_secs,
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
    let window = req.window.unwrap_or(state.window).max(1);
    let horizon = req.horizon.unwrap_or(state.horizon).max(1);
    let k = req.k.unwrap_or(state.k).max(1);
    let dt_secs = req.dt_secs.unwrap_or(state.dt_secs);

    run_forecast(
        state.data.clone(),
        state.synthetic,
        window,
        horizon,
        k,
        dt_secs,
    )
    .map(Json)
    .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e.to_string()))
}
