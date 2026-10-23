//! cTrader Open API client — **stub for MVP**.
//!
//! Live WebSocket / Protobuf auth is deferred until Open API credentials are
//! available. This crate only defines the intended surface and maps spot/bar
//! quotes into [`galoa_core::PriceEvent`] once real payloads arrive.
//!
//! See design doc: close = mid(bid,ask) when both present, else bid for spots;
//! trendbar close for bars. Never commit secrets; use env only.

use galoa_core::{PriceEvent, SymbolId};
use thiserror::Error;

/// Quote field policy for spot events (no true OHLC close).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum QuoteField {
    #[default]
    Mid,
    Bid,
    Ask,
}

#[derive(Debug, Error)]
pub enum CTraderError {
    #[error("cTrader live ingest is not enabled in this MVP build (use Gold CSV / generate)")]
    NotImplemented,
    #[error("missing credentials: set CTRADER_CLIENT_ID / CTRADER_CLIENT_SECRET via env (never commit)")]
    MissingCredentials,
}

/// Placeholder config — fields mirror future env wiring; unused until live phase.
#[derive(Clone, Debug, Default)]
pub struct CTraderConfig {
    pub quote_field: QuoteField,
    pub demo: bool,
}

/// Stub client. All network methods return [`CTraderError::NotImplemented`].
#[derive(Clone, Debug, Default)]
pub struct CTraderClient {
    pub config: CTraderConfig,
}

impl CTraderClient {
    pub fn new(config: CTraderConfig) -> Self {
        Self { config }
    }

    /// Would connect, auth, heartbeat, subscribe — deferred.
    pub async fn connect_and_subscribe(&self, _symbol: &str) -> Result<(), CTraderError> {
        let _ = (
            std::env::var("CTRADER_CLIENT_ID"),
            std::env::var("CTRADER_CLIENT_SECRET"),
        );
        // Intentionally do not require secrets to be present for offline MVP.
        Err(CTraderError::NotImplemented)
    }

    /// Map a spot quote into a core price event (unit-tested path for later live use).
    pub fn map_spot(
        &self,
        symbol_id: SymbolId,
        ts_ns: i64,
        bid: Option<f64>,
        ask: Option<f64>,
    ) -> Option<PriceEvent> {
        let close = match self.config.quote_field {
            QuoteField::Mid => match (bid, ask) {
                (Some(b), Some(a)) => Some((b + a) / 2.0),
                (Some(b), None) => Some(b),
                (None, Some(a)) => Some(a),
                _ => None,
            },
            QuoteField::Bid => bid,
            QuoteField::Ask => ask,
        }?;
        Some(PriceEvent::new(symbol_id, ts_ns, close))
    }

    /// Map a trendbar close (already decoded to absolute price) into a price event.
    pub fn map_trendbar_close(
        &self,
        symbol_id: SymbolId,
        ts_ns: i64,
        close: f64,
    ) -> PriceEvent {
        PriceEvent::new(symbol_id, ts_ns, close)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mid_prefers_both() {
        let c = CTraderClient::default();
        let e = c.map_spot(1, 0, Some(10.0), Some(12.0)).unwrap();
        assert_eq!(e.close, 11.0);
    }

    #[test]
    fn mid_falls_back_to_bid() {
        let c = CTraderClient::default();
        let e = c.map_spot(1, 0, Some(10.0), None).unwrap();
        assert_eq!(e.close, 10.0);
    }
}
