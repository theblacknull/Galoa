use serde::{Deserialize, Serialize};

/// Internal symbol id (cTrader `symbolId` later; test data uses a fixed id).
pub type SymbolId = u32;

/// First-focus symbol name for Gold.
pub const XAUUSD: &str = "XAUUSD";

/// Default symbol id used for synthetic / replay XAUUSD series.
pub const DEFAULT_SYMBOL: SymbolId = 1;

/// Core price event: one scalar close per timestamp.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PriceEvent {
    pub symbol_id: SymbolId,
    /// Unix time in nanoseconds.
    pub ts_ns: i64,
    pub close: f64,
}

impl PriceEvent {
    pub fn new(symbol_id: SymbolId, ts_ns: i64, close: f64) -> Self {
        Self {
            symbol_id,
            ts_ns,
            close,
        }
    }
}
