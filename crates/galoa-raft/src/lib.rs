//! RAFT — Retrieval Augmented Time Series Forecasting (arXiv:2505.04163).
//!
//! Time-domain sliding-patch retrieval (Pearson default), multi-period
//! downsample `{1,2,4}`, softmax top-m aggregation, optional linear/ridge head.
//! Dedicated crate so SpecReTF (spectral) and RAFT (time-domain) stay separate.

mod downsample;
mod eval;
mod head;
mod retrieve;
mod similarity;

pub use downsample::{avg_pool, repeat_upsample};
pub use eval::{
    run_raft_eval, run_raft_signals, ForecastSignal, MethodMetrics, RaftEvalConfig, RaftEvalReport,
    RaftSignal,
};
pub use head::{forecast_with_head, HeadKind, RidgeHead};
pub use retrieve::{
    forecast_retrieval_only, offset_subtract, retrieve_multiperiod, retrieve_period, softmax_tau,
    PeriodRetrieval, RaftError, RaftNeighbor, RaftParams, RaftRetrievalResult, DEFAULT_HORIZON,
    DEFAULT_LOOKBACK, DEFAULT_PERIODS, DEFAULT_TAU, DEFAULT_TOP_M,
};
pub use similarity::{neg_euclidean, pearson, SimilarityKind};
