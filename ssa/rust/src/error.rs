//! Error type shared by the whole crate.

use std::fmt;

/// Errors raised by the SSA/MSSA routines.
///
/// Wherever the Python reference raises a `ValueError` / `LinAlgError`,
/// the Rust port returns one of these variants instead.
#[derive(Debug, Clone, PartialEq)]
pub enum SsaError {
    /// Invalid argument (mirrors Python `ValueError`).
    InvalidArgument(String),
    /// Component index outside `0..d` (mirrors numpy `IndexError`).
    IndexOutOfRange { index: usize, len: usize },
    /// Numerical failure (mirrors numpy `LinAlgError`).
    LinAlg(String),
    /// I/O or parsing failure (dataset loader).
    Io(String),
}

impl fmt::Display for SsaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SsaError::InvalidArgument(m) => write!(f, "invalid argument: {m}"),
            SsaError::IndexOutOfRange { index, len } => {
                write!(f, "index {index} is out of bounds for {len} components")
            }
            SsaError::LinAlg(m) => write!(f, "linear algebra error: {m}"),
            SsaError::Io(m) => write!(f, "io error: {m}"),
        }
    }
}

impl std::error::Error for SsaError {}

pub type Result<T> = std::result::Result<T, SsaError>;
