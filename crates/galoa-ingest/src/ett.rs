//! Multivariate ETT / Autoformer CSV loader for SpecReTF paper benchmarks.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EttError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("empty or header-only CSV: {0}")]
    Empty(String),
    #[error("parse float at line {line}: {msg}")]
    Parse { line: usize, msg: String },
    #[error("expected ≥1 numeric column after date")]
    NoFeatures,
}

/// Multivariate series: `values[t][c]`, column names (excluding date).
#[derive(Clone, Debug)]
pub struct MultivariateSeries {
    pub columns: Vec<String>,
    /// Row-major time × channel.
    pub values: Vec<Vec<f64>>,
}

impl MultivariateSeries {
    pub fn n_steps(&self) -> usize {
        self.values.len()
    }

    pub fn n_channels(&self) -> usize {
        self.columns.len()
    }

    /// Column-major view: `channels[c][t]`.
    pub fn channels(&self) -> Vec<Vec<f64>> {
        let t = self.n_steps();
        let c = self.n_channels();
        let mut out = vec![vec![0.0; t]; c];
        for (ti, row) in self.values.iter().enumerate() {
            for (ci, &v) in row.iter().enumerate() {
                out[ci][ti] = v;
            }
        }
        out
    }
}

/// Load Autoformer-style CSV: first column date (skipped), remaining floats.
pub fn load_multivariate_csv(path: &Path) -> Result<MultivariateSeries, EttError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    let header = lines
        .next()
        .ok_or_else(|| EttError::Empty(path.display().to_string()))??;
    let cols: Vec<String> = header
        .split(',')
        .skip(1)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if cols.is_empty() {
        return Err(EttError::NoFeatures);
    }
    let n_c = cols.len();
    let mut values = Vec::new();
    for (i, line) in lines.enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < n_c + 1 {
            return Err(EttError::Parse {
                line: i + 2,
                msg: format!("got {} fields, need {}", parts.len(), n_c + 1),
            });
        }
        let mut row = Vec::with_capacity(n_c);
        for j in 0..n_c {
            let s = parts[j + 1].trim();
            let v: f64 = s.parse().map_err(|e| EttError::Parse {
                line: i + 2,
                msg: format!("{e} ({s})"),
            })?;
            row.push(v);
        }
        values.push(row);
    }
    if values.is_empty() {
        return Err(EttError::Empty(path.display().to_string()));
    }
    Ok(MultivariateSeries {
        columns: cols,
        values,
    })
}

/// Informer/Autoformer border lengths for ETTh* (hourly).
pub fn etth_borders(seq_len: usize) -> [(usize, usize); 3] {
    let train_end = 12 * 30 * 24; // 8640
    let val_end = train_end + 4 * 30 * 24; // 11520
    let test_end = val_end + 4 * 30 * 24; // 14400
    [
        (0, train_end),
        (train_end.saturating_sub(seq_len), val_end),
        (val_end.saturating_sub(seq_len), test_end),
    ]
}

/// ETTm* (15-min): 4× denser than hourly borders.
pub fn ettm_borders(seq_len: usize) -> [(usize, usize); 3] {
    let train_end = 12 * 30 * 24 * 4;
    let val_end = train_end + 4 * 30 * 24 * 4;
    let test_end = val_end + 4 * 30 * 24 * 4;
    [
        (0, train_end),
        (train_end.saturating_sub(seq_len), val_end),
        (val_end.saturating_sub(seq_len), test_end),
    ]
}

/// Exchange-rate style: 70% / 10% / 20% of full length (Autoformer custom).
pub fn percent_borders(n: usize, seq_len: usize) -> [(usize, usize); 3] {
    let n_train = (n as f64 * 0.7) as usize;
    let n_test = (n as f64 * 0.2) as usize;
    let n_vali = n - n_train - n_test;
    let train_end = n_train;
    let val_end = n_train + n_vali;
    let test_end = n;
    [
        (0, train_end),
        (train_end.saturating_sub(seq_len), val_end),
        (val_end.saturating_sub(seq_len), test_end),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn etth_border_lengths() {
        let b = etth_borders(720);
        assert_eq!(b[0], (0, 8640));
        assert_eq!(b[1].1, 11520);
        assert_eq!(b[2].1, 14400);
        assert_eq!(b[2].0, 11520 - 720);
    }
}
