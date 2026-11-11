//! Bundled dataset (port of `datasets/load_global_average_temperatures_df.py`).
//!
//! Global land temperatures by major city
//! (<https://www.kaggle.com/datasets/shishu1421/global-temperature/data>).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::error::{Result, SsaError};

/// Default location of the bundled CSV.
pub fn default_csv_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets").join("GlobalLandTemperaturesByMajorCity.csv")
}

/// One row of the CSV (`dt,AverageTemperature,AverageTemperatureUncertainty,
/// City,Country,Latitude,Longitude`).  Missing numbers are `None`.
#[derive(Debug, Clone, PartialEq)]
pub struct TemperatureRecord {
    pub dt: String,
    pub average_temperature: Option<f64>,
    pub average_temperature_uncertainty: Option<f64>,
    pub city: String,
    pub country: String,
    pub latitude: String,
    pub longitude: String,
}

impl TemperatureRecord {
    fn is_complete(&self) -> bool {
        self.average_temperature.is_some()
            && self.average_temperature_uncertainty.is_some()
            && !self.dt.is_empty()
            && !self.city.is_empty()
            && !self.country.is_empty()
            && !self.latitude.is_empty()
            && !self.longitude.is_empty()
    }
}

/// Loads the bundled dataset.
///
/// * `raw = true`: every row, as read from the CSV (Python `rawDS=True`).
/// * `raw = false`: rows with any missing field are dropped (Python
///   `dropna(axis=0)`).  Python then transposes the frame so that dates
///   become columns; in Rust the records are kept row-wise, use
///   [`city_series`] / [`cities_matrix`] to obtain series.
pub fn load_global_average_temperatures(raw: bool) -> Result<Vec<TemperatureRecord>> {
    load_global_average_temperatures_from(default_csv_path(), raw)
}

/// Same as [`load_global_average_temperatures`] from an explicit path.
pub fn load_global_average_temperatures_from(
    path: impl AsRef<Path>,
    raw: bool,
) -> Result<Vec<TemperatureRecord>> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)
        .map_err(|e| SsaError::Io(format!("{}: {e}", path.display())))?;
    let mut lines = text.lines();
    let header = lines.next().ok_or_else(|| SsaError::Io("empty CSV".into()))?;
    let expected = "dt,AverageTemperature,AverageTemperatureUncertainty,City,Country,Latitude,Longitude";
    if header.trim_start_matches('\u{feff}').trim_end() != expected {
        return Err(SsaError::Io(format!("unexpected CSV header: {header}")));
    }
    let mut out = Vec::new();
    for (no, line) in lines.enumerate() {
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() != 7 {
            return Err(SsaError::Io(format!("line {}: expected 7 fields, got {}", no + 2, f.len())));
        }
        let num = |s: &str| -> Result<Option<f64>> {
            if s.is_empty() {
                Ok(None)
            } else {
                s.parse::<f64>()
                    .map(Some)
                    .map_err(|e| SsaError::Io(format!("line {}: {e}", no + 2)))
            }
        };
        let rec = TemperatureRecord {
            dt: f[0].to_string(),
            average_temperature: num(f[1])?,
            average_temperature_uncertainty: num(f[2])?,
            city: f[3].to_string(),
            country: f[4].to_string(),
            latitude: f[5].to_string(),
            longitude: f[6].to_string(),
        };
        if raw || rec.is_complete() {
            out.push(rec);
        }
    }
    Ok(out)
}

/// `(dates, temperatures)` of one city, in file order (complete rows only).
pub fn city_series(records: &[TemperatureRecord], city: &str) -> (Vec<String>, Vec<f64>) {
    records
        .iter()
        .filter(|r| r.city == city)
        .filter_map(|r| r.average_temperature.map(|t| (r.dt.clone(), t)))
        .unzip()
}

/// Aligns several cities on their common dates: returns the sorted dates and
/// an `(n_dates, n_cities)` matrix ready for [`crate::Mssa`].
pub fn cities_matrix(records: &[TemperatureRecord], cities: &[&str]) -> (Vec<String>, crate::Matrix) {
    let mut per_city: Vec<BTreeMap<&str, f64>> = vec![BTreeMap::new(); cities.len()];
    for r in records {
        if let (Some(pos), Some(t)) = (cities.iter().position(|c| *c == r.city), r.average_temperature) {
            per_city[pos].insert(r.dt.as_str(), t);
        }
    }
    let dates: Vec<String> = match per_city.first() {
        Some(first) => first
            .keys()
            .filter(|d| per_city.iter().all(|m| m.contains_key(*d)))
            .map(|d| d.to_string())
            .collect(),
        None => Vec::new(),
    };
    let m = crate::Matrix::from_fn(dates.len(), cities.len(), |i, j| per_city[j][dates[i].as_str()]);
    (dates, m)
}
