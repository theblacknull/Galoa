//! Weighting strategies for the spectral channels (port of `weights_strategies.py`).
//!
//! In Python the strategy-specific parameters are passed as `**kwargs` to
//! `fit`; in Rust they are stored in the strategy value itself.

use crate::error::{Result, SsaError};
use crate::matrix::Matrix;

/// Assigns one weight to each of the `S` series of an `(N, S)` matrix.
pub trait WeightingStrategy: Send + Sync {
    /// Returns a weight vector of length `S`.
    fn compute_weights(&self, x: &Matrix) -> Result<Vec<f64>>;
}

/// Classical SSA/MSSA: every channel has weight 1.
#[derive(Debug, Clone, Copy, Default)]
pub struct UniformWeights;

impl WeightingStrategy for UniformWeights {
    fn compute_weights(&self, x: &Matrix) -> Result<Vec<f64>> {
        Ok(vec![1.0; x.ncols()])
    }
}

/// User supplied weights (must be strictly positive).
#[derive(Debug, Clone)]
pub struct PrecomputedWeights {
    pub weights: Vec<f64>,
}

impl PrecomputedWeights {
    pub fn new(weights: impl Into<Vec<f64>>) -> Self {
        Self { weights: weights.into() }
    }
}

impl WeightingStrategy for PrecomputedWeights {
    fn compute_weights(&self, _x: &Matrix) -> Result<Vec<f64>> {
        Ok(self.weights.clone())
    }
}

/// Gaussian kernel of the distance: `exp(-0.5 (d / bandwidth)^2)`.
#[derive(Debug, Clone)]
pub struct DistanceWeights {
    pub distances: Vec<f64>,
    pub bandwidth: f64,
}

impl DistanceWeights {
    /// `bandwidth` defaults to 1.0 in Python.
    pub fn new(distances: impl Into<Vec<f64>>, bandwidth: f64) -> Self {
        Self { distances: distances.into(), bandwidth }
    }
}

impl WeightingStrategy for DistanceWeights {
    fn compute_weights(&self, _x: &Matrix) -> Result<Vec<f64>> {
        Ok(gaussian(&self.distances, self.bandwidth))
    }
}

/// Distance kernel multiplied by `cluster_penalty` for neighbours that are
/// not in the focal cluster.
#[derive(Debug, Clone)]
pub struct ClusterAwareWeights<C: PartialEq + Clone + Send + Sync> {
    pub distances: Vec<f64>,
    pub focal_cluster: C,
    pub neighbour_clusters: Vec<C>,
    pub bandwidth: f64,
    pub cluster_penalty: f64,
}

impl<C: PartialEq + Clone + Send + Sync> ClusterAwareWeights<C> {
    /// Python defaults: `bandwidth = 1.0`, `cluster_penalty = 0.5`.
    pub fn new(
        distances: impl Into<Vec<f64>>,
        focal_cluster: C,
        neighbour_clusters: impl Into<Vec<C>>,
        bandwidth: f64,
        cluster_penalty: f64,
    ) -> Self {
        Self {
            distances: distances.into(),
            focal_cluster,
            neighbour_clusters: neighbour_clusters.into(),
            bandwidth,
            cluster_penalty,
        }
    }
}

impl<C: PartialEq + Clone + Send + Sync> WeightingStrategy for ClusterAwareWeights<C> {
    fn compute_weights(&self, _x: &Matrix) -> Result<Vec<f64>> {
        if self.distances.len() != self.neighbour_clusters.len() {
            // numpy would fail to broadcast the two arrays.
            return Err(SsaError::InvalidArgument(format!(
                "distances ({}) and neighbour_clusters ({}) must have the same length",
                self.distances.len(),
                self.neighbour_clusters.len()
            )));
        }
        let spatial = gaussian(&self.distances, self.bandwidth);
        Ok(spatial
            .into_iter()
            .zip(&self.neighbour_clusters)
            .map(|(w, c)| if *c == self.focal_cluster { w } else { w * self.cluster_penalty })
            .collect())
    }
}

fn gaussian(distances: &[f64], bandwidth: f64) -> Vec<f64> {
    distances.iter().map(|d| (-0.5 * (d / bandwidth).powi(2)).exp()).collect()
}
