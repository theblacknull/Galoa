//! Similarity metrics for RAFT key–query matching (paper §3.2 + App. C.1).

/// Pearson correlation in `[-1, 1]`. Returns `0` if either series is constant.
pub fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let mut mean_a = 0.0;
    let mut mean_b = 0.0;
    for i in 0..n {
        mean_a += a[i];
        mean_b += b[i];
    }
    mean_a /= n as f64;
    mean_b /= n as f64;

    let mut num = 0.0;
    let mut da2 = 0.0;
    let mut db2 = 0.0;
    for i in 0..n {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        num += da * db;
        da2 += da * da;
        db2 += db * db;
    }
    let denom = (da2 * db2).sqrt();
    if denom < 1e-15 {
        0.0
    } else {
        (num / denom).clamp(-1.0, 1.0)
    }
}

/// Negative Euclidean distance (higher = more similar), matching App. C.1 “Negative L2”.
pub fn neg_euclidean(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..n {
        let d = a[i] - b[i];
        s += d * d;
    }
    -s.sqrt()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SimilarityKind {
    #[default]
    Pearson,
    Euclidean,
}

impl SimilarityKind {
    pub fn score(self, a: &[f64], b: &[f64]) -> f64 {
        match self {
            Self::Pearson => pearson(a, b),
            Self::Euclidean => neg_euclidean(a, b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn pearson_identical_is_one() {
        let a = [1.0, 2.0, 3.0, 4.0];
        assert_relative_eq!(pearson(&a, &a), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn pearson_opposite_is_minus_one() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [4.0, 3.0, 2.0, 1.0];
        assert_relative_eq!(pearson(&a, &b), -1.0, epsilon = 1e-12);
    }

    #[test]
    fn euclid_identical_is_zero_neg() {
        let a = [1.0, 2.0, 3.0];
        assert_relative_eq!(neg_euclidean(&a, &a), 0.0, epsilon = 1e-12);
    }
}
