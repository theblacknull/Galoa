//! Multi-period average pooling (paper §3.3, periods `{1,2,4}`).

/// Non-overlapping average pool with period `p`. Drops a trailing partial chunk.
pub fn avg_pool(x: &[f64], p: usize) -> Vec<f64> {
    if p == 0 {
        return Vec::new();
    }
    if p == 1 {
        return x.to_vec();
    }
    let n = x.len() / p;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let start = i * p;
        let chunk = &x[start..start + p];
        let m = chunk.iter().sum::<f64>() / p as f64;
        out.push(m);
    }
    out
}

/// Upsample a length-`⌊F/p⌋` vector back to `F` by repeating each value `p` times
/// (identity analog of paper’s linear `g^(p)` / official `repeat_interleave`).
pub fn repeat_upsample(v: &[f64], p: usize, target_len: usize) -> Vec<f64> {
    if p == 0 || target_len == 0 {
        return vec![0.0; target_len];
    }
    if p == 1 {
        let mut out = v.to_vec();
        out.resize(target_len, *v.last().unwrap_or(&0.0));
        return out;
    }
    let mut out = Vec::with_capacity(target_len);
    for &val in v {
        for _ in 0..p {
            if out.len() >= target_len {
                break;
            }
            out.push(val);
        }
        if out.len() >= target_len {
            break;
        }
    }
    while out.len() < target_len {
        out.push(*v.last().unwrap_or(&0.0));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_p2() {
        let x = [1.0, 3.0, 5.0, 7.0, 9.0];
        assert_eq!(avg_pool(&x, 2), vec![2.0, 6.0]);
    }

    #[test]
    fn upsample_roundtrip_shape() {
        let v = avg_pool(&[1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0], 2);
        let u = repeat_upsample(&v, 2, 8);
        assert_eq!(u.len(), 8);
        assert_eq!(u, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]);
    }
}
